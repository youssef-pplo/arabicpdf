"""PDF page type detection for ArabicPDF.

Classifies each page of an Arabic PDF as one of:
- DIGITAL_CLEAN:     Valid, extractable Arabic text.
- DIGITAL_CORRUPTED: Text layer exists but is garbled / wrong encoding.
- SCANNED:           Image-only page with no usable text layer.
- MIXED:             Page has both digital-text and scanned-image regions.

Uses PyMuPDF (``fitz``) for PDF parsing and delegates Arabic-specific
heuristics to :mod:`arabicpdf.arabic_utils`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None

from arabicpdf.config import PageType
from arabicpdf.arabic_utils import is_valid_arabic, arabic_char_ratio, char_entropy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_MIN_TEXT_LEN = 10  # Below this the page is considered text-empty
_ARABIC_CLEAN_MIN = 0.5  # Minimum Arabic-char ratio for DIGITAL_CLEAN
_ARABIC_CORRUPT_MAX = 0.3  # Below this → DIGITAL_CORRUPTED
_ENTROPY_LOW = 1.5  # Abnormally low entropy → garbled
_ENTROPY_HIGH = 7.0  # Abnormally high entropy → garbled
_ENTROPY_NORMAL_LOW = 3.0  # Lower bound of "normal" entropy
_ENTROPY_NORMAL_HIGH = 6.0  # Upper bound of "normal" entropy


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass
class PageClassification:
    """Classification result for a single PDF page.

    Attributes:
        page_num:    1-indexed page number.
        page_type:   Detected :class:`PageType`.
        raw_text:    Extracted text (if any).
        confidence:  Confidence of the classification (0.0 – 1.0).
        has_images:  Whether the page contains embedded images.
    """

    page_num: int
    page_type: PageType
    raw_text: Optional[str] = None
    confidence: float = 0.0
    has_images: bool = False


# ---------------------------------------------------------------------------
# Single-page classifier
# ---------------------------------------------------------------------------
def classify_page(pdf_path: str, page_num: int) -> PageClassification:
    """Classify a single page of a PDF document.

    Args:
        pdf_path: Filesystem path to the PDF.
        page_num: **1-indexed** page number.

    Returns:
        A :class:`PageClassification` describing the page.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        ValueError: If *page_num* is out of range.
    """
    if fitz is None:
        raise ImportError(
            "PyMuPDF is required for PDF detection. Install it with: pip install pymupdf"
        )
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("Failed to open PDF '%s': %s", pdf_path, exc)
        return PageClassification(
            page_num=page_num,
            page_type=PageType.SCANNED,
            raw_text=None,
            confidence=0.0,
            has_images=False,
        )

    try:
        total_pages = doc.page_count
        # Convert 1-indexed → 0-indexed
        idx = page_num - 1
        if idx < 0 or idx >= total_pages:
            raise ValueError(
                f"page_num {page_num} out of range "
                f"(document has {total_pages} page(s))"
            )

        page = doc[idx]

        # ----- text extraction -------------------------------------------------
        try:
            raw_text: str = page.get_text("text") or ""
        except Exception:
            logger.warning(
                "Text extraction failed for page %d of '%s'", page_num, pdf_path
            )
            raw_text = ""

        # ----- image detection --------------------------------------------------
        try:
            images = page.get_images(full=True)
            has_images = len(images) > 0
        except Exception:
            logger.warning(
                "Image detection failed for page %d of '%s'", page_num, pdf_path
            )
            images = []
            has_images = False

        # ----- classification logic --------------------------------------------
        text_stripped = raw_text.strip()

        if len(text_stripped) < _MIN_TEXT_LEN:
            # Very little / no text → scanned if images present, otherwise
            # still mark scanned (blank or unreadable page).
            return PageClassification(
                page_num=page_num,
                page_type=PageType.SCANNED,
                raw_text=raw_text if raw_text else None,
                confidence=0.9 if has_images else 0.6,
                has_images=has_images,
            )

        # There *is* text — compute Arabic heuristics
        ratio = arabic_char_ratio(text_stripped)
        entropy = char_entropy(text_stripped)

        page_type, confidence = _decide_type(ratio, entropy, has_images)

        return PageClassification(
            page_num=page_num,
            page_type=page_type,
            raw_text=raw_text,
            confidence=confidence,
            has_images=has_images,
        )

    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Internal decision function
# ---------------------------------------------------------------------------
def _decide_type(
    arabic_ratio: float,
    entropy: float,
    has_images: bool,
) -> tuple[PageType, float]:
    """Return ``(PageType, confidence)`` given computed metrics.

    Decision tree:

    1. Extreme entropy (< 1.5 or > 7.0) → DIGITAL_CORRUPTED
    2. arabic_ratio ≥ 0.5 **and** normal entropy (3.0–6.0) → DIGITAL_CLEAN
    3. arabic_ratio < 0.3 → DIGITAL_CORRUPTED
    4. arabic_ratio 0.3–0.5 → MIXED
    5. Fall-through with abnormal-but-not-extreme entropy → DIGITAL_CORRUPTED
    """

    # 1. Extreme entropy — almost certainly garbled
    if entropy < _ENTROPY_LOW or entropy > _ENTROPY_HIGH:
        return PageType.DIGITAL_CORRUPTED, 0.85

    # 2. Healthy Arabic text
    if arabic_ratio >= _ARABIC_CLEAN_MIN:
        if _ENTROPY_NORMAL_LOW <= entropy <= _ENTROPY_NORMAL_HIGH:
            return PageType.DIGITAL_CLEAN, 0.95
        # Arabic ratio is good but entropy slightly off — still mostly clean
        # but lower confidence.
        return PageType.DIGITAL_CLEAN, 0.75

    # 3. Very low Arabic ratio — garbled encoding
    if arabic_ratio < _ARABIC_CORRUPT_MAX:
        return PageType.DIGITAL_CORRUPTED, 0.90

    # 4. In-between ratio (0.3 – 0.5) — mixed content
    if has_images:
        # Mixed digital + scanned regions
        return PageType.MIXED, 0.80
    # No images but ratio is ambiguous — lean toward corrupted
    return PageType.MIXED, 0.65


# ---------------------------------------------------------------------------
# Whole-document classifier
# ---------------------------------------------------------------------------
def classify_document(
    pdf_path: str,
    pages: list[int] | None = None,
) -> list[PageClassification]:
    """Classify all (or selected) pages of a PDF document.

    Args:
        pdf_path: Filesystem path to the PDF.
        pages:    Optional list of **1-indexed** page numbers to classify.
                  ``None`` means all pages.

    Returns:
        A list of :class:`PageClassification` objects, one per requested page,
        ordered by page number.
    """
    # Determine page count first
    if fitz is None:
        raise ImportError(
            "PyMuPDF is required for PDF detection. Install it with: pip install pymupdf"
        )
    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()
    except Exception as exc:
        logger.error("Failed to open PDF '%s': %s", pdf_path, exc)
        return []

    if pages is None:
        page_nums = list(range(1, total_pages + 1))
    else:
        # Filter to valid page numbers and sort
        page_nums = sorted(
            p for p in pages if 1 <= p <= total_pages
        )
        skipped = set(pages) - set(page_nums)
        if skipped:
            logger.warning(
                "Skipping out-of-range page(s) %s (document has %d page(s))",
                sorted(skipped),
                total_pages,
            )

    results: list[PageClassification] = []
    for pn in page_nums:
        try:
            classification = classify_page(pdf_path, pn)
        except Exception as exc:
            logger.error("Error classifying page %d: %s", pn, exc)
            classification = PageClassification(
                page_num=pn,
                page_type=PageType.SCANNED,
                raw_text=None,
                confidence=0.0,
                has_images=False,
            )
        results.append(classification)

    return results
