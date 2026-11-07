"""
Layer 1: Direct text extraction from digital Arabic PDFs.

Attempts extraction via qalam (preferred for Arabic) with automatic
fallback to PyMuPDF + arabic-reshaper/bidi post-processing.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Qalam-based extraction
# ---------------------------------------------------------------------------

def extract_with_qalam(
    pdf_path: str,
    page_num: int | None = None,
) -> list[tuple[int, str | None]]:
    """Extract text from a PDF using the *qalam* library (Arabic-optimised).

    Args:
        pdf_path: Path to the PDF file.
        page_num: 0-based page index.  ``None`` means all pages.

    Returns:
        List of ``(page_number, text_or_None)`` tuples.

    Raises:
        ImportError: If *qalam* is not installed.
        FileNotFoundError: If *pdf_path* does not exist.
    """
    try:
        import qalam  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The 'qalam' package is required for Arabic-optimised extraction "
            "but is not installed.  Install it with:\n"
            "    pip install qalam"
        )

    pdf = Path(pdf_path)
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    results: list[tuple[int, str | None]] = []

    try:
        doc = qalam.Document(str(pdf))

        if page_num is not None:
            # Single page (page_num is 0-based in extractor interface, qalam.page is 1-based)
            try:
                page_obj = doc.page(page_num + 1)
                text: str | None = page_obj.text.strip() if page_obj.text else None
                results.append((page_num, text if text else None))
            except Exception as exc:
                logger.warning("qalam failed on page %d: %s", page_num, exc)
                results.append((page_num, None))
        else:
            # All pages
            for i, page_obj in enumerate(doc.pages):
                try:
                    text = page_obj.text.strip() if page_obj.text else None
                    results.append((i, text if text else None))
                except Exception as exc:
                    logger.warning("qalam failed on page %d: %s", i, exc)
                    results.append((i, None))

    except Exception as exc:
        logger.error("qalam could not process '%s': %s", pdf_path, exc)
        raise

    return results


# ---------------------------------------------------------------------------
# PyMuPDF-based extraction
# ---------------------------------------------------------------------------

def extract_with_pymupdf(
    pdf_path: str,
    page_num: int | None = None,
) -> list[tuple[int, str]]:
    """Extract text from a PDF using PyMuPDF (``fitz``).

    Args:
        pdf_path: Path to the PDF file.
        page_num: 0-based page index.  ``None`` means all pages.

    Returns:
        List of ``(page_number, text)`` tuples.

    Raises:
        ImportError: If *PyMuPDF* is not installed.
        FileNotFoundError: If *pdf_path* does not exist.
    """
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "The 'PyMuPDF' package is required for PDF text extraction "
                "but is not installed.  Install it with:\n"
                "    pip install PyMuPDF"
            )

    pdf = Path(pdf_path)
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    results: list[tuple[int, str]] = []

    try:
        doc = fitz.open(str(pdf))

        pages = [page_num] if page_num is not None else range(len(doc))

        for i in pages:
            page = doc.load_page(i)
            text: str = page.get_text("text")
            results.append((i, text))

        doc.close()
    except Exception as exc:
        logger.error("PyMuPDF could not process '%s': %s", pdf_path, exc)
        raise

    return results


# ---------------------------------------------------------------------------
# Arabic text post-processing
# ---------------------------------------------------------------------------

def fix_arabic_text(text: str) -> str:
    """Reshape and reorder Arabic text for correct display.

    Uses *arabic_reshaper* (glyph joining) and *python-bidi*
    (bidirectional reordering).  If either library is missing the
    original text is returned unchanged with a warning.

    Args:
        text: Raw extracted text that may have broken Arabic shaping.

    Returns:
        Corrected text, or the original text if reshaping libraries
        are unavailable.
    """
    try:
        import arabic_reshaper  # type: ignore[import-untyped]
    except ImportError:
        warnings.warn(
            "'arabic_reshaper' is not installed — Arabic text may appear "
            "incorrectly shaped.  Install it with:  pip install arabic-reshaper",
            stacklevel=2,
        )
        return text

    try:
        from bidi.algorithm import get_display  # type: ignore[import-untyped]
    except ImportError:
        warnings.warn(
            "'python-bidi' is not installed — Arabic text may appear in "
            "reversed order.  Install it with:  pip install python-bidi",
            stacklevel=2,
        )
        return text

    fixed_lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            fixed_lines.append(line)
            continue
        try:
            reshaped: str = arabic_reshaper.reshape(line)
            display: str = get_display(reshaped)
            fixed_lines.append(display)
        except Exception as exc:
            logger.debug("Could not reshape line: %s", exc)
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def extract_direct(
    pdf_path: str,
    page_num: int | None = None,
) -> list[tuple[int, str | None]]:
    """Extract text directly from a digital PDF (Layer 1).

    Strategy:
        1. Try *qalam* first (best Arabic support).
        2. On failure or unavailability, fall back to *PyMuPDF* with
           ``fix_arabic_text`` post-processing.

    Args:
        pdf_path: Path to the PDF file.
        page_num: 0-based page index.  ``None`` means all pages.

    Returns:
        List of ``(page_number, extracted_text_or_None)`` tuples.
    """
    # --- Attempt 1: qalam ------------------------------------------------
    try:
        results = extract_with_qalam(pdf_path, page_num)
        logger.info("Extraction via qalam succeeded for '%s'.", pdf_path)
        return results
    except ImportError:
        logger.info("qalam not installed — falling back to PyMuPDF.")
    except Exception as exc:
        logger.warning(
            "qalam extraction failed for '%s': %s — falling back to PyMuPDF.",
            pdf_path,
            exc,
        )

    # --- Attempt 2: PyMuPDF + arabic reshaping ----------------------------
    try:
        raw_results = extract_with_pymupdf(pdf_path, page_num)
    except ImportError:
        raise ImportError(
            "Neither 'qalam' nor 'PyMuPDF' is installed.  At least one is "
            "required for direct text extraction.  Install with:\n"
            "    pip install qalam        # Arabic-optimised\n"
            "    pip install PyMuPDF      # general-purpose fallback"
        )

    fixed_results: list[tuple[int, str | None]] = []
    for pg, text in raw_results:
        if text and text.strip():
            fixed_results.append((pg, fix_arabic_text(text)))
        else:
            fixed_results.append((pg, None))

    logger.info("Extraction via PyMuPDF succeeded for '%s'.", pdf_path)
    return fixed_results
