"""Core pipeline orchestration for ArabicPDF."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None

from arabicpdf.config import (
    Config,
    ConversionResult,
    Mode,
    PageResult,
    PageType,
)
from arabicpdf.detector import classify_page
from arabicpdf.extractor import extract_direct
from arabicpdf.preprocessor import ArabicPreprocessor, render_pdf_page
from arabicpdf.ocr import get_available_engines, OCRResult
from arabicpdf.consensus import multi_engine_consensus
from arabicpdf.corrector import correct_text, hunspell_check
from arabicpdf.postprocessor import postprocess
from arabicpdf.arabic_utils import is_valid_arabic

logger = logging.getLogger(__name__)


def convert(
    pdf_path: str | Path,
    config: Optional[Config] = None,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ConversionResult:
    """Convert an Arabic PDF file into text with detailed metadata.

    Args:
        pdf_path: Path to the input PDF file.
        config: Optional configuration settings. If omitted, default
            balanced configuration is used.
        progress_callback: Optional callback receiving (completed_pages,
            total_selected_pages, current_stage). Called before slow stages
            and after each completed page. Conversion is silent by default.

    Returns:
        ConversionResult containing combined text, per-page results,
        and confidence metrics.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If specified pages are out of range.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if fitz is None:
        raise ImportError(
            "PyMuPDF is required to process PDFs. "
            "Install it with: pip install pymupdf"
        )

    cfg = config or Config()
    preprocessor = None  # Lazily instantiated when OCR is needed

    # Open PDF to get page count
    doc = fitz.open(str(path))
    total_pages = len(doc)
    doc.close()

    if total_pages == 0:
        if progress_callback is not None:
            progress_callback(0, 0, "No pages to process")
        return ConversionResult(text="", pages=[], confidence=1.0)

    # Determine pages to process (1-indexed for user)
    if cfg.pages is not None:
        selected_pages = []
        for p in cfg.pages:
            if p < 1 or p > total_pages:
                raise ValueError(
                    f"Page {p} is out of range for document with {total_pages} page(s)."
                )
            selected_pages.append(p)
    else:
        selected_pages = list(range(1, total_pages + 1))

    page_results: list[PageResult] = []

    def report(stage: str) -> None:
        if progress_callback is not None:
            progress_callback(len(page_results), len(selected_pages), stage)

    # Get available OCR engines
    report("Checking OCR engines")
    available_engines = get_available_engines()
    if not available_engines:
        logger.debug(
            "No local OCR engines available (Tesseract/PaddleOCR). "
            "Scanned or corrupted pages will rely on fallback direct extraction."
        )

    for page_num in selected_pages:
        page_idx = page_num - 1  # 0-indexed for internal libraries
        flags: list[str] = []

        # 1. Classify page
        report(f"Page {page_num}: classifying")
        classification = classify_page(str(path), page_num)
        page_type = classification.page_type

        page_text = ""
        method = "none"
        confidence = classification.confidence

        # 2. Try direct extraction first (Qalam excels even on corrupted digital PDFs)
        direct_success = False
        report(f"Page {page_num}: extracting text")
        try:
            direct_res = extract_direct(str(path), page_idx)
            if direct_res and direct_res[0][1]:
                candidate_text = direct_res[0][1].strip()
                if is_valid_arabic(candidate_text, threshold=cfg.arabic_threshold):
                    page_text = candidate_text
                    method = "direct"
                    confidence = max(0.95, classification.confidence)
                    direct_success = True
        except Exception as e:
            logger.debug("Direct extraction failed on page %d: %s", page_num, e)

        # 2b. Check if this is an intentional blank page (no text, no images)
        if not direct_success and not classification.has_images and not classification.raw_text:
            page_text = ""
            method = "blank"
            confidence = 1.0
            direct_success = True

        # 3. If direct extraction was not used or failed, run OCR pipeline
        if not direct_success:
            if not available_engines:
                # Fallback to raw text if any
                if classification.raw_text:
                    page_text = classification.raw_text
                    method = "direct_unverified"
                    confidence = 0.3
                    flags.append("No OCR engine available; using unverified text layer.")
                else:
                    page_text = ""
                    method = "failed"
                    confidence = 0.0
                    flags.append("No text layer and no OCR engine available.")
            else:
                try:
                    # Render high-resolution image
                    report(f"Page {page_num}: rendering for OCR")
                    raw_img = render_pdf_page(str(path), page_idx, dpi=cfg.render_dpi)
                    report(f"Page {page_num}: preprocessing image")
                    if preprocessor is None:
                        preprocessor = ArabicPreprocessor()
                    preprocessed_img = preprocessor.process(raw_img, strategy="default")

                    # Run OCR
                    ocr_results: list[OCRResult] = []
                    if cfg.mode == Mode.FAST:
                        # Use first available engine
                        report(f"Page {page_num}: OCR ({type(available_engines[0]).__name__})")
                        res = available_engines[0].ocr(preprocessed_img)
                        if res.text:
                            ocr_results.append(res)
                    else:
                        # Balanced / Max accuracy: run all available engines
                        for engine in available_engines:
                            report(f"Page {page_num}: OCR ({type(engine).__name__})")
                            try:
                                res = engine.ocr(preprocessed_img)
                                if res.text.strip():
                                    ocr_results.append(res)
                            except Exception as e:
                                logger.debug(
                                    "Engine %s failed on page %d: %s",
                                    getattr(engine, "__class__", type(engine)).__name__,
                                    page_num,
                                    e,
                                )

                    # If fast mode or only one result returned
                    if len(ocr_results) == 1:
                        page_text = ocr_results[0].text
                        confidence = ocr_results[0].confidence
                        method = f"ocr_{ocr_results[0].engine}"
                    elif len(ocr_results) > 1:
                        # Multi-engine consensus voting
                        report(f"Page {page_num}: combining OCR results")
                        engine_tuples = [
                            (res.text, res.confidence) for res in ocr_results
                        ]
                        merged_text, consensus_conf = multi_engine_consensus(
                            engine_tuples, check_valid_word=hunspell_check
                        )
                        page_text = merged_text
                        confidence = consensus_conf
                        engines_used = "+".join(r.engine for r in ocr_results)
                        method = f"ocr_consensus({engines_used})"
                    else:
                        page_text = ""
                        confidence = 0.0
                        method = "ocr_empty"
                        flags.append("OCR produced no text.")

                except Exception as e:
                    logger.error("OCR pipeline failed on page %d: %s", page_num, e)
                    page_text = classification.raw_text or ""
                    confidence = 0.2
                    method = "ocr_error_fallback"
                    flags.append(f"OCR error: {e}")

        # 4. Linguistic correction & spell checking
        if page_text and (cfg.fix_dot_confusion or cfg.fix_spelling):
            report(f"Page {page_num}: correcting spelling")
            try:
                corrected, changes = correct_text(
                    page_text,
                    fix_dots=cfg.fix_dot_confusion,
                    fix_spelling=cfg.fix_spelling,
                    max_distance=cfg.correction_max_distance,
                )
                if changes:
                    method += "+corrected"
                    for chg in changes[:5]:  # record first few changes
                        flags.append(
                            f"Corrected '{chg['original']}' -> '{chg['corrected']}' ({chg['reason']})"
                        )
                    if len(changes) > 5:
                        flags.append(f"+{len(changes) - 5} additional word corrections")
                page_text = corrected
            except Exception as e:
                logger.debug("Correction error on page %d: %s", page_num, e)

        # 5. Text post-processing (cleanup, normalization, tashkeel handling)
        report(f"Page {page_num}: cleaning text")
        final_text = postprocess(
            page_text,
            preserve_diacritics=cfg.preserve_diacritics,
            do_normalize_alef=cfg.normalize_alef,
        )

        page_results.append(
            PageResult(
                number=page_num,
                text=final_text,
                method=method,
                confidence=round(confidence, 4),
                page_type=page_type,
                flags=flags,
            )
        )
        report(f"Page {page_num}: complete")

    # Combine full document text
    combined_text = "\n\n".join(
        p.text for p in page_results if p.text.strip()
    )

    avg_confidence = (
        sum(p.confidence for p in page_results) / len(page_results)
        if page_results
        else 0.0
    )

    return ConversionResult(
        text=combined_text,
        pages=page_results,
        confidence=round(avg_confidence, 4),
    )


def to_text(
    pdf_path: str | Path,
    pages: Optional[list[int]] = None,
    mode: str | Mode = Mode.BALANCED,
    preserve_diacritics: bool = True,
    normalize_alef: bool = False,
    fix_spelling: bool = True,
    render_dpi: int = 300,
) -> str:
    """Convenience function to extract Arabic text directly from a PDF.

    Args:
        pdf_path: Path to the input PDF file.
        pages: Optional list of 1-indexed pages to convert.
        mode: Processing mode ("fast", "balanced", or "max_accuracy").
        preserve_diacritics: Whether to retain tashkeel/harakat.
        normalize_alef: Whether to map all alef variants to bare alef (ا).
        fix_spelling: Whether to run dictionary & dot confusion correction.
        render_dpi: DPI resolution for rendering scanned pages.

    Returns:
        The extracted Arabic text string.
    """
    if isinstance(mode, str):
        mode = Mode(mode)

    cfg = Config(
        mode=mode,
        pages=pages,
        preserve_diacritics=preserve_diacritics,
        normalize_alef=normalize_alef,
        fix_spelling=fix_spelling,
        render_dpi=render_dpi,
    )
    result = convert(pdf_path, config=cfg)
    return result.text
