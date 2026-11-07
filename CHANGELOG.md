# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-11-07

### Added
- **Core Pipeline**:
  - `to_text()`: One-line function to convert any Arabic PDF into clean text.
  - `convert()`: Full conversion pipeline returning rich `ConversionResult` with per-page confidence and extraction method metadata.
- **Page-Level Classifier**:
  - Automatic detection of `DIGITAL_CLEAN`, `DIGITAL_CORRUPTED`, `SCANNED`, and `BLANK` pages.
  - High-accuracy heuristics using Arabic Unicode character ratios and Shannon character entropy.
- **Direct Extraction Engine**:
  - Native integration with Rust-powered `qalam` engine for 100% accurate digital Arabic PDF extraction with intact RTL logical order and ligatures.
  - Automatic fallback to PyMuPDF with `arabic-reshaper` and `python-bidi`.
- **Image Preprocessing**:
  - OpenCV-based pipeline tuned specifically for Arabic script.
  - Sub-degree deskewing (<0.5° precision).
  - Fast non-local means speckle denoising to eliminate phantom Arabic dots.
  - Sauvola adaptive binarization for uneven lighting and yellowed pages.
  - Custom sharpening kernel to preserve delicate Arabic dots (*nuqat*).
- **Dual OCR Engine & Consensus Voting**:
  - Pluggable OCR architecture supporting Tesseract (`ara`) and PaddleOCR (Arabic mobile models).
  - Sequence alignment (`difflib.SequenceMatcher`) algorithm to merge independent OCR readings.
- **Linguistic Dot-Confusion Resolution**:
  - Dictionary-based resolver arbitrating common Arabic dot confusion groups (`ب/ت/ث/ن/ي`, `ج/ح/خ`, `د/ذ`, `ر/ز`, `س/ش`, `ص/ض`, `ط/ظ`, `ع/غ`).
  - Automated fixing of `ة` vs `ه` (*ta marbuta* / *ha*) and *alef* variants (`أ/إ/آ/ا`).
- **Post-Processing & Normalization**:
  - Full Unicode NFC normalization, tatweel (*kashida*) removal, optional diacritic stripping, and whitespace cleanup.
- **CLI Tool**:
  - `arabicpdf convert <file.pdf>` with options for `--pages`, `--mode`, `--dpi`, `--strip-diacritics`, `--output`, and `--details`.
  - `arabicpdf inspect <file.pdf>` for inspecting page classifications without running OCR.
- **Testing & Packaging**:
  - Complete test suite with 23 unit tests covering utils, consensus, corrector, postprocessor, and end-to-end extraction.
  - Automated setup script for Linux and macOS (`scripts/setup.sh`).
