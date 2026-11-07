# 🪶 ArabicPDF

[![PyPI version](https://img.shields.io/pypi/v/arabicpdf.svg?color=blue)](https://pypi.org/project/arabicpdf/)
[![Python versions](https://img.shields.io/pypi/pyversions/arabicpdf.svg)](https://pypi.org/project/arabicpdf/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/youssef-pplo/arabicpdf/actions/workflows/ci.yml/badge.svg)](https://github.com/youssef-pplo/arabicpdf/actions)

**Convert any Arabic PDF to clean text with near-100% accuracy. Lightweight. 100% Local. No GPU required.**

---

## 🌟 Why ArabicPDF?

Extracting Arabic text from PDFs is notoriously tricky:
- **Digital PDFs** frequently suffer from corrupted `/ToUnicode` mappings, reversed Right-to-Left (RTL) words, or broken cursive ligatures (producing disconnected glyph soup).
- **Scanned PDFs** suffer from dot confusion (نُقَط) where minuscule speckles turn `ح` into `خ` or `ر` into `ز`, or faint printing merges `ب`/`ت`/`ث`/`ن`/`ي`.
- Existing solutions either demand massive **25GB+ deep learning models and high-end GPUs**, or use basic extractors that output gibberish.

**ArabicPDF solves this with smart algorithms rather than brute-force weight downloads:**
- **~150 MB total footprint:** Runs smoothly on low-end laptops, older machines, and CPU-only environments.
- **Zero Cloud APIs & Zero Telemetry:** 100% local, offline, private, and free forever.
- **Smart 4-Tier Pipeline:**
  1. **Page-Level Classifier:** Automatically identifies digital clean, corrupted text layers, or scanned image pages.
  2. **Direct Extraction (Qalam & PyMuPDF):** 100% accurate extraction on digital-born PDFs with proper RTL order.
  3. **Arabic-Tuned Image Preprocessing (OpenCV):** Precise deskewing (<0.5°), speckle noise removal, Sauvola adaptive binarization, and dot-sharpening filters.
  4. **Dual OCR Engine Consensus:** Tesseract and PaddleOCR run independently; their outputs are aligned word-for-word using sequence alignment.
  5. **Linguistic Dot-Confusion Resolver:** Automatically arbitrates character confusions (ب/ت/ث/ن/ي, ج/ح/خ, د/ذ, ر/ز, س/ش, ص/ض, ط/ظ, ع/غ) against an Arabic dictionary.

---

## 🚀 Quick Start

### 1. System Requirements

ArabicPDF uses lightweight system libraries for OCR and spell-checking:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-ara hunspell-ar poppler-utils

# macOS (Homebrew)
brew install tesseract tesseract-lang poppler hunspell

# Or simply run our automated setup script:
./scripts/setup.sh
```

### 2. Install ArabicPDF

```bash
pip install arabicpdf
```

---

## 💻 Python API

### One-line extraction
```python
import arabicpdf

# Extract text directly
text = arabicpdf.to_text("document.pdf")
print(text)
```

### Advanced usage with detailed metadata
```python
import arabicpdf
from arabicpdf import Config, Mode

config = Config(
    mode=Mode.BALANCED,         # "fast", "balanced", or "max_accuracy"
    pages=[1, 2, 3],            # Process specific pages (1-indexed)
    preserve_diacritics=True,   # Keep tashkeel (فَتْحَة، ضَمَّة...)
    normalize_alef=False,       # Set True to normalize أ/إ/آ → ا
    fix_spelling=True,          # Apply Arabic dictionary & dot correction
    render_dpi=300,             # 300 DPI optimal for Arabic dots
)

result = arabicpdf.convert("document.pdf", config=config)

print(f"Overall confidence: {result.confidence * 100:.1f}%")

for page in result.pages:
    print(f"\n--- Page {page.number} [{page.method}] (Confidence: {page.confidence * 100:.1f}%) ---")
    print(page.text)
    if page.flags:
        print(f"Notes: {page.flags}")
```

---

## 🖥️ Command Line Interface (CLI)

ArabicPDF comes with a fast CLI tool:

```bash
# Convert a PDF and print to terminal
arabicpdf convert document.pdf

# Save extracted text to a file
arabicpdf convert document.pdf -o output.txt

# Extract specific page range with custom DPI
arabicpdf convert document.pdf --pages 1-5 --dpi 400 -o output.txt

# Inspect PDF structure and classification without full OCR
arabicpdf inspect document.pdf
```

---

## ⚙️ Architecture & Accuracy Breakdown

| Document Type | Primary Method | Expected Accuracy | Footprint |
| :--- | :--- | :--- | :--- |
| **Clean Digital PDF** | Qalam / PyMuPDF + Reshaper | **100%** | ~5 MB |
| **Corrupted Encoding PDF** | Image Render → OpenCV → Dual OCR → Dot Resolver | **95% – 98%** | ~150 MB |
| **High-Quality Scan (300 DPI)** | OpenCV Preprocessing → Dual OCR Consensus → Dictionary | **94% – 97%** | ~150 MB |
| **Degraded / Faded Scan** | CLAHE Contrast → Adaptive Sauvola → Multi-Pass OCR | **90% – 94%** | ~150 MB |

### Optional Upgrade Tiers
If you have additional disk space or a GPU and want even higher accuracy for extreme edge cases:
- `pip install arabicpdf[tier2]` — adds **Surya OCR** (+2 GB) for complex multi-column newspaper layouts and tables.
- `pip install arabicpdf[tier3]` — adds **CAMeL Tools** (+500 MB) for full Arabic morphological and grammatical analysis.
- `pip install arabicpdf[tier4]` — adds **Qari-OCR** (+8 GB, requires GPU) for state-of-the-art Arabic Vision-Language Model processing.

---

## 🧪 Testing

To run the test suite:

```bash
python3 -m unittest discover -s tests -v
```

All core unit tests run instantly with zero dependencies.

---

## 📄 License

MIT License. Free for commercial and non-commercial use.
