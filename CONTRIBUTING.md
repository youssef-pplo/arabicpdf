# Contributing to ArabicPDF

Thank you for your interest in improving **ArabicPDF**! We welcome contributions of all kinds: bug fixes, documentation improvements, new features, and benchmark evaluations.

---

## 🛠️ Development Setup

### 1. Clone the repository
```bash
git clone https://github.com/youssef-pplo/arabicpdf.git
cd arabicpdf
```

### 2. Set up a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install in editable mode with development dependencies
```bash
pip install --upgrade pip
pip install -e ".[dev,paddle]"
```

### 4. Install system dependencies (Tesseract & Poppler)
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-ara hunspell-ar poppler-utils

# macOS
brew install tesseract tesseract-lang poppler hunspell
```

---

## 🧪 Running Tests

Run the test suite using Python's built-in `unittest` runner or `pytest`:

```bash
# Using unittest
python3 -m unittest discover -s tests -v

# Using pytest
pytest tests/ -v
```

---

## 📐 Project Structure

```
arabicpdf/
├── arabicpdf/
│   ├── __init__.py         # Public exports (to_text, convert, Config, Mode)
│   ├── core.py             # Pipeline orchestrator
│   ├── config.py           # Configuration models (Pydantic + dataclass fallback)
│   ├── detector.py         # PDF type classifier
│   ├── extractor.py        # Direct extraction (Qalam + PyMuPDF)
│   ├── preprocessor.py     # OpenCV image enhancement
│   ├── ocr.py              # OCR engine wrappers (Tesseract, PaddleOCR)
│   ├── consensus.py        # Sequence alignment & consensus voting
│   ├── corrector.py        # Hunspell & dot confusion resolver
│   ├── postprocessor.py    # Unicode normalization & cleanup
│   ├── arabic_utils.py     # Unicode ranges, Levenshtein, tokenization
│   └── cli.py              # Command-line entry point
├── tests/                  # Unit and integration test suite
├── scripts/                # Installation scripts
└── pyproject.toml          # Packaging specification
```

---

## 📝 Pull Request Guidelines

1. **Keep it focused**: Each pull request should address a single bug or feature.
2. **Add tests**: Any bug fix or new capability must include corresponding tests in `tests/`.
3. **Preserve lightness**: Avoid adding heavy mandatory runtime dependencies. Prefer smart algorithms or optional dependency extras.
4. **Follow PEP 8**: Ensure your code is clean, readable, and properly type-hinted.

---

## 📜 Code of Conduct

Please treat everyone in the community with respect, kindness, and professionalism.
