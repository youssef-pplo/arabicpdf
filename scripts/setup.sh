#!/usr/bin/env bash
# Development setup script for ArabicPDF (Zero system dependencies required!)
set -e

echo "=== ArabicPDF Development Setup ==="

# Check python version
PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Error: $PYTHON_BIN not found. Please install Python 3.10 or higher."
    exit 1
fi

echo "Creating virtual environment in .venv..."
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate

echo "Upgrading pip and installing arabicpdf in editable mode..."
pip install --upgrade pip
pip install -e ".[dev]"

echo ""
echo "=== ArabicPDF is ready! ==="
echo "Activate environment: source .venv/bin/activate"
echo "Run tests: python -m unittest discover -s tests -v"
echo "Convert a PDF: arabicpdf convert document.pdf"
