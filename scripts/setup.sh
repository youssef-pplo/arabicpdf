#!/usr/bin/env bash
# System dependencies setup script for ArabicPDF
set -e

echo "=== ArabicPDF System Setup ==="

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v apt-get &> /dev/null; then
        echo "Detected Debian/Ubuntu system."
        echo "Installing Tesseract with Arabic support, Hunspell with Arabic dictionary, and Poppler..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-ara hunspell-ar poppler-utils libhunspell-dev
        echo "System packages installed successfully."
    elif command -v dnf &> /dev/null; then
        echo "Detected Fedora/RHEL system."
        sudo dnf install -y tesseract tesseract-langpack-ara hunspell-ar poppler-utils hunspell-devel
        echo "System packages installed successfully."
    elif command -v pacman &> /dev/null; then
        echo "Detected Arch Linux system."
        sudo pacman -S --noconfirm tesseract tesseract-data-ara poppler hunspell
        echo "System packages installed successfully."
    else
        echo "Unsupported Linux package manager. Please manually install: tesseract-ocr, tesseract-ocr-ara, hunspell-ar, poppler-utils"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &> /dev/null; then
        echo "Detected macOS system with Homebrew."
        brew install tesseract tesseract-lang poppler hunspell
        echo "Homebrew packages installed successfully."
    else
        echo "Homebrew not found. Please install Homebrew or manually install tesseract, poppler, hunspell."
    fi
else
    echo "Unsupported OS. For Windows, please install Tesseract with Arabic language pack and add it to your PATH."
fi

echo "=== Setup complete ==="
