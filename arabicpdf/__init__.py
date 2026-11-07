"""ArabicPDF — Convert any Arabic PDF to text. Lightweight. Local. Accurate."""

from __future__ import annotations

from arabicpdf.config import (
    Config,
    ConversionResult,
    Mode,
    PageResult,
    PageType,
)
from arabicpdf.core import convert, to_text

__version__ = "0.1.0"

__all__ = [
    "to_text",
    "convert",
    "Config",
    "Mode",
    "PageType",
    "PageResult",
    "ConversionResult",
    "__version__",
]
