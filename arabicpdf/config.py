"""Configuration for ArabicPDF pipeline.

Uses Pydantic when installed, with graceful fallback to standard library
dataclasses for ultra-lightweight environments.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Any

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    from dataclasses import dataclass, field


class Mode(str, Enum):
    """Processing mode — trades speed for accuracy."""

    FAST = "fast"  # Qalam only (digital) or single OCR (scanned)
    BALANCED = "balanced"  # Dual OCR + consensus + correction
    MAX_ACCURACY = "max_accuracy"  # All available engines + full correction


class PageType(str, Enum):
    """Classification of a PDF page."""

    DIGITAL_CLEAN = "digital_clean"  # Valid extractable Arabic text
    DIGITAL_CORRUPTED = "digital_corrupted"  # Text layer exists but garbled
    SCANNED = "scanned"  # Image-only, no text layer
    MIXED = "mixed"  # Some regions digital, some scanned


if HAS_PYDANTIC:
    class Config(BaseModel):
        """Pipeline configuration."""

        mode: Mode = Field(default=Mode.BALANCED, description="Processing mode")
        pages: Optional[list[int]] = Field(
            default=None, description="Pages to process (1-indexed). None = all."
        )
        preserve_diacritics: bool = Field(
            default=True, description="Preserve tashkeel/harakat"
        )
        normalize_alef: bool = Field(
            default=False, description="Normalize أ/إ/آ → ا"
        )
        fix_spelling: bool = Field(
            default=True, description="Apply Hunspell spell correction"
        )
        fix_dot_confusion: bool = Field(
            default=True, description="Apply Arabic dot-confusion resolver"
        )
        render_dpi: int = Field(
            default=300,
            ge=150,
            le=1200,
            description="DPI for rendering PDF pages to images",
        )
        arabic_threshold: float = Field(
            default=0.6,
            ge=0.0,
            le=1.0,
            description="Minimum ratio of Arabic chars to consider text valid",
        )
        correction_max_distance: int = Field(
            default=2,
            ge=1,
            le=5,
            description="Max Levenshtein distance for spell corrections",
        )

    class PageResult(BaseModel):
        """Result for a single page."""

        number: int
        text: str
        method: str  # "direct", "ocr", "ocr+corrected"
        confidence: float  # 0.0 to 1.0
        page_type: PageType
        flags: list[str] = Field(default_factory=list)

    class ConversionResult(BaseModel):
        """Full conversion result."""

        text: str
        pages: list[PageResult]
        confidence: float  # Average across pages

        @property
        def page_count(self) -> int:
            return len(self.pages)

else:
    @dataclass
    class Config:
        """Pipeline configuration (dataclass fallback)."""

        mode: Mode = Mode.BALANCED
        pages: Optional[list[int]] = None
        preserve_diacritics: bool = True
        normalize_alef: bool = False
        fix_spelling: bool = True
        fix_dot_confusion: bool = True
        render_dpi: int = 300
        arabic_threshold: float = 0.6
        correction_max_distance: int = 2

    @dataclass
    class PageResult:
        """Result for a single page (dataclass fallback)."""

        number: int
        text: str
        method: str
        confidence: float
        page_type: PageType
        flags: list[str] = field(default_factory=list)

    @dataclass
    class ConversionResult:
        """Full conversion result (dataclass fallback)."""

        text: str
        pages: list[PageResult]
        confidence: float

        @property
        def page_count(self) -> int:
            return len(self.pages)
