"""Post-processing: Arabic text cleanup and normalization."""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Arabic reshaper + BiDi (optional dependencies)
# ---------------------------------------------------------------------------

_reshaper_available: bool | None = None
_bidi_available: bool | None = None


def _try_import_reshaper():
    global _reshaper_available
    if _reshaper_available is not None:
        return _reshaper_available
    try:
        import arabic_reshaper  # noqa: F401

        _reshaper_available = True
    except ImportError:
        _reshaper_available = False
        logger.info("arabic-reshaper not installed; reshaping disabled.")
    return _reshaper_available


def _try_import_bidi():
    global _bidi_available
    if _bidi_available is not None:
        return _bidi_available
    try:
        from bidi.algorithm import get_display  # noqa: F401

        _bidi_available = True
    except ImportError:
        _bidi_available = False
        logger.info("python-bidi not installed; BiDi reordering disabled.")
    return _bidi_available


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------

# Tatweel / kashida — used for justification, not meaningful
TATWEEL = "\u0640"

# Arabic diacritics (tashkeel)
DIACRITICS = re.compile(
    "[\u064B-\u065F\u0610-\u061A\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)

# Zero-width characters that sometimes appear
ZERO_WIDTH = re.compile("[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]")

# Multiple spaces / weird whitespace
MULTI_SPACE = re.compile(r"[ \t]+")
MULTI_NEWLINE = re.compile(r"\n{3,}")


def normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalization."""
    return unicodedata.normalize("NFC", text)


def remove_tatweel(text: str) -> str:
    """Remove tatweel/kashida characters (ـ)."""
    return text.replace(TATWEEL, "")


def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritics (tashkeel)."""
    return DIACRITICS.sub("", text)


def remove_zero_width(text: str) -> str:
    """Remove zero-width and bidirectional control characters."""
    return ZERO_WIDTH.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces and excessive newlines."""
    text = MULTI_SPACE.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    # Trim each line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)


def normalize_alef(text: str) -> str:
    """Normalize all alef variants (أ إ آ ٱ) to bare alef (ا)."""
    text = text.replace("\u0623", "\u0627")  # أ → ا
    text = text.replace("\u0625", "\u0627")  # إ → ا
    text = text.replace("\u0622", "\u0627")  # آ → ا
    text = text.replace("\u0671", "\u0627")  # ٱ → ا
    return text


# ---------------------------------------------------------------------------
# Arabic reshaping & BiDi
# ---------------------------------------------------------------------------


def reshape_arabic(text: str) -> str:
    """Reshape Arabic text — reconnect disconnected glyphs."""
    if not _try_import_reshaper():
        return text

    import arabic_reshaper

    lines = text.split("\n")
    reshaped = []
    for line in lines:
        try:
            reshaped.append(arabic_reshaper.reshape(line))
        except Exception:
            reshaped.append(line)
    return "\n".join(reshaped)


def apply_bidi(text: str) -> str:
    """Apply BiDi algorithm for correct RTL display ordering."""
    if not _try_import_bidi():
        return text

    from bidi.algorithm import get_display

    lines = text.split("\n")
    bidi_lines = []
    for line in lines:
        try:
            bidi_lines.append(get_display(line))
        except Exception:
            bidi_lines.append(line)
    return "\n".join(bidi_lines)


# ---------------------------------------------------------------------------
# Full post-processing pipeline
# ---------------------------------------------------------------------------


def postprocess(
    text: str,
    preserve_diacritics: bool = True,
    do_normalize_alef: bool = False,
    reshape: bool = False,
    bidi: bool = False,
) -> str:
    """Full post-processing pipeline for extracted Arabic text.

    Args:
        text: Raw extracted text.
        preserve_diacritics: If False, strip tashkeel/harakat.
        do_normalize_alef: If True, normalize أ/إ/آ → ا.
        reshape: If True, apply Arabic reshaping (for display).
        bidi: If True, apply BiDi reordering (for display).

    Returns:
        Cleaned text.
    """
    if not text:
        return text

    # 1. Unicode normalization (NFC)
    text = normalize_unicode(text)

    # 2. Remove zero-width and control characters
    text = remove_zero_width(text)

    # 3. Remove tatweel (kashida)
    text = remove_tatweel(text)

    # 4. Optionally strip diacritics
    if not preserve_diacritics:
        text = remove_diacritics(text)

    # 5. Optionally normalize alef
    if do_normalize_alef:
        text = normalize_alef(text)

    # 6. Normalize whitespace
    text = normalize_whitespace(text)

    # 7. Optionally reshape + bidi (for display purposes)
    if reshape:
        text = reshape_arabic(text)
    if bidi:
        text = apply_bidi(text)

    return text.strip()
