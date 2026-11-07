"""Arabic spell correction and dot-confusion resolution."""

from __future__ import annotations

import logging
from typing import Optional

from arabicpdf.arabic_utils import (
    CONFUSION_MAP,
    DOT_GROUPS,
    TA_MARBUTA_HA,
    ALEF_VARIANTS,
    is_arabic_word,
    levenshtein,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hunspell wrapper (optional dependency)
# ---------------------------------------------------------------------------

_hunspell_checker = None
_hunspell_available: Optional[bool] = None


def _get_hunspell():
    """Lazily load Hunspell Arabic dictionary."""
    global _hunspell_checker, _hunspell_available

    if _hunspell_available is not None:
        return _hunspell_checker

    try:
        import hunspell as _hunspell_mod

        # Try common paths for Arabic dictionary
        dict_paths = [
            ("/usr/share/hunspell/ar.aff", "/usr/share/hunspell/ar.dic"),
            ("/usr/share/myspell/ar.aff", "/usr/share/myspell/ar.dic"),
            (
                "/usr/share/hunspell/dicts/ar.aff",
                "/usr/share/hunspell/dicts/ar.dic",
            ),
        ]

        for aff, dic in dict_paths:
            try:
                _hunspell_checker = _hunspell_mod.HunSpell(dic, aff)
                _hunspell_available = True
                logger.info("Hunspell Arabic dictionary loaded from %s", dic)
                return _hunspell_checker
            except Exception:
                continue

        logger.warning(
            "Hunspell is installed but no Arabic dictionary found. "
            "Install it: sudo apt install hunspell-ar"
        )
        _hunspell_available = False
        return None

    except ImportError:
        logger.info(
            "pyhunspell not installed. Spell correction disabled. "
            'Install with: pip install "arabicpdf[hunspell]"'
        )
        _hunspell_available = False
        return None


def hunspell_check(word: str) -> bool:
    """Check if a word is valid according to Hunspell."""
    checker = _get_hunspell()
    if checker is None:
        return True  # If no checker, assume valid
    try:
        return checker.spell(word)
    except Exception:
        return True


def hunspell_suggest(word: str) -> list[str]:
    """Get spelling suggestions from Hunspell."""
    checker = _get_hunspell()
    if checker is None:
        return []
    try:
        return list(checker.suggest(word))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Dot-confusion resolver
# ---------------------------------------------------------------------------


def resolve_dot_confusion(word: str) -> Optional[str]:
    """Try swapping confused Arabic letters to find a valid word.

    80% of Arabic OCR errors are dot errors: ب↔ت↔ث↔ن↔ي etc.
    This function tries each confusion partner and checks if the
    result is a valid word in the dictionary.
    """
    if hunspell_check(word):
        return None  # Word is already valid

    # Try single-character swaps with confusion partners
    for i, char in enumerate(word):
        if char not in CONFUSION_MAP:
            continue
        for replacement in CONFUSION_MAP[char]:
            candidate = word[:i] + replacement + word[i + 1 :]
            if hunspell_check(candidate):
                logger.debug(
                    "Dot confusion fix: '%s' -> '%s' (pos %d: %s->%s)",
                    word,
                    candidate,
                    i,
                    char,
                    replacement,
                )
                return candidate

    # Try two-character swaps (less common but happens)
    for i, char_i in enumerate(word):
        if char_i not in CONFUSION_MAP:
            continue
        for j, char_j in enumerate(word):
            if j <= i or char_j not in CONFUSION_MAP:
                continue
            for rep_i in CONFUSION_MAP[char_i]:
                for rep_j in CONFUSION_MAP[char_j]:
                    candidate = (
                        word[:i]
                        + rep_i
                        + word[i + 1 : j]
                        + rep_j
                        + word[j + 1 :]
                    )
                    if hunspell_check(candidate):
                        logger.debug(
                            "Double dot fix: '%s' -> '%s'", word, candidate
                        )
                        return candidate

    return None


# ---------------------------------------------------------------------------
# Ta marbuta / Ha confusion
# ---------------------------------------------------------------------------


def fix_ta_marbuta(word: str) -> Optional[str]:
    """Fix ة↔ه confusion at word end.

    Common OCR error: ta marbuta (ة) misread as ha (ه) and vice versa.
    """
    if not word:
        return None

    if hunspell_check(word):
        return None

    if word[-1] == "ه":
        candidate = word[:-1] + "ة"
        if hunspell_check(candidate):
            return candidate
    elif word[-1] == "ة":
        candidate = word[:-1] + "ه"
        if hunspell_check(candidate):
            return candidate

    return None


# ---------------------------------------------------------------------------
# Alef variant normalization / correction
# ---------------------------------------------------------------------------


def fix_alef_variants(word: str) -> Optional[str]:
    """Try different alef variants (أ/إ/آ/ا) to find valid word."""
    if hunspell_check(word):
        return None

    alef_chars = set(ALEF_VARIANTS)

    for i, char in enumerate(word):
        if char not in alef_chars:
            continue
        for variant in ALEF_VARIANTS:
            if variant == char:
                continue
            candidate = word[:i] + variant + word[i + 1 :]
            if hunspell_check(candidate):
                return candidate

    return None


# ---------------------------------------------------------------------------
# Spell correction with Hunspell suggestions
# ---------------------------------------------------------------------------


def spell_correct(word: str, max_distance: int = 2) -> Optional[str]:
    """Correct a misspelled word using Hunspell suggestions.

    Only accepts corrections within `max_distance` edit distance
    to avoid replacing with unrelated words.
    """
    if hunspell_check(word):
        return None

    suggestions = hunspell_suggest(word)
    if not suggestions:
        return None

    # Find the closest suggestion
    best = None
    best_dist = max_distance + 1

    for suggestion in suggestions:
        dist = levenshtein(word, suggestion)
        if dist <= max_distance and dist < best_dist:
            best = suggestion
            best_dist = dist

    return best


# ---------------------------------------------------------------------------
# Full correction pipeline
# ---------------------------------------------------------------------------


def correct_text(
    text: str,
    fix_dots: bool = True,
    fix_spelling: bool = True,
    max_distance: int = 2,
) -> tuple[str, list[dict]]:
    """Apply all corrections to Arabic text.

    Returns (corrected_text, list_of_changes).
    Each change is a dict: {"original": str, "corrected": str, "reason": str}
    """
    words = text.split()
    corrected_words = []
    changes = []

    for word in words:
        if not is_arabic_word(word):
            corrected_words.append(word)
            continue

        original = word
        correction = None
        reason = ""

        # 1. Try dot confusion fix (most common Arabic OCR error)
        if fix_dots and not hunspell_check(word):
            correction = resolve_dot_confusion(word)
            if correction:
                reason = "dot_confusion"

        # 2. Try ta marbuta / ha fix
        if correction is None and not hunspell_check(word):
            correction = fix_ta_marbuta(word)
            if correction:
                reason = "ta_marbuta"

        # 3. Try alef variant fix
        if correction is None and not hunspell_check(word):
            correction = fix_alef_variants(word)
            if correction:
                reason = "alef_variant"

        # 4. General spell correction (last resort)
        if correction is None and fix_spelling and not hunspell_check(word):
            correction = spell_correct(word, max_distance=max_distance)
            if correction:
                reason = "spell_correction"

        if correction:
            corrected_words.append(correction)
            changes.append(
                {
                    "original": original,
                    "corrected": correction,
                    "reason": reason,
                }
            )
        else:
            corrected_words.append(word)

    return " ".join(corrected_words), changes
