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
# Built-in Arabic dictionary (zero system dependencies, pure Python)
# ---------------------------------------------------------------------------

import gzip
from pathlib import Path

_BUILTIN_WORDS: Optional[set[str]] = None


def _get_builtin_words() -> set[str]:
    """Lazily load the bundled Arabic dictionary (290,000+ words)."""
    global _BUILTIN_WORDS
    if _BUILTIN_WORDS is not None:
        return _BUILTIN_WORDS

    dict_path = Path(__file__).resolve().parent / "data" / "arabic_words.txt.gz"
    if dict_path.is_file():
        try:
            with gzip.open(dict_path, "rt", encoding="utf-8") as f:
                _BUILTIN_WORDS = {line.strip() for line in f if line.strip()}
            logger.debug(
                "Loaded %d words from built-in Arabic lexicon.",
                len(_BUILTIN_WORDS),
            )
            return _BUILTIN_WORDS
        except Exception as exc:
            logger.debug("Could not read built-in dictionary: %s", exc)

    _BUILTIN_WORDS = set()
    return _BUILTIN_WORDS


def is_known_arabic_word(word: str) -> bool:
    """Check if word is recognized in the built-in Arabic dictionary."""
    words = _get_builtin_words()
    if not words:
        checker = _get_hunspell()
        if checker is not None:
            try:
                return checker.spell(word)
            except Exception:
                return True
        return True

    if word in words:
        return True

    # Common Arabic attached prefixes
    for prefix in ("وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ك", "ل"):
        if word.startswith(prefix) and len(word) > len(prefix) + 1:
            if word[len(prefix) :] in words:
                return True

    # Common Arabic attached suffixes
    for suffix in ("ها", "هم", "هن", "نا", "كم", "كن", "ه", "ي", "ة"):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            if word[: -len(suffix)] in words:
                return True

    return False


# ---------------------------------------------------------------------------
# Hunspell wrapper (optional fallback if system dictionary is installed)
# ---------------------------------------------------------------------------

_hunspell_checker = None
_hunspell_available: Optional[bool] = None


def _get_hunspell():
    """Lazily load system Hunspell Arabic dictionary if installed."""
    global _hunspell_checker, _hunspell_available

    if _hunspell_available is not None:
        return _hunspell_checker

    try:
        import hunspell as _hunspell_mod

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
                logger.debug("Hunspell Arabic dictionary loaded from %s", dic)
                return _hunspell_checker
            except Exception:
                continue

        _hunspell_available = False
        return None

    except ImportError:
        _hunspell_available = False
        return None


def hunspell_check(word: str) -> bool:
    """Check if a word is valid according to the dictionary."""
    return is_known_arabic_word(word)


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
