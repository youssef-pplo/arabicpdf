"""
Arabic text utilities for ArabicPDF.

Provides Unicode-aware helpers for detecting, analysing, and tokenising
Arabic text. All functions are pure-Python with no external dependencies.

Key capabilities
─────────────────
• Unicode range checks (is_arabic_char, is_arabic_word, is_valid_arabic)
• Text analysis (arabic_char_ratio, count_arabic_words, has_diacritics, char_entropy)
• Dot-confusion data (DOT_GROUPS, CONFUSION_MAP, variant sets)
• Levenshtein edit-distance (no external deps)
• Arabic-aware tokenisation
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Final

# ──────────────────────────────────────────────────────────────────────
# 1. Unicode range checks
# ──────────────────────────────────────────────────────────────────────

#: Compiled pattern matching the main Arabic Unicode blocks:
#:   U+0600–U+06FF  Arabic
#:   U+0750–U+077F  Arabic Supplement
#:   U+FB50–U+FDFF  Arabic Presentation Forms-A
#:   U+FE70–U+FEFF  Arabic Presentation Forms-B
ARABIC_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def is_arabic_char(char: str) -> bool:
    """Check whether a single character falls within Arabic Unicode ranges.

    Parameters
    ----------
    char:
        A single character string.

    Returns
    -------
    bool
        ``True`` if the character is in one of the Arabic Unicode blocks.

    Raises
    ------
    ValueError
        If *char* is not exactly one character long.
    """
    if len(char) != 1:
        raise ValueError(f"Expected a single character, got {len(char)} characters")
    return ARABIC_PATTERN.match(char) is not None


def is_arabic_word(word: str) -> bool:
    """Check whether a word is *predominantly* Arabic.

    A word is considered Arabic when more than half of its non-whitespace
    characters belong to Arabic Unicode ranges.

    Parameters
    ----------
    word:
        The word to test.

    Returns
    -------
    bool
        ``True`` if the word is predominantly Arabic.
    """
    if not word:
        return False
    chars = [ch for ch in word if not ch.isspace()]
    if not chars:
        return False
    arabic_count = sum(1 for ch in chars if ARABIC_PATTERN.match(ch))
    return arabic_count > len(chars) / 2


def is_valid_arabic(text: str, threshold: float = 0.6) -> bool:
    """Decide whether *text* is mostly valid Arabic.

    Computes the ratio of Arabic characters to total non-space characters
    and checks that it meets or exceeds *threshold*.

    Parameters
    ----------
    text:
        The text to evaluate.
    threshold:
        Minimum ratio of Arabic characters (default ``0.6``).

    Returns
    -------
    bool
        ``True`` if the Arabic-character ratio ≥ *threshold*.
    """
    return arabic_char_ratio(text) >= threshold


# ──────────────────────────────────────────────────────────────────────
# 2. Arabic text analysis
# ──────────────────────────────────────────────────────────────────────

def arabic_char_ratio(text: str) -> float:
    """Return the ratio of Arabic characters to total alphabetic letters.

    Punctuation (e.g. leader dots in table of contents) and digits
    do not penalise the ratio.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    float
        A value in [0.0, 1.0].  Returns ``0.0`` for empty / whitespace-only
        text.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if letters:
        arabic_letters = sum(1 for ch in letters if ARABIC_PATTERN.match(ch))
        return arabic_letters / len(letters)

    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return 0.0
    arabic_count = sum(1 for ch in non_space if ARABIC_PATTERN.match(ch))
    return arabic_count / len(non_space)


def count_arabic_words(text: str) -> int:
    """Count the number of words in *text* that are predominantly Arabic.

    Words are split on whitespace via :func:`str.split`.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    int
        Number of Arabic words.
    """
    return sum(1 for word in text.split() if is_arabic_word(word))


#: Arabic diacritical marks (tashkeel), U+064B – U+0652.
_DIACRITICS_PATTERN: Final[re.Pattern[str]] = re.compile(r"[\u064B-\u0652]")


def has_diacritics(text: str) -> bool:
    """Check whether *text* contains any Arabic diacritics (tashkeel).

    The diacritics checked are U+064B (fathatan) through U+0652 (sukun).

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    bool
        ``True`` if at least one diacritical mark is present.
    """
    return _DIACRITICS_PATTERN.search(text) is not None


def char_entropy(text: str) -> float:
    """Compute the Shannon entropy of the character distribution in *text*.

    Garbled / incorrectly-decoded text tends to have abnormally high or
    low entropy compared to natural Arabic.

    Parameters
    ----------
    text:
        Input text (whitespace is included in the distribution).

    Returns
    -------
    float
        Entropy in bits.  Returns ``0.0`` for empty text.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy


# ──────────────────────────────────────────────────────────────────────
# 3. Arabic dot-confusion data
# ──────────────────────────────────────────────────────────────────────

#: Groups of Arabic letters that share the same base shape and differ
#: only in the number / placement of dots.
DOT_GROUPS: Final[list[str]] = [
    "بتثنی",   # ba / ta / tha / nun / ya (Farsi)
    "جحخ",      # jim / ḥa / kha
    "دذ",       # dal / dhal
    "رز",       # ra / zay
    "سش",       # sin / shin
    "صض",       # ṣad / ḍad
    "طظ",       # ṭa / ẓa
    "عغ",       # ʿain / ghain
    "فق",       # fa / qaf
]

#: Map each Arabic letter to the set of letters it can be confused with
#: (same base shape, different dots).  Built from :data:`DOT_GROUPS`.
CONFUSION_MAP: Final[dict[str, set[str]]] = {}
for _group in DOT_GROUPS:
    for _char in _group:
        CONFUSION_MAP[_char] = set(_group) - {_char}

#: Common alef variants that may need normalisation.
ALEF_VARIANTS: Final[str] = "اأإآٱ"

#: Ta marbuta and Ha — often confused in OCR output.
TA_MARBUTA_HA: Final[str] = "ةه"

#: Alef maqsura and Ya — often confused in OCR output.
ALEF_MAQSURA_YA: Final[str] = "ىي"


# ──────────────────────────────────────────────────────────────────────
# 4. Levenshtein distance
# ──────────────────────────────────────────────────────────────────────

def levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings.

    Uses the classic dynamic-programming algorithm with *O(min(m, n))*
    space (single-row optimisation).  No external dependencies.

    Parameters
    ----------
    s1:
        First string.
    s2:
        Second string.

    Returns
    -------
    int
        Minimum number of single-character edits (insertions, deletions,
        or substitutions) needed to transform *s1* into *s2*.
    """
    # Ensure s1 is the shorter string for space optimisation.
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    n = len(s1)
    m = len(s2)

    # Previous row of distances.
    previous: list[int] = list(range(n + 1))

    for j in range(1, m + 1):
        current: list[int] = [j] + [0] * n
        for i in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            current[i] = min(
                current[i - 1] + 1,       # insertion
                previous[i] + 1,           # deletion
                previous[i - 1] + cost,    # substitution
            )
        previous = current

    return previous[n]


# ──────────────────────────────────────────────────────────────────────
# 5. Text tokenisation
# ──────────────────────────────────────────────────────────────────────

#: Arabic-specific punctuation characters that should act as token
#: boundaries in addition to whitespace.
_ARABIC_PUNCTUATION: Final[str] = (
    "\u060C"   # Arabic comma ،
    "\u061B"   # Arabic semicolon ؛
    "\u061F"   # Arabic question mark ؟
    "\u06D4"   # Arabic full stop ۔
    "\u0640"   # tatweel (kashida) ـ
)

#: Pattern that splits on whitespace and Arabic / Latin punctuation,
#: keeping only non-empty tokens.
_TOKENIZE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[" + re.escape(_ARABIC_PUNCTUATION) + r"\s\.\,\;\:\!\?\"\'\(\)\[\]\{\}<>«»/\\]+"
)


def tokenize_arabic(text: str) -> list[str]:
    """Split *text* into word tokens, handling Arabic punctuation.

    Splits on whitespace as well as common Arabic and Latin punctuation.
    Empty tokens are discarded.

    Parameters
    ----------
    text:
        Input text (may contain mixed Arabic / Latin content).

    Returns
    -------
    list[str]
        Non-empty word tokens, in their original order.
    """
    return [tok for tok in _TOKENIZE_PATTERN.split(text) if tok]
