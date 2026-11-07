"""Dual / multi-engine OCR consensus voting algorithm.

When two or more OCR engines produce different readings of the same Arabic
text, this module aligns the outputs word-by-word and picks the most
plausible variant using Arabic-aware heuristics (dot-confusion resolution,
valid-word voting, length preference).
"""

from __future__ import annotations

from difflib import SequenceMatcher
import logging
from typing import Callable, Optional

from arabicpdf.arabic_utils import is_arabic_word, CONFUSION_MAP, levenshtein

logger = logging.getLogger(__name__)

# Type alias for the optional word-validity callback (e.g. Hunspell check).
_WordChecker = Optional[Callable[[str], bool]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid(word: str, check: _WordChecker) -> bool:
    """Return *True* if *word* is considered a valid Arabic word.

    Uses the caller-supplied *check* callback when available, falling back
    to the lightweight heuristic :func:`is_arabic_word`.
    """
    if check is not None:
        return check(word)
    return is_arabic_word(word)


def _count_valid(words: list[str], check: _WordChecker) -> int:
    """Count how many entries in *words* pass the validity test."""
    return sum(1 for w in words if _valid(w, check))


def _chars_differ_by_one(word_a: str, word_b: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Check whether two words differ in exactly one character position.

    Returns ``(True, char_a, char_b)`` when they do, where *char_a* /
    *char_b* are the differing characters.  Otherwise ``(False, None, None)``.
    """
    if len(word_a) != len(word_b):
        return False, None, None

    diffs: list[tuple[str, str]] = []
    for ca, cb in zip(word_a, word_b):
        if ca != cb:
            diffs.append((ca, cb))
            if len(diffs) > 1:
                return False, None, None

    if len(diffs) == 1:
        return True, diffs[0][0], diffs[0][1]
    return False, None, None


# ---------------------------------------------------------------------------
# Core consensus
# ---------------------------------------------------------------------------

def resolve_arabic_confusion(
    word_a: str,
    word_b: str,
    check_valid_word: _WordChecker = None,
) -> str:
    """Pick the better word when two engines disagree on a single token.

    Strategy
    --------
    1. If the words differ by exactly one character *and* that character pair
       appears in :data:`CONFUSION_MAP` (Arabic dot-confusion), prefer the
       variant that passes the validity check.
    2. If both are valid or neither is valid, prefer the **longer** word –
       OCR engines sometimes drop dots, producing a shorter (incorrect) form.
    3. As a final fallback, return *word_a* (the first engine's output).
    """
    single, char_a, char_b = _chars_differ_by_one(word_a, word_b)

    if single and char_a is not None and char_b is not None:
        # Check if the differing characters form a known confusion pair.
        is_confusion = (
            char_b in CONFUSION_MAP.get(char_a, set())
            or char_a in CONFUSION_MAP.get(char_b, set())
        )
        if is_confusion:
            va = _valid(word_a, check_valid_word)
            vb = _valid(word_b, check_valid_word)
            if va and not vb:
                return word_a
            if vb and not va:
                return word_b
            # Both valid or both invalid – fall through to length heuristic.

    # Length heuristic: longer word is usually more correct (dots preserved).
    if len(word_b) > len(word_a):
        return word_b
    return word_a


def align_and_merge(
    text_a: str,
    text_b: str,
    check_valid_word: _WordChecker = None,
) -> tuple[str, float]:
    """Align two OCR outputs word-by-word and merge them via majority voting.

    Parameters
    ----------
    text_a:
        OCR output from the first engine.
    text_b:
        OCR output from the second engine.
    check_valid_word:
        Optional callback ``(word) -> bool`` used to decide whether a word
        is valid Arabic.  Falls back to :func:`is_arabic_word`.

    Returns
    -------
    tuple[str, float]
        ``(merged_text, confidence)`` where *confidence* ∈ [0, 1] reflects
        the fraction of words the two engines agreed on.
    """
    words_a = text_a.split()
    words_b = text_b.split()

    if not words_a and not words_b:
        return "", 1.0
    if not words_a:
        return text_b, 0.0
    if not words_b:
        return text_a, 0.0

    matcher = SequenceMatcher(None, words_a, words_b)
    merged: list[str] = []
    agreed = 0
    total = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Both engines agree – high confidence.
            segment = words_a[i1:i2]
            merged.extend(segment)
            agreed += len(segment)
            total += len(segment)

        elif tag == "replace":
            # Engines disagree on this span – pick the better version.
            chunk_a = words_a[i1:i2]
            chunk_b = words_b[j1:j2]

            # If the chunks are the same length we can do per-word confusion
            # resolution; otherwise fall back to counting valid words.
            if len(chunk_a) == len(chunk_b):
                for wa, wb in zip(chunk_a, chunk_b):
                    best = resolve_arabic_confusion(wa, wb, check_valid_word)
                    merged.append(best)
                total += len(chunk_a)
            else:
                valid_a = _count_valid(chunk_a, check_valid_word)
                valid_b = _count_valid(chunk_b, check_valid_word)
                if valid_b > valid_a:
                    merged.extend(chunk_b)
                    total += len(chunk_b)
                else:
                    merged.extend(chunk_a)
                    total += len(chunk_a)

        elif tag == "insert":
            # Words present only in B – include if they look like valid Arabic.
            chunk = words_b[j1:j2]
            valid_count = _count_valid(chunk, check_valid_word)
            if valid_count > 0:
                merged.extend(chunk)
                total += len(chunk)
            else:
                logger.debug("Dropping insert-only chunk (no valid words): %s", chunk)
                total += len(chunk)  # still count for confidence denominator

        elif tag == "delete":
            # Words present only in A – include if they look like valid Arabic.
            chunk = words_a[i1:i2]
            valid_count = _count_valid(chunk, check_valid_word)
            if valid_count > 0:
                merged.extend(chunk)
                total += len(chunk)
            else:
                logger.debug("Dropping delete-only chunk (no valid words): %s", chunk)
                total += len(chunk)

    confidence = agreed / total if total > 0 else 0.0
    merged_text = " ".join(merged)
    logger.debug(
        "align_and_merge: agreed=%d total=%d confidence=%.2f",
        agreed, total, confidence,
    )
    return merged_text, confidence


# ---------------------------------------------------------------------------
# Multi-engine consensus
# ---------------------------------------------------------------------------

def multi_engine_consensus(
    results: list[tuple[str, float]],
    check_valid_word: _WordChecker = None,
) -> tuple[str, float]:
    """Merge outputs from 2+ OCR engines into a single consensus text.

    Parameters
    ----------
    results:
        A list of ``(text, engine_confidence)`` tuples, one per OCR engine.
        *engine_confidence* is a 0–1 score indicating how confident the
        engine itself was in its output.
    check_valid_word:
        Optional word-validity callback forwarded to :func:`align_and_merge`.

    Returns
    -------
    tuple[str, float]
        ``(consensus_text, overall_confidence)``
    """
    if not results:
        return "", 0.0

    if len(results) == 1:
        return results[0]

    # Sort by engine confidence (descending) so the best engine is the
    # initial anchor.  This biases pairwise merges towards higher-quality
    # sources without ignoring lower-confidence engines entirely.
    sorted_results = sorted(results, key=lambda r: r[1], reverse=True)

    current_text, current_conf = sorted_results[0]
    cumulative_weight = current_conf

    for text, engine_conf in sorted_results[1:]:
        merged_text, merge_conf = align_and_merge(
            current_text, text, check_valid_word,
        )

        # Weighted combination: blend the merge confidence with the
        # running confidence, weighting by engine scores.
        new_weight = cumulative_weight + engine_conf
        if new_weight > 0:
            current_conf = (
                (current_conf * cumulative_weight + merge_conf * engine_conf)
                / new_weight
            )
        else:
            current_conf = merge_conf

        cumulative_weight = new_weight
        current_text = merged_text

    logger.info(
        "multi_engine_consensus: %d engines → confidence=%.2f",
        len(results), current_conf,
    )
    return current_text, current_conf
