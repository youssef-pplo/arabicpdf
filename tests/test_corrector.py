"""Tests for corrector module (compatible with unittest and pytest)."""

import unittest
from unittest.mock import patch
from arabicpdf.corrector import (
    resolve_dot_confusion,
    fix_ta_marbuta,
    fix_alef_variants,
    correct_text,
)


class TestCorrector(unittest.TestCase):
    @patch("arabicpdf.corrector.hunspell_check")
    def test_fix_ta_marbuta_basic(self, mock_hunspell):
        mock_hunspell.side_effect = lambda word: word == "مدرسة"

        fixed = fix_ta_marbuta("مدرسه")
        self.assertEqual(fixed, "مدرسة")

    @patch("arabicpdf.corrector.hunspell_check")
    def test_resolve_dot_confusion_mocked(self, mock_hunspell):
        mock_hunspell.side_effect = lambda word: word in {"كتاب", "طالب"}

        fixed = resolve_dot_confusion("كتان")
        self.assertEqual(fixed, "كتاب")

    @patch("arabicpdf.corrector.hunspell_check")
    def test_correct_text_pipeline(self, mock_hunspell):
        valid_words = {"هذه", "مدرسة", "كبيرة"}
        mock_hunspell.side_effect = lambda word: word in valid_words

        text = "هذه مدرسه كبيرة"
        corrected, changes = correct_text(text, fix_dots=True, fix_spelling=True)
        self.assertIn("مدرسة", corrected)
        self.assertGreaterEqual(len(changes), 1)


if __name__ == "__main__":
    unittest.main()
