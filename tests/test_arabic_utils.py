"""Tests for arabic_utils (compatible with unittest and pytest)."""

import unittest
from arabicpdf.arabic_utils import (
    is_arabic_char,
    is_arabic_word,
    is_valid_arabic,
    arabic_char_ratio,
    count_arabic_words,
    has_diacritics,
    char_entropy,
    levenshtein,
    tokenize_arabic,
    CONFUSION_MAP,
)


class TestArabicUtils(unittest.TestCase):
    def test_is_arabic_char(self):
        self.assertTrue(is_arabic_char("ك"))
        self.assertTrue(is_arabic_char("م"))
        self.assertFalse(is_arabic_char("a"))
        self.assertFalse(is_arabic_char("1"))
        self.assertFalse(is_arabic_char(" "))

    def test_is_arabic_word(self):
        self.assertTrue(is_arabic_word("مرحبا"))
        self.assertFalse(is_arabic_word("hello"))
        self.assertFalse(is_arabic_word("12345"))
        self.assertTrue(is_arabic_word("كتاب"))

    def test_is_valid_arabic(self):
        arabic_text = "بسم الله الرحمن الرحيم هذا نص عربي تجريبي."
        self.assertTrue(is_valid_arabic(arabic_text, threshold=0.6))

        mixed_text = "Hello world هذا تجربة 12345"
        self.assertFalse(is_valid_arabic(mixed_text, threshold=0.8))

        corrupted_text = "âêîôû abcdef ????"
        self.assertFalse(is_valid_arabic(corrupted_text, threshold=0.5))

    def test_arabic_char_ratio(self):
        pure_arabic = "مرحبا بالعالم"
        self.assertGreaterEqual(arabic_char_ratio(pure_arabic), 0.9)

        pure_english = "Hello world"
        self.assertEqual(arabic_char_ratio(pure_english), 0.0)

    def test_has_diacritics(self):
        with_tashkeel = "كَتَبَ الطَّالِبُ الدَّرْسَ"
        without_tashkeel = "كتب الطالب الدرس"
        self.assertTrue(has_diacritics(with_tashkeel))
        self.assertFalse(has_diacritics(without_tashkeel))

    def test_levenshtein(self):
        self.assertEqual(levenshtein("كتاب", "كتاب"), 0)
        self.assertEqual(levenshtein("كتاب", "كتان"), 1)
        self.assertEqual(levenshtein("مرحبا", "مرحباا"), 1)
        self.assertEqual(levenshtein("بيت", "دار"), 3)

    def test_confusion_map(self):
        self.assertIn("ت", CONFUSION_MAP["ب"])
        self.assertIn("ث", CONFUSION_MAP["ب"])
        self.assertIn("ن", CONFUSION_MAP["ب"])
        self.assertIn("خ", CONFUSION_MAP["ح"])
        self.assertIn("ج", CONFUSION_MAP["ح"])

    def test_tokenize_arabic(self):
        text = "قال: مرحبا بكم، أهلا وسهلا!"
        tokens = tokenize_arabic(text)
        self.assertIn("قال", tokens)
        self.assertIn("مرحبا", tokens)
        self.assertIn("بكم", tokens)
        self.assertIn("أهلا", tokens)
        self.assertIn("وسهلا", tokens)


if __name__ == "__main__":
    unittest.main()
