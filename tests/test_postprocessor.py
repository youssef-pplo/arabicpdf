"""Tests for postprocessor module (compatible with unittest and pytest)."""

import unittest
from arabicpdf.postprocessor import (
    normalize_unicode,
    remove_tatweel,
    remove_diacritics,
    remove_zero_width,
    normalize_whitespace,
    normalize_alef,
    postprocess,
)


class TestPostprocessor(unittest.TestCase):
    def test_remove_tatweel(self):
        text_with_tatweel = "جــــمـيـــل"
        cleaned = remove_tatweel(text_with_tatweel)
        self.assertEqual(cleaned, "جميل")

    def test_remove_diacritics(self):
        text_with_tashkeel = "كَتَبَ القَلَمُ"
        cleaned = remove_diacritics(text_with_tashkeel)
        self.assertEqual(cleaned, "كتب القلم")

    def test_normalize_alef(self):
        text = "أحمد وإبراهيم مع آمنة"
        normalized = normalize_alef(text)
        self.assertEqual(normalized, "احمد وابراهيم مع امنة")

    def test_normalize_whitespace(self):
        text = "  كلمة    أخرى  \n\n\n\n  سطر جديد  "
        cleaned = normalize_whitespace(text)
        self.assertNotIn("    ", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

    def test_postprocess_full(self):
        raw = "  أهــــلاً   وسهـــلاً   بِكُمْ  "
        processed = postprocess(
            raw,
            preserve_diacritics=False,
            do_normalize_alef=True,
        )
        self.assertNotIn("ــــ", processed)
        self.assertEqual(processed, "اهلا وسهلا بكم")


if __name__ == "__main__":
    unittest.main()
