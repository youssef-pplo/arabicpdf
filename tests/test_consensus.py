"""Tests for OCR consensus voting (compatible with unittest and pytest)."""

import unittest
from arabicpdf.consensus import align_and_merge, multi_engine_consensus, resolve_arabic_confusion


class TestConsensus(unittest.TestCase):
    def test_align_and_merge_identical(self):
        text_a = "بسم الله الرحمن الرحيم"
        text_b = "بسم الله الرحمن الرحيم"
        merged, conf = align_and_merge(text_a, text_b)
        self.assertEqual(merged, "بسم الله الرحمن الرحيم")
        self.assertEqual(conf, 1.0)

    def test_align_and_merge_with_disagreement(self):
        text_a = "هذا كتان جميل"
        text_b = "هذا كتاب جميل"

        def mock_checker(w):
            return w in {"هذا", "كتاب", "جميل"}

        merged, conf = align_and_merge(text_a, text_b, check_valid_word=mock_checker)
        self.assertIn("كتاب", merged)
        self.assertIn("جميل", merged)

    def test_multi_engine_consensus(self):
        engine_results = [
            ("هذا كتاب رائع", 0.95),
            ("هذا كتاب رائع", 0.90),
            ("هذا كتان رائع", 0.70),
        ]
        merged, conf = multi_engine_consensus(engine_results)
        self.assertIn("كتاب", merged)
        self.assertIn("رائع", merged)
        self.assertGreater(conf, 0.8)

    def test_resolve_arabic_confusion(self):
        def mock_checker(w):
            return w == "كتاب"

        resolved = resolve_arabic_confusion("كتان", "كتاب", check_valid_word=mock_checker)
        self.assertEqual(resolved, "كتاب")


if __name__ == "__main__":
    unittest.main()
