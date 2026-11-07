"""End-to-end tests for ArabicPDF (compatible with unittest and pytest)."""

import tempfile
import unittest
from pathlib import Path

try:
    import pymupdf as fitz
    FITZ_AVAILABLE = True
except ImportError:
    try:
        import fitz
        FITZ_AVAILABLE = True
    except ImportError:
        FITZ_AVAILABLE = False


class TestCore(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        real_doc = Path(__file__).resolve().parent.parent / "document.pdf"
        if real_doc.is_file():
            self.pdf_path = real_doc
        else:
            self.pdf_path = Path(self.tmp_dir.name) / "sample_arabic.pdf"
            if FITZ_AVAILABLE:
                doc = fitz.open()
                page = doc.new_page()
                page.insert_text(fitz.Point(50, 72), "مرحبا", fontsize=18)
                doc.save(str(self.pdf_path))
                doc.close()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_invalid_pdf_path(self):
        from arabicpdf import to_text
        with self.assertRaises(FileNotFoundError):
            to_text("non_existent_file_xyz_123.pdf")

    def test_convert_digital_pdf(self):
        if not FITZ_AVAILABLE:
            self.skipTest("PyMuPDF (fitz) is not installed in current environment")

        from arabicpdf import convert, Config
        cfg = Config(pages=[1])
        result = convert(self.pdf_path, config=cfg)
        self.assertEqual(result.page_count, 1)
        self.assertGreater(result.confidence, 0.0)
        page_res = result.pages[0]
        self.assertEqual(page_res.number, 1)
        self.assertGreater(len(page_res.text), 0)

    def test_to_text_convenience(self):
        if not FITZ_AVAILABLE:
            self.skipTest("PyMuPDF (fitz) is not installed in current environment")

        from arabicpdf import to_text, Mode
        text = to_text(self.pdf_path, pages=[1], mode=Mode.FAST)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()
