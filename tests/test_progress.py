"""Progress reporting must stay accurate and out of extracted text."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from arabicpdf import Config, ConversionResult
from arabicpdf.cli import _ConversionProgress, main
from arabicpdf.core import convert


class TestProgress(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.pdf = Path(self.tmp_dir.name) / "sample.pdf"
        self.pdf.touch()

    def test_selected_pages_report_before_work_and_only_count_completed_pages(self):
        events = []
        pdf_backend = MagicMock()
        pdf_backend.open.return_value.__len__.return_value = 5

        def engines():
            self.assertEqual(events[-1], (0, 2, "Checking OCR engines"))
            return []

        def classify(path, number):
            self.assertEqual(events[-1][2], f"Page {number}: classifying")
            return SimpleNamespace(
                page_type="digital_clean", confidence=1.0,
                has_images=False, raw_text="",
            )

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("arabicpdf.core.fitz", pdf_backend),
            patch("arabicpdf.core.get_available_engines", side_effect=engines),
            patch("arabicpdf.core.classify_page", side_effect=classify),
            patch("arabicpdf.core.extract_direct", return_value=[]),
            redirect_stdout(stdout), redirect_stderr(stderr),
        ):
            result = convert(
                self.pdf, config=Config(pages=[4, 2]),
                progress_callback=lambda *event: events.append(event),
            )

        self.assertEqual([p.number for p in result.pages], [4, 2])
        self.assertIn((1, 2, "Page 4: complete"), events)
        self.assertEqual(events[-1], (2, 2, "Page 2: complete"))
        self.assertEqual([e[0] for e in events], sorted(e[0] for e in events))
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def _run_cli(self, *extra_args):
        result = ConversionResult(text="مرحبا بالعالم", pages=[], confidence=1.0)

        def fake_convert(path, config, progress_callback):
            if progress_callback:
                progress_callback(0, 2, "Checking OCR engines")
                progress_callback(0, 2, "Page 1: OCR")
                progress_callback(1, 2, "Page 1: complete")
                progress_callback(2, 2, "Page 2: complete")
            return result

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("arabicpdf.cli.convert", side_effect=fake_convert),
            redirect_stdout(stdout), redirect_stderr(stderr),
        ):
            self.assertEqual(main(["convert", str(self.pdf), *extra_args]), 0)
        return stdout.getvalue(), stderr.getvalue()

    def test_cli_progress_does_not_pollute_stdout(self):
        stdout, stderr = self._run_cli()
        self.assertEqual(stdout, "مرحبا بالعالم\n")
        self.assertIn("Preparing conversion", stderr)
        self.assertIn("Page 1: OCR", stderr)
        self.assertIn("100%", stderr)
        self.assertIn("2/2", stderr)

    def test_cli_output_file_contains_only_text(self):
        output = Path(self.tmp_dir.name) / "output.txt"
        stdout, stderr = self._run_cli("-o", str(output))
        self.assertEqual(stdout, "")
        self.assertEqual(output.read_text(encoding="utf-8"), "مرحبا بالعالم")
        self.assertIn("100%", stderr)

    def test_no_progress(self):
        stdout, stderr = self._run_cli("--no-progress")
        self.assertEqual(stdout, "مرحبا بالعالم\n")
        self.assertEqual(stderr, "")

    def test_timer_refreshes_while_work_is_blocked_and_stops_on_failure(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True

        stderr = Terminal()
        refreshed = Event()
        with redirect_stderr(stderr):
            progress = _ConversionProgress()
            with patch.object(progress._bar, "refresh", side_effect=refreshed.set):
                with self.assertRaisesRegex(RuntimeError, "OCR failed"):
                    with progress:
                        self.assertTrue(refreshed.wait(2), "Timer did not refresh")
                        raise RuntimeError("OCR failed")
        self.assertFalse(progress._thread.is_alive())
        self.assertTrue(progress._bar.disable)
        self.assertIn("Failed", stderr.getvalue())
        self.assertNotIn("100%", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
