"""
OCR engine wrappers for Arabic PDF text extraction.

Provides a unified interface over multiple lightweight, local-only OCR
backends: Tesseract, PaddleOCR, and (optionally) Surya.  Each engine is
safe to construct even when its underlying dependency is missing — in that
case ``available`` is ``False`` and ``ocr()`` returns an empty result
instead of raising.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    """Container for a single OCR pass."""

    text: str = ""
    confidence: float = 0.0  # 0.0 – 1.0
    engine: str = ""


# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------

class TesseractEngine:
    """Wrapper around *pytesseract* (Tesseract CLI)."""

    def __init__(self) -> None:
        self._available: bool = False

        if np is None:
            return

        # 1. Check the Python binding
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            logger.warning(
                "pytesseract is not installed. "
                "Install it with: pip install pytesseract"
            )
            return

        # 2. Check the tesseract binary
        if shutil.which("tesseract") is None:
            logger.warning(
                "The 'tesseract' binary was not found on PATH. "
                "Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract"
            )
            return

        # 3. Check Arabic language pack
        try:
            result = subprocess.run(
                ["tesseract", "--list-langs"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            langs = result.stdout.lower()
            if "ara" not in langs:
                logger.warning(
                    "Tesseract Arabic language pack ('ara') is not installed. "
                    "Install it, e.g.: sudo apt install tesseract-ocr-ara"
                )
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not verify Tesseract languages: %s", exc)
            return

        self._available = True
        logger.debug("TesseractEngine initialised successfully.")

    # -- properties ----------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    # -- OCR -----------------------------------------------------------------

    def ocr(self, image: np.ndarray) -> OCRResult:
        """Run Tesseract OCR on a NumPy image array.

        Parameters
        ----------
        image:
            BGR or greyscale image as a NumPy array.

        Returns
        -------
        OCRResult
            Extracted text, average confidence (0–1), and engine name.
        """
        if not self._available:
            logger.error("TesseractEngine is not available; returning empty result.")
            return OCRResult(engine="tesseract")

        try:
            import pytesseract

            config = "--oem 1 --psm 6 -l ara -c preserve_interword_spaces=1"

            # Extract text
            text: str = pytesseract.image_to_string(image, config=config)

            # Compute average word-level confidence
            try:
                data = pytesseract.image_to_data(
                    image, config=config, output_type=pytesseract.Output.DICT
                )
                confidences = [
                    int(c)
                    for c in data.get("conf", [])
                    if str(c).lstrip("-").isdigit() and int(c) >= 0
                ]
                avg_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
            except Exception:  # noqa: BLE001
                avg_conf = 0.0

            return OCRResult(
                text=text.strip(),
                confidence=round(min(max(avg_conf, 0.0), 1.0), 4),
                engine="tesseract",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Tesseract OCR failed: %s", exc, exc_info=True)
            return OCRResult(engine="tesseract")


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------

class PaddleOCREngine:
    """Wrapper around *PaddleOCR* with lazy model loading."""

    def __init__(self) -> None:
        self._available: bool = False
        self._engine: Any = None  # lazily created PaddleOCR instance

        if np is None:
            return

        try:
            import paddleocr  # noqa: F401
            self._available = True
            logger.debug("PaddleOCR package found; engine will be loaded on first use.")
        except ImportError:
            logger.warning(
                "paddleocr is not installed. "
                "Install it with: pip install paddleocr"
            )

    # -- properties ----------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    # -- lazy init -----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Create the PaddleOCR instance on first use."""
        if self._engine is not None:
            return

        from paddleocr import PaddleOCR  # type: ignore[import-untyped]

        try:
            self._engine = PaddleOCR(lang="ar")
        except TypeError:
            try:
                self._engine = PaddleOCR(
                    use_angle_cls=True,
                    lang="ar",
                    use_gpu=False,
                    show_log=False,
                )
            except Exception as e:
                logger.error("Failed to initialize PaddleOCR: %s", e)
                raise
        logger.debug("PaddleOCR model loaded.")

    # -- OCR -----------------------------------------------------------------

    def ocr(self, image: np.ndarray) -> OCRResult:
        """Run PaddleOCR on a NumPy image array.

        Parameters
        ----------
        image:
            BGR or greyscale image as a NumPy array.

        Returns
        -------
        OCRResult
            Extracted text, average confidence (0–1), and engine name.
        """
        if not self._available:
            logger.error("PaddleOCREngine is not available; returning empty result.")
            return OCRResult(engine="paddleocr")

        try:
            self._ensure_loaded()

            results = self._engine.ocr(image, cls=True)

            # PaddleOCR returns: list[list[ [box, (text, conf)] ]] or None
            if not results:
                return OCRResult(engine="paddleocr")

            lines: list[str] = []
            confidences: list[float] = []

            for page in results:
                if not page:
                    continue
                for line_info in page:
                    # Each line_info is [box_coords, (text, confidence)]
                    if not line_info or len(line_info) < 2:
                        continue
                    text_conf = line_info[1]
                    if text_conf is None:
                        continue
                    text_str, conf = text_conf
                    if text_str:
                        lines.append(str(text_str))
                        confidences.append(float(conf))

            joined = "\n".join(lines)
            avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

            return OCRResult(
                text=joined.strip(),
                confidence=round(min(max(avg_conf, 0.0), 1.0), 4),
                engine="paddleocr",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("PaddleOCR failed: %s", exc, exc_info=True)
            return OCRResult(engine="paddleocr")


# ---------------------------------------------------------------------------
# Surya  (tier-2 / optional)
# ---------------------------------------------------------------------------

class SuryaEngine:
    """Wrapper around *Surya* OCR (optional tier-2 engine)."""

    def __init__(self) -> None:
        self._available: bool = False

        try:
            import surya  # noqa: F401
            self._available = True
            logger.debug("Surya package found.")
        except ImportError:
            logger.info(
                "surya is not installed (optional). "
                "Install it with: pip install surya-ocr"
            )

    # -- properties ----------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    # -- OCR -----------------------------------------------------------------

    def ocr(self, image: np.ndarray) -> OCRResult:
        """Run Surya OCR on a NumPy image array.

        Parameters
        ----------
        image:
            BGR or greyscale image as a NumPy array.

        Returns
        -------
        OCRResult
            Extracted text, average confidence (0–1), and engine name.

        Raises
        ------
        RuntimeError
            If Surya is not installed.
        """
        if not self._available:
            raise RuntimeError(
                "SuryaEngine is not available — the 'surya-ocr' package is not installed."
            )

        try:
            from PIL import Image as PILImage
            from surya.ocr import run_ocr  # type: ignore[import-untyped]
            from surya.model.detection import segformer  # type: ignore[import-untyped]
            from surya.model.recognition import model as rec_model_mod  # type: ignore[import-untyped]

            # Convert numpy array → PIL Image
            pil_image = PILImage.fromarray(image)

            # Load models (Surya caches these internally)
            det_processor, det_model = segformer.load_processor(), segformer.load_model()
            rec_processor, rec_model = rec_model_mod.load_processor(), rec_model_mod.load_model()

            langs = [["ar"]]
            results = run_ocr(
                [pil_image],
                langs,
                det_model,
                det_processor,
                rec_model,
                rec_processor,
            )

            if not results:
                return OCRResult(engine="surya")

            page = results[0]
            lines: list[str] = []
            confidences: list[float] = []
            for text_line in page.text_lines:
                lines.append(text_line.text)
                confidences.append(text_line.confidence)

            joined = "\n".join(lines)
            avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

            return OCRResult(
                text=joined.strip(),
                confidence=round(min(max(avg_conf, 0.0), 1.0), 4),
                engine="surya",
            )
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Surya OCR failed: %s", exc, exc_info=True)
            return OCRResult(engine="surya")


# ---------------------------------------------------------------------------
# Discovery helper
# ---------------------------------------------------------------------------

def get_available_engines() -> list[TesseractEngine | PaddleOCREngine | SuryaEngine]:
    """Return a list of OCR engine instances whose dependencies are satisfied.

    Engines are probed in order: Tesseract → PaddleOCR → Surya.
    Only engines with ``available == True`` are included.
    """
    available: list[TesseractEngine | PaddleOCREngine | SuryaEngine] = []

    for engine_cls in (TesseractEngine, PaddleOCREngine, SuryaEngine):
        engine = engine_cls()
        if engine.available:
            available.append(engine)

    if not available:
        logger.warning(
            "No OCR engines are available. Install at least one of: "
            "pytesseract + tesseract-ocr-ara, paddleocr, surya-ocr"
        )
    else:
        names = [type(e).__name__ for e in available]
        logger.info("Available OCR engines: %s", ", ".join(names))

    return available
