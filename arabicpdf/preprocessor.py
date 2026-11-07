"""
Image preprocessing module for Arabic OCR.

Uses OpenCV to prepare PDF page images for maximum OCR accuracy on Arabic text.
Arabic script is particularly sensitive to preprocessing because:
- Dots (nuqat) distinguish many letters: ب/ت/ث/ن/ي differ ONLY by dot count/position
- Speckle noise can create phantom dots: ح→خ, ر→ز, د→ذ
- RTL text flow requires careful deskewing
- Faded manuscripts need contrast enhancement without losing thin strokes
"""

from __future__ import annotations

from typing import Literal

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


# ---------------------------------------------------------------------------
# Preprocessing strategies
# ---------------------------------------------------------------------------

STRATEGIES = ["default", "high_contrast", "no_binarize", "sharpen_first", "denoise_heavy"]

StrategyName = Literal[
    "default", "high_contrast", "no_binarize", "sharpen_first", "denoise_heavy"
]


class ArabicPreprocessor:
    """Prepares page images for Arabic OCR via configurable preprocessing pipelines.

    Each *strategy* is an ordered sequence of image transformations chosen to
    balance noise removal with preservation of the fine Arabic dot patterns
    that distinguish otherwise-identical letter shapes.

    Example::

        preprocessor = ArabicPreprocessor()
        clean = preprocessor.process(raw_image, strategy="high_contrast")
    """

    def __init__(self) -> None:
        if cv2 is None or np is None:
            raise ImportError(
                "OpenCV and NumPy are required for ArabicPreprocessor. "
                "Install them with: pip install opencv-python-headless numpy"
            )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def process(self, image: np.ndarray, strategy: str = "default") -> np.ndarray:
        """Apply a named preprocessing strategy to *image*.

        Parameters
        ----------
        image:
            Input image as a NumPy array (BGR or grayscale).
        strategy:
            One of :data:`STRATEGIES`.  Defaults to ``"default"``.

        Returns
        -------
        np.ndarray
            The preprocessed image ready for OCR.

        Raises
        ------
        ValueError
            If *strategy* is not a recognised name.
        """
        dispatch = {
            "default": self._strategy_default,
            "high_contrast": self._strategy_high_contrast,
            "no_binarize": self._strategy_no_binarize,
            "sharpen_first": self._strategy_sharpen_first,
            "denoise_heavy": self._strategy_denoise_heavy,
        }

        if strategy not in dispatch:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {', '.join(STRATEGIES)}"
            )

        return dispatch[strategy](image)

    # --------------------------------------------------------------------- #
    # Individual transforms
    # --------------------------------------------------------------------- #

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert a BGR image to grayscale.  No-op if already single-channel."""
        if len(img.shape) == 2:
            return img
        if img.shape[2] == 1:
            return img[:, :, 0]
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """Fix small page rotations using the minimum-area bounding rectangle
        of non-white (dark) pixels.

        Skips correction when the detected angle is below 0.5° to avoid
        unnecessary interpolation that could blur Arabic dots.
        """
        # Find coordinates of all dark pixels (below mid-gray threshold)
        coords = np.column_stack(np.where(img < 128))

        if coords.shape[0] < 10:
            # Not enough dark pixels to estimate rotation
            return img

        # cv2.minAreaRect expects (x, y) points
        rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
        angle: float = rect[2]

        # minAreaRect returns angles in [-90, 0).  Normalise to [-45, 45].
        if angle < -45:
            angle += 90

        if abs(angle) < 0.5:
            return img

        h, w = img.shape[:2]
        centre = (w / 2.0, h / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)

        return cv2.warpAffine(
            img,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _denoise(self, img: np.ndarray, h: int = 10) -> np.ndarray:
        """Remove speckle noise with non-local means denoising.

        This is *critical* for Arabic: random speckle can turn into phantom
        dots, changing the identity of a letter entirely (e.g. ح→خ, ر→ز).

        Parameters
        ----------
        h:
            Filter strength.  Higher removes more noise but may blur dots.
            Default 10 is a safe balance; ``denoise_heavy`` strategy uses 20.
        """
        return cv2.fastNlMeansDenoising(
            img,
            h=h,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        """Adaptive Gaussian thresholding.

        Better than global Otsu for documents with uneven lighting (scans,
        photographs of pages).  ``blockSize=25`` and ``C=12`` are tuned for
        typical Arabic printed text at 300 DPI.
        """
        return cv2.adaptiveThreshold(
            img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=25,
            C=12,
        )

    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        """Unsharp-mask sharpening to preserve Arabic dots (nuqat).

        The difference between ب/ت/ث/ن/ي is *only* the number and position
        of dots.  Sharpening ensures these small features survive binarisation
        and denoising.
        """
        # Classic 3×3 unsharp-mask kernel
        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )
        return cv2.filter2D(img, ddepth=-1, kernel=kernel)

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Contrast-Limited Adaptive Histogram Equalisation (CLAHE).

        Recovers faded or low-contrast text without over-amplifying noise.
        ``clipLimit=2.0`` and ``tileGridSize=(8, 8)`` work well for full-page
        scans at 300 DPI.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def _remove_borders(self, img: np.ndarray, margin: int = 10) -> np.ndarray:
        """Remove dark borders and shadows at page edges.

        Sets the outermost *margin* pixels on every side to white (255),
        eliminating scanner shadows and book-spine darkness that can confuse
        OCR layout analysis.
        """
        result = img.copy()
        result[:margin, :] = 255       # top
        result[-margin:, :] = 255      # bottom
        result[:, :margin] = 255       # left
        result[:, -margin:] = 255      # right
        return result

    # --------------------------------------------------------------------- #
    # Strategy pipelines
    # --------------------------------------------------------------------- #

    def _strategy_default(self, image: np.ndarray) -> np.ndarray:
        """grayscale → deskew → denoise → binarize → sharpen"""
        img = self._to_grayscale(image)
        img = self._deskew(img)
        img = self._denoise(img)
        img = self._binarize(img)
        img = self._sharpen(img)
        return img

    def _strategy_high_contrast(self, image: np.ndarray) -> np.ndarray:
        """grayscale → deskew → enhance_contrast → denoise → binarize → sharpen"""
        img = self._to_grayscale(image)
        img = self._deskew(img)
        img = self._enhance_contrast(img)
        img = self._denoise(img)
        img = self._binarize(img)
        img = self._sharpen(img)
        return img

    def _strategy_no_binarize(self, image: np.ndarray) -> np.ndarray:
        """grayscale → deskew → denoise → enhance_contrast

        Skips binarisation — some text (especially handwritten or
        calligraphic Arabic) is better recognised in grayscale.
        """
        img = self._to_grayscale(image)
        img = self._deskew(img)
        img = self._denoise(img)
        img = self._enhance_contrast(img)
        return img

    def _strategy_sharpen_first(self, image: np.ndarray) -> np.ndarray:
        """grayscale → sharpen → deskew → denoise → binarize

        Sharpens *before* denoising so that Arabic dots are reinforced
        early and survive subsequent filtering.
        """
        img = self._to_grayscale(image)
        img = self._sharpen(img)
        img = self._deskew(img)
        img = self._denoise(img)
        img = self._binarize(img)
        return img

    def _strategy_denoise_heavy(self, image: np.ndarray) -> np.ndarray:
        """grayscale → deskew → heavy denoise (h=20) → binarize → sharpen

        Aggressive denoising for very noisy scans.  Uses ``h=20`` instead
        of the default ``h=10`` — may blur some fine dots on clean originals.
        """
        img = self._to_grayscale(image)
        img = self._deskew(img)
        img = self._denoise(img, h=20)
        img = self._binarize(img)
        img = self._sharpen(img)
        return img


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def render_pdf_page(pdf_path: str, page_num: int, dpi: int = 300) -> np.ndarray:
    """Render a single PDF page to a NumPy image array at the given DPI.

    Uses PyMuPDF (``fitz``) for rendering.  The returned array is in BGR
    colour order (matching OpenCV conventions) so it can be passed directly
    to :class:`ArabicPreprocessor`.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file on disk.
    page_num:
        Zero-based page index.
    dpi:
        Resolution for rasterisation.  300 DPI is the recommended minimum
        for Arabic OCR; 150 DPI may lose dots.

    Returns
    -------
    np.ndarray
        The rendered page as an 8-bit BGR image.

    Raises
    ------
    ImportError
        If PyMuPDF is not installed.
    IndexError
        If *page_num* is out of range for the document.
    FileNotFoundError
        If *pdf_path* does not exist.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF rendering. "
            "Install it with: pip install pymupdf"
        )

    doc = fitz.open(pdf_path)

    if page_num < 0 or page_num >= len(doc):
        doc.close()
        raise IndexError(
            f"Page {page_num} is out of range for document with "
            f"{len(doc)} page(s)."
        )

    page = doc[page_num]

    # Scale matrix: default PDF resolution is 72 DPI
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    # Convert pixmap samples (RGB) to a NumPy array, then to BGR for OpenCV
    img_array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, 3
    )
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    doc.close()
    return img_bgr
