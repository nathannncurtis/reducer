"""
rasterize.py - PDF to image conversion using PyMuPDF.

Fast in-memory rendering, no external dependencies.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
try:
    import fitz  # pip install pymupdf
except ImportError:
    import pymupdf as fitz  # apt install python3-pymupdf

logger = logging.getLogger(__name__)


def get_page_count(pdf_path: Path) -> int:
    """Get total page count."""
    with fitz.open(pdf_path) as doc:
        return len(doc)


def get_page_dimensions(pdf_path: Path, page_num: int) -> Tuple[float, float]:
    """Get page dimensions in PDF points (1/72 inch)."""
    with fitz.open(pdf_path) as doc:
        page = doc[page_num]
        rect = page.rect
        return rect.width, rect.height


def rasterize_page(
    pdf_path: Path,
    page_num: int,
    dpi: int = 200
) -> Tuple[np.ndarray, float, float]:
    """
    Rasterize a single PDF page to RGB image.

    Args:
        pdf_path: Path to PDF
        page_num: 0-indexed page number
        dpi: Target DPI (50-300)

    Returns:
        Tuple of (RGB numpy array, page_width_pts, page_height_pts)
    """
    pdf_path = Path(pdf_path)
    dpi = max(50, min(dpi, 300))  # Clamp to 50-300

    with fitz.open(pdf_path) as doc:
        page = doc[page_num]

        # Get page dimensions in points
        rect = page.rect
        page_width_pts = rect.width
        page_height_pts = rect.height

        # Calculate zoom factor (72 DPI is PDF default)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        # Render to pixmap (in-memory)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        # Convert to numpy array (RGB)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, 3
        ).copy()  # Copy to own the memory

        logger.debug(
            f"Rasterized page {page_num}: {pixmap.width}x{pixmap.height} @ {dpi} DPI"
        )

        return image, page_width_pts, page_height_pts
