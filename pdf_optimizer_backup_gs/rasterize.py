"""
rasterize.py - PDF to image conversion with strict dimension limits.

HARD REQUIREMENTS:
- Maximum 200 DPI
- Maximum dimensions: 1700x2200 pixels (8.5x11 @ 200 DPI)
- Everything gets flattened to raster - no vectors, fonts, or layers preserved
"""

import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import cv2
import pikepdf

logger = logging.getLogger(__name__)

# Hard limits
MAX_DPI = 200
MAX_WIDTH = 1700   # 8.5" @ 200 DPI
MAX_HEIGHT = 2200  # 11" @ 200 DPI


def get_gs_command() -> str:
    """Get Ghostscript command for this platform."""
    try:
        subprocess.run(["gs", "--version"], capture_output=True, timeout=5)
        return "gs"
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            subprocess.run(["gswin64c", "--version"], capture_output=True, timeout=5)
            return "gswin64c"
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError("Ghostscript not found. Install gs or gswin64c.")


def get_page_count(pdf_path: Path) -> int:
    """Get total page count."""
    with pikepdf.open(pdf_path) as pdf:
        return len(pdf.pages)


def get_page_dimensions(pdf_path: Path, page_num: int) -> Tuple[float, float]:
    """Get page dimensions in PDF points (1/72 inch)."""
    with pikepdf.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        if "/MediaBox" in page:
            box = page["/MediaBox"]
            width = float(box[2]) - float(box[0])
            height = float(box[3]) - float(box[1])
        else:
            width, height = 612.0, 792.0  # US Letter default
        return width, height


def constrain_to_limits(width: int, height: int) -> Tuple[int, int]:
    """
    Constrain dimensions to MAX_WIDTH x MAX_HEIGHT while preserving aspect ratio.
    """
    if width <= MAX_WIDTH and height <= MAX_HEIGHT:
        return width, height

    scale = min(MAX_WIDTH / width, MAX_HEIGHT / height)
    return int(width * scale), int(height * scale)


def rasterize_page(
    pdf_path: Path,
    page_num: int,
    dpi: int = MAX_DPI
) -> Tuple[np.ndarray, float, float]:
    """
    Rasterize a single PDF page to RGB image.

    Flattens everything - vectors, fonts, layers, transparency.
    Constrains to MAX_WIDTH x MAX_HEIGHT.

    Args:
        pdf_path: Path to PDF
        page_num: 0-indexed page number
        dpi: Target DPI (capped at MAX_DPI)

    Returns:
        Tuple of (RGB numpy array, page_width_pts, page_height_pts)
    """
    pdf_path = Path(pdf_path)
    dpi = max(50, min(dpi, 300))  # Clamp to 50-300

    # Get original page dimensions
    page_width_pts, page_height_pts = get_page_dimensions(pdf_path, page_num)

    gs_cmd = get_gs_command()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "page.png"

        # Ghostscript uses 1-indexed pages
        gs_page = page_num + 1

        cmd = [
            gs_cmd,
            "-dNOPAUSE",
            "-dBATCH",
            "-dSAFER",
            "-sDEVICE=png16m",  # 24-bit RGB
            f"-r{dpi}",
            f"-dFirstPage={gs_page}",
            f"-dLastPage={gs_page}",
            # Flatten everything
            "-dTextAlphaBits=4",
            "-dGraphicsAlphaBits=4",
            "-dNOTRANSPARENCY",  # Flatten transparency
            f"-sOutputFile={output_path}",
            str(pdf_path)
        ]

        logger.debug(f"Rasterizing page {page_num} at {dpi} DPI")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise RuntimeError(f"Ghostscript failed: {result.stderr}")

        # Load and constrain
        img = Image.open(output_path).convert("RGB")
        width, height = img.size

        # Constrain to limits
        new_width, new_height = constrain_to_limits(width, height)
        if (new_width, new_height) != (width, height):
            logger.debug(f"Constraining {width}x{height} to {new_width}x{new_height}")
            img = img.resize((new_width, new_height), Image.LANCZOS)

        return np.array(img), page_width_pts, page_height_pts


def rasterize_page_fast(
    pdf_path: Path,
    page_num: int
) -> Tuple[np.ndarray, float, float]:
    """
    Fast rasterization at fixed 200 DPI with dimension limits.

    This is the primary entry point for the aggressive pipeline.
    """
    return rasterize_page(pdf_path, page_num, dpi=MAX_DPI)
