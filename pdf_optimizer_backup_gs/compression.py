"""
compression.py - PDF page compression.

Supports:
- JPEG for color/grayscale pages
- CCITT G4 for B&W pages (1-bit, very small)
"""

import io
import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

# Target size per page
TARGET_SIZE_PER_PAGE = 10 * 1024  # 10 KB

# Saturation threshold for grayscale conversion
# Only convert to grayscale if mean saturation is below this
GRAYSCALE_SATURATION_THRESHOLD = 10  # Out of 255


@dataclass
class CompressedPage:
    """Compressed page data ready for PDF embedding."""
    page_num: int
    image_data: bytes
    width: int
    height: int
    page_width_pts: float
    page_height_pts: float
    is_color: bool
    is_g4: bool = False  # True if CCITT G4, False if JPEG

    @property
    def total_size(self) -> int:
        return len(self.image_data)


def is_grayscale_image(image: np.ndarray) -> bool:
    """
    Check if image is effectively grayscale based on saturation.

    Only returns True if the entire page has very low saturation.
    Does NOT try to detect "color regions" or "photos".
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return True  # Already grayscale

    # Convert to HSV and check saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mean_saturation = np.mean(hsv[:, :, 1])

    is_gray = mean_saturation < GRAYSCALE_SATURATION_THRESHOLD
    logger.debug(f"Mean saturation: {mean_saturation:.1f}, is_grayscale: {is_gray}")

    return is_gray


def compress_1bit(image: np.ndarray) -> Tuple[bytes, int, int]:
    """
    Compress image as 1-bit black/white.

    Thresholds to B&W using Otsu's method, packs bits.
    Will be compressed with FlateDecode in PDF (very small for text).

    Returns:
        (packed_bytes, width, height)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    height, width = gray.shape

    # Threshold to 1-bit using Otsu (0=black, 255=white)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Pack bits: 8 pixels per byte, 1=white 0=black
    # PDF expects 1=white for DeviceGray with 1 bit
    packed = np.packbits(binary, axis=1)
    raw_data = packed.tobytes()

    return raw_data, width, height


def compress_jpeg(
    image: np.ndarray,
    quality: int = 15,
    target_dpi: int = 200,
    current_dpi: int = 200
) -> Tuple[bytes, int, int, bool]:
    """
    Compress image as JPEG.

    Args:
        image: RGB or grayscale numpy array
        quality: JPEG quality (1-100, lower = smaller)
        target_dpi: Target DPI for downsampling
        current_dpi: Current image DPI

    Returns:
        (jpeg_bytes, width, height, is_color)
    """
    # Downsample if needed
    if target_dpi < current_dpi:
        scale = target_dpi / current_dpi
        new_width = max(1, int(image.shape[1] * scale))
        new_height = max(1, int(image.shape[0] * scale))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    height, width = image.shape[:2]

    # Check if effectively grayscale
    is_color = not is_grayscale_image(image)

    # Convert to PIL
    if len(image.shape) == 2:
        # Already grayscale
        img = Image.fromarray(image, mode="L")
        is_color = False
    elif not is_color:
        # Convert to grayscale since it's effectively gray anyway
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img = Image.fromarray(gray, mode="L")
    else:
        # Keep as color
        if image.shape[2] == 4:
            image = image[:, :, :3]
        img = Image.fromarray(image, mode="RGB")

    # Encode as JPEG
    buffer = io.BytesIO()
    img.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        subsampling=2  # 4:2:0 chroma subsampling
    )

    return buffer.getvalue(), width, height, is_color


def compress_page(
    image: np.ndarray,
    page_num: int,
    page_width_pts: float,
    page_height_pts: float,
    quality: int = 15,
    target_dpi: int = 150,
    current_dpi: int = 200,
    use_g4: bool = False
) -> CompressedPage:
    """
    Compress a rasterized page.

    Args:
        image: Rasterized page (RGB numpy array)
        page_num: Page number
        page_width_pts: Page width in PDF points
        page_height_pts: Page height in PDF points
        quality: JPEG quality (lower = smaller file)
        target_dpi: Target DPI (downsample if current > target)
        current_dpi: Current image DPI
        use_g4: Use 1-bit compression for B&W pages

    Returns:
        CompressedPage with image data
    """
    # Check if page is grayscale
    is_color = not is_grayscale_image(image)

    # Downsample if needed
    if target_dpi < current_dpi:
        scale = target_dpi / current_dpi
        new_width = max(1, int(image.shape[1] * scale))
        new_height = max(1, int(image.shape[0] * scale))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Use 1-bit for B&W pages if requested
    if use_g4 and not is_color:
        raw_data, width, height = compress_1bit(image)
        logger.info(
            f"Page {page_num}: {len(raw_data):,} bytes | "
            f"{width}x{height} | 1-bit B&W"
        )
        return CompressedPage(
            page_num=page_num,
            image_data=raw_data,
            width=width,
            height=height,
            page_width_pts=page_width_pts,
            page_height_pts=page_height_pts,
            is_color=False,
            is_g4=True
        )

    # Use JPEG for color pages (or all pages if not use_g4)
    jpeg_data, width, height, is_color = compress_jpeg(
        image,
        quality=quality,
        target_dpi=target_dpi,
        current_dpi=target_dpi  # Already downsampled above
    )

    logger.info(
        f"Page {page_num}: {len(jpeg_data):,} bytes | "
        f"{width}x{height} | color={is_color} | q={quality}"
    )

    return CompressedPage(
        page_num=page_num,
        image_data=jpeg_data,
        width=width,
        height=height,
        page_width_pts=page_width_pts,
        page_height_pts=page_height_pts,
        is_color=is_color,
        is_g4=False
    )


def compress_page_adaptive(
    image: np.ndarray,
    page_num: int,
    page_width_pts: float,
    page_height_pts: float,
    target_size: int = TARGET_SIZE_PER_PAGE,
    current_dpi: int = 200
) -> CompressedPage:
    """
    Compress page with adaptive quality to hit target size.

    Fast: only tries a few combinations.
    """
    # Fixed 200 DPI, quality 75+ for sharpness
    settings = [
        (200, 85),   # 200 DPI, quality 85
        (200, 80),   # 200 DPI, quality 80
        (200, 75),   # 200 DPI, quality 75 - minimum
    ]

    for target_dpi, quality in settings:
        result = compress_page(
            image,
            page_num,
            page_width_pts,
            page_height_pts,
            quality=quality,
            target_dpi=target_dpi,
            current_dpi=current_dpi
        )

        # Check if we hit target (with tolerance)
        if result.total_size <= target_size * 1.5:
            return result

    # Return last result (smallest)
    return result
