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


def compress_1bit(image: np.ndarray) -> Tuple[bytes, int, int, bool]:
    """
    Compress image as 1-bit black/white using CCITT G4 via img2pdf.

    Thresholds to B&W using Otsu's method, then uses img2pdf for
    proper CCITT G4 fax encoding.

    Returns:
        (image_bytes, width, height, is_ccitt) - is_ccitt=True means CCITTFaxDecode
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    height, width = gray.shape
    logger.debug(f"compress_1bit: image size {width}x{height}")

    # Threshold to 1-bit using Otsu (0=black, 255=white)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert to PIL 1-bit image
    pil_img = Image.fromarray(binary, mode='L').convert('1')

    # Try to use img2pdf for proper CCITT G4
    try:
        import img2pdf

        # Save as TIFF with G4 compression
        tiff_buffer = io.BytesIO()
        pil_img.save(tiff_buffer, format='TIFF', compression='group4')
        tiff_data = tiff_buffer.getvalue()
        logger.debug(f"compress_1bit: TIFF G4 size = {len(tiff_data)} bytes")

        # Use img2pdf to create PDF with proper G4
        pdf_bytes = img2pdf.convert(tiff_data)
        logger.debug(f"compress_1bit: img2pdf PDF size = {len(pdf_bytes)} bytes")

        # Extract the image stream from the PDF
        import pikepdf
        temp_pdf = pikepdf.open(io.BytesIO(pdf_bytes))
        page = temp_pdf.pages[0]

        # Find the image XObject
        if '/Resources' in page and '/XObject' in page.Resources:
            for name, xobj in page.Resources.XObject.items():
                logger.debug(f"compress_1bit: found XObject {name}, type={type(xobj)}")
                # Get the raw compressed stream (G4 data)
                g4_data = xobj.read_raw_bytes()
                logger.debug(f"compress_1bit: G4 raw bytes = {len(g4_data)}")
                temp_pdf.close()
                return g4_data, width, height, True
        else:
            logger.debug("compress_1bit: No XObject found in img2pdf output")

        temp_pdf.close()
    except Exception as e:
        logger.warning(f"img2pdf G4 encoding failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # Fallback to packed bits (will use FlateDecode)
    packed = np.packbits(binary, axis=1)
    return packed.tobytes(), width, height, False


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
        img_data, width, height, is_ccitt = compress_1bit(image)
        mode = "CCITT G4" if is_ccitt else "1-bit zlib"
        logger.info(
            f"Page {page_num}: {len(img_data):,} bytes | "
            f"{width}x{height} | {mode}"
        )
        return CompressedPage(
            page_num=page_num,
            image_data=img_data,
            width=width,
            height=height,
            page_width_pts=page_width_pts,
            page_height_pts=page_height_pts,
            is_color=False,
            is_g4=is_ccitt  # True = CCITTFaxDecode, False = FlateDecode
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
