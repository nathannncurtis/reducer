"""
segmentation.py - Robust foreground/background separation for MRC compression.

REQUIREMENTS:
- Boost contrast BEFORE thresholding
- Adaptive thresholding with Otsu fallback
- Morphology closing to repair broken characters
- Fallback detection: if <5% black pixels, segmentation failed
- Color detection: >0.5% pixels with chroma above threshold = color

Output:
- Foreground mask: 1-bit binary (255=text, 0=background)
- Background: Original image with text areas filled/smoothed
- is_color: Whether page contains meaningful color
- foreground_coverage: Percentage of pixels detected as foreground
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

# Thresholds
MIN_FOREGROUND_COVERAGE = 0.03  # 3% - below this, segmentation failed
MAX_FOREGROUND_COVERAGE = 0.60  # 60% - above this, probably wrong
COLOR_CHROMA_THRESHOLD = 15     # Minimum chroma to be considered "color"
COLOR_PIXEL_THRESHOLD = 0.005   # 0.5% of pixels must have color


@dataclass
class SegmentationResult:
    """Result of page segmentation."""
    foreground_mask: np.ndarray    # Binary mask (255=foreground, 0=background)
    background: np.ndarray          # Background layer (RGB or grayscale)
    is_color: bool                  # True if page has meaningful color
    foreground_coverage: float      # Fraction of pixels in foreground
    segmentation_ok: bool           # True if segmentation succeeded


def detect_color(image: np.ndarray) -> bool:
    """
    Detect if image contains meaningful color.

    Rule: >0.5% of pixels must have chroma value above threshold.
    This catches colored logos, stamps, highlights, etc.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False

    # Convert to LAB colorspace for better chroma detection
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Chroma = sqrt(a^2 + b^2) where a,b are color channels
    # LAB a channel is at index 1, b channel is at index 2
    # They're centered at 128 in OpenCV's 8-bit LAB
    a = lab[:, :, 1].astype(np.float32) - 128
    b = lab[:, :, 2].astype(np.float32) - 128
    chroma = np.sqrt(a**2 + b**2)

    # Count pixels with significant chroma
    color_pixels = np.sum(chroma > COLOR_CHROMA_THRESHOLD)
    total_pixels = image.shape[0] * image.shape[1]
    color_ratio = color_pixels / total_pixels

    is_color = color_ratio > COLOR_PIXEL_THRESHOLD
    logger.debug(f"Color detection: {color_ratio*100:.2f}% chromatic pixels, is_color={is_color}")

    return is_color


def boost_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Boost contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    This helps thresholding work better on low-contrast scans.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def adaptive_threshold(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Apply adaptive thresholding with multiple fallback strategies.

    Strategy:
    1. Try Sauvola-style adaptive threshold
    2. If that fails (low coverage), try Otsu
    3. If still failing, try aggressive Otsu with contrast boost

    Returns (binary_mask, coverage_ratio)
    """
    height, width = gray.shape
    total_pixels = height * width

    # Strategy 1: Adaptive Gaussian thresholding (Sauvola-like)
    # Block size must be odd
    block_size = max(11, (min(height, width) // 50) | 1)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Invert so text is white (255)
        block_size,
        C=10  # Constant subtracted from mean
    )

    coverage = np.sum(binary > 0) / total_pixels
    logger.debug(f"Adaptive threshold: {coverage*100:.2f}% foreground")

    if MIN_FOREGROUND_COVERAGE <= coverage <= MAX_FOREGROUND_COVERAGE:
        return binary, coverage

    # Strategy 2: Otsu's method
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coverage_otsu = np.sum(binary_otsu > 0) / total_pixels
    logger.debug(f"Otsu threshold: {coverage_otsu*100:.2f}% foreground")

    if MIN_FOREGROUND_COVERAGE <= coverage_otsu <= MAX_FOREGROUND_COVERAGE:
        return binary_otsu, coverage_otsu

    # Strategy 3: Contrast-boosted Otsu
    boosted = boost_contrast(gray)
    _, binary_boosted = cv2.threshold(boosted, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coverage_boosted = np.sum(binary_boosted > 0) / total_pixels
    logger.debug(f"Boosted Otsu: {coverage_boosted*100:.2f}% foreground")

    if MIN_FOREGROUND_COVERAGE <= coverage_boosted <= MAX_FOREGROUND_COVERAGE:
        return binary_boosted, coverage_boosted

    # Strategy 4: More aggressive - lower the bar
    # Use a fixed threshold based on mean pixel value
    mean_val = np.mean(gray)
    threshold_val = mean_val * 0.85  # 85% of mean
    _, binary_fixed = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
    coverage_fixed = np.sum(binary_fixed > 0) / total_pixels
    logger.debug(f"Fixed threshold ({threshold_val:.0f}): {coverage_fixed*100:.2f}% foreground")

    # Return the best result we have
    results = [
        (binary, coverage),
        (binary_otsu, coverage_otsu),
        (binary_boosted, coverage_boosted),
        (binary_fixed, coverage_fixed)
    ]

    # Pick the one closest to target range
    valid_results = [(b, c) for b, c in results if c >= MIN_FOREGROUND_COVERAGE]
    if valid_results:
        # Pick one with coverage in valid range, preferring lower coverage
        valid_results.sort(key=lambda x: x[1])
        return valid_results[0]

    # All failed - return the one with highest coverage
    results.sort(key=lambda x: x[1], reverse=True)
    return results[0]


def repair_characters(binary: np.ndarray) -> np.ndarray:
    """
    Repair broken characters using morphological closing.
    This fills small gaps in strokes without significantly changing shapes.
    """
    # Small kernel for closing - repairs broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return closed


def create_background(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    is_color: bool
) -> np.ndarray:
    """
    Create background layer by filling foreground regions.

    Uses inpainting to fill text areas with surrounding colors.
    Light smoothing to reduce JPEG artifacts.
    """
    # Dilate mask slightly to ensure text edges are covered
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv2.dilate(foreground_mask, kernel, iterations=1)

    # Inpaint the foreground regions
    if len(image.shape) == 3:
        # Color image
        background = cv2.inpaint(image, dilated_mask, 3, cv2.INPAINT_TELEA)
    else:
        # Grayscale
        background = cv2.inpaint(image, dilated_mask, 3, cv2.INPAINT_TELEA)

    # Light smoothing to reduce JPEG blocking artifacts
    # Use bilateral filter to preserve edges while smoothing
    if len(background.shape) == 3:
        background = cv2.bilateralFilter(background, 5, 50, 50)
    else:
        background = cv2.bilateralFilter(background, 5, 50, 50)

    return background


def segment_page(image: np.ndarray) -> SegmentationResult:
    """
    Segment page into foreground (text) and background layers.

    Args:
        image: RGB numpy array from rasterization

    Returns:
        SegmentationResult with foreground mask, background, and metadata
    """
    height, width = image.shape[:2]

    # Detect if color page
    is_color = detect_color(image)

    # Convert to grayscale for thresholding
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Boost contrast before thresholding
    gray_boosted = boost_contrast(gray)

    # Apply adaptive thresholding
    binary, coverage = adaptive_threshold(gray_boosted)

    # Check if segmentation succeeded
    segmentation_ok = MIN_FOREGROUND_COVERAGE <= coverage <= MAX_FOREGROUND_COVERAGE

    if not segmentation_ok:
        logger.warning(
            f"Segmentation may have failed: {coverage*100:.2f}% foreground "
            f"(expected {MIN_FOREGROUND_COVERAGE*100:.0f}-{MAX_FOREGROUND_COVERAGE*100:.0f}%)"
        )

    # Repair broken characters
    binary = repair_characters(binary)

    # Recalculate coverage after repair
    coverage = np.sum(binary > 0) / (height * width)

    # Create background layer
    if is_color:
        background = create_background(image, binary, is_color=True)
    else:
        # Convert to grayscale for background
        background = create_background(gray, binary, is_color=False)

    logger.info(
        f"Segmentation: is_color={is_color}, "
        f"foreground={coverage*100:.1f}%, "
        f"ok={segmentation_ok}"
    )

    return SegmentationResult(
        foreground_mask=binary,
        background=background,
        is_color=is_color,
        foreground_coverage=coverage,
        segmentation_ok=segmentation_ok
    )


def create_fallback_single_layer(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Create a single-layer fallback when MRC segmentation fails.

    Returns the image (possibly converted to grayscale) and is_color flag.
    """
    is_color = detect_color(image)

    if is_color:
        return image, True
    else:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), False
        return image, False
