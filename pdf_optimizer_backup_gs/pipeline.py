"""
pipeline.py - Simple PDF compression pipeline.

NO MRC. NO segmentation. NO masks. NO JBIG2.

Pipeline:
1. Rasterize page at 200 DPI
2. Compress as single JPEG
3. Wrap in PDF

Target: ~10 KB per page.
"""

import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from .rasterize import rasterize_page, get_page_count
from .compression import compress_page, CompressedPage
from .pdf_writer import PDFWriter

logger = logging.getLogger(__name__)


@dataclass
class PageStats:
    """Statistics for a processed page."""
    page_num: int
    success: bool
    error: Optional[str] = None
    process_time: float = 0.0
    compressed_size: int = 0
    is_color: bool = False


@dataclass
class OptimizationResult:
    """Result of optimizing a PDF."""
    input_path: Path
    output_path: Path
    success: bool
    error: Optional[str] = None

    page_count: int = 0
    pages_ok: int = 0
    pages_failed: int = 0

    input_size: int = 0
    output_size: int = 0
    total_time: float = 0.0

    page_stats: List[PageStats] = field(default_factory=list)

    @property
    def reduction_pct(self) -> float:
        if self.input_size == 0:
            return 0
        return (1 - self.output_size / self.input_size) * 100

    @property
    def avg_page_size(self) -> float:
        if self.page_count == 0:
            return 0
        return self.output_size / self.page_count

    def summary(self) -> str:
        return (
            f"Input:  {self.input_path.name} ({self.input_size:,} bytes)\n"
            f"Output: {self.output_path.name} ({self.output_size:,} bytes)\n"
            f"Reduction: {self.reduction_pct:.1f}%\n"
            f"Pages: {self.pages_ok}/{self.page_count}\n"
            f"Avg page size: {self.avg_page_size:,.0f} bytes\n"
            f"Time: {self.total_time:.1f}s"
        )


def process_page(
    pdf_path: Path,
    page_num: int,
    quality: int = 75,
    dpi: int = 200,
    use_g4: bool = False
) -> tuple:
    """
    Process a single page: rasterize → JPEG compress.

    No segmentation. No masks. Just compress the raster.
    """
    stats = PageStats(page_num=page_num, success=False)

    try:
        start = time.time()

        # Rasterize at specified DPI
        image, width_pts, height_pts = rasterize_page(pdf_path, page_num, dpi=dpi)

        # Compress page
        compressed = compress_page(
            image,
            page_num,
            width_pts,
            height_pts,
            quality=quality,
            target_dpi=dpi,
            current_dpi=dpi,
            use_g4=use_g4
        )

        stats.compressed_size = compressed.total_size
        stats.is_color = compressed.is_color
        stats.process_time = time.time() - start
        stats.success = True

        return stats, compressed

    except Exception as e:
        logger.error(f"Page {page_num} failed: {e}")
        stats.error = str(e)
        return stats, None


def optimize_pdf(
    input_path: Path,
    output_path: Path,
    max_workers: int = 0,
    quality: int = 75,
    dpi: int = 200,
    use_g4: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> OptimizationResult:
    """
    Optimize a PDF by rasterizing and compressing each page.

    Args:
        input_path: Input PDF
        output_path: Output PDF
        max_workers: Parallel workers (0 = auto)
        quality: JPEG quality (1-100)
        dpi: Render DPI (50-300)
        use_g4: Use CCITT G4 for B&W pages
        progress_callback: Optional callback(current, total)

    Returns:
        OptimizationResult with statistics
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    result = OptimizationResult(
        input_path=input_path,
        output_path=output_path,
        success=False
    )

    try:
        start_time = time.time()

        # Get input info
        result.input_size = input_path.stat().st_size
        result.page_count = get_page_count(input_path)

        # Auto-detect workers
        if max_workers <= 0:
            max_workers = multiprocessing.cpu_count()

        logger.info(
            f"Processing {input_path.name}: {result.page_count} pages, "
            f"{result.input_size:,} bytes, {max_workers} workers"
        )

        # Process pages
        compressed_pages = [None] * result.page_count
        page_stats = []

        if max_workers == 1:
            # Sequential processing
            for page_num in range(result.page_count):
                stats, compressed = process_page(input_path, page_num, quality, dpi, use_g4)
                page_stats.append(stats)
                if compressed:
                    compressed_pages[page_num] = compressed
                if progress_callback:
                    progress_callback(page_num + 1, result.page_count)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_page, input_path, i, quality, dpi, use_g4): i
                    for i in range(result.page_count)
                }

                completed = 0
                for future in as_completed(futures):
                    page_num = futures[future]
                    stats, compressed = future.result()
                    page_stats.append(stats)
                    if compressed:
                        compressed_pages[page_num] = compressed
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, result.page_count)

        # Sort stats by page number
        page_stats.sort(key=lambda s: s.page_num)
        result.page_stats = page_stats

        # Count successes
        result.pages_ok = sum(1 for p in compressed_pages if p is not None)
        result.pages_failed = result.page_count - result.pages_ok

        # Build output PDF
        if result.pages_ok > 0:
            writer = PDFWriter()
            for page in compressed_pages:
                if page is not None:
                    writer.add_page(page)
            writer.save(output_path)
            result.output_size = output_path.stat().st_size
            result.success = True

        result.total_time = time.time() - start_time

        logger.info(f"\n{result.summary()}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        result.error = str(e)

    return result
