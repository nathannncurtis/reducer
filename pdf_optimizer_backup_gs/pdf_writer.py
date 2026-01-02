"""
pdf_writer.py - PDF assembly from compressed pages.

Supports:
- JPEG images (DCTDecode)
- 1-bit B&W images (FlateDecode)

Each page is exactly one image scaled to fill the page.
"""

import logging
import zlib
from pathlib import Path
from typing import List

import pikepdf
from pikepdf import Pdf, Stream, Dictionary, Name, Array

from .compression import CompressedPage

logger = logging.getLogger(__name__)


class PDFWriter:
    """
    Assembles JPEG pages into a minimal PDF.

    Each page contains exactly one JPEG image.
    No text layers, no masks, no layering.
    """

    def __init__(self):
        self.pdf = Pdf.new()
        self.pages: List[CompressedPage] = []

    def add_page(self, compressed: CompressedPage):
        """Add a page to the PDF."""
        # Create a blank page
        self.pdf.add_blank_page(
            page_size=(compressed.page_width_pts, compressed.page_height_pts)
        )
        page = self.pdf.pages[-1]

        if compressed.is_g4:
            # 1-bit B&W image with FlateDecode
            # Compress with zlib
            compressed_data = zlib.compress(compressed.image_data, level=9)

            image_dict = Dictionary({
                '/Type': Name.XObject,
                '/Subtype': Name.Image,
                '/Width': compressed.width,
                '/Height': compressed.height,
                '/ColorSpace': Name.DeviceGray,
                '/BitsPerComponent': 1,
                '/Filter': Name.FlateDecode,
            })
            img_stream = Stream(self.pdf, compressed_data, image_dict)
        else:
            # JPEG image
            colorspace = Name.DeviceRGB if compressed.is_color else Name.DeviceGray

            image_dict = Dictionary({
                '/Type': Name.XObject,
                '/Subtype': Name.Image,
                '/Width': compressed.width,
                '/Height': compressed.height,
                '/ColorSpace': colorspace,
                '/BitsPerComponent': 8,
                '/Filter': Name.DCTDecode,
            })
            img_stream = Stream(self.pdf, compressed.image_data, image_dict)

        # Create XObjects dict
        xobjects = Dictionary({})
        xobjects['/Im0'] = self.pdf.make_indirect(img_stream)

        # Set resources
        page.Resources = Dictionary({'/XObject': xobjects})

        # Create content stream - just draw the image scaled to page
        content = f"""
q
{compressed.page_width_pts:.4f} 0 0 {compressed.page_height_pts:.4f} 0 0 cm
/Im0 Do
Q
"""
        page.Contents = self.pdf.make_indirect(
            Stream(self.pdf, content.strip().encode("ascii"))
        )

        self.pages.append(compressed)

        mode = "1-bit" if compressed.is_g4 else ("color" if compressed.is_color else "gray")
        logger.debug(
            f"Added page {compressed.page_num}: "
            f"{compressed.total_size:,} bytes ({mode})"
        )

    def save(self, output_path: Path):
        """Save PDF to file."""
        output_path = Path(output_path)

        self.pdf.save(
            output_path,
            compress_streams=True,
            object_stream_mode=pikepdf.ObjectStreamMode.generate
        )

        logger.info(f"Saved {len(self.pages)} pages to {output_path}")

    def get_total_size(self) -> int:
        """Get total content size (before PDF overhead)."""
        return sum(p.total_size for p in self.pages)


def create_pdf(pages: List[CompressedPage], output_path: Path) -> int:
    """
    Create PDF from compressed pages.

    Returns output file size in bytes.
    """
    writer = PDFWriter()
    for page in pages:
        writer.add_page(page)
    writer.save(output_path)
    return output_path.stat().st_size
