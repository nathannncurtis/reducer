"""
pdf_writer.py - PDF assembly from compressed pages.

Supports:
- JPEG images (DCTDecode)
- 1-bit B&W images (FlateDecode)
- OCR text layer extraction and re-rendering with standard fonts
"""

import logging
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

import pikepdf
from pikepdf import Pdf, Stream, Dictionary, Name, Array

try:
    import fitz  # PyMuPDF
except ImportError:
    import pymupdf as fitz

from .compression import CompressedPage

logger = logging.getLogger(__name__)


class PDFWriter:
    """
    Assembles compressed pages into a PDF.

    Optionally extracts OCR text from original and re-renders with Helvetica.
    """

    def __init__(self, original_pdf: Optional[Path] = None):
        """
        Initialize PDF writer.

        Args:
            original_pdf: Path to original PDF for OCR text extraction
        """
        self.pdf = Pdf.new()
        self.pages: List[CompressedPage] = []
        self.original_pdf_path = None
        self.preserve_ocr = False
        self._fitz_doc = None

        if original_pdf is not None:
            self.original_pdf_path = Path(original_pdf)
            if self.original_pdf_path.exists():
                try:
                    self._fitz_doc = fitz.open(self.original_pdf_path)
                    self.preserve_ocr = True
                    logger.debug(f"Opened original PDF for OCR extraction: {original_pdf}")
                except Exception as e:
                    logger.warning(f"Could not open original PDF for OCR: {e}")

    def add_page(self, compressed: CompressedPage):
        """Add a page to the PDF, optionally preserving OCR."""
        # Create a blank page
        self.pdf.add_blank_page(
            page_size=(compressed.page_width_pts, compressed.page_height_pts)
        )
        page = self.pdf.pages[-1]

        if compressed.is_g4:
            # 1-bit B&W image with CCITT G4 (data already compressed by img2pdf)
            image_dict = Dictionary({
                '/Type': Name.XObject,
                '/Subtype': Name.Image,
                '/Width': compressed.width,
                '/Height': compressed.height,
                '/ColorSpace': Name.DeviceGray,
                '/BitsPerComponent': 1,
                '/Filter': Name.CCITTFaxDecode,
                '/DecodeParms': Dictionary({
                    '/K': -1,  # G4 encoding
                    '/Columns': compressed.width,
                    '/Rows': compressed.height,
                    '/BlackIs1': True,
                })
            })
            img_stream = Stream(self.pdf, compressed.image_data, image_dict)
        elif not compressed.is_color and len(compressed.image_data) > 0 and compressed.image_data[0:2] != b'\xff\xd8':
            # 1-bit B&W fallback with FlateDecode (zlib compression)
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

        # Create XObjects dict with our image
        xobjects = Dictionary({})
        xobjects['/Im0'] = self.pdf.make_indirect(img_stream)

        # Start with image drawing content
        image_content = f"""
q
{compressed.page_width_pts:.4f} 0 0 {compressed.page_height_pts:.4f} 0 0 cm
/Im0 Do
Q
"""

        # Try to extract and re-render OCR text from original
        ocr_content = ""
        has_ocr = False

        if self.preserve_ocr and self._fitz_doc is not None:
            try:
                if compressed.page_num < len(self._fitz_doc):
                    ocr_content = self._extract_text_as_helvetica(
                        compressed.page_num,
                        compressed.page_width_pts,
                        compressed.page_height_pts
                    )
                    has_ocr = bool(ocr_content)
            except Exception as e:
                logger.debug(f"Could not extract OCR from page {compressed.page_num}: {e}")

        # Set resources - add Helvetica font if we have OCR
        resources = Dictionary({'/XObject': xobjects})
        if has_ocr:
            # Use built-in Helvetica (no embedding needed)
            resources['/Font'] = Dictionary({
                '/F1': Dictionary({
                    '/Type': Name.Font,
                    '/Subtype': Name.Type1,
                    '/BaseFont': Name.Helvetica,
                    '/Encoding': Name.WinAnsiEncoding,
                })
            })
        page.Resources = resources

        # Combine image content + OCR content
        full_content = image_content.strip()
        if ocr_content:
            full_content = full_content + "\n" + ocr_content

        page.Contents = self.pdf.make_indirect(
            Stream(self.pdf, full_content.encode("latin-1", errors="replace"))
        )

        self.pages.append(compressed)

        mode = "1-bit" if compressed.is_g4 else ("color" if compressed.is_color else "gray")
        ocr_status = "+OCR" if has_ocr else ""
        logger.debug(
            f"Added page {compressed.page_num}: "
            f"{compressed.total_size:,} bytes ({mode}{ocr_status})"
        )

    def _extract_text_as_helvetica(
        self,
        page_num: int,
        page_width: float,
        page_height: float
    ) -> str:
        """
        Extract text from original PDF and generate invisible Helvetica text layer.

        Uses PyMuPDF to get text with positions, then creates PDF content
        stream with invisible text (render mode 3) using Helvetica font.
        """
        fitz_page = self._fitz_doc[page_num]

        # Get text as dictionary with position info
        text_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        # Build content stream with invisible text
        content_parts = []
        content_parts.append("BT")  # Begin text
        content_parts.append("3 Tr")  # Render mode 3 = invisible
        content_parts.append("/F1 1 Tf")  # Helvetica (size set via Tm matrix)

        # Get original page dimensions for coordinate mapping
        orig_rect = fitz_page.rect
        scale_x = page_width / orig_rect.width if orig_rect.width > 0 else 1
        scale_y = page_height / orig_rect.height if orig_rect.height > 0 else 1

        word_count = 0
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # Skip non-text blocks
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    # Get position and size
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    font_size = span.get("size", 10)

                    # Scale coordinates to match our page size
                    x = bbox[0] * scale_x
                    # PDF y-origin is bottom, fitz is top
                    y = page_height - (bbox[3] * scale_y)
                    scaled_size = font_size * scale_y

                    # Escape special PDF characters
                    escaped_text = self._escape_pdf_text(text)

                    # Position and render text using text matrix (more compact)
                    content_parts.append(f"{scaled_size:.1f} 0 0 {scaled_size:.1f} {x:.1f} {y:.1f} Tm ({escaped_text}) Tj")

                    word_count += 1

        content_parts.append("ET")  # End text

        if word_count == 0:
            return ""

        logger.debug(f"Page {page_num}: extracted {word_count} text spans")
        return "\n".join(content_parts)

    def _escape_pdf_text(self, text: str) -> str:
        """Escape special characters for PDF string literal."""
        # Handle characters that need escaping in PDF strings
        result = text.replace("\\", "\\\\")
        result = result.replace("(", "\\(")
        result = result.replace(")", "\\)")
        # Filter to WinAnsiEncoding compatible characters
        # Replace non-encodable chars with space
        filtered = ""
        for c in result:
            try:
                c.encode("latin-1")
                filtered += c
            except UnicodeEncodeError:
                filtered += " "
        return filtered

    def save(self, output_path: Path):
        """Save PDF to file."""
        output_path = Path(output_path)

        # Close fitz document if open
        if self._fitz_doc is not None:
            self._fitz_doc.close()
            self._fitz_doc = None

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
