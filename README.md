# PDF Optimizer - Rasterize & Compress

A Python-based PDF compression tool that rasterizes pages and compresses them
aggressively. Designed for **scanned documents** or PDFs that have already been
processed/OCR'd elsewhere. Target: ~10 KB per page.

## Features

- **Full rasterization**: Every page becomes a single compressed image (no vectors, fonts, or layers)
- **JPEG compression**: Color and grayscale pages compressed as JPEG
- **CCITT G4 compression**: Optional 1-bit fax encoding for B&W pages (massive size reduction)
- **OCR preservation**: Extracts existing OCR text from the source PDF and re-renders it as an invisible Helvetica layer
- **Grayscale detection**: Pages with low color saturation automatically convert to grayscale JPEG (smaller than RGB)
- **Parallel processing**: Multi-threaded page processing for faster batch operations

## Installation

### Python Dependencies

```bash
pip install -r requirements.txt
```

### External Dependencies

- [pikepdf](https://github.com/pikepdf/pikepdf) - PDF manipulation
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - PDF rasterization and OCR extraction
- [OpenCV](https://opencv.org/) - Image processing
- [Pillow](https://python-pillow.org/) - Image encoding
- [img2pdf](https://github.com/josch/img2pdf) - CCITT G4 encoding (optional, for `-g` flag)

## Quick Start

```bash
# Basic optimization (JPEG, 200 DPI, quality 75)
python optimize_pdf.py scan.pdf -o compressed.pdf

# Use CCITT G4 for B&W pages (much smaller for text-only docs)
python optimize_pdf.py scan.pdf -o compressed.pdf -g

# Lower quality for smaller files
python optimize_pdf.py scan.pdf -o compressed.pdf -q 50

# Higher DPI for better quality
python optimize_pdf.py scan.pdf -o compressed.pdf -d 300

# Parallel processing with 8 workers
python optimize_pdf.py scan.pdf -o compressed.pdf --workers 8

# Batch processing
python optimize_pdf.py *.pdf --output-dir ./compressed/

# Remove OCR layer entirely
python optimize_pdf.py scan.pdf -o compressed.pdf --ocr-remove
```

## Command-Line Options

### Input/Output
- `input` - Input PDF file(s)
- `-o, --output` - Output file (single input only)
- `--output-dir` - Output directory (for multiple files)

### Compression
- `-q, --quality` - JPEG quality 1-100 (default: 75)
- `-d, --dpi` - Render DPI 50-300 (default: 200)
- `-g, --g4` - Use CCITT G4 (1-bit) for B&W pages, JPEG for color

### OCR
- `--ocr-remove` - Remove OCR text layer (default: preserve if present)

### Processing
- `--workers` - Parallel workers (0 = auto-detect CPU count)
- `-v, --verbose` - Verbose output

## How It Works

### 1. Rasterization
Each PDF page is rasterized to an RGB image at the configured DPI (default 200).
Uses PyMuPDF for rendering.

### 2. Color Detection
The page is analyzed for color content by checking mean saturation in HSV color space.
Pages with saturation below threshold 10 are treated as grayscale.

### 3. Compression

**Without `-g` flag (default):**
- All pages compressed as JPEG
- Grayscale pages use single-channel JPEG (smaller than RGB)
- Color pages use RGB JPEG with 4:2:0 chroma subsampling

**With `-g` flag:**
- B&W pages: Otsu thresholding → 1-bit → CCITT G4 fax encoding (very small)
- Color pages: Still use JPEG

### 4. OCR Preservation
If the source PDF has OCR text, it's extracted using PyMuPDF and re-rendered
as invisible text (render mode 3) using the built-in Helvetica font. This
preserves searchability without embedding custom fonts.

### 5. PDF Assembly
pikepdf assembles pages into the output PDF with proper image XObjects.

## Typical Results

On scanned documents (black text on white):

| Mode | Size Reduction | Notes |
|------|---------------|-------|
| JPEG only | 60-80% | Works for everything |
| JPEG + G4 (`-g`) | 80-95% | Best for B&W text documents |

The `-g` flag makes a huge difference on already-B&W scans since CCITT G4
is designed specifically for fax/document images.

## Limitations

- **Rasterizes everything**: Vector graphics, fonts, and layers are flattened to images
- **No new OCR**: Only preserves existing OCR from the source PDF (does not run Tesseract)
- **Quality loss**: Lossy JPEG compression reduces image quality
- **Memory usage**: Large pages are fully rasterized into memory

## Development

### Project Structure

```
reducer/
├── optimize_pdf.py          # CLI entry point
├── requirements.txt
├── README.md
└── pdf_optimizer/
    ├── __init__.py
    ├── rasterize.py         # PDF → images (PyMuPDF)
    ├── compression.py       # JPEG/G4 encoding
    ├── pdf_writer.py        # PDF assembly + OCR extraction
    ├── pipeline.py          # Orchestration
    └── segmentation.py      # (unused, from earlier MRC approach)
```

### Extending

To add new compression methods:
1. Add encoder function to `compression.py`
2. Add XObject handling to `pdf_writer.py`
3. Wire into `compress_page()` and `PDFWriter.add_page()`


## Acknowledgments

- [pikepdf](https://github.com/pikepdf/pikepdf) - PDF manipulation
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - PDF rendering and text extraction
- [OpenCV](https://opencv.org/) - Image processing
- [img2pdf](https://github.com/josch/img2pdf) - CCITT G4 encoding
