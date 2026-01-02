#!/usr/bin/env python3
"""
optimize_pdf.py - Size-first PDF compression CLI.

TARGET: ~10 KB per page
PHILOSOPHY: Rasterize everything, crush it, make it OCRable.

Usage:
    python optimize_pdf.py input.pdf -o output.pdf
    python optimize_pdf.py input.pdf --workers 8
    python optimize_pdf.py *.pdf --output-dir ./compressed/
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from pdf_optimizer.pipeline import optimize_pdf


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Size-first PDF compression. Target: ~10 KB per page.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_pdf.py scan.pdf -o compressed.pdf
  python optimize_pdf.py scan.pdf --workers 8
  python optimize_pdf.py *.pdf --output-dir ./out/

The output PDF will be:
  - Fully rasterized (no vectors, fonts, layers)
  - MRC compressed (JBIG2 text + JPEG background) when possible
  - Single-layer JPEG fallback when MRC fails
  - Color preserved where it exists
  - OCRable but optimized for size over appearance
"""
    )

    parser.add_argument(
        "input",
        nargs="+",
        type=Path,
        help="Input PDF file(s)"
    )

    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file (single input only)"
    )
    output.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (for multiple files)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers (0 = auto, default: auto-detect CPU count)"
    )

    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=75,
        help="JPEG quality 1-100 (default: 75)"
    )

    parser.add_argument(
        "-d", "--dpi",
        type=int,
        default=200,
        help="Render DPI 50-300 (default: 200)"
    )

    parser.add_argument(
        "-g", "--g4",
        action="store_true",
        help="Use CCITT G4 (1-bit) for B&W pages, JPEG for color"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()


def print_progress(current: int, total: int):
    """Print progress bar."""
    width = 40
    filled = int(width * current / total)
    bar = "=" * filled + "-" * (width - filled)
    pct = current / total * 100
    print(f"\r[{bar}] {current}/{total} ({pct:.0f}%)", end="", file=sys.stderr)
    if current == total:
        print(file=sys.stderr)


def main():
    args = parse_args()
    setup_logging(args.verbose)

    # Validate inputs
    valid_inputs = []
    for p in args.input:
        if not p.exists():
            print(f"Error: File not found: {p}", file=sys.stderr)
            continue
        if p.suffix.lower() != ".pdf":
            print(f"Warning: Skipping non-PDF: {p}", file=sys.stderr)
            continue
        valid_inputs.append(p)

    if not valid_inputs:
        print("Error: No valid PDF files", file=sys.stderr)
        sys.exit(1)

    # Determine output
    if len(valid_inputs) > 1:
        if args.output:
            print("Error: Use --output-dir for multiple files", file=sys.stderr)
            sys.exit(1)
        if not args.output_dir:
            args.output_dir = Path(".")

    # Process single file
    if len(valid_inputs) == 1:
        input_path = valid_inputs[0]
        output_path = args.output or input_path.with_stem(input_path.stem + "_optimized")

        result = optimize_pdf(
            input_path,
            output_path,
            max_workers=args.workers,
            quality=args.quality,
            dpi=args.dpi,
            use_g4=args.g4,
            progress_callback=print_progress
        )

        if result.success:
            print(f"\n{result.summary()}")
            sys.exit(0)
        else:
            print(f"Error: {result.error}", file=sys.stderr)
            sys.exit(1)
    else:
        # Batch processing
        args.output_dir.mkdir(parents=True, exist_ok=True)

        total_in = 0
        total_out = 0
        successes = 0

        for i, input_path in enumerate(valid_inputs):
            output_path = args.output_dir / f"{input_path.stem}_optimized.pdf"
            print(f"\n[{i+1}/{len(valid_inputs)}] {input_path.name}")

            result = optimize_pdf(
                input_path,
                output_path,
                max_workers=args.workers,
                quality=args.quality,
                dpi=args.dpi,
                use_g4=args.g4,
                progress_callback=print_progress
            )

            total_in += result.input_size
            if result.success:
                total_out += result.output_size
                successes += 1

        print(f"\n{'='*50}")
        print(f"Batch complete: {successes}/{len(valid_inputs)} files")
        print(f"Total: {total_in:,} -> {total_out:,} bytes")
        if total_in > 0:
            print(f"Reduction: {(1 - total_out/total_in)*100:.1f}%")

        sys.exit(0 if successes == len(valid_inputs) else 1)


if __name__ == "__main__":
    main()
