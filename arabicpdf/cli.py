"""Command-line interface for ArabicPDF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arabicpdf import __version__, convert, to_text, Config, Mode
from arabicpdf.detector import classify_document


def _parse_pages(pages_str: str | None) -> list[int] | None:
    """Parse a page range string like '1,3,5-7' into a sorted list of ints."""
    if not pages_str:
        return None
    pages: set[int] = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s.strip()), int(end_s.strip())
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return sorted(pages)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="arabicpdf",
        description="Convert any Arabic PDF to clean text locally with high accuracy.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: convert
    convert_parser = subparsers.add_parser(
        "convert", help="Extract text from an Arabic PDF"
    )
    convert_parser.add_argument(
        "pdf_path", type=str, help="Path to input Arabic PDF"
    )
    convert_parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output text file path (default: stdout)"
    )
    convert_parser.add_argument(
        "-m",
        "--mode",
        choices=["fast", "balanced", "max_accuracy"],
        default="balanced",
        help="Extraction mode (default: balanced)",
    )
    convert_parser.add_argument(
        "-p", "--pages", type=str, default=None, help="Page numbers or ranges (e.g. '1,3,5-8')"
    )
    convert_parser.add_argument(
        "--dpi", type=int, default=300, help="Rendering DPI for OCR (default: 300)"
    )
    convert_parser.add_argument(
        "--strip-diacritics",
        action="store_true",
        help="Remove Arabic vowels/harakat from output",
    )
    convert_parser.add_argument(
        "--normalize-alef",
        action="store_true",
        help="Normalize all alef forms (أ/إ/آ) to bare alef (ا)",
    )
    convert_parser.add_argument(
        "--no-spell-check",
        action="store_true",
        help="Disable dictionary-based spell and dot correction",
    )
    convert_parser.add_argument(
        "--details",
        action="store_true",
        help="Print detailed page-level extraction metadata",
    )

    # Command: inspect / info
    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect PDF page types (digital vs scanned vs corrupted)"
    )
    inspect_parser.add_argument(
        "pdf_path", type=str, help="Path to input Arabic PDF"
    )

    args = parser.parse_args(argv)

    if not args.command:
        # Default behavior: if a single positional argument is passed without subcommand
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            args.command = "convert"
            args.pdf_path = sys.argv[1]
            args.output = None
            args.mode = "balanced"
            args.pages = None
            args.dpi = 300
            args.strip_diacritics = False
            args.normalize_alef = False
            args.no_spell_check = False
            args.details = False
        else:
            parser.print_help()
            return 1

    if args.command == "inspect":
        pdf_file = Path(args.pdf_path)
        if not pdf_file.is_file():
            sys.stderr.write(f"Error: File not found: {args.pdf_path}\n")
            return 1
        print(f"Inspecting '{args.pdf_path}'...\n")
        classifications = classify_document(str(pdf_file))
        for c in classifications:
            print(
                f"Page {c.page_num:3d}: Type={c.page_type.value:18s} "
                f"Confidence={c.confidence*100:5.1f}%  HasImages={str(c.has_images):5s}"
            )
        return 0

    if args.command == "convert":
        pdf_file = Path(args.pdf_path)
        if not pdf_file.is_file():
            sys.stderr.write(f"Error: File not found: {args.pdf_path}\n")
            return 1

        pages_list = _parse_pages(args.pages)
        cfg = Config(
            mode=Mode(args.mode),
            pages=pages_list,
            preserve_diacritics=not args.strip_diacritics,
            normalize_alef=args.normalize_alef,
            fix_spelling=not args.no_spell_check,
            fix_dot_confusion=not args.no_spell_check,
            render_dpi=args.dpi,
        )

        result = convert(str(pdf_file), config=cfg)

        if args.details:
            sys.stderr.write(f"Processed {result.page_count} page(s). Document confidence: {result.confidence*100:.1f}%\n")
            for p in result.pages:
                flags_msg = f" | Flags: {', '.join(p.flags)}" if p.flags else ""
                sys.stderr.write(
                    f" - Page {p.number}: Method={p.method} Conf={p.confidence*100:.1f}%{flags_msg}\n"
                )

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result.text, encoding="utf-8")
            sys.stderr.write(f"Saved extracted text to: {args.output}\n")
        else:
            sys.stdout.write(result.text)
            if not result.text.endswith("\n"):
                sys.stdout.write("\n")

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
