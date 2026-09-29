#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

PYTHON_PLACEHOLDER = "# <<PYTHON_API_TOC>>"

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

def replace_placeholder(template_text: str, placeholder: str, replacement: str) -> str:
    pattern = rf"^[ \t]*{re.escape(placeholder)}[ \t]*\r?\n?"
    if not re.search(pattern, template_text, flags=re.MULTILINE):
        raise RuntimeError(f"Placeholder not found in template: {placeholder}")
    return re.sub(
        pattern,
        replacement.rstrip() + "\n",
        template_text,
        count=1,
        flags=re.MULTILINE,
    )

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build myst.yml from myst.template.yml and generated TOC snippets."
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("documentation/myst.template.yml"),
    )
    parser.add_argument(
        "--python-snippet",
        type=Path,
        default=Path("documentation/book/src/_generated/api/_toc.yml.inc"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("documentation/myst.yml"),
    )
    args = parser.parse_args()

    template_text = load_text(args.template)

    if not args.python_snippet.is_file():
        raise FileNotFoundError(f"Python TOC snippet not found: {args.python_snippet}")

    python_snippet = load_text(args.python_snippet)
    output_text = replace_placeholder(template_text, PYTHON_PLACEHOLDER, python_snippet)

    save_text(args.output, output_text)
    print(f"Wrote {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())