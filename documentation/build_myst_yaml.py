#!/usr/bin/env python3
"""
Build ``myst.yml`` from a template and a generated API table of contents.

The template (``myst.template.yml``) holds a single placeholder line,
``# <<PYTHON_API_TOC>>``. This script replaces that line with the contents of
the generated snippet ``_toc.yml.inc`` and writes the result to ``myst.yml``.

Run it from the repository root so the default relative paths resolve::

    python documentation/build_myst_yaml.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PYTHON_PLACEHOLDER = "# <<PYTHON_API_TOC>>"
"""Line in the template that the generated Python API TOC replaces."""


def load_text(path: Path) -> str:
    """
    Read a file as UTF-8 text.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    str
        The file contents.
    """
    return path.read_text(encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    """
    Write text to a file as UTF-8, overwriting any existing content.

    Parameters
    ----------
    path : Path
        File to write.
    text : str
        Contents to write.
    """
    path.write_text(text, encoding="utf-8")


def replace_placeholder(template_text: str, placeholder: str, replacement: str) -> str:
    """
    Replace the first line containing only ``placeholder`` with ``replacement``.

    The match covers the whole line, including leading and trailing spaces or
    tabs and the line ending. The replacement ends with exactly one newline.

    Parameters
    ----------
    template_text : str
        Template text that contains the placeholder line.
    placeholder : str
        Literal placeholder text. The function escapes it, so regex
        characters match literally.
    replacement : str
        Text to insert in place of the placeholder line.

    Returns
    -------
    str
        The template text with the first placeholder line replaced.

    Raises
    ------
    RuntimeError
        If no line in the template matches the placeholder.
    """
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
    """
    Parse command-line arguments and write the output file.

    The command line accepts three options. ``--template`` names the template
    file and defaults to ``documentation/myst.template.yml``.
    ``--python-snippet`` names the generated API TOC snippet and defaults to
    ``documentation/book/src/_generated/api/_toc.yml.inc``. ``--output``
    names the file to write and defaults to ``documentation/myst.yml``.

    Returns
    -------
    int
        Process exit code. Always 0 on success.

    Raises
    ------
    FileNotFoundError
        If the TOC snippet file does not exist.
    RuntimeError
        If the template lacks the placeholder line.
    """
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
