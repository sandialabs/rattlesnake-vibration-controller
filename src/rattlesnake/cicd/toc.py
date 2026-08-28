#!/usr/bin/env python3
"""
MyST Table of Contents Validator

Jupyter Book's ``--strict`` flag does not fail a build when a table of contents
entry is missing.  MyST records that error against ``myst.yml`` itself.  The
strict check only inspects errors recorded against pages that made it into the
project, so it never sees them.  MyST prints the errors and then exits zero.

This module checks the table of contents directly.  Run it before the book
build so CI fails on missing content:

    python src/rattlesnake/cicd/toc.py --myst_file documentation/myst.yml
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Final, NamedTuple, Optional

import yaml

# Extensions MyST appends when a table of contents entry omits one.
VALID_FILE_EXTENSIONS: Final[tuple[str, ...]] = (
    ".md",
    ".ipynb",
    ".tex",
    ".myst.json",
)


class Entry(NamedTuple):
    """A single table of contents entry."""

    value: str
    is_pattern: bool


def load(*, myst_file: str) -> tuple[Path, list]:
    """
    Read the table of contents from a MyST configuration file.

    Args:
        myst_file: Path to the myst.yml configuration file.

    Returns:
        A tuple of the directory entries resolve against and the raw
        table of contents items.

    Raises:
        ValueError: If the file has no usable table of contents.
    """
    path: Path = Path(myst_file)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"{myst_file} does not contain a YAML mapping")

    project = config.get("project") or {}
    items = project.get("toc", config.get("toc"))

    if items is None:
        raise ValueError(f"No table of contents found in {myst_file}")
    if not isinstance(items, list):
        raise ValueError(f"Table of contents in {myst_file} is not a list")

    return path.parent, items


def entries(*, items: Any) -> list[Entry]:
    """
    Flatten a table of contents into its file and pattern entries.

    Entries that name neither a file nor a pattern, such as external URLs and
    title-only groupings, contribute nothing themselves.  Their children are
    still walked.

    Args:
        items: A table of contents list, or None.

    Returns:
        Every file and pattern entry, in document order.
    """
    collected: list[Entry] = []

    for item in items or []:
        if not isinstance(item, dict):
            continue
        if "file" in item:
            collected.append(Entry(value=str(item["file"]), is_pattern=False))
        elif "pattern" in item:
            collected.append(Entry(value=str(item["pattern"]), is_pattern=True))
        collected.extend(entries(items=item.get("children")))

    return collected


def _matches_unique(*, base_dir: Path, value: str) -> list[str]:
    """
    Find the files MyST would infer for an extensionless entry.

    Matches are deduplicated case-insensitively, because a case-insensitive
    filesystem reports both ``page.md`` and ``page.MD`` for the same file.

    Args:
        base_dir: Directory the entry resolves against.
        value: The entry as written in the table of contents.

    Returns:
        The unique candidate file names.
    """
    unique: dict[str, str] = {}

    for extension in VALID_FILE_EXTENSIONS:
        for candidate in (f"{value}{extension}", f"{value}{extension.upper()}"):
            if (base_dir / candidate).exists():
                unique.setdefault(candidate.lower(), candidate)

    return list(unique.values())


def pattern_check(*, base_dir: Path, value: str) -> Optional[str]:
    """
    Check that a table of contents pattern matches at least one file.

    Args:
        base_dir: Directory the pattern resolves against.
        value: The pattern as written in the table of contents.

    Returns:
        A problem description, or None if the pattern matches.
    """
    if any(base_dir.glob(value)):
        return None
    return f"Pattern from table of contents did not match any files: {value}"


def file_check(*, base_dir: Path, value: str) -> Optional[str]:
    """
    Check that a table of contents file entry resolves to a page.

    The rules mirror MyST's own resolution so this reports the same problems
    MyST reports, rather than a stricter or looser set.

    Args:
        base_dir: Directory the entry resolves against.
        value: The entry as written in the table of contents.

    Returns:
        A problem description, or None if the entry resolves.
    """
    target: Path = base_dir / value

    if target.exists():
        if target.is_dir():
            return f"Folder referenced as file in table of contents: {value}"
        return None

    matches: list[str] = _matches_unique(base_dir=base_dir, value=value)

    if len(matches) > 1:
        return f"Multiple files match table of contents entry: {value}"
    if matches:
        return None
    if Path(value).suffix:
        return f"Table of contents entry does not exist: {value}"
    return f"Unable to resolve table of contents entry: {value}"


def entry_check(*, base_dir: Path, entry: Entry) -> Optional[str]:
    """
    Check that a single table of contents entry resolves to content.

    Args:
        base_dir: Directory the entry resolves against.
        entry: The entry to check.

    Returns:
        A problem description, or None if the entry resolves.
    """
    if entry.is_pattern:
        return pattern_check(base_dir=base_dir, value=entry.value)
    return file_check(base_dir=base_dir, value=entry.value)


def validate(*, myst_file: str) -> list[str]:
    """
    Check every table of contents entry in a MyST configuration file.

    Args:
        myst_file: Path to the myst.yml configuration file.

    Returns:
        A problem description for each entry that does not resolve.  An empty
        list means the table of contents is sound.
    """
    base_dir, items = load(myst_file=myst_file)

    problems: list[str] = []
    for entry in entries(items=items):
        problem = entry_check(base_dir=base_dir, entry=entry)
        if problem is not None:
            problems.append(problem)

    return problems


def report(*, myst_file: str) -> int:
    """
    Validate a table of contents and print the outcome.

    On GitHub Actions each problem is also emitted as an error annotation so it
    surfaces in the workflow summary.

    Args:
        myst_file: Path to the myst.yml configuration file.

    Returns:
        Exit code, 0 when every entry resolves and 1 otherwise.
    """
    try:
        problems: list[str] = validate(myst_file=myst_file)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"[X] Table of contents check failed: {error}")
        return 1

    if not problems:
        print(f"[OK] All table of contents entries resolve in {myst_file}")
        return 0

    annotate: bool = os.environ.get("GITHUB_ACTIONS") == "true"
    print(f"[X] {len(problems)} table of contents problem(s) in {myst_file}:")
    for problem in problems:
        print(f"    - {problem}")
        if annotate:
            print(f"::error file={myst_file}::{problem}")

    return 1


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Validate the MyST table of contents."
    )
    parser.add_argument(
        "--myst_file", required=True, help="Path to myst.yml configuration file"
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    return report(myst_file=args.myst_file)


if __name__ == "__main__":
    sys.exit(main())
