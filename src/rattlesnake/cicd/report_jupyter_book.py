#!/usr/bin/env python3
"""
Jupyter Book Metadata Generator

This module updates the Jupyter Book configuration file (myst.yml) with
CI/CD metadata (timestamp, branch, commit hash).
"""

import argparse
import sys
from rattlesnake.cicd.utilities import (
    get_multiline_timestamp,
    ReportMetadata,
)


def generate_footer_md(
    metadata: ReportMetadata,
) -> str:
    """
    Generate a Markdown snippet with CI/CD metadata.

    Args:
        metadata: CI/CD metadata

    Returns:
        Markdown string formatted for the primary_sidebar_footer block
    """
    # Use 6-space indentation as found in myst.yml for the block content
    indent: str = "      "
    ts_lines = get_multiline_timestamp(metadata.timestamp)

    # Use HTML links instead of Markdown links since we are inside a <div>
    github_url = f"https://github.com/{metadata.github_repo}"
    run_url = f"{github_url}/actions/runs/{metadata.run_id}"
    branch_url = f"{github_url}/tree/{metadata.ref_name}"
    commit_url = f"{github_url}/commit/{metadata.github_sha}"

    return (
        f"\n"
        f"{indent}---\n"
        f'{indent}<div style="font-size: 0.7em;">\n'
        f"{indent}{ts_lines[0]}<br>\n"
        f"{indent}&nbsp;&nbsp;{ts_lines[1]}<br>\n"
        f"{indent}&nbsp;&nbsp;{ts_lines[2]}<br>\n"
        f"{indent}&nbsp;&nbsp;{ts_lines[3]}<br>\n"
        f'{indent}Run ID: <a href="{run_url}">{metadata.run_id}</a><br>\n'
        f'{indent}Branch: <a href="{branch_url}">{metadata.ref_name}</a><br>\n'
        f'{indent}Commit: <a href="{commit_url}">{metadata.github_sha[:7]}</a><br>\n'
        f"{indent}</div>\n"
    )


def update_myst_file(file_path: str, footer_md: str) -> None:
    """
    Append metadata footer to the myst.yml file.

    Args:
        file_path: Path to the myst.yml file
        footer_md: Markdown snippet to append

    Raises:
        FileNotFoundError: If myst.yml is not found
        IOError: If writing to the file fails
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "primary_sidebar_footer: |" not in content:
            print(f"Warning: 'primary_sidebar_footer: |' not found in {file_path}")
            return

        # Simple append to the end of the file since primary_sidebar_footer is the last part
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(footer_md)

    except FileNotFoundError as e:
        raise FileNotFoundError(f'File not found: "{file_path}"') from e
    except IOError as e:
        raise IOError(f'Error updating file "{file_path}": {e}') from e


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Inject CI/CD metadata into myst.yml",
    )
    parser.add_argument("--myst_file", required=True, help="Path to myst.yml")
    parser.add_argument(
        "--timestamp", required=True, help="UTC timestamp, e.g., 20240101_120000_UTC"
    )
    parser.add_argument("--run_id", required=True, help="GitHub Actions run ID")
    parser.add_argument("--ref_name", required=True, help="Git branch name")
    parser.add_argument("--github_sha", required=True, help="GitHub commit SHA")
    parser.add_argument("--github_repo", required=True, help="GitHub repository name")

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args: argparse.Namespace = parse_arguments()

    metadata = ReportMetadata(
        timestamp=args.timestamp,
        run_id=args.run_id,
        ref_name=args.ref_name,
        github_sha=args.github_sha,
        github_repo=args.github_repo,
    )

    try:
        footer_md: str = generate_footer_md(metadata)
        update_myst_file(args.myst_file, footer_md)
        print(f"[OK] Successfully updated Jupyter Book metadata in {args.myst_file}")
    except (FileNotFoundError, IOError) as e:
        print(f"[X] File Error: {e}")
        return 1
    except ValueError as e:  # Catch potential parsing errors
        print(f"[X] Input Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
