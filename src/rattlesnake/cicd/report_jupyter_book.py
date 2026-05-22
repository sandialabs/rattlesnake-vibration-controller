#!/usr/bin/env python3
"""
Jupyter Book Metadata Generator

This module updates the Jupyter Book configuration file (myst.yml) with
CI/CD metadata (timestamp, branch, commit hash).
"""

import argparse

from rattlesnake.cicd.utilities import (
    ReportMetadata,
    add_common_args,
    get_multiline_timestamp,
    report_cli_entrypoint,
)


def generate_footer(metadata: ReportMetadata) -> str:
    """
    Construct the metadata block for myst.yml.

    Args:
        metadata: CI/CD metadata

    Returns:
        Formatted HTML/Markdown string for the footer
    """
    github_url = f"https://github.com/{metadata.github_repo}"
    run_url = f"{github_url}/actions/runs/{metadata.run_id}"
    branch_url = f"{github_url}/tree/{metadata.ref_name}"
    commit_url = f"{github_url}/commit/{metadata.github_sha}"

    ts_lines = get_multiline_timestamp(metadata.timestamp)
    indent = "      "

    return (
        f'{indent}<div style="font-size: 0.7em;">\n'
        f"{indent}{ts_lines[0]}<br>\n"
        f"{indent}&nbsp;&nbsp;{ts_lines[1]}<br>\n"
        f"{indent}&nbsp;&nbsp;{ts_lines[2]}<br>\n"
        f"{indent}&nbsp;&nbsp;{ts_lines[3]}<br>\n"
        f"{indent}&nbsp;&nbsp;{ts_lines[4]}<br>\n"
        f'{indent}Run ID: <a href="{run_url}">{metadata.run_id}</a><br>\n'
        f'{indent}Branch: <a href="{branch_url}">{metadata.ref_name}</a><br>\n'
        f'{indent}Commit: <a href="{commit_url}">{metadata.github_sha[:7]}</a><br>\n'
        f"{indent}</div>\n"
    )


def update_myst_file(myst_file: str, footer_md: str, debug: bool = False) -> None:
    """
    Read myst.yml and inject the footer.
    """
    with open(myst_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    in_footer = False
    footer_updated = False

    for line in lines:
        if "primary_sidebar_footer: |" in line:
            new_lines.append(line)
            new_lines.append(footer_md)
            footer_updated = True
            in_footer = True
            continue

        if in_footer:
            if line.strip() and not line.startswith("    "):
                in_footer = False
            # else:
            #     continue  # was discarding existing footer lines (static links),
            #                # causing CI metadata to replace rather than prepend them

        new_lines.append(line)

    if not footer_updated:
        print("Warning: 'primary_sidebar_footer: |' not found in myst.yml")
    else:
        with open(myst_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print("[OK] Successfully updated Jupyter Book metadata")
        if debug:
            print("--- myst.yml new_lines (debug) ---")
            for line in new_lines:
                print(repr(line))
            print("--- end new_lines ---")


def generate_report(args: argparse.Namespace, metadata: ReportMetadata) -> None:
    """
    Orchestrate report generation.
    """
    footer_md = generate_footer(metadata)
    extended_debug: bool = (
        True  # Set to True to print the new myst.yml content for debugging
    )
    # Suppress injection to see primary_sidebar_footer in myst.yml for testing
    update_myst_file(args.myst_file, footer_md, debug=extended_debug)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Update Jupyter Book metadata."
    )
    parser.add_argument(
        "--myst_file", required=True, help="Path to myst.yml configuration file"
    )
    add_common_args(parser)
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    return report_cli_entrypoint(parse_arguments, generate_report, exit_on_error=False)


if __name__ == "__main__":
    main()
