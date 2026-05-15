"""
Lint HTML Report Generator (Functional Version)

This module extracts lint output and generates a custom HTML report.
"""

import argparse
import os
import re

from rattlesnake.cicd.utilities import (
    ReportMetadata,
    add_common_args,
    get_html_report_footer,
    get_html_report_header,
    get_score_color_lint,
    report_cli_entrypoint,
    write_report,
)


def get_lint_content(input_file: str) -> str:
    """
    Read lint output from file.

    Args:
        input_file: Path to the lint output file

    Returns:
        Content of the lint output file
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The input file '{input_file}' was not found.")
    with open(input_file, "r", encoding="utf-8") as f:
        return f.read()


def get_lint_sections(content: str) -> tuple[list[str], list[str]]:
    """
    Parse lint output into issues and summary sections.

    Args:
        content: The lint output content

    Returns:
        A tuple of (issues, summary_lines)
    """
    # Split by any sequence of 10 or more dashes (common in pylint output)
    sections = re.split(r"-{10,}", content)

    # Issues are almost always before the first separator
    issues = sections[0].strip().split("\n")
    issues = [i for i in issues if i.strip()]

    # If there are no separators, the whole file might be issues or just a score
    if len(sections) == 1:
        lines = content.split("\n")
        return lines, lines

    # For summary, take everything after the first separator
    summary_text: str = "\n".join(sections[1:])
    summary = summary_text.split("\n")

    return issues, summary


def get_score_from_summary(summary_lines: list[str]) -> str:
    """
    Extract lint score from summary.

    Args:
        summary_lines: Lines from the summary section

    Returns:
        The extracted score as a string
    """
    score_pattern = r"Your code has been rated at (\d+\.\d+)/10"
    for line in summary_lines:
        match = re.search(score_pattern, line)
        if match:
            return match.group(1)
    return "0.00"


def get_html_issues_table(issues: list[str], metadata: ReportMetadata) -> str:
    """
    Generate issues table.

    Args:
        issues: List of lint issues
        metadata: CI/CD metadata

    Returns:
        HTML table string
    """
    if not issues:
        return '<div class="section"><p>No issues found! Great job.</p></div>'

    table_html = """
        <div class="section">
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Line:Col</th>
                    <th>Type</th>
                    <th>Message</th>
                </tr>
            </thead>
            <tbody>"""

    msg_types = {
        "C": "Convention",
        "R": "Refactor",
        "W": "Warning",
        "E": "Error",
        "F": "Fatal",
    }

    # Pattern: path/to/file.py:line:col: TYPE: Message
    pattern = r"(.+):(\d+):(\d+): ([A-Z])\d+: (.+)"

    for issue in issues:
        match = re.search(pattern, issue)
        if match:
            file_path, line, col, type_code, text = match.groups()
            type_name = msg_types.get(type_code, type_code)

            # GitHub direct link
            file_url = (
                f"https://github.com/{metadata.github_repo}/blob/"
                f"{metadata.github_sha}/{file_path}#L{line}"
            )

            row_color = ""
            if type_code == "E":
                row_color = ' style="background-color: #ffeef0"'
            elif type_code == "W":
                row_color = ' style="background-color: #fff5b1"'

            table_html += f"""
                <tr{row_color}>
                    <td><a href="{file_url}">{file_path}</a></td>
                    <td>{line}:{col}</td>
                    <td>{type_name}</td>
                    <td>{text}</td>
                </tr>"""

    table_html += """
            </tbody>
        </table>
        </div>"""
    return table_html


def generate_report(args: argparse.Namespace, metadata: ReportMetadata) -> None:
    """
    Orchestrate report generation.

    Args:
        args: Parsed command line arguments
        metadata: CI/CD metadata
    """
    content = get_lint_content(args.input_file)
    issues, summary = get_lint_sections(content)
    score = get_score_from_summary(summary)

    score_color = get_score_color_lint(score)
    html = get_html_report_header(
        title="Lint Report",
        score_label="Pylint Score",
        score_value=f"{score}/10",
        score_color=score_color,
        metadata=metadata,
    )
    html += get_html_issues_table(issues, metadata)
    html += get_html_report_footer()

    write_report(html, args.output_file)
    print(f"[OK] Lint report generated: {args.output_file}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate a custom HTML report from lint output.",
    )
    parser.add_argument("--input_file", required=True, help="Path to lint output file")
    parser.add_argument(
        "--output_file", required=True, help="Path for output HTML file"
    )

    add_common_args(parser)
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    return report_cli_entrypoint(parse_arguments, generate_report, exit_on_error=False)


if __name__ == "__main__":
    main()
