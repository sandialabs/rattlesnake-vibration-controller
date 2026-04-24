"""
Lint HTML Report Generator (Functional Version)

This module extracts lint output and generates a custom HTML report.
"""

import argparse
import os
import re
import sys

from rattlesnake.cicd.utilities import (
    ReportMetadata,
    add_common_args,
    extend_timestamp,
    get_score_color_lint,
    report_main_runner,
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
    sections = content.split("-------------------")
    issues = sections[0].strip().split("\n")
    issues = [i for i in issues if i.strip()]
    summary = sections[1].split("\n") if len(sections) > 1 else []
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


def get_html_header(score: str, metadata: ReportMetadata) -> str:
    """
    Generate the HTML header.

    Args:
        score: The pylint score
        metadata: CI/CD metadata

    Returns:
        HTML header string
    """
    score_color = get_score_color_lint(score)
    timestamp_ext = extend_timestamp(metadata.timestamp)

    # Pre-calculate GitHub URLs
    github_url = f"https://github.com/{metadata.github_repo}"
    run_url = f"{github_url}/actions/runs/{metadata.run_id}"
    branch_url = f"{github_url}/tree/{metadata.ref_name}"
    commit_url = f"{github_url}/commit/{metadata.github_sha}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lint Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f6f8fa; line-height: 1.6;
            background: lightgray;
        }}
        .container {{
            max-width: 1200px; margin: 0 auto;
        }}
        .header {{
            background: white; padding: 30px; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px;
        }}
        .score {{
            font-size: 2.5em; font-weight: bold; color: {score_color};
        }}
        .metadata {{
            color: #6a737d; font-size: 0.9em; margin-top: 10px;
        }}
        .nav {{
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px;
        }}
        .nav a {{
            background: #0366d6; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 6px; margin-right: 10px;
            display: inline-block; margin-bottom: 5px;
        }}
        .nav a:hover {{
            background: #0256cc;
        }}
        .section {{
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px;
        }}
        table {{
            width: 100%; border-collapse: collapse; background: #fff;
            border: 1px solid #e1e4e8; border-radius: 6px;
        }}
        th {{
            text-align: left; padding: 12px; background: #f6f8fa;
            border-bottom: 1px solid #e1e4e8;
        }}
        td {{ padding: 12px; border-bottom: 1px solid #e1e4e8; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Lint Report</h1>
            <div class="score">Pylint Score: {score}/10</div>
            <div class="metadata">
                <div>&nbsp;</div>
                <div><strong>Generated:</strong> {timestamp_ext}</div>
                <div><strong>Run ID:</strong>
                    <a href="{run_url}">{metadata.run_id}</a></div>
                <div><strong>Branch:</strong>
                    <a href="{branch_url}">{metadata.ref_name}</a></div>
                <div><strong>Commit:</strong>
                    <a href="{commit_url}">{metadata.github_sha[:7]}</a></div>
                <div><strong>Repository:</strong>
                    <a href="{github_url}">{metadata.github_repo}</a></div>
            </div>
        </div>

        <div class="nav">
            <a href="../../index.html">← Back to Dashboard</a>
        </div>
"""


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


def get_html_footer() -> str:
    """
    Generate the HTML footer.

    Returns:
        HTML footer string
    """
    return """
    </div>
    <footer style="text-align: center; margin: 40px 0; color: #6a737d;">
        <p>Generated by GitHub Actions</p>
    </footer>
</body>
</html>"""


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

    html = get_html_header(score, metadata)
    html += get_html_issues_table(issues, metadata)
    html += get_html_footer()

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(html)
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
    args: argparse.Namespace = parse_arguments()
    return report_main_runner(generate_report, args)


if __name__ == "__main__":
    sys.exit(main())
