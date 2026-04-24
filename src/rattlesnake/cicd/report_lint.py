"""
Lint HTML Report Generator (Functional Version)

This module extracts lint output and generates a custom HTML report.
"""

import argparse
import os
import re
import sys
from typing import List, NamedTuple, Tuple

from rattlesnake.cicd.utilities import get_multiline_timestamp, get_score_color_lint


class ReportMetadata(NamedTuple):
    """Container for CI/CD metadata used in reports."""
    timestamp: str
    run_id: str
    ref_name: str
    github_sha: str
    github_repo: str


def get_lint_content(input_file: str) -> str:
    """
    Read lint output from file.

    Args:
        input_file: Path to the lint output file

    Returns:
        Content of the lint output file
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        return f.read()


def get_lint_sections(lint_content: str) -> Tuple[List[str], List[str]]:
    """
    Parse lint output to extract issues and summary.

    Args:
        lint_content: Raw lint output content

    Returns:
        A tuple containing (issues_lines, summary_lines)
    """
    issues: List[str] = []
    summary: List[str] = []
    lines: List[str] = lint_content.split("\n")

    in_summary: bool = False

    for line in lines:
        if not line.strip():
            continue

        if "-------------------" in line or "Report" in line:
            in_summary = True

        if not in_summary:
            # Extract issues that match the lint format
            if re.match(r"^[^:]+:\d+:\d+: [A-Z]\d+: .*", line):
                issues.append(line)
        else:
            summary.append(line)

    return issues, summary


def get_score_from_summary(summary_lines: List[str]) -> str:
    """
    Extract the lint score from summary lines.

    Args:
        summary_lines: List of lines from the summary section

    Returns:
        The score as a string (e.g., "8.50")
    """
    for line in summary_lines:
        match = re.search(r"Your code has been rated at (\d+\.\d+)/10", line)
        if match:
            return match.group(1)
    return "0.00"


def get_html_header(score: str, timestamp_short: str) -> str:
    """
    Generate the HTML header and summary section.

    Args:
        score: The lint score
        timestamp_short: Timestamp string for processing

    Returns:
        HTML string for the header
    """
    color: str = get_score_color_lint(score)
    ts_lines: List[str] = get_multiline_timestamp(timestamp_short)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lint Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                         Helvetica, Arial, sans-serif;
            line-height: 1.6; color: #333; max-width: 1200px;
            margin: 0 auto; padding: 20px;
        }}
        h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .summary-card {{
            background: #f8f9fa; border-radius: 8px; padding: 20px;
            margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex; justify-content: space-between; align-items: center;
        }}
        .score-box {{ text-align: center; }}
        .score-value {{ font-size: 48px; font-weight: bold; color: {color}; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
        .issue-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        .issue-table th, .issue-table td {{
            text-align: left; padding: 12px; border-bottom: 1px solid #eee;
        }}
        .type-error {{ color: #c0392b; font-weight: bold; }}
        .type-warning {{ color: #f39c12; font-weight: bold; }}
        .type-convention {{ color: #2980b9; font-weight: bold; }}
        .type-refactor {{ color: #8e44ad; font-weight: bold; }}
        .file-link {{ color: #3498db; text-decoration: none; }}
        .file-link:hover {{ text-decoration: underline; }}
        .code-callout {{
            background: #f1f2f6;
            border-left: 5px solid #2f3542;
            padding: 15px;
            margin-bottom: 30px;
            border-radius: 0 4px 4px 0;
        }}
        .code-callout code {{
            display: block;
            background: #2f3542;
            color: #ffffff;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-family: 'Courier New', Courier, monospace;
        }}
    </style>
</head>
<body>
    <h1>Lint Analysis Report</h1>

    <div class="summary-card">
        <div>
            <h2>Summary</h2>
            <div class="timestamp">
                {ts_lines[0]}<br>
                &nbsp;&nbsp;{ts_lines[1]}<br>
                &nbsp;&nbsp;{ts_lines[2]}<br>
                &nbsp;&nbsp;{ts_lines[3]}
            </div>
        </div>
        <div class="score-box">
            <div>Global Score</div>
            <div class="score-value">{score}/10</div>
        </div>
    </div>

    <div class="code-callout">
        Want to compare your local pylint score?
        <code>pylint src/rattlesnake</code>
    </div>
"""


def get_html_issues_table(issues: List[str], metadata: ReportMetadata) -> str:
    """
    Generate the HTML table for lint issues.

    Args:
        issues: List of raw lint issue lines
        metadata: Report metadata containing repo and sha

    Returns:
        HTML string for the issues table
    """
    if not issues:
        return "<p>✅ No issues found! Great job.</p>"

    html: List[str] = [
        """
    <h2>Detailed Issues</h2>
    <table class="issue-table">
        <thead>
            <tr>
                <th>Type</th>
                <th>File</th>
                <th>Line</th>
                <th>Message</th>
            </tr>
        </thead>
        <tbody>
    """
    ]

    for issue in issues:
        # Format: path/to/file.py:line:col: C0114: ...
        match = re.match(r"^([^:]+):(\d+):(\d+): ([A-Z])\d+: (.*)$", issue)
        if match:
            file_path, line, _, category_code, message = match.groups()

            category_map = {
                "E": ("Error", "type-error"),
                "W": ("Warning", "type-warning"),
                "C": ("Convention", "type-convention"),
                "R": ("Refactor", "type-refactor"),
            }

            cat_name, cat_class = category_map.get(category_code, ("Other", ""))

            # Create GitHub link
            # Note: We assume the file_path from lint is relative to repo root
            github_base = f"https://github.com/{metadata.github_repo}/blob"
            github_url = f"{github_base}/{metadata.github_sha}/{file_path}#L{line}"

            html.append(
                f"""
            <tr>
                <td><span class="{cat_class}">{cat_name}</span></td>
                <td><a href="{github_url}" class="file-link">{file_path}</a></td>
                <td>{line}</td>
                <td>{message}</td>
            </tr>
            """
            )

    html.append("        </tbody>\n    </table>")
    return "".join(html)


def get_html_footer(metadata: ReportMetadata) -> str:
    """
    Generate the HTML footer with CI/CD metadata.

    Args:
        metadata: Report metadata containing run_id and ref_name

    Returns:
        HTML string for the footer
    """
    run_url: str = f"https://github.com/{metadata.github_repo}/actions/runs/{metadata.run_id}"
    return f"""
    <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #666; font-size: 0.8em;">
        <p>Generated by GitHub Actions Workflow<br>
        Run ID: <a href="{run_url}">{metadata.run_id}</a> | Branch: {metadata.ref_name}</p>
    </div>
</body>
</html>
"""


def generate_report(
    input_file: str,
    output_file: str,
    metadata: ReportMetadata,
) -> None:
    """
    Main function to orchestrate report generation.

    Args:
        input_file: Path to lint output
        output_file: Path for the generated HTML
        metadata: CI/CD metadata
    """
    content: str = get_lint_content(input_file)
    issues, summary = get_lint_sections(content)
    score: str = get_score_from_summary(summary)

    html: str = (
        get_html_header(score, metadata.timestamp)
        + get_html_issues_table(issues, metadata)
        + get_html_footer(metadata)
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)


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
    parser.add_argument("--output_file", required=True, help="Path for output HTML file")
    parser.add_argument("--timestamp", required=True, help="Build timestamp")
    parser.add_argument("--run_id", required=True, help="GitHub actions run ID")
    parser.add_argument("--ref_name", required=True, help="Branch/tag name")
    parser.add_argument("--github_sha", required=True, help="Full git commit SHA")
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
        generate_report(
            args.input_file,
            args.output_file,
            metadata,
        )
        print(f"[OK] Lint report generated: {args.output_file}")
    except (FileNotFoundError, IOError) as e:
        print(f"[X] File Error: {e}")
        return 1
    except ValueError as e:  # Catch potential parsing errors
        print(f"[X] Input Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
