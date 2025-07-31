#!/usr/bin/env python3
"""
Pylint HTML Report Generator (Functional Version)

This module extracts pylint output and generates a custom HTML report.
"""

import os
import html
import re
import argparse
import sys
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import pytz


def get_score_color(pylint_score: str) -> str:
    """
    Determine color based on pylint score.

    Args:
        pylint_score: The pylint score as string

    Returns:
        Hex color code for the score
    """
    # TODO: Merge this with same function in pylint_badge_color.py
    try:
        score_val: float = float(pylint_score)
        if score_val >= 8.0:
            return "brightgreen"  # #28a745
        elif score_val >= 6.0:
            return "yellow"  # #ffc107
        elif score_val >= 4.0:
            return "orange"  # #dc3545
        else:
            return "red"  # #dc3545
    except ValueError:
        return "gray"  # #6c757d for invalid scores


def get_formatted_timestamp() -> str:
    """
    Get formatted timestamp with UTC, EST, and MST times.

    Returns:
        Formatted timestamp string
    """
    # Get the current UTC time
    utc_now: datetime = datetime.now(pytz.utc)

    # Define the time zones
    est: pytz.BaseTzInfo = pytz.timezone("America/New_York")
    mst: pytz.BaseTzInfo = pytz.timezone("America/Denver")

    # Convert UTC time to EST and MST
    est_now: datetime = utc_now.astimezone(est)
    mst_now: datetime = utc_now.astimezone(mst)

    # Format the output
    formatted_time: str = (
        utc_now.strftime("%Y-%m-%d %H:%M:%S UTC")
        + f" ({est_now.strftime('%Y-%m-%d %H:%M:%S EST')} / {mst_now.strftime('%Y-%m-%d %H:%M:%S MST')})"
    )

    return formatted_time


def read_pylint_content(input_file: str) -> str:
    """
    Read pylint output from file.

    Args:
        input_file: Path to the pylint output file

    Returns:
        Content of the pylint output file

    Raises:
        SystemExit: If file cannot be read
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f'Error: Input file "{input_file}" not found.')
        sys.exit(1)
    except IOError as e:
        print(f'Error reading input file "{input_file}": {e}')
        sys.exit(1)


def parse_pylint_output(pylint_content: str) -> Tuple[List[str], List[str]]:
    """
    Parse pylint output to extract issues and summary.

    Args:
        pylint_content: Raw pylint output content

    Returns:
        Tuple of (issues_list, summary_lines)
    """
    lines: List[str] = pylint_content.split("\n")
    issues: List[str] = []
    summary_started: bool = False
    summary_lines: List[str] = []

    for line in lines:
        if line.startswith("************* Module"):
            _: str = line.replace("************* Module ", "")
        # elif ":" in line and any(
        #     # codes: error, warning, convention, refactor
        #     x in line
        #     # for x in ["error", "warning", "convention", "refactor"]
        #     for x in ["error", "warning", convention_code, "refactor"]
        # ):
        # extract any line with codes that match convention, warning, error, or refactor,
        # e.g., C0114:, W0611:, E1101:, or R0913:
        elif re.search(r"\b[CWER]\d{4}:", line):
            # Extract issues that match the pylint format
            issues.append(line)
        elif summary_started or "Your code has been rated at" in line:
            summary_started = True
            summary_lines.append(line)

    return issues, summary_lines


def count_issues_by_type(issues: List[str]) -> Dict[str, int]:
    """
    Count issues by type (error, warning, convention).

    Args:
        issues: List of pylint issue strings

    Returns:
        Dictionary with counts for each issue type
    """
    # error_count: int = len([i for i in issues if "error" in i.lower()])
    # warning_count: int = len([i for i in issues if "warning" in i.lower()])
    # convention_count: int = len([i for i in issues if "convention" in i.lower()])
    convention_count: int = len([i for i in issues if re.search(r"C\d{4}:", i)])
    warning_count: int = len([i for i in issues if re.search(r"W\d{4}:", i)])
    error_count: int = len([i for i in issues if re.search(r"E\d{4}:", i)])
    refactor_count: int = len([i for i in issues if re.search(r"R\d{4}:", i)])

    return {
        "convention": convention_count,
        "warning": warning_count,
        "error": error_count,
        "refactor": refactor_count,
    }


def generate_issues_html(issues: List[str]) -> str:
    """
    Generate HTML for the issues section.

    Args:
        issues: List of pylint issues

    Returns:
        HTML string for issues section
    """
    if not issues:
        return "<p>No issues found! 🎉</p>"

    issues_list: List[str] = []

    for issue in issues:
        # if "convention" in issue.lower():
        if re.search(r"C\d{4}:", issue):
            css_class = "convention"
        # elif "warning" in issue.lower():
        elif re.search(r"W\d{4}:", issue):
            css_class = "warning"
        # elif "error" in issue.lower():
        elif re.search(r"E\d{4}:", issue):
            css_class = "error"
        else:
            css_class = "refactor"
        issues_list.append(f'<div class="issue {css_class}">{html.escape(issue)}</div>')

    return f'<div class="issues-list">{"".join(issues_list)}</div>'


def generate_html_report(
    pylint_content: str,
    issues: List[str],
    summary_lines: List[str],
    pylint_score: str,
    run_id: str,
    ref_name: str,
    github_sha: str,
    github_repo: str,
) -> str:
    """
    Generate the complete HTML report.

    Args:
        pylint_content: Raw pylint output content
        issues: List of pylint issues
        summary_lines: Summary lines from pylint output
        pylint_score: Pylint score
        run_id: GitHub run ID
        ref_name: Git reference name
        github_sha: GitHub SHA
        github_repo: GitHub repository name

    Returns:
        Complete HTML report as string
    """
    formatted_time: str = get_formatted_timestamp()
    score_color: str = get_score_color(pylint_score)
    issue_counts: Dict[str, int] = count_issues_by_type(issues)
    issues_html: str = generate_issues_html(issues)

    html_content: str = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pylint Report - Rattlesnake</title>
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
            background: white; padding: 25px; border-radius: 8px; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px;
        }}
        .issues-list {{
            max-height: 500px; overflow-y: auto; 
            border: 1px solid #e1e4e8; border-radius: 6px;
        }}
        .issue {{
            padding: 10px; border-bottom: 1px solid #e1e4e8;
            font-family: 'SFMono-Regular', 'Consolas', monospace;
            font-size: 0.9em;
        }}
        .issue:last-child {{
            border-bottom: none;
        }}
        .issue.error {{ background: #ffeef0; }}
        .issue.warning {{ background: #fff8e1; }}
        .issue.convention {{ background: #e8f4fd; }}
        .issue.refactor {{ background: #f0f9ff; }}
        .summary {{
            background: #f6f8fa; padding: 20px; border-radius: 6px; 
            border-left: 4px solid #0366d6; font-family: monospace;
            white-space: pre-wrap;
        }}
        .stats {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin-top: 20px;
        }}
        .stat-card {{
            background: #f6f8fa; padding: 15px; border-radius: 6px; text-align: center;
        }}
        .stat-number {{
            font-size: 1.8em; font-weight: bold; color: #0366d6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Rattlesnake Pylint Report</h1>
            <div class="score">{pylint_score}/10</div>
            <div class="metadata">
                <div><strong>Generated:</strong> {formatted_time}</div>
                <div><strong>Run ID:</strong> {run_id}</div>
                <div><strong>Run ID:</strong> <a href="https://github.com/{github_repo}/actions/runs/{run_id}"> {run_id}</a></div>
                <div><strong>Branch:</strong> {ref_name}</div>
                <div><strong>Commit:</strong> {github_sha[:7]}</div>
                <div><strong>Repository:</strong> <a href="https://github.com/{github_repo}">{github_repo}</a></div>
            </div>
        </div>
        
        <div class="nav">
            <a href="#summary">Summary</a>
            <a href="#issues">Issues ({len(issues)})</a>
            <a href="#statistics">Full Report</a>
            <a href="https://github.com/{github_repo}/actions/runs/{run_id}">View Workflow Run</a>
            <a href="https://github.com/{github_repo}">Repository</a>
        </div>
        
        <div class="section">
            <h2 id="summary">📊 Summary</h2>
            <div class="summary">{"".join(summary_lines)}</div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{len(issues)}</div>
                    <div>Total Issues</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{issue_counts["convention"]}</div>
                    <div>Conventions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{issue_counts["warning"]}</div>
                    <div>Warnings</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{issue_counts["error"]}</div>
                    <div>Errors</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{issue_counts["refactor"]}</div>
                    <div>Refactor</div>
                </div>
            </div>
        </div>
        
        <!--div class="section">
            <h2 id="issues">🔍 Issues Detail</h2>
            {issues_html}
        </div-->
        
        <div class="section">
            <h2 id="statistics">📈 Full Report</h2>
            <details>
                <summary>Click to view complete pylint output</summary>
                <pre style="background: #f6f8fa; padding: 20px; border-radius: 6px; overflow-x: auto;">{html.escape(pylint_content)}</pre>
            </details>
        </div>
    </div>
    <footer style="text-align: center; margin: 40px 0; color: #6a737d;">
        <p>Generated by GitHub Actions • <a href="https://github.com/{github_repo}">Rattlesnake Project</a></p>
    </footer>
</body>
</html>"""

    return html_content


def write_html_report(html_content: str, output_file: str) -> None:
    """
    Write HTML content to file.

    Args:
        html_content: The HTML content to write
        output_file: Path for the output HTML file

    Raises:
        SystemExit: If file cannot be written
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
    except IOError as e:
        print(f'Error writing output file "{output_file}": {e}')
        sys.exit(1)


def create_pylint_html_report(
    input_file: str,
    output_file: str,
    pylint_score: str,
    run_id: str,
    ref_name: str,
    github_sha: str,
    github_repo: str,
) -> Tuple[int, Dict[str, int]]:
    """
    Main function to create HTML report from pylint output.

    Args:
        input_file: Path to the pylint output text file
        output_file: Path for the generated HTML report
        pylint_score: Pylint score
        run_id: GitHub Actions run ID
        ref_name: Git reference name (branch)
        github_sha: GitHub commit SHA
        github_repo: GitHub repository name

    Returns:
        Tuple of (total_issues, issue_counts_dict)
    """
    breakpoint()
    # Read the pylint output
    pylint_content: str = read_pylint_content(input_file)

    # Parse pylint output
    issues, summary_lines = parse_pylint_output(pylint_content=pylint_content)

    # Generate HTML report
    html_content: str = generate_html_report(
        pylint_content,
        issues,
        summary_lines,
        pylint_score,
        run_id,
        ref_name,
        github_sha,
        github_repo,
    )

    # Write HTML report
    write_html_report(html_content, output_file)

    # Count issues by type
    issue_counts: Dict[str, int] = count_issues_by_type(issues)

    return len(issues), issue_counts


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate enhanced HTML report from pylint output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python pylint_report.py \\
    --input_file pylint_output_20240101_120000_UTC.txt \\
    --output_file pylint_report.html \\
    --pylint_score 8.5 \\
    --run_id 1234567890 \\
    --ref_name main \\
    --github_sha abc123def456 \\
    --github_repo owner/repo-name
        """,
    )

    parser.add_argument("--input_file", required=True, help="Input pylint output file")

    parser.add_argument("--output_file", required=True, help="Output HTML report file")

    parser.add_argument(
        "--pylint_score", required=True, help='Pylint score (e.g., "8.5")'
    )

    # parser.add_argument(
    #     "--badge_color",
    #     required=True,
    #     help='Badge color (e.g., "green", "yellow", "red")',
    # )

    parser.add_argument("--run_id", required=True, help="GitHub Actions run ID")

    parser.add_argument(
        "--ref_name", required=True, help="Git reference name (branch name)"
    )

    parser.add_argument("--github_sha", required=True, help="GitHub commit SHA")

    parser.add_argument(
        "--github_repo", required=True, help="GitHub repository name (owner/repo-name)"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    args: argparse.Namespace = parse_arguments()

    try:
        total_issues, issue_counts = create_pylint_html_report(
            args.input_file,
            args.output_file,
            args.pylint_score,
            args.run_id,
            args.ref_name,
            args.github_sha,
            args.github_repo,
        )

        print(f"✅ Enhanced HTML report generated: {args.output_file}")
        print(f"📊 Pylint score: {args.pylint_score}/10")
        print(f"🔍 Total issues found: {total_issues}")
        print(f"   - Errors: {issue_counts['error']}")
        print(f"   - Warnings: {issue_counts['warning']}")
        print(f"   - Conventions: {issue_counts['convention']}")

    except Exception as e:
        print(f"❌ Error generating HTML report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
