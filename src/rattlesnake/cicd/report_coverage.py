"""This module extracts key coverage metrics from a coverage output file."""

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from rattlesnake.cicd.utilities import (
    extend_timestamp,
    get_score_color_coverage,
    write_report,
)


@dataclass(frozen=True)
class CoverageMetric:
    """Represents coverage metrics for a codebase.

    Attributes:
        lines_valid (int): The total number of valid lines in the codebase.
        lines_covered (int): The number of lines that are covered by tests.
        coverage (float): The coverage percentage, calculated as
            (lines_covered / lines_valid) * 100. Defaults to 0.0.
        color (str): The color code (e.g., red, green), based on the coverage.
    """

    lines_valid: int = 0
    lines_covered: int = 0

    @property
    def coverage(self) -> float:
        """
        Calculates the coverage percentage.

        The coverage is calculated as `(lines_covered / lines_valid) * 100`.
        Returns 0.0 if `lines_valid` is zero to prevent division by zero errors.
        """

        return (self.lines_covered / self.lines_valid * 100) if self.lines_valid > 0 else 0.0

    @property
    def color(self) -> str:
        """
        Determines the badge color based on the coverage percentage.
        """
        return get_score_color_coverage(str(self.coverage))


def get_coverage_metric(coverage_file: Path) -> CoverageMetric:
    """
    Gets the lines-valid, lines-covered, and coverage percentage as
    a list strings.
    """

    cm = CoverageMetric()

    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        lines_valid = int(root.attrib["lines-valid"])
        lines_covered = int(root.attrib["lines-covered"])
        cm = CoverageMetric(
            lines_valid=lines_valid,
            lines_covered=lines_covered,
        )  # overwrite default
    except (FileNotFoundError, ET.ParseError, KeyError) as e:
        print(f"Error processing coverage file: {e}")

    return cm


def get_report_html(
    coverage_metric: CoverageMetric,
    timestamp: str,
    run_id: str,
    ref_name: str,
    github_sha: str,
    github_repo: str,
) -> str:
    """
    Generates an HTML report from the coverage metrics.

    Args:
        coverage_metric: CoverageMetric object containing coverage data
        timestamp: The timestamp from bash when lint ran, in format YYYYMMDD_HHMMSS_UTC
            e.g., 20250815_211112_UTC
        run_id: GitHub Actions run ID
        ref_name: Git reference name (branch)
        github_sha: GitHub commit SHA
        github_repo: GitHub repository name

    Returns:
        Complete HTML report as a string
    """
    timestamp_ext = extend_timestamp(timestamp)
    score_color: str = coverage_metric.color

    # Programmatically construct the full report URL
    try:
        owner, repo_name = github_repo.split("/")
        full_report_url = (
            f"https://{owner}.github.io/{repo_name}/reports/coverage/htmlcov/index.html"
        )
    except ValueError:
        # Fallback or default URL in case the repo format is unexpected
        full_report_url = "#"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coverage Report</title>
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Coverage Report</h1>
            <div class="score">Coverage: {coverage_metric.coverage:.2f}%</div>
            <div class="metadata">
                <div><strong>Lines Covered:</strong> {coverage_metric.lines_covered}</div>
                <div><strong>Total Lines:</strong> {coverage_metric.lines_valid}</div>
                <div>&nbsp;</div>
                <div><strong>Generated:</strong> {timestamp_ext}</div>
                <div><strong>Run ID:</strong> <a href="https://github.com/{github_repo}/actions/runs/{run_id}"> {run_id}</a></div>
                <div><strong>Branch:</strong> <a href="https://github.com/{github_repo}/tree/{ref_name}"> {ref_name}</a></div>
                <div><strong>Commit:</strong> <a href="https://github.com/{github_repo}/commit/{github_sha}"> {github_sha[:7]}</a></div>
                <div><strong>Repository:</strong> <a href="https://github.com/{github_repo}">{github_repo}</a></div>
                <div>&nbsp;</div>
                <div><strong>Full report:</strong> <a href="{full_report_url}">HTML</a></div>
            </div>
        </div>
    </div>
    <footer style="text-align: center; margin: 40px 0; color: #6a737d;">
        <p>Generated by GitHub Actions</p>
    </footer>
</body>
</html>"""

    return html_content


def run_coverage_report(
    input_file: str,
    output_file: str,
    timestamp: str,
    run_id: str,
    ref_name: str,
    github_sha: str,
    github_repo: str,
) -> CoverageMetric:
    """
    Main function to create HTML report from coverage output.

    Args:
        input_file: Path to the coverage output text file
        output_file: Path for the generated HTML report
        timestamp: The timestamp from bash when lint ran, in format YYYYMMDD_HHMMSS_UTC
            e.g., 20250815_211112_UTC
        run_id: GitHub Actions run ID
        ref_name: Git reference name (branch)
        github_sha: GitHub commit SHA
        github_repo: GitHub repository name

    Returns:
        CoverageMetric
    """
    # Get the coverage metric
    coverage_metric = get_coverage_metric(coverage_file=Path(input_file))
    print(f"run_coverage_report: coverage_metric={coverage_metric}")

    # Generate HTML report
    html_content: str = get_report_html(
        coverage_metric,
        timestamp,
        run_id,
        ref_name,
        github_sha,
        github_repo,
    )

    # Write the HTML report
    write_report(html_content, output_file)

    return coverage_metric


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate enhanced HTML report from coverage output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python coverage_report.py \
    --input_file coverage.xml \
    --output_file coverage_report.html \
    --timestamp 20240101_120000_UTC \
    --run_id 1234567890 \
    --ref_name main \
    --github_sha abc123def456 \
    --github_repo owner/repo-name
        """,
    )

    parser.add_argument("--input_file", required=True, help="Input coverage output file")

    parser.add_argument("--output_file", required=True, help="Output HTML report file")

    parser.add_argument(
        "--timestamp", required=True, help="UTC timestamp, e.g., 20240101_120000_UTC"
    )

    parser.add_argument("--run_id", required=True, help="GitHub Actions run ID")

    parser.add_argument("--ref_name", required=True, help="Git reference name (branch name)")

    parser.add_argument("--github_sha", required=True, help="GitHub commit SHA")

    parser.add_argument(
        "--github_repo", required=True, help="GitHub repository name (owner/repo-name)"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args: argparse.Namespace = parse_arguments()

    try:
        cm: CoverageMetric = run_coverage_report(
            args.input_file,
            args.output_file,
            args.timestamp,
            args.run_id,
            args.ref_name,
            args.github_sha,
            args.github_repo,
        )

        print(f"[OK] Coverage HTML report generated: {args.output_file}")
        print(f"[I] - valid lines of code: {cm.lines_valid}")
        print(f"[I] - lines covered: {cm.lines_covered}")
        print(f"[I] - coverage: {cm.coverage} percent")

    except FileNotFoundError:
        print(f"[X] Error: The input file '{args.input_file}' was not found.")
        return 1
    except IOError as e:
        print(f"[X] I/O error occurred: {e}")
        return 1
    except ValueError as e:
        print(f"[X] Input Error : {e}")
        return 1

    return 0  # Success exit code


if __name__ == "__main__":
    sys.exit(main())
