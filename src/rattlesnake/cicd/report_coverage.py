"""This module extracts key coverage metrics from a coverage output file."""

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from rattlesnake.cicd.utilities import (
    ReportMetadata,
    add_common_args,
    extend_timestamp,
    get_score_color_coverage,
    report_main_runner,
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
    metadata: ReportMetadata,
) -> str:
    """
    Generates an HTML report from the coverage metrics.

    Args:
        coverage_metric: CoverageMetric object containing coverage data
        metadata: CI/CD metadata (timestamp, run_id, etc.)

    Returns:
        Complete HTML report as a string
    """
    timestamp_ext = extend_timestamp(metadata.timestamp)
    score_color: str = coverage_metric.color

    # Programmatically construct the full report URL
    try:
        owner, repo_name = metadata.github_repo.split("/")
        full_report_url = (
            f"https://{owner}.github.io/{repo_name}/reports/coverage/htmlcov/index.html"
        )
    except ValueError:
        # Fallback or default URL in case the repo format is unexpected
        full_report_url = "#"

    # Pre-calculate GitHub URLs to stay under line length limits
    github_url = f"https://github.com/{metadata.github_repo}"
    run_url = f"{github_url}/actions/runs/{metadata.run_id}"
    branch_url = f"{github_url}/tree/{metadata.ref_name}"
    commit_url = f"{github_url}/commit/{metadata.github_sha}"

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
                <div><strong>Run ID:</strong>
                    <a href="{run_url}">{metadata.run_id}</a></div>
                <div><strong>Branch:</strong>
                    <a href="{branch_url}">{metadata.ref_name}</a></div>
                <div><strong>Commit:</strong>
                    <a href="{commit_url}">{metadata.github_sha[:7]}</a></div>
                <div><strong>Repository:</strong>
                    <a href="{github_url}">{metadata.github_repo}</a></div>
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
    metadata: ReportMetadata,
) -> CoverageMetric:
    """
    Main function to create HTML report from coverage output.

    Args:
        input_file: Path to the coverage output text file
        output_file: Path for the generated HTML report
        metadata: CI/CD metadata

    Returns:
        CoverageMetric
    """
    # Get the coverage metric
    coverage_metric = get_coverage_metric(coverage_file=Path(input_file))
    print(f"run_coverage_report: coverage_metric={coverage_metric}")

    # Generate HTML report
    html_content: str = get_report_html(
        coverage_metric,
        metadata,
    )

    # Write the HTML report
    write_report(html_content, output_file)

    print(f"[OK] Coverage HTML report generated: {output_file}")
    print(f"[I] - valid lines of code: {coverage_metric.lines_valid}")
    print(f"[I] - lines covered: {coverage_metric.lines_covered}")
    print(f"[I] - coverage: {coverage_metric.coverage:.2f} percent")

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
    )

    parser.add_argument("--input_file", required=True, help="Input coverage output file")
    parser.add_argument("--output_file", required=True, help="Output HTML report file")

    add_common_args(parser)

    return parser.parse_args()


def generate_report(args: argparse.Namespace, metadata: ReportMetadata) -> None:
    """
    Adapter for report_main_runner.
    """
    run_coverage_report(args.input_file, args.output_file, metadata)


def main() -> int:
    """
    Main entry point for the script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args: argparse.Namespace = parse_arguments()
    return report_main_runner(generate_report, args)


if __name__ == "__main__":
    sys.exit(main())
