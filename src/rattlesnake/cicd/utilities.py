#!/usr/bin/env python3
"""
Utilities for CICD processes.
"""

import argparse
import json
import re
from datetime import datetime
from typing import NamedTuple

import pytz
import requests


class ReportMetadata(NamedTuple):
    """Container for CI/CD metadata used in reports."""
    timestamp: str
    run_id: str
    ref_name: str
    github_sha: str
    github_repo: str


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Add shared CI/CD arguments to an argument parser.

    Args:
        parser: The argparse.ArgumentParser to update.
    """
    parser.add_argument(
        "--timestamp", required=True, help="UTC timestamp, e.g., 20240101_120000_UTC"
    )
    parser.add_argument("--run_id", required=True, help="GitHub Actions run ID")
    parser.add_argument("--ref_name", required=True, help="Git reference name (branch)")
    parser.add_argument("--github_sha", required=True, help="GitHub commit SHA")
    parser.add_argument("--github_repo", required=True, help="GitHub repository name")


def add_badge_args(parser: argparse.ArgumentParser) -> None:
    """
    Add shared arguments for badge generation scripts.

    Args:
        parser: The argparse.ArgumentParser to update.
    """
    parser.add_argument("--output_dir", help="Directory to save badges")
    parser.add_argument("--github_repo", help="owner/repo")
    parser.add_argument("--deploy_subdir", help="main or dev")
    parser.add_argument("--run_id", help="GitHub Run ID")
    parser.add_argument("--github_server_url", default="https://github.com")


def report_metadata_creation(args: argparse.Namespace) -> ReportMetadata:
    """
    Create ReportMetadata from parsed arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        ReportMetadata instance.
    """
    return ReportMetadata(
        timestamp=args.timestamp,
        run_id=args.run_id,
        ref_name=args.ref_name,
        github_sha=args.github_sha,
        github_repo=args.github_repo,
    )


def report_main_runner(main_func, args: argparse.Namespace) -> int:
    """
    Common execution loop for report generation scripts.

    Args:
        main_func: Function that takes (args, metadata) and returns None.
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    metadata: ReportMetadata = report_metadata_creation(args)

    try:
        main_func(args, metadata)
    except (FileNotFoundError, IOError) as e:
        print(f"[X] File Error: {e}")
        return 1
    except ValueError as e:
        print(f"[X] Input Error: {e}")
        return 1
    return 0


def badge_image_download(url: str, output_path: str) -> bool:
    """
    Download a badge SVG from a URL.

    Args:
        url: The URL of the badge.
        output_path: Local file path to save the SVG.

    Returns:
        True if successful, False otherwise.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True
    except requests.RequestException as e:
        print(f"[X] Failed to download badge from {url}: {e}")
        return False


def badge_metadata_json_write(metadata: dict, output_path: str) -> None:
    """
    Write badge metadata to a JSON file.

    Args:
        metadata: Dictionary containing metadata.
        output_path: Local file path to save the JSON.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except (IOError, TypeError) as e:
        print(f"[X] Failed to write JSON metadata to {output_path}: {e}")


def get_score_color_lint(pylint_score: str) -> str:
    """
    Determine color based on pylint score.

    Args:
        pylint_score: The pylint score as string, e.g., "8.5", "7.0", etc.

    Returns:
        Hex color code for the score, as a string.
    """
    try:
        score_val: float = float(pylint_score)
        if score_val >= 8.0:
            return "brightgreen"
        if score_val >= 6.0:
            return "yellow"
        if score_val >= 4.0:
            return "orange"
        return "red"
    except ValueError:
        return "gray"


def get_score_color_coverage(coverage_score: str) -> str:
    """
    Determines the color based on a coverage score.

    Args:
        coverage_score:  The coverage score as a string, e.g., "92.5"

    Returns:
        The color for the badge as a string.
    """
    try:
        score_val: float = float(coverage_score)
        if score_val >= 90:
            return "brightgreen"
        if score_val >= 80:
            return "green"
        if score_val >= 70:
            return "yellow"
        if score_val >= 60:
            return "orange"
        return "red"
    except ValueError:
        return "gray"


def _get_timezone_strings(utc_now: datetime) -> tuple[str, str, str, str]:
    """
    Helper to get formatted UTC, EST, MST, and PST strings from a UTC datetime.

    Args:
        utc_now: The UTC datetime object.

    Returns:
        A tuple of four strings: (UTC string, EST string, MST string, PST string).
    """
    # Define the time zones
    timezone_est: pytz.BaseTzInfo = pytz.timezone("America/New_York")
    timezone_mst: pytz.BaseTzInfo = pytz.timezone("America/Denver")
    timezone_pst: pytz.BaseTzInfo = pytz.timezone("America/Los_Angeles")

    # Convert UTC time to EST, MST, and PST
    est_now: datetime = utc_now.astimezone(timezone_est)
    mst_now: datetime = utc_now.astimezone(timezone_mst)
    pst_now: datetime = utc_now.astimezone(timezone_pst)

    # Format the output
    df: str = "%Y-%m-%d %H:%M:%S"  # Date format
    utc_str: str = utc_now.strftime(f"{df} UTC")
    est_str: str = est_now.strftime(f"{df} EST")
    mst_str: str = mst_now.strftime(f"{df} MST")
    pst_str: str = pst_now.strftime(f"{df} PST")

    return utc_str, est_str, mst_str, pst_str


def get_timestamp() -> str:
    """
    Get formatted timestamp with UTC, EST, MST, and PST times.

    Returns:
        Formatted timestamp string
    """
    # Get the current UTC time
    utc_now: datetime = datetime.now(pytz.utc)

    # Get the formatted strings for each timezone
    utc_str, est_str, mst_str, pst_str = _get_timezone_strings(utc_now)

    # Combine the formatted times
    timestamp: str = f"{utc_str} ({est_str} / {mst_str} / {pst_str})"

    return timestamp


def extend_timestamp(short: str) -> str:
    """
    Given a timestamp string from CI/CD in the form of
    20250815_211112_UTC, extend the timestamp to include EST, MST, and PST times
    and return the extended string.

    Args:
        short: the UTC bash string, for example: 20250815_211112_UTC

    Returns:
        Extended timestamp string.
    """
    # Call the multiline function to get the individual strings
    lines: list[str] = get_multiline_timestamp(short)

    # lines[1] is UTC, lines[2] is EST, lines[3] is MST, lines[4] is PST
    return f"{lines[1]} ({lines[2]} / {lines[3]} / {lines[4]})"


def get_multiline_timestamp(short: str) -> list[str]:
    """
    Given a timestamp string from CI/CD in the form of
    20250815_211112_UTC, return a list of strings for multiple timezones.

    Args:
        short: the UTC bash string, for example: 20250815_211112_UTC

    Returns:
        List of 4 formatted strings.
    """
    # Regex pattern to match the required format: YYYYMMDD_HHMMSS_TZ
    pattern: re.Pattern = re.compile(r"^(\d{8})_(\d{6})_(UTC|GMT|Z)$")
    match = pattern.match(short)

    if not match:
        raise ValueError(f"Invalid timestamp format: '{short}'")

    # Extract the date and time parts from the regex match
    date_part, time_part, _ = match.groups()

    # Combine the parts into a format that can be parsed by datetime
    datetime_str: str = f"{date_part}_{time_part}_UTC"
    input_format: str = "%Y%m%d_%H%M%S_%Z"

    # Convert the input string to a datetime object
    utc_datetime: datetime = datetime.strptime(datetime_str, input_format)

    # Make the datetime object timezone-aware
    utc_now: datetime = pytz.utc.localize(utc_datetime)

    # Get the formatted strings for each timezone
    utc_str, est_str, mst_str, pst_str = _get_timezone_strings(utc_now)

    return ["Generated:", utc_str, est_str, mst_str, pst_str]


def write_report(html_content: str, output_file: str) -> None:
    """
    Write HTML content to file.

    Args:
        html_content: The HTML content to write
        output_file: Path for the output HTML file

    Raises:
        IOError: If the file cannot be written.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
    except IOError as e:
        raise IOError(f'Error writing output file "{output_file}": {e}') from e
