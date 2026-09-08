"""
Generates the Tests badge (SVG) and metadata (JSON) for CI/CD.
"""

import argparse
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from rattlesnake.cicd.utilities import (
    add_badge_args,
    badge_image_download,
    badge_metadata_json_write,
)


def parse_junit(input_file: str) -> tuple[int, int, int, int]:
    """Parses a JUnit XML report to get test totals.

    Args:
        input_file: Path to a JUnit XML report (pytest --junitxml=...).

    Returns:
        A (total, passed, failed, skipped) tuple.
    """
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()

        if root.tag == "testsuites":
            suites = root.findall("testsuite")
        elif root.tag == "testsuite":
            suites = [root]
        else:
            raise KeyError(f"Unexpected root tag: {root.tag}")

        total = sum(int(suite.get("tests", 0)) for suite in suites)
        failures = sum(int(suite.get("failures", 0)) for suite in suites)
        errors = sum(int(suite.get("errors", 0)) for suite in suites)
        skipped = sum(int(suite.get("skipped", 0)) for suite in suites)

        failed = failures + errors
        passed = total - failed - skipped
        return total, passed, failed, skipped
    except (ET.ParseError, KeyError, FileNotFoundError, ValueError) as e:
        print(f"[!] Error parsing JUnit XML report: {e}")
        return 0, 0, 0, 0


def main():
    """Main method for creating the badge."""
    parser = argparse.ArgumentParser(description="Generate Tests badge and metadata.")
    parser.add_argument(
        "--input_file", required=True, help="JUnit XML report file (junit.xml)"
    )
    add_badge_args(parser)
    args = parser.parse_args()

    total, passed, failed, skipped = parse_junit(args.input_file)

    message = f"{passed} pass {failed} fail"
    if skipped > 0:
        message += f" {skipped} skip"

    color = "brightgreen" if failed == 0 else "red"

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        label = quote("tests")
        encoded_message = quote(message)
        badge_url = (
            f"https://img.shields.io/badge/{label}-{encoded_message}-{color}.svg"
        )
        output_svg = str(Path(args.output_dir) / "tests.svg")
        if badge_image_download(url=badge_url, output_path=output_svg):
            print(f"[OK] Tests SVG badge saved to {args.output_dir}")

        if all([args.github_repo, args.deploy_subdir, args.run_id]):
            metadata = {
                "total": total,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "message": message,
                "color": color,
                "workflow_url": (
                    f"{args.github_server_url}/{args.github_repo}/actions/"
                    f"workflows/ci.yml?query=branch%3A{args.deploy_subdir}"
                ),
                "run_id": args.run_id,
                "artifact_url": (
                    f"{args.github_server_url}/{args.github_repo}/actions/"
                    f"runs/{args.run_id}"
                ),
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            output_json = str(Path(args.output_dir) / "tests-info.json")
            badge_metadata_json_write(metadata=metadata, output_path=output_json)
            print(f"[OK] Tests JSON metadata saved to {args.output_dir}")

    print(f"Tests badge processing complete: {message} ({color})")


if __name__ == "__main__":
    main()
