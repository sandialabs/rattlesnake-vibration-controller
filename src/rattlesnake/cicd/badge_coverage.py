"""
Generates the Coverage badge (SVG) and metadata (JSON) for CI/CD.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests
from rattlesnake.cicd.utilities import (
    add_badge_args,
    get_score_color_coverage,
)


def extract_coverage(input_file: str) -> float:
    """Parses a coverage.xml file to get the coverage percentage."""
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
        # line-rate is a decimal (e.g., 0.925), so multiply by 100
        coverage = float(root.attrib["line-rate"]) * 100
        return coverage
    except (ET.ParseError, KeyError, FileNotFoundError, ValueError) as e:
        print(f"[!] Error parsing coverage.xml: {e}")
        return 0.0


def export_to_github_env(coverage: float, color: str):
    """Exports coverage metrics to GITHUB_ENV if available."""
    env_path = os.environ.get("GITHUB_ENV")
    if env_path:
        with open(env_path, "a", encoding="utf-8") as f:
            f.write(f"COVERAGE={coverage:.1f}\n")
            f.write(f"BADGE_COLOR_COV={color}\n")
        print(f"    [C] Exported COVERAGE={coverage:.1f} and BADGE_COLOR_COV={color}")


def main():
    """Main method for creating the badge."""
    parser = argparse.ArgumentParser(description="Generate Coverage badge and metadata.")
    parser.add_argument("--input_file", help="coverage.xml file (to extract percentage)")
    parser.add_argument("--coverage", type=float, help="Coverage percentage (direct input)")
    parser.add_argument("--export_env", action="store_true", help="Export to GITHUB_ENV")

    # Add common badge arguments from utilities
    add_badge_args(parser)

    args = parser.parse_args()

    # Determine the coverage
    if args.coverage is not None:
        coverage = args.coverage
    elif args.input_file:
        coverage = extract_coverage(args.input_file)
    else:
        print("[X] Error: Must provide either --input_file or --coverage")
        sys.exit(1)

    color = get_score_color_coverage(str(coverage))

    # Optional export to GITHUB_ENV
    if args.export_env:
        export_to_github_env(coverage, color)

    # If output_dir is provided, generate SVG and JSON
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Download SVG badge
        # Percentage is formatted to 1 decimal place for the badge
        badge_url = f"https://img.shields.io/badge/coverage-{coverage:.1f}%25-{color}.svg"
        try:
            response = requests.get(badge_url, timeout=10)
            response.raise_for_status()
            with open(Path(args.output_dir) / "coverage.svg", "wb") as f:
                f.write(response.content)
            print(f"[OK] Coverage SVG badge saved to {args.output_dir}")
        except requests.RequestException as e:
            print(f"[X] Failed to download badge: {e}")

        # Generate JSON metadata if other required args are present
        if all([args.github_repo, args.deploy_subdir, args.run_id]):
            owner, repo = args.github_repo.split("/")
            metadata = {
                "coverage": f"{coverage:.1f}",
                "color": color,
                "pages_url": (
                    f"https://{owner}.github.io/{repo}/"
                    f"{args.deploy_subdir}/reports/coverage/"
                ),
                "workflow_url": (
                    f"{args.github_server_url}/{args.github_repo}/"
                    "actions/workflows/ci.yml"
                ),
                "run_id": args.run_id,
                "artifact_url": (
                    f"{args.github_server_url}/{args.github_repo}/"
                    f"actions/runs/{args.run_id}"
                ),
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            with open(
                Path(args.output_dir) / "coverage-info.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(metadata, f, indent=2)
            print(f"[OK] Coverage JSON metadata saved to {args.output_dir}")

    print(f"Coverage badge processing complete: {coverage:.1f}% ({color})")


if __name__ == "__main__":
    main()
