"""
Generates the Lint badge (SVG) and metadata (JSON) for CI/CD.
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from rattlesnake.cicd.utilities import (
    add_badge_args,
    badge_image_download,
    badge_metadata_json_write,
    get_score_color_lint,
)


def extract_score(input_file: str) -> float:
    """Extracts the lint score from the output text file."""
    pattern = re.compile(r"Your code has been rated at (\d+\.\d+)/10")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
            match = pattern.search(content)
            if match:
                return float(match.group(1))
    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"[!] Error reading lint output: {e}")
    return 0.0


def export_to_github_env(color: str):
    """Exports the badge color to GITHUB_ENV if available."""
    env_path = os.environ.get("GITHUB_ENV")
    if env_path:
        with open(env_path, "a", encoding="utf-8") as f:
            f.write(f"BADGE_COLOR={color}\n")
        print(f"    [C] Exported BADGE_COLOR={color} to GITHUB_ENV")


def main():
    """Main method for creating the badge."""
    parser = argparse.ArgumentParser(description="Generate Lint badge and metadata.")
    parser.add_argument("--input_file", help="Lint text output file (to extract score)")
    parser.add_argument("--score", type=float, help="Lint score (direct input)")
    parser.add_argument("--export_env", action="store_true", help="Export color to GITHUB_ENV")

    # Add common badge arguments from utilities
    add_badge_args(parser)

    args = parser.parse_args()

    # Determine the score
    if args.score is not None:
        score = args.score
    elif args.input_file:
        score = extract_score(args.input_file)
    else:
        print("[X] Error: Must provide either --input_file or --score")
        sys.exit(1)

    color = get_score_color_lint(str(score))

    # Optional export to GITHUB_ENV
    if args.export_env:
        export_to_github_env(color)

    # If output_dir is provided, generate SVG and JSON
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Download SVG badge
        badge_url = f"https://img.shields.io/badge/lint-{score}-{color}.svg"
        output_svg = str(Path(args.output_dir) / "lint.svg")
        if badge_image_download(url=badge_url, output_path=output_svg):
            print(f"[OK] Lint SVG badge saved to {args.output_dir}")

        # Generate JSON metadata if other required args are present
        if all([args.github_repo, args.deploy_subdir, args.run_id]):
            owner, repo = args.github_repo.split("/")
            metadata = {
                "score": str(score),
                "color": color,
                "pages_url": f"https://{owner}.github.io/{repo}/{args.deploy_subdir}/reports/lint/",
                "workflow_url": (
                    f"{args.github_server_url}/{args.github_repo}/"
                    "actions/workflows/ci.yml"
                ),
                "run_id": args.run_id,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            output_json = str(Path(args.output_dir) / "lint-info.json")
            badge_metadata_json_write(metadata=metadata, output_path=output_json)
            print(f"[OK] Lint JSON metadata saved to {args.output_dir}")

    print(f"Lint badge processing complete: Score={score}, Color={color}")


if __name__ == "__main__":
    main()
