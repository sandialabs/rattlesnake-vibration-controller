"""
Generates the Lint badge (SVG) and metadata (JSON) for CI/CD.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from rattlesnake.cicd.utilities import get_score_color_lint


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
    parser.add_argument("--output_dir", help="Directory to save badges")
    parser.add_argument("--github_repo", help="owner/repo")
    parser.add_argument("--deploy_subdir", help="main or dev")
    parser.add_argument("--run_id", help="GitHub Run ID")
    parser.add_argument("--github_server_url", default="https://github.com")
    parser.add_argument("--export_env", action="store_true", help="Export color to GITHUB_ENV")

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
        try:
            response = requests.get(badge_url, timeout=10)
            response.raise_for_status()
            with open(Path(args.output_dir) / "lint.svg", "wb") as f:
                f.write(response.content)
            print(f"[OK] Lint SVG badge saved to {args.output_dir}")
        except requests.RequestException as e:
            print(f"[X] Failed to download badge: {e}")

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

            with open(Path(args.output_dir) / "lint-info.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            print(f"[OK] Lint JSON metadata saved to {args.output_dir}")

    print(f"Lint badge processing complete: Score={score}, Color={color}")


if __name__ == "__main__":
    main()
