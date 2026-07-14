#!/usr/bin/env python3
"""
Preflight script to verify likely CI/CD readiness through extensive local testing.

By default, runs the same lint and test scope as CI on a non-main/dev branch:
pylint on all of src/rattlesnake and pytest on tests/ --ignore=tests/long.
Use --all-tests to match what CI runs on main/dev (includes tests/long/).

This script has an early check that attempts to connect to the package index (PyPI)
to detect common network issues (e.g. corporate firewalls with SSL interception).
If such issues are detected, it provides feedback to the user on how to proceed
(e.g. setting SSL_CERT_FILE or using --no-sync to skip dependency synchronization).
This helps users understand and resolve connectivity issues before they cause failures
in the actual CI/CD pipeline.

The `--no-sync` flag allows users to skip the dependency synchronization step,
which is useful when they are behind a firewall or have already installed dependencies.

Usage:
    uv run preflight [--no-sync] [--skip-network-check] [--all-tests]
                     [--coverage] [--tag TAG] [--docs] [--force]

Examples:
    uv run preflight                         # Default: matches CI's non-main/dev scope
    uv run preflight --all-tests             # Full suite: matches CI on main/dev (slow)
    uv run preflight --coverage              # Default scope + coverage report
    uv run preflight --all-tests --coverage  # Full suite + coverage report
    uv run preflight --tag v1.0.0rc1         # Validate tag, then run default scope
    uv run preflight --tag v1.0.0 --all-tests  # Validate tag, then run full suite
    uv run preflight --docs                  # Build Jupyter Book (requires network)
    uv run preflight --no-sync               # Skip dependency sync (offline/firewall)
    uv run preflight --force                 # Continue even if network/sync checks fail
    uv run preflight --skip-network-check    # Skip the initial connectivity check

"""

import argparse
import fnmatch
import os
import pathlib
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request

import yaml

# Path to the CI workflow file, relative to this file's repo root.
_CI_YML = pathlib.Path(__file__).parents[2] / ".github" / "workflows" / "ci.yml"


def _local_context() -> dict[str, str]:
    """Build a substitution map from GitHub Actions expressions to local values."""

    def _git(args: list[str]) -> str:
        result = subprocess.run(
            ["git"] + args, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    sha = _git(["rev-parse", "HEAD"])
    commit_msg = _git(["log", "-1", "--format=%s"])

    remote_url = _git(["remote", "get-url", "origin"])
    # Convert SSH or HTTPS remote URL to "org/repo" form
    repo = re.sub(r".*[:/]([^/]+/[^/]+?)(?:\.git)?$", r"\1", remote_url)

    return {
        "github.run_id": "local",
        "github.ref_name": branch,
        "github.ref": f"refs/heads/{branch}",
        "github.sha": sha,
        "github.repository": repo,
        "github.server_url": "https://github.com",
        "github.event.repository.name": repo.split("/")[-1] if "/" in repo else repo,
        "github.event.head_commit.message || "
        "github.event.pull_request.title": commit_msg,
        "github.event.inputs.test_level": "",
        "env.DEPLOY_SUBDIR": "dev",  # local runs are treated as dev
    }


def _substitute(
    text: str, context: dict[str, str]
) -> tuple[str, list[tuple[str, str]]]:
    """
    Replace all ${{ expr }} tokens in *text* using *context*.
    Returns (substituted_text, list_of_(from, to) pairs for every replacement made).
    """
    replacements: list[tuple[str, str]] = []

    def replacer(match: re.Match) -> str:
        expr = match.group(1).strip()
        if expr in context:
            replacements.append((match.group(0), context[expr]))
            return context[expr]
        return match.group(0)  # leave unknown expressions as-is

    result = re.sub(r"\$\{\{\s*(.*?)\s*\}\}", replacer, text)
    return result, replacements


def commands_from_ci_yml(job_name: str, step_name: str) -> str | None:
    """
    Parse .github/workflows/ci.yml and return the shell script for a specific
    job/step, with all GitHub Actions expressions substituted for local values.

    Prints the source file, job/step being harvested, and each substitution made.
    Returns None if the job or step name is not found.
    """
    if not _CI_YML.exists():
        print(f"[!] CI workflow not found at {_CI_YML}")
        return None

    print(f"  Harvesting from: {_CI_YML}")
    print(f"  Job: '{job_name}'  /  Step: '{step_name}'")

    with _CI_YML.open() as fh:
        workflow = yaml.safe_load(fh)

    jobs = workflow.get("jobs", {})
    job = jobs.get(job_name)
    if job is None:
        print(f"[!] Job '{job_name}' not found in {_CI_YML.name}")
        return None

    for step in job.get("steps", []):
        if step.get("name") == step_name and "run" in step:
            raw = step["run"]
            substituted, replacements = _substitute(raw, _local_context())
            if replacements:
                print("  Substitutions:")
                for from_expr, to_val in replacements:
                    to_display = to_val if to_val else '""'
                    print(f"    {from_expr}  ->  {to_display}")
            else:
                print("  Substitutions: (none)")
            return substituted

    print(f"[!] Step '{step_name}' not found in job '{job_name}'")
    return None


def _filter_patterns_from_ci_yml() -> dict[str, list[str]]:
    """
    Parse the dorny/paths-filter 'filters:' block from ci.yml and return a dict
    of {category: [glob_patterns]}, e.g. {'docs': [...], 'code': [...]}.
    Returns an empty dict if the step or filters key is not found.
    """
    if not _CI_YML.exists():
        return {}

    with _CI_YML.open() as fh:
        workflow = yaml.safe_load(fh)

    for step in workflow.get("jobs", {}).get("changes", {}).get("steps", []):
        if step.get("name") == "Filter changes" and "with" in step:
            filters_raw = step["with"].get("filters", "")
            parsed = yaml.safe_load(filters_raw)
            if isinstance(parsed, dict):
                # Strip surrounding quotes that dorny uses in its YAML format
                return {
                    category: [p.strip("'") for p in patterns]
                    for category, patterns in parsed.items()
                }

    return {}


def changes_from_git() -> tuple[bool, bool]:
    """
    Detect whether docs or code files have changed, mirroring the dorny/paths-filter
    logic in ci.yml's 'changes' job. Patterns are harvested directly from ci.yml so
    they stay in sync automatically.

    Checks the last commit plus any uncommitted changes in the working tree.
    Returns (docs_changed, code_changed).
    """
    print(f"  Harvesting change-filter patterns from: {_CI_YML}")

    patterns = _filter_patterns_from_ci_yml()
    if not patterns:
        print("  [!] Could not parse filter patterns — assuming all changed.")
        return True, True

    for category, globs in patterns.items():
        print(f"  Patterns for '{category}': {globs}")

    def _git(args: list[str]) -> str:
        result = subprocess.run(
            ["git"] + args, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    committed = _git(["diff", "--name-only", "HEAD~1", "HEAD"]).splitlines()
    uncommitted = _git(["diff", "--name-only", "HEAD"]).splitlines()
    changed: set[str] = set(committed + uncommitted)

    print(f"  Changed files ({len(changed)} total):")
    for f in sorted(changed):
        print(f"    {f}")

    results: dict[str, bool] = {}
    for category, globs in patterns.items():
        matched = [f for f in changed if any(fnmatch.fnmatch(f, pat) for pat in globs)]
        results[category] = bool(matched)
        status = "YES" if matched else "no"
        print(f"  '{category}' changed: {status}", end="")
        if matched:
            print(f"  (matched: {matched})", end="")
        print()

    return results.get("docs", False), results.get("code", False)


def check_connectivity(timeout: int = 5) -> tuple:
    """
    Check if the network (specifically the package index and file host) is reachable.
    Returns (success, error_message).
    """
    endpoints = [
        os.environ.get("UV_INDEX_URL", "https://pypi.org/simple/"),
        "https://files.pythonhosted.org/",
    ]

    for url in endpoints:
        try:
            urllib.request.urlopen(url, timeout=timeout)
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e.reason):
                return (
                    False,
                    f"SSL certificate verification failed for {url} (Unknown Issuer).",
                )
            return False, f"Connection failed to {url}: {e.reason}"
        except Exception as e:
            return False, f"Unexpected error connecting to {url}: {e}"

    return True, ""


def run_step(
    command: list[str], description: str, capture_output: bool = False
) -> subprocess.CompletedProcess:
    """
    Executes a shell command and prints the result.
    """
    print(f"\n--- {description} ---")
    try:
        # We allow output to flow to stdout/stderr for real-time feedback
        # unless capture_output is True.
        result = subprocess.run(
            command, capture_output=capture_output, text=True, check=True
        )
        print("Result: [SUCCESS]")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Result: [FAILED] (Exit code: {e.returncode})")
        if capture_output:
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)
        return e
    except FileNotFoundError:
        print("Result: [ERROR] (Command not found.)")
        sys.exit(1)


def run_bash_script(script: str, description: str) -> bool:
    """
    Run a multi-line bash script string and stream output to the terminal.
    Returns True on success, False on failure.
    """
    print(f"\n--- {description} ---")
    result = subprocess.run(["bash", "-c", script], text=True, check=False)
    if result.returncode == 0:
        print("Result: [SUCCESS]")
    else:
        print(f"Result: [FAILED] (Exit code: {result.returncode})")
    return result.returncode == 0


def run_pytest_ci(no_sync: bool = False) -> bool:
    """
    Harvest and run the pytest step from ci.yml's pytest_matrix job.
    Adapts 'python -m pytest' to 'uv run pytest' for local execution.
    """
    script = commands_from_ci_yml("pytest_matrix", "Run tests")
    if script is None:
        return False

    uv_pytest = f"uv run {'--no-sync ' if no_sync else ''}pytest"
    script = script.replace("python -m pytest", uv_pytest)
    print(f"  Local adaptation: 'python -m pytest'  ->  '{uv_pytest}'")

    return run_bash_script(script, "pytest (harvested from ci.yml / pytest_matrix)")


def validate_tag_checks(tag: str) -> bool:
    """
    Validate a tag against the same three rules enforced by the validate_tag CI job:
      1. Current branch must be 'main' or 'dev'.
      2. Tag must conform to PEP 440.
      3. Tag must be strictly newer than all existing tags.

    Requires 'packaging' (a direct project dependency). Run 'uv sync' if not available.
    Returns True if all checks pass, False otherwise.
    """
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        print(
            "[!] 'packaging' is not installed. Run 'uv sync --all-extras --dev' first."
        )
        return False

    print(f"\n--- Tag Validation: '{tag}' ---")

    # Check 1: current branch must be main or dev
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    branch = result.stdout.strip()
    if branch not in ("main", "dev"):
        print(
            f"  [FAIL] Branch: current branch is '{branch}'. Must be 'main' or 'dev'."
        )
        return False
    print(f"  [OK]   Branch: on '{branch}'.")

    # Check 2: PEP 440 format
    version_str = tag.lstrip("v")
    try:
        new_version = Version(version_str)
        print(f"  [OK]   PEP 440: '{tag}' is valid.")
    except InvalidVersion:
        print(f"  [FAIL] PEP 440: '{tag}' is not a valid PEP 440 version.")
        print(
            "         Valid examples: v1.0.0  v1.1.0a1  v1.1.0rc1"
            "  v1.0.0.post1  v1.1.0.dev1"
        )
        return False

    # Check 3: version must be strictly newer than all existing tags
    result = subprocess.run(["git", "tag"], capture_output=True, text=True, check=False)
    existing = []
    for t in result.stdout.strip().splitlines():
        t = t.strip()
        if t == tag:
            continue
        try:
            existing.append((Version(t.lstrip("v")), t))
        except InvalidVersion:
            pass  # skip tags that don't conform to PEP 440

    if not existing:
        print(f"  [OK]   Monotonicity: '{tag}' will be the first release.")
        return True

    latest_version, latest_tag = max(existing, key=lambda x: x[0])
    if new_version <= latest_version:
        print(
            f"  [FAIL] Monotonicity: '{tag}' ({new_version}) is not newer"
            f" than '{latest_tag}' ({latest_version})."
        )
        return False
    print(
        f"  [OK]   Monotonicity: '{tag}' ({new_version}) is newer"
        f" than '{latest_tag}' ({latest_version})."
    )
    return True


def main() -> None:
    """
    Main entry point for preflight checks.
    """
    parser = argparse.ArgumentParser(
        description="Preflight checks for CI/CD readiness."
    )
    parser.add_argument(
        "--all-tests",
        action="store_true",
        help=(
            "Run all tests in 'tests/' and lint all source files"
            " (matches CI on main/dev)."
        ),
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help=(
            "Add coverage reporting to the pytest run"
            " (--cov=rattlesnake --cov-report=term-missing)."
        ),
    )
    parser.add_argument(
        "--tag",
        metavar="TAG",
        default=None,
        help=(
            "Validate TAG before pushing a release"
            " (branch, PEP 440, monotonicity). Runs before lint/tests."
        ),
    )
    parser.add_argument(
        "--docs",
        action="store_true",
        help=(
            "Build the Jupyter Book documentation"
            " (requires network access to api.mystmd.org)."
        ),
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help=(
            "Skip 'uv' dependency synchronization"
            " (useful when offline or behind a firewall)."
        ),
    )
    parser.add_argument(
        "--skip-network-check",
        action="store_true",
        help="Skip the initial network connectivity check.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Continue even if network checks or synchronization fails.",
    )
    args = parser.parse_args()

    print("Starting Rattlesnake Preflight Checks...")

    # 1. Check if 'uv' is installed
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[X] Error: 'uv' is not installed or not in PATH.")
        print("    Please install uv: https://github.com/astral-sh/uv")
        sys.exit(1)

    # 2. Network connectivity check
    if not args.skip_network_check and not args.no_sync:
        print("Checking network connectivity...", end=" ", flush=True)
        success, err = check_connectivity()
        if success:
            print("[OK]")
        else:
            print("[FAILED]")
            print(f"\n[!] Network check failed: {err}")
            print("    It is highly likely that 'uv' will fail to sync dependencies.")

            print("\n    Suggestions:")
            if "CERTIFICATE" in err:
                print("    - Set SSL_CERT_FILE to your corporate CA bundle.")
                print(
                    "    - On macOS, ensure your certificate is in the System Keychain."
                )
                print(
                    "    - Or use '--no-sync' if your environment is already prepared."
                )
            else:
                print("    - Ensure HTTP_PROXY and HTTPS_PROXY are set correctly.")
                print("    - Use '--no-sync' to bypass network synchronization.")

            if not args.force:
                print("\n[!] Aborting to avoid long 'uv' error messages.")
                print("    Use '--force' if you believe this check is incorrect.")
                print("-" * 20)
                sys.exit(1)
            print("-" * 20)

    # 3. Environment Synchronization (Fail Fast)
    if not args.no_sync:
        print(
            "\nChecking/Syncing environment dependencies (including extras)...",
            flush=True,
        )
        # We sync all extras to ensure tools like pytest and ruff are available
        sync_res = subprocess.run(
            ["uv", "sync", "--all-extras", "--dev"],
            capture_output=True,
            text=True,
            check=False,
        )
        if sync_res.returncode != 0:
            print("Result: [FAILED]")
            print(
                "\n[!] 'uv sync' failed. This usually indicates a network or SSL issue."
            )

            # Extract common error patterns from uv output
            if (
                "UnknownIssuer" in sync_res.stderr
                or "invalid peer certificate" in sync_res.stderr
            ):
                print("\n    Detected SSL Certificate Issue (Unknown Issuer).")
                print(
                    "    - Solution: export SSL_CERT_FILE=/path/to/your/ca-bundle.pem"
                )
            elif "Connect" in sync_res.stderr or "timeout" in sync_res.stderr:
                print("\n    Detected Connection Issue.")
                print(
                    "    - Solution: Check your proxy settings"
                    " (HTTP_PROXY/HTTPS_PROXY)."
                )

            print("\n    Full error from 'uv':")
            print("-" * 20)
            print(sync_res.stderr.strip())
            print("-" * 20)

            if not args.force:
                print(
                    "\n[!] Aborting preflight. Fix the environment or use '--no-sync'."
                )
                sys.exit(1)
        else:
            print("Result: [SUCCESS]")

    # 4. Tag validation (fail fast before lint/tests, matching CI's validate_tag job)
    if args.tag:
        if not validate_tag_checks(args.tag):
            print("\n" + "=" * 40)
            print("PREFLIGHT FAILED: Fix tag errors before pushing.")
            sys.exit(1)

    # 5. Run actual validation steps
    sync_flag = "--no-sync" if args.no_sync else ""
    # TODO: CBH re-enable once test_acquisition hanging issue is resolved with Dan
    # cov_flags = "--cov=rattlesnake --cov-report=term-missing" if args.coverage else ""

    if args.all_tests:
        steps = [
            (
                f"uv run {sync_flag} ruff format --check src/rattlesnake/",
                "Full Ruff Format Check",
            ),
            (
                f"uv run {sync_flag} pylint src/rattlesnake",
                "Full Pylint Analysis",
            ),
            # TODO: CBH re-enable once test_acquisition hang is resolved with Dan
            # (f"uv run {sync_flag} pytest tests/ {cov_flags}",
            #  "Full Test Suite"
            #  + (" (with coverage)" if args.coverage else "")),
        ]
    else:
        steps = [
            (
                f"uv run {sync_flag} ruff format --check src/rattlesnake/",
                "Ruff Format Check",
            ),
            (
                f"uv run {sync_flag} pylint src/rattlesnake",
                "Pylint Analysis",
            ),
            # TODO: CBH re-enable once test_acquisition hang is resolved with Dan
            # (f"uv run {sync_flag} pytest tests/ --ignore=tests/long {cov_flags}",
            #  "Tests (tests/ --ignore=tests/long)"
            #  + (" with coverage" if args.coverage else "")),
        ]

    all_passed = True
    for cmd_str, desc in steps:
        cmd = shlex.split(cmd_str)
        # Filter out empty strings from shlex.split if any
        cmd = [c for c in cmd if c]

        res = run_step(cmd, desc)
        if isinstance(res, subprocess.CalledProcessError):
            all_passed = False

    # 6. Jupyter Book build (optional — requires network access)
    if args.docs:
        print("\n--- Jupyter Book Build (--html --strict) ---")
        print("Note: Requires network access to api.mystmd.org.")
        uv_sync_flag = ["--no-sync"] if args.no_sync else []
        docs_result = subprocess.run(
            ["uv", "run"]
            + uv_sync_flag
            + ["jupyter", "book", "build", "--html", "--strict"],
            cwd="documentation",
            text=True,
            check=False,
        )
        if docs_result.returncode != 0:
            print(f"Result: [FAILED] (Exit code: {docs_result.returncode})")
            all_passed = False
        else:
            print("Result: [SUCCESS]")

    print("\n" + "=" * 40)
    if all_passed:
        print("PREFLIGHT PASSED: Ready to push!")
        sys.exit(0)
    else:
        print("PREFLIGHT FAILED: Fix errors before pushing.")
        if not args.no_sync:
            print(
                "\n[TIP] If synchronization failed,"
                " try 'python preflight.py --no-sync'."
            )
        if not args.all_tests:
            print(
                "\n[TIP] To run the full suite (including tests/long/),"
                " use 'python preflight.py --all-tests'."
            )
        if not args.coverage:
            print(
                "\n[TIP] To include a coverage report,"
                " add '--coverage' to your command."
            )
        if not args.docs:
            print(
                "\n[TIP] To check the documentation build,"
                " add '--docs' (requires network access)."
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
