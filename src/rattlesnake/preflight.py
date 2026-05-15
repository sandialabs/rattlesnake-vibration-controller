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
    uv run preflight [--no-sync] [--skip-network-check] [--all-tests] [--coverage] [--tag TAG] [--docs] [--force]

Examples:
    uv run preflight                         # Default: matches CI's non-main/dev scope
    uv run preflight --all-tests             # Full suite: matches CI on main/dev (slow)
    uv run preflight --coverage              # Default scope + coverage report
    uv run preflight --all-tests --coverage  # Full suite + coverage report
    uv run preflight --tag v1.0.0rc1         # Validate tag, then run default scope
    uv run preflight --tag v1.0.0 --all-tests  # Validate tag, then run full suite
    uv run preflight --docs                  # Build Jupyter Book (requires network access)
    uv run preflight --no-sync               # Skip dependency sync (useful when offline/firewall)
    uv run preflight --force                 # Continue even if network/sync checks fail
    uv run preflight --skip-network-check    # Skip the initial connectivity check

"""

import argparse
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.request


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
            "         Valid examples: v1.0.0  v1.1.0a1  v1.1.0rc1  v1.0.0.post1  v1.1.0.dev1"
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
            f"  [FAIL] Monotonicity: '{tag}' ({new_version}) is not newer than '{latest_tag}' ({latest_version})."
        )
        return False
    print(
        f"  [OK]   Monotonicity: '{tag}' ({new_version}) is newer than '{latest_tag}' ({latest_version})."
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
        help="Run all tests in 'tests/' and lint all source files (matches CI on main/dev).",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Add coverage reporting to the pytest run (--cov=rattlesnake --cov-report=term-missing).",
    )
    parser.add_argument(
        "--tag",
        metavar="TAG",
        default=None,
        help="Validate TAG before pushing a release (branch, PEP 440, monotonicity). Runs before lint/tests.",
    )
    parser.add_argument(
        "--docs",
        action="store_true",
        help="Build the Jupyter Book documentation (requires network access to api.mystmd.org).",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip 'uv' dependency synchronization (useful when offline or behind a firewall).",
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
                    "    - Solution: Check your proxy settings (HTTP_PROXY/HTTPS_PROXY)."
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
            # TODO: CBH re-enable once test_acquisition hanging issue is resolved with Dan
            # (f"uv run {sync_flag} pytest tests/ {cov_flags}", "Full Test Suite" + (" (with coverage)" if args.coverage else "")),
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
            # TODO: CBH re-enable once test_acquisition hanging issue is resolved with Dan
            # (f"uv run {sync_flag} pytest tests/ --ignore=tests/long {cov_flags}", "Tests (tests/ --ignore=tests/long)" + (" with coverage" if args.coverage else "")),
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
                "\n[TIP] If synchronization failed, try 'python preflight.py --no-sync'."
            )
        if not args.all_tests:
            print(
                "\n[TIP] To run the full suite (including tests/long/), use 'python preflight.py --all-tests'."
            )
        if not args.coverage:
            print(
                "\n[TIP] To include a coverage report, add '--coverage' to your command."
            )
        if not args.docs:
            print(
                "\n[TIP] To check the documentation build, add '--docs' (requires network access)."
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
