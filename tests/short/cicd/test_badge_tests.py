"""
Unit tests for badge_tests.py — Tests Badge Generator.
"""

import json
from unittest.mock import patch

from rattlesnake.cicd.badge_tests import main, parse_junit


def test_parse_junit_testsuite_root(tmp_path):
    """Test parsing a JUnit XML report with a bare <testsuite> root."""
    junit_xml = (
        '<testsuite name="pytest" tests="654" failures="0" errors="0" '
        'skipped="0"></testsuite>'
    )
    file = tmp_path / "junit.xml"
    file.write_text(junit_xml)
    assert parse_junit(str(file)) == (654, 654, 0, 0)


def test_parse_junit_testsuites_root(tmp_path):
    """Test parsing a JUnit XML report with a <testsuites><testsuite> root."""
    junit_xml = (
        "<testsuites>"
        '<testsuite name="pytest" tests="10" failures="1" errors="1" '
        'skipped="2"></testsuite>'
        "</testsuites>"
    )
    file = tmp_path / "junit.xml"
    file.write_text(junit_xml)
    # total=10, failed=failures+errors=2, skipped=2, passed=10-2-2=6
    assert parse_junit(str(file)) == (10, 6, 2, 2)


def test_parse_junit_invalid(tmp_path):
    """Test parsing a malformed XML file."""
    file = tmp_path / "bad.xml"
    file.write_text("<invalid")
    assert parse_junit(str(file)) == (0, 0, 0, 0)


@patch("requests.get")
def test_main_success_all_pass(mock_get, tmp_path):
    """Test main function for a fully-passing suite badge generation."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b"svg-content"

    junit_xml = (
        '<testsuite name="pytest" tests="654" failures="0" errors="0" '
        'skipped="0"></testsuite>'
    )
    input_file = tmp_path / "junit.xml"
    input_file.write_text(junit_xml)
    output_dir = tmp_path / "badges"

    test_args = [
        "badge_tests.py",
        "--input_file",
        str(input_file),
        "--output_dir",
        str(output_dir),
        "--github_repo",
        "owner/repo",
        "--deploy_subdir",
        "dev",
        "--run_id",
        "123",
    ]
    with patch("sys.argv", test_args):
        main()

    assert (output_dir / "tests.svg").exists()
    assert (output_dir / "tests.svg").read_bytes() == b"svg-content"
    assert (output_dir / "tests-info.json").exists()
    with open(output_dir / "tests-info.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
        assert metadata["total"] == 654
        assert metadata["passed"] == 654
        assert metadata["failed"] == 0
        assert metadata["skipped"] == 0
        assert metadata["message"] == "654 pass 0 fail"
        assert metadata["color"] == "brightgreen"


@patch("requests.get")
def test_main_with_failures(mock_get, tmp_path):
    """Test main function for a badge generation with failing/skipped tests."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b"svg-content"

    junit_xml = (
        '<testsuite name="pytest" tests="10" failures="2" errors="1" '
        'skipped="1"></testsuite>'
    )
    input_file = tmp_path / "junit.xml"
    input_file.write_text(junit_xml)
    output_dir = tmp_path / "badges"

    test_args = [
        "badge_tests.py",
        "--input_file",
        str(input_file),
        "--output_dir",
        str(output_dir),
        "--github_repo",
        "owner/repo",
        "--deploy_subdir",
        "main",
        "--run_id",
        "456",
    ]
    with patch("sys.argv", test_args):
        main()

    with open(output_dir / "tests-info.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
        assert metadata["failed"] == 3
        assert metadata["passed"] == 6
        assert metadata["skipped"] == 1
        assert metadata["message"] == "6 pass 3 fail 1 skip"
        assert metadata["color"] == "red"
