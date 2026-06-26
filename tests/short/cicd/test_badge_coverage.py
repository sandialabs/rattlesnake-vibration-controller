"""
Unit tests for badge_coverage.py — Coverage Badge Generator.
"""

import json
import os
from unittest.mock import patch

import pytest
from rattlesnake.cicd.badge_coverage import extract_coverage, main


@pytest.fixture
def mock_github_env_file(tmp_path):
    """Pytest fixture to set up a mock GITHUB_ENV file."""
    mock_env_file = tmp_path / "github_env"
    os.environ["GITHUB_ENV"] = str(mock_env_file)
    yield mock_env_file
    if "GITHUB_ENV" in os.environ:
        del os.environ["GITHUB_ENV"]


def test_extract_coverage_valid(tmp_path):
    """Test extracting coverage from a valid XML file."""
    coverage_xml = '<coverage line-rate="0.925"></coverage>'
    file = tmp_path / "coverage.xml"
    file.write_text(coverage_xml)
    assert extract_coverage(str(file)) == 92.5


def test_extract_coverage_invalid(tmp_path):
    """Test extracting coverage from a malformed XML file."""
    file = tmp_path / "bad.xml"
    file.write_text("<invalid")
    assert extract_coverage(str(file)) == 0.0


@patch("requests.get")
def test_main_success(mock_get, tmp_path, mock_github_env_file):
    """Test main function for a successful badge generation."""
    # Mock shields.io response
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b"svg-content"

    coverage_xml = '<coverage line-rate="0.85"></coverage>'
    input_file = tmp_path / "coverage.xml"
    input_file.write_text(coverage_xml)
    output_dir = tmp_path / "badges"

    # Mock command line arguments
    test_args = [
        "badge_coverage.py",
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
        "--export_env",
    ]

    with patch("sys.argv", test_args):
        main()

    # Check SVG file
    assert (output_dir / "coverage.svg").exists()
    assert (output_dir / "coverage.svg").read_bytes() == b"svg-content"

    # Check JSON file
    assert (output_dir / "coverage-info.json").exists()
    with open(output_dir / "coverage-info.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
        assert metadata["coverage"] == "85.0"
        assert metadata["color"] == "green"

    # Check GITHUB_ENV
    with open(mock_github_env_file, "r", encoding="utf-8") as f:
        env_content = f.read()
        assert "COVERAGE=85.0" in env_content
        assert "BADGE_COLOR_COV=green" in env_content
