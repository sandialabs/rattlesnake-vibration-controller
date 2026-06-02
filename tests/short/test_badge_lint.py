"""
Unit tests for badge_lint.py — Lint Badge Generator.
"""

import json
import os
from unittest.mock import patch

import pytest
from rattlesnake.cicd.badge_lint import extract_score, main


@pytest.fixture
def mock_github_env_file(tmp_path):
    """Pytest fixture to set up a mock GITHUB_ENV file."""
    mock_env_file = tmp_path / "github_env"
    os.environ["GITHUB_ENV"] = str(mock_env_file)
    yield mock_env_file
    if "GITHUB_ENV" in os.environ:
        del os.environ["GITHUB_ENV"]


def test_extract_score_valid(tmp_path):
    """Test extracting score from a valid lint output file."""
    lint_output = "Your code has been rated at 9.50/10"
    file = tmp_path / "lint_output.txt"
    file.write_text(lint_output)
    assert extract_score(str(file)) == 9.50


def test_extract_score_invalid(tmp_path):
    """Test extracting score from a missing or malformed file."""
    file = tmp_path / "bad.txt"
    file.write_text("no score here")
    assert extract_score(str(file)) == 0.0


@patch("requests.get")
def test_main_success(mock_get, tmp_path, mock_github_env_file):
    """Test main function for a successful lint badge generation."""
    # Mock shields.io response
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b"svg-content"

    lint_output = "Your code has been rated at 8.75/10"
    input_file = tmp_path / "lint_output.txt"
    input_file.write_text(lint_output)
    output_dir = tmp_path / "badges"

    # Mock command line arguments
    test_args = [
        "badge_lint.py",
        "--input_file", str(input_file),
        "--output_dir", str(output_dir),
        "--github_repo", "owner/repo",
        "--deploy_subdir", "dev",
        "--run_id", "123",
        "--export_env"
    ]

    with patch("sys.argv", test_args):
        main()

    # Check SVG file
    assert (output_dir / "lint.svg").exists()
    assert (output_dir / "lint.svg").read_bytes() == b"svg-content"

    # Check JSON file
    assert (output_dir / "lint-info.json").exists()
    with open(output_dir / "lint-info.json", "r") as f:
        metadata = json.load(f)
        assert metadata["score"] == "8.75"
        assert metadata["color"] == "brightgreen"

    # Check GITHUB_ENV
    with open(mock_github_env_file, "r") as f:
        env_content = f.read()
        assert "BADGE_COLOR=brightgreen" in env_content
