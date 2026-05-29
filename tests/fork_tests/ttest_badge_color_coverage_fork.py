from pathlib import Path
import os

import pytest

from rattlesnake.cicd.badge_color_coverage import get_coverage_and_color


@pytest.fixture
def mock_github_env_file(tmp_path):
    """
    Pytest fixture to set up and tear down a mock GITHUB_ENV file.
    """
    mock_env_file = tmp_path / "github_env"
    os.environ["GITHUB_ENV"] = str(mock_env_file)
    yield mock_env_file
    # Teardown: remove the environment variable after the test
    if "GITHUB_ENV" in os.environ:
        del os.environ["GITHUB_ENV"]


def test_get_coverage_and_color_valid_file(tmp_path, capsys, mock_github_env_file):
    """
    Tests get_coverage_and_color with a valid coverage.xml file.
    """
    coverage_xml_content = '<coverage line-rate="0.925"></coverage>'
    coverage_file = tmp_path / "coverage.xml"
    coverage_file.write_text(coverage_xml_content)

    get_coverage_and_color(str(coverage_file))

    with open(mock_github_env_file, "r") as f:
        content = f.read()
        assert "COVERAGE=92.5" in content
        assert "BADGE_COLOR_COV=brightgreen" in content

    captured = capsys.readouterr()
    assert "Coverage: 92.5%" in captured.out


def test_get_coverage_and_color_missing_file(tmp_path, capsys, mock_github_env_file):
    """
    Tests get_coverage_and_color with a missing coverage.xml file.
    """
    get_coverage_and_color("non_existent_file.xml")

    with open(mock_github_env_file, "r", encoding="utf8") as f:
        content = f.read()
        assert "COVERAGE=0.0" in content
        assert "BADGE_COLOR_COV=red" in content

    captured = capsys.readouterr()
    assert "Coverage: 0.0%" in captured.out


def test_get_coverage_and_color_malformed_xml(tmp_path, capsys, mock_github_env_file):
    """
    Tests get_coverage_and_color with a malformed coverage.xml file.
    """
    malformed_xml_content = '<coverage line-rate="0.invalid"></coverage>'
    coverage_file = tmp_path / "malformed.xml"
    coverage_file.write_text(malformed_xml_content)

    get_coverage_and_color(str(coverage_file))

    with open(mock_github_env_file, "r", encoding="utf8") as f:
        content = f.read()
        assert "COVERAGE=0.0" in content
        assert "BADGE_COLOR_COV=red" in content

    captured = capsys.readouterr()
    assert "Coverage: 0.0%" in captured.out


def test_get_coverage_and_color_no_github_env(tmp_path, capsys):
    """
    Tests get_coverage_and_color when GITHUB_ENV is not set.
    """
    coverage_xml_content = '<coverage line-rate="0.50"></coverage>'
    coverage_file = tmp_path / "coverage.xml"
    coverage_file.write_text(coverage_xml_content)

    # Ensure GITHUB_ENV is not set for this specific test
    if "GITHUB_ENV" in os.environ:
        original_github_env = os.environ["GITHUB_ENV"]
        del os.environ["GITHUB_ENV"]
    else:
        original_github_env = None

    get_coverage_and_color(str(coverage_file))

    captured = capsys.readouterr()
    assert "Coverage: 50.0%" in captured.out

    # Restore GITHUB_ENV if it was originally set
    if original_github_env is not None:
        os.environ["GITHUB_ENV"] = original_github_env
