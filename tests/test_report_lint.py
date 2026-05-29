"""
Unit tests for report_lint.py — Lint HTML Report Generator.

This test suite verifies the correctness of the functions used to parse
lint output and generate a custom HTML report.
"""

import types

import pytest
from rattlesnake.cicd.report_lint import (
    get_lint_content,
    get_lint_sections,
    get_score_from_summary,
    main,
)
from rattlesnake.cicd.utilities import ReportMetadata, get_html_report_footer, get_html_report_header


def test_get_lint_content_success(tmp_path):
    """Test that get_lint_content successfully reads a file."""
    content = "lint data"
    file_path = tmp_path / "lint_output.txt"
    file_path.write_text(content)
    assert get_lint_content(str(file_path)) == content


def test_get_lint_content_file_not_found():
    """Test that get_lint_content raises FileNotFoundError when file is missing."""
    with pytest.raises(FileNotFoundError):
        get_lint_content("non_existent_file.txt")


def test_get_lint_sections():
    """Test parsing lint output into issues and summary sections."""
    lint_output = (
        "path/to/file.py:10:5: C0114: Missing module docstring\n"
        "-------------------\n"
        "Your code has been rated at 9.50/10"
    )
    issues, summary = get_lint_sections(lint_output)
    assert len(issues) == 1
    assert "C0114" in issues[0]
    # summary includes everything after the first separator
    assert any("Your code has been rated at 9.50/10" in line for line in summary)


def test_get_lint_sections_long_dashes():
    """Test parsing lint output with longer dash separators (reproduction of bug)."""
    lint_output = (
        "path/to/file.py:10:5: C0114: Missing module docstring\n"
        "-----------------------------------\n"
        "Your code has been rated at 9.51/10"
    )
    issues, summary = get_lint_sections(lint_output)
    assert len(issues) == 1
    assert "C0114" in issues[0]
    assert any("Your code has been rated at 9.51/10" in line for line in summary)
    assert get_score_from_summary(summary) == "9.51"


def test_get_lint_sections_no_separator():
    """Test parsing lint output with no separators."""
    lint_output = "Your code has been rated at 10.00/10"
    issues, summary = get_lint_sections(lint_output)
    assert any("Your code has been rated at 10.00/10" in line for line in summary)
    assert get_score_from_summary(summary) == "10.00"


def test_get_score_from_summary():
    """Test extracting the lint score from summary lines."""
    summary = ["Some noise", "Your code has been rated at 8.75/10", "More noise"]
    assert get_score_from_summary(summary) == "8.75"


def test_get_score_from_summary_not_found():
    """Test extracting score when it's missing from summary."""
    assert get_score_from_summary(["No score here"]) == "0.00"


def test_get_html_header():
    """Test generating the HTML header."""
    metadata = ReportMetadata(
        timestamp="20240101_120000_UTC",
        run_id="run123",
        ref_name="main",
        github_sha="sha",
        github_repo="owner/repo",
    )
    header = get_html_report_header("Title", "Label", "9.00", "green", metadata)
    assert "<title>Title</title>" in header
    assert "9.00" in header
    assert "run123" in header


def test_get_html_footer():
    """Test generating the HTML footer."""
    footer = get_html_report_footer()
    assert "GitHub Actions" in footer


def test_main_success(monkeypatch, capsys, tmp_path):
    """Test the main function for a successful run."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("Your code has been rated at 10.00/10")
    output_file = tmp_path / "output.html"

    mock_args = types.SimpleNamespace(
        input_file=str(input_file),
        output_file=str(output_file),
        timestamp="20240101_120000_UTC",
        run_id="run1",
        ref_name="main",
        github_sha="sha1",
        github_repo="owner/repo",
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)

    exit_code = main()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[OK] Lint report generated" in captured.out
    assert output_file.exists()


def test_main_error(monkeypatch, capsys):
    """Test the main function when an error occurs."""
    mock_args = types.SimpleNamespace(
        input_file="missing.txt",
        output_file="output.html",
        timestamp="20240101_120000_UTC",
        run_id="run1",
        ref_name="main",
        github_sha="sha1",
        github_repo="owner/repo",
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)

    exit_code = main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "[X] File Error:" in captured.out
