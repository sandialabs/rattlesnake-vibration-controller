"""
The module tests the pylint report module.

This test suite verifies the correctness of the utility functions used to
parse pylint output, format issue counts, and generate an HTML report.

Example use:
    source .venv/bin/activate
    pytest tests/test_report_pylint.py -v
    pytest tests/test_report_pylint.py::test_some_specific_function -v
    pytest --cov=src/rattlesnake --cov-report=xml --cov-report=html --cov-report=term-missing
"""

import types
from pathlib import Path
from typing import Final

import pytest

from rattlesnake.cicd.report_pylint import (
    get_issue_counts,
    get_issues_list_html,
    get_pylint_content,
    get_pylint_sections,
    get_report_html,
    get_score_from_summary,
    main,
    run_pylint_report,
    write_report,
)
from rattlesnake.cicd.utilities import get_score_color_lint


def test_get_pylint_content_success():
    """Test that get_pylint_content successfully reads a file."""
    content = "Hello, world!"
    file_path = Path(__file__).parent / "test_file.txt"
    file_path.write_text(content)
    assert file_path.is_file()
    assert get_pylint_content(str(file_path)) == content
    file_path.unlink()  # clean up and delete the temporary file


def test_get_pylint_content_file_not_found():
    """Test that get_pylint_content raises FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError) as excinfo:
        get_pylint_content("non_existent_file.txt")
    assert 'Input file not found: "non_existent_file.txt"' in str(excinfo.value)


def test_write_report_success(tmp_path):
    """Test that write_report successfully writes to a file."""
    content = "<html></html>"
    file_path = tmp_path / "report.html"
    write_report(content, str(file_path))
    assert file_path.read_text() == content


def test_write_report_io_error(monkeypatch):
    """Test that write_report raises IOError when file writing fails."""

    def mock_open_raises_io_error(*args, **kwargs):
        raise IOError("Permission denied")

    monkeypatch.setattr("builtins.open", mock_open_raises_io_error)
    with pytest.raises(IOError) as excinfo:
        write_report("<html></html>", "protected_file.html")
    assert 'Error writing output file "protected_file.html"' in str(excinfo.value)
    assert "Permission denied" in str(excinfo.value)


def test_main_success(monkeypatch, capsys):
    """Test the main function for a successful run."""
    mock_args = types.SimpleNamespace(
        input_file="dummy_input.txt",
        output_file="dummy_output.html",
        timestamp="20240101_120000_UTC",
        run_id="123",
        ref_name="main",
        github_sha="abc",
        github_repo="owner/repo",
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)
    monkeypatch.setattr(
        "rattlesnake.cicd.report_pylint.run_pylint_report",
        lambda *args, **kwargs: (
            10,
            {"convention": 1, "warning": 2, "error": 3, "refactor": 4},
            9.5,
        ),
    )

    main()
    captured = capsys.readouterr()
    assert "✅ Pylint HTML report generated: dummy_output.html" in captured.out
    assert "📊 Pylint score: 9.5/10" in captured.out
    assert "- Conventions: 1" in captured.out
    assert "- Warnings: 2" in captured.out
    assert "- Errors: 3" in captured.out
    assert "- Refactors: 4" in captured.out


def test_main_file_not_found(monkeypatch, capsys):
    """Test the main function when the input file is not found."""
    mock_args = types.SimpleNamespace(
        input_file="non_existent.txt",
        output_file="dummy_output.html",
        timestamp="20240101_120000_UTC",
        run_id="123",
        ref_name="main",
        github_sha="abc",
        github_repo="owner/repo",
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)
    monkeypatch.setattr(
        "rattlesnake.cicd.report_pylint.run_pylint_report",
        lambda *args, **kwargs: exec('raise FileNotFoundError("File not found")'),
    )

    exit_code = main()
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "❌ Error: The input file 'non_existent.txt' was not found." in captured.out


def test_main_io_error(monkeypatch, capsys):
    """Test the main function when an IOError occurs."""
    mock_args = types.SimpleNamespace(
        input_file="dummy_input.txt",
        output_file="dummy_output.html",
        timestamp="20240101_120000_UTC",
        run_id="123",
        ref_name="main",
        github_sha="abc",
        github_repo="owner/repo",
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)
    monkeypatch.setattr(
        "rattlesnake.cicd.report_pylint.run_pylint_report",
        lambda *args, **kwargs: exec('raise IOError("Permission denied")'),
    )

    exit_code = main()
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "❌ I/O error occurred: Permission denied" in captured.out


def test_main_unexpected_error(monkeypatch, capsys):
    """Test the main function for an unexpected error."""
    mock_args = types.SimpleNamespace(
        input_file="dummy_input.txt",
        output_file="dummy_output.html",
        timestamp="20240101_120000_UTC",
        run_id="123",
        ref_name="main",
        github_sha="abc",
        github_repo="owner/repo",
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)
    monkeypatch.setattr(
        "rattlesnake.cicd.report_pylint.run_pylint_report",
        lambda *args, **kwargs: exec('raise Exception("Something went wrong")'),
    )

    exit_code = main()
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "❌ An unexpected error occurred: Something went wrong" in captured.out


def test_get_score_color():
    """Run unit tests for the score_color function."""

    # Test for a high pylint score (>= 8.0)
    assert get_score_color_lint("8.5") == "brightgreen", "Test failed for score 8.5"
    # Test for a medium pylint score (6.0 <= score < 8.0)
    assert get_score_color_lint("7.0") == "yellow", "Test failed for score 7.0"
    # Test for a low pylint score (4.0 <= score < 6.0)
    assert get_score_color_lint("5.0") == "orange", "Test failed for score 5.0"
    # Test for a very low pylint score (< 4.0)
    assert get_score_color_lint("3.0") == "red", "Test failed for score 3.0"
    # Test for an invalid pylint score (non-numeric)
    assert get_score_color_lint("invalid") == "gray", "Test failed for invalid score"
    # Test for an empty string as pylint score
    assert get_score_color_lint("") == "gray", "Test failed for empty string"


def test_get_issue_counts():
    """Test parsing of pylint output and issue counting.

    Verifies that issues are extracted correctly and categorized
    into error, warning, and convention counts.
    """
    content = """
************* Module example
example.py:1:0: C0114: Missing module docstring (missing-module-docstring)
example.py:3:0: C0116: Missing function or method docstring (missing-function-docstring)
example.py:3:0: W0611: Unused import os (unused-import)
example.py:4:0: E1101: Instance of 'datetime' has no 'nowz' member (no-member)
example.py:5:0: R0913: Too many arguments (8/5) (too-many-arguments)

Your code has been rated at 6.00/10 (previous run: 7.00/10, -1.00)
"""
    issues, summary = get_pylint_sections(content)
    counts = get_issue_counts(issues)

    assert len(issues) == 5
    assert any("example.py" in line for line in issues)
    assert any("rated at" in line for line in summary)
    assert counts["error"] == 1
    assert counts["warning"] == 1
    assert counts["convention"] == 2
    assert counts["refactor"] == 1


def test_get_score_from_summary():
    """Unit test for get_score_from_summary function."""

    # Test cases
    test_cases = [
        # Test with a valid score
        (["Your code has been rated at 6.00/10", "", ""], "6.00"),
        (["Your code has been rated at 8.75/10", "", ""], "8.75"),
        # Test with multiple lines, only one containing the score
        (["Some other message", "Your code has been rated at 7.50/10", ""], "7.50"),
        # Test with an empty summary
        ([], "0.00"),
        # Test with no rating line
        (["No rating here", "", ""], "0.00"),
        # Test with an invalid format
        (["Your code has been rated at invalid/10", "", ""], "0.00"),
    ]

    for summary_lines, expected_score in test_cases:
        assert get_score_from_summary(summary_lines) == expected_score


def test_get_issues_list_html():
    """Test that get_issues_list_html returns valid HTML content.

    Verifies that issues are wrapped in the correct HTML structure and
    tagged with appropriate CSS classes.
    """
    issues = [
        "example.py:1:0: C0114: Missing module docstring (missing-module-docstring)",
        "example.py:3:0: W0611: Unused import os (unused-import)",
        "example.py:4:0: E1101: Instance of 'datetime' has no 'nowz' member (no-member)",
        "example.py:4:0: E1101: Instance of 'datetime' has no 'nowz' member (no-member)",
        "example.py:5:0: R0913: Too many arguments (8/5) (too-many-arguments)",
    ]
    html_out = get_issues_list_html(issues)

    assert '<div class="issues-list">' in html_out

    assert 'class="issue convention"' in html_out
    assert 'class="issue warning"' in html_out
    assert 'class="issue error"' in html_out
    assert 'class="issue refactor"' in html_out


def test_get_report_html_no_issues():
    """Test get_report_html with no issues."""

    issues_list = []  # empty list

    found = get_issues_list_html(issues=issues_list)
    known = "<p>No issues found! 🎉</p>"

    assert found == known


def test_get_report_html():
    """Test get_report_html with minimal valid inputs.

    Ensures generated HTML includes expected static and dynamic content
    like report title, issues, and summary.
    """
    pylint_content = "example.py:1:0: C0114: Missing module docstring (missing-module-docstring)"
    issues = [pylint_content]
    summary = ["Your code has been rated at 9.00/10"]
    report = get_report_html(
        pylint_content=pylint_content,
        issues=issues,
        summary_lines=summary,
        pylint_score="9.00",
        timestamp="20250815_211112_UTC",
        run_id="123",
        ref_name="main",
        github_sha="abc123def456",
        github_repo="testuser/testrepo",
    )

    assert "<!DOCTYPE html>" in report
    assert "Pylint Report" in report
    assert "example.py" in report
    assert "Your code has been rated at 9.00/10" in report


def test_run_pylint_report():
    """Tests the main report creation."""

    function_debug: Final[bool] = False  # set to True to avoid deleting the temporary output file

    fin = Path(__file__).parent / "files" / "pylint_output_20250729_150018_UTC.txt"
    assert fin.is_file(), "Input file does not exist."

    # Run the pylint report generation
    # This will create an HTML report in the same directory as this test file
    fout = Path(__file__).parent / "files" / "pylint_report_temp.html"

    _total_issues, _issue_counts, _pylint_score = run_pylint_report(
        input_file=str(fin),
        output_file=str(fout),
        timestamp="20250815_211112_UTC",
        run_id="1234567890",
        ref_name="main",
        github_sha="abc123def456",
        github_repo="testuser/testrepo",
    )

    # Check if the output file was created
    assert fout.is_file(), "Output HTML report was not created."
    print(f"Created temporary file: {fout}")

    if not function_debug:
        # Clean up the output file after the test
        fout.unlink(missing_ok=True)
        print(f"Deleted temporary file: {fout}")
    else:
        print(f"Retained output file: {fout}")
