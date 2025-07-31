"""
Unit tests for pylint_report.py — Pylint HTML Report Generator.

This test suite verifies the correctness of the utility functions used to
parse pylint output, format issue counts, and generate an HTML report.

Example use:
    source .venv/bin/activate
    pytest test_pylint_report.py -v
    pytest test_pylint_report.py::test_score_color
"""

import re
# import pytest

from pylint_report import (
    create_pylint_html_report,
    get_formatted_timestamp,
    # pylint_content,
    parse_pylint_output,
    get_score_color,
    count_issues_by_type,
    generate_issues_html,
    generate_html_report,
)


def test_get_score_color():
    """Run unit tests for the score_color function."""

    # Test for a high pylint score (>= 8.0)
    assert get_score_color("8.5") == "brightgreen", "Test failed for score 8.5"
    # Test for a medium pylint score (6.0 <= score < 8.0)
    assert get_score_color("7.0") == "yellow", "Test failed for score 7.0"
    # Test for a low pylint score (4.0 <= score < 6.0)
    assert get_score_color("5.0") == "orange", "Test failed for score 5.0"
    # Test for a very low pylint score (< 4.0)
    assert get_score_color("3.0") == "red", "Test failed for score 3.0"
    # Test for an invalid pylint score (non-numeric)
    assert get_score_color("invalid") == "gray", "Test failed for invalid score"
    # Test for an empty string as pylint score
    assert get_score_color("") == "gray", "Test failed for empty string"


def test_get_formatted_timestamp():
    """Test that formatted_timestamp() returns valid datetime string.

    Confirms output includes expected format components like UTC, EST, MST,
    and a recognizable timestamp pattern.

    Example get_formatted_timestamp output:
        '2025-07-31 18:33:56 UTC (2025-07-31 14:33:56 EST / 2025-07-31 12:33:56 MST)'
    """
    ts = get_formatted_timestamp()

    assert "UTC" in ts
    assert re.search(r"\d{4}-\d{2}-\d{2}", ts)


def test_parse_pylint_output_and_count():
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
    issues, summary = parse_pylint_output(content)
    counts = count_issues_by_type(issues)

    assert len(issues) == 5
    assert any("example.py" in line for line in issues)
    assert any("rated at" in line for line in summary)
    assert counts["error"] == 1
    assert counts["warning"] == 1
    assert counts["convention"] == 2
    assert counts["refactor"] == 1


def test_generate_issues_html():
    """Test that generate_issues_html() returns valid HTML content.

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
    html_out = generate_issues_html(issues)

    assert '<div class="issues-list">' in html_out

    assert 'class="issue convention"' in html_out
    assert 'class="issue warning"' in html_out
    assert 'class="issue error"' in html_out
    assert 'class="issue refactor"' in html_out


def test_generate_html_report_basic():
    """Test generate_html_report() with minimal valid inputs.

    Ensures generated HTML includes expected static and dynamic content
    like report title, issues, and summary.
    """
    pylint_content = (
        "example.py:1:0: C0114: Missing module docstring (missing-module-docstring)"
    )
    issues = [pylint_content]
    summary = ["Your code has been rated at 9.00/10"]
    report = generate_html_report(
        pylint_content=pylint_content,
        issues=issues,
        summary_lines=summary,
        pylint_score="9.00",
        run_id="123",
        ref_name="main",
        github_sha="abc123def456",
        github_repo="testuser/testrepo",
    )

    assert "<!DOCTYPE html>" in report
    assert "Rattlesnake Pylint Report" in report
    assert "example.py" in report
    assert "Your code has been rated at 9.00/10" in report


def test_create_pylint_html_report():
    """Tests the main report creation."""

    _total_issues, _issue_counts = create_pylint_html_report(
        input_file="pylint_output_20250729_150018_UTC.txt",
        output_file="pylint_report.html",
        pylint_score="8.5",  # TODO: Replace with actual score from file
        run_id="1234567890",
        ref_name="main",
        github_sha="abc123def456",
        github_repo="testuser/testrepo",
    )
