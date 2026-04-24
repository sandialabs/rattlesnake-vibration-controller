"""
Unit tests for report_coverage.py — Coverage HTML Report Generator.

This test suite verifies the correctness of the functions used to parse
coverage.xml output and generate a custom HTML report.
"""

import types
from pathlib import Path

import pytest
from rattlesnake.cicd.report_coverage import (
    CoverageMetric,
    get_coverage_metric,
    get_report_html,
    main,
    run_coverage_report,
)


def test_coverage_metric_calculation():
    """Test the coverage percentage calculation in CoverageMetric."""
    cm = CoverageMetric(lines_valid=100, lines_covered=75)
    assert cm.coverage == 75.0
    assert cm.color == "yellow"

    cm_zero = CoverageMetric(lines_valid=0, lines_covered=0)
    assert cm_zero.coverage == 0.0


def test_get_coverage_metric_success(tmp_path):
    """Test extracting metrics from a valid coverage.xml file."""
    coverage_xml = '<coverage lines-valid="100" lines-covered="85"></coverage>'
    file_path = tmp_path / "coverage.xml"
    file_path.write_text(coverage_xml)

    cm = get_coverage_metric(file_path)
    assert cm.lines_valid == 100
    assert cm.lines_covered == 85
    assert cm.coverage == 85.0


def test_get_coverage_metric_missing_file():
    """Test get_coverage_metric with a non-existent file."""
    cm = get_coverage_metric(Path("non_existent.xml"))
    assert cm.lines_valid == 0
    assert cm.lines_covered == 0


def test_get_report_html():
    """Test generating the HTML report string."""
    cm = CoverageMetric(lines_valid=100, lines_covered=95)
    html = get_report_html(
        cm,
        "20240101_120000_UTC",
        "run1",
        "main",
        "sha123456789",
        "owner/repo",
    )
    assert "<h1>Coverage Report</h1>" in html
    assert "Coverage: 95.00%" in html
    assert "run1" in html
    assert "sha1234" in html


def test_run_coverage_report(tmp_path):
    """Test the orchestration of the coverage report generation."""
    coverage_xml = '<coverage lines-valid="100" lines-covered="90"></coverage>'
    input_file = tmp_path / "coverage.xml"
    input_file.write_text(coverage_xml)
    output_file = tmp_path / "report.html"

    cm = run_coverage_report(
        str(input_file),
        str(output_file),
        "20240101_120000_UTC",
        "run1",
        "main",
        "sha1",
        "owner/repo",
    )

    assert cm.coverage == 90.0
    assert output_file.exists()
    assert "Coverage: 90.00%" in output_file.read_text()


def test_main_success(monkeypatch, capsys, tmp_path):
    """Test the main function for a successful run."""
    coverage_xml = '<coverage lines-valid="100" lines-covered="100"></coverage>'
    input_file = tmp_path / "coverage.xml"
    input_file.write_text(coverage_xml)
    output_file = tmp_path / "report.html"

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
    assert "[OK] Coverage HTML report generated" in captured.out
    assert output_file.exists()


def test_main_error(monkeypatch, capsys):
    """Test the main function when an error occurs."""
    mock_args = types.SimpleNamespace(
        input_file="missing.xml",
        output_file="output.html",
        timestamp="20240101_120000_UTC",
        run_id="run1",
        ref_name="main",
        github_sha="sha1",
        github_repo="owner/repo",
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)

    exit_code = main()
    # Even if file is missing, it prints error but might return 0 or 1 depending on implementation
    # Based on our implementation, it returns 0 after printing "Error processing coverage file"
    # Wait, looking at main() in report_coverage.py, it doesn't catch the FileNotFoundError
    # if it's thrown inside get_coverage_metric (which it isn't, it's caught there).
    # Actually, main() has a try/except FileNotFoundError.
    # But get_coverage_metric catches FileNotFoundError and returns cm with 0 lines.
    # So main() won't see FileNotFoundError.
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Error processing coverage file" in captured.out
