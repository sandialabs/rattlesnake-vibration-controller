"""
This module tests the pytest and coverage report module.

Example use:
    source .venv/bin/activate
    pytest tests/test_report_pytest.py -v
    pytest tests/test_report_pytest.py::test_some_specific_function -v
    pytest --cov=src/rattlesnake --cov-report=html --cov-report=xml --cov-report=term-missing
"""

from pathlib import Path
import types
from typing import Final

import pytest

from rattlesnake.cicd.report_pytest import (
    CoverageMetric,
    get_coverage_metric,
    get_report_html,
    main,
    run_pytest_report,
)


def test_get_coverage_metrics_simple():
    """Tests the coverage metrics calculation on a simple example."""
    cm = CoverageMetric(lines_valid=100, lines_covered=80)

    assert cm.lines_valid == 100
    assert cm.lines_covered == 80
    assert cm.coverage == 80.0


def test_get_coverage_metrics_valid_file():
    """Tests a correct coverage metrics are returns from a valid coverage file."""

    fin = Path(__file__).parent / "files" / "coverage_output_20250807_241800_UTC.xml"
    assert fin.is_file()
    result = get_coverage_metric(fin)

    cm = CoverageMetric(lines_valid=10499, lines_covered=23)

    assert cm.lines_valid == 10499
    assert cm.lines_covered == 23
    assert cm.coverage == pytest.approx(
        0.2190684827126393, rel=1e-9
    )  # relative tolerance

    assert result.lines_valid == cm.lines_valid
    assert result.lines_covered == cm.lines_covered
    assert result.coverage == cm.coverage

    # Ensure the coverage percentage is calculated correctly
    assert result.coverage == (cm.lines_covered / cm.lines_valid * 100)


def test_get_coverage_metrics_bad_attributes():
    """Tests a correct coverage metrics are returns from an invalid coverage file."""

    fin = Path(__file__).parent / "files" / "coverage_bad_attributes.xml"
    assert fin.is_file()
    result = get_coverage_metric(fin)

    expected = CoverageMetric(lines_valid=0, lines_covered=0)

    assert result.lines_valid == expected.lines_valid
    assert result.lines_covered == expected.lines_covered
    assert result.coverage == expected.coverage

    # Ensure the coverage percentage is 0.0 when there are no valid lines
    assert result.coverage == 0.0


def test_get_coverage_metric_file_not_found():
    """Tests that get_coverage_metric handles a non-existent file."""
    fin = Path("non_existent_file.xml")
    result = get_coverage_metric(fin)
    expected = CoverageMetric(lines_valid=0, lines_covered=0)
    assert result.lines_valid == expected.lines_valid
    assert result.lines_covered == expected.lines_covered
    assert result.coverage == expected.coverage


def test_get_report_html():
    """Test get_report_html with minimal valid inputs.

    Ensures generated HTML includes expected static and dynamic content
    like report title, coverage, and summary.
    """
    coverage_metric = CoverageMetric(lines_valid=200, lines_covered=150)
    report = get_report_html(
        coverage_metric=coverage_metric,
        timestamp="20250815_211112_UTC",
        run_id="123",
        ref_name="main",
        github_sha="abc123def456",
        github_repo="testuser/testrepo",
    )

    assert "<!DOCTYPE html>" in report
    assert "Pytest Report" in report
    assert "Coverage: 75.00%" in report
    assert "<strong>Lines Covered:</strong> 150" in report
    assert "<strong>Total Lines:</strong> 200" in report
    assert 'href="https://github.com/testuser/testrepo/actions/runs/123"' in report
    assert 'href="https://github.com/testuser/testrepo/tree/main"' in report
    assert 'href="https://github.com/testuser/testrepo/commit/abc123def456"' in report


def test_run_pytest_report():
    """Tests the main report creation."""

    function_debug: Final[bool] = (
        False  # set to True to avoid deleting the temporary output file
    )

    fin = Path(__file__).parent / "files" / "coverage_output_20250807_241800_UTC.xml"
    assert fin.is_file(), "Input file does not exist"

    # Run the pytest report generation
    # This will create an HTML report in the same directory as this test file
    fout = Path(__file__).parent / "files" / "pytest_report_temp.html"

    aa, bb, cc, dd, ee = (
        "20250807_021110_UTC",  # timestamp
        "123",  # run_id
        "main",  # ref_name
        "abc123",  # github_sha
        "testuser/testrepo",  # github_repo
    )

    cm: CoverageMetric = run_pytest_report(
        input_file=str(fin),
        output_file=str(fout),
        timestamp=aa,
        run_id=bb,
        ref_name=cc,
        github_sha=dd,
        github_repo=ee,
    )

    # Generate HTML report
    _html_content: str = get_report_html(
        coverage_metric=cm,
        timestamp=aa,
        run_id=bb,
        ref_name=cc,
        github_sha=dd,
        github_repo=ee,
    )

    assert fout.is_file(), "Output HTML report was not created"
    print(f"Created temporary report: {fout}")

    if not function_debug:
        # Clean up the output file after test
        fout.unlink(missing_ok=True)
        print(f"Deleted temporary file: {fout}")
    else:
        print(f"Retained output file: {fout}")


def test_main_success(monkeypatch, capsys):
    """Test the main function for a successful run."""
    mock_args = types.SimpleNamespace(
        input_file="dummy_input.xml",
        output_file="dummy_output.html",
        timestamp="20240101_120000_UTC",
        run_id="123",
        ref_name="main",
        github_sha="abc",
        github_repo="owner/repo",
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)
    monkeypatch.setattr(
        "rattlesnake.cicd.report_pytest.run_pytest_report",
        lambda *args, **kwargs: CoverageMetric(lines_valid=100, lines_covered=85),
    )

    main()
    captured = capsys.readouterr()
    assert "✅ Pytest HTML report generated: dummy_output.html" in captured.out
    assert "📊 - valid lines of code: 100" in captured.out
    assert "🔍 - lines covered: 85" in captured.out
    assert "🎉 - coverage: 85.0" in captured.out


def test_main_file_not_found(monkeypatch, capsys):
    """Test the main function when the input file is not found."""
    mock_args = types.SimpleNamespace(
        input_file="non_existent.xml",
        output_file="dummy_output.html",
        timestamp="20240101_120000_UTC",
        run_id="123",
        ref_name="main",
        github_sha="abc",
        github_repo="owner/repo",
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)
    monkeypatch.setattr(
        "rattlesnake.cicd.report_pytest.run_pytest_report",
        lambda *args, **kwargs: exec('raise FileNotFoundError("File not found")'),
    )

    exit_code = main()
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "❌ Error: The input file 'non_existent.xml' was not found." in captured.out


def test_main_io_error(monkeypatch, capsys):
    """Test the main function when an IOError occurs."""
    mock_args = types.SimpleNamespace(
        input_file="dummy_input.xml",
        output_file="dummy_output.html",
        timestamp="20240101_120000_UTC",
        run_id="123",
        ref_name="main",
        github_sha="abc",
        github_repo="owner/repo",
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)
    monkeypatch.setattr(
        "rattlesnake.cicd.report_pytest.run_pytest_report",
        lambda *args, **kwargs: exec('raise IOError("Permission denied")'),
    )

    exit_code = main()
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "❌ I/O error occurred: Permission denied" in captured.out


def test_main_unexpected_error(monkeypatch, capsys):
    """Test the main function for an unexpected error."""
    mock_args = types.SimpleNamespace(
        input_file="dummy_input.xml",
        output_file="dummy_output.html",
        timestamp="20240101_120000_UTC",
        run_id="123",
        ref_name="main",
        github_sha="abc",
        github_repo="owner/repo",
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)
    monkeypatch.setattr(
        "rattlesnake.cicd.report_pytest.run_pytest_report",
        lambda *args, **kwargs: exec('raise Exception("Something went wrong")'),
    )

    exit_code = main()
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "❌ An unexpected error occurred: Something went wrong" in captured.out
