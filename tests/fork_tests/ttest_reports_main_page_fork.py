"""
Unit tests for generate_reports_landing_page.py — GitHub Pages Landing Page Generator.

This test suite verifies the correctness of the functions used to generate
the main landing page for code quality reports.
"""

import types

import pytest
from rattlesnake.cicd.reports_main_page import (
    get_report_html,
    main,
    run_reports_main_page,
    write_report,
)


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


def test_get_report_html():
    """Test get_report_html with valid inputs."""
    report = get_report_html(
        github_repo="test/repo",
        pylint_score="9.5",
    )
    assert "<!DOCTYPE html>" in report
    assert "Rattlesnake Code Quality Reports" in report
    assert "https://github.com/test/repo" in report
    assert "<strong>Latest Score:</strong> 9.5/10" in report


def test_run_generate_landing_page(tmp_path):
    """Tests the main report creation function."""
    output_file = tmp_path / "index.html"
    run_reports_main_page(
        github_repo="owner/repo",
        pylint_score="8.8",
        output_file=str(output_file),
    )
    assert output_file.is_file()
    content = output_file.read_text()
    assert "<h1>Rattlesnake Code Quality</h1>" in content
    assert "<strong>Latest Score:</strong> 8.8/10" in content
    assert 'href="https://github.com/owner/repo"' in content


def test_main_success(monkeypatch, capsys):
    """Test the main function for a successful run."""
    mock_args = types.SimpleNamespace(
        github_repo="owner/repo",
        pylint_score="9.9",
        output_file="landing.html",
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)

    def mock_run(*args, **kwargs):
        pass

    monkeypatch.setattr(
        "rattlesnake.cicd.reports_main_page.run_reports_main_page",
        mock_run,
    )

    exit_code = main()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "✅ GitHub Pages reports main page generated: landing.html" in captured.out


def test_main_io_error(monkeypatch, capsys):
    """Test the main function when an IOError occurs."""
    mock_args = types.SimpleNamespace(
        github_repo="owner/repo",
        pylint_score="9.9",
        output_file="landing.html",
    )
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)

    def mock_run_raises_io_error(*args, **kwargs):
        raise IOError("Disk full")

    monkeypatch.setattr(
        "rattlesnake.cicd.reports_main_page.run_reports_main_page",
        mock_run_raises_io_error,
    )

    exit_code = main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "❌ I/O error occurred: Disk full" in captured.err


def test_main_unexpected_error(monkeypatch, capsys):
    """Test the main function for an unexpected error."""
    mock_args = types.SimpleNamespace(
        github_repo="owner/repo",
        pylint_score="9.9",
        output_file="landing.html",
    )
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)

    def mock_run_raises_exception(*args, **kwargs):
        raise Exception("Something bad happened")

    monkeypatch.setattr(
        "rattlesnake.cicd.reports_main_page.run_reports_main_page",
        mock_run_raises_exception,
    )

    exit_code = main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "❌ An unexpected error occurred: Something bad happened" in captured.err
