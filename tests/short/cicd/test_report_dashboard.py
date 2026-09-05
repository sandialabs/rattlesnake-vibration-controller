"""
Unit tests for report_dashboard.py — CI/CD Dashboard Generator.
"""

import types

# import pytest  # unused import
from rattlesnake.cicd.report_dashboard import generate_dashboard_html, main
from rattlesnake.cicd.utilities import ReportMetadata


def test_generate_dashboard_html():
    """Test that generate_dashboard_html returns expected content."""
    metadata = ReportMetadata(
        timestamp="20260424_215553_UTC",
        run_id="12345",
        ref_name="main",
        github_sha="abc1234",
        github_repo="owner/repo",
    )
    html = generate_dashboard_html(metadata)

    assert "<!DOCTYPE html>" in html
    assert "Project Dashboard" in html
    assert "Rattlesnake" in html
    assert "https://github.com/owner/repo" in html
    assert "main/badges/lint.svg" in html
    assert "dev/badges/coverage.svg" in html
    assert "dev/badges/tests.svg" in html
    assert "2026-04-24" in html
    assert "(expires 2026-07-23)" in html


def test_main_success(tmp_path, monkeypatch, capsys):
    """Test the main function for a successful run."""
    output_file = tmp_path / "index.html"
    github_repo = "test/repo"

    mock_args = types.SimpleNamespace(
        output_file=str(output_file),
        timestamp="20260424_215553_UTC",
        run_id="12345",
        ref_name="main",
        github_sha="abc1234",
        github_repo=github_repo,
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)

    main()

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert github_repo in content
    captured = capsys.readouterr()
    assert "[OK] Project Dashboard generated" in captured.out
