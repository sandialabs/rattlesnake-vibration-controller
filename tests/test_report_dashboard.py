"""
Unit tests for report_dashboard.py — CI/CD Dashboard Generator.
"""

import types

# import pytest  # unused import
from rattlesnake.cicd.report_dashboard import generate_dashboard_html, main


def test_generate_dashboard_html():
    """Test that generate_dashboard_html returns expected content."""
    github_repo = "owner/repo"
    html = generate_dashboard_html(github_repo)

    assert "<!DOCTYPE html>" in html
    assert "Project Dashboard" in html
    assert "Rattlesnake" in html
    assert f"https://github.com/{github_repo}" in html
    assert f"https://github.com/{github_repo}/tree/main" in html
    assert f"https://github.com/{github_repo}/tree/dev" in html
    assert "main/badges/lint.svg" in html
    assert "dev/badges/coverage.svg" in html


def test_main_success(tmp_path, monkeypatch, capsys):
    """Test the main function for a successful run."""
    output_file = tmp_path / "index.html"
    github_repo = "test/repo"

    mock_args = types.SimpleNamespace(github_repo=github_repo, output_file=str(output_file))

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)

    main()

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert github_repo in content
    captured = capsys.readouterr()
    assert "[OK] Project Dashboard generated" in captured.out
