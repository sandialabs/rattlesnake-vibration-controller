"""
Tests for the Jupyter Book metadata reporter.
"""

import types

# import pytest  # unused import

from rattlesnake.cicd.report_jupyter_book import (
    generate_footer_md,
    update_myst_file,
    main,
)
from rattlesnake.cicd.utilities import ReportMetadata


def test_generate_footer_md():
    """Test the generation of the Markdown footer."""
    metadata = ReportMetadata(
        timestamp="20240324_120000_UTC",
        run_id="12345678",
        ref_name="main",
        github_sha="abc123456789",
        github_repo="owner/repo",
    )

    footer = generate_footer_md(metadata)

    assert "---" in footer
    assert '<div style="font-size: 0.7em;">' in footer
    assert "Generated:<br>" in footer
    assert "&nbsp;&nbsp;2024-03-24 12:00:00 UTC<br>" in footer
    assert "&nbsp;&nbsp;2024-03-24 08:00:00 EST<br>" in footer
    assert "&nbsp;&nbsp;2024-03-24 06:00:00 MST<br>" in footer
    assert 'Run ID: <a href="https://github.com/owner/repo/actions/runs/12345678">12345678</a><br>' in footer
    assert 'Branch: <a href="https://github.com/owner/repo/tree/main">main</a><br>' in footer
    assert 'Commit: <a href="https://github.com/owner/repo/commit/abc123456789">abc1234</a><br>' in footer
    assert "owner/repo" in footer
    assert "Repository:" not in footer


def test_update_myst_file_success(tmp_path):
    """Test updating a myst.yml file."""
    myst_content = """
site:
  parts:
    primary_sidebar_footer: |
      [Link](https://example.com)
"""
    myst_file = tmp_path / "myst.yml"
    myst_file.write_text(myst_content)

    footer_md = "      ---\n      <div style=\"font-size: 0.7em;\">Generated: 2024-03-24</div>"
    update_myst_file(str(myst_file), footer_md)

    updated_content = myst_file.read_text()
    assert "[Link](https://example.com)" in updated_content
    assert "---" in updated_content
    assert '<div style="font-size: 0.7em;">Generated: 2024-03-24</div>' in updated_content


def test_update_myst_file_no_footer(tmp_path, capsys):
    """Test update when primary_sidebar_footer is missing."""
    myst_content = "site: {}"
    myst_file = tmp_path / "myst.yml"
    myst_file.write_text(myst_content)

    update_myst_file(str(myst_file), "footer")
    
    captured = capsys.readouterr()
    assert "Warning: 'primary_sidebar_footer: |' not found" in captured.out


def test_main_success(monkeypatch, capsys, tmp_path):
    """Test the main function for a successful run."""
    myst_content = """
site:
  parts:
    primary_sidebar_footer: |
      [Link](https://example.com)
"""
    myst_file = tmp_path / "myst.yml"
    myst_file.write_text(myst_content)

    mock_args = types.SimpleNamespace(
        myst_file=str(myst_file),
        timestamp="20240324_120000_UTC",
        run_id="123",
        ref_name="main",
        github_sha="abc123456",
        github_repo="owner/repo",
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)

    exit_code = main()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[OK] Successfully updated Jupyter Book metadata" in captured.out
    
    updated_content = myst_file.read_text()
    assert "Generated:<br>" in updated_content
    assert "&nbsp;&nbsp;2024-03-24 12:00:00 UTC<br>" in updated_content


def test_main_error(monkeypatch, capsys):
    """Test the main function when an error occurs."""
    mock_args = types.SimpleNamespace(
        myst_file="non_existent.yml",
        timestamp="20240324_120000_UTC",
        run_id="123",
        ref_name="main",
        github_sha="abc123456",
        github_repo="owner/repo",
    )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)

    exit_code = main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "[X] File Error:" in captured.out
