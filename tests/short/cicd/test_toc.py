"""
Tests for the MyST table of contents validator.
"""

from pathlib import Path

import pytest
import yaml

from rattlesnake.cicd import toc


REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_MYST_FILE = REPO_ROOT / "documentation" / "myst.yml"


def myst_file_write(tmp_path: Path, toc_items: list) -> str:
    """
    Write a minimal myst.yml containing the given table of contents.

    Args:
        tmp_path: Directory to write into.
        toc_items: The table of contents list.

    Returns:
        Path to the written myst.yml.
    """
    myst_file = tmp_path / "myst.yml"
    config = {"version": 1, "project": {"title": "Test", "toc": toc_items}}
    myst_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(myst_file)


def page_write(tmp_path: Path, relative_path: str) -> Path:
    """
    Create a markdown page inside a temporary book.

    Args:
        tmp_path: Directory to write into.
        relative_path: Page path relative to tmp_path.

    Returns:
        The created file path.
    """
    page = tmp_path / relative_path
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("# Page\n", encoding="utf-8")
    return page


def test_entries_flattens_nested_children():
    """Nested children are collected in document order."""
    items = [
        {"file": "a.md"},
        {
            "file": "b.md",
            "children": [
                {"file": "c.md"},
                {"title": "Group", "children": [{"file": "d.md"}]},
            ],
        },
    ]

    assert [entry.value for entry in toc.entries(items=items)] == [
        "a.md",
        "b.md",
        "c.md",
        "d.md",
    ]


def test_entries_skips_urls_and_titles():
    """URL and title-only entries contribute nothing, but children are walked."""
    items = [
        {"url": "https://example.com", "title": "External"},
        {"title": "Developer's Guide", "children": [{"file": "deep.md"}]},
    ]

    entries = toc.entries(items=items)

    assert len(entries) == 1
    assert entries[0] == toc.Entry(value="deep.md", is_pattern=False)


def test_entries_records_patterns():
    """Pattern entries are collected and flagged as patterns."""
    entries = toc.entries(items=[{"pattern": "chapters/*.md"}])

    assert entries == [toc.Entry(value="chapters/*.md", is_pattern=True)]


def test_entries_handles_missing_items():
    """A None or empty table of contents yields no entries."""
    assert toc.entries(items=None) == []
    assert toc.entries(items=[]) == []


def test_validate_passes_when_every_entry_exists(tmp_path):
    """A table of contents pointing at real files reports no problems."""
    page_write(tmp_path, "book/src/index.md")
    page_write(tmp_path, "book/src/chapter.md")
    myst_file = myst_file_write(
        tmp_path,
        [
            {"file": "book/src/index.md"},
            {"file": "book/src/chapter.md"},
        ],
    )

    assert toc.validate(myst_file=myst_file) == []


def test_validate_reports_missing_entry(tmp_path):
    """The missing-content case that --strict fails to catch is reported."""
    page_write(tmp_path, "book/src/index.md")
    page_write(tmp_path, "book/src/ui_documentation.md")
    myst_file = myst_file_write(
        tmp_path,
        [
            {"file": "book/src/index.md"},
            {
                "file": "book/src/ui_documentation.md",
                "children": [{"file": "book/src/_generated/sine_run_doc.md"}],
            },
        ],
    )

    problems = toc.validate(myst_file=myst_file)

    assert problems == [
        "Table of contents entry does not exist: book/src/_generated/sine_run_doc.md"
    ]


def test_validate_infers_extension(tmp_path):
    """An extensionless entry resolves when exactly one candidate exists."""
    page_write(tmp_path, "book/src/index.md")
    myst_file = myst_file_write(tmp_path, [{"file": "book/src/index"}])

    assert toc.validate(myst_file=myst_file) == []


def test_validate_reports_unresolvable_extensionless_entry(tmp_path):
    """An extensionless entry with no candidate file is reported."""
    myst_file = myst_file_write(tmp_path, [{"file": "book/src/index"}])

    assert toc.validate(myst_file=myst_file) == [
        "Unable to resolve table of contents entry: book/src/index"
    ]


def test_validate_reports_folder_used_as_file(tmp_path):
    """A directory named where a file belongs is reported."""
    (tmp_path / "book" / "src").mkdir(parents=True)
    myst_file = myst_file_write(tmp_path, [{"file": "book/src"}])

    assert toc.validate(myst_file=myst_file) == [
        "Folder referenced as file in table of contents: book/src"
    ]


def test_validate_reports_empty_pattern(tmp_path):
    """A pattern matching nothing is reported."""
    page_write(tmp_path, "book/src/index.md")
    myst_file = myst_file_write(
        tmp_path,
        [{"file": "book/src/index.md"}, {"pattern": "book/src/_generated/*.md"}],
    )

    assert toc.validate(myst_file=myst_file) == [
        "Pattern from table of contents did not match any files: "
        "book/src/_generated/*.md"
    ]


def test_validate_accepts_matching_pattern(tmp_path):
    """A pattern matching at least one file reports no problem."""
    page_write(tmp_path, "book/src/index.md")
    page_write(tmp_path, "book/src/_generated/sine_run_doc.md")
    myst_file = myst_file_write(
        tmp_path,
        [{"file": "book/src/index.md"}, {"pattern": "book/src/_generated/*.md"}],
    )

    assert toc.validate(myst_file=myst_file) == []


def test_load_rejects_file_without_toc(tmp_path):
    """A configuration file with no table of contents raises."""
    myst_file = tmp_path / "myst.yml"
    myst_file.write_text(
        yaml.safe_dump({"version": 1, "project": {}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="No table of contents"):
        toc.load(myst_file=str(myst_file))


def test_load_rejects_non_mapping(tmp_path):
    """A configuration file that is not a mapping raises."""
    myst_file = tmp_path / "myst.yml"
    myst_file.write_text("- just\n- a list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain a YAML mapping"):
        toc.load(myst_file=str(myst_file))


def test_report_returns_zero_when_sound(tmp_path, capsys):
    """A sound table of contents exits 0 and says so."""
    page_write(tmp_path, "book/src/index.md")
    myst_file = myst_file_write(tmp_path, [{"file": "book/src/index.md"}])

    assert toc.report(myst_file=myst_file) == 0
    assert "[OK]" in capsys.readouterr().out


def test_report_returns_one_and_lists_problems(tmp_path, capsys, monkeypatch):
    """A broken table of contents exits 1 and annotates the problems."""
    page_write(tmp_path, "book/src/index.md")
    myst_file = myst_file_write(
        tmp_path,
        [{"file": "book/src/index.md"}, {"file": "book/src/missing.md"}],
    )
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    assert toc.report(myst_file=myst_file) == 1

    output = capsys.readouterr().out
    assert "1 table of contents problem(s)" in output
    assert "- Table of contents entry does not exist: book/src/missing.md" in output
    assert f"::error file={myst_file}::" in output


def test_report_returns_one_for_unreadable_file(tmp_path, capsys):
    """A missing configuration file exits 1 rather than raising."""
    assert toc.report(myst_file=str(tmp_path / "absent.yml")) == 1
    assert "[X] Table of contents check failed" in capsys.readouterr().out


def test_main_uses_command_line_argument(tmp_path, monkeypatch):
    """The CLI entry point validates the file named on the command line."""
    page_write(tmp_path, "book/src/index.md")
    myst_file = myst_file_write(tmp_path, [{"file": "book/src/index.md"}])
    monkeypatch.setattr("sys.argv", ["toc.py", "--myst_file", myst_file])

    assert toc.main() == 0


@pytest.mark.skipif(
    not REPO_MYST_FILE.is_file(), reason="documentation/ is not part of the sdist"
)
def test_repo_myst_file_parses():
    """The repository's own myst.yml exposes a table of contents we can read."""
    base_dir, items = toc.load(myst_file=str(REPO_MYST_FILE))
    values = [entry.value for entry in toc.entries(items=items)]

    assert base_dir == REPO_MYST_FILE.parent
    assert "book/src/introduction.md" in values
    assert "book/src/_generated/sine_run_doc.md" in values
