"""
Tests for the preflight documentation checks.
"""

import subprocess

from rattlesnake import preflight


class FakeBuild:
    """Stand-in for the Popen handle of a Jupyter Book build."""

    def __init__(self, lines: list[str], returncode: int = 0):
        self.stdout = iter(lines)
        self.returncode = returncode

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def build_patch(monkeypatch, lines: list[str], returncode: int = 0) -> None:
    """
    Replace the Jupyter Book build with a canned run.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        lines: Lines the build writes to stdout.
        returncode: Exit code the build reports.
    """
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: FakeBuild(lines, returncode),
    )


def toc_patch(monkeypatch, exit_code: int) -> None:
    """
    Replace the table of contents check with a canned result.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        exit_code: Exit code the check reports.
    """
    monkeypatch.setattr(preflight.toc, "report", lambda **kwargs: exit_code)


def test_docs_checks_passes_on_a_clean_build(monkeypatch):
    """A sound table of contents and an error-free build pass."""
    toc_patch(monkeypatch, 0)
    build_patch(monkeypatch, ["📚 Built 28 pages for project in 4.57 s.\n"])

    assert preflight.docs_checks(no_sync=True) is True


def test_docs_checks_fails_on_errors_the_strict_flag_ignores(monkeypatch, capsys):
    """
    MyST errors recorded against myst.yml fail the check.

    These are the errors --strict prints and then ignores, so the build itself
    reports success. Without the log scan they reach the deployed book.
    """
    toc_patch(monkeypatch, 0)
    build_patch(
        monkeypatch,
        [
            "⛔️ myst.yml Table of contents entry does not exist: "
            "book/src/_generated/sine_run_doc.md\n",
            "📚 Built 28 pages for project in 4.57 s.\n",
        ],
    )

    assert preflight.docs_checks(no_sync=True) is False
    assert "MyST reported 1 error(s)" in capsys.readouterr().out


def test_docs_checks_fails_on_a_broken_table_of_contents(monkeypatch):
    """A table of contents problem fails the check even when the build is clean."""
    toc_patch(monkeypatch, 1)
    build_patch(monkeypatch, ["📚 Built 28 pages for project in 4.57 s.\n"])

    assert preflight.docs_checks(no_sync=True) is False


def test_docs_checks_fails_on_a_failed_build(monkeypatch):
    """A nonzero build exit code fails the check."""
    toc_patch(monkeypatch, 0)
    build_patch(monkeypatch, ["Site has 5 errors, stopping build.\n"], returncode=1)

    assert preflight.docs_checks(no_sync=True) is False
