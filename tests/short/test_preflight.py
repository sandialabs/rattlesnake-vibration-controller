"""
Tests for the preflight documentation checks.
"""

import io
import subprocess
import sys

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


def narrow_stdout_patch(monkeypatch, *, encoding: str) -> io.BytesIO:
    """
    Replace sys.stdout with a stream restricted to the given encoding.

    Reproduces the default Windows console codec ("charmap"/cp1252), which
    cannot encode MyST's emoji. pytest's own file-descriptor capture goes
    through that same codec, so this is not a synthetic scenario: it is what
    actually failed on the Windows CI runner.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        encoding: The encoding to restrict sys.stdout to.

    Returns:
        The underlying buffer, for inspecting what was actually written.
    """
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding=encoding, newline="")
    monkeypatch.setattr(sys, "stdout", stream)
    return buffer


def test_console_print_falls_back_when_the_stream_cannot_encode_the_text(monkeypatch):
    """A narrow stdout encoding does not raise; unencodable text degrades instead."""
    buffer = narrow_stdout_patch(monkeypatch, encoding="cp1252")

    preflight._console_print("📚 Built 28 pages for project in 4.57 s.\n", end="")
    sys.stdout.flush()

    written = buffer.getvalue().decode("cp1252")
    assert "Built 28 pages" in written
    assert "📚" not in written


def test_docs_checks_survives_a_narrow_stdout_encoding(monkeypatch):
    """
    The exact failure seen on Windows CI: stdout cannot encode MyST's emoji.

    Before the fix, printing the build's emoji-laden output crashed with
    UnicodeEncodeError under cp1252 instead of reporting a check result.
    """
    toc_patch(monkeypatch, 0)
    build_patch(monkeypatch, ["📚 Built 28 pages for project in 4.57 s.\n"])
    narrow_stdout_patch(monkeypatch, encoding="cp1252")

    assert preflight.docs_checks(no_sync=True) is True
