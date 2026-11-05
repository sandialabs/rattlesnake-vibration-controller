"""
Unit tests for ``rattlesnake.data.registry``, which generates ``_registry.py``.

Every test builds its own small directory in ``tmp_path``, except three. The
``examples`` fixture makes its directory a git repository and stages its files,
because ``file_list`` lists only files that git tracks. Tests that use it skip
when git is not installed.

Two of the three exceptions read the dictionary that ``rattlesnake.data``
already imported.
``test_committed_registry_matches_generated_format`` compares it with the
committed text. ``test_committed_registry_matches_files_on_disk`` compares it
with the data files in the checkout, and skips outside a git checkout, as in an
extracted sdist. ``test_suffixes_read_repository`` reads
``pyproject.toml`` in the checkout.

Tests that rely on the default suffixes also read that ``pyproject.toml``.

The tests cover seven areas.

Keyword-only signatures
    Every public function except ``main`` takes keyword arguments only, so a
    positional call raises ``TypeError``.
``file_list`` and ``build``
    Select tracked files by suffix, search subdirectories, sort names, use
    ``/`` separators, and hash each file. Skip untracked files and tracked
    files that are missing from disk. Reject a directory outside git.
``render`` and ``generate``
    Write a ``registry`` dictionary with one line per file. The text is valid
    Python, ``read`` loads it back, and ``ruff format`` leaves it unchanged.
    The formatting test is the guard against a formatting diff. It skips when
    ruff is not installed.
``suffixes_read``
    Reads the suffixes from the shared ``exclude`` globs in
    ``pyproject.toml``. It rejects a build target with its own ``exclude``
    list and a glob that is not ``EXCLUDE_PREFIX`` followed by a suffix.
``read``
    Loads the dictionary with ``ast.literal_eval`` and never runs the file.
``difference_list``
    Reports a file that is ``missing`` from the registry, a ``stale`` entry
    whose file is gone, and a ``changed`` hash.
``main``
    Writes the file by default. With ``--check`` it leaves the file alone and
    returns 1 for a stale registry. It returns 2 for a missing directory or a
    rejected ``pyproject.toml``.

Unit Tests
----------
test_functions_require_keyword_arguments
    Raises ``TypeError`` when a public function receives a positional argument.
test_file_list
    Keeps only the registered suffixes, from every subdirectory, sorted.
test_file_list_missing_directory
    Raises ``FileNotFoundError`` for a directory that does not exist.
test_file_list_skips_untracked
    Leaves out a file with a registered suffix that git does not track.
test_file_list_skips_deleted
    Leaves out a tracked file that is missing from disk.
test_file_list_outside_git
    Raises ``FileNotFoundError`` for a directory that is not in a git checkout.
test_build
    Maps each file name to the SHA-256 hash of its content.
test_render
    Writes the header and one correct ``name: hash`` line per entry.
test_render_empty
    Writes ``registry = {}`` for an empty registry.
test_generated_source_is_ruff_formatted
    Confirms that ``ruff format`` leaves the generated text unchanged.
test_generate_round_trip
    Reads the generated file back into the dictionary that ``build`` returns.
test_committed_registry_matches_generated_format
    Confirms the committed ``_registry.py`` has exactly the generated text.
test_committed_registry_matches_files_on_disk
    Confirms the committed registry lists every data file with its current hash.
test_suffixes_read
    Returns the sorted suffixes from the shared ``exclude`` list.
test_suffixes_read_repository
    Reads the checkout's ``pyproject.toml`` without an error.
test_suffixes_read_rejects_target_exclude
    Raises ``ValueError`` when the wheel or sdist sets its own ``exclude``.
test_suffixes_read_rejects_glob
    Raises ``ValueError`` for a glob that does not end in a plain suffix.
test_read_rejects_other_files
    Raises ``ValueError`` when no string dictionary named ``registry`` exists.
test_read_does_not_execute
    Loads the dictionary without running the surrounding code.
test_difference_list
    Reports missing, stale, and changed entries.
test_main_writes_registry
    Writes the generated text to ``--output`` and returns 0.
test_main_check
    Returns 0 for a current file, 1 for a stale file, and never writes.
test_main_missing_directory
    Returns 2 for a directory that does not exist.
test_main_rejected_pyproject
    Returns 2 when ``suffixes_read`` rejects ``pyproject.toml``.
"""

import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
from collections.abc import Callable

import pytest

from rattlesnake import data
from rattlesnake.data import registry

IN_GIT_CHECKOUT = (registry.PYPROJECT_PATH.parent / ".git").exists()


def git_run(*, directory: pathlib.Path, arguments: list[str]) -> None:
    """
    Run a git command in a directory.

    Parameters
    ----------
    directory : pathlib.Path
        Working directory for git.
    arguments : list of str
        Arguments after ``git``, such as ``["add", "--all"]``.
    """
    subprocess.run(["git", *arguments], cwd=directory, capture_output=True, check=True)


@pytest.fixture(name="examples")
def examples_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Build a small examples directory as a git repository with staged files.

    The fixture stages the files but never commits them. ``git ls-files``
    reads the index, so staging is enough, and a commit would need a git user
    name and email that a fresh CI runner lacks.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    pathlib.Path
        Directory that holds two registered files in subdirectories, one
        registered file at the top level, and two files that the registry
        must ignore.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    directory = tmp_path / "examples"
    (directory / "b" / "deep").mkdir(parents=True)
    (directory / "a").mkdir()
    (directory / "b" / "deep" / "two.nc4").write_bytes(b"two")
    (directory / "a" / "one.npz").write_bytes(b"one")
    (directory / "zero.npy").write_bytes(b"zero")
    (directory / "a" / "sheet.xlsx").write_bytes(b"ignored")
    (directory / "a" / "code.py").write_bytes(b"ignored")
    git_run(directory=directory, arguments=["init", "--quiet"])
    git_run(directory=directory, arguments=["add", "--all"])
    return directory


def sha256(*, content: bytes) -> str:
    """
    Return the SHA-256 hash of bytes.

    Parameters
    ----------
    content : bytes
        Bytes to hash.

    Returns
    -------
    str
        Lowercase hexadecimal hash.
    """
    return hashlib.sha256(content).hexdigest()


def pyproject_write(
    *, path: pathlib.Path, exclude: list[str], target: str | None = None
) -> pathlib.Path:
    """
    Write a ``pyproject.toml`` that holds only hatchling ``exclude`` lists.

    Parameters
    ----------
    path : pathlib.Path
        File to write.
    exclude : list of str
        Globs for the shared ``[tool.hatch.build]`` list.
    target : str, optional
        Build target, such as ``"wheel"``, that also gets its own ``exclude``
        list with one ``.xlsx`` glob. Defaults to no target list.

    Returns
    -------
    pathlib.Path
        ``path``, for chaining.
    """
    text = f"[tool.hatch.build]\nexclude = {json.dumps(exclude)}\n"
    if target is not None:
        text += (
            f"[tool.hatch.build.targets.{target}]\n"
            f'exclude = ["{registry.EXCLUDE_PREFIX}.xlsx"]\n'
        )
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "function",
    [
        registry.file_list,
        registry.build,
        registry.render,
        registry.generate,
        registry.read,
        registry.difference_list,
        registry.suffixes_read,
    ],
)
def test_functions_require_keyword_arguments(function: Callable[..., object]) -> None:
    """
    Raise ``TypeError`` when a public function receives a positional argument.

    Parameters
    ----------
    function : Callable
        Public function in ``rattlesnake.data.registry``. ``main`` is excluded
        because it takes the argument list that ``argparse`` expects.
    """
    with pytest.raises(TypeError):
        function("positional")


def test_file_list(examples: pathlib.Path) -> None:
    """
    Keep only the registered suffixes, from every subdirectory, sorted.
    """
    assert registry.file_list(directory=examples) == [
        "a/one.npz",
        "b/deep/two.nc4",
        "zero.npy",
    ]
    assert registry.file_list(directory=examples, suffixes=(".npy",)) == ["zero.npy"]


def test_file_list_missing_directory(tmp_path: pathlib.Path) -> None:
    """
    Raise ``FileNotFoundError`` for a directory that does not exist.
    """
    with pytest.raises(FileNotFoundError):
        registry.file_list(directory=tmp_path / "missing")


def test_file_list_skips_untracked(examples: pathlib.Path) -> None:
    """
    Leave out a file with a registered suffix that git does not track.

    A run output, such as a ``SAVE_CONTROL_DATA`` file, is untracked. Its hash
    changes on every run, so it must never be registered.
    """
    (examples / "a" / "run_output.nc4").write_bytes(b"run output")
    assert registry.file_list(directory=examples) == [
        "a/one.npz",
        "b/deep/two.nc4",
        "zero.npy",
    ]


def test_file_list_skips_deleted(examples: pathlib.Path) -> None:
    """
    Leave out a tracked file that is missing from disk.

    Git still lists a file deleted with ``rm`` instead of ``git rm``. There is
    nothing to hash, so ``--check`` reports the entry as ``stale``.
    """
    (examples / "zero.npy").unlink()
    assert registry.file_list(directory=examples) == ["a/one.npz", "b/deep/two.nc4"]


def test_file_list_outside_git(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Raise ``FileNotFoundError`` for a directory that is not in a git checkout.

    ``GIT_CEILING_DIRECTORIES`` stops git from finding a repository above
    ``tmp_path``, in case the temporary directory sits inside one.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    (tmp_path / "one.nc4").write_bytes(b"one")
    with pytest.raises(FileNotFoundError, match="not a git checkout"):
        registry.file_list(directory=tmp_path, suffixes=(".nc4",))


def test_build(examples: pathlib.Path) -> None:
    """
    Map each file name to the SHA-256 hash of its content.
    """
    assert registry.build(directory=examples) == {
        "a/one.npz": sha256(content=b"one"),
        "b/deep/two.nc4": sha256(content=b"two"),
        "zero.npy": sha256(content=b"zero"),
    }


def test_render() -> None:
    """
    Write the header and one correct ``name: hash`` line per entry.
    """
    entries = {"a/one.npz": "1" * 64, "zero.npy": "2" * 64}
    expected = (
        registry.HEADER
        + "registry = {\n"
        + f'    "a/one.npz": "{"1" * 64}",\n'
        + f'    "zero.npy": "{"2" * 64}",\n'
        + "}\n"
    )
    assert registry.render(entries=entries) == expected


def test_render_empty() -> None:
    """
    Write ``registry = {}`` for an empty registry.
    """
    assert registry.render(entries={}) == registry.HEADER + "registry = {}\n"


@pytest.mark.skipif(
    importlib.util.find_spec("ruff") is None, reason="ruff is not installed"
)
@pytest.mark.parametrize("populated", [True, False])
def test_generated_source_is_ruff_formatted(
    populated: bool, examples: pathlib.Path
) -> None:
    """
    Confirm that ``ruff format`` leaves the generated text unchanged.

    The test formats the text as if it were ``_registry.py``, so ruff applies
    the project's settings from ``pyproject.toml``. Each entry is longer than
    the 88-character limit, which checks that ruff does not rewrap it.

    Parameters
    ----------
    populated : bool
        ``True`` to generate from files, ``False`` to generate an empty
        registry from a directory in the same repository with no tracked files.
    """
    empty = examples / "empty"
    empty.mkdir()
    text = registry.generate(directory=examples if populated else empty)
    assert ("registry = {}" in text) is not populated
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--stdin-filename",
            str(registry.REGISTRY_PATH),
            "-",
        ],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == text


def test_generate_round_trip(examples: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """
    Read the generated file back into the dictionary that ``build`` returns.
    """
    path = tmp_path / "_registry.py"
    path.write_text(registry.generate(directory=examples), encoding="utf-8")
    assert registry.read(path=path) == registry.build(directory=examples)


def test_committed_registry_matches_generated_format() -> None:
    """
    Confirm the committed ``_registry.py`` has exactly the generated text.

    This test catches a hand edit, because ``render`` would write different
    text for the same entries.
    """
    text = registry.REGISTRY_PATH.read_text(encoding="utf-8")
    assert text == registry.render(entries=data.REGISTRY)


def test_suffixes_read(tmp_path: pathlib.Path) -> None:
    """
    Return the sorted suffixes from the shared ``exclude`` list.

    The list holds ``.npz`` before ``.nc4``, so the test also checks the sort.
    """
    globs = [registry.EXCLUDE_PREFIX + suffix for suffix in (".npz", ".nc4")]
    path = pyproject_write(path=tmp_path / "pyproject.toml", exclude=globs)
    assert registry.suffixes_read(path=path) == (".nc4", ".npz")


@pytest.mark.skipif(
    not registry.PYPROJECT_PATH.is_file(), reason="not running from a source checkout"
)
def test_suffixes_read_repository() -> None:
    """
    Read the checkout's ``pyproject.toml`` without an error.

    ``suffixes_read`` raises when a build target sets its own ``exclude`` list,
    so this test fails when someone adds one. Such a list would put every data
    file back into that archive.
    """
    suffixes = registry.suffixes_read()
    assert suffixes
    assert all(suffix.startswith(".") for suffix in suffixes)


@pytest.mark.parametrize("target", ["wheel", "sdist"])
def test_suffixes_read_rejects_target_exclude(
    target: str, tmp_path: pathlib.Path
) -> None:
    """
    Raise ``ValueError`` when the wheel or sdist sets its own ``exclude``.

    Hatchling uses a target's own ``exclude`` list in place of the shared one.
    A wheel built that way held all 39 data files.

    Parameters
    ----------
    target : str
        Build target that gets its own ``exclude`` list.
    """
    path = pyproject_write(
        path=tmp_path / "pyproject.toml",
        exclude=[registry.EXCLUDE_PREFIX + ".nc4"],
        target=target,
    )
    with pytest.raises(ValueError, match=f"targets.{target}"):
        registry.suffixes_read(path=path)


@pytest.mark.parametrize(
    "glob",
    [
        "/src/rattlesnake/examples/*.nc4",
        "/src/rattlesnake/examples/**/*nc4",
        "/src/rattlesnake/examples/**/*.tar.gz",
        "/src/rattlesnake/examples/**/*.np?",
        "/src/rattlesnake/examples/**/*.",
    ],
)
def test_suffixes_read_rejects_glob(glob: str, tmp_path: pathlib.Path) -> None:
    """
    Raise ``ValueError`` for a glob that does not end in a plain suffix.

    The cases cover a wrong prefix, a missing dot, a two-part suffix, a
    wildcard, and an empty suffix. ``pathlib`` reports only the last suffix of
    a name, so ``file_list`` could never match ``.tar.gz``.

    Parameters
    ----------
    glob : str
        Glob to put in the shared ``exclude`` list.
    """
    path = pyproject_write(path=tmp_path / "pyproject.toml", exclude=[glob])
    with pytest.raises(ValueError, match="is not"):
        registry.suffixes_read(path=path)


@pytest.mark.skipif(not IN_GIT_CHECKOUT, reason="not running from a git checkout")
def test_committed_registry_matches_files_on_disk() -> None:
    """
    Confirm the committed registry lists every data file with its current hash.

    This test catches a data file that someone added, removed, or changed
    without regenerating ``_registry.py``. Otherwise CI stays green, the release
    ships the old hash, and every user's ``fetch`` fails with ``ValueError``
    after four attempts. A git checkout holds every data file, so the test needs
    no network access. An extracted sdist holds no data files and no ``.git``,
    so the test skips there.
    """
    differences = registry.difference_list(
        current=data.REGISTRY, generated=registry.build()
    )
    assert differences == [], (
        "_registry.py is stale. Run: uv run python -m rattlesnake.data.registry\n"
        + "\n".join(differences)
    )


@pytest.mark.parametrize(
    "source",
    [
        "other = {}\n",
        "registry = 5\n",
        "registry = {'a': 1}\n",
        "registry = {1: 'a'}\n",
        "registry: dict = {}\n",
    ],
)
def test_read_rejects_other_files(source: str, tmp_path: pathlib.Path) -> None:
    """
    Raise ``ValueError`` when no string dictionary named ``registry`` exists.

    Parameters
    ----------
    source : str
        Python source that does not assign a dictionary of strings to
        ``registry``.
    """
    path = tmp_path / "_registry.py"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="registry dictionary"):
        registry.read(path=path)


def test_read_does_not_execute(tmp_path: pathlib.Path) -> None:
    """
    Load the dictionary without running the surrounding code.

    The file would create a marker file if Python ran it.
    """
    marker = tmp_path / "ran"
    path = tmp_path / "_registry.py"
    path.write_text(
        f"open({str(marker)!r}, 'w').close()\nregistry = {{'a.npz': 'x'}}\n",
        encoding="utf-8",
    )
    assert registry.read(path=path) == {"a.npz": "x"}
    assert not marker.exists()


def test_difference_list() -> None:
    """
    Report missing, stale, and changed entries, and nothing for a match.
    """
    current = {"same.npz": "1" * 64, "old.npz": "2" * 64, "edit.npy": "3" * 64}
    generated = {"same.npz": "1" * 64, "new.nc4": "4" * 64, "edit.npy": "5" * 64}
    assert registry.difference_list(current=current, generated=generated) == [
        "changed  edit.npy",
        "missing  new.nc4",
        "stale    old.npz",
    ]
    assert registry.difference_list(current=current, generated=current) == []


def test_main_writes_registry(
    examples: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Write the generated text to ``--output`` and return 0.
    """
    output = tmp_path / "_registry.py"
    status = registry.main(["--directory", str(examples), "--output", str(output)])
    assert status == 0
    assert output.read_text(encoding="utf-8") == registry.generate(directory=examples)
    assert "Wrote 3 entries" in capsys.readouterr().out


def test_main_check(
    examples: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Return 0 for a current file, 1 for a stale file, and never write.
    """
    output = tmp_path / "_registry.py"
    arguments = ["--directory", str(examples), "--output", str(output)]
    assert registry.main(arguments + ["--check"]) == 1
    assert not output.exists()
    assert "missing  a/one.npz" in capsys.readouterr().out

    registry.main(arguments)
    assert registry.main(arguments + ["--check"]) == 0
    assert "up to date" in capsys.readouterr().out

    (examples / "a" / "one.npz").write_bytes(b"edited")
    before = output.read_text(encoding="utf-8")
    assert registry.main(arguments + ["--check"]) == 1
    assert "changed  a/one.npz" in capsys.readouterr().out
    assert output.read_text(encoding="utf-8") == before


def test_main_missing_directory(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    Return 2 for a directory that does not exist.
    """
    status = registry.main(
        ["--directory", str(tmp_path / "missing"), "--output", str(tmp_path / "r.py")]
    )
    assert status == 2
    assert "not a directory" in capsys.readouterr().err


def test_main_rejected_pyproject(
    examples: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Return 2 when ``suffixes_read`` rejects ``pyproject.toml``, and write nothing.
    """
    path = pyproject_write(
        path=tmp_path / "pyproject.toml",
        exclude=[registry.EXCLUDE_PREFIX + ".nc4"],
        target="wheel",
    )
    monkeypatch.setattr(registry, "PYPROJECT_PATH", path)
    output = tmp_path / "_registry.py"
    status = registry.main(["--directory", str(examples), "--output", str(output)])
    assert status == 2
    assert "own exclude" in capsys.readouterr().err
    assert not output.exists()
