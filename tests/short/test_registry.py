"""
Unit tests for ``rattlesnake.data.registry``, which generates ``_registry.py``.

Every test builds its own small directory in ``tmp_path``. Only
``test_committed_registry_matches_generated_format`` reads the real registry,
and it reads the dictionary that ``rattlesnake.data`` already imported.

The tests cover six areas.

Keyword-only signatures
    Every public function except ``main`` takes keyword arguments only, so a
    positional call raises ``TypeError``.
``file_list`` and ``build``
    Select files by suffix, search subdirectories, sort names, use ``/``
    separators, and hash each file.
``render`` and ``generate``
    Write a ``registry`` dictionary with one line per file. The text is valid
    Python, ``read`` loads it back, and ``ruff format`` leaves it unchanged.
    The formatting test is the guard against a formatting diff. It skips when
    ruff is not installed.
``read``
    Loads the dictionary with ``ast.literal_eval`` and never runs the file.
``difference_list``
    Reports a file that is ``missing`` from the registry, a ``stale`` entry
    whose file is gone, and a ``changed`` hash.
``main``
    Writes the file by default. With ``--check`` it leaves the file alone and
    returns 1 for a stale registry. It returns 2 for a missing directory.

Unit Tests
----------
test_functions_require_keyword_arguments
    Raises ``TypeError`` when a public function receives a positional argument.
test_file_list
    Keeps only the registered suffixes, from every subdirectory, sorted.
test_file_list_missing_directory
    Raises ``FileNotFoundError`` for a directory that does not exist.
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
"""

import hashlib
import importlib.util
import pathlib
import subprocess
import sys
from collections.abc import Callable

import pytest

from rattlesnake import data
from rattlesnake.data import registry


@pytest.fixture
def examples(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Build a small examples directory with data files and other files.

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
    directory = tmp_path / "examples"
    (directory / "b" / "deep").mkdir(parents=True)
    (directory / "a").mkdir()
    (directory / "b" / "deep" / "two.nc4").write_bytes(b"two")
    (directory / "a" / "one.npz").write_bytes(b"one")
    (directory / "zero.npy").write_bytes(b"zero")
    (directory / "a" / "sheet.xlsx").write_bytes(b"ignored")
    (directory / "a" / "code.py").write_bytes(b"ignored")
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


@pytest.mark.parametrize(
    "function",
    [
        registry.file_list,
        registry.build,
        registry.render,
        registry.generate,
        registry.read,
        registry.difference_list,
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
    populated: bool, examples: pathlib.Path, tmp_path: pathlib.Path
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
        registry.
    """
    empty = tmp_path / "empty"
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
