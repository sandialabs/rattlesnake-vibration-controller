"""
Unit tests for ``hatch_build.py``, the build hook that records the commit SHA.

The dev environment does not install ``hatchling``. The tests load
``hatch_build.py`` with a stub for its one ``hatchling`` import, so they run
everywhere. The tests that need git create a throwaway repository in
``tmp_path``.

Unit Tests
----------
test_commit_read_requires_keyword_arguments
    Raises ``TypeError`` for a positional argument.
test_commit_read_without_git
    Returns ``None`` for a directory with no ``.git`` entry.
test_commit_read_in_repository
    Returns the full SHA of ``HEAD``.
test_initialize_writes_commit
    Writes the SHA to ``COMMIT_FILE`` and lists it as a build artifact.
test_initialize_keeps_existing_file_without_git
    Keeps the file an sdist already holds when no repository exists.
test_initialize_writes_none_without_git
    Writes ``commit = None`` when neither a repository nor a file exists.
"""

import importlib.util
import pathlib
import shutil
import subprocess
import sys
import types

import pytest

HATCH_BUILD_PATH = pathlib.Path(__file__).resolve().parents[2] / "hatch_build.py"

if not HATCH_BUILD_PATH.is_file():
    pytest.skip(
        "hatch_build.py exists only in a source checkout", allow_module_level=True
    )


def hatch_build_load() -> types.ModuleType:
    """
    Load ``hatch_build.py`` with a stub for ``hatchling``.

    Returns
    -------
    types.ModuleType
        The loaded ``hatch_build`` module.
    """
    interface = types.ModuleType("hatchling.builders.hooks.plugin.interface")
    interface.BuildHookInterface = object  # type: ignore[attr-defined]
    stubs = {
        "hatchling": types.ModuleType("hatchling"),
        "hatchling.builders": types.ModuleType("hatchling.builders"),
        "hatchling.builders.hooks": types.ModuleType("hatchling.builders.hooks"),
        "hatchling.builders.hooks.plugin": types.ModuleType(
            "hatchling.builders.hooks.plugin"
        ),
        "hatchling.builders.hooks.plugin.interface": interface,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("hatch_build", HATCH_BUILD_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


hatch_build = hatch_build_load()


@pytest.fixture
def repository(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    """
    Create a git repository with one commit.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    tuple of pathlib.Path and str
        Repository root and the SHA of its only commit.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    root = tmp_path / "repository"
    root.mkdir()

    def git(*arguments: str) -> str:
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.com",
                *arguments,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init", "--quiet")
    git("commit", "--quiet", "--allow-empty", "--message", "initial")
    return root, git("rev-parse", "HEAD")


def hook_initialize(*, root: pathlib.Path) -> dict[str, list[str]]:
    """
    Run ``CommitBuildHook.initialize`` against a root directory.

    Parameters
    ----------
    root : pathlib.Path
        Directory that stands in for the repository root.

    Returns
    -------
    dict of str to list of str
        The build data after the hook ran.
    """
    (root / "src" / "rattlesnake").mkdir(parents=True, exist_ok=True)
    build_data: dict[str, list[str]] = {"artifacts": []}
    hook = types.SimpleNamespace(root=str(root))
    hatch_build.CommitBuildHook.initialize(hook, "standard", build_data)
    return build_data


def test_commit_read_requires_keyword_arguments(tmp_path: pathlib.Path) -> None:
    """
    Raise ``TypeError`` when ``commit_read`` receives a positional argument.
    """
    with pytest.raises(TypeError):
        hatch_build.commit_read(tmp_path)  # pylint: disable=too-many-function-args


def test_commit_read_without_git(tmp_path: pathlib.Path) -> None:
    """
    Return ``None`` for a directory with no ``.git`` entry, as in an sdist.
    """
    assert hatch_build.commit_read(root=tmp_path) is None


def test_commit_read_in_repository(repository: tuple[pathlib.Path, str]) -> None:
    """
    Return the full 40-character SHA of ``HEAD``.
    """
    root, sha = repository
    assert hatch_build.commit_read(root=root) == sha
    assert len(sha) == 40


def test_initialize_writes_commit(repository: tuple[pathlib.Path, str]) -> None:
    """
    Write the SHA to ``COMMIT_FILE`` and list the file as a build artifact.
    """
    root, sha = repository
    build_data = hook_initialize(root=root)
    text = (root / hatch_build.COMMIT_FILE).read_text(encoding="utf-8")
    assert f"commit = {sha!r}" in text
    assert build_data["artifacts"] == [f"/{hatch_build.COMMIT_FILE}"]


def test_initialize_keeps_existing_file_without_git(tmp_path: pathlib.Path) -> None:
    """
    Keep the ``COMMIT_FILE`` that an extracted sdist already holds.
    """
    path = tmp_path / hatch_build.COMMIT_FILE
    path.parent.mkdir(parents=True)
    path.write_text("commit = 'from-sdist'\n", encoding="utf-8")
    hook_initialize(root=tmp_path)
    assert path.read_text(encoding="utf-8") == "commit = 'from-sdist'\n"


def test_initialize_writes_none_without_git(tmp_path: pathlib.Path) -> None:
    """
    Write ``commit = None`` when neither a repository nor a file exists.
    """
    hook_initialize(root=tmp_path)
    text = (tmp_path / hatch_build.COMMIT_FILE).read_text(encoding="utf-8")
    assert "commit = None" in text
