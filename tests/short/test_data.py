"""
Unit tests for ``rattlesnake.data``, which locates large data files.

None of these tests use the internet. Tests that exercise a download serve a
temporary directory from a local HTTP server on a free port.

The tests cover seven areas.

Keyword-only signatures
    Every public function takes keyword arguments only, so a positional call
    raises ``TypeError``.
``git_ref`` and ``commit_installed``
    Map a package build to the git ref that holds its data files. A release
    version maps to ``v<version>``. A development version maps to the commit
    SHA that ``hatch_build.py`` recorded, or to ``dev`` without one.
``hash_check``
    Accepts a matching SHA-256 hash and rejects a wrong hash or a missing file.
``PACKAGE_DIRECTORY``
    Equals the directory of the ``rattlesnake.examples`` package. The data
    moved from ``data.py`` to ``data/__init__.py``, which changed the parent
    count that locates this directory. A wrong count makes the registry tests
    skip instead of fail, so a separate test compares against the import
    system.
``REGISTRY``
    Holds at least one entry. Every name is a relative path and every hash is
    64 hexadecimal characters. In a source checkout, each registered file must
    exist and match its hash. The checkout tests skip when ``pyproject.toml``
    is absent, as in an installed wheel.
``fetch``
    Prefers a packaged file whose hash matches, downloads a missing file into
    the cache, rejects a download with a wrong hash, and raises ``KeyError``
    for a name that is not in the registry.
``cache_fill`` and ``fetch_all``
    Put every registered file in the cache directory. A cached file with a
    matching hash stays. Otherwise a packaged file with a matching hash is
    copied. Otherwise the file downloads. ``fetch_all`` reports progress.
``rattlesnake-data`` and ``python -m rattlesnake.data``
    The console script points at ``__main__.main``. The command requires ``--all``, prints one line per file, stops on a connection failure,
    and continues past an HTTP error or a hash mismatch.

Unit Tests
----------
test_functions_require_keyword_arguments
    Raises ``TypeError`` when a public function receives a positional argument.
test_git_ref
    Maps release and development versions to the expected git ref.
test_git_ref_without_commit
    Falls back to ``dev`` for a development version with no recorded commit.
test_git_ref_installed_version
    Returns a ``v``-prefixed tag, a commit SHA, or ``dev`` for this install.
test_commit_installed
    Reads the SHA from ``rattlesnake._commit``.
test_commit_installed_missing_module
    Returns ``None`` when ``rattlesnake._commit`` does not exist.
test_hash_check
    Accepts, rejects, and handles a missing file.
test_cache_directory_default
    Matches the per-platform cache path that the installation page documents.
test_package_directory_is_examples_package
    Compares ``PACKAGE_DIRECTORY`` with the import system's answer.
test_registry_entries_are_well_formed
    Checks that the registry is not empty and each name and hash is valid.
test_registry_file_exists_in_checkout
    Fails when a registered file is missing from a source checkout.
test_registry_matches_repository_file
    Confirms each registry hash matches the file in the repository.
test_fetch_names_are_registered
    Fails when source code calls ``fetch`` with a name the registry lacks.
test_fetch_unknown_name
    Raises ``KeyError`` for an unregistered name.
test_fetch_prefers_packaged_file
    Returns the packaged file without a download.
test_fetch_downloads_into_cache
    Downloads a missing file into the cache.
test_fetch_rejects_corrupt_download
    Raises ``ValueError`` and caches nothing on a hash mismatch.
test_fetch_all_downloads_into_cache
    Downloads every file when the package holds none.
test_fetch_all_copies_packaged_files
    Copies packaged files into the cache without a download.
test_fetch_all_keeps_cached_files
    Skips files that the cache already holds.
test_cache_fill_replaces_corrupt_cached_file
    Overwrites a cached file whose hash is wrong with the packaged copy.
test_fetch_all_progress
    Calls the progress function once per file, in name order.
test_main_console_script
    ``pyproject.toml`` maps ``rattlesnake-data`` to ``__main__.main``.
test_main_requires_all
    Exits with status 2 when ``--all`` is missing.
test_main_fills_cache
    Exits 0 and prints one line per file.
test_main_stops_on_connection_error
    Exits 1 after the first file when GitHub is unreachable.
test_main_continues_past_file_errors
    Exits 1, lists the failed files, and still fetches the others.
"""

import ast
import functools
import hashlib
import os
import http.server
import importlib.util
import pathlib
import re
import sys
import threading
import tomllib
import types
from collections.abc import Callable, Iterator
from typing import NoReturn

import pytest
import requests

from rattlesnake import data
from rattlesnake.data import __main__ as data_main

REPOSITORY_ROOT = pathlib.Path(data.__file__).resolve().parents[3]
IN_CHECKOUT = (REPOSITORY_ROOT / "pyproject.toml").is_file()
SHA = "6c7224ca3fdb489fad2b3f53637634a0cf5906f2"


@pytest.mark.parametrize(
    "function",
    [
        data.git_ref,
        data.hash_compute,
        data.hash_check,
        data.cache_directory,
        data.fetch,
        data.fetcher_create,
        data.cache_fill,
        data.fetch_all,
    ],
)
def test_functions_require_keyword_arguments(function: Callable[..., object]) -> None:
    """
    Raise ``TypeError`` when a public function receives a positional argument.

    Parameters
    ----------
    function : Callable
        Public function in ``rattlesnake.data``.
    """
    with pytest.raises(TypeError):
        function("positional")


@pytest.mark.parametrize(
    "version, expected",
    [
        ("4.0.0rc4", "v4.0.0rc4"),
        ("4.0.0", "v4.0.0"),
        ("4.0.0rc5.dev5", SHA),
        ("4.0.0rc5.dev5+g6c7224c", SHA),
    ],
)
def test_git_ref(version: str, expected: str) -> None:
    """
    Map package versions to the git ref that holds their data files.

    Parameters
    ----------
    version : str
        Package version passed to ``git_ref``, with ``commit=SHA``.
    expected : str
        Expected ref. A release version maps to ``v<version>`` and ignores the
        commit. A development version maps to the commit SHA.
    """
    assert data.git_ref(version=version, commit=SHA) == expected


def test_git_ref_without_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Fall back to ``dev`` for a development version with no recorded commit.
    """
    monkeypatch.setattr(data, "commit_installed", lambda: None)
    assert data.git_ref(version="4.0.0rc5.dev5") == data.DEV_REF


def test_git_ref_installed_version() -> None:
    """
    Return a ``v``-prefixed tag, a 40-character commit SHA, or ``dev``.
    """
    ref = data.git_ref()
    assert (
        ref == data.DEV_REF or ref.startswith("v") or re.fullmatch(r"[0-9a-f]{40}", ref)
    )


def test_commit_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Read the SHA from ``rattlesnake._commit``.
    """
    module = types.ModuleType("rattlesnake._commit")
    module.commit = SHA  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rattlesnake._commit", module)
    assert data.commit_installed() == SHA


def test_commit_installed_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Return ``None`` when ``rattlesnake._commit`` does not exist.

    A ``None`` entry in ``sys.modules`` makes the import raise ``ImportError``.
    """
    monkeypatch.setitem(sys.modules, "rattlesnake._commit", None)
    assert data.commit_installed() is None


def test_cache_directory_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Match the per-platform cache path that the installation page documents.

    The Example Data Files section of ``acquiring_and_running_rattlesnake.md``
    lists one default cache directory per operating system. CI runs this test on macOS, Linux, and
    Windows, so a wrong row in that table fails on the matching runner.
    """
    monkeypatch.delenv(data.ENV_VAR, raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    path = data.cache_directory(ref="dev")
    home = pathlib.Path.home()
    if sys.platform == "darwin":
        assert path == home / "Library" / "Caches" / "rattlesnake" / "dev"
    elif sys.platform == "win32":
        local = pathlib.Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
        assert path == local / "rattlesnake" / "rattlesnake" / "Cache" / "dev"
    else:
        assert path == home / ".cache" / "rattlesnake" / "dev"


def test_hash_check(tmp_path: pathlib.Path) -> None:
    """
    Accept a matching hash and reject a wrong hash or a missing file.
    """
    path = tmp_path / "file.bin"
    path.write_bytes(b"rattlesnake")
    expected = hashlib.sha256(b"rattlesnake").hexdigest()
    assert data.hash_check(path=path, expected=expected)
    assert not data.hash_check(path=path, expected="0" * 64)
    assert not data.hash_check(path=tmp_path / "missing.bin", expected=expected)


def test_package_directory_is_examples_package() -> None:
    """
    Point ``PACKAGE_DIRECTORY`` at the ``rattlesnake.examples`` package.

    A wrong parent count in ``data/__init__.py`` would send every lookup to the
    wrong directory. The registry tests would then skip instead of fail, so
    this test compares against the import system's own answer.
    """
    spec = importlib.util.find_spec("rattlesnake.examples")
    assert spec is not None
    assert spec.submodule_search_locations is not None
    expected = pathlib.Path(spec.submodule_search_locations[0]).resolve()
    assert data.PACKAGE_DIRECTORY == expected


def test_registry_entries_are_well_formed() -> None:
    """
    Check that the registry is not empty, each name is a relative path, and each
    hash is SHA-256.
    """
    assert data.REGISTRY
    for name, digest in data.REGISTRY.items():
        assert not pathlib.PurePosixPath(name).is_absolute(), name
        assert ".." not in pathlib.PurePosixPath(name).parts, name
        assert re.fullmatch(r"[0-9a-f]{64}", digest), name


@pytest.mark.skipif(not IN_CHECKOUT, reason="not running from a source checkout")
@pytest.mark.parametrize("name", sorted(data.REGISTRY))
def test_registry_file_exists_in_checkout(name: str) -> None:
    """
    Fail when a registered file is missing from a source checkout.

    ``test_registry_matches_repository_file`` skips a missing file so that it
    can run against an installed wheel. In a checkout, a missing file means a
    wrong registry name or a wrong ``PACKAGE_DIRECTORY``, so this test fails.

    Parameters
    ----------
    name : str
        Key in ``REGISTRY``.
    """
    assert (data.PACKAGE_DIRECTORY / name).is_file()


@pytest.mark.parametrize("name", sorted(data.REGISTRY))
def test_registry_matches_repository_file(name: str) -> None:
    """
    Confirm that each registry hash matches the file in the repository.

    This test catches a data file that changed without a registry update. It
    skips when the file is not in the installation, as in a wheel built
    without the registered files.

    Parameters
    ----------
    name : str
        Key in ``REGISTRY``.
    """
    path = data.PACKAGE_DIRECTORY / name
    if not path.is_file():
        pytest.skip(f"{name} is not in this installation")
    assert data.hash_check(path=path, expected=data.REGISTRY[name])


def fetch_name_list(*, path: pathlib.Path) -> list[tuple[int, str | None]]:
    """
    List the ``name`` arguments of every ``fetch`` call in a source file.

    Parameters
    ----------
    path : pathlib.Path
        Python source file.

    Returns
    -------
    list of tuple of int and (str or None)
        Line number and name for each call. A name passed as a module-level
        string constant, such as ``HARDWARE_FILE_NAME``, resolves to its value.
        A name that is neither a literal nor such a constant is ``None``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        called = (
            function.attr
            if isinstance(function, ast.Attribute)
            else getattr(function, "id", None)
        )
        if called != "fetch":
            continue
        for keyword in node.keywords:
            if keyword.arg != "name":
                continue
            value = keyword.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                names.append((node.lineno, value.value))
            elif isinstance(value, ast.Name):
                names.append((node.lineno, constants.get(value.id)))
            else:
                names.append((node.lineno, None))
    return names


def test_fetch_names_are_registered() -> None:
    """
    Fail when source code calls ``fetch`` with a name the registry lacks.

    A renamed or deleted data file leaves the registry consistent with the
    files on disk, so the registry tests pass. A ``fetch`` call that still uses
    the old name fails only when that example runs, with ``KeyError``. This
    test scans every module outside ``rattlesnake.data`` and catches the
    stale name in CI instead. A name the test cannot resolve also fails, so
    every call stays checkable.
    """
    source = pathlib.Path(data.__file__).resolve().parents[1]
    problems = []
    count = 0
    for path in sorted(source.rglob("*.py")):
        if path.parent.name == "data" and path.parent.parent == source:
            continue
        for line, name in fetch_name_list(path=path):
            count += 1
            location = f"{path.relative_to(source.parent)}:{line}"
            if name is None:
                problems.append(f"{location}: name is not a literal or module constant")
            elif name not in data.REGISTRY:
                problems.append(f"{location}: {name!r} is not in the registry")
    assert count > 0, "found no fetch calls; the scan is looking in the wrong place"
    assert problems == [], "\n".join(problems)


def test_fetch_unknown_name() -> None:
    """
    Raise ``KeyError`` for a name that is not in the registry.
    """
    with pytest.raises(KeyError):
        data.fetch(name="not/in/registry.npz")


def test_fetch_prefers_packaged_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Return the packaged file without downloading when its hash matches.

    The test replaces ``pooch.create`` with a function that fails, so any
    download attempt fails the test.
    """
    name = "hardware/sdynpy_system/sdynpy_system.npz"
    content = b"packaged"
    packaged = tmp_path / name
    packaged.parent.mkdir(parents=True)
    packaged.write_bytes(content)
    monkeypatch.setattr(data, "PACKAGE_DIRECTORY", tmp_path)
    monkeypatch.setitem(data.REGISTRY, name, hashlib.sha256(content).hexdigest())

    def fail(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("fetch must not download when the packaged file matches")

    monkeypatch.setattr(data.pooch, "create", fail)
    assert data.fetch(name=name) == str(packaged)


@pytest.fixture(name="http_server")
def http_server_fixture(tmp_path: pathlib.Path) -> Iterator[str]:
    """
    Serve ``tmp_path`` over HTTP on a free local port.

    Pooch accepts only ``http``, ``https``, ``ftp``, ``sftp``, and ``doi``
    URLs, so a ``file://`` URL cannot stand in for a download.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory to serve.

    Yields
    ------
    str
        Base URL of the server, ending in ``/``.
    """

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        """
        Serve files without writing a log line for each request.
        """

        def log_message(self, format: str, *args: object) -> None:
            """
            Discard the log message.

            Parameters
            ----------
            format : str
                Format string for the message.
            *args : object
                Values for the format string.
            """

    handler = functools.partial(QuietHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()
    thread.join()


def test_fetch_downloads_into_cache(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, http_server: str
) -> None:
    """
    Download a missing file into the cache and return the cached path.

    The packaged directory is empty, so the file must come from the server.
    ``RATTLESNAKE_DATADIR`` points the cache at ``tmp_path``.
    """
    name = "hardware/sdynpy_system/sdynpy_system.npz"
    content = b"downloaded"
    (tmp_path / name).parent.mkdir(parents=True)
    (tmp_path / name).write_bytes(content)
    cache = tmp_path / "cache"
    monkeypatch.setattr(data, "PACKAGE_DIRECTORY", tmp_path / "empty")
    monkeypatch.setattr(data, "BASE_URL", http_server)
    monkeypatch.setitem(data.REGISTRY, name, hashlib.sha256(content).hexdigest())
    monkeypatch.setenv(data.ENV_VAR, str(cache))
    path = data.fetch(name=name)
    assert path == str(cache / name)
    assert (cache / name).read_bytes() == content


def test_fetch_rejects_corrupt_download(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, http_server: str
) -> None:
    """
    Raise ``ValueError`` and keep nothing in the cache when the hash differs.

    The server holds bytes that do not match the registry hash. Pooch checks
    the hash after the download and deletes the file when it does not match.
    """
    name = "hardware/sdynpy_system/sdynpy_system.npz"
    (tmp_path / name).parent.mkdir(parents=True)
    (tmp_path / name).write_bytes(b"corrupt")
    monkeypatch.setattr(data, "PACKAGE_DIRECTORY", tmp_path / "empty")
    monkeypatch.setattr(data, "BASE_URL", http_server)
    monkeypatch.setitem(data.REGISTRY, name, hashlib.sha256(b"expected").hexdigest())
    monkeypatch.setenv(data.ENV_VAR, str(tmp_path / "cache"))
    with pytest.raises(ValueError, match="hash"):
        data.fetch(name=name)
    assert not (tmp_path / "cache" / name).exists()


NAME_LIST = [
    "environment/read/read_v4.nc4",
    "hardware/sdynpy_system/sdynpy_system.npz",
    "frame_wing/data/srs_specification.npz",
]


def registry_prepare(
    *, root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, bytes]:
    """
    Replace the registry with three small files and point the cache at a folder.

    The function writes the three files under ``root``, which the local server
    can serve. It empties the package directory and sets ``RATTLESNAKE_DATADIR``
    to ``root / "cache"``.

    Parameters
    ----------
    root : pathlib.Path
        Directory that holds the files to serve.
    monkeypatch : pytest.MonkeyPatch
        Fixture that replaces the registry and the environment.

    Returns
    -------
    dict of str to bytes
        File contents by registry name.
    """
    content_by_name = {name: name.encode() for name in NAME_LIST}
    for name, content in content_by_name.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)
    monkeypatch.setattr(
        data,
        "REGISTRY",
        {name: hashlib.sha256(c).hexdigest() for name, c in content_by_name.items()},
    )
    monkeypatch.setattr(data, "PACKAGE_DIRECTORY", root / "empty")
    monkeypatch.setenv(data.ENV_VAR, str(root / "cache"))
    return content_by_name


def download_forbid(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Make any Pooch download fail the test.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture that replaces ``pooch.Pooch.fetch``.
    """

    def fail(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("the test must not download")

    monkeypatch.setattr(data.pooch.Pooch, "fetch", fail)


def test_fetch_all_downloads_into_cache(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, http_server: str
) -> None:
    """
    Download every file into the cache when the package holds none.
    """
    content_by_name = registry_prepare(root=tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setattr(data, "BASE_URL", http_server)
    paths = data.fetch_all()
    assert paths == [str(tmp_path / "cache" / name) for name in sorted(content_by_name)]
    for name, content in content_by_name.items():
        assert (tmp_path / "cache" / name).read_bytes() == content


def test_fetch_all_copies_packaged_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Copy packaged files into the cache, as in a clone, without a download.

    Without the copy, the cache stays empty and a staged directory is useless
    on a machine that installed a wheel.
    """
    content_by_name = registry_prepare(root=tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setattr(data, "PACKAGE_DIRECTORY", tmp_path)
    download_forbid(monkeypatch)
    paths = data.fetch_all()
    assert len(paths) == len(content_by_name)
    for name, content in content_by_name.items():
        assert (tmp_path / "cache" / name).read_bytes() == content


def test_fetch_all_keeps_cached_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Skip files that the cache already holds with a matching hash.

    The package directory is empty and downloads fail, so only the cached
    copies can satisfy the call.
    """
    content_by_name = registry_prepare(root=tmp_path, monkeypatch=monkeypatch)
    for name, content in content_by_name.items():
        (tmp_path / "cache" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "cache" / name).write_bytes(content)
    download_forbid(monkeypatch)
    assert len(data.fetch_all()) == len(content_by_name)


def test_cache_fill_replaces_corrupt_cached_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Overwrite a cached file whose hash is wrong with the packaged copy.
    """
    content_by_name = registry_prepare(root=tmp_path, monkeypatch=monkeypatch)
    name = NAME_LIST[0]
    monkeypatch.setattr(data, "PACKAGE_DIRECTORY", tmp_path)
    (tmp_path / "cache" / name).parent.mkdir(parents=True)
    (tmp_path / "cache" / name).write_bytes(b"corrupt")
    download_forbid(monkeypatch)
    fetcher = data.fetcher_create(ref="ref")
    assert data.cache_fill(name=name, fetcher=fetcher) == str(tmp_path / "cache" / name)
    assert (tmp_path / "cache" / name).read_bytes() == content_by_name[name]


def test_fetch_all_progress(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Call the progress function once per file, in name order.
    """
    registry_prepare(root=tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setattr(data, "PACKAGE_DIRECTORY", tmp_path)
    calls = []
    data.fetch_all(
        progress=lambda index, total, name: calls.append((index, total, name))
    )
    assert calls == [(i, 3, name) for i, name in enumerate(sorted(NAME_LIST), 1)]


@pytest.mark.skipif(not IN_CHECKOUT, reason="not running from a source checkout")
def test_main_console_script() -> None:
    """
    Map ``rattlesnake-data`` to ``__main__.main`` in ``pyproject.toml``.
    """
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as file:
        scripts = tomllib.load(file)["project"]["scripts"]
    assert scripts["rattlesnake-data"] == "rattlesnake.data.__main__:main"


def test_main_requires_all() -> None:
    """
    Exit with status 2 when ``--all`` is missing.
    """
    with pytest.raises(SystemExit) as error:
        data_main.main([])
    assert error.value.code == 2


def test_main_fills_cache(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    http_server: str,
) -> None:
    """
    Exit 0 and print one line per file, then the cache directory.
    """
    registry_prepare(root=tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setattr(data, "BASE_URL", http_server)
    assert data_main.main(["--all"]) == 0
    lines = capsys.readouterr().out.splitlines()
    for index, name in enumerate(sorted(NAME_LIST), 1):
        assert lines[index - 1].startswith(f"[{index}/3] {name}")
    assert f"Cache directory: {tmp_path / 'cache'}" in lines
    assert "3 of 3 files ready" in lines[-1]


def test_main_stops_on_connection_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Exit 1 after the first file when GitHub is unreachable.

    A connection failure affects every file, so waiting through the retries of
    the other files would only delay the same message.
    """
    registry_prepare(root=tmp_path, monkeypatch=monkeypatch)
    calls = []

    def unreachable(*, name: str, fetcher: object) -> NoReturn:
        calls.append(name)
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(data, "cache_fill", unreachable)
    assert data_main.main(["--all"]) == 1
    assert calls == [sorted(NAME_LIST)[0]]
    assert "Cannot reach GitHub" in capsys.readouterr().err


def test_main_continues_past_file_errors(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    http_server: str,
) -> None:
    """
    Exit 1, list the failed files, and still fetch the others.

    One file returns HTTP 404 and another fails its hash check. The third file
    must still arrive in the cache.
    """
    registry_prepare(root=tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setattr(data, "BASE_URL", http_server)
    not_found, bad_hash, good = sorted(NAME_LIST)
    original = data.cache_fill

    def cache_fill_faulty(*, name: str, fetcher: object) -> str:
        if name == not_found:
            raise requests.exceptions.HTTPError("404")
        if name == bad_hash:
            raise ValueError("hash mismatch")
        return original(name=name, fetcher=fetcher)

    monkeypatch.setattr(data, "cache_fill", cache_fill_faulty)
    assert data_main.main(["--all"]) == 1
    err = capsys.readouterr().err
    assert "2 of 3 files failed" in err
    assert f"  {not_found}" in err
    assert f"  {bad_hash}" in err
    assert (tmp_path / "cache" / good).is_file()
