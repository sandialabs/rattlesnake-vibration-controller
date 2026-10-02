"""
Unit tests for ``rattlesnake.data``, which locates large data files.

None of these tests use the internet. Tests that exercise a download serve a
temporary directory from a local HTTP server on a free port.

The tests cover five areas.

Keyword-only signatures
    Every public function takes keyword arguments only, so a positional call
    raises ``TypeError``.
``git_ref``
    Maps a package version to the git ref that holds its data files. A tagged
    version maps to ``v<version>``. A development version maps to ``dev``.
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

Unit Tests
----------
test_functions_require_keyword_arguments
    Raises ``TypeError`` when a public function receives a positional argument.
test_git_ref
    Maps tagged and development versions to the expected git ref.
test_git_ref_installed_version
    Returns ``dev`` or a ``v``-prefixed tag for the installed version.
test_hash_check
    Accepts, rejects, and handles a missing file.
test_package_directory_is_examples_package
    Compares ``PACKAGE_DIRECTORY`` with the import system's answer.
test_registry_entries_are_well_formed
    Checks that the registry is not empty and each name and hash is valid.
test_registry_file_exists_in_checkout
    Fails when a registered file is missing from a source checkout.
test_registry_matches_repository_file
    Confirms each registry hash matches the file in the repository.
test_fetch_unknown_name
    Raises ``KeyError`` for an unregistered name.
test_fetch_prefers_packaged_file
    Returns the packaged file without a download.
test_fetch_downloads_into_cache
    Downloads a missing file into the cache.
test_fetch_rejects_corrupt_download
    Raises ``ValueError`` and caches nothing on a hash mismatch.
"""

import functools
import hashlib
import http.server
import importlib.util
import pathlib
import re
import threading
from collections.abc import Callable, Iterator
from typing import NoReturn

import pytest

from rattlesnake import data

REPOSITORY_ROOT = pathlib.Path(data.__file__).resolve().parents[3]
IN_CHECKOUT = (REPOSITORY_ROOT / "pyproject.toml").is_file()


@pytest.mark.parametrize(
    "function",
    [
        data.git_ref,
        data.hash_compute,
        data.hash_check,
        data.cache_directory,
        data.fetch,
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
        ("4.0.0rc4.dev2", "dev"),
        ("4.0.0rc4.dev2+g1234567", "dev"),
    ],
)
def test_git_ref(version: str, expected: str) -> None:
    """
    Map package versions to the git ref that holds their data files.

    Parameters
    ----------
    version : str
        Package version passed to ``git_ref``.
    expected : str
        Expected ref. A tagged version maps to ``v<version>``. A development
        version maps to ``dev``.
    """
    assert data.git_ref(version=version) == expected


def test_git_ref_installed_version() -> None:
    """
    Return either ``dev`` or a ``v``-prefixed tag for the installed version.
    """
    ref = data.git_ref()
    assert ref == "dev" or ref.startswith("v")


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


@pytest.fixture
def http_server(tmp_path: pathlib.Path) -> Iterator[str]:
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
