"""
Locate large data files that are not shipped in the PyPI distribution.

Each file listed in ``_registry.py`` is resolved in this order. First, the copy in
the installed package tree, when its SHA-256 hash matches. Second, the local
cache. Third, a download from the GitHub repository with Pooch.

The download URL pins the file to the installed build. A release version
``X.Y.Z`` reads from the tag ``vX.Y.Z``. A development version reads from the
commit it was built from. ``hatch_build.py`` records that commit SHA in
``rattlesnake._commit`` at build time.

Set ``RATTLESNAKE_DATADIR`` to use a different cache directory, for example a
directory that was staged ahead of time on a machine without network access.
``rattlesnake-data --all`` stages a cache directory in one step. For
each registered file, it keeps a cached copy whose hash matches, otherwise
copies a matching packaged file into the cache, otherwise downloads the file.

Attributes
----------
BASE_URL : str
    Download URL template. The ``{ref}`` field takes the value from ``git_ref``.
    The path after ``raw/{ref}/`` is the examples directory in the repository.
DEV_REF : str
    Git ref that a development version downloads from when it has no recorded
    commit SHA.
ENV_VAR : str
    Name of the environment variable that overrides the cache directory.
PACKAGE_DIRECTORY : pathlib.Path
    Directory in the installed package that may already hold registered files.
REGISTRY : dict of str to str
    Maps each file name, relative to ``PACKAGE_DIRECTORY``, to its SHA-256 hash.
    It is the ``registry`` dictionary in ``_registry.py``. Do not edit that file
    by hand. Run ``python -m rattlesnake.data.registry`` to regenerate it.
"""

import hashlib
import importlib
import importlib.metadata
import os
import pathlib
import shutil
from collections.abc import Callable

import pooch

from rattlesnake.data._registry import registry as REGISTRY

BASE_URL = (
    "https://github.com/sandialabs/rattlesnake-vibration-controller/raw/"
    "{ref}/src/rattlesnake/examples/"
)
DEV_REF = "dev"
ENV_VAR = "RATTLESNAKE_DATADIR"
PACKAGE_DIRECTORY = pathlib.Path(__file__).resolve().parents[1] / "examples"


def commit_installed() -> str | None:
    """
    Return the commit SHA that ``hatch_build.py`` recorded at build time.

    Returns
    -------
    str or None
        Full SHA from ``rattlesnake._commit``. ``None`` when that module is
        missing, as in a source tree that was never built or installed, or when
        the build had no git repository.
    """
    try:
        module = importlib.import_module("rattlesnake._commit")
    except ImportError:
        return None
    return getattr(module, "commit", None)


def git_ref(*, version: str | None = None, commit: str | None = None) -> str:
    """
    Return the git ref that holds the data files for a package build.

    Parameters
    ----------
    version : str, optional
        Package version. Defaults to the installed version of
        ``rattlesnake-vibration-controller``.
    commit : str, optional
        Commit SHA of the build. Defaults to ``commit_installed()``.

    Returns
    -------
    str
        ``"v" + version`` for a release version, such as ``v4.0.0rc4``. The
        commit SHA for a development version, such as ``4.0.0rc5.dev5``.
        ``"dev"`` for a development version with no recorded commit.
    """
    if version is None:
        version = importlib.metadata.version("rattlesnake-vibration-controller")
    if ".dev" not in version and "+" not in version:
        return f"v{version}"
    if commit is None:
        commit = commit_installed()
    return commit or DEV_REF


def hash_compute(*, path: str | os.PathLike[str]) -> str:
    """
    Compute the SHA-256 hash of a file.

    Parameters
    ----------
    path : str or os.PathLike
        File to read. The function reads it in 1 MiB blocks.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 hash.
    """
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_check(*, path: str | os.PathLike[str], expected: str) -> bool:
    """
    Check whether a file exists and has the expected SHA-256 hash.

    Parameters
    ----------
    path : str or os.PathLike
        File to check.
    expected : str
        Expected SHA-256 hash as a hexadecimal string.

    Returns
    -------
    bool
        ``True`` if the file exists and its hash equals ``expected``.
    """
    if not pathlib.Path(path).is_file():
        return False
    return hash_compute(path=path) == expected


def cache_directory(*, ref: str) -> pathlib.Path:
    """
    Return the directory that holds downloaded files for a git ref.

    Parameters
    ----------
    ref : str
        Git ref from ``git_ref``.

    Returns
    -------
    pathlib.Path
        The directory named by ``RATTLESNAKE_DATADIR`` when it is set.
        Otherwise a per-ref directory in the user cache.
    """
    override = os.environ.get(ENV_VAR)
    if override:
        return pathlib.Path(override).expanduser()
    return pathlib.Path(pooch.os_cache("rattlesnake")) / ref


def fetch(*, name: str) -> str:
    """
    Return the local path of a registered data file, downloading it if needed.

    Parameters
    ----------
    name : str
        Key in ``REGISTRY``, for example
        ``"hardware/data/sdynpy_system.npz"``.

    Returns
    -------
    str
        Path of a local file whose hash matches the registry.

    Raises
    ------
    KeyError
        If ``name`` is not in ``REGISTRY``.
    requests.exceptions.ConnectionError
        If the file is not available locally and GitHub cannot be reached.
    requests.exceptions.HTTPError
        If GitHub returns an error, such as 404 for a commit it does not have.
    ValueError
        If every download attempt returns a file whose hash does not match.
    """
    expected = REGISTRY[name]
    packaged = PACKAGE_DIRECTORY / name
    if hash_check(path=packaged, expected=expected):
        return str(packaged)
    return fetcher_create(ref=git_ref()).fetch(name)


def fetcher_create(*, ref: str) -> pooch.Pooch:
    """
    Return a Pooch downloader for the registered files at a git ref.

    Parameters
    ----------
    ref : str
        Git ref from ``git_ref``.

    Returns
    -------
    pooch.Pooch
        Downloader that stores files in ``cache_directory(ref=ref)`` and retries
        each failed download 3 times.
    """
    return pooch.create(
        path=cache_directory(ref=ref),
        base_url=BASE_URL.format(ref=ref),
        registry=REGISTRY,
        retry_if_failed=3,
    )


def cache_fill(*, name: str, fetcher: pooch.Pooch) -> str:
    """
    Put one registered file in the cache directory of a downloader.

    The function tries three steps in order. First, it keeps a cached file whose
    hash matches. Second, it copies a packaged file whose hash matches. Third, it
    downloads the file. Unlike ``fetch``, it never returns the packaged path, so
    the cache holds the file afterward.

    Parameters
    ----------
    name : str
        Key in ``REGISTRY``.
    fetcher : pooch.Pooch
        Downloader from ``fetcher_create``. Its path is the cache directory.

    Returns
    -------
    str
        Path of the file in the cache directory.

    Raises
    ------
    KeyError
        If ``name`` is not in ``REGISTRY``.
    requests.exceptions.ConnectionError
        If the file must be downloaded and GitHub cannot be reached.
    requests.exceptions.HTTPError
        If GitHub returns an error, such as 404 for a commit it does not have.
    ValueError
        If every download attempt returns a file whose hash does not match.
    """
    expected = REGISTRY[name]
    target = pathlib.Path(fetcher.path) / name
    packaged = PACKAGE_DIRECTORY / name
    if hash_check(path=target, expected=expected):
        return str(target)
    if hash_check(path=packaged, expected=expected):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(packaged, target)
        return str(target)
    return fetcher.fetch(name)


def fetch_all(*, progress: Callable[[int, int, str], None] | None = None) -> list[str]:
    """
    Put every registered file in the cache directory.

    The function stops at the first failure. The command
    ``rattlesnake-data --all`` continues past a failed file instead.

    Parameters
    ----------
    progress : Callable, optional
        Called after each file with the 1-based index, the total count, and the
        file name.

    Returns
    -------
    list of str
        Paths of the files in the cache directory, sorted by name.

    Raises
    ------
    requests.exceptions.ConnectionError
        If a file must be downloaded and GitHub cannot be reached.
    requests.exceptions.HTTPError
        If GitHub returns an error for a file.
    ValueError
        If every download attempt for a file returns a hash mismatch.
    """
    fetcher = fetcher_create(ref=git_ref())
    names = sorted(REGISTRY)
    paths = []
    for index, name in enumerate(names, start=1):
        paths.append(cache_fill(name=name, fetcher=fetcher))
        if progress is not None:
            progress(index, len(names), name)
    return paths
