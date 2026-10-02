"""
Locate large data files that are not shipped in the PyPI distribution.

Each file listed in ``_registry.py`` is resolved in this order. First, the copy in
the installed package tree, when its SHA-256 hash matches. Second, the local
cache. Third, a download from the GitHub repository with Pooch.

The download URL pins the file to the installed release. A tagged version
``X.Y.Z`` reads from the tag ``vX.Y.Z``. An untagged development version reads
from the ``dev`` branch.

Set ``RATTLESNAKE_DATADIR`` to use a different cache directory, for example a
directory that was staged ahead of time on a machine without network access.

Attributes
----------
BASE_URL : str
    Download URL template. The ``{ref}`` field takes the value from ``git_ref``.
    The path after ``raw/{ref}/`` is the examples directory in the repository.
DEV_REF : str
    Git ref that development versions download from.
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
import importlib.metadata
import os
import pathlib

import pooch

from rattlesnake.data._registry import registry as REGISTRY

BASE_URL = (
    "https://github.com/sandialabs/rattlesnake-vibration-controller/raw/"
    "{ref}/src/rattlesnake/examples/"
)
DEV_REF = "dev"
ENV_VAR = "RATTLESNAKE_DATADIR"
PACKAGE_DIRECTORY = pathlib.Path(__file__).resolve().parents[1] / "examples"


def git_ref(*, version: str | None = None) -> str:
    """
    Return the git ref that holds the data files for a package version.

    Parameters
    ----------
    version : str, optional
        Package version. Defaults to the installed version of
        ``rattlesnake-vibration-controller``.

    Returns
    -------
    str
        ``"dev"`` for a development version, such as ``4.0.0rc2.dev70``.
        ``"v" + version`` for a tagged version, such as ``v4.0.0rc4``.
    """
    if version is None:
        version = importlib.metadata.version("rattlesnake-vibration-controller")
    if ".dev" in version or "+" in version:
        return DEV_REF
    return f"v{version}"


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
        ``"hardware/sdynpy_system/sdynpy_system.npz"``.

    Returns
    -------
    str
        Path of a local file whose hash matches the registry.

    Raises
    ------
    KeyError
        If ``name`` is not in ``REGISTRY``.
    ConnectionError
        If the file is not available locally and the download fails.
    """
    expected = REGISTRY[name]
    packaged = PACKAGE_DIRECTORY / name
    if hash_check(path=packaged, expected=expected):
        return str(packaged)
    ref = git_ref()
    fetcher = pooch.create(
        path=cache_directory(ref=ref),
        base_url=BASE_URL.format(ref=ref),
        registry=REGISTRY,
        retry_if_failed=3,
    )
    return fetcher.fetch(name)
