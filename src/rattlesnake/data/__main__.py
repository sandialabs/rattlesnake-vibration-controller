"""
Fill the data cache so Rattlesnake can run without network access.

Run the command on a machine with network access::

    RATTLESNAKE_DATADIR=/path/to/stage python -m rattlesnake.data --all

Copy the directory to the offline machine and set ``RATTLESNAKE_DATADIR``
there. For each registered file, the command keeps a cached copy whose hash
matches, otherwise copies a matching file from the installed package,
otherwise downloads the file. A rerun skips the files that are already cached.
"""

import argparse
import sys
import time

import requests

from rattlesnake import data

NETWORK_ERRORS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)


def main(argv: list[str] | None = None) -> int:
    """
    Put every registered data file in the cache directory.

    The command prints one line per file. A connection failure stops the
    command, because it affects every file. An HTTP error or a hash mismatch
    affects one file, so the command records it, continues, and lists the failed
    files at the end.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        0 when every file is in the cache directory. 1 after a connection
        failure or after any other failed file. ``argparse`` exits with 2 when
        ``--all`` is missing.
    """
    parser = argparse.ArgumentParser(
        prog="python -m rattlesnake.data",
        description="Put every registered data file in the cache directory.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        required=True,
        help="Fetch every registered data file.",
    )
    parser.parse_args(argv)

    ref = data.git_ref()
    fetcher = data.fetcher_create(ref=ref)
    names = sorted(data.REGISTRY)
    width = len(str(len(names)))
    failures = []
    start = time.perf_counter()
    for index, name in enumerate(names, start=1):
        label = f"[{index:>{width}}/{len(names)}] {name}"
        file_start = time.perf_counter()
        try:
            data.cache_fill(name=name, fetcher=fetcher)
        except NETWORK_ERRORS as error:
            print(f"{label}  FAILED: {error}", file=sys.stderr)
            print(
                "Cannot reach GitHub. Check the network, or run this command on a "
                "machine with network access.",
                file=sys.stderr,
            )
            return 1
        except (requests.exceptions.HTTPError, ValueError) as error:
            failures.append(name)
            print(f"{label}  FAILED: {error}", file=sys.stderr)
            continue
        print(f"{label}  {time.perf_counter() - file_start:.1f} s")

    print(f"Cache directory: {fetcher.path}")
    print(
        f"{len(names) - len(failures)} of {len(names)} files ready in {time.perf_counter() - start:.1f} s"
    )
    if failures:
        print(f"{len(failures)} of {len(names)} files failed:", file=sys.stderr)
        for name in failures:
            print(f"  {name}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
