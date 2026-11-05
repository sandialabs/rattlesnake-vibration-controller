"""Tests for the example stream metadata builders in ``STREAM_DICT``."""

import os
from pathlib import Path

import pytest

from rattlesnake.examples.example_registry import STREAM_DICT
from rattlesnake.process.streaming import StreamType

WRITING_TYPES = [
    StreamType.IMMEDIATELY,
    StreamType.MANUAL,
    StreamType.TEST_LEVEL,
    StreamType.PROFILE_INSTRUCTION,
]


@pytest.mark.parametrize("stream_type", WRITING_TYPES)
def test_stream_file_in_output_directory(
    *, stream_type: StreamType, tmp_path: Path
) -> None:
    """Place ``example_stream.nc4`` in the given ``output_directory``.

    Parameters
    ----------
    stream_type : StreamType
        Stream type whose builder is under test.
    tmp_path : Path
        Pytest temporary directory passed as ``output_directory``.
    """
    metadata = STREAM_DICT[stream_type]("env", output_directory=tmp_path)

    assert Path(metadata.stream_file) == tmp_path / "example_stream.nc4"


@pytest.mark.parametrize("stream_type", WRITING_TYPES)
def test_stream_file_defaults_to_cwd(
    *, stream_type: StreamType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Place ``example_stream.nc4`` in the cwd when no directory is given.

    Parameters
    ----------
    stream_type : StreamType
        Stream type whose builder is under test.
    tmp_path : Path
        Pytest temporary directory that becomes the cwd.
    monkeypatch : pytest.MonkeyPatch
        Fixture that changes the cwd to ``tmp_path``.
    """
    monkeypatch.chdir(tmp_path)

    metadata = STREAM_DICT[stream_type]("env")

    assert os.path.samefile(Path(metadata.stream_file).parent, tmp_path)
    assert Path(metadata.stream_file).name == "example_stream.nc4"


def test_no_stream_accepts_output_directory(*, tmp_path: Path) -> None:
    """Accept and ignore ``output_directory`` for ``StreamType.NO_STREAM``.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory passed as ``output_directory``.
    """
    metadata = STREAM_DICT[StreamType.NO_STREAM]("env", output_directory=tmp_path)

    assert metadata.stream_type is StreamType.NO_STREAM
