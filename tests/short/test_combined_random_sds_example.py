"""Tests for the output location in the combined Random and SDS example."""

from pathlib import Path

import pytest

from rattlesnake.engine import RattlesnakeController
from rattlesnake.examples.frame_wing.combined_random_sds import (
    build_combined_controller,
)


def record_start_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> list:
    """Spy on ``start_acquisition`` and skip the slow system identification run.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture that installs the spy.

    Returns
    -------
    list
        Stream metadata from every ``start_acquisition`` call, in order.
    """
    recorded = []
    original = RattlesnakeController.start_acquisition

    def spy(self, stream_metadata, *args, **kwargs):
        recorded.append(stream_metadata)
        return original(self, stream_metadata, *args, **kwargs)

    monkeypatch.setattr(RattlesnakeController, "start_acquisition", spy)
    monkeypatch.setattr(RattlesnakeController, "run_system_id", lambda *a, **k: None)
    return recorded


def test_stream_file_in_output_directory(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Place ``streaming_example.nc4`` in the given ``output_directory``.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory passed as ``output_directory``.
    monkeypatch : pytest.MonkeyPatch
        Fixture that installs the ``start_acquisition`` spy.
    """
    recorded = record_start_acquisition(monkeypatch)

    controller = build_combined_controller(output_directory=tmp_path)
    controller.shutdown()

    stream_metadata = recorded[-1]
    assert Path(stream_metadata.stream_file) == tmp_path / "streaming_example.nc4"


def test_stream_file_defaults_to_cwd(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Place ``streaming_example.nc4`` in the cwd when no directory is given.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory that becomes the cwd.
    monkeypatch : pytest.MonkeyPatch
        Fixture that changes the cwd to ``tmp_path`` and installs the
        ``start_acquisition`` spy.
    """
    monkeypatch.chdir(tmp_path)
    recorded = record_start_acquisition(monkeypatch)

    controller = build_combined_controller()
    controller.shutdown()

    stream_metadata = recorded[-1]
    assert (
        Path(stream_metadata.stream_file).resolve()
        == (tmp_path / "streaming_example.nc4").resolve()
    )
