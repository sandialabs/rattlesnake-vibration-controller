import multiprocessing as mp
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.process.streaming import (
    StreamMetadata,
    StreamingProcess,
    StreamType,
    streaming_process,
)
from rattlesnake.testing.mock_utilities import (
    mock_event_container,
    mock_queue_container,
    skeleton_environment_metadata,
    skeleton_hardware_metadata,
)
from rattlesnake.utilities import GlobalCommands, RattlesnakeError


# region Fixtures
@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def streaming_setup(request):
    """
    Create queue and event containers for threaded and multiprocessing modes.
    """
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)

    return use_thread, queue_container, event_container


@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def streaming(request):
    """
    Create a ``StreamingProcess`` in threaded and multiprocessing modes.
    """
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)

    return StreamingProcess(
        "Process Name",
        queue_container,
        event_container.streaming_ready_event,
    )


@pytest.fixture
def hardware_metadata():
    """
    Create mock hardware metadata for streaming tests.
    """
    return skeleton_hardware_metadata()


@pytest.fixture
def environment_metadata_dict():
    """
    Create mock environment metadata dictionary for streaming tests.
    """
    return {"Environment 0": skeleton_environment_metadata()}


# endregion


# region StreamType
def test_stream_type_unique_integer_values():
    """
    Iterates through each enum member to confirm unique integer values.
    """
    values = [member.value for member in StreamType]

    assert all(isinstance(value, int) for value in values)
    assert len(values) == len(set(values))


def test_stream_type_expected_members():
    """
    Verifies that expected streaming modes are defined.
    """
    assert StreamType.NO_STREAM.value == 0
    assert StreamType.IMMEDIATELY.value == 1
    assert StreamType.PROFILE_INSTRUCTION.value == 2
    assert StreamType.TEST_LEVEL.value == 3
    assert StreamType.MANUAL.value == 4


# endregion


# region StreamMetadata
def test_stream_metadata_init_defaults():
    """
    Verifies that stream metadata initializes required attributes with default
    values.
    """
    stream_metadata = StreamMetadata()

    assert isinstance(stream_metadata, StreamMetadata)
    assert stream_metadata.stream_type == StreamType.NO_STREAM
    assert stream_metadata.stream_file is None
    assert stream_metadata.test_level_environment_name is None


def test_stream_metadata_init_values(tmp_path):
    """
    Confirms that initialization stores supplied stream metadata values.
    """
    stream_file = tmp_path / "stream.nc4"

    stream_metadata = StreamMetadata(
        stream_type=StreamType.MANUAL,
        stream_file=stream_file,
        test_level_environment_name="Environment 0",
    )

    assert stream_metadata.stream_type == StreamType.MANUAL
    assert stream_metadata.stream_file == stream_file
    assert stream_metadata.test_level_environment_name == "Environment 0"


@pytest.mark.parametrize(
    "stream_type",
    [
        StreamType.MANUAL,
        StreamType.IMMEDIATELY,
        StreamType.PROFILE_INSTRUCTION,
    ],
)
def test_stream_metadata_validate_streaming_modes_with_valid_file(
    stream_type,
    tmp_path,
):
    """
    Verifies that non-test-level streaming modes pass validation when a valid
    stream file path is supplied.
    """
    stream_metadata = StreamMetadata(
        stream_type=stream_type,
        stream_file=tmp_path / "stream.nc4",
    )

    stream_metadata.validate()


def test_stream_metadata_validate_no_stream_without_file():
    """
    Verifies that ``NO_STREAM`` does not require a stream file.
    """
    stream_metadata = StreamMetadata(
        stream_type=StreamType.NO_STREAM,
        stream_file=None,
    )

    stream_metadata.validate()


@pytest.mark.parametrize(
    "stream_file",
    [
        None,
        123,
        object(),
    ],
)
def test_stream_metadata_validate_enabled_stream_requires_valid_file(stream_file):
    """
    Verifies that streaming modes other than ``NO_STREAM`` require a valid file
    path.
    """
    stream_metadata = StreamMetadata(
        stream_type=StreamType.MANUAL,
        stream_file=stream_file,
    )

    with pytest.raises(RattlesnakeError):
        stream_metadata.validate()


def test_stream_metadata_validate_parent_directory_must_exist(tmp_path):
    """
    Verifies that validation fails when the stream file parent directory does
    not exist.
    """
    stream_metadata = StreamMetadata(
        stream_type=StreamType.MANUAL,
        stream_file=tmp_path / "missing_directory" / "stream.nc4",
    )

    with pytest.raises(RattlesnakeError):
        stream_metadata.validate()


def test_stream_metadata_validate_path_object(tmp_path):
    """
    Verifies that ``Path`` objects are accepted as stream file paths.
    """
    stream_metadata = StreamMetadata(
        stream_type=StreamType.MANUAL,
        stream_file=Path(tmp_path / "stream.nc4"),
    )

    stream_metadata.validate()


def test_stream_metadata_validate_test_level_requires_environment(tmp_path):
    """
    Verifies that ``TEST_LEVEL`` streaming requires a valid environment name.
    """
    stream_metadata = StreamMetadata(
        stream_type=StreamType.TEST_LEVEL,
        stream_file=tmp_path / "stream.nc4",
        test_level_environment_name=None,
    )

    with pytest.raises(RattlesnakeError):
        stream_metadata.validate()


@pytest.mark.parametrize(
    "test_level_environment_name",
    [
        "",
        "Environment 0",
    ],
)
def test_stream_metadata_validate_test_level_with_environment_name(
    test_level_environment_name,
    tmp_path,
):
    """
    Verifies that ``TEST_LEVEL`` streaming passes validation when a string
    environment name is supplied.
    """
    stream_metadata = StreamMetadata(
        stream_type=StreamType.TEST_LEVEL,
        stream_file=tmp_path / "stream.nc4",
        test_level_environment_name=test_level_environment_name,
    )

    stream_metadata.validate()


def test_stream_metadata_validate_test_level_environment_must_be_string(tmp_path):
    """
    Verifies that ``TEST_LEVEL`` streaming rejects non-string environment
    names.
    """
    stream_metadata = StreamMetadata(
        stream_type=StreamType.TEST_LEVEL,
        stream_file=tmp_path / "stream.nc4",
        test_level_environment_name=123,
    )

    with pytest.raises(RattlesnakeError):
        stream_metadata.validate()


# endregion


# region StreamingProcess
@pytest.mark.parametrize("use_thread", [True, False])
def test_streaming_init(use_thread):
    """
    Verifies that ``StreamingProcess`` initializes successfully and is an
    ``AbstractMessageProcess``.
    """
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)

    stream_process = StreamingProcess(
        "Process Name",
        queue_container,
        event_container.streaming_ready_event,
    )

    assert isinstance(stream_process, StreamingProcess)
    assert isinstance(stream_process, AbstractMessageProcess)

    assert stream_process.process_name == "Process Name"
    assert stream_process.netcdf_handle is None
    assert stream_process.stream_variable == "time_data"
    assert stream_process.stream_dimension == "time_samples"
    assert stream_process.stream_index == 0
    assert stream_process.ready_event.is_set()


def test_streaming_command_map(streaming):
    """
    Verifies that streaming-specific global commands are mapped to the correct
    process methods.
    """
    assert streaming.command_map[GlobalCommands.INITIALIZE_STREAMING] == (
        streaming.initialize
    )
    assert streaming.command_map[GlobalCommands.STREAMING_DATA] == (
        streaming.write_data
    )
    assert streaming.command_map[GlobalCommands.FINALIZE_STREAMING] == (
        streaming.finalize
    )
    assert streaming.command_map[GlobalCommands.CREATE_NEW_STREAM] == (
        streaming.create_new_stream
    )


@pytest.mark.parametrize(
    "stream_type",
    [
        StreamType.IMMEDIATELY,
        StreamType.PROFILE_INSTRUCTION,
        StreamType.TEST_LEVEL,
        StreamType.MANUAL,
    ],
)
@mock.patch("rattlesnake.process.streaming.save_rattlesnake_to_netcdf")
@mock.patch("rattlesnake.process.streaming.nc.Dataset")
def test_streaming_process_initialize_creates_file(
    mock_dataset,
    mock_save_rattlesnake_to_netcdf,
    stream_type,
    streaming,
    hardware_metadata,
    environment_metadata_dict,
    tmp_path,
):
    """
    Verifies that streaming initialization creates a netCDF dataset and saves
    metadata for enabled streaming modes.
    """
    stream_file = tmp_path / "stream.nc4"
    stream_metadata = StreamMetadata(
        stream_type=stream_type,
        stream_file=stream_file,
        test_level_environment_name="Environment 0",
    )
    data = (stream_metadata, hardware_metadata, environment_metadata_dict)

    streaming.stream_variable = "old_variable"
    streaming.stream_dimension = "old_dimension"
    streaming.stream_index = 10
    streaming.clear_ready()

    streaming.initialize(data)

    mock_dataset.assert_called_once_with(
        stream_file,
        "w",
        format="NETCDF4",
        clobber=True,
    )
    assert streaming.netcdf_handle is mock_dataset.return_value

    mock_save_rattlesnake_to_netcdf.assert_called_once_with(
        mock_dataset.return_value,
        hardware_metadata,
        environment_metadata_dict,
    )

    assert streaming.stream_variable == "time_data"
    assert streaming.stream_dimension == "time_samples"
    assert streaming.stream_index == 0
    assert streaming.ready_event.is_set()


@mock.patch("rattlesnake.process.streaming.save_rattlesnake_to_netcdf")
@mock.patch("rattlesnake.process.streaming.nc.Dataset")
def test_streaming_process_initialize_no_stream(
    mock_dataset,
    mock_save_rattlesnake_to_netcdf,
    streaming,
    hardware_metadata,
    environment_metadata_dict,
):
    """
    Verifies that ``NO_STREAM`` initialization does not create a file and marks
    the process ready.
    """
    stream_metadata = StreamMetadata(stream_type=StreamType.NO_STREAM)
    data = (stream_metadata, hardware_metadata, environment_metadata_dict)

    streaming.clear_ready()

    streaming.initialize(data)

    mock_dataset.assert_not_called()
    mock_save_rattlesnake_to_netcdf.assert_not_called()
    assert streaming.netcdf_handle is None
    assert streaming.ready_event.is_set()


def test_streaming_process_write_data(streaming):
    """
    Verifies that data is written to the active netCDF variable at the end of
    the current time dimension.
    """
    data = np.ones((2, 5))

    mock_dimension = mock.MagicMock()
    mock_dimension.size = 2

    mock_variable = mock.MagicMock()
    mock_dataset = mock.MagicMock()
    mock_dataset.dimensions = {"time_samples": mock_dimension}
    mock_dataset.variables = {"time_data": mock_variable}

    streaming.netcdf_handle = mock_dataset
    streaming.stream_dimension = "time_samples"
    streaming.stream_variable = "time_data"

    streaming.write_data(data)

    mock_variable.__setitem__.assert_called_once_with(
        (slice(None, None, None), slice(2, None, None)),
        data,
    )


def test_streaming_process_write_data_uses_current_stream(streaming):
    """
    Verifies that ``write_data`` uses the currently selected stream variable
    and dimension.
    """
    data = np.ones((2, 3))

    mock_dimension = mock.MagicMock()
    mock_dimension.size = 7

    mock_variable = mock.MagicMock()
    mock_dataset = mock.MagicMock()
    mock_dataset.dimensions = {"time_samples_1": mock_dimension}
    mock_dataset.variables = {"time_data_1": mock_variable}

    streaming.netcdf_handle = mock_dataset
    streaming.stream_dimension = "time_samples_1"
    streaming.stream_variable = "time_data_1"

    streaming.write_data(data)

    mock_variable.__setitem__.assert_called_once_with(
        (slice(None, None, None), slice(7, None, None)),
        data,
    )


def test_streaming_process_write_data_no_init(streaming):
    """
    Verifies that calling ``write_data`` without an open netCDF handle returns
    without error.
    """
    streaming.netcdf_handle = None

    result = streaming.write_data(np.ones((2, 5)))

    assert result is None


def test_streaming_process_create_new_stream(streaming):
    """
    Verifies that a new stream dimension and variable are created.
    """
    mock_dataset = mock.MagicMock()
    streaming.netcdf_handle = mock_dataset

    streaming.create_new_stream(None)

    assert streaming.stream_index == 1
    assert streaming.stream_dimension == "time_samples_1"
    assert streaming.stream_variable == "time_data_1"

    mock_dataset.createDimension.assert_called_once_with("time_samples_1", None)
    mock_dataset.createVariable.assert_called_once_with(
        "time_data_1",
        "f8",
        ("response_channels", "time_samples_1"),
    )


def test_streaming_process_create_multiple_new_streams(streaming):
    """
    Verifies that repeated new-stream requests increment stream names.
    """
    mock_dataset = mock.MagicMock()
    streaming.netcdf_handle = mock_dataset

    streaming.create_new_stream(None)
    streaming.create_new_stream(None)

    assert streaming.stream_index == 2
    assert streaming.stream_dimension == "time_samples_2"
    assert streaming.stream_variable == "time_data_2"

    assert mock_dataset.createDimension.call_args_list == [
        mock.call("time_samples_1", None),
        mock.call("time_samples_2", None),
    ]
    assert mock_dataset.createVariable.call_args_list == [
        mock.call("time_data_1", "f8", ("response_channels", "time_samples_1")),
        mock.call("time_data_2", "f8", ("response_channels", "time_samples_2")),
    ]


def test_streaming_process_create_new_stream_no_netcdf(streaming):
    """
    Verifies that calling ``create_new_stream`` without an open netCDF handle
    returns without error.
    """
    streaming.netcdf_handle = None

    result = streaming.create_new_stream(None)

    assert result is None
    assert streaming.stream_index == 0
    assert streaming.stream_variable == "time_data"
    assert streaming.stream_dimension == "time_samples"


def test_streaming_process_finalize(streaming):
    """
    Verifies that an open netCDF handle is closed and cleared.
    """
    mock_dataset = mock.MagicMock()
    streaming.netcdf_handle = mock_dataset

    streaming.finalize(None)

    mock_dataset.close.assert_called_once()
    assert streaming.netcdf_handle is None


def test_streaming_process_finalize_no_netcdf(streaming):
    """
    Verifies that finalizing without an open netCDF handle returns without
    error.
    """
    streaming.netcdf_handle = None

    result = streaming.finalize(None)

    assert result is None
    assert streaming.netcdf_handle is None


@mock.patch("rattlesnake.process.streaming.StreamingProcess.finalize")
def test_streaming_process_quit(mock_finalize, streaming):
    """
    Verifies that ``quit`` finalizes the stream and returns ``True``.
    """
    result = streaming.quit(None)

    mock_finalize.assert_called_once_with(None)
    assert result is True


# endregion


# region streaming_process
@pytest.mark.parametrize("use_thread", [True, False])
@mock.patch("rattlesnake.process.streaming.StreamingProcess")
def test_streaming_process_func(mock_streaming_process_class, use_thread):
    """
    Verifies that ``streaming_process`` constructs a ``StreamingProcess`` and
    calls its ``run`` method.
    """
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)

    streaming_process(
        queue_container,
        event_container.streaming_ready_event,
        event_container.streaming_close_event,
    )

    mock_streaming_process_class.assert_called_once_with(
        "Streaming",
        queue_container,
        event_container.streaming_ready_event,
    )

    mock_instance = mock_streaming_process_class.return_value
    mock_instance.run.assert_called_once_with(event_container.streaming_close_event)


# endregion
