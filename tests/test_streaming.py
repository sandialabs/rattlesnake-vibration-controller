from rattlesnake.process.streaming import StreamType, StreamMetadata, StreamingProcess, streaming_process
from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from mock_objects.mock_hardware import MockHardwareMetadata
from mock_objects.mock_environment import MockEnvironmentMetadata
from mock_objects.mock_utilities import mock_queue_container, mock_event_container
import pytest
import numpy as np
import multiprocessing as mp
from unittest import mock


# region: Fixtures
@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def streaming(request):
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    streaming_process = StreamingProcess("Process Name", queue_container, event_container.streaming_ready_event)
    return streaming_process


# region: StreamMetadata
def test_stream_metadata_init():
    stream_metadata = StreamMetadata()

    assert isinstance(stream_metadata, StreamMetadata)
    assert hasattr(stream_metadata, "stream_type")
    assert hasattr(stream_metadata, "stream_file")
    assert hasattr(stream_metadata, "test_level_environment_name")


@pytest.mark.parametrize(
    "stream_type, stream_file, test_level, path_exists, expected",
    [
        (StreamType.MANUAL, "filepath", None, True, True),
        (StreamType.MANUAL, None, None, True, ValueError),
        (StreamType.MANUAL, "filepath", None, False, ValueError),
        (StreamType.NO_STREAM, None, None, False, True),
        (StreamType.IMMEDIATELY, "filepath", None, True, True),
        (StreamType.TEST_LEVEL, "filepath", "Environment 0", True, True),
        (StreamType.TEST_LEVEL, "filepath", None, True, ValueError),
    ],
)
@mock.patch("rattlesnake.process.streaming.Path")
def test_stream_metadata_validate(mock_path, stream_type, stream_file, test_level, path_exists, expected):
    stream_metadata = StreamMetadata()
    stream_metadata.stream_type = stream_type
    stream_metadata.stream_file = stream_file
    stream_metadata.test_level_environment_name = test_level

    mock_path.return_value.parent.exists.return_value = path_exists

    if expected is ValueError:
        with pytest.raises(ValueError):
            stream_metadata.validate()
    elif expected is True:
        assert stream_metadata.validate()
    else:
        assert False


# region: StreamingProcess
# Test StreamingProcess intialization
@pytest.mark.parametrize("use_thread", [True, False])
def test_streaming_init(use_thread):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    streaming_process = StreamingProcess("Process Name", queue_container, event_container.streaming_ready_event)

    # Test if object is the correct class
    assert isinstance(streaming_process, StreamingProcess)
    assert isinstance(streaming_process, AbstractMessageProcess)


@pytest.mark.parametrize(
    "stream_type",
    [
        StreamType.NO_STREAM,
        StreamType.IMMEDIATELY,
        StreamType.PROFILE_INSTRUCTION,
        StreamType.TEST_LEVEL,
        StreamType.MANUAL,
    ],
)
@mock.patch("rattlesnake.process.streaming.nc.Dataset")
def test_streaming_process_initialize(mock_dataset, stream_type, streaming):
    mock_dataset.return_value.createGroup.return_value = "Group Handle"
    stream_metadata = StreamMetadata()
    stream_metadata.stream_file = "filename"
    stream_metadata.stream_type = stream_type
    hardware_metadata = MockHardwareMetadata()
    environment_metadata = MockEnvironmentMetadata()
    mock_store = mock.MagicMock()
    environment_metadata.store_to_netcdf = mock_store
    environment_metadata_dict = {"Environment 0": environment_metadata}
    data = (stream_metadata, hardware_metadata, environment_metadata_dict)

    streaming.clear_ready()
    streaming.initialize(data)

    assert streaming.ready_event.is_set()
    if stream_type == StreamType.NO_STREAM:
        mock_dataset.return_value.createDimension.assert_not_called()
    else:
        dimension_calls = [
            mock.call("response_channels", 2),
            mock.call("output_channels", 1),
            mock.call("time_samples", None),
            mock.call("num_environments", 1),
        ]
        mock_dataset.return_value.createDimension.assert_has_calls(dimension_calls)
        assert streaming.netcdf_handle.sample_rate == 1000
        mock_store.assert_called_once_with("Group Handle")


def test_streaming_process_write_data(streaming):
    data = "data"
    mock_dataset = mock.MagicMock()
    mock_dataset.dimensions = {"time_samples": np.array([0, 0])}
    streaming.netcdf_handle = mock_dataset

    streaming.write_data(data)

    mock_dataset.variables["time_data"].__setitem__.assert_called_with((slice(None, None, None), slice(2, None, None)), data)


def test_streaming_process_create_new_stream(streaming):
    mock_dataset = mock.MagicMock()
    streaming.netcdf_handle = mock_dataset

    streaming.create_new_stream(None)

    mock_dataset.createDimension.assert_called_with("time_samples_1", None)
    mock_dataset.createVariable.assert_called_with("time_data_1", "f8", ("response_channels", "time_samples_1"))


def test_streaming_process_create_new_stream_no_netcdf(streaming):
    streaming.netcdf_handle = None

    streaming.create_new_stream(None)

    assert True


def test_streaming_process_finalize(streaming):
    mock_dataset = mock.MagicMock()
    streaming.netcdf_handle = mock_dataset

    streaming.finalize(None)

    mock_dataset.close.assert_called
    assert streaming.netcdf_handle == None


def test_streaming_process_write_data(streaming):
    data = "data"
    mock_dataset = mock.MagicMock()
    mock_dataset.dimensions = {"time_samples": np.array([0, 0])}
    streaming.netcdf_handle = mock_dataset

    streaming.write_data(data)

    mock_dataset.variables["time_data"].__setitem__.assert_called_with((slice(None, None, None), slice(2, None, None)), data)


def test_streaming_process_write_data_no_init(streaming):
    data = "data"
    streaming.netcdf_handle = None
    result = streaming.write_data(data)

    assert result is None


def test_streaming_process_create_new_stream(streaming):
    mock_dataset = mock.MagicMock()
    streaming.netcdf_handle = mock_dataset

    streaming.create_new_stream(None)

    mock_dataset.createDimension.assert_called_with("time_samples_1", None)
    mock_dataset.createVariable.assert_called_with("time_data_1", "f8", ("response_channels", "time_samples_1"))


def test_streaming_process_finalize(streaming):
    mock_dataset = mock.MagicMock()
    streaming.netcdf_handle = mock_dataset

    streaming.finalize(None)

    mock_dataset.close.assert_called
    assert streaming.netcdf_handle == None


@mock.patch("rattlesnake.process.streaming.StreamingProcess.finalize")
def test_streaming_process_quit(mock_finalize, streaming):
    quit_var = streaming.quit(None)

    mock_finalize.assert_called()
    assert quit_var == True


# region: streaming_process
@pytest.mark.parametrize("use_thread", [True, False])
@mock.patch("rattlesnake.process.streaming.StreamingProcess")
def test_output_process_func(mock_stream, use_thread):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    streaming_process(queue_container, event_container.streaming_ready_event, event_container.streaming_close_event)

    mock_instance = mock_stream.return_value
    mock_instance.run.assert_called()
