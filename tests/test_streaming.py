from rattlesnake.process.streaming import StreamingProcess, streaming_process
from rattlesnake.utilities import (
    VerboseMessageQueue,
    QueueContainer,
    Channel,
    DataAcquisitionParameters,
)
from rattlesnake.environment.environment_utilities import ControlTypes
from functions.common_functions import create_data_acquisition_parameters
from unittest import mock
import multiprocessing as mp
import pytest
import numpy as np


# Create log_file_queue
@pytest.fixture
def log_file_queue():
    return mp.Queue()


# Create queue_container
@pytest.fixture
def queue_container(log_file_queue):
    queue_container = QueueContainer(
        VerboseMessageQueue(log_file_queue, "Controller Communication Queue"),
        VerboseMessageQueue(log_file_queue, "Acquisition Command Queue"),
        VerboseMessageQueue(log_file_queue, "Output Command Queue"),
        VerboseMessageQueue(log_file_queue, "Streaming Command Queue"),
        log_file_queue,
        mp.Queue(),
        mp.Queue(),
        mp.Queue(),
        {"Modal": VerboseMessageQueue(log_file_queue, "Environment Command Queue")},
        {"Modal": mp.Queue()},
        {"Modal": mp.Queue()},
    )
    return queue_container


@pytest.fixture
def streaming_process_obj(queue_container):
    streaming_process_obj = StreamingProcess("Process Name", queue_container)
    return streaming_process_obj


@pytest.fixture()
def data_acquisition_parameters():
    return create_data_acquisition_parameters()


# Test StreamingProcess intialization
def test_streaming_process_init(queue_container):
    streaming_process = StreamingProcess("Process Name", queue_container)

    # Test if object is the correct class
    assert isinstance(streaming_process, StreamingProcess)


@mock.patch("rattlesnake.process.streaming.nc.Dataset")
def test_streaming_process_initialize(
    mock_dataset, streaming_process_obj, data_acquisition_parameters
):
    mock_metadata = mock.MagicMock()
    mock_dataset().createGroup.return_value = "Group Handle"
    data = ("Filename", data_acquisition_parameters, {"Environment Name": mock_metadata})

    streaming_process_obj.initialize(data)

    dimension_calls = [
        mock.call("response_channels", 2),
        mock.call("output_channels", 1),
        mock.call("time_samples", None),
        mock.call("num_environments", 1),
    ]
    mock_dataset().createDimension.assert_has_calls(dimension_calls)
    assert streaming_process_obj.netcdf_handle.sample_rate == 2000
    mock_metadata.store_to_netcdf.assert_called_with("Group Handle")


def test_streaming_process_write_data(streaming_process_obj):
    data = "data"
    mock_dataset = mock.MagicMock()
    mock_dataset.dimensions = {"time_samples": np.array([0, 0])}
    streaming_process_obj.netcdf_handle = mock_dataset

    streaming_process_obj.write_data(data)

    mock_dataset.variables["time_data"].__setitem__.assert_called_with(
        (slice(None, None, None), slice(2, None, None)), data
    )


def test_streaming_process_create_new_stream(streaming_process_obj):
    mock_dataset = mock.MagicMock()
    streaming_process_obj.netcdf_handle = mock_dataset

    streaming_process_obj.create_new_stream(None)

    mock_dataset.createDimension.assert_called_with("time_samples_1", None)
    mock_dataset.createVariable.assert_called_with(
        "time_data_1", "f8", ("response_channels", "time_samples_1")
    )


def test_streaming_process_finalize(streaming_process_obj):
    mock_dataset = mock.MagicMock()
    streaming_process_obj.netcdf_handle = mock_dataset

    streaming_process_obj.finalize(None)

    mock_dataset.close.assert_called
    assert streaming_process_obj.netcdf_handle == None


@mock.patch("rattlesnake.process.streaming.StreamingProcess.finalize")
def test_streaming_process_quit(mock_finalize, streaming_process_obj):
    quit_var = streaming_process_obj.quit(None)

    mock_finalize.assert_called()
    assert quit_var == True


# Test streaming_process function
# Prevent run while loop from starting
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.run")
def test_streaming_process_func(mock_run, queue_container):
    streaming_process(queue_container)

    # Test if the run function was called
    mock_run.assert_called()


if __name__ == "__main__":
    log_file_queue = mp.Queue()

    queue_container = QueueContainer(
        VerboseMessageQueue(log_file_queue, "Controller Communication Queue"),
        VerboseMessageQueue(log_file_queue, "Acquisition Command Queue"),
        VerboseMessageQueue(log_file_queue, "Output Command Queue"),
        VerboseMessageQueue(log_file_queue, "Streaming Command Queue"),
        log_file_queue,
        mp.Queue(),
        mp.Queue(),
        mp.Queue(),
        {"Modal": VerboseMessageQueue(log_file_queue, "Environment Command Queue")},
        {"Modal": mp.Queue()},
        {"Modal": mp.Queue()},
    )

    streaming_process_obj = StreamingProcess("Process Name", queue_container)

    data_acquisition_parameters = create_data_acquisition_parameters()

    # test_streaming_process_initialize(streaming_process_obj = streaming_process_obj, data_acquisition_parameters = data_acquisition_parameters)
    test_streaming_process_create_new_stream(streaming_process_obj=streaming_process_obj)
