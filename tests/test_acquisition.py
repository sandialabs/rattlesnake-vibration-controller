from rattlesnake.environment.environment_utilities import ControlTypes
from rattlesnake.utilities import (
    VerboseMessageQueue,
    QueueContainer,
    DataAcquisitionParameters,
    Channel,
    GlobalCommands,
)
from rattlesnake.process.acquisition import AcquisitionProcess, acquisition_process
from rattlesnake.hardware.abstract_hardware import HardwareAcquisition
from rattlesnake.user_interface.ui_utilities import UICommands
from functions.common_functions import create_hardware_dict_acquisition
from functions.acquisition_functions import create_acquire_log_calls
from unittest import mock
import multiprocessing as mp
import pytest
import numpy as np


# Create log_file_queue
@pytest.fixture
def log_file_queue():
    return mp.Queue()


@pytest.fixture
def hardware_dict():
    return create_hardware_dict_acquisition()


# Create modal environment type
@pytest.fixture
def environments():
    control_type = ControlTypes(6)
    return [[control_type, control_type.name.title()]]


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
def acquisition_process_obj(queue_container, environments):
    acquisition_process = AcquisitionProcess(
        "Process Name", queue_container, environments, mp.Value("i", 0)
    )
    return acquisition_process


@pytest.fixture
def environment_channels():
    return [[True]]


@pytest.fixture
def channel_list():
    return [
        Channel.from_channel_table_row(
            [
                "221",
                "Y+",
                "",
                "19644",
                "X+",
                "",
                "",
                "",
                "",
                "",
                "Virtual",
                "",
                "Accel",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        ),
        Channel.from_channel_table_row(
            [
                "221",
                "Y+",
                "",
                "19644",
                "X+",
                "",
                "",
                "",
                "",
                "",
                "Virtual",
                "",
                "Force",
                "",
                "",
                "",
                "",
                "",
                "Phys_dev",
                "",
                "5",
                "10",
            ]
        ),
    ]


@pytest.fixture()
def data_acquisition_parameters(channel_list):
    sample_rate = 2000
    time_per_read = 0.25
    time_per_write = 0.25
    output_oversample = 2
    hardware_selector_idx = 6
    hardware_file = "ExampleFile.nc4"
    environments = ["Modal"]
    environment_booleans = [[True]]
    acquisition_processes = 1
    task_trigger = 0
    task_trigger_output_channel = ""
    data_acquisition_parameters = DataAcquisitionParameters(
        channel_list,
        sample_rate,
        round(sample_rate * time_per_read),
        round(sample_rate * time_per_write * output_oversample),
        hardware_selector_idx,
        hardware_file,
        environments,
        environment_booleans,
        output_oversample,
        maximum_acquisition_processes=acquisition_processes,
        task_trigger=task_trigger,
        task_trigger_output_channel=task_trigger_output_channel,
    )

    return data_acquisition_parameters


# Test AcquisitonProcess initialization
def test_acquisition_process_init(queue_container, environments):
    acquisition_process = AcquisitionProcess(
        "Process Name", queue_container, environments, mp.Value("i", 0)
    )

    # Make sure it is the correct class
    assert isinstance(acquisition_process, AcquisitionProcess)
    # Test the acquisition_active property
    assert acquisition_process.acquisition_active == False
    # Test the acquisiton_active setter
    acquisition_process.acquisition_active = True
    assert acquisition_process.acquisition_active == True


@pytest.mark.parametrize("hardware", [None, mock.MagicMock()])
@pytest.mark.parametrize("hardware_idx", [0, 1, 2, 4, 5])
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_process_initialize_data_acquisition(
    mock_log,
    hardware_dict,
    hardware,
    hardware_idx,
    data_acquisition_parameters,
    environment_channels,
    acquisition_process_obj,
):
    data_acquisition_parameters.hardware = hardware_idx

    with mock.patch(hardware_dict[hardware_idx]) as mock_hardware:
        acquisition_process_obj.hardware = hardware
        acquisition_process_obj.initialize_data_acquisition(
            (data_acquisition_parameters, environment_channels)
        )

        # Test if hardware setup was called
        mock_hardware().set_up_data_acquisition_parameters_and_channels.assert_called_with(
            data_acquisition_parameters, data_acquisition_parameters.channel_list
        )

    # Test if log message was stored
    mock_log.assert_called_with("Initializing Data Acquisition")
    # Test if output indices were stored
    assert acquisition_process_obj.output_indices[0] == 1
    # Test if warning limit was stored
    np.testing.assert_array_almost_equal(acquisition_process_obj.warning_limits, [float("inf"), 5])
    # Test if abort limit was stored
    np.testing.assert_array_almost_equal(acquisition_process_obj.abort_limits, [float("inf"), 10])
    # Test if data array was initialized
    np.testing.assert_array_almost_equal(acquisition_process_obj.read_data, np.zeros((2, 2000)))


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_process_stop_environment(mock_log, acquisition_process_obj):
    data = "Modal"

    mock_hardware = mock.MagicMock()
    acquisition_process_obj.hardware = mock_hardware
    acquisition_process_obj.stop_environment(data)

    mock_log.assert_called_with("Deactivating Environment {:}".format(data))
    mock_hardware.get_acquisition_delay.assert_called()


@pytest.mark.parametrize("prev_streamed", [True, False])
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
def test_acqusition_process_start_streaming(mock_put, prev_streamed, acquisition_process_obj):
    acquisition_process_obj.has_streamed = prev_streamed
    acquisition_process_obj.start_streaming(None)
    if prev_streamed:
        mock_put.assert_called_with("Process Name", (GlobalCommands.CREATE_NEW_STREAM, None))
    assert acquisition_process_obj.streaming == True
    assert acquisition_process_obj.has_streamed == True


def test_acquisition_process_stop_streaming(acquisition_process_obj):
    acquisition_process_obj.stop_streaming(None)

    assert acquisition_process_obj.streaming == False


@mock.patch("rattlesnake.process.acquisition.align_signals")
@mock.patch("rattlesnake.process.acquisition.AcquisitionProcess.add_data_to_buffer")
@mock.patch("rattlesnake.process.acquisition.AcquisitionProcess.get_first_output_data")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
@mock.patch("rattlesnake.process.acquisition.mp.queues.Queue.put")
@mock.patch("rattlesnake.process.acquisition.mp.queues.Queue.get_nowait")
@mock.patch("rattlesnake.process.acquisition.time")
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_acquire_signal(
    mock_log,
    mock_time,
    mock_get,
    mock_put,
    mock_vput,
    mock_first,
    mock_add,
    mock_align,
    acquisition_process_obj,
):
    mock_get.side_effect = [("Modal", None), (None, None)]
    mock_time.side_effect = [0, 10, 20]
    mock_hardware = mock.MagicMock()
    mock_hardware.read.return_value = np.ones((2, 100))
    acquisition_process_obj.hardware = mock_hardware
    acquisition_process_obj.shutdown_flag = False
    acquisition_process_obj.warning_limits = [10, 10]
    acquisition_process_obj.abort_limits = [10, 10]
    mock_environment = mock.MagicMock()
    mock_environment.__getitem__.return_value = 1
    acquisition_process_obj.environment_first_data = mock_environment
    acquisition_process_obj.environment_active_flags["Modal"] = True
    acquisition_process_obj.read_data = np.zeros((2, 100))
    mock_align.return_value = (None, 2, None, None)
    acquisition_process_obj.environment_acquisition_channels = {"Modal": [0, 1]}
    acquisition_process_obj.streaming = True

    acquisition_process_obj.acquire_signal(None)

    log_calls = create_acquire_log_calls()
    mock_log.assert_has_calls(log_calls)
    assert mock_put.call_args_list[0][0][0][0] == UICommands.MONITOR
    np.testing.assert_array_equal(mock_put.call_args_list[1][0][0][0], np.zeros((2, 98)))
    vput_calls = [mock.call("Process Name", (GlobalCommands.RUN_HARDWARE, None))]
    mock_vput.assert_has_calls(vput_calls)
    assert mock_vput.call_args_list[1][0][1][0] == GlobalCommands.STREAMING_DATA
    np.testing.assert_array_equal(mock_vput.call_args_list[1][0][1][1], np.ones((2, 100)))
    np.testing.assert_array_equal(mock_add.call_args_list[0][0], np.ones((1, 2, 100)))


def test_add_data_to_buffer(acquisition_process_obj):
    data = np.zeros((1, 100))
    acquisition_process_obj.read_data = np.zeros((1, 100))
    acquisition_process_obj.add_data_to_buffer(data)

    np.testing.assert_array_equal(acquisition_process_obj.read_data, data)


@mock.patch("rattlesnake.process.acquisition.flush_queue")
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_process_get_first_output_data(
    mock_log, mock_flush, queue_container, acquisition_process_obj
):
    mock_flush.return_value = [("Modal", "Data")]

    acquisition_process_obj.get_first_output_data()

    mock_flush.assert_called_with(queue_container.input_output_sync_queue)
    mock_log.assert_called_with("Listening for first data for environment Modal")
    assert acquisition_process_obj.environment_first_data["Modal"] == "Data"
    assert acquisition_process_obj.any_environments_started == True


def test_acquisition_process_stop_acquisition(acquisition_process_obj):
    acquisition_process_obj.stop_acquisition(None)

    assert acquisition_process_obj.shutdown_flag == True


@mock.patch("rattlesnake.process.acquisition.flush_queue")
@mock.patch("rattlesnake.process.acquisition.AcquisitionProcess.log")
def test_acquisition_process_quit(mock_log, mock_flush, acquisition_process_obj):
    mock_hardware = mock.MagicMock()
    acquisition_process_obj.hardware = mock_hardware

    acquisition_process_obj.quit(None)

    mock_log.assert_called_with("Flushed 0 items out of queues")
    mock_hardware.close.assert_called()


# Test the acquisition_process function
# Prevent the run while loop from starting
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.run")
def test_acquisition_process_func(mock_run, queue_container, environments):
    acquisition_process(queue_container, environments, mp.Value("i", 0))

    # Test that the run function was called
    mock_run.assert_called()


if __name__ == "__main__":
    log_file_queue = mp.Queue()

    environments = [[ControlTypes(6), ControlTypes(6).name.title()]]

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

    acquisition_process = AcquisitionProcess(
        "Process Name", queue_container, environments, mp.Value("i", 0)
    )

    environment_channels = [[True]]

    channel_list = [
        Channel.from_channel_table_row(
            [
                "221",
                "Y+",
                "",
                "19644",
                "X+",
                "",
                "",
                "",
                "",
                "",
                "Virtual",
                "",
                "Accel",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        ),
        Channel.from_channel_table_row(
            [
                "221",
                "Y+",
                "",
                "19644",
                "X+",
                "",
                "",
                "",
                "",
                "",
                "Virtual",
                "",
                "Force",
                "",
                "",
                "",
                "",
                "",
                "Phys_dev",
                "",
                "5",
                "10",
            ]
        ),
    ]
    sample_rate = 2000
    time_per_read = 0.25
    time_per_write = 0.25
    output_oversample = 2
    hardware_selector_idx = 6
    hardware_file = "ExampleFile.nc4"
    environments = ["Modal"]
    environment_booleans = [[True]]
    acquisition_processes = 1
    data_acquisition_parameters = DataAcquisitionParameters(
        channel_list,
        sample_rate,
        round(sample_rate * time_per_read),
        round(sample_rate * time_per_write * output_oversample),
        hardware_selector_idx,
        hardware_file,
        environments,
        environment_booleans,
        output_oversample,
        acquisition_processes,
    )

    # test_acquisition_stop_environment(prev_streamed = True, acquisition_process_obj = acquisition_process)
    # test_add_data_to_buffer(acquisition_process_obj=acquisition_process)
    # test_get_first_output_data(queue_container=queue_container,acquisition_process_obj=acquisition_process)
    test_acquisition_acquire_signal(acquisition_process_obj=acquisition_process)
