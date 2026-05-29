from rattlesnake.process.acquisition import AcquisitionProcess, acquisition_process
from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.user_interface.ui_utilities import UICommands
from rattlesnake.utilities import GlobalCommands
from mock_objects.mock_hardware import MockHardwareMetadata, acquisition_dict, metadata_attr_dict, IMPLEMENTED_HARDWARE
from mock_objects.mock_environment import MockEnvironmentMetadata
from mock_objects.mock_utilities import mock_queue_container, mock_event_container
import pytest
import multiprocessing as mp
import numpy as np
from unittest import mock


# region: Fixtures
@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def acquisition(request):
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    acquisition = AcquisitionProcess(
        "Process Name",
        queue_container,
        event_container.acquisition_active_event,
        event_container.streaming_active_event,
        event_container.acquisition_ready_event,
    )
    return acquisition


# region: AcquisitionProcess
# Test AcquisitonProcess initialization
@pytest.mark.parametrize("use_thread", [True, False])
def test_acquisition_init(use_thread):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    acquisition = AcquisitionProcess(
        "Process Name",
        queue_container,
        event_container.acquisition_active_event,
        event_container.streaming_active_event,
        event_container.acquisition_ready_event,
    )

    # Make sure it is the correct class
    assert isinstance(acquisition, AcquisitionProcess)
    assert isinstance(acquisition, AbstractMessageProcess)


def test_acquisition_properties(acquisition):
    # Test the acquisition_active property
    assert acquisition.acquisition_active == False
    # Test the acquisiton_active setter
    acquisition.set_active()
    assert acquisition.acquisition_active == True
    acquisition.clear_active()
    assert acquisition.acquisition_active == False


@pytest.mark.parametrize("hardware_type", [*IMPLEMENTED_HARDWARE])
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_process_initialize_hardware(mock_log, hardware_type, acquisition):
    hardware_metadata = MockHardwareMetadata()
    hardware_lookup = acquisition_dict()
    attr_lookup = metadata_attr_dict()
    hardware_metadata.hardware_type = hardware_type
    mock_existing_hardware = mock.MagicMock()
    acquisition.hardware = mock_existing_hardware

    with mock.patch(hardware_lookup[hardware_type]) as mock_hardware:
        for attr in attr_lookup[hardware_type]:
            setattr(hardware_metadata, attr, 0)

        acquisition.clear_ready()
        acquisition.initialize_hardware(hardware_metadata)
        mock_hardware().initialize_hardware.assert_called()

    mock_log.assert_called_with("Initializing Hardware")
    # Test if output indices were stored
    assert acquisition.output_indices[0] == 1
    # Test if warning limit was stored
    np.testing.assert_array_almost_equal(acquisition.warning_limits, [float("inf"), float("inf")])
    # Test if abort limit was stored
    np.testing.assert_array_almost_equal(acquisition.abort_limits, [float("inf"), float("inf")])
    # Test if data array was initialized
    np.testing.assert_array_almost_equal(acquisition.read_data, np.zeros((2, 1000)))
    # Test if acquisition was set to ready
    assert acquisition.ready_event.is_set()
    mock_existing_hardware.close.assert_called_once()


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_process_initialize_hardware_value_error(mock_log, acquisition):
    hardware_metadata = MockHardwareMetadata()

    with pytest.raises(TypeError):
        acquisition.initialize_hardware(hardware_metadata)


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_process_initialize_environment(mock_log, acquisition):
    hardware_metadata = MockHardwareMetadata()
    acquisition.hardware_metadata = hardware_metadata
    environment_metadata = MockEnvironmentMetadata()
    acquisition.clear_ready()
    acquisition.initialize_environment({"Environment 0": environment_metadata})

    mock_log.assert_called_with("Initializing Environment")
    assert acquisition.environment_list == ["Environment 0"]
    assert acquisition.environment_acquisition_channels["Environment 0"] == [0, 1]
    assert acquisition.environment_active_flags["Environment 0"] == False
    assert acquisition.environment_last_data["Environment 0"] == False
    assert acquisition.environment_samples_remaining_to_read["Environment 0"] == 0
    assert acquisition.environment_first_data["Environment 0"] == None
    assert acquisition.ready_event.is_set()


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_process_stop_environment(mock_log, acquisition):
    hardware_metadata = MockHardwareMetadata()
    acquisition.hardware_metadata = hardware_metadata
    acquisition.environment_list = ["Environment 0"]
    acquisition.environment_acquisition_channels["Environment 0"] = [0, 1]
    acquisition.environment_active_flags["Environment 0"] = False
    acquisition.environment_last_data["Environment 0"] = False
    acquisition.environment_samples_remaining_to_read["Environment 0"] = 0
    acquisition.environment_first_data["Environment 0"] = None

    mock_hardware = mock.MagicMock()
    acquisition.hardware = mock_hardware
    acquisition.stop_environment("Environment 0")

    mock_log.assert_called_with("Deactivating Environment Environment 0")
    mock_hardware.get_acquisition_delay.assert_called()


@pytest.mark.parametrize("prev_streamed", [True, False])
def test_acqusition_process_start_streaming(prev_streamed, acquisition):
    mock_stream_queue = mock.MagicMock()
    acquisition.queue_container.streaming_command_queue = mock_stream_queue
    acquisition.has_streamed = prev_streamed
    acquisition.start_streaming(None)
    if prev_streamed:
        mock_stream_queue.put.assert_called_with("Process Name", (GlobalCommands.CREATE_NEW_STREAM, None))
    assert acquisition.streaming == True
    assert acquisition.has_streamed == True


def test_acquisition_process_stop_streaming(acquisition):
    acquisition.stop_streaming(None)

    assert acquisition.streaming == False


@mock.patch("rattlesnake.process.acquisition.align_signals")
@mock.patch("rattlesnake.process.acquisition.AcquisitionProcess.add_data_to_buffer")
@mock.patch("rattlesnake.process.acquisition.AcquisitionProcess.get_first_output_data")
@mock.patch("rattlesnake.process.acquisition.time")
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_acquire_signal(
    mock_log,
    mock_time,
    mock_first,
    mock_add,
    mock_align,
    acquisition,
):
    hardware_metadata = MockHardwareMetadata()
    acquisition.hardware_metadata = hardware_metadata
    acquisition.environment_list = ["Environment 0"]
    acquisition.environment_acquisition_channels["Environment 0"] = [0, 1]
    acquisition.environment_active_flags["Environment 0"] = False
    acquisition.environment_last_data["Environment 0"] = False
    acquisition.environment_samples_remaining_to_read["Environment 0"] = 0
    acquisition.environment_first_data["Environment 0"] = None

    mock_input_output_queue = mock.MagicMock()
    mock_input_output_queue.get_nowait.side_effect = [("Environment 0", None), (None, None)]
    mock_gui_queue_put = mock.MagicMock()
    mock_acquisition_queue = mock.MagicMock()
    mock_streaming_queue_put = mock.MagicMock()
    mock_data_in_queue = mock.MagicMock()

    acquisition.queue_container.input_output_sync_queue = mock_input_output_queue
    acquisition.queue_container.gui_update_queue.put = mock_gui_queue_put
    acquisition.queue_container.acquisition_command_queue = mock_acquisition_queue
    acquisition.queue_container.streaming_command_queue.put = mock_streaming_queue_put
    acquisition.queue_container.environment_data_in_queues["Environment 0"] = mock_data_in_queue

    mock_time.side_effect = [0, 10, 20]
    mock_hardware = mock.MagicMock()
    mock_hardware.read.return_value = np.ones((2, 100))
    acquisition.hardware = mock_hardware
    acquisition.shutdown_flag = False
    acquisition.warning_limits = [10, 10]
    acquisition.abort_limits = [10, 10]
    mock_environment = mock.MagicMock()
    mock_environment.__getitem__.return_value = 1
    acquisition.environment_first_data = mock_environment
    acquisition.environment_active_flags["Environment 0"] = True
    acquisition.read_data = np.zeros((2, 100))
    mock_align.return_value = (None, 2, None, None)
    acquisition.environment_acquisition_channels = {"Environment 0": [0, 1]}
    acquisition.streaming = True

    acquisition.acquire_signal(None)

    log_calls = [
        mock.call("Waiting for Output to Start"),
        mock.call("Listening for first data for environment Environment 0"),
        mock.call("Detected Output Started"),
        mock.call("Starting Hardware Acquisition"),
        mock.call("Acquiring Data for ['Environment 0'] environments"),
        mock.call("Correlation check for environment Environment 0 took 10.00 seconds"),
        mock.call("Found First Data for Environment Environment 0"),
        mock.call("Sending (2, 98) data to Environment 0 environment"),
    ]
    mock_log.assert_has_calls(log_calls)
    assert mock_gui_queue_put.call_args_list[0][0][0][0] == UICommands.MONITOR
    np.testing.assert_array_equal(mock_gui_queue_put.call_args_list[0][0][0][1], np.ones((2,)))
    acquisition_calls = [mock.call("Process Name", (GlobalCommands.RUN_HARDWARE, None))]
    mock_acquisition_queue.put.assert_has_calls(acquisition_calls)
    assert mock_streaming_queue_put.call_args_list[0][0][1][0] == GlobalCommands.STREAMING_DATA
    np.testing.assert_array_equal(mock_streaming_queue_put.call_args_list[0][0][1][1], np.ones((2, 100)))
    np.testing.assert_array_equal(mock_add.call_args_list[0][0], np.ones((1, 2, 100)))


def test_add_data_to_buffer(acquisition):
    data = np.zeros((1, 100))
    acquisition.read_data = np.zeros((1, 100))
    acquisition.add_data_to_buffer(data)

    np.testing.assert_array_equal(acquisition.read_data, data)


@mock.patch("rattlesnake.process.acquisition.flush_queue")
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_process_get_first_output_data(mock_log, mock_flush, acquisition):
    acquisition.environment_list = ["Environment 0"]
    acquisition.environment_acquisition_channels["Environment 0"] = [0, 1]
    acquisition.environment_active_flags["Environment 0"] = False
    acquisition.environment_last_data["Environment 0"] = False
    acquisition.environment_samples_remaining_to_read["Environment 0"] = 0
    acquisition.environment_first_data["Environment 0"] = None
    mock_flush.return_value = [("Environment 0", "Data")]

    acquisition.get_first_output_data()

    mock_flush.assert_called_with(acquisition.queue_container.input_output_sync_queue)
    mock_log.assert_called_with("Listening for first data for environment Environment 0")
    assert acquisition.environment_first_data["Environment 0"] == "Data"
    assert acquisition.any_environments_started == True


def test_acquisition_process_stop_acquisition(acquisition):
    acquisition.stop_acquisition(None)

    assert acquisition.shutdown_flag == True


@mock.patch("rattlesnake.process.acquisition.flush_queue")
@mock.patch("rattlesnake.process.acquisition.AcquisitionProcess.log")
def test_acquisition_process_quit(mock_log, mock_flush, acquisition):
    mock_hardware = mock.MagicMock()
    acquisition.hardware = mock_hardware

    acquisition.quit(None)

    mock_log.assert_called_with("Flushed 0 items out of queues")
    mock_hardware.close.assert_called()


# region: acquisition_process
# Prevent the run while loop from starting
@pytest.mark.parametrize("use_thread", [True, False])
@mock.patch("rattlesnake.process.acquisition.AcquisitionProcess")
def test_acquisition_process_func(mock_acquisition, use_thread):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    acquisition_active = mp.Value("i", 0)
    acquisition_process(
        queue_container,
        acquisition_active,
        event_container.acquisition_ready_event,
        event_container.streaming_active_event,
        event_container.acquisition_close_event,
    )

    mock_instance = mock_acquisition.return_value
    mock_instance.run.assert_called()
