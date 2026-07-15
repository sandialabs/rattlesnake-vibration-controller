import multiprocessing as mp
import threading
from unittest import mock

import numpy as np
import pytest

from rattlesnake.environment.abstract_environment import EnvironmentMetadata
from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.process.acquisition import AcquisitionProcess, acquisition_process
from rattlesnake.testing.mock_utilities import (
    skeleton_environment_metadata,
    skeleton_hardware_metadata,
    mock_event_container,
    mock_queue_container,
)
from rattlesnake.utilities import GlobalCommands


# region Helpers
def get_ping_alive_event(event_container, use_thread):
    """
    Return ``ping_alive_event`` from the event container when available.

    Some mock event containers may not define this event, so create a local
    compatible event as a fallback.
    """
    if hasattr(event_container, "ping_alive_event"):
        return event_container.ping_alive_event

    return threading.Event() if use_thread else mp.Event()


# endregion


# region Fixtures
@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def acquisition_setup(request):
    """
    Create queue and event containers for threaded and multiprocessing modes.
    """
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    ping_alive_event = get_ping_alive_event(event_container, use_thread)

    return use_thread, queue_container, event_container, ping_alive_event


@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def acquisition(request):
    """
    Create an ``AcquisitionProcess`` in threaded and multiprocessing modes.
    """
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    ping_alive_event = get_ping_alive_event(event_container, use_thread)

    return AcquisitionProcess(
        "Process Name",
        queue_container,
        event_container.acquisition_active_event,
        event_container.streaming_active_event,
        event_container.acquisition_ready_event,
        ping_alive_event,
    )


@pytest.fixture
def hardware_metadata():
    """
    Create mock hardware metadata for acquisition tests.
    """
    return skeleton_hardware_metadata()


@pytest.fixture
def environment_metadata():
    """
    Create mock environment metadata for acquisition routing tests.
    """
    return skeleton_environment_metadata()


# endregion


# region AcquisitionProcess Initialization
@pytest.mark.parametrize("use_thread", [True, False])
def test_acquisition_init(use_thread):
    """
    Verifies that ``AcquisitionProcess`` initializes successfully and is an
    ``AbstractMessageProcess``.
    """
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    ping_alive_event = get_ping_alive_event(event_container, use_thread)

    acquisition = AcquisitionProcess(
        "Process Name",
        queue_container,
        event_container.acquisition_active_event,
        event_container.streaming_active_event,
        event_container.acquisition_ready_event,
        ping_alive_event,
    )

    assert isinstance(acquisition, AcquisitionProcess)
    assert isinstance(acquisition, AbstractMessageProcess)

    assert acquisition.process_name == "Process Name"
    assert acquisition.queue_container is queue_container
    assert acquisition.ping_alive_event is ping_alive_event

    assert acquisition.startup is True
    assert acquisition.shutdown_flag is False
    assert acquisition.any_environments_started is False

    assert acquisition.sample_rate is None
    assert acquisition.read_size is None

    assert acquisition.environment_list == []
    assert acquisition.environment_acquisition_channels == {}
    assert acquisition.environment_active_flags == {}
    assert acquisition.environment_last_data == {}
    assert acquisition.environment_samples_remaining_to_read == {}
    assert acquisition.environment_first_data == {}

    assert acquisition.hardware is None
    assert acquisition.hardware_metadata is None

    assert acquisition.has_streamed is False
    assert acquisition.read_data is None
    assert acquisition.output_indices is None
    assert acquisition.abort_limits is None
    assert acquisition.warning_limits is None


def test_acquisition_command_map(acquisition):
    """
    Verifies that acquisition-specific global commands are mapped to the
    correct process methods.
    """
    assert acquisition.command_map[GlobalCommands.INITIALIZE_HARDWARE] == (
        acquisition.initialize_hardware
    )
    assert acquisition.command_map[GlobalCommands.RUN_HARDWARE] == (
        acquisition.acquire_signal
    )
    assert acquisition.command_map[GlobalCommands.STOP_HARDWARE] == (
        acquisition.stop_acquisition
    )
    assert acquisition.command_map[GlobalCommands.INITIALIZE_ENVIRONMENT] == (
        acquisition.initialize_environment
    )
    assert acquisition.command_map[GlobalCommands.STOP_ENVIRONMENT] == (
        acquisition.stop_environment
    )
    assert acquisition.command_map[GlobalCommands.START_STREAMING] == (
        acquisition.start_streaming
    )
    assert acquisition.command_map[GlobalCommands.STOP_STREAMING] == (
        acquisition.stop_streaming
    )


# endregion


# region State Properties and Events
def test_acquisition_active_property(acquisition):
    """
    Verifies that ``acquisition_active`` reflects the acquisition active event.
    """
    acquisition.clear_active()

    assert acquisition.acquisition_active is False

    acquisition.set_active()

    assert acquisition.acquisition_active is True


def test_acquisition_set_active(acquisition):
    """
    Verifies that calling ``set_active`` sets the acquisition active event.
    """
    acquisition.clear_active()

    acquisition.set_active()

    assert acquisition.acquisition_active is True


def test_acquisition_clear_active(acquisition):
    """
    Verifies that calling ``clear_active`` clears the acquisition active event.
    """
    acquisition.set_active()

    acquisition.clear_active()

    assert acquisition.acquisition_active is False


def test_acquisition_streaming_property(acquisition):
    """
    Verifies that ``streaming`` reflects the streaming active event.
    """
    acquisition.clear_streaming()

    assert acquisition.streaming is False

    acquisition.set_streaming()

    assert acquisition.streaming is True


def test_acquisition_set_streaming(acquisition):
    """
    Verifies that calling ``set_streaming`` sets the streaming active event.
    """
    acquisition.clear_streaming()

    acquisition.set_streaming()

    assert acquisition.streaming is True


def test_acquisition_clear_streaming(acquisition):
    """
    Verifies that calling ``clear_streaming`` clears the streaming active event.
    """
    acquisition.set_streaming()

    acquisition.clear_streaming()

    assert acquisition.streaming is False


# endregion


# region State Synchronization
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_process_initialize_hardware(
    mock_log,
    acquisition,
    hardware_metadata,
):
    """
    Verifies that hardware initialization stores sampling parameters, creates
    hardware, initializes limits and buffers, and sets the ready event.
    """
    mock_existing_hardware = mock.MagicMock()
    acquisition.hardware = mock_existing_hardware
    acquisition.clear_ready()

    mock_hardware = mock.MagicMock()
    mock_hardware_class = mock.MagicMock(return_value=mock_hardware)

    with mock.patch.dict(
        "rattlesnake.process.acquisition.HARDWARE_ACQUISITION",
        {hardware_metadata.hardware_type: mock_hardware_class},
        clear=True,
    ):
        acquisition.initialize_hardware(hardware_metadata)

    mock_log.assert_called_with("Initializing Hardware")

    mock_existing_hardware.close.assert_called_once()
    mock_hardware_class.assert_called_once_with(
        acquisition.ping_alive_event,
        acquisition.queue_container.single_process_hardware_queue,
    )
    mock_hardware.initialize_hardware.assert_called_once_with(hardware_metadata)

    assert acquisition.sample_rate == hardware_metadata.sample_rate
    assert acquisition.read_size == hardware_metadata.samples_per_read
    assert acquisition.hardware is mock_hardware
    assert acquisition.hardware_metadata is hardware_metadata

    assert isinstance(acquisition.abort_limits, np.ndarray)
    assert isinstance(acquisition.warning_limits, np.ndarray)
    assert acquisition.abort_limits.shape == (len(hardware_metadata.channel_list),)
    assert acquisition.warning_limits.shape == (len(hardware_metadata.channel_list),)

    expected_output_indices = [
        index
        for index, channel in enumerate(hardware_metadata.channel_list)
        if (channel.feedback_device is not None)
        and not (
            channel.feedback_device.startswith("#")
            or channel.feedback_device.strip() == ""
        )
    ]
    assert acquisition.output_indices == expected_output_indices

    expected_buffer_size = 4 * np.max(
        [
            hardware_metadata.samples_per_read,
            hardware_metadata.samples_per_write // hardware_metadata.output_oversample,
        ]
    )
    assert acquisition.read_data.shape == (
        len(hardware_metadata.channel_list),
        expected_buffer_size,
    )

    assert acquisition.ready_event.is_set()


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_process_initialize_hardware_without_existing_hardware(
    mock_log,
    acquisition,
    hardware_metadata,
):
    """
    Verifies that hardware initialization works when no previous hardware
    object exists.
    """
    acquisition.hardware = None
    acquisition.clear_ready()

    mock_hardware = mock.MagicMock()
    mock_hardware_class = mock.MagicMock(return_value=mock_hardware)

    with mock.patch.dict(
        "rattlesnake.process.acquisition.HARDWARE_ACQUISITION",
        {hardware_metadata.hardware_type: mock_hardware_class},
        clear=True,
    ):
        acquisition.initialize_hardware(hardware_metadata)

    mock_log.assert_called_with("Initializing Hardware")
    mock_hardware.initialize_hardware.assert_called_once_with(hardware_metadata)
    assert acquisition.hardware is mock_hardware
    assert acquisition.ready_event.is_set()


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_process_initialize_environment(
    mock_log,
    acquisition,
    hardware_metadata,
    environment_metadata,
):
    """
    Verifies that environment acquisition routing state is initialized from
    environment metadata.
    """
    acquisition.hardware_metadata = hardware_metadata
    acquisition.clear_ready()

    acquisition.initialize_environment({"Environment 0": environment_metadata})

    mock_log.assert_called_with("Initializing Environment")

    assert acquisition.environment_list == ["Environment 0"]
    assert acquisition.environment_acquisition_channels["Environment 0"] == [0, 1]
    assert acquisition.environment_active_flags["Environment 0"] is False
    assert acquisition.environment_last_data["Environment 0"] is False
    assert acquisition.environment_samples_remaining_to_read["Environment 0"] == 0
    assert acquisition.environment_first_data["Environment 0"] is None
    assert acquisition.ready_event.is_set()


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_process_initialize_multiple_environments(
    mock_log,
    acquisition,
    hardware_metadata,
):
    """
    Verifies that multiple environments are initialized independently.
    """
    acquisition.hardware_metadata = hardware_metadata

    metadata_0 = skeleton_environment_metadata(channel_list_bools=[True, False])
    metadata_1 = skeleton_environment_metadata(channel_list_bools=[False, True])

    acquisition.initialize_environment(
        {
            "Environment 0": metadata_0,
            "Environment 1": metadata_1,
        }
    )

    mock_log.assert_called_with("Initializing Environment")

    assert acquisition.environment_list == ["Environment 0", "Environment 1"]
    assert acquisition.environment_acquisition_channels["Environment 0"] == [0]
    assert acquisition.environment_acquisition_channels["Environment 1"] == [1]

    for environment_name in ["Environment 0", "Environment 1"]:
        assert acquisition.environment_active_flags[environment_name] is False
        assert acquisition.environment_last_data[environment_name] is False
        assert acquisition.environment_samples_remaining_to_read[environment_name] == 0
        assert acquisition.environment_first_data[environment_name] is None


# endregion


# region Commands
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_process_stop_environment(mock_log, acquisition):
    """
    Verifies that stopping an environment clears active state and prepares
    final data delivery.
    """
    acquisition.environment_list = ["Environment 0"]
    acquisition.environment_acquisition_channels["Environment 0"] = [0, 1]
    acquisition.environment_active_flags["Environment 0"] = True
    acquisition.environment_last_data["Environment 0"] = False
    acquisition.environment_samples_remaining_to_read["Environment 0"] = 0
    acquisition.environment_first_data["Environment 0"] = None

    mock_hardware = mock.MagicMock()
    mock_hardware.get_acquisition_delay.return_value = 17
    acquisition.hardware = mock_hardware

    acquisition.stop_environment("Environment 0")

    mock_log.assert_called_with("Deactivating Environment Environment 0")
    mock_hardware.get_acquisition_delay.assert_called_once()

    assert acquisition.environment_active_flags["Environment 0"] is False
    assert acquisition.environment_last_data["Environment 0"] is True
    assert acquisition.environment_samples_remaining_to_read["Environment 0"] == 17


@pytest.mark.parametrize("previously_streamed", [True, False])
def test_acquisition_process_start_streaming(previously_streamed, acquisition):
    """
    Verifies that starting streaming sets the streaming state and requests a new
    stream when streaming had previously occurred.
    """
    mock_streaming_command_queue = mock.MagicMock()
    acquisition.queue_container.streaming_command_queue = mock_streaming_command_queue
    acquisition.has_streamed = previously_streamed

    acquisition.start_streaming(None)

    assert acquisition.streaming is True
    assert acquisition.has_streamed is True

    if previously_streamed:
        mock_streaming_command_queue.put.assert_called_once_with(
            "Process Name",
            (GlobalCommands.CREATE_NEW_STREAM, None),
        )
    else:
        mock_streaming_command_queue.put.assert_not_called()


def test_acquisition_process_stop_streaming(acquisition):
    """
    Verifies that stopping streaming clears the streaming state.
    """
    acquisition.set_streaming()

    acquisition.stop_streaming(None)

    assert acquisition.streaming is False


def test_add_data_to_buffer(acquisition):
    """
    Verifies that new acquired data is added to the end of the rolling buffer.
    """
    acquisition.read_data = np.zeros((1, 5))
    data = np.array([[1.0, 2.0]])

    acquisition.add_data_to_buffer(data)

    np.testing.assert_array_equal(
        acquisition.read_data,
        np.array([[0.0, 0.0, 0.0, 1.0, 2.0]]),
    )


def test_add_data_to_buffer_full_buffer(acquisition):
    """
    Verifies that data replacing the full buffer overwrites the buffer.
    """
    acquisition.read_data = np.zeros((1, 3))
    data = np.array([[1.0, 2.0, 3.0]])

    acquisition.add_data_to_buffer(data)

    np.testing.assert_array_equal(acquisition.read_data, data)


def test_add_data_to_buffer_empty_data(acquisition):
    """
    Verifies that empty acquired data does not modify the rolling buffer.
    """
    acquisition.read_data = np.array([[1.0, 2.0, 3.0]])
    original = acquisition.read_data.copy()
    data = np.empty((1, 0))

    acquisition.add_data_to_buffer(data)

    np.testing.assert_array_equal(acquisition.read_data, original)


@mock.patch("rattlesnake.process.acquisition.flush_queue")
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_acquisition_process_get_first_output_data(
    mock_log,
    mock_flush_queue,
    acquisition,
):
    """
    Verifies that first-output synchronization data is read from the sync queue
    and stored by environment name.
    """
    acquisition.environment_list = ["Environment 0"]
    acquisition.environment_acquisition_channels["Environment 0"] = [0, 1]
    acquisition.environment_active_flags["Environment 0"] = False
    acquisition.environment_last_data["Environment 0"] = False
    acquisition.environment_samples_remaining_to_read["Environment 0"] = 0
    acquisition.environment_first_data["Environment 0"] = None

    mock_flush_queue.return_value = [("Environment 0", "Data")]

    acquisition.get_first_output_data()

    mock_flush_queue.assert_called_once_with(
        acquisition.queue_container.input_output_sync_queue
    )
    mock_log.assert_called_with(
        "Listening for first data for environment Environment 0"
    )

    assert acquisition.environment_first_data["Environment 0"] == "Data"
    assert acquisition.any_environments_started is True


@mock.patch("rattlesnake.process.acquisition.flush_queue")
def test_acquisition_process_get_first_output_data_multiple_environments(
    mock_flush_queue,
    acquisition,
):
    """
    Verifies that first-output data can be stored for multiple environments.
    """
    acquisition.environment_first_data = {
        "Environment 0": None,
        "Environment 1": None,
    }

    mock_flush_queue.return_value = [
        ("Environment 0", "Data 0"),
        ("Environment 1", "Data 1"),
    ]

    acquisition.get_first_output_data()

    assert acquisition.environment_first_data["Environment 0"] == "Data 0"
    assert acquisition.environment_first_data["Environment 1"] == "Data 1"
    assert acquisition.any_environments_started is True


def test_acquisition_process_stop_acquisition(acquisition):
    """
    Verifies that calling ``stop_acquisition`` sets ``shutdown_flag``.
    """
    acquisition.shutdown_flag = False

    acquisition.stop_acquisition(None)

    assert acquisition.shutdown_flag is True


# endregion


# region Shutdown
@mock.patch("rattlesnake.process.acquisition.flush_queue")
@mock.patch("rattlesnake.process.acquisition.AcquisitionProcess.log")
def test_acquisition_process_quit(
    mock_log,
    mock_flush_queue,
    acquisition,
):
    """
    Verifies that quitting flushes queues, closes hardware, and returns
    ``True``.
    """
    mock_hardware = mock.MagicMock()
    acquisition.hardware = mock_hardware

    mock_environment_queue = mock.MagicMock()
    mock_acquisition_queue = mock.MagicMock()
    acquisition.queue_container.environment_data_in_queues = {
        "Environment 0": mock_environment_queue
    }
    acquisition.queue_container.acquisition_command_queue = mock_acquisition_queue

    mock_flush_queue.side_effect = [
        ["env item 0", "env item 1"],
        ["acq item"],
    ]

    result = acquisition.quit(None)

    assert result is True
    mock_hardware.close.assert_called_once()

    assert mock_flush_queue.call_args_list == [
        mock.call(mock_environment_queue),
        mock.call(mock_acquisition_queue),
    ]
    mock_log.assert_called_with("Flushed 3 items out of queues")


@mock.patch("rattlesnake.process.acquisition.flush_queue")
@mock.patch("rattlesnake.process.acquisition.AcquisitionProcess.log")
def test_acquisition_process_quit_without_hardware(
    mock_log,
    mock_flush_queue,
    acquisition,
):
    """
    Verifies that quitting succeeds when no hardware object exists.
    """
    acquisition.hardware = None
    acquisition.queue_container.environment_data_in_queues = {}
    mock_acquisition_queue = mock.MagicMock()
    acquisition.queue_container.acquisition_command_queue = mock_acquisition_queue

    mock_flush_queue.return_value = []

    result = acquisition.quit(None)

    assert result is True
    mock_flush_queue.assert_called_once_with(mock_acquisition_queue)
    mock_log.assert_called_with("Flushed 0 items out of queues")


# endregion


# region acquisition_process
@pytest.mark.parametrize("use_thread", [True, False])
@mock.patch("rattlesnake.process.acquisition.AcquisitionProcess")
def test_acquisition_process_func(mock_acquisition_process_class, use_thread):
    """
    Verifies that ``acquisition_process`` constructs an ``AcquisitionProcess``
    and calls its ``run`` method.
    """
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    ping_alive_event = get_ping_alive_event(event_container, use_thread)

    acquisition_process(
        queue_container,
        event_container.acquisition_active_event,
        event_container.streaming_active_event,
        event_container.acquisition_ready_event,
        event_container.acquisition_close_event,
        ping_alive_event,
    )

    mock_acquisition_process_class.assert_called_once_with(
        "Acquisition",
        queue_container,
        event_container.acquisition_active_event,
        event_container.streaming_active_event,
        event_container.acquisition_ready_event,
        ping_alive_event,
    )

    mock_instance = mock_acquisition_process_class.return_value
    mock_instance.run.assert_called_once_with(event_container.acquisition_close_event)


# endregion
