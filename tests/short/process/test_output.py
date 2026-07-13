import multiprocessing as mp
import threading
from unittest import mock

import numpy as np
import pytest

from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.process.output import OutputProcess, output_process
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
def output_setup(request):
    """
    Create queue and event containers for threaded and multiprocessing modes.
    """
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    ping_alive_event = get_ping_alive_event(event_container, use_thread)

    return use_thread, queue_container, event_container, ping_alive_event


@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def output(request):
    """
    Create an ``OutputProcess`` in threaded and multiprocessing modes.
    """
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    ping_alive_event = get_ping_alive_event(event_container, use_thread)

    return OutputProcess(
        "Process Name",
        queue_container,
        event_container.output_active_event,
        event_container.output_ready_event,
        ping_alive_event,
    )


@pytest.fixture
def hardware_metadata():
    """
    Create mock hardware metadata for output tests.
    """
    return skeleton_hardware_metadata()


@pytest.fixture
def environment_metadata():
    """
    Create mock environment metadata for output routing tests.
    """
    return skeleton_environment_metadata()


# endregion


# region OutputProcess Initialization
@pytest.mark.parametrize("use_thread", [True, False])
def test_output_init(use_thread):
    """
    Verifies that ``OutputProcess`` initializes successfully and is an
    ``AbstractMessageProcess``.
    """
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    ping_alive_event = get_ping_alive_event(event_container, use_thread)

    output = OutputProcess(
        "Process Name",
        queue_container,
        event_container.output_active_event,
        event_container.output_ready_event,
        ping_alive_event,
    )

    assert isinstance(output, OutputProcess)
    assert isinstance(output, AbstractMessageProcess)

    assert output.process_name == "Process Name"
    assert output.queue_container is queue_container
    assert output.ping_alive_event is ping_alive_event

    assert output.startup is True
    assert output.shutdown_flag is False

    assert output.sample_rate is None
    assert output.write_size is None
    assert output.num_outputs is None
    assert output.output_oversample is None

    assert output.environment_list == []
    assert output.environment_output_channels is None
    assert output.environment_active_flags == {}
    assert output.environment_starting_up_flags == {}
    assert output.environment_shutting_down_flags == {}
    assert output.environment_data_out_remainders is None
    assert output.environment_first_data == {}

    assert output.hardware is None
    assert output.hardware_metadata is None


def test_output_command_map(output):
    """
    Verifies that output-specific global commands are mapped to the correct
    process methods.
    """
    assert output.command_map[GlobalCommands.INITIALIZE_HARDWARE] == (
        output.initialize_hardware
    )
    assert output.command_map[GlobalCommands.RUN_HARDWARE] == output.output_signal
    assert output.command_map[GlobalCommands.STOP_HARDWARE] == output.stop_output
    assert output.command_map[GlobalCommands.INITIALIZE_ENVIRONMENT] == (
        output.initialize_environment
    )
    assert output.command_map[GlobalCommands.START_ENVIRONMENT] == (
        output.start_environment
    )


# endregion


# region State Properties and Events
def test_output_active_property(output):
    """
    Verifies that ``output_active`` reflects the output active event.
    """
    output.clear_active()

    assert output.output_active is False

    output.set_active()

    assert output.output_active is True


def test_output_set_active(output):
    """
    Verifies that calling ``set_active`` sets the output active event.
    """
    output.clear_active()

    output.set_active()

    assert output.output_active is True


def test_output_clear_active(output):
    """
    Verifies that calling ``clear_active`` clears the output active event.
    """
    output.set_active()

    output.clear_active()

    assert output.output_active is False


# endregion


# region State Synchronization
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_output_process_initialize_hardware(
    mock_log,
    output,
    hardware_metadata,
):
    """
    Verifies that hardware initialization stores sampling parameters, creates
    hardware, initializes the hardware object, computes output count, stores
    metadata, and sets the ready event.
    """
    mock_existing_hardware = mock.MagicMock()
    output.hardware = mock_existing_hardware
    output.clear_ready()

    mock_hardware = mock.MagicMock()
    mock_hardware_class = mock.MagicMock(return_value=mock_hardware)

    with mock.patch.dict(
        "rattlesnake.process.output.HARDWARE_OUTPUT",
        {hardware_metadata.hardware_type: mock_hardware_class},
        clear=True,
    ):
        output.initialize_hardware(hardware_metadata)

    mock_log.assert_called_with("Initializing Hardware")

    mock_existing_hardware.close.assert_called_once()
    mock_hardware_class.assert_called_once_with(
        output.ping_alive_event,
        output.queue_container.single_process_hardware_queue,
    )
    mock_hardware.initialize_hardware.assert_called_once_with(hardware_metadata)

    assert output.sample_rate == hardware_metadata.sample_rate
    assert output.write_size == hardware_metadata.samples_per_write
    assert output.output_oversample == hardware_metadata.output_oversample
    assert output.hardware is mock_hardware
    assert output.hardware_metadata is hardware_metadata

    expected_output_indices = [
        index
        for index, channel in enumerate(hardware_metadata.channel_list)
        if (channel.feedback_device is not None)
        and not (
            channel.feedback_device.startswith("#")
            or channel.feedback_device.strip() == ""
        )
    ]
    assert output.num_outputs == len(expected_output_indices)

    assert output.ready_event.is_set()


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_output_process_initialize_hardware_without_existing_hardware(
    mock_log,
    output,
    hardware_metadata,
):
    """
    Verifies that hardware initialization works when no previous hardware
    object exists.
    """
    output.hardware = None
    output.clear_ready()

    mock_hardware = mock.MagicMock()
    mock_hardware_class = mock.MagicMock(return_value=mock_hardware)

    with mock.patch.dict(
        "rattlesnake.process.output.HARDWARE_OUTPUT",
        {hardware_metadata.hardware_type: mock_hardware_class},
        clear=True,
    ):
        output.initialize_hardware(hardware_metadata)

    mock_log.assert_called_with("Initializing Hardware")
    mock_hardware.initialize_hardware.assert_called_once_with(hardware_metadata)
    assert output.hardware is mock_hardware
    assert output.ready_event.is_set()


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_output_process_initialize_environment(
    mock_log,
    output,
    hardware_metadata,
    environment_metadata,
):
    """
    Verifies that environment output routing state is initialized from
    environment metadata.
    """
    output.hardware_metadata = hardware_metadata
    output.clear_ready()

    output.initialize_environment({"Environment 0": environment_metadata})

    mock_log.assert_called_with("Initializing Environment")

    assert output.environment_list == ["Environment 0"]
    assert output.environment_active_flags["Environment 0"] is False
    assert output.environment_starting_up_flags["Environment 0"] is False
    assert output.environment_shutting_down_flags["Environment 0"] is False
    assert output.environment_first_data["Environment 0"] is False

    np.testing.assert_array_equal(
        output.environment_output_channels["Environment 0"],
        np.array([0]),
    )
    np.testing.assert_array_equal(
        output.environment_data_out_remainders["Environment 0"],
        np.zeros((1, 0)),
    )

    assert output.ready_event.is_set()


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_output_process_initialize_multiple_environments(
    mock_log,
    output,
    hardware_metadata,
):
    """
    Verifies that multiple environments are initialized independently.
    """
    output.hardware_metadata = hardware_metadata

    metadata_0 = skeleton_environment_metadata(channel_list_bools=[True, True])
    metadata_1 = skeleton_environment_metadata(channel_list_bools=[False, True])

    output.initialize_environment(
        {
            "Environment 0": metadata_0,
            "Environment 1": metadata_1,
        }
    )

    mock_log.assert_called_with("Initializing Environment")

    assert output.environment_list == ["Environment 0", "Environment 1"]

    for environment_name in ["Environment 0", "Environment 1"]:
        assert output.environment_active_flags[environment_name] is False
        assert output.environment_starting_up_flags[environment_name] is False
        assert output.environment_shutting_down_flags[environment_name] is False
        assert output.environment_first_data[environment_name] is False
        np.testing.assert_array_equal(
            output.environment_output_channels[environment_name],
            np.array([0]),
        )
        np.testing.assert_array_equal(
            output.environment_data_out_remainders[environment_name],
            np.zeros((1, 0)),
        )


# endregion


# region Commands
@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_process_stop_output(mock_log, output):
    """
    Verifies that requesting output shutdown logs the action and sets
    ``shutdown_flag``.
    """
    output.shutdown_flag = False

    output.stop_output(None)

    mock_log.assert_called_with("Starting Shutdown Procedure")
    assert output.shutdown_flag is True


@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_process_start_environment(mock_log, output):
    """
    Verifies that starting an environment sets startup state, clears shutdown
    state, and leaves active state false until enough output data is available.
    """
    output.environment_list = ["Environment 0"]
    output.environment_active_flags["Environment 0"] = False
    output.environment_starting_up_flags["Environment 0"] = False
    output.environment_shutting_down_flags["Environment 0"] = True
    output.environment_first_data["Environment 0"] = False

    output.start_environment("Environment 0")

    mock_log.assert_called_with("Started Environment Environment 0")
    assert output.environment_starting_up_flags["Environment 0"] is True
    assert output.environment_shutting_down_flags["Environment 0"] is False
    assert output.environment_active_flags["Environment 0"] is False


@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_signal_writes_active_environment_data_and_starts_hardware(
    mock_log,
    output,
):
    """
    Verifies that ``output_signal`` writes active environment data to hardware,
    sends first-data synchronization information to acquisition, starts
    hardware on startup, sets output active state, and schedules the next
    output iteration.
    """
    output.num_outputs = 1
    output.write_size = 4
    output.output_oversample = 1
    output.startup = True
    output.shutdown_flag = False

    output.environment_list = ["Environment 0"]
    output.environment_output_channels = {"Environment 0": np.array([0])}
    output.environment_active_flags = {"Environment 0": True}
    output.environment_starting_up_flags = {"Environment 0": False}
    output.environment_shutting_down_flags = {"Environment 0": False}
    output.environment_data_out_remainders = {
        "Environment 0": np.ones((1, 4)),
    }
    output.environment_first_data = {"Environment 0": True}

    mock_hardware = mock.MagicMock()
    mock_hardware.ready_for_new_output.return_value = True
    output.hardware = mock_hardware

    mock_input_output_sync_queue = mock.MagicMock()
    mock_output_command_queue = mock.MagicMock()
    output.queue_container.input_output_sync_queue = mock_input_output_sync_queue
    output.queue_container.output_command_queue = mock_output_command_queue

    output.output_signal(None)

    mock_hardware.ready_for_new_output.assert_not_called()
    mock_hardware.write.assert_called_once()
    np.testing.assert_array_equal(
        mock_hardware.write.call_args.args[0],
        np.ones((1, 4)),
    )
    mock_hardware.start.assert_called_once()

    assert output.startup is False
    assert output.output_active is True
    assert output.environment_first_data["Environment 0"] is False
    np.testing.assert_array_equal(
        output.environment_data_out_remainders["Environment 0"],
        np.zeros((1, 0)),
    )

    first_sync_call = mock_input_output_sync_queue.put.call_args_list[0]
    assert first_sync_call.args[0][0] == "Environment 0"
    np.testing.assert_array_equal(first_sync_call.args[0][1], np.ones((1, 4)))

    second_sync_call = mock_input_output_sync_queue.put.call_args_list[1]
    assert second_sync_call.args[0] == (None, True)

    mock_output_command_queue.put.assert_called_once_with(
        output.process_name,
        (GlobalCommands.RUN_HARDWARE, None),
    )

    assert any(
        "Writing 1 x 4 data to hardware" in call.args[0]
        for call in mock_log.call_args_list
    )


@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_signal_gets_environment_data_from_queue(
    mock_log,
    output,
):
    """
    Verifies that ``output_signal`` retrieves environment output data from an
    environment data-out queue and appends it to the environment remainder
    buffer.
    """
    output.num_outputs = 1
    output.write_size = 4
    output.output_oversample = 1
    output.startup = False
    output.shutdown_flag = False

    output.environment_list = ["Environment 0"]
    output.environment_output_channels = {"Environment 0": np.array([0])}
    output.environment_active_flags = {"Environment 0": True}
    output.environment_starting_up_flags = {"Environment 0": False}
    output.environment_shutting_down_flags = {"Environment 0": False}
    output.environment_data_out_remainders = {
        "Environment 0": np.zeros((1, 0)),
    }
    output.environment_first_data = {"Environment 0": False}

    mock_environment_queue = mock.MagicMock()
    mock_environment_queue.get_nowait.return_value = (np.ones((1, 2)), False)
    output.queue_container.environment_data_out_queues = {
        "Environment 0": mock_environment_queue,
    }

    mock_hardware = mock.MagicMock()
    mock_hardware.ready_for_new_output.return_value = True
    output.hardware = mock_hardware

    mock_output_command_queue = mock.MagicMock()
    output.queue_container.output_command_queue = mock_output_command_queue

    output.output_signal(None)

    mock_environment_queue.get_nowait.assert_called_once()
    np.testing.assert_array_equal(
        output.environment_data_out_remainders["Environment 0"],
        np.ones((1, 2)),
    )
    mock_hardware.write.assert_not_called()
    mock_output_command_queue.put.assert_called_once_with(
        output.process_name,
        (GlobalCommands.RUN_HARDWARE, None),
    )

    assert any(
        "Got 1 x 2 data from Environment 0 Environment" in call.args[0]
        for call in mock_log.call_args_list
    )


@mock.patch("rattlesnake.process.output.flush_queue")
@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_signal_shutdown_when_all_environments_inactive(
    mock_log,
    mock_flush_queue,
    output,
):
    """
    Verifies that ``output_signal`` stops hardware and clears output active
    state when shutdown is requested and all environment output has drained.
    """
    output.num_outputs = 1
    output.write_size = 4
    output.output_oversample = 1
    output.startup = False
    output.shutdown_flag = True
    output.set_active()

    output.environment_list = ["Environment 0"]
    output.environment_output_channels = {"Environment 0": np.array([0])}
    output.environment_active_flags = {"Environment 0": False}
    output.environment_starting_up_flags = {"Environment 0": False}
    output.environment_shutting_down_flags = {"Environment 0": False}
    output.environment_data_out_remainders = {
        "Environment 0": np.zeros((1, 0)),
    }
    output.environment_first_data = {"Environment 0": False}

    mock_hardware = mock.MagicMock()
    mock_hardware.ready_for_new_output.return_value = True
    output.hardware = mock_hardware

    mock_output_command_queue = mock.MagicMock()
    output.queue_container.output_command_queue = mock_output_command_queue

    output.output_signal(None)

    mock_hardware.stop.assert_called_once()
    mock_flush_queue.assert_called_once_with(
        output.queue_container.input_output_sync_queue
    )

    assert output.startup is True
    assert output.shutdown_flag is False
    assert output.output_active is False

    mock_output_command_queue.put.assert_not_called()
    mock_log.assert_any_call("Stopping Hardware")


@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_signal_deactivates_drained_environment(
    mock_log,
    output,
):
    """
    Verifies that an environment marked as shutting down is deactivated once
    its output remainder has drained and acquisition is notified.
    """
    output.num_outputs = 1
    output.write_size = 4
    output.output_oversample = 1
    output.startup = False
    output.shutdown_flag = False

    output.environment_list = ["Environment 0"]
    output.environment_output_channels = {"Environment 0": np.array([0])}
    output.environment_active_flags = {"Environment 0": True}
    output.environment_starting_up_flags = {"Environment 0": False}
    output.environment_shutting_down_flags = {"Environment 0": True}
    output.environment_data_out_remainders = {
        "Environment 0": np.zeros((1, 0)),
    }
    output.environment_first_data = {"Environment 0": False}

    mock_hardware = mock.MagicMock()
    mock_hardware.ready_for_new_output.return_value = True
    output.hardware = mock_hardware

    mock_acquisition_command_queue = mock.MagicMock()
    mock_output_command_queue = mock.MagicMock()
    output.queue_container.acquisition_command_queue = mock_acquisition_command_queue
    output.queue_container.output_command_queue = mock_output_command_queue

    output.output_signal(None)

    assert output.environment_active_flags["Environment 0"] is False
    assert output.environment_starting_up_flags["Environment 0"] is False
    assert output.environment_shutting_down_flags["Environment 0"] is False

    mock_acquisition_command_queue.put.assert_called_once_with(
        output.process_name,
        (GlobalCommands.STOP_ENVIRONMENT, "Environment 0"),
    )
    mock_output_command_queue.put.assert_called_once_with(
        output.process_name,
        (GlobalCommands.RUN_HARDWARE, None),
    )


# endregion


# region Shutdown
@mock.patch("rattlesnake.process.output.flush_queue")
@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_process_quit(
    mock_log,
    mock_flush_queue,
    output,
):
    """
    Verifies that quitting flushes queues, closes hardware, and returns
    ``True``.
    """
    mock_hardware = mock.MagicMock()
    output.hardware = mock_hardware

    mock_environment_queue = mock.MagicMock()
    mock_output_queue = mock.MagicMock()
    mock_sync_queue = mock.MagicMock()
    mock_single_process_hardware_queue = mock.MagicMock()

    output.queue_container.environment_data_out_queues = {
        "Environment 0": mock_environment_queue,
    }
    output.queue_container.output_command_queue = mock_output_queue
    output.queue_container.input_output_sync_queue = mock_sync_queue
    output.queue_container.single_process_hardware_queue = (
        mock_single_process_hardware_queue
    )

    mock_flush_queue.side_effect = [
        ["env item 0", "env item 1"],
        ["output item"],
        [],
        ["hardware item"],
    ]

    result = output.quit(None)

    assert result is True
    mock_hardware.close.assert_called_once()

    assert mock_flush_queue.call_args_list == [
        mock.call(mock_environment_queue),
        mock.call(mock_output_queue),
        mock.call(mock_sync_queue),
        mock.call(mock_single_process_hardware_queue),
    ]
    mock_log.assert_called_with("Flushed 4 items out of queues")


@mock.patch("rattlesnake.process.output.flush_queue")
@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_process_quit_without_hardware(
    mock_log,
    mock_flush_queue,
    output,
):
    """
    Verifies that quitting succeeds when no hardware object exists.
    """
    output.hardware = None

    mock_output_queue = mock.MagicMock()
    mock_sync_queue = mock.MagicMock()
    mock_single_process_hardware_queue = mock.MagicMock()

    output.queue_container.environment_data_out_queues = {}
    output.queue_container.output_command_queue = mock_output_queue
    output.queue_container.input_output_sync_queue = mock_sync_queue
    output.queue_container.single_process_hardware_queue = (
        mock_single_process_hardware_queue
    )

    mock_flush_queue.return_value = []

    result = output.quit(None)

    assert result is True

    assert mock_flush_queue.call_args_list == [
        mock.call(mock_output_queue),
        mock.call(mock_sync_queue),
        mock.call(mock_single_process_hardware_queue),
    ]
    mock_log.assert_called_with("Flushed 0 items out of queues")


# endregion


# region output_process
@pytest.mark.parametrize("use_thread", [True, False])
@mock.patch("rattlesnake.process.output.OutputProcess")
def test_output_process_func(mock_output_process_class, use_thread):
    """
    Verifies that ``output_process`` constructs an ``OutputProcess`` and calls
    its ``run`` method.
    """
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    ping_alive_event = get_ping_alive_event(event_container, use_thread)

    output_process(
        queue_container,
        event_container.output_active_event,
        event_container.output_ready_event,
        event_container.output_close_event,
        ping_alive_event,
    )

    mock_output_process_class.assert_called_once_with(
        "Output",
        queue_container,
        event_container.output_active_event,
        event_container.output_ready_event,
        ping_alive_event,
    )

    mock_instance = mock_output_process_class.return_value
    mock_instance.run.assert_called_once_with(event_container.output_close_event)


# endregion
