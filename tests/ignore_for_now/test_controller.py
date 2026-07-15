import multiprocessing as mp
from unittest import mock

import pytest

from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.process.controller import ControllerProcess, controller_process
from rattlesnake.process.streaming import StreamMetadata, StreamType
from rattlesnake.testing.mock_utilities import (
    mock_event_container,
    mock_queue_container,
)
from rattlesnake.utilities import GlobalCommands


# region Fixtures
@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def controller_setup(request):
    """
    Create queue and event containers for threaded and multiprocessing modes.
    """
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)

    return use_thread, queue_container, event_container


@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def controller(request):
    """
    Create a ``ControllerProcess`` in threaded and multiprocessing modes.
    """
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)

    return ControllerProcess(
        "Process Name",
        queue_container,
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.streaming_active_event,
        event_container.environment_active_events,
        event_container.environment_sysid_active_events,
        event_container.controller_ready_event,
    )


# endregion


# region ControllerProcess Initialization
@pytest.mark.parametrize("use_thread", [True, False])
def test_controller_init(use_thread):
    """
    Verifies that ``ControllerProcess`` initializes successfully and is an
    ``AbstractMessageProcess``.
    """
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)

    controller = ControllerProcess(
        "Process Name",
        queue_container,
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.streaming_active_event,
        event_container.environment_active_events,
        event_container.environment_sysid_active_events,
        event_container.controller_ready_event,
    )

    assert isinstance(controller, ControllerProcess)
    assert isinstance(controller, AbstractMessageProcess)

    assert controller.process_name == "Process Name"
    assert controller.queue_container is queue_container
    assert (
        controller._acquisition_active_event is event_container.acquisition_active_event
    )
    assert controller._output_active_event is event_container.output_active_event
    assert controller._streaming_active_event is event_container.streaming_active_event
    assert (
        controller._environment_active_event
        is event_container.environment_active_events
    )
    assert (
        controller._environment_sysid_active_event
        is event_container.environment_sysid_active_events
    )
    assert controller.ready_event is event_container.controller_ready_event
    assert controller.ready_event.is_set()

    assert isinstance(controller.stream_metadata, StreamMetadata)
    assert controller.stream_metadata.stream_type == StreamType.NO_STREAM


def test_controller_command_map(controller):
    """
    Verifies that controller-specific global commands are mapped to the correct
    process methods.
    """
    assert (
        controller.command_map[GlobalCommands.RUN_HARDWARE] == controller.run_hardware
    )
    assert controller.command_map[GlobalCommands.STOP_HARDWARE] == (
        controller.stop_hardware
    )
    assert controller.command_map[GlobalCommands.START_SYSTEM_ID_NOISE] == (
        controller.start_system_id_noise
    )
    assert controller.command_map[GlobalCommands.START_SYSTEM_ID_TRANSFER] == (
        controller.start_system_id_transfer
    )
    assert controller.command_map[GlobalCommands.STOP_SYSTEM_ID] == (
        controller.stop_system_id
    )
    assert controller.command_map[GlobalCommands.START_ENVIRONMENT] == (
        controller.start_environment
    )
    assert controller.command_map[GlobalCommands.STOP_ENVIRONMENT] == (
        controller.stop_environment
    )
    assert controller.command_map[GlobalCommands.START_STREAMING] == (
        controller.start_streaming
    )
    assert controller.command_map[GlobalCommands.STOP_STREAMING] == (
        controller.stop_streaming
    )
    assert controller.command_map[GlobalCommands.STREAM_AT_TARGET_LEVEL] == (
        controller.at_target_level
    )
    assert controller.command_map[GlobalCommands.STREAM_MANUAL] == (
        controller.manual_stream
    )
    assert controller.command_map[GlobalCommands.PROFILE_CLOSEOUT] == (
        controller.profile_closeout
    )
    assert controller.command_map[GlobalCommands.SEND_ENVIRONMENT_COMMAND] == (
        controller.send_environment_command
    )


# endregion


# region State Properties
def test_controller_acquisition_active_property(controller):
    """
    Verifies that ``acquisition_active`` reflects the acquisition active event.
    """
    controller._acquisition_active_event.clear()

    assert controller.acquisition_active is False

    controller._acquisition_active_event.set()

    assert controller.acquisition_active is True


def test_controller_output_active_property(controller):
    """
    Verifies that ``output_active`` reflects the output active event.
    """
    controller._output_active_event.clear()

    assert controller.output_active is False

    controller._output_active_event.set()

    assert controller.output_active is True


def test_controller_streaming_active_property(controller):
    """
    Verifies that ``streaming_active`` reflects the streaming active event.
    """
    controller._streaming_active_event.clear()

    assert controller.streaming_active is False

    controller._streaming_active_event.set()

    assert controller.streaming_active is True


def test_controller_environments_active_property(controller):
    """
    Verifies that ``environments_active`` returns names of environments whose
    active events are set.
    """
    environment_0_event = mock.MagicMock()
    environment_0_event.is_set.return_value = True

    environment_1_event = mock.MagicMock()
    environment_1_event.is_set.return_value = False

    controller._environment_active_event = {
        "Environment 0": environment_0_event,
        "Environment 1": environment_1_event,
    }

    assert controller.environments_active == ["Environment 0"]


def test_controller_environments_sysid_active_property(controller):
    """
    Verifies that ``environments_sysid_active`` returns names of environments
    whose system identification active events are set.
    """
    environment_0_event = mock.MagicMock()
    environment_0_event.is_set.return_value = False

    environment_1_event = mock.MagicMock()
    environment_1_event.is_set.return_value = True

    controller._environment_sysid_active_event = {
        "Environment 0": environment_0_event,
        "Environment 1": environment_1_event,
    }

    assert controller.environments_sysid_active == ["Environment 1"]


# endregion


# region Hardware
@pytest.mark.parametrize("stream_type", [*StreamType])
def test_controller_run_hardware(stream_type, controller):
    """
    Verifies that acquisition and output run commands are sent and that
    immediate streaming is started for ``StreamType.IMMEDIATELY``.
    """
    mock_acquisition_active = mock.MagicMock()
    mock_acquisition_active.is_set.return_value = False
    mock_output_active = mock.MagicMock()
    mock_output_active.is_set.return_value = False

    controller._acquisition_active_event = mock_acquisition_active
    controller._output_active_event = mock_output_active

    mock_acquisition_queue = mock.MagicMock()
    mock_output_queue = mock.MagicMock()
    controller.queue_container.acquisition_command_queue = mock_acquisition_queue
    controller.queue_container.output_command_queue = mock_output_queue
    controller.start_streaming = mock.MagicMock()

    stream_metadata = StreamMetadata(stream_type=stream_type)

    controller.run_hardware(stream_metadata)

    assert controller.stream_metadata is stream_metadata

    mock_acquisition_queue.put.assert_called_once_with(
        "Controller",
        (GlobalCommands.RUN_HARDWARE, None),
    )
    mock_output_queue.put.assert_called_once_with(
        "Controller",
        (GlobalCommands.RUN_HARDWARE, None),
    )

    if stream_type == StreamType.IMMEDIATELY:
        controller.start_streaming.assert_called_once_with(True)
    else:
        controller.start_streaming.assert_not_called()


@pytest.mark.parametrize(
    "acquisition_active, output_active",
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_controller_run_hardware_error(
    acquisition_active,
    output_active,
    controller,
):
    """
    Verifies that starting hardware raises errors when acquisition or output is
    already active.
    """
    mock_acquisition_active = mock.MagicMock()
    mock_acquisition_active.is_set.return_value = acquisition_active
    mock_output_active = mock.MagicMock()
    mock_output_active.is_set.return_value = output_active

    controller._acquisition_active_event = mock_acquisition_active
    controller._output_active_event = mock_output_active

    controller.queue_container.acquisition_command_queue = mock.MagicMock()
    controller.queue_container.output_command_queue = mock.MagicMock()
    controller.start_streaming = mock.MagicMock()

    stream_metadata = StreamMetadata(stream_type=StreamType.NO_STREAM)

    with pytest.raises(RuntimeError):
        controller.run_hardware(stream_metadata)


@pytest.mark.parametrize(
    "acquisition_active, output_active, expected_error",
    [
        (False, False, RuntimeError),
        (False, True, RuntimeError),
        (True, False, RuntimeError),
        (True, True, None),
    ],
)
def test_controller_stop_hardware(
    acquisition_active,
    output_active,
    expected_error,
    controller,
):
    """
    Verifies that acquisition and output stop commands are sent, and that
    errors are raised when acquisition or output is inactive.
    """
    mock_acquisition_active = mock.MagicMock()
    mock_acquisition_active.is_set.return_value = acquisition_active
    mock_output_active = mock.MagicMock()
    mock_output_active.is_set.return_value = output_active
    mock_streaming_active = mock.MagicMock()
    mock_streaming_active.is_set.return_value = False

    controller._acquisition_active_event = mock_acquisition_active
    controller._output_active_event = mock_output_active
    controller._streaming_active_event = mock_streaming_active
    controller._environment_active_event = {}
    controller._environment_sysid_active_event = {}

    mock_acquisition_queue = mock.MagicMock()
    mock_output_queue = mock.MagicMock()
    mock_streaming_queue = mock.MagicMock()

    controller.queue_container.acquisition_command_queue = mock_acquisition_queue
    controller.queue_container.output_command_queue = mock_output_queue
    controller.queue_container.streaming_command_queue = mock_streaming_queue
    controller.stream_metadata = StreamMetadata(stream_type=StreamType.NO_STREAM)

    if expected_error is RuntimeError:
        with pytest.raises(RuntimeError):
            controller.stop_hardware(None)
    else:
        controller.stop_hardware(None)

    mock_acquisition_queue.put.assert_called_once_with(
        "Controller",
        (GlobalCommands.STOP_HARDWARE, None),
    )
    mock_output_queue.put.assert_called_once_with(
        "Controller",
        (GlobalCommands.STOP_HARDWARE, None),
    )


def test_controller_stop_hardware_stops_active_environments_and_sysid(controller):
    """
    Verifies that active environments and active system identification tasks are
    stopped before hardware shutdown.
    """
    controller._acquisition_active_event.set()
    controller._output_active_event.set()
    controller._streaming_active_event.clear()

    environment_active = mock.MagicMock()
    environment_active.is_set.return_value = True
    environment_inactive = mock.MagicMock()
    environment_inactive.is_set.return_value = False

    sysid_active = mock.MagicMock()
    sysid_active.is_set.return_value = True
    sysid_inactive = mock.MagicMock()
    sysid_inactive.is_set.return_value = False

    controller._environment_active_event = {
        "Environment 0": environment_active,
        "Environment 1": environment_inactive,
    }
    controller._environment_sysid_active_event = {
        "Environment 0": sysid_inactive,
        "Environment 1": sysid_active,
    }

    controller.stop_environment = mock.MagicMock()
    controller.stop_system_id = mock.MagicMock()

    mock_acquisition_queue = mock.MagicMock()
    mock_output_queue = mock.MagicMock()
    controller.queue_container.acquisition_command_queue = mock_acquisition_queue
    controller.queue_container.output_command_queue = mock_output_queue
    controller.stream_metadata = StreamMetadata(stream_type=StreamType.NO_STREAM)

    controller.stop_hardware(None)

    controller.stop_system_id.assert_called_once_with("Environment 1")
    controller.stop_environment.assert_called_once_with("Environment 0")

    mock_acquisition_queue.put.assert_called_once_with(
        "Controller",
        (GlobalCommands.STOP_HARDWARE, None),
    )
    mock_output_queue.put.assert_called_once_with(
        "Controller",
        (GlobalCommands.STOP_HARDWARE, None),
    )


@pytest.mark.parametrize("stream_type", [*StreamType])
@pytest.mark.parametrize("streaming_active", [True, False])
def test_controller_stop_hardware_streaming(
    stream_type,
    streaming_active,
    controller,
):
    """
    Verifies that active streaming is stopped and configured streams are
    finalized during hardware shutdown.
    """
    controller._acquisition_active_event.set()
    controller._output_active_event.set()

    mock_streaming_active = mock.MagicMock()
    mock_streaming_active.is_set.return_value = streaming_active
    controller._streaming_active_event = mock_streaming_active

    controller._environment_active_event = {}
    controller._environment_sysid_active_event = {}

    controller.stop_streaming = mock.MagicMock()

    mock_acquisition_queue = mock.MagicMock()
    mock_output_queue = mock.MagicMock()
    mock_streaming_queue = mock.MagicMock()

    controller.queue_container.acquisition_command_queue = mock_acquisition_queue
    controller.queue_container.output_command_queue = mock_output_queue
    controller.queue_container.streaming_command_queue = mock_streaming_queue
    controller.stream_metadata = StreamMetadata(stream_type=stream_type)

    controller.stop_hardware(None)

    if streaming_active:
        controller.stop_streaming.assert_called_once_with()
    else:
        controller.stop_streaming.assert_not_called()

    if stream_type is not StreamType.NO_STREAM:
        mock_streaming_queue.put.assert_called_once_with(
            "Process Name",
            (GlobalCommands.FINALIZE_STREAMING, None),
        )
    else:
        mock_streaming_queue.put.assert_not_called()


# endregion


# region Environment
@pytest.mark.parametrize(
    "environment_active, expected_error",
    [
        (False, None),
        (True, RuntimeError),
    ],
)
def test_controller_start_environment(
    environment_active,
    expected_error,
    controller,
):
    """
    Verifies that output and environment start commands are sent, and that
    starting an already active environment raises ``RuntimeError``.
    """
    active_event = mock.MagicMock()
    active_event.is_set.return_value = environment_active
    controller._environment_active_event = {"Environment 0": active_event}

    mock_output_queue = mock.MagicMock()
    mock_environment_queue = mock.MagicMock()

    controller.queue_container.output_command_queue = mock_output_queue
    controller.queue_container.environment_command_queues = {
        "Environment 0": mock_environment_queue
    }

    instruction = mock.MagicMock()

    if expected_error is RuntimeError:
        with pytest.raises(RuntimeError):
            controller.start_environment(("Environment 0", instruction))
        mock_output_queue.put.assert_not_called()
        mock_environment_queue.put.assert_not_called()
    else:
        controller.start_environment(("Environment 0", instruction))

        mock_output_queue.put.assert_called_once_with(
            "Controller",
            (GlobalCommands.START_ENVIRONMENT, "Environment 0"),
        )
        mock_environment_queue.put.assert_called_once_with(
            "Controller",
            (GlobalCommands.START_ENVIRONMENT, instruction),
        )


@pytest.mark.parametrize(
    "environment_active, expected_error",
    [
        (False, RuntimeError),
        (True, None),
    ],
)
def test_controller_stop_environment(
    environment_active,
    expected_error,
    controller,
):
    """
    Verifies that the environment stop command is sent for active environments
    and that stopping an inactive environment raises ``RuntimeError``.
    """
    active_event = mock.MagicMock()
    active_event.is_set.return_value = environment_active
    controller._environment_active_event = {"Environment 0": active_event}

    mock_environment_queue = mock.MagicMock()
    controller.queue_container.environment_command_queues = {
        "Environment 0": mock_environment_queue
    }

    if expected_error is RuntimeError:
        with pytest.raises(RuntimeError):
            controller.stop_environment("Environment 0")
        mock_environment_queue.put.assert_not_called()
    else:
        controller.stop_environment("Environment 0")

        mock_environment_queue.put.assert_called_once_with(
            "Controller",
            (GlobalCommands.STOP_ENVIRONMENT, None),
        )


def test_controller_send_environment_command(controller):
    """
    Verifies that an arbitrary command and payload are forwarded to the
    requested environment command queue.
    """
    mock_environment_queue = mock.MagicMock()
    controller.queue_container.environment_command_queues = {
        "Environment 0": mock_environment_queue
    }

    command = object()
    command_data = {"value": 1}

    controller.send_environment_command(("Environment 0", command, command_data))

    mock_environment_queue.put.assert_called_once_with(
        "Controller",
        (command, command_data),
    )


# endregion


# region System Identification
def test_controller_start_system_id_noise(controller):
    """
    Verifies that output receives a start-environment command and the
    environment receives a start-system-identification-noise command.
    """
    mock_output_queue = mock.MagicMock()
    mock_environment_queue = mock.MagicMock()

    controller.queue_container.output_command_queue = mock_output_queue
    controller.queue_container.environment_command_queues = {
        "Environment 0": mock_environment_queue
    }

    controller.start_system_id_noise("Environment 0")

    mock_output_queue.put.assert_called_once_with(
        "Controller",
        (GlobalCommands.START_ENVIRONMENT, "Environment 0"),
    )
    mock_environment_queue.put.assert_called_once_with(
        "Controller",
        (GlobalCommands.START_SYSTEM_ID_NOISE, None),
    )


def test_controller_start_system_id_transfer(controller):
    """
    Verifies that output receives a start-environment command and the
    environment receives a start-system-identification-transfer command.
    """
    mock_output_queue = mock.MagicMock()
    mock_environment_queue = mock.MagicMock()

    controller.queue_container.output_command_queue = mock_output_queue
    controller.queue_container.environment_command_queues = {
        "Environment 0": mock_environment_queue
    }

    controller.start_system_id_transfer("Environment 0")

    mock_output_queue.put.assert_called_once_with(
        "Controller",
        (GlobalCommands.START_ENVIRONMENT, "Environment 0"),
    )
    mock_environment_queue.put.assert_called_once_with(
        "Controller",
        (GlobalCommands.START_SYSTEM_ID_TRANSFER, None),
    )


def test_controller_stop_system_id(controller):
    """
    Verifies that the stop-system-identification command is forwarded to the
    requested environment.
    """
    mock_environment_queue = mock.MagicMock()
    controller.queue_container.environment_command_queues = {
        "Environment 0": mock_environment_queue
    }

    controller.stop_system_id("Environment 0")

    mock_environment_queue.put.assert_called_once_with(
        "Controller",
        (GlobalCommands.STOP_SYSTEM_ID, True),
    )


# endregion


# region Streaming
@pytest.mark.parametrize("stream_type", [*StreamType])
@pytest.mark.parametrize("override", [True, False, None])
def test_controller_start_streaming(stream_type, override, controller):
    """
    Verifies that streaming starts when override is truthy or when stream type
    is ``StreamType.PROFILE_INSTRUCTION``.
    """
    mock_acquisition_queue = mock.MagicMock()
    controller.queue_container.acquisition_command_queue = mock_acquisition_queue
    controller.stream_metadata.stream_type = stream_type

    controller.start_streaming(override)

    if override or stream_type == StreamType.PROFILE_INSTRUCTION:
        mock_acquisition_queue.put.assert_called_once_with(
            "Controller",
            (GlobalCommands.START_STREAMING, None),
        )
    else:
        mock_acquisition_queue.put.assert_not_called()


def test_controller_stop_streaming(controller):
    """
    Verifies that a stop-streaming command is sent to acquisition.
    """
    mock_acquisition_queue = mock.MagicMock()
    controller.queue_container.acquisition_command_queue = mock_acquisition_queue

    controller.stop_streaming(None)

    mock_acquisition_queue.put.assert_called_once_with(
        "Controller",
        (GlobalCommands.STOP_STREAMING, None),
    )


@pytest.mark.parametrize("stream_type", [*StreamType])
@pytest.mark.parametrize("environment_name", ["Environment 0", "Wrong Environment"])
def test_controller_at_target_level_match(
    stream_type,
    environment_name,
    controller,
):
    """
    Verifies that streaming starts only when stream type is
    ``StreamType.TEST_LEVEL`` and the environment name matches.
    """
    controller.stream_metadata.stream_type = stream_type
    controller.stream_metadata.test_level_environment_name = "Environment 0"
    controller.start_streaming = mock.MagicMock()

    controller.at_target_level(environment_name)

    if stream_type == StreamType.TEST_LEVEL and environment_name == "Environment 0":
        controller.start_streaming.assert_called_once_with(True)
    else:
        controller.start_streaming.assert_not_called()


@pytest.mark.parametrize("stream_type", [*StreamType])
def test_controller_manual_stream(stream_type, controller):
    """
    Verifies that manual streaming starts only when stream type is
    ``StreamType.MANUAL``.
    """
    controller.stream_metadata.stream_type = stream_type
    controller.start_streaming = mock.MagicMock()

    controller.manual_stream(None)

    if stream_type == StreamType.MANUAL:
        controller.start_streaming.assert_called_once_with(True)
    else:
        controller.start_streaming.assert_not_called()


# endregion


# region Profile
def test_controller_profile_closeout(controller):
    """
    Verifies that profile closeout sets the controller ready event.
    """
    controller.clear_ready()

    controller.profile_closeout(None)

    assert controller.ready_event.is_set()


# endregion


# region Process
@pytest.mark.parametrize("use_thread", [True, False])
@mock.patch("rattlesnake.process.controller.ControllerProcess")
def test_controller_process_func(mock_controller_process_class, use_thread):
    """
    Verifies that ``controller_process`` constructs a ``ControllerProcess`` and
    calls its ``run`` method.
    """
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)

    controller_process(
        queue_container,
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.streaming_active_event,
        event_container.environment_active_events,
        event_container.environment_sysid_active_events,
        event_container.controller_ready_event,
        event_container.controller_close_event,
    )

    mock_controller_process_class.assert_called_once_with(
        "Controller",
        queue_container,
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.streaming_active_event,
        event_container.environment_active_events,
        event_container.environment_sysid_active_events,
        event_container.controller_ready_event,
    )

    mock_instance = mock_controller_process_class.return_value
    mock_instance.run.assert_called_once_with(event_container.controller_close_event)


# endregion
