from rattlesnake.process.controller import ControllerProcess, controller_process
from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.process.streaming import StreamMetadata, StreamType
from rattlesnake.utilities import GlobalCommands
from mock_objects.mock_utilities import mock_queue_container, mock_event_container
import pytest
import multiprocessing as mp
from unittest import mock


# region: Fixtures
@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def controller(request):
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    controller = ControllerProcess(
        "Process Name",
        queue_container,
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.streaming_active_event,
        event_container.environment_active_events,
        event_container.controller_ready_event,
    )
    return controller


# region: ControllerProcess
# Test AcquisitonProcess initialization
@pytest.mark.parametrize("use_thread", [True, False])
def test_controller_init(use_thread):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    controller = ControllerProcess(
        "Process Name",
        queue_container,
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.streaming_active_event,
        event_container.environment_active_events,
        event_container.controller_ready_event,
    )

    # Make sure it is the correct class
    assert isinstance(controller, ControllerProcess)
    assert isinstance(controller, AbstractMessageProcess)


@pytest.mark.parametrize("stream_type", [*StreamType])
def test_controller_run_hardware(stream_type, controller):
    # Mock active acquistion
    mock_acquisition_active = mock.MagicMock()
    mock_acquisition_active.is_set.return_value = False
    mock_output_active = mock.MagicMock()
    mock_output_active.is_set.return_value = False
    controller._acquisition_active_event = mock_acquisition_active
    controller._output_active_event = mock_output_active
    # Mock command queues
    mock_acquisition = mock.MagicMock()
    mock_output = mock.MagicMock()
    controller.queue_container.acquisition_command_queue = mock_acquisition
    controller.queue_container.output_command_queue = mock_output
    controller.start_streaming = mock.MagicMock()
    # Build stream metadata
    stream_metadata = StreamMetadata()
    stream_metadata.stream_type = stream_type
    controller.run_hardware(stream_metadata)

    mock_acquisition.put.assert_called_once_with("Controller", (GlobalCommands.RUN_HARDWARE, None))
    mock_output.put.assert_called_once_with("Controller", (GlobalCommands.RUN_HARDWARE, None))
    if stream_type == StreamType.IMMEDIATELY:
        controller.start_streaming.assert_called_once_with(True)
    else:
        controller.start_streaming.assert_not_called()


@pytest.mark.parametrize(
    "acquisition_active, output_active, expected",
    [(False, False, True), (False, True, RuntimeError), (True, False, RuntimeError), (True, True, RuntimeError)],
)
def test_controller_run_hardware_error(acquisition_active, output_active, expected, controller):
    # Mock active acqusition
    mock_acquisition_active = mock.MagicMock()
    mock_acquisition_active.is_set.return_value = acquisition_active
    mock_output_active = mock.MagicMock()
    mock_output_active.is_set.return_value = output_active
    controller._acquisition_active_event = mock_acquisition_active
    controller._output_active_event = mock_output_active
    # Mock command queues
    mock_acquisition = mock.MagicMock()
    mock_output = mock.MagicMock()
    controller.queue_container.acquisition_command_queue = mock_acquisition
    controller.queue_container.output_command_queue = mock_output
    controller.start_streaming = mock.MagicMock()
    # Build stream metadata
    stream_metadata = StreamMetadata()
    stream_metadata.stream_type = StreamType.NO_STREAM

    if expected is RuntimeError:
        with pytest.raises(RuntimeError):
            controller.run_hardware(stream_metadata)
    else:
        controller.run_hardware(stream_metadata)
        mock_acquisition.put.assert_called_once_with("Controller", (GlobalCommands.RUN_HARDWARE, None))
        mock_output.put.assert_called_once_with("Controller", (GlobalCommands.RUN_HARDWARE, None))
        controller.start_streaming.assert_not_called()


@pytest.mark.parametrize(
    "acquisition_active, output_active, environment_active, expected",
    [
        (False, False, False, RuntimeError),
        (False, True, False, RuntimeError),
        (True, False, False, RuntimeError),
        (True, True, False, True),
        (True, True, True, True),
    ],
)
def test_controller_stop_hardware(acquisition_active, output_active, environment_active, expected, controller):
    mock_acquisition_active = mock.MagicMock()
    mock_acquisition_active.is_set.return_value = acquisition_active
    mock_output_active = mock.MagicMock()
    mock_output_active.is_set.return_value = output_active
    mock_environment_active = mock.MagicMock()
    mock_environment_active.is_set.return_value = environment_active
    controller._acquisition_active_event = mock_acquisition_active
    controller._output_active_event = mock_output_active
    controller._environment_active_event["Environment 0"] = mock_environment_active

    mock_acquisition = mock.MagicMock()
    mock_output = mock.MagicMock()
    controller.queue_container.acquisition_command_queue = mock_acquisition
    controller.queue_container.output_command_queue = mock_output
    mock_stop = mock.MagicMock()
    controller.stop_environment = mock_stop

    if expected is RuntimeError:
        with pytest.raises(RuntimeError):
            controller.stop_hardware(None)
    else:
        controller.stop_hardware(None)
        if environment_active:
            mock_stop.assert_called_with("Environment 0")
        mock_acquisition.put.assert_called_once_with("Controller", (GlobalCommands.STOP_HARDWARE, None))
        mock_output.put.assert_called_once_with("Controller", (GlobalCommands.STOP_HARDWARE, None))


@pytest.mark.parametrize(
    "environment_active, expected",
    [(False, True), (True, RuntimeError)],
)
def test_controller_start_environment(environment_active, expected, controller):
    mock_environment_active = mock.MagicMock()
    mock_environment_active.is_set.return_value = environment_active
    controller._environment_active_event["Environment 0"] = mock_environment_active

    queue_name = "Environment 0"
    mock_instruction = mock.MagicMock()
    mock_acquisition = mock.MagicMock()
    mock_output = mock.MagicMock()
    mock_environment = mock.MagicMock()
    controller.queue_container.acquisition_command_queue = mock_acquisition
    controller.queue_container.output_command_queue = mock_output
    controller.queue_container.environment_command_queues = {queue_name: mock_environment}

    if expected is RuntimeError:
        with pytest.raises(RuntimeError):
            controller.start_environment((queue_name, mock_instruction))
    else:
        controller.start_environment((queue_name, mock_instruction))
        mock_output.put.assert_called_once_with("Controller", (GlobalCommands.START_ENVIRONMENT, queue_name))
        mock_environment.put.assert_called_once_with("Controller", (GlobalCommands.START_ENVIRONMENT, mock_instruction))


@pytest.mark.parametrize(
    "environment_active, expected",
    [(False, RuntimeError), (True, True)],
)
def test_controller_stop_environment(environment_active, expected, controller):
    mock_environment_active = mock.MagicMock()
    mock_environment_active.is_set.return_value = environment_active
    controller._environment_active_event["Environment 0"] = mock_environment_active

    queue_name = "Environment 0"
    mock_environment = mock.MagicMock()
    controller.queue_container.environment_command_queues = {queue_name: mock_environment}

    if expected is RuntimeError:
        with pytest.raises(RuntimeError):
            controller.stop_environment(queue_name)
    else:
        controller.stop_environment(queue_name)
        mock_environment.put.assert_called_once_with("Controller", (GlobalCommands.STOP_ENVIRONMENT, None))


@pytest.mark.parametrize("stream_type", [*StreamType])
@pytest.mark.parametrize("override", [True, False, None])
def test_controller_start_streaming(stream_type, override, controller):
    mock_acquisition = mock.MagicMock()
    controller.queue_container.acquisition_command_queue = mock_acquisition
    controller.stream_metadata.stream_type = stream_type
    controller.start_streaming(override)

    if override:
        mock_acquisition.put.assert_called_once_with("Controller", (GlobalCommands.START_STREAMING, None))
    elif stream_type == StreamType.PROFILE_INSTRUCTION:
        mock_acquisition.put.assert_called_once_with("Controller", (GlobalCommands.START_STREAMING, None))
    else:
        mock_acquisition.put.assert_not_called()


def test_controller_stop_streaming(controller):
    mock_acquisition = mock.MagicMock()
    controller.queue_container.acquisition_command_queue = mock_acquisition
    controller.stop_streaming(None)

    mock_acquisition.put.assert_called_once_with("Controller", (GlobalCommands.STOP_STREAMING, None))


@pytest.mark.parametrize("stream_type", [*StreamType])
@pytest.mark.parametrize("environment_name", ["Environment 0", "Wrong Environment"])
def test_controller_at_target_level_match(stream_type, environment_name, controller):
    controller.stream_metadata.stream_type = stream_type
    controller.stream_metadata.test_level_environment_name = "Environment 0"
    controller.start_streaming = mock.MagicMock()
    controller.at_target_level(environment_name)

    if stream_type == StreamType.TEST_LEVEL and environment_name == "Environment 0":
        controller.start_streaming.assert_called_once_with(True)
    else:
        controller.start_streaming.assert_not_called()


def test_controller_profile_closeout(controller):
    controller.clear_ready()
    controller.profile_closeout(None)

    assert controller.ready_event.is_set()


# region: controller_process
# Prevent the run while loop from starting
@pytest.mark.parametrize("use_thread", [True, False])
@mock.patch("rattlesnake.process.controller.ControllerProcess")
def test_controller_process_func(mock_controller, use_thread):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    controller_process(
        queue_container,
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.streaming_active_event,
        event_container.environment_active_events,
        event_container.controller_ready_event,
        event_container.controller_close_event,
    )

    mock_instance = mock_controller.return_value
    mock_instance.run.assert_called()
