from rattlesnake.engine import RattlesnakeController, RattlesnakeState
from rattlesnake.utilities import GlobalCommands, RattlesnakeError
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.abstract_environment import EnvironmentMetadata
from rattlesnake.process.streaming import StreamMetadata
from mock_objects.mock_hardware import MockHardwareMetadata
from mock_objects.mock_environment import MockEnvironmentMetadata
import pytest
from unittest import mock


# region Fixtures
@pytest.fixture(
    params=[(True, True), (True, False), (False, True), (False, False)],
    ids=[
        "threaded, blocking",
        "threaded, non_blocking",
        "non_threaded, blocking",
        "non_threaded, non_blocking",
    ],
)
@mock.patch("rattlesnake.engine.mp.Process")
@mock.patch("rattlesnake.engine.threading.Thread")
@mock.patch("rattlesnake.engine.RattlesnakeController.wait_for_events")
def rattlesnake_package(mock_wait_event, mock_thread, mock_process, request):
    threaded, blocking = request.param
    rattlesnake = RattlesnakeController(threaded=threaded, timeout=1)
    if not blocking:
        rattlesnake.clear_blocking()
    return (rattlesnake, threaded, blocking)


# region Rattlesnake
@pytest.mark.parametrize("threaded", [True, False])
@pytest.mark.parametrize("blocking", [True, False])
@mock.patch("rattlesnake.engine.mp.Process")
@mock.patch("rattlesnake.engine.threading.Thread")
@mock.patch("rattlesnake.engine.RattlesnakeController.wait_for_events")
def test_rattlesnake_init(
    mock_wait_event, mock_thread, mock_process, threaded, blocking
):
    mock_wait_event.return_value = None
    rattlesnake = RattlesnakeController(threaded=threaded, timeout=1)
    if not blocking:
        rattlesnake.clear_blocking()

    assert isinstance(rattlesnake, RattlesnakeController)
    if blocking:
        mock_wait_event.assert_called()


@pytest.mark.parametrize(
    "hardware_metadata, environment_metadata, acquisition_active, output_active, environment_active, expected_state",
    [
        (None, {}, False, False, {}, RattlesnakeState.INIT),  # fallback state
        (MockHardwareMetadata(), {}, False, False, {}, RattlesnakeState.HARDWARE_STORE),
        (
            MockHardwareMetadata(),
            {"Environment 0": MockEnvironmentMetadata()},
            False,
            False,
            False,
            RattlesnakeState.ENVIRONMENT_STORE,
        ),
        (
            MockHardwareMetadata(),
            {"Environment 0": MockEnvironmentMetadata()},
            True,
            True,
            False,
            RattlesnakeState.HARDWARE_ACTIVE,
        ),
        (
            MockHardwareMetadata(),
            {"Environment 0": MockEnvironmentMetadata()},
            True,
            True,
            True,
            RattlesnakeState.ENVIRONMENT_ACTIVE,
        ),
    ],
)
def test_rattlesnake_state(
    hardware_metadata,
    environment_metadata,
    acquisition_active,
    output_active,
    environment_active,
    expected_state,
    rattlesnake_package,
):
    rattlesnake, threaded, blocking = rattlesnake_package
    rattlesnake.hardware_metadata = hardware_metadata
    rattlesnake.environment_metadata = environment_metadata

    if acquisition_active:
        rattlesnake.event_container.acquisition_active_event.set()
    if output_active:
        rattlesnake.event_container.output_active_event.set()
    if environment_active:
        rattlesnake.event_container.environment_active_events["Environment 0"].set()

    assert rattlesnake.state == expected_state


def test_rattlesnake_properties(rattlesnake_package):
    rattlesnake, threaded, blocking = rattlesnake_package

    assert rattlesnake.threaded == threaded
    assert rattlesnake.blocking == blocking
    assert rattlesnake.timeout == 1
    assert rattlesnake.hardware_metadata == None
    assert rattlesnake.environment_metadata == {}


@pytest.mark.parametrize(
    "ready_event_return, active_event_return, active_event_check, expected",
    [
        ([True, True], [True, True], True, True),
        ([True, True], [], True, True),
        ([], [True, True], True, True),
        ([True, True], [False, True], True, TimeoutError),
        ([False, True], [True, True], True, TimeoutError),
        ([True, True], [False, False], False, True),
        ([True, True], [], False, True),
        ([], [False, False], False, True),
        ([True, True], [False, True], False, TimeoutError),
        ([False, True], [True, True], False, TimeoutError),
        ([True, True], [True, True], None, TimeoutError),
        ([True, True], [], None, True),
        ([], [True, True], None, TimeoutError),
    ],
)
@mock.patch("rattlesnake.engine.time.time")
def test_rattlesnake_wait_for_events(
    mock_time,
    ready_event_return,
    active_event_return,
    active_event_check,
    expected,
    rattlesnake_package,
):
    rattlesnake, threaded, blocking = rattlesnake_package
    rattlesnake._timeout = 1
    mock_time.side_effect = [0, 2]

    ready_event_list = []
    for return_value in ready_event_return:
        mock_ready_event = mock.MagicMock()
        mock_ready_event.is_set.return_value = return_value
        ready_event_list.append(mock_ready_event)

    active_event_list = []
    for return_value in active_event_return:
        mock_active_event = mock.MagicMock()
        mock_active_event.is_set.return_value = return_value
        active_event_list.append(mock_active_event)

    if expected is TimeoutError:
        with pytest.raises(TimeoutError):
            rattlesnake.wait_for_events(
                ready_event_list,
                active_event_list,
                active_event_check=active_event_check,
            )
        # Only on timeout should ready_event.set() be called
        for ready_event in ready_event_list:
            ready_event.set.assert_called()
    else:
        rattlesnake.wait_for_events(
            ready_event_list, active_event_list, active_event_check=active_event_check
        )
        # On success, ready_event.set() should NOT be called
        for ready_event in ready_event_list:
            ready_event.set.assert_not_called()


@pytest.mark.parametrize(
    "state, instance, expected",
    [
        (RattlesnakeState.INIT, None, RattlesnakeError),
        (RattlesnakeState.INIT, HardwareMetadata, True),
        (RattlesnakeState.HARDWARE_STORE, HardwareMetadata, True),
        (RattlesnakeState.ENVIRONMENT_STORE, HardwareMetadata, True),
        (RattlesnakeState.HARDWARE_ACTIVE, HardwareMetadata, RattlesnakeError),
        (RattlesnakeState.ENVIRONMENT_ACTIVE, HardwareMetadata, RattlesnakeError),
    ],
)
def test_rattlesnake_initialize_hardware(
    state, instance, expected, rattlesnake_package
):
    rattlesnake, threaded, blocking = rattlesnake_package
    mock_wait_event = mock.MagicMock()
    mock_environment_manager = mock.MagicMock()
    mock_acquisiton = mock.MagicMock()
    mock_output = mock.MagicMock()
    rattlesnake.wait_for_events = mock_wait_event
    rattlesnake.environment_manager = mock_environment_manager
    rattlesnake.queue_container.acquisition_command_queue = mock_acquisiton
    rattlesnake.queue_container.output_command_queue = mock_output

    hardware_metadata = mock.MagicMock(spec=instance)

    with mock.patch.object(
        RattlesnakeController, "state", new_callable=mock.PropertyMock
    ) as mock_state:
        mock_state.return_value = state

        if expected == RattlesnakeError:
            with pytest.raises(RattlesnakeError):
                rattlesnake.initialize_hardware(hardware_metadata)
        else:
            rattlesnake.initialize_hardware(hardware_metadata)

            hardware_metadata.validate.assert_called()
            mock_environment_manager.initialize_hardware.assert_called_with(
                hardware_metadata
            )
            mock_acquisiton.put.assert_called_with(
                "Rattlesnake", (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata)
            )
            mock_output.put.assert_called_with(
                "Rattlesnake", (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata)
            )
            if blocking:
                mock_wait_event.assert_called()


@pytest.mark.parametrize(
    "state, expected",
    [
        (RattlesnakeState.INIT, RuntimeError),
        (RattlesnakeState.HARDWARE_STORE, True),
        (RattlesnakeState.ENVIRONMENT_STORE, True),
        (RattlesnakeState.HARDWARE_ACTIVE, RuntimeError),
        (RattlesnakeState.ENVIRONMENT_ACTIVE, RuntimeError),
    ],
)
def test_rattlesnake_set_environment(state, expected, rattlesnake_package):
    rattlesnake, threaded, blocking = rattlesnake_package
    mock_wait_event = mock.MagicMock()
    mock_environment_manager = mock.MagicMock()
    mock_acquisiton = mock.MagicMock()
    mock_output = mock.MagicMock()
    rattlesnake.wait_for_events = mock_wait_event
    rattlesnake.environment_manager = mock_environment_manager
    rattlesnake.queue_container.acquisition_command_queue = mock_acquisiton
    rattlesnake.queue_container.output_command_queue = mock_output

    environment_metadata = mock.MagicMock(spec=EnvironmentMetadata)
    environment_metadata_list = [environment_metadata]

    with mock.patch.object(
        RattlesnakeController, "state", new_callable=mock.PropertyMock
    ) as mock_state:
        mock_state.return_value = state

        if expected is RuntimeError:
            with pytest.raises(RuntimeError):
                rattlesnake.initialize_environments(environment_metadata_list)
        else:
            rattlesnake.initialize_environments(environment_metadata_list)

            mock_environment_manager.validate_environment_metadata.assert_called_with(
                environment_metadata_list
            )
            mock_environment_manager.initialize_environments.assert_called()
            mock_acquisiton.put.assert_called_with(
                "Rattlesnake",
                (
                    GlobalCommands.INITIALIZE_ENVIRONMENT,
                    rattlesnake.environment_metadata,
                ),
            )
            mock_output.put.assert_called_with(
                "Rattlesnake",
                (
                    GlobalCommands.INITIALIZE_ENVIRONMENT,
                    rattlesnake.environment_metadata,
                ),
            )
            if blocking:
                mock_wait_event.assert_called()


def test_rattlesnake_set_empty_environment(rattlesnake_package):
    rattlesnake, threaded, blocking = rattlesnake_package
    mock_wait_event = mock.MagicMock()
    mock_environment_manager = mock.MagicMock()
    mock_environment_manager.environment_metadata = {}
    mock_acquisiton = mock.MagicMock()
    mock_output = mock.MagicMock()
    rattlesnake.wait_for_events = mock_wait_event
    rattlesnake.environment_manager = mock_environment_manager
    rattlesnake.queue_container.acquisition_command_queue = mock_acquisiton
    rattlesnake.queue_container.output_command_queue = mock_output

    environment_metadata_list = []

    with mock.patch.object(
        RattlesnakeController, "state", new_callable=mock.PropertyMock
    ) as mock_state:
        mock_state.return_value = RattlesnakeState.ENVIRONMENT_STORE

        rattlesnake.initialize_environments(environment_metadata_list)

        mock_environment_manager.validate_environment_metadata.assert_called_with(
            environment_metadata_list, None
        )
        mock_environment_manager.initialize_environments.assert_called()

        mock_acquisiton.put.assert_called_with(
            "Rattlesnake",
            (GlobalCommands.INITIALIZE_ENVIRONMENT, rattlesnake.environment_metadata),
        )
        mock_output.put.assert_called_with(
            "Rattlesnake",
            (GlobalCommands.INITIALIZE_ENVIRONMENT, rattlesnake.environment_metadata),
        )
        if blocking:
            mock_wait_event.assert_called()


@pytest.mark.parametrize(
    "state, instance, expected",
    [
        (RattlesnakeState.INIT, StreamMetadata, RuntimeError),
        (RattlesnakeState.HARDWARE_STORE, StreamMetadata, RuntimeError),
        (RattlesnakeState.ENVIRONMENT_STORE, StreamMetadata, True),
        (RattlesnakeState.ENVIRONMENT_STORE, None, TypeError),
        (RattlesnakeState.HARDWARE_ACTIVE, StreamMetadata, RuntimeError),
        (RattlesnakeState.ENVIRONMENT_ACTIVE, StreamMetadata, RuntimeError),
    ],
)
def test_rattlesnake_start_acquisition(state, instance, expected, rattlesnake_package):
    rattlesnake, threaded, blocking = rattlesnake_package
    mock_wait_event = mock.MagicMock()
    mock_streaming = mock.MagicMock()
    mock_controller = mock.MagicMock()
    rattlesnake.wait_for_events = mock_wait_event
    rattlesnake.queue_container.streaming_command_queue = mock_streaming
    rattlesnake.queue_container.controller_command_queue = mock_controller

    stream_metadata = mock.MagicMock(spec=instance)

    with mock.patch.object(
        RattlesnakeController, "state", new_callable=mock.PropertyMock
    ) as mock_state:
        mock_state.return_value = state

        if expected == RuntimeError:
            with pytest.raises(RuntimeError):
                rattlesnake.start_acquisition(stream_metadata)
        elif expected == TypeError:
            with pytest.raises(TypeError):
                rattlesnake.start_acquisition(stream_metadata)
        else:
            rattlesnake.start_acquisition(stream_metadata)

            stream_metadata.validate.assert_called()
            mock_streaming.put.assert_called_with(
                "Rattlesnake",
                (
                    GlobalCommands.INITIALIZE_STREAMING,
                    (
                        stream_metadata,
                        rattlesnake.hardware_metadata,
                        rattlesnake.environment_metadata,
                    ),
                ),
            )
            mock_controller.put.assert_called_with(
                "Rattlesnake", (GlobalCommands.RUN_HARDWARE, stream_metadata)
            )
            if blocking:
                mock_wait_event.assert_called()


@pytest.mark.parametrize(
    "state, expected",
    [
        (RattlesnakeState.INIT, RuntimeError),
        (RattlesnakeState.HARDWARE_STORE, RuntimeError),
        (RattlesnakeState.ENVIRONMENT_STORE, RuntimeError),
        (RattlesnakeState.HARDWARE_ACTIVE, True),
        (RattlesnakeState.ENVIRONMENT_ACTIVE, RuntimeError),
    ],
)
def test_rattlesnake_start_profile(state, expected, rattlesnake_package):
    rattlesnake, threaded, blocking = rattlesnake_package
    mock_profile = mock.MagicMock()
    profile_event_list = [mock_profile]

    mock_wait_event = mock.MagicMock()
    mock_environment_manager = mock.MagicMock()
    mock_profile_manager = mock.MagicMock()
    rattlesnake.wait_for_events = mock_wait_event
    rattlesnake.environment_manager = mock_environment_manager
    rattlesnake.profile_manager = mock_profile_manager

    with mock.patch.object(
        RattlesnakeController, "state", new_callable=mock.PropertyMock
    ) as mock_state:
        mock_state.return_value = state

        if expected is RuntimeError:
            with pytest.raises(RuntimeError):
                rattlesnake.start_profile(profile_event_list)
        else:
            rattlesnake.start_profile(profile_event_list)
            mock_environment_manager.validate_profile_events.assert_called_with(
                profile_event_list
            )
            mock_profile_manager.validate_profile_list(profile_event_list)
            mock_profile_manager.start_profile.assert_called_with(profile_event_list)


@pytest.mark.parametrize(
    "first_alive, second_alive", [(False, False), (True, False), (True, True)]
)
@mock.patch("rattlesnake.engine.flush_queue")
def test_rattlesnake_shutdown(
    mock_flush, first_alive, second_alive, rattlesnake_package
):
    rattlesnake, threaded, blocking = rattlesnake_package
    mock_log_file_queue = mock.MagicMock()
    mock_controller_queue = mock.MagicMock()
    mock_acquisition_queue = mock.MagicMock()
    mock_output_queue = mock.MagicMock()
    mock_streaming_queue = mock.MagicMock()
    mock_log_file = mock.MagicMock()
    mock_controller = mock.MagicMock()
    mock_controller.is_alive.side_effect = [first_alive, second_alive]
    mock_acquisition = mock.MagicMock()
    mock_acquisition.is_alive.side_effect = [first_alive, second_alive]
    mock_output = mock.MagicMock()
    mock_output.is_alive.side_effect = [first_alive, second_alive]
    mock_streaming = mock.MagicMock()
    mock_streaming.is_alive.side_effect = [first_alive, second_alive]
    mock_environment_manager = mock.MagicMock()

    rattlesnake.queue_container.log_file_queue = mock_log_file_queue
    rattlesnake.queue_container.controller_command_queue = mock_controller_queue
    rattlesnake.queue_container.acquisition_command_queue = mock_acquisition_queue
    rattlesnake.queue_container.output_command_queue = mock_output_queue
    rattlesnake.queue_container.streaming_command_queue = mock_streaming_queue
    rattlesnake.log_file_process = mock_log_file
    rattlesnake.controller_proc = mock_controller
    rattlesnake.acquisition_proc = mock_acquisition
    rattlesnake.output_proc = mock_output
    rattlesnake.streaming_proc = mock_streaming
    rattlesnake.environment_manager = mock_environment_manager

    with mock.patch.object(
        RattlesnakeController, "state", new_callable=mock.PropertyMock
    ) as mock_state:
        mock_state.return_value = RattlesnakeState.ENVIRONMENT_ACTIVE
        mock_stop = mock.MagicMock()
        rattlesnake.stop_acquisition = mock_stop

        rattlesnake.shutdown()

        mock_stop.assert_called()

        mock_controller_queue.put.assert_called_with(
            "Rattlesnake", (GlobalCommands.QUIT, None)
        )
        mock_acquisition_queue.put.assert_called_with(
            "Rattlesnake", (GlobalCommands.QUIT, None)
        )
        mock_output_queue.put.assert_called_with(
            "Rattlesnake", (GlobalCommands.QUIT, None)
        )
        mock_streaming_queue.put.assert_called_with(
            "Rattlesnake", (GlobalCommands.QUIT, None)
        )

        mock_controller.join.assert_called()
        mock_acquisition.join.assert_called()
        mock_output.join.assert_called()
        mock_streaming.join.assert_called()
        mock_environment_manager.close_environments.assert_called()
        mock_log_file.join.assert_called()

        if first_alive:
            assert rattlesnake.event_container.controller_close_event.is_set()
            assert rattlesnake.event_container.acquisition_close_event.is_set()
            assert rattlesnake.event_container.output_close_event.is_set()
            assert rattlesnake.event_container.streaming_close_event.is_set()

        if first_alive and second_alive and not threaded:
            mock_controller.terminate.assert_called()
            mock_acquisition.terminate.assert_called()
            mock_output.terminate.assert_called()
            mock_streaming.terminate.assert_called()
