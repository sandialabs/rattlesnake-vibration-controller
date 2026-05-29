from rattlesnake.profile_manager import (
    ProfileManager,
    ProfileEvent,
    EXTRA_CLOSEOUT_TIME,
)
from rattlesnake.utilities import GlobalCommands
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.abstract_environment import EnvironmentInstructions
from rattlesnake.environment.time_environment import TimeCommands
from mock_objects.mock_utilities import mock_queue_container, fake_time
import pytest
from unittest import mock


# region: Fixtures
@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def profile_manager(request):
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    profile_manager = ProfileManager(queue_container)
    return profile_manager


# region: ProfileEvent
def test_profile_event_init():
    timestamp = 0
    environment_name = "Global"
    command = GlobalCommands.START_STREAMING
    profile_event = ProfileEvent(timestamp, environment_name, command)

    assert isinstance(profile_event, ProfileEvent)


def test_profile_event_properties():
    timestamp = 0
    environment_name = "Global"
    command = GlobalCommands.START_STREAMING
    profile_event = ProfileEvent(timestamp, environment_name, command)
    profile_event._environment_type = EnvironmentType.TIME
    profile_event._queue_name = "Environment 0"

    assert profile_event.environment_type == EnvironmentType.TIME
    assert profile_event.queue_name == "Environment 0"


@pytest.mark.parametrize(
    "timestamp, environment_name, command, environment_type, queue_name, expected",
    [
        (
            0,
            "Environment Name",
            GlobalCommands.START_STREAMING,
            "Global",
            "Global",
            True,
        ),
        (
            0,
            "Environment Name",
            GlobalCommands.START_ENVIRONMENT,
            EnvironmentType.TIME,
            "Environment 0",
            TypeError,
        ),
        (
            -10,
            "Environment Name",
            GlobalCommands.START_STREAMING,
            "Global",
            "Global",
            ValueError,
        ),
        (
            "timestamp",
            "Environment Name",
            GlobalCommands.START_STREAMING,
            "Global",
            "Global",
            ValueError,
        ),
        (0, 10, GlobalCommands.START_STREAMING, "Global", "Global", TypeError),
        (
            0,
            "Environment Name",
            GlobalCommands.START_ENVIRONMENT,
            "Global",
            "Global",
            TypeError,
        ),
        (
            0,
            "Environment Name",
            GlobalCommands.START_STREAMING,
            EnvironmentType.TIME,
            "Environment 0",
            TypeError,
        ),
        (
            0,
            "Environment Name",
            GlobalCommands.START_STREAMING,
            "Not a environment",
            "Global",
            TypeError,
        ),
        (0, "Environment Name", "Not a command", "Global", "Global", TypeError),
        (
            0,
            "Environment Name",
            GlobalCommands.START_STREAMING,
            "Global",
            None,
            ValueError,
        ),
        (
            0,
            "Environment Name",
            TimeCommands.SET_TEST_LEVEL,
            EnvironmentType.TIME,
            "Environment 0",
            TypeError,
        ),
    ],
)
def test_profile_event_validate(
    timestamp, environment_name, command, environment_type, queue_name, expected
):
    profile_event = ProfileEvent(timestamp, environment_name, command)
    profile_event._environment_type = environment_type
    profile_event._queue_name = queue_name

    if expected is TypeError:
        with pytest.raises(TypeError):
            profile_event.validate()
    elif expected is ValueError:
        with pytest.raises(ValueError):
            profile_event.validate()
    else:
        valid_profile = profile_event.validate()
        assert valid_profile == True


# region: ProfileManager
@pytest.mark.parametrize("use_thread", [True, False])
def test_profile_manager_init(use_thread):
    queue_container = mock_queue_container(use_thread)
    profile_manager = ProfileManager(queue_container)

    assert isinstance(profile_manager, ProfileManager)


def test_profile_manager_properties(profile_manager):
    mock_log_file_queue = mock.MagicMock()
    mock_controller = mock.MagicMock()
    profile_manager._log_file_queue = mock_log_file_queue
    profile_manager._controller_command_queue = mock_controller

    assert profile_manager.log_file_queue == mock_log_file_queue
    assert profile_manager._controller_command_queue == mock_controller


@pytest.mark.parametrize(
    "profile_event_list, profile_queue_names, valid_list, expected",
    [
        ([], [], [], True),
        (
            [ProfileEvent(0, "Global", GlobalCommands.START_STREAMING)],
            ["Global"],
            [True],
            True,
        ),
        (
            [ProfileEvent(0, "Environment Name", TimeCommands.SET_TEST_LEVEL)],
            ["Environment 0"],
            [True],
            True,
        ),
        (
            [None],
            ["Global"],
            [True],
            TypeError,
        ),
        (
            [ProfileEvent(0, "Environment Name", TimeCommands.SET_TEST_LEVEL)],
            ["Environment 0"],
            [False],
            ValueError,
        ),
        (
            [ProfileEvent(0, "Global", "Not a command")],
            ["Global"],
            [True],
            KeyError,
        ),
    ],
)
def test_profile_manager_validate_profile_list(
    profile_event_list, profile_queue_names, valid_list, expected, profile_manager
):
    for profile_event, queue_name, valid in zip(
        profile_event_list, profile_queue_names, valid_list
    ):
        if isinstance(profile_event, ProfileEvent):
            mock_valid = mock.MagicMock()
            mock_valid.return_value = valid
            profile_event.validate = mock_valid
            profile_event._queue_name = queue_name

    if expected is KeyError:
        with pytest.raises(KeyError):
            profile_manager.validate_profile_list(profile_event_list)
    elif expected is TypeError:
        with pytest.raises(TypeError):
            profile_manager.validate_profile_list(profile_event_list)
    elif expected is ValueError:
        with pytest.raises(ValueError):
            profile_manager.validate_profile_list(profile_event_list)
    else:
        valid_profile_event = profile_manager.validate_profile_list(profile_event_list)
        assert valid_profile_event


@mock.patch("rattlesnake.profile_manager.threading.Timer")
def test_profile_manager_start_profile(mock_timer, profile_manager):
    global_event = ProfileEvent(0, "Global", GlobalCommands.START_STREAMING)
    global_event._queue_name = "Global"
    environment_event = ProfileEvent(2, "Environment Name", TimeCommands.SET_NO_REPEAT)
    environment_event._queue_name = "Environment 0"
    start_event = ProfileEvent(2, "Environment Name", GlobalCommands.START_ENVIRONMENT)
    start_event._queue_name = "Environment 0"
    profile_event_list = [global_event, environment_event, start_event]
    profile_manager.start_profile(profile_event_list)

    expected_calls = [
        mock.call(
            global_event.timestamp,
            profile_manager.fire_profile_event,
            args=(global_event.queue_name, global_event.command, global_event.data),
        ),
        mock.call(
            environment_event.timestamp,
            profile_manager.fire_profile_event,
            args=(
                environment_event.queue_name,
                environment_event.command,
                environment_event.data,
            ),
        ),
        mock.call(
            start_event.timestamp,
            profile_manager.fire_profile_event,
            args=(start_event.queue_name, start_event.command, start_event.data),
        ),
        mock.call(
            start_event.timestamp + EXTRA_CLOSEOUT_TIME,
            profile_manager.fire_closeout_event,
        ),
    ]

    assert mock_timer.call_args_list == expected_calls


def test_profile_manager_fire_profile_event(profile_manager):
    mock_function = mock.MagicMock()
    profile_manager.command_map = {GlobalCommands.START_STREAMING: mock_function}

    profile_manager.fire_profile_event(
        "Global",
        GlobalCommands.START_STREAMING,
        None,
    )

    mock_function.assert_called_once_with(
        "Global", GlobalCommands.START_STREAMING, None
    )


@mock.patch("rattlesnake.profile_manager.threading.Timer")
def test_profile_manager_stop_profile(mock_timer, profile_manager):
    timer_1 = mock.MagicMock()
    timer_2 = mock.MagicMock()
    profile_manager.profile_timers = [timer_1, timer_2]
    profile_manager.stop_profile()

    timer_1.cancel.assert_called_once()
    timer_2.cancel.assert_called_once()
    mock_timer.assert_called_once_with(
        EXTRA_CLOSEOUT_TIME,
        profile_manager.fire_closeout_event,
    )

    mock_timer.return_value.start.assert_called_once()
    assert mock_timer.return_value in profile_manager.profile_timers


def test_profile_manager_stop_hardware(profile_manager):
    instructions = EnvironmentInstructions("Environment Type", "Environment 0")
    profile_manager.environment_instructions = {"Environment 0": instructions}
    mock_controller = mock.MagicMock()
    profile_manager._controller_command_queue = mock_controller
    profile_manager.stop_hardware("Global", GlobalCommands.STOP_HARDWARE, None)

    mock_controller.put.assert_called_with(
        "Profile Manager", (GlobalCommands.STOP_HARDWARE, None)
    )


def test_profile_manager_start_streaming(profile_manager):
    mock_controller = mock.MagicMock()
    profile_manager._controller_command_queue = mock_controller
    profile_manager.start_streaming("Global", GlobalCommands.START_STREAMING, None)

    mock_controller.put.assert_called_once_with(
        "Profile Manager", (GlobalCommands.START_STREAMING, False)
    )


def test_profile_manager_stop_streaming(profile_manager):
    mock_controller = mock.MagicMock()
    profile_manager._controller_command_queue = mock_controller
    profile_manager.stop_streaming("Global", GlobalCommands.STOP_STREAMING, None)

    mock_controller.put.assert_called_once_with(
        "Profile Manager", (GlobalCommands.STOP_STREAMING, None)
    )


def test_start_environment(profile_manager):
    instructions = EnvironmentInstructions("Environment Type", "Environment 0")
    mock_controller = mock.MagicMock()
    profile_manager._controller_command_queue = mock_controller
    profile_manager.start_environment(
        "Environment 0", GlobalCommands.START_ENVIRONMENT, instructions
    )

    mock_controller.put.assert_called_once_with(
        "Profile Manager",
        (GlobalCommands.START_ENVIRONMENT, ("Environment 0", instructions)),
    )


def test_stop_environment(profile_manager):
    mock_controller = mock.MagicMock()
    profile_manager._controller_command_queue = mock_controller
    profile_manager.stop_environment(
        "Environment 0", GlobalCommands.STOP_ENVIRONMENT, None
    )

    mock_controller.put.assert_called_once_with(
        "Profile Manager", (GlobalCommands.STOP_ENVIRONMENT, "Environment 0")
    )


def test_fire_closeout_event(profile_manager):
    mock_controller = mock.MagicMock()
    profile_manager._controller_command_queue = mock_controller
    profile_manager.fire_closeout_event()

    mock_controller.put.assert_called_once_with(
        "Profile Manager", (GlobalCommands.PROFILE_CLOSEOUT, None)
    )


@mock.patch("rattlesnake.profile_manager.datetime")
def test_profile_manager_log(mock_time, profile_manager):
    mock_log_file_queue = mock.MagicMock()
    profile_manager._log_file_queue = mock_log_file_queue
    mock_time.now = fake_time
    profile_manager.log("Test Message")

    mock_log_file_queue.put.assert_called_once_with(
        "Datetime: Profile Manager -- Test Message\n"
    )
