from unittest import mock

import pytest

from rattlesnake.environment.abstract_environment import EnvironmentInstructions
from rattlesnake.environment.environment_registry import ENVIRONMENT_COMMANDS
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.time_environment import TimeCommands
from rattlesnake.environment.skeleton_environment import SkeletonCommands
from rattlesnake.profile_manager import (
    EXTRA_CLOSEOUT_TIME,
    TASK_NAME,
    VALID_COMMANDS,
    VALID_DATA,
    ProfileEvent,
    ProfileManager,
)
from rattlesnake.testing.mock_utilities import (
    fake_time,
    mock_queue_container,
    skeleton_environment_instructions,
)
from rattlesnake.user_interface.ui_utilities import UICommands
from rattlesnake.utilities import GlobalCommands, RattlesnakeError


# region Fixtures
@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def profile_manager(request):
    """
    Create a ``ProfileManager`` in threaded and multiprocessing queue modes.
    """
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)

    return ProfileManager(queue_container)


# endregion


# region Profile Constants
def test_profile_constants():
    """
    Verifies key profile manager constants.
    """
    assert EXTRA_CLOSEOUT_TIME > 0
    assert TASK_NAME == "Profile Manager"


def test_valid_commands_contains_global_commands():
    """
    Verifies that global profile commands are registered.
    """
    assert "Global" in VALID_COMMANDS
    assert GlobalCommands.STOP_HARDWARE in VALID_COMMANDS["Global"]
    assert GlobalCommands.START_STREAMING in VALID_COMMANDS["Global"]
    assert GlobalCommands.STOP_STREAMING in VALID_COMMANDS["Global"]


def test_valid_commands_contains_environment_commands():
    """
    Verifies that registered environment command classes contribute valid
    profile commands.
    """
    for environment_type, command_class in ENVIRONMENT_COMMANDS.items():
        assert environment_type in VALID_COMMANDS
        assert GlobalCommands.START_ENVIRONMENT in VALID_COMMANDS[environment_type]
        assert GlobalCommands.STOP_ENVIRONMENT in VALID_COMMANDS[environment_type]
        assert (
            UICommands.SET_ENVIRONMENT_INSTRUCTIONS in VALID_COMMANDS[environment_type]
        )

        for command in command_class.valid_profile_commands():
            assert command in VALID_COMMANDS[environment_type]


def test_valid_data_contains_global_data_requirements():
    """
    Verifies that global profile commands have expected data requirements.
    """
    assert VALID_DATA[GlobalCommands.STOP_HARDWARE] is type(None)
    assert VALID_DATA[GlobalCommands.START_STREAMING] is type(None)
    assert VALID_DATA[GlobalCommands.STOP_STREAMING] is type(None)
    assert VALID_DATA[GlobalCommands.START_ENVIRONMENT] is EnvironmentInstructions
    assert VALID_DATA[GlobalCommands.STOP_ENVIRONMENT] is type(None)
    assert VALID_DATA[UICommands.SET_ENVIRONMENT_INSTRUCTIONS] is (
        EnvironmentInstructions
    )


# endregion


# region ProfileEvent
def test_profile_event_init():
    """
    Verifies that a profile event can be initialized.
    """
    profile_event = ProfileEvent(
        timestamp=0,
        environment_name="Global",
        command=GlobalCommands.START_STREAMING,
    )

    assert isinstance(profile_event, ProfileEvent)
    assert profile_event.timestamp == 0
    assert profile_event.environment_name == "Global"
    assert profile_event.command == GlobalCommands.START_STREAMING
    assert profile_event.data is None
    assert profile_event.environment_type is None
    assert profile_event.queue_name is None


def test_profile_event_properties():
    """
    Verifies that assigned environment type and queue name properties return
    expected values.
    """
    profile_event = ProfileEvent(
        timestamp=0,
        environment_name="Global",
        command=GlobalCommands.START_STREAMING,
    )
    profile_event._environment_type = EnvironmentType.TIME
    profile_event._queue_name = "Environment 0"

    assert profile_event.environment_type == EnvironmentType.TIME
    assert profile_event.queue_name == "Environment 0"


@pytest.mark.parametrize(
    "timestamp, environment_name, command, data, environment_type, queue_name, expected",
    [
        (
            0,
            "Global",
            GlobalCommands.START_STREAMING,
            None,
            "Global",
            "Global",
            True,
        ),
        (
            0,
            "Global",
            GlobalCommands.STOP_STREAMING,
            None,
            "Global",
            "Global",
            True,
        ),
        (
            0,
            "Global",
            GlobalCommands.STOP_HARDWARE,
            None,
            "Global",
            "Global",
            True,
        ),
        (
            -1,
            "Global",
            GlobalCommands.START_STREAMING,
            None,
            "Global",
            "Global",
            RattlesnakeError,
        ),
        (
            "bad timestamp",
            "Global",
            GlobalCommands.START_STREAMING,
            None,
            "Global",
            "Global",
            RattlesnakeError,
        ),
        (
            0,
            123,
            GlobalCommands.START_STREAMING,
            None,
            "Global",
            "Global",
            RattlesnakeError,
        ),
        (
            0,
            "Global",
            GlobalCommands.START_ENVIRONMENT,
            None,
            "Global",
            "Global",
            RattlesnakeError,
        ),
        (
            0,
            "Global",
            GlobalCommands.START_STREAMING,
            None,
            "Not an environment type",
            "Global",
            RattlesnakeError,
        ),
        (
            0,
            "Global",
            "Not a command",
            None,
            "Global",
            "Global",
            RattlesnakeError,
        ),
        (
            0,
            "Global",
            GlobalCommands.START_STREAMING,
            None,
            "Global",
            None,
            RattlesnakeError,
        ),
        (
            0,
            "Global",
            GlobalCommands.START_STREAMING,
            "invalid data",
            "Global",
            "Global",
            RattlesnakeError,
        ),
    ],
)
def test_profile_event_validate_global_events(
    timestamp,
    environment_name,
    command,
    data,
    environment_type,
    queue_name,
    expected,
):
    """
    Verifies global profile event validation for valid and invalid timestamps,
    environment names, commands, queue names, environment types, and data
    types.
    """
    profile_event = ProfileEvent(timestamp, environment_name, command, data)
    profile_event._environment_type = environment_type
    profile_event._queue_name = queue_name

    if expected is RattlesnakeError:
        with pytest.raises(RattlesnakeError):
            profile_event.validate()
    else:
        profile_event.validate()


def test_profile_event_validate_start_environment_instruction():
    """
    Verifies that a start-environment profile event accepts matching
    environment instructions.
    """
    instructions = skeleton_environment_instructions(environment_name="Mock Environment")

    profile_event = ProfileEvent(
        timestamp=0,
        environment_name="Mock Environment",
        command=GlobalCommands.START_ENVIRONMENT,
        data=instructions,
    )
    profile_event._environment_type = instructions.environment_type
    profile_event._queue_name = "Environment 0"

    profile_event.validate()


def test_profile_event_validate_start_environment_instruction_wrong_name():
    """
    Verifies that environment instruction data must match the profile event
    environment name.
    """
    instructions = skeleton_environment_instructions(environment_name="Other Environment")

    profile_event = ProfileEvent(
        timestamp=0,
        environment_name="Mock Environment",
        command=GlobalCommands.START_ENVIRONMENT,
        data=instructions,
    )
    profile_event._environment_type = instructions.environment_type
    profile_event._queue_name = "Environment 0"

    with pytest.raises(RattlesnakeError):
        profile_event.validate()


def test_profile_event_validate_start_environment_instruction_wrong_type():
    """
    Verifies that environment instruction data must match the profile event
    environment type.
    """
    instructions = skeleton_environment_instructions(environment_name="Mock Environment")

    profile_event = ProfileEvent(
        timestamp=0,
        environment_name="Mock Environment",
        command=GlobalCommands.START_ENVIRONMENT,
        data=instructions,
    )
    profile_event._environment_type = EnvironmentType.TIME
    profile_event._queue_name = "Environment 0"

    with pytest.raises(RattlesnakeError):
        profile_event.validate()


def test_profile_event_validate_environment_profile_command():
    """
    Verifies that an environment-specific profile command validates when its
    data type matches the environment command definition.
    """
    profile_event = ProfileEvent(
        timestamp=0,
        environment_name="Mock Environment",
        command=SkeletonCommands.EXAMPLE_SET_TEST_LEVEL,
        data=None,
    )
    profile_event._environment_type = EnvironmentType.SKELETON
    profile_event._queue_name = "Environment 0"

    profile_event.validate()


def test_profile_event_validate_environment_profile_command_wrong_data():
    """
    Verifies that environment-specific profile command data types are checked.
    """
    profile_event = ProfileEvent(
        timestamp=0,
        environment_name="Mock Environment",
        command=SkeletonCommands.EXAMPLE_SET_TEST_LEVEL,
        data="not none",
    )
    profile_event._environment_type = EnvironmentType.SKELETON
    profile_event._queue_name = "Environment 0"

    with pytest.raises(RattlesnakeError):
        profile_event.validate()


# endregion


# region ProfileManager Initialization and Properties
@pytest.mark.parametrize("use_thread", [True, False])
def test_profile_manager_init(use_thread):
    """
    Verifies that a profile manager can be initialized.
    """
    queue_container = mock_queue_container(use_thread)
    profile_manager = ProfileManager(queue_container)

    assert isinstance(profile_manager, ProfileManager)
    assert profile_manager.log_file_queue is queue_container.log_file_queue
    assert (
        profile_manager.controller_command_queue
        is queue_container.controller_command_queue
    )
    assert profile_manager.profile_timers == []
    assert profile_manager.gui_timer is None


def test_profile_manager_properties(profile_manager):
    """
    Verifies that queue properties return expected queues.
    """
    mock_log_file_queue = mock.MagicMock()
    mock_controller_command_queue = mock.MagicMock()

    profile_manager._log_file_queue = mock_log_file_queue
    profile_manager._controller_command_queue = mock_controller_command_queue

    assert profile_manager.log_file_queue is mock_log_file_queue
    assert profile_manager.controller_command_queue is mock_controller_command_queue


def test_profile_manager_command_map(profile_manager):
    """
    Verifies that controller-level and environment-level profile commands are
    mapped to handlers.
    """
    assert profile_manager.command_map[GlobalCommands.STOP_HARDWARE] == (
        profile_manager.stop_hardware
    )
    assert profile_manager.command_map[GlobalCommands.START_STREAMING] == (
        profile_manager.start_streaming
    )
    assert profile_manager.command_map[GlobalCommands.STOP_STREAMING] == (
        profile_manager.stop_streaming
    )
    assert profile_manager.command_map[GlobalCommands.START_ENVIRONMENT] == (
        profile_manager.start_environment
    )
    assert profile_manager.command_map[GlobalCommands.STOP_ENVIRONMENT] == (
        profile_manager.stop_environment
    )

    assert profile_manager.command_map[SkeletonCommands.EXAMPLE_RUN_ENVIRONMENT] == (
        profile_manager.send_environment_command
    )
    assert profile_manager.command_map[SkeletonCommands.EXAMPLE_SET_TEST_LEVEL] == (
        profile_manager.send_environment_command
    )


# endregion


# region Profile List Validation
@pytest.mark.parametrize(
    "profile_event_list, queue_names, environment_types, expected",
    [
        ([], [], [], True),
        (
            [ProfileEvent(0, "Global", GlobalCommands.START_STREAMING)],
            ["Global"],
            ["Global"],
            True,
        ),
        (
            [ProfileEvent(0, "Mock Environment", GlobalCommands.START_ENVIRONMENT)],
            ["Environment 0"],
            [EnvironmentType.SKELETON],
            RattlesnakeError,
        ),
        (
            [
                ProfileEvent(
                    0,
                    "Mock Environment",
                    GlobalCommands.START_ENVIRONMENT,
                    skeleton_environment_instructions(environment_name="Mock Environment"),
                )
            ],
            ["Environment 0"],
            [EnvironmentType.SKELETON],
            True,
        ),
        (
            [None],
            ["Global"],
            ["Global"],
            RattlesnakeError,
        ),
        (
            [ProfileEvent(0, "Global", "Not a command")],
            ["Global"],
            ["Global"],
            RattlesnakeError,
        ),
    ],
)
def test_profile_manager_validate_profile_list(
    profile_event_list,
    queue_names,
    environment_types,
    expected,
    profile_manager,
):
    """
    Verifies that valid profile lists pass validation and invalid event types,
    invalid commands, and invalid command data raise ``RattlesnakeError``.
    """
    for profile_event, queue_name, environment_type in zip(
        profile_event_list,
        queue_names,
        environment_types,
    ):
        if isinstance(profile_event, ProfileEvent):
            profile_event._queue_name = queue_name
            profile_event._environment_type = environment_type

    if expected is RattlesnakeError:
        with pytest.raises(RattlesnakeError):
            profile_manager.validate_profile_list(profile_event_list)
    else:
        profile_manager.validate_profile_list(profile_event_list)


def test_profile_manager_validate_profile_list_sorts_by_timestamp(profile_manager):
    """
    Verifies that profile events are sorted by timestamp during validation.
    """
    event_late = ProfileEvent(10, "Global", GlobalCommands.START_STREAMING)
    event_late._queue_name = "Global"
    event_late._environment_type = "Global"

    event_early = ProfileEvent(1, "Global", GlobalCommands.STOP_STREAMING)
    event_early._queue_name = "Global"
    event_early._environment_type = "Global"

    profile_event_list = [event_late, event_early]

    profile_manager.validate_profile_list(profile_event_list)

    assert profile_event_list == [event_early, event_late]


def test_profile_manager_validate_profile_list_unimplemented_command(profile_manager):
    """
    Verifies that a valid profile event raises if the profile manager command
    map does not implement its command.
    """
    profile_event = ProfileEvent(0, "Global", GlobalCommands.STOP_STREAMING)
    profile_event._queue_name = "Global"
    profile_event._environment_type = "Global"

    profile_manager.command_map.pop(GlobalCommands.STOP_STREAMING)

    with pytest.raises(RattlesnakeError):
        profile_manager.validate_profile_list([profile_event])


# endregion


# region Profile Scheduling
@mock.patch("rattlesnake.profile_manager.threading.Timer")
def test_profile_manager_start_profile(mock_timer, profile_manager):
    """
    Verifies that timers are created for each profile event and that a closeout
    timer is scheduled after the final event.
    """
    global_event = ProfileEvent(0, "Global", GlobalCommands.START_STREAMING)
    global_event._queue_name = "Global"
    global_event._environment_type = "Global"

    environment_event = ProfileEvent(
        2,
        "Mock Environment",
        SkeletonCommands.EXAMPLE_SET_TEST_LEVEL,
        None,
    )
    environment_event._queue_name = "Environment 0"
    environment_event._environment_type = EnvironmentType.NONE

    start_event = ProfileEvent(
        3,
        "Mock Environment",
        GlobalCommands.START_ENVIRONMENT,
        skeleton_environment_instructions(environment_name="Mock Environment"),
    )
    start_event._queue_name = "Environment 0"
    start_event._environment_type = EnvironmentType.NONE

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
    assert mock_timer.return_value.start.call_count == 4
    assert profile_manager.profile_timers == [mock_timer.return_value] * 4


@mock.patch("rattlesnake.profile_manager.threading.Timer")
def test_profile_manager_start_profile_empty_list(mock_timer, profile_manager):
    """
    Verifies that starting an empty profile still schedules a closeout event.
    """
    profile_manager.start_profile([])

    mock_timer.assert_called_once_with(
        EXTRA_CLOSEOUT_TIME,
        profile_manager.fire_closeout_event,
    )
    mock_timer.return_value.start.assert_called_once()
    assert profile_manager.profile_timers == [mock_timer.return_value]


def test_profile_manager_fire_profile_event(profile_manager):
    """
    Verifies that firing an event dispatches to the mapped command handler.
    """
    mock_function = mock.MagicMock()
    profile_manager.command_map = {GlobalCommands.START_STREAMING: mock_function}

    profile_manager.fire_profile_event(
        "Global",
        GlobalCommands.START_STREAMING,
        None,
    )

    mock_function.assert_called_once_with(
        "Global",
        GlobalCommands.START_STREAMING,
        None,
    )


@mock.patch("rattlesnake.profile_manager.threading.Timer")
def test_profile_manager_stop_profile(mock_timer, profile_manager):
    """
    Verifies that existing timers are canceled and that a closeout timer is
    created, started, and stored.
    """
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


# endregion


# region Command Routing
def test_profile_manager_stop_hardware(profile_manager):
    """
    Verifies that the stop-hardware command is sent to the controller.
    """
    mock_controller = mock.MagicMock()
    profile_manager._controller_command_queue = mock_controller

    profile_manager.stop_hardware("Global", GlobalCommands.STOP_HARDWARE, None)

    mock_controller.put.assert_called_once_with(
        "Profile Manager",
        (GlobalCommands.STOP_HARDWARE, None),
    )


def test_profile_manager_start_streaming(profile_manager):
    """
    Verifies that the start-streaming command is sent to the controller.
    """
    mock_controller = mock.MagicMock()
    profile_manager._controller_command_queue = mock_controller

    profile_manager.start_streaming("Global", GlobalCommands.START_STREAMING, None)

    mock_controller.put.assert_called_once_with(
        "Profile Manager",
        (GlobalCommands.START_STREAMING, False),
    )


def test_profile_manager_stop_streaming(profile_manager):
    """
    Verifies that the stop-streaming command is sent to the controller.
    """
    mock_controller = mock.MagicMock()
    profile_manager._controller_command_queue = mock_controller

    profile_manager.stop_streaming("Global", GlobalCommands.STOP_STREAMING, None)

    mock_controller.put.assert_called_once_with(
        "Profile Manager",
        (GlobalCommands.STOP_STREAMING, None),
    )


def test_profile_manager_start_environment(profile_manager):
    """
    Verifies that the start-environment command and instructions are sent to
    the controller.
    """
    instructions = skeleton_environment_instructions(environment_name="Mock Environment")
    mock_controller = mock.MagicMock()
    profile_manager._controller_command_queue = mock_controller

    profile_manager.start_environment(
        "Environment 0",
        GlobalCommands.START_ENVIRONMENT,
        instructions,
    )

    mock_controller.put.assert_called_once_with(
        "Profile Manager",
        (GlobalCommands.START_ENVIRONMENT, ("Environment 0", instructions)),
    )


def test_profile_manager_stop_environment(profile_manager):
    """
    Verifies that the stop-environment command is sent to the controller.
    """
    mock_controller = mock.MagicMock()
    profile_manager._controller_command_queue = mock_controller

    profile_manager.stop_environment(
        "Environment 0",
        GlobalCommands.STOP_ENVIRONMENT,
        None,
    )

    mock_controller.put.assert_called_once_with(
        "Profile Manager",
        (GlobalCommands.STOP_ENVIRONMENT, "Environment 0"),
    )


def test_profile_manager_send_environment_command(profile_manager):
    """
    Verifies that an environment-specific profile command is routed through the
    controller.
    """
    mock_controller = mock.MagicMock()
    profile_manager._controller_command_queue = mock_controller

    profile_manager.send_environment_command(
        "Environment 0",
        SkeletonCommands.EXAMPLE_FLOAT_COMMAND,
        1.0,
    )

    mock_controller.put.assert_called_once_with(
        "Profile Manager",
        (
            GlobalCommands.SEND_ENVIRONMENT_COMMAND,
            ("Environment 0", SkeletonCommands.EXAMPLE_FLOAT_COMMAND, 1.0),
        ),
    )


def test_profile_manager_fire_closeout_event(profile_manager):
    """
    Verifies that the profile closeout command is sent to the controller.
    """
    mock_controller = mock.MagicMock()
    profile_manager._controller_command_queue = mock_controller

    profile_manager.fire_closeout_event()

    mock_controller.put.assert_called_once_with(
        "Profile Manager",
        (GlobalCommands.PROFILE_CLOSEOUT, None),
    )


# endregion


# region Logging
@mock.patch("rattlesnake.profile_manager.datetime")
def test_profile_manager_log(mock_datetime, profile_manager):
    """
    Verifies that calling ``log`` writes the expected formatted message to the
    log file queue.
    """
    mock_log_file_queue = mock.MagicMock()
    profile_manager._log_file_queue = mock_log_file_queue
    mock_datetime.now = fake_time

    profile_manager.log("Test Message")

    mock_log_file_queue.put.assert_called_once_with(
        "Datetime: Profile Manager -- Test Message\n"
    )


# endregion
