from pathlib import Path
from unittest import mock

import pytest

from rattlesnake.engine import RattlesnakeController, RattlesnakeState
from rattlesnake.environment.abstract_environment import (
    EnvironmentInstructions,
    EnvironmentMetadata,
)
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.process.streaming import StreamMetadata, StreamType
from rattlesnake.testing.mock_utilities import (
    skeleton_environment_instructions,
    skeleton_environment_metadata,
    skeleton_hardware_metadata,
)
from rattlesnake.utilities import GlobalCommands, RattlesnakeError


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
@mock.patch("rattlesnake.engine.mp.Manager")
@mock.patch("rattlesnake.engine.RattlesnakeController.wait_for_events")
def rattlesnake_package(
    mock_wait_for_events,
    mock_manager,
    mock_thread,
    mock_process,
    request,
):
    """
    Create a ``RattlesnakeController`` in threaded/non-threaded and
    blocking/non-blocking configurations without starting real subprocesses.
    """
    threaded, blocking = request.param
    mock_wait_for_events.return_value = None

    rattlesnake = RattlesnakeController(threaded=threaded, timeout=1)

    if not blocking:
        rattlesnake.clear_blocking()

    return rattlesnake, threaded, blocking


@pytest.fixture
def hardware_metadata():
    """
    Create mock hardware metadata.
    """
    return skeleton_hardware_metadata()


@pytest.fixture
def environment_metadata():
    """
    Create mock environment metadata.
    """
    return skeleton_environment_metadata()


@pytest.fixture
def stream_metadata():
    """
    Create default stream metadata.
    """
    return StreamMetadata()


# endregion


# region Initialization and Properties
@pytest.mark.parametrize("threaded", [True, False])
@pytest.mark.parametrize("blocking", [True, False])
@mock.patch("rattlesnake.engine.mp.Process")
@mock.patch("rattlesnake.engine.threading.Thread")
@mock.patch("rattlesnake.engine.mp.Manager")
@mock.patch("rattlesnake.engine.RattlesnakeController.wait_for_events")
def test_rattlesnake_init(
    mock_wait_for_events,
    mock_manager,
    mock_thread,
    mock_process,
    threaded,
    blocking,
):
    """
    Verifies that the controller initializes in threaded and non-threaded
    modes and optionally waits for startup events.
    """
    mock_wait_for_events.return_value = None

    rattlesnake = RattlesnakeController(threaded=threaded, timeout=1)

    if not blocking:
        rattlesnake.clear_blocking()

    assert isinstance(rattlesnake, RattlesnakeController)
    assert rattlesnake.threaded is threaded
    assert rattlesnake.timeout == 1

    if blocking:
        mock_wait_for_events.assert_called_once()
    else:
        assert rattlesnake.blocking is False


def test_rattlesnake_properties(rattlesnake_package):
    """
    Verifies core controller properties after initialization.
    """
    rattlesnake, threaded, blocking = rattlesnake_package

    assert rattlesnake.threaded is threaded
    assert rattlesnake.blocking is blocking
    assert rattlesnake.timeout == 1
    assert rattlesnake.hardware_metadata is None
    assert rattlesnake.environment_metadata == {}
    assert rattlesnake.last_stream_metadata is None
    assert rattlesnake.last_profile_event_list == []
    assert rattlesnake.has_gui is False


def test_rattlesnake_set_and_clear_blocking(rattlesnake_package):
    """
    Verifies that blocking mode can be enabled and disabled.
    """
    rattlesnake, _, _ = rattlesnake_package

    rattlesnake.set_blocking()
    assert rattlesnake.blocking is True

    rattlesnake.clear_blocking()
    assert rattlesnake.blocking is False


def test_rattlesnake_alive_event(rattlesnake_package):
    """
    Verifies that the ping-alive event can be set and cleared.
    """
    rattlesnake, _, _ = rattlesnake_package

    rattlesnake.clear_alive()
    assert rattlesnake.is_alive is False

    rattlesnake.set_alive()
    assert rattlesnake.is_alive is True

    rattlesnake.clear_alive()
    assert rattlesnake.is_alive is False


def test_rattlesnake_setup_gui(rattlesnake_package):
    """
    Verifies that GUI setup disables blocking and marks the controller as GUI
    backed.
    """
    rattlesnake, _, _ = rattlesnake_package

    rattlesnake.set_blocking()
    rattlesnake.setup_gui()

    assert rattlesnake.blocking is False
    assert rattlesnake.has_gui is True


def test_rattlesnake_log(rattlesnake_package):
    """
    Verifies that controller log messages are written to the log file queue.
    """
    rattlesnake, _, _ = rattlesnake_package
    mock_log_file_queue = mock.MagicMock()
    rattlesnake.queue_container.log_file_queue = mock_log_file_queue

    rattlesnake.log("hello")

    mock_log_file_queue.put.assert_called_once()
    assert "Rattlesnake -- hello" in mock_log_file_queue.put.call_args.args[0]


# endregion


# region State
@pytest.mark.parametrize(
    "hardware_metadata_value, environment_metadata_value, acquisition_active, "
    "output_active, environment_active, sysid_active, expected_state",
    [
        (None, {}, False, False, False, False, RattlesnakeState.INIT),
        (
            skeleton_hardware_metadata(),
            {},
            False,
            False,
            False,
            False,
            RattlesnakeState.HARDWARE_STORE,
        ),
        (
            skeleton_hardware_metadata(),
            {"Environment 0": skeleton_environment_metadata()},
            False,
            False,
            False,
            False,
            RattlesnakeState.ENVIRONMENT_STORE,
        ),
        (
            skeleton_hardware_metadata(),
            {"Environment 0": skeleton_environment_metadata()},
            True,
            True,
            False,
            False,
            RattlesnakeState.HARDWARE_ACTIVE,
        ),
        (
            skeleton_hardware_metadata(),
            {"Environment 0": skeleton_environment_metadata()},
            True,
            True,
            True,
            False,
            RattlesnakeState.ENVIRONMENT_ACTIVE,
        ),
        (
            skeleton_hardware_metadata(),
            {"Environment 0": skeleton_environment_metadata()},
            True,
            True,
            False,
            True,
            RattlesnakeState.SYS_ID_ACTIVE,
        ),
    ],
)
def test_rattlesnake_state(
    hardware_metadata_value,
    environment_metadata_value,
    acquisition_active,
    output_active,
    environment_active,
    sysid_active,
    expected_state,
    rattlesnake_package,
):
    """
    Verifies that the controller reports the expected state for combinations
    of stored metadata and active process events.
    """
    rattlesnake, _, _ = rattlesnake_package

    rattlesnake.hardware_metadata = hardware_metadata_value
    rattlesnake.environment_metadata = environment_metadata_value

    if acquisition_active:
        rattlesnake.event_container.acquisition_active_event.set()
    else:
        rattlesnake.event_container.acquisition_active_event.clear()

    if output_active:
        rattlesnake.event_container.output_active_event.set()
    else:
        rattlesnake.event_container.output_active_event.clear()

    if environment_active:
        rattlesnake.event_container.environment_active_events["Environment 0"].set()
    else:
        rattlesnake.event_container.environment_active_events["Environment 0"].clear()

    if sysid_active:
        rattlesnake.event_container.environment_sysid_active_events[
            "Environment 0"
        ].set()
    else:
        rattlesnake.event_container.environment_sysid_active_events[
            "Environment 0"
        ].clear()

    assert rattlesnake.state == expected_state


# endregion


# region Wait for Events
@pytest.mark.parametrize(
    "ready_event_return, active_event_return, active_event_check, expected",
    [
        ([True, True], [True, True], True, True),
        ([True, True], [], True, True),
        ([], [True, True], True, True),
        ([True, True], [False, True], True, RattlesnakeError),
        ([False, True], [True, True], True, RattlesnakeError),
        ([True, True], [False, False], False, True),
        ([True, True], [], False, True),
        ([], [False, False], False, True),
        ([True, True], [False, True], False, RattlesnakeError),
        ([False, True], [True, True], False, RattlesnakeError),
        ([True, True], [True, True], None, RattlesnakeError),
        ([True, True], [], None, True),
        ([], [True, True], None, RattlesnakeError),
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
    """
    Verifies successful waits and timeout behavior for ready and active events.
    """
    rattlesnake, _, _ = rattlesnake_package
    rattlesnake._timeout = 1
    mock_time.side_effect = [0, 2]

    ready_event_list = []
    for return_value in ready_event_return:
        ready_event = mock.MagicMock()
        ready_event.is_set.return_value = return_value
        ready_event_list.append(ready_event)

    active_event_list = []
    for return_value in active_event_return:
        active_event = mock.MagicMock()
        active_event.is_set.return_value = return_value
        active_event_list.append(active_event)

    if expected is RattlesnakeError:
        with pytest.raises(RattlesnakeError):
            rattlesnake.wait_for_events(
                ready_event_list,
                active_event_list,
                active_event_check=active_event_check,
            )

        for ready_event in ready_event_list:
            ready_event.set.assert_called()
    else:
        rattlesnake.wait_for_events(
            ready_event_list,
            active_event_list,
            active_event_check=active_event_check,
        )

        for ready_event in ready_event_list:
            ready_event.set.assert_not_called()


@mock.patch("rattlesnake.engine.time.time")
def test_rattlesnake_wait_for_events_alive_ping_resets_timeout(
    mock_time,
    rattlesnake_package,
):
    """
    Verifies that the alive ping resets the wait timeout.
    """
    rattlesnake, _, _ = rattlesnake_package
    rattlesnake._timeout = 10

    ready_event = mock.MagicMock()
    ready_event.is_set.side_effect = [False, True]

    rattlesnake.set_alive()
    mock_time.side_effect = [0, 100, 101]

    rattlesnake.wait_for_events([ready_event], [])

    assert rattlesnake.is_alive is False


# endregion


# region Loading and Saving
@mock.patch("rattlesnake.engine.load_metadata_from_netcdf")
@mock.patch("rattlesnake.engine.nc4.Dataset")
@mock.patch("rattlesnake.engine.os.access")
def test_rattlesnake_load_from_netcdf_template(
    mock_access,
    mock_dataset,
    mock_load_metadata_from_netcdf,
    rattlesnake_package,
    hardware_metadata,
    environment_metadata,
):
    """
    Verifies that netCDF templates are loaded and used to initialize hardware
    and environments.
    """
    rattlesnake, _, _ = rattlesnake_package
    rattlesnake.clear_blocking()

    mock_access.return_value = True
    mock_load_metadata_from_netcdf.return_value = (
        hardware_metadata,
        [environment_metadata],
    )
    rattlesnake.initialize_hardware = mock.MagicMock()
    rattlesnake.initialize_environments = mock.MagicMock()
    rattlesnake.initialize_profile_event_list = mock.MagicMock()

    rattlesnake.load_rattlesnake_from_template("template.nc4")

    mock_dataset.assert_called_once_with("template.nc4")
    mock_load_metadata_from_netcdf.assert_called_once_with(mock_dataset.return_value)
    rattlesnake.initialize_hardware.assert_called_once_with(hardware_metadata)
    rattlesnake.initialize_environments.assert_called_once_with([environment_metadata])
    rattlesnake.initialize_profile_event_list.assert_called_once_with([])
    assert rattlesnake.blocking is False


@mock.patch("rattlesnake.engine.load_metadata_from_workbook")
@mock.patch("rattlesnake.engine.openpyxl.load_workbook")
@mock.patch("rattlesnake.engine.os.access")
def test_rattlesnake_load_from_workbook_template(
    mock_access,
    mock_load_workbook,
    mock_load_metadata_from_workbook,
    rattlesnake_package,
    hardware_metadata,
    environment_metadata,
):
    """
    Verifies that Excel templates are loaded and used to initialize hardware,
    environments, and profile events.
    """
    rattlesnake, _, _ = rattlesnake_package

    profile_event_list = [mock.MagicMock()]
    mock_access.return_value = True
    mock_load_metadata_from_workbook.return_value = (
        hardware_metadata,
        [environment_metadata],
        profile_event_list,
    )

    rattlesnake.initialize_hardware = mock.MagicMock()
    rattlesnake.initialize_environments = mock.MagicMock()
    rattlesnake.initialize_profile_event_list = mock.MagicMock()

    rattlesnake.load_rattlesnake_from_template("template.xlsx")

    mock_load_workbook.assert_called_once_with("template.xlsx")
    mock_load_metadata_from_workbook.assert_called_once_with(
        mock_load_workbook.return_value
    )
    rattlesnake.initialize_hardware.assert_called_once_with(hardware_metadata)
    rattlesnake.initialize_environments.assert_called_once_with([environment_metadata])
    rattlesnake.initialize_profile_event_list.assert_called_once_with(
        profile_event_list
    )


@mock.patch("rattlesnake.engine.os.access")
def test_rattlesnake_load_template_permission_error(mock_access, rattlesnake_package):
    """
    Verifies that unreadable template files raise ``RattlesnakeError``.
    """
    rattlesnake, _, _ = rattlesnake_package
    mock_access.return_value = False

    with pytest.raises(RattlesnakeError):
        rattlesnake.load_rattlesnake_from_template("template.xlsx")


@mock.patch("rattlesnake.engine.save_rattlesnake_to_workbook")
@mock.patch("rattlesnake.engine.openpyxl.Workbook")
def test_rattlesnake_save_to_template(
    mock_workbook_class,
    mock_save_rattlesnake_to_workbook,
    rattlesnake_package,
    hardware_metadata,
    environment_metadata,
):
    """
    Verifies that controller metadata is saved to an Excel template.
    """
    rattlesnake, _, _ = rattlesnake_package
    rattlesnake.hardware_metadata = hardware_metadata
    rattlesnake.environment_metadata = {"Environment 0": environment_metadata}
    rattlesnake.last_profile_event_list = ["profile"]

    rattlesnake.save_rattlesnake_to_template("template.xlsx")

    mock_save_rattlesnake_to_workbook.assert_called_once()
    workbook_arg, hardware_arg, environment_values_arg, profile_arg = (
        mock_save_rattlesnake_to_workbook.call_args.args
    )

    assert workbook_arg is mock_workbook_class.return_value
    assert hardware_arg is hardware_metadata
    assert list(environment_values_arg) == [environment_metadata]
    assert profile_arg == ["profile"]
    mock_workbook_class.return_value.save.assert_called_once_with("template.xlsx")


def test_rattlesnake_save_to_template_invalid_extension(rattlesnake_package):
    """
    Verifies that only ``.xlsx`` templates can be saved.
    """
    rattlesnake, _, _ = rattlesnake_package

    with pytest.raises(RattlesnakeError):
        rattlesnake.save_rattlesnake_to_template("template.nc4")


# endregion


# region Hardware
@pytest.mark.parametrize(
    "state, instance, expected",
    [
        (RattlesnakeState.INIT, None, RattlesnakeError),
        (RattlesnakeState.INIT, HardwareMetadata, True),
        (RattlesnakeState.HARDWARE_STORE, HardwareMetadata, True),
        (RattlesnakeState.ENVIRONMENT_STORE, HardwareMetadata, True),
        (RattlesnakeState.HARDWARE_ACTIVE, HardwareMetadata, RattlesnakeError),
        (RattlesnakeState.ENVIRONMENT_ACTIVE, HardwareMetadata, RattlesnakeError),
        (RattlesnakeState.SYS_ID_ACTIVE, HardwareMetadata, RattlesnakeError),
    ],
)
def test_rattlesnake_initialize_hardware(
    state,
    instance,
    expected,
    rattlesnake_package,
):
    """
    Verifies state validation, metadata validation, command routing, and
    blocking wait behavior for hardware initialization.
    """
    rattlesnake, _, blocking = rattlesnake_package

    rattlesnake.wait_for_events = mock.MagicMock()
    rattlesnake.environment_manager = mock.MagicMock()
    rattlesnake.queue_container.acquisition_command_queue = mock.MagicMock()
    rattlesnake.queue_container.output_command_queue = mock.MagicMock()

    hardware_metadata = mock.MagicMock(spec=instance)

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.return_value = state

        if expected is RattlesnakeError:
            with pytest.raises(RattlesnakeError):
                rattlesnake.initialize_hardware(hardware_metadata)
        else:
            rattlesnake.initialize_hardware(hardware_metadata)

            hardware_metadata.validate.assert_called_once_with()
            rattlesnake.environment_manager.initialize_hardware.assert_called_once_with(
                hardware_metadata
            )
            rattlesnake.queue_container.acquisition_command_queue.put.assert_called_once_with(
                "Rattlesnake",
                (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata),
            )
            rattlesnake.queue_container.output_command_queue.put.assert_called_once_with(
                "Rattlesnake",
                (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata),
            )

            if blocking:
                rattlesnake.wait_for_events.assert_called_once()


# endregion


# region Environment Initialization
@pytest.mark.parametrize(
    "state, should_raise",
    [
        (RattlesnakeState.INIT, True),
        (RattlesnakeState.HARDWARE_STORE, False),
        (RattlesnakeState.ENVIRONMENT_STORE, False),
        (RattlesnakeState.HARDWARE_ACTIVE, True),
        (RattlesnakeState.ENVIRONMENT_ACTIVE, True),
        (RattlesnakeState.SYS_ID_ACTIVE, True),
    ],
)
def test_rattlesnake_initialize_environment(
    state,
    should_raise,
    rattlesnake_package,
):
    """
    Verifies validation, environment initialization, command routing, return
    value, and blocking wait behavior.
    """
    rattlesnake, _, blocking = rattlesnake_package

    rattlesnake.wait_for_events = mock.MagicMock()
    rattlesnake.environment_manager = mock.MagicMock()
    rattlesnake.queue_container.acquisition_command_queue = mock.MagicMock()
    rattlesnake.queue_container.output_command_queue = mock.MagicMock()

    hardware_metadata = mock.MagicMock(spec=HardwareMetadata)
    rattlesnake.hardware_metadata = hardware_metadata

    environment_metadata = mock.MagicMock(spec=EnvironmentMetadata)
    environment_metadata_list = [environment_metadata]
    initialized_environment_metadata = {"Environment 0": environment_metadata}
    rattlesnake.environment_manager.initialize_environments.return_value = (
        initialized_environment_metadata
    )

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.return_value = state

        if should_raise:
            with pytest.raises(RattlesnakeError):
                rattlesnake.initialize_environments(environment_metadata_list)
        else:
            returned = rattlesnake.initialize_environments(environment_metadata_list)

            rattlesnake.environment_manager.validate_environment_metadata.assert_called_once_with(
                environment_metadata_list,
                hardware_metadata,
            )
            rattlesnake.environment_manager.initialize_environments.assert_called_once_with(
                environment_metadata_list,
                hardware_metadata,
            )
            rattlesnake.queue_container.acquisition_command_queue.put.assert_called_once_with(
                "Rattlesnake",
                (
                    GlobalCommands.INITIALIZE_ENVIRONMENT,
                    initialized_environment_metadata,
                ),
            )
            rattlesnake.queue_container.output_command_queue.put.assert_called_once_with(
                "Rattlesnake",
                (
                    GlobalCommands.INITIALIZE_ENVIRONMENT,
                    initialized_environment_metadata,
                ),
            )
            assert returned == initialized_environment_metadata

            if blocking:
                rattlesnake.wait_for_events.assert_called_once()


def test_rattlesnake_initialize_empty_environment(rattlesnake_package):
    """
    Verifies that an empty environment metadata list can be initialized.
    """
    rattlesnake, _, blocking = rattlesnake_package

    rattlesnake.wait_for_events = mock.MagicMock()
    rattlesnake.environment_manager = mock.MagicMock()
    rattlesnake.queue_container.acquisition_command_queue = mock.MagicMock()
    rattlesnake.queue_container.output_command_queue = mock.MagicMock()
    rattlesnake.hardware_metadata = None

    initialized_environment_metadata = {}
    rattlesnake.environment_manager.initialize_environments.return_value = (
        initialized_environment_metadata
    )

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.return_value = RattlesnakeState.ENVIRONMENT_STORE

        returned = rattlesnake.initialize_environments([])

    rattlesnake.environment_manager.validate_environment_metadata.assert_called_once_with(
        [],
        None,
    )
    rattlesnake.environment_manager.initialize_environments.assert_called_once_with(
        [],
        None,
    )
    assert returned == initialized_environment_metadata

    if blocking:
        rattlesnake.wait_for_events.assert_called_once()


# endregion


# region System Identification
@pytest.mark.parametrize(
    "state, should_raise",
    [
        (RattlesnakeState.INIT, True),
        (RattlesnakeState.HARDWARE_STORE, True),
        (RattlesnakeState.ENVIRONMENT_STORE, False),
        (RattlesnakeState.HARDWARE_ACTIVE, False),
        (RattlesnakeState.ENVIRONMENT_ACTIVE, True),
        (RattlesnakeState.SYS_ID_ACTIVE, True),
    ],
)
def test_rattlesnake_initialize_system_id(
    state,
    should_raise,
    rattlesnake_package,
    hardware_metadata,
):
    """
    Verifies system identification metadata initialization state validation and
    environment-manager interactions.
    """
    rattlesnake, _, blocking = rattlesnake_package

    sysid_metadata = mock.MagicMock()
    rattlesnake.hardware_metadata = hardware_metadata
    rattlesnake.wait_for_events = mock.MagicMock()
    rattlesnake.environment_manager = mock.MagicMock()
    rattlesnake.environment_manager.validate_system_id_metadata.return_value = (
        "Environment 0"
    )
    environment_metadata = {"Environment 0": skeleton_environment_metadata()}
    rattlesnake.environment_manager.initialize_system_id.return_value = (
        environment_metadata
    )

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.return_value = state

        if should_raise:
            with pytest.raises(RattlesnakeError):
                rattlesnake.initialize_system_id(sysid_metadata, "Mock Environment")
        else:
            rattlesnake.initialize_system_id(sysid_metadata, "Mock Environment")

            rattlesnake.environment_manager.validate_system_id_metadata.assert_called_once_with(
                sysid_metadata,
                hardware_metadata,
                "Mock Environment",
            )
            rattlesnake.environment_manager.initialize_system_id.assert_called_once_with(
                sysid_metadata,
                "Environment 0",
            )
            assert rattlesnake.environment_metadata == environment_metadata

            if blocking:
                rattlesnake.wait_for_events.assert_called_once()


@pytest.mark.parametrize(
    "method_name, command",
    [
        ("start_system_id_noise", GlobalCommands.START_SYSTEM_ID_NOISE),
        (
            "start_system_id_transfer_function",
            GlobalCommands.START_SYSTEM_ID_TRANSFER,
        ),
    ],
)
@pytest.mark.parametrize(
    "state, should_raise",
    [
        (RattlesnakeState.ENVIRONMENT_STORE, True),
        (RattlesnakeState.HARDWARE_ACTIVE, False),
        (RattlesnakeState.ENVIRONMENT_ACTIVE, True),
    ],
)
def test_rattlesnake_start_system_id_methods(
    method_name,
    command,
    state,
    should_raise,
    rattlesnake_package,
):
    """
    Verifies noise and transfer-function system identification start methods.
    """
    rattlesnake, _, blocking = rattlesnake_package

    rattlesnake.wait_for_events = mock.MagicMock()
    rattlesnake.environment_manager = mock.MagicMock()
    rattlesnake.environment_manager.queue_names_dict = {
        "Mock Environment": "Environment 0"
    }
    rattlesnake.queue_container.controller_command_queue = mock.MagicMock()

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.return_value = state

        method = getattr(rattlesnake, method_name)

        if should_raise:
            with pytest.raises(RattlesnakeError):
                method("Mock Environment")
        else:
            method("Mock Environment")

            rattlesnake.queue_container.controller_command_queue.put.assert_called_once_with(
                "Rattlesnake",
                (command, "Environment 0"),
            )

            if blocking:
                rattlesnake.wait_for_events.assert_called_once()


def test_rattlesnake_start_system_id_unknown_environment(rattlesnake_package):
    """
    Verifies that starting system identification for an unknown environment
    raises ``RattlesnakeError``.
    """
    rattlesnake, _, _ = rattlesnake_package
    rattlesnake.environment_manager = mock.MagicMock()
    rattlesnake.environment_manager.queue_names_dict = {}

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.return_value = RattlesnakeState.HARDWARE_ACTIVE

        with pytest.raises(RattlesnakeError):
            rattlesnake.start_system_id_noise("Missing Environment")


@pytest.mark.parametrize(
    "state, should_raise",
    [
        (RattlesnakeState.HARDWARE_ACTIVE, True),
        (RattlesnakeState.SYS_ID_ACTIVE, False),
    ],
)
def test_rattlesnake_stop_system_id(state, should_raise, rattlesnake_package):
    """
    Verifies that stopping system identification sends the correct controller
    command and waits for the sysid active event to clear.
    """
    rattlesnake, _, blocking = rattlesnake_package

    rattlesnake.wait_for_events = mock.MagicMock()
    rattlesnake.environment_manager = mock.MagicMock()
    rattlesnake.environment_manager.queue_names_dict = {
        "Mock Environment": "Environment 0"
    }
    rattlesnake.queue_container.controller_command_queue = mock.MagicMock()

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.return_value = state

        if should_raise:
            with pytest.raises(RattlesnakeError):
                rattlesnake.stop_system_id("Mock Environment")
        else:
            rattlesnake.stop_system_id("Mock Environment")

            rattlesnake.queue_container.controller_command_queue.put.assert_called_once_with(
                "Rattlesnake",
                (GlobalCommands.STOP_SYSTEM_ID, "Environment 0"),
            )

            if blocking:
                rattlesnake.wait_for_events.assert_called_once()


def test_rattlesnake_save_system_id_to_file(rattlesnake_package):
    """
    Verifies that saving system identification forwards the save command to the
    environment and waits for readiness.
    """
    rattlesnake, _, _ = rattlesnake_package
    rattlesnake.wait_for_events = mock.MagicMock()
    rattlesnake.environment_manager = mock.MagicMock()
    rattlesnake.environment_manager.queue_names_dict = {
        "Mock Environment": "Environment 0"
    }
    mock_environment_queue = mock.MagicMock()
    rattlesnake.queue_container.environment_command_queues = {
        "Environment 0": mock_environment_queue
    }

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.return_value = RattlesnakeState.ENVIRONMENT_STORE

        rattlesnake.save_system_id_to_file("Mock Environment", "sysid.npz")

    mock_environment_queue.put.assert_called_once_with(
        "Rattlesnake",
        (GlobalCommands.SAVE_SYSTEM_ID, "sysid.npz"),
    )
    rattlesnake.wait_for_events.assert_called_once()


def test_rattlesnake_load_system_id_from_package(rattlesnake_package):
    """
    Verifies that loading a system identification package validates the package,
    forwards it to the environment, and waits for stored confirmation.
    """
    rattlesnake, _, _ = rattlesnake_package
    rattlesnake.wait_for_events = mock.MagicMock()
    rattlesnake.environment_manager = mock.MagicMock()
    rattlesnake.environment_manager.validate_system_id_package.return_value = (
        "Environment 0"
    )

    mock_environment_queue = mock.MagicMock()
    rattlesnake.queue_container.environment_command_queues = {
        "Environment 0": mock_environment_queue
    }

    sysid_package = mock.MagicMock()

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.return_value = RattlesnakeState.ENVIRONMENT_STORE

        rattlesnake.load_system_id_from_package("Mock Environment", sysid_package)

    rattlesnake.environment_manager.validate_system_id_package.assert_called_once_with(
        "Mock Environment",
        sysid_package,
    )
    mock_environment_queue.put.assert_called_once_with(
        "Rattlesnake",
        (GlobalCommands.LOAD_SYSTEM_ID, sysid_package),
    )
    rattlesnake.wait_for_events.assert_called_once()


def test_rattlesnake_preview_system_id_noise_from_hardware_active(
    rattlesnake_package,
):
    """
    Verifies that previewing noise stops active acquisition first, disables
    automatic shutdown, initializes sysid, starts acquisition, and starts noise.
    """
    rattlesnake, _, _ = rattlesnake_package
    sysid_metadata = mock.MagicMock()

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.side_effect = [
            RattlesnakeState.HARDWARE_ACTIVE,
            RattlesnakeState.ENVIRONMENT_STORE,
        ]

        rattlesnake.stop_acquisition = mock.MagicMock()
        rattlesnake.initialize_system_id = mock.MagicMock()
        rattlesnake.start_acquisition = mock.MagicMock()
        rattlesnake.start_system_id_noise = mock.MagicMock()

        rattlesnake.preview_system_id_noise(sysid_metadata, "Mock Environment")

    assert sysid_metadata.auto_shutdown is False
    rattlesnake.stop_acquisition.assert_called_once_with()
    rattlesnake.initialize_system_id.assert_called_once_with(
        sysid_metadata,
        "Mock Environment",
    )
    rattlesnake.start_acquisition.assert_called_once()
    rattlesnake.start_system_id_noise.assert_called_once_with("Mock Environment")


def test_rattlesnake_preview_system_id_transfer(rattlesnake_package):
    """
    Verifies transfer-function preview orchestration.
    """
    rattlesnake, _, _ = rattlesnake_package
    sysid_metadata = mock.MagicMock()

    rattlesnake.initialize_system_id = mock.MagicMock()
    rattlesnake.start_acquisition = mock.MagicMock()
    rattlesnake.start_system_id_transfer_function = mock.MagicMock()

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.return_value = RattlesnakeState.ENVIRONMENT_STORE

        rattlesnake.preview_system_id_transfer(sysid_metadata, "Mock Environment")

    assert sysid_metadata.auto_shutdown is False
    rattlesnake.initialize_system_id.assert_called_once_with(
        sysid_metadata,
        "Mock Environment",
    )
    rattlesnake.start_acquisition.assert_called_once()
    rattlesnake.start_system_id_transfer_function.assert_called_once_with(
        "Mock Environment"
    )


@pytest.mark.parametrize("stream_file", [None, "sysid_stream.nc4"])
def test_rattlesnake_run_system_id(stream_file, rattlesnake_package):
    """
    Verifies complete system identification orchestration with and without
    streaming.
    """
    rattlesnake, _, _ = rattlesnake_package
    rattlesnake.clear_blocking()

    sysid_metadata = mock.MagicMock()
    if stream_file is None:
        if hasattr(sysid_metadata, "stream_file"):
            delattr(sysid_metadata, "stream_file")
    else:
        sysid_metadata.stream_file = stream_file

    rattlesnake.environment_manager = mock.MagicMock()
    rattlesnake.environment_manager.queue_names_dict = {
        "Mock Environment": "Environment 0"
    }

    rattlesnake.initialize_system_id = mock.MagicMock()
    rattlesnake.start_acquisition = mock.MagicMock()
    rattlesnake.start_streaming = mock.MagicMock()
    rattlesnake.stop_streaming = mock.MagicMock()
    rattlesnake.start_system_id_noise = mock.MagicMock()
    rattlesnake.start_system_id_transfer_function = mock.MagicMock()
    rattlesnake.stop_acquisition = mock.MagicMock()

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.return_value = RattlesnakeState.ENVIRONMENT_STORE

        rattlesnake.run_system_id(sysid_metadata, "Mock Environment")

    assert sysid_metadata.auto_shutdown is True
    rattlesnake.initialize_system_id.assert_called_once_with(
        sysid_metadata,
        "Mock Environment",
    )
    rattlesnake.start_acquisition.assert_called_once()
    rattlesnake.start_system_id_noise.assert_called_once_with("Mock Environment")
    rattlesnake.start_system_id_transfer_function.assert_called_once_with(
        "Mock Environment"
    )
    rattlesnake.stop_acquisition.assert_called_once_with()

    if stream_file:
        assert rattlesnake.start_streaming.call_count == 2
        assert rattlesnake.stop_streaming.call_count == 2
    else:
        rattlesnake.start_streaming.assert_not_called()
        rattlesnake.stop_streaming.assert_not_called()


@pytest.mark.parametrize(
    "state",
    [RattlesnakeState.HARDWARE_ACTIVE, RattlesnakeState.SYS_ID_ACTIVE],
)
def test_rattlesnake_stop_system_id_run(state, rattlesnake_package):
    """
    Verifies system-identification-run stop behavior.
    """
    rattlesnake, _, _ = rattlesnake_package
    rattlesnake.stop_acquisition = mock.MagicMock()
    rattlesnake.stop_system_id = mock.MagicMock()

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.return_value = state

        rattlesnake.stop_system_id_run("Mock Environment")

    if state == RattlesnakeState.HARDWARE_ACTIVE:
        rattlesnake.stop_acquisition.assert_called_once_with()
        rattlesnake.stop_system_id.assert_not_called()
    else:
        rattlesnake.stop_system_id.assert_called_once_with("Mock Environment")
        rattlesnake.stop_acquisition.assert_called_once_with()


# endregion


# region Acquisition
@pytest.mark.parametrize(
    "state, valid_stream_metadata, should_raise",
    [
        (RattlesnakeState.INIT, True, True),
        (RattlesnakeState.HARDWARE_STORE, True, True),
        (RattlesnakeState.ENVIRONMENT_STORE, True, False),
        (RattlesnakeState.ENVIRONMENT_STORE, False, True),
        (RattlesnakeState.HARDWARE_ACTIVE, True, True),
        (RattlesnakeState.ENVIRONMENT_ACTIVE, True, True),
        (RattlesnakeState.SYS_ID_ACTIVE, True, True),
    ],
)
def test_rattlesnake_start_acquisition(
    state,
    valid_stream_metadata,
    should_raise,
    rattlesnake_package,
):
    """
    Verifies state validation, stream metadata validation, streaming
    initialization command routing, controller run command routing, and
    blocking wait behavior.
    """
    rattlesnake, _, blocking = rattlesnake_package

    rattlesnake.wait_for_events = mock.MagicMock()
    rattlesnake.queue_container.streaming_command_queue = mock.MagicMock()
    rattlesnake.queue_container.controller_command_queue = mock.MagicMock()

    if valid_stream_metadata:
        stream_metadata = StreamMetadata()
        stream_metadata.validate = mock.MagicMock()
    else:
        stream_metadata = None

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.return_value = state

        if should_raise:
            with pytest.raises(RattlesnakeError):
                rattlesnake.start_acquisition(stream_metadata)
        else:
            rattlesnake.start_acquisition(stream_metadata)

            stream_metadata.validate.assert_called_once_with()
            rattlesnake.queue_container.streaming_command_queue.put.assert_called_once_with(
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
            rattlesnake.queue_container.controller_command_queue.put.assert_called_once_with(
                "Rattlesnake",
                (GlobalCommands.RUN_HARDWARE, stream_metadata),
            )
            assert rattlesnake.last_stream_metadata is stream_metadata

            if blocking:
                rattlesnake.wait_for_events.assert_called_once()


@pytest.mark.parametrize(
    "state, stream_type_valid, should_raise",
    [
        (RattlesnakeState.ENVIRONMENT_STORE, True, False),
        (RattlesnakeState.HARDWARE_STORE, True, True),
        (RattlesnakeState.ENVIRONMENT_STORE, False, True),
    ],
)
def test_rattlesnake_set_stream_metadata(
    state,
    stream_type_valid,
    should_raise,
    rattlesnake_package,
):
    """
    Verifies storing stream metadata for UI use.
    """
    rattlesnake, _, _ = rattlesnake_package

    if stream_type_valid:
        stream_metadata = StreamMetadata()
        stream_metadata.validate = mock.MagicMock()
    else:
        stream_metadata = object()

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state.return_value = state

        if should_raise:
            with pytest.raises(RattlesnakeError):
                rattlesnake.set_stream_metadata(stream_metadata)
        else:
            rattlesnake.set_stream_metadata(stream_metadata)

            stream_metadata.validate.assert_called_once_with()
            assert rattlesnake.last_stream_metadata is stream_metadata


@pytest.mark.parametrize(
    "state, should_raise",
    [
        (RattlesnakeState.INIT, True),
        (RattlesnakeState.ENVIRONMENT_STORE, True),
        (RattlesnakeState.HARDWARE_ACTIVE, False),
        (RattlesnakeState.ENVIRONMENT_ACTIVE, False),
        (RattlesnakeState.SYS_ID_ACTIVE, False),
    ],
)
def test_rattlesnake_stop_acquisition(
    state,
    should_raise,
    rattlesnake_package,
):
    """
    Verifies stop-acquisition state validation, profile stop, controller
    command routing, and blocking wait behavior.
    """
    rattlesnake, _, blocking = rattlesnake_package

    rattlesnake.wait_for_events = mock.MagicMock()
    rattlesnake.profile_manager = mock.MagicMock()
    rattlesnake.environment_manager = mock.MagicMock()
    rattlesnake.environment_manager.active_event_list = []
    rattlesnake.queue_container.controller_command_queue = mock.MagicMock()

    with mock.patch.object(
        RattlesnakeController,
        "state",
        new_callable=mock.PropertyMock,
    ) as mock_state:
        mock_state
