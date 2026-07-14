import multiprocessing as mp
import threading
from unittest import mock

import pytest

from rattlesnake.environment.abstract_environment import (
    EnvironmentInstructions,
    EnvironmentMetadata,
)
from rattlesnake.environment.environment_registry import SYSID_ENVIRONMENTS
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment_manager import EnvironmentManager
from rattlesnake.process.abstract_sysid_data_analysis import (
    SysIdDataPackage,
    SysIdMetadata,
)
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.testing.mock_utilities import (
    fake_time,
    mock_event_container,
    mock_queue_container,
    skeleton_environment_instructions,
    skeleton_environment_metadata,
    skeleton_hardware_metadata,
)
from rattlesnake.utilities import GlobalCommands, RattlesnakeError


def first_sysid_environment_type():
    """
    Return one registered system-identification environment type.

    Tests that require a system-identification environment are skipped if the
    registry does not contain any system-identification environments.
    """
    if not SYSID_ENVIRONMENTS:
        pytest.skip("No system-identification environments are registered")

    return SYSID_ENVIRONMENTS[0]


# endregion


# region Fixtures
@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def environment_manager(request):
    """
    Create an ``EnvironmentManager`` in threaded and multiprocessing modes.
    """
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)

    return EnvironmentManager(queue_container, event_container, use_thread)


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
def environment_instructions():
    """
    Create mock environment instructions.
    """
    return skeleton_environment_instructions()


# endregion


# region Initialization and Properties
@pytest.mark.parametrize("use_thread", [True, False])
def test_environment_manager_init(use_thread):
    """
    Verifies that the environment manager initializes successfully and stores
    the requested threaded mode.
    """
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)

    environment_manager = EnvironmentManager(
        queue_container,
        event_container,
        use_thread,
    )

    assert isinstance(environment_manager, EnvironmentManager)
    assert environment_manager.threaded is use_thread

    assert environment_manager.queue_container is queue_container
    assert environment_manager.event_container is event_container
    assert (
        environment_manager.environment_active_events
        is event_container.environment_active_events
    )
    assert (
        environment_manager.environment_ready_events
        is event_container.environment_ready_events
    )
    assert (
        environment_manager.environment_close_events
        is event_container.environment_close_events
    )
    assert (
        environment_manager.environment_sysid_active_events
        is event_container.environment_sysid_active_events
    )
    assert (
        environment_manager.environment_sysid_stored_events
        is event_container.environment_sysid_stored_events
    )

    if use_thread:
        assert environment_manager.new_process is threading.Thread
        assert environment_manager.new_event is threading.Event
    else:
        assert environment_manager.new_process is mp.Process
        assert environment_manager.new_event is mp.Event


@pytest.mark.parametrize(
    "queue_names, expected_available",
    [
        (
            ["Environment 0", "Environment 1"],
            ["Environment 2", "Environment 3"],
        ),
        (
            [],
            ["Environment 0", "Environment 1", "Environment 2", "Environment 3"],
        ),
        (
            ["Environment 0", "Environment 1", "Environment 2", "Environment 3"],
            [],
        ),
    ],
)
def test_environment_manager_available_queues(
    queue_names,
    expected_available,
    environment_manager,
):
    """
    Verifies that available queues exclude assigned queue names.
    """
    environment_manager.queue_names = queue_names

    assert environment_manager.available_queues == expected_available


def test_environment_manager_num_queues(environment_manager):
    """
    Verifies that the total queue count includes assigned and available queues.
    """
    environment_manager.queue_names = ["Environment 0", "Environment 1"]

    assert environment_manager.num_queues == 4


def test_environment_manager_queue_names_dict(environment_manager):
    """
    Verifies that user-facing environment names map to internal queue names.
    """
    environment_manager.queue_names = ["Environment 0", "Environment 1"]
    environment_manager.environment_names = {
        "Environment 0": "Env A",
        "Environment 1": "Env B",
    }

    assert environment_manager.queue_names_dict == {
        "Env A": "Environment 0",
        "Env B": "Environment 1",
    }


@mock.patch("rattlesnake.environment_manager.datetime")
def test_environment_manager_log(mock_datetime, environment_manager):
    """
    Verifies that environment manager log messages are written to the log file
    queue.
    """
    mock_log_file_queue = mock.MagicMock()
    environment_manager.queue_container.log_file_queue = mock_log_file_queue
    mock_datetime.now = fake_time

    environment_manager.log("hello")

    mock_log_file_queue.put.assert_called_once_with(
        "Datetime: Environment Manager -- hello\n"
    )


# endregion


# region State Sync
def test_environment_manager_ready_event_list(environment_manager):
    """
    Verifies that the ready event list contains events for assigned
    environments.
    """
    environment_manager.queue_names = ["Environment 0", "Environment 1"]

    assert environment_manager.ready_event_list == [
        environment_manager.environment_ready_events["Environment 0"],
        environment_manager.environment_ready_events["Environment 1"],
    ]


def test_environment_manager_active_event_list(environment_manager):
    """
    Verifies that the active event list contains events for assigned
    environments.
    """
    environment_manager.queue_names = ["Environment 0", "Environment 1"]

    assert environment_manager.active_event_list == [
        environment_manager.environment_active_events["Environment 0"],
        environment_manager.environment_active_events["Environment 1"],
    ]


def test_environment_manager_acquisition_ready_environments_non_sysid(
    environment_manager,
):
    """
    Verifies that non-system-identification environments are acquisition ready.
    """
    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_types = {"Environment 0": EnvironmentType.NONE}

    assert environment_manager.acquisition_ready_environments == {"Environment 0": True}


def test_environment_manager_acquisition_ready_environments_sysid(
    environment_manager,
):
    """
    Verifies that system-identification environments are acquisition ready only
    when system-identification data has been stored.
    """
    sysid_environment_type = first_sysid_environment_type()

    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_types = {"Environment 0": sysid_environment_type}

    environment_manager.environment_sysid_stored_events["Environment 0"].clear()
    assert environment_manager.acquisition_ready_environments == {
        "Environment 0": False
    }

    environment_manager.environment_sysid_stored_events["Environment 0"].set()
    assert environment_manager.acquisition_ready_environments == {"Environment 0": True}


def test_environment_manager_sysid_active_environments(environment_manager):
    """
    Verifies that user-facing names of environments with active system
    identification are returned.
    """
    environment_manager.environment_names = {
        "Environment 0": "Env A",
        "Environment 1": "Env B",
    }

    environment_manager.environment_sysid_active_events["Environment 0"].set()
    environment_manager.environment_sysid_active_events["Environment 1"].clear()

    assert environment_manager.sysid_active_environments == ["Env A"]


def test_environment_manager_clear_sysid_events(environment_manager):
    """
    Verifies that system-identification stored events are cleared for assigned
    environments.
    """
    environment_manager.queue_names = ["Environment 0", "Environment 1"]
    environment_manager.environment_sysid_stored_events["Environment 0"].set()
    environment_manager.environment_sysid_stored_events["Environment 1"].set()

    environment_manager.clear_sysid_events()

    assert not environment_manager.environment_sysid_stored_events[
        "Environment 0"
    ].is_set()
    assert not environment_manager.environment_sysid_stored_events[
        "Environment 1"
    ].is_set()


def test_environment_manager_set_ready_events(environment_manager):
    """
    Verifies that assigned environment ready events are set.
    """
    environment_manager.queue_names = ["Environment 0", "Environment 1"]
    environment_manager.environment_ready_events["Environment 0"].clear()
    environment_manager.environment_ready_events["Environment 1"].clear()

    environment_manager.set_ready_events()

    assert environment_manager.environment_ready_events["Environment 0"].is_set()
    assert environment_manager.environment_ready_events["Environment 1"].is_set()


def test_environment_manager_initialize_hardware(
    environment_manager,
    hardware_metadata,
):
    """
    Verifies that hardware initialization commands are sent to assigned
    environment queues.
    """
    environment_manager.queue_names = ["Environment 0", "Environment 1"]

    mock_command_queue_0 = mock.MagicMock()
    mock_command_queue_1 = mock.MagicMock()
    environment_manager.queue_container.environment_command_queues["Environment 0"] = (
        mock_command_queue_0
    )
    environment_manager.queue_container.environment_command_queues["Environment 1"] = (
        mock_command_queue_1
    )

    environment_manager.initialize_hardware(hardware_metadata)

    mock_command_queue_0.put.assert_called_once_with(
        "Environment Manager",
        (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata),
    )
    mock_command_queue_1.put.assert_called_once_with(
        "Environment Manager",
        (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata),
    )
    assert not environment_manager.environment_ready_events["Environment 0"].is_set()
    assert not environment_manager.environment_ready_events["Environment 1"].is_set()


def test_environment_manager_initialize_system_id(environment_manager):
    """
    Verifies that system-identification metadata is copied into environment
    metadata and an initialize command is sent to the environment.
    """
    queue_name = "Environment 0"
    sysid_metadata = mock.MagicMock(spec=SysIdMetadata)

    metadata = skeleton_environment_metadata()
    environment_manager.environment_metadata = {queue_name: metadata}

    mock_command_queue = mock.MagicMock()
    environment_manager.queue_container.environment_command_queues[queue_name] = (
        mock_command_queue
    )

    updated_metadata = environment_manager.initialize_system_id(
        sysid_metadata,
        queue_name,
    )

    assert updated_metadata is not environment_manager.environment_metadata
    assert updated_metadata[queue_name] is not metadata
    assert updated_metadata[queue_name].sysid_metadata is sysid_metadata

    mock_command_queue.put.assert_called_once_with(
        "Environment Manager",
        (GlobalCommands.INITIALIZE_SYSTEM_ID, sysid_metadata),
    )


# endregion


# region Environment Initialization
@pytest.mark.parametrize(
    "existing_metadata, existing_types, queue_names, expected",
    [
        (
            {"Environment 0": skeleton_environment_metadata()},
            {"Environment 0": EnvironmentType.SKELETON},
            ["Environment 0"],
            "reuse",
        ),
        (
            {},
            {},
            [],
            "add",
        ),
        (
            {
                "Environment 0": skeleton_environment_metadata(),
                "Environment 1": skeleton_environment_metadata(),
            },
            {
                "Environment 0": EnvironmentType.SKELETON,
                "Environment 1": EnvironmentType.SKELETON,
            },
            ["Environment 0", "Environment 1"],
            "remove",
        ),
    ],
)
def test_environment_manager_initialize_environments(
    existing_metadata,
    existing_types,
    queue_names,
    expected,
    environment_manager,
    hardware_metadata,
):
    """
    Verifies that matching existing environments receive initialization
    commands, new environments are added, and unmapped environments are
    removed.
    """
    metadata = skeleton_environment_metadata(environment_name="New Environment")
    metadata_list = [metadata]

    environment_manager.queue_names = list(queue_names)
    environment_manager.environment_types = dict(existing_types)
    environment_manager.environment_metadata = dict(existing_metadata)

    mock_command_queue = mock.MagicMock()
    environment_manager.queue_container.environment_command_queues["Environment 0"] = (
        mock_command_queue
    )

    environment_manager.add_environment = mock.MagicMock(return_value="Environment 2")
    environment_manager.remove_environment = mock.MagicMock()

    returned_metadata = environment_manager.initialize_environments(
        metadata_list,
        hardware_metadata,
    )

    if expected == "reuse":
        assert returned_metadata == {"Environment 0": metadata}
        assert environment_manager.environment_names["Environment 0"] == (
            "New Environment"
        )
        mock_command_queue.assign_environment.assert_called_once_with("New Environment")
        mock_command_queue.put.assert_has_calls(
            [
                mock.call(
                    "Environment Manager",
                    (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata),
                ),
                mock.call(
                    "Environment Manager",
                    (GlobalCommands.INITIALIZE_ENVIRONMENT, metadata),
                ),
            ]
        )
        environment_manager.add_environment.assert_not_called()
        environment_manager.remove_environment.assert_not_called()

    elif expected == "add":
        environment_manager.add_environment.assert_called_once_with(
            metadata,
            hardware_metadata,
        )
        assert returned_metadata == {"Environment 2": metadata}

    elif expected == "remove":
        assert returned_metadata == {"Environment 0": metadata}
        environment_manager.remove_environment.assert_called_once_with("Environment 1")


def test_environment_manager_initialize_environments_clears_sysid_events(
    environment_manager,
    hardware_metadata,
):
    """
    Verifies that environment initialization clears stale system-identification
    stored events.
    """
    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_types = {"Environment 0": EnvironmentType.SKELETON}
    environment_manager.environment_sysid_stored_events["Environment 0"].set()

    mock_command_queue = mock.MagicMock()
    environment_manager.queue_container.environment_command_queues["Environment 0"] = (
        mock_command_queue
    )

    metadata = skeleton_environment_metadata()

    environment_manager.initialize_environments([metadata], hardware_metadata)

    assert not environment_manager.environment_sysid_stored_events[
        "Environment 0"
    ].is_set()


# endregion


# region Validation
@pytest.mark.parametrize(
    "environment_name_list, instance_list, expected",
    [
        (
            ["Environment 0", "Environment 1"],
            [EnvironmentMetadata, EnvironmentMetadata],
            True,
        ),
        (
            ["Environment 0", "Environment 1"],
            [None, EnvironmentMetadata],
            RattlesnakeError,
        ),
        (
            ["Environment 0", "Environment 0"],
            [EnvironmentMetadata, EnvironmentMetadata],
            RattlesnakeError,
        ),
        (
            ["0", "1", "2", "3", "4"],
            [
                EnvironmentMetadata,
                EnvironmentMetadata,
                EnvironmentMetadata,
                EnvironmentMetadata,
                EnvironmentMetadata,
            ],
            RattlesnakeError,
        ),
    ],
)
def test_environment_manager_validate_environment_metadata(
    environment_name_list,
    instance_list,
    expected,
    environment_manager,
    hardware_metadata,
):
    """
    Verifies valid metadata passes validation and invalid object types,
    duplicate names, or too many environments raise ``RattlesnakeError``.
    """
    environment_manager.queue_names = ["Environment 0", "Environment 1"]

    metadata_list = []
    for environment_name, instance in zip(environment_name_list, instance_list):
        metadata = mock.MagicMock(spec=instance)
        metadata.environment_name = environment_name
        metadata_list.append(metadata)

    if expected is RattlesnakeError:
        with pytest.raises(RattlesnakeError):
            environment_manager.validate_environment_metadata(
                metadata_list,
                hardware_metadata,
            )
    else:
        environment_manager.validate_environment_metadata(
            metadata_list,
            hardware_metadata,
        )
        for metadata in metadata_list:
            metadata.validate.assert_called_once_with(hardware_metadata)


def test_environment_manager_validate_environment_metadata_real_metadata(
    environment_manager,
    hardware_metadata,
):
    """
    Verifies that concrete mock metadata validates successfully.
    """
    metadata_list = [
        skeleton_environment_metadata(environment_name="Env A"),
        skeleton_environment_metadata(environment_name="Env B"),
    ]

    environment_manager.validate_environment_metadata(
        metadata_list,
        hardware_metadata,
    )


@pytest.mark.parametrize(
    "sysid_metadata_valid, environment_exists, environment_supports_sysid, expected",
    [
        (True, True, True, True),
        (False, True, True, RattlesnakeError),
        (True, False, True, RattlesnakeError),
        (True, True, False, RattlesnakeError),
    ],
)
def test_environment_manager_validate_system_id_metadata(
    sysid_metadata_valid,
    environment_exists,
    environment_supports_sysid,
    expected,
    environment_manager,
    hardware_metadata,
):
    """
    Verifies system-identification metadata validation.
    """
    sysid_environment_type = first_sysid_environment_type()

    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_names = (
        {"Environment 0": "Skeleton Environment"} if environment_exists else {}
    )
    environment_manager.environment_types = {
        "Environment 0": (
            sysid_environment_type
            if environment_supports_sysid
            else EnvironmentType.NONE
        )
    }

    sysid_metadata = (
        mock.MagicMock(spec=SysIdMetadata) if sysid_metadata_valid else object()
    )

    if expected is RattlesnakeError:
        with pytest.raises(RattlesnakeError):
            environment_manager.validate_system_id_metadata(
                sysid_metadata,
                hardware_metadata,
                "Skeleton Environment",
            )
    else:
        queue_name = environment_manager.validate_system_id_metadata(
            sysid_metadata,
            hardware_metadata,
            "Skeleton Environment",
        )
        assert queue_name == "Environment 0"


@pytest.mark.parametrize(
    "environment_exists, supports_sysid, valid_package_type, response_match, "
    "reference_match, expected",
    [
        (True, True, True, True, True, True),
        (False, True, True, True, True, RattlesnakeError),
        (True, False, True, True, True, RattlesnakeError),
        (True, True, False, True, True, RattlesnakeError),
        (True, True, True, False, True, RattlesnakeError),
        (True, True, True, True, False, RattlesnakeError),
    ],
)
def test_environment_manager_validate_system_id_package(
    environment_exists,
    supports_sysid,
    valid_package_type,
    response_match,
    reference_match,
    expected,
    environment_manager,
):
    """
    Verifies system-identification package validation.
    """
    sysid_environment_type = first_sysid_environment_type()

    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_names = (
        {"Environment 0": "Skeleton Environment"} if environment_exists else {}
    )
    environment_manager.environment_types = {
        "Environment 0": (
            sysid_environment_type if supports_sysid else EnvironmentType.NONE
        )
    }

    environment_metadata = mock.MagicMock()
    environment_metadata.num_response_channels = 2
    environment_metadata.num_reference_channels = 1
    environment_manager.environment_metadata = {"Environment 0": environment_metadata}

    if valid_package_type:
        data_package = mock.MagicMock(spec=SysIdDataPackage)
        data_package.num_response_channels = 2 if response_match else 3
        data_package.num_reference_channels = 1 if reference_match else 2
    else:
        data_package = object()

    if expected is RattlesnakeError:
        with pytest.raises(RattlesnakeError):
            environment_manager.validate_system_id_package(
                "Skeleton Environment",
                data_package,
            )
    else:
        queue_name = environment_manager.validate_system_id_package(
            "Skeleton Environment",
            data_package,
        )
        assert queue_name == "Environment 0"
        data_package.validate.assert_called_once_with()


@pytest.mark.parametrize(
    "environment_name, environment_type, instance, sysid_environment, "
    "sysid_stored, expected",
    [
        (
            "Skeleton Environment",
            EnvironmentType.NONE,
            EnvironmentInstructions,
            False,
            True,
            True,
        ),
        (
            "Wrong Name",
            EnvironmentType.NONE,
            EnvironmentInstructions,
            False,
            True,
            RattlesnakeError,
        ),
        (
            "Skeleton Environment",
            object(),
            EnvironmentInstructions,
            False,
            True,
            RattlesnakeError,
        ),
        (
            "Skeleton Environment",
            EnvironmentType.NONE,
            None,
            False,
            True,
            RattlesnakeError,
        ),
        (
            "Skeleton Environment",
            "SYSID",
            EnvironmentInstructions,
            True,
            False,
            RattlesnakeError,
        ),
        ("Skeleton Environment", "SYSID", EnvironmentInstructions, True, True, True),
    ],
)
def test_environment_manager_validate_instructions(
    environment_name,
    environment_type,
    instance,
    sysid_environment,
    sysid_stored,
    expected,
    environment_manager,
):
    """
    Verifies that valid instructions return a queue name and invalid
    instruction inputs raise ``RattlesnakeError``.
    """
    sysid_environment_type = (
        first_sysid_environment_type() if sysid_environment else None
    )
    stored_environment_type = (
        sysid_environment_type if sysid_environment else EnvironmentType.NONE
    )
    instruction_environment_type = (
        sysid_environment_type if environment_type == "SYSID" else environment_type
    )

    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_names = {"Environment 0": "Skeleton Environment"}
    environment_manager.environment_types = {"Environment 0": stored_environment_type}

    if sysid_stored:
        environment_manager.environment_sysid_stored_events["Environment 0"].set()
    else:
        environment_manager.environment_sysid_stored_events["Environment 0"].clear()

    instructions = mock.MagicMock(spec=instance)
    instructions.environment_name = environment_name
    instructions.environment_type = instruction_environment_type

    if expected is RattlesnakeError:
        with pytest.raises(RattlesnakeError):
            environment_manager.validate_environment_instructions(instructions)
    else:
        queue_name = environment_manager.validate_environment_instructions(instructions)
        assert queue_name == "Environment 0"
        instructions.validate.assert_called_once_with()


@pytest.mark.parametrize(
    "environment_name, instance, expected",
    [
        ("Environment Name", ProfileEvent, True),
        ("Invalid Name", ProfileEvent, RattlesnakeError),
        ("Environment Name", None, RattlesnakeError),
        ("Global", ProfileEvent, "Global"),
    ],
)
def test_environment_manager_validate_profile_events(
    environment_name,
    instance,
    expected,
    environment_manager,
):
    """
    Verifies profile events are annotated correctly and invalid events raise
    ``RattlesnakeError``.
    """
    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_names = {"Environment 0": "Environment Name"}
    environment_manager.environment_types = {"Environment 0": EnvironmentType.NONE}

    profile_event = mock.MagicMock(spec=instance)
    profile_event.environment_name = environment_name
    profile_event_list = [profile_event]

    if expected is RattlesnakeError:
        with pytest.raises(RattlesnakeError):
            environment_manager.validate_profile_events(profile_event_list)
    elif expected == "Global":
        environment_manager.validate_profile_events(profile_event_list)
        assert profile_event._queue_name == "Global"
        assert profile_event._environment_type == "Global"
    else:
        environment_manager.validate_profile_events(profile_event_list)
        assert profile_event._queue_name == "Environment 0"
        assert profile_event._environment_type == EnvironmentType.NONE


# endregion


# region Environment Operations
def test_environment_manager_clear_environment(environment_manager):
    """
    Verifies that environment tracking dictionaries are cleared and
    ``close_environments`` is called.
    """
    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_names = {"Environment 0": "Environment Name"}
    environment_manager.environment_types = {"Environment 0": EnvironmentType.NONE}
    environment_manager.environment_metadata = {
        "Environment 0": skeleton_environment_metadata()
    }
    environment_manager.environment_processes = {"Environment 0": mock.MagicMock()}

    environment_manager.close_environments = mock.MagicMock()

    environment_manager.clear_environments()

    assert environment_manager.queue_names == []
    assert environment_manager.environment_names == {}
    assert environment_manager.environment_types == {}
    assert environment_manager.environment_metadata == {}
    assert environment_manager.environment_processes == {}
    environment_manager.close_environments.assert_called_once_with()


def test_environment_manager_add_environment(
    environment_manager,
    hardware_metadata,
):
    """
    Verifies that adding an environment starts a process, stores mappings, sends
    initialization commands, and clears relevant events.
    """
    metadata = skeleton_environment_metadata(environment_name="Skeleton Environment")
    metadata.environment_type = EnvironmentType.NONE

    mock_process = mock.MagicMock()
    mock_process_class = mock.MagicMock(return_value=mock_process)
    environment_manager.new_process = mock_process_class

    mock_command_queue = mock.MagicMock()
    environment_manager.queue_container.environment_command_queues["Environment 0"] = (
        mock_command_queue
    )

    mock_process_function = mock.MagicMock()

    with mock.patch.dict(
        "rattlesnake.environment_manager.ENVIRONMENT_PROCESS",
        {EnvironmentType.NONE: mock_process_function},
        clear=False,
    ):
        queue_name = environment_manager.add_environment(
            metadata,
            hardware_metadata,
        )

    assert queue_name == "Environment 0"
    assert environment_manager.queue_names == ["Environment 0"]
    assert environment_manager.environment_names["Environment 0"] == (
        "Skeleton Environment"
    )
    assert environment_manager.environment_types["Environment 0"] == (
        EnvironmentType.NONE
    )
    assert environment_manager.environment_processes["Environment 0"] is mock_process

    mock_command_queue.assign_environment.assert_called_once_with(
        "Skeleton Environment"
    )
    mock_process_class.assert_called_once()
    mock_process.start.assert_called_once_with()

    mock_command_queue.put.assert_has_calls(
        [
            mock.call(
                "Environment Manager",
                (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata),
            ),
            mock.call(
                "Environment Manager",
                (GlobalCommands.INITIALIZE_ENVIRONMENT, metadata),
            ),
        ]
    )

    assert not environment_manager.environment_active_events["Environment 0"].is_set()
    assert not environment_manager.environment_sysid_active_events[
        "Environment 0"
    ].is_set()
    assert not environment_manager.environment_sysid_stored_events[
        "Environment 0"
    ].is_set()


def test_environment_manager_add_environment_error(
    environment_manager,
    hardware_metadata,
):
    """
    Verifies that adding an environment raises ``RattlesnakeError`` when no
    queues are available.
    """
    metadata = skeleton_environment_metadata()

    environment_manager.queue_names = [
        "Environment 0",
        "Environment 1",
        "Environment 2",
        "Environment 3",
    ]

    with pytest.raises(RattlesnakeError):
        environment_manager.add_environment(metadata, hardware_metadata)


@mock.patch("rattlesnake.environment_manager.datetime")
@pytest.mark.parametrize(
    "first_alive, second_alive", [(False, False), (True, False), (True, True)]
)
def test_environment_manager_remove_environment(
    mock_datetime,
    first_alive,
    second_alive,
    environment_manager,
):
    """
    Verifies that removing an environment sends a quit command, joins the
    process, force-closes unresponsive processes, and clears mappings.
    """
    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_names = {"Environment 0": "Environment Name"}
    environment_manager.environment_types = {"Environment 0": EnvironmentType.NONE}
    environment_manager.environment_metadata = {
        "Environment 0": skeleton_environment_metadata()
    }

    mock_log_file_queue = mock.MagicMock()
    mock_command_queue = mock.MagicMock()
    mock_process = mock.MagicMock()
    mock_process.is_alive.side_effect = [first_alive, second_alive]
    mock_close_event = mock.MagicMock()

    environment_manager.queue_container.log_file_queue = mock_log_file_queue
    environment_manager.queue_container.environment_command_queues["Environment 0"] = (
        mock_command_queue
    )
    environment_manager.environment_processes = {"Environment 0": mock_process}
    environment_manager.environment_close_events = {"Environment 0": mock_close_event}

    mock_datetime.now = fake_time

    environment_manager.remove_environment("Environment 0")

    assert environment_manager.queue_names == []
    assert environment_manager.environment_names == {}
    assert environment_manager.environment_types == {}
    assert environment_manager.environment_metadata == {}
    assert environment_manager.environment_processes == {}

    mock_command_queue.put.assert_called_once_with(
        "Environment Manager",
        (GlobalCommands.QUIT, None),
    )
    mock_process.join.assert_called()

    if first_alive:
        mock_close_event.set.assert_called_once_with()

    if first_alive and second_alive and not environment_manager.threaded:
        mock_process.terminate.assert_called_once_with()


def test_environment_manager_remove_environment_invalid_queue_name(
    environment_manager,
):
    """
    Verifies that removing an unassigned queue name raises
    ``RattlesnakeError``.
    """
    environment_manager.queue_names = ["Environment 0"]

    with pytest.raises(RattlesnakeError):
        environment_manager.remove_environment("Invalid Queue Name")


@pytest.mark.parametrize(
    "first_alive, second_alive", [(False, False), (True, False), (True, True)]
)
@mock.patch("rattlesnake.environment_manager.datetime")
def test_environment_manager_close_environment(
    mock_datetime,
    first_alive,
    second_alive,
    environment_manager,
):
    """
    Verifies that closing environments sends quit commands, joins processes,
    force-closes unresponsive processes, and flushes command queues.
    """
    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_names = {"Environment 0": "Environment Name"}
    environment_manager.environment_types = {"Environment 0": EnvironmentType.NONE}
    environment_manager.environment_metadata = {
        "Environment 0": skeleton_environment_metadata()
    }

    mock_log_file_queue = mock.MagicMock()
    mock_command_queue = mock.MagicMock()
    mock_process = mock.MagicMock()
    mock_process.is_alive.side_effect = [first_alive, second_alive]
    mock_close_event = mock.MagicMock()

    environment_manager.queue_container.log_file_queue = mock_log_file_queue
    environment_manager.queue_container.environment_command_queues["Environment 0"] = (
        mock_command_queue
    )
    environment_manager.environment_processes = {"Environment 0": mock_process}
    environment_manager.environment_close_events = {"Environment 0": mock_close_event}

    mock_datetime.now = fake_time

    environment_manager.close_environments()

    mock_command_queue.put.assert_called_once_with(
        "Environment Manager",
        (GlobalCommands.QUIT, None),
    )
    mock_process.join.assert_called()
    mock_command_queue.flush.assert_called_once_with("Environment Manager")

    if first_alive:
        mock_close_event.set.assert_called_once_with()

    if first_alive and second_alive and not environment_manager.threaded:
        mock_process.terminate.assert_called_once_with()


# endregion
