from rattlesnake.environment_manager import EnvironmentManager
from rattlesnake.utilities import GlobalCommands
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.environment.abstract_environment import EnvironmentMetadata, EnvironmentInstructions
from mock_objects.mock_hardware import MockHardwareMetadata
from mock_objects.mock_environment import MockEnvironmentType, MockEnvironmentMetadata, MockEnvironmentInstructions, IMPLEMENTED_ENVIRONMENT
from mock_objects.mock_utilities import mock_queue_container, mock_event_container, fake_time
import pytest
import multiprocessing as mp
from unittest import mock


# region: Fixtures
@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def environment_manager(request):
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    environment_manager = EnvironmentManager(queue_container, event_container, use_thread)

    return environment_manager


# region: EnvironmentManager
@pytest.mark.parametrize("use_thread", [True, False])
def test_environment_manager_init(use_thread):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    environment_manager = EnvironmentManager(queue_container, event_container, use_thread)

    assert isinstance(environment_manager, EnvironmentManager)
    assert environment_manager.threaded == use_thread


@pytest.mark.parametrize(
    "queue_names, available_queues",
    [
        (["Environment 0", "Environment 1"], ["Environment 2", "Environment 3"]),
        ([], ["Environment 0", "Environment 1", "Environment 2", "Environment 3"]),
    ],
)
def test_environment_manager_available_queues(queue_names, available_queues, environment_manager):
    environment_manager.queue_names = queue_names

    assert environment_manager.available_queues == available_queues
    assert environment_manager.num_queues == 4


def test_environment_manager_queue_names_dict(environment_manager):
    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_names["Environment 0"] = "Environment Name"

    queue_names_dict = environment_manager.queue_names_dict
    assert queue_names_dict["Environment Name"] == "Environment 0"


def test_environment_manager_ready_event_list(environment_manager):
    environment_manager.queue_names = ["Environment 0", "Environment 1"]
    ready_event_list = [
        environment_manager.environment_ready_events["Environment 0"],
        environment_manager.environment_ready_events["Environment 1"],
    ]

    assert environment_manager.ready_event_list == ready_event_list


def test_environment_manager_set_ready_events(environment_manager):
    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_ready_events["Environment 0"].clear()
    environment_manager.set_ready_events()

    assert environment_manager.environment_ready_events["Environment 0"].is_set()


def test_environment_manager_initialize_hardware(environment_manager):
    hardware_metadata = MockHardwareMetadata()
    environment_manager.queue_names = ["Environment 0"]
    mock_command_queue = mock.MagicMock()
    environment_manager.queue_container.environment_command_queues["Environment 0"] = mock_command_queue
    environment_manager.initialize_hardware(hardware_metadata)

    mock_command_queue.put.assert_called_with("Environment Manager", (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata))
    assert environment_manager.hardware_metadata == hardware_metadata


@pytest.mark.parametrize(
    "existing_metadata, valid_metadata, expected",
    [
        (({"Environment 0": MockEnvironmentMetadata()}, {"Environment 0": MockEnvironmentType.ENVIRONMENT}, ["Environment 0"]), True, "Put"),
        (({}, {}, []), False, TypeError),
        (({}, {}, []), True, "Add"),
        (
            (
                {"Environment 0": MockEnvironmentMetadata(), "Environment 1": MockEnvironmentMetadata()},
                {"Environment 0": MockEnvironmentType.ENVIRONMENT, "Environment 1": MockEnvironmentType.ENVIRONMENT},
                ["Environment 0", "Environment 1"],
            ),
            True,
            "Remove",
        ),
    ],
)
def test_environment_manager_initialize_environment(existing_metadata, valid_metadata, expected, environment_manager):
    metadata_dict, type_dict, queue_names = existing_metadata

    hardware_metadata = MockHardwareMetadata()
    environment_manager.hardware_metadata = hardware_metadata

    environment_metadata = MockEnvironmentMetadata()
    environment_metadata.validate = mock.MagicMock(return_value=valid_metadata)
    metadata_list = [environment_metadata]

    environment_manager.queue_names = queue_names
    environment_manager.environment_types = type_dict
    environment_manager.environment_metadata = metadata_dict

    mock_command_queue = mock.MagicMock()
    environment_manager.queue_container.environment_command_queues["Environment 0"] = mock_command_queue

    environment_manager.add_environment = mock.MagicMock()
    environment_manager.remove_environment = mock.MagicMock()

    if expected is TypeError:
        with pytest.raises(TypeError):
            environment_manager.initialize_environments(metadata_list)
    else:
        environment_manager.initialize_environments(metadata_list)
        if expected == "Put":
            expected_environment_calls = [
                mock.call.put("Environment Manager", (GlobalCommands.INITIALIZE_HARDWARE, environment_manager.hardware_metadata)),
                mock.call.put("Environment Manager", (GlobalCommands.INITIALIZE_ENVIRONMENT, environment_metadata)),
            ]
            mock_command_queue.assert_has_calls(expected_environment_calls)
        elif expected == "Add":
            environment_manager.add_environment.assert_called()
        elif expected == "Remove":
            environment_manager.remove_environment.assert_called_with("Environment 1")


@pytest.mark.parametrize(
    "environment_name_list, validate_list, instance_list, expected",
    [
        (["Environment 0", "Environment 1"], [True, True], [EnvironmentMetadata, EnvironmentMetadata], True),
        (["Environment 0", "Environment 1"], [True, True], [None, EnvironmentMetadata], TypeError),
        (["Environment 0", "Environment 0"], [True, True], [EnvironmentMetadata, EnvironmentMetadata], ValueError),
        (["Environment Name"], [False], [EnvironmentMetadata], ValueError),
        (
            ["0", "1", "2", "3", "4"],
            [True, True, True, True, True],
            [EnvironmentMetadata, EnvironmentMetadata, EnvironmentMetadata, EnvironmentMetadata, EnvironmentMetadata],
            IndexError,
        ),
    ],
)
def test_environment_manager_validate_environment_metadata(environment_name_list, validate_list, instance_list, expected, environment_manager):
    environment_manager.queue_names = ["Environment 0", "Environment 1"]
    metadata_list = []

    for environment_name, validate, instance in zip(
        environment_name_list,
        validate_list,
        instance_list,
    ):
        mock_metadata = mock.MagicMock(spec=instance)
        mock_metadata.environment_name = environment_name
        mock_metadata.validate.return_value = validate
        metadata_list.append(mock_metadata)

    if expected is TypeError:
        with pytest.raises(TypeError):
            environment_manager.validate_environment_metadata(metadata_list)
    elif expected is ValueError:
        with pytest.raises(ValueError):
            environment_manager.validate_environment_metadata(metadata_list)
    elif expected is IndexError:
        with pytest.raises(IndexError):
            environment_manager.validate_environment_metadata(metadata_list)
    elif expected:
        valid_metadata_list = environment_manager.validate_environment_metadata(metadata_list)
        assert valid_metadata_list


@pytest.mark.parametrize(
    "environment_name, environment_type, validate, instance, expected",
    [
        ("Mock Environment", MockEnvironmentType.ENVIRONMENT, True, EnvironmentInstructions, True),
        ("Wrong Name", MockEnvironmentType.ENVIRONMENT, True, EnvironmentInstructions, KeyError),
        ("Mock Environment", None, True, EnvironmentInstructions, TypeError),
        ("Mock Environment", MockEnvironmentType.ENVIRONMENT, True, None, TypeError),
        ("Mock Environment", MockEnvironmentType.ENVIRONMENT, False, EnvironmentInstructions, ValueError),
    ],
)
def test_environment_manager_validate_instructions(environment_name, environment_type, validate, instance, expected, environment_manager):
    environment_instructions = mock.MagicMock(spec=instance)
    environment_instructions.environment_name = environment_name
    environment_instructions.environment_type = environment_type
    environment_instructions.validate.return_value = validate
    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_names = {"Environment 0": "Mock Environment"}
    environment_manager.environment_types = {"Environment 0": MockEnvironmentType.ENVIRONMENT}

    if expected is True:
        valid_instruction = environment_manager.validate_environment_instructions(environment_instructions)
        assert valid_instruction
    elif expected is KeyError:
        with pytest.raises(KeyError):
            environment_manager.validate_environment_instructions(environment_instructions)
    elif expected is TypeError:
        with pytest.raises(TypeError):
            environment_manager.validate_environment_instructions(environment_instructions)
    elif expected is ValueError:
        with pytest.raises(ValueError):
            environment_manager.validate_environment_instructions(environment_instructions)


@pytest.mark.parametrize(
    "environment_name, instance, expected",
    [
        ("Environment Name", ProfileEvent, True),
        ("Invalid Name", ProfileEvent, KeyError),
        ("Environment Name", None, TypeError),
        ("Global", ProfileEvent, "Global"),
    ],
)
def test_environment_mananger_validate_profile_events(environment_name, instance, expected, environment_manager):
    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_names["Environment 0"] = "Environment Name"
    environment_manager.environment_types["Environment 0"] = "Environment Type"

    profile_event = mock.MagicMock(spec=instance)
    profile_event.environment_name = environment_name
    profile_event_list = [profile_event]

    if expected is TypeError:
        with pytest.raises(TypeError):
            environment_manager.validate_profile_events(profile_event_list)
    elif expected is KeyError:
        with pytest.raises(KeyError):
            environment_manager.validate_profile_events(profile_event_list)
    elif expected == "Global":
        valid_profile = environment_manager.validate_profile_events(profile_event_list)
        profile_event = profile_event_list[0]
        assert valid_profile
        assert profile_event._queue_name == "Global"
        assert profile_event._environment_type == "Global"
    elif expected:
        valid_profile = environment_manager.validate_profile_events(profile_event_list)
        profile_event = profile_event_list[0]
        assert valid_profile
        assert profile_event._queue_name == "Environment 0"
        assert profile_event._environment_type == "Environment Type"


def test_environment_manager_clear_environment(environment_manager):
    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_names = {"Environment 0": "Environment Name"}
    environment_manager.environment_types = {"Environment 0": MockEnvironmentType.ENVIRONMENT}
    environment_manager.environment_metadata = {"Environment 0": MockEnvironmentMetadata()}
    environment_manager.environment_processes = {"Environment 0": mp.Process()}
    environment_manager.environment_events = {"Environment 0": mp.Event()}
    mock_close_process = mock.MagicMock()
    environment_manager.close_environments = mock_close_process
    environment_manager.clear_environments()

    assert environment_manager.queue_names == []
    assert environment_manager.environment_names == {}
    assert environment_manager.environment_types == {}
    assert environment_manager.environment_metadata == {}
    assert environment_manager.environment_processes == {}
    mock_close_process.assert_called()


@pytest.mark.parametrize("environment_type", [None, *IMPLEMENTED_ENVIRONMENT])
def test_environment_manager_add_environment(environment_type, environment_manager):
    metadata = MockEnvironmentMetadata()
    metadata.environment_type = environment_type

    mock_process = mock.MagicMock()
    mock_event = mock.MagicMock()
    environment_manager.new_process = mock.MagicMock(return_value=mock_process)
    environment_manager.new_event = mock.MagicMock(return_value=mock_event)

    environment_manager.add_environment(metadata)
    if environment_type == None:
        assert True
    else:
        assert environment_manager.queue_names == ["Environment 0"]
        assert environment_manager.environment_names["Environment 0"] == "Mock Environment"
        assert environment_manager.environment_types["Environment 0"] == environment_type
        assert environment_manager.environment_processes["Environment 0"] == mock_process
        assert not environment_manager.environment_close_events["Environment 0"].is_set()
        assert environment_manager.environment_metadata["Environment 0"] == metadata
        assert metadata.queue_name == "Environment 0"


@pytest.mark.parametrize("queue_names, expected", [([], True), (["Environment 0", "Environment 1", "Environment 2", "Environment 3"], KeyError)])
def test_environment_mananger_add_environment_key_error(queue_names, expected, environment_manager):
    metadata = MockEnvironmentMetadata()
    metadata.environment_type = None

    environment_manager.queue_names = queue_names

    if expected is KeyError:
        with pytest.raises(KeyError):
            environment_manager.add_environment(metadata)
    elif expected:
        environment_manager.add_environment(metadata)
        assert True


@mock.patch("rattlesnake.environment_manager.datetime")
@pytest.mark.parametrize("is_alive", [True, False])
def test_environment_manager_remove_environment(mock_time, is_alive, environment_manager):
    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_names = {"Environment 0": "Environment Name"}
    environment_manager.environment_types = {"Environment 0": MockEnvironmentType.ENVIRONMENT}
    environment_manager.environment_metadata = {"Environment 0": MockEnvironmentMetadata()}

    mock_log_file_queue = mock.MagicMock()
    mock_command_queue = mock.MagicMock()
    mock_process = mock.MagicMock()
    mock_process.is_alive.return_value = is_alive
    mock_event = mock.MagicMock()
    environment_manager.queue_container.log_file_queue = mock_log_file_queue
    environment_manager.queue_container.environment_command_queues["Environment 0"] = mock_command_queue
    environment_manager.environment_processes = {"Environment 0": mock_process}
    environment_manager.environment_close_events = {"Environment 0": mock_event}
    mock_time.now = fake_time

    environment_manager.remove_environment("Environment 0")
    assert environment_manager.queue_names == []
    assert environment_manager.environment_names == {}
    assert environment_manager.environment_types == {}
    assert environment_manager.environment_metadata == {}
    assert environment_manager.environment_processes == {}

    mock_command_queue.put.assert_called_with("Environment Manager", (GlobalCommands.QUIT, None))
    mock_process.join.assert_called()
    if is_alive:
        mock_event.set.assert_called()


def test_environment_manager_remove_environment_invalid_queue_name(environment_manager):
    environment_manager.queue_names = ["Environment 0"]

    with pytest.raises(KeyError):
        environment_manager.remove_environment("Invalid Queue Name")


@pytest.mark.parametrize("is_alive", [True, False])
@mock.patch("rattlesnake.environment_manager.datetime")
def test_environment_mananger_close_environment(mock_time, is_alive, environment_manager):
    environment_manager.queue_names = ["Environment 0"]
    environment_manager.environment_names = {"Environment 0": "Environment Name"}
    environment_manager.environment_types = {"Environment 0": MockEnvironmentType.ENVIRONMENT}
    environment_manager.environment_metadata = {"Environment 0": MockEnvironmentMetadata()}

    mock_log_file_queue = mock.MagicMock()
    mock_command_queue = mock.MagicMock()
    mock_process = mock.MagicMock()
    mock_process.is_alive.return_value = is_alive
    mock_event = mock.MagicMock()
    environment_manager.queue_container.log_file_queue = mock_log_file_queue
    environment_manager.queue_container.environment_command_queues["Environment 0"] = mock_command_queue
    environment_manager.environment_processes = {"Environment 0": mock_process}
    environment_manager.environment_close_events = {"Environment 0": mock_event}
    mock_time.now = fake_time

    environment_manager.close_environments()

    mock_command_queue.put.assert_called_with("Environment Manager", (GlobalCommands.QUIT, None))
    mock_process.join.assert_called()
    if is_alive:
        mock_event.set.assert_called()
