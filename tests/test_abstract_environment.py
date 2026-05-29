import multiprocessing as mp
from queue import Empty as QueueEmpty
from unittest import mock

import netCDF4 as nc4
import pytest

from mock_objects.mock_environment import (
    IMPLEMENTED_ENVIRONMENT,
    MockEnvironment,
    MockEnvironmentInstructions,
    MockEnvironmentMetadata,
    environment_dict,
    environment_metadata_dict,
    build_environment,
)
from mock_objects.mock_hardware import MockHardwareMetadata
from mock_objects.mock_utilities import (
    fake_time,
    mock_channel_list,
    mock_channel_list_bools,
    mock_event_container,
    mock_queue_container,
)
from rattlesnake.utilities import RattlesnakeError
from rattlesnake.environment.abstract_environment import (
    Environment,
    EnvironmentInstructions,
    EnvironmentMetadata,
    process,
)
from rattlesnake.environment.environment_registry import SYSID_ENVIRONMENTS
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.hardware.hardware_utilities import Channel


# region Fixtures
@pytest.fixture
def environment_metadata():
    return MockEnvironmentMetadata()


@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def environment(request):
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    environment = MockEnvironment(
        "Environment Name",
        "Environment 0",
        queue_container,
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.environment_active_events["Environment 0"],
        event_container.environment_ready_events["Environment 0"],
    )

    return environment


# region Metadata
def test_environment_metadata_init():
    environment_metadata = MockEnvironmentMetadata()

    assert isinstance(environment_metadata, EnvironmentMetadata)
    assert hasattr(environment_metadata, "sample_rate")
    assert hasattr(environment_metadata, "channel_list_bools")
    assert hasattr(environment_metadata, "environment_name")
    assert hasattr(environment_metadata, "environment_type")
    assert hasattr(environment_metadata, "queue_name")


@pytest.mark.parametrize(
    "channel_list_bools, expected",
    [
        ([True, True], [0, 1]),
        ([True, False], [0]),
    ],
)
def test_environment_metadata_channel_indices(
    channel_list_bools, expected, environment_metadata
):
    environment_metadata.channel_list_bools = channel_list_bools

    channel_list_indices = environment_metadata.map_channel_indices()
    assert channel_list_indices == expected


@pytest.mark.parametrize(
    "environment_name, environment_type, expected",
    [
        ("Environment Name", EnvironmentType.TIME, True),
        (0, EnvironmentType.TIME, RattlesnakeError),
        ("Environment Name", 0, RattlesnakeError),
    ],
)
def test_environment_metadata_validate(
    environment_name, environment_type, expected, environment_metadata
):
    mock_hardware_metadata = MockHardwareMetadata()
    environment_metadata.environment_name = environment_name
    environment_metadata.environment_type = environment_type

    if expected == True:
        environment_metadata.validate(mock_hardware_metadata)
        assert True
    elif expected == RattlesnakeError:
        with pytest.raises(RattlesnakeError):
            environment_metadata.validate(mock_hardware_metadata)


def test_environment_metadata_save_metadata_to_netcdf(environment_metadata):
    dataset = nc4.Dataset("temp.nc", mode="w", diskless=True, persist=False)
    netcdf_group = dataset.createGroup("temp_group")

    environment_metadata.save_metadata_to_netcdf(netcdf_group)

    assert True


# region Instructions
def test_environment_instructions_init():
    environment_instructions = MockEnvironmentInstructions()

    assert isinstance(environment_instructions, EnvironmentInstructions)
    assert hasattr(environment_instructions, "environment_type")
    assert hasattr(environment_instructions, "environment_name")


def test_environment_instructions_validate():
    environment_instructions = MockEnvironmentInstructions()

    environment_instructions.validate()
    assert True


# region Environment
@pytest.mark.parametrize("use_thread", [True, False])
def test_environment_init(use_thread):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    environment = MockEnvironment(
        "Environment Name",
        "Environment 0",
        queue_container,
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.environment_active_events["Environment 0"],
        event_container.environment_ready_events["Environment 0"],
    )

    assert isinstance(environment, Environment)


def test_environment_properties(environment):
    environment.acquisition_active
    environment.output_active
    environment.environment_command_queue
    environment.data_in_queue
    environment.data_out_queue
    environment.gui_update_queue
    environment.controller_command_queue
    environment.log_file_queue
    environment.queue_name
    environment.command_map
    environment._ready_event
    environment._active_event

    assert True


def test_environment_setters(environment):
    assert environment.active == False
    environment.set_active()
    assert environment.active == True
    environment.clear_active()
    assert environment.active == False


def test_environment_set_ready(environment):
    environment._ready_event.clear()
    environment.set_ready()

    assert environment._ready_event.is_set()


def test_environment_clear_ready(environment):
    environment._ready_event.set()
    environment.clear_ready()

    assert not environment._ready_event.is_set()


def test_environment_functions(environment, environment_metadata):
    hardware_metadata = MockHardwareMetadata()

    environment.initialize_hardware(hardware_metadata)
    environment.initialize_environment(environment_metadata)
    environment.stop_environment(None)
    environment.quit(None)

    assert True


@mock.patch("rattlesnake.environment.abstract_environment.datetime")
def test_environment_log(mock_time, environment):
    mock_log_file_queue = mock.MagicMock()
    environment._log_file_queue = mock_log_file_queue
    mock_time.now = fake_time
    environment.log("Test Message")

    mock_log_file_queue.put.assert_called_once_with(
        "Datetime: Environment Name -- Test Message\n"
    )


def test_environment_map_command(environment):
    key = "Test Key"

    def function():
        return "Test Function"

    environment.map_command(key, function)

    # Test that the key maps to the function
    data = environment.command_map[key]
    assert data == function


@pytest.mark.parametrize(
    "mock_function, mock_key",
    [
        (mock.MagicMock(return_value=False), "Test Key"),
        (mock.MagicMock(side_effect=KeyError), "Test Key"),
        (mock.MagicMock(return_value=False), "Not a key"),
    ],
)
# Force get command to return values
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.get")
# Prevent from writing to log_file_queue
@mock.patch("rattlesnake.environment.abstract_environment.Environment.log")
def test_environment_run(mock_log, mock_get, mock_function, mock_key, environment):
    # Add the key function and quit function to the command map
    environment._command_map = {
        mock_key: mock_function,
        "Quit Key": environment.quit,
    }

    # Make the get command return "Test Key", then "Quit Key"
    mock_get.side_effect = [(QueueEmpty), ("Test Key", None), ("Quit Key", None)]
    mock_shutdown = mock.MagicMock()
    mock_shutdown.is_set.return_value = False

    environment.run(mock_shutdown)

    # Test that the function was called if the key exists
    if mock_key == "Test Key":
        mock_function.assert_called()
    # Test that the quit command was ran
    mock_log.assert_called_with("Stopping Process")


# region Process
@pytest.mark.parametrize("use_thread", [True, False])
@mock.patch("rattlesnake.environment.abstract_environment.Environment")
def test_run_process(mock_process_class, use_thread):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    process(
        "Environment Name",
        "Environment 0",
        queue_container.environment_command_queues["Environment 0"],
        queue_container.gui_update_queue,
        queue_container.controller_command_queue,
        queue_container.log_file_queue,
        queue_container.environment_data_in_queues["Environment 0"],
        queue_container.environment_data_out_queues["Environment 0"],
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.environment_active_events["Environment 0"],
        event_container.environment_ready_events["Environment 0"],
        event_container.environment_close_events["Environment 0"],
        event_container.environment_sysid_active_events["Environment 0"],
        event_container.environment_sysid_stored_events["Environment 0"],
        use_thread,
    )

    mock_instance = mock_process_class.return_value
    mock_instance.run.assert_called()


# region Ready Checks
@pytest.mark.parametrize("environment_type", [*IMPLEMENTED_ENVIRONMENT])
@pytest.mark.parametrize("use_thread", [True, False])
def test_environment_init(environment_type, use_thread):
    environment_lookup = environment_dict()
    mock_queues = mock.MagicMock()
    event_container = mock_event_container(use_thread)
    new_environment = environment_lookup[environment_type]
    event_container.environment_ready_events["Environment 0"].clear()
    if environment_type in SYSID_ENVIRONMENTS:
        new_environment(
            "Environment Name",
            "Queue Name",
            mock_queues,
            event_container.acquisition_active_event,
            event_container.output_active_event,
            event_container.environment_active_events["Environment 0"],
            event_container.environment_ready_events["Environment 0"],
            event_container.environment_sysid_active_events["Environment 0"],
            event_container.environment_sysid_stored_events["Environment 0"],
        )
    else:
        new_environment(
            "Environment Name",
            "Queue Name",
            mock_queues,
            event_container.acquisition_active_event,
            event_container.output_active_event,
            event_container.environment_active_events["Environment 0"],
            event_container.environment_ready_events["Environment 0"],
        )
    assert event_container.environment_ready_events["Environment 0"].is_set()
    assert not event_container.environment_active_events["Environment 0"].is_set()
    assert not event_container.environment_sysid_active_events["Environment 0"].is_set()
    assert not event_container.environment_sysid_stored_events["Environment 0"].is_set()


@pytest.mark.parametrize("environment_type", [*IMPLEMENTED_ENVIRONMENT])
@pytest.mark.parametrize("use_thread", [True, False])
def test_environment_initialize_hardware(environment_type, use_thread):
    mock_queues = mock.MagicMock()
    event_container = mock_event_container(use_thread)
    environment = build_environment(environment_type, mock_queues, event_container)

    hardware_metadata = MockHardwareMetadata()
    event_container.environment_ready_events["Environment 0"].clear()
    environment.initialize_hardware(hardware_metadata)

    assert event_container.environment_ready_events["Environment 0"].is_set()


@pytest.mark.parametrize("environment_type", [*IMPLEMENTED_ENVIRONMENT])
@pytest.mark.parametrize("use_thread", [True, False])
def test_environment_initialize_environment(environment_type, use_thread):
    mock_queues = mock.MagicMock()
    event_container = mock_event_container(use_thread)
    environment = build_environment(environment_type, mock_queues, event_container)
    hardware_metadata = MockHardwareMetadata()
    environment.initialize_hardware(hardware_metadata)

    mock_environment_metadata = mock.MagicMock()
    event_container.environment_ready_events["Environment 0"].clear()
    environment.initialize_environment(mock_environment_metadata)

    assert event_container.environment_ready_events["Environment 0"].is_set()


@pytest.mark.parametrize("environment_type", [*IMPLEMENTED_ENVIRONMENT])
@pytest.mark.parametrize("use_thread", [True, False])
def test_environment_stop_environment(environment_type, use_thread):
    environment_lookup = environment_dict()
    mock_queues = mock.MagicMock()
    event_container = mock_event_container(use_thread)
    environment = build_environment(environment_type, mock_queues, event_container)
    hardware_metadata = MockHardwareMetadata()
    environment.initialize_hardware(hardware_metadata)
    mock_environment_metadata = mock.MagicMock()
    environment.initialize_environment(mock_environment_metadata)

    mock_data = mock.MagicMock()
    environment.stop_environment(mock_data)

    # Theres not really a good way to check if the environment active
    # event got cleared because it usually happens in the control loop
    assert True
