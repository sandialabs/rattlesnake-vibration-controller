from rattlesnake.process.output import OutputProcess, output_process
from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.utilities import GlobalCommands
from mock_objects.mock_hardware import MockHardwareMetadata, output_dict, metadata_attr_dict, UNIMPLEMENTED_HARDWARE
from mock_objects.mock_environment import MockEnvironmentMetadata
from mock_objects.mock_utilities import mock_queue_container, mock_event_container
import pytest
import multiprocessing as mp
import numpy as np
from unittest import mock


# region: Fixtures
IMPLEMENTED_HARDWARE = [hardware for hardware in HardwareType if hardware not in UNIMPLEMENTED_HARDWARE]


@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def output(request):
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    output = OutputProcess(
        "Process Name",
        queue_container,
        event_container.output_active_event,
        event_container.output_ready_event,
    )
    return output


# region: OutputProcess
# Test OutputProcess initialization
@pytest.mark.parametrize("use_thread", [True, False])
def test_output_init(use_thread):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    output = OutputProcess(
        "Process Name",
        queue_container,
        event_container.output_active_event,
        event_container.output_ready_event,
    )

    # Make sure it is the correct class
    assert isinstance(output, OutputProcess)
    assert isinstance(output, AbstractMessageProcess)


def test_output_properties(output):
    # Test if object is the correct class
    assert isinstance(output, OutputProcess)
    # Test the output_active property
    assert output.output_active == False
    # Test the output_active setter
    output.set_active()
    assert output.output_active == True
    output.clear_active()
    assert output.output_active == False


@pytest.mark.parametrize("hardware_type", [*IMPLEMENTED_HARDWARE])
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_output_process_initialize_hardware(mock_log, hardware_type, output):
    hardware_metadata = MockHardwareMetadata()
    hardware_lookup = output_dict()
    attr_lookup = metadata_attr_dict()
    hardware_metadata.hardware_type = hardware_type
    mock_existing_hardware = mock.MagicMock()
    output.hardware = mock_existing_hardware

    with mock.patch(hardware_lookup[hardware_type]) as mock_hardware:
        for attr in attr_lookup[hardware_type]:
            setattr(hardware_metadata, attr, 0)

        output.clear_ready()
        output.initialize_hardware(hardware_metadata)
        mock_hardware().initialize_hardware.assert_called()

    mock_log.assert_called_with("Initializing Hardware")
    assert output.hardware_metadata == hardware_metadata
    assert output.ready_event.is_set()
    mock_existing_hardware.close.assert_called_once()


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_output_process_initialize_hardware_value_error(mock_log, output):
    hardware_metadata = MockHardwareMetadata()

    with pytest.raises(TypeError):
        output.initialize_hardware(hardware_metadata)


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_output_process_initialize_environment(mock_log, output):
    hardware_metadata = MockHardwareMetadata()
    output.hardware_metadata = hardware_metadata
    environment_metadata = MockEnvironmentMetadata()
    output.clear_ready()
    output.initialize_environment({"Environment 0": environment_metadata})

    mock_log.assert_called_with("Initializing Environment")
    assert output.environment_list == ["Environment 0"]
    assert output.environment_active_flags["Environment 0"] == False
    assert output.environment_starting_up_flags["Environment 0"] == False
    assert output.environment_shutting_down_flags["Environment 0"] == False
    assert output.environment_first_data["Environment 0"] == False
    assert output.environment_output_channels["Environment 0"] == [0]
    np.testing.assert_array_equal(output.environment_data_out_remainders["Environment 0"], np.zeros((1, 0)))
    assert output.ready_event.is_set()


@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_process_stop_output(mock_log, output):
    output.stop_output(None)

    mock_log.assert_called_with("Starting Shutdown Procedure")
    assert output.shutdown_flag == True


@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_process_start_environment(mock_log, output):

    output.environment_list = ["Environment 0"]
    output.environment_active_flags["Environment 0"] = False
    output.environment_starting_up_flags["Environment 0"] = False
    output.environment_shutting_down_flags["Environment 0"] = False
    output.environment_first_data["Environment 0"] = False

    output.start_environment("Environment 0")

    mock_log.assert_called_with("Started Environment Environment 0")
    assert output.environment_starting_up_flags["Environment 0"] == True
    assert output.environment_shutting_down_flags["Environment 0"] == False
    assert output.environment_active_flags["Environment 0"] == False


@mock.patch("rattlesnake.process.output.flush_queue")
@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_process_quit(mock_log, mock_flush, output):
    mock_hardware = mock.MagicMock()
    output.hardware = mock_hardware

    output.quit(None)

    mock_log.assert_called_with("Flushed 0 items out of queues")
    mock_hardware.close.assert_called()


# region: output_process
# Prevent run while loop from starting
@pytest.mark.parametrize("use_thread", [True, False])
@mock.patch("rattlesnake.process.output.OutputProcess")
def test_output_process_func(mock_output, use_thread):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    output_process(
        queue_container,
        event_container.output_active_event,
        event_container.output_ready_event,
        event_container.output_close_event,
    )

    mock_instance = mock_output.return_value
    mock_instance.run.assert_called()
