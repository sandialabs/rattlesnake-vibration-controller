from rattlesnake.process.output import OutputProcess, output_process
from rattlesnake.utilities import (
    VerboseMessageQueue,
    QueueContainer,
    Channel,
    DataAcquisitionParameters,
)
from rattlesnake.environment.environment_utilities import ControlTypes
from functions.common_functions import create_hardware_dict_output
from unittest import mock
import multiprocessing as mp
import pytest
import numpy as np


# Create log_file_queue
@pytest.fixture
def log_file_queue():
    return mp.Queue()


# Create modal environment
@pytest.fixture
def environments():
    control_type = ControlTypes(6)
    return [[control_type, control_type.name.title()]]


# Create a queue container
@pytest.fixture
def queue_container(log_file_queue):
    queue_container = QueueContainer(
        VerboseMessageQueue(log_file_queue, "Controller Communication Queue"),
        VerboseMessageQueue(log_file_queue, "Acquisition Command Queue"),
        VerboseMessageQueue(log_file_queue, "Output Command Queue"),
        VerboseMessageQueue(log_file_queue, "Streaming Command Queue"),
        log_file_queue,
        mp.Queue(),
        mp.Queue(),
        mp.Queue(),
        {"Modal": VerboseMessageQueue(log_file_queue, "Environment Command Queue")},
        {"Modal": mp.Queue()},
        {"Modal": mp.Queue()},
    )
    return queue_container


@pytest.fixture
def output_process_obj(queue_container, environments):
    output_process_obj = OutputProcess(
        "Process Name", queue_container, environments, mp.Value("i", 0)
    )
    return output_process_obj


@pytest.fixture
def hardware_dict():
    return create_hardware_dict_output()


@pytest.fixture
def channel_list():
    return [
        Channel.from_channel_table_row(
            [
                "221",
                "Y+",
                "",
                "19644",
                "X+",
                "",
                "",
                "",
                "",
                "",
                "Virtual",
                "",
                "Accel",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        ),
        Channel.from_channel_table_row(
            [
                "221",
                "Y+",
                "",
                "19644",
                "X+",
                "",
                "",
                "",
                "",
                "",
                "Virtual",
                "",
                "Force",
                "",
                "",
                "",
                "",
                "",
                "Phys_dev",
                "",
                "5",
                "10",
            ]
        ),
    ]


@pytest.fixture()
def data_acquisition_parameters(channel_list):
    sample_rate = 2000
    time_per_read = 0.25
    time_per_write = 0.25
    output_oversample = 2
    hardware_selector_idx = 6
    hardware_file = "ExampleFile.nc4"
    environments = ["Modal"]
    environment_booleans = [[True]]
    acquisition_processes = 1
    task_trigger = 0
    task_trigger_output_channel = ""
    data_acquisition_parameters = DataAcquisitionParameters(
        channel_list,
        sample_rate,
        round(sample_rate * time_per_read),
        round(sample_rate * time_per_write * output_oversample),
        hardware_selector_idx,
        hardware_file,
        environments,
        environment_booleans,
        output_oversample,
        maximum_acquisition_processes=acquisition_processes,
        task_trigger=task_trigger,
        task_trigger_output_channel=task_trigger_output_channel,
    )

    return data_acquisition_parameters


# Test the OutputProcess intialization
def test_output_process_init(queue_container, environments):
    output_process = OutputProcess("Process Name", queue_container, environments, mp.Value("i", 0))

    # Test if object is the correct class
    assert isinstance(output_process, OutputProcess)
    # Test the output_active property
    assert output_process.output_active == False
    # Test the output_active setter
    output_process.output_active = True
    assert output_process.output_active == True


@pytest.mark.parametrize("hardware", [None, mock.MagicMock()])
@pytest.mark.parametrize("hardware_idx", [0, 1, 2, 4, 5])
@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_process_initialize_data_acquisition(
    mock_log, hardware, hardware_idx, output_process_obj, data_acquisition_parameters, hardware_dict
):
    data_acquisition_parameters.hardware = hardware_idx

    with mock.patch(hardware_dict[hardware_idx]) as mock_hardware:
        data = (data_acquisition_parameters, {"Environment Name": [1, 2]})
        output_process_obj.hardware = hardware

        output_process_obj.initialize_data_acquisition(data)

        mock_hardware().set_up_data_output_parameters_and_channels.assert_called_with(
            data_acquisition_parameters, data_acquisition_parameters.channel_list
        )

    # Test if log message was stored
    mock_log.assert_called_with("Initializing Data Acquisition")

    assert output_process_obj.environment_output_channels["Environment Name"] == np.array([0])


# @mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
# @mock.patch("rattlesnake.process.acquisition.mp.queues.Queue.put")
# @mock.patch("rattlesnake.process.acquisition.mp.queues.Queue.get_nowait")
# @mock.patch("rattlesnake.process.output.OutputProcess.log")
# def test_output_process_output_signal(mock_log, mock_get, mock_put, mock_vput, output_process_obj):
#     mock_hardware = mock.MagicMock()
#     output_process_obj.hardware = mock_hardware

#     output_process_obj.environment_data_out_remainders['Modal'] = [0]


#     output_process_obj.output_signal(None)


@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_process_stop_output(mock_log, output_process_obj):
    output_process_obj.stop_output(None)

    mock_log.assert_called_with("Starting Shutdown Procedure")
    assert output_process_obj.shutdown_flag == True


@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_process_start_environment(mock_log, output_process_obj):
    output_process_obj.start_environment("Modal")

    mock_log.assert_called_with("Started Environment Modal")
    assert output_process_obj.environment_starting_up_flags["Modal"] == True
    assert output_process_obj.environment_shutting_down_flags["Modal"] == False
    assert output_process_obj.environment_active_flags["Modal"] == False


@mock.patch("rattlesnake.process.acquisition.flush_queue")
@mock.patch("rattlesnake.process.output.OutputProcess.log")
def test_output_process_quit(mock_log, mock_flush, output_process_obj):
    mock_hardware = mock.MagicMock()
    output_process_obj.hardware = mock_hardware

    output_process_obj.quit(None)

    mock_log.assert_called_with("Flushed 0 items out of queues")
    mock_hardware.close.assert_called()


# Test the output_process function
# Prevent run while loop from starting
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.run")
def test_output_process_func(mock_run, queue_container, environments):
    output_process(queue_container, environments, mp.Value("i", 0))

    # Test if the run function was called
    mock_run.assert_called()


if __name__ == "__main__":
    log_file_queue = mp.Queue()

    queue_container = QueueContainer(
        VerboseMessageQueue(log_file_queue, "Controller Communication Queue"),
        VerboseMessageQueue(log_file_queue, "Acquisition Command Queue"),
        VerboseMessageQueue(log_file_queue, "Output Command Queue"),
        VerboseMessageQueue(log_file_queue, "Streaming Command Queue"),
        log_file_queue,
        mp.Queue(),
        mp.Queue(),
        mp.Queue(),
        {"Modal": VerboseMessageQueue(log_file_queue, "Environment Command Queue")},
        {"Modal": mp.Queue()},
        {"Modal": mp.Queue()},
    )

    control_type = ControlTypes(6)
    environments = [[control_type, control_type.name.title()]]

    output_process_obj = OutputProcess(
        "Process Name", queue_container, environments, mp.Value("i", 0)
    )

    channel_list = [
        Channel.from_channel_table_row(
            [
                "221",
                "Y+",
                "",
                "19644",
                "X+",
                "",
                "",
                "",
                "",
                "",
                "Virtual",
                "",
                "Accel",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        ),
        Channel.from_channel_table_row(
            [
                "221",
                "Y+",
                "",
                "19644",
                "X+",
                "",
                "",
                "",
                "",
                "",
                "Virtual",
                "",
                "Force",
                "",
                "",
                "",
                "",
                "",
                "Phys_dev",
                "",
                "5",
                "10",
            ]
        ),
    ]

    sample_rate = 2000
    time_per_read = 0.25
    time_per_write = 0.25
    output_oversample = 2
    hardware_selector_idx = 6
    hardware_file = "ExampleFile.nc4"
    environment = ["Modal"]
    environment_booleans = [[True]]
    acquisition_processes = 1
    data_acquisition_parameters = DataAcquisitionParameters(
        channel_list,
        sample_rate,
        round(sample_rate * time_per_read),
        round(sample_rate * time_per_write * output_oversample),
        hardware_selector_idx,
        hardware_file,
        environment,
        environment_booleans,
        output_oversample,
        acquisition_processes,
    )

    # test_output_initialize_data_acquisition(hardware = mock.MagicMock(), hardware_idx = 2, output_process_obj=output_process_obj, data_acquisition_parameters=data_acquisition_parameters, hardware_dict=create_hardware_dict_output())
    test_output_process_output_signal(output_process_obj=output_process_obj)
