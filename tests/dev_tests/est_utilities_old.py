"""
Rattlesnake Queues Testing

This code unit tests the log_file_queue and the Verbose Queue object. This
code makes sure that the correct objects are put and get from these queues.

The following code is tested:
utilities.py
- log_file_queue messages
- VerboseMessageQueue.__init__
- VerboseMessageQueue.generate_message_id
- VerboseMessageQueue.put
- VerboseMessageQueue.get
- VerboseMessageQueue.flush
"""

import multiprocessing as mp
import random
import string
from unittest import mock

import pytest
from functions.common_functions import fake_time
from functions.queues_functions import clear_log_queue, clear_verbose_queue

# from rattlesnake.components.environments import ControlTypes  # unused import
from rattlesnake.utilities import (
    Channel,
    DataAcquisitionParameters,
    GlobalCommands,
    QueueContainer,
    VerboseMessageQueue,
    load_python_module,
    log_file_task,
)

# from qtpy import QtWidgets  # unused import


# Create a log file queue
@pytest.fixture
def log_file_queue():
    return mp.Queue()


# Initialize the verbose message queue
@pytest.fixture
def verbose_queue(log_file_queue):
    return VerboseMessageQueue(log_file_queue, "VerboseQueue")


# Generate an example channel_table_string
@pytest.fixture
def channel_table_row():
    channel_table_row = [
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
    return channel_table_row


# Initialize a channel with the template row
@pytest.fixture
def channel(channel_table_row):
    return Channel.from_channel_table_row(channel_table_row)


# Test Channel initialization
def test_channel_init():
    channel = Channel(
        "node_number",
        "node_direction",
        "comment",
        "serial_number",
        "triax_dof",
        "sensitivity",
        "unit",
        "make",
        "model",
        "expiration",
        "physical_device",
        "physical_channel",
        "channel_type",
        "minimum_value",
        "maximum_value",
        "coupling",
        "excitation_source",
        "excitation",
        "feedback_device",
        "feedback_channel",
        "warning_level",
        "abort_level",
    )

    # Test if object is a Channel
    assert isinstance(channel, Channel)


# Test initializing a Channel object with function


def test_from_channel_table_row(channel_table_row):
    # Create channel from first row of channel_table template
    channel = Channel.from_channel_table_row(channel_table_row)

    # Test that channel variable is a Channel object
    assert isinstance(channel, Channel)


# Test initializing a DataAcquisitionParameters object
def test_data_acquisition_parameters_initialization(channel):
    # Initialize DataAcquisitionParameters object
    channel_list = [channel]
    sample_rate = 2048
    time_per_read = 5
    time_per_write = 5
    output_oversample = 10
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

    # Test if variable is a DataAcquisitionParameters object
    assert isinstance(data_acquisition_parameters, DataAcquisitionParameters)
    # Test nyquist_frequency property
    assert data_acquisition_parameters.nyquist_frequency == sample_rate / 2
    # Test output_sample_rate property
    assert data_acquisition_parameters.output_sample_rate == sample_rate * output_oversample


# Test log_file_task
# Prevent a file from opening
@mock.patch("builtins.open", new_callable=mock.mock_open)
def test_log_file_process(mock_file, log_file_queue):
    # Put messages into the queue
    message = "This is a test"
    log_file_queue.put(message)
    log_file_queue.put("quit")

    log_file_task(log_file_queue)

    # Test if the message was stored
    mock_file().write.assert_any_call(message)
    # Test if the log quit out correctly
    mock_file().write.assert_called_with("Program quitting, logging terminated.")
    # Test if the file was flushed
    mock_file().flush.assert_called()


# Test initialziation of queue container
def test_queue_container_init(log_file_queue):
    acquisition_command_queue = VerboseMessageQueue(log_file_queue, "Acquisition Command Queue")
    output_command_queue = VerboseMessageQueue(log_file_queue, "Output Command Queue")
    streaming_command_queue = VerboseMessageQueue(log_file_queue, "Streaming Command Queue")
    input_output_sync_queue = mp.Queue()
    single_process_hardware_queue = mp.Queue()
    gui_update_queue = mp.Queue()
    controller_communication_queue = VerboseMessageQueue(
        log_file_queue, "Controller Communication Queue"
    )
    environment_name = "Environment Name"
    environment_command_queues = {}
    environment_data_in_queues = {}
    environment_data_out_queues = {}
    environment_command_queues[environment_name] = VerboseMessageQueue(
        log_file_queue, environment_name + " Command Queue"
    )
    environment_data_in_queues[environment_name] = mp.Queue()
    environment_data_out_queues[environment_name] = mp.Queue()

    queue_container = QueueContainer(
        controller_communication_queue,
        acquisition_command_queue,
        output_command_queue,
        streaming_command_queue,
        log_file_queue,
        input_output_sync_queue,
        single_process_hardware_queue,
        gui_update_queue,
        environment_command_queues,
        environment_data_in_queues,
        environment_data_out_queues,
    )

    assert isinstance(queue_container, QueueContainer)


# Test the initialization of the verbose mesage queue
def test_verbose_queue_init(log_file_queue):
    verbose_queue = VerboseMessageQueue(log_file_queue, "VerboseQueue")

    assert isinstance(verbose_queue, VerboseMessageQueue)


# Test message_id generation
def test_verbose_message_id(verbose_queue, random_seed=42):
    # Seed the message id
    random.seed(random_seed)
    message_id = verbose_queue.generate_message_id()
    random.seed(random_seed)
    assert_id = "".join(random.choice(string.ascii_letters) for _ in range(6))

    # Test if the message id matches the template
    assert message_id == assert_id


# Test verbose message queue put
# Mock the message id return string
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.generate_message_id")
# Prevent the Queue object form putting stuff into it
@mock.patch("rattlesnake.utilities.mp.queues.Queue.put")
def test_verbose_message_queue_put(mock_put, mock_id, verbose_queue):
    # Mock message id
    message_id = "1"
    mock_id.return_value = message_id
    # Create objects to put into queue
    task_name = "Test verbose queue"
    message_data_tuple = (GlobalCommands.QUIT, "Information")

    verbose_queue.put(task_name, message_data_tuple)

    # Test if objects were put into queue
    mock_put.assert_called_with((message_id, message_data_tuple))


# Test verbose message queue get
# Prevent the Queue object from getting from an empty queue
@mock.patch("rattlesnake.utilities.mp.queues.Queue.get")
def test_verbose_message_queue_get(mock_get, verbose_queue):
    # Mock the data to get from the queue
    message_id = "1"
    task_name = "Test verbose queue"
    message_data_tuple = (GlobalCommands.QUIT, "Information")
    mock_get.return_value = (message_id, message_data_tuple)

    data = verbose_queue.get(task_name)

    # Test if data from get matches mock return value
    assert data == message_data_tuple


# # Test verbose message queue flush and log_file_queue
# # Mock the message_id
# @mock.patch('rattlesnake.components.utilities.VerboseMessageQueue.generate_message_id')
# # Mock the datetime for the logging queue
# @mock.patch('rattlesnake.components.utilities.datetime')
# def test_verbose_message_queue_flush(mock_time, mock_id, log_file_queue, verbose_queue):
#     # Create multiprocessing string to store logging messages
#     log_string = mp.Array('c', 300)
#     log_string.value = b''
#     # Create data to put into queue
#     task_name = "Test verbose flush"
#     message_data_tuple = (GlobalCommands.QUIT, 'This should have data')

#     # Mock datetime and id for logging
#     mock_time.now = fake_time
#     mock_id.return_value = '1'

#     # Put object in queue and flush
#     verbose_queue.put(task_name, message_data_tuple)
#     data = verbose_queue.flush(task_name)

#     # Clear log_file_queue and store messages to log_string
#     log_file_process = mp.Process(
#         target=clear_log_queue, args=(log_file_queue, log_string))
#     log_file_process.start()
#     log_file_process.join()

#     # This is inconsistent at best, I need to look into this further
#     # Test if the log message is correct
#     # assert log_string.value == b'Datetime: Test verbose flush put QUIT (1) to VerboseQueue\nDatetime: Test verbose flush flushed VerboseQueue\nDatetime: Test verbose flush got QUIT (1) from VerboseQueue during flush\n'

#     # Test if flushed data matches input data
#     assert data[0] == message_data_tuple


# Test log_file_queue for verbose push and verbose get)
# Mock the message_id for the log message
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.generate_message_id")
# Mock the datetime in the log message
@mock.patch("rattlesnake.utilities.datetime")
def test_verbose_message_queue_log(mock_time, mock_id, log_file_queue, verbose_queue):
    # Generate multiprocessing arrays to store data and log message to
    verbose_array = mp.Array("i", 1)
    verbose_value = 10
    log_string = mp.Array("c", 200)
    log_string.value = b""
    # Create data to put into queue
    task_name = "Test verbose queue"
    message_data_tuple = (GlobalCommands.QUIT, verbose_value)

    # Mock datetime and id for log message
    mock_time.now = fake_time
    mock_id.return_value = "1"

    # Put data into verbose queue and clear the queue, store data into verbose_array
    verbose_queue.put(task_name, message_data_tuple)
    verbose_process = mp.Process(
        target=clear_verbose_queue, args=(verbose_queue, "Get Queue", verbose_array)
    )
    verbose_process.start()
    verbose_process.join()

    # Clear the log_file_queue and store the messages to log_string
    log_file_process = mp.Process(target=clear_log_queue, args=(log_file_queue, log_string))
    log_file_process.start()
    log_file_process.join()

    # Test if log message matches template
    assert (
        log_string.value
        == b"Datetime: Test verbose queue put QUIT (1) to VerboseQueue\nDatetime: Get Queue got QUIT (1) from VerboseQueue\n"
    )


@mock.patch("rattlesnake.utilities.importlib.util.module_from_spec")
@mock.patch("rattlesnake.utilities.importlib.util.spec_from_file_location")
def test_load_python_module(mock_from_spec, mock_module):
    mock_spec = mock.MagicMock()
    mock_from_spec.return_value = mock_spec
    mock_module.return_value = "module"

    module = load_python_module("users/document/filename.py")

    mock_from_spec.assert_called_with("filename", "users/document/filename.py")
    mock_module.assert_called_with(mock_spec)
    mock_spec.loader.exec_module.assert_called()
    assert module == "module"


if __name__ == "__main__":
    log_file_queue = mp.Queue()
    verbose_queue = VerboseMessageQueue(log_file_queue, "VerboseQueue")
    # test_verbose_message_queue_flush(
    #     log_file_queue=log_file_queue, verbose_queue=verbose_queue)

    # test_log_file_process(log_file_queue=log_file_queue)
    # test_verbose_message_id(verbose_queue=verbose_queue)
    # test_verbose_message_queue_get(verbose_queue=verbose_queue)
    # test_verbose_message_queue_log(log_file_queue=log_file_queue, verbose_queue = verbose_queue)
    test_load_python_module()
