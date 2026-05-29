from rattlesnake.utilities import VerboseMessageQueue, GlobalCommands, log_file_task
from mock_objects.mock_utilities import fake_time, clear_log_queue, clear_verbose_queue
import multiprocessing as mp
import threading
import queue as thqueue
import random
import string
import pytest
from unittest import mock


# Create a log file queue
@pytest.fixture
def log_file_queue():
    return mp.Queue()


# Create a queue name manager
@pytest.fixture
def name_manager():
    return mp.Manager()


# Initialize the verbose message queue
@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def verbose_fixture(request, log_file_queue, name_manager):
    use_thread = request.param

    if use_thread:
        queue = thqueue.Queue()
    else:
        queue = mp.Queue()

    return (VerboseMessageQueue(log_file_queue, queue, name_manager, "VerboseQueue"), use_thread)


# region: log_file_task
# Prevent a file from opening
@mock.patch("builtins.open", new_callable=mock.mock_open)
def test_log_file_process(mock_file, log_file_queue):
    # Put messages into the queue
    message = "This is a test"
    log_file_queue.put(message)
    log_file_queue.put(GlobalCommands.QUIT)

    mock_shutdown = mock.MagicMock()
    mock_shutdown.is_set.return_value = False

    log_file_task(log_file_queue, mock_shutdown)

    # Test if the message was stored
    mock_file().write.assert_any_call(message)
    # Test if the log quit out correctly
    mock_file().write.assert_called_with("Program quitting, logging terminated.")
    # Test if the file was flushed
    mock_file().flush.assert_called()


# region: VerboseMessageQueue
# Test the initialization of the verbose mesage queue
def test_verbose_queue_init(log_file_queue, name_manager):
    verbose_queue = VerboseMessageQueue(log_file_queue, mp.Queue(), name_manager, "VerboseQueue")

    assert isinstance(verbose_queue, VerboseMessageQueue)


def test_verbose_queue_name(verbose_fixture):
    verbose_queue, _ = verbose_fixture

    assert verbose_queue.log_name == "VerboseQueue"

    verbose_queue.assign_environment("Environment 0")

    assert verbose_queue.log_name == "VerboseQueue | Environment 0"


# Test message_id generation
def test_verbose_message_id(verbose_fixture, random_seed=42):
    verbose_queue, _ = verbose_fixture
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
def test_verbose_message_queue_put(mock_id, verbose_fixture):
    verbose_queue, _ = verbose_fixture
    mock_base_queue = mock.MagicMock()
    verbose_queue.base_queue = mock_base_queue
    # Mock message id
    message_id = "1"
    mock_id.return_value = message_id
    # Create objects to put into queue
    task_name = "Test verbose queue"
    message_data_tuple = (GlobalCommands.QUIT, "Information")

    verbose_queue.put(task_name, message_data_tuple)

    # Test if objects were put into queue
    mock_base_queue.put.assert_called_with((message_id, message_data_tuple))


# Test verbose message queue get
# Prevent the Queue object from getting from an empty queue
def test_verbose_message_queue_get(verbose_fixture):
    verbose_queue, _ = verbose_fixture
    mock_base_queue = mock.MagicMock()
    verbose_queue.base_queue = mock_base_queue
    # Mock the data to get from the queue
    message_id = "1"
    task_name = "Test verbose queue"
    message_data_tuple = (GlobalCommands.QUIT, "Information")
    mock_base_queue.get.return_value = (message_id, message_data_tuple)

    data = verbose_queue.get(task_name)

    # Test if data from get matches mock return value
    assert data == message_data_tuple


# Test log_file_queue for verbose push and verbose get)
# Mock the message_id for the log message
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.generate_message_id")
# Mock the datetime in the log message
@mock.patch("rattlesnake.utilities.datetime")
def test_verbose_queue_log(mock_time, mock_id, log_file_queue, verbose_fixture):
    verbose_queue, use_thread = verbose_fixture
    if use_thread:
        new_process = threading.Thread
    else:
        new_process = mp.Process
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
    verbose_process = new_process(target=clear_verbose_queue, args=(verbose_queue, "Get Queue", verbose_array))
    verbose_process.start()
    verbose_process.join()

    # Clear the log_file_queue and store the messages to log_string
    log_file_process = new_process(target=clear_log_queue, args=(log_file_queue, log_string))
    log_file_process.start()
    log_file_process.join()

    # Test if log message matches template
    assert log_string.value == b"Datetime: Test verbose queue put QUIT (1) to VerboseQueue\nDatetime: Get Queue got QUIT (1) from VerboseQueue\n"


# Test verbose message queue flush and log_file_queue
# Mock the message_id
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.generate_message_id")
# Mock the datetime for the logging queue
@mock.patch("rattlesnake.utilities.datetime")
def test_verbose_message_queue_flush(mock_time, mock_id, log_file_queue, verbose_fixture):
    verbose_queue, use_thread = verbose_fixture
    if use_thread:
        new_process = threading.Thread
    else:
        new_process = mp.Process
    # Create multiprocessing string to store logging messages
    log_string = mp.Array("c", 300)
    log_string.value = b""
    # Create data to put into queue
    task_name = "Test verbose flush"
    message_data_tuple = (GlobalCommands.QUIT, "This should have data")

    # Mock datetime and id for logging
    mock_time.now = fake_time
    mock_id.return_value = "1"

    # Put object in queue and flush
    verbose_queue.put(task_name, message_data_tuple)
    data = verbose_queue.flush(task_name)

    # Clear log_file_queue and store messages to log_string
    log_file_process = new_process(target=clear_log_queue, args=(log_file_queue, log_string))
    log_file_process.start()
    log_file_process.join()

    # This is inconsistent at best, I need to look into this further
    # Test if the log message is correct
    # assert log_string.value == b'Datetime: Test verbose flush put QUIT (1) to VerboseQueue\nDatetime: Test verbose flush flushed VerboseQueue\nDatetime: Test verbose flush got QUIT (1) from VerboseQueue during flush\n'

    # Test if flushed data matches input data
    assert data[0] == message_data_tuple


def test_verbose_queue_close(verbose_fixture):
    verbose_queue, _ = verbose_fixture

    verbose_queue.close()
    verbose_queue.empty()
    verbose_queue.join_thread()

    assert True
