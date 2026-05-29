from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.utilities import GlobalCommands
from mock_objects.mock_utilities import mock_queue_container, mock_event_container, fake_time
import pytest
import multiprocessing as mp
from unittest import mock
from queue import Empty as QueueEmpty


# region: Fixtures
@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def abstract_message_process(request):
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    abstract_message_process = AbstractMessageProcess(
        "Process Name",
        queue_container.log_file_queue,
        queue_container.controller_command_queue,
        queue_container.gui_update_queue,
        event_container.controller_ready_event,
    )
    return abstract_message_process


# region: AbstractMessageProcess
@pytest.mark.parametrize("use_thread", [True, False])
def test_message_process_init(use_thread):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    abstract_message_process = AbstractMessageProcess(
        "Process Name",
        queue_container.log_file_queue,
        queue_container.controller_command_queue,
        queue_container.gui_update_queue,
        event_container.controller_ready_event,
    )

    assert isinstance(abstract_message_process, AbstractMessageProcess)


def test_message_process_properties(abstract_message_process):
    abstract_message_process.gui_update_queue
    abstract_message_process.command_queue
    abstract_message_process.log_file_queue

    assert abstract_message_process.process_name == "Process Name"
    assert abstract_message_process.command_map == {GlobalCommands.QUIT: abstract_message_process.quit}


@mock.patch("rattlesnake.process.abstract_message_process.datetime")
def test_message_process_log(mock_time, abstract_message_process):
    mock_log_file_queue = mock.MagicMock()
    abstract_message_process._log_file_queue = mock_log_file_queue
    mock_time.now = fake_time
    abstract_message_process.log("Test Message")

    mock_log_file_queue.put.assert_called_once_with("Datetime: Process Name -- Test Message\n")


def test_message_process_set_ready(abstract_message_process):
    abstract_message_process._ready_event.clear()
    abstract_message_process.set_ready()

    assert abstract_message_process._ready_event.is_set()


def test_message_process_clear_ready(abstract_message_process):
    abstract_message_process._ready_event.set()
    abstract_message_process.clear_ready()

    assert not abstract_message_process._ready_event.is_set()


# Test the map_command function
def test_abstract_message_process_map_command(abstract_message_process):
    # Create custom key and function
    key = "Test Key"

    def function():
        return "Test Function"

    abstract_message_process.map_command(key, function)

    # Test if the key returns the correct function
    data = abstract_message_process.command_map[key]
    assert data == function


# Test the run function
# Loop through keys and functions that return KeyErrors
@pytest.mark.parametrize(
    "mock_function, mock_key",
    [
        (mock.MagicMock(return_value=False), "Test Key"),
        (mock.MagicMock(side_effect=KeyError), "Test Key"),
        (mock.MagicMock(return_value=False), "Not a key"),
    ],
)
# Force command_queue.get function to return data
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.get")
# Prevent from storing to log_file_queue
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
def test_abstract_message_process_run(mock_log, mock_get, mock_function, mock_key, abstract_message_process):
    # Create a command_map
    abstract_message_process._command_map = {
        mock_key: mock_function,
        "Quit Key": abstract_message_process.quit,
    }

    # Give command_queue.get a "Test Key" then "Quit Key"
    mock_get.side_effect = [(QueueEmpty), ("Test Key", None), ("Quit Key", None)]
    shutdown_event = mp.Event()
    abstract_message_process.run(shutdown_event)

    # If a correct key was given, check that function was ran
    if mock_key == "Test Key":
        mock_function.assert_called()
    # Test that the run function stopped
    mock_log.assert_called_with("Stopping Process")
