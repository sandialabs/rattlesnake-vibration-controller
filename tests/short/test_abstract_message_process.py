"""
Tests for Abstract Message Process

This module contains tests for the AbstractMessageProcess class, verifying
initialization, logging, command mapping, and the main run loop.
"""

import multiprocessing as mp
from unittest import mock
import pytest

from functions.common_functions import fake_time
from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.utilities import GlobalCommands, VerboseMessageQueue


# Create log_file_queue
@pytest.fixture()
def log_file_queue():
    """
    Fixture for a log file queue.
    """
    return mp.Queue()


# Create command queue
@pytest.fixture()
def abstract_command_queue(log_file_queue):
    """
    Fixture for an abstract command queue.
    """
    return VerboseMessageQueue(log_file_queue, "Abstract Command Queue")


# Create Gui update queue
@pytest.fixture()
def gui_update_queue():
    """
    Fixture for a GUI update queue.
    """
    return mp.Queue()


# Initialize an AbstractMessageProcess
@pytest.fixture()
def abstract_message_process(log_file_queue, abstract_command_queue, gui_update_queue):
    """
    Fixture for an AbstractMessageProcess instance.
    """
    return AbstractMessageProcess(
        "Process Name", log_file_queue, abstract_command_queue, gui_update_queue
    )


# Test
def test_abstract_message_process_init(log_file_queue, abstract_command_queue, gui_update_queue):
    """
    Test the initialization of the AbstractMessageProcess class.
    """
    abstract_message_process = AbstractMessageProcess(
        "Process Name", log_file_queue, abstract_command_queue, gui_update_queue
    )

    # Test if initialized class is indeed that class
    assert isinstance(abstract_message_process, AbstractMessageProcess)
    # Test the process name property
    assert abstract_message_process.process_name == "Process Name"
    # Test the command_map property
    assert abstract_message_process.command_map == {
        GlobalCommands.QUIT: abstract_message_process.quit
    }
    # Test the gui_update_queue property
    assert abstract_message_process.gui_update_queue == gui_update_queue
    # Test the command_queue property
    assert abstract_message_process.command_queue == abstract_command_queue
    # Test the log_file_queue property
    assert abstract_message_process.log_file_queue == log_file_queue


# Test the AbstractMessageProcess log function
# Prevent anything from being written to log_file_queue
@mock.patch("rattlesnake.process.abstract_message_process.Queue.put")
# Replace date and time with a string
@mock.patch("rattlesnake.process.abstract_message_process.datetime")
def test_abstract_message_process_log(mock_time, mock_put, abstract_message_process):
    """
    Test the logging functionality of the AbstractMessageProcess class.
    """
    message = "Test Message"
    mock_time.now = fake_time

    abstract_message_process.log(message)

    # Test if the correct string was stored to log_file_queue
    mock_put.assert_called_with("{:}: {:} -- {:}\n".format("Datetime", "Process Name", message))


# Test if the quit function works
def test_abstract_message_process_quit(abstract_message_process):
    """
    Test the quit functionality of the AbstractMessageProcess class.
    """
    data = abstract_message_process.quit(None)

    # Test that the quit function returns True
    assert data is True


# Test the map_command function
def test_abstract_message_process_map_command(abstract_message_process):
    """
    Test the mapping of custom commands to the AbstractMessageProcess.
    """
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
def test_abstract_message_process_run(
    mock_log, mock_get, mock_function, mock_key, abstract_message_process
):
    """
    Test the main run loop of the AbstractMessageProcess class.
    """
    # Create a command_map
    abstract_message_process._command_map = {
        mock_key: mock_function,
        "Quit Key": abstract_message_process.quit,
    }

    # Give command_queue.get a "Test Key" then "Quit Key"
    mock_get.side_effect = [("Test Key", None), ("Quit Key", None)]

    abstract_message_process.run()

    # If a correct key was given, check that function was ran
    if mock_key == "Test Key":
        mock_function.assert_called()
    # Test that the run function stopped
    mock_log.assert_called_with("Stopping Process")


if __name__ == "__main__":
    log_file_queue = mp.Queue()
    abstract_command_queue = VerboseMessageQueue(log_file_queue, "Spectral Command Queue")
    abstract_message_process = AbstractMessageProcess(
        "Process Name", log_file_queue, abstract_command_queue, mp.Queue()
    )

    test_abstract_message_process_run(
        mock_function=mock.MagicMock(return_value=False),
        abstract_message_process=abstract_message_process,
    )
    # test_abstract_message_process_map_command(abstract_message_process=abstract_message_process)
