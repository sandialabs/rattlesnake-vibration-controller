import multiprocessing as mp
import queue as thqueue
import time
from unittest import mock

import pytest

from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.testing.mock_utilities import (
    fake_time,
    mock_event_container,
    mock_queue_container,
)
from rattlesnake.user_interface.ui_utilities import UICommands
from rattlesnake.utilities import GlobalCommands


# region Fixtures
@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def abstract_message_process(request):
    """
    Create an ``AbstractMessageProcess`` in threaded and multiprocessing modes.
    """
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)

    return AbstractMessageProcess(
        "Process Name",
        queue_container.log_file_queue,
        queue_container.controller_command_queue,
        queue_container.gui_update_queue,
        event_container.controller_ready_event,
    )


# endregion


# region AbstractMessageProcess
@pytest.mark.parametrize("use_thread", [True, False])
def test_message_process_init(use_thread):
    """
    Verifies that an ``AbstractMessageProcess`` can be initialized in threaded
    and non-threaded configurations.
    """
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)

    message_process = AbstractMessageProcess(
        "Process Name",
        queue_container.log_file_queue,
        queue_container.controller_command_queue,
        queue_container.gui_update_queue,
        event_container.controller_ready_event,
    )

    assert isinstance(message_process, AbstractMessageProcess)
    assert message_process.process_name == "Process Name"
    assert message_process.log_file_queue is queue_container.log_file_queue
    assert message_process.command_queue is queue_container.controller_command_queue
    assert message_process.gui_update_queue is queue_container.gui_update_queue
    assert message_process.ready_event is event_container.controller_ready_event
    assert message_process.ready_event.is_set()


def test_message_process_properties(abstract_message_process):
    """
    Verifies that process properties return expected values and that the
    default command map contains ``GlobalCommands.QUIT``.
    """
    abstract_message_process.gui_update_queue
    abstract_message_process.command_queue
    abstract_message_process.log_file_queue
    abstract_message_process.ready_event

    assert abstract_message_process.process_name == "Process Name"
    assert abstract_message_process.command_map == {
        GlobalCommands.QUIT: abstract_message_process.quit
    }


@mock.patch("rattlesnake.process.abstract_message_process.datetime")
def test_message_process_log(mock_datetime, abstract_message_process):
    """
    Verifies that calling ``log`` places the expected formatted log message on
    the log file queue.
    """
    mock_log_file_queue = mock.MagicMock()
    abstract_message_process._log_file_queue = mock_log_file_queue
    mock_datetime.now = fake_time

    abstract_message_process.log("Test Message")

    mock_log_file_queue.put.assert_called_once_with(
        "Datetime: Process Name -- Test Message\n"
    )


def test_message_process_set_ready(abstract_message_process):
    """
    Verifies that calling ``set_ready`` sets the ready event.
    """
    abstract_message_process._ready_event.clear()
    abstract_message_process.set_ready()

    assert abstract_message_process._ready_event.is_set()


def test_message_process_clear_ready(abstract_message_process):
    """
    Verifies that calling ``clear_ready`` clears the ready event.
    """
    abstract_message_process._ready_event.set()
    abstract_message_process.clear_ready()

    assert not abstract_message_process._ready_event.is_set()


def test_abstract_message_process_map_command(abstract_message_process):
    """
    Confirms that a custom command key can be added to the command map and maps
    to the provided callable.
    """
    key = "Test Key"

    def function(data):
        return data

    abstract_message_process.map_command(key, function)

    assert abstract_message_process.command_map[key] == function


def test_message_process_quit(abstract_message_process):
    """
    Verifies that ``quit`` returns ``True`` so the command loop stops.
    """
    assert abstract_message_process.quit(None) is True


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.get")
def test_abstract_message_process_run_dispatches_mapped_command(
    mock_get,
    mock_log,
    abstract_message_process,
):
    """
    Verifies that the command loop dispatches a mapped command and passes the
    command data to the mapped function.
    """
    handler = mock.MagicMock(return_value=False)
    payload = object()
    abstract_message_process.map_command("Test Key", handler)
    mock_get.side_effect = [
        ("Test Key", payload),
        (GlobalCommands.QUIT, None),
    ]
    shutdown_event = mp.Event()

    abstract_message_process.run(shutdown_event)

    handler.assert_called_once_with(payload)
    mock_log.assert_any_call("Stopping Process")


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.get")
def test_abstract_message_process_run_ignores_empty_queue(
    mock_get,
    mock_log,
    abstract_message_process,
):
    """
    Verifies that empty queue exceptions are ignored and the command loop
    continues until a quit command is received.
    """
    mock_get.side_effect = [
        thqueue.Empty(),
        (GlobalCommands.QUIT, None),
    ]
    shutdown_event = mp.Event()

    abstract_message_process.run(shutdown_event)

    mock_log.assert_any_call("Stopping Process")


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.get")
def test_abstract_message_process_run_undefined_command(
    mock_get,
    mock_log,
    abstract_message_process,
):
    """
    Verifies that undefined commands are logged and do not stop the process.
    """
    mock_get.side_effect = [
        ("Undefined Command", None),
        (GlobalCommands.QUIT, None),
    ]
    shutdown_event = mp.Event()

    abstract_message_process.run(shutdown_event)

    assert any(
        "Undefined Message Undefined Command" in call.args[0]
        for call in mock_log.call_args_list
    )
    mock_log.assert_any_call("Stopping Process")


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.get")
def test_abstract_message_process_run_command_exception(
    mock_get,
    mock_log,
    abstract_message_process,
):
    """
    Verifies that exceptions raised by mapped command functions are logged and
    sent to the GUI update queue as ``UICommands.ERROR`` messages.
    """
    mock_gui_update_queue = mock.MagicMock()
    abstract_message_process._gui_update_queue = mock_gui_update_queue

    def boom(data):
        raise RuntimeError("BOOM")

    abstract_message_process.map_command("Boom", boom)
    mock_get.side_effect = [
        ("Boom", None),
        (GlobalCommands.QUIT, None),
    ]
    shutdown_event = mp.Event()

    abstract_message_process.run(shutdown_event)
    time.sleep(1)

    assert any("ERROR" in call.args[0] for call in mock_log.call_args_list)
    mock_gui_update_queue.put.assert_called_once()
    gui_message, gui_data = mock_gui_update_queue.put.call_args.args[0]
    assert gui_message == UICommands.ERROR
    assert gui_data[0] == "Process Name Error"
    assert "RuntimeError: BOOM" in gui_data[1]
    mock_log.assert_any_call("Stopping Process")


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.get")
def test_abstract_message_process_run_exits_correctly(
    mock_get,
    mock_log,
    abstract_message_process,
):
    """
    Verifies that the command loop exits when a mapped function returns a
    truthy halt flag.
    """
    halt_handler = mock.MagicMock(return_value=True)
    abstract_message_process.map_command("Halt", halt_handler)
    mock_get.side_effect = [
        ("Halt", None),
    ]
    shutdown_event = mp.Event()

    abstract_message_process.run(shutdown_event)

    halt_handler.assert_called_once_with(None)
    mock_log.assert_any_call("Stopping Process")


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.get")
def test_abstract_message_process_run_exits_on_shutdown_event(
    mock_get,
    mock_log,
    abstract_message_process,
):
    """
    Verifies that the command loop exits without reading from the command queue
    when the supplied shutdown event is already set.
    """
    shutdown_event = mp.Event()
    shutdown_event.set()

    abstract_message_process.run(shutdown_event)

    mock_get.assert_not_called()
    assert any("Starting Process" in call.args[0] for call in mock_log.call_args_list)
    assert not any(
        "Stopping Process" in call.args[0] for call in mock_log.call_args_list
    )


@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.get")
def test_abstract_message_process_run_without_shutdown_event(
    mock_get,
    mock_log,
    abstract_message_process,
):
    """
    Verifies that the command loop can run without a shutdown event and exits
    when a quit command is received.
    """
    mock_get.side_effect = [
        (GlobalCommands.QUIT, None),
    ]

    abstract_message_process.run()

    mock_log.assert_any_call("Stopping Process")


@pytest.mark.parametrize(
    "command_return,expected_stop",
    [
        (False, False),
        (None, False),
        (0, False),
        (True, True),
        (1, True),
    ],
)
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.log")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.get")
def test_abstract_message_process_run_halt_flag_truthiness(
    mock_get,
    mock_log,
    command_return,
    expected_stop,
    abstract_message_process,
):
    """
    Verifies that only truthy mapped-command return values stop the command
    loop.
    """
    handler = mock.MagicMock(return_value=command_return)
    abstract_message_process.map_command("Command", handler)
    if expected_stop:
        mock_get.side_effect = [
            ("Command", None),
        ]
    else:
        mock_get.side_effect = [
            ("Command", None),
            (GlobalCommands.QUIT, None),
        ]
    shutdown_event = mp.Event()

    abstract_message_process.run(shutdown_event)

    handler.assert_called_once_with(None)
    mock_log.assert_any_call("Stopping Process")


@pytest.mark.parametrize("use_thread", [True, False])
def test_message_process_ready_event_is_set_on_init(use_thread):
    """
    Verifies that the ready event supplied during initialization is set by the
    constructor.
    """
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    event_container.controller_ready_event.clear()

    message_process = AbstractMessageProcess(
        "Process Name",
        queue_container.log_file_queue,
        queue_container.controller_command_queue,
        queue_container.gui_update_queue,
        event_container.controller_ready_event,
    )

    assert message_process.ready_event is event_container.controller_ready_event
    assert message_process.ready_event.is_set()


# endregion
