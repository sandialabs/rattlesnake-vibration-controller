"""
This file contains a number of helper classes for the general controller.

Rattlesnake Vibration Control Software
Copyright (C) 2021  National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import importlib.util
import multiprocessing.managers  # pylint: disable=unused-import
import multiprocessing.synchronize  # pylint: disable=unused-import
import os
import random
import re
import socket
import string
import subprocess
import sys
import threading
import time
import multiprocessing as mp
import multiprocessing.queues as mpqueue
import queue as thqueue
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, TextIO, Tuple

import netCDF4
import requests
import numpy as np
import scipy.signal as sig
from openpyxl.worksheet.worksheet import Worksheet
from scipy.interpolate import interp1d
from scipy.io import loadmat

if TYPE_CHECKING:
    from rattlesnake.hardware.abstract_hardware import HardwareMetadata
    from rattlesnake.environment.abstract_environment import EnvironmentMetadata

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

# region Global
# Define base directory
this_path = os.path.split(__file__)[0]
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    DIRECTORY = sys._MEIPASS  # pylint: disable=protected-access
else:
    DIRECTORY = this_path


def open_and_lock_file(lock_filename: str, encoding: str = "utf-8") -> TextIO | None:
    """
    Returns a new file if the filename is not a locked file. Returns None
    if the file is in use by another program

    The file is opened in append mode and an exclusive, non-blocking lock is
    requested on it. If the lock is acquired, the file is truncated and the
    current process id is written to it.

    Parameters
    ----------
    lock_filename : str
        Path to the lock file to open and lock. It is created if it does not
        exist.
    encoding : str, optional
        Text encoding used to open the file. Defaults to ``"utf-8"``.

    Returns
    -------
    TextIO or None
        The open, locked file handle, or ``None`` if another process already
        holds the lock.

    Unit Tests
    ----------
    """
    lock_file = open(lock_filename, "a+", encoding=encoding)
    lock_file.seek(0)

    try:
        if sys.platform == "win32":
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_file.close()
        return None

    lock_file.truncate()
    lock_file.write(f"pid={os.getpid()}\n")
    lock_file.flush()

    return lock_file


def unlock_file(lock_file: TextIO) -> None:
    """
    Releases the lock acquired by ``open_and_lock_file`` and closes the file.

    Parameters
    ----------
    lock_file : TextIO
        A file handle returned by ``open_and_lock_file``. The file is closed
        even if releasing the lock fails.

    Unit Tests
    ----------
    """
    try:
        if sys.platform == "win32":
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def get_unique_log_filename() -> Tuple[TextIO, str]:
    """
    Finds a log filename that is not in use by another Rattlesnake instance.

    Tries ``Rattlesnake.log`` first, then ``Rattlesnake_0.log``, ``Rattlesnake_1.log``,
    and so on, until a matching ``.lock`` file can be locked. This allows multiple
    instances of Rattlesnake to run in a Jupyter Notebook without writing to the same
    log file.

    Returns
    -------
    lock : TextIO
        The locked lock file. It should be released with ``unlock_file`` when logging is
        finished.
    base_filename : str
        The log filename associated with the acquired lock.

    Unit Tests
    ----------
    """
    base_filename = "Rattlesnake.log"
    lock_filename = "Rattlesnake.log.lock"
    lock = open_and_lock_file(lock_filename)
    ind = 0
    while lock is None:
        base_filename = f"Rattlesnake_{ind}.log"
        lock_filename = f"Rattlesnake_{ind}.log.lock"
        lock = open_and_lock_file(lock_filename)
        ind = ind + 1

    return lock, base_filename


def log_file_task(
    log_queue: mp.Queue,
    shutdown_event: mp.synchronize.Event,
) -> None:
    """
    A multiprocessing function that collects logging data and writes to file.

    Messages are read from ``log_queue`` and written to a unique log file until
    ``shutdown_event`` is set or a ``GlobalCommands.QUIT`` message is received. Messages
    containing ``" ERROR"`` are also printed to the console. This task is always run on
    a seperate process, even when the ``RattlesnakeController`` is in threading mode.

    Parameters
    ----------
    log_queue : mp.queues.Queue
        The multiprocessing queue to collect logging messages from
    shutdown_event : mp.synchronize.Event
        Event that stops the logging loop when set.

    Unit Tests
    ----------
    """
    lock, log_filename = get_unique_log_filename()

    try:
        with open(log_filename, "w", encoding="utf-8") as f:
            while not shutdown_event.is_set():
                try:
                    output = log_queue.get(timeout=0.2)
                except (mpqueue.Empty, thqueue.Empty):
                    continue

                if output == GlobalCommands.QUIT:
                    f.write("Program quitting, logging terminated.\n")
                    break

                text = str(output)
                if " ERROR" in text:
                    print(text)

                num_newlines = text.count("\n")
                if num_newlines > 1:
                    text = text.replace("\n", "////", num_newlines - 1)
                f.write(text)
                f.flush()

    finally:
        unlock_file(lock)


class RattlesnakeError(Exception):
    """
    Exception used when wanting to print error statement to the Rattlesnake user
    interface dialog box.

    The validation errors should prompt the user on what to change in order to fix the
    input

    Unit Tests
    ----------
    """


class GlobalCommands(Enum):
    """
    Enumeration of command codes passed between subtasks.

    Most of these are sent from ``RattlesnakeController`` and ``Controller`` tasks and
    have implementations in many subprocesses.

    Unit Tests
    ----------
    """

    QUIT = 0  # Stop individual processes
    INITIALIZE_HARDWARE = 1  # Store hardware metadata to processes
    RUN_HARDWARE = 2  # Start running acquisition/output process
    STOP_HARDWARE = 3  # Stops running acquisition/output process
    INITIALIZE_ENVIRONMENT = 4  # Stores metadata to processes
    START_ENVIRONMENT = 5  # Tells output to start that environment
    STOP_ENVIRONMENT = 6  # Tells output to stop that environment
    INITIALIZE_SYSTEM_ID = (
        7  # Stores system id metadata to environment and system id process
    )
    START_SYSTEM_ID_NOISE = 8  # Start up system identification noise
    START_SYSTEM_ID_TRANSFER = 9  # Start up system identification transfer function
    STOP_SYSTEM_ID = 10  # Stop system identification process
    INITIALIZE_STREAMING = 11  # Creates stream file to store to
    CREATE_NEW_STREAM = 12  # Create new stream of data in file
    START_STREAMING = 13  # Acquisition sends data to stream process
    STREAMING_DATA = 14  # Continue storing data
    STOP_STREAMING = 15  # Acquisition stops sending data to stream process
    FINALIZE_STREAMING = 16  # Close out of stream file
    INITIALIZE_PROFILE = 17  # Send profile metadata to controller
    START_PROFILE = 18  # Start test from profile
    STOP_PROFILE = 19  # Stop test from profile
    PROFILE_CLOSEOUT = 20  # Tells controller the profile events are over
    STREAM_AT_TARGET_LEVEL = (
        21  # Notifies controller that environment has hit its target level
    )
    STREAM_MANUAL = 22  # Notifies controller that manual streaming has been enabled
    SEND_ENVIRONMENT_COMMAND = 23  # Sends environment specific command to environment
    SAVE_SYSTEM_ID = 24  # Tells environment to save its system id to a file
    LOAD_SYSTEM_ID = 25  # Tells environment to load a system id from a file
    RESTART_ENVIRONMENT_OUTPUT = (
        26  # Tells the output to start listening again for an environment
    )

    @property
    def label(self) -> str:
        """
        Human-readable name of the command, used by the UI for profile event list
        display.

        Underscores in the enum name are replaced with spaces and the result is
        title-cased (e.g. ``START_STREAMING`` becomes ``"Start Streaming"``).

        Returns
        -------
        str
            The display label for the command.

        Unit Tests
        ----------
        """
        return self.name.replace("_", " ").title()


class VerboseMessageQueue:
    """
    A queue class that contains automatic logging information

    Wraps a multiprocessing or threading queue so that ``put`` and ``get``
    operations are written to the log file. Each message is tagged with a
    random message id so a put can be matched to its corresponding get in the
    log. Repeated puts of the same message within ``time_threshold`` seconds
    are not logged to keep the log file readable.

    The message queue has an optional name manager parameter that is used to assign
    and environment name to the queue when it is assigned to an environment during
    the ``intialize_environment`` function.

    Attributes
    ----------
    base_queue : mp.queues.Queue or queue.Queue
        The underlying queue that stores ``(message_id, message_data_tuple)``
        pairs.
    log_queue : mp.queues.Queue
        Queue that log messages are written to.
    base_name : str
        Name of the queue used in log messages.
    environment_name : multiprocessing.managers.ValueProxy or None
        Shared value holding the name of the environment assigned to this
        queue, or ``None`` if no ``name_manager`` was provided.
    last_put_message : Enum or None
        The most recent message put to the queue.
    last_put_time : float
        Time of the most recent logged put.
    last_get_message : Enum or None
        The most recent message retrieved from the queue.
    last_get_time : float
        Time of the most recent logged get.
    last_flush : float
        Time of the most recent logged flush.
    time_threshold : float
        Seconds within which repeated puts of the same message are not logged.

    Unit Tests
    ----------
    """

    def __init__(
        self,
        log_queue: mp.Queue,
        base_queue: mp.Queue | thqueue.Queue,
        base_name: str = "",
        name_manager: mp.managers.SyncManager | None = None,
    ) -> None:
        """
        A queue class that contains automatic logging information.

        Parameters
        ----------
        log_queue : mp.queues.Queue
            A queue that a logging task will read from where the operations of
            the queue will be logged.
        base_queue : mp.queues.Queue or queue.Queue
            The underlying queue that data is put to and retrieved from. A
            ``queue.Queue`` is used when Rattlesnake runs in threaded mode.
        base_name : str, optional
            The name of the queue that will be included in the logging
            information. Defaults to ``""``.
        name_manager : multiprocessing.managers.SyncManager, optional
            Manager used to create a shared value for the environment name so
            that it can be updated with ``assign_environment`` after the queue
            has been passed to other processes. If ``None``, no environment
            name is tracked. Defaults to ``None``.

        Unit Tests
        ----------
        """
        self.base_queue = base_queue
        self.log_queue = log_queue
        self.base_name = base_name
        if name_manager:
            self.environment_name = name_manager.Value(str, "")
        else:
            self.environment_name = None
        self.last_put_message = None
        self.last_put_time = -float("inf")
        self.last_get_message = None
        self.last_get_time = -float("inf")
        self.last_flush = -float("inf")
        self.time_threshold = 1.0

    @property
    def log_name(self) -> str:
        """
        The name used to identify this queue in log messages.

        If an environment name has been assigned, the name will return, otherwise the
        base name will be returned.

        Returns
        -------
        str
            The name of the queue for log messages.

        Unit Tests
        ----------
        """
        if self.environment_name:
            try:
                env = self.environment_name.value
                return f"{self.base_name} | {env}" if env else self.base_name
            except Exception:
                return self.base_name

        return self.base_name

    def assign_environment(self, env_name: str) -> None:
        """
        Associates this queue with a named environment for logging purposes.

        This requires the queue to have been created with a ``name_manager``. It is used
        by ``EnvironmentManager`` when it assigns environments to queues.

        Parameters
        ----------
        env_name : str
            Name of the environment to include in log messages.

        Unit Tests
        ----------
        """
        self.environment_name.value = env_name

    def generate_message_id(
        self, size: int = 6, chars: str = string.ascii_letters + string.digits
    ) -> str:
        """
        Generates a random identifier for log file messages.

        Parameters
        ----------
        size : int, optional
            Number of characters in the identifier. Defaults to 6.
        chars : str, optional
            Characters to choose from. Defaults to ASCII letters and digits.

        Returns
        -------
        str
            The random identifier.

        Unit Tests
        ----------
        """
        return "".join(random.choice(chars) for _ in range(size))

    def put(
        self,
        task_name: str,
        message_data_tuple: Tuple[Enum, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Puts data into a verbose queue.

        When the put is logged, the following line is written to ``log_queue``:
            <timestamp>: <task_name> put <message> (<message_id>) to <log_name>


        Parameters
        ----------
        task_name : str
            Task name that is performing the put operation
        message_data_tuple : Tuple
            A (message,data) tuple where message is the instruction and data is any
            optional data to be passed along with the instruction.
        *args :
            Additional arguments that will be passed to the mp.queues.Queue.put function
        **kwargs :
            Additional arguments that will be passed to the mp.queues.Queue.put function

        Unit Tests
        ----------
        """
        put_time = time.time()
        if (
            self.last_put_message != message_data_tuple[0]
            or put_time - self.last_put_time > self.time_threshold
        ):
            message_id = self.generate_message_id(8)
            self.log_queue.put(
                f"{datetime.now()}: {task_name} put "
                f"{message_data_tuple[0].name} ({message_id}) to {self.log_name}\n"
            )
            self.last_put_message = message_data_tuple[0]
            self.last_put_time = put_time
        else:
            message_id = ""
        self.base_queue.put((message_id, message_data_tuple), *args, **kwargs)

    def get(self, task_name: str, *args: Any, **kwargs: Any) -> Tuple[Enum, Any]:
        """
        Gets data from a verbose queue.

        When the get is logged, the following line is written to ``log_queue``:
            <timestamp>: <task_name> got <message> (<message_id>) from <log_name>.

        Parameters
        ----------
        task_name : str
            Name of the task that is retrieving data from the queue
        *args :
            Additional arguments that will be passed to the mp.queues.Queue.get function
        **kwargs :
            Additional arguments that will be passed to the mp.queues.Queue.get function

        Returns
        -------
        message_data_tuple : Tuple[Enum, Any]
            A (message,data) tuple

        Raises
        ------
        queue.Empty
            If the queue is empty and the get is non-blocking or times out.

        Unit Tests
        ----------
        """
        message_id, message_data_tuple = self.base_queue.get(*args, **kwargs)
        if message_id != "":
            self.log_queue.put(
                f"{datetime.now()}: {task_name} got "
                f"{message_data_tuple[0].name} ({message_id}) from {self.log_name}\n"
            )
        return message_data_tuple

    def flush(self, task_name: str) -> List[Tuple[Enum, Any]]:
        """
        Flushes a verbose queue getting all data currently in the queue.

        After execution the queue should be empty barring race conditions.

        Parameters
        ----------
        task_name : str
            Name of the task that is flushing the queue

        Returns
        -------
        data : List[Tuple[Enum, Any]]
            A list of all (message,data) tuples currently in the queue.

        Unit Tests
        ----------
        """
        flush_time = time.time()
        if flush_time - self.last_flush > 0.1:
            self.log_queue.put(
                f"{datetime.now()}: {task_name} flushed {self.log_name}\n"
            )
            self.last_flush = flush_time
        data = []
        while True:
            try:
                message_id, this_data = self.base_queue.get(False)
                data.append(this_data)
                if message_id != "":
                    self.log_queue.put(
                        f"{datetime.now()}: {task_name} got {data[-1][0].name} ("
                        f"{message_id if message_id != '' else 'put not logged'})"
                        f" from {self.log_name} during flush\n"
                    )
            except mp.queues.Empty:
                return data

    def empty(self) -> bool:
        """
        Return true if the queue is empty.

        Returns
        -------
        bool
            ``True`` if the underlying queue is empty. This is not reliable
            across processes due to multiprocessing semantics.

        Unit Tests
        ----------
        """
        return self.base_queue.empty()

    def close(self) -> None:
        """
        Closes the base queue.

        Does nothing if the underlying queue does not support ``close``.

        Unit Tests
        ----------
        """
        if hasattr(self.base_queue, "close"):
            self.base_queue.close()

    def join_thread(self) -> None:
        """
        Joins the base thread.

        Joins the background feeder thread of the underlying multiprocessing
        queue. Does nothing if the underlying queue does not support
        ``join_thread``.

        Unit Tests
        ----------
        """
        if hasattr(self.base_queue, "join_thread"):
            self.base_queue.join_thread()


class QueueContainer:
    """
    A container class for the queues that the ``RattlesnakeController`` will manage.

    Unit Tests
    ----------
    """

    def __init__(
        self,
        controller_command_queue: VerboseMessageQueue,
        acquisition_command_queue: VerboseMessageQueue,
        output_command_queue: VerboseMessageQueue,
        streaming_command_queue: VerboseMessageQueue,
        log_file_queue: mp.Queue,
        input_output_sync_queue: mp.Queue,
        single_process_hardware_queue: mp.Queue,
        gui_update_queue: mp.Queue,
        environment_command_queues: Dict[str, VerboseMessageQueue],
        environment_data_in_queues: Dict[str, mp.Queue],
        environment_data_out_queues: Dict[str, mp.Queue],
    ) -> None:
        """A container class for the queues that the ``RattlesnakeController`` will
        manage.

        The controller uses many queues to pass data between the various pieces. This
        class organizes those queues into one common namespace.

        Parameters
        ----------
        controller_command_queue : VerboseMessageQueue
            Queue that is read by the controller for global controller commands.
        acquisition_command_queue : VerboseMessageQueue
            Queue that is read by the acquisition subtask for acquisition commands.
        output_command_queue : VerboseMessageQueue
            Queue that is read by the output subtask for output commands.
        streaming_command_queue : VerboseMessageQueue
            Queue that is read by the streaming subtask for streaming commands.
        log_file_queue : mp_queues.Queue
            Queue for putting logging messages that will be read by the logging subtask
            and written to a file.
        input_output_sync_queue : mp_queues.Queue
            Queue that is used to synchronize input and output signals.
        single_process_hardware_queue : mp_queues.Queue
            Queue that is used to connect the acquisition and output subtasks for
            hardware implementations that cannot have acquisition and output in separate
            processes.
        gui_update_queue : mp_queues.Queue
            Queue where various subtasks put instructions for updating the widgets in
            the user interface.
        environment_command_queues : Dict[str,VerboseMessageQueue]
            A dictionary where the keys are environment names and the values are
            VerboseMessageQueues that connect the main controller to the environment
            subtasks for sending instructions.
        environment_data_in_queues : Dict[str,multiprocessing.queues.Queue]
            A dictionary where the keys are environment names and the values are
            multiprocessing queues that connect the acquisition subtask to the
            environment subtask.  Each environment will retrieve acquired data from this
            queue.
        environment_data_out_queues : Dict[str,multiprocessing.queues.Queue]
            A dictionary where the keys are environment names and the values are
            multiprocessing queues that connect the output subtask to the environment
            subtask.  Each environment will put data that it wants the controller to
            generate in this queue.

        Unit Tests
        ----------
        """
        self.controller_command_queue = controller_command_queue
        self.acquisition_command_queue = acquisition_command_queue
        self.output_command_queue = output_command_queue
        self.streaming_command_queue = streaming_command_queue
        self.log_file_queue = log_file_queue
        self.input_output_sync_queue = input_output_sync_queue
        self.single_process_hardware_queue = single_process_hardware_queue
        self.gui_update_queue = gui_update_queue
        self.environment_command_queues = environment_command_queues
        self.environment_data_in_queues = environment_data_in_queues
        self.environment_data_out_queues = environment_data_out_queues


class EventContainer:
    """
    A container class for the multiprocessing events that the controller uses to
    coordinate readiness, shutdown, and activity state across subtasks.

    Events fall into five groups:

    - **Ready** events are set by a process when it has finished handling a command. The
      sender clears the event before sending the command and waits for it to be set
      again.
    - **Close** events are set to tell a process to shut down.
    - **Active** events are set while a process (or an environment) is running, and
      cleared when it stops.
    - **Storage** events are set when an environment has stored its system
      identification data.
    - The **alive** event is set by long-running processes to reset the controller's
      timeout while it waits on them.

    Unit Tests
    ----------
    """

    def __init__(
        self,
        controller_ready_event: mp.synchronize.Event,
        acquisition_ready_event: mp.synchronize.Event,
        output_ready_event: mp.synchronize.Event,
        streaming_ready_event: mp.synchronize.Event,
        environment_ready_events: Dict[str, mp.synchronize.Event],
        log_close_event: mp.synchronize.Event,
        controller_close_event: mp.synchronize.Event,
        acquisition_close_event: mp.synchronize.Event,
        output_close_event: mp.synchronize.Event,
        streaming_close_event: mp.synchronize.Event,
        environment_close_events: Dict[str, mp.synchronize.Event],
        acquisition_active_event: mp.synchronize.Event,
        output_active_event: mp.synchronize.Event,
        streaming_active_event: mp.synchronize.Event,
        environment_active_events: Dict[str, mp.synchronize.Event],
        environment_sysid_active_events: Dict[str, mp.synchronize.Event],
        environment_sysid_stored_events: Dict[str, mp.synchronize.Event],
        ping_alive_event: mp.synchronize.Event,
    ) -> None:
        """
        A container class for the events that the controller will manage.

        Parameters
        ----------
        controller_ready_event : mp.synchronize.Event
            Cleared when a profile is started or stopped, and set by the
            controller process when it receives ``PROFILE_CLOSEOUT``. Set
            whenever no profile is running.
        acquisition_ready_event : mp.synchronize.Event
            Set by the acquisition process when it has finished handling a
            command.
        output_ready_event : mp.synchronize.Event
            Set by the output process when it has finished handling a command.
        streaming_ready_event : mp.synchronize.Event
            Set by the streaming process when it has finished handling a
            command.
        environment_ready_events : Dict[str, mp.synchronize.Event]
            A dictionary where the keys are queue names and the values are
            events set by each environment process when it has finished
            handling a command.
        log_close_event : mp.synchronize.Event
            Set to shut down the log process.
        controller_close_event : mp.synchronize.Event
            Set to shut down the controller process.
        acquisition_close_event : mp.synchronize.Event
            Set to shut down the acquisition process.
        output_close_event : mp.synchronize.Event
            Set to shut down the output process.
        streaming_close_event : mp.synchronize.Event
            Set to shut down the streaming process.
        environment_close_events : Dict[str, mp.synchronize.Event]
            A dictionary where the keys are queue names and the values are
            events set to shut down each environment process.
        acquisition_active_event : mp.synchronize.Event
            Set while the acquisition process is acquiring data from hardware.
        output_active_event : mp.synchronize.Event
            Set while the output process is writing data to hardware.
        streaming_active_event : mp.synchronize.Event
            Set while acquired data is being streamed to file.
        environment_active_events : Dict[str, mp.synchronize.Event]
            A dictionary where the keys are queue names and the values are
            events set while each environment is running.
        environment_sysid_active_events : Dict[str, mp.synchronize.Event]
            A dictionary where the keys are queue names and the values are
            events set while each environment is running system
            identification.
        environment_sysid_stored_events : Dict[str, mp.synchronize.Event]
            A dictionary where the keys are queue names and the values are
            events set when each environment has stored its system
            identification data.
        ping_alive_event : mp.synchronize.Event
            Set by long-running processes to tell the controller to reset its
            timeout while waiting on other events.

        Unit Tests
        ----------
        """
        # Ready Events
        self.controller_ready_event = controller_ready_event
        self.acquisition_ready_event = acquisition_ready_event
        self.output_ready_event = output_ready_event
        self.streaming_ready_event = streaming_ready_event
        self.environment_ready_events = environment_ready_events
        # Close Events
        self.log_close_event = log_close_event
        self.controller_close_event = controller_close_event
        self.acquisition_close_event = acquisition_close_event
        self.output_close_event = output_close_event
        self.streaming_close_event = streaming_close_event
        self.environment_close_events = environment_close_events
        # Active Events
        self.acquisition_active_event = acquisition_active_event
        self.output_active_event = output_active_event
        self.streaming_active_event = streaming_active_event
        self.environment_active_events = environment_active_events
        self.environment_sysid_active_events = environment_sysid_active_events
        # Storage Events
        self.environment_sysid_stored_events = environment_sysid_stored_events
        # Alive Event
        self.ping_alive_event = ping_alive_event


def flush_queue(
    queue: mp.Queue | thqueue.Queue | VerboseMessageQueue,
    timeout: float | None = None,
) -> List[Any]:
    """
    Flushes a queue by getting all the data currently in it.

    Parameters
    ----------
    queue : mp.queues.Queue or queue.Queue or VerboseMessageQueue
        The queue to flush.
    timeout : float, optional
        If ``None``, the queue is read without blocking and the flush stops as
        soon as it is empty. Otherwise each get blocks for up to ``timeout``
        seconds, and the flush stops once a get times out. Defaults to
        ``None``.

    Returns
    -------
    data : List[Any]
        A list of all data that were in the queue at flush.

    Unit Tests
    ----------
    """
    data = []
    while True:
        try:
            if isinstance(queue, VerboseMessageQueue):
                data.append(
                    queue.get(
                        "Flush",
                        block=False if timeout is None else True,
                        timeout=timeout,
                    )
                )
            else:
                data.append(
                    queue.get(block=False if timeout is None else True, timeout=timeout)
                )
        except (thqueue.Empty, mpqueue.Empty):
            return data


def gui_queue_cleanup(
    gui_update_queue: mp.Queue | thqueue.Queue,
    shutdown_event: threading.Event,
    gui_active_check: Callable[[], bool],
    max_queue_size: int = 500,
    poll_interval: float = 0.25,
) -> None:
    """
    This task runs when Rattlesnake does not have a GUI and will keep the GUI update
    queue from growing when a max size is hit. This prevents a memory leak when
    Rattlesnake is running purely in headless mode.

    Every ``poll_interval`` seconds, if ``gui_active_check`` returns ``False``, the
    oldest messages are discarded until the queue holds no more than ``max_queue_size``
    items. The loop exits when ``shutdown_event`` is set.

    Parameters
    ----------
    gui_update_queue : mp.queues.Queue or queue.Queue
        The queue that processes put GUI update instructions to.
    shutdown_event : threading.Event
        Event that stops the cleanup loop when set.
    gui_active_check : Callable[[], bool]
        Function that returns ``True`` when a GUI is attached and consuming the queue,
        in which case nothing is discarded.
    max_queue_size : int, optional
        The maximum number of items left in the queue after cleanup. Defaults to 500.
    poll_interval : float, optional
        Seconds between cleanup checks. Defaults to 0.25.

    Unit Tests
    ----------
    """
    while not shutdown_event.wait(poll_interval):
        if gui_active_check():
            continue
        try:
            while gui_update_queue.qsize() > max_queue_size:
                gui_update_queue.get_nowait()
        except (thqueue.Empty, mpqueue.Empty):
            continue


# Maps a direction string (e.g. "X+", "-RY") to a signed integer direction, where 1-3
# are translations, 4-6 are rotations, and the sign is the direction. Used by the random
# vibration environment and UI, and the SDynPy virtual hardware.
_direction_map: Dict[str | None, int] = {
    "X+": 1,
    "X": 1,
    "+X": 1,
    "Y+": 2,
    "Y": 2,
    "+Y": 2,
    "Z+": 3,
    "Z": 3,
    "+Z": 3,
    "RX+": 4,
    "RX": 4,
    "+RX": 4,
    "RY+": 5,
    "RY": 5,
    "+RY": 5,
    "RZ+": 6,
    "RZ": 6,
    "+RZ": 6,
    "X-": -1,
    "-X": -1,
    "Y-": -2,
    "-Y": -2,
    "Z-": -3,
    "-Z": -3,
    "RX-": -4,
    "-RX": -4,
    "RY-": -5,
    "-RY": -5,
    "RZ-": -6,
    "-RZ": -6,
    "": 0,
    None: 0,
}
# Inverse of _direction_map, mapping a signed integer direction back to its canonical
# string. Used by the SDynPy virtual hardware.
_direction_inv_map: Dict[int, str] = {
    0: "",
    1: "X+",
    2: "Y+",
    3: "Z+",
    4: "RX+",
    5: "RY+",
    6: "RZ+",
    -1: "X-",
    -2: "Y-",
    -3: "Z-",
    -4: "RX-",
    -5: "RY-",
    -6: "RZ-",
}


class IPAddress:
    """
    Container for information about IPAddress, including host name, ipv4, and ipv6
    address.

    Used to compile backup addresses to fall back on when the connection does not
    respond within the timeout windows.

    Attributes
    ----------
    host_name : str or None
        Host name of the device, in the form ``"BK<type>-<serial>"`` for LAN-XI modules.
    ipv4_address : str or None
        IPv4 address of the device.
    ipv6_address : str or None
        IPv6 address of the device, wrapped in brackets and including the scope id if
        present (e.g. ``"[fe80::1%5]"``).
    valid_ip : bool
        ``True`` if the device has been confirmed reachable.
    module_info : dict or None
        JSON response from the device's ``/rest/rec/module/info`` endpoint.
    sync_type : dict or None
        JSON response from the device's ``/rest/rec/syncmode`` endpoint.
    validation_timeout : float
        Timeout in seconds for REST requests to the device.

    Unit Tests
    ----------
    """

    def __init__(
        self,
        host_name: str | None = None,
        ipv4_address: str | None = None,
        ipv6_address: str | None = None,
        valid_ip: bool = False,
    ) -> None:
        """
        Initializes an address with no module or sync information.

        Parameters
        ----------
        host_name : str, optional
            Host name of the device. Defaults to ``None``.
        ipv4_address : str, optional
            IPv4 address of the device. Defaults to ``None``.
        ipv6_address : str, optional
            IPv6 address of the device. Defaults to ``None``.
        valid_ip : bool, optional
            Whether the address is already known to be valid. Defaults to
            ``False``.

        Unit Tests
        ----------
        """
        self.host_name = host_name
        self.ipv4_address = ipv4_address
        self.ipv6_address = ipv6_address
        self.valid_ip = valid_ip
        self.module_info = None
        self.sync_type = None
        self.validation_timeout = 5

    def get_ip_from_host_name(self) -> None:
        """
        Resolves ipv4_address and ipv6_address from host_name via DNS lookup.

        Sets ``valid_ip`` to ``False`` if there is no host name or the lookup fails. A
        successful lookup does not set ``valid_ip`` to ``True``.

        Unit Tests
        ----------
        """
        if not self.host_name:
            self.valid_ip = False
            return

        try:
            # Get the address info for the hostname
            socket_info = socket.getaddrinfo(self.host_name, None)
            for family, _, _, _, sockaddr in socket_info:
                if family == socket.AF_INET:
                    self.ipv4_address = sockaddr[0]

                elif family == socket.AF_INET6:
                    ipv6 = sockaddr[0]
                    scope_id = sockaddr[3]

                    if scope_id:
                        self.ipv6_address = f"[{ipv6}%{scope_id}]"
                    else:
                        self.ipv6_address = f"[{ipv6}]"
        except Exception:
            self.valid_ip = False

    def get_host_name_from_ip(self) -> None:
        """
        Queries the device REST API to derive host_name from the IP address

        The IPv6 address is used if present, otherwise the IPv4 address. On success,
        ``host_name``, ``module_info`` and ``sync_type`` are filled in and ``valid_ip``
        is set to ``True``. On failure, ``valid_ip`` is set to ``False``.

        Unit Tests
        ----------
        """
        if self.ipv6_address:
            host = "http://" + self.ipv6_address
        elif self.ipv4_address:
            host = "http://" + self.ipv4_address
        else:
            self.valid_ip = False
            return

        try:
            response = requests.get(
                host + "/rest/rec/module/info",
                timeout=self.validation_timeout,
            )
            self.module_info = response.json()
            response = requests.get(
                host + "/rest/rec/syncmode",
                timeout=self.validation_timeout,
            )
            self.sync_type = response.json()
            module_type = self.module_info["module"]["type"]["number"]
            module_serial = self.module_info["module"]["serial"]
            self.host_name = f"BK{module_type}-{module_serial}"
            self.valid_ip = True
        except Exception:
            self.valid_ip = False

    def validate(self) -> None:
        """
        Queries the device REST API to confirm the address is reachable and valid.

        The IPv6 address is used if present, otherwise the IPv4 address. On success,
        ``module_info`` and ``sync_type`` are filled in and ``valid_ip`` is set to
        ``True``. On failure, ``valid_ip`` is set to ``False``. Unlike
        ``get_host_name_from_ip``, ``host_name`` is not changed.

        Unit Tests
        ----------
        """
        if self.ipv6_address:
            host = "http://" + self.ipv6_address
        elif self.ipv4_address:
            host = "http://" + self.ipv4_address
        else:
            self.valid_ip = False
            return

        try:
            response = requests.get(
                host + "/rest/rec/module/info",
                timeout=self.validation_timeout,
            )
            self.module_info = response.json()
            response = requests.get(
                host + "/rest/rec/syncmode",
                timeout=self.validation_timeout,
            )
            self.sync_type = response.json()
            self.valid_ip = True
        except Exception:
            self.valid_ip = False

    def __eq__(self, other: object) -> bool:
        """
        The equality of addresses takes priority of ipv6 > ipv4 > bknum. This order is
        chosen based off the priority of validation as well as how reliable each
        connection is.

        Parameters
        ----------
        other : object
            The object to compare against.

        Returns
        -------
        bool
            ``True`` if the highest-priority field that both addresses have filled in
            matches. ``False`` if it does not match or no field is filled in on both.
            ``NotImplemented`` if ``other`` is not an ``IPAddress``.

        Unit Tests
        ----------
        """
        if not isinstance(other, IPAddress):
            return NotImplemented
        if self.ipv6_address is not None and other.ipv6_address is not None:
            return self.ipv6_address == other.ipv6_address
        if self.ipv4_address is not None and other.ipv4_address is not None:
            return self.ipv4_address == other.ipv4_address
        if self.host_name is not None and other.host_name is not None:
            return self.host_name == other.host_name
        return False

    # Equality is content-based on mutable fields, so instances are not safe
    # to use as dict/set keys.
    __hash__ = None

    def merge(self, other: IPAddress) -> Tuple[IPAddress, bool]:
        """
        Combines this address with another address that ``__eq__`` considers the same
        device, keeping whichever fields are filled in.

        Where both addresses have a value for a field, this address's value is kept.

        Parameters
        ----------
        other : IPAddress
            The address to merge with.

        Returns
        -------
        merged : IPAddress
            A new address with the combined fields. If there was a conflict,
            ``valid_ip`` is ``False`` and ``module_info`` and ``sync_type`` are cleared
            so the address is re-validated.
        conflict : bool
            ``True`` if both addresses had different values for the same field.

        Unit Tests
        ----------
        """
        merged = IPAddress()
        conflict = False
        for attr in ("ipv6_address", "ipv4_address", "host_name"):
            self_value = getattr(self, attr)
            other_value = getattr(other, attr)
            if (
                self_value is not None
                and other_value is not None
                and self_value != other_value
            ):
                conflict = True
            setattr(merged, attr, self_value if self_value is not None else other_value)

        if conflict:
            merged.valid_ip = False
            merged.module_info = None
            merged.sync_type = None
        else:
            merged.valid_ip = self.valid_ip or other.valid_ip
            merged.module_info = self.module_info or other.module_info
            merged.sync_type = self.sync_type or other.sync_type

        return merged, conflict


def autofill_single_ip_address(ip_address: IPAddress) -> IPAddress:
    """
    Worker function for a single ip_address object.

    Fills in any missing host name or IP addresses for ``ip_address`` by querying the
    device and DNS, without touching UI widgets so it can be run in a worker thread.
    Addresses that are already valid are returned unchanged.

    Parameters
    ----------
    ip_address : IPAddress
        The address to fill in. Modified in place.

    Returns
    -------
    IPAddress
        The same ``ip_address`` object, after its fields have been filled in.

    Unit Tests
    ----------
    """
    if ip_address.valid_ip:
        return ip_address

    if ip_address.ipv6_address:
        ip_address.get_host_name_from_ip()
        ip_address.get_ip_from_host_name()
    elif ip_address.ipv4_address:
        ip_address.get_host_name_from_ip()
        ip_address.get_ip_from_host_name()
    if ip_address.host_name:
        ip_address.get_ip_from_host_name()

    print(
        f"host: {ip_address.host_name}, ipv4: {ip_address.ipv4_address}, "
        f"ipv6: {ip_address.ipv6_address}"
    )

    return ip_address


def search_for_lanxi_devices(timeout: float) -> List[IPAddress]:
    """
    Repeatedly polls the network for B&K LAN-XI devices until timeout has been reached.

    Parameters
    ----------
    timeout : float
        The number of seconds to search for devices before returning

    Returns
    -------
    list[IPAddress]
        A list of unique IPAddress objects for the LAN-XI devices found

    Unit Tests
    ----------
    """
    unique_addresses = {}
    start_time = time.perf_counter()
    while True:
        now = time.perf_counter()
        elapsed = now - start_time
        if elapsed >= timeout:
            break

        devices = find_lanxi_devices()
        for device in devices:
            unique_addresses[device.ipv4_address] = device

        print(f"found={len(devices)} devices")

    return list(unique_addresses.values())


def find_lanxi_devices() -> List[IPAddress]:
    """
    Scans the ARP table for candidate LAN-XI IP addresses and validates them.

    Candidate addresses are extracted from the local ARP table by matching the
    169.254.x.x link-local address range used by LAN-XI devices, then each candidate is
    queried concurrently to confirm it is a valid LAN-XI module.

    Returns
    -------
    list[IPAddress]
        A list of IPAddress objects for the LAN-XI devices found

    Notes
    -----
    Only used by ``search_for_lanxi_devices``.

    Unit Tests
    ----------
    """
    result = subprocess.run(["arp", "-a"], capture_output=True, text=True, check=False)

    addresses = set()

    # Match 169.254.x.x where x is 1 to 3 digits
    pattern = re.compile(r"\b169\.254\.(\d{1,3})\.(\d{1,3})\b")

    for line in result.stdout.splitlines():
        match = pattern.search(line)
        if match:
            third = int(match.group(1))
            fourth = int(match.group(2))

            # Ensure valid IPv4 octet range
            if 0 <= third <= 255 and 0 <= fourth <= 255:
                addresses.add(f"169.254.{third}.{fourth}")

    candidates = sorted(addresses, key=lambda ip: tuple(map(int, ip.split("."))))

    max_workers = min(10, len(addresses))
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(check_lanxi_candidate, ipv4): ipv4 for ipv4 in candidates
        }

        for future in as_completed(futures):
            host_name, ipv4, info, sync, valid = future.result()
            if valid:
                address = IPAddress(host_name, ipv4)
                address.module_info = info
                address.sync_type = sync
                address.get_ip_from_host_name()
                results.append(address)

    return results


def check_lanxi_candidate(
    ipv4_address: str,
) -> Tuple[str | None, str, dict | None, dict | None, bool]:
    """
    Queries a candidate IP address to check if it is a valid LAN-XI device.

    Parameters
    ----------
    ipv4_address : str
        The candidate IPv4 address to query

    Returns
    -------
    host_name : str or None
        The device host name if valid, otherwise None
    ipv4_address : str
        The IPv4 address that was queried
    info : dict or None
        The module info JSON response if valid, otherwise None
    sync : dict or None
        The sync mode JSON response if valid, otherwise None
    valid : bool
        True if the candidate responded as a valid LAN-XI device

    Unit Tests
    ----------
    """
    host = f"http://{ipv4_address}"
    try:
        response = requests.get(
            host + "/rest/rec/module/info",
            timeout=0.3,
        )
        response.raise_for_status()
        info = response.json()
        response = requests.get(
            host + "/rest/rec/syncmode",
            timeout=0.3,
        )
        sync = response.json()

        host_name = f"BK{info['module']['type']['number']}-{info['module']['serial']}"
        valid = True
    except Exception:
        host_name = None
        info = None
        sync = None
        valid = False

    return (host_name, ipv4_address, info, sync, valid)


def add_unique_ip_address(
    unique_addresses: List[IPAddress], candidate: IPAddress
) -> bool:
    """
    Adds ``candidate`` into ``unique_addresses`` in place, merging it into
    an existing entry (per ``IPAddress.__eq__``/``merge``) instead of
    appending a duplicate when an equivalent address is already present.

    Parameters
    ----------
    unique_addresses : list[IPAddress]
        The list to add to. Modified in place.
    candidate : IPAddress
        The address to add or merge.

    Returns
    -------
    bool
        True if the candidate was merged into an existing entry and that
        merge disagreed on a field, meaning the merged entry needs to be
        re-validated against the device.

    Unit Tests
    ----------
    """
    for index, existing in enumerate(unique_addresses):
        if existing == candidate:
            merged, conflict = existing.merge(candidate)
            unique_addresses[index] = merged
            return conflict
    unique_addresses.append(candidate)
    return False


# endregion


# region Loading
def load_time_history(signal_path: str, sample_rate: float) -> np.ndarray:
    """
    Loads a time history from a given file

    The signal can be loaded from numpy files (.npz, .npy) or matlab files (.mat). For
    .mat and .npz files, the time data can be included in the file in the 't' field, or
    it can be excluded and the sample_rate input argument will be used.  If time data is
    specified, it will be linearly interpolated to the sample rate of the controller.

    For these file types, the signal should be stored in the 'signal' field.  For .npy
    files, only one array is stored, so it is treated as the signal, and the sample_rate
    input argument is used to construct the time data.

    If the signal has an odd number of samples, the last sample is dropped so the length
    is even.

    Parameters
    ----------
    signal_path : str
        Path to the file from which to load the time history
    sample_rate : float
        The sample rate of the loaded signal.

    Returns
    -------
    signal : np.ndarray
        A signal loaded from the file

    Raises
    ------
    ValueError
        If the file extension is not ``.npy``, ``.npz`` or ``.mat``.

    Unit Tests
    ----------
    """
    _, extension = os.path.splitext(signal_path)
    if extension.lower() == ".npy":
        signal = np.load(signal_path)
    elif extension.lower() == ".npz":
        data = np.load(signal_path)
        signal = data["signal"]
        try:
            times = data["t"].squeeze()
            fn = interp1d(times, signal)
            abscissa = np.arange(
                0, max(times) + 1 / sample_rate - 1e-10, 1 / sample_rate
            )
            abscissa = abscissa[abscissa <= max(times)]
            signal = fn(abscissa)
        except KeyError:
            pass
    elif extension.lower() == ".mat":
        data = loadmat(signal_path)
        signal = data["signal"]
        try:
            times = data["t"].squeeze()
            fn = interp1d(times, signal)
            abscissa = np.arange(
                0, max(times) + 1 / sample_rate - 1e-10, 1 / sample_rate
            )
            abscissa = abscissa[abscissa <= max(times)]
            signal = fn(abscissa)
        except KeyError:
            pass
    else:
        raise ValueError(
            f"Could Not Determine the file type from the filename "
            f"{signal_path}: {extension}"
        )
    if signal.shape[-1] % 2 == 1:
        signal = signal[..., :-1]
    return signal


def load_csv_matrix(file: str) -> List[List[str]]:
    """
    Loads a matrix from a CSV file

    Parameters
    ----------
    file : str
        Path to the file that will be loaded

    Returns
    -------
    data : list[list[str]]
        A 2D nested list of strings containing the matrix in the CSV file.


    Unit Tests
    ----------
    """
    with open(file, "r", encoding="utf-8") as f:
        data = []
        for line in f:
            data.append([])
            for v in line.split(","):
                data[-1].append(v.strip())
    return data


def save_csv_matrix(data: Iterable[Iterable[str]], file: str) -> None:
    """
    Saves 2D matrix data to a csv file.

    Parameters
    ----------
    data : 2D iterable of str
        A 2D nested iterable of strings that will be written to a file.
    file : str
        The path to a file where the data will be written.


    Unit Tests
    ----------
    """
    text = "\n".join([",".join(row) for row in data])
    with open(file, "w", encoding="utf-8") as f:
        f.write(text)


def load_python_module(module_path: str) -> ModuleType:
    """
    Loads in the Python file at the specified path as a module at runtime.

    Parameters
    ----------
    module_path : str
        Path to the module to be loaded

    Returns
    -------
    module : ModuleType
        A reference to the loaded module

    Unit Tests
    ----------
    """
    _, file = os.path.split(module_path)
    file, _ = os.path.splitext(file)
    spec = importlib.util.spec_from_file_location(file, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_control_module(control_script: str) -> ModuleType:
    """
    Loads a control law module from a file path or an importable module name.

    If ``control_script`` ends in ``.py`` or is an existing file, it is loaded from that
    path with ``load_python_module``. Otherwise it is treated as a dotted module name
    and imported with ``importlib.import_module``.

    Parameters
    ----------
    control_script : str
        Path to a Python file, or the name of an importable module, containing the
        control law.

    Returns
    -------
    module : ModuleType
        A reference to the loaded module

    Unit Tests
    ----------
    """
    if not control_script.endswith(".py") and not os.path.isfile(control_script):
        return importlib.import_module(control_script)
    return load_python_module(control_script)


def worksheet_cell_str(value: Any, default: str = "") -> str:
    """
    This is used to cleanse inputs before writing them to excel as the GUI automatically
    converts everything to a string but the headless mode will keep it as
    ints/floats/etc. which changes how openpyxl stores the information to an excel
    sheet.

    Parameters
    ----------
    value : Any
        The value to convert. Strings are stripped, floats with integer values are
        written without a decimal point, and datetimes are written as an ISO date
        (``YYYY-MM-DD``).
    default : str, optional
        Value returned when ``value`` is ``None`` or an empty string. Defaults to
        ``""``.

    Returns
    -------
    str
        The string to write to the worksheet cell.

    Unit Tests
    ----------
    """
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        return value if value else default
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, datetime):
        return value.date().isoformat()
    return str(value)


def read_transformation_matrix_from_worksheet(
    worksheet: "Worksheet", start_row: int, num_rows: int, start_col: int
) -> np.ndarray | None:
    """
    Reads a numeric matrix from a block of cells in an Excel worksheet

    Each row is read starting at start_col until the first blank, comment (starting with
    '#'), or "none" cell is encountered.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        The worksheet to read the matrix from.
    start_row : int
        The 1-indexed row where the matrix begins.
    num_rows : int
        The number of rows to read.
    start_col : int
        The 1-indexed column where each row of the matrix begins.

    Returns
    -------
    np.ndarray or None
        The matrix read from the worksheet, or None if the first cell is blank or
        contains "none".

    Unit Tests
    ----------
    """
    first_cell = worksheet.cell(start_row, start_col).value
    if first_cell is None or (str(first_cell).strip().lower() == "none"):
        return None

    matrix = []
    for i in range(num_rows):
        # Read the entire row until the first blank cell
        row = []
        col_idx = start_col
        while True:
            value = worksheet.cell(start_row + i, col_idx).value
            if value is None or (
                isinstance(value, str)
                and (value.startswith("#") or value.strip() == "")
            ):
                break
            row.append(float(value))
            col_idx += 1

        matrix.append(row)

    if not matrix:
        return None
    return np.array(matrix, dtype=float)


def save_rattlesnake_to_netcdf(
    netcdf_dataset: "netCDF4.Dataset",
    hardware_metadata: "HardwareMetadata | None" = None,
    environment_metadata_dict: "Dict[str, EnvironmentMetadata] | None" = None,
) -> None:
    """
    Saves hardware and environment metadata to an open netCDF4 dataset

    Parameters
    ----------
    netcdf_dataset : netCDF4.Dataset
        An open, writable netCDF4 dataset to save the metadata to
    hardware_metadata : HardwareMetadata, optional
        The hardware metadata to save
    environment_metadata_dict : dict, optional
        A dictionary where the keys are environment names and the values are the
        environment metadata objects to save, one per environment

    Notes
    -----
    This is in utilities instead of load_utilities since the modal environment saves
    netcdf files internally.

    Unit Tests
    ----------
    """
    if hardware_metadata:
        hardware_metadata.save_metadata_to_netcdf(netcdf_dataset)
        netcdf_dataset.createDimension(
            "num_environments", len(environment_metadata_dict)
        )
    if environment_metadata_dict:
        var = netcdf_dataset.createVariable(
            "environment_names", str, ("num_environments",)
        )
        environment_booleans = []
        for i, metadata in enumerate(environment_metadata_dict.values()):
            var[i] = metadata.environment_name
            environment_booleans.append(metadata.channel_list_bools)
        var = netcdf_dataset.createVariable(
            "environment_types", int, ("num_environments",)
        )
        for i, metadata in enumerate(environment_metadata_dict.values()):
            var[i] = metadata.environment_type.value
        var = netcdf_dataset.createVariable(
            "environment_active_channels",
            "i1",
            ("response_channels", "num_environments"),
        )
        var[...] = np.array(environment_booleans, dtype="int8").T
        for environment_metadata in environment_metadata_dict.values():
            group_handle = netcdf_dataset.createGroup(
                environment_metadata.environment_name
            )
            environment_metadata.save_metadata_to_netcdf(group_handle)


# endregion


# region Math Operations
def coherence(
    cpsd_matrix: np.ndarray, row_column: Tuple[int, int] | None = None
) -> np.ndarray:
    """
    Compute coherence from a CPSD matrix.

    Parameters
    ----------
    cpsd_matrix : np.ndarray
        A 3D complex numpy array where the first index corresponds to the frequency line
        and the second and third indices correspond to the rows and columns of the
        matrix.
    row_column : Tuple[int, int], optional
        Optional argument to compute the coherence at just a single (row,column) pair.
        (Default value = Compute Entire Matrix).

    Returns
    -------
    coherence : np.ndarray
        3D array of coherence values where the [i,j,k] entry corresponds to the
        coherence of the CPSD matrix for the ith frequency line, jth row, and kth
        column. If ``row_column`` is given, a 1D array of coherence values at each
        frequency line for that pair.

    Unit Tests
    ----------
    """
    if row_column is None:
        diag = np.einsum("ijj->ij", cpsd_matrix)
        return np.real(
            np.abs(cpsd_matrix) ** 2 / (diag[:, :, np.newaxis] * diag[:, np.newaxis, :])
        )

    row, column = row_column
    return np.real(
        np.abs(cpsd_matrix[:, row, column]) ** 2
        / (cpsd_matrix[:, row, row] * cpsd_matrix[:, column, column])
    )


def cpsd_to_time_history(
    cpsd_matrix: np.ndarray,
    sample_rate: float,
    df: float,
    output_oversample: int = 1,
) -> np.ndarray:
    # pylint: disable=invalid-name
    """
    Generates a time history realization from a CPSD matrix.

    Parameters
    ----------
    cpsd_matrix : np.ndarray
        A 3D complex np.ndarray representing a CPSD matrix where the first dimension is
        the frequency line and the second two dimensions are the rows and columns of the
        matrix at each frequency line.
    sample_rate : float
        The sample rate of the controller in samples per second
    df : float
        The frequency spacing of the cpsd matrix
    output_oversample : int, optional
        Factor by which the output is oversampled relative to ``sample_rate``, by
        zero-padding the spectrum. Defaults to 1.

    Returns
    -------
    output : np.ndarray
        A numpy array containing the generated signals, with shape ``(num_signals,
        num_samples)``.

    Notes
    -----
    Not currently used anywhere. ``CPSDSignalGenerator`` has its own implementation.

    [1] R. Schultz and G. Nelson, "Input signal synthesis for open-loop
    multiple-input/multiple-output testing," Proceedings of the International Modal
    Analysis Conference, 2019.

    Unit Tests
    ----------
    """
    # Compute SVD broadcasting over all frequency lines
    [U, S, Vh] = np.linalg.svd(cpsd_matrix, full_matrices=False)
    # Reform using the sqrt of the S matrix
    Lsvd = U * np.sqrt(S[:, np.newaxis, :]) @ Vh
    # Compute Random Process
    W = np.sqrt(0.5) * (
        np.random.randn(*cpsd_matrix.shape[:-1], 1)
        + 1j * np.random.randn(*cpsd_matrix.shape[:-1], 1)
    )
    Xv = 1 / np.sqrt(df) * Lsvd @ W
    # Ensure that the signal is real by setting the nyquist and DC component to 0
    Xv[[0, -1], :, :] = 0
    # Compute the IFFT, using the real version makes it so you don't need
    # negative frequencies
    zero_padding = np.zeros(
        [(output_oversample - 1) * (Xv.shape[0] - 1)] + list(Xv.shape[1:]),
        dtype=Xv.dtype,
    )
    xv = (
        np.fft.irfft(np.concatenate((Xv, zero_padding), axis=0) / np.sqrt(2), axis=0)
        * output_oversample
        * sample_rate
    )
    output = xv[:, :, 0].T
    return output


def reduce_array_by_coordinate(
    array: np.ndarray,
    coordinate: np.ndarray,
    control_coordinate: np.ndarray,
    excitation_coordinate: np.ndarray | None = None,
) -> np.ndarray:
    """
    Picks out entries in an array based on coordinate strings.

    Entries are matched on node and the absolute value of direction. If the signs of the
    matched directions differ, the entry is negated.

    Parameters
    ----------
    array : np.ndarray
        Array to parse.
    coordinate : np.ndarray
        Coordinate names associated with the array.
    control_coordinate : np.ndarray
        Coordinate names associated with control degrees of freedom.
    excitation_coordinate : np.ndarray, optional
        Coordinate names associated with excitation degrees of freedom. If not given,
        ``control_coordinate`` is used.

    Returns
    -------
    np.ndarray
        An array sorted by the provided coordinate strings.

    Raises
    ------
    ValueError
        If requested coordinate strings do not exist in the array strings.

    Unit Tests
    ----------
    """
    if excitation_coordinate is None:
        excitation_coordinate = control_coordinate.copy()
    # transforming control_coordinate from array of shape (N,) to (N, N, 2)
    # (equivalent to SDynPy outer_product)
    if array.ndim == 3:
        control_coordinate = np.array(
            np.meshgrid(control_coordinate, excitation_coordinate)
        ).T
    elif array.ndim == 2:
        control_coordinate = np.tile(control_coordinate, (2, 1)).T
    output_shape = control_coordinate.shape[:-1]
    ordinate = np.moveaxis(array, 0, -1)
    flat_array = ordinate.reshape(-1, ordinate.shape[-1])
    flat_coord = coordinate.flatten().reshape(-1, 2)
    index_array = np.empty(output_shape, dtype=int)
    positive_coordinates = flat_coord.copy()
    positive_coordinates["direction"] = abs(flat_coord["direction"])
    positive_control_coordinates = control_coordinate.copy()
    positive_control_coordinates["direction"] = abs(control_coordinate["direction"])
    for index in np.ndindex(output_shape):
        positive_key = positive_control_coordinates[index]
        try:
            index_array[index] = np.where(
                np.all(positive_coordinates == positive_key, axis=-1)
            )[0][0]
        except IndexError as exc:
            raise ValueError(
                f"Coordinate {str(control_coordinate[index])} not found in data array"
            ) from exc
    return_array = flat_array[index_array]
    return_coord = flat_coord[index_array]

    ordinate_multiplication_array = np.prod(
        np.sign(return_coord["direction"]) * np.sign(control_coordinate["direction"]),
        axis=-1,
    )
    # Set up for broadcasting
    ordinate_multiplication_array = ordinate_multiplication_array[..., np.newaxis]
    # Remove zeros and replace with 1s because we don't flip signs if
    # there is no direction associated with the coordinate
    ordinate_multiplication_array[ordinate_multiplication_array == 0] = 1
    return_array *= ordinate_multiplication_array
    return np.moveaxis(return_array, -1, 0)


def db2scale(decibel: float | np.ndarray) -> float | np.ndarray:
    """
    Converts a decibel value to a scale factor.

    Parameters
    ----------
    decibel : float or np.ndarray
        Value in decibel scale.

    Returns
    -------
    scale : float or np.ndarray
        Value in linear scale.

    Unit Tests
    ----------
    """
    return 10 ** (decibel / 20)


def power2db(power: float | np.ndarray) -> float | np.ndarray:
    """
    Converts a power quantity to decibels.

    Parameters
    ----------
    power : float or np.ndarray
        Power quantity in linear units.

    Returns
    -------
    float or np.ndarray
        ``10 * log10(power)``.

    Unit Tests
    ----------
    """
    return 10 * np.log10(power)


def scale2db(scale: float | np.ndarray) -> float | np.ndarray:
    """
    Converts a scale quantity to decibels.

    Parameters
    ----------
    scale : float or np.ndarray
        Amplitude (root-power) quantity in linear units.

    Returns
    -------
    float or np.ndarray
        ``20 * log10(scale)``.

    Unit Tests
    ----------
    """
    return 20 * np.log10(scale)


def rms_time(
    signal: np.ndarray,
    axis: int | Tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> float | np.ndarray:
    """
    Computes RMS over a time signal.

    Parameters
    ----------
    signal : np.ndarray
        Signal over which to compute the root-mean-square value.
    axis : int or Tuple[int, ...], optional
        The dimension over which the mean is performed (Default value = None).
    keepdims : bool, optional
        Whether to keep the dimension over which mean is computed (Default value =
        False).

    Returns
    -------
    rms : numpy scalar or numpy.ndarray
        The root-mean-square value of signal

    Unit Tests
    ----------
    """
    return np.sqrt(np.mean(signal**2, axis=axis, keepdims=keepdims))


def rms_csd(csd: np.ndarray, df: float) -> np.ndarray:
    """
    Computes RMS of a CPSD matrix.

    Parameters
    ----------
    csd : np.ndarray
        3D complex Numpy array where the first dimension is the frequency line and the
        second two dimensions are the rows and columns of the CPSD matrix.
    df : float
        Frequency spacing of the CPSD matrix.

    Returns
    -------
    rms : numpy.ndarray
        The root-mean-square value of each signal in the CPSD matrix, computed from the
        diagonal.

    Unit Tests
    ----------
    """
    return np.sqrt(np.einsum("ijj->j", csd).real * df)


def trac(th_1: np.ndarray, th_2: np.ndarray | None = None) -> np.ndarray:
    """
    Computes the time response assurance criterion.

    Signals are compared along the last axis. Pairs where either signal is all zeros
    have a trac of 1.

    Parameters
    ----------
    th_1 : np.ndarray
        Signals to compute the trac on.
    th_2 : np.ndarray, optional
        Signals to compute the trac against th_1 on.  If not specified, the trac of th_1
        to itself is computed.

    Returns
    -------
    np.ndarray
        Trac values for each signal or pair of signals, with the shape of ``th_1`` minus
        its last axis.

    Unit Tests
    ----------
    """
    if th_2 is None:
        th_2 = th_1
    th_1_original_shape = th_1.shape
    th_1_flattened = th_1.reshape(-1, th_1.shape[-1])
    th_2_flattened = th_2.reshape(-1, th_2.shape[-1])
    numerator = np.abs(np.sum(th_1_flattened * th_2_flattened.conj(), axis=-1)) ** 2
    denominator = np.sum(th_1_flattened * th_1_flattened.conj(), axis=-1) * np.sum(
        th_2_flattened * th_2_flattened.conj(), axis=-1
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        trac_val = numerator / denominator
    trac_val = np.where(denominator == 0, 1.0, trac_val)
    return trac_val.reshape(th_1_original_shape[:-1])


def moving_sum(signal: np.ndarray, n: int) -> np.ndarray:
    """
    Computes a moving sum of the specified number of items.

    Parameters
    ----------
    signal : np.ndarray
        The signal(s) to compute the moving sum on. The sum is computed along the last
        axis.
    n : int
        The number of items to use in the moving sum.

    Returns
    -------
    np.ndarray
        The moving sum computed at each time step in the signal. The last axis has
        length ``signal.shape[-1] - n + 1``.

    Unit Tests
    ----------
    """
    return_value = np.cumsum(signal, axis=-1)
    return_value[..., n:] = return_value[..., n:] - return_value[..., :-n]
    return return_value[..., n - 1 :]


def corr_norm_signal_spec(signal: np.ndarray, specification: np.ndarray) -> np.ndarray:
    """
    Computes correlation weighted by the norm of the signals.

    Parameters
    ----------
    signal : np.ndarray
        The signal to compute the correlation on.
    specification : np.ndarray
        The signal to compute the correlation against.

    Returns
    -------
    np.ndarray
        The weighted correlation signal

    Unit Tests
    ----------
    """
    correlation = sig.correlate(signal, specification, mode="valid").squeeze()
    norm_specification = np.linalg.norm(specification)
    norm_signal = np.sqrt(
        np.sum(moving_sum(signal**2, specification.shape[-1]), axis=0)
    )
    norm_signal[norm_signal == 0] = 1e14
    return correlation / norm_specification / norm_signal


def corr_norm_spec2(signal: np.ndarray, specification: np.ndarray) -> np.ndarray:
    """
    Computes correlation weighted by the norm of the specification signal.

    Parameters
    ----------
    signal : np.ndarray
        The signal to compute the correlation on.
    specification : np.ndarray
        The signal to compute the correlation against.

    Returns
    -------
    np.ndarray
        The weighted correlation signal.

    Unit Tests
    ----------
    """
    correlation = sig.correlate(signal, specification, mode="valid").squeeze()
    norm_specification = np.linalg.norm(specification)
    return correlation / norm_specification**2


def norm_ratio(signal: np.ndarray, specification: np.ndarray) -> np.ndarray:
    """
    Computes the ratio of the norms of two signals.

    Parameters
    ----------
    signal : np.ndarray
        The signal to compute the correlation on.
    specification : np.ndarray
        The signal to compute the correlation against.

    Returns
    -------
    np.ndarray
        The norm ratio signal.

    Unit Tests
    ----------
    """
    norm_specification = np.linalg.norm(specification)
    norm_signal = np.sqrt(
        np.sum(moving_sum(signal**2, specification.shape[-1]), axis=0)
    )
    return 1 - np.abs((norm_signal / norm_specification) ** 2 - 1)


def correlation_norm_spec_ratio(
    signal: np.ndarray, specification: np.ndarray
) -> np.ndarray:
    """
    Computes correlation weighted by the ratio of the norms of the signals.

    Parameters
    ----------
    signal : np.ndarray
        The signal to compute the correlation on.
    specification : np.ndarray
        The signal to compute the correlation against.

    Returns
    -------
    np.ndarray
        The weighted correlation signal.

    Unit Tests
    ----------
    """
    correlation = sig.correlate(signal, specification, mode="valid").squeeze()
    norm_specification = np.linalg.norm(specification)
    norm_signal = np.sqrt(
        np.sum(moving_sum(signal**2, specification.shape[-1]), axis=0)
    )
    return correlation / norm_specification**2 - abs(
        1 - (norm_signal / norm_specification) ** 2
    )


def correlation_norm_signal_spec_ratio(
    signal: np.ndarray, specification: np.ndarray
) -> np.ndarray:
    """
    Computes correlation weighted by the ratio of the norms of the signals.

    Parameters
    ----------
    signal : np.ndarray
        The signal to compute the correlation on.
    specification : np.ndarray
        The signal to compute the correlation against.

    Returns
    -------
    np.ndarray
        The weighted correlation signal.

    Unit Tests
    ----------
    """
    correlation = sig.correlate(signal, specification, mode="valid").squeeze()
    norm_specification = np.linalg.norm(specification)
    norm_signal = np.sqrt(
        np.sum(moving_sum(signal**2, specification.shape[-1]), axis=0)
    )
    norm_signal_divide = norm_signal.copy()
    norm_signal_divide[norm_signal_divide == 0] = 1e14
    return correlation / norm_specification / norm_signal_divide - abs(
        1 - (norm_signal / norm_specification) ** 2
    )


def align_signals(
    measurement_buffer: np.ndarray,
    specification: np.ndarray,
    correlation_threshold: float = 0.9,
    perform_subsample: bool = True,
    correlation_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> Tuple[np.ndarray | None, int | None, float | None, float]:
    """
    Computes the time shift between two signals in time.

    Parameters
    ----------
    measurement_buffer : np.ndarray
        Signal coming from the measurement.
    specification : np.ndarray
        Signal to align the measurement to.
    correlation_threshold : float, optional
        Threshold for a "good" correlation, by default 0.9.
    perform_subsample : bool, optional
        If True, computes a time shift that could be between samples using the phase of
        the FFT of the signals, by default True.
    correlation_metric : Callable[[np.ndarray, np.ndarray], np.ndarray], optional
        An optional function to use to change the matching criterion, by default A
        simple correlation is used.

    Returns
    -------
    spec_portion_aligned : np.ndarray or None
        The portion of the measurement that lines up with the specification. ``None`` if
        the correlation is below ``correlation_threshold``.
    delay : int or None
        The number of samples into the measurement where the specification starts.
        ``None`` if the correlation is below ``correlation_threshold``.
    mean_phase_slope : float or None
        The slope of the phase computed in the FFT from the subsample alignment.  Will
        be None if subsample matching is not used or the correlation is below
        ``correlation_threshold``.
    found_correlation : float
        The value of the correlation metric used to find the match.

    Unit Tests
    ----------
    """
    if correlation_metric is None:
        maximum_possible_correlation = np.sum(specification**2)
        correlation = (
            sig.correlate(measurement_buffer, specification, mode="valid").squeeze()
            / maximum_possible_correlation
        )
    else:
        correlation = correlation_metric(measurement_buffer, specification)
    correlation = np.atleast_1d(correlation)
    delay = np.argmax(correlation)
    found_correlation = correlation[delay]
    print(f"Max Correlation: {found_correlation}")
    if not (found_correlation >= correlation_threshold):
        return None, None, None, found_correlation
    # np.savez('alignment_debug.npz',measurement_buffer=measurement_buffer,
    #          specification = specification,
    #          correlation_threshold = correlation_threshold)
    specification_portion = measurement_buffer[
        :, delay : delay + specification.shape[-1]
    ]

    if perform_subsample:
        # Compute ffts for subsample alignment
        spec_fft = np.fft.rfft(specification, axis=-1)
        spec_portion_fft = np.fft.rfft(specification_portion, axis=-1)

        with np.errstate(invalid="ignore", divide="ignore"):
            phase_difference = np.angle(spec_portion_fft / spec_fft)
        phase_difference = np.where(spec_fft == 0, np.nan, phase_difference)
        phase_slope = (
            phase_difference[..., 1:-1] / np.arange(phase_difference.shape[-1])[1:-1]
        )
        mean_phase_slope = np.nanmedian(
            phase_slope
        )  # Use Median to discard outliers due to potentially noisy phase

        spec_portion_aligned_fft = spec_portion_fft * np.exp(
            -1j * mean_phase_slope * np.arange(spec_portion_fft.shape[-1])
        )
        spec_portion_aligned = np.fft.irfft(spec_portion_aligned_fft)
    else:
        spec_portion_aligned = specification_portion.copy()
        mean_phase_slope = None
    return spec_portion_aligned, delay, mean_phase_slope, found_correlation


def shift_signal(
    signal: np.ndarray, samples_to_keep: int, sample_delay: int, phase_slope: float
) -> np.ndarray:
    """
    Applies a time shift to a signal by modifying the phase of the FFT.

    Parameters
    ----------
    signal : np.ndarray
        The signal to shift.
    samples_to_keep : int
        The number of samples to keep in the shifted signal.
    sample_delay : int
        The number of samples to delay.
    phase_slope : float
        The slope of the phase if subsample shift is used.

    Returns
    -------
    np.ndarray
        The shifted signal.

    Notes
    -----
    Only used by the SDS and transient environments, with the ``delay`` and
    ``mean_phase_slope`` returned by ``align_signals``.

    Unit Tests
    ----------
    """
    signal_sample_aligned = signal[..., sample_delay : sample_delay + samples_to_keep]
    sample_aligned_fft = np.fft.rfft(signal_sample_aligned, axis=-1)
    subsample_aligned_fft = sample_aligned_fft * np.exp(
        -1j * phase_slope * np.arange(sample_aligned_fft.shape[-1])
    )
    return np.fft.irfft(subsample_aligned_fft)


def wrap(data: float | np.ndarray, period: float = 2 * np.pi) -> float | np.ndarray:
    """
    Wraps angle data into the range ``[-period/2, period/2)``.

    With the default period this is ``[-pi, pi)``.

    Parameters
    ----------
    data : float or np.ndarray
        The angle data to wrap.
    period : float, optional
        The period to wrap over. Defaults to ``2 * pi``.

    Returns
    -------
    float or np.ndarray
        The wrapped data.

    Unit Tests
    ----------
    """
    return (data + period / 2) % period - period / 2


class OverlapBuffer:
    """
    Class to hold a buffer stored in a numpy array.

    This buffer supports overlap; when you pull data out, it doesn't remove the data
    from the buffer.

    New data is added to the end of the buffer along ``buffer_axis``, pushing the oldest
    data off the front. ``buffer_position`` counts how many samples from the end of the
    buffer have not yet been read.

    Unit Tests
    ----------
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        buffer_axis: int = -1,
        starting_value: Any = 0,
        dtype: str | np.dtype = "float64",
    ) -> None:
        """
        Creates a buffer object.

        Parameters
        ----------
        shape : tuple
            Shape of the underlying data array.
        buffer_axis : int, optional
            Index corresponding to the buffer axis. The default is -1.
        starting_value : optional
            Initial value of the array.  Can be any value or array that can be
            broadcast into the shape of the array. The default is 0.
        dtype : numpy dtype, optional
            The data type of the buffer array. The default is 'float64'.

        Unit Tests
        ----------
        """
        self._buffer_data = np.empty(shape, dtype)
        self._buffer_data[:] = starting_value
        self._buffer_axis = (
            buffer_axis % self.buffer_data.ndim
        )  # Makes a positive index
        self._buffer_position = 0

    @property
    def buffer_position(self) -> int:
        """
        The current buffer position.

        Returns
        -------
        int
            Number of samples at the end of the buffer that have not been
            read.

        Unit Tests
        ----------
        """
        return self._buffer_position

    @property
    def buffer_axis(self) -> int:
        """
        The axis of the data that is used as buffer dimension.

        Returns
        -------
        int
            The non-negative index of the buffer axis.

        Unit Tests
        ----------
        """
        return self._buffer_axis

    @property
    def buffer_data(self) -> np.ndarray:
        """
        Gets the data currently on the buffer.

        Returns
        -------
        np.ndarray
            The underlying buffer array.

        Unit Tests
        ----------
        """
        return self._buffer_data

    def add_data_noshift(self, data: np.ndarray) -> None:
        """
        Adds data to the buffer without shifting the buffer.

        If ``data`` is longer than the buffer along ``buffer_axis``, only the most
        recent samples are kept.

        Parameters
        ----------
        data : np.ndarray
            Data to add. Must match the buffer shape on every axis except
            ``buffer_axis``.

        Unit Tests
        ----------
        """
        data = np.array(data)
        # Make sure the data will fit into the buffer
        data_slice = tuple(
            [
                (
                    slice(-self.buffer_data.shape[self.buffer_axis], None)
                    if i == self.buffer_axis
                    else slice(None)
                )
                for i in range(self.buffer_data.ndim)
            ]
        )
        data = data[data_slice]
        # Figure out how much we need to roll the buffer
        new_data_size = data.shape[self.buffer_axis]
        old_data_slice = tuple(
            [
                slice(new_data_size, None) if i == self.buffer_axis else slice(None)
                for i in range(self.buffer_data.ndim)
            ]
        )
        self.buffer_data[:] = np.concatenate(
            (self.buffer_data[old_data_slice], data), axis=self.buffer_axis
        )

    def add_data(self, data: np.ndarray) -> None:
        """
        Adds data to the buffer and shifts the buffer position.

        The buffer position is increased by the number of samples added, up to the size
        of the buffer.

        Parameters
        ----------
        data : np.ndarray
            Data to add. Must match the buffer shape on every axis except
            ``buffer_axis``.

        Unit Tests
        ----------
        """
        self.add_data_noshift(data)
        self._buffer_position += data.shape[self.buffer_axis]
        if self.buffer_position > self.buffer_data.shape[self.buffer_axis]:
            self._buffer_position = self.buffer_data.shape[self.buffer_axis]

    def get_data_noshift(self, num_samples: int) -> np.ndarray:
        """
        Gets data from the buffer without shifting the buffer position.

        Parameters
        ----------
        num_samples : int
            Number of samples to read, starting from the oldest unread sample.

        Returns
        -------
        np.ndarray
            A view of the requested samples.

        Raises
        ------
        ValueError
            If ``num_samples`` is greater than ``buffer_position``.

        Unit Tests
        ----------
        """
        data_start = -self.buffer_position
        data_end = -self.buffer_position + num_samples
        if data_end > 0:
            raise ValueError(
                f"Too many samples requested {num_samples} > "
                f"buffer position of {self.buffer_position}"
            )
        data_slice = tuple(
            [
                (
                    slice(data_start, None if data_end == 0 else data_end)
                    if i == self.buffer_axis
                    else slice(None)
                )
                for i in range(self.buffer_data.ndim)
            ]
        )
        return self.buffer_data[data_slice]

    def get_data(self, num_samples: int, buffer_shift: int | None = None) -> np.ndarray:
        """
        Gets data from the buffer and updates the position.

        Parameters
        ----------
        num_samples : int
            Number of samples to read, starting from the oldest unread sample.
        buffer_shift : int, optional
            Amount to shift the buffer position by after reading. If ``None``, the
            position is moved back by ``num_samples`` so the read samples are consumed.
            A smaller shift leaves samples to be read again, giving overlap. Defaults to
            ``None``.

        Returns
        -------
        np.ndarray
            A view of the requested samples.

        Raises
        ------
        ValueError
            If ``num_samples`` is greater than ``buffer_position``.

        Unit Tests
        ----------
        """
        data = self.get_data_noshift(num_samples)
        if buffer_shift is None:
            self.shift_buffer_position(-num_samples)
        else:
            self.shift_buffer_position(buffer_shift)
        return data

    def shift_buffer_position(self, samples: int) -> None:
        """
        Moves the buffer positions.

        The result is clamped between 0 and the size of the buffer.

        Parameters
        ----------
        samples : int
            Number of samples to move the position by. Negative values mark samples as
            read.

        Unit Tests
        ----------
        """
        self._buffer_position += samples
        if self._buffer_position < 0:
            self._buffer_position = 0
        if self._buffer_position > self.buffer_data.shape[self.buffer_axis]:
            self._buffer_position = self.buffer_data.shape[self.buffer_axis]

    def set_buffer_position(self, position: int = 0) -> None:
        """
        Sets the buffer positions.

        The result is clamped between 0 and the size of the buffer.

        Parameters
        ----------
        position : int, optional
            The new buffer position. Defaults to 0.

        Unit Tests
        ----------
        """
        self._buffer_position = position
        if self._buffer_position < 0:
            self._buffer_position = 0
        if self._buffer_position > self.buffer_data.shape[self.buffer_axis]:
            self._buffer_position = self.buffer_data.shape[self.buffer_axis]

    def __getitem__(self, key: Any) -> Any:
        """
        Indexes directly into the underlying buffer array.

        Parameters
        ----------
        key : Any
            Any index accepted by a numpy array.

        Returns
        -------
        Any
            The result of indexing ``buffer_data``.

        Unit Tests
        ----------
        """
        return self.buffer_data[key]

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Gets the shape of the buffer.

        Returns
        -------
        Tuple[int, ...]
            The shape of the underlying buffer array.

        Unit Tests
        ----------
        """
        return self.buffer_data.shape


# endregion
