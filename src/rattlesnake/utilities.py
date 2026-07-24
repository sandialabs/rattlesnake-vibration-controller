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

import importlib.util
import msvcrt
import multiprocessing as mp
import multiprocessing.queues as mpqueue
import multiprocessing.synchronize  # pylint: disable=unused-import
import os
import queue as thqueue
import random
import re
import socket
import string
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from typing import Dict, Tuple

import numpy as np
import requests
import scipy.signal as sig
from scipy.interpolate import interp1d
from scipy.io import loadmat

# region Global
# Define base directory
this_path = os.path.split(__file__)[0]
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    DIRECTORY = sys._MEIPASS  # pylint: disable=protected-access
else:
    DIRECTORY = this_path


def open_and_lock_file(log_filename: str, encoding: str = "utf-8"):
    """
    Returns a new file if the filename is not a locked file. Returns None
    if the file is in use by another program
    """
    if not os.path.exists(log_filename):
        open(log_filename, "a", encoding=encoding).close()
    f = open(log_filename, "r+", encoding=encoding)
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.close()
        return None
    f.seek(0)
    f.truncate(0)
    return f


def log_file_task(
    queue: mp.Queue,
    shutdown_event,
):
    """A multiprocessing function that collects logging data and writes to file

    Parameters
    ----------
    queue : mp.queues.Queue
        The multiprocessing queue to collect logging messages from
    """
    f = open_and_lock_file("Rattlesnake.log")
    ind = 0
    while f is None:
        filename = f"Rattlesnake_{ind}.log"
        f = open_and_lock_file(filename)
        ind = ind+1
    with f:
        while not shutdown_event.is_set():
            output = queue.get()
            if " ERROR" in str(output):
                print(output)
            if output == GlobalCommands.QUIT:
                f.write("Program quitting, logging terminated.")
                break
            num_newlines = output.count("\n")
            if num_newlines > 1:
                output = output.replace("\n", "////", num_newlines - 1)
            f.write(output)
            f.flush()


class RattlesnakeError(Exception):
    """Generic exception raised for Rattlesnake-specific error conditions."""


class GlobalCommands(Enum):
    """Enumeration of command codes passed between controller subtasks."""

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
    SAVE_SYSTEM_ID = 24
    LOAD_SYSTEM_ID = 25

    @property
    def label(self):
        """Used by UI as names for"""
        return self.name.replace("_", " ").title()


class VerboseMessageQueue:
    """A queue class that contains automatic logging information"""

    def __init__(self, log_queue, base_queue, base_name: str = "", name_manager=None):
        """
        A queue class that contains automatic logging information

        Parameters
        ----------
        log_queue : mp.queues.Queue :
            A queue that a logging task will read from where the operations of
            the queue will be logged.
        queue_name : str :
            The name of the queue that will be included in the logging information

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
    def log_name(self):
        """The name used to identify this queue in log messages"""
        if self.environment_name:
            env = self.environment_name.value
            return f"{self.base_name} | {env}" if env else self.base_name

        return self.base_name

    def assign_environment(self, env_name: str):
        """Associates this queue with a named environment for logging purposes"""
        self.environment_name.value = env_name

    def generate_message_id(self, size=6, chars=string.ascii_letters + string.digits):
        """Generates a random identifier for log file messages"""
        return "".join(random.choice(chars) for _ in range(size))

    def put(self, task_name, message_data_tuple, *args, **kwargs):
        """Puts data to a verbose queue

        Parameters
        ----------
        task_name : str
            Task name that is performing the put operation
        message_data_tuple : Tuple
            A (message,data) tuple where message is the instruction and data is
            any optional data to be passed along with the instruction.
        *args :
            Additional arguments that will be passed to the mp.queues.Queue.put
            function
        **kwargs :
            Additional arguments that will be passed to the mp.queues.Queue.put
            function

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

    def get(self, task_name, *args, **kwargs):
        """Gets data from a verbose queue

        Parameters
        ----------
        task_name : str :
            Name of the task that is retrieving data from the queue
        *args :
            Additional arguments that will be passed to the mp.queues.Queue.get
            function
        **kwargs :
            Additional arguments that will be passed to the mp.queues.Queue.get
            function


        Returns
        -------
        message_data_tuple :
            A (message,data) tuple

        """
        message_id, message_data_tuple = self.base_queue.get(*args, **kwargs)
        if message_id != "":
            self.log_queue.put(
                f"{datetime.now()}: {task_name} got "
                f"{message_data_tuple[0].name} ({message_id}) from {self.log_name}\n"
            )
        return message_data_tuple

    def flush(self, task_name):
        """Flushes a verbose queue getting all data currently in the queue

        After execution the queue should be empty barring race conditions.

        Parameters
        ----------
        task_name : str :
            Name of the task that is flushing the queue


        Returns
        -------
        data : iterable of message_data_tuples :
            A list of all (message,data) tuples currently in the queue.

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

    def empty(self):
        """Return true if the queue is empty."""
        return self.base_queue.empty()

    def close(self):
        """Closes queue"""
        if hasattr(self.base_queue, "close"):
            self.base_queue.close()

    def join_thread(self):
        """Joins thread"""
        if hasattr(self.base_queue, "join_thread"):
            self.base_queue.join_thread()


class QueueContainer:
    """A container class for the queues that the controller will manage"""

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
    ):
        """A container class for the queues that the controller will manage.

        The controller uses many queues to pass data between the various pieces.
        This class organizes those queues into one common namespace.

        Parameters
        ----------
        controller_command_queue : VerboseMessageQueue
            Queue that is read by the controller for global controller commands
        acquisition_command_queue : VerboseMessageQueue
            Queue that is read by the acquisition subtask for acquisition commands
        output_command_queue : VerboseMessageQueue
            Queue that is read by the output subtask for output commands
        streaming_command_queue : VerboseMessageQueue
            Queue that is read by the streaming subtask for streaming commands
        log_file_queue : mp_queues.Queue
            Queue for putting logging messages that will be read by the logging
            subtask and written to a file.
        input_output_sync_queue : mp_queues.Queue
            Queue that is used to synchronize input and output signals
        single_process_hardware_queue : mp_queues.Queue
            Queue that is used to connect the acquisition and output subtasks
            for hardware implementations that cannot have acquisition and
            output in separate processes.
        gui_update_queue : mp_queues.Queue
            Queue where various subtasks put instructions for updating the
            widgets in the user interface
        environment_command_queues : Dict[str,VerboseMessageQueue]
            A dictionary where the keys are environment names and the values are
            VerboseMessageQueues that connect the main controller to the
            environment subtasks for sending instructions.
        environment_data_in_queues : Dict[str,multiprocessing.queues.Queue]
            A dictionary where the keys are environment names and the values are
            multiprocessing queues that connect the acquisition subtask to the
            environment subtask.  Each environment will retrieve acquired data
            from this queue.
        environment_data_out_queues : Dict[str,multiprocessing.queues.Queue]
            A dictionary where the keys are environment names and the values are
            multiprocessing queues that connect the output subtask to the
            environment subtask.  Each environment will put data that it wants
            the controller to generate in this queue.

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
    """A container class for the multiprocessing events that the controller
    uses to coordinate readiness, shutdown, and activity state across
    subtasks."""

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
    ):
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


def flush_queue(queue, timeout=None):
    """Flushes a queue by getting all the data currently in it.

    Parameters
    ----------
    queue : mp.queues.Queue or VerboseMessageQueue:
        The queue to flush


    Returns
    -------
    data : iterable
        A list of all data that were in the queue at flush

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
    gui_update_queue,
    shutdown_event,
    gui_active_check,
    max_queue_size=500,
    poll_interval=0.25,
):
    while not shutdown_event.wait(poll_interval):
        if gui_active_check():
            continue
        try:
            while gui_update_queue.qsize() > max_queue_size:
                gui_update_queue.get_nowait()
        except (thqueue.Empty, mpqueue.Empty):
            continue


_direction_map = {
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
_direction_inv_map = {
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


def autofill_single_ip_address(ip_address):
    """
    Worker function for a single ip_address object.
    Runs the same logic as the original loop, but without touching UI widgets.
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


def search_for_lanxi_devices(timeout):
    """Repeatedly polls the network for B&K LAN-XI devices until timeout

    Parameters
    ----------
    timeout : float
        The number of seconds to search for devices before returning

    Returns
    -------
    list[IPAddress]
        A list of unique IPAddress objects for the LAN-XI devices found
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


def find_lanxi_devices():
    """Scans the ARP table for candidate LAN-XI IP addresses and validates them

    Candidate addresses are extracted from the local ARP table by matching the
    169.254.x.x link-local address range used by LAN-XI devices, then each
    candidate is queried concurrently to confirm it is a valid LAN-XI module.

    Returns
    -------
    list[IPAddress]
        A list of IPAddress objects for the LAN-XI devices found
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


def check_lanxi_candidate(ipv4_address):
    """Queries a candidate IP address to check if it is a valid LAN-XI device

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


class IPAddress:
    """Container for information about IPAddress, mainly used to make
    sure each address has a values for relevant information"""

    def __init__(
        self,
        host_name: str | None = None,
        ipv4_address: str | None = None,
        ipv6_address: str | None = None,
        valid_ip: bool = False,
    ):
        self.host_name = host_name
        self.ipv4_address = ipv4_address
        self.ipv6_address = ipv6_address
        self.valid_ip = valid_ip
        self.module_info = None
        self.sync_type = None
        self.validation_timeout = 5

    def get_ip_from_host_name(self):
        """Resolves ipv4_address and ipv6_address from host_name via DNS lookup"""
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

    def get_host_name_from_ip(self):
        """Queries the device REST API to derive host_name from the IP address"""
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

    def validate(self):
        """Queries the device REST API to confirm the address is reachable and valid"""
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


# endregion


# region Loading
def load_time_history(signal_path, sample_rate):
    """Loads a time history from a given file

    The signal can be loaded from numpy files (.npz, .npy) or matlab files (.mat).
    For .mat and .npz files, the time data can be included in the file in the
    't' field, or it can be excluded and the sample_rate input argument will
    be used.  If time data is specified, it will be linearly interpolated to the
    sample rate of the controller.
    For these file types, the signal should be stored in the 'signal'
    field.  For .npy files, only one array is stored, so it is treated as the
    signal, and the sample_rate input argument is used to construct the time
    data.

    Parameters
    ----------
    signal_path : str:
        Path to the file from which to load the time history

    sample_rate : str:
        The sample rate of the loaded signal.

    Returns
    -------
    signal : np.ndarray:
        A signal loaded from the file

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


def load_csv_matrix(file):
    """Loads a matrix from a CSV file

    Parameters
    ----------
    file : str :
        Path to the file that will be loaded


    Returns
    -------
    data : list[list[str]]
        A 2D nested list of strings containing the matrix in the CSV file.

    """
    with open(file, "r", encoding="utf-8") as f:
        data = []
        for line in f:
            data.append([])
            for v in line.split(","):
                data[-1].append(v.strip())
    return data


def save_csv_matrix(data, file):
    """Saves 2D matrix data to a file

    Parameters
    ----------
    data : 2D iterable of str:
        A 2D nested iterable of strings that will be written to a file
    file : str :
        The path to a file where the data will be written.

    """
    text = "\n".join([",".join(row) for row in data])
    with open(file, "w", encoding="utf-8") as f:
        f.write(text)


def load_python_module(module_path):
    """Loads in the Python file at the specified path as a module at runtime

    Parameters
    ----------
    module_path : str:
        Path to the module to be loaded


    Returns
    -------
    module : module:
        A reference to the loaded module
    """
    _, file = os.path.split(module_path)
    file, _ = os.path.splitext(file)
    spec = importlib.util.spec_from_file_location(file, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worksheet_cell_str(value, default=""):
    """
    This is used to cleanse inputs before writing them to excel as
    the GUI automatically converts everything to a string but the
    headless mode will keep it as ints/floats/etc. which changes how
    openpyxl stores the information to an excel sheet.
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
    worksheet, start_row, num_rows, start_col
):
    """Reads a numeric matrix from a block of cells in an Excel worksheet

    Each row is read starting at start_col until the first blank, comment
    (starting with '#'), or "none" cell is encountered.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        The worksheet to read the matrix from
    start_row : int
        The 1-indexed row where the matrix begins
    num_rows : int
        The number of rows to read
    start_col : int
        The 1-indexed column where each row of the matrix begins

    Returns
    -------
    np.ndarray or None
        The matrix read from the worksheet, or None if the first cell is
        blank or contains "none"
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


# This is really dumb but since modal environment saves netcdf
# internally, this has to be in utilities instead of load_utilities
# for circular import reasons
def save_rattlesnake_to_netcdf(
    netcdf_dataset,
    hardware_metadata=None,
    environment_metadata_dict=None,
):
    """Saves hardware and environment metadata to an open netCDF4 dataset

    Parameters
    ----------
    netcdf_dataset : netCDF4.Dataset
        An open, writable netCDF4 dataset to save the metadata to
    hardware_metadata : HardwareMetadata, optional
        The hardware metadata to save
    environment_metadata_dict : dict, optional
        A dictionary where the keys are environment names and the values are
        the environment metadata objects to save, one per environment
    """
    hardware_metadata.save_metadata_to_netcdf(netcdf_dataset)
    netcdf_dataset.createDimension("num_environments", len(environment_metadata_dict))
    var = netcdf_dataset.createVariable("environment_names", str, ("num_environments",))
    environment_booleans = []
    for i, metadata in enumerate(environment_metadata_dict.values()):
        var[i] = metadata.environment_name
        environment_booleans.append(metadata.channel_list_bools)
    var = netcdf_dataset.createVariable("environment_types", int, ("num_environments",))
    for i, metadata in enumerate(environment_metadata_dict.values()):
        var[i] = metadata.environment_type.value
    var = netcdf_dataset.createVariable(
        "environment_active_channels",
        "i1",
        ("response_channels", "num_environments"),
    )
    var[...] = np.array(environment_booleans, dtype="int8").T
    for environment_metadata in environment_metadata_dict.values():
        group_handle = netcdf_dataset.createGroup(environment_metadata.environment_name)
        environment_metadata.save_metadata_to_netcdf(group_handle)


# endregion


# region Math Operations
def coherence(cpsd_matrix: np.ndarray, row_column: Tuple[int, int] | None = None):
    """Compute coherence from a CPSD matrix

    Parameters
    ----------
    cpsd_matrix : np.ndarray :
        A 3D complex numpy array where the first index corresponds to the
        frequency line and the second and third indices correspond to the rows
        and columns of the matrix.
    row_column : Tuple[int, int] :
        Optional argument to compute the coherence at just a single (row,column)
        pair.  (Default value = Compute Entire Matrix)

    Returns
    -------
    coherence : np.ndarray :
        3D array of coherence values where the [i,j,k] entry corresponds to the
        coherence of the CPSD matrix for the ith frequency line, jth row, and
        kth column.

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


def cpsd_to_time_history(cpsd_matrix, sample_rate, df, output_oversample=1):
    # pylint: disable=invalid-name
    """Generates a time history realization from a CPSD matrix

    Parameters
    ----------
    cpsd_matrix : np.ndarray :
        A 3D complex np.ndarray representing a CPSD matrix where the first
        dimension is the frequency line and the second two dimensions are the
        rows and columns of the matrix at each frequency line.
    sample_rate: float :
        The sample rate of the controller in samples per second
    df : float :
        The frequency spacing of the cpsd matrix


    Returns
    -------
    output : np.ndarray :
        A numpy array containing the generated signals

    Notes
    -----
    Uses the process described in [1]_

    .. [1] R. Schultz and G. Nelson, "Input signal synthesis for open-loop
       multiple-input/multiple-output testing," Proceedings of the International
       Modal Analysis Conference, 2019.

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
    excitation_coordinate: np.ndarray = None,
):
    """Picks out entries in an array based on coordinate strings

    Parameters
    ----------
    array : np.ndarray
        Array to parse
    coordinate : np.ndarray
        Coordinate names associated with the array
    control_coordinate : np.ndarray
        Coordinate names associated with control degrees of freedom
    excitation_coordinate : np.ndarray, optional
        Coordinate names associated with excitation degrees of freedom

    Returns
    -------
    np.ndarray
        An array sorted by the provided coordinate strings

    Raises
    ------
    ValueError
        If requested coordinate strings do not exist in the array strings
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


def db2scale(decibel):
    """Converts a decibel value to a scale factor

    Parameters
    ----------
    decibel : float :
        Value in decibels


    Returns
    -------
    scale : float :
        Value in linear

    """
    return 10 ** (decibel / 20)


def power2db(power):
    """Converts a power quantity to decibels"""
    return 10 * np.log10(power)


def scale2db(scale):
    """Converts a scale quantity to decibels"""
    return 20 * np.log10(scale)


def rms_time(signal, axis=None, keepdims=False):
    """Computes RMS over a time signal

    Parameters
    ----------
    signal : np.ndarray :
        Signal over which to compute the root-mean-square value
    axis : int :
        The dimension over which the mean is performed (Default value = None)
    keepdims : bool :
        Whether to keep the dimension over which mean is computed
        (Default value = False)

    Returns
    -------
    rms : numpy scalar or numpy.ndarray
        The root-mean-square value of signal

    """
    return np.sqrt(np.mean(signal**2, axis=axis, keepdims=keepdims))


def rms_csd(csd, df):
    """Computes RMS of a CPSD matrix

    Parameters
    ----------
    csd : np.ndarray :
        3D complex Numpy array where the first dimension is the frequency line
        and the second two dimensions are the rows and columns of the CPSD
        matrix.
    df : float :
        Frequency spacing of the CPSD matrix

    Returns
    -------
    rms : numpy scalar or numpy.ndarray
        The root-mean-square value of signals in the CPSD matrix

    """
    return np.sqrt(np.einsum("ijj->j", csd).real * df)


def trac(th_1, th_2=None):
    """Computes the time response assurance criterion

    Parameters
    ----------
    th_1 : np.ndarray
        Signals to compute the trac on.
    th_2 : np.ndarray, optional
        Signals to compute the trac against th_1 on.  If not specified, the
        trac of th_1 to itself is computed

    Returns
    -------
    np.ndarray
        Trac values for each signal or pair of signals
    """
    if th_2 is None:
        th_2 = th_1
    th_1_original_shape = th_1.shape
    th_1_flattened = th_1.reshape(-1, th_1.shape[-1])
    th_2_flattened = th_2.reshape(-1, th_2.shape[-1])
    trac_val = np.abs(np.sum(th_1_flattened * th_2_flattened.conj(), axis=-1)) ** 2 / (
        (np.sum(th_1_flattened * th_1_flattened.conj(), axis=-1))
        * np.sum(th_2_flattened * th_2_flattened.conj(), axis=-1)
    )
    return trac_val.reshape(th_1_original_shape[:-1])


def moving_sum(signal, n):
    """Computes a moving sum of the specified number of items

    Parameters
    ----------
    signal : np.ndarray
        The signal(s) to compute the moving sum on
    n : int
        The number of items to use in the moving sum

    Returns
    -------
    np.array
        The moving sum computed at each time step in the signal
    """
    return_value = np.cumsum(signal, axis=-1)
    return_value[..., n:] = return_value[..., n:] - return_value[..., :-n]
    return return_value[..., n - 1 :]


def corr_norm_signal_spec(signal, specification):
    """Computes correlation weighted by the norm of the signals

    Parameters
    ----------
    signal : np.ndarray
        The signal to compute the correlation on
    specification : np.ndarray
        The signal to compute the correlation against

    Returns
    -------
    np.ndarray
        The weighted correlation signal
    """
    correlation = sig.correlate(signal, specification, mode="valid").squeeze()
    norm_specification = np.linalg.norm(specification)
    norm_signal = np.sqrt(
        np.sum(moving_sum(signal**2, specification.shape[-1]), axis=0)
    )
    norm_signal[norm_signal == 0] = 1e14
    return correlation / norm_specification / norm_signal


def corr_norm_spec2(signal, specification):
    """Computes correlation weighted by the norm of the specification signal

    Parameters
    ----------
    signal : np.ndarray
        The signal to compute the correlation on
    specification : np.ndarray
        The signal to compute the correlation against

    Returns
    -------
    np.ndarray
        The weighted correlation signal
    """
    correlation = sig.correlate(signal, specification, mode="valid").squeeze()
    norm_specification = np.linalg.norm(specification)
    return correlation / norm_specification**2


def norm_ratio(signal, specification):
    """Computes the ratio of the norms of two signals

    Parameters
    ----------
    signal : np.ndarray
        The signal to compute the correlation on
    specification : np.ndarray
        The signal to compute the correlation against

    Returns
    -------
    np.ndarray
        The norm ratio signal
    """
    norm_specification = np.linalg.norm(specification)
    norm_signal = np.sqrt(
        np.sum(moving_sum(signal**2, specification.shape[-1]), axis=0)
    )
    return 1 - np.abs((norm_signal / norm_specification) ** 2 - 1)


def correlation_norm_spec_ratio(signal, specification):
    """Computes correlation weighted by the ratio of the norms of the signals

    Parameters
    ----------
    signal : np.ndarray
        The signal to compute the correlation on
    specification : np.ndarray
        The signal to compute the correlation against

    Returns
    -------
    np.ndarray
        The weighted correlation signal
    """
    correlation = sig.correlate(signal, specification, mode="valid").squeeze()
    norm_specification = np.linalg.norm(specification)
    norm_signal = np.sqrt(
        np.sum(moving_sum(signal**2, specification.shape[-1]), axis=0)
    )
    return correlation / norm_specification**2 - abs(
        1 - (norm_signal / norm_specification) ** 2
    )


def correlation_norm_signal_spec_ratio(signal, specification):
    """Computes correlation weighted by the ratio of the norms of the signals

    Parameters
    ----------
    signal : np.ndarray
        The signal to compute the correlation on
    specification : np.ndarray
        The signal to compute the correlation against

    Returns
    -------
    np.ndarray
        The weighted correlation signal
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
    measurement_buffer,
    specification,
    correlation_threshold=0.9,
    perform_subsample=True,
    correlation_metric=None,
):
    """Computes the time shift between two signals in time

    Parameters
    ----------
    measurement_buffer : np.ndarray
        Signal coming from the measurement
    specification : np.ndarray
        Signal to align the measurement to
    correlation_threshold : float, optional
        Threshold for a "good" correlation, by default 0.9
    perform_subsample : bool, optional
        If True, computes a time shift that could be between samples using
        the phase of the FFT of the signals, by default True
    correlation_metric : function, optional
        An optional function to use to change the matching criterion, by
        default A simple correlation is used

    Returns
    -------
    spec_portion_aligned : np.ndarray
        The portion of the measurement that lines up with the specification
    delay : float
        The time difference between the measurement and specification
    mean_phase_slope : float
        The slope of the phase computed in the FFT from the subsample
        alignment.  Will be None if subsample matching is not used
    found_correlation : float
        The value of the correlation metric used to find the match
    """
    if correlation_metric is None:
        maximum_possible_correlation = np.sum(specification**2)
        correlation = (
            sig.correlate(measurement_buffer, specification, mode="valid").squeeze()
            / maximum_possible_correlation
        )
    else:
        correlation = correlation_metric(measurement_buffer, specification)
    delay = np.argmax(correlation)
    found_correlation = correlation[delay]
    print(f"Max Correlation: {found_correlation}")
    if found_correlation < correlation_threshold:
        return None, None, None, None
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

        # Compute phase angle differences for subpixel alignment
        phase_difference = np.angle(spec_portion_fft / spec_fft)
        phase_slope = (
            phase_difference[..., 1:-1] / np.arange(phase_difference.shape[-1])[1:-1]
        )
        mean_phase_slope = np.median(
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


def shift_signal(signal, samples_to_keep, sample_delay, phase_slope):
    """Applies a time shift to a signal by modifying the phase of the FFT

    Parameters
    ----------
    signal : np.ndarray
        The signal to shift
    samples_to_keep : int
        The number of samples to keep in the shifted signal
    sample_delay : int
        The number of samples to delay
    phase_slope : float
        The slope of the phase if subsample shift is used

    Returns
    -------
    np.ndarray
        The shifted signal
    """
    signal_sample_aligned = signal[..., sample_delay : sample_delay + samples_to_keep]
    sample_aligned_fft = np.fft.rfft(signal_sample_aligned, axis=-1)
    subsample_aligned_fft = sample_aligned_fft * np.exp(
        -1j * phase_slope * np.arange(sample_aligned_fft.shape[-1])
    )
    return np.fft.irfft(subsample_aligned_fft)


def wrap(data, period=2 * np.pi):
    """Wraps angle data between -pi/2 and pi/2"""
    return (data + period / 2) % period - period / 2


class OverlapBuffer:
    """Class to hold a buffer stored in a numpy array.

    This buffer supports overlap; when you pull data out, it doesn't remove the
    data from the buffer."""

    def __init__(self, shape, buffer_axis=-1, starting_value=0, dtype="float64"):
        """
        Creates a buffer object

        Parameters
        ----------
        shape : tuple
            Shape of the underlying data array
        buffer_axis : int, optional
            Index corresponding to the buffer axis. The default is -1.
        starting_value : optional
            Initial value of the array.  Can be any value or array that can be
            broadcast into the shape of the array. The default is 0.
        dtype : numpy dtype, optional
            The data type of the buffer array. The default is 'float64'.
        """
        self._buffer_data = np.empty(shape, dtype)
        self._buffer_data[:] = starting_value
        self._buffer_axis = (
            buffer_axis % self.buffer_data.ndim
        )  # Makes a positive index
        self._buffer_position = 0

    @property
    def buffer_position(self):
        """The current buffer position"""
        return self._buffer_position

    @property
    def buffer_axis(self):
        """The axis of the data that is used as buffer dimension"""
        return self._buffer_axis

    @property
    def buffer_data(self):
        """Gets the data currently on the buffer"""
        return self._buffer_data

    def add_data_noshift(self, data):
        """Adds data to the buffer without shifting the buffer"""
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

    def add_data(self, data):
        """Adds data to the buffer and shifts the buffer position"""
        self.add_data_noshift(data)
        self._buffer_position += data.shape[self.buffer_axis]
        if self.buffer_position > self.buffer_data.shape[self.buffer_axis]:
            self._buffer_position = self.buffer_data.shape[self.buffer_axis]

    def get_data_noshift(self, num_samples):
        """Gets data from the buffer without shifting the buffer position"""
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

    def get_data(self, num_samples, buffer_shift=None):
        """Gets data from the buffer and updates the position"""
        data = self.get_data_noshift(num_samples)
        if buffer_shift is None:
            self.shift_buffer_position(-num_samples)
        else:
            self.shift_buffer_position(buffer_shift)
        return data

    def shift_buffer_position(self, samples):
        """Moves the buffer positions"""
        self._buffer_position += samples
        if self._buffer_position < 0:
            self._buffer_position = 0
        if self._buffer_position > self.buffer_data.shape[self.buffer_axis]:
            self._buffer_position = self.buffer_data.shape[self.buffer_axis]

    def set_buffer_position(self, position=0):
        """Sets the buffer positions"""
        self._buffer_position = position
        if self._buffer_position < 0:
            self._buffer_position = 0
        if self._buffer_position > self.buffer_data.shape[self.buffer_axis]:
            self._buffer_position = self.buffer_data.shape[self.buffer_axis]

    def __getitem__(self, key):
        return self.buffer_data[key]

    @property
    def shape(self):
        """Gets the shape of the buffer"""
        return self.buffer_data.shape


# endregion
