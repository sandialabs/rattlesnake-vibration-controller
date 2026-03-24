# -*- coding: utf-8 -*-
"""
Implementation for the HBK LAN-XI Hardware

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

import multiprocessing as mp
import re
import socket
import time
from typing import List

import numpy as np
import requests

from rattlesnake.hardware.abstract_hardware import HardwareAcquisition, HardwareOutput
from rattlesnake.hardware.lanxi_stream import OpenapiHeader, OpenapiMessage
from rattlesnake.utilities import Channel, DataAcquisitionParameters

OUTPUT_RATE = 131072
LANXI_STATE_TIMEOUT = 255.0
LANXI_STATE_REPORT = 10
VALID_FILTERS = ["DC", "0.7 Hz", "7.0 Hz", "22.4 Hz", "Intensity"]
VALID_RANGES = ["0.316", "1", "10", "31.6"]
HEADER_LENGTH = 28
BUFFER_SIZE = 2.5

LANXI_STATE_SHUTDOWN = {
    "RecorderRecording": "/rest/rec/measurements/stop",
    "RecorderStreaming": "/rest/rec/finish",
    "RecorderOpened": "/rest/rec/close",
    "Idle": None,
}

IPV4_PATTERN = r"^((25[0-5]|(2[0-4]|1[0-9]|[1-9]|)[0-9])(\.(?!$)|$)){4}$"
IPV6_PATTERN = r"\[\s*([0-9a-fA-F]{1,4}:){0,7}(:[0-9a-fA-F]{1,4})*%?\d*\s*\]"

# TO DO List
# TODO Get responses each time a get or put is done so we know if it was successful
# TODO Shut down the data acquisition more quickly


class LanXIError(Exception):
    """Exception to signify an error using LAN-XI"""


def read_lanxi(socket_handle: socket.socket):
    """
    Reads and interprets data from a Lan-XI

    Parameters
    ----------
    socket_handle : socket.socket
        Handle to the socket that the communication is happening over.

    Returns
    -------
    message_type : OpenapiStream.Header.EMessageType
        Enum determining what the data type is
    data : np.ndarray or dict
        The data read from the device.  Will be a np.ndarray for a signal and
        a dictionary for a interpretation

    """
    # print('Reading data from {:}:{:}'.format(*socket_handle.getpeername()))
    data = socket_handle.recv(HEADER_LENGTH, socket.MSG_WAITALL)
    if len(data) == 0:
        raise LanXIError("Socket is not connected anymore")
    wstream = OpenapiHeader.from_bytes(data)
    content_length = wstream.message_length + HEADER_LENGTH
    # We use the header's content_length to collect the rest of a package
    while len(data) < content_length:
        packet = socket_handle.recv(content_length - len(data))
        data += packet
    # Now we parse the package
    try:
        package = OpenapiMessage.from_bytes(data)
    except EOFError as e:
        print(f"Data Invalid {data}")
        raise e
    if package.header.message_type == OpenapiMessage.Header.EMessageType.e_interpretation:
        interpretation_dict = {}
        for interpretation in package.message.interpretations:
            interpretation_dict[interpretation.descriptor_type] = interpretation.value
        return (package.header.message_type, interpretation_dict)
    elif (
        package.header.message_type == OpenapiMessage.Header.EMessageType.e_signal_data
    ):  # If the data contains signal data
        array = []
        for signal in package.message.signals:  # For each signal in the package
            array.append(np.array([x.calc_value for x in signal.values]) / 2**23)
        return package.header.message_type, np.concatenate(array, axis=-1)
    # If 'quality data' message, then record information on data quality issues
    elif package.header.message_type == OpenapiMessage.Header.EMessageType.e_data_quality:
        ip, port = socket_handle.getpeername()
        for q in package.message.qualities:
            if q.validity_flags.overload:
                print(f"Overload Detected on {ip}:{port}")
            if q.validity_flags.invalid:
                print(f"Invalid Data Detected on {ip}:{port}")
            if q.validity_flags.overrun:
                print(f"Overrun Detected on {ip}:{port}")
        return None, None
    else:
        raise LanXIError(f"Unknown Message Type: {package.header.message_type}")


def lanxi_multisocket_reader(
    socket_handles: List[socket.socket],
    active_channels_list: List[int],
    data_queues: List[mp.queues.Queue],
):
    """
    Reads data from all channels on multiple modules.

    This function is designed to be run by a multiprocessing Process

    Parameters
    ----------
    socket_handles : List[socket.socket]
        A list of the sockets for the cards on this module.
    active_channels_list : List[int]
        A list of number of active channels on each module.
    data_queues : List[mp.queues.Queue]
        A set of queues to pass data back to the main process.

    """
    print(
        "Starting to record from:\n  {:}".format(
            "\n  ".join(
                ["{:}:{:}".format(*socket_handle.getpeername()) for socket_handle in socket_handles]
            )
        )
    )
    try:
        while True:
            for socket_handle, active_channels, data_queue in zip(
                socket_handles, active_channels_list, data_queues
            ):
                socket_data = []
                socket_data_types = []
                while len(socket_data) < active_channels:
                    message_type, data = read_lanxi(socket_handle)
                    # print('Reading {:}:{:} Data Type {:}'.format(
                    #       *socket_handle.getpeername(),message_type))
                    if message_type is not None:
                        socket_data.append(data)
                        socket_data_types.append(message_type)
                # Make sure they are all the same type
                assert all([data_type == socket_data_types[0] for data_type in socket_data_types])
                if socket_data_types[0] == OpenapiMessage.Header.EMessageType.e_interpretation:
                    # print('{:}:{:} Putting Interpretation to Queue'.format(
                    #       *socket_handle.getpeername()))
                    data_queue.put(("Interpretation", socket_data))
                elif socket_data_types[0] == OpenapiMessage.Header.EMessageType.e_signal_data:
                    # print('{:}:{:} Putting Signal to Queue'.format(
                    #        *socket_handle.getpeername()))
                    data_queue.put(("Signal", socket_data))
                else:
                    raise ValueError(
                        "Unknown Signal Type {:} in {:}:{:}".format(  # pylint: disable=consider-using-f-string
                            socket_data_types[0], *socket_handle.getpeername()
                        )
                    )
    except LanXIError:
        for socket_handle, data_queue in zip(socket_handles, data_queues):
            # The socket has closed, so gracefully close down
            ip, port = socket_handle.getpeername()
            print(f"Closing Socket {ip}:{port}")
            while True:
                try:
                    print(f"Emptying Queue {ip}:{port}")
                    data_queue.get(False)
                except mp.queues.Empty:
                    print(f"Returning {ip}:{port}")
                    break
        return


def create_harware_maps(acquisition_map, output_map, channel_list):
    """Creates mapping between the LAN-XI channels and the rattlesnake channel list

    Parameters
    ----------
    acquisition_map : dict
        A dictionary that will be populated with the acquisition map information.  The dictionary
        will be nested, with the first key being the physical device and the second key being the
        physical channel.  The value will be a tuple containing the channel index and the Channel
        itself.
    output_map : dict
        A dictionary that will be populated with the acquisition map information.  The dictionary
        will be nested, with the first key being the feedback device and the second key being the
        feedback channel.  The value will be a tuple containing the channel index and the Channel
        itself.
    channel_list : list of Channel objects
        A list of channels in the rattlesnake test
    """
    for i, channel in enumerate(channel_list):
        if channel.physical_device not in acquisition_map:
            acquisition_map[channel.physical_device] = {}
        acquisition_map[channel.physical_device][int(channel.physical_channel)] = (
            i,
            channel,
        )
    for i, channel in enumerate(
        [channel for channel in channel_list if channel.feedback_device is not None]
    ):
        if channel.feedback_device not in output_map:
            output_map[channel.feedback_device] = {}
        output_map[channel.feedback_device][int(channel.feedback_channel)] = i, channel


def wait_for_ptp_state(host: str, state: str):
    """Waits until hardware is at a current state

    Parameters
    ----------
    host : str
        The address of the host to wait for
    state : str
        The name of the state to wait until.

    Returns
    -------
    bool
        True if the state has changed, False if the hardware timed out.

    """
    start_time = time.time()
    current_state = ""
    iteration = 0
    while True:
        response = requests.get("http://" + host + "/rest/rec/onchange", timeout=60)
        state_data = response.json()
        current_state = state_data["ptpStatus"]
        if current_state == state:
            result = True
            break
        if time.time() - start_time > LANXI_STATE_TIMEOUT:
            result = False
            break
        time.sleep(1)
        iteration += 1
        if iteration % LANXI_STATE_REPORT == 0:
            print(f"Host {host} at {current_state} state, waiting for {state}")
    if not result:
        raise LanXIError(
            f"Wait for PTP State {state} timed out on host {host}.  Last retrieved "
            f"state: {current_state}"
        )
    return result


def wait_for_recorder_state(host: str, state: str):
    """Waits until hardware is at a current state

    Parameters
    ----------
    host : str
        The address of the host to wait for
    state : str
        The name of the state to wait until.

    Returns
    -------
    bool
        True if the state has changed, False if the hardware timed out.

    """
    start_time = time.time()
    current_state = ""
    iteration = 0
    while True:
        response = requests.get("http://" + host + "/rest/rec/onchange", timeout=60)
        state_data = response.json()
        current_state = state_data["moduleState"]
        if current_state == state:
            result = True
            break
        if current_state == "PostFailed":
            result = False
            break
        if time.time() - start_time > LANXI_STATE_TIMEOUT:
            result = False
            break
        time.sleep(1)
        iteration += 1
        if iteration % LANXI_STATE_REPORT == 0:
            print(f"Host {host} at {current_state} state, waiting for {state}")
    if not result:
        raise LanXIError(
            f"Wait for Recorder State {state} timed out on host {host}.  Last retrieved "
            f"state: {current_state}"
        )
    return result


def wait_for_input_state(host: str, state: str):
    """Waits until hardware is at a current state

    Parameters
    ----------
    host : str
        The address of the host to wait for
    state : str
        The name of the state to wait until.

    Returns
    -------
    bool
        True if the state has changed, False if the hardware timed out.

    """
    start_time = time.time()
    current_state = ""
    iteration = 0
    while True:
        response = requests.get("http://" + host + "/rest/rec/onchange", timeout=60)
        state_data = response.json()
        current_state = state_data["inputStatus"]
        if current_state == state:
            result = True
            break
        if time.time() - start_time > LANXI_STATE_TIMEOUT:
            result = False
            break
        time.sleep(1)
        iteration += 1
        if iteration % LANXI_STATE_REPORT == 0:
            print(f"Host {host} at {current_state} state, waiting for {state}")
    if not result:
        raise LanXIError(
            f"Wait for Input State {state} timed out on host {host}.  Last retrieved state: "
            f"{current_state}"
        )
    return result


def close_recorder(host):
    """Closes the host based on its current state"""
    response = requests.get("http://" + host + "/rest/rec/onchange", timeout=60)
    state_data = response.json()
    current_state = state_data["moduleState"]
    if current_state == "RecorderRecording":
        print(f"Stopping Measurement on {host}")
        requests.put("http://" + host + "/rest/rec/measurements/stop", timeout=60)
        wait_for_recorder_state(host, "RecorderStreaming")
        close_recorder(host)
    elif current_state == "RecorderConfiguring":
        response = requests.get("http://" + host + "/rest/rec/channels/input/default", timeout=60)
        channel_settings = response.json()
        response = requests.put(
            "http://" + host + "/rest/rec/channels/input", json=channel_settings, timeout=60
        )
        wait_for_recorder_state(host, "RecorderStreaming")
        close_recorder(host)
    elif current_state == "RecorderStreaming":
        print(f"Finishing Streaming on {host}")
        requests.put("http://" + host + "/rest/rec/finish", timeout=60)
        wait_for_recorder_state(host, "RecorderOpened")
        close_recorder(host)
    elif current_state == "RecorderOpened":
        print(f"Closing Recorder on {host}")
        requests.put("http://" + host + "/rest/rec/close", timeout=60)
        wait_for_recorder_state(host, "Idle")
        close_recorder(host)
    elif current_state == "Idle":
        print(f"Recorder {host} Idle")
    else:
        raise LanXIError(f"Unknown State {current_state} on {host}")
    return


# region: Acqusition
class LanXIAcquisition(HardwareAcquisition):
    """Class defining the interface between LAN-XI acquisition and the controller

    This class defines the interfaces between the controller and the
    data acquisition portion of the hardware.  It is run by the Acquisition
    process, and must define how to get data from the test hardware into the
    controller."""

    def __init__(self, maximum_processes):
        """
        Constructs the LAN-XI Acquisition class and specifies values to null.
        """
        self.acquisition_map = {}
        self.output_map = {}
        self.sockets = {}
        self.processes = {}
        self.process_data_queues = {}
        self.interpretations = None
        self.master_address = None
        self.slave_addresses = set([])
        self.samples_per_read = None
        self.last_acquisition_time = None
        self.maximum_processes = maximum_processes
        self.modules_per_process = None
        self.total_processes = None
        self.acquisition_delay = None

    # region: Abstract Methods
    def set_up_data_acquisition_parameters_and_channels(
        self, test_data: DataAcquisitionParameters, channel_data: List[Channel]
    ):
        """
        Initialize the hardware and set up channels and sampling properties

        The function must create channels on the hardware corresponding to
        the channels in the test.  It must also set the sampling rates.

        Parameters
        ----------
        test_data : DataAcquisitionParameters :
            A container containing the data acquisition parameters for the
            controller set by the user.
        channel_data : List[Channel] :
            A list of ``Channel`` objects defining the channels in the test
        Returns
        -------
        None.

        """
        # Now create a hardware map that will help us do bookkeeping
        create_harware_maps(self.acquisition_map, self.output_map, channel_data)
        # Go through the channel table and get the hardware and channel
        # information
        host_addresses = [channel.physical_device for channel in channel_data]
        host_addresses += [
            channel.feedback_device
            for channel in channel_data
            if (
                not (channel.feedback_device is None)
                and not (channel.feedback_device.strip() == "")
            )
        ]
        self.master_address = host_addresses[0]
        self.slave_addresses = set(
            [address for address in host_addresses if not address == self.master_address]
        )
        self.samples_per_read = test_data.samples_per_read
        modules_per_process_floor = len(self.acquisition_map) // self.maximum_processes
        modules_per_process_remainder = len(self.acquisition_map) % self.maximum_processes
        if modules_per_process_remainder == 0:
            self.modules_per_process = modules_per_process_floor
        else:
            self.modules_per_process = modules_per_process_floor + 1
        self.total_processes = (len(self.acquisition_map) // self.modules_per_process) + (
            0 if len(self.acquisition_map) % self.modules_per_process == 0 else 1
        )
        self.acquisition_delay = (
            (BUFFER_SIZE + 2) * test_data.samples_per_write / test_data.output_oversample
        )

    def start(self):
        """Method to start acquiring data from the hardware"""
        self.sockets = {}
        self.processes = {}
        self.process_data_queues = {}
        # Apply the trigger for multi-frame acquisition
        if len(set(self.acquisition_map) | set(self.output_map)) > 1:
            requests.put("http://" + self.master_address + "/rest/rec/apply", timeout=60)
        # Collect the sockets
        for acquisition_device in self.acquisition_map:
            response = requests.get(
                "http://" + acquisition_device + "/rest/rec/destination/socket", timeout=60
            )
            port = response.json()["tcpPort"]
            # Connect to the socket
            is_ipv4 = re.search(IPV4_PATTERN, acquisition_device) is not None
            is_ipv6 = re.search(IPV6_PATTERN, acquisition_device) is not None
            if is_ipv4:
                self.sockets[acquisition_device] = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP
                )
            elif is_ipv6:
                self.sockets[acquisition_device] = socket.socket(
                    socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP
                )
            else:  # This will crash but is fixed in overhaul version so...
                self.sockets[acquisition_device] = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP
                )
            self.sockets[acquisition_device].connect((acquisition_device, port))
        for slave_address in self.slave_addresses:
            if slave_address in self.acquisition_map:
                requests.post("http://" + slave_address + "/rest/rec/measurements", timeout=60)
        if self.master_address in self.acquisition_map:
            requests.post("http://" + self.master_address + "/rest/rec/measurements", timeout=60)
        print("Started Measurements")
        # Wait for the module state to be recorder streaming
        for slave_address in self.slave_addresses:
            if slave_address in self.acquisition_map:
                wait_for_recorder_state(slave_address, "RecorderRecording")
        if self.master_address in self.acquisition_map:
            wait_for_recorder_state(self.master_address, "RecorderRecording")

        # Here we need to start the processes
        # Split it up into multiple processes
        socket_handles = []
        active_channels_list = []
        data_queues = []
        for acquisition_device, channel_dict in self.acquisition_map.items():
            self.process_data_queues[acquisition_device] = mp.Queue()
            active_channels = len(channel_dict)

            socket_handles.append(self.sockets[acquisition_device])
            active_channels_list.append(active_channels)
            data_queues.append(self.process_data_queues[acquisition_device])

            if len(socket_handles) % self.modules_per_process == 0:
                self.processes[acquisition_device] = mp.Process(
                    target=lanxi_multisocket_reader,
                    args=(socket_handles, active_channels_list, data_queues),
                )
                self.processes[acquisition_device].start()
                socket_handles = []
                active_channels_list = []
                data_queues = []
        if len(socket_handles) > 0:
            self.processes[acquisition_device] = mp.Process(
                target=lanxi_multisocket_reader,
                args=(socket_handles, active_channels_list, data_queues),
            )
            self.processes[acquisition_device].start()
            socket_handles = []
            active_channels_list = []
            data_queues = []

    def get_acquisition_delay(self) -> int:
        """
        Get the number of samples between output and acquisition.

        This function returns the number of samples that need to be read to
        ensure that the last output is read by the acquisition.  If there is
        buffering in the output, this delay should be adjusted accordingly.

        Returns
        -------
        int
            Number of samples between when a dataset is written to the output
            and when it has finished playing.

        """
        return self.acquisition_delay

    def read(self):
        """Method to read a frame of data from the hardware"""
        samples = 0
        full_read_data = []
        while samples < self.samples_per_read:
            read_data = []
            acquisition_indices = []
            for acquisition_device, _ in self.process_data_queues.items():
                # We are going to loop until we get a signal which should be every time except the
                # first, which will pass the interpretation.
                while True:
                    # Get the data from the queues
                    # print('Reading from queue')
                    data_type, data = self.process_data_queues[acquisition_device].get()
                    if data_type == "Interpretation":
                        if self.interpretations is None:
                            self.interpretations = {}
                        self.interpretations[acquisition_device] = data  # Store the interpretation
                    elif data_type == "Signal":
                        for signal, channel_number, interpretation in zip(
                            data,
                            sorted(self.acquisition_map[acquisition_device]),
                            self.interpretations[acquisition_device],
                        ):
                            acquisition_index, _ = self.acquisition_map[acquisition_device][
                                channel_number
                            ]
                            array = (
                                signal
                                * interpretation[
                                    OpenapiMessage.Interpretation.EDescriptorType.scale_factor
                                ]  # This is the scale factor
                                + interpretation[
                                    OpenapiMessage.Interpretation.EDescriptorType.offset
                                ]  # This is the offset
                            )
                            read_data.append(array)
                            acquisition_indices.append(acquisition_index)
                        break  # Exit the loop because we found the signal
            # Check if all the data are the same length
            index_map = np.empty(len(acquisition_indices), dtype=int)
            index_map[acquisition_indices] = np.arange(len(acquisition_indices))
            read_data = np.array(read_data)[index_map]
            full_read_data.append(read_data)
            samples += read_data.shape[-1]
        full_read_data = np.concatenate(full_read_data, axis=-1)
        current_time = time.time()
        if self.last_acquisition_time is not None:
            dtime = current_time - self.last_acquisition_time
            print(f"Took {dtime:0.4f}s to read {full_read_data.shape[-1]} samples")
        self.last_acquisition_time = current_time
        return full_read_data

    def read_remaining(self):
        """Method to read the rest of the data on the acquisition from the hardware"""
        return np.zeros(
            (
                sum(
                    [
                        1
                        for acquisition_device, acquisition_dict in self.acquisition_map.items()
                        for channel_number in acquisition_dict
                    ]
                ),
                1,
            )
        )

    def stop(self):
        """Method to stop the acquisition"""
        if self.master_address in self.acquisition_map:
            requests.put(
                "http://" + self.master_address + "/rest/rec/measurements/stop", timeout=60
            )
        for slave_address in self.slave_addresses:
            if slave_address in self.acquisition_map:
                requests.put("http://" + slave_address + "/rest/rec/measurements/stop", timeout=60)
        # Wait for the module state to be recorder streaming
        for slave_address in self.slave_addresses:
            if slave_address in self.acquisition_map:
                wait_for_recorder_state(slave_address, "RecorderStreaming")
        if self.master_address in self.acquisition_map:
            wait_for_recorder_state(self.master_address, "RecorderStreaming")
        # Join the processes
        for acquisition_device, process in self.processes.items():
            print(f"Recovering process {acquisition_device}")
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
            print(f"Process {acquisition_device} recovered")
        print("All processes recovered, ready for next acquire.")
        self.processes = {}
        self.process_data_queues = {}
        self.interpretations = None
        self.last_acquisition_time = None

    def close(self):
        """Method to close down the hardware"""
        if len(self.processes) > 0:  # This means we are still running!
            self.stop()

    # region: Functions
    def _get_states(self):
        for host in list(self.slave_addresses) + [self.master_address]:
            response = requests.get("http://" + host + "/rest/rec/onchange", timeout=60)
            state_data = response.json()
            print(
                f"Host {host}: Recorder State {state_data['moduleState']}, Input State "
                f"{state_data['inputStatus']}, PTP State {state_data['ptpStatus']}, Recording Mode"
            )

    def _reboot_all(self):
        for host in list(self.slave_addresses) + [self.master_address]:
            requests.put("http://" + host + "/rest/rec/reboot", timeout=60)


# region: Output
class LanXIOutput(HardwareOutput):
    """Abstract class defining the interface between the controller and output

    This class defines the interfaces between the controller and the
    output or source portion of the hardware.  It is run by the Output
    process, and must define how to get write data to the hardware from the
    control system"""

    def __init__(self, maximum_processes):
        """Method to start up the hardware"""
        self.sockets = {}
        self.acquisition_map = {}
        self.output_map = {}
        self.master_address = None
        self.slave_addresses = set([])
        self.oversample_factor = None
        self.output_rate = None
        self.bandwidth_string = None
        self.transfer_size = 4096 * 4
        self.sample_rate = None
        self.write_size = None
        self.empty_time = 0.0
        self.generator_sample_rate = None
        self.buffer_size = 5
        self.ready_signal_factor = BUFFER_SIZE
        self.maximum_processes = maximum_processes

    # region: Abstract Methods
    def set_up_data_output_parameters_and_channels(
        self, test_data: DataAcquisitionParameters, channel_data: List[Channel]
    ):
        # Create a hardware map that will help us do bookkeeping
        create_harware_maps(self.acquisition_map, self.output_map, channel_data)
        self.write_size = test_data.samples_per_write
        self.sample_rate = test_data.sample_rate
        # Go through the channel table and get the hardware and channel
        # information
        host_addresses = [channel.physical_device for channel in channel_data]
        host_addresses += [
            channel.feedback_device
            for channel in channel_data
            if (
                not (channel.feedback_device is None)
                and not (channel.feedback_device.strip() == "")
            )
        ]
        self.master_address = host_addresses[0]
        self.slave_addresses = set(
            [address for address in host_addresses if not address == self.master_address]
        )
        print("\nInitial States:")
        self._get_states()

        # time.sleep(10)

        # Close all devices to start from scratch
        print("Resetting Data Acquisition System")
        self.close(reboot=False)

        # time.sleep(10)

        # If there are any slave addresses, need to perform PTP sync
        if len(self.slave_addresses) > 0:
            print("PTP Mode")
            master_json = {
                "synchronization": {
                    "mode": "ptp",
                    "domain": 42,
                    "preferredMaster": True,
                }
            }
            requests.put(
                "http://" + self.master_address + "/rest/rec/syncmode", json=master_json, timeout=60
            )
            slave_json = {
                "synchronization": {
                    "mode": "ptp",
                    "domain": 42,
                    "preferredMaster": False,
                }
            }
            for slave_address in self.slave_addresses:
                requests.put(
                    "http://" + slave_address + "/rest/rec/syncmode", json=slave_json, timeout=60
                )
            print("Waiting for PTP Sync...")
            # Wait until PTP locks
            for slave_address in self.slave_addresses:
                wait_for_ptp_state(slave_address, "Locked")
            wait_for_ptp_state(self.master_address, "Locked")
            print("PTP Synced!")
            single_module = False
        else:
            print("Single Module Mode")
            master_json = {"synchronization": {"mode": "stand-alone"}}
            requests.put(
                "http://" + self.master_address + "/rest/rec/syncmode", json=master_json, timeout=60
            )
            single_module = True
        print("\nStates after synchronization")
        self._get_states()

        # Now we open the recorders
        open_json = {
            # May need to investigate this further, but for now we won't use TEDS
            "performTransducerDetection": False,
            "singleModule": single_module,
        }
        for slave_address in self.slave_addresses:
            requests.put("http://" + slave_address + "/rest/rec/open", json=open_json, timeout=60)
        requests.put("http://" + self.master_address + "/rest/rec/open", json=open_json, timeout=60)
        print("\nStates after Open")
        self._get_states()

        # Now get the sample rate
        for i, address in enumerate(self.acquisition_map):
            response = requests.get("http://" + address + "/rest/rec/module/info", timeout=60)
            module_info = response.json()
            if i == 0:
                supported_sample_rates = module_info["supportedSampleRates"]
            else:
                supported_sample_rates = [
                    v for v in supported_sample_rates if v in module_info["supportedSampleRates"]
                ]
        print(f"Supported Sample Rates {supported_sample_rates}")
        bandwidth = test_data.sample_rate / 2.56
        if bandwidth > 1000:
            self.bandwidth_string = f"{bandwidth / 1000:0.1f} kHz"
        else:
            self.bandwidth_string = str(round(bandwidth)) + " Hz"
        print(f"Sample Rate: {test_data.sample_rate} Hz (Bandwidth {self.bandwidth_string})")
        if test_data.sample_rate not in supported_sample_rates:
            raise LanXIError(
                f"Invalid Sample Rate {test_data.sample_rate}, must be one of "
                f"{supported_sample_rates}"
            )

        # Get the Generator Sample Rate
        self.generator_sample_rate = round(np.log2(OUTPUT_RATE / test_data.sample_rate))
        if self.generator_sample_rate > 3:
            self.generator_sample_rate = 3
        elif self.generator_sample_rate < 0:
            self.generator_sample_rate = 0
        self.output_rate = OUTPUT_RATE // 2 ** (self.generator_sample_rate)
        # Now prep the generators
        self.set_generators()

        # Now we need to set up the recording configuration
        for slave_address in self.slave_addresses:
            if slave_address in self.acquisition_map:
                requests.put("http://" + slave_address + "/rest/rec/create", timeout=60)
            else:
                print(
                    f"Skipping Creating Slave Address Recorder {slave_address}, not in acquisition"
                )
        if self.master_address in self.acquisition_map:
            requests.put("http://" + self.master_address + "/rest/rec/create", timeout=60)
        else:
            print(
                f"Skipping Creating Master Address Recorder "
                f"{self.master_address}, not in acquisition"
            )
        for slave_address in self.slave_addresses:
            if slave_address in self.acquisition_map:
                wait_for_recorder_state(slave_address, "RecorderConfiguring")
        if self.master_address in self.acquisition_map:
            wait_for_recorder_state(self.master_address, "RecorderConfiguring")
        print("\nStates after Recorder Create")
        self._get_states()

        # State is now in Recorder Configuring
        print("Recorder in Configuring State")
        # Now we have to go through and create the channels
        for acquisition_device, device_dictionary in self.acquisition_map.items():
            response = requests.get(
                "http://" + acquisition_device + "/rest/rec/channels/input/default", timeout=60
            )
            channel_settings = response.json()
            # Go through and disable all channels
            for channel_json in channel_settings["channels"]:
                channel_json["enabled"] = False
            for channel_number, (_, channel) in device_dictionary.items():
                _, channel_json = [
                    (i, channel_json)
                    for i, channel_json in enumerate(channel_settings["channels"])
                    if channel_json["channel"] == channel_number
                ][0]
                channel_json["bandwidth"] = self.bandwidth_string
                channel_json["ccld"] = False if channel.excitation_source is None else True
                channel_json["transducer"]["requiresCcld"] = channel_json["ccld"]
                if channel_json["ccld"]:
                    print(f"Device {acquisition_device} channel {channel_number} has CCLD enabled")
                channel_json["destinations"] = ["socket"]
                channel_json["enabled"] = True
                channel_coupling = "DC" if channel.coupling is None else channel.coupling
                if channel_coupling not in VALID_FILTERS:
                    raise LanXIError(f"For LAN-XI, Coupling must be sent to one of {VALID_FILTERS}")
                channel_json["filter"] = channel_coupling
                if channel.maximum_value not in VALID_RANGES:
                    raise LanXIError(f"For LAN-XI, Maximum Value must be one of {VALID_RANGES}")
                channel_json["range"] = channel.maximum_value + " Vpeak"
                channel_json["transducer"]["sensitivity"] = float(channel.sensitivity) / 1000
                # The metadata doesn't really matter here, so we just use an arbitrary number
                # Otherwise it shold be something like this:
                # ('' if channel.serial_number is None else channel.serial_number)
                # +('' if channel.triax_dof is None else channel.triax_dof)
                channel_json["transducer"]["serialNumber"] = 9999
                channel_json["transducer"]["type"]["model"] = (
                    "" if channel.make is None else channel.make
                ) + ("" if channel.model is None else " " + channel.model)
                channel_json["transducer"]["unit"] = channel.unit
            response = requests.put(
                "http://" + acquisition_device + "/rest/rec/channels/input",
                json=channel_settings,
                timeout=60,
            )
            print(
                f"Setting inputs to {acquisition_device} Channels, {response.status_code} "
                f"{response.text}"
            )
        print("\nStates after Channel Input")
        self._get_states()

        # Now check for synchronization
        if len(self.slave_addresses) > 0:
            for slave_address in self.slave_addresses:
                if slave_address in self.acquisition_map:
                    wait_for_input_state(slave_address, "Settled")
            if self.master_address in self.acquisition_map:
                wait_for_input_state(self.master_address, "Settled")
            print("Recorder Settled, Synchronizing...")

            for slave_address in self.slave_addresses:
                if slave_address in self.acquisition_map:
                    requests.put("http://" + slave_address + "/rest/rec/synchronize", timeout=60)
            if self.master_address in self.acquisition_map:
                requests.put("http://" + self.master_address + "/rest/rec/synchronize", timeout=60)
            for slave_address in self.slave_addresses:
                if slave_address in self.acquisition_map:
                    wait_for_input_state(slave_address, "Synchronized")
            if self.master_address in self.acquisition_map:
                wait_for_input_state(self.master_address, "Synchronized")
            print("Recorder Synchronized, Starting Streaming...")

            for slave_address in self.slave_addresses:
                if slave_address in self.acquisition_map:
                    requests.put("http://" + slave_address + "/rest/rec/startstreaming", timeout=60)
            if self.master_address in self.acquisition_map:
                requests.put(
                    "http://" + self.master_address + "/rest/rec/startstreaming", timeout=60
                )

        # Wait for the module state to be recorder streaming
        for slave_address in self.slave_addresses:
            if slave_address in self.acquisition_map:
                wait_for_recorder_state(slave_address, "RecorderStreaming")
        if self.master_address in self.acquisition_map:
            wait_for_recorder_state(self.master_address, "RecorderStreaming")
        print("Recorder Streaming")
        self._get_states()

        print("\n\nData Acquisition Ready for Acquire")

    def start(self):
        """Method to start outputting data to the hardware"""
        self.empty_time += time.time()
        # Now start the generators
        master_json = None
        for generator_device, generator_channel_dict in self.output_map.items():
            json = {
                "outputs": [{"number": channel_number} for channel_number in generator_channel_dict]
            }
            if generator_device == self.master_address:
                master_json = (
                    json  # Pull this out because the master should be assigned last I think.
                )
                continue
            requests.put(
                "http://" + generator_device + "/rest/rec/generator/start", json=json, timeout=60
            )
        if master_json is not None:
            requests.put(
                "http://" + self.master_address + "/rest/rec/generator/start",
                json=master_json,
                timeout=60,
            )
        print("States after Generator Started")
        self._get_states()

    def write(self, data):
        """Method to write a frame of data to the hardware"""
        for output_device, socket_dict in self.sockets.items():
            for channel_number, socket_handle in socket_dict.items():
                output_index, _ = self.output_map[output_device][channel_number]
                this_data = (data[output_index] / 10 * 8372224).astype("int32").tobytes()
                while len(this_data) > 0:
                    sent_bytes = socket_handle.send(this_data[: self.transfer_size])
                    this_data = this_data[sent_bytes:]
        self.empty_time += self.write_size / self.output_rate

    def stop(self):
        """Method to stop the output"""
        master_json = None
        for generator_device, generator_channel_dict in self.output_map.items():
            json = {
                "outputs": [{"number": channel_number} for channel_number in generator_channel_dict]
            }
            if generator_device == self.master_address:
                master_json = (
                    json  # Pull this out because the master should be assigned last I think.
                )
                continue
            requests.put(
                "http://" + generator_device + "/rest/rec/generator/stop", json=json, timeout=60
            )
        if master_json is not None:
            requests.put(
                "http://" + self.master_address + "/rest/rec/generator/stop",
                json=master_json,
                timeout=60,
            )
        self.empty_time = 0.0
        self.set_generators()

    # region: Functions
    def set_generators(self):
        """Sets the generator states"""
        if len(self.output_map) == 0:
            return
        master_json = None
        for generator_device, generator_channel_dict in self.output_map.items():
            json = {
                "outputs": [{"number": channel_number} for channel_number in generator_channel_dict]
            }
            if generator_device == self.master_address:
                master_json = (
                    json  # Pull this out because the master should be assigned last I think.
                )
                continue
            requests.put(
                "http://" + generator_device + "/rest/rec/generator/prepare", json=json, timeout=60
            )
        if master_json is not None:
            requests.put(
                "http://" + self.master_address + "/rest/rec/generator/prepare",
                json=master_json,
                timeout=60,
            )
        print("\nStates after Generator Prepare")
        self._get_states()

        # Configure the generator channels
        master_json = None
        for generator_device, generator_channel_dict in self.output_map.items():
            json = {
                "bufferSize": self.buffer_size * self.write_size,  # TODO: Re-evaluate this number
                "outputs": [
                    {
                        "number": channel_number,
                        "floating": False,
                        "gain": 1.0,
                        "inputs": [
                            {
                                "number": 1,
                                "signalType": "stream",
                                "gain": 1.0,
                                "offset": 0.0,
                                "samplingRate": self.generator_sample_rate,
                            },
                            {"number": 2, "signalType": "none"},
                        ],
                    }
                    for channel_number in generator_channel_dict
                ],
            }
            if generator_device == self.master_address:
                # Pull this out because the master should be assigned last I think.
                master_json = json
                continue
            requests.put(
                "http://" + generator_device + "/rest/rec/generator/output", json=json, timeout=60
            )
        if master_json is not None:
            requests.put(
                "http://" + self.master_address + "/rest/rec/generator/output",
                json=master_json,
                timeout=60,
            )
        print("\nStates after Generator Output")
        self._get_states()

        # Now pull the socket information for the outputs
        for generator_device, generator_dict in self.output_map.items():
            response = requests.get(
                "http://" + generator_device + "/rest/rec/generator/output", timeout=60
            )
            output_data = response.json()
            for channel_number in generator_dict:
                output = [out for out in output_data["outputs"] if out["number"] == channel_number][
                    0
                ]
                if generator_device not in self.sockets:
                    self.sockets[generator_device] = {}

                is_ipv4 = re.search(IPV4_PATTERN, generator_device) is not None
                is_ipv6 = re.search(IPV6_PATTERN, generator_device) is not None
                if is_ipv4:
                    self.sockets[generator_device][output["number"]] = socket.socket(
                        socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP
                    )
                elif is_ipv6:
                    self.sockets[generator_device][output["number"]] = socket.socket(
                        socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP
                    )
                else:  # This will crash but is fixed in overhaul version so...
                    self.sockets[generator_device][output["number"]] = socket.socket(
                        socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP
                    )
                self.sockets[generator_device][output["number"]].connect(
                    (generator_device, output["inputs"][0]["port"])
                )
                print(
                    f"Output Connected to Device {generator_device} Channel {output['number']} "
                    f"on Port {output['inputs'][0]['port']}"
                )
                self.oversample_factor = round(
                    OUTPUT_RATE / (2 ** output["inputs"][0]["samplingRate"]) / self.sample_rate
                )
            print(f"Output overampling factor: {self.oversample_factor}x")

    def close(self, reboot=False):
        """Method to close down the hardware"""
        for _, socket_dict in self.sockets.items():
            for _, socket_handle in socket_dict.items():
                socket_handle.close()
        if reboot:
            self._reboot_all()
        else:
            self._close_recorders(
                [self.master_address] + [address for address in self.slave_addresses]
            )
        # self._reboot_all()
        # self._close_recorder(self.master_address)
        # for slave_address in self.slave_addresses:
        #     self._close_recorder(slave_address)

    def ready_for_new_output(self):
        """Method that returns true if the hardware should accept a new signal"""
        # print('Time until output buffer empty {:}, time per write {:}'.format(
        #       self.empty_time - time.time(),self.write_size / self.output_rate))
        if (self.empty_time - time.time()) < (
            self.write_size / self.output_rate
        ) * self.ready_signal_factor:  # TODO: Might need to increase buffer
            # print('Need new output')
            return True
        else:
            # print('No output needed')
            return False

    def _close_recorders(self, hosts):
        with mp.Pool(
            len(hosts)
        ) as pool:  # Not sure if this can be len(hosts) or if it should be self.maximum_processes
            pool.map(close_recorder, hosts)
        # host_states = {}
        # while True:
        #     for host in hosts:
        #         # print('Getting state from host {:}'.format(host))
        #         response = requests.get('http://'+host+'/rest/rec/onchange')
        #         state_data = response.json()
        #         current_state = state_data['moduleState']
        #         # print('Got state from host {:}'.format(host))
        #         if host in host_states and host_states[host] == current_state:
        #             continue
        #         host_states[host] = current_state
        #         try:
        #             operation = LANXI_STATE_SHUTDOWN[current_state]
        #         except KeyError:
        #             print('Unknown State {:} for host {:}.  Rebooting'.format(current_state,host))
        #             requests.put('http://'+host+'/rest/rec/reboot')
        #             continue
        #         if not operation is None:
        #             print('Host {:} at {:} state: {:}'.format(host,current_state,operation))
        #             requests.put('http://'+host+operation)
        #     # Check if all hosts are idle
        #     if all([v == 'Idle' for k,v in host_states.items()]):
        #         print('All hosts are idle')
        #         break
        #     time.sleep(0.2)

    def _get_states(self):
        for host in list(self.slave_addresses) + [self.master_address]:
            response = requests.get("http://" + host + "/rest/rec/onchange", timeout=60)
            state_data = response.json()
            print(
                f"Host {host}: Recorder State {state_data['moduleState']}, Input State "
                f"{state_data['inputStatus']}, PTP State {state_data['ptpStatus']}, Recording Mode"
            )

    def _reboot_all(self):
        for host in list(self.slave_addresses) + [self.master_address]:
            requests.put("http://" + host + "/rest/rec/reboot", timeout=60)
