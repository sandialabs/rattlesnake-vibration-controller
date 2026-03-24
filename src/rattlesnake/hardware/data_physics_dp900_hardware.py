# -*- coding: utf-8 -*-
"""
Hardware definition that allows for the Data Physics DP900 Device to be run
with Rattlesnake.

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
import time
from typing import List

import numpy as np

from rattlesnake.hardware.abstract_hardware import HardwareAcquisition, HardwareOutput
from rattlesnake.utilities import Channel, DataAcquisitionParameters, flush_queue

BUFFER_SIZE_FACTOR = 3
SLEEP_FACTOR = 10


# region: Acqusition
class DataPhysicsDP900Acquisition(HardwareAcquisition):
    """Class defining the interface between the controller and Data Physics
    DP900 hardware

    This class defines the interfaces between the controller and the
    Data Physics hardware that runs their open API.  It is run by the Acquisition
    process, and must define how to get data from the test hardware into the
    controller."""

    def __init__(self, dll_path: str, queue: mp.queues.Queue):
        """
        Initializes the data physics hardware interface.

        Parameters
        ----------
        dll_path : str
            Path to the Dp900Matlab.dll file that defines
        queue : mp.queues.Queue
            Multiprocessing queue used to pass output data from the output task
            to the acquisition task because DP900 runs on a single processor

        Returns
        -------
        None.

        """
        from .data_physics_dp900_interface import DP900, DP900Coupling, DP900Status

        self.DP900Coupling = DP900Coupling
        self.DP900Status = DP900Status
        self.output_active = False
        self.dp900 = DP900(dll_path)
        self.buffer_size = 2**24
        self.input_bnc_indices = []
        self.output_bnc_indices = []
        self.input_channel_table_indices = []
        self.output_channel_table_indices = []
        self.channel_sorting = None
        self.output_sorting = None
        self.data_acquisition_parameters = None
        self.output_data_queue = queue
        self.time_per_read = None
        self.last_write_time = None

    # region: Store Metadata
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
        # Store data acquisition parameters for later
        self.data_acquisition_parameters = test_data
        self.time_per_read = test_data.samples_per_read / test_data.sample_rate

        # End the measurement if necessary
        if self.dp900.status == self.DP900Status.RUNNING:
            self.dp900.stop()
        if (
            self.dp900.status == self.DP900Status.STOPPED
            or self.dp900.status == self.DP900Status.INIT
        ):
            self.dp900.end()
        if self.dp900.status == self.DP900Status.DISCONNECTED:
            self.dp900.connect("")

        # Get the system information
        system_list = self.dp900.get_system_list(online=True)

        # Get the channel information
        input_bncs = self.dp900.get_input_channel_bncs()
        output_bncs = self.dp900.get_output_channel_bncs()

        # Set up defaults that will be overwritten
        self.output_active = False
        self.input_bnc_indices = []
        self.output_bnc_indices = []
        self.input_channel_table_indices = []
        self.output_channel_table_indices = []

        input_couplings = []
        input_ranges = []
        input_sensitivities = []
        output_ranges = []
        output_sensitivities = []
        all_bncs = []
        systems = []

        # Set up channel parameters
        for ct_index, channel in enumerate(channel_data):
            system = channel.physical_device
            if system not in system_list:
                raise ValueError(
                    f"System {system} is not a valid system.  Must be one of {system_list}"
                )
            if system not in systems:
                systems.append(system)
                if len(systems) > 1:
                    raise ValueError(
                        "Multi-chassis tests are not currently supported in Rattlesnake"
                    )
            # Figure out if the channel is an output channel or just acquisition
            is_output = not (channel.feedback_device is None) and not (
                channel.feedback_device.strip() == ""
            )

            # Get the channel index from the bnc number
            if is_output:
                self.output_active = True
                all_bncs.append(int(channel.physical_channel))
                bnc_index = np.flatnonzero(output_bncs == int(channel.physical_channel))
                if len(bnc_index) > 1:
                    raise ValueError(
                        f"More than one matching channel for BNC {channel.physical_channel} "
                        "(how did this happen?)"
                    )
                if len(bnc_index) < 1:
                    raise ValueError(
                        f"BNC {channel.physical_channel} was not found in the list of output "
                        f"BNCs {output_bncs}.  Please run DP900Config to correctly set input "
                        f"and output channels."
                    )
                bnc_index = bnc_index[0]
                self.output_bnc_indices.append(bnc_index)
                self.output_channel_table_indices.append(ct_index)
                output_ranges.append(float(channel.maximum_value))
                output_sensitivities.append(1)
            else:
                all_bncs.append(int(channel.physical_channel))
                bnc_index = np.flatnonzero(input_bncs == int(channel.physical_channel))
                if len(bnc_index) > 1:
                    raise ValueError(
                        f"More than one matching channel for BNC {channel.physical_channel} "
                        f"(how did this happen?)"
                    )
                if len(bnc_index) < 1:
                    raise ValueError(
                        f"BNC {channel.physical_channel} was not found in the list of input "
                        f"BNCs {input_bncs}.  Please run DP900Config to correctly set input "
                        f"and output channels."
                    )
                bnc_index = bnc_index[0]
                self.input_bnc_indices.append(bnc_index)
                self.input_channel_table_indices.append(ct_index)
                input_ranges.append(float(channel.maximum_value))
                input_sensitivities.append(float(channel.sensitivity))
                # Set the values in the arrays appropriately given the channel index
                if channel.coupling.lower() in ["ac differential", "ac diff", "ac"]:
                    input_couplings.append(self.DP900Coupling.AC_DIFFERENTIAL)
                elif channel.coupling.lower() in ["dc differential", "dc diff", "dc"]:
                    input_couplings.append(self.DP900Coupling.DC_DIFFERENTIAL)
                elif channel.coupling.lower() in [
                    "ac single ended",
                    "ac single-ended",
                    "ac single",
                ]:
                    input_couplings.append(self.DP900Coupling.AC_SINGLE_ENDED)
                elif channel.coupling.lower() in [
                    "dc single ended",
                    "dc single-ended",
                    "dc single",
                ]:
                    input_couplings.append(self.DP900Coupling.DC_SINGLE_ENDED)
                elif channel.coupling.lower() in ["iepe", "icp", "ac icp", "ccld"]:
                    input_couplings.append(self.DP900Coupling.AC_COUPLED_IEPE)

        self.dp900.set_system_list(systems)

        # Set the sample rate
        self.dp900.set_sample_rate(test_data.sample_rate)

        # Set the buffer size
        self.dp900.set_buffer_size(self.buffer_size)
        # print('Buffer Size: {:}'.format(self.buffer_size))

        # Now we need to re-order the items to pass to the parameter setup
        # functions
        input_sorting = np.argsort(self.input_bnc_indices)
        input_channels = np.array(self.input_bnc_indices)[input_sorting] + 1
        input_couplings = np.array(input_couplings)[input_sorting]
        input_ranges = np.array(input_ranges)[input_sorting]
        input_sensitivities = np.array(input_sensitivities)[input_sorting]
        self.dp900.setup_input_parameters(
            input_couplings, input_channels, input_sensitivities, input_ranges
        )

        if self.output_active:
            self.output_sorting = np.argsort(self.output_bnc_indices)
            output_channels = np.array(self.output_bnc_indices)[self.output_sorting] + 1
            output_ranges = np.array(output_ranges)[self.output_sorting]
            output_sensitivities = np.array(output_sensitivities)[self.output_sorting]

            # Now send the data to the dp900 device
            self.dp900.setup_output_parameters(output_sensitivities, output_ranges, output_channels)

            # Since the outputs are at the end, we want to adjust the sorting to
            # put the outputs at the end
            # all_bncs = np.array(all_bncs)
            # all_bncs[self.output_channel_table_indices] += 100000000

        # We should then be able to use this channel sorting to unsort the
        # read data
        self.channel_sorting = np.argsort(all_bncs)

        self.dp900.set_save_recording(False)

    # region: Abstract Methods
    def start(self):
        """Method to start acquiring data from the hardware"""
        self.dp900.init()
        self.dp900.start()

    def read(self) -> np.ndarray:
        """Method to read a frame of data from the hardware that returns
        an appropriately sized np.ndarray"""
        while (
            self.dp900.get_available_input_data_samples()
            < self.data_acquisition_parameters.samples_per_read
        ):
            # Check if we need to output anything
            if self.output_active:
                self.get_and_write_output_data()
            # Pause for a bit to allow more samples to accumulate
            time.sleep(self.time_per_read / SLEEP_FACTOR)
        # Read the data now that we have enough samples
        read_data = self.dp900.read_input_data(self.data_acquisition_parameters.samples_per_read)
        # Now we need to sort the data correctly to give it back to the channel table
        read_data[self.channel_sorting] = read_data.copy()
        return read_data

    def read_remaining(self) -> np.ndarray:
        """Method to read the rest of the data on the acquisition from the hardware
        that returns an appropriately sized np.ndarray"""
        # Check if we need to output anything
        if self.output_active:
            self.get_and_write_output_data()
        # Check how many samples are available
        samples_available = 0
        # Wait until some arrive
        while samples_available == 0:
            samples_available = self.dp900.get_available_input_data_samples()
            # Pause for a bit to allow more samples to accumulate
            time.sleep(self.time_per_read / SLEEP_FACTOR)
        # Read that many data samples and put it to the "read_data" array
        # Make sure we rearrange the channels correctly per the rattlesnake
        # channel table using self.input_channel_order
        read_data = self.dp900.read_input_data(samples_available)
        read_data[self.channel_sorting] = read_data.copy()
        return read_data

    def stop(self):
        """Method to stop the acquisition"""
        self.dp900.stop()
        self.dp900.end()
        flush_queue(self.output_data_queue)

    def close(self):
        """Method to close down the hardware"""
        self.dp900.disconnect()
        flush_queue(self.output_data_queue)

    def get_acquisition_delay(self) -> int:
        """Get the number of samples between output and acquisition

        This function is designed to handle buffering done in the output
        hardware, ensuring that all data written to the output is read by the
        acquisition.  If a output hardware has a buffer, there may be a non-
        negligable delay between when output is written to the device and
        actually played out from the device."""
        return BUFFER_SIZE_FACTOR * self.data_acquisition_parameters.samples_per_write

    # region: Functions
    def get_and_write_output_data(self, block: bool = False):
        """
        Checks to see if there is any data on the output queue that needs to be
        written to the hardware.

        Parameters
        ----------
        block : bool, optional
            If True, this function will wait until the data appears with a timeout
            of 10 seconds.  Otherwise it will simply return if there is no
            data available. The default is False.

        Raises
        ------
        RuntimeError
            Raised if the timeout occurs while waiting for data while blocking

        Returns
        -------
        None.

        """
        samples_on_buffer = self.dp900.get_total_output_samples_on_buffer()
        write_threshold = 3 * self.data_acquisition_parameters.samples_per_write
        # print('{:} Samples on Output Buffer, (<{:} to output more)'.format(
        #     samples_on_buffer,write_threshold))
        # TODO: Uncomment this
        if not block and samples_on_buffer >= write_threshold:
            # print('Too much data on buffer, not putting new data')
            return
        try:
            data = self.output_data_queue.get(block, timeout=10)
            # print('Got New Data from queue')
        except mp.queues.Empty as e:
            # print('Did not get new data from queue')
            if block:
                raise RuntimeError(
                    "Did not receive output in a reasonable amount of time, check output "
                    "process and output hardware for issues"
                ) from e
            # Otherwise just return because there's no data available
            return
        # If we did get output, we need to put it into a numpy array that we can
        # send to the daq
        this_write_time = time.time()
        outputs = data[self.output_sorting]
        # if self.last_write_time is not None:
        #     print('Time since last write: {:}'.format(this_write_time - self.last_write_time))
        self.last_write_time = this_write_time
        # Send the outputs to the daq
        self.dp900.write_output_data(outputs)
        return


# region: Output
class DataPhysicsDP900Output(HardwareOutput):
    """Abstract class defining the interface between the controller and output

    This class defines the interfaces between the controller and the
    output or source portion of the hardware.  It is run by the Output
    process, and must define how to get write data to the hardware from the
    control system"""

    def __init__(self, queue: mp.queues.Queue):
        """
        Initializes the hardware by simply storing the data passing queue

        Parameters
        ----------
        queue : mp.queues.Queue
            Queue used to pass data from output to acquisition

        Returns
        -------
        None.

        """
        self.queue = queue

    # region: Abstract Methods
    def set_up_data_output_parameters_and_channels(
        self, test_data: DataAcquisitionParameters, channel_data: List[Channel]
    ):
        """
        Initialize the hardware and set up sources and sampling properties

        The function must create channels on the hardware corresponding to
        the sources in the test.  It must also set the sampling rates.

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

    def start(self):
        """Method to start outputting data to the hardware"""
        # TODO: Remove this
        # self.last_check_time = time.time()

    def write(self, data):
        """Method to write a np.ndarray with a frame of data to the hardware"""
        self.queue.put(data)

    def stop(self):
        """Method to stop the output"""
        flush_queue(self.queue)

    def close(self):
        """Method to close down the hardware"""
        flush_queue(self.queue)

    def ready_for_new_output(self) -> bool:
        """Method that returns true if the hardware should accept a new signal

        Returns ``True`` if the data-passing queue is empty."""
        return self.queue.empty()
