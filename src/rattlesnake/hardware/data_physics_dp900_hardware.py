# -*- coding: utf-8 -*-
"""
Hardware definition that allows for the Data Physics DP900 Device to be run
with Rattlesnake.

The DP900 API runs acquisition and output on a single process. The output class
passes its data to the acquisition class through a multiprocessing queue, and the
acquisition class writes that data to the device while it waits for acquired data.

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

import multiprocessing.queues  # pylint: disable=unused-import
import multiprocessing.synchronize  # pylint: disable=unused-import
import time
import multiprocessing as mp
from typing import List, Self

import openpyxl
import netCDF4 as nc4
import numpy as np

from rattlesnake.utilities import flush_queue
from rattlesnake.hardware.hardware_utilities import Channel, HardwareType
from rattlesnake.hardware.data_physics_dp900_interface import (
    DP900,
    DP900Coupling,
    DP900Status,
)
from rattlesnake.hardware.abstract_hardware import (
    HardwareMetadata,
    HardwareAcquisition,
    HardwareOutput,
)

BUFFER_SIZE_FACTOR = 3  # Frames of output that may be buffered on the device
SLEEP_FACTOR = 10  # Fraction of a read frame to sleep while waiting for samples

HARDWARE_TYPE = HardwareType.DP_900


# region Metadata
class DataPhysicsDP900Metadata(HardwareMetadata):
    """
    Metadata that defines how the Data Physics DP900 hardware is set up.

    Extends ``HardwareMetadata`` with the path to the DP900 library file.

    Unit Tests
    ----------
    """

    def __init__(
        self,
        channel_list: List[Channel],
        sample_rate: int,
        time_per_read: float,
        time_per_write: float,
        hardware_file: str,
    ) -> None:
        """
        Initializes the DP900 metadata.

        Parameters
        ----------
        channel_list : List[Channel]
            Channels in the channel table.
        sample_rate : int
            Sample rate of the data acquisition in samples per second.
        time_per_read : float
            Time, in seconds, of data acquired in each read frame.
        time_per_write : float
            Time, in seconds, of data output in each write frame.
        hardware_file : str
            Path to the ``Dp900Matlab.dll`` library file used to run the DP900 device.

        Unit Tests
        ----------
        """
        super().__init__(
            HARDWARE_TYPE,
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
        )
        self.hardware_file = hardware_file

    # region Validation
    def validate(self) -> None:
        """
        Check that the metadata fully defines the DP900 hardware.

        Only the checks shared by all hardware are performed.

        Raises
        ------
        RattlesnakeError
            If ``channel_list`` contains duplicate channels, or ``output_oversample``
            is not a number greater than 0.

        Unit Tests
        ----------
        """
        return super().validate()

    # endregion

    # region Loading
    def save_metadata_to_netcdf(self, netcdf_dataset: nc4.Dataset) -> None:
        """
        Save the DP900 metadata to a netCDF4 dataset.

        Saves the metadata shared by all hardware and the ``hardware_file`` attribute.

        Parameters
        ----------
        netcdf_dataset : netCDF4.Dataset
            An open, writable netCDF4 dataset.

        Unit Tests
        ----------
        """
        super().save_metadata_to_netcdf(netcdf_dataset)

        netcdf_dataset.hardware_file = self.hardware_file

    @classmethod
    def load_metadata_from_netcdf(cls, netcdf_dataset: nc4.Dataset) -> Self:
        """
        Load DP900 metadata from a netCDF4 dataset.

        Parameters
        ----------
        netcdf_dataset : netCDF4.Dataset
            An open netCDF4 dataset saved by ``save_metadata_to_netcdf``.

        Returns
        -------
        DataPhysicsDP900Metadata
            Metadata loaded from the dataset.

        Unit Tests
        ----------
        """
        (
            hardware_type,
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
            output_oversample,
        ) = super().load_metadata_from_netcdf(netcdf_dataset)

        hardware_file = netcdf_dataset.hardware_file

        return cls(
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
            hardware_file,
        )

    def save_metadata_to_workbook(
        self, workbook: openpyxl.workbook.workbook.Workbook
    ) -> None:
        """
        Save the DP900 metadata to a workbook.

        Saves the metadata shared by all hardware and writes ``hardware_file`` to the
        "Hardware File" row of the "Hardware" sheet.

        Parameters
        ----------
        workbook : openpyxl.workbook.workbook.Workbook
            Workbook to write the metadata to.

        Unit Tests
        ----------
        """
        super().save_metadata_to_workbook(workbook)

        hardware_worksheet = workbook["Hardware"]
        hardware_worksheet.cell(2, 2, self.hardware_file)

    @classmethod
    def load_metadata_from_workbook(
        cls, workbook: openpyxl.workbook.workbook.Workbook
    ) -> Self:
        """
        Load DP900 metadata from a workbook.

        Reads the metadata shared by all hardware and the "Hardware File" row of the
        "Hardware" sheet.

        Parameters
        ----------
        workbook : openpyxl.workbook.workbook.Workbook
            Workbook containing a channel table and a "Hardware" sheet.

        Returns
        -------
        DataPhysicsDP900Metadata
            Metadata loaded from the workbook.

        Raises
        ------
        RattlesnakeError
            If the workbook has more than one channel table sheet or none.
        KeyError
            If the workbook does not have a "Hardware" sheet.

        Unit Tests
        ----------
        """
        (
            hardware_type,
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
            output_oversample,
        ) = super().load_metadata_from_workbook(workbook)

        hardware_file = None

        hardware_worksheet = workbook["Hardware"]
        for row in hardware_worksheet.rows:
            name = str(row[0].value).lower().strip().replace(" ", "_")
            value = row[1].value
            if value is None or value == "":
                continue
            match name:
                case "hardware_file":
                    hardware_file = value
                case _:
                    continue

        return cls(
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
            hardware_file,
        )

    # endregion


# endregion


# region Acqusition
class DataPhysicsDP900Acquisition(HardwareAcquisition):
    """
    Class defining the interface between the Acquisition process and Data Physics DP900
    hardware.

    This class defines the interfaces between the controller and the Data Physics
    hardware that runs their open API. It is run by the Acquisition process, and must
    define how to get data from the test hardware into the controller.

    Because the DP900 runs acquisition and output on a single process, this class also
    writes output to the device. ``DataPhysicsDP900Output`` puts output frames on a
    queue, and this class writes them to the device while it waits for acquired data.

    Unit Tests
    ----------
    """

    def __init__(
        self, ping_alive_event: mp.synchronize.Event, queue: mp.queues.Queue
    ) -> None:
        """
        Initializes the Data Physics DP900 acquisition interface.

        The device is not connected until ``initialize_hardware`` is called.

        Parameters
        ----------
        ping_alive_event : multiprocessing.synchronize.Event
            Event used to reset the controller's timeout during long operations. Not
            used by the DP900.
        queue : multiprocessing.queues.Queue
            Queue used to pass output data from the output process to the acquisition
            process, because the DP900 runs on a single process.

        Unit Tests
        ----------
        """

        self.DP900Coupling = DP900Coupling
        self.DP900Status = DP900Status
        self.output_active = False
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

    def initialize_hardware(self, test_data: DataPhysicsDP900Metadata) -> None:
        """
        Initialize the hardware and set up channels and sampling properties.

        Loads the DP900 library, resets the device to a disconnected state and
        reconnects, then sets up the input and output channels, sample rate, and
        buffer size. Channels with a ``feedback_device`` are set up as outputs, and all
        other channels as inputs. The input coupling is read from each channel's
        ``coupling`` value.

        The device orders channels by BNC number, so the sorting needed to return data
        in channel table order is stored for ``read`` and ``read_remaining``.

        Parameters
        ----------
        test_data : DataPhysicsDP900Metadata
            Metadata containing the sampling properties, channel list, and library file
            path for the DP900.

        Raises
        ------
        ValueError
            If a channel's ``physical_device`` is not an online system, the channels
            use more than one system, or a channel's ``physical_channel`` does not
            match exactly one BNC configured in DP900Config.

        Unit Tests
        ----------
        """
        # Store data acquisition parameters for later
        self.dp900 = DP900(test_data.hardware_file)
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
        for ct_index, channel in enumerate(test_data.channel_list):
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
                (
                    channel.feedback_device.startswith("#")
                    or channel.feedback_device.strip() == ""
                )
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
                if str(channel.coupling).lower() in [
                    "ac differential",
                    "ac diff",
                    "ac",
                ]:
                    input_couplings.append(self.DP900Coupling.AC_DIFFERENTIAL)
                elif str(channel.coupling).lower() in [
                    "dc differential",
                    "dc diff",
                    "dc",
                ]:
                    input_couplings.append(self.DP900Coupling.DC_DIFFERENTIAL)
                elif str(channel.coupling).lower() in [
                    "ac single ended",
                    "ac single-ended",
                    "ac single",
                ]:
                    input_couplings.append(self.DP900Coupling.AC_SINGLE_ENDED)
                elif str(channel.coupling).lower() in [
                    "dc single ended",
                    "dc single-ended",
                    "dc single",
                ]:
                    input_couplings.append(self.DP900Coupling.DC_SINGLE_ENDED)
                elif str(channel.coupling).lower() in ["iepe", "icp", "ac icp", "ccld"]:
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
            self.dp900.setup_output_parameters(
                output_sensitivities, output_ranges, output_channels
            )

            # Since the outputs are at the end, we want to adjust the sorting to
            # put the outputs at the end
            # all_bncs = np.array(all_bncs)
            # all_bncs[self.output_channel_table_indices] += 100000000

        # We should then be able to use this channel sorting to unsort the
        # read data
        self.channel_sorting = np.argsort(all_bncs)

        self.dp900.set_save_recording(False)

    def start(self) -> None:
        """
        Start acquiring data from the hardware.

        Unit Tests
        ----------
        """
        self.dp900.init()
        self.dp900.start()

    def read(self) -> np.ndarray:
        """
        Read a frame of data from the hardware.

        Waits until a full frame of samples is available. While waiting, any output
        on the queue is written to the device.

        Returns
        -------
        np.ndarray
            A ``num_channels x samples_per_read`` array of acquired data in channel
            table order.

        Unit Tests
        ----------
        """
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
        read_data = self.dp900.read_input_data(
            self.data_acquisition_parameters.samples_per_read
        )
        # Now we need to sort the data correctly to give it back to the channel table
        read_data[self.channel_sorting] = read_data.copy()
        return read_data

    def read_remaining(self) -> np.ndarray:
        """
        Read the rest of the data on the hardware after the acquisition stops.

        Writes any output on the queue to the device, then waits until at least one
        sample is available and reads every available sample.

        Returns
        -------
        np.ndarray
            A ``num_channels x num_samples`` array of the remaining acquired data in
            channel table order.

        Unit Tests
        ----------
        """
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

    def stop(self) -> None:
        """
        Stop the acquisition and end the measurement.

        Any output left on the queue is discarded.

        Unit Tests
        ----------
        """
        self.dp900.stop()
        self.dp900.end()
        flush_queue(self.output_data_queue)

    def close(self) -> None:
        """
        Disconnect from the hardware.

        Any output left on the queue is discarded.

        Unit Tests
        ----------
        """
        self.dp900.disconnect()
        flush_queue(self.output_data_queue)

    def get_acquisition_delay(self) -> int:
        """
        Get the number of samples between output and acquisition.

        This method handles buffering done in the output hardware, ensuring that all
        data written to the output is read by the acquisition. If the output hardware
        has a buffer, there may be a non-negligible delay between when output is
        written to the device and when it is played out from the device.

        Returns
        -------
        int
            ``BUFFER_SIZE_FACTOR`` times ``samples_per_write``.

        Unit Tests
        ----------
        """
        return BUFFER_SIZE_FACTOR * self.data_acquisition_parameters.samples_per_write

    def get_and_write_output_data(self, block: bool = False) -> None:
        """
        Write output data from the queue to the hardware.

        If not blocking and the device already has at least three write frames on its
        output buffer, no data is taken from the queue. Output data is sorted into the
        device's BNC order before it is written.

        Parameters
        ----------
        block : bool, optional
            If ``True``, wait up to 10 seconds for data to appear on the queue.
            Otherwise return immediately if there is no data available. Defaults to
            ``False``.

        Raises
        ------
        RuntimeError
            If ``block`` is ``True`` and no data arrives before the timeout.

        Unit Tests
        ----------
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


# endregion


# region Output
class DataPhysicsDP900Output(HardwareOutput):
    """
    Class defining the interface between the Output process and Data Physics DP900
    output.

    This class defines the interfaces between the controller and the output or source
    portion of the hardware. It is run by the Output process, and must define how to
    write data from the control system to the hardware.

    Because the DP900 runs acquisition and output on a single process, this class does
    not communicate with the device. It puts output frames on a queue that
    ``DataPhysicsDP900Acquisition`` reads and writes to the device.

    Unit Tests
    ----------
    """

    def __init__(
        self, ping_alive_event: mp.synchronize.Event, queue: mp.queues.Queue
    ) -> None:
        """
        Initializes the output interface by storing the data passing queue.

        Parameters
        ----------
        ping_alive_event : multiprocessing.synchronize.Event
            Event used to reset the controller's timeout during long operations. Not
            used by the DP900.
        queue : multiprocessing.queues.Queue
            Queue used to pass data from the output process to the acquisition process.

        Unit Tests
        ----------
        """
        self.queue = queue

    def initialize_hardware(self, test_data: DataPhysicsDP900Metadata) -> None:
        """
        Initialize the hardware and set up sources and sampling properties.

        Does nothing, because ``DataPhysicsDP900Acquisition`` sets up the output
        channels.

        Parameters
        ----------
        test_data : DataPhysicsDP900Metadata
            Metadata containing the sampling properties and channel list for the DP900.

        Unit Tests
        ----------
        """

    def start(self) -> None:
        """
        Start outputting data to the hardware.

        Does nothing, because ``DataPhysicsDP900Acquisition`` starts the device.

        Unit Tests
        ----------
        """
        # TODO: Remove this
        # self.last_check_time = time.time()

    def write(self, data: np.ndarray) -> None:
        """
        Put a frame of output data on the queue for the acquisition to write.

        Parameters
        ----------
        data : np.ndarray
            A ``num_output_channels x samples_per_write`` array to write to the output
            hardware.

        Unit Tests
        ----------
        """
        self.queue.put(data)

    def stop(self) -> None:
        """
        Stop the output by discarding any output left on the queue.

        Unit Tests
        ----------
        """
        flush_queue(self.queue)

    def close(self) -> None:
        """
        Close down the output by discarding any output left on the queue.

        Unit Tests
        ----------
        """
        flush_queue(self.queue)

    def ready_for_new_output(self) -> bool:
        """
        Return whether the hardware is ready to accept a new frame of output.

        Returns
        -------
        bool
            ``True`` if the data passing queue is empty, otherwise ``False``.

        Unit Tests
        ----------
        """
        return self.queue.empty()


# endregion
