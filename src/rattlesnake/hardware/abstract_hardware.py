# -*- coding: utf-8 -*-
"""
Abstract hardware definitions that can be used to implement new hardware devices.

Each hardware device implements three classes: a ``HardwareMetadata`` subclass that
defines and validates the hardware setup, a ``HardwareAcquisition`` subclass run by
the acquisition process, and a ``HardwareOutput`` subclass run by the output process.

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

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import netCDF4 as nc4
import openpyxl

from rattlesnake.utilities import RattlesnakeError, worksheet_cell_str
from rattlesnake.hardware.hardware_utilities import Channel, HardwareType
from rattlesnake.user_interface.ui_utilities import HardwareAssistModules


# region Metadata
class HardwareMetadata:
    """
    Abstract class that contains values to fully define how the hardware is setup.

    This class contains attributes required for the environment to interact with the
    acquisition, output, and streaming processes. The class should be extended with
    attributes required for the ``HardwareAcquisition`` and ``HardwareOutput`` classes
    specific to that ``HardwareType``.

    Subclasses must implement validation, the channel table assist methods, and loading
    and saving of their metadata. The base implementations of those methods handle the
    attributes shared by all hardware, so subclasses should call them through
    ``super()`` and extend the result.

    Unit Tests
    ----------
    """

    def __init__(
        self,
        hardware_type: HardwareType,
        channel_list: List[Channel],
        sample_rate: int,
        time_per_read: float,
        time_per_write: float,
        *,
        output_oversample: int = 1,
    ) -> None:
        """
        Initializes the metadata shared by all hardware devices.

        Parameters
        ----------
        hardware_type : HardwareType
            Type of hardware device the metadata defines.
        channel_list : List[Channel]
            Channels in the channel table.
        sample_rate : int
            Sample rate of the data acquisition in samples per second.
        time_per_read : float
            Time, in seconds, of data acquired in each read frame.
        time_per_write : float
            Time, in seconds, of data output in each write frame.
        output_oversample : int, optional
            Factor by which the output sample rate exceeds the acquisition sample rate.
            Used by virtual hardware for integration oversampling. Defaults to ``1``.

        Unit Tests
        ----------
        """
        self.hardware_type = hardware_type
        self._channel_list = channel_list
        self.sample_rate = sample_rate
        self.time_per_read = time_per_read
        self.time_per_write = time_per_write
        # Used for virtual hardware but still required for normal hardware
        self.output_oversample = output_oversample

    @property
    def channel_list(self) -> List[Channel]:
        """
        Get or set the channels in the channel table.

        Returns
        -------
        List[Channel]
            Channels in the channel table.

        Unit Tests
        ----------
        """
        return self._channel_list

    @channel_list.setter
    def channel_list(self, value: List[Channel]) -> None:
        self._channel_list = value

    @property
    def samples_per_read(self) -> int:
        """
        Return the number of samples in each read frame.

        Returns
        -------
        int
            ``sample_rate * time_per_read`` rounded to the nearest integer.

        Unit Tests
        ----------
        """
        return round(self.sample_rate * self.time_per_read)

    @property
    def samples_per_write(self) -> int:
        """
        Return the number of samples in each write frame.

        Returns
        -------
        int
            ``sample_rate * time_per_write * output_oversample`` rounded to the nearest
            integer.

        Unit Tests
        ----------
        """
        return round(self.sample_rate * self.time_per_write * self.output_oversample)

    @property
    def nyquist_frequency(self) -> float:
        """
        Return the Nyquist frequency of the data acquisition.

        Returns
        -------
        float
            Half of ``sample_rate``, in Hz.

        Unit Tests
        ----------
        """
        return self.sample_rate / 2

    @property
    def output_sample_rate(self) -> int:
        """
        Return the sample rate of the output.

        Returns
        -------
        int
            ``sample_rate * output_oversample``, in samples per second.

        Unit Tests
        ----------
        """
        return self.sample_rate * self.output_oversample

    # region Validation
    @abstractmethod
    def validate(self) -> None:
        """
        Check that the metadata fully defines the hardware and is valid for that device.

        If possible, subclasses should check which devices are connected to the computer
        and make sure the metadata is a valid input to the ``initialize_hardware``
        method of the ``HardwareAcquisition`` and ``HardwareOutput`` classes.

        Raise detailed errors while validating. These errors show up in log files for
        debugging and do not stop the main process from running.

        The base implementation checks the attributes shared by all hardware.

        Raises
        ------
        RattlesnakeError
            If ``channel_list`` contains duplicate channels, or ``output_oversample`` is
            not a number greater than 0.

        Unit Tests
        ----------
        """
        if len(self.channel_list) != len(set(self.channel_list)):
            raise RattlesnakeError("Duplicate channels found in channel_list")

        if (
            not isinstance(self.output_oversample, (int, float))
            or self.output_oversample <= 0
        ):
            raise RattlesnakeError(f"output_oversample must be a number greater than 0")

    @abstractmethod
    def valid_channel_dict(self, channel: Channel) -> Dict[str, list]:
        """
        Return the valid values of each channel table column for a channel.

        Used to fill the channel table assist widgets. Valid values can depend on the
        channel's other values, such as the physical channels available on its
        ``physical_device``. The base implementation returns an empty list for every
        column, and subclasses fill in the columns they can validate. An example for
        this is:
            "node_number": [1, 2, 3, 4],
            "node_direction": [],

        Parameters
        ----------
        channel : Channel
            Channel to find the valid values for.

        Returns
        -------
        Dict[str, list]
            Mapping from each attribute in ``Channel.channel_attr_list`` to its list of
            valid values.

        An empty list means the column has no restricted values.

        Unit Tests
        ----------
        """
        valid_dict = {}
        for attr in Channel().channel_attr_list:
            valid_dict[attr] = []
        return valid_dict

    @property
    @abstractmethod
    def assist_mode_modules(self) -> Dict[str, HardwareAssistModules]:
        """
        Return the assist widget used for each channel table column.

        The base implementation uses ``HardwareAssistModules.NONE`` for every column,
        and subclasses change the columns that have a combo box or spin box.

        Returns
        -------
        Dict[str, HardwareAssistModules]
            Mapping from each attribute in ``Channel.channel_attr_list`` to its assist
            widget type.

        Unit Tests
        ----------
        """
        assist_modules = {}
        for attr in Channel().channel_attr_list:
            assist_modules[attr] = HardwareAssistModules.NONE
        return assist_modules

    # endregion

    # region Loading
    @classmethod
    def load_channel_table_from_netcdf(
        cls, netcdf_dataset: nc4.Dataset
    ) -> List[Channel]:
        """
        Load the channel table from the ``channels`` group of a netCDF4 dataset.

        Blank string values are loaded as ``None``, and empty channels are skipped.

        Parameters
        ----------
        netcdf_dataset : netCDF4.Dataset
            An open netCDF4 dataset containing a ``channels`` group and a
            ``response_channels`` dimension.

        Returns
        -------
        List[Channel]
            Channels loaded from the dataset.

        Unit Tests
        ----------
        """
        channel_table = netcdf_dataset["channels"]

        channel_list = []
        num_channels = netcdf_dataset.dimensions["response_channels"].size
        channel_attr_list = Channel().channel_attr_list
        for row_idx in range(num_channels):
            channel = Channel()
            for attr in channel_attr_list:
                value = channel_table[attr][row_idx]
                value = None if isinstance(value, str) and not value.strip() else value
                setattr(channel, attr, value)

            if not channel.is_empty:  # optional safety check
                channel_list.append(channel)

        return channel_list

    @classmethod
    def save_channel_table_to_workbook(
        self, channel_list: List[Channel], workbook: openpyxl.workbook.workbook.Workbook
    ) -> None:
        """
        Write a channel table to the active sheet of a workbook.

        The active sheet is renamed to ``"Channel Table"``. Group headers are written to
        the first row, column headers to the second row, and one channel per row
        starting at the third row. ``None`` values are written as empty strings.

        Parameters
        ----------
        channel_list : List[Channel]
            Channels to write to the channel table.
        workbook : openpyxl.workbook.workbook.Workbook
            Workbook to write the channel table to.

        Unit Tests
        ----------
        """
        worksheet = workbook.active
        worksheet.title = "Channel Table"
        worksheet.cell(row=1, column=2, value="Test Article Definition")
        worksheet.merge_cells(start_row=1, start_column=2, end_row=1, end_column=4)
        worksheet.cell(row=1, column=5, value="Instrument Definition")
        worksheet.merge_cells(start_row=1, start_column=5, end_row=1, end_column=11)
        worksheet.cell(row=1, column=12, value="Channel Definition")
        worksheet.merge_cells(start_row=1, start_column=12, end_row=1, end_column=19)
        worksheet.cell(row=1, column=20, value="Output Feedback")
        worksheet.merge_cells(start_row=1, start_column=20, end_row=1, end_column=21)
        worksheet.cell(row=1, column=22, value="Limits")
        worksheet.merge_cells(start_row=1, start_column=22, end_row=1, end_column=23)
        for col_idx, val in enumerate(
            [
                "Channel Index",
                "Node Number",
                "Node Direction",
                "Comment",
                "Serial Number",
                "Triax DoF",
                "Sensitivity (mV/EU)",
                "Engineering Unit",
                "Make",
                "Model",
                "Calibration Exp Date",
                "Physical Device",
                "Physical Channel",
                "Type",
                "Minimum Value (V)",
                "Maximum Value (V)",
                "Coupling",
                "Current Excitation Source",
                "Current Excitation Value",
                "Physical Device",
                "Physical Channel",
                "Warning Level (EU)",
                "Abort Level (EU)",
            ]
        ):
            worksheet.cell(row=2, column=1 + col_idx, value=val)
        # Fill out values
        channel_attr_list = Channel().channel_attr_list
        for row, channel in enumerate(channel_list):
            row_idx = row + 3
            worksheet.cell(row=row_idx, column=1, value=row)
            for col, attr in enumerate(channel_attr_list):
                col_idx = col + 2
                val = getattr(channel, attr)
                val = str(val) if val is not None else ""
                worksheet.cell(row=row_idx, column=col_idx, value=val)

    @classmethod
    def load_channel_table_from_workbook(
        cls, workbook: openpyxl.workbook.workbook.Workbook
    ) -> List[Channel]:
        """
        Load the channel table from a workbook.

        If the workbook has more than one sheet, the channel table is the sheet with
        ``"channel"`` in its name. Channels are read starting at the third row until the
        first empty row.

        Parameters
        ----------
        workbook : openpyxl.workbook.workbook.Workbook
            Workbook containing a channel table sheet.

        Returns
        -------
        List[Channel]
            Channels loaded from the channel table.

        Raises
        ------
        RattlesnakeError
            If the workbook has more than one channel table sheet or none.

        Unit Tests
        ----------
        """
        sheets = workbook.sheetnames

        if len(sheets) > 1:
            sheets = [sheet for sheet in sheets if "channel" in sheet.lower()]
        if len(sheets) > 1:
            raise RattlesnakeError(
                "Multiple channel table sheets located in Excel Spreadsheet"
            )
        if len(sheets) == 0:
            raise RattlesnakeError(
                "Excel Spreadsheet does not contain a channel table sheet"
            )

        worksheet = workbook[sheets[0]]

        channel_list = []
        channel_attr_list = Channel().channel_attr_list
        for row in worksheet.iter_rows(min_row=3, min_col=2, max_col=23):
            channel = Channel()
            for col, cell in enumerate(row):
                value = worksheet_cell_str(cell.value, default=None)
                setattr(channel, channel_attr_list[col], value)
            if channel.is_empty:
                break
            channel_list.append(channel)

        return channel_list

    @classmethod
    def save_blank_hardware_to_workbook(
        cls, workbook: openpyxl.workbook.workbook.Workbook
    ) -> None:
        """
        Write a blank hardware sheet template to a workbook.

        Writes the ``"Hardware"`` sheet labels in the first column and instructions for
        filling in each value. The sheet is created if it does not exist.

        This currently implements every possible hardware attribute even if it is unused
        by that specific hardware type.

        Parameters
        ----------
        workbook : openpyxl.workbook.workbook.Workbook
            Workbook to write the hardware sheet to.

        Unit Tests
        ----------
        """
        if "Hardware" in workbook.sheetnames:
            hardware_worksheet = workbook["Hardware"]
        else:
            hardware_worksheet = workbook.create_sheet("Hardware")
        hardware_worksheet.cell(1, 1, "Hardware Type")
        hardware_worksheet.cell(1, 2, "# Enter hardware index here")
        hardware_worksheet.cell(
            1,
            3,
            "Hardware Indices: 0 - NI DAQmx; 1 - LAN XI; 2 - Data Physics Quattro; "
            "3 - Data Physics 900 Series; 4 - Exodus Modal Solution; 5 - State Space Integration; "
            "6 - SDynPy System Integration",
        )
        hardware_worksheet.cell(2, 1, "Hardware File")
        hardware_worksheet.cell(
            2,
            3,
            "# Path to Hardware File (Depending on Hardware Device: 0 - Not Used; 1 - Not Used; "
            "2 - Path to DpQuattro.dll library file; 3 - Not Used; 4 - Path to Exodus Eigensolution; "
            "5 - Path to State Space File; 6 - Path to SDynPy system file)",
        )
        hardware_worksheet.cell(3, 1, "Sample Rate")
        hardware_worksheet.cell(3, 3, "# Sample Rate of Data Acquisition System")
        hardware_worksheet.cell(4, 1, "Time Per Read")
        hardware_worksheet.cell(
            4, 3, "# Number of seconds per Read from the Data Acquisition System"
        )
        hardware_worksheet.cell(5, 1, "Time Per Write")
        hardware_worksheet.cell(
            5, 3, "# Number of seconds per Write to the Data Acquisition System"
        )
        hardware_worksheet.cell(6, 1, "Maximum Acquisition Processes")
        hardware_worksheet.cell(
            6,
            3,
            "# Maximum Number of Acquisition Processes to start to pull data from hardware",
        )
        hardware_worksheet.cell(
            6,
            4,
            "Only Used by LAN-XI Hardware.  This row can be deleted if LAN-XI is not used",
        )
        hardware_worksheet.cell(7, 1, "Integration Oversampling")
        hardware_worksheet.cell(
            7, 3, "# For virtual control, an integration oversampling can be specified"
        )
        hardware_worksheet.cell(
            7,
            3,
            "Only used for virtual control (Exodus, State Space, or SDynPy).  "
            "This row can be deleted if these are not used.",
        )
        hardware_worksheet.cell(8, 1, "Task Trigger")
        hardware_worksheet.cell(8, 3, "# Start trigger type")
        hardware_worksheet.cell(
            8,
            3,
            "Task Triggers: 0 - Internal, 1 - PFI0 with external trigger, 2 - PFI0 with Analog Output "
            "trigger.  Only used for NI hardware.  This row can be deleted if NI is not used.",
        )
        hardware_worksheet.cell(9, 1, "Task Trigger Output Channel")
        hardware_worksheet.cell(
            9, 3, "# Physical device and channel that generates a trigger signal"
        )
        hardware_worksheet.cell(
            9,
            4,
            "Only used if Task Triggers is 2.  Only used for NI hardware.  "
            "This row can be deleted if it is not used.",
        )
        hardware_worksheet.cell(10, 1, "Damping Ratio")
        hardware_worksheet.cell(
            10,
            3,
            "# Damping ratio for Exodus Modal Solution virtual hardware. Only fill out if you "
            "are using Exodus Modal Solution virtual hardware.",
        )

    @classmethod
    @abstractmethod
    def load_metadata_from_workbook(
        cls, workbook: openpyxl.workbook.workbook.Workbook
    ) -> Tuple[HardwareType, List[Channel], int, float, float, int]:
        """
        Load the metadata shared by all hardware from a workbook.

        Reads the channel table and the "Hardware" sheet, where each row has a label
        in the first column and a value in the second. Values that are missing keep
        their defaults. Subclasses should call this method, read their own values, and
        return an instance of the subclass.

        Parameters
        ----------
        workbook : openpyxl.workbook.workbook.Workbook
            Workbook containing a channel table and a ``"Hardware"`` sheet.

        Returns
        -------
        hardware_type : HardwareType
            Type of hardware device. Defaults to ``HardwareType.NONE``.
        channel_list : List[Channel]
            Channels loaded from the channel table.
        sample_rate : int
            Sample rate in samples per second. Defaults to ``1000``.
        time_per_read : float
            Time, in seconds, of each read frame. Defaults to ``0.1``.
        time_per_write : float
            Time, in seconds, of each write frame. Defaults to ``0.1``.
        output_oversample : int
            Integration oversampling factor. Defaults to ``1``.

        Raises
        ------
        RattlesnakeError
            If the workbook has more than one channel table sheet or none.
        KeyError
            If the workbook does not have a ``"Hardware"`` sheet.

        Unit Tests
        ----------
        """
        channel_list = cls.load_channel_table_from_workbook(workbook)

        # This is a holdover from previous worksheet logic
        # Need default values
        hardware_type = HardwareType.NONE
        sample_rate = 1000
        time_per_read = 0.1
        time_per_write = 0.1
        output_oversample = 1

        # Hardware
        hardware_sheet = workbook["Hardware"]
        for row in hardware_sheet.rows:
            name = str(row[0].value).lower().strip().replace(" ", "_")
            value = row[1].value
            if value is None or value == "":
                continue
            match name:
                case "hardware_type":
                    hardware_type_int = int(value)
                    hardware_type = HardwareType(hardware_type_int)
                case "sample_rate":
                    sample_rate = int(value)
                case "time_per_read":
                    time_per_read = float(value)
                case "time_per_write":
                    time_per_write = float(value)
                case "integration_oversampling":
                    output_oversample = int(value)
                case _:
                    continue

        return (
            hardware_type,
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
            output_oversample,
        )

    @abstractmethod
    def save_metadata_to_netcdf(self, netcdf_dataset: nc4.Dataset) -> None:
        """
        Save the metadata attributes shared by all hardware devices to a netCDF4
        dataset.

        Creates the channel and time sample dimensions, the ``time_data`` streaming
        variable, the hardware attributes, and a channel table ``channels`` group.
        Subclasses should call this method and then save their own attributes.

        Parameters
        ----------
        netcdf_dataset : netCDF4.Dataset
            An open, writable netCDF4 dataset.

        Unit Tests
        ----------
        """
        stream_variable = "time_data"
        stream_dimension = "time_samples"
        # Create dimensions
        netcdf_dataset.createDimension("response_channels", len(self.channel_list))
        netcdf_dataset.createDimension(
            "output_channels",
            len(
                [
                    channel
                    for channel in self.channel_list
                    if channel.feedback_device is not None
                ]
            ),
        )
        netcdf_dataset.createDimension(stream_dimension, None)
        # Create attributes
        netcdf_dataset.file_version = "3.0.0"
        netcdf_dataset.sample_rate = self.sample_rate
        netcdf_dataset.time_per_write = self.samples_per_write / self.output_sample_rate
        netcdf_dataset.time_per_read = self.samples_per_read / self.sample_rate
        netcdf_dataset.hardware = self.hardware_type.value
        netcdf_dataset.output_oversample = self.output_oversample
        # Create Variables
        netcdf_dataset.createVariable(
            stream_variable, "f8", ("response_channels", stream_dimension)
        )

        # Create channel table variables
        labels = [
            ["node_number", str],
            ["node_direction", str],
            ["comment", str],
            ["serial_number", str],
            ["triax_dof", str],
            ["sensitivity", str],
            ["unit", str],
            ["make", str],
            ["model", str],
            ["expiration", str],
            ["physical_device", str],
            ["physical_channel", str],
            ["channel_type", str],
            ["minimum_value", str],
            ["maximum_value", str],
            ["coupling", str],
            ["excitation_source", str],
            ["excitation", str],
            ["feedback_device", str],
            ["feedback_channel", str],
            ["warning_level", str],
            ["abort_level", str],
        ]
        for label, netcdf_datatype in labels:
            var = netcdf_dataset.createVariable(
                "/channels/" + label, netcdf_datatype, ("response_channels",)
            )
            channel_data = [getattr(channel, label) for channel in self.channel_list]
            if netcdf_datatype == "i1":
                channel_data = np.array([1 if val else 0 for val in channel_data])
            else:
                channel_data = ["" if val is None else val for val in channel_data]
            for i, cd in enumerate(channel_data):
                var[i] = str(cd)

    @classmethod
    @abstractmethod
    def load_metadata_from_netcdf(
        cls, netcdf_dataset: nc4.Dataset
    ) -> Tuple[HardwareType, List[Channel], int, float, float, int]:
        """
        Load the metadata shared by all hardware from a netCDF4 dataset.

        Subclasses should call this method, read their own attributes, and return an
        instance of the subclass.

        Parameters
        ----------
        netcdf_dataset : netCDF4.Dataset
            An open netCDF4 dataset saved by ``save_metadata_to_netcdf``.

        Returns
        -------
        hardware_type : HardwareType
            Type of hardware device.
        channel_list : List[Channel]
            Channels loaded from the ``channels`` group.
        sample_rate : int
            Sample rate in samples per second.
        time_per_read : float
            Time, in seconds, of each read frame.
        time_per_write : float
            Time, in seconds, of each write frame.
        output_oversample : int
            Integration oversampling factor.

        Unit Tests
        ----------
        """
        channel_list = cls.load_channel_table_from_netcdf(netcdf_dataset)

        # Hardware
        hardware_type = HardwareType(netcdf_dataset.hardware)
        sample_rate = int(netcdf_dataset.sample_rate)
        time_per_read = float(netcdf_dataset.time_per_read)
        time_per_write = float(netcdf_dataset.time_per_write)
        output_oversample = int(netcdf_dataset.output_oversample)

        return (
            hardware_type,
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
            output_oversample,
        )

    @abstractmethod
    def save_metadata_to_workbook(
        self, workbook: openpyxl.workbook.workbook.Workbook
    ) -> None:
        """
        Save the metadata shared by all hardware to a workbook.

        Writes the channel table to the active sheet, and the hardware type, sample
        rate, time per read, and time per write to the second column of the "Hardware"
        sheet. The hardware sheet is created if it does not exist. Subclasses should
        call this method and then save their own values.

        Parameters
        ----------
        workbook : openpyxl.workbook.workbook.Workbook
            Workbook to write the metadata to.

        Unit Tests
        ----------
        """
        self.save_channel_table_to_workbook(self.channel_list, workbook)

        if "Hardware" in workbook.sheetnames:
            hardware_worksheet = workbook["Hardware"]
        else:
            hardware_worksheet = workbook.create_sheet("Hardware")

        # Fill out values
        hardware_type = self.hardware_type
        hardware_worksheet.cell(1, 2, str(hardware_type.value))
        hardware_worksheet.cell(3, 2, str(self.sample_rate))
        hardware_worksheet.cell(4, 2, str(self.time_per_read))
        hardware_worksheet.cell(5, 2, str(self.time_per_write))

    # endregion


# endregion


# region Acquisition
class HardwareAcquisition(ABC):
    """
    Abstract class defining the interface between the controller and acquisition.

    This class defines the interfaces between the controller and the
    data acquisition process for the hardware.  It is owned by the Acquisition
    process, and must define how to get data from the test hardware into the
    controller.

    Unit Tests
    ----------
    """

    @abstractmethod
    def initialize_hardware(self, metadata: HardwareMetadata) -> None:
        """
        Initialize the hardware and set up channels and sampling properties.

        The method must set up channels on the hardware corresponding to the channels in
        the channel list. It must also set the sampling rates and hardware specific
        properties.

        Parameters
        ----------
        metadata : HardwareMetadata
            Hardware specific metadata containing the sampling properties and channel
            list to store to the ``HardwareAcquisition``.

        Unit Tests
        ----------
        """

    @abstractmethod
    def start(self) -> None:
        """
        Start acquiring data from the hardware.

        Implements any hardware specific startup sequences required by the hardware
        device.

        Unit Tests
        ----------
        """

    @abstractmethod
    def read(self) -> np.ndarray:
        """
        Read a frame of data from the hardware.

        Returns
        -------
        np.ndarray
            A ``num_channels x samples_per_read`` array of acquired data.

        Unit Tests
        ----------
        """

    @abstractmethod
    def read_remaining(self) -> np.ndarray:
        """
        Read the rest of the data on the hardware after the acquisition stops.

        Returns
        -------
        np.ndarray
            A ``num_channels x num_samples`` array of the remaining acquired data.

        Unit Tests
        ----------
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the acquisition.

        Unit Tests
        ----------
        """

    @abstractmethod
    def close(self) -> None:
        """
        Close down the hardware.

        Unit Tests
        ----------
        """

    @abstractmethod
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
            Number of samples the acquisition must keep reading after the last output
            is written.

        Unit Tests
        ----------
        """


# endregion


# region Output
class HardwareOutput(ABC):
    """
    Abstract class defining the interface between the controller and output.

    This class defines the interfaces between the controller and the output process or
    source portion of the hardware. It is owned by the Output process, and must define
    how to write data from the control system to the hardware.

    Unit Tests
    ----------
    """

    @abstractmethod
    def initialize_hardware(self, metadata: HardwareMetadata) -> None:
        """
        Initialize the hardware and set up sources and sampling properties.

        The method must create channels on the hardware corresponding to the sources in
        the test. It must also set the sampling rates.

        Parameters
        ----------
        metadata : HardwareMetadata
            Hardware specific metadata that defines the sampling properties and channel
            list for a given hardware.

        Unit Tests
        ----------
        """

    @abstractmethod
    def start(self) -> None:
        """
        Start outputting data to the hardware.

        Unit Tests
        ----------
        """

    @abstractmethod
    def write(self, data: np.ndarray) -> None:
        """
        Write a frame of data to the hardware.

        Parameters
        ----------
        data : np.ndarray
            A ``num_channels x buffer_size`` array to write to the output hardware.

        Unit Tests
        ----------
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the output.

        Unit Tests
        ----------
        """

    @abstractmethod
    def close(self) -> None:
        """
        Close down the hardware.

        Unit Tests
        ----------
        """

    @abstractmethod
    def ready_for_new_output(self) -> bool:
        """
        Return whether the hardware is ready to accept a new frame of output.

        Returns
        -------
        bool
            ``True`` if the hardware should accept a new signal, otherwise ``False``.

        Unit Tests
        ----------
        """


# endregion
