# -*- coding: utf-8 -*-
"""
Abstract hardware definition that can be used to implement new hardware devices

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
from typing import List

import numpy as np
import netCDF4 as nc4
import openpyxl

from rattlesnake.utilities import RattlesnakeError
from rattlesnake.hardware.hardware_utilities import Channel, HardwareType


class HardwareMetadata:
    """
    Abstract class that contains values to fully define how the hardware is setup.

    This class contains attributes required to run acquisition, output, and streaming
    processes. The class should also contain extra attributes required for the
    HardwareAcquisition and HardwareOutput class specific to that HardwareType.
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
    ):
        self.hardware_type = hardware_type
        self.channel_list = channel_list
        self.sample_rate = sample_rate
        self.time_per_read = time_per_read
        self.time_per_write = time_per_write
        # Used for virtual hardware but still required for normal hardware
        self.output_oversample = output_oversample

    @property
    def samples_per_read(self):
        """Property returning the number of samples per read frame."""
        return round(self.sample_rate * self.time_per_read)

    @property
    def samples_per_write(self):
        """Property returning the number of samples per write frame."""
        return round(self.sample_rate * self.time_per_write * self.output_oversample)

    @property
    def nyquist_frequency(self):
        """Property returning the Nyquist frequency of the data acquisition."""
        return self.sample_rate / 2

    @property
    def output_sample_rate(self):
        """Property returning the output sample rate."""
        return self.sample_rate * self.output_oversample

    @property
    @abstractmethod
    def extra_attr_list(self) -> List[str]:
        """
        Method that returns a list of extra attributes that should be stored to the
        netcdf4 output file so that the hardware can be defined when loading Rattlesnake
        from a file.
        """
        return []

    @abstractmethod
    def validate(self) -> True:
        """
        Method to check if the metadata object fully defines the hardware and is valid
        for that machine

        If possible should check which devices are connected to the machine at a given
        time and make sure that they are valid inputs to the initialize_hardware function
        of the HardwareAcquisition and HardwareOutput classes.

        Throw detailed errors while validating, these errors will show up in log files for
        debugging and will not stop the main process from running.
        """
        if len(self.channel_list) != len(set(self.channel_list)):
            raise RattlesnakeError("Duplicate channels found in channel_list")

        return True

    @abstractmethod
    def valid_channel_dict(self, channel: Channel):
        valid_dict = {}
        for attr in Channel().channel_attr_list:
            valid_dict[attr] = []
        return valid_dict

    @abstractmethod
    def store_to_netcdf(self, netcdf_group_handle: nc4._netCDF4.Group) -> None:
        pass

    @classmethod
    @abstractmethod
    def retrieve_metadata_from_netcdf(cls, netcdf_handle: nc4._netCDF4.Group):
        pass

    @abstractmethod
    def store_to_worksheet(self, worksheet: openpyxl.worksheet.worksheet.Worksheet):
        pass

    @classmethod
    @abstractmethod
    def retrieve_metadata_from_worksheet(
        cls, worksheet: openpyxl.worksheet.worksheet.Worksheet
    ):
        pass


# region: HardwareAcquisition
class HardwareAcquisition(ABC):
    """
    Abstract class defining the interface between the controller and acquisition.

    This class defines the interfaces between the controller and the
    data acquisition portion of the hardware.  It is run by the Acquisition
    process, and must define how to get data from the test hardware into the
    controller.
    """

    @abstractmethod
    def initialize_hardware(self, metadata: HardwareMetadata) -> None:
        """
        Initialize the hardware and set up channels and sampling properties.

        The function must create channels on the hardware corresponding to
        the channels in the test.  It must also set the sampling rates.

        Parameters
        ----------
        metadata : HardwareMetadata
            Hardware specific metadata class containing the sampling properties
            and channel list to store to the HardwareAcquisition.
        """

    @abstractmethod
    def start(self) -> None:
        """Method to start acquiring data from the hardware."""

    @abstractmethod
    def read(self) -> np.ndarray:
        """Method to read a frame of data from the hardware that returns
        an appropriately sized np.ndarray."""

    @abstractmethod
    def read_remaining(self) -> np.ndarray:
        """Method to read the rest of the data on the acquisition from the hardware
        that returns an appropriately sized np.ndarray."""

    @abstractmethod
    def stop(self) -> None:
        """Method to stop the acquisition."""

    @abstractmethod
    def close(self) -> None:
        """Method to close down the hardware."""

    @abstractmethod
    def get_acquisition_delay(self) -> int:
        """Get the number of samples between output and acquisition.

        This function is designed to handle buffering done in the output
        hardware, ensuring that all data written to the output is read by the
        acquisition.  If a output hardware has a buffer, there may be a non-
        negligable delay between when output is written to the device and
        actually played out from the device."""


# region: HardwareOutput
class HardwareOutput(ABC):
    """Abstract class defining the interface between the controller and output

    This class defines the interfaces between the controller and the
    output or source portion of the hardware.  It is run by the Output
    process, and must define how to get write data to the hardware from the
    control system"""

    @abstractmethod
    def initialize_hardware(self, metadata: HardwareMetadata) -> None:
        """
        Initialize the hardware and set up sources and sampling properties

        The function must create channels on the hardware corresponding to
        the sources in the test.  It must also set the sampling rates.

        Parameters
        ----------
        metadata : HardwareMetadata :
            Hardware specific metdata class that defines the sampling properties
            and channel list for a given hardware.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """Method to start outputting data to the hardware"""
        pass

    @abstractmethod
    def write(self, data) -> None:
        """
        Method to write a np.ndarray with a frame of data to the hardware

        Parameters
        ----------
        data : np.ndarray :
        num_channels x buffer_size array to write to the output hardware
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Method to stop the output"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Method to close down the hardware"""
        pass

    @abstractmethod
    def ready_for_new_output(self) -> bool:
        """Method that returns true if the hardware should accept a new signal"""
        pass
