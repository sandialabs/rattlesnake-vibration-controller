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

from rattlesnake.hardware.hardware_utilities import Channel


class DataAcquisitionParameters:
    """Container to hold the global data acquisition parameters of the controller"""

    def __init__(
        self,
        channel_list: List[Channel],
        sample_rate,
        samples_per_read,
        samples_per_write,
        hardware,
        hardware_file,
        environment_names,
        environment_bools,
        output_oversample,
        **extra_parameters,
    ):
        """Container to hold the global data acquisition parameters of the controller

        Parameters
        ----------
        channel_list : List[Channel]:
            An iterable containing Channel objects for each channel in the
            controller.
        sample_rate : int :
            Number of samples per second that the data acquisition runs at
        samples_per_read : int :
            Number of samples the data acquisition will acquire with each read.
            Smaller numbers here will result in finer resolution for starting
            and stopping environments, but will be more computationally
            intensive to run.
        samples_per_write : int :
            Number of samples the data acquisition will output with each write.
            Smaller numbers here will result in finer resolution for starting
            and stopping environments, but will be more computationally
            intensive to run.
        hardware : int :
            Hardware index corresponding to the QCombobox selector on the
            Channel Table tab of the main controller
        hardware_file : str :
            Path to an optional file that completes the hardware specification,
            for example, a finite element model results.
        environment_names : List[str]:
            A list of the names of environments in the controller
        environment_bools : np.ndarray :
            A 2D array specifying which channels are active in which environment.
            If the [i,j] component is True, then the ith channel is active in
            the jth environment.
        output_oversample : int
            Oversample factor of the output generator
        maximum_acquisition_processes : int
            Maximum number of processes to spin up to read data off the acquisition
        """
        self.channel_list = channel_list
        self.sample_rate = sample_rate
        self.samples_per_read = samples_per_read
        self.samples_per_write = samples_per_write
        self.hardware = hardware
        self.hardware_file = hardware_file
        self.environment_names = environment_names
        self.environment_active_channels = environment_bools
        self.output_oversample = output_oversample
        self.extra_parameters = extra_parameters

    @property
    def nyquist_frequency(self):
        """Property returning the Nyquist frequency of the data acquisition."""
        return self.sample_rate / 2

    @property
    def output_sample_rate(self):
        """Property returning the output sample rate"""
        return self.sample_rate * self.output_oversample


# region: Acqusition
class HardwareAcquisition(ABC):
    """Abstract class defining the interface between the controller and acquisition

    This class defines the interfaces between the controller and the
    data acquisition portion of the hardware.  It is run by the Acquisition
    process, and must define how to get data from the test hardware into the
    controller."""

    @abstractmethod
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

    @abstractmethod
    def start(self):
        """Method to start acquiring data from the hardware"""

    @abstractmethod
    def read(self) -> np.ndarray:
        """Method to read a frame of data from the hardware that returns
        an appropriately sized np.ndarray"""

    @abstractmethod
    def read_remaining(self) -> np.ndarray:
        """Method to read the rest of the data on the acquisition from the hardware
        that returns an appropriately sized np.ndarray"""

    @abstractmethod
    def stop(self):
        """Method to stop the acquisition"""

    @abstractmethod
    def close(self):
        """Method to close down the hardware"""

    @abstractmethod
    def get_acquisition_delay(self) -> int:
        """Get the number of samples between output and acquisition

        This function is designed to handle buffering done in the output
        hardware, ensuring that all data written to the output is read by the
        acquisition.  If a output hardware has a buffer, there may be a non-
        negligable delay between when output is written to the device and
        actually played out from the device."""


# region: Output
class HardwareOutput(ABC):
    """Abstract class defining the interface between the controller and output

    This class defines the interfaces between the controller and the
    output or source portion of the hardware.  It is run by the Output
    process, and must define how to get write data to the hardware from the
    control system"""

    @abstractmethod
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

    @abstractmethod
    def start(self):
        """Method to start outputting data to the hardware"""

    @abstractmethod
    def write(self, data):
        """Method to write a np.ndarray with a frame of data to the hardware"""

    @abstractmethod
    def stop(self):
        """Method to stop the output"""

    @abstractmethod
    def close(self):
        """Method to close down the hardware"""

    @abstractmethod
    def ready_for_new_output(self) -> bool:
        """Method that returns true if the hardware should accept a new signal"""
