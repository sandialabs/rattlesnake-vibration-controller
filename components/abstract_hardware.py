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

from abc import ABC,abstractmethod
from .utilities import Channel,DataAcquisitionParameters
from typing import List
import numpy as np

class HardwareAcquisition(ABC):
    """Abstract class defining the interface between the controller and acquisition
    
    This class defines the interfaces between the controller and the
    data acquisition portion of the hardware.  It is run by the Acquisition
    process, and must define how to get data from the test hardware into the
    controller."""
    
    @abstractmethod
    def set_up_data_acquisition_parameters_and_channels(self,
                                                        test_data : DataAcquisitionParameters,
                                                        channel_data : List[Channel]):
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
        pass
    
    @abstractmethod
    def start(self):
        """Method to start acquiring data from the hardware"""
        pass
    
    @abstractmethod
    def read(self) -> np.ndarray:
        """Method to read a frame of data from the hardware that returns
        an appropriately sized np.ndarray"""
        pass
    
    @abstractmethod
    def read_remaining(self) -> np.ndarray:
        """Method to read the rest of the data on the acquisition from the hardware
        that returns an appropriately sized np.ndarray"""
        pass
    
    @abstractmethod
    def stop(self):
        """Method to stop the acquisition"""
        pass
    
    @abstractmethod
    def close(self):
        """Method to close down the hardware"""
        pass
    
    @abstractmethod
    def get_acquisition_delay(self) -> int:
        """Get the number of samples between output and acquisition
        
        This function is designed to handle buffering done in the output
        hardware, ensuring that all data written to the output is read by the
        acquisition.  If a output hardware has a buffer, there may be a non-
        negligable delay between when output is written to the device and
        actually played out from the device."""
    
    

class HardwareOutput(ABC):
    """Abstract class defining the interface between the controller and output
    
    This class defines the interfaces between the controller and the
    output or source portion of the hardware.  It is run by the Output
    process, and must define how to get write data to the hardware from the
    control system"""
    
    @abstractmethod
    def set_up_data_output_parameters_and_channels(self,
                                                        test_data : DataAcquisitionParameters,
                                                        channel_data : List[Channel]):
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
        pass
    
    @abstractmethod
    def start(self):
        """Method to start outputting data to the hardware"""
        pass
    
    @abstractmethod
    def write(self,data):
        """Method to write a np.ndarray with a frame of data to the hardware"""
        pass
    
    @abstractmethod
    def stop(self):
        """Method to stop the output"""
        pass
    
    @abstractmethod
    def close(self):
        """Method to close down the hardware"""
        pass
    
    @abstractmethod
    def ready_for_new_output(self) -> bool:
        """Method that returns true if the hardware should accept a new signal"""
        pass