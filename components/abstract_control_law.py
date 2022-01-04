# -*- coding: utf-8 -*-
"""
Abstract base law that new Random Vibration Environment control laws can inherit
from.

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
import numpy as np

class AbstractControlClass(ABC):
    @abstractmethod
    def __init__(self,specification : np.ndarray, # The specification to control to
                 extra_control_parameters : str, # Extra parameters specified by the controller
                 transfer_function : np.ndarray = None,  # Transfer Functions
                 buzz_cpsd : np.ndarray = None, # Buzz test in case cross terms are to be computed
                 last_response_cpsd : np.ndarray = None, # Last Response for Error Correction
                 last_output_cpsd : np.ndarray = None, # Last output for Drive-based control
                 ):
        """
        Initializes the control law
        
        Note that to facilitate the updating of the control law while the test
        is running, the init function will take all control data.  If control
        is not running, these will be Nones.

        Parameters
        ----------
        specification : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x control channels representing the specification defined
            by a CPSD matrix
        extra_control_parameters : str
            A string containing any extra parameters that might need to be
            passed to the control law.  This should be parsed by the __init__
            function.
        transfer_function : np.ndarray, optional
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x excitation sources representing the FRF matrix measured
            by the system identification process between drive voltages and
            control response. Will only be passed if the control is switched
            mid-run.  The default is None.
        buzz_cpsd : np.ndarray, optional
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x control channels representing the CPSD matrix measured
            by the system identification process.  Will only be passed if the
            control is switched mid-run.  The default is None.
        last_response_cpsd : np.ndarray, optional
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x control channels representing the last CPSD matrix of 
            control channel responses.  Will only be passed if the
            control is switched mid-run.  The default is None.
        last_output_cpsd : np.ndarray, optional
            A complex 3d numpy ndarray with dimensions frequency lines x drive
            channels x drive channels representing the last CPSD matrix of 
            drive outputs.  Will only be passed if the
            control is switched mid-run.  The default is None.
        """
        pass
    
    @abstractmethod
    def system_id_update(self,transfer_function : np.ndarray, # The transfer function from the system identification
                         buzz_cpsd : np.ndarray # The CPSD from the system identification
                         ):
        """
        Updates the control law with the data from the system identification

        Parameters
        ----------
        transfer_function : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x excitation sources representing the FRF matrix measured
            by the system identification process between drive voltages and
            control response
        buzz_cpsd : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x control channels representing the CPSD matrix measured
            by the system identification process

        """
        pass
    
    @abstractmethod
    def control(self,transfer_function : np.ndarray, # The last update of the transfer function
                last_response_cpsd : np.ndarray = None,  # Last Response for Error Correction
                last_output_cpsd : np.ndarray = None # Last output for Drive-based control
                ) -> np.ndarray:
        """
        Perform the control operations

        Parameters
        ----------
        transfer_function : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x excitation sources representing the FRF matrix measured
            by the system identification process between drive voltages and
            control response
        last_response_cpsd : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x control channels representing the last CPSD matrix of 
            control channel responses.  If no previous data exists (first time
            through control) it will be None.  The default is None.
        last_output_cpsd : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x drive
            channels x drive channels representing the last CPSD matrix of 
            drive outputs.  If no previous data exists (first time
            through control) it will be None.  The default is None.

        Returns
        -------
        next_output_cpsd : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x drive
            channels x drive channels representing the new CPSD matrix of 
            drive outputs that should be played to the shakers.
        """
        pass