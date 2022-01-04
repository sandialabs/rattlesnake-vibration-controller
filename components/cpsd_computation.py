# -*- coding: utf-8 -*-
"""
Controller subsystem that handles computing CPSD matrices from the responses and
output signals.

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

from .utilities import flush_queue,DataAcquisitionParameters
import numpy as np
import scipy.signal as sig
from enum import Enum
from .abstract_message_process import AbstractMessageProcess
from .random_vibration_environment import RandomEnvironmentQueues,RandomVibrationParameters
import time
        
WAIT_TIME = 0.05

class CPSDMessages(Enum):
    """Collection of instructions that the CPSD Computation Process might get"""
    INITIALIZE_DATA_ACQUISITION = 0
    INITIALIZE_TEST_PARAMETERS = 1
    RUN_CPSD = 2
    CLEAR_CPSD = 3
    STOP_CPSD = 4

class CPSDComputationProcess(AbstractMessageProcess):
    """Class defining a subprocess that computes a CPSD from a time history."""
    def __init__(self,process_name : str, queues : RandomEnvironmentQueues):
        """
        Constructor for the CPSD Computation Process
        
        Sets up the ``command_map`` and initializes internal data

        Parameters
        ----------
        process_name : str
            Name for the process that will be used in the Log file.
        queues : RandomEnvironmentQueues
            A container containing the queues for communication within the
            Random Vibration Environment

        """
        super().__init__(process_name,queues.log_file_queue,queues.cpsd_command_queue,queues.gui_update_queue)
        self.map_command(CPSDMessages.INITIALIZE_DATA_ACQUISITION,self.initialize_data_acquisition)
        self.map_command(CPSDMessages.INITIALIZE_TEST_PARAMETERS,self.initialize_test_parameters)
        self.map_command(CPSDMessages.RUN_CPSD,self.run_cpsd)
        self.map_command(CPSDMessages.CLEAR_CPSD,self.clear_cpsd)
        self.map_command(CPSDMessages.STOP_CPSD,self.stop_cpsd)
        self.queue_container = queues
        self.averages = None
        self.window = None
        self.window_correction = None
        self.response_data = None
        self.reference_data = None
        self.samples_per_frame = None
        self.sample_rate = None
        self.num_control_channels = None
        self.num_output_channels = None
        self.response_cpsd = None
        self.reference_cpsd = None
        
    def initialize_data_acquisition(self,data : DataAcquisitionParameters):
        """Sets up the process based off of the global data acquisition settings

        Parameters
        ----------
        data : DataAcquisitionParameters :
            Global data acquisition settings, including the sampling rate and
            channel information.

        """
        self.log('Initializing Data Acquisition')
        self.num_control_channels = len([channel for channel in data.channel_list if channel.control])
        self.num_output_channels = len([channel for channel in data.channel_list if not channel.feedback_device is None])
        self.sample_rate = data.sample_rate
    
    def initialize_test_parameters(self,data : RandomVibrationParameters):
        """Initializes the signal processing parameters from the environment.

        Parameters
        ----------
        data : RandomVibrationParameters :
            Container containing the setting specific to the environment.

        """
        self.averages = data.frames_in_cpsd
        self.samples_per_frame = data.samples_per_frame
        self.window = sig.windows.get_window(data.cpsd_window.lower(),self.samples_per_frame,fftbins=True)
        self.window_correction = 1/np.mean(self.window**2)
        self.frequency_spacing = data.frequency_spacing
        if not data.response_transformation_matrix is None:
            self.num_control_channels = data.response_transformation_matrix.shape[0]
        if not data.output_transformation_matrix is None:
            self.num_output_channels = data.output_transformation_matrix.shape[0]
        # Initialize as NaNs only if the CPSD is the wrong size
        if (self.response_data is None
            or self.response_data.shape != (self.averages,self.num_control_channels,data.fft_lines)
            or self.reference_data is None
            or self.reference_data.shape != (self.averages,self.num_output_channels,data.fft_lines)):
            self.response_data = np.nan*np.ones((self.averages,self.num_control_channels,
                                                      data.fft_lines),dtype=complex)
            self.reference_data = np.nan*np.ones((self.averages,self.num_output_channels,
                                                       data.fft_lines),dtype=complex)
        
    
    def run_cpsd(self,data):
        """Continuously compute CPSDs from time histories.
        
        This function accepts data from the ``data_for_cpsd_queue`` and computes
        CPSD matrices from the time data.  It uses a rolling buffer to append
        data.  The oldest data is pushed out of the buffer by the newest data.
        The test level is also passed with the response data and output
        data.  The test level is used to ensure that no frame uses
        discontinuous data.  

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        data = flush_queue(self.queue_container.data_for_cpsd_queue,timeout = WAIT_TIME)
        if len(data) == 0:
            time.sleep(WAIT_TIME)
            self.queue_container.cpsd_command_queue.put(self.process_name,(CPSDMessages.RUN_CPSD,None))
            return
        frames_received = len(data)
        response_data,reference_data = [value for value in zip(*data)]
        self.log('Received {:} Frames'.format(frames_received))
        self.response_data = np.concatenate((self.response_data[frames_received:],response_data),axis=0)
        self.reference_data = np.concatenate((self.reference_data[frames_received:],reference_data),axis=0)
        self.log('Buffered Frames (Resp Shape: {:}, Ref Shape: {:})'.format(self.response_data.shape,self.reference_data.shape))
        # Exclude any with NaNs
        exclude_averages = np.any(np.isnan(self.response_data),axis=(-1,-2))
        self.log('Computed Number Averages {:}'.format((~exclude_averages).sum()))
        # Return if there is actually no data
        if np.all(exclude_averages):
            self.queue_container.cpsd_command_queue.put(self.process_name,(CPSDMessages.RUN_CPSD,None))
            return
        # Compute spectral matrices
        #TODO: Clean up CPSDs to remove negative eigenvalues
        self.log('Computing CPSD Matrices')
        response_spectral_matrix = np.einsum('aif,ajf->fij',
             self.response_data[~exclude_averages],
             np.conj(self.response_data[~exclude_averages]))/self.response_data[~exclude_averages].shape[0]
        reference_spectral_matrix = np.einsum('aif,ajf->fij',
             self.reference_data[~exclude_averages],
             np.conj(self.reference_data[~exclude_averages]))/self.reference_data[~exclude_averages].shape[0]
        # Normalize
        response_spectral_matrix *= (self.frequency_spacing*self.window_correction/
                                     self.sample_rate**2)
        response_spectral_matrix[1:-1] *= 2
        reference_spectral_matrix *= (self.frequency_spacing*self.window_correction/
                                     self.sample_rate**2)
        reference_spectral_matrix[1:-1] *= 2
        self.log('Computed CPSD Matrices')
        self.response_cpsd = response_spectral_matrix
        self.reference_cpsd = reference_spectral_matrix
        cpsd_frames = self.averages - np.sum(exclude_averages)
        self.log('Sending Updated CPSDs')
        self.queue_container.updated_cpsd_queue.put((self.response_cpsd,self.reference_cpsd,cpsd_frames))
        self.log('Updated CPSDs Sent')
        # Keep running
        self.queue_container.cpsd_command_queue.put(self.process_name,(CPSDMessages.RUN_CPSD,None))
        
    def clear_cpsd(self,data):
        """Clears all data in the buffer so the CPSD starts fresh from new data

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.response_data[:] = np.nan
        self.reference_data[:] = np.nan
        self.response_cpsd = None
        self.reference_cpsd = None
    
    def stop_cpsd(self,data):
        """Stops computing CPSDs from time data.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        time.sleep(WAIT_TIME)
        self.queue_container.cpsd_command_queue.flush(self.process_name)
        flush_queue(self.queue_container.updated_cpsd_queue)

def cpsd_computation_process(environment_name : str,
                             queues : RandomEnvironmentQueues):
    """Function passed to multiprocessing as the CPSD computation process
    
    This process creates the ``CPSDComputationProcess`` object and calls the
    ``run`` function.


    Parameters
    ----------
    environment_name : str :
        Name of the environment that this subprocess belongs to.
    queues : RandomEnvironmentQueues :
        Set of queues that a random vibration environment uses to communicate

    """
    
    cpsd_computation_instance = CPSDComputationProcess(environment_name + ' CPSD Computation', queues)
    
    cpsd_computation_instance.run()
