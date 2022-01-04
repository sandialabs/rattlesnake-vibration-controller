# -*- coding: utf-8 -*-
"""
Controller subsystem that handles computation of FRFs

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
from .utilities import (flush_queue,DataAcquisitionParameters,VerboseMessageQueue,GlobalCommands)
import scipy.signal as sig
import numpy as np
from enum import Enum
from .abstract_message_process import AbstractMessageProcess
import time

WAIT_TIME = 0.05

class FRFMessages(Enum):
    """Collection of instructions that the FRF Computation Process might get"""
    INITIALIZE_DATA_ACQUISITION = 0
    INITIALIZE_TEST_PARAMETERS = 1
    RUN_FRF = 2
    CLEAR_FRF = 3
    SHOW_FRF = 4
    STOP_FRF = 5

class FRFComputationProcess(AbstractMessageProcess):
    """Class defining a subprocess that computes a FRF from a time history."""
    def __init__(self,process_name : str, 
                 frf_command_queue : VerboseMessageQueue,
                 data_for_frf_queue : mp.queues.Queue,
                 updated_frf_queue : mp.queues.Queue,
                 gui_update_queue : mp.queues.Queue,
                 log_file_queue : mp.queues.Queue,
                 environment_name : str):
        """
        Constructor for the FRF Computation Process
        
        Sets up the ``command_map`` and initializes internal data

        Parameters
        ----------
        process_name : str
            Name for the process that will be used in the Log file.
        frf_command_queue : VerboseMessageQueue :
            The queue containing instructions for the FRF process
        data_for_frf_queue : mp.queues.Queue :
            Queue containing input data for the FRF computation
        updated_frf_queue : mp.queues.Queue :
            Queue where frf process will put computed frfs
        gui_update_queue : mp.queues.Queue :
            Queue for gui updates
        log_file_queue : mp.queues.Queue :
            Queue for writing to the log file
        environment_name : str
            Name of the environment that controls this subprocess.

        """
        super().__init__(process_name,log_file_queue,frf_command_queue,gui_update_queue)
        self.map_command(FRFMessages.INITIALIZE_DATA_ACQUISITION,self.initialize_data_acquisition)
        self.map_command(FRFMessages.INITIALIZE_TEST_PARAMETERS,self.initialize_test_parameters)
        self.map_command(FRFMessages.RUN_FRF,self.run_frf)
        self.map_command(FRFMessages.CLEAR_FRF,self.clear_frf)
        self.map_command(FRFMessages.SHOW_FRF,self.show_frf)
        self.map_command(FRFMessages.STOP_FRF,self.stop_frf)
        self.environment_name = environment_name
        self.data_for_frf_queue = data_for_frf_queue
        self.frf_command_queue = frf_command_queue
        self.updated_frf_queue = updated_frf_queue
        self.average_scheme = None
        self.averages = None
        self.exponential_average_coefficient = None
        self.technique = None
        self.response_data = None
        self.reference_data = None
        self.samples_per_frame = None
        self.sample_rate = None
        self.num_control_channels = None
        self.num_output_channels = None
        self.frf = None
        
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
    
    def initialize_test_parameters(self,data):
        """Initializes the signal processing parameters from the environment.

        Parameters
        ----------
        data :
            Container containing the setting specific to the environment.

        """
        self.average_scheme = data.averaging_type
        self.averages = data.system_id_averages
        self.exponential_average_coefficient = data.averaging_coefficient
        self.technique = data.frf_technique
        self.samples_per_frame = data.samples_per_frame

        if not data.response_transformation_matrix is None:
            self.num_control_channels = data.response_transformation_matrix.shape[0]
        if not data.output_transformation_matrix is None:
            self.num_output_channels = data.output_transformation_matrix.shape[0]
        # Initialize the NaNs only if the FRF is the wrong size.
        if (self.response_data is None
            or self.response_data.shape != (self.averages,self.num_control_channels,data.fft_lines)
            or self.reference_data is None
            or self.reference_data.shape != (self.averages,self.num_output_channels,data.fft_lines)):
            self.response_data = np.nan*np.ones((self.averages,self.num_control_channels,
                                                      data.fft_lines),dtype=complex)
            self.reference_data = np.nan*np.ones((self.averages,self.num_output_channels,
                                                       data.fft_lines),dtype=complex)
    
    def run_frf(self,data):
        """Continuously compute FRFs from time histories.
        
        This function accepts data from the ``data_for_frf_queue`` and computes
        FRF matrices from the time data.  It uses a rolling buffer to append
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
        data = flush_queue(self.data_for_frf_queue,timeout = WAIT_TIME)
        if len(data) == 0:
            time.sleep(WAIT_TIME)
            self.frf_command_queue.put(self.process_name,(FRFMessages.RUN_FRF,None))
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
            self.frf_command_queue.put(self.process_name,(FRFMessages.RUN_FRF,None))
            return
        if self.technique == 'H1':
            # We want to compute X*F^H = [X1;X2;X3][F1^H F2^H F3^H]
            self.log('Computing H1 FRF')
            Gxf = np.einsum('aif,ajf->fij',self.response_data[~exclude_averages],np.conj(self.reference_data[~exclude_averages]))
            Gff = np.einsum('aif,ajf->fij',self.reference_data[~exclude_averages],np.conj(self.reference_data[~exclude_averages]))
            self.log('Computed Cross and Auto Spectra')
            # np.savez('test_data/debug/frf_data_check.npz',
            #         response_data = self.response_data,
            #         reference_data = self.reference_data,
            #         average_indices = self.time_indices_for_averages,
            #         Gxf = Gxf,
            #         Gff = Gff)
            # # Add small values to any matrices that are singular
            # singular_matrices = np.abs(np.linalg.det(Gff)) < 2*np.finfo(Gff.dtype).eps
            # Gff[singular_matrices] += np.eye(Gff.shape[-1])*np.finfo(Gff.dtype).eps
            try:
                Gffpinv = np.linalg.pinv(Gff,rcond=1e-12,hermitian=True)
                self.frf = Gxf@Gffpinv
                self.log('Computed FRF')
            except np.linalg.LinAlgError:
                self.log('Singular Matrix in FRF Calculation.')
        else:
            raise NotImplementedError('Method {:} has not been implemented yet!'.format(self.technique))
        
        frf_frames = self.averages - np.sum(exclude_averages)
        
        if not self.frf is None:
            self.log('Sending Updated FRF')
            self.updated_frf_queue.put((self.frf,frf_frames))
            self.gui_update_queue.put((self.environment_name,('FRF',(np.fft.rfftfreq(self.samples_per_frame,1/self.sample_rate),self.frf))))
            self.log('Updated FRFs Sent')
        # Keep running
        self.frf_command_queue.put(self.process_name,(FRFMessages.RUN_FRF,None))
        
    def clear_frf(self,data):
        """Clears all data in the buffer so the FRF starts fresh from new data

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.response_data[:] = np.nan
        self.reference_data[:] = np.nan
        self.frf = None
    
    def show_frf(self,data):
        """Sends FRF information to the GUI update queue to be plotted

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.gui_update_queue.put((self.environment_name,('FRF',(np.fft.rfftfreq(self.samples_per_frame,1/self.sample_rate),self.frf))))
    
    def stop_frf(self,data):
        """Stops computing FRFs from time data.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        time.sleep(WAIT_TIME)
        flushed_data = self.frf_command_queue.flush(self.process_name)
        # Put back any quit message that may have been pulled off
        for message,data in flushed_data:
            if message == GlobalCommands.QUIT:
                self.frf_command_queue.put(self.process_name,(message,data))
        flush_queue(self.updated_frf_queue)

def frf_computation_process(environment_name : str,
                            frf_command_queue : VerboseMessageQueue,
                            data_for_frf_queue : mp.queues.Queue,
                            updated_frf_queue : mp.queues.Queue,
                            gui_update_queue : mp.queues.Queue,
                            log_file_queue : mp.queues.Queue,
                            ):
    """Function passed to multiprocessing as the FRF computation process
    
    This process creates the ``FRFComputationProcess`` object and calls the
    ``run`` function.


    Parameters
    ----------
    environment_name : str :
        Name of the environment that this subprocess belongs to.
    frf_command_queue : VerboseMessageQueue :
        The queue containing instructions for the FRF process
    data_for_frf_queue : mp.queues.Queue :
        Queue containing input data for the FRF computation
    updated_frf_queue : mp.queues.Queue :
        Queue where frf process will put computed frfs
    gui_update_queue : mp.queues.Queue :
        Queue for gui updates
    log_file_queue : mp.queues.Queue :
        Queue for writing to the log file

    """
    
    frf_computation_instance = FRFComputationProcess(environment_name + ' FRF Computation', 
                                                     frf_command_queue,
                                                     data_for_frf_queue,
                                                     updated_frf_queue,
                                                     gui_update_queue,
                                                     log_file_queue, environment_name)
    
    frf_computation_instance.run()
