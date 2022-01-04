# -*- coding: utf-8 -*-
"""
This subprocess of the Random Vibration environment collects data from the 
data in queue and sends it to the FRF and CPSD calculation scripts while also
tracking the test level.

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

from enum import Enum
from .abstract_message_process import AbstractMessageProcess
from .random_vibration_environment import RandomEnvironmentQueues,RandomVibrationParameters
from .utilities import DataAcquisitionParameters,flush_queue,OverlapBuffer
import multiprocessing as mp
import copy

import numpy as np
from scipy.fft import rfft
import scipy.signal as sig

class RandomDataCollectorMessages(Enum):
    """Commands that the Random Data Collector Process can accept"""
    INITIALIZE_DATA_ACQUISITION = 0
    INITIALIZE_TEST_PARAMETERS = 1
    ACQUIRE = 2
    STOP = 3
    SET_TEST_LEVEL = 4

class RandomDataCollectorProcess(AbstractMessageProcess):
    """Class that takes data from the data_in_queue and distributes to the environment
    
    This class keeps track of the test level used when acquiring data so the 
    data can be scaled back to full level for control.  It will also skip
    frames that are acquired while the system is ramping."""
    def __init__(self,process_name : str, queues : RandomEnvironmentQueues, environment_name):
        """
        Constructs the data collector class for a Random Vibration Environment

        Parameters
        ----------
        process_name : str
            A name to assign the process, primarily for logging purposes.
        queues : RandomEnvironmentQueues
            A list of Random Environment queues for communcation with other parts
            of the environment and the controller
        environment_name : str
            The name of the environment that this process is generating signals for.

        """
        super().__init__(process_name,queues.log_file_queue,queues.collector_command_queue,queues.gui_update_queue)
        self.map_command(RandomDataCollectorMessages.INITIALIZE_DATA_ACQUISITION,
                         self.initialize_data_acquisition)
        self.map_command(RandomDataCollectorMessages.INITIALIZE_TEST_PARAMETERS,
                         self.initialize_test_parameters)
        self.map_command(RandomDataCollectorMessages.ACQUIRE,self.acquire)
        self.map_command(RandomDataCollectorMessages.STOP,self.stop)
        self.map_command(RandomDataCollectorMessages.SET_TEST_LEVEL,self.set_test_level)
        self.environment_name = environment_name
        self.data_acquisition_parameters = None
        self.test_parameters = None
        self.skip_frames = 0
        self.test_level = None
        self.control_data_buffer = None
        self.output_data_buffer = None
        self.buffer_position = 0
        self.queue_container = queues
        self.control_channels = None
        self.output_channels = None
        self.cpsd_window = None
        self.frf_window = None
        
        
    def initialize_data_acquisition(self,data : DataAcquisitionParameters):
        """Stores global Data Acquisition Parameters to the process

        Parameters
        ----------
        data : DataAcquisitionParameters :
            The data acquisition parameters of the controller.

        """
        self.data_acquisition_parameters = data
        self.control_channels = [index for index,channel in enumerate(self.data_acquisition_parameters.channel_list) if channel.control]
        self.output_channels = [index for index,channel in enumerate(self.data_acquisition_parameters.channel_list) if not channel.feedback_device is None]
        
    def initialize_test_parameters(self,data : RandomVibrationParameters):
        """Stores environment signal processing parameters to the process

        Parameters
        ----------
        data : RandomVibrationParameters :
            Signal processing parameters for the environment

        """
        self.test_parameters = data
        # Set up the buffer
        # Only clear the buffer if the data processing parameters have changed, otherwise we will zero out some data.
        control_buffer_shape = (len(self.control_channels) if self.test_parameters.response_transformation_matrix is None else self.test_parameters.response_transformation_matrix.shape[0],
                                self.test_parameters.samples_per_frame*2)
        output_buffer_shape = (len(self.output_channels) if self.test_parameters.output_transformation_matrix is None else self.test_parameters.output_transformation_matrix.shape[0],
                                self.test_parameters.samples_per_frame*2)
        self.frf_window = sig.windows.get_window(self.test_parameters.frf_window.lower(),self.test_parameters.samples_per_frame,fftbins=True)
        self.cpsd_window = sig.windows.get_window(self.test_parameters.cpsd_window.lower(),self.test_parameters.samples_per_frame,fftbins=True)
        if (self.control_data_buffer is None or self.output_data_buffer is None or
            self.control_data_buffer.shape != control_buffer_shape or
            self.output_data_buffer.shape != output_buffer_shape):
            self.control_data_buffer = OverlapBuffer(control_buffer_shape)
            self.output_data_buffer = OverlapBuffer(output_buffer_shape)
        
    def acquire(self,data):
        """Acquires data from the data_in_queue and sends to the environment
        
        This function will take data and scale it by the test level, or skip
        sending the data if the test level is currently changing.  It will
        also apply the transformation matrices if they are defined.  
        
        It will stop itself if the last data is acquired.

        Parameters
        ----------
        data : Ignored
            Unused argument required due to the expectation that functions called
            by the RandomDataCollector.run function will have one argument
            accepting any data passed along with the instruction.
        """
        try:
            acquisition_data,last_data = self.queue_container.data_in_queue.get(timeout=10)
            self.log('Acquired Data')
        except mp.queues.Empty:
            # Keep running until stopped
#            self.log('No Incoming Data!')
            self.command_queue.put(self.process_name,(RandomDataCollectorMessages.ACQUIRE,None))
            return
        control_data = acquisition_data[self.control_channels]
        if not self.test_parameters.response_transformation_matrix is None:
            control_data = self.test_parameters.response_transformation_matrix@control_data
        output_data = acquisition_data[self.output_channels]
        if not self.test_parameters.output_transformation_matrix is None:
            output_data = self.test_parameters.output_transformation_matrix@output_data
        self.log('Parsed Channels')
        #        np.savez('test_data/collector_data_check.npz',output_data = output_data,acquisition_data = acquisition_data,output_channels = self.output_channels)
        # Send the data up to the GUI
        self.queue_container.gui_update_queue.put((self.environment_name,('time_data',(control_data,output_data))))
        self.log('Sent Data to GUI')
        if self.skip_frames > 0:
            self.skip_frames -= 1
            self.log('Skipped Frame, {:} left'.format(self.skip_frames))
            # Reset buffer positions to zero
            self.control_data_buffer.set_buffer_position()
            self.output_data_buffer.set_buffer_position()
        elif self.test_level != 0.0:
            # Add data to the buffer
            self.control_data_buffer.add_data(control_data)
            self.output_data_buffer.add_data(output_data)
            # Check if we have enough data to send the next dataset
            # print('\nBuffer Position: {:}'.format(self.buffer_position))
            while self.control_data_buffer.buffer_position >= self.test_parameters.samples_per_frame:
                self.log('Sending Data')
                control_data = self.control_data_buffer.get_data(
                    self.test_parameters.samples_per_frame,
                    -self.test_parameters.samples_per_acquire)
                output_data = self.output_data_buffer.get_data(
                    self.test_parameters.samples_per_frame,
                    -self.test_parameters.samples_per_acquire)
                self.log('Extracted Data, Computing FFTs')
                control_fft = rfft(control_data*self.frf_window/self.test_level,axis=-1)
                output_fft = rfft(output_data*self.frf_window/self.test_level,axis=-1)
                self.queue_container.data_for_frf_queue.put(copy.deepcopy(
                        (control_fft,
                         output_fft)))
                control_fft = rfft(control_data*self.cpsd_window/self.test_level,axis=-1)
                output_fft = rfft(output_data*self.cpsd_window/self.test_level,axis=-1)
                self.queue_container.data_for_cpsd_queue.put(copy.deepcopy(
                        (control_fft,
                         output_fft)))
                self.log('Sent Data')
        # Keep running until stopped
        if not last_data:
            self.command_queue.put(self.process_name,(RandomDataCollectorMessages.ACQUIRE,None))
        else:
            self.stop(None)
        
    def stop(self,data):
        """Stops acquiring data from the data_in_queue and flushes queues.

        Parameters
        ----------
        data : Ignored
            Unused argument required due to the expectation that functions called
            by the RandomDataCollector.run function will have one argument
            accepting any data passed along with the instruction.
        """
        self.log('Stopping Data Collection')
        flush_queue(self.queue_container.data_for_cpsd_queue)
        flush_queue(self.queue_container.data_for_frf_queue)
        self.command_queue.flush(self.process_name)
        self.control_data_buffer.set_buffer_position()
        self.output_data_buffer.set_buffer_position()
    
    def set_test_level(self,data):
        """Updates the value of the current test level due and sets the number
        of frames to skip.

        Parameters
        ----------
        data : tuple
            Tuple containing the number of frames to skip and the new test
            level

        """
        self.skip_frames,self.test_level = data
        self.log('Setting Test Level to {:}, skipping next {:} frames'.format(self.test_level,self.skip_frames))
        
def random_data_collector_process(environment_name : str,queue_container: RandomEnvironmentQueues):
    """Random vibration data collector process function called by multiprocessing
    
    This function defines the Random Vibration Data Collector process that
    gets run by the multiprocessing module when it creates a new process.  It
    creates a RandomDataCollectorProcess object and runs it.

    Parameters
    ----------
    environment_name : str :
        Name of the environment associated with this signal generation process
        
    queue_container: RandomEnvironmentQueues :
        A container of queues that allows communcation with the random vibration
        environment as well as the rest of the controller.
    """

    data_collector_instance = RandomDataCollectorProcess(environment_name + ' Data Collector',queue_container, environment_name)
    
    data_collector_instance.run()
