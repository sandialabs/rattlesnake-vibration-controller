# -*- coding: utf-8 -*-
"""
Hardware definition that allows for the Data Physics Quattro Device to be run
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

from .abstract_hardware import HardwareAcquisition,HardwareOutput
from .utilities import Channel,DataAcquisitionParameters,flush_queue
import numpy as np
from typing import List
import multiprocessing as mp
from .data_physics_interface import DPQuattro,QuattroCoupling,QuattroStatus
import time

BUFFER_SIZE_FACTOR = 3
SLEEP_FACTOR = 10

class DataPhysicsAcquisition(HardwareAcquisition):
    """Class defining the interface between the controller and Data Physics 
    hardware
    
    This class defines the interfaces between the controller and the
    Data Physics hardware that runs their open API.  It is run by the Acquisition
    process, and must define how to get data from the test hardware into the
    controller."""
    def __init__(self, dll_path : str, queue : mp.queues.Queue):
        """
        Initializes the data physics hardware interface.

        Parameters
        ----------
        dll_path : str
            Path to the DpQuattro.dll file that defines 
        queue : mp.queues.Queue
            Multiprocessing queue used to pass output data from the output task
            to the acquisition task because Quattro runs on a single processor

        Returns
        -------
        None.

        """
        self.quattro = DPQuattro(dll_path)
        self.buffer_size = 2**24
        self.input_channel_order = []
        self.input_couplings = [QuattroCoupling.DC_DIFFERENTIAL]*4
        self.input_ranges = [10.0]*4
        self.input_sensitivities = [1.0]*4
        self.output_channel_order = []
        self.output_ranges = [10.0]*2
        self.output_sensitivities = [1.0]*2
        self.data_acquisition_parameters = None
        self.output_data_queue = queue
        self.read_data = None
        self.time_per_read = None
    
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
        # Store data acquisition parameters for later
        self.data_acquisition_parameters = test_data
        self.time_per_read = test_data.samples_per_read/test_data.sample_rate
        
        # End the measurement if necessary
        if self.quattro.status == QuattroStatus.RUNNING:
            self.quattro.stop()
        if self.quattro.status == QuattroStatus.STOPPED or self.quattro.status == QuattroStatus.INIT:
            self.quattro.end()
        if self.quattro.status == QuattroStatus.DISCONNECTED:
            self.quattro.connect()
        
        # Set up defaults that will be overwritten
        self.input_channel_order = []
        self.input_couplings = [QuattroCoupling.DC_DIFFERENTIAL]*4
        self.input_ranges = [10.0]*4
        self.input_sensitivities = [1.0]*4
        self.output_channel_order = []
        self.output_ranges = [10.0]*2
        self.output_sensitivities = [1.0]*2
        
        # Set the sample rate
        self.quattro.set_sample_rate(test_data.sample_rate)
        
        # Set the buffer size ensuring that it is more that 4096
        # self.buffer_size = (BUFFER_SIZE_FACTOR+1)*max(test_data.samples_per_write,test_data.samples_per_read)
        # if self.buffer_size < 8192:
        #     self.buffer_size = 8192
        self.quattro.set_buffer_size(
            self.buffer_size
            )
        # print('Buffer Size: {:}'.format(self.buffer_size))
        
        # Set up channel parameters
        for channel in channel_data:
            # Figure out if the channel is an output channel or just acquisition
            is_output = not (channel.feedback_device is None) and not (channel.feedback_device.strip() == '')
            
            # Get the channel index from physical device
            channel_index = int(channel.physical_channel)-1
            
            # Track the channel order so we can rearrange measurements upon read
            self.input_channel_order.append(channel_index)
            
            # Set the values in the arrays appropriately given the channel index
            if channel.coupling.lower() in ['ac differential',
                                            'ac diff',
                                            'ac']:
                self.input_couplings[channel_index] = QuattroCoupling.AC_DIFFERENTIAL
            elif channel.coupling.lower() in ['dc differential',
                                             'dc diff',
                                             'dc']:
                self.input_couplings[channel_index] = QuattroCoupling.DC_DIFFERENTIAL
            elif channel.coupling.lower() in ['ac single ended',
                                              'ac single-ended',
                                              'ac single']:
                self.input_couplings[channel_index] = QuattroCoupling.AC_SINGLE_ENDED
            elif channel.coupling.lower() in ['dc single ended',
                                              'dc single-ended',
                                              'dc single']:
                self.input_couplings[channel_index] = QuattroCoupling.DC_SINGLE_ENDED
            elif channel.coupling.lower() in ['iepe','icp','ac icp','ccld']:
                self.input_couplings[channel_index] = QuattroCoupling.AC_COUPLED_IEPE
            self.input_sensitivities[channel_index] = 1.0 if is_output else float(channel.sensitivity)/1000
            self.input_ranges[channel_index] = 10.0 if is_output else float(channel.maximum_value)
            
            # Set up the output
            if is_output:
                channel_index = int(channel.feedback_channel)-1
                self.output_channel_order.append(channel_index)
                self.output_ranges[channel_index] = float(channel.maximum_value)
            
        # Now send the data to the quattro device
        self.quattro.setup_input_parameters(
            self.input_couplings,self.input_sensitivities,self.input_ranges)
        self.quattro.setup_output_parameters(
            self.output_sensitivities,self.output_ranges)
    
    def start(self):
        """Method to start acquiring data from the hardware"""
        self.read_data = []
        self.quattro.initialize()
        self.quattro.start()
    
    def read(self) -> np.ndarray:
        """Method to read a frame of data from the hardware that returns
        an appropriately sized np.ndarray"""
        while np.sum([data.shape[-1] for data in self.read_data]) < self.data_acquisition_parameters.samples_per_read:
            # Check if we need to output anything
            self.get_and_write_output_data()
            # Check how many samples are available
            samples_available = self.quattro.get_available_input_data_samples()
            # print('{:} Samples Available to Read'.format(samples_available))
            # Read that many data samples and put it to the "read_data" array
            # Make sure we rearrange the channels correctly per the rattlesnake
            # channel table using self.input_channel_order
            if samples_available > 0:
                self.read_data.append(self.quattro.read_input_data(samples_available)[self.input_channel_order])
            # Pause for a bit to allow more samples to accumulate
            time.sleep(self.time_per_read/SLEEP_FACTOR)
        # After we finish getting enough samples for a read, we can split the 
        # read data into the number of samples requested, and put the remainder
        # as the start of the next self.read_data list.
        read_data = np.concatenate(self.read_data,axis=-1)
        self.read_data = [read_data[:,self.data_acquisition_parameters.samples_per_read:]]
        return read_data[:,:self.data_acquisition_parameters.samples_per_read]
    
    def read_remaining(self) -> np.ndarray:
        """Method to read the rest of the data on the acquisition from the hardware
        that returns an appropriately sized np.ndarray"""
        # Check if we need to output anything
        self.get_and_write_output_data()
        # Check how many samples are available
        samples_available = 0
        # Wait until some arrive
        while samples_available == 0:
            samples_available = self.quattro.get_available_input_data_samples()
        # Read that many data samples and put it to the "read_data" array
        # Make sure we rearrange the channels correctly per the rattlesnake
        # channel table using self.input_channel_order
        self.read_data.append(self.quattro.read_input_data(samples_available)[self.input_channel_order])
        # Pause for a bit to allow more samples to accumulate
        time.sleep(self.time_per_read/SLEEP_FACTOR)
        # After we finish getting enough samples for a read, we can split the 
        # read data into the number of samples requested, and put the remainder
        # as the start of the next self.read_data list.
        read_data = np.concatenate(self.read_data,axis=-1)
        self.read_data = [read_data[:,self.data_acquisition_parameters.samples_per_read:]]
        read_data = np.concatenate(self.read_data,axis=-1)
        return read_data
    
    def stop(self):
        """Method to stop the acquisition"""
        self.quattro.stop()
        self.quattro.end()
    
    def close(self):
        """Method to close down the hardware"""
        self.quattro.disconnect()
    
    def get_acquisition_delay(self) -> int:
        """Get the number of samples between output and acquisition
        
        This function is designed to handle buffering done in the output
        hardware, ensuring that all data written to the output is read by the
        acquisition.  If a output hardware has a buffer, there may be a non-
        negligable delay between when output is written to the device and
        actually played out from the device."""
        return BUFFER_SIZE_FACTOR*self.data_acquisition_parameters.samples_per_write
    
    def get_and_write_output_data(self, block : bool = False):
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
        samples_on_buffer = self.quattro.get_total_output_samples_on_buffer()
        # print('{:} Samples on Output Buffer, (<{:} to output more)'.format(samples_on_buffer,self.data_acquisition_parameters.samples_per_write))
        if not block and samples_on_buffer >= self.data_acquisition_parameters.samples_per_write:
            # print('Too much data on buffer, not putting new data')
            return
        try:
            data = self.output_data_queue.get(block,timeout=10)
            # print('Got New Data from queue')
        except mp.queues.Empty:
            # print('Did not get new data from queue')
            if block:
                raise RuntimeError('Did not receive output in a reasonable amount of time, check output process and output hardware for issues')
            # Otherwise just return because there's no data available
            return
        # If we did get output, we need to put it into a numpy array that we can
        # send to the daq
        outputs = np.zeros((2,self.data_acquisition_parameters.samples_per_write))
        outputs[self.output_channel_order] = data
        # Send the outputs to the daq
        self.quattro.write_output_data(outputs)
        return

class DataPhysicsOutput(HardwareOutput):
    """Abstract class defining the interface between the controller and output
    
    This class defines the interfaces between the controller and the
    output or source portion of the hardware.  It is run by the Output
    process, and must define how to get write data to the hardware from the
    control system"""
    
    def __init__(self,queue : mp.queues.Queue):
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
    
    def start(self):
        """Method to start outputting data to the hardware"""
        pass
    
    def write(self,data):
        """Method to write a np.ndarray with a frame of data to the hardware"""
        self.queue.put(data)
    
    def stop(self):
        """Method to stop the output"""
        pass
    
    def close(self):
        """Method to close down the hardware"""
        pass
    
    def ready_for_new_output(self) -> bool:
        """Method that returns true if the hardware should accept a new signal
        
        Returns ``True`` if the data-passing queue is empty."""
        return self.queue.empty()