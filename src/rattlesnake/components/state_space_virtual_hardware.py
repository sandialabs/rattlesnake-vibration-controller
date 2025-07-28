# -*- coding: utf-8 -*-
"""
Synthetic "hardware" that allows the responses to be simulated by integrating
linear equations of motion using state space matrices, A, B, C, and D.

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
import scipy.signal as signal
from scipy.io import loadmat
import time
import os

class StateSpaceAcquisition(HardwareAcquisition):
    """Class defining the interface between the controller and synthetic acquisition
    
    This class defines the interfaces between the controller and the
    data acquisition portion of the hardware.  In this case, the hardware is
    actually simulated by integrating state space matrices, A, B, C, and D.
    It is run by the Acquisition process, and must define how to get data from
    the test hardware into the controller.
    """
    def __init__(self,state_space_file : str,queue : mp.queues.Queue):
        '''Loads in the state space file and sets initial parameters to null values
        

        Parameters
        ----------
        state_space_file : str : 
            Path to the file containing state space matrices A, B, C, and D.
        queue : mp.queues.Queue
            A queue that passes input data from the StateSpaceOutput class to
            this class.  Normally, this data transfer would occur through
            the physical test object: the exciters would excite the test object
            with the specified excitation and the Acquisition would record the
            responses to that excitation.  In the synthetic case, we need to
            pass the output data to the acquisition which does the integration.

        '''
        file_base,extension = os.path.splitext(state_space_file)

        if extension.lower() == '.npz':
            data = np.load(state_space_file)
        elif extension.lower() == '.mat':
            data = loadmat(state_space_file)
        else:
            raise ValueError('Unknown extension to file {:}, should be {:} or {:}, not {:}'.format(state_space_file,
                                                                                                   '.npz','.mat',extension))
        self.system = signal.StateSpace(data['A'],data['B'],data['C'],data['D'])
        self.times = None
        self.state = np.zeros(data['A'].shape[0])
        self.frame_time = None
        self.queue = queue
        self.force_buffer = None
        self.integration_oversample = None
        self.acquisition_delay = None
    
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
        self.create_response_channels(channel_data)
        self.set_parameters(test_data)
    
    def create_response_channels(self,channel_data : List[Channel]):
        """Method to set up response channels
        
        This function takes channels from the supplied list of channels and
        extracts the mode shape coefficients corresponding to those channels.

        Parameters
        ----------
        channel_data : List[Channel] :
            A list of ``Channel`` objects defining the channels in the test

        """
#        print('{:} Channels'.format(len(channel_data)))
        self.response_channels = np.array([channel.feedback_device is None or channel.feedback_device == ''  for channel in channel_data],dtype='bool')
        # Need to add a signal buffer in case the write size is not equal to
        # the read size
        self.force_buffer = np.zeros((0,np.sum(~self.response_channels)))
    
    def set_parameters(self,test_data : DataAcquisitionParameters):
        """Method to set up sampling rate and other test parameters
        
        For the synthetic case, we will set up the integration parameters using
        the sample rates provided.

        Parameters
        ----------
        test_data : DataAcquisitionParameters :
            A container containing the data acquisition parameters for the
            controller set by the user.

        """
        self.integration_oversample = test_data.output_oversample
        self.times = np.arange(test_data.samples_per_read*self.integration_oversample)/(test_data.sample_rate*self.integration_oversample)
        self.frame_time = test_data.samples_per_read/test_data.sample_rate
        self.acquisition_delay = test_data.samples_per_write/test_data.output_oversample
        
        
    def start(self):
        """Method to start acquiring data.
        
        For the synthetic case, it simply initializes the state of the system to zero"""
        self.state[:] = 0
    
    def get_acquisition_delay(self) -> int:
        """
        Get the number of samples between output and acquisition.
        
        This function returns the number of samples that need to be read to
        ensure that the last output is read by the acquisition.  If there is
        buffering in the output, this delay should be adjusted accordingly.

        Returns
        -------
        int
            Number of samples between when a dataset is written to the output
            and when it has finished playing.

        """
        return self.acquisition_delay
    
    def read(self):
        """Method to read a frame of data from the hardware
        
        This function gets the force from the output queue and adds it to the
        buffer of time signals that represents the force.  It then integrates
        a frame of time and sends it to the acquisition.
        
        Returns
        -------
        read_data : 
            2D Data read from the controller with shape ``n_channels`` x
            ``n_samples``
        
        """
        start_time = time.time()
        while self.force_buffer.shape[0] < self.times.size:
            try:
                forces = self.queue.get(timeout=self.frame_time)
            except mp.queues.Empty: # If we don't get an output in time, this likely means output has stopped so just put zeros.
                forces = np.zeros((self.force_buffer.shape[-1],self.times.size))
            self.force_buffer = np.concatenate((self.force_buffer,forces.T),axis=0)
            
        # Now extract a force that is the correct size
        this_force = self.force_buffer[:self.times.size]
        # And leave the rest for next time
        self.force_buffer = self.force_buffer[self.times.size:]
            
        times_out,sys_out,x_out = signal.lsim(self.system,this_force,self.times,self.state)
        
        self.state[:] = x_out[-1]
        
        integration_time = time.time() - start_time
        remaining_time = self.frame_time - integration_time
        if remaining_time > 0.0:
            time.sleep(remaining_time)

        return sys_out.T[...,::self.integration_oversample]
    
    def read_remaining(self):
        """Method to read the rest of the data on the acquisition
        
        This function simply returns one sample of zeros.
        
        Returns
        -------
        read_data : 
            2D Data read from the controller with shape ``n_channels`` x
            ``n_samples``
        """
        return np.zeros((len(self.response_channels),1))
    
    def stop(self):
        """Method to stop the acquisition.
        
        This simply sets the state to zero."""
        self.state[:] = 0
    
    def close(self):
        """Method to close down the hardware
        
        """
        pass
    
    
class StateSpaceOutput(HardwareOutput):
    """Class defining the interface between the controller and synthetic output
    
    Note that the only thing that this class does is pass data to the acquisition
    hardware task which actually performs the integration.  Therefore, many of
    the functions here are actually empty."""
    def __init__(self,queue : mp.queues.Queue):
        """
        Initializes the hardware by simply storing the data passing queue.

        Parameters
        ----------
        queue : mp.queues.Queue
            Queue used to pass data from output to acquisition for integration.
            See ``StateSpaceAcquisition.__init__``

        """
        self.queue = queue
    
    def set_up_data_output_parameters_and_channels(self,
                                                        test_data : DataAcquisitionParameters,
                                                        channel_data : List[Channel]):
        """
        Initialize the hardware and set up sources and sampling properties
        
        This does nothing for the synthetic hardware

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
        """Method to start acquiring data
        
        Does nothing for synthetic hardware."""
        pass
    
    def write(self,data : np.ndarray):
        """Method to write a frame of data
        
        For the synthetic excitation, this simply puts the data into the data-
        passing queue.

        Parameters
        ----------
        data : np.ndarray
            Data to write to the output.

        """
        self.queue.put(data)
    
    def stop(self):
        """Method to stop the acquisition
        
        Does nothing for synthetic hardware."""
        flush_queue(self.queue)
    
    def close(self):
        """Method to close down the hardware
        
        Does nothing for synthetic hardware."""
        pass

    def ready_for_new_output(self):
        """Signals that the hardware is ready for new output
        
        Returns ``True`` if the data-passing queue is empty.
        """
        return self.queue.empty()
