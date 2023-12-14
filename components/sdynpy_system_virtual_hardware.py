"""
Synthetic "hardware" that allows the responses to be simulated by integrating
linear equations of motion defined by a SDynPy System Object

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
import time
import os

_direction_map = {'X+': 1, 'X': 1, '+X': 1,
                  'Y+': 2, 'Y': 2, '+Y': 2,
                  'Z+': 3, 'Z': 3, '+Z': 3,
                  'RX+': 4, 'RX': 4, '+RX': 4,
                  'RY+': 5, 'RY': 5, '+RY': 5,
                  'RZ+': 6, 'RZ': 6, '+RZ': 6,
                  'X-': -1, '-X': -1,
                  'Y-': -2, '-Y': -2,
                  'Z-': -3, '-Z': -3,
                  'RX-': -4, '-RX': -4,
                  'RY-': -5, '-RY': -5,
                  'RZ-': -6, '-RZ': -6,
                  '': 0, None:0}

class SDynPySystemAcquisition(HardwareAcquisition):
    """Class defining the interface between the controller and synthetic acquisition
    
    This class defines the interfaces between the controller and the data
    acquisition portion of the hardware.  In this case, the hardware is simulated
    by integrating state space matrices derived from a SDynPy system object.
    It is run by the acquisition process, and must define how to get data from
    the test hardware into the controller.
    """
    
    def __init__(self,system_file : str, queue : mp.queues.Queue):
        """
        Loads in the SDynPy system file and sets initial parameters to null
        values.

        Parameters
        ----------
        system_file : str
            Path to the file containing state space the SDynPy system object
        queue : mp.queues.Queue
            A queue that passes input data from the SDynPySystemOutput class to
            this class.  Normally, this data transfer would occur through
            the physical test object: the exciters would excite the test object
            with the specified excitation and the Acquisition would record the
            responses to that excitation.  In the synthetic case, we need to
            pass the output data to the acquisition which does the integration.

        Returns
        -------
        None.

        """
        self.sdynpy_system_data = {key:val for key,val in np.load(system_file).items()}
        self.system = None
        self.times = None
        self.state = None
        self.frame_time = None
        self.queue = queue
        self.force_buffer = None
        self.integration_oversample = None
        self.response_channels = None
        self.output_channels = None
        # Create a dictionary of channels for faster lookup
        self.channel_indices = {tuple([abs(v) for v in val]):index for index,val in enumerate(self.sdynpy_system_data['coordinate'])}
        
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
        self.output_channels = ~self.response_channels
        # Need to add a signal buffer in case the write size is not equal to
        # the read size
        self.force_buffer = np.zeros((0,np.sum(~self.response_channels)))
        
        # Figure out which channels go with which indices
        channel_indices = []
        channel_signs = []
        for channel in channel_data:
            node_number = int(channel.node_number)
            direction = _direction_map[channel.node_direction]
            channel_index = self.channel_indices[(node_number,abs(direction))]
            channel_indices.append(channel_index)
            channel_signs.append(np.sign(direction)*np.sign(self.sdynpy_system_data['coordinate'][channel_index]['direction']))
        channel_indices = np.array(channel_indices)
        channel_signs = np.array(channel_signs)
        
        # Now we need to actually go through and set up the A, B, C, D state matrices
        M = self.sdynpy_system_data['mass']
        C = self.sdynpy_system_data['damping']
        K = self.sdynpy_system_data['stiffness']
        
        # Now we need to pull out the transformation matrices
        phi = self.sdynpy_system_data['transformation'][channel_indices,:]
        # Multiply by the signs
        phi *= channel_signs[:,np.newaxis]
        
        # Separate into responses and excitations; here input is into the system
        phi_excitation = phi[self.output_channels,:].copy()
        phi_response = phi[self.response_channels,:].copy()
        
        # Set up some other parameters
        ndofs = M.shape[0]
        tdofs_response = phi_response.shape[0]
        tdofs_input = phi_excitation.shape[0]
        
        # Assembly the full state matrices
        
        # A = [[     0,     I],
        #      [M^-1*K,M^-1*C]]

        A_state = np.block([[np.zeros((ndofs, ndofs)), np.eye(ndofs)],
                            [-np.linalg.solve(M, K), -np.linalg.solve(M, C)]])

        # B = [[     0],
        #      [  M^-1]]

        B_state = np.block([[np.zeros((ndofs, tdofs_input))],
                            [np.linalg.solve(M, phi_excitation.T)]])

        # C = [[     I,     0],   # Displacements
        #      [     0,     I],   # Velocities
        #      [M^-1*K,M^-1*C],   # Accelerations
        #      [     0,     0]]   # Forces

        C_all = np.block([[phi_response, np.zeros((tdofs_response, ndofs))],
                            [np.zeros((tdofs_response, ndofs)), phi_response],
                            [-phi_response @ np.linalg.solve(M, K), -phi_response @ np.linalg.solve(M, C)],
                            [np.zeros((tdofs_input, ndofs)), np.zeros((tdofs_input, ndofs))]])

        # D = [[     0],   # Displacements
        #      [     0],   # Velocities
        #      [  M^-1],   # Accelerations
        #      [     I]]   # Forces

        D_all = np.block([[np.zeros((tdofs_response, tdofs_input))],
                            [np.zeros((tdofs_response, tdofs_input))],
                            [phi_response @ np.linalg.solve(M, phi_excitation.T)],
                            [np.eye(tdofs_input)]])
        
        # Split into different types
        displacement_indices = np.arange(tdofs_response)
        velocity_indices = np.arange(tdofs_response) + tdofs_response
        acceleration_indices = np.arange(tdofs_response) + 2 * tdofs_response
        force_indices = np.arange(tdofs_input) + 3 * tdofs_response
        
        C_disp = C_all[displacement_indices]
        C_vel = C_all[velocity_indices]
        C_accel = C_all[acceleration_indices]
        C_force = C_all[force_indices]
        
        D_disp = D_all[displacement_indices]
        D_vel = D_all[velocity_indices]
        D_accel = D_all[acceleration_indices]
        D_force = D_all[force_indices]
        
        # Now assemble the full response C and D matrices based on the data type
        C_response = []
        D_response = []
        response_index = 0
        for i,channel in enumerate(channel_data):
            if self.output_channels[i]:
                continue
            if channel.channel_type.lower() in ['disp','displacement']:
                C_response.append(C_disp[response_index])
                D_response.append(D_disp[response_index])
            elif channel.channel_type.lower() in ['vel','velocity']:
                C_response.append(C_vel[response_index])
                D_response.append(D_vel[response_index])
            elif channel.channel_type.lower() in ['accel','acceleration','acc']:
                C_response.append(C_accel[response_index])
                D_response.append(D_accel[response_index])
            else:
                print("Unknown Channel Type for Channel {:}: {:}".format(i+1,channel.channel_type()))
                C_response.append(C_disp[response_index])
                D_response.append(D_disp[response_index])
            response_index += 1
        C_response = np.array(C_response)
        D_response = np.array(D_response)
        
        # Now assemble the final C and D matrices
        C_state = np.empty((len(channel_data), C_response.shape[-1]))
        C_state[self.response_channels,:] = C_response
        C_state[self.output_channels,:] = C_force
        D_state = np.empty((len(channel_data), D_response.shape[-1]))
        D_state[self.response_channels,:] = D_response
        D_state[self.output_channels,:] = D_force
        self.system = signal.StateSpace(A_state,B_state,C_state,D_state)
        self.state = np.zeros(A_state.shape[0])
        # np.savez('SDynPy_State.npz', A=A_state, B=B_state, C = C_state, D = D_state)
        
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
    
class SDynPySystemOutput(HardwareOutput):
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
