# -*- coding: utf-8 -*-
"""
Synthetic "hardware" that allows the responses to be simulated by integrating
modal equations of motion nearly real-time.

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
import netCDF4
import multiprocessing as mp
import scipy.signal as signal
import time

DEBUG = False

NOISE_LEVEL = 0.00000001

if DEBUG:
    import glob
    import os
    test_files_output_names = r'debug\output_data_{:}.npz'
    test_files_acquisition_names = r'debug\acquire_data_{:}.npz'
    test_files_system_name = r'debug\system.npz'
    # Delete existing test files
    for str_format in [test_files_acquisition_names,test_files_output_names]:
        files = glob.glob(str_format.format('*'))
        for file in files:
            try:
                os.remove(file)
            except (FileNotFoundError,PermissionError):
                pass
    

class ExodusAcquisition(HardwareAcquisition):
    """Class defining the interface between the controller and synthetic acquisition
    
    This class defines the interfaces between the controller and the
    data acquisition portion of the hardware.  In this case, the hardware is
    actually simulated by integrating the modal equations of motion of a finite
    element model containing a modal solution.  It is run by the Acquisition
    process, and must define how to get data from the test hardware into the
    controller."""
    def __init__(self,exodus_file : str,queue : mp.queues.Queue):
        '''Loads in the exodus file and sets initial parameters to null values
        

        Parameters
        ----------
        exodus_file : str : 
            Path to the exodus file.
        queue : mp.queues.Queue
            A queue that passes input data from the ExodusOutput class to the
            Exodus input class.  Normally, this data transfer would occur through
            the physical test object: the exciters would excite the test object
            with the specified excitation and the Acquisition would record the
            responses to that excitation.  In the synthetic case, we need to
            pass the output data to the acquisition which does the integration.

        '''
        self.exo = exodus(exodus_file)
        self.phi = None
        self.phi_full = None
        self.response_channels = None
        self.system = None
        self.times = None
        self.state = None
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
        displacements = self.exo.get_displacements()
        node_numbers = self.exo.get_node_num_map()
        try:
            self.damping = float(channel_data[0].comment)
            print('{:} Damping'.format(self.damping))
        except Exception:
            self.damping = 0.01
        self.response_channels = np.array([channel.feedback_device is None or channel.feedback_device == ''  for channel in channel_data],dtype='bool')
        self.phi_full = np.array([self._create_channel(channel,displacements,node_numbers) for channel in channel_data])
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
        # Get the number of modes that we will keep (bandwidth*1.5)
        frequencies = self.exo.get_times()
        frequencies[frequencies < 0] = 0 # Eliminate any negative frequencies
        keep_modes = frequencies < test_data.nyquist_frequency*1.5
        # Oversampling due to integration
        self.integration_oversample = test_data.output_oversample
        self.phi = self.phi_full[...,keep_modes]
        nmodes = self.phi.shape[-1]
        frequencies = frequencies[keep_modes]
        M = np.eye(nmodes)
        K = np.diag((frequencies*2*np.pi)**2)
        C = np.diag((2*2*np.pi*frequencies*self.damping))
        A,B,C,D = MCK_to_StateSpace(M,C,K)
        self.system = signal.StateSpace(A,B,C,D)
        self.times = np.arange(test_data.samples_per_read*self.integration_oversample)/(test_data.sample_rate*self.integration_oversample)
        self.frame_time = test_data.samples_per_read/test_data.sample_rate
        self.state = np.zeros(nmodes*2)
        self.acquisition_delay = test_data.samples_per_write/test_data.output_oversample
        if DEBUG:
            np.savez(test_files_system_name,A=A,B=B,C=C,D=D,times=self.times)
        
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
        
        # print('Got Force')
        # print('this_force shape: {:}'.format(this_force.shape))
        
        modal_forces = np.einsum('ij,ki->kj',self.phi[~self.response_channels],this_force)
        # print('modal_forces shape: {:}'.format(modal_forces.shape))
        
        # print('Integrating...')
        # print('system: {:}'.format(self.system))
        # print('forces: {:}\n{:}'.format(modal_forces.shape,modal_forces))
        # print('times: {:}\n{:}'.format(self.times.shape,self.times))
        # print('state: {:}\n{:}'.format(self.state.shape,self.state))
            
        times_out,sys_out,x_out = signal.lsim(self.system,modal_forces,self.times,self.state)
        # print('output: {:}\n{:}'.format(sys_out.shape,sys_out))
    
        sys_accels = sys_out[:,2*self.phi.shape[-1]:3*self.phi.shape[-1]]
    
        accelerations = self.phi[self.response_channels]@sys_accels.T
        
        self.state[:] = sys_out[-1,0:2*self.phi.shape[-1]]
        
        # Now we need to combine accelerations with forces in the same way
        output = np.zeros((len(self.response_channels),len(self.times))) # n channels x n times
        output[self.response_channels] = accelerations
        output[~self.response_channels] = this_force.T
        
        # print('output: {:}\n{:}'.format(output.shape,output))
        
        integration_time = time.time() - start_time
        remaining_time = self.frame_time - integration_time
        if remaining_time > 0.0:
            time.sleep(remaining_time)
        
        if DEBUG:
            # Find current files
            num_files = len(glob.glob(test_files_acquisition_names.format('*')))
            np.savez(test_files_acquisition_names.format(num_files),full_output=output,
                     integration_oversample = self.integration_oversample,
                     modal_forces = modal_forces,forces = this_force,
                     integration_output = sys_out,times=times_out,state = self.state,
                     response_channels = self.response_channels)
        return output[...,::self.integration_oversample] + NOISE_LEVEL*np.random.randn(*output[...,::self.integration_oversample].shape)
    
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
        
        This simply closes the exodus file."""
        self.exo.close()
    
    def _create_channel(self,channel: Channel,displacement,node_numbers):
        """Helper function to create a channel from the exodus file.
        
        This function parses the channel information and then extracts the
        mode shape row corresponding to that channel.
        
        Parameters
        ----------
        channel: Channel :
            ``Channel`` object specifying the channel information
            
        displacement :
            The 3D mode shape matrix consisting of direction x mode x node
            dimensions.
            
        node_numbers :
            The node numbers in the finite element model, either increasing
            from one to the largest node or from the node num map.

        Returns
        -------
        phi_row : 
            A row of the mode shape matrix corresponding to the channel

        """
        node_number = int(channel.node_number)
        node_index = np.where(node_numbers == node_number)[0][0]
        if channel.node_direction.lower().replace(' ','') in ['x+','+x']:
            direction = np.array([1,0,0])
        elif channel.node_direction.lower().replace(' ','') in ['x-','-x']:
            direction = np.array([-1,0,0])
        elif channel.node_direction.lower().replace(' ','') in ['y+','+y']:
            direction = np.array([0,1,0])
        elif channel.node_direction.lower().replace(' ','') in ['y-','-y']:
            direction = np.array([0,-1,0])
        elif channel.node_direction.lower().replace(' ','') in ['z+','+z']:
            direction = np.array([0,0,1])
        elif channel.node_direction.lower().replace(' ','') in ['z-','-z']:
            direction = np.array([0,0,-1])
        else:
            direction = np.array([float(val) for val in channel.node_direction.split(',')])
            direction /= np.linalg.norm(direction)
        phi_row = np.einsum('i,ij',direction,displacement[...,node_index])
        return phi_row
    
class ExodusOutput(HardwareOutput):
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
            See ``ExodusAcquisition.__init__``

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
        if DEBUG:
            # Find current files
            num_files = len(glob.glob(test_files_output_names.format('*')))
            np.savez(test_files_output_names.format(num_files),forces = data)
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
    
class ExodusError(Exception):
    """An exception to specify an error has occured in the Exodus reader"""
    pass

class exodus:
    """A class with limited exodus file read capabilities"""
    def __init__(self,filename):
        """
        Read in an exodus file given by ``filename``

        Parameters
        ----------
        filename : str
            Path to the exodus file to read.

        """
        self.filename = filename
        self._ncdf_handle = netCDF4.Dataset(filename,'r')
    
    @property
    def num_dimensions(self):
        """Property to get the number of dimensions in an exodus file"""
        return self._ncdf_handle.dimensions['num_dim'].size
    
    @property
    def num_nodes(self):
        """Property to get the number of nodes in an exodus file"""
        return self._ncdf_handle.dimensions['num_nodes'].size
    
    @property
    def num_times(self):
        """Property to get the number of time steps in an exodus file"""
        return self._ncdf_handle.dimensions['time_step'].size
    
    def get_node_num_map(self):
        """Retrieve the list of local node IDs from the exodus file.

        Returns
        -------
        node_num_map : np.array
            Returns a 1D array with size num_nodes, denoting the node number
            for the node in each index

        Notes
        -----
        If there is no node_num_map in the exodus file, this function simply
        returns an array from 1 to self.num_nodes
        """
        if 'node_num_map' in self._ncdf_handle.variables:
            return self._ncdf_handle.variables['node_num_map'][:]
        else:
            return np.ma.MaskedArray(np.arange(self.num_nodes)+1)
    
    def get_displacements(self,displacement_name='Disp',capital_coordinates=True):
        """Get the displacements from the Exodus file.

        Parameters
        ----------
        displacement_name :
             The prefix of the displacement variable (Default value = 'Disp')
        capital_coordinates :
             Whether or not the X,Y,Z values appended to the displacement name
             are capitalized.  (Default value = True)

        Returns
        -------
        displacement_array : np.ndarray
            Returns a 3D array with size num_dimensions, num_times,num_nodes
            containing the displacements in each direction at each mode shape
            at each node.
        """
        return np.array([self.get_node_variable_values(displacement_name+(val.upper() if capital_coordinates else val.lower()))
                         for val in 'xyz'[:self.num_dimensions]])
    
    def get_coords(self):
        """Retrieve the coordinates of the nodes in the exodus file.
        
        Returns
        -------
        coordinate_array : np.ndarray
            Returns a 2D array with size num_dimensions, num_nodes
            containing the position of each node.
            """
        # TODO Add error checking
        coord_names = ('coordx','coordy','coordz')[:self.num_dimensions]
        raw_list = [self._ncdf_handle.variables[name][:] for name in coord_names]
        coords = np.array(raw_list)
        return coords
    
    def get_node_variable_names(self):
        """Gets a tuple of nodal variable names from the exodus file
        
        Returns
        -------
        node_var_names : tuple(str)
            A tuple of strings corresponding to the variable names in the
            model."""
        try:
            raw_records = self._ncdf_handle.variables['name_nod_var']
        except KeyError:
            raise ExodusError('Node Variable Names are not defined!')
        node_var_names = tuple(''.join(value.decode() for value in line if not isinstance(value,np.ma.core.MaskedConstant)) for line in raw_records)
        return node_var_names
    
    def get_node_variable_values(self,name_or_index,step=None):
        """Gets the node variable values for the specified timestep

        Parameters
        ----------
        name_or_index : str or int
            Name or Index of the nodal variable that is desired.  If
            type(name_or_index) == str, then it is assumed to be the name.  If
            type(name_or_index) == int, then it is assumed to be the index.
        step : int
            Time step at which to recover the nodal variable (Default value = None)

        Returns
        -------
        node_var_values : np.ndarray
            A 1D or 2D numpy array of variable values depending on whether a
            time step is specified or not.
        
        """
        if isinstance(name_or_index,(int,np.integer)):
            index = name_or_index
        elif isinstance(name_or_index,(str,np.character)):
            try:
                index = self.get_node_variable_names().index(name_or_index)
            except ValueError:
                raise ExodusError('Name {} not found in self.get_node_variable_names().  Options are {}'.format(name_or_index,self.get_node_variable_names()))
        vals_nod_var_name = 'vals_nod_var{:d}'.format(index+1)
        if step is not None:
            if step >= self.num_times:
                raise ExodusError('Invalid Time Step')
            return self._ncdf_handle.variables[vals_nod_var_name][step,:]
        else:
            return self._ncdf_handle.variables[vals_nod_var_name][:]
        

    def get_times(self):
        """Gets the time values from the exodus file
        
        Returns
        -------
        times : np.ndarray
            A vector of time values in the exodus file.  These may be
            frequencies if the exodus file contains a modal or frequency-based
            solution.
        """
        return self._ncdf_handle.variables['time_whole'][:]
    
    def close(self):
        """Closes the exodus file"""
        self._ncdf_handle.close()
    
def MCK_to_StateSpace(M,C,K):
    """Creates a state-space representation from a mass, stiffness, and damping matrix.

    Parameters
    ----------
    M : np.ndarray
        A square array defining the system mass matrix.
    C : np.ndarray
        A square array defining the system damping matrix
    K : np.ndarray
        A square array defining the system stiffness matrix        

    Returns
    -------
    A : np.ndarray
        The state space A matrix
    B : np.ndarray
        The state space B matrix
    C : np.ndarray
        The state space C matrix
    D : np.ndarray
        The state space D matrix
    """
    
    ndofs = M.shape[0]
    
    # A = [[      0,      I],
    #      [-M^-1*K,-M^-1*C]]
    
    A_state = np.block([[ np.zeros((ndofs,ndofs)),          np.eye(ndofs)],
                    [   -np.linalg.solve(M,K),  -np.linalg.solve(M,C)]])

    # B = [[     0,  M^-1]]
    
    B_state = np.block([[ np.zeros((ndofs,ndofs))],   [np.linalg.inv(M)]])
    
    # C = [[      I,      0],   # Displacements
    #      [      0,      I],   # Velocities
    #      [-M^-1*K,-M^-1*C],   # Accelerations
    #      [      0,      0]]   # Forces
    
    C_state = np.block([[            np.eye(ndofs), np.zeros((ndofs,ndofs))],
                        [  np.zeros((ndofs,ndofs)),           np.eye(ndofs)],
                        [    -np.linalg.solve(M,K),   -np.linalg.solve(M,C)],
                        [  np.zeros((ndofs,ndofs)), np.zeros((ndofs,ndofs))]])
    
    # D = [[     0],   # Displacements
    #      [     0],   # Velocities
    #      [  M^-1],   # Accelerations
    #      [     I]]   # Forces
    
    D_state = np.block([[np.zeros((ndofs,ndofs))],
                        [np.zeros((ndofs,ndofs))],
                        [       np.linalg.inv(M)],
                        [          np.eye(ndofs)]])
    
    return A_state,B_state,C_state,D_state

# def integrate_MCK(M,C,K,times,forces,x0=None):
#     """Integrates a system defined by a Mass, Stiffness, and Damping Matrix

#     Parameters
#     ----------
#     M : np.ndarray
#         A square array defining the system mass matrix.
#     C : np.ndarray
#         A square array defining the system damping matrix
#     K : np.ndarray
#         A square array defining the system stiffness matrix     
#     times :
#         A vector of times that will be passed to the ``scipy.signal.lsim``
#         function.
#     forces :
#         A vector of forces that will be passed to the ``scipy.signal.lsim``
#         function
#     x0 :
#         The initial state of the system that will be used by the ``scipy.signal.lsim``
#         function (Default value = None, meaning the system starts at rest)

#     Returns
#     -------

#     """
#     A,B,C,D = MCK_to_StateSpace(M,C,K)
#     linear_system = signal.StateSpace(A,B,C,D)
    
#     times_out,sys_out,x_out = signal.lsim(linear_system,forces,times,x0)
    
#     sys_disps = sys_out[:,0*M.shape[0]:1*M.shape[0]]
#     sys_vels = sys_out[:,1*M.shape[0]:2*M.shape[0]]
#     sys_accels = sys_out[:,2*M.shape[0]:3*M.shape[0]]
#     sys_forces = sys_out[:,3*M.shape[0]:4*M.shape[0]]
    
#     return sys_disps,sys_vels,sys_accels,sys_forces

# def integrate_modes(natural_frequencies,damping_ratios,modal_forces,times,q0,qd0):
#     """Integrate a set of natural frequencies, damping ratios, and modal forces

#     Parameters
#     ----------
#     natural_frequencies : np.ndarray
#         A 1D array of natural frequencies for each mode
#     damping_ratios : np.ndarray
#         A 1D array of critical damping ratios for each mode
#     modal_forces : np.ndarray
#         A set of modal forces that will be passed to the integrate_MCK function
#     times : np.ndarray
#         A set of times that will be passed to the integrate_MCK function
#     q0 : np.ndarray
#         Initial values of the modal coefficients
#     qd0 : np.ndarray
#         Initial velocities of the modal coefficients
        

#     Returns
#     -------

#     """
#     nmodes = natural_frequencies.size
#     M = np.eye(nmodes)
#     K = np.diag((natural_frequencies*2*np.pi)**2)
#     C = np.diag((2*2*np.pi*natural_frequencies*damping_ratios))
#     q = integrate_MCK(M,C,K,times,modal_forces,np.concatenate((q0,qd0)))[:2]
#     return q