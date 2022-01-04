# -*- coding: utf-8 -*-
"""
Controller Subsystem that handles the reading of data from the hardware.

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
from .utilities import QueueContainer,GlobalCommands,flush_queue
from .abstract_message_process import AbstractMessageProcess

class AcquisitionProcess(AbstractMessageProcess):
    """Class defining the acquisition behavior of the controller
    
    This class will handle reading data from the hardware and then sending it
    to the individual environment processes.
    
    See AbstractMesssageProcess for inherited class members.
    """
    def __init__(self,process_name : str, queue_container : QueueContainer,environments : list):
        """
        Constructor for the AcquisitionProcess class
        
        Sets up the ``command_map`` and initializes all data members.

        Parameters
        ----------
        process_name : str
            The name of the process.
        queue_container : QueueContainer
            A container containing the queues used to communicate between
            controller processes
        environments : list
            A list of ``(ControlType,environment_name)`` pairs that define the
            environments in the controller.


        """
        super().__init__(process_name,queue_container.log_file_queue,queue_container.acquisition_command_queue,queue_container.gui_update_queue)
        self.map_command(GlobalCommands.INITIALIZE_DATA_ACQUISITION,self.initialize_data_acquisition)
        self.map_command(GlobalCommands.RUN_HARDWARE,self.acquire_signal)
        self.map_command(GlobalCommands.STOP_HARDWARE,self.stop_acquisition)
        self.map_command(GlobalCommands.START_ENVIRONMENT,self.start_environment)
        self.map_command(GlobalCommands.STOP_ENVIRONMENT,self.stop_environment)
        self.map_command(GlobalCommands.START_STREAMING,self.start_streaming)
        self.map_command(GlobalCommands.STOP_STREAMING,self.stop_streaming)
        
        # Communication
        self.queue_container = queue_container
        self.startup = True
        self.shutdown_flag = False
        self.any_environments_started = False
        # Sampling data
        self.sample_rate = None
        self.read_size = None
        # Environment Data
        self.environment_list = [environment[1] for environment in environments]
        self.environment_acquisition_channels = None
        self.environment_active_flags = {environment:False for environment in self.environment_list}
        self.environment_last_data = {environment:False for environment in self.environment_list}
        self.environment_samples_remaining_to_read = {environment:0 for environment in self.environment_list}
        # Hardware data
        self.hardware = None
        # Streaming Information
        self.streaming = False
        # Persistent data
        self.read_data = None
        
    def initialize_data_acquisition(self,data):
        """Sets up the acquisition according to the specified parameters

        Parameters
        ----------
        data : tuple
            A tuple consisting of data acquisition parameters and the channels
            used by each environment.

        """
        self.log('Initializing Data Acquisition')
        # Pull out invormation from the queue
        data_acquisition_parameters,self.environment_acquisition_channels = data
        # Store pertinent data
        self.sample_rate = data_acquisition_parameters.sample_rate
        self.read_size = data_acquisition_parameters.samples_per_read
        # Check which type of hardware we have
        if not self.hardware is None:
            self.hardware.close()
        if data_acquisition_parameters.hardware == 0:
            from .nidaqmx_hardware_multitask import NIDAQmxAcquisition
            self.hardware = NIDAQmxAcquisition()
        elif data_acquisition_parameters.hardware == 1:
            from .lanxi_hardware_multiprocessing import LanXIAcquisition
            self.hardware = LanXIAcquisition(data_acquisition_parameters.maximum_acquisition_processes)
        elif data_acquisition_parameters.hardware == 2:
            from .exodus_modal_solution_hardware import ExodusAcquisition
            self.hardware = ExodusAcquisition(data_acquisition_parameters.hardware_file,self.queue_container.single_process_hardware_queue)
        elif data_acquisition_parameters.hardware == 3:
            from .state_space_virtual_hardware import StateSpaceAcquisition
            self.hardware = StateSpaceAcquisition(data_acquisition_parameters.hardware_file,self.queue_container.single_process_hardware_queue)
        else:
            raise ValueError('Invalid Hardware or Hardware Not Implemented!')
        # Initialize hardware and create channels
        self.hardware.set_up_data_acquisition_parameters_and_channels(data_acquisition_parameters, data_acquisition_parameters.channel_list)
    
    def start_environment(self,data):
        """Sets the flag stating the specified environment is active

        Parameters
        ----------
        data : str
            The environment name that should be activated.


        """
        self.log('Activating Environment {:}'.format(data))
        self.environment_active_flags[data] = True
        
    def stop_environment(self,data):
        """Sets flags stating that the specified environment will be ending.

        Parameters
        ----------
        data : str
            The environment name that should be deactivated

        """
        self.log('Deactivating Environment {:}'.format(data))
        self.environment_active_flags[data] = False
        self.environment_last_data[data] = True
        self.environment_samples_remaining_to_read[data] = self.hardware.get_acquisition_delay()
        
    def start_streaming(self,data):
        """Sets the flag to tell the acquisition to write data to disk

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.streaming = True
        
    def stop_streaming(self,data):
        """Sets the flag to tell the acquisition to not write data to disk

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.streaming = False
    
    def acquire_signal(self,data):
        """The main acquisition loop of the controller.
        
        If it is the first time through this loop, startup will be set to True
        and the hardware will be started.
        
        If it is the last time through this loop, the hardware will be shut
        down.
        
        The function will simply read the data from the hardware and pass it
        to any active environment and to the streaming process if the process
        is active.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        if self.startup:
            self.any_environments_started = False
            self.log('Waiting for Output to Start')
            try:
                self.queue_container.input_output_sync_queue.get(timeout=30)
            except mp.queues.Empty:
                self.queue_container.gui_update_queue.put(('error',('Acquisition Error','Acquisition timed out waiting for output to start.  Check output task for errors!')))
                return
            self.log('Starting Hardware Acquisition')
            self.hardware.start()
            self.startup = False
        if (self.shutdown_flag # We're shutting down
            and all([not flag for environment,flag in self.environment_active_flags.items()]) # All the environments are inactive
            and all([not flag for environment,flag in self.environment_last_data.items()]) # None of the environments are expecting their last data
            ):
            self.log('Acquiring Remaining Data')
            self.read_data = self.hardware.read_remaining()
            self.hardware.stop()
            self.shutdown_flag = False
            self.startup = True
            # print('{:} {:}'.format(self.streaming,self.any_environments_started))
            if self.streaming and self.any_environments_started:
                self.queue_container.streaming_command_queue.put(self.process_name,(GlobalCommands.STREAMING_DATA,self.read_data.copy()))
                self.queue_container.streaming_command_queue.put(self.process_name,(GlobalCommands.FINALIZE_STREAMING,None))
                self.streaming = False
        else:
            self.log('Acquiring Data for {:} environments'.format([name for name,flag in self.environment_active_flags.items() if flag]))
            self.read_data = self.hardware.read()
            # Send the data to the different channels
            for environment in self.environment_list:
                # Only send data if the environment is active or if it is the last data we will send.
                if self.environment_active_flags[environment] or self.environment_last_data[environment]:
                    self.any_environments_started = True
                    environment_data = self.read_data[self.environment_acquisition_channels[environment]].copy()
                    if self.environment_last_data[environment]:
                        self.environment_samples_remaining_to_read[environment] -= self.read_size
                        self.log('Reading last data for {:}, {:} samples remaining'.format(environment,self.environment_samples_remaining_to_read[environment]))
                    environment_finished = self.environment_last_data[environment] and self.environment_samples_remaining_to_read[environment] <= 0
                    self.queue_container.environment_data_in_queues[environment].put((environment_data,environment_finished))
                    if environment_finished:
                        self.environment_last_data[environment] = False
                        self.log('Delivered last data to {:}'.format(environment))
                    
                        
    #                    np.savez('test_data/acquisition_data_check.npz',read_data = self.read_data,environment_data = environment_data,environment_channels = self.environment_acquisition_channels[environment])
            self.queue_container.acquisition_command_queue.put(self.process_name,(GlobalCommands.RUN_HARDWARE,None))
            # print('{:} {:}'.format(self.streaming,self.any_environments_started))
            if self.streaming and self.any_environments_started:
                self.queue_container.streaming_command_queue.put(self.process_name,(GlobalCommands.STREAMING_DATA,self.read_data.copy()))
    
    def stop_acquisition(self,data):
        """Sets a flag telling the acquisition that it should start shutting down

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.shutdown_flag = True

    def quit(self,data):
        """Stops the process and shuts down the hardware if necessary.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``
        """
        # Pull any data off the queues that have been put to
        queue_flush_sum = 0
        for queue in [q for name,q in self.queue_container.environment_data_in_queues.items()] + [self.queue_container.acquisition_command_queue]:
            queue_flush_sum += len(flush_queue(queue))
        self.log('Flushed {:} items out of queues'.format(queue_flush_sum))
        if not self.hardware is None:
            self.hardware.close()
        return True
    
def acquisition_process(queue_container : QueueContainer,environments : list):
    """Function passed to multiprocessing as the acquisition process
    
    This process creates the ``AcquisitionProcess`` object and calls the ``run``
    command.

    Parameters
    ----------
    queue_container : QueueContainer
        A container containing the queues used to communicate between
        controller processes
    environments : list
        A list of ``(ControlType,environment_name)`` pairs that define the
        environments in the controller.

    """

    acquisition_instance = AcquisitionProcess('Acquisition',queue_container,environments)
    
    acquisition_instance.run()