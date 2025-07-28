# -*- coding: utf-8 -*-
"""
Controller subsystem that handles the output from the hardware to the
shaker amplifiers or other excitation device.

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

from .utilities import QueueContainer,GlobalCommands,flush_queue,VerboseMessageQueue,rms_time
from .abstract_message_process import AbstractMessageProcess
import numpy as np
import multiprocessing as mp


task_name = 'Output'

volume_threshold = 1.01

class OutputProcess(AbstractMessageProcess):
    """Class defining the output behavior of the controller
    
    This class will handle collecting data from the environments and writing
    data to be output to the hardware
    
    See AbstractMessageProcess for inherited members.
    """
    def __init__(self,process_name : str, queue_container : QueueContainer,environments : list, output_active : mp.Value):
        """
        Constructor for the OutputProcess Class
        
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
        super().__init__(process_name,queue_container.log_file_queue,queue_container.output_command_queue,queue_container.gui_update_queue)
        self.map_command(GlobalCommands.INITIALIZE_DATA_ACQUISITION,self.initialize_data_acquisition)
        self.map_command(GlobalCommands.RUN_HARDWARE,self.output_signal)
        self.map_command(GlobalCommands.STOP_HARDWARE,self.stop_output)
        self.map_command(GlobalCommands.START_ENVIRONMENT,self.start_environment)
        # Communication
        self.queue_container = queue_container
        self.startup = True
        self.shutdown_flag = False
        # Sampling data
        self.sample_rate = None
        self.write_size = None
        self.num_outputs = None
        self.output_oversample = None
        # Environment Data
        self.environment_list = self.environment_list = [environment[1] for environment in environments]
        self.environment_output_channels = None
        self.environment_active_flags = {environment:False for environment in self.environment_list}
        self.environment_starting_up_flags = {environment:False for environment in self.environment_list}
        self.environment_shutting_down_flags = {environment:False for environment in self.environment_list}
        self.environment_data_out_remainders = None
        self.environment_first_data = {environment:False for environment in self.environment_list}
        # Hardware data
        self.hardware = None
        # Shared memory to record activity
        self._output_active = output_active
        # print('output setup')
        
    @property
    def output_active(self):
        return bool(self._output_active.value)
    
    @output_active.setter
    def output_active(self,val):
        # print('output currently active: {:}'.format(self.output_active))
        # print('setting output active')
        if val:
            self._output_active.value = 1
        else:
            self._output_active.value = 0
        # print('set output active')
    
    def initialize_data_acquisition(self,data):
        """
        Sets up the output according to the specified parameters

        Parameters
        ----------
        data : tuple
            A tuple consisting of data acquisition parameters and the channels
            used by each environment.

        """
        self.log('Initializing Data Acquisition')
        # Pull out invormation from the queue
        data_acquisition_parameters,environment_channels = data
        # Store pertinent data
        self.sample_rate = data_acquisition_parameters.sample_rate
        self.write_size = data_acquisition_parameters.samples_per_write
        self.output_oversample = data_acquisition_parameters.output_oversample
        # Check which type of hardware we have
        if not self.hardware is None:
            self.hardware.close()
        if data_acquisition_parameters.hardware == 0:
            from .nidaqmx_hardware_multitask import NIDAQmxOutput
            self.hardware = NIDAQmxOutput()
        elif data_acquisition_parameters.hardware == 1:
            from .lanxi_hardware_multiprocessing import LanXIOutput
            self.hardware = LanXIOutput(data_acquisition_parameters.maximum_acquisition_processes)
        elif data_acquisition_parameters.hardware == 2:
            from .data_physics_hardware import DataPhysicsOutput
            self.hardware = DataPhysicsOutput(self.queue_container.single_process_hardware_queue)
        elif data_acquisition_parameters.hardware == 4:
            from .exodus_modal_solution_hardware import ExodusOutput
            self.hardware = ExodusOutput(self.queue_container.single_process_hardware_queue)
        elif data_acquisition_parameters.hardware == 5:
            from .state_space_virtual_hardware import StateSpaceOutput
            self.hardware = StateSpaceOutput(self.queue_container.single_process_hardware_queue)
        elif data_acquisition_parameters.hardware == 6:
            from .sdynpy_system_virtual_hardware import SDynPySystemOutput
            self.hardware = SDynPySystemOutput(self.queue_container.single_process_hardware_queue)
        else:
            raise ValueError('Invalid Hardware or Hardware Not Implemented!')
        # Initialize hardware and create channels
        self.hardware.set_up_data_output_parameters_and_channels(data_acquisition_parameters, data_acquisition_parameters.channel_list)
        # Get the environment output channels in reference to all the output channels
        output_indices = [index for index,channel in enumerate(data_acquisition_parameters.channel_list)
                          if not (channel.feedback_device is None) and not (channel.feedback_device.strip() == '')]
        self.num_outputs = len(output_indices)
        self.environment_output_channels = {}
        self.environment_data_out_remainders = {}
        for environment,active_indices in environment_channels.items():
            common_indices,out_inds,env_inds = np.intersect1d(
                    output_indices,active_indices,return_indices = True)
            self.environment_output_channels[environment] = out_inds
            self.environment_data_out_remainders[environment] = np.zeros((len(common_indices),0))
        
    def output_signal(self,data):
        """The main output loop of the controller.
        
        If it is the first time through the loop, ``self.startup`` will be set
        to ``True`` and the hardware will be started.
        
        The output task must be notified that a given channel has started before
        it will check the channel for data.
        
        The function receives data from each of the environment data_out queues
        as well as a flag specifying whether the signal is the last signal.  If
        so, the output will deactivate the channel and stop looking for data
        from it until the next time it is activated.
        
        If the output is shut down, it will activate a ``shutdown_flag``, after
        which the output will wait for all environments to pass their last data
        and be deactivated.  Once all environments are deactivated, the output
        will stop and shutdown the hardware.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        # Skip hardware operations if there are no channels
        skip_hardware = self.num_outputs == 0
        # Go through each environment and collect data no matter what.
        # Start with ready to write and set to false if any environments are not
        ready_to_write = True
        for environment in self.environment_list:
            # If the task isn't active or currently shutting down, we don't need more data, so just skip it.
            if (not self.environment_active_flags[environment] and not self.environment_starting_up_flags[environment]) or self.environment_shutting_down_flags[environment]:
                continue
            # Check if we need more data from that environment
            if self.environment_data_out_remainders[environment].shape[-1] < self.write_size:
                if not self.environment_starting_up_flags[environment]:
                    ready_to_write = False
                try:
                    # Try to grab data from the queue and add it to the remainders.
                    environment_data,last_run = self.queue_container.environment_data_out_queues[environment].get_nowait()
                except mp.queues.Empty:
                    # If data is not ready yet, simply continue to the next environment and we'll try on the next time around.
                    continue
                self.log('Got {:} data from {:} Environment'.format(' x '.join(['{:}'.format(shape) for shape in environment_data.shape]),environment))
                if last_run:
                    self.log('Deactivating {:} Environment'.format(environment))
                    self.environment_shutting_down_flags[environment] = True
                    if self.environment_starting_up_flags[environment]:
                        self.environment_starting_up_flags[environment] = False
                        self.environment_active_flags[environment] = True
                        self.environment_first_data[environment] = True
                self.environment_data_out_remainders[environment] = np.concatenate((self.environment_data_out_remainders[environment],environment_data),axis=-1)
            else:
                if self.environment_starting_up_flags[environment]:
                    self.environment_starting_up_flags[environment] = False
                    self.environment_active_flags[environment] = True
                    self.log('Received First Complete Data Write for {:} Environment'.format(environment))
                    self.environment_first_data[environment] = True
                
        # If we got through that previous loop still ready to write, we can
        # output the next signal if the hardware is ready
        if ready_to_write and (self.startup or skip_hardware or self.hardware.ready_for_new_output()):
            self.log('Ready to Write: Environment Remainders {:}'.format([(environment,remainder.shape[-1])
                                for environment,remainder in self.environment_data_out_remainders.items() if self.environment_active_flags[environment]]))
            write_data = np.zeros((self.num_outputs,self.write_size))
            for environment in self.environment_list:
                # If the task is shutting down and all the data has been drained from it, make it inactive and just skip it.
                if self.environment_shutting_down_flags[environment] and self.environment_data_out_remainders[environment].shape[-1] == 0:
                    self.environment_active_flags[environment] = False
                    self.environment_starting_up_flags[environment] = False
                    self.queue_container.acquisition_command_queue.put(self.process_name,(GlobalCommands.STOP_ENVIRONMENT,environment))
                    self.environment_shutting_down_flags[environment] = False
                    continue
                # If the task is inactive, also just skip it
                elif not self.environment_active_flags[environment]:
                    continue
                # Get the indices corresponding to the output channels
                output_indices = self.environment_output_channels[environment]
                # Determine how many time steps are available to write
                output_timesteps = min(self.environment_data_out_remainders[environment].shape[-1],self.write_size)
                # Write one portion of the environment output to write_data
                write_data[output_indices,:output_timesteps] += self.environment_data_out_remainders[environment][:,:output_timesteps]
                self.environment_data_out_remainders[environment] = self.environment_data_out_remainders[environment][:,output_timesteps:]
            # Now that we have each environment accounted for in the output we
            # can write to the hardware
            self.log('Writing {:} data to hardware'.format(' x '.join('{:}'.format(shape) for shape in write_data.shape)))
            if not skip_hardware:
                self.log('Output Writing Data to Hardware RMS: \n  {:}'.format((rms_time(write_data,axis=-1))))
                for environment in self.environment_first_data:
                    if self.environment_first_data[environment]:
                        self.log('Sending first data for environment {:} to Acquisition for syncing'.format(environment))
                        self.queue_container.input_output_sync_queue.put((environment,write_data[...,::self.output_oversample].copy()))
                        self.environment_first_data[environment] = False
                self.hardware.write(write_data)
            else:
                if self.environment_first_data[environment]:
                    self.queue_container.input_output_sync_queue.put((environment,0))
                    self.environment_first_data[environment] = False
#            np.savez('test_data/output_data_check.npz',output_data = write_data)
            # Now check and see if we are starting up and start the hardare if so
            if self.startup:
                self.log('Starting Hardware Output')
                if not skip_hardware:
                    self.hardware.start()
                # Send something to the sync queue to tell acquisition to start now
                # We will send None because it is unique and we won't have an
                # environment with that name
                self.queue_container.input_output_sync_queue.put((None,True))
                self.startup = False
                self.output_active = True
                # print('started output')
        # Now check if we need to shut down.
        if (self.shutdown_flag # Time to shut down
            and all([not flag for environment,flag in self.environment_active_flags.items()]) # Check that all environments are not active
            and all([not flag for environment,flag in self.environment_starting_up_flags.items()]) # Check that all environments are not starting up
            and all([remainder.shape[-1] == 0 for environment,remainder in self.environment_data_out_remainders.items()]) # Check that all data is written
            ):
            self.log('Stopping Hardware')
            if not skip_hardware:
                self.hardware.stop()
            self.startup = True
            self.shutdown_flag = False
            flush_queue(self.queue_container.input_output_sync_queue)
            self.output_active = False
        else:
            # Otherwise keep going
            self.queue_container.output_command_queue.put(self.process_name,(GlobalCommands.RUN_HARDWARE,None))
    
    def stop_output(self,data):
        """Sets a flag telling the output that it should start shutting down

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.log('Starting Shutdown Procedure')
        self.shutdown_flag = True
    
    def start_environment(self,data):
        """Sets the flag stating the specified environment is active

        Parameters
        ----------
        data : str
            The environment name that should be activated.

        """
        self.log('Started Environment {:}'.format(data))
        self.environment_starting_up_flags[data] = True
        self.environment_shutting_down_flags[data] = False
        self.environment_active_flags[data] = False
    
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
        # for name in dir(self.queue_container):
        #     queue_data = getattr(self.queue_container,name) 
        #     if isinstance(queue_data,VerboseMessageQueue) or isinstance(queue_data,mp.queues.Queue):
        #         queue_flush_sum += len(flush_queue(queue))
        #     elif isinstance(queue_data,)
        for queue in [q for name,q in self.queue_container.environment_data_out_queues.items()] + [self.queue_container.output_command_queue,
                                                                                                   self.queue_container.input_output_sync_queue,
                                                                                                   self.queue_container.single_process_hardware_queue]:
            queue_flush_sum += len(flush_queue(queue))
        self.log('Flushed {:} items out of queues'.format(queue_flush_sum))
        if not self.hardware is None:
            self.hardware.close()
        return True

def output_process(queue_container : QueueContainer,environments : list, output_active : mp.Value):
    """Function passed to multiprocessing as the output process
    
    This process creates the ``OutputProcess`` object and calls the ``run``
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

    output_instance = OutputProcess('Output',queue_container,environments, output_active)
    
    output_instance.run()