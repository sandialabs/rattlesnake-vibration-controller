# -*- coding: utf-8 -*-
"""
Controller subsystem that handles the creation of the signals from an output
CPSD matrix, including Constant Overlap and Add.

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
import numpy as np
from .utilities import (cola,flush_queue,
                        cpsd_to_time_history,rms_time,pseudorandom_signal,
                        DataAcquisitionParameters,GlobalCommands)
from .abstract_message_process import AbstractMessageProcess
from enum import Enum
from .random_vibration_environment import RandomEnvironmentQueues,RandomVibrationParameters,RandomVibrationCommands
from .random_vibration_data_collector import RandomDataCollectorMessages
import copy

test_level_threshold = 1.01

class RandomSignalGenerationMessages(Enum):
    """Commands that the Random Vibration Signal Generation Process can accept"""
    INITIALIZE_DATA_ACQUISITION = 0
    INITIALIZE_TEST_PARAMETERS = 1
    RUN_TRANSFER_FUNCTION = 2
    RUN_CONTROL = 3
    MUTE = 4
    ADJUST_TEST_LEVEL = 5
    START_SHUTDOWN = 6

class RandomSignalGenerationProcess(AbstractMessageProcess):
    """Class encapsulating the signal generation process for random vibration
    
    This class handles the signal generation of the Random Vibration environment.
    It accepts data from the data analysis process and transforms that data into
    time histories that are put into the data_out_queue."""
    def __init__(self,process_name : str, queues : RandomEnvironmentQueues,environment_name:str):
        """
        Class encapsulating the signal generation process for random vibration

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
        super().__init__(process_name,queues.log_file_queue,queues.signal_generation_command_queue,queues.gui_update_queue)
        self.map_command(RandomSignalGenerationMessages.INITIALIZE_DATA_ACQUISITION,self.initialize_data_acquisition)
        self.map_command(RandomSignalGenerationMessages.INITIALIZE_TEST_PARAMETERS,self.initialize_test_parameters)
        self.map_command(RandomSignalGenerationMessages.RUN_TRANSFER_FUNCTION,self.run_transfer_function)
        self.map_command(RandomSignalGenerationMessages.RUN_CONTROL,self.run_control)
        self.map_command(RandomSignalGenerationMessages.MUTE,self.mute)
        self.map_command(RandomSignalGenerationMessages.ADJUST_TEST_LEVEL,self.adjust_test_level)
        self.map_command(RandomSignalGenerationMessages.START_SHUTDOWN,self.start_shutdown)
        self.environment_name = environment_name
        self.queue_container = queues
        self.signal_samples = None
        self.end_samples = None
        self.cola_window = None
        self.cola_window_exponent = None
        self.sample_rate = None
        self.df = None
        self.num_output_channels = None
        self.cola_queue = None
        self.last_cpsd = None
        self.output_transformation_matrix = None
        self.test_level_target = 1
        self.current_test_level = 0
        self.ramp_samples = None
        self.skip_frames = None
        self.test_level_change = 0
        self.startup = True
        self.shutdown_flag = False
        self.output_oversample = None
        
    def initialize_data_acquisition(self,data : DataAcquisitionParameters):
        """Stores global Data Acquisition Parameters to the process

        Parameters
        ----------
        data : DataAcquisitionParameters :
            The data acquisition parameters of the controller.

        """
        self.log('Initializing Data Acquisition')
        self.num_output_channels = len([channel for channel in data.channel_list if not channel.feedback_device is None])
        self.sample_rate = data.sample_rate
        self.output_oversample = data.output_oversample
        self.samples_per_read = data.samples_per_read
    
    def initialize_test_parameters(self,data : RandomVibrationParameters):
        """Stores environment signal processing parameters to the process

        Parameters
        ----------
        data : RandomVibrationParameters :
            Signal processing parameters for the environment

        """
        self.signal_samples = data.samples_per_output
        self.end_samples = data.overlapped_output_samples
        self.cola_window = data.cola_window.lower()
        self.cola_window_exponent = data.cola_window_exponent
        self.df = data.frequency_spacing
        self.output_transformation_matrix = None if data.output_transformation_matrix is None else np.linalg.pinv(data.output_transformation_matrix)
        self.ramp_samples = data.test_level_ramp_time*self.sample_rate*self.output_oversample
        self.skip_frames = int(np.ceil(self.ramp_samples//self.output_oversample/self.samples_per_read))+2
        # Reset number of output channels if there is a transformation
        # matrix
        if not self.output_transformation_matrix is None:
            self.num_output_channels = self.output_transformation_matrix.shape[1]
        # Only reset the data to zeros if it is not the right shape
        if (self.cola_queue is None 
            or self.cola_queue.shape != (2,self.num_output_channels,data.samples_per_frame*self.output_oversample)
            or self.last_cpsd is None
            or self.last_cpsd.shape != (data.fft_lines,self.num_output_channels,self.num_output_channels)
            ):
            self.cola_queue = np.zeros((2,self.num_output_channels,
                                            data.samples_per_frame*self.output_oversample),dtype='float32')
            self.last_cpsd = np.zeros((data.fft_lines,
                        self.num_output_channels,
                        self.num_output_channels),dtype='complex64')
                        
    def run_transfer_function(self,data):
        """Function to handle generation of signals for system identification.

        The function serves several roles, including startup, run, and shutdown
        of the signal generation process.
        
        During startup, the function will initialize the constant overlap and
        add (COLA) queue.  It will also send messages to disable and enable 
        widgets on the user interface.
        
        When running, it will accept an RMS voltage level from the data analysis
        process, from which it will generate a gaussian random signal with that
        RMS level and put it to the COLA queue.  The function determines if a
        new signal is required by checking if the data_out_queue is empty.
        The function determines whether or not to stop by checking if the
        shutdown_flag has been set and the test level is zero.  If this is the
        case, it will perform shutdown operations.  If it is not, the function
        will simply put the same commands that caused the function to run in the
        first place back into the signal_generation_command_queue which will
        cause the function to run again when the instruction is pulled from the
        queue.  The function will therefore be called indefinitely unless other
        commands are issued to the controller and the shutdown critera are met.
        
        During shutdown it will reverse the enabling and disabling of widgets,
        and perform more operations described in the RandomSignalGenerationProcess.shutdown
        function.

        Parameters
        ----------
        data : Ignored
            Unused argument required due to the expectation that functions called
            by the RandomSignalGenerationProcess.Run function will have one argument
            accepting any data passed along with the instruction.

        """
        if self.startup:
            self.queue_container.gui_update_queue.put((self.environment_name,('disable','preview_transfer_function_button')))
            self.queue_container.gui_update_queue.put((self.environment_name,('disable','acquire_transfer_function_button')))
            self.queue_container.gui_update_queue.put((self.environment_name,('enable','stop_transfer_function_button')))
            self.log('Waiting for RMS level from Data Analysis')
            try:
                data = self.queue_container.cpsd_to_generate_queue.get(timeout=10)
            except mp.queues.Empty:
                self.queue_container.gui_update_queue.put(
                    ('error',
                     ('{:} Error'.format(self.process_name),
                      '{:} timed out while waiting for first set of data from data analysis'.format(self.process_name))))
                return
            self.last_cpsd[0,0,0] = data
            # Fill the COLA queue the first time through
            # self.cola_queue[-1,...] = data*np.random.randn(self.num_output_channels,
            #                             (self.signal_samples+self.end_samples)*self.output_oversample)
            # np.savez('test_data/signal_generation_transfer_function_argument_check.npz',
            #           fmin = 0.0, fmax = self.sample_rate/2, df = self.df,
            #           sample_rate = self.sample_rate*self.output_oversample,
            #           rms = data.real,
            #           nsignals = self.num_output_channels)
            self.cola_queue[-1,...] = pseudorandom_signal(0.0, self.sample_rate/2, self.df,
                                      self.sample_rate*self.output_oversample, data.real,self.num_output_channels)
            self.startup=False
        else:
            data = flush_queue(self.queue_container.cpsd_to_generate_queue)
            if len(data) == 0:
                data = self.last_cpsd[0,0,0]
            else:
                data = data[-1]
                self.last_cpsd[0,0,0] = data
        if self.queue_container.data_out_queue.empty(): # May need to change this to some kind of buffer if it takes longer to COLA the signal than it does to output the signal
            # Create the new signal
            # signal = data.real*np.random.randn(self.num_output_channels,
            #                                     (self.signal_samples+self.end_samples)*self.output_oversample)
            signal = pseudorandom_signal(0.0, self.sample_rate/2, self.df, self.sample_rate*self.output_oversample, data.real,self.num_output_channels)
            # Add it to the COLA queue
            self.cola_queue = np.roll(self.cola_queue,-1,axis=0)
            self.cola_queue[-1,...] = signal
            # Perform the overlap and add
            output_signal = cola(self.signal_samples*self.output_oversample,self.end_samples*self.output_oversample,
                                 self.cola_queue,self.cola_window,
                                 self.cola_window_exponent)
            # Get the current test level to see if we need to shut down yet.  We get it here so we output at least one round of all zeros prior to shutting down
            last_run = self.shutdown_flag and self.current_test_level == 0.0
            # Pass the data to the output
            self.output(output_signal,last_run)
            # Run again
            if last_run:
                self.shutdown()
                self.queue_container.gui_update_queue.put((self.environment_name,('enable','preview_transfer_function_button')))
                self.queue_container.gui_update_queue.put((self.environment_name,('enable','acquire_transfer_function_button')))
                self.queue_container.gui_update_queue.put((self.environment_name,('disable','stop_transfer_function_button')))
                return
        self.queue_container.signal_generation_command_queue.put(self.process_name,(RandomSignalGenerationMessages.RUN_TRANSFER_FUNCTION,None))
    
    
    def run_control(self,data):
        """Function to handle generation of signals for controlling the environment.

        The function serves several roles, including startup, run, and shutdown
        of the signal generation process.
        
        During startup, the function will initialize the constant overlap and
        add (COLA) queue.
        
        When running, it will accept a CPSD matrix from the data analysis
        process, from which it will generate a set of signals that satisfy that
        CPSD matrix.  The function determines if a
        new signal is required by checking if the data_out_queue is empty.
        The function determines whether or not to stop by checking if the
        shutdown_flag has been set and the test level is zero.  If this is the
        case, it will perform shutdown operations.  If it is not, the function
        will simply put the same commands that caused the function to run in the
        first place back into the signal_generation_command_queue which will
        cause the function to run again when the instruction is pulled from the
        queue.  The function will therefore be called indefinitely unless other
        commands are issued to the controller and the shutdown critera are met.
        
        During shutdown it will perform more operations described in the
        RandomSignalGenerationProcess.shutdown function.

        Parameters
        ----------
        data : None
            Unused argument required due to the expectation that functions called
            by the RandomSignalGenerationProcess.run function will have one argument
            accepting any data passed along with the instruction.

        """
        if self.startup:
            self.log('Waiting for CPSD from Data Analysis')
            try:
                data = self.queue_container.cpsd_to_generate_queue.get(timeout=10)
            except mp.queues.Empty:
                self.queue_container.gui_update_queue.put(
                    ('error',
                     ('{:} Error'.format(self.process_name),
                      '{:} timed out while waiting for first set of data from data analysis'.format(self.process_name))))
                return
            self.last_cpsd[...] = data
            # Fill the COLA queue the first time through
            self.cola_queue[0,...] = cpsd_to_time_history(data,self.sample_rate,self.df,self.output_oversample)
            self.cola_queue[1,...] = cpsd_to_time_history(data,self.sample_rate,self.df,self.output_oversample)
            self.startup=False
        else:
            data = flush_queue(self.queue_container.cpsd_to_generate_queue)
            if len(data) > 0:
                self.log('Received data from Data Analysis task')
                data = data[-1]
                self.last_cpsd[...] = data
        # Create the time history, I think I'm going to do this either way, and then only write it if we need to
        self.log('Starting Time History Generation')
        signal = cpsd_to_time_history(self.last_cpsd,self.sample_rate,self.df,self.output_oversample)
        # Do 3-sigma clipping
        rms_vals = rms_time(signal,axis=-1)
        for sig,rms_val in zip(signal,rms_vals):
            sig[sig > rms_val*3] = rms_val*3
            sig[sig < -rms_val*3] = -rms_val*3
        self.log('Finished Time History Generation')
        # Now check if we need to use it
        if self.queue_container.data_out_queue.empty():
            self.log('Starting COLA')
            self.cola_queue = np.roll(self.cola_queue,-1,axis=0)
            self.cola_queue[-1,...] = signal
            output_signal = cola(self.signal_samples*self.output_oversample,self.end_samples*self.output_oversample,
                                  self.cola_queue,self.cola_window,
                                  self.cola_window_exponent)
            self.log('Finished COLA')
            # Get the current test level to see if we need to shut down yet.  We get it here so we output at least one round of all zeros prior to shutting down
            last_run = self.shutdown_flag and self.current_test_level == 0.0
            # Pass the data to the output
            self.output(output_signal,last_run)
            # Run again
            if last_run:
                self.shutdown()
                return
        self.queue_container.signal_generation_command_queue.put(self.process_name,(RandomSignalGenerationMessages.RUN_CONTROL,None))
    
    def mute(self,data):
        """Immediately mute the signal generation task
        
        This function should primarily only be called at the beginning of an
        analysis to ensure the system ramps up from zero excitation.  Muting
        the signal generation during excitation will shock load the exciters and
        test article and may damage those hardware.

        Parameters
        ----------
        data : None
            Unused argument required due to the expectation that functions called
            by the RandomSignalGenerationProcess.Run function will have one argument
            accepting any data passed along with the instruction.

        """
        self.current_test_level = 0.0
        self.test_level_target = 0.0
        self.test_level_change = 0.0
        
    def adjust_test_level(self,data):
        """Sets a new target test level and computes the test level change per sample

        Parameters
        ----------
        data : float:
            The new test level target scale factor.

        """
        self.test_level_target = data
        self.test_level_change = (self.test_level_target - self.current_test_level)/self.ramp_samples
        if self.test_level_change != 0.0:
            self.log('Changed test level to {:} from {:}, {:} change per sample'.format(self.test_level_target,self.current_test_level,self.test_level_change))
            self.queue_container.collector_command_queue.put(self.process_name,(RandomDataCollectorMessages.SET_TEST_LEVEL,(self.skip_frames,self.test_level_target)))

    def start_shutdown(self,data):
        """Starts the shutdown process for the signal generation process
        
        This will set the shutdown flag to true and adjust the test level to
        zero.  Note that this does not immediately stop the generation because
        the actual test level will take some time to ramp down to zero as to
        not shock load the test system.

        Parameters
        ----------
        data : None
            Unused argument required due to the expectation that functions called
            by the RandomSignalGenerationProcess.Run function will have one argument
            accepting any data passed along with the instruction.

        """
        if self.shutdown_flag == True or self.startup == True: # This means we weren't supposed to shutdown, it was an extra signal.
            return
        # Disable the volume controls
        self.queue_container.gui_update_queue.put((self.environment_name,('disable','voltage_scale_factor_selector')))
        self.queue_container.gui_update_queue.put((self.environment_name,('disable','current_test_level_selector')))
        self.shutdown_flag = True
        # Get any commands that might be in the queue currently
        commands = self.queue_container.signal_generation_command_queue.flush(self.process_name)
        commands = [command[0] for command in commands]
        # Turn down the volume to zero
        self.queue_container.signal_generation_command_queue.put(self.process_name,(RandomSignalGenerationMessages.ADJUST_TEST_LEVEL,0.0))
        # Put the run command back onto the stack.
        if RandomSignalGenerationMessages.RUN_CONTROL in commands:
            self.queue_container.signal_generation_command_queue.put(self.process_name,(RandomSignalGenerationMessages.RUN_CONTROL,None))
            # If we're doing control we don't necessarily want to stop the hardware because other control strategies might still be running
        if RandomSignalGenerationMessages.RUN_TRANSFER_FUNCTION in commands:
            self.queue_container.signal_generation_command_queue.put(self.process_name,(RandomSignalGenerationMessages.RUN_TRANSFER_FUNCTION,None))
            self.queue_container.controller_communication_queue.put(self.process_name,(GlobalCommands.STOP_HARDWARE,None))

    def shutdown(self):
        """Performs final cleanup operations when the system has shut down
        
        This function is called when the signal generation has been instructed
        to shut down and the test level has reached zero.  The signal generation
        is the first process in the Random Vibration environment to stop when
        shutdown is called, so it notifies the environment process to stop the
        acquisition and analysis tasks because it is no longer generating signals
        """
        self.log('Shutting Down Signal Generation')
        self.queue_container.signal_generation_command_queue.flush(self.process_name)
        # Tell the other processes to shut down as well
        self.queue_container.environment_command_queue.put(self.process_name,(RandomVibrationCommands.STOP_ACQUISITION_AND_ANALYSIS,None))
        # Enable the volume controls
        self.queue_container.gui_update_queue.put((self.environment_name,('enable','voltage_scale_factor_selector')))
        self.queue_container.gui_update_queue.put((self.environment_name,('enable','current_test_level_selector')))
        self.startup = True
        self.shutdown_flag = False

    def output(self,write_data,last_signal=False):
        """Puts data to the data_out_queue and handles test level changes
        
        This function keeps track of the environment test level and scales the
        output signals accordingly prior to placing them into the data_out_queue.
        This function also handles the ramping between two test levels.

        Parameters
        ----------
        write_data : np.ndarray
            A numpy array containing the signals to be written.
            
        last_signal :
            Specifies if the signal being written is the last signal that will
            be generated due to the signal generation shutting down.  This is
            passed to the output task to tell it that there will be no more
            signals from this environment until it is restarted. (Default value
            = False)
        """
#        np.savez('test_data/signal_generation_initial_output_data_check.npz',write_data = write_data)
        # Perform the output transformation if necessary
        if not self.output_transformation_matrix is None:
            self.log('Applying Transformation')
            write_data = self.output_transformation_matrix@write_data
        # Compute the test_level scaling for this dataset
        if self.test_level_change == 0.0:
            test_level = self.current_test_level
            self.log('Test Level at {:}'.format(test_level))
        else:
            test_level = self.current_test_level + (np.arange(self.signal_samples*self.output_oversample)+1)*self.test_level_change
            # Compute distance in steps from the target test_level and find where it is near the target
            full_level_index = np.nonzero(abs(test_level - self.test_level_target)/abs(self.test_level_change) < test_level_threshold)[0]
            # Check if any are
            if len(full_level_index) > 0:
                # If so, set all test_levels after that one to the target test_level
                test_level[full_level_index[0]+1:] = self.test_level_target
                # And update that our current test_level is now the target test_level
                self.current_test_level = self.test_level_target
                self.test_level_change = 0.0
            else:
                # Otherwise, our current test_level is the last entry in the test_level scaling
                self.current_test_level = test_level[-1]
            self.log('Test level from {:} to {:}'.format(test_level[0],test_level[-1]))
        # Write the test level-scaled data to the task
        self.log('Sending data to data_out queue')
        # np.savez('test_data/signal_generation_output_data_check.npz',write_data = write_data,test_level = test_level)
        self.queue_container.data_out_queue.put((copy.deepcopy(write_data*test_level),last_signal))
        
   

def random_signal_generation_process(environment_name : str, queue_container: RandomEnvironmentQueues):
    """Random vibration signal generation process function called by multiprocessing
    
    This function defines the Random Vibration Signal Generation process that
    gets run by the multiprocessing module when it creates a new process.  It
    creates a RandomSignalGenerationProcess object and runs it.

    Parameters
    ----------
    environment_name : str :
        Name of the environment associated with this signal generation process
        
    queue_container: RandomEnvironmentQueues :
        A container of queues that allows communcation with the random vibration
        environment as well as the rest of the controller.
    """

    signal_generation_instance = RandomSignalGenerationProcess(environment_name+' Signal Generation',queue_container,environment_name)
    
    signal_generation_instance.run()
