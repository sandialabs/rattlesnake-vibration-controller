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
                        DataAcquisitionParameters,GlobalCommands,VerboseMessageQueue)
from .abstract_message_process import AbstractMessageProcess
from enum import Enum
import copy

test_level_threshold = 1.01

class SysIdSignalGenerationCommands(Enum):
    """Commands that the System ID Signal Generation Process can accept"""
    INITIALIZE_PARAMETERS = 0
    RUN_TRANSFER_FUNCTION = 1
    MUTE = 2
    ADJUST_TEST_LEVEL = 3
    START_SHUTDOWN = 4
    QUIT_EVENTUALLY = 5

class SysIdParameters:
    def __init__(self):
        pass

class SysIdSignalGenerationProcess(AbstractMessageProcess):
    """Class encapsulating the signal generation process for system identification
    
    This class handles the signal generation of the system identification.
    It accepts level data from the controller and transforms that data into
    time histories that are put into the data_out_queue."""
    def __init__(self,process_name : str, instruction_queue : VerboseMessageQueue,
                 log_file_queue : mp.queues.Queue,
                 gui_update_queue : mp.queues.Queue,
                 data_out_queue : mp.queues.Queue, 
                 controller_communication_queue : VerboseMessageQueue,
                 environment_name:str):
        """
        Class encapsulating the signal generation process for system identification

        Parameters
        ----------
        process_name : str
            A name to assign the process, primarily for logging purposes.
        instruction_queue : VerboseMessageQueue
            A queue from which instructions will be obtained
        log_file_queue : mp.queues.Queue
            A queue to which log file messages are written
        gui_update_queue : mp.queues.Queue
            A queue to which gui updates are written
        data_out_queue : mp.queues.Queue
            A queue to which time signal data will be put
        controller_communication_queue : VerboseMessageQueue
            A queue to which global communications to the controller are put
        environment_name : str
            The name of the environment that this process is generating signals for.

        """
        super().__init__(process_name,log_file_queue,instruction_queue,gui_update_queue)
        self.map_command(SysIdSignalGenerationCommands.INITIALIZE_DATA_ACQUISITION,self.initialize_data_acquisition)
        self.map_command(SysIdSignalGenerationCommands.INITIALIZE_TEST_PARAMETERS,self.initialize_test_parameters)
        self.map_command(SysIdSignalGenerationCommands.RUN_TRANSFER_FUNCTION,self.run_transfer_function)
        self.map_command(SysIdSignalGenerationCommands.MUTE,self.mute)
        self.map_command(SysIdSignalGenerationCommands.ADJUST_TEST_LEVEL,self.adjust_test_level)
        self.map_command(SysIdSignalGenerationCommands.START_SHUTDOWN,self.start_shutdown)
        self.map_command(SysIdSignalGenerationCommands.QUIT_EVENTUALLY,self.quit_eventually)
        self.environment_name = environment_name
        self.data_out_queue = data_out_queue
        self.instruction_queue = instruction_queue
        self.controller_communication_queue = controller_communication_queue
        self.signal_samples = None
        self.end_samples = None
        self.cola_window = None
        self.cola_window_exponent = None
        self.sample_rate = None
        self.df = None
        self.num_output_channels = None
        self.cola_queue = None
        self.rms = None
        self.output_transformation_matrix = None
        self.test_level_target = 1
        self.current_test_level = 0
        self.ramp_samples = None
        self.test_level_change = 0
        self.startup = True
        self.shutdown_flag = False
        self.output_oversample = None
        self.quit_flag = False
        
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
    
    def initialize_test_parameters(self,data):
        """Stores environment signal processing parameters to the process

        Parameters
        ----------
        data : 
            A 3-tuple containing the signal length, frequency spacing, ramp samples,
            and output transformation matrix

        """
        signal_length,df,ramp_samples,output_transformation_matrix = data
        self.signal_samples = int(signal_length*(0.5))
        self.end_samples = signal_length - self.signal_samples
        self.cola_window = 'hann'
        self.cola_window_exponent = 0.5
        self.df = df
        self.output_transformation_matrix = None if output_transformation_matrix is None else np.linalg.pinv(output_transformation_matrix)
        self.ramp_samples = ramp_samples*self.output_oversample
        # Reset number of output channels if there is a transformation
        # matrix
        if not self.output_transformation_matrix is None:
            self.num_output_channels = self.output_transformation_matrix.shape[1]
        # Only reset the data to zeros if it is not the right shape
        if (self.cola_queue is None 
            or self.cola_queue.shape != (2,self.num_output_channels,signal_length*self.output_oversample)
            ):
            self.cola_queue = np.zeros((2,self.num_output_channels,
                                            signal_length*self.output_oversample),dtype='float32')
                        
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
        and perform more operations described in the SysIdSignalGenerationProcess.shutdown
        function.

        Parameters
        ----------
        data : Ignored
            Unused argument required due to the expectation that functions called
            by the SysIdSignalGenerationProcess.Run function will have one argument
            accepting any data passed along with the instruction.

        """
        if self.startup:
            self.rms = data
            # Fill the COLA queue the first time through
            # np.savez('test_data/signal_generation_transfer_function_argument_check.npz',
            #           fmin = 0.0, fmax = self.sample_rate/2, df = self.df,
            #           sample_rate = self.sample_rate*self.output_oversample,
            #           rms = data.real,
            #           nsignals = self.num_output_channels)
            self.cola_queue[-1,...] = pseudorandom_signal(0.0, self.sample_rate/2, self.df, self.sample_rate*self.output_oversample, self.rms,self.num_output_channels)
            self.startup=False
        if self.data_out_queue.empty(): # May need to change this to some kind of buffer if it takes longer to COLA the signal than it does to output the signal
            # Create the new signal
            signal = pseudorandom_signal(0.0, self.sample_rate/2, self.df, self.sample_rate*self.output_oversample, self.rms,self.num_output_channels)
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
                return
        self.instruction_queue.put(self.process_name,(SysIdSignalGenerationCommands.RUN_TRANSFER_FUNCTION,None))
    
    
    
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
            by the SysIdSignalGenerationProcess.Run function will have one argument
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
            by the SysIdSignalGenerationProcess.Run function will have one argument
            accepting any data passed along with the instruction.

        """
        if self.shutdown_flag == True or self.startup == True: # This means we weren't supposed to shutdown, it was an extra signal.
            return
        # Disable the volume controls
        self.gui_update_queue.put((self.environment_name,('disable','voltage_scale_factor_selector')))
        self.gui_update_queue.put((self.environment_name,('disable','test_level_selector')))
        self.shutdown_flag = True
        # Get any commands that might be in the queue currently
        commands = self.instruction_queue.flush(self.process_name)
        commands = [command[0] for command in commands]
        # Turn down the volume to zero
        self.instruction_queue.put(self.process_name,(SysIdSignalGenerationCommands.ADJUST_TEST_LEVEL,0.0))
        # Put the run command back onto the stack.
        if SysIdSignalGenerationCommands.RUN_TRANSFER_FUNCTION in commands:
            self.instruction_queue.put(self.process_name,(SysIdSignalGenerationCommands.RUN_TRANSFER_FUNCTION,None))
            self.controller_communication_queue.put(self.process_name,(GlobalCommands.STOP_HARDWARE,None))

    def shutdown(self):
        """Performs final cleanup operations when the system has shut down
        
        This function is called when the signal generation has been instructed
        to shut down and the test level has reached zero.
        """
        self.log('Shutting Down Signal Generation')
        self.instruction_queue.flush(self.process_name)
        # Enable the volume controls
        self.gui_update_queue.put((self.environment_name,('enable','voltage_scale_factor_selector')))
        self.gui_update_queue.put((self.environment_name,('enable','test_level_selector')))
        self.startup = True
        self.shutdown_flag = False
        if self.quit_flag:
            self.instruction_queue.put(self.process_name,(GlobalCommands.QUIT,None))

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
        self.data_out_queue.put((copy.deepcopy(write_data*test_level),last_signal))
        
    def quit_eventually(self,data):
        """
        Tells the process that it should quit after shutdown

        Parameters
        ----------
        data :  None
            Unused argument required due to the expectation that functions called
            by the SysIdSignalGenerationProcess.Run function will have one argument
            accepting any data passed along with the instruction.

        Returns
        -------
        None.

        """
        self.quit_flag = True

def system_id_signal_generation_process(environment_name : str,  
                                        instruction_queue : VerboseMessageQueue,
                                        log_file_queue : mp.queues.Queue,
                                        gui_update_queue : mp.queues.Queue,
                                        data_out_queue : mp.queues.Queue, 
                                        controller_communication_queue : VerboseMessageQueue):
    """Signal generation process function called by multiprocessing
    
    This function defines the Signal Generation process for system identification that
    gets run by the multiprocessing module when it creates a new process.  It
    creates a SysIdSignalGenerationProcess object and runs it.

    Parameters
    ----------
    environment_name : str :
        Name of the environment associated with this signal generation process
    instruction_queue : VerboseMessageQueue
        A queue from which instructions will be obtained
    log_file_queue : mp.queues.Queue
        A queue to which log file messages are written
    gui_update_queue : mp.queues.Queue
        A queue to which gui updates are written
    data_out_queue : mp.queues.Queue
        A queue to which time signal data will be put
    controller_communication_queue : VerboseMessageQueue
        A queue to which global communications to the controller are put
    """

    signal_generation_instance = SysIdSignalGenerationProcess(environment_name+' Signal Generation',
                                                               instruction_queue,
                                                               log_file_queue,
                                                               gui_update_queue,
                                                               data_out_queue, 
                                                               controller_communication_queue,
                                                               environment_name)
    
    signal_generation_instance.run()
