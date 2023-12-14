# -*- coding: utf-8 -*-
"""
Process that handles signal generation

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
from abc import ABC,abstractmethod,abstractproperty
from .utilities import (flush_queue,GlobalCommands,VerboseMessageQueue,rms_time)
from .abstract_message_process import AbstractMessageProcess
from .signal_generation import SignalGenerator
from enum import Enum
import copy
import scipy.signal as sig

test_level_threshold = 1.01

class SignalGenerationCommands(Enum):
    """Commands that the Random Vibration Signal Generation Process can accept"""
    INITIALIZE_PARAMETERS = 0
    INITIALIZE_SIGNAL_GENERATOR = 1
    GENERATE_SIGNALS = 2
    START_SHUTDOWN = 3
    SHUTDOWN = 4
    MUTE = 5
    ADJUST_TEST_LEVEL = 6
    SET_TEST_LEVEL = 7
    SHUTDOWN_ACHIEVED = 8

class SignalGenerationMetadata:
    def __init__(self,samples_per_write,level_ramp_samples,
                 output_transformation_matrix = None,
                 new_signal_sample_threshold = None,
                 disabled_signals = None):
        self.ramp_samples = level_ramp_samples
        self.output_transformation_matrix = output_transformation_matrix
        self.samples_per_write = samples_per_write
        self.new_signal_sample_threshold = self.samples_per_write if new_signal_sample_threshold is None else new_signal_sample_threshold
        self.diabled_signals = [] if disabled_signals is None else disabled_signals

    def __eq__(self,other):
        try:
            return np.all([np.all(self.__dict__[field] == other.__dict__[field]) for field in self.__dict__])
        except (AttributeError,KeyError):
            return False
        
class SignalGenerationProcess(AbstractMessageProcess):
    """Class encapsulating the signal generation process for random vibration
    
    This class handles the signal generation of the Random Vibration environment.
    It accepts data from the data analysis process and transforms that data into
    time histories that are put into the data_out_queue."""
    def __init__(self,process_name : str, 
                 command_queue : VerboseMessageQueue,
                 data_in_queue : mp.queues.Queue,
                 data_out_queue : mp.queues.Queue,
                 environment_command_queue : VerboseMessageQueue,
                 log_file_queue : mp.queues.Queue,
                 gui_update_queue : mp.queues.Queue,
                 environment_name : str):
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
        super().__init__(process_name,log_file_queue,command_queue,gui_update_queue)
        self.map_command(SignalGenerationCommands.INITIALIZE_PARAMETERS,self.initialize_parameters)
        self.map_command(SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,self.initialize_signal_generator)
        self.map_command(SignalGenerationCommands.GENERATE_SIGNALS,self.generate_signals)
        self.map_command(SignalGenerationCommands.START_SHUTDOWN,self.start_shutdown)
        self.map_command(SignalGenerationCommands.SHUTDOWN,self.shutdown)
        self.map_command(SignalGenerationCommands.MUTE,self.mute)
        self.map_command(SignalGenerationCommands.ADJUST_TEST_LEVEL,self.adjust_test_level)
        self.map_command(SignalGenerationCommands.SET_TEST_LEVEL, self.set_test_level)
        self.environment_name = environment_name
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.environment_command_queue = environment_command_queue
        self.ramp_samples = None
        self.output_transformation_matrix = None
        self.samples_per_write = None
        self.new_signal_sample_threshold = None
        self.test_level_target = 1.0
        self.current_test_level = 0.0
        self.test_level_change = 0.0
        self.signal_remainder = None
        self.startup = True
        self.shutdown_flag = False
        self.done_generating = False
        self.signal_generator = None
    
    def initialize_parameters(self,data : SignalGenerationMetadata):
        """Stores environment signal processing parameters to the process

        Parameters
        ----------
        data : Metadata :
            Signal processing parameters for the environment

        """
        self.log('Initializing Test Parameters')
        self.ramp_samples = data.ramp_samples
        self.output_transformation_matrix = None if data.output_transformation_matrix is None else np.linalg.pinv(data.output_transformation_matrix)
        self.samples_per_write = data.samples_per_write
        self.new_signal_sample_threshold = data.new_signal_sample_threshold
        self.disabled_signals = data.diabled_signals
    
    def initialize_signal_generator(self,signal_generator : SignalGenerator):
        self.signal_generator = signal_generator
        self.signal_remainder = None
    
    def generate_signals(self,data):
        """Function to handle generation of signals for controlling the environment.

        Parameters
        ----------
        data : None
            Unused argument required due to the expectation that functions called
            by the RandomSignalGenerationProcess.generate_signals function will have one argument
            accepting any data passed along with the instruction.

        """
        # Check to make sure that the signal generator is defined
        if self.signal_generator is None:
            raise RuntimeError('Signal Generator object not yet defined!')
        # Check to see if we are just starting up
        if self.startup:
            self.log('Starting up output')
            # Check if we are ready to output immediately, otherwise, wait for
            # data to come in.
            if not self.signal_generator.ready_for_next_output:
                self.log('Waiting for Input Data')
                try:
                    data = self.data_in_queue.get(timeout=10)
                except mp.queues.Empty:
                    self.gui_update_queue.put(
                        ('error',
                         ('{:} Error'.format(self.process_name),
                          '{:} timed out while waiting for first set of parameters'.format(self.process_name))))
                    return
                self.signal_generator.update_parameters(*data)
            self.startup=False
        # Check and see if there is any data in the queue that can be used to
        # update the signal generator
        update_data = flush_queue(self.data_in_queue)
        if len(update_data) > 0:
            # Assign the most recent data to the signal generator
            self.log('Got Updated Parameters')
            self.signal_generator.update_parameters(*update_data[-1])
        if ((self.signal_remainder is None
            or self.signal_remainder.shape[-1] < self.new_signal_sample_threshold)
            and not self.done_generating):
            self.log('Generating Frame of Data')
            new_signal,self.done_generating = self.signal_generator.generate_frame()
            self.log('Generated Signal with RMS \n  {:}'.format(rms_time(new_signal,axis=-1)))
            # During the first run through, signal_remainder will be None
            if self.signal_remainder is None:
                self.signal_remainder = new_signal
            else:
                # Otherwise we just concatenate the new data at the end
                self.signal_remainder = np.concatenate((self.signal_remainder,new_signal),axis=-1)
        # Now check if we need to send it to the output task
        if self.data_out_queue.empty():
            self.log('Outputting Data')
            # Determine if this is the last output.  This will be the last output
            # if the shutdown flag is set and the current test level is zero,
            # but also if we are done generating and there is no more signal in
            # the measurement frame.
            signal_to_output = self.signal_remainder[...,:self.samples_per_write]
            self.signal_remainder = self.signal_remainder[...,self.samples_per_write:]
            last_run = ((self.shutdown_flag and self.current_test_level == 0.0)
                        or (self.done_generating and self.signal_remainder.shape[-1] == 0))
            self.output(signal_to_output,last_run)
            # Run again
            if last_run:
                self.log('Received Last Run, Shutting Down')
                self.shutdown()
                return
        self.command_queue.put(self.process_name,(SignalGenerationCommands.GENERATE_SIGNALS,None))

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
        if len(self.disabled_signals) > 0:
            write_data = write_data.copy()
            write_data[self.disabled_signals] = 0
        if not self.output_transformation_matrix is None:
            self.log('Applying Transformation')
            write_data = self.output_transformation_matrix@write_data
        # Compute the test_level scaling for this dataset
        if self.test_level_change == 0.0:
            test_level = self.current_test_level
            self.log('Test Level at {:}'.format(test_level))
        else:
            test_level = self.current_test_level + (np.arange(write_data.shape[-1])+1)*self.test_level_change
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
        # # TODO: Delete This Later
        # self.write_index += 1
        # np.savez('signal_generation_output_data_check_{:}.npz'.format(self.write_index),write_data = write_data,test_level = test_level)
        self.log('Sending Output with RMS \n  {:}'.format(rms_time(write_data*test_level,axis=-1)))
        self.data_out_queue.put((copy.deepcopy(write_data*test_level),last_signal))

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
            by the SignalGenerationProcess.run function will have one argument
            accepting any data passed along with the instruction.

        """
        self.current_test_level = 0.0
        self.test_level_target = 0.0
        self.test_level_change = 0.0

    def set_test_level(self,data):
        """Immediately set the level of the signal generation task
        
        This function should primarily only be called at the beginning of an
        analysis to ensure the system ramps up from zero excitation.  Setting
        the signal generation during excitation will shock load the exciters and
        test article and may damage those hardware.

        Parameters
        ----------
        data : level
            The level of the signal generator

        """
        self.current_test_level = data
        self.test_level_target = data
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
            self.log('Changed test level from {:} to {:}, {:} change per sample'.format(self.current_test_level,self.test_level_target,self.test_level_change))

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
        self.shutdown_flag = True
        self.adjust_test_level(0.0)
        # Get any commands that might be in the queue currently
        commands = self.command_queue.flush(self.process_name)
        commands = [command[0] for command in commands]
        # Put the run command back onto the stack.
        if SignalGenerationCommands.GENERATE_SIGNALS in commands:
            self.command_queue.put(self.process_name,(SignalGenerationCommands.GENERATE_SIGNALS,None))

    def shutdown(self):
        """Performs final cleanup operations when the system has shut down
        
        This function is called when the signal generation has been instructed
        to shut down and the test level has reached zero.  The signal generation
        is the first process in the Random Vibration environment to stop when
        shutdown is called, so it notifies the environment process to stop the
        acquisition and analysis tasks because it is no longer generating signals
        """
        self.log('Shutting Down Signal Generation')
        self.command_queue.flush(self.process_name)
        # Tell the other processes to shut down as well
        self.environment_command_queue.put(self.process_name,(SignalGenerationCommands.SHUTDOWN_ACHIEVED,None))
        self.startup = True
        self.shutdown_flag = False
        self.done_generating = False
        

def signal_generation_process(environment_name : str, 
                              command_queue : VerboseMessageQueue,
                              data_in_queue : mp.queues.Queue,
                              data_out_queue : mp.queues.Queue,
                              environment_command_queue : VerboseMessageQueue,
                              log_file_queue : mp.queues.Queue,
                              gui_update_queue : mp.queues.Queue,
                              process_name : str = None):
    """Signal generation process function called by multiprocessing
    
    This function defines the Signal Generation process that
    gets run by the multiprocessing module when it creates a new process.  It
    creates a SignalGenerationProcess object and runs it.

    Parameters
    ----------
    environment_name : str :
        Name of the environment associated with this signal generation process
    """

    signal_generation_instance = SignalGenerationProcess(
        environment_name+' Signal Generation' if process_name is None
        else process_name,
        command_queue,data_in_queue,
        data_out_queue,environment_command_queue,log_file_queue,
        gui_update_queue,environment_name)
    
    signal_generation_instance.run()
