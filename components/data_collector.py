# -*- coding: utf-8 -*-
"""
A general subprocess that collects data, splits it into measurement frames, then
sends it to the environment.

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
from .utilities import flush_queue,rms_time,VerboseMessageQueue
from .utilities import load_python_module, rms_time
import multiprocessing as mp
import copy
from typing import List
from time import sleep

import numpy as np
from scipy.fft import rfft
import scipy.signal as sig

DEBUG = False

if DEBUG:
    import pickle

class FrameBuffer:
    def __init__(self,num_channels,trigger_index,pretrigger,positive_slope,trigger_level,
                 hysteresis_level,hysteresis_samples,samples_per_frame,maximum_overlap,manual_accept,
                 trigger_enabled,trigger_only_first,wait_samples,
                 dtype='float64',starting_value=np.nan,buffer_size_frame_multiplier = 2):
        self.samples_per_frame = samples_per_frame
        self.trigger_index = trigger_index
        self.pretrigger_samples = int(pretrigger*samples_per_frame) if trigger_enabled else 0
        self.positive_slope = positive_slope
        self.trigger_level = trigger_level
        self.hysteresis_level = hysteresis_level
        self.hysteresis_samples = hysteresis_samples
        self.samples_per_frame = samples_per_frame
        self.overlap_samples = samples_per_frame - int(maximum_overlap*samples_per_frame)
        self.manual_accept = manual_accept
        self.waiting_for_accept = False
        self._buffer = starting_value*np.ones((num_channels,int(np.ceil(buffer_size_frame_multiplier*samples_per_frame))))
        self.buffer_size_frame_multiplier = buffer_size_frame_multiplier
        self.wait_samples = wait_samples
        self.last_trigger = self.overlap_samples-self.wait_samples
        self.last_reset = self.overlap_samples+1-self.wait_samples
        self.trigger_enabled = trigger_enabled
        self.trigger_only_first = trigger_only_first
        self.first_trigger = True
    
    @property
    def buffer_data(self):
        return self._buffer
    
    def add_data(self,data):
        data = np.array(data)
        self.last_trigger += data.shape[-1]
        self.last_reset += data.shape[-1]
        # Make sure the data will fit into the buffer
        data = data[...,-self.buffer_data.shape[-1]:]
        # Figure out how much we need to roll the buffer
        self.buffer_data[:] = np.concatenate((self.buffer_data[...,data.shape[-1]:],
                                           data),axis=-1)
    
    def find_triggers(self):
        # print('Finding Triggers, first trigger {:}'.format(self.first_trigger))
        if self.manual_accept and self.waiting_for_accept:
            # print('Waiting for Accept')
            return []
        if self.trigger_enabled and ((self.trigger_only_first and self.first_trigger) or 
                                     not self.trigger_only_first):
            # print('Getting trigger based on signal')
            trigger_data = self.buffer_data[self.trigger_index,self.pretrigger_samples:self.samples_per_frame+self.pretrigger_samples]
            if self.positive_slope:
                indices = (trigger_data[:-1] < self.trigger_level) & (trigger_data[1:] > self.trigger_level)
                reset_indices = trigger_data < self.hysteresis_level
            else:
                indices = (trigger_data[:-1] > self.trigger_level) & (trigger_data[1:] < self.trigger_level)
                reset_indices = trigger_data > self.hysteresis_level
            if self.hysteresis_samples > 1:
                zeros = ~reset_indices
                iszero = np.concatenate(([0], np.equal(zeros, 0).view(np.int8), [0]))
                absdiff = np.abs(np.diff(iszero))
                ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
                reset_indices = np.array([r[-1]-1 for r in ranges if r[1]-r[0] > self.hysteresis_samples-2])
            triggers = list(self.buffer_size_frame_multiplier*self.samples_per_frame - (np.where(indices)[0]+1+self.pretrigger_samples))
            resets = np.concatenate([[self.last_reset],self.buffer_size_frame_multiplier*self.samples_per_frame - reset_indices - self.pretrigger_samples])
            
            final_triggers = []
            
            while len(triggers) > 0:
                trigger = triggers.pop(0)
                # Check to see if the trigger is far enough away from the last one
                if self.last_trigger - trigger < self.overlap_samples:
                    continue
                # Check to see if there has been a reset since the last trigger
                if not np.any((resets < self.last_trigger) & (resets > trigger)):
                    continue
                if (self.trigger_only_first and not self.first_trigger):
                    continue
                final_triggers.append(trigger)
                self.first_trigger = False
                self.last_trigger = trigger
                
            self.last_reset = resets.min()
            
            if self.manual_accept and len(final_triggers) > 0:
                self.last_trigger = self.overlap_samples
                self.last_reset = self.overlap_samples-1
                self.waiting_for_accept = True
                return [final_triggers[0]]
            else:
                return final_triggers
        else:
            # Get the next triggers that are in the data
            # print('Getting trigger based on spacing')
            last_trigger_rectified = self.buffer_size_frame_multiplier*self.samples_per_frame - self.last_trigger
            triggers_available = int((self.samples_per_frame - last_trigger_rectified)/self.overlap_samples)
            final_triggers = [self.last_trigger - (i+1)*self.overlap_samples for i in range(triggers_available)]
            if len(final_triggers) > 0:
                self.last_trigger = final_triggers[-1]
            return final_triggers
    
    def reset_trigger(self):
        self.last_trigger = self.overlap_samples-self.wait_samples
        self.last_reset = self.overlap_samples-1-self.wait_samples
    
    def accept(self):
        self.last_trigger = self.overlap_samples
        self.last_reset = self.overlap_samples-1
        self.waiting_for_accept = False
    
    def add_data_get_frame(self,data):
        self.add_data(data)
        # print('Last Trigger: {:}'.format(self.last_trigger))
        triggers = self.find_triggers()
        # print('Triggers: {:}'.format(triggers))
        frame_indices = (self.buffer_size_frame_multiplier*self.samples_per_frame - np.array(triggers)[:,np.newaxis] + np.arange(self.samples_per_frame) - self.pretrigger_samples).astype(int)
        return np.moveaxis(self.buffer_data[:,frame_indices],1,0)
    
    def __getitem__(self,key):
        return self._buffer[key]
    
    def __setitem__(self,key,val):
        self._buffer[key] = val
    
class DataCollectorCommands(Enum):
    """Commands that the Random Data Collector Process can accept"""
    INITIALIZE_COLLECTOR = 1
    FORCE_INITIALIZE_COLLECTOR = 2
    ACQUIRE = 3
    STOP = 4
    ACCEPT = 5
    SET_TEST_LEVEL = 6
    ACCEPTED = 7
    SHUTDOWN_ACHIEVED = 8

class AcquisitionType(Enum):
    FREE_RUN = 0
    TRIGGER_EVERY_FRAME = 1
    TRIGGER_FIRST_FRAME = 2
    
class Acceptance(Enum):
    MANUAL = 0
    AUTOMATIC = 1
    
class TriggerSlope(Enum):
    POSITIVE = 0
    NEGATIVE = 1
    
class Window(Enum):
    RECTANGLE = 0
    HANN = 1
    HAMMING = 2
    FLATTOP = 3
    TUKEY = 4
    BLACKMANHARRIS = 5
    EXPONENTIAL = 6
    EXPONENTIAL_FORCE = 7

class CollectorMetadata:
    def __init__(self,
                 num_channels,
                 response_channel_indices,
                 reference_channel_indices,
                 acquisition_type,
                 acceptance,
                 acceptance_function,
                 overlap_fraction,
                 trigger_channel_index,
                 trigger_slope,
                 trigger_level,
                 trigger_hysteresis,
                 trigger_hysteresis_samples,
                 pretrigger_fraction,
                 frame_size,
                 window,
                 window_parameter_1 = 0,
                 window_parameter_2 = 0,
                 window_parameter_3 = 0,
                 wait_samples = 0,
                 response_transformation_matrix = None,
                 reference_transformation_matrix = None):
        self.num_channels = num_channels
        self.response_channel_indices = response_channel_indices
        self.reference_channel_indices = reference_channel_indices
        self.acquisition_type = acquisition_type
        self.acceptance = acceptance
        self.acceptance_function = acceptance_function
        self.overlap_fraction = overlap_fraction
        self.trigger_channel_index = trigger_channel_index
        self.trigger_slope = trigger_slope
        self.trigger_level = trigger_level
        self.trigger_hysteresis = trigger_hysteresis
        self.trigger_hysteresis_samples = trigger_hysteresis_samples
        self.pretrigger_fraction = pretrigger_fraction
        self.frame_size = frame_size
        self.window = window
        self.window_parameter_1 = window_parameter_1
        self.window_parameter_2 = window_parameter_2
        self.window_parameter_3 = window_parameter_3
        self.response_transformation_matrix = response_transformation_matrix
        self.reference_transformation_matrix = reference_transformation_matrix
        self.wait_samples = wait_samples
        
    def __eq__(self,other):
        try:
            return np.all([np.all(self.__dict__[field] == other.__dict__[field]) for field in self.__dict__])
        except (AttributeError,KeyError):
            return False

class DataCollectorProcess(AbstractMessageProcess):
    """Class that takes data from the data_in_queue and distributes to the environment
    
    This class keeps track of the test level used when acquiring data so the 
    data can be scaled back to full level for control.  It will also skip
    frames that are acquired while the system is ramping."""
    def __init__(self,process_name : str, 
                 command_queue : VerboseMessageQueue,
                 data_in_queue : mp.queues.Queue,
                 data_out_queues : List[mp.queues.Queue],
                 environment_command_queue : VerboseMessageQueue,
                 log_file_queue : mp.queues.Queue,
                 gui_update_queue : mp.queues.Queue,
                 environment_name):
        """
        Constructs the data collector class

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
        self.map_command(DataCollectorCommands.INITIALIZE_COLLECTOR,
                         self.initialize_collector)
        self.map_command(DataCollectorCommands.FORCE_INITIALIZE_COLLECTOR,
                         self.force_initialize_collector)
        self.map_command(DataCollectorCommands.ACQUIRE,self.acquire)
        self.map_command(DataCollectorCommands.STOP,self.stop)
        self.map_command(DataCollectorCommands.ACCEPT,self.accept)
        self.map_command(DataCollectorCommands.SET_TEST_LEVEL,self.set_test_level)
        self.environment_command_queue = environment_command_queue
        self.environment_name = environment_name
        self.collector_metadata = None
        self.frame_buffer = None
        self.reference_window = None
        self.response_window = None
        self.window_correction_factor = None
        self.acceptance_function = None
        self.skip_frames = 0
        self.test_level = None
        self.data_in_queue = data_in_queue
        self.data_out_queues = data_out_queues
        if DEBUG:
            self.received_data_index = 0
        
    def initialize_collector(self,data : CollectorMetadata):
        if not (self.collector_metadata == data):
            self.force_initialize_collector(data)
        
    def force_initialize_collector(self,data : CollectorMetadata):
        # Flush the outputs to make sure that there's nothing hanging out on
        # the queue when we start up.
        for queue in self.data_out_queues:
            flush_queue(queue)
        self.collector_metadata = data
        self.frame_buffer = FrameBuffer(
            self.collector_metadata.num_channels,
            self.collector_metadata.trigger_channel_index,
            self.collector_metadata.pretrigger_fraction,
            self.collector_metadata.trigger_slope == TriggerSlope.POSITIVE,
            self.collector_metadata.trigger_level,
            self.collector_metadata.trigger_hysteresis,
            self.collector_metadata.trigger_hysteresis_samples,
            self.collector_metadata.frame_size,
            self.collector_metadata.overlap_fraction,
            self.collector_metadata.acceptance == Acceptance.MANUAL,
            self.collector_metadata.acquisition_type != AcquisitionType.FREE_RUN,
            self.collector_metadata.acquisition_type == AcquisitionType.TRIGGER_FIRST_FRAME,
            self.collector_metadata.wait_samples)
        if self.collector_metadata.acceptance_function is None:
            self.acceptance_function = lambda x: True
        else:
            module = load_python_module(self.collector_metadata.acceptance_function[0])
            self.acceptance_function = getattr(module,self.collector_metadata.acceptance_function[1])
            self.log('Loaded acceptance function')
        if self.collector_metadata.window == Window.RECTANGLE:
            self.reference_window = 1
            self.response_window = 1
        elif self.collector_metadata.window == Window.HANN:
            self.reference_window = sig.get_window(
                'hann',self.collector_metadata.frame_size)
            self.response_window = self.reference_window.copy()
        elif self.collector_metadata.window == Window.HAMMING:
            self.reference_window = sig.get_window(
                'hamming',self.collector_metadata.frame_size)
            self.response_window = self.reference_window.copy()
        elif self.collector_metadata.window == Window.FLATTOP:
            self.reference_window = sig.get_window(
                'flattop',self.collector_metadata.frame_size)
            self.response_window = self.reference_window.copy()
        elif self.collector_metadata.window == Window.TUKEY:
            self.reference_window = sig.get_window(
                ('tukey',self.collector_metadata.window_parameter_1),
                 self.collector_metadata.frame_size)
            self.response_window = self.reference_window.copy()
        elif self.collector_metadata.window == Window.BLACKMANHARRIS:
            self.reference_window = sig.get_window(
                'blackmanharris',self.collector_metadata.frame_size)
            self.response_window = self.reference_window.copy()
        elif self.collector_metadata.window == Window.EXPONENTIAL:
            self.reference_window = sig.get_window(
                ('exponential',self.collector_metadata.window_parameter_1,
                 self.collector_metadata.window_parameter_2),
                 self.collector_metadata.frame_size)
            self.response_window = self.reference_window.copy()
        elif self.collector_metadata.window == Window.EXPONENTIAL_FORCE:
            self.reference_window = sig.get_window(
                ('exponential',self.collector_metadata.window_parameter_2,
                 self.collector_metadata.window_parameter_3),
                 self.collector_metadata.frame_size)
            self.response_window = self.reference_window.copy()
            non_pulse_samples = (np.arange(self.collector_metadata.frame_size)+1) / self.collector_metadata.frame_size > self.collector_metadata.window_parameter_1
            self.reference_window[non_pulse_samples] = 0
        else:
            raise ValueError('Invalid Window Type')
        self.window_correction = np.sqrt(1 / np.mean(self.response_window**2))
        if DEBUG:
            with open('debug_data/collector_metadata.pkl','wb') as f:
                pickle.dump(self.collector_metadata,f)
            with open('debug_data/framebuffer.pkl','wb') as f:
                pickle.dump(self.frame_buffer,f)
            self.received_data_index = 0
        
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
            acquisition_data,last_data = self.data_in_queue.get(timeout=10)
            self.log('Acquired Data with shape {:} and Last Data {:}'.format(acquisition_data.shape,last_data))
            self.log('Data Average RMS: {:0.4f}'.format(rms_time(acquisition_data)))
        except mp.queues.Empty:
            # Keep running until stopped
#            self.log('No Incoming Data!')
            self.command_queue.put(self.process_name,(DataCollectorCommands.ACQUIRE,None))
            return
        # Add data to buffer
        self.log('Putting Data to Buffer')
        output_frames = self.frame_buffer.add_data_get_frame(acquisition_data)
        if DEBUG:
            np.save('debug_data/acquisition_data_{:05d}.npy'.format(self.received_data_index),
                    acquisition_data)
            np.save('debug_data/output_frames_{:05d}.npy'.format(self.received_data_index),
                    output_frames)
            with open('debug_data/framebuffer_{:05d}.pkl'.format(self.received_data_index),'wb') as f:
                pickle.dump(self.frame_buffer,f)
            self.received_data_index += 1
        if output_frames.shape[0] > 0:
            self.log('Measurement Frames Received ({:})'.format(output_frames.shape[0]))
            for frame in output_frames:
                if self.skip_frames > 0:
                    self.skip_frames -= 1
                    self.log('Skipped Frame, {:} left to skip'.format(self.skip_frames))
                    # Reset the buffer.  It isn't clear if this is needed, and in the current implementation it breaks things...
                    # self.frame_buffer.reset_trigger()
                    continue
                frame = np.copy(frame)
                accepted = self.acceptance_function(frame)
                response_frame = frame[self.collector_metadata.response_channel_indices]
                reference_frame = frame[self.collector_metadata.reference_channel_indices]
                if self.collector_metadata.response_transformation_matrix is not None:
                    response_frame = self.collector_metadata.response_transformation_matrix@response_frame
                if self.collector_metadata.reference_transformation_matrix is not None:
                    reference_frame = self.collector_metadata.reference_transformation_matrix@reference_frame
                self.log('Received output from framebuffer with RMS: \n  {:}'.format(rms_time(reference_frame,axis=-1)))
                # Apply window functions
                response_frame*= self.response_window/self.test_level
                reference_frame*= self.reference_window/self.test_level
                if accepted and not self.frame_buffer.manual_accept:
                    self.gui_update_queue.put((self.environment_name,('time_frame',(frame,True))))
                    self.log('Sending data')
                    # Separate into response and reference
                    response_fft = rfft(response_frame,axis=-1)*self.window_correction
                    reference_fft = rfft(reference_frame,axis=-1)*self.window_correction
                    for queue in self.data_out_queues:
                        queue.put(copy.deepcopy(
                            (response_fft,
                             reference_fft)))
                    self.log('Sent Data')
                elif self.frame_buffer.manual_accept:
                    self.last_frame = frame
                    self.gui_update_queue.put((self.environment_name,('time_frame',(frame,False))))
                else:
                    self.gui_update_queue.put((self.environment_name,('time_frame',(frame,False))))
        # Keep running until stopped
        if not last_data:
            self.command_queue.put(self.process_name,(DataCollectorCommands.ACQUIRE,None))
        else:
            self.stop(None)
        
    def accept(self,keep_frame):
        self.log('Received Accept Signal {:}'.format(keep_frame))
        self.frame_buffer.accept()
        if keep_frame:
            self.log('Sending data manually')
            self.gui_update_queue.put((self.environment_name,('time_frame',(self.last_frame,True))))
            frame_fft = rfft(self.last_frame,axis=-1)*self.window_correction
            # Separate into response and reference
            reference_fft = frame_fft[self.collector_metadata.reference_channel_indices]
            response_fft = frame_fft[self.collector_metadata.response_channel_indices]
            for queue in self.data_out_queues:
                queue.put(copy.deepcopy(
                    (response_fft,
                     reference_fft)))
            self.log('Sent Data')
        self.last_frame = None
        self.environment_command_queue.put(self.process_name,(DataCollectorCommands.ACCEPTED,keep_frame))
        
    def stop(self,data):
        """Stops acquiring data from the data_in_queue and flushes queues.

        Parameters
        ----------
        data : Ignored
            Unused argument required due to the expectation that functions called
            by the RandomDataCollector.run function will have one argument
            accepting any data passed along with the instruction.
        """
        sleep(0.05)
        self.log('Stopping Data Collection')
        for queue in self.data_out_queues:
            flush_queue(queue)
        self.command_queue.flush(self.process_name)
        self.frame_buffer.reset_trigger()
        self.environment_command_queue.put(self.process_name,(DataCollectorCommands.SHUTDOWN_ACHIEVED,None))
        
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
        
def data_collector_process(environment_name : str,
                           command_queue : VerboseMessageQueue,
                           data_in_queue : mp.queues.Queue,
                           data_out_queues : List[mp.queues.Queue],
                           environment_command_queue : VerboseMessageQueue,
                           log_file_queue : mp.queues.Queue,
                           gui_update_queue : mp.queues.Queue,
                           process_name : str = None):
    """Random vibration data collector process function called by multiprocessing
    
    This function defines the Random Vibration Data Collector process that
    gets run by the multiprocessing module when it creates a new process.  It
    creates a ModalDataCollectorProcess object and runs it.

    Parameters
    ----------
    environment_name : str :
        Name of the environment associated with this signal generation process
    """

    data_collector_instance = DataCollectorProcess(
        environment_name + ' Data Collector' if process_name is None else process_name,
        command_queue,
        data_in_queue,
        data_out_queues,
        environment_command_queue,
        log_file_queue,
        gui_update_queue, environment_name)
    
    data_collector_instance.run()
