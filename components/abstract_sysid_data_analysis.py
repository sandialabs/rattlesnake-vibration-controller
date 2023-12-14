"""
Defines data analysis performed for environments that use system identification

Abstract environment that can be used to create new environment control strategies
in the controller that use system identification.

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
from .utilities import (flush_queue,GlobalCommands,VerboseMessageQueue)
import multiprocessing as mp

class SysIDDataAnalysisCommands(Enum):
    INITIALIZE_PARAMETERS = 0
    RUN_NOISE = 1
    RUN_TRANSFER_FUNCTION = 2
    START_SHUTDOWN_AND_RUN_SYSID = 3
    START_SHUTDOWN = 4
    STOP_SYSTEM_ID = 5
    SHUTDOWN_ACHIEVED = 6
    SYSTEM_ID_COMPLETE = 7
    
from .abstract_sysid_environment import AbstractSysIdMetadata
    
class AbstractSysIDAnalysisProcess(AbstractMessageProcess):
    
    def __init__(self,process_name : str, 
                 command_queue : VerboseMessageQueue,
                 data_in_queue : mp.queues.Queue,
                 data_out_queue : mp.queues.Queue,
                 environment_command_queue : VerboseMessageQueue,
                 log_file_queue : mp.queues.Queue,
                 gui_update_queue : mp.queues.Queue,
                 environment_name : str):
        super().__init__(process_name,log_file_queue,command_queue,
                         gui_update_queue)
        self.map_command(SysIDDataAnalysisCommands.INITIALIZE_PARAMETERS,self.initialize_sysid_parameters)
        self.map_command(SysIDDataAnalysisCommands.RUN_NOISE,self.run_sysid_noise)
        self.map_command(SysIDDataAnalysisCommands.RUN_TRANSFER_FUNCTION,self.run_sysid_transfer_function)
        self.map_command(SysIDDataAnalysisCommands.STOP_SYSTEM_ID,self.stop_sysid)
        self.environment_name = environment_name
        self.environment_command_queue = environment_command_queue
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.parameters = None
        self.frames = None
        self.frequencies = None
        self.sysid_frf = None
        self.sysid_coherence = None
        self.sysid_response_cpsd = None
        self.sysid_reference_cpsd = None
        self.sysid_response_noise = None
        self.sysid_reference_noise = None
        self.sysid_condition = None
        self.startup = True
        
    def initialize_sysid_parameters(self,data : AbstractSysIdMetadata):
        self.parameters = data
    
    def run_sysid_noise(self,auto_shutdown):
        if self.startup:
            self.startup = False
            self.frames = 0
        spectral_data = flush_queue(self.data_in_queue)
        if len(spectral_data) > 0:
            self.log('Obtained Spectral Data')
            (self.frames,
             self.frequencies,
             frf,coherence,
             self.sysid_response_noise,
             self.sysid_reference_noise,
             condition) = spectral_data[-1]
            self.gui_update_queue.put((self.environment_name,
                                       ('noise_update',
                                        (self.frames,
                                        self.parameters.sysid_noise_averages,
                                        self.frequencies,
                                        self.sysid_response_noise,
                                        self.sysid_reference_noise))))
        if auto_shutdown and self.parameters.sysid_noise_averages == self.frames:
            self.environment_command_queue.put(
                self.process_name,
                (SysIDDataAnalysisCommands.START_SHUTDOWN_AND_RUN_SYSID,None))
            self.stop_sysid(None)
        else:
            self.command_queue.put(self.process_name,
                                   (SysIDDataAnalysisCommands.RUN_NOISE,auto_shutdown))
    
    def run_sysid_transfer_function(self,auto_shutdown):
        if self.startup:
            self.startup = False
            self.frames = 0
        spectral_data = flush_queue(self.data_in_queue)
        if len(spectral_data) > 0:
            self.log('Obtained Spectral Data')
            (self.frames,
             self.frequencies,
             self.sysid_frf,
             self.coherence,
             self.sysid_response_cpsd,
             self.sysid_reference_cpsd,
             self.sysid_condition) = spectral_data[-1]
            self.gui_update_queue.put((self.environment_name,
                                       ('sysid_update',
                                        (self.frames,
                                        self.parameters.sysid_averages,
                                        self.frequencies,
                                        self.sysid_frf,
                                        self.coherence,
                                        self.sysid_response_cpsd,
                                        self.sysid_reference_cpsd,
                                        self.sysid_condition))))
        if auto_shutdown and self.parameters.sysid_averages == self.frames:
            self.environment_command_queue.put(
                self.process_name,
                (SysIDDataAnalysisCommands.START_SHUTDOWN,False))
            self.stop_sysid(None)
            self.environment_command_queue.put(
                self.process_name,
                (SysIDDataAnalysisCommands.SYSTEM_ID_COMPLETE,
                 (self.frames,
                  self.parameters.sysid_averages,
                  self.frequencies,
                  self.sysid_frf,
                  self.coherence,
                  self.sysid_response_cpsd,
                  self.sysid_reference_cpsd,
                  self.sysid_condition,
                  self.sysid_response_noise,
                  self.sysid_reference_noise)))
        else:
            self.command_queue.put(self.process_name,
                                   (SysIDDataAnalysisCommands.RUN_TRANSFER_FUNCTION,auto_shutdown))
    
    def stop_sysid(self,data):
        # Remove any run_transfer_function or run_control from the queue
        instructions = self.command_queue.flush(self.process_name)
        for instruction in instructions:
            if not instruction[0] in [SysIDDataAnalysisCommands.RUN_NOISE,SysIDDataAnalysisCommands.RUN_TRANSFER_FUNCTION]:
                self.command_queue.put(self.process_name,instruction)
        flush_queue(self.data_out_queue)
        self.startup = True
        self.environment_command_queue.put(
            self.process_name,
            (SysIDDataAnalysisCommands.SHUTDOWN_ACHIEVED,None))

def sysid_data_analysis_process(environment_name : str,
                                 command_queue : VerboseMessageQueue,
                                 data_in_queue : mp.queues.Queue,
                                 data_out_queue : mp.queues.Queue,
                                 environment_command_queue : VerboseMessageQueue,
                                 gui_update_queue : mp.queues.Queue,
                                 log_file_queue : mp.queues.Queue,
                                 process_name = None
                                 ):
    data_analysis_instance = AbstractSysIDAnalysisProcess(
        environment_name + ' Data Analysis' if process_name is None else process_name,
        command_queue,data_in_queue,
        data_out_queue,environment_command_queue,log_file_queue,
        gui_update_queue,environment_name)
    
    data_analysis_instance.run()