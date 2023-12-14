# -*- coding: utf-8 -*-
"""
This file defines a skeleton of an environment that utilizes system
identification.  This file should be modified to construct a full environment.

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

from .abstract_sysid_environment import AbstractSysIdMetadata,AbstractSysIdUI,AbstractSysIdEnvironment
from .environments import (ControlTypes,environment_definition_ui_paths,
                           environment_prediction_ui_paths,
                           environment_run_ui_paths)
from .utilities import (VerboseMessageQueue, DataAcquisitionParameters,
                        load_python_module, GlobalCommands, db2scale)

from enum import Enum
import multiprocessing as mp
from multiprocessing.queues import Queue
from qtpy import QtWidgets,uic
import netCDF4 as nc4

control_type = ControlTypes.Skeleton
maximum_name_length = 50

#%% Queues

class SkeletonQueues:
    """A container class for the queues that this environment will manage."""
    def __init__(self,
                 environment_name : str,
                 environment_command_queue : VerboseMessageQueue,
                 gui_update_queue : Queue,
                 controller_communication_queue : VerboseMessageQueue,
                 data_in_queue : Queue,
                 data_out_queue : Queue,
                 log_file_queue : VerboseMessageQueue
                 ):
        """A container class for the queues that random vibration will manage.
        
        The environment uses many queues to pass data between the various pieces.
        This class organizes those queues into one common namespace.
        

        Parameters
        ----------
        environment_name : str
            Name of the environment
        environment_command_queue : VerboseMessageQueue
            Queue that is read by the environment for environment commands
        gui_update_queue : mp.queues.Queue
            Queue where various subtasks put instructions for updating the
            widgets in the user interface
        controller_communication_queue : VerboseMessageQueue
            Queue that is read by the controller for global controller commands
        data_in_queue : mp.queues.Queue
            Multiprocessing queue that connects the acquisition subtask to the
            environment subtask.  Each environment will retrieve acquired data
            from this queue.
        data_out_queue : mp.queues.Queue
            Multiprocessing queue that connects the output subtask to the
            environment subtask.  Each environment will put data that it wants
            the controller to generate in this queue.
        log_file_queue : VerboseMessageQueue
            Queue for putting logging messages that will be read by the logging
            subtask and written to a file.
        """
        self.environment_command_queue = environment_command_queue
        self.gui_update_queue = gui_update_queue
        self.data_analysis_command_queue = VerboseMessageQueue(log_file_queue,environment_name + ' Data Analysis Command Queue')
        self.signal_generation_command_queue = VerboseMessageQueue(log_file_queue,environment_name + ' Signal Generation Command Queue')
        self.spectral_command_queue = VerboseMessageQueue(log_file_queue,environment_name + ' Spectral Computation Command Queue')
        self.collector_command_queue = VerboseMessageQueue(log_file_queue,environment_name + ' Data Collector Command Queue')
        self.controller_communication_queue = controller_communication_queue
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.data_for_spectral_computation_queue = mp.Queue()
        self.updated_spectral_quantities_queue = mp.Queue()
        self.log_file_queue = log_file_queue

#%% Metadata

class SkeletonMetadata(AbstractSysIdMetadata):
    def __init__(self):
        pass
    
    @property
    def number_of_channels(self):
        pass
    
    @property
    def response_channel_indices(self):
        pass
    
    @property
    def reference_channel_indices(self):
        pass
    
    @property
    def response_transformation_matrix(self):
        pass
    
    @property
    def reference_transformation_matrix(self):
        pass
    
    @property
    def sample_rate(self):
        pass
    
    def store_to_netcdf(self,netcdf_group_handle : nc4._netCDF4.Group):
        super().store_to_netcdf(netcdf_group_handle)

#%% UI

from .spectral_processing import (spectral_processing_process,
                                  SpectralProcessingCommands,
                                  SpectralProcessingMetadata,
                                  AveragingTypes,Estimator)
from .signal_generation_process import (signal_generation_process,
                                        SignalGenerationCommands,
                                        SignalGenerationMetadata)
from .data_collector import (data_collector_process,DataCollectorCommands,CollectorMetadata,
                             AcquisitionType,Acceptance,TriggerSlope,Window)

class SkeletonUI(AbstractSysIdUI):
    def __init__(self,
                 environment_name : str,
                 definition_tabwidget : QtWidgets.QTabWidget,
                 system_id_tabwidget : QtWidgets.QTabWidget,
                 test_predictions_tabwidget : QtWidgets.QTabWidget,
                 run_tabwidget : QtWidgets.QTabWidget,
                 environment_command_queue : VerboseMessageQueue,
                 controller_communication_queue : VerboseMessageQueue,
                 log_file_queue : Queue):
        super().__init__(environment_name,
             environment_command_queue,controller_communication_queue,log_file_queue,
             system_id_tabwidget)
        # Add the page to the control definition tabwidget
        self.definition_widget = QtWidgets.QWidget()
        uic.loadUi(environment_definition_ui_paths[control_type],self.definition_widget)
        definition_tabwidget.addTab(self.definition_widget,self.environment_name)
        # Add the page to the control prediction tabwidget
        self.prediction_widget = QtWidgets.QWidget()
        uic.loadUi(environment_prediction_ui_paths[control_type],self.prediction_widget)
        test_predictions_tabwidget.addTab(self.prediction_widget,self.environment_name)
        # Add the page to the run tabwidget
        self.run_widget = QtWidgets.QWidget()
        uic.loadUi(environment_run_ui_paths[control_type],self.run_widget)
        run_tabwidget.addTab(self.run_widget,self.environment_name)
    
    def collect_environment_definition_parameters(self):
        pass
    
    def create_environment_template(self, environment_name, workbook):
        pass
    
    def initialize_data_acquisition(self, data_acquisition_parameters):
        pass
    
    def initialize_environment(self):
        pass
    
    @property
    def initialized_control_names(self):
        pass
    
    @property
    def initialized_output_names(self):
        pass
    
    def retrieve_metadata(self, netcdf_handle):
        pass
    
    def set_parameters_from_template(self, worksheet):
        pass
    
    def start_control(self):
        pass
    
    def stop_control(self):
        pass
    
    def update_gui(self,queue_data):
        if super().update_gui(queue_data):
            return

#%% Environment

class SkeletonEnvironment(AbstractSysIdEnvironment):
    
    def __init__(self,
                 environment_name : str,
                 queue_container : SkeletonQueues):
        super().__init__(
                environment_name,
                queue_container.environment_command_queue,
                queue_container.gui_update_queue,
                queue_container.controller_communication_queue,
                queue_container.log_file_queue,
                queue_container.collector_command_queue,
                queue_container.signal_generation_command_queue,
                queue_container.spectral_command_queue,
                queue_container.data_analysis_command_queue,
                queue_container.data_in_queue,
                queue_container.data_out_queue)
    
    def start_control(self,data):
        pass
    
    def stop_environment(self,data):
        pass

#%% Process

def skeleton_process(environment_name : str,
                 input_queue : VerboseMessageQueue,
                 gui_update_queue : Queue,
                 controller_communication_queue : VerboseMessageQueue,
                 log_file_queue : Queue,
                 data_in_queue : Queue,
                 data_out_queue : Queue):
    # Create vibration queues
    queue_container = SkeletonQueues(environment_name,
                                      input_queue,
                                      gui_update_queue,
                                      controller_communication_queue,
                                      data_in_queue,
                                      data_out_queue,
                                      log_file_queue)
    
    spectral_proc = mp.Process(target=spectral_processing_process,
                                args=(environment_name,
                                      queue_container.spectral_command_queue,
                                      queue_container.data_for_spectral_computation_queue,
                                      queue_container.updated_spectral_quantities_queue,
                                      queue_container.environment_command_queue,
                                      queue_container.gui_update_queue,
                                      queue_container.log_file_queue))
    spectral_proc.start()
    analysis_proc = mp.Process(target=data_analysis_process,
                                args=(environment_name,
                                      queue_container.data_analysis_command_queue,
                                      queue_container.updated_spectral_quantities_queue,
                                      queue_container.data_analysis_output_queue,
                                      queue_container.environment_command_queue,
                                      queue_container.gui_update_queue,
                                      queue_container.log_file_queue))
    analysis_proc.start()
    siggen_proc = mp.Process(target=signal_generation_process,args=(environment_name,
                                                                    queue_container.signal_generation_command_queue,
                                                                    queue_container.data_analysis_output_queue,
                                                                    queue_container.data_out_queue,
                                                                    queue_container.environment_command_queue,
                                                                    queue_container.log_file_queue,
                                                                    queue_container.gui_update_queue))
    siggen_proc.start()
    collection_proc = mp.Process(target=data_collector_process,
                                  args=(environment_name,
                                        queue_container.collector_command_queue,
                                        queue_container.data_in_queue,
                                        [queue_container.data_for_spectral_computation_queue],
                                        queue_container.environment_command_queue,
                                        queue_container.log_file_queue,
                                        queue_container.gui_update_queue))
    
    # collection_proc.start()
    process_class = SkeletonEnvironment(
            environment_name,
            queue_container)
    process_class.run()
    
    # Rejoin all the processes
    process_class.log('Joining Subprocesses')
    process_class.log('Joining Spectral Computation')
    spectral_proc.join()
    process_class.log('Joining Data Analysis')
    analysis_proc.join()
    process_class.log('Joining Signal Generation')
    siggen_proc.join()
    process_class.log('Joining Data Collection')
    collection_proc.join()