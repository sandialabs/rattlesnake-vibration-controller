# -*- coding: utf-8 -*-
"""
This file defines a transient environment that utilizes system
identification.

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
                        load_python_module, GlobalCommands, db2scale, rms_time,
                        trac,align_signals,shift_signal)
from .ui_utilities import (load_time_history,PlotTimeWindow,
                           TransformationMatrixWindow,multiline_plotter,
                           colororder)
from enum import Enum
import multiprocessing as mp
import inspect
from multiprocessing.queues import Queue
from qtpy import QtWidgets,uic,QtCore
from qtpy.QtCore import Qt
import netCDF4 as nc4
import numpy as np
import traceback
import os
import importlib
import scipy.signal as sig

#%% Global Variables
control_type = ControlTypes.TRANSIENT
maximum_name_length = 50
buffer_size_samples_per_read_multiplier = 2

#%% Commands
class TransientCommands(Enum):
    START_CONTROL = 0
    STOP_CONTROL = 1
    PERFORM_CONTROL_PREDICTION = 3

#%% Queues

class TransientQueues:
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
        self.time_history_to_generate_queue = mp.Queue()
        self.log_file_queue = log_file_queue

#%% Metadata

class TransientMetadata(AbstractSysIdMetadata):
    def __init__(self,
                 number_of_channels,
                 sample_rate,
                 control_signal,
                 ramp_time,
                 control_python_script,
                 control_python_function,
                 control_python_function_type,
                 control_python_function_parameters,
                 control_channel_indices,
                 output_channel_indices,
                 response_transformation_matrix,
                 output_transformation_matrix):
        super().__init__()
        self.number_of_channels = number_of_channels
        self.sample_rate = sample_rate
        self.control_signal = control_signal
        self.test_level_ramp_time = ramp_time
        self.control_python_script = control_python_script
        self.control_python_function = control_python_function
        self.control_python_function_type = control_python_function_type
        self.control_python_function_parameters = control_python_function_parameters
        self.control_channel_indices = control_channel_indices
        self.output_channel_indices = output_channel_indices
        self.response_transformation_matrix = response_transformation_matrix
        self.reference_transformation_matrix = output_transformation_matrix
    
    @property
    def ramp_samples(self):
        return int(self.ramp_time*self.sample_rate)
    
    @property
    def number_of_channels(self):
        return self._number_of_channels
    
    @number_of_channels.setter
    def number_of_channels(self,value):
        self._number_of_channels = value
    
    @property
    def response_channel_indices(self):
        return self.control_channel_indices
    
    @property
    def reference_channel_indices(self):
        return self.output_channel_indices
    
    @property
    def response_transformation_matrix(self):
        return self._response_transformation_matrix
    
    @response_transformation_matrix.setter
    def response_transformation_matrix(self,value):
        self._response_transformation_matrix = value
    
    @property
    def reference_transformation_matrix(self):
        return self._reference_transformation_matrix
    
    @reference_transformation_matrix.setter
    def reference_transformation_matrix(self,value):
        self._reference_transformation_matrix = value
    
    @property
    def sample_rate(self):
        return self._sample_rate
    
    @sample_rate.setter
    def sample_rate(self,value):
        self._sample_rate = value
    
    @property
    def signal_samples(self):
        return self.control_signal.shape[-1]
    
    def store_to_netcdf(self,netcdf_group_handle : nc4._netCDF4.Group):
        super().store_to_netcdf(netcdf_group_handle)
        netcdf_group_handle.test_level_ramp_time = self.test_level_ramp_time
        netcdf_group_handle.control_python_script = self.control_python_script
        netcdf_group_handle.control_python_function = self.control_python_function
        netcdf_group_handle.control_python_function_type = self.control_python_function_type
        netcdf_group_handle.control_python_function_parameters = self.control_python_function_parameters
        # Save the output signal
        netcdf_group_handle.createDimension('control_channels',len(self.control_channel_indices))
        netcdf_group_handle.createDimension('specification_channels',len(self.control_channel_indices))
        netcdf_group_handle.createDimension('signal_samples',self.signal_samples)
        var = netcdf_group_handle.createVariable('control_signal','f8',('specification_channels','signal_samples'))
        var[...] = self.control_signal
        # Control Channels
        var = netcdf_group_handle.createVariable('control_channel_indices','i4',('control_channels'))
        var[...] = self.control_channel_indices
        # Transformation Matrix
        if not self.response_transformation_matrix is None:
            netcdf_group_handle.createDimension('response_transformation_rows',self.response_transformation_matrix.shape[0])
            netcdf_group_handle.createDimension('response_transformation_cols',self.response_transformation_matrix.shape[1])
            var = netcdf_group_handle.createVariable('response_transformation_matrix','f8',('response_transformation_rows','response_transformation_cols'))
            var[...] = self.response_transformation_matrix
        if not self.reference_transformation_matrix is None:
            netcdf_group_handle.createDimension('reference_transformation_rows',self.reference_transformation_matrix.shape[0])
            netcdf_group_handle.createDimension('reference_transformation_cols',self.reference_transformation_matrix.shape[1])
            var = netcdf_group_handle.createVariable('reference_transformation_matrix','f8',('reference_transformation_rows','reference_transformation_cols'))
            var[...] = self.reference_transformation_matrix
        
#%% UI

from .spectral_processing import (spectral_processing_process,
                                  SpectralProcessingCommands,
                                  SpectralProcessingMetadata,
                                  AveragingTypes,Estimator)
from .signal_generation_process import (signal_generation_process,
                                        SignalGenerationCommands,
                                        SignalGenerationMetadata)
from .signal_generation import TransientSignalGenerator
from .data_collector import FrameBuffer,data_collector_process
from .abstract_sysid_data_analysis import sysid_data_analysis_process

class TransientUI(AbstractSysIdUI):
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
        
        self.specification_signal = None
        self.show_signal_checkboxes = None
        self.plot_data_items = {}
        self.plot_windows = []
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.python_control_module = None
        self.physical_channel_names = None
        self.physical_output_indices = None
        self.excitation_prediction = None
        self.response_prediction = None
        self.last_control_data = None
        self.last_output_data = None
        
        self.control_selector_widgets = [
                self.prediction_widget.response_selector,
                self.run_widget.control_channel_selector]
        self.output_selector_widgets = [
                self.prediction_widget.excitation_selector,]
        
        # Set common look and feel for plots
        plotWidgets = [self.definition_widget.signal_display_plot,
                       self.prediction_widget.excitation_display_plot,
                       self.prediction_widget.response_display_plot,
                       self.run_widget.output_signal_plot,
                       self.run_widget.response_signal_plot]
        for plotWidget in plotWidgets:
            plot_item = plotWidget.getPlotItem()
            plot_item.showGrid(True,True,0.25)
            plot_item.enableAutoRange()
            plot_item.getViewBox().enableAutoRange(enable=True)
            
        self.connect_callbacks()
        
        # Complete the profile commands
        self.command_map['Set Test Level'] = self.change_test_level_from_profile
        self.command_map['Set Repeat'] = self.set_repeat_from_profile
        self.command_map['Set No Repeat'] = self.set_norepeat_from_profile
        
    def connect_callbacks(self):
        # Definition
        self.definition_widget.load_signal_button.clicked.connect(self.load_signal)
        self.definition_widget.transformation_matrices_button.clicked.connect(self.define_transformation_matrices)
        self.definition_widget.show_all_button.clicked.connect(self.show_all_signals)
        self.definition_widget.show_none_button.clicked.connect(self.show_no_signals)
        self.definition_widget.control_channels_selector.itemChanged.connect(self.update_control_channels)
        self.definition_widget.control_script_load_file_button.clicked.connect(self.select_python_module)
        self.definition_widget.control_function_input.currentIndexChanged.connect(self.update_generator_selector)
        self.definition_widget.check_selected_button.clicked.connect(self.check_selected_control_channels)
        self.definition_widget.uncheck_selected_button.clicked.connect(self.uncheck_selected_control_channels)
        # Prediction
        self.prediction_widget.excitation_selector.currentIndexChanged.connect(self.plot_predictions)
        self.prediction_widget.response_selector.currentIndexChanged.connect(self.plot_predictions)
        self.prediction_widget.response_error_list.itemClicked.connect(self.update_response_error_prediction_selector)
        self.prediction_widget.excitation_voltage_list.itemClicked.connect(self.update_excitation_prediction_selector)
        self.prediction_widget.maximum_voltage_button.clicked.connect(self.show_max_voltage_prediction)
        self.prediction_widget.minimum_voltage_button.clicked.connect(self.show_min_voltage_prediction)
        self.prediction_widget.maximum_error_button.clicked.connect(self.show_max_error_prediction)
        self.prediction_widget.minimum_error_button.clicked.connect(self.show_min_error_prediction)
        self.prediction_widget.recompute_predictions_button.clicked.connect(self.recompute_predictions)
        # Run Test 
        self.run_widget.start_test_button.clicked.connect(self.start_control)
        self.run_widget.stop_test_button.clicked.connect(self.stop_control)
        self.run_widget.create_window_button.clicked.connect(self.create_window)
        self.run_widget.show_all_channels_button.clicked.connect(self.show_all_channels)
        self.run_widget.tile_windows_button.clicked.connect(self.tile_windows)
        self.run_widget.close_windows_button.clicked.connect(self.close_windows)
        self.run_widget.control_response_error_list.itemDoubleClicked.connect(self.show_window)
        self.run_widget.save_current_control_data_button.clicked.connect(self.save_control_data)
    
    # %% Data Acquisition
    
    def initialize_data_acquisition(self, data_acquisition_parameters):
        super().initialize_data_acquisition(data_acquisition_parameters)
        # Initialize the plots
        for plot in [self.definition_widget.signal_display_plot,
                       self.prediction_widget.excitation_display_plot,
                       self.prediction_widget.response_display_plot,
                       self.run_widget.output_signal_plot,
                       self.run_widget.response_signal_plot]:
            plot.getPlotItem().clear()
        
        # Set up channel names
        self.physical_channel_names = ['{:} {:} {:}'.format('' if channel.channel_type is None else channel.channel_type,channel.node_number,channel.node_direction)[:maximum_name_length]
            for channel in data_acquisition_parameters.channel_list]
        self.physical_output_indices = [i for i,channel in enumerate(data_acquisition_parameters.channel_list) if channel.feedback_device]
        # Set up widgets
        self.definition_widget.sample_rate_display.setValue(data_acquisition_parameters.sample_rate)
        self.system_id_widget.samplesPerFrameSpinBox.setValue(data_acquisition_parameters.sample_rate)
        self.definition_widget.control_channels_selector.clear()
        for channel_name in self.physical_channel_names:
            item = QtWidgets.QListWidgetItem()
            item.setText(channel_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.definition_widget.control_channels_selector.addItem(item)
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.define_transformation_matrices(None,False)
        self.definition_widget.input_channels_display.setValue(len(self.physical_channel_names))
        self.definition_widget.output_channels_display.setValue(len(self.physical_output_indices))
        self.definition_widget.control_channels_display.setValue(0)
    
    @property
    def physical_output_names(self):
        return [self.physical_channel_names[i] for i in self.physical_output_indices]
    
    #%% Environment
    
    @property
    def physical_control_indices(self):
        return [i for i in range(
            self.definition_widget.control_channels_selector.count()) if 
            self.definition_widget.control_channels_selector.item(i).checkState() == Qt.Checked]
    
    @property
    def physical_control_names(self):
        return [self.physical_channel_names[i] for i in self.physical_control_indices]
    
    @property
    def initialized_control_names(self):
        if self.environment_parameters.response_transformation_matrix is None:
            return [self.physical_channel_names[i] for i in self.environment_parameters.control_channel_indices]
        else:
            return ['Transformed Response {:}'.format(i+1) for i in range(self.environment_parameters.response_transformation_matrix.shape[0])]

    @property
    def initialized_output_names(self):
        if self.environment_parameters.reference_transformation_matrix is None:
            return self.physical_output_names
        else:
            return ['Transformed Drive {:}'.format(i+1) for i in range(self.environment_parameters.reference_transformation_matrix.shape[0])]
    
    def update_control_channels(self):
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.specification_signal = None
        self.definition_widget.control_channels_display.setValue(len(self.physical_control_indices))
        self.define_transformation_matrices(None,False)
        self.show_signal()
    
    def collect_environment_definition_parameters(self):
        if self.python_control_module is None:
            control_module = None
            control_function = None
            control_function_type = None
            control_function_parameters = None
        else:
            control_module = self.definition_widget.control_script_file_path_input.text()
            control_function = self.definition_widget.control_function_input.itemText(self.definition_widget.control_function_input.currentIndex())
            control_function_type = self.definition_widget.control_function_generator_selector.currentIndex()
            control_function_parameters = self.definition_widget.control_parameters_text_input.toPlainText()
        return TransientMetadata(
            len(self.data_acquisition_parameters.channel_list),
            self.definition_widget.sample_rate_display.value(),
            self.specification_signal,
            self.definition_widget.ramp_selector.value(),
            control_module,
            control_function,
            control_function_type,
            control_function_parameters, 
            self.physical_control_indices,
            self.physical_output_indices,
            self.response_transformation_matrix,
            self.output_transformation_matrix)
    
    def load_signal(self,clicked,filename=None):
        """Loads a time signal using a dialog or the specified filename

        Parameters
        ----------
        clicked :
            The clicked event that triggered the callback.
        filename :
            File name defining the specification for bypassing the callback when
            loading from a file (Default value = None).

        """
        if filename is None:
            filename,file_filter = QtWidgets.QFileDialog.getOpenFileName(self.definition_widget,'Select Signal File',filter='Numpy or Mat (*.npy *.npz *.mat)')
            if filename == '':
                return
        self.definition_widget.signal_file_name_display.setText(filename)
        self.specification_signal = load_time_history(filename,self.definition_widget.sample_rate_display.value())
        self.setup_specification_table()
        self.show_signal()
    
    def setup_specification_table(self):
        """Sets up the specification table for the Transient Environment
        
        This function computes the RMS and max values for the signals and then
        creates entries in the table for each signal"""
        self.definition_widget.signal_samples_display.setValue(self.specification_signal.shape[-1])
        self.definition_widget.signal_time_display.setValue(self.specification_signal.shape[-1]/self.definition_widget.sample_rate_display.value())
        maxs = np.max(np.abs(self.specification_signal),axis=-1)
        rmss = rms_time(self.specification_signal,axis=-1)
        # Add rows to the signal table
        self.definition_widget.signal_information_table.setRowCount(self.specification_signal.shape[0])
        self.show_signal_checkboxes = []
        for i,(name,mx,rms) in enumerate(zip(self.physical_control_names,maxs,rmss)):
            item = QtWidgets.QTableWidgetItem()
            item.setText(name)
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.definition_widget.signal_information_table.setItem(i,1,item)
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.show_signal)
            self.show_signal_checkboxes.append(checkbox)
            self.definition_widget.signal_information_table.setCellWidget(i,0,checkbox)
            item = QtWidgets.QTableWidgetItem()
            item.setText('{:0.2f}'.format(mx))
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.definition_widget.signal_information_table.setItem(i,2,item)
            item = QtWidgets.QTableWidgetItem()
            item.setText('{:0.2f}'.format(rms))
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.definition_widget.signal_information_table.setItem(i,3,item)
    
    def show_signal(self):
        """Shows the signal on the user interface"""
        pi = self.definition_widget.signal_display_plot.getPlotItem()
        pi.clear()
        if self.specification_signal is None:
            self.definition_widget.signal_information_table.setRowCount(0)
            return
        abscissa = np.arange(self.specification_signal.shape[-1])/self.definition_widget.sample_rate_display.value()
        for i,(curve,checkbox) in enumerate(zip(self.specification_signal,self.show_signal_checkboxes)):
            pen = {'color':colororder[i%len(colororder)]}
            if checkbox.isChecked():
                pi.plot(abscissa,curve,pen=pen)
            else:
                pi.plot((0,0),(0,0),pen=pen)
    
    def show_all_signals(self):
        # print('Showing All Signals')
        for checkbox in self.show_signal_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
        self.show_signal()
        
    def show_no_signals(self):
        # print('Showing No Signals')
        for checkbox in self.show_signal_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        self.show_signal()
    
    def define_transformation_matrices(self,clicked,dialog = True):
        """Defines the transformation matrices using the dialog box"""
        if dialog:
            (response_transformation,output_transformation,result
             ) = TransformationMatrixWindow.define_transformation_matrices(
                    self.response_transformation_matrix,
                    self.definition_widget.control_channels_display.value(),
                    self.output_transformation_matrix,
                    self.definition_widget.output_channels_display.value(),
                    self.definition_widget)
        else:
            response_transformation = self.response_transformation_matrix
            output_transformation = self.output_transformation_matrix
            result = True
        if result:
            # Update the control names
            for widget in self.control_selector_widgets:
                widget.blockSignals(True)
                widget.clear()
            if response_transformation is None:
                for i,control_name in enumerate(self.physical_control_names):
                    for widget in self.control_selector_widgets:
                        widget.addItem('{:}: {:}'.format(i+1,control_name))
                self.definition_widget.transform_channels_display.setValue(len(self.physical_control_names))
            else:
                for i in range(response_transformation.shape[0]):
                    for widget in self.control_selector_widgets:
                        widget.addItem('{:}: {:}'.format(i+1,'Virtual Response'))
                self.definition_widget.transform_channels_display.setValue(response_transformation.shape[0])
            for widget in self.control_selector_widgets:
                widget.blockSignals(False)
            # Update the output names
            for widget in self.output_selector_widgets:
                widget.blockSignals(True)
                widget.clear()
            if output_transformation is None:
                for i,drive_name in enumerate(self.physical_output_names):
                    for widget in self.output_selector_widgets:
                        widget.addItem('{:}: {:}'.format(i+1,drive_name))
                self.definition_widget.transform_outputs_display.setValue(len(self.physical_output_names))
            else:
                for i in range(output_transformation.shape[0]):
                    for widget in self.output_selector_widgets:
                        widget.addItem('{:}: {:}'.format(i+1,'Virtual Drive'))
                self.definition_widget.transform_outputs_display.setValue(output_transformation.shape[0])
            for widget in self.output_selector_widgets:
                widget.blockSignals(False)
            # Clear the signals
            self.definition_widget.signal_information_table.clear()
            self.definition_widget.signal_display_plot.clear()
            self.definition_widget.signal_file_name_display.clear()
            self.definition_widget.signal_information_table.setRowCount(0)
            self.show_signal_checkboxes = None
            self.response_transformation_matrix = response_transformation
            self.output_transformation_matrix = output_transformation
    
    def select_python_module(self,clicked,filename=None):
        """Loads a Python module using a dialog or the specified filename

        Parameters
        ----------
        clicked :
            The clicked event that triggered the callback.
        filename :
            File name defining the Python module for bypassing the callback when
            loading from a file (Default value = None).

        """
        if filename is None:
            filename,file_filter = QtWidgets.QFileDialog.getOpenFileName(self.definition_widget,'Select Python Module',filter='Python Modules (*.py)')
            if filename == '':
                return
        self.python_control_module = load_python_module(filename)
        functions = [function for function in inspect.getmembers(self.python_control_module)
                     if (inspect.isfunction(function[1]) and len(inspect.signature(function[1]).parameters)>=6)
                     or inspect.isgeneratorfunction(function[1])
                     or (inspect.isclass(function[1]) and all([method in function[1].__dict__ for method in ['system_id_update','control']]))]
        self.log('Loaded module {:} with functions {:}'.format(self.python_control_module.__name__,[function[0] for function in functions]))
        self.definition_widget.control_function_input.clear()
        self.definition_widget.control_script_file_path_input.setText(filename)
        for function in functions:
            self.definition_widget.control_function_input.addItem(function[0])
    
    def update_generator_selector(self):
        """Updates the function/generator selector based on the function selected"""
        if self.python_control_module is None:
            return
        function = getattr(self.python_control_module,self.definition_widget.control_function_input.itemText(self.definition_widget.control_function_input.currentIndex()))
        if inspect.isgeneratorfunction(function):
            self.definition_widget.control_function_generator_selector.setCurrentIndex(1)
        elif inspect.isclass(function):
            self.definition_widget.control_function_generator_selector.setCurrentIndex(2)
        else:
            self.definition_widget.control_function_generator_selector.setCurrentIndex(0)
    
    def initialize_environment(self):
        super().initialize_environment()
        # Make sure everything is defined 
        if self.environment_parameters.control_signal is None:
            raise ValueError('Control Signal is not defined for {:}!'.format(self.environment_name))
        if self.environment_parameters.control_python_script is None:
            raise ValueError('Control function has not been loaded for {:}'.format(self.environment_name))
        self.system_id_widget.samplesPerFrameSpinBox.setMaximum(self.specification_signal.shape[-1])
        for widget in [self.prediction_widget.response_selector,
            self.run_widget.control_channel_selector]:
            widget.blockSignals(True)
            widget.clear()
            for i,control_name in enumerate(self.initialized_control_names):
                widget.addItem('{:}: {:}'.format(i+1,control_name))
            widget.blockSignals(False)
        for widget in [self.prediction_widget.excitation_selector]:
            widget.blockSignals(True)
            widget.clear()
            for i,drive_name in enumerate(self.initialized_output_names):
                widget.addItem('{:}: {:}'.format(i+1,drive_name))
            widget.blockSignals(False)
        # Set up the prediction plots
        self.prediction_widget.excitation_display_plot.getPlotItem().clear()
        self.prediction_widget.response_display_plot.getPlotItem().clear()
        self.plot_data_items['response_prediction'] = multiline_plotter(
                np.arange(self.environment_parameters.control_signal.shape[-1])/self.environment_parameters.sample_rate,
                np.zeros((2,self.environment_parameters.control_signal.shape[-1])),
                widget = self.prediction_widget.response_display_plot,
                other_pen_options={'width':1},
                names = ['Prediction','Spec']
                )
        self.plot_data_items['excitation_prediction'] = multiline_plotter(
                np.arange(self.environment_parameters.control_signal.shape[-1])/self.environment_parameters.sample_rate,
                np.zeros((1,self.environment_parameters.control_signal.shape[-1])),
                widget = self.prediction_widget.excitation_display_plot,
                other_pen_options={'width':1},
                names = ['Prediction']
                )
        # Set up the run plots
        self.run_widget.output_signal_plot.getPlotItem().clear()
        self.run_widget.response_signal_plot.getPlotItem().clear()
        buffer_multiplier = 1+(
            self.data_acquisition_parameters.samples_per_read*buffer_size_samples_per_read_multiplier/
            self.environment_parameters.control_signal.shape[-1])
        buffer_size = int(np.ceil(self.environment_parameters.control_signal.shape[-1]*buffer_multiplier))
        self.plot_data_items['output_signal_measurement'] = multiline_plotter(
                np.arange(buffer_size)/self.data_acquisition_parameters.sample_rate,
                np.zeros((len(self.initialized_control_names),buffer_size)),
                widget=self.run_widget.output_signal_plot,
                other_pen_options={'width':1},
                names = self.initialized_control_names)
        self.plot_data_items['signal_range'] = self.run_widget.response_signal_plot.getPlotItem().plot(np.zeros(5),np.zeros(5),pen = {'color': "k", 'width': 1},name='Signal Lower Bound')
        self.plot_data_items['control_signal_measurement'] = multiline_plotter(
                np.arange(buffer_size)/self.data_acquisition_parameters.sample_rate,
                np.zeros((len(self.initialized_output_names),buffer_size)),
                widget=self.run_widget.response_signal_plot,
                other_pen_options={'width':1},
                names = self.initialized_output_names)
        
        return self.environment_parameters
    
    def check_selected_control_channels(self):
        for item in self.definition_widget.control_channels_selector.selectedItems():
            item.setCheckState(Qt.Checked)
        
    def uncheck_selected_control_channels(self):
        for item in self.definition_widget.control_channels_selector.selectedItems():
            item.setCheckState(Qt.Unchecked)
    
    #%% Predictions
    def plot_predictions(self):
        times = np.arange(self.specification_signal.shape[-1])/self.data_acquisition_parameters.sample_rate
        index = self.prediction_widget.excitation_selector.currentIndex()
        self.plot_data_items['excitation_prediction'][0].setData(times,
                            self.excitation_prediction[index])
        index = self.prediction_widget.response_selector.currentIndex()
        self.plot_data_items['response_prediction'][0].setData(times,
                            self.response_prediction[index])
        self.plot_data_items['response_prediction'][1].setData(times,
                            self.specification_signal[index])
    
    def show_max_voltage_prediction(self):
        widget = self.prediction_widget.excitation_voltage_list
        index = np.argmax([float(widget.item(v).text())
                           for v in range(widget.count())])
        self.prediction_widget.excitation_selector.setCurrentIndex(index)
        
    def show_min_voltage_prediction(self):
        widget = self.prediction_widget.excitation_voltage_list
        index = np.argmin([float(widget.item(v).text())
                           for v in range(widget.count())])
        self.prediction_widget.excitation_selector.setCurrentIndex(index)
        
    def show_max_error_prediction(self):
        widget = self.prediction_widget.response_error_list
        index = np.argmax([float(widget.item(v).text())
                           for v in range(widget.count())])
        self.prediction_widget.response_selector.setCurrentIndex(index)
        
    def show_min_error_prediction(self):
        widget = self.prediction_widget.response_error_list
        index = np.argmin([float(widget.item(v).text())
                           for v in range(widget.count())])
        self.prediction_widget.response_selector.setCurrentIndex(index)

    def update_response_error_prediction_selector(self,item):
        index = self.prediction_widget.response_error_list.row(item)
        self.prediction_widget.response_selector.setCurrentIndex(index)

    def update_excitation_prediction_selector(self,item):
        index = self.prediction_widget.excitation_voltage_list.row(item)
        self.prediction_widget.excitation_selector.setCurrentIndex(index)
    
    def recompute_predictions(self):
        self.environment_command_queue.put(
            self.log_name,
            (TransientCommands.PERFORM_CONTROL_PREDICTION,False))
    
    #%% Control
    
    def start_control(self):
        self.enable_control(False)
        self.controller_communication_queue.put(
            self.log_name,
            (GlobalCommands.START_ENVIRONMENT,self.environment_name))
        self.environment_command_queue.put(
            self.log_name,
            (TransientCommands.START_CONTROL,
             (db2scale(self.run_widget.test_level_selector.value()),
              self.run_widget.repeat_signal_checkbox.isChecked())))
        if self.run_widget.test_level_selector.value() >= 0:
            self.controller_communication_queue.put(self.log_name,(GlobalCommands.AT_TARGET_LEVEL,self.environment_name))
    
    def stop_control(self):
        self.environment_command_queue.put(self.log_name,(TransientCommands.STOP_CONTROL,
                                                          None))
    
    def enable_control(self,enabled):
        for widget in [self.run_widget.test_level_selector,
                       self.run_widget.repeat_signal_checkbox,
                       self.run_widget.start_test_button]:
            widget.setEnabled(enabled)
        for widget in [self.run_widget.stop_test_button]:
            widget.setEnabled(not enabled)
    
    def change_test_level_from_profile(self,test_level):
        self.run_widget.test_level_selector.setValue(int(test_level))
    
    def set_repeat_from_profile(self,data):
        self.run_widget.repeat_signal_checkbox.setChecked(True)
    
    def set_norepeat_from_profile(self,data):
        self.run_widget.repeat_signal_checkbox.setChecked(False)

    def create_window(self,event,control_index = None):
        """Creates a subwindow to show a specific channel information

        Parameters
        ----------
        event :
            
        control_index :
            Row index in the specification matrix to display (Default value = None)

        """
        if control_index is None:
            control_index = self.run_widget.control_channel_selector.currentIndex()
        self.plot_windows.append(
                PlotTimeWindow(None,control_index,
                          self.environment_parameters.control_signal,
                          self.data_acquisition_parameters.sample_rate,
                          self.run_widget.control_channel_selector.itemText(control_index),
                          ))
        if self.last_control_data is not None:
            self.plot_windows[-1].update_plot(self.last_control_data)
    
    def show_all_channels(self):
        """Creates a subwindow for each ASD in the CPSD matrix"""
        for i in range(self.environment_parameters.control_signal.shape[0]):
            self.create_window(None,i)
        self.tile_windows()
    
    def tile_windows(self):
        """Tile subwindow equally across the screen"""
        screen_rect = QtWidgets.QApplication.desktop().screenGeometry()
        # Go through and remove any closed windows
        self.plot_windows = [window for window in self.plot_windows if window.isVisible()]
        num_windows = len(self.plot_windows)
        ncols = int(np.ceil(np.sqrt(num_windows)))
        nrows = int(np.ceil(num_windows/ncols))
        window_width = int(screen_rect.width()/ncols)
        window_height = int(screen_rect.height()/nrows)
        for index,window in enumerate(self.plot_windows):
            window.resize(window_width,window_height)
            row_ind = index // ncols
            col_ind = index % ncols
            window.move(col_ind*window_width,row_ind*window_height)
    
    def show_window(self,item):
        index = self.run_widget.control_response_error_list.row(item)
        self.create_window(None,index)
    
    def close_windows(self):
        """Close all subwindows"""
        for window in self.plot_windows:
            window.close()

    def update_control_plots(self):
        # Go through and remove any closed windows
        self.plot_windows = [window for window in self.plot_windows if window.isVisible()]
        for window in self.plot_windows:
            window.update_plot(self.last_control_data)

    def save_control_data(self):
        """Save Time Data from the Controller"""
        filename,file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self.definition_widget,'Select File to Save Spectral Data',
            filter='NetCDF File (*.nc4)')
        if filename == '':
            return
        labels = [['node_number',str],
                  ['node_direction',str],
                  ['comment',str],
                  ['serial_number',str],
                  ['triax_dof',str],
                  ['sensitivity',str],
                  ['unit',str],
                  ['make',str],
                  ['model',str],
                  ['expiration',str],
                  ['physical_device',str],
                  ['physical_channel',str],
                  ['channel_type',str],
                  ['minimum_value',str],
                  ['maximum_value',str],
                  ['coupling',str],
                  ['excitation_source',str],
                  ['excitation',str],
                  ['feedback_device',str],
                  ['feedback_channel',str],
                  ['warning_level',str],
                  ['abort_level',str],
                  ]
        global_data_parameters : DataAcquisitionParameters
        global_data_parameters = self.data_acquisition_parameters
        netcdf_handle = nc4.Dataset(filename,'w',format='NETCDF4',clobber=True)
        # Create dimensions
        netcdf_handle.createDimension('response_channels',len(global_data_parameters.channel_list))
        netcdf_handle.createDimension('output_channels',len([channel for channel in global_data_parameters.channel_list if not channel.feedback_device is None]))
        netcdf_handle.createDimension('time_samples',None)
        netcdf_handle.createDimension('num_environments',len(global_data_parameters.environment_names))
        # Create attributes
        netcdf_handle.file_version = '3.0.0'
        netcdf_handle.sample_rate = global_data_parameters.sample_rate
        netcdf_handle.time_per_write = global_data_parameters.samples_per_write/global_data_parameters.output_sample_rate
        netcdf_handle.time_per_read = global_data_parameters.samples_per_read/global_data_parameters.sample_rate
        netcdf_handle.hardware = global_data_parameters.hardware
        netcdf_handle.hardware_file = 'None' if global_data_parameters.hardware_file is None else global_data_parameters.hardware_file
        netcdf_handle.maximum_acquisition_processes = global_data_parameters.maximum_acquisition_processes
        netcdf_handle.output_oversample = global_data_parameters.output_oversample
        # Create Variables
        var = netcdf_handle.createVariable('environment_names',str,('num_environments',))
        this_environment_index = None
        for i,name in enumerate(global_data_parameters.environment_names):
            var[i] = name
            if name == self.environment_name:
                this_environment_index = i
        var = netcdf_handle.createVariable('environment_active_channels','i1',('response_channels','num_environments'))
        var[...] = global_data_parameters.environment_active_channels.astype('int8')[
            global_data_parameters.environment_active_channels[:,this_environment_index],:]
        # Create channel table variables
        
        for (label,netcdf_datatype) in labels:
            var = netcdf_handle.createVariable('/channels/'+label,netcdf_datatype,('response_channels',))
            channel_data = [getattr(channel,label) for channel in global_data_parameters.channel_list]
            if netcdf_datatype == 'i1':
                channel_data = np.array([1 if val else 0 for val in channel_data])
            else:
                channel_data = ['' if val is None else val for val in channel_data]
            for i,cd in enumerate(channel_data):
                var[i] = cd
        # Save the environment to the file
        group_handle = netcdf_handle.createGroup(self.environment_name)
        self.environment_parameters.store_to_netcdf(group_handle)
        # Create Variables for Spectral Data
        group_handle.createDimension('drive_channels',self.last_transfer_function.shape[2])
        group_handle.createDimension('fft_lines',self.environment_parameters.sysid_frame_size//2 + 1)
        var = group_handle.createVariable('frf_data_real','f8',('fft_lines','specification_channels','drive_channels'))
        var[...] = self.last_transfer_function.real
        var = group_handle.createVariable('frf_data_imag','f8',('fft_lines','specification_channels','drive_channels'))
        var[...] = self.last_transfer_function.imag
        var = group_handle.createVariable('frf_coherence','f8',('fft_lines','specification_channels'))
        var[...] = self.last_coherence.real
        var = group_handle.createVariable('response_cpsd_real','f8',('fft_lines','specification_channels','specification_channels'))
        var[...] = self.last_response_cpsd.real
        var = group_handle.createVariable('response_cpsd_imag','f8',('fft_lines','specification_channels','specification_channels'))
        var[...] = self.last_response_cpsd.imag
        var = group_handle.createVariable('drive_cpsd_real','f8',('fft_lines','drive_channels','drive_channels'))
        var[...] = self.last_reference_cpsd.real
        var = group_handle.createVariable('drive_cpsd_imag','f8',('fft_lines','drive_channels','drive_channels'))
        var[...] = self.last_reference_cpsd.imag
        var = group_handle.createVariable('response_noise_cpsd_real','f8',('fft_lines','specification_channels','specification_channels'))
        var[...] = self.last_response_noise.real
        var = group_handle.createVariable('response_noise_cpsd_imag','f8',('fft_lines','specification_channels','specification_channels'))
        var[...] = self.last_response_noise.imag
        var = group_handle.createVariable('drive_noise_cpsd_real','f8',('fft_lines','drive_channels','drive_channels'))
        var[...] = self.last_reference_noise.real
        var = group_handle.createVariable('drive_noise_cpsd_imag','f8',('fft_lines','drive_channels','drive_channels'))
        var[...] = self.last_reference_noise.imag
        var = group_handle.createVariable('control_response','f8',('specification_channels','signal_samples'))
        var[...] = self.last_control_data
        var = group_handle.createVariable('control_drives','f8',('drive_channels','signal_samples'))
        var[...] = self.last_output_data
        netcdf_handle.close()

    #%% Misc
    
    def retrieve_metadata(self, netcdf_handle):
        super().retrieve_metadata(netcdf_handle)
        # Get the group
        group = netcdf_handle.groups[self.environment_name]
        # Spinboxes
        self.definition_widget.ramp_selector.setValue(group.test_level_ramp_time)
        # Control channels
        for i in group.variables['control_channel_indices'][...]:
            item = self.definition_widget.control_channels_selector.item(i)
            item.setCheckState(Qt.Checked)
        # Other Data
        try:
            self.response_transformation_matrix = group.variables['response_transformation_matrix'][...].data
        except KeyError:
            self.response_transformation_matrix = None
        try:
            self.output_transformation_matrix = group.variables['output_transformation_matrix'][...].data
        except KeyError:
            self.output_transformation_matrix = None
        self.define_transformation_matrices(None,dialog=False)
        self.specification_signal = group.variables['control_signal'][...].data
        self.select_python_module(None,group.control_python_script)
        self.definition_widget.control_function_input.setCurrentIndex(self.definition_widget.control_function_input.findText(group.control_python_function))
        self.definition_widget.control_parameters_text_input.setText(group.control_python_function_parameters)
        self.setup_specification_table()
        self.show_signal()
    
    def update_gui(self,queue_data):
        if super().update_gui(queue_data):
            return
        message,data = queue_data
        if message == 'time_data':
            response_data,output_data,signal_delay = data
            max_y = -1e15
            min_y = 1e15
            for curve,this_data in zip(self.plot_data_items['control_signal_measurement'],response_data):
                x,y = curve.getData()
                if np.max(y) > max_y:
                    max_y = np.max(y)
                if np.min(y) < min_y:
                    min_y = np.min(y)
                y = np.concatenate((y[this_data.size:],this_data[-x.size:]),axis=0)
                curve.setData(x,y)
            # Display the data
            for curve,this_output in zip(self.plot_data_items['output_signal_measurement'],output_data):
                x,y = curve.getData()
                y = np.concatenate((y[this_output.size:],this_output[-x.size:]),axis=0)
                curve.setData(x,y)
            if signal_delay is None:
                self.plot_data_items['signal_range'].setData(
                    np.zeros(5),np.zeros(5)
                    )
            else:
                self.plot_data_items['signal_range'].setData(
                    np.array((x[signal_delay],x[signal_delay],
                              x[signal_delay+self.environment_parameters.control_signal.shape[-1]-1],
                              x[signal_delay+self.environment_parameters.control_signal.shape[-1]-1],
                              x[signal_delay])),1.05*np.array((min_y,max_y,max_y,min_y,min_y))
                    )
        elif message == 'control_data':
            self.last_control_data,self.last_output_data = data
            self.update_control_plots()
        elif message == 'control_predictions':
            (times,
            self.excitation_prediction,
            self.response_prediction,
            prediction) = data
            self.plot_predictions()
        elif message == 'enable_control':
            self.enable_control(True)
        elif message == 'enable':
            widget = None
            for parent in [self.definition_widget,self.run_widget,self.system_id_widget,self.prediction_widget]:
                try:
                    widget = getattr(parent,data)
                    break
                except AttributeError:
                    continue
            if widget is None:
                raise ValueError('Cannot Enable Widget {:}: not found in UI'.format(data))
            widget.setEnabled(True)
        elif message == 'disable':
            widget = None
            for parent in [self.definition_widget,self.run_widget,self.system_id_widget,self.prediction_widget]:
                try:
                    widget = getattr(parent,data)
                    break
                except AttributeError:
                    continue
            if widget is None:
                raise ValueError('Cannot Disable Widget {:}: not found in UI'.format(data))
            widget.setEnabled(False)
        else:
            widget = None
            for parent in [self.definition_widget,self.run_widget,self.system_id_widget,self.prediction_widget]:
                try:
                    widget = getattr(parent,message)
                    break
                except AttributeError:
                    continue
            if widget is None:
                raise ValueError('Cannot Update Widget {:}: not found in UI'.format(message))
            if type(widget) is QtWidgets.QDoubleSpinBox:
                widget.setValue(data)
            elif type(widget) is QtWidgets.QSpinBox:
                widget.setValue(data)
            elif type(widget) is QtWidgets.QLineEdit:
                widget.setText(data)
            elif type(widget) is QtWidgets.QListWidget:
                widget.clear()
                widget.addItems(['{:.3f}'.format(d) for d in data])
    
    def set_parameters_from_template(self, worksheet):
        self.definition_widget.ramp_selector.setValue(float(worksheet.cell(3,2).value))
        self.select_python_module(None,worksheet.cell(4,2).value)
        self.definition_widget.control_function_input.setCurrentIndex(self.definition_widget.control_function_input.findText(worksheet.cell(5,2).value))
        self.definition_widget.control_parameters_text_input.setText('' if worksheet.cell(6,2).value is None else str(worksheet.cell(6,2).value))
        column_index = 2
        while True:
            value = worksheet.cell(7,column_index).value
            if value is None or (isinstance(value,str) and value.strip() == ''):
                break
            item = self.definition_widget.control_channels_selector.item(int(value)-1)
            item.setCheckState(Qt.Checked)
            column_index += 1
        self.system_id_widget.samplesPerFrameSpinBox.setValue(int(worksheet.cell(8,2).value))
        self.system_id_widget.averagingTypeComboBox.setCurrentIndex(self.system_id_widget.averagingTypeComboBox.findText(worksheet.cell(9,2).value))
        self.system_id_widget.noiseAveragesSpinBox.setValue(int(worksheet.cell(10,2).value))
        self.system_id_widget.systemIDAveragesSpinBox.setValue(int(worksheet.cell(11,2).value))
        self.system_id_widget.averagingCoefficientDoubleSpinBox.setValue(float(worksheet.cell(12,2).value))
        self.system_id_widget.estimatorComboBox.setCurrentIndex(self.system_id_widget.estimatorComboBox.findText(worksheet.cell(13,2).value))
        self.system_id_widget.levelDoubleSpinBox.setValue(float(worksheet.cell(14,2).value))
        self.system_id_widget.levelRampTimeDoubleSpinBox.setValue(float(worksheet.cell(15,2).value))
        self.system_id_widget.signalTypeComboBox.setCurrentIndex(self.system_id_widget.signalTypeComboBox.findText(worksheet.cell(16,2).value))
        self.system_id_widget.windowComboBox.setCurrentIndex(self.system_id_widget.windowComboBox.findText(worksheet.cell(17,2).value))
        self.system_id_widget.overlapDoubleSpinBox.setValue(float(worksheet.cell(18,2).value))
        self.system_id_widget.onFractionDoubleSpinBox.setValue(float(worksheet.cell(19,2).value))
        self.system_id_widget.pretriggerDoubleSpinBox.setValue(float(worksheet.cell(20,2).value))
        self.system_id_widget.rampFractionDoubleSpinBox.setValue(float(worksheet.cell(21,2).value))
        
        # Now we need to find the transformation matrices' sizes
        response_channels = self.definition_widget.control_channels_display.value()
        output_channels = self.definition_widget.output_channels_display.value()
        output_transform_row = 23
        if isinstance(worksheet.cell(22,2).value,str) and worksheet.cell(22,2).value.lower() == 'none':
            self.response_transformation_matrix = None
        else:
            while True:
                if worksheet.cell(output_transform_row,1).value == 'Output Transformation Matrix:':
                    break
                output_transform_row += 1
            response_size = output_transform_row-22
            response_transformation = []
            for i in range(response_size):
                response_transformation.append([])
                for j in range(response_channels):
                    response_transformation[-1].append(float(worksheet.cell(22+i,2+j).value))
            self.response_transformation_matrix = np.array(response_transformation)
        if isinstance(worksheet.cell(output_transform_row,2).value,str) and worksheet.cell(output_transform_row,2).value.lower() == 'none':
            self.output_transformation_matrix = None
        else:
            output_transformation = []
            i = 0
            while True:
                if worksheet.cell(output_transform_row+i,2).value is None or (isinstance(worksheet.cell(output_transform_row+i,2).value,str) and worksheet.cell(output_transform_row+i,2).value.strip() == ''):
                    break
                output_transformation.append([])
                for j in range(output_channels):
                    output_transformation[-1].append(float(worksheet.cell(output_transform_row+i,2+j).value))
                i += 1
            self.output_transformation_matrix = np.array(output_transformation)
        self.define_transformation_matrices(None,dialog=False)
        self.load_signal(None,worksheet.cell(2,2).value)
    
    @staticmethod
    def create_environment_template(environment_name, workbook):
        worksheet = workbook.create_sheet(environment_name)
        worksheet.cell(1,1,'Control Type')
        worksheet.cell(1,2,'Transient')
        worksheet.cell(1,4,'Note: Replace cells with hash marks (#) to provide the requested parameters.')
        worksheet.cell(2,1,'Signal File')
        worksheet.cell(2,2,'# Path to the file that contains the time signal that will be output')
        worksheet.cell(3,1,'Ramp Time')
        worksheet.cell(3,2,'# Time for the environment to ramp between levels or from start or to stop.')
        worksheet.cell(4,1,'Control Python Script:')
        worksheet.cell(4,2,'# Path to the Python script containing the control law')
        worksheet.cell(5,1,'Control Python Function:')
        worksheet.cell(5,2,'# Function name within the Python Script that will serve as the control law')
        worksheet.cell(6,1,'Control Parameters:')
        worksheet.cell(6,2,'# Extra parameters used in the control law')
        worksheet.cell(7,1,'Control Channels (1-based):')
        worksheet.cell(7,2,'# List of channels, one per cell on this row')
        worksheet.cell(8,1,'System ID Samples per Frame')
        worksheet.cell(8,2,'# Number of Samples per Measurement Frame in the System Identification')
        worksheet.cell(9,1,'System ID Averaging:')
        worksheet.cell(9,2,'# Averaging Type, should be Linear or Exponential')
        worksheet.cell(10,1,'Noise Averages:')
        worksheet.cell(10,2,'# Number of Averages used when characterizing noise')
        worksheet.cell(11,1,'System ID Averages:')
        worksheet.cell(11,2,'# Number of Averages used when computing the FRF')
        worksheet.cell(12,1,'Exponential Averaging Coefficient:')
        worksheet.cell(12,2,'# Averaging Coefficient for Exponential Averaging (if used)')
        worksheet.cell(13,1,'System ID Estimator:')
        worksheet.cell(13,2,'# Technique used to compute system ID.  Should be one of H1, H2, H3, or Hv.')
        worksheet.cell(14,1,'System ID Level (V RMS):')
        worksheet.cell(14,2,'# RMS Value of Flat Voltage Spectrum used for System Identification.')
        worksheet.cell(15,1,'System ID Ramp Time')
        worksheet.cell(15,2,'# Time for the system identification to ramp between levels or from start or to stop.')
        worksheet.cell(16,1,'System ID Signal Type:')
        worksheet.cell(16,2,'# Signal to use for the system identification')
        worksheet.cell(17,1,'System ID Window:')
        worksheet.cell(17,2,'# Window used to compute FRFs during system ID.  Should be one of Hann or None')
        worksheet.cell(18,1,'System ID Overlap %:')
        worksheet.cell(18,2,'# Overlap to use in the system identification')
        worksheet.cell(19,1,'System ID Burst On %:')
        worksheet.cell(19,2,'# Percentage of a frame that the burst random is on for')
        worksheet.cell(20,1,'System ID Burst Pretrigger %:')
        worksheet.cell(20,2,'# Percentage of a frame that occurs before the burst starts in a burst random signal')
        worksheet.cell(21,1,'System ID Ramp Fraction %:')
        worksheet.cell(21,2,'# Percentage of the "System ID Burst On %" that will be used to ramp up to full level')
        worksheet.cell(22,1,'Response Transformation Matrix:')
        worksheet.cell(22,2,'# Transformation matrix to apply to the response channels.  Type None if there is none.  Otherwise, make this a 2D array in the spreadsheet and move the Output Transformation Matrix line down so it will fit.  The number of columns should be the number of physical control channels.')
        worksheet.cell(23,1,'Output Transformation Matrix:')
        worksheet.cell(23,2,'# Transformation matrix to apply to the outputs.  Type None if there is none.  Otherwise, make this a 2D array in the spreadsheet.  The number of columns should be the number of physical output channels in the environment.')

#%% Environment

class TransientEnvironment(AbstractSysIdEnvironment):
    
    def __init__(self,
                 environment_name : str,
                 queue_container : TransientQueues,
                 acquisition_active : mp.Value,
                 output_active : mp.Value):
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
                queue_container.data_out_queue,
                acquisition_active,
                output_active)
        self.map_command(TransientCommands.PERFORM_CONTROL_PREDICTION,self.perform_control_prediction)
        self.map_command(TransientCommands.START_CONTROL,self.start_control)
        self.map_command(TransientCommands.STOP_CONTROL,self.stop_environment)
        # Persistent data
        self.data_acquisition_parameters = None
        self.environment_parameters = None
        self.queue_container = queue_container
        self.frames = None
        self.frequencies = None
        self.frf = None
        self.coherence = None
        self.sysid_response_cpsd = None
        self.sysid_reference_cpsd = None
        self.sysid_condition = None
        self.sysid_response_noise = None
        self.sysid_reference_noise = None
        self.control_function_type = None
        self.extra_control_parameters = None
        self.control_function = None
        self.aligned_output = None
        self.aligned_response = None
        self.next_drive = None
        self.predicted_response = None
        self.startup = True
        self.shutdown_flag = False
        self.repeat = False
        self.test_level = 0
        self.control_buffer = None
        self.output_buffer = None
        self.last_signal_found = None
    
    def initialize_environment_test_parameters(self, environment_parameters : TransientMetadata):
        if self.environment_parameters is None or self.environment_parameters.control_signal.shape != environment_parameters.control_signal.shape:
            self.frames = None
            self.frequencies = None
            self.frf = None
            self.coherence = None
            self.sysid_response_cpsd = None
            self.sysid_reference_cpsd = None
            self.sysid_condition = None
            self.sysid_response_noise = None
            self.sysid_reference_noise = None
            self.control_function_type = None
            self.extra_control_parameters = None
            self.control_function = None
            self.aligned_output = None
            self.aligned_response = None
            self.next_drive = None
            self.predicted_response = None
        super().initialize_environment_test_parameters(environment_parameters)
        self.environment_parameters : TransientMetadata
        # Load in the control law
        path,file = os.path.split(environment_parameters.control_python_script)
        file,ext = os.path.splitext(file)
        spec = importlib.util.spec_from_file_location(file, environment_parameters.control_python_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.control_function_type = environment_parameters.control_python_function_type
        self.extra_control_parameters = environment_parameters.control_python_function_parameters
        if self.control_function_type == 1: # Generator
            # Get the generator function
            generator_function = getattr(module,environment_parameters.control_python_function)()
            # Get us to the first yield statement
            next(generator_function)
            # Define the control function as the generator's send function
            self.control_function = generator_function.send
        elif self.control_function_type == 2: # Class
            self.control_function = getattr(module,environment_parameters.control_python_function)(
                self.data_acquisition_parameters.sample_rate,
                self.environment_parameters.control_signal,
                self.data_acquisition_parameters.output_oversample,
                self.extra_control_parameters, # Required parameters
                self.environment_parameters.sysid_frequency_spacing, # Frequency Spacing
                self.frf, # Transfer Functions
                self.sysid_response_noise,  # Noise levels and correlation 
                self.sysid_reference_noise, # from the system identification
                self.sysid_response_cpsd,  # Response levels and correlation
                self.sysid_reference_cpsd, # from the system identification
                self.coherence, # Coherence from the system identification
                self.frames, # Number of frames in the CPSD and FRF matrices
                self.environment_parameters.sysid_averages, # Total frames that could be in the CPSD and FRF matrices
                self.aligned_output, # Last excitation signal for drive-based control
                self.aligned_response) # Last response signal for error-based correction
        else: # Function
            self.control_function = getattr(module,environment_parameters.control_python_function)

    def system_id_complete(self,data):
        self.log('Finished System Identification')
        self.controller_communication_queue.put(
            self.environment_name,
            (GlobalCommands.COMPLETED_SYSTEM_ID,self.environment_name))
        (self.frames,
        avg,
        self.frequencies,
        self.frf,
        self.coherence,
        self.sysid_response_cpsd,
        self.sysid_reference_cpsd,
        self.sysid_condition,
        self.sysid_response_noise,
        self.sysid_reference_noise) = data
        # Perform the control prediction
        self.perform_control_prediction(True)
        
    def perform_control_prediction(self,sysid_update):
        if self.frf is None:
            self.gui_update_queue.put(('error',('Perform System Identification','Perform System ID before performing test predictions')))
            return
        if self.control_function_type == 1: # Generator
            output_time_history = self.control_function((
                self.data_acquisition_parameters.sample_rate,
                self.environment_parameters.control_signal,
                self.environment_parameters.sysid_frequency_spacing,
                self.frf, # Transfer Functions
                self.sysid_response_noise,  # Noise levels and correlation 
                self.sysid_reference_noise, # from the system identification
                self.sysid_response_cpsd,  # Response levels and correlation
                self.sysid_reference_cpsd, # from the system identification
                self.coherence, # Coherence from the system identification
                self.frames, # Number of frames in the CPSD and FRF matrices
                self.environment_parameters.sysid_averages, # Total frames that could be in the CPSD and FRF matrices
                self.data_acquisition_parameters.output_oversample,
                self.extra_control_parameters, # Required parameters
                self.next_drive, # Last excitation signal for drive-based control
                self.predicted_response, # Last response signal for error correction
                 ))
        elif self.control_function_type == 2: # Class
            if sysid_update:
                self.control_function.system_id_update(
                    self.environment_parameters.sysid_frequency_spacing,
                    self.frf, # Transfer Functions
                    self.sysid_response_noise,  # Noise levels and correlation 
                    self.sysid_reference_noise, # from the system identification
                    self.sysid_response_cpsd,  # Response levels and correlation
                    self.sysid_reference_cpsd, # from the system identification
                    self.coherence, # Coherence from the system identification
                    self.frames, # Number of frames in the CPSD and FRF matrices
                    self.environment_parameters.sysid_averages, # Total frames that could be in the CPSD and FRF matrices
                    )
            output_time_history = self.control_function.control(self.next_drive,self.predicted_response)
        else: # Function
            output_time_history = self.control_function(
                self.data_acquisition_parameters.sample_rate,
                self.environment_parameters.control_signal,
                self.environment_parameters.sysid_frequency_spacing,
                self.frf, # Transfer Functions
                self.sysid_response_noise,  # Noise levels and correlation 
                self.sysid_reference_noise, # from the system identification
                self.sysid_response_cpsd,  # Response levels and correlation
                self.sysid_reference_cpsd, # from the system identification
                self.coherence, # Coherence from the system identification
                self.frames, # Number of frames in the CPSD and FRF matrices
                self.environment_parameters.sysid_averages, # Total frames that could be in the CPSD and FRF matrices
                self.data_acquisition_parameters.output_oversample,
                self.extra_control_parameters, # Required parameters
                self.next_drive, # Last excitation signal for drive-based control
                self.predicted_response, # Last response signal for error correction
                    )
        self.next_drive = output_time_history
        self.show_test_prediction()
        
    def show_test_prediction(self):
        # print('Drive Signals {:}'.format(self.next_drive.shape))
        drive_signals = self.next_drive[:,::self.data_acquisition_parameters.output_oversample]
        impulse_responses = np.moveaxis(np.fft.irfft(self.frf,axis=0),0,-1)

        self.predicted_response = np.zeros((impulse_responses.shape[0],drive_signals.shape[-1]))

        for i,impulse_response_row in enumerate(impulse_responses):
            for j,(impulse,drive) in enumerate(zip(impulse_response_row,drive_signals)):
                # print('Convolving {:},{:}'.format(i,j))
                self.predicted_response[i,:] += sig.convolve(drive, impulse,'full')[:drive_signals.shape[-1]]
        
        # print('Response Prediction {:}'.format(self.predicted_response.shape))
        # print('Control Signal {:}'.format(self.environment_parameters.control_signal.shape))
        time_trac = trac(self.predicted_response,self.environment_parameters.control_signal)
        peak_voltages = np.max(np.abs(self.next_drive),axis=-1)
        self.gui_update_queue.put((self.environment_name,('excitation_voltage_list',peak_voltages)))
        self.gui_update_queue.put((self.environment_name,('response_error_list',time_trac)))
        self.gui_update_queue.put(
            (self.environment_name,
             ('control_predictions',
              (np.arange(self.environment_parameters.control_signal.shape[-1])/self.data_acquisition_parameters.sample_rate,
               drive_signals,
               self.predicted_response,
               self.environment_parameters.control_signal))))
    
    def get_signal_generation_metadata(self):
        return SignalGenerationMetadata(
            samples_per_write = self.data_acquisition_parameters.samples_per_write,
            level_ramp_samples = self.environment_parameters.test_level_ramp_time * self.environment_parameters.sample_rate,
            output_transformation_matrix = self.environment_parameters.reference_transformation_matrix,
            )
    
    def start_control(self,data):
        if self.startup:
            self.test_level,self.repeat = data
            self.log('Starting Environment')
            self.siggen_shutdown_achieved = False
            # Set up the signal generation
            self.queue_container.signal_generation_command_queue.put(
                self.environment_name,
                (SignalGenerationCommands.INITIALIZE_PARAMETERS,
                 self.get_signal_generation_metadata()))
            self.queue_container.signal_generation_command_queue.put(
                self.environment_name,
                (SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,
                 TransientSignalGenerator(self.next_drive,self.repeat)))
            self.queue_container.signal_generation_command_queue.put(
                self.environment_name,
                (SignalGenerationCommands.SET_TEST_LEVEL,self.test_level))
            # Tell the signal generation to start generating signals
            self.queue_container.signal_generation_command_queue.put(
                self.environment_name,
                (SignalGenerationCommands.GENERATE_SIGNALS,None))
            # Set up the measurement buffers
            n_control_channels = (len(self.environment_parameters.control_channel_indices)
                                  if self.environment_parameters.response_transformation_matrix is None
                                  else self.environment_parameters.response_transformation_matrix.shape[0])
            n_output_channels = (len(self.environment_parameters.output_channel_indices)
                                 if self.environment_parameters.reference_transformation_matrix is None
                                 else self.environment_parameters.reference_transformation_matrix.shape[0])
            self.control_buffer = FrameBuffer(
                n_control_channels,
                0,
                0,
                False,
                0,
                0,
                0,
                self.environment_parameters.control_signal.shape[-1],
                0,
                False,
                False,
                False,
                0,
                buffer_size_frame_multiplier=1+(
                    self.data_acquisition_parameters.samples_per_read*buffer_size_samples_per_read_multiplier/
                    self.environment_parameters.control_signal.shape[-1]),
                starting_value = 0.0)
            self.output_buffer = FrameBuffer(
                n_output_channels,
                0,
                0,
                False,
                0,
                0,
                0,
                self.environment_parameters.control_signal.shape[-1],
                0,
                False,
                False,
                False,
                0,
                buffer_size_frame_multiplier=1+(
                    self.data_acquisition_parameters.samples_per_read*buffer_size_samples_per_read_multiplier/
                    self.environment_parameters.control_signal.shape[-1]),
                starting_value = 0.0)
            self.startup = False
        # See if any data has come in
        try:
            acquisition_data,last_acquisition = self.queue_container.data_in_queue.get_nowait()
            if self.last_signal_found is not None:
                self.last_signal_found -= self.data_acquisition_parameters.samples_per_read
            if last_acquisition:
                self.log('Acquired Last Data, Signal Generation Shutdown Achieved: {:}'.format(self.siggen_shutdown_achieved))
            else:
                self.log('Acquired Data')
            scale_factor = 0.0 if self.test_level < 1e-10 else 1/self.test_level
            control_data = acquisition_data[self.environment_parameters.control_channel_indices]*scale_factor
            if not self.environment_parameters.response_transformation_matrix is None:
                control_data = self.environment_parameters.response_transformation_matrix@control_data
            output_data = acquisition_data[self.environment_parameters.output_channel_indices]*scale_factor
            if not self.environment_parameters.reference_transformation_matrix is None:
                output_data = self.environment_parameters.reference_transformation_matrix@output_data
            # Add the data to the buffers
            self.control_buffer.add_data(control_data)
            self.output_buffer.add_data(output_data)
            # Find alignment with the specification via output
            self.log('Aligning signal with specification')
            self.aligned_output,sample_delay,phase_change = align_signals(self.output_buffer[:],self.next_drive[:,::self.data_acquisition_parameters.output_oversample],correlation_threshold = 0.5)
            self.queue_container.gui_update_queue.put((self.environment_name,('time_data',(control_data,output_data,sample_delay)))) # Sample_delay will be None if the alignment is not found
            if not self.aligned_output is None:
                self.log('Alignment Found at {:} samples'.format(sample_delay))
                self.aligned_response = shift_signal(self.control_buffer[:],self.environment_parameters.control_signal.shape[-1],sample_delay,phase_change)
                time_trac = trac(self.aligned_response,self.environment_parameters.control_signal)
                self.gui_update_queue.put((self.environment_name,('control_response_error_list',time_trac)))
                self.queue_container.gui_update_queue.put((self.environment_name,('control_data',(self.aligned_response,self.aligned_output)))) # Sample_delay will be None if the alignment is not found
                # Do the next control
                self.log('Last Signal Found: {:}, Current Signal Found: {:}'.format(self.last_signal_found,sample_delay))
                # We don't want to keep a signal if it starts during the last signal.  Multiply by 0.8 to give a little wiggle room in case the
                # last signal wasn't found exactly at the right place.
                if self.last_signal_found is None or (self.last_signal_found + self.environment_parameters.control_signal.shape[-1]*0.8) < sample_delay:
                    self.next_drive = self.aligned_output
                    self.predicted_response = self.aligned_response
                    self.log('Computing next signal via control law')
                    self.perform_control_prediction(False)
                    self.last_signal_found = sample_delay
                else:
                    self.log('Signal was found previously, not controlling')
        except mp.queues.Empty:
            last_acquisition = False
        # See if we need to keep going
        if self.siggen_shutdown_achieved and last_acquisition:
            self.shutdown()
        else:
            self.queue_container.environment_command_queue.put(self.environment_name,(TransientCommands.START_CONTROL,None))

    def shutdown(self):
        self.log('Environment Shut Down')
        self.gui_update_queue.put((self.environment_name,('enable_control',None)))
        self.startup = True

    def stop_environment(self,data):
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.START_SHUTDOWN,None))

#%% Process

def transient_process(environment_name : str,
                 input_queue : VerboseMessageQueue,
                 gui_update_queue : Queue,
                 controller_communication_queue : VerboseMessageQueue,
                 log_file_queue : Queue,
                 data_in_queue : Queue,
                 data_out_queue : Queue,
                 acquisition_active : mp.Value,
                 output_active : mp.Value):
    try:
        # Create vibration queues
        queue_container = TransientQueues(environment_name,
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
        analysis_proc = mp.Process(target=sysid_data_analysis_process,
                                    args=(environment_name,
                                          queue_container.data_analysis_command_queue,
                                          queue_container.updated_spectral_quantities_queue,
                                          queue_container.time_history_to_generate_queue,
                                          queue_container.environment_command_queue,
                                          queue_container.gui_update_queue,
                                          queue_container.log_file_queue))
        analysis_proc.start()
        siggen_proc = mp.Process(target=signal_generation_process,args=(environment_name,
                                                                        queue_container.signal_generation_command_queue,
                                                                        queue_container.time_history_to_generate_queue,
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
        collection_proc.start()
        
        process_class = TransientEnvironment(
                environment_name,
                queue_container,
                acquisition_active,
                output_active)
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
    except Exception:
        print(traceback.format_exc())