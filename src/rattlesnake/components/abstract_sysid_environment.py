# -*- coding: utf-8 -*-
"""
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

from .abstract_environment import AbstractMetadata,AbstractEnvironment,AbstractUI
from .data_collector import (DataCollectorCommands,CollectorMetadata,
                             AcquisitionType,Acceptance,TriggerSlope,Window)
from .spectral_processing import (SpectralProcessingCommands,SpectralProcessingMetadata,
                                  AveragingTypes,Estimator)
from .signal_generation_process import SignalGenerationCommands,SignalGenerationMetadata
from .signal_generation import (
    SignalGenerator,RandomSignalGenerator,PseudorandomSignalGenerator,
    BurstRandomSignalGenerator,ChirpSignalGenerator)

import os
from abc import abstractmethod
from .utilities import (VerboseMessageQueue,DataAcquisitionParameters,
                        GlobalCommands,error_message_qt)
from .environments import system_identification_ui_path
import time
import numpy as np
import multiprocessing as mp
from qtpy import QtWidgets,uic
from multiprocessing.queues import Queue
import netCDF4 as nc4
import openpyxl
from enum import Enum
from copy import deepcopy
from scipy.io import savemat

class SystemIdCommands(Enum):
    PREVIEW_NOISE = 0
    PREVIEW_TRANSFER_FUNCTION = 1
    START_SYSTEM_ID = 2
    STOP_SYSTEM_ID = 3
    CHECK_FOR_COMPLETE_SHUTDOWN = 4

class AbstractSysIdMetadata(AbstractMetadata):
    """Abstract class for storing metadata for an environment.
    
    This class is used as a storage container for parameters used by an
    environment.  It is returned by the environment UI's
    ``collect_environment_definition_parameters`` function as well as its 
    ``initialize_environment`` function.  Various parts of the controller and
    environment will query the class's data members for parameter values.
    
    Classes inheriting from AbstractMetadata must define:
      1. store_to_netcdf - A function defining the way the parameters are
         stored to a netCDF file saved during streaming operations.
        """
    
    def __init__(self):
        self.sysid_frame_size = None
        self.sysid_averaging_type = None
        self.sysid_noise_averages = None
        self.sysid_averages = None
        self.sysid_exponential_averaging_coefficient = None
        self.sysid_estimator = None
        self.sysid_level = None
        self.sysid_level_ramp_time = None
        self.sysid_signal_type = None
        self.sysid_window = None
        self.sysid_overlap = None
        self.sysid_burst_on = None
        self.sysid_pretrigger = None
        self.sysid_burst_ramp_fraction = None
        
    @property
    @abstractmethod
    def number_of_channels(self):
        pass
        
    @property
    @abstractmethod
    def response_channel_indices(self):
        pass
    
    @property
    @abstractmethod
    def reference_channel_indices(self):
        pass
    
    @property
    def num_response_channels(self):
        return (
            len(self.response_channel_indices)
            if
            self.response_transformation_matrix is None
            else
            self.response_transformation_matrix.shape[0]
            )
    
    @property
    def num_reference_channels(self):
        return (
            len(self.reference_channel_indices)
            if
            self.reference_transformation_matrix is None
            else
            self.reference_transformation_matrix.shape[0]
            )
    
    @property
    @abstractmethod
    def response_transformation_matrix(self):
        pass
    
    @property
    @abstractmethod
    def reference_transformation_matrix(self):
        pass
    
    @property
    def sysid_frequency_spacing(self):
        return self.sample_rate/self.sysid_frame_size
    
    @property
    @abstractmethod
    def sample_rate(self):
        pass
    
    @property
    def sysid_fft_lines(self):
        return self.sysid_frame_size//2 + 1
    
    @property
    def sysid_skip_frames(self):
        return int(
            np.ceil(
                self.sysid_level_ramp_time*self.sample_rate/
                (self.sysid_frame_size*(1-self.sysid_overlap))
                ))
    
    @abstractmethod
    def store_to_netcdf(self,netcdf_group_handle : nc4._netCDF4.Group):
        """Store parameters to a group in a netCDF streaming file.
        
        This function stores parameters from the environment into the netCDF
        file in a group with the environment's name as its name.  The function
        will receive a reference to the group within the dataset and should
        store the environment's parameters into that group in the form of
        attributes, dimensions, or variables.
        
        This function is the "write" counterpart to the retrieve_metadata 
        function in the AbstractUI class, which will read parameters from
        the netCDF file to populate the parameters in the user interface.

        Parameters
        ----------
        netcdf_group_handle : nc4._netCDF4.Group
            A reference to the Group within the netCDF dataset where the
            environment's metadata is stored.

        """
        netcdf_group_handle.sysid_frame_size = self.sysid_frame_size
        netcdf_group_handle.sysid_averaging_type = self.sysid_averaging_type
        netcdf_group_handle.sysid_noise_averages = self.sysid_noise_averages
        netcdf_group_handle.sysid_averages = self.sysid_averages
        netcdf_group_handle.sysid_exponential_averaging_coefficient = self.sysid_exponential_averaging_coefficient
        netcdf_group_handle.sysid_estimator = self.sysid_estimator
        netcdf_group_handle.sysid_level = self.sysid_level
        netcdf_group_handle.sysid_level_ramp_time = self.sysid_level_ramp_time
        netcdf_group_handle.sysid_signal_type = self.sysid_signal_type
        netcdf_group_handle.sysid_window = self.sysid_window
        netcdf_group_handle.sysid_overlap = self.sysid_overlap
        netcdf_group_handle.sysid_burst_on = self.sysid_burst_on
        netcdf_group_handle.sysid_pretrigger = self.sysid_pretrigger
        netcdf_group_handle.sysid_burst_ramp_fraction = self.sysid_burst_ramp_fraction

    def __eq__(self,other):
        try:
            return np.all([np.all(self.__dict__[field] == other.__dict__[field]) for field in self.__dict__])
        except (AttributeError,KeyError):
            return False

from .abstract_sysid_data_analysis import SysIDDataAnalysisCommands

class AbstractSysIdUI(AbstractUI):
    """Abstract User Interface class defining the interface with the controller
    
    This class is used to define the interface between the User Interface of a
    environment in the controller and the main controller."""
    
    @abstractmethod
    def __init__(self,
                 environment_name : str,
                 environment_command_queue : VerboseMessageQueue,
                 controller_communication_queue : VerboseMessageQueue,
                 log_file_queue : Queue,system_id_tabwidget : QtWidgets.QTabWidget):
        """
        Stores data required by the controller to interact with the UI
        
        This class stores data required by the controller to interact with the
        user interface for a given environment.  This includes the environment
        name and queues to pass information between the controller and
        environment.  It additionally initializes the ``command_map`` which is
        used by the Test Profile functionality to map profile instructions to
        operations on the user interface.


        Parameters
        ----------
        environment_name : str
            The name of the environment
        environment_command_queue : VerboseMessageQueue
            A queue that will provide instructions to the corresponding
            environment
        controller_communication_queue : VerboseMessageQueue
            The queue that relays global communication messages to the controller
        log_file_queue : Queue
            The queue that will be used to put messages to the log file.


        """
        super().__init__(environment_name,
             environment_command_queue,controller_communication_queue,log_file_queue)
        # Add the page to the system id tabwidget
        self.system_id_widget = QtWidgets.QWidget()
        uic.loadUi(system_identification_ui_path,self.system_id_widget)
        system_id_tabwidget.addTab(self.system_id_widget,self.environment_name)
        self.connect_sysid_callbacks()
        
        self.data_acquisition_parameters = None
        self.environment_parameters = None
        self.frequencies = None
        self.last_time_response = None
        self.last_transfer_function = None
        self.last_response_noise = None
        self.last_reference_noise = None
        self.last_response_cpsd = None
        self.last_reference_cpsd = None
        self.last_coherence = None
        self.last_condition = None
        
        self.time_response_plot = self.system_id_widget.time_data_graphicslayout.addPlot(row=0,column=0)
        self.time_response_plot.setLabel('left','Response')
        self.time_response_plot.setLabel('bottom','Time (s)')
        self.time_reference_plot = self.system_id_widget.time_data_graphicslayout.addPlot(row=0,column=1)
        self.time_reference_plot.setLabel('left','Reference')
        self.time_reference_plot.setLabel('bottom','Time (s)')
        self.level_response_plot = self.system_id_widget.levels_graphicslayout.addPlot(row=0,column=0)
        self.level_response_plot.setLabel('left','Response PSD')
        self.level_response_plot.setLabel('bottom','Frequency (Hz)')
        self.level_reference_plot = self.system_id_widget.levels_graphicslayout.addPlot(row=0,column=1)
        self.level_reference_plot.setLabel('left','Reference PSD')
        self.level_reference_plot.setLabel('bottom','Frequency (Hz)')
        self.transfer_function_phase_plot = self.system_id_widget.transfer_function_graphics_layout.addPlot(row=0,column=0)
        self.transfer_function_phase_plot.setLabel('left','Phase')
        self.transfer_function_phase_plot.setLabel('bottom','Frequency (Hz)')
        self.transfer_function_magnitude_plot = self.system_id_widget.transfer_function_graphics_layout.addPlot(row=0,column=1)
        self.transfer_function_magnitude_plot.setLabel('left','Amplitude')
        self.transfer_function_magnitude_plot.setLabel('bottom','Frequency (Hz)')
        self.impulse_response_plot = self.system_id_widget.impulse_graphicslayout.addPlot(row=0,column=0)
        self.impulse_response_plot.setLabel('left','Impulse Response')
        self.impulse_response_plot.setLabel('bottom','Time (s)')
        self.coherence_plot = self.system_id_widget.coherence_graphicslayout.addPlot(row=0,column=0)
        self.coherence_plot.setLabel('left','Multiple Coherence')
        self.coherence_plot.setLabel('bottom','Frequency (Hz)')
        self.condition_plot = self.system_id_widget.coherence_graphicslayout.addPlot(row=0,column=1)
        self.condition_plot.setLabel('left','Condition Number')
        self.condition_plot.setLabel('bottom','Frequency (Hz)')
        self.coherence_plot.vb.setLimits(yMin=0,yMax=1)
        self.coherence_plot.vb.disableAutoRange(axis='y')
        for plot in [self.level_response_plot,
                     self.level_reference_plot,
                     self.transfer_function_magnitude_plot,
                     self.condition_plot]:
            plot.setLogMode(False,True)
        self.show_hide_coherence()
        self.show_hide_impulse()
        self.show_hide_levels()
        self.show_hide_time_data()
        self.show_hide_transfer_function()
    
    def connect_sysid_callbacks(self):
        self.system_id_widget.preview_noise_button.clicked.connect(self.preview_noise)
        self.system_id_widget.preview_system_id_button.clicked.connect(self.preview_transfer_function)
        self.system_id_widget.start_button.clicked.connect(self.acquire_transfer_function)
        self.system_id_widget.stop_button.clicked.connect(self.stop_system_id)
        self.system_id_widget.select_transfer_function_stream_file_button.clicked.connect(self.select_transfer_function_stream_file)
        self.system_id_widget.response_selector.itemSelectionChanged.connect(self.update_sysid_plots)
        self.system_id_widget.reference_selector.itemSelectionChanged.connect(self.update_sysid_plots)
        self.system_id_widget.coherence_checkbox.stateChanged.connect(self.show_hide_coherence)
        self.system_id_widget.levels_checkbox.stateChanged.connect(self.show_hide_levels)
        self.system_id_widget.time_data_checkbox.stateChanged.connect(self.show_hide_time_data)
        self.system_id_widget.impulse_checkbox.stateChanged.connect(self.show_hide_impulse)
        self.system_id_widget.transfer_function_checkbox.stateChanged.connect(self.show_hide_transfer_function)
        self.system_id_widget.signalTypeComboBox.currentIndexChanged.connect(self.update_signal_type)
        self.system_id_widget.save_system_id_matrices_button.clicked.connect(self.save_sysid_matrix_file)
        
    @abstractmethod
    def initialize_data_acquisition(self,data_acquisition_parameters : DataAcquisitionParameters):
        """Update the user interface with data acquisition parameters
        
        This function is called when the Data Acquisition parameters are
        initialized.  This function should set up the environment user interface
        accordingly.

        Parameters
        ----------
        data_acquisition_parameters : DataAcquisitionParameters :
            Container containing the data acquisition parameters, including
            channel table and sampling information.

        """
        self.log('Initializing Data Acquisition')
        # Store for later
        self.data_acquisition_parameters = data_acquisition_parameters
    
    @abstractmethod
    def collect_environment_definition_parameters(self) -> AbstractSysIdMetadata:
        """
        Collect the parameters from the user interface defining the environment

        Returns
        -------
        AbstractSysIdMetadata
            A metadata or parameters object containing the parameters defining
            the corresponding environment.

        """
        pass
    
    def update_sysid_metadata(self,metadata : AbstractSysIdMetadata):
        metadata.sysid_frame_size = self.system_id_widget.samplesPerFrameSpinBox.value()
        metadata.sysid_averaging_type = self.system_id_widget.averagingTypeComboBox.itemText(self.system_id_widget.averagingTypeComboBox.currentIndex())
        metadata.sysid_noise_averages = self.system_id_widget.noiseAveragesSpinBox.value()
        metadata.sysid_averages = self.system_id_widget.systemIDAveragesSpinBox.value()
        metadata.sysid_exponential_averaging_coefficient = self.system_id_widget.averagingCoefficientDoubleSpinBox.value()
        metadata.sysid_estimator = self.system_id_widget.estimatorComboBox.itemText(self.system_id_widget.estimatorComboBox.currentIndex())
        metadata.sysid_level = self.system_id_widget.levelDoubleSpinBox.value()
        metadata.sysid_level_ramp_time = self.system_id_widget.levelRampTimeDoubleSpinBox.value()
        metadata.sysid_signal_type = self.system_id_widget.signalTypeComboBox.itemText(self.system_id_widget.signalTypeComboBox.currentIndex())
        metadata.sysid_window = self.system_id_widget.windowComboBox.itemText(self.system_id_widget.windowComboBox.currentIndex())
        metadata.sysid_overlap = self.system_id_widget.overlapDoubleSpinBox.value()/100 if metadata.sysid_signal_type == 'Random' else 0.0
        metadata.sysid_burst_on = self.system_id_widget.onFractionDoubleSpinBox.value()/100
        metadata.sysid_pretrigger = self.system_id_widget.pretriggerDoubleSpinBox.value()/100
        metadata.sysid_burst_ramp_fraction = self.system_id_widget.rampFractionDoubleSpinBox.value()/100
        # for key in dir(metadata):
        #     if '__' == key[:2]:
        #         continue
        #     print('Key: {:}'.format(key))
        #     print('Value: {:}'.format(getattr(metadata,key)))
    
    @property
    @abstractmethod
    def initialized_control_names(self):
        pass
    
    @property
    @abstractmethod
    def initialized_output_names(self):
        pass
    
    @abstractmethod
    def initialize_environment(self) -> AbstractMetadata:
        """
        Update the user interface with environment parameters
        
        This function is called when the Environment parameters are initialized.
        This function should set up the user interface accordingly.  It must
        return the parameters class of the environment that inherits from
        AbstractMetadata.

        Returns
        -------
        AbstractMetadata
            An AbstractMetadata-inheriting object that contains the parameters
            defining the environment.

        """
        self.environment_parameters = self.collect_environment_definition_parameters()
        self.update_sysid_metadata(self.environment_parameters)
        self.system_id_widget.reference_selector.blockSignals(True)
        self.system_id_widget.response_selector.blockSignals(True)
        self.system_id_widget.reference_selector.clear()
        self.system_id_widget.response_selector.clear()
        for i,control_name in enumerate(self.initialized_control_names):
            self.system_id_widget.response_selector.addItem('{:}: {:}'.format(i+1,control_name))
        for i,drive_name in enumerate(self.initialized_output_names):
            self.system_id_widget.reference_selector.addItem('{:}: {:}'.format(i+1,drive_name))
        self.system_id_widget.reference_selector.blockSignals(False)
        self.system_id_widget.response_selector.blockSignals(False)
        self.system_id_widget.reference_selector.setCurrentRow(0)
        self.system_id_widget.response_selector.setCurrentRow(0)
        self.update_signal_type()
        return self.environment_parameters
    
    def preview_noise(self):
        self.log('Starting Noise Preview')
        self.update_sysid_metadata(self.environment_parameters)
        for widget in [self.system_id_widget.preview_noise_button,
                       self.system_id_widget.preview_system_id_button,
                       self.system_id_widget.start_button,
                       self.system_id_widget.samplesPerFrameSpinBox,
                       self.system_id_widget.averagingTypeComboBox,
                       self.system_id_widget.noiseAveragesSpinBox,
                       self.system_id_widget.systemIDAveragesSpinBox,
                       self.system_id_widget.averagingCoefficientDoubleSpinBox,
                       self.system_id_widget.estimatorComboBox,
                       self.system_id_widget.levelDoubleSpinBox,
                       self.system_id_widget.signalTypeComboBox,
                       self.system_id_widget.windowComboBox,
                       self.system_id_widget.overlapDoubleSpinBox,
                       self.system_id_widget.onFractionDoubleSpinBox,
                       self.system_id_widget.pretriggerDoubleSpinBox,
                       self.system_id_widget.rampFractionDoubleSpinBox,
                       self.system_id_widget.stream_transfer_function_data_checkbox,
                       self.system_id_widget.select_transfer_function_stream_file_button,
                       self.system_id_widget.transfer_function_stream_file_display,
                       self.system_id_widget.levelRampTimeDoubleSpinBox,
                       self.system_id_widget.save_system_id_matrices_button]:
            widget.setEnabled(False)
        for widget in [self.system_id_widget.stop_button]:
            widget.setEnabled(True)
        self.environment_command_queue.put(self.log_name,(SystemIdCommands.PREVIEW_NOISE,self.environment_parameters))
        
    def preview_transfer_function(self):
        self.log('Starting System ID Preview')
        self.update_sysid_metadata(self.environment_parameters)
        for widget in [self.system_id_widget.preview_noise_button,
                       self.system_id_widget.preview_system_id_button,
                       self.system_id_widget.start_button,
                       self.system_id_widget.samplesPerFrameSpinBox,
                       self.system_id_widget.averagingTypeComboBox,
                       self.system_id_widget.noiseAveragesSpinBox,
                       self.system_id_widget.systemIDAveragesSpinBox,
                       self.system_id_widget.averagingCoefficientDoubleSpinBox,
                       self.system_id_widget.estimatorComboBox,
                       self.system_id_widget.levelDoubleSpinBox,
                       self.system_id_widget.signalTypeComboBox,
                       self.system_id_widget.windowComboBox,
                       self.system_id_widget.overlapDoubleSpinBox,
                       self.system_id_widget.onFractionDoubleSpinBox,
                       self.system_id_widget.pretriggerDoubleSpinBox,
                       self.system_id_widget.rampFractionDoubleSpinBox,
                       self.system_id_widget.stream_transfer_function_data_checkbox,
                       self.system_id_widget.select_transfer_function_stream_file_button,
                       self.system_id_widget.transfer_function_stream_file_display,
                       self.system_id_widget.levelRampTimeDoubleSpinBox,
                       self.system_id_widget.save_system_id_matrices_button]:
            widget.setEnabled(False)
        for widget in [self.system_id_widget.stop_button]:
            widget.setEnabled(True)
        self.environment_command_queue.put(self.log_name,(SystemIdCommands.PREVIEW_TRANSFER_FUNCTION,(self.environment_parameters)))
    
    def acquire_transfer_function(self):
        self.log('Starting System ID')
        self.update_sysid_metadata(self.environment_parameters)
        for widget in [self.system_id_widget.preview_noise_button,
                       self.system_id_widget.preview_system_id_button,
                       self.system_id_widget.start_button,
                       self.system_id_widget.samplesPerFrameSpinBox,
                       self.system_id_widget.averagingTypeComboBox,
                       self.system_id_widget.noiseAveragesSpinBox,
                       self.system_id_widget.systemIDAveragesSpinBox,
                       self.system_id_widget.averagingCoefficientDoubleSpinBox,
                       self.system_id_widget.estimatorComboBox,
                       self.system_id_widget.levelDoubleSpinBox,
                       self.system_id_widget.signalTypeComboBox,
                       self.system_id_widget.windowComboBox,
                       self.system_id_widget.overlapDoubleSpinBox,
                       self.system_id_widget.onFractionDoubleSpinBox,
                       self.system_id_widget.pretriggerDoubleSpinBox,
                       self.system_id_widget.rampFractionDoubleSpinBox,
                       self.system_id_widget.stream_transfer_function_data_checkbox,
                       self.system_id_widget.select_transfer_function_stream_file_button,
                       self.system_id_widget.transfer_function_stream_file_display,
                       self.system_id_widget.levelRampTimeDoubleSpinBox,
                       self.system_id_widget.save_system_id_matrices_button]:
            widget.setEnabled(False)
        for widget in [self.system_id_widget.stop_button]:
            widget.setEnabled(True)
        if self.system_id_widget.stream_transfer_function_data_checkbox.isChecked():
            stream_name = self.system_id_widget.transfer_function_stream_file_display.text()
        else:
            stream_name = None
        self.environment_command_queue.put(self.log_name,(SystemIdCommands.START_SYSTEM_ID,(self.environment_parameters,stream_name)))
    
    def stop_system_id(self):
        self.log('Stopping System ID')
        self.system_id_widget.stop_button.setEnabled(False)
        self.environment_command_queue.put(self.log_name,(SystemIdCommands.STOP_SYSTEM_ID,True))
    
    def select_transfer_function_stream_file(self):
        """Select a file to save transfer function data to"""
        filename,file_filter = QtWidgets.QFileDialog.getSaveFileName(self.system_id_widget,'Select NetCDF File to Save Transfer Function Data',filter='NetCDF File (*.nc4)')
        if filename == '':
            return
        self.system_id_widget.transfer_function_stream_file_display.setText(filename)
        self.system_id_widget.stream_transfer_function_data_checkbox.setChecked(True)
    
    def update_sysid_plots(self,update_time = True,
                           update_transfer_function = True,
                           update_noise = True):
        # Figure out the selected entries
        response_indices = [i for i in range(
            self.system_id_widget.response_selector.count()) if 
            self.system_id_widget.response_selector.item(i).isSelected()]
        reference_indices = [i for i in range(
            self.system_id_widget.reference_selector.count()) if 
            self.system_id_widget.reference_selector.item(i).isSelected()]
        # print(response_indices)
        # print(reference_indices)
        if update_time:
            self.time_response_plot.clear()
            self.time_reference_plot.clear()
            if self.last_time_response is not None:
                response_frame_indices = np.array(self.environment_parameters.response_channel_indices)[response_indices]
                reference_frame_indices = np.array(self.environment_parameters.reference_channel_indices)[reference_indices]
                response_time_data = self.last_time_response[response_frame_indices]
                reference_time_data = self.last_time_response[reference_frame_indices]
                times = np.arange(response_time_data.shape[-1])/self.data_acquisition_parameters.sample_rate
                for i,time_data in enumerate(response_time_data):
                    self.time_response_plot.plot(times,time_data,pen=i)
                for i,time_data in enumerate(reference_time_data):
                    self.time_reference_plot.plot(times,time_data,pen=i)
        if update_transfer_function:
            self.transfer_function_phase_plot.clear()
            self.transfer_function_magnitude_plot.clear()
            self.condition_plot.clear()
            self.coherence_plot.clear()
            self.impulse_response_plot.clear()
            if self.last_transfer_function is not None and len(response_indices) > 0 and len(reference_indices) > 0:
                # print(self.last_transfer_function)
                # print(np.array(response_indices)[:,np.newaxis])
                # print(np.array(reference_indices))
                frf_section = np.reshape(
                    self.last_transfer_function[
                        ...,
                        np.array(response_indices)[:,np.newaxis],
                        np.array(reference_indices)],
                    (self.frequencies.size,-1)).T
                impulse_response = np.fft.irfft(frf_section,axis=-1)
                for i,(frf,imp) in enumerate(zip(frf_section,impulse_response)):
                    self.transfer_function_phase_plot.plot(self.frequencies,
                                                           np.angle(frf)*180/np.pi,pen=i)
                    self.transfer_function_magnitude_plot.plot(self.frequencies,
                                                               np.abs(frf),pen=i)
                    self.impulse_response_plot.plot(np.arange(imp.size)/self.environment_parameters.sample_rate,imp,pen=i)
                for i,coherence in enumerate(self.last_coherence[...,response_indices].T):
                    self.coherence_plot.plot(self.frequencies,coherence,pen=i)
            if self.last_condition is not None:
                self.condition_plot.plot(self.frequencies,
                                         self.last_condition,pen=0)
        if update_noise:
            reference_noise = None if self.last_reference_noise is None or len(reference_indices) == 0 else self.last_reference_noise[...,reference_indices,reference_indices].real
            response_noise = None if self.last_response_noise is None or len(response_indices) == 0 else self.last_response_noise[...,response_indices,response_indices].real
            reference_level = None if self.last_reference_cpsd is None or len(reference_indices) == 0 else self.last_reference_cpsd[...,reference_indices,reference_indices].real
            response_level = None if self.last_response_cpsd is None or len(response_indices) == 0 else self.last_response_cpsd[...,response_indices,response_indices].real
            self.level_reference_plot.clear()
            self.level_response_plot.clear()
            for i in range(len(reference_indices)):
                if reference_noise is not None:
                    self.level_reference_plot.plot(self.frequencies,reference_noise[:,i],pen=i)
                if reference_level is not None:
                    try:
                        self.level_reference_plot.plot(self.frequencies,reference_level[:,i],pen=i)
                    except Exception:
                        pass
            for i in range(len(response_indices)):
                if response_noise is not None:
                    self.level_response_plot.plot(self.frequencies,response_noise[:,i],pen=i)
                if response_level is not None:
                    try:
                        self.level_response_plot.plot(self.frequencies,response_level[:,i],pen=i)
                    except Exception:
                        pass
    
    def show_hide_coherence(self):
        if self.system_id_widget.coherence_checkbox.isChecked():
            self.system_id_widget.coherence_groupbox.show()
        else:
            self.system_id_widget.coherence_groupbox.hide()
    
    def show_hide_levels(self):
        if self.system_id_widget.levels_checkbox.isChecked():
            self.system_id_widget.levels_groupbox.show()
        else:
            self.system_id_widget.levels_groupbox.hide()
    
    def show_hide_time_data(self):
        if self.system_id_widget.time_data_checkbox.isChecked():
            self.system_id_widget.time_data_groupbox.show()
        else:
            self.system_id_widget.time_data_groupbox.hide()
            
    def show_hide_transfer_function(self):
        if self.system_id_widget.transfer_function_checkbox.isChecked():
            self.system_id_widget.transfer_function_groupbox.show()
        else:
            self.system_id_widget.transfer_function_groupbox.hide()
    
    def show_hide_impulse(self):
        if self.system_id_widget.impulse_checkbox.isChecked():
            self.system_id_widget.impulse_groupbox.show()
        else:
            self.system_id_widget.impulse_groupbox.hide()
    
    def update_signal_type(self):
        if self.system_id_widget.signalTypeComboBox.currentIndex() == 0: # Random
            self.system_id_widget.windowComboBox.setCurrentIndex(0)
            self.system_id_widget.overlapDoubleSpinBox.show()
            self.system_id_widget.overlapLabel.show()
            self.system_id_widget.onFractionLabel.hide()
            self.system_id_widget.onFractionDoubleSpinBox.hide()
            self.system_id_widget.pretriggerLabel.hide()
            self.system_id_widget.pretriggerDoubleSpinBox.hide()
            self.system_id_widget.rampFractionLabel.hide()
            self.system_id_widget.rampFractionDoubleSpinBox.hide()
        elif self.system_id_widget.signalTypeComboBox.currentIndex() == 1: # Pseudorandom
            self.system_id_widget.windowComboBox.setCurrentIndex(1)
            self.system_id_widget.overlapDoubleSpinBox.hide()
            self.system_id_widget.overlapLabel.hide()
            self.system_id_widget.onFractionLabel.hide()
            self.system_id_widget.onFractionDoubleSpinBox.hide()
            self.system_id_widget.pretriggerLabel.hide()
            self.system_id_widget.pretriggerDoubleSpinBox.hide()
            self.system_id_widget.rampFractionLabel.hide()
            self.system_id_widget.rampFractionDoubleSpinBox.hide()
        elif self.system_id_widget.signalTypeComboBox.currentIndex() == 2: # Burst
            self.system_id_widget.windowComboBox.setCurrentIndex(1)
            self.system_id_widget.overlapDoubleSpinBox.hide()
            self.system_id_widget.overlapLabel.hide()
            self.system_id_widget.onFractionLabel.show()
            self.system_id_widget.onFractionDoubleSpinBox.show()
            self.system_id_widget.pretriggerLabel.show()
            self.system_id_widget.pretriggerDoubleSpinBox.show()
            self.system_id_widget.rampFractionLabel.show()
            self.system_id_widget.rampFractionDoubleSpinBox.show()
        elif self.system_id_widget.signalTypeComboBox.currentIndex() == 3: # Chirp
            self.system_id_widget.windowComboBox.setCurrentIndex(1)
            self.system_id_widget.overlapDoubleSpinBox.hide()
            self.system_id_widget.overlapLabel.hide()
            self.system_id_widget.onFractionLabel.hide()
            self.system_id_widget.onFractionDoubleSpinBox.hide()
            self.system_id_widget.pretriggerLabel.hide()
            self.system_id_widget.pretriggerDoubleSpinBox.hide()
            self.system_id_widget.rampFractionLabel.hide()
            self.system_id_widget.rampFractionDoubleSpinBox.hide()
            
    
    @abstractmethod
    def retrieve_metadata(self, netcdf_handle : nc4._netCDF4.Dataset):
        """Collects environment parameters from a netCDF dataset.

        This function retrieves parameters from a netCDF dataset that was written
        by the controller during streaming.  It must populate the widgets
        in the user interface with the proper information.

        This function is the "read" counterpart to the store_to_netcdf 
        function in the AbstractMetadata class, which will write parameters to
        the netCDF file to document the metadata.
        
        Note that the entire dataset is passed to this function, so the function
        should collect parameters pertaining to the environment from a Group
        in the dataset sharing the environment's name, e.g.
        
        ``group = netcdf_handle.groups[self.environment_name]``
        ``self.definition_widget.parameter_selector.setValue(group.parameter)``
        
        Parameters
        ----------
        netcdf_handle : nc4._netCDF4.Dataset :
            The netCDF dataset from which the data will be read.  It should have
            a group name with the enviroment's name.

        """
        # Get the group
        group = netcdf_handle.groups[self.environment_name]
        self.system_id_widget.samplesPerFrameSpinBox.setValue(group.sysid_frame_size)
        self.system_id_widget.averagingTypeComboBox.setCurrentIndex(self.system_id_widget.averagingTypeComboBox.findText(group.sysid_averaging_type))
        self.system_id_widget.noiseAveragesSpinBox.setValue(group.sysid_noise_averages)
        self.system_id_widget.systemIDAveragesSpinBox.setValue(group.sysid_averages)
        self.system_id_widget.averagingCoefficientDoubleSpinBox.setValue(group.sysid_exponential_averaging_coefficient)
        self.system_id_widget.estimatorComboBox.setCurrentIndex(self.system_id_widget.estimatorComboBox.findText(group.sysid_estimator))
        self.system_id_widget.levelDoubleSpinBox.setValue(group.sysid_level)
        self.system_id_widget.levelRampTimeDoubleSpinBox.setValue(group.sysid_level_ramp_time)
        self.system_id_widget.signalTypeComboBox.setCurrentIndex(self.system_id_widget.signalTypeComboBox.findText(group.sysid_signal_type))
        self.system_id_widget.windowComboBox.setCurrentIndex(self.system_id_widget.windowComboBox.findText(group.sysid_window ))
        self.system_id_widget.overlapDoubleSpinBox.setValue(group.sysid_overlap*100)
        self.system_id_widget.onFractionDoubleSpinBox.setValue(group.sysid_burst_on*100)
        self.system_id_widget.pretriggerDoubleSpinBox.setValue(group.sysid_pretrigger*100)
        self.system_id_widget.rampFractionDoubleSpinBox.setValue(group.sysid_burst_ramp_fraction*100)
    
    @abstractmethod
    def update_gui(self,queue_data : tuple):
        """Update the environment's graphical user interface
        
        This function will receive data from the gui_update_queue that
        specifies how the user interface should be updated.  Data will usually
        be received as ``(instruction,data)`` pairs, where the ``instruction`` notes
        what operation should be taken or which widget should be modified, and
        the ``data`` notes what data should be used in the update.

        Parameters
        ----------
        queue_data : tuple
            A tuple containing ``(instruction,data)`` pairs where ``instruction``
            defines and operation or widget to be modified and ``data`` contains
            the data used to perform the operation.
        """
        message,data = queue_data
        self.log('Got GUI Message {:}'.format(message))
        # print('Update GUI Got {:}'.format(message))
        if message == 'time_frame':
            self.last_time_response,accept = data
            self.update_sysid_plots(update_time = True,
                                    update_transfer_function = False,
                                    update_noise = False)
        elif message == 'noise_update':
            (frames,total_frames,self.frequencies,
             self.last_response_noise,self.last_reference_noise) = data
            self.update_sysid_plots(update_time = False,
                                    update_transfer_function = False,
                                    update_noise = True)
            self.system_id_widget.current_frames_spinbox.setValue(frames)
            self.system_id_widget.total_frames_spinbox.setValue(total_frames)
            self.system_id_widget.progressBar.setValue(int(frames/total_frames*100))
        elif message == 'sysid_update':
            (frames,total_frames,self.frequencies,
             self.last_transfer_function,self.last_coherence,
             self.last_response_cpsd,self.last_reference_cpsd,
             self.last_condition) = data
            # print(self.last_transfer_function.shape)
            # print(self.last_coherence.shape)
            # print(self.last_response_cpsd.shape)
            # print(self.last_reference_cpsd.shape)
            self.update_sysid_plots(update_time = False,
                                    update_transfer_function = True,
                                    update_noise = True)
            self.system_id_widget.current_frames_spinbox.setValue(frames)
            self.system_id_widget.total_frames_spinbox.setValue(total_frames)
            self.system_id_widget.progressBar.setValue(int(frames/total_frames*100))
        elif message == 'enable_system_id':
            for widget in [self.system_id_widget.preview_noise_button,
                           self.system_id_widget.preview_system_id_button,
                           self.system_id_widget.start_button,
                           self.system_id_widget.samplesPerFrameSpinBox,
                           self.system_id_widget.averagingTypeComboBox,
                           self.system_id_widget.noiseAveragesSpinBox,
                           self.system_id_widget.systemIDAveragesSpinBox,
                           self.system_id_widget.averagingCoefficientDoubleSpinBox,
                           self.system_id_widget.estimatorComboBox,
                           self.system_id_widget.levelDoubleSpinBox,
                           self.system_id_widget.signalTypeComboBox,
                           self.system_id_widget.windowComboBox,
                           self.system_id_widget.overlapDoubleSpinBox,
                           self.system_id_widget.onFractionDoubleSpinBox,
                           self.system_id_widget.pretriggerDoubleSpinBox,
                           self.system_id_widget.rampFractionDoubleSpinBox,
                           self.system_id_widget.stream_transfer_function_data_checkbox,
                           self.system_id_widget.select_transfer_function_stream_file_button,
                           self.system_id_widget.transfer_function_stream_file_display,
                           self.system_id_widget.levelRampTimeDoubleSpinBox,
                           self.system_id_widget.save_system_id_matrices_button]:
                widget.setEnabled(True)
            for widget in [self.system_id_widget.stop_button]:
                widget.setEnabled(False)
        elif message == 'disable_system_id':
            for widget in [self.system_id_widget.preview_noise_button,
                           self.system_id_widget.preview_system_id_button,
                           self.system_id_widget.start_button,
                           self.system_id_widget.samplesPerFrameSpinBox,
                           self.system_id_widget.averagingTypeComboBox,
                           self.system_id_widget.noiseAveragesSpinBox,
                           self.system_id_widget.systemIDAveragesSpinBox,
                           self.system_id_widget.averagingCoefficientDoubleSpinBox,
                           self.system_id_widget.estimatorComboBox,
                           self.system_id_widget.levelDoubleSpinBox,
                           self.system_id_widget.signalTypeComboBox,
                           self.system_id_widget.windowComboBox,
                           self.system_id_widget.overlapDoubleSpinBox,
                           self.system_id_widget.onFractionDoubleSpinBox,
                           self.system_id_widget.pretriggerDoubleSpinBox,
                           self.system_id_widget.rampFractionDoubleSpinBox,
                           self.system_id_widget.stream_transfer_function_data_checkbox,
                           self.system_id_widget.select_transfer_function_stream_file_button,
                           self.system_id_widget.transfer_function_stream_file_display,
                           self.system_id_widget.levelRampTimeDoubleSpinBox,
                           self.system_id_widget.save_system_id_matrices_button]:
                widget.setEnabled(False)
            for widget in [self.system_id_widget.stop_button]:
                widget.setEnabled(True)
        else:
            return False
        return True
        
        
    @staticmethod
    @abstractmethod
    def create_environment_template(environment_name : str, workbook : openpyxl.workbook.workbook.Workbook):
        """Creates a template worksheet in an Excel workbook defining the
        environment.
        
        This function creates a template worksheet in an Excel workbook that
        when filled out could be read by the controller to re-create the 
        environment.
        
        This function is the "write" counterpart to the 
        ``set_parameters_from_template`` function in the ``AbstractUI`` class,
        which reads the values from the template file to populate the user
        interface.

        Parameters
        ----------
        environment_name : str :
            The name of the environment that will specify the worksheet's name
        workbook : openpyxl.workbook.workbook.Workbook :
            A reference to an ``openpyxl`` workbook.

        """
        pass
    
    @abstractmethod
    def set_parameters_from_template(self, worksheet : openpyxl.worksheet.worksheet.Worksheet):
        """
        Collects parameters for the user interface from the Excel template file
        
        This function reads a filled out template worksheet to create an
        environment.  Cells on this worksheet contain parameters needed to
        specify the environment, so this function should read those cells and
        update the UI widgets with those parameters.
        
        This function is the "read" counterpart to the 
        ``create_environment_template`` function in the ``AbstractUI`` class,
        which writes a template file that can be filled out by a user.
        

        Parameters
        ----------
        worksheet : openpyxl.worksheet.worksheet.Worksheet
            An openpyxl worksheet that contains the environment template.
            Cells on this worksheet should contain the parameters needed for the
            user interface.

        """
        pass
    
    def save_sysid_matrix_file(self):
        if self.last_transfer_function is None or self.last_response_noise is None:
            error_message_qt('Run System Identification First!',
                             'System Identification Matrices not yet created.\n\nRun System Identification First!')
            return
        filename,file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self.system_id_widget,'Select File to Save Transfer Function Matrices',
            filter='NetCDF File (*.nc4);;MatLab File (*.mat);;Numpy File (*.npz)')
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
        if file_filter == 'NetCDF File (*.nc4)':
            netcdf_handle = nc4.Dataset(filename,'w',format='NETCDF4',clobber=True)
            # Create dimensions
            netcdf_handle.createDimension('response_channels',len(self.data_acquisition_parameters.channel_list))
            
            netcdf_handle.createDimension('num_environments',len(self.data_acquisition_parameters.environment_names))
            # Create attributes
            netcdf_handle.file_version = '3.0.0'
            netcdf_handle.sample_rate = self.data_acquisition_parameters.sample_rate
            netcdf_handle.time_per_write = self.data_acquisition_parameters.samples_per_write/self.data_acquisition_parameters.output_sample_rate
            netcdf_handle.time_per_read = self.data_acquisition_parameters.samples_per_read/self.data_acquisition_parameters.sample_rate
            netcdf_handle.hardware = self.data_acquisition_parameters.hardware
            netcdf_handle.hardware_file = 'None' if self.data_acquisition_parameters.hardware_file is None else self.data_acquisition_parameters.hardware_file
            netcdf_handle.maximum_acquisition_processes = self.data_acquisition_parameters.maximum_acquisition_processes
            netcdf_handle.output_oversample = self.data_acquisition_parameters.output_oversample
            # Create Variables
            var = netcdf_handle.createVariable('environment_names',str,('num_environments',))
            this_environment_index = None
            for i,name in enumerate(self.data_acquisition_parameters.environment_names):
                var[i] = name
                if name == self.environment_name:
                    this_environment_index = i
            var = netcdf_handle.createVariable('environment_active_channels','i1',('response_channels','num_environments'))
            var[...] = self.data_acquisition_parameters.environment_active_channels.astype('int8')[
                self.data_acquisition_parameters.environment_active_channels[:,this_environment_index],:]
            # Create channel table variables
            for (label,netcdf_datatype) in labels:
                var = netcdf_handle.createVariable('/channels/'+label,netcdf_datatype,('response_channels',))
                channel_data = [getattr(channel,label) for channel in self.data_acquisition_parameters.channel_list]
                if netcdf_datatype == 'i1':
                    channel_data = np.array([1 if val else 0 for val in channel_data])
                else:
                    channel_data = ['' if val is None else val for val in channel_data]
                for i,cd in enumerate(channel_data):
                    var[i] = cd
            group_handle = netcdf_handle.createGroup(self.environment_name)
            self.environment_parameters.store_to_netcdf(group_handle)
            try:
                group_handle.createDimension('control_channels',self.last_transfer_function.shape[1])
            except RuntimeError:
                pass
            try:
                group_handle.createDimension('output_channels',self.last_transfer_function.shape[2])
            except RuntimeError:
                pass
            try:
                group_handle.createDimension('fft_lines',self.last_transfer_function.shape[0])
            except RuntimeError:
                pass
            var = group_handle.createVariable('frf_data_real','f8',('fft_lines','control_channels','output_channels'))
            var[...] = self.last_transfer_function.real
            var = group_handle.createVariable('frf_data_imag','f8',('fft_lines','control_channels','output_channels'))
            var[...] = self.last_transfer_function.imag
            var = group_handle.createVariable('frf_coherence','f8',('fft_lines','control_channels'))
            var[...] = self.last_coherence.real
            var = group_handle.createVariable('response_cpsd_real','f8',('fft_lines','control_channels','control_channels'))
            var[...] = self.last_response_cpsd.real
            var = group_handle.createVariable('response_cpsd_imag','f8',('fft_lines','control_channels','control_channels'))
            var[...] = self.last_response_cpsd.imag
            var = group_handle.createVariable('reference_cpsd_real','f8',('fft_lines','output_channels','output_channels'))
            var[...] = self.last_reference_cpsd.real
            var = group_handle.createVariable('reference_cpsd_imag','f8',('fft_lines','output_channels','output_channels'))
            var[...] = self.last_reference_cpsd.imag
            var = group_handle.createVariable('response_noise_cpsd_real','f8',('fft_lines','control_channels','control_channels'))
            var[...] = self.last_response_noise.real
            var = group_handle.createVariable('response_noise_cpsd_imag','f8',('fft_lines','control_channels','control_channels'))
            var[...] = self.last_response_noise.imag
            var = group_handle.createVariable('reference_noise_cpsd_real','f8',('fft_lines','output_channels','output_channels'))
            var[...] = self.last_reference_noise.real
            var = group_handle.createVariable('reference_noise_cpsd_imag','f8',('fft_lines','output_channels','output_channels'))
            var[...] = self.last_reference_noise.imag
        else:
            field_dict = {}
            field_dict['version'] = '3.0.0'
            field_dict['sample_rate'] = self.data_acquisition_parameters.sample_rate
            field_dict['time_per_write'] = self.data_acquisition_parameters.samples_per_write/self.data_acquisition_parameters.output_sample_rate
            field_dict['time_per_read'] = self.data_acquisition_parameters.samples_per_read/self.data_acquisition_parameters.sample_rate
            field_dict['hardware'] = self.data_acquisition_parameters.hardware
            field_dict['hardware_file'] = 'None' if self.data_acquisition_parameters.hardware_file is None else self.data_acquisition_parameters.hardware_file
            field_dict['maximum_acquisition_processes'] = self.data_acquisition_parameters.maximum_acquisition_processes
            field_dict['output_oversample'] = self.data_acquisition_parameters.output_oversample
            field_dict['frf_data'] = self.last_transfer_function
            field_dict['response_cpsd'] = self.last_response_cpsd
            field_dict['reference_cpsd'] = self.last_reference_cpsd
            field_dict['coherence'] = self.last_coherence
            field_dict['response_noise_cpsd'] = self.last_response_noise
            field_dict['reference_noise_cpsd'] = self.last_reference_noise
            field_dict['response_indices'] = self.environment_parameters.response_channel_indices
            field_dict['reference_indices'] = self.environment_parameters.reference_channel_indices
            field_dict['response_transformation_matrix'] = np.nan if self.environment_parameters.response_transformation_matrix is None else self.environment_parameters.response_transformation_matrix
            field_dict['reference_transformation_matrix'] = np.nan if self.environment_parameters.reference_transformation_matrix is None else self.environment_parameters.reference_transformation_matrix
            field_dict['sysid_frequency_spacing'] = self.environment_parameters.sysid_frequency_spacing
            for key,value in self.environment_parameters.__dict__.items():
                try:
                    if 'sysid_' in key:
                        field_dict[key] = np.array(value)
                except TypeError:
                    continue
            for label,datatype in labels:
                field_dict['channel_'+label] = np.array([
                    '' if getattr(channel,label) is None else getattr(channel,label) for channel
                    in self.data_acquisition_parameters.channel_list])
            # print(field_dict)
            if file_filter == 'MatLab File (*.mat)':
                for field in ['frf_data',
                              'response_cpsd',
                              'reference_cpsd',
                              'coherence',
                              'response_noise_cpsd',
                              'reference_noise_cpsd']:
                    field_dict[field] = np.moveaxis(field_dict[field],0,-1)
                savemat(filename,field_dict)
            elif file_filter == 'Numpy File (*.npz)':
                np.savez(filename,**field_dict)
                
    def disable_system_id_daq_armed(self):
        for widget in [self.system_id_widget.preview_noise_button,
                       self.system_id_widget.preview_system_id_button,
                       self.system_id_widget.start_button,
                       self.system_id_widget.samplesPerFrameSpinBox,
                       self.system_id_widget.averagingTypeComboBox,
                       self.system_id_widget.noiseAveragesSpinBox,
                       self.system_id_widget.systemIDAveragesSpinBox,
                       self.system_id_widget.averagingCoefficientDoubleSpinBox,
                       self.system_id_widget.estimatorComboBox,
                       self.system_id_widget.levelDoubleSpinBox,
                       self.system_id_widget.signalTypeComboBox,
                       self.system_id_widget.windowComboBox,
                       self.system_id_widget.overlapDoubleSpinBox,
                       self.system_id_widget.onFractionDoubleSpinBox,
                       self.system_id_widget.pretriggerDoubleSpinBox,
                       self.system_id_widget.rampFractionDoubleSpinBox,
                       self.system_id_widget.stream_transfer_function_data_checkbox,
                       self.system_id_widget.select_transfer_function_stream_file_button,
                       self.system_id_widget.transfer_function_stream_file_display,
                       self.system_id_widget.levelRampTimeDoubleSpinBox,
                       self.system_id_widget.save_system_id_matrices_button]:
            widget.setEnabled(False)
        for widget in [self.system_id_widget.stop_button]:
            widget.setEnabled(False)
        
    def enable_system_id_daq_disarmed(self):
        for widget in [self.system_id_widget.preview_noise_button,
                       self.system_id_widget.preview_system_id_button,
                       self.system_id_widget.start_button,
                       self.system_id_widget.samplesPerFrameSpinBox,
                       self.system_id_widget.averagingTypeComboBox,
                       self.system_id_widget.noiseAveragesSpinBox,
                       self.system_id_widget.systemIDAveragesSpinBox,
                       self.system_id_widget.averagingCoefficientDoubleSpinBox,
                       self.system_id_widget.estimatorComboBox,
                       self.system_id_widget.levelDoubleSpinBox,
                       self.system_id_widget.signalTypeComboBox,
                       self.system_id_widget.windowComboBox,
                       self.system_id_widget.overlapDoubleSpinBox,
                       self.system_id_widget.onFractionDoubleSpinBox,
                       self.system_id_widget.pretriggerDoubleSpinBox,
                       self.system_id_widget.rampFractionDoubleSpinBox,
                       self.system_id_widget.stream_transfer_function_data_checkbox,
                       self.system_id_widget.select_transfer_function_stream_file_button,
                       self.system_id_widget.transfer_function_stream_file_display,
                       self.system_id_widget.levelRampTimeDoubleSpinBox,
                       self.system_id_widget.save_system_id_matrices_button]:
            widget.setEnabled(True)
        for widget in [self.system_id_widget.stop_button]:
            widget.setEnabled(False)

class AbstractSysIdEnvironment(AbstractEnvironment):
    """Abstract Environment class defining the interface with the controller
    
    This class is used to define the operation of an environment within the
    controller, which must be completed by subclasses inheriting from this
    class.  Children of this class will sit in a While loop in the
    ``AbstractEnvironment.run()`` function.  While in this loop, the
    Environment will pull instructions and data from the
    ``command_queue`` and then use the ``command_map`` to map those instructions
    to functions in the class.
    
    All child classes inheriting from AbstractEnvironment will require functions
    to be defined for global operations of the controller, which are already
    mapped in the ``command_map``.  Any additional operations must be defined
    by functions and then added to the command_map when initilizing the child
    class.
    
    All functions called via the ``command_map`` must accept one input argument
    which is the data passed along with the command.  For functions that do not
    require additional data, this argument can be ignored, but it must still be
    present in the function's calling signature.
    
    The run function will continue until one of the functions called by
    ``command_map`` returns a truthy value, which signifies the controller to
    quit.  Therefore, any functions mapped to ``command_map`` that should not
    instruct the program to quit should not return any value that could be
    interpreted as true."""
    
    def __init__(self,
                 environment_name : str,
                 command_queue : VerboseMessageQueue,
                 gui_update_queue : Queue,
                 controller_communication_queue : VerboseMessageQueue,
                 log_file_queue : Queue,
                 collector_command_queue : VerboseMessageQueue,
                 signal_generator_command_queue : VerboseMessageQueue,
                 spectral_processing_command_queue : VerboseMessageQueue,
                 data_analysis_command_queue : VerboseMessageQueue,
                 data_in_queue : Queue,
                 data_out_queue : Queue, 
                 acquisition_active : mp.Value,
                 output_active : mp.Value):
        super().__init__(
                environment_name,
                command_queue,
                gui_update_queue,
                controller_communication_queue,
                log_file_queue,
                data_in_queue,
                data_out_queue,
                acquisition_active,
                output_active)
        self.map_command(SystemIdCommands.PREVIEW_NOISE,self.preview_noise)
        self.map_command(SystemIdCommands.PREVIEW_TRANSFER_FUNCTION,self.preview_transfer_function)
        self.map_command(SystemIdCommands.START_SYSTEM_ID,self.start_noise)
        self.map_command(SystemIdCommands.STOP_SYSTEM_ID,self.stop_system_id)
        self.map_command(SignalGenerationCommands.SHUTDOWN_ACHIEVED,self.siggen_shutdown_achieved_fn)
        self.map_command(DataCollectorCommands.SHUTDOWN_ACHIEVED,self.collector_shutdown_achieved_fn)
        self.map_command(SpectralProcessingCommands.SHUTDOWN_ACHIEVED,self.spectral_shutdown_achieved_fn)
        self.map_command(SysIDDataAnalysisCommands.SHUTDOWN_ACHIEVED,self.analysis_shutdown_achieved_fn)
        self.map_command(SysIDDataAnalysisCommands.START_SHUTDOWN,self.stop_system_id)
        self.map_command(SysIDDataAnalysisCommands.START_SHUTDOWN_AND_RUN_SYSID,self.start_shutdown_and_run_sysid)
        self.map_command(SysIDDataAnalysisCommands.SYSTEM_ID_COMPLETE,self.system_id_complete)
        self.map_command(SystemIdCommands.CHECK_FOR_COMPLETE_SHUTDOWN,self.check_for_sysid_shutdown)
        self._waiting_to_start_transfer_function = False
        self.collector_command_queue = collector_command_queue
        self.signal_generator_command_queue = signal_generator_command_queue
        self.spectral_processing_command_queue = spectral_processing_command_queue
        self.data_analysis_command_queue = data_analysis_command_queue
        self.data_acquisition_parameters = None
        self.environment_parameters = None
        self.collector_shutdown_achieved = True
        self.spectral_shutdown_achieved = True
        self.siggen_shutdown_achieved = True
        self.analysis_shutdown_achieved = True
        self._sysid_stream_name = None

    def initialize_data_acquisition_parameters(self,data_acquisition_parameters : DataAcquisitionParameters):
        """Initialize the data acquisition parameters in the environment.
        
        The environment will receive the global data acquisition parameters from
        the controller, and must set itself up accordingly.

        Parameters
        ----------
        data_acquisition_parameters : DataAcquisitionParameters :
            A container containing data acquisition parameters, including
            channels active in the environment as well as sampling parameters.
        """
        self.data_acquisition_parameters = data_acquisition_parameters
    
    def initialize_environment_test_parameters(self,environment_parameters : AbstractSysIdMetadata):
        """
        Initialize the environment parameters specific to this environment
        
        The environment will recieve parameters defining itself from the
        user interface and must set itself up accordingly.

        Parameters
        ----------
        environment_parameters : AbstractMetadata
            A container containing the parameters defining the environment

        """
        self.environment_parameters = environment_parameters
    
    def get_sysid_data_collector_metadata(self) -> CollectorMetadata:
        num_channels =  self.environment_parameters.number_of_channels
        response_channel_indices = self.environment_parameters.response_channel_indices
        reference_channel_indices = self.environment_parameters.reference_channel_indices
        if self.environment_parameters.sysid_signal_type in ['Random','Pseudorandom','Chirp']:
            acquisition_type = AcquisitionType.FREE_RUN
        else:
            acquisition_type = AcquisitionType.TRIGGER_FIRST_FRAME
        acceptance = Acceptance.AUTOMATIC
        acceptance_function = None
        if self.environment_parameters.sysid_signal_type == 'Random':
            overlap_fraction = self.environment_parameters.sysid_overlap
        else:
            overlap_fraction = 0
        if self.environment_parameters.sysid_signal_type == 'Burst Random':
            trigger_channel_index = reference_channel_indices[0]
        else:
            trigger_channel_index = 0
        trigger_slope = TriggerSlope.POSITIVE
        trigger_level = self.environment_parameters.sysid_level/100
        trigger_hysteresis = self.environment_parameters.sysid_level/200
        trigger_hysteresis_samples = ((1-self.environment_parameters.sysid_burst_on)
                                      *self.environment_parameters.sysid_frame_size)//2
        pretrigger_fraction = self.environment_parameters.sysid_pretrigger
        frame_size = self.environment_parameters.sysid_frame_size
        window = Window.HANN if self.environment_parameters.sysid_window == 'Hann' else Window.RECTANGLE
        
        return CollectorMetadata(
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
            response_transformation_matrix = self.environment_parameters.response_transformation_matrix,
            reference_transformation_matrix = self.environment_parameters.reference_transformation_matrix)
    
    def get_sysid_spectral_processing_metadata(self,is_noise=False) -> SpectralProcessingMetadata:
        averaging_type = AveragingTypes.LINEAR if self.environment_parameters.sysid_averaging_type == 'Linear' else AveragingTypes.EXPONENTIAL
        averages = self.environment_parameters.sysid_noise_averages if is_noise else self.environment_parameters.sysid_averages
        exponential_averaging_coefficient = self.environment_parameters.sysid_exponential_averaging_coefficient
        if self.environment_parameters.sysid_estimator == 'H1':
            frf_estimator = Estimator.H1
        elif self.environment_parameters.sysid_estimator == 'H2':
            frf_estimator = Estimator.H2
        elif self.environment_parameters.sysid_estimator == 'H3':
            frf_estimator = Estimator.H3
        elif self.environment_parameters.sysid_estimator == 'Hv':
            frf_estimator = Estimator.HV
        num_response_channels = self.environment_parameters.num_response_channels
        num_reference_channels = self.environment_parameters.num_reference_channels
        frequency_spacing = self.environment_parameters.sysid_frequency_spacing
        sample_rate = self.environment_parameters.sample_rate
        num_frequency_lines = self.environment_parameters.sysid_fft_lines
        return SpectralProcessingMetadata(
            averaging_type, 
            averages, 
            exponential_averaging_coefficient,
            frf_estimator, 
            num_response_channels, 
            num_reference_channels, 
            frequency_spacing, 
            sample_rate,
            num_frequency_lines)
    
    def get_sysid_signal_generation_metadata(self) -> SignalGenerationMetadata:
        return SignalGenerationMetadata(
            samples_per_write = self.data_acquisition_parameters.samples_per_write,
            level_ramp_samples = self.environment_parameters.sysid_level_ramp_time * self.environment_parameters.sample_rate,
            output_transformation_matrix = self.environment_parameters.reference_transformation_matrix,
            )
    
    def get_sysid_signal_generator(self) -> SignalGenerator:
        if self.environment_parameters.sysid_signal_type == 'Random':
            return RandomSignalGenerator(
                rms = self.environment_parameters.sysid_level,
                sample_rate = self.environment_parameters.sample_rate,
                num_samples_per_frame = self.environment_parameters.sysid_frame_size,
                num_signals = self.environment_parameters.num_reference_channels,
                low_frequency_cutoff = None,
                high_frequency_cutoff = None,
                cola_overlap = 0.5,
                cola_window = 'hann',
                cola_exponent = 0.5,
                output_oversample = self.data_acquisition_parameters.output_oversample)
        elif self.environment_parameters.sysid_signal_type == 'Pseudorandom':
            return PseudorandomSignalGenerator(
                rms = self.environment_parameters.sysid_level,
                sample_rate = self.environment_parameters.sample_rate,
                num_samples_per_frame = self.environment_parameters.sysid_frame_size,
                num_signals = self.environment_parameters.num_reference_channels,
                low_frequency_cutoff = None,
                high_frequency_cutoff = None,
                output_oversample = self.data_acquisition_parameters.output_oversample)
        elif self.environment_parameters.sysid_signal_type == 'Burst Random':
            return BurstRandomSignalGenerator(
                rms = self.environment_parameters.sysid_level,
                sample_rate = self.environment_parameters.sample_rate,
                num_samples_per_frame = self.environment_parameters.sysid_frame_size,
                num_signals = self.environment_parameters.num_reference_channels,
                low_frequency_cutoff = None,
                high_frequency_cutoff = None,
                on_fraction = self.environment_parameters.sysid_burst_on, 
                ramp_fraction = self.environment_parameters.sysid_burst_ramp_fraction, 
                output_oversample = self.data_acquisition_parameters.output_oversample)
        elif self.environment_parameters.sysid_signal_type == 'Chirp':
            return ChirpSignalGenerator(
                level = self.environment_parameters.sysid_level,
                sample_rate = self.environment_parameters.sample_rate,
                num_samples_per_frame = self.environment_parameters.sysid_frame_size,
                num_signals = self.environment_parameters.num_reference_channels,
                low_frequency_cutoff = self.environment_parameters.sysid_frequency_spacing,
                high_frequency_cutoff = self.environment_parameters.sample_rate/2,
                output_oversample = self.data_acquisition_parameters.output_oversample)
    
    def preview_noise(self,data):
        self.log('Starting Noise Preview')
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False
        self.analysis_shutdown_achieved = False
        self.environment_parameters = data
        # Start up controller
        self.controller_communication_queue.put(self.environment_name,(GlobalCommands.RUN_HARDWARE,None))
        self.controller_communication_queue.put(self.environment_name,(GlobalCommands.START_ENVIRONMENT,self.environment_name))
        
        # Set up the collector
        collector_metadata = deepcopy(self.get_sysid_data_collector_metadata())
        collector_metadata.acquisition_type = AcquisitionType.FREE_RUN
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.FORCE_INITIALIZE_COLLECTOR,
             collector_metadata))
        
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.SET_TEST_LEVEL,
             (self.environment_parameters.sysid_skip_frames,1)))
        time.sleep(0.01)
        
        # Set up the signal generation
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.INITIALIZE_PARAMETERS,
             self.get_sysid_signal_generation_metadata()))
        
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,
             self.get_sysid_signal_generator()))
        
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.MUTE,None))
        
        # Tell the collector to start acquiring data
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.ACQUIRE,None))
        
        # Tell the signal generation to start generating signals
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.GENERATE_SIGNALS,None))
        
        # Set up the data analysis
        self.data_analysis_command_queue.put(
            self.environment_name,
            (SysIDDataAnalysisCommands.INITIALIZE_PARAMETERS,self.environment_parameters))
        
        # Start the data analysis running
        self.data_analysis_command_queue.put(
            self.environment_name,
            (SysIDDataAnalysisCommands.RUN_NOISE,False))
        
        # Set up the spectral processing
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.INITIALIZE_PARAMETERS,
             self.get_sysid_spectral_processing_metadata(is_noise=True)))
        
        # Tell the spectral analysis to clear and start acquiring
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING,None))
        
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,None))
    
    def preview_transfer_function(self,data):
        self.log('Starting System ID Preview')
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False
        self.analysis_shutdown_achieved = False
        self.environment_parameters = data
        # Start up controller
        self.controller_communication_queue.put(self.environment_name,(GlobalCommands.RUN_HARDWARE,None))
        # Wait for the environment to start up
        while not (self.acquisition_active and self.output_active):
            # print('Waiting for Acquisition and Output to Start up') 
            time.sleep(0.1)
        self.controller_communication_queue.put(self.environment_name,(GlobalCommands.START_ENVIRONMENT,self.environment_name))
        
        # Set up the collector
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.FORCE_INITIALIZE_COLLECTOR,
             self.get_sysid_data_collector_metadata()))
        
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.SET_TEST_LEVEL,
             (self.environment_parameters.sysid_skip_frames,1)))
        time.sleep(0.01)
        
        # Set up the signal generation
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.INITIALIZE_PARAMETERS,
             self.get_sysid_signal_generation_metadata()))
        
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,
             self.get_sysid_signal_generator()))
        
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.MUTE,None))
        
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.ADJUST_TEST_LEVEL,1.0))
        
        # Tell the collector to start acquiring data
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.ACQUIRE,None))
        
        # Tell the signal generation to start generating signals
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.GENERATE_SIGNALS,None))
        
        # Set up the data analysis
        self.data_analysis_command_queue.put(
            self.environment_name,
            (SysIDDataAnalysisCommands.INITIALIZE_PARAMETERS,self.environment_parameters))
        
        # Start the data analysis running
        self.data_analysis_command_queue.put(
            self.environment_name,
            (SysIDDataAnalysisCommands.RUN_TRANSFER_FUNCTION,False))
        
        # Set up the spectral processing
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.INITIALIZE_PARAMETERS,
             self.get_sysid_spectral_processing_metadata(is_noise=False)))
        
        # Tell the spectral analysis to clear and start acquiring
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING,None))
        
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,None))
    
    def start_noise(self,data):
        self.log('Starting Noise Measurement for System ID')
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False
        self.analysis_shutdown_achieved = False
        self.environment_parameters,self._sysid_stream_name = data
        self.controller_communication_queue.put(self.environment_name,(GlobalCommands.UPDATE_METADATA,(self.environment_name,self.environment_parameters)))
        # Start up controller
        if self._sysid_stream_name is not None:
            name,ext = os.path.splitext(self._sysid_stream_name)
            self.controller_communication_queue.put(self.environment_name,(GlobalCommands.INITIALIZE_STREAMING,name+'_noise'+ext))
            self.controller_communication_queue.put(self.environment_name,(GlobalCommands.START_STREAMING,None))
        self.controller_communication_queue.put(self.environment_name,(GlobalCommands.RUN_HARDWARE,None))
        self.controller_communication_queue.put(self.environment_name,(GlobalCommands.START_ENVIRONMENT,self.environment_name))
        
        # Set up the collector
        collector_metadata = deepcopy(self.get_sysid_data_collector_metadata())
        collector_metadata.acquisition_type = AcquisitionType.FREE_RUN
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.FORCE_INITIALIZE_COLLECTOR,
             collector_metadata))
        
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.SET_TEST_LEVEL,
             (self.environment_parameters.sysid_skip_frames,1)))
        time.sleep(0.01)
        
        # Set up the signal generation
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.INITIALIZE_PARAMETERS,
             self.get_sysid_signal_generation_metadata()))
        
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,
             self.get_sysid_signal_generator()))
        
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.MUTE,None))
        
        # Tell the collector to start acquiring data
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.ACQUIRE,None))
        
        # Tell the signal generation to start generating signals
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.GENERATE_SIGNALS,None))
        
        # Set up the data analysis
        self.data_analysis_command_queue.put(
            self.environment_name,
            (SysIDDataAnalysisCommands.INITIALIZE_PARAMETERS,self.environment_parameters))
        
        # Start the data analysis running
        self.data_analysis_command_queue.put(
            self.environment_name,
            (SysIDDataAnalysisCommands.RUN_NOISE,True))
        
        # Set up the spectral processing
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.INITIALIZE_PARAMETERS,
             self.get_sysid_spectral_processing_metadata(is_noise=True)))
        
        # Tell the spectral analysis to clear and start acquiring
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING,None))
        
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,None))
    
    def start_transfer_function(self,data):
        self.log('Starting Transfer Function for System ID')
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False
        self.analysis_shutdown_achieved = False
        self.environment_parameters = data
        # Start up controller
        if self._sysid_stream_name is not None:
            name,ext = os.path.splitext(self._sysid_stream_name)
            self.controller_communication_queue.put(self.environment_name,(GlobalCommands.INITIALIZE_STREAMING,name+'_tf'+ext))
            self.controller_communication_queue.put(self.environment_name,(GlobalCommands.START_STREAMING,None))
        self.controller_communication_queue.put(self.environment_name,(GlobalCommands.RUN_HARDWARE,None))
        # Wait for the environment to start up
        while not (self.acquisition_active and self.output_active):
            # print('Waiting for Acquisition and Output to Start up') 
            time.sleep(0.1)
        self.controller_communication_queue.put(self.environment_name,(GlobalCommands.START_ENVIRONMENT,self.environment_name))
        
        # Set up the collector
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.FORCE_INITIALIZE_COLLECTOR,
             self.get_sysid_data_collector_metadata()))
        
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.SET_TEST_LEVEL,
             (self.environment_parameters.sysid_skip_frames,1)))
        time.sleep(0.01)
        
        # Set up the signal generation
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.INITIALIZE_PARAMETERS,
             self.get_sysid_signal_generation_metadata()))
        
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,
             self.get_sysid_signal_generator()))
        
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.MUTE,None))
        
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.ADJUST_TEST_LEVEL,1.0))
        
        # Tell the collector to start acquiring data
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.ACQUIRE,None))
        
        # Tell the signal generation to start generating signals
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.GENERATE_SIGNALS,None))
        
        # Set up the data analysis
        self.data_analysis_command_queue.put(
            self.environment_name,
            (SysIDDataAnalysisCommands.INITIALIZE_PARAMETERS,self.environment_parameters))
        
        # Start the data analysis running
        self.data_analysis_command_queue.put(
            self.environment_name,
            (SysIDDataAnalysisCommands.RUN_TRANSFER_FUNCTION,True))
        
        # Set up the spectral processing
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.INITIALIZE_PARAMETERS,
             self.get_sysid_spectral_processing_metadata(is_noise=False)))
        
        # Tell the spectral analysis to clear and start acquiring
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING,None))
        
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,None))
        
    def stop_system_id(self,stop_data_analysis):
        self.log('Stop Transfer Function')
        self.controller_communication_queue.put(self.environment_name,(GlobalCommands.STOP_HARDWARE,None))
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.SET_TEST_LEVEL,
             (self.environment_parameters.sysid_skip_frames*10,1)))
        self.signal_generator_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.START_SHUTDOWN,None))
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.STOP_SPECTRAL_PROCESSING,None))
        if stop_data_analysis:
            self.data_analysis_command_queue.put(
                self.environment_name,
                (SysIDDataAnalysisCommands.STOP_SYSTEM_ID,None))
        self.environment_command_queue.put(
            self.environment_name,
            (SystemIdCommands.CHECK_FOR_COMPLETE_SHUTDOWN,None))
        
    def siggen_shutdown_achieved_fn(self,data):
        self.siggen_shutdown_achieved = True
        
    def collector_shutdown_achieved_fn(self,data):
        self.collector_shutdown_achieved = True
        
    def spectral_shutdown_achieved_fn(self,data):
        self.spectral_shutdown_achieved = True
        
    def analysis_shutdown_achieved_fn(self,data):
        self.analysis_shutdown_achieved = True
        
    def check_for_sysid_shutdown(self,data):
        if (self.siggen_shutdown_achieved and self.collector_shutdown_achieved
            and self.spectral_shutdown_achieved and
            self.analysis_shutdown_achieved and not self.acquisition_active
            and not self.output_active):
            self.log('Shutdown Achieved')
            if self._waiting_to_start_transfer_function:
                self.start_transfer_function(self.environment_parameters)
            else:
                self.gui_update_queue.put((self.environment_name,('enable_system_id',None)))
                self._sysid_stream_name = None
            self._waiting_to_start_transfer_function = False
        else:
            # Recheck some time later
            time.sleep(1)
            waiting_for = []
            if not self.siggen_shutdown_achieved:
                waiting_for.append('Signal Generation')
            if not self.collector_shutdown_achieved:
                waiting_for.append('Collector')
            if not self.spectral_shutdown_achieved:
                waiting_for.append('Spectral Processing')
            if not self.analysis_shutdown_achieved:
                waiting_for.append('Data Analysis')
            if self.output_active:
                waiting_for.append('Output Shutdown')
            if self.acquisition_active:
                waiting_for.append('Acquisition Shutdown')
            self.log('Waiting for {:}'.format(' and '.join(waiting_for)))
            self.environment_command_queue.put(
                self.environment_name,
                (SystemIdCommands.CHECK_FOR_COMPLETE_SHUTDOWN,None))
    
    def start_shutdown_and_run_sysid(self,data):
        self.log('Shutting down and then Running System ID Afterwards')
        self._waiting_to_start_transfer_function = True
        self.stop_system_id(False)
    
    def system_id_complete(self,data):
        self.log('Finished System Identification')
        self.controller_communication_queue.put(
            self.environment_name,
            (GlobalCommands.COMPLETED_SYSTEM_ID,self.environment_name))
    
    @abstractmethod
    def stop_environment(self,data):
        """Stop the environment gracefully
        
        This function defines the operations to shut down the environment
        gracefully so there is no hard stop that might damage test equipment
        or parts.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        pass
    
    def quit(self,data):
        for queue in [self.queue_container.spectral_command_queue,
                       self.queue_container.data_analysis_command_queue,
                      self.queue_container.signal_generation_command_queue,
                      self.queue_container.collector_command_queue]:
            queue.put(self.environment_name,(GlobalCommands.QUIT,None))
        # Return true to stop the task
        return True