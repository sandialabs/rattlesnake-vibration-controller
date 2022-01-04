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

from PyQt5 import QtWidgets,uic
from abc import abstractmethod
from .utilities import Channel,VerboseMessageQueue,GlobalCommands,DataAcquisitionParameters,OverlapBuffer,db2scale
from .abstract_environment import AbstractMetadata,AbstractEnvironment,AbstractUI
from .generic_frf_computation import FRFParameters,FRFCommands,frf_computation_process
from .generic_system_id_signal_generation import SysIdParameters,SysIdSignalGenerationCommands,system_id_signal_generation_process
from .environments import system_identification_ui_path
from typing import List
from multiprocessing.queues import Queue
from datetime import datetime
import traceback
import os
import netCDF4 as nc4
import openpyxl
import numpy as np

MAXIMUM_NAME_LENGTH = 50

class SysIdCommands:
    START_TRANSFER_FUNCTION = 0
    STOP_TRANSFER_FUNCTION = 1
    SHOW_TEST_PREDICTION = 2
    ADJUST_TEST_LEVEL = 3
    PERFORM_CONTROL_PREDICTION = 4
    SHOW_FRF = 5

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
    
    @abstractmethod
    def get_system_id_parameters(self, data_acquisition_parameters : DataAcquisitionParameters
                                 ) -> SysIdParameters:
        """
        Extracts system identification parameters from the metadata object
        
        Parameters
        ----------
        data_acquisition_parameters : DataAcquisitionParameters
            Global Data Acquisition Parameters

        Returns
        -------
        SysIdParameters
            Object containing parameters required for the system identification
            signal generation subprocess
        """
        pass
    
    @abstractmethod
    def get_frf_parameters(self, data_acquisition_parameters : DataAcquisitionParameters
                                 ) -> FRFParameters:
        """
        Extracts FRF parameters from the metadata object
        
        Parameters
        ----------
        data_acquisition_parameters : DataAcquisitionParameters
            Global Data Acquisition Parameters
            
        Returns
        -------
        FRFParameters
            Object containing parameters required for the frf computation
            subprocess

        """
        pass

class AbstractSysIdUI(AbstractUI):
    """Abstract User Interface class defining the interface with the controller
    
    This class is used to define the interface between the User Interface of a
    environment in the controller and the main controller."""
    
    @abstractmethod
    def __init__(self,
                 environment_name : str,
                 environment_command_queue : VerboseMessageQueue,
                 controller_communication_queue : VerboseMessageQueue,
                 log_file_queue : Queue,
                 system_id_tabwidget : QtWidgets.QTabWidget):
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
        system_id_tabwidget : QtWidgets.QTabWidget
            QTabWidget containing the environment subtabs on the System
            Identification main tab.
        """
        super().__init__(environment_name, environment_command_queue,
                         controller_communication_queue, log_file_queue)
        # Add the page to the system id tabwidget
        self.system_id_widget = QtWidgets.QWidget()
        uic.loadUi(system_identification_ui_path,self.system_id_widget)
        system_id_tabwidget.addTab(self.system_id_widget,self.environment_name)
        
        # Initialize persistent attributes
        self.data_acquisition_parameters = None
        self.environment_parameters = None
        self.frf_parameters = None
        self.sysid_parameters = None
        self.physical_control_names = None
        self.physical_output_names = None
        self.plot_data_items = {}
        
        # Set common look and feel for plots
        plotWidgets = [self.system_id_widget.response_timehistory_plot,
                       self.system_id_widget.drive_timehistory_plot,
                       self.system_id_widget.transfer_function_phase_plot,
                       self.system_id_widget.transfer_function_amplitude_plot]
        for plotWidget in plotWidgets:
            plot_item = plotWidget.getPlotItem()
            plot_item.showGrid(True,True,0.25)
            plot_item.enableAutoRange()
            plot_item.getViewBox().enableAutoRange(enable=True)
        logscalePlotWidgets = [self.system_id_widget.transfer_function_amplitude_plot]
        for plotWidget in logscalePlotWidgets:
            plot_item = plotWidget.getPlotItem()
            plot_item.setLogMode(False,True)
        
        # Connect callbacks
        self.system_id_widget.preview_transfer_function_button.clicked.connect(self.preview_transfer_function)
        self.system_id_widget.acquire_transfer_function_button.clicked.connect(self.acquire_transfer_function)
        self.system_id_widget.stop_transfer_function_button.clicked.connect(self.stop_transfer_function)
        self.system_id_widget.select_transfer_function_stream_file_button.clicked.connect(self.select_transfer_function_stream_file)
        self.system_id_widget.voltage_scale_factor_selector.valueChanged.connect(self.change_transfer_function_test_level)
        self.system_id_widget.transfer_function_response_selector.currentIndexChanged.connect(self.show_frf)
        self.system_id_widget.transfer_function_reference_selector.currentIndexChanged.connect(self.show_frf)
    
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
        self.log('Initializing Data Acquisition Parameters')
        # Get channel information
        self.data_acquisition_parameters = data_acquisition_parameters
        channels = data_acquisition_parameters.channel_list
        self.physical_control_names
        self.physical_control_names = ['{:} {:} {:}'.format('' if channel.channel_type is None else channel.channel_type,channel.node_number,channel.node_direction)[:MAXIMUM_NAME_LENGTH]
            for channel in channels if channel.control]
        self.physical_output_names = ['{:} {:} {:}'.format('' if channel.channel_type is None else channel.channel_type,channel.node_number,channel.node_direction)[:MAXIMUM_NAME_LENGTH]
            for channel in channels if channel.feedback_device]
    
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
        self.log('Initializing Environment Parameters')
        self.environment_parameters = self.collect_environment_definition_parameters()
        self.sysid_parameters : SysIdParameters = self.environment_parameters.get_system_id_parameters(self.data_acquisition_parameters)
        self.frf_parameters : FRFParameters = self.environment_parameters.get_frf_parameters(self.data_acquisition_parameters)
        if self.sysid_parameters.response_transformation_matrix is None:
            for i in range(self.sysid_parameters.response_transformation_matrix.shape[0]):
                widget.addItem('{:}: {:}'.format(i+1,'Virtual Response'))
            else:

    ### System Identification Callbacks
    def preview_transfer_function(self):
        """Starts the transfer function in preview mode"""
        self.log('Starting Transfer Function Preview')
        self.system_id_widget.preview_transfer_function_button.setEnabled(False)
        self.system_id_widget.acquire_transfer_function_button.setEnabled(False)
        self.system_id_widget.stop_transfer_function_button.setEnabled(True)
        self.controller_communication_queue.put(self.log_name,(GlobalCommands.RUN_HARDWARE,None))
        self.controller_communication_queue.put(self.log_name,(GlobalCommands.START_ENVIRONMENT,self.environment_name))
        self.environment_command_queue.put(self.log_name,(SysIdCommands.START_TRANSFER_FUNCTION,
                      (False,db2scale(self.system_id_widget.voltage_scale_factor_selector.value()))))
    
    def acquire_transfer_function(self):
        """Starts the transfer function in acquire mode"""
        self.log('Starting Transfer Function Acquire')
        self.system_id_widget.preview_transfer_function_button.setEnabled(False)
        self.system_id_widget.acquire_transfer_function_button.setEnabled(False)
        self.system_id_widget.stop_transfer_function_button.setEnabled(True)
        if self.system_id_widget.stream_transfer_function_data_checkbox.isChecked():
            self.controller_communication_queue.put(self.log_name,(GlobalCommands.INITIALIZE_STREAMING,self.system_id_widget.transfer_function_stream_file_display.text()))
            self.controller_communication_queue.put(self.log_name,(GlobalCommands.START_STREAMING,None))
        self.controller_communication_queue.put(self.log_name,(GlobalCommands.RUN_HARDWARE,None))
        self.controller_communication_queue.put(self.log_name,(GlobalCommands.START_ENVIRONMENT,self.environment_name))
        self.environment_command_queue.put(self.log_name,(SysIdCommands.START_TRANSFER_FUNCTION,
                      (True,db2scale(self.system_id_widget.voltage_scale_factor_selector.value()))))
        
    def stop_transfer_function(self):
        """Stops the transfer function acquisition"""
        self.log('Stopping Transfer Function')
        self.environment_command_queue.put(self.log_name,(SysIdCommands.STOP_TRANSFER_FUNCTION,None))
    
    def select_transfer_function_stream_file(self):
        """Select a file to save transfer function data to"""
        filename,file_filter = QtWidgets.QFileDialog.getSaveFileName(self.system_id_widget,'Select NetCDF File to Save Transfer Function Data',filter='NetCDF File (*.nc4)')
        if filename == '':
            return
        self.system_id_widget.transfer_function_stream_file_display.setText(filename)
        self.system_id_widget.stream_transfer_function_data_checkbox.setChecked(True)
    
    def change_transfer_function_test_level(self):
        """Updates the test level for the transfer function"""
        self.environment_command_queue.put(self.log_name,(SysIdCommands.ADJUST_TEST_LEVEL,db2scale(self.system_id_widget.voltage_scale_factor_selector.value())))
    
    def show_frf(self):
        """Tells the environment to show the FRF"""
        self.environment_command_queue.put(self.log_name,(SysIdCommands.SHOW_FRF,None))

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
            
        Raises
        ------
        ValueError :
            If the command or widget is not found.
            
        Notes
        -----
        In the subclass's update_gui function, first call super().update_gui
        within a try-except block.  If no ValueError is raised, then the update
        was successfully handle by the superclass.  If ValueError was not
        raised, the update needs to be handled by the subclass.
        """
        message,data = queue_data
        if message == 'sysid_time_data':
            # Display the data
            control_data,output_data = data
            for curve,this_data in zip(self.plot_data_items['control_responses'],control_data):
                x,y = curve.getData()
                y = np.concatenate((y[this_data.size:],this_data[-x.size:]),axis=0)
                curve.setData(x,y)
            # Display the data
            for curve,this_output in zip(self.plot_data_items['drive_outputs'],output_data):
                x,y = curve.getData()
                y = np.concatenate((y[this_output.size:],this_output[-x.size:]),axis=0)
                curve.setData(x,y)
        elif message == 'FRF':
            # Display the data
            self.plot_data_items['transfer_function_phase'].setData(data[0],
                                np.angle(data[1][:,self.system_id_widget.transfer_function_response_selector.currentIndex(),
                                                   self.system_id_widget.transfer_function_reference_selector.currentIndex()]))
            self.plot_data_items['transfer_function_amplitude'].setData(data[0],
                                np.abs(data[1][:,self.system_id_widget.transfer_function_response_selector.currentIndex(),
                                                 self.system_id_widget.transfer_function_reference_selector.currentIndex()]))
        elif message == 'enable':
            try:
                widget = getattr(self.system_id_widget,data)
            except AttributeError:
                raise ValueError('Cannot Enable Widget {:}: not found in UI'.format(data))
            widget.setEnabled(True)
        elif message == 'disable':
            try:
                widget = getattr(self.system_id_widget,data)
            except AttributeError:
                raise ValueError('Cannot Disable Widget {:}: not found in UI'.format(data))
            widget.setEnabled(False)
        else:
            try:
                widget = getattr(self.system_id_widget,message)
            except AttributeError:
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
                 data_in_queue : Queue,
                 data_out_queue : Queue):
        self._environment_name = environment_name
        self._command_queue = command_queue
        self._gui_update_queue = gui_update_queue
        self._controller_communication_queue = controller_communication_queue
        self._log_file_queue = log_file_queue
        self._data_in_queue = data_in_queue
        self._data_out_queue = data_out_queue
        self._command_map = {GlobalCommands.QUIT:self.quit,
                             GlobalCommands.INITIALIZE_DATA_ACQUISITION:self.initialize_data_acquisition_parameters,
                             GlobalCommands.INITIALIZE_ENVIRONMENT_PARAMETERS:self.initialize_environment_test_parameters,
                             GlobalCommands.STOP_ENVIRONMENT:self.stop_environment}

    @abstractmethod
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
        pass
    
    @abstractmethod
    def initialize_environment_test_parameters(self,environment_parameters : AbstractMetadata):
        """
        Initialize the environment parameters specific to this environment
        
        The environment will recieve parameters defining itself from the
        user interface and must set itself up accordingly.

        Parameters
        ----------
        environment_parameters : AbstractMetadata
            A container containing the parameters defining the environment

        """
        
        pass
    
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
    
    @property
    def environment_command_queue(self) -> VerboseMessageQueue:
        """The queue that provides commands to the environment."""
        return self._command_queue
    
    @property
    def data_in_queue(self) -> Queue:
        """The queue from which data is delivered to the environment"""
        return self._data_in_queue
    
    @property
    def data_out_queue(self) -> Queue:
        """The queue to which data is written that will be output to exciters"""
        return self._data_out_queue
    
    @property
    def gui_update_queue(self) -> Queue:
        """The queue that GUI update instructions are written to"""
        return self._gui_update_queue
    
    @property
    def controller_communication_queue(self) -> Queue:
        """The queue that global controller updates are written to"""
        return self._controller_communication_queue
    
    @property
    def log_file_queue(self) -> Queue:
        """The queue that log file messages are written to"""
        return self._log_file_queue
    
    def log(self,message : str):
        """Write a message to the log file
        
        This function puts a message onto the ``log_file_queue`` so it will
        eventually be written to the log file.
        
        When written to the log file, the message will include the date and
        time that the message was queued, the name of the environment, and
        then the message itself.

        Parameters
        ----------
        message : str :
            A message that will be written to the log file.
        """
        self.log_file_queue.put('{:}: {:} -- {:}\n'.format(datetime.now(),self.environment_name,message))
    
    @property
    def environment_name(self) -> str:
        """A string defining the name of the environment"""
        return self._environment_name
    
    @property
    def command_map(self) -> dict:
        """A dictionary that maps commands received by the ``command_queue`` to functions in the class"""
        return self._command_map
    
    def map_command(self,key,function):
        """A function that maps an instruction to a function in the ``command_map``

        Parameters
        ----------
        key :
            The instruction that will be pulled from the ``command_queue``
            
        function :
            A reference to the function that will be called when the ``key``
            message is received.

        """
        self._command_map[key] = function

    def run(self):
        """The main function that is run by the environment's process
        
        A function that is called by the environment's process function that
        sits in a while loop waiting for instructions on the command queue.
        
        When the instructions are recieved, they are separated into
        ``(message,data)`` pairs.  The ``message`` is used in conjuction with
        the ``command_map`` to identify which function should be called, and
        the ``data`` is passed to that function as the argument.  If the
        function returns a truthy value, it signals to the ``run`` function
        that it is time to stop the loop and exit.


        """
        self.log('Starting Process with PID {:}'.format(os.getpid()))
        while True:
            # Get the message from the queue
            message,data = self.environment_command_queue.get(self.environment_name)
            # Call the function corresponding to that message with the data as argument
            try:
                function = self.command_map[message]
            except KeyError:
                self.log('Undefined Message {:}, acceptable messages are {:}'.format(message,[key for key in self.command_map]))
                continue
            try:
                halt_flag = function(data)
            except Exception:
                tb = traceback.format_exc()
                self.log('ERROR\n\n {:}'.format(tb))
                self.gui_update_queue.put(('error',('{:} Error'.format(self.environment_name),'!!!UNKNOWN ERROR!!!\n\n{:}'.format(tb))))
                halt_flag = False
            # If we get a true value, stop.
            if halt_flag:
                self.log('Stopping Process')
                break

    def quit(self,data):
        """Returns True to stop the ``run`` while loop and exit the process

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        Returns
        -------
        True :
            This function returns True to signal to the ``run`` while loop
            that it is time to close down the environment.

        """
        return True
    
def run_process(environment_name : str,
                input_queue : VerboseMessageQueue,
                gui_update_queue : Queue,
                controller_communication_queue : VerboseMessageQueue,
                log_file_queue : Queue,
                data_in_queue : Queue,
                data_out_queue : Queue):
    """A function called by ``multiprocessing.Process`` to start the environment
    
    This function should not be called directly, but used as a template for
    other environments to start up.

    Parameters
    ----------
    environment_name : str :
        The name of the environment
        
    input_queue : VerboseMessageQueue :
        The command queue for the environment
        
    gui_update_queue : Queue :
        The queue that accepts GUI update ``(message,data)`` pairs.
        
    controller_communication_queue : VerboseMessageQueue :
        The queue where global instructions to the controller can be written
        
    log_file_queue : Queue :
        The queue where logging messages can be written
        
    data_in_queue : Queue :
        The queue from which the environment will receive data from the
        acquisition hardware
        
    data_out_queue : Queue :
        The queue to which the environment should write data so it will be output
        to the excitation devices in the output hardware
        
    """
    process_class = AbstractEnvironment(
            environment_name,
            input_queue,
            gui_update_queue,
            controller_communication_queue,
            log_file_queue,
            data_in_queue,
            data_out_queue)
    process_class.run()