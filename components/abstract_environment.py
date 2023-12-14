# -*- coding: utf-8 -*-
"""
Abstract environment that can be used to create new environment control strategies
in the controller.

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

from qtpy import QtWidgets
from abc import ABC,abstractmethod
from .utilities import Channel,VerboseMessageQueue,GlobalCommands,DataAcquisitionParameters
from typing import List
from multiprocessing.queues import Queue
import multiprocessing as mp
from datetime import datetime
import traceback
import os
import netCDF4 as nc4
import openpyxl

class AbstractMetadata(ABC):
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
        pass

class AbstractUI(ABC):
    """Abstract User Interface class defining the interface with the controller
    
    This class is used to define the interface between the User Interface of a
    environment in the controller and the main controller."""
    
    @abstractmethod
    def __init__(self,
                 environment_name : str,
                 environment_command_queue : VerboseMessageQueue,
                 controller_communication_queue : VerboseMessageQueue,
                 log_file_queue : Queue):
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
        self._environment_name = environment_name
        self._log_name = environment_name + ' UI'
        self._log_file_queue = log_file_queue
        self._environment_command_queue = environment_command_queue
        self._controller_communication_queue = controller_communication_queue
        self._command_map = {'Start Control':self.start_control,
                             'Stop Control':self.stop_control}
        
    @property
    def command_map(self) -> dict:
        """Dictionary mapping profile instructions to functions of the UI that
        are called when the instruction is executed."""
        return self._command_map
    
    @abstractmethod
    def start_control(self):
        """Runs the corresponding environment in the controller"""
        pass
    
    @abstractmethod
    def stop_control(self):
        """Stops the corresponding environment in the controller"""
        pass
    
    @abstractmethod
    def collect_environment_definition_parameters(self) -> AbstractMetadata:
        """
        Collect the parameters from the user interface defining the environment

        Returns
        -------
        AbstractMetadata
            A metadata or parameters object containing the parameters defining
            the corresponding environment.

        """
        pass
    
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
        pass
    
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
        pass
    
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
        pass
    
    @property
    def log_file_queue(self) -> Queue:
        """A property containing a reference to the queue accepting messages
        that will be written to the log file"""
        return self._log_file_queue
    
    @property
    def environment_command_queue(self) -> VerboseMessageQueue:
        """A property containing a reference to the queue accepting commands
        that will be delivered to the environment"""
        return self._environment_command_queue
    
    @property
    def controller_communication_queue(self) -> VerboseMessageQueue:
        """A property containing a reference to the queue accepting global
        commands that will be delivered to the controller"""
        return self._controller_communication_queue
    
    @property
    def environment_name(self):
        """A property containing the environment's name"""
        return self._environment_name
    
    @property
    def log_name(self):
        """A property containing the name that the UI will be referenced by in
        the log file, which will typically be ``self.environment_name + ' UI'``"""
        return self._log_name
    
    def log(self,message : str):
        """Write a message to the log file
        
        This function puts a message onto the ``log_file_queue`` so it will
        eventually be written to the log file.
        
        When written to the log file, the message will include the date and
        time that the message was queued, the name that the UI uses in the log
        file (``self.log_file``), and then the message itself.

        Parameters
        ----------
        message : str :
            A message that will be written to the log file.

        """
        self.log_file_queue.put('{:}: {:} -- {:}\n'.format(datetime.now(),self.log_name,message))
        
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

class AbstractEnvironment(ABC):
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
                 data_out_queue : Queue,
                 acquisition_active : mp.Value,
                 output_active : mp.Value):
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
        self._acquisition_active = acquisition_active
        self._output_active = output_active

    @property
    def acquisition_active(self):
        # print('Checking if Acquisition Active: {:}'.format(bool(self._acquisition_active.value)))
        return bool(self._acquisition_active.value)

    @property
    def output_active(self):
        # print('Checking if Output Active: {:}'.format(bool(self._output_active.value)))
        return bool(self._output_active.value)

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
                data_out_queue : Queue,
                acquisition_active : mp.Value,
                output_active : mp.Value):
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
            data_out_queue,
            acquisition_active,
            output_active)
    process_class.run()