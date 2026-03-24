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

import multiprocessing as mp
import multiprocessing.sharedctypes  # pylint: disable=unused-import
import os
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from multiprocessing.queues import Queue
import netCDF4 as nc4
from rattlesnake.utilities import (
    DataAcquisitionParameters,
    GlobalCommands,
    VerboseMessageQueue,
)
from rattlesnake.user_interface.ui_utilities import UICommands

PICKLE_ON_ERROR = False

if PICKLE_ON_ERROR:
    import pickle


# region Metadata
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
    def store_to_netcdf(
        self, netcdf_group_handle: nc4._netCDF4.Group
    ):  # pylint: disable=c-extension-no-member
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


# endregion


# region Environment
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

    def dump_to_dict(self):
        """Dumps the environment to a dictionary to be pickled if an error occurs"""
        if PICKLE_ON_ERROR:
            state = self.__dict__.copy()
            for key, value in state.items():
                try:
                    pickle.dumps(value)
                    print(f"{key} is pickleable")
                except Exception:  # pylint: disable=broad-exception-caught
                    print(f"{key} is not pickleable")
                    state[key] = None
            return state
        else:
            return self.__dict__.copy()

    def __init__(
        self,
        environment_name: str,
        command_queue: VerboseMessageQueue,
        gui_update_queue: Queue,
        controller_communication_queue: VerboseMessageQueue,
        log_file_queue: Queue,
        data_in_queue: Queue,
        data_out_queue: Queue,
        acquisition_active: mp.sharedctypes.Synchronized,
        output_active: mp.sharedctypes.Synchronized,
    ):
        self._environment_name = environment_name
        self._command_queue = command_queue
        self._gui_update_queue = gui_update_queue
        self._controller_communication_queue = controller_communication_queue
        self._log_file_queue = log_file_queue
        self._data_in_queue = data_in_queue
        self._data_out_queue = data_out_queue
        self._command_map = {
            GlobalCommands.QUIT: self.quit,
            GlobalCommands.INITIALIZE_DATA_ACQUISITION: self.initialize_data_acquisition_parameters,
            GlobalCommands.INITIALIZE_ENVIRONMENT_PARAMETERS: self.initialize_environment_test_parameters,
            GlobalCommands.STOP_ENVIRONMENT: self.stop_environment,
        }
        self._acquisition_active = acquisition_active
        self._output_active = output_active

    @property
    def acquisition_active(self):
        """Flag to check if acquisition is active"""
        # print('Checking if Acquisition Active: {:}'.format(bool(self._acquisition_active.value)))
        return bool(self._acquisition_active.value)

    @property
    def output_active(self):
        """Flag to check if output is active"""
        # print('Checking if Output Active: {:}'.format(bool(self._output_active.value)))
        return bool(self._output_active.value)

    @abstractmethod
    def initialize_data_acquisition_parameters(
        self, data_acquisition_parameters: DataAcquisitionParameters
    ):
        """Initialize the data acquisition parameters in the environment.

        The environment will receive the global data acquisition parameters from
        the controller, and must set itself up accordingly.

        Parameters
        ----------
        data_acquisition_parameters : DataAcquisitionParameters :
            A container containing data acquisition parameters, including
            channels active in the environment as well as sampling parameters.
        """

    @abstractmethod
    def initialize_environment_test_parameters(self, environment_parameters: AbstractMetadata):
        """
        Initialize the environment parameters specific to this environment

        The environment will recieve parameters defining itself from the
        user interface and must set itself up accordingly.

        Parameters
        ----------
        environment_parameters : AbstractMetadata
            A container containing the parameters defining the environment

        """

    @abstractmethod
    def stop_environment(self, data):
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

    def log(self, message: str):
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
        self.log_file_queue.put(f"{datetime.now()}: {self.environment_name} -- {message}\n")

    @property
    def environment_name(self) -> str:
        """A string defining the name of the environment"""
        return self._environment_name

    @property
    def command_map(self) -> dict:
        """A dictionary that maps commands received by the ``command_queue`` to functions in the class"""
        return self._command_map

    def map_command(self, key, function):
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
        self.log(f"Starting Process with PID {os.getpid()}")
        while True:
            # Get the message from the queue
            message, data = self.environment_command_queue.get(self.environment_name)
            # Call the function corresponding to that message with the data as argument
            try:
                function = self.command_map[message]
            except KeyError:
                self.log(
                    f"Undefined Message {message}, acceptable messages are {[key for key in self.command_map]}"
                )
                continue
            try:
                halt_flag = function(data)
            except Exception:  # pylint: disable=broad-exception-caught
                tb = traceback.format_exc()
                self.log(f"ERROR\n\n {tb}")
                self.gui_update_queue.put(
                    (
                        UICommands.ERROR,
                        (
                            f"{self.environment_name} Error",
                            f"!!!UNKNOWN ERROR!!!\n\n{tb}",
                        ),
                    )
                )
                if PICKLE_ON_ERROR:
                    with open(
                        f"debug_data/{self.environment_name}_error_state.txt", "w", encoding="utf-8"
                    ) as f:
                        f.write(f"{tb}")
                    with open(f"debug_data/{self.environment_name}_error_state.pkl", "wb") as f:
                        dic = self.dump_to_dict()
                        pickle.dump(dic, f)
                    print("Done Writing Pickle File from Error...")
                halt_flag = False
            # If we get a true value, stop.
            if halt_flag:
                self.log("Stopping Process")
                break

    def quit(self, data):  # pylint: disable=unused-argument
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


# region: Process
def run_process(
    environment_name: str,
    input_queue: VerboseMessageQueue,
    gui_update_queue: Queue,
    controller_communication_queue: VerboseMessageQueue,
    log_file_queue: Queue,
    data_in_queue: Queue,
    data_out_queue: Queue,
    acquisition_active: mp.sharedctypes.Synchronized,
    output_active: mp.sharedctypes.Synchronized,
):
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
    process_class = AbstractEnvironment(  # pylint: disable=abstract-class-instantiated
        environment_name,
        input_queue,
        gui_update_queue,
        controller_communication_queue,
        log_file_queue,
        data_in_queue,
        data_out_queue,
        acquisition_active,
        output_active,
    )
    process_class.run()
