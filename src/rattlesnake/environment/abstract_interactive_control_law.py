# -*- coding: utf-8 -*-
"""
This is an abstract interface to that defines an interactive control law in
Rattlesnake that can have it's own graphical user interface and interactive
elements in the control law.  This control law will need to interface with the
environment in order to send parameters and receive data.  It will also need to
handle building up a graphical user interface.

The general workflow for this function should be:
    1. The control law is selected in the environment user interface with a new
       type associated with it: "Interactive".
    2. The control law must define four classes.
        a. The UI class will handle specification of parameters
           and/or visualization of results.  Minor calculations may also be
           performed in the UI class, accepting that the UI will lock up while the
           calculations are performed unless sent to a separate process.
        b. The control calculation class will handle the major calculations
           involved with computing the next output dataset.
        c. The control law must define a "parameters" class that contains
           parameters that the control class will need.  These can be things
           like regularization parameters, weighting parameters, transformation
           matrices; whatever the control law needs to perform its calculation.
        d. The control law must define a "results" class that contains
           information that needs to be passed from the control back to the
           user interface.
    2. The control law will need functions to handle the passing of the data
       objects between UI and calculation class.  These should be
       `send_parameters` and `update_ui` functions for the UI and
       `update_parameters` and `send_results` functions for the calculation.
       The environment will handle calling these functions.
    3. Communication between the UI and the calculation will occur via channels
       set up in the environment.  The data analysis queue will be used to send
       information to the control law, and the GUI update queue will be used to
       send information back to the UI.
    4. When the Initialize Environment button is clicked, the control law must
       initialize itself and build a graphical user interface for the control
       law.  Extra parameters from the Rattlesnake box will be sent to this
       initialization function, and the GUI builder can use this for initial
       values for the control law, or for whatever other reason there might be.
    5. When the "Start" button is clicked on the System Identification, the
       control law UI will send it's current parameter state to the calculation
       class.  The calculation class will update itself.
    6. When the system identification completes, the control calculation will
       receive the system identification information and then perform a control
       calculation for the prediction step.  After this control calculation,
       results will be obtained and sent back to the UI.
    7. During control, the control law will repeatedly send updates back to the
       UI.  The UI will not automatically send parameters unless told to do so
       through a callback in its UI.


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

from abc import ABC, abstractmethod
from enum import Enum
from rattlesnake.utilities import GlobalCommands


class ControlLawUICommands(Enum):
    INTERACTIVE_CONTROL_UPDATE = 0


class AbstractControlLawUI(ABC):
    """A user interface to allow users to create interactive control laws"""

    @abstractmethod
    def __init__(self, process_name, send_parameters_queue, window, parent_ui_class):
        """Initializes an interactive control law

        Parameters
        ----------
        process_name : str
            The process name associated with this user interface
        send_parameters_queue : np.Queue
            A Multiprocessing queue into which parameters defined by the UI will be put to be
            used by the environment process
        window : QDialog
            The dialog window that the user interface will be placed in
        parent_ui_class : RandomVibrationUI
            The user interface object for the environment that spawned this control law.
        """
        self.send_parameters_queue = send_parameters_queue
        self.process_name = process_name
        self.window = window
        self.parent_ui_class = parent_ui_class
        self.data_acquisition_parameters = None
        self.environment_parameters = None

    def initialize_parameters(self, data_acquisition_parameters, environment_parameters):
        """Stores the data acquisition and environment parameters to the UI

        Parameters
        ----------
        data_acquisition_parameters : DataAcquisitionParameters
            The global data acquisition parameters like sample rate and channel table.
        environment_parameters : RandomVibrationMetadata
            Parameters defining the environment
        """
        self.data_acquisition_parameters = data_acquisition_parameters
        self.environment_parameters = environment_parameters

    def send_parameters(self):
        """Sends parameters from the UI to the environment process"""
        self.send_parameters_queue.put(
            self.process_name,
            (
                GlobalCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS,
                self.collect_parameters(),
            ),
        )

    def run_callback(self, command, *args):
        """Tells the environment process to run a specific command"""
        self.send_parameters_queue.put(
            self.process_name,
            (GlobalCommands.SEND_INTERACTIVE_COMMAND, (command, args)),
        )

    @abstractmethod
    def collect_parameters(self) -> dict:
        """Collects parameters from the UI to send to the environment process"""

    @abstractmethod
    def update_ui_control(self, results: dict):
        """Updates the UI with results from the control law

        Parameters
        ----------
        results : dict
            A dictionary containing information the UI might need to update itself
        """

    @abstractmethod
    def update_ui_sysid(
        self,
        sysid_frf,  # Transfer Functions
        sysid_response_noise,  # Noise levels and correlation
        sysid_reference_noise,  # from the system identification
        sysid_response_cpsd,  # Response levels and correlation
        sysid_reference_cpsd,  # from the system identification
        sysid_coherence,  # Coherence from the system identification
    ):
        """Updates the UI with information from the system identification

        Parameters
        ----------
        sysid_frf : np.ndarray
            The system transfer functions
        sysid_response_noise : np.ndarray
            The noise CPSD matrix at the control channels from the system identification
        sysid_reference_noise : np.ndarray
            The noise CPSD matrix at the drive channels from the system identification
        sysid_response_cpsd : np.ndarray
            The Buzz CPSD at the control channels from the system identification
        sysid_reference_cpsd : np.ndarray
            The Buzz CPSD at the drive channels from the system identification
        sysid_coherence : np.ndarray
            The multiple coherence for each of the control channels from the system identification
        """

    def close(self):
        """Closes the UI window"""
        self.window.close()


class AbstractControlLawComputation(ABC):
    """Computation process for the interactive control law that runs on the environment
    data analysis process"""

    @abstractmethod
    def __init__(self, environment_name, gui_update_queue):
        """Initializes the control process

        Parameters
        ----------
        environment_name : str
            The name of the environment that the control law is running in
        gui_update_queue : mp.Queue
            The queue into which GUI updates will be put
        """
        self.environment_name = environment_name
        self.gui_update_queue = gui_update_queue
        self._command_map = {}

    def send_results(self):
        """Sends results of the control calculation to the UI to update itself"""
        self.gui_update_queue.put(
            (
                self.environment_name,
                (ControlLawUICommands.INTERACTIVE_CONTROL_UPDATE, self.collect_results()),
            )
        )

    @abstractmethod
    def update_parameters(self, parameters: dict):
        """Updates the control computation based on parameters sent from the UI

        Parameters
        ----------
        parameters : dict
            A dictionary containing relevant parameters from the user interface
        """

    @abstractmethod
    def collect_results(self) -> dict:
        """Collects results from the computation to send to the user interface for updates

        Returns
        -------
        dict
            A dictionary containing relevant results to display on the user interface
        """

    @abstractmethod
    def control(self):
        """Executes the control calculation"""

    @abstractmethod
    def system_id_update(self):
        """Updates the control law based on parameters received from the system identification"""

    @staticmethod
    @abstractmethod
    def get_ui_class():
        """Returns the User Interface class corresponding to this calculation class"""

    @property
    def command_map(self) -> dict:
        """A dictionary that maps commands received by the ``command_queue``
        to functions in the class"""
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

    def send_command(self, command: tuple):
        """A function used by the environment process to execute commands sent from the UI
        The UI object uses the `send_parameters_queue` (which is also the environment_command_queue)
        to send custom instructions (set up using the map_command function) to this computation
        object.

        Parameters
        ----------
        command : tuple
            (command enumeration, command arguments)

        Returns
        -------
        Any
        """
        func, args = command
        function = self._command_map[func]
        return function(*args)
