import multiprocessing as mp
from abc import ABC, abstractmethod
from datetime import datetime

from qtpy import QtWidgets, QtCore

from rattlesnake.engine import RattlesnakeController
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.abstract_environment import (
    EnvironmentMetadata,
    EnvironmentInstructions,
)
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.user_interface.ui_utilities import UICommands, EventWatcher


# region User Interface
class EnvironmentUI(ABC):
    """Abstract User Interface class defining the interface with the controller

    This class is used to define the interface between the User Interface of a
    environment in the controller and the main controller."""

    @abstractmethod
    def __init__(
        self,
        environment_type: EnvironmentType,
        environment_name: str,
        rattlesnake: RattlesnakeController,
    ):
        self.environment_type = environment_type
        self.environment_name = environment_name
        self.rattlesnake = rattlesnake
        self.hardware_metadata = None
        self.definition_widget = None
        self.system_id_widget = None
        self.prediction_widget = None
        self.run_widget = None
        self.event_thread = None
        self.event_watcher = None

    # region State Sync
    def get_channel_list_bools(self, global_channel_list):
        channel_set = set(self.hardware_metadata.channel_list)
        channel_list_bools = [channel in channel_set for channel in global_channel_list]
        return channel_list_bools

    @abstractmethod
    def initialize_hardware(self, hardware_metadata: HardwareMetadata):
        """Update the user interface with data acquisition parameters

        This function is called when the Data Acquisition parameters are
        initialized.  This function should set up the environment user interface
        accordingly.

        Parameters
        ----------
        hardware_metadata : HardwareMetadata :
            Container containing the data acquisition parameters, including
            channel table and sampling information.

        """
        self.hardware_metadata = hardware_metadata

    @abstractmethod
    def initialize_environment(self, environment_metadata: EnvironmentMetadata):
        """Updates the later environment tabs with environment metadata. This is called after
        get_environment_metadata when initializing from UI. If loading from a template or
        existing rattlesnake state, this will be called after set_environment_metadata.
        This allows you to update the system id, test predictions, and run tab with data from
        the environment definitions tab."""

    @abstractmethod
    def get_environment_metadata(self, global_channel_list) -> EnvironmentMetadata:
        """
        Collect the parameters from the user interface defining the environment

        Parameters
        ----------
        global_channel_list : List[Channel] :
            List of all hardware channels. Since environments deal with subsets of channel list,
            this is required to build channel list bools

        Returns
        -------
        EnvironmentMetadata
            An EnvironmentMetadata-inheriting object that contains the parameters
            defining the environment.
        """

    @abstractmethod
    def set_environment_metadata(self, metadata: EnvironmentMetadata):
        """
        Update the user interface from environment metadata

        This function is called when the Environment parameters are initialized.
        This function should set up the user interface accordingly.  It must
        return the parameters class of the environment that inherits from
        AbstractMetadata.
        """

    @abstractmethod
    def get_environment_instructions(self) -> EnvironmentInstructions:
        """
        Compiles environment instructions to give to the main environment class
        when start_environment is called

        Returns
        -------
        EnvironmentInstructions
            An EnvironmentInstructions-inheriting object that contians parameters
            in the environment likely to change between runs
        """

    @abstractmethod
    def set_environment_instructions(self, instructions: EnvironmentInstructions):
        """
        Updates the user interface with environment instructions

        This function is called when wanting to sync the environment ui with an
        EnvironmentInstructions object. This will most likely set widgets in the
        environment's run_tab to the values in the EnvironmentInstructions
        """

    # endregion

    # region Process
    @property
    def active(self):
        try:
            queue_name = self.rattlesnake.environment_manager.queue_names_dict[
                self.environment_name
            ]
            return self.rattlesnake.environment_manager.event_container.environment_active_events[
                queue_name
            ].is_set()
        except:
            return False

    @abstractmethod
    def start_environment(self):
        """
        This method in the UI class should follow this structure:
        1. Disable start_environment button
        2. Call super().start_environment
        """
        try:
            instructions = self.get_environment_instructions()
            queue_name = self.rattlesnake.environment_manager.queue_names_dict[
                self.environment_name
            ]
            self.rattlesnake.start_environment(instructions)
        except Exception as e:
            self.start_environment_error(e)
            return

        ready_event_list = []
        active_event_list = [
            self.rattlesnake.event_container.environment_active_events[queue_name]
        ]
        self.create_event_watcher(
            ready_event_list, active_event_list, active_event_check=True
        )
        self.event_watcher.ready.connect(self.start_environment_ready)
        self.event_watcher.error.connect(self.start_environment_error)
        self.event_thread.start()

    @abstractmethod
    def start_environment_ready(self):
        if self.active:
            self.display_environment_started()
        else:
            self.display_environment_ended()

        self.clean_up_event_watcher()

    @abstractmethod
    def start_environment_error(self, error):
        """
        This method defines how to recover UI if the instruction/environment did
        not start up correctly. Should follow this structure:
        1. Enable stop_environment and start_environment button
        2. Call super().start_environment_error or display error some other way
        """
        if self.active:
            self.display_environment_started()
        else:
            self.display_environment_ended()

        self.clean_up_event_watcher()
        self.display_error(error)

    @abstractmethod
    def stop_environment(self):
        """
        This method in the UI class should follow this structure:
        1. Disable stop_environment button
        2. Call super().stop_environment
        """
        try:
            queue_name = self.rattlesnake.environment_manager.queue_names_dict[
                self.environment_name
            ]
            self.rattlesnake.stop_environment(self.environment_name)
        except Exception as e:
            self.stop_environment_error(e)
            return

        ready_event_list = []
        active_event_list = [
            self.rattlesnake.event_container.environment_active_events[queue_name]
        ]
        self.create_event_watcher(
            ready_event_list, active_event_list, active_event_check=False
        )
        self.event_watcher.ready.connect(self.stop_environment_ready)
        self.event_watcher.error.connect(self.stop_environment_error)
        self.event_thread.start()

    @abstractmethod
    def stop_environment_ready(self):
        if self.active:
            self.display_environment_started()
        else:
            self.display_environment_ended()

        self.clean_up_event_watcher()

    @abstractmethod
    def stop_environment_error(self, error):
        """
        This method defines how to recover UI if the instruction/environment did
        not start up correctly. Should follow this structure:
        1. Enable stop_environment and start_environment button
        2. Call super().start_environment_error or display error some other way
        """
        if self.active:
            self.display_environment_started()
        else:
            self.display_environment_ended()

        self.clean_up_event_watcher()
        self.display_error(error)

    @abstractmethod
    def display_environment_started(self):
        """
        This command is called when the environment process officially
        starts up. Needs to prevent user from starting environment again until
        display_environment ended has been called.
        """

    @abstractmethod
    def display_environment_ended(self):
        """
        This command is called when the environment process has officially
        shut down. Needs to enable the user to start up the process again.
        """

    # region Commands
    @property
    def log_file_queue(self) -> mp.Queue:
        """A property containing a reference to the queue accepting messages
        that will be written to the log file"""
        return self.rattlesnake.queue_container.log_file_queue

    @property
    def gui_update_queue(self) -> mp.Queue:
        return self.rattlesnake.queue_container.gui_update_queue

    @property
    def log_name(self):
        """A property containing the name that the UI will be referenced by in
        the log file, which will typically be ``self.environment_name + ' UI'``"""
        return self.environment_name + " UI"

    @abstractmethod
    def update_gui(self, queue_data: tuple):
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
        command, data = queue_data
        match command:
            case UICommands.ENVIRONMENT_STARTED:
                self.display_environment_started()
            case UICommands.ENVIRONMENT_ENDED:
                self.display_environment_ended()
            case UICommands.SET_ENVIRONMENT_INSTRUCTIONS:
                self.set_environment_instructions(data)
            case UICommands.SET_ATTR:
                attr, data = data
                widget = getattr(self, attr)
                if isinstance(widget, QtWidgets.QDoubleSpinBox):
                    widget.setValue(data)
                elif isinstance(widget, QtWidgets.QSpinBox):
                    widget.setValue(data)
                elif isinstance(widget, QtWidgets.QLineEdit):
                    widget.setText(data)
                elif isinstance(widget, QtWidgets.QListWidget):
                    widget.clear()
                    widget.addItems([f"{d:.3f}" for d in data])
                else:
                    print(
                        f"{widget} is not found in {self.environment_name} uesr interface"
                    )
            case _:
                return False

    def log(self, message: str):
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
        self.log_file_queue.put(f"{datetime.now()}: {self.log_name} -- {message}\n")

    def create_event_watcher(
        self, ready_event_list, active_event_list, *, active_event_check: bool = None
    ):
        if getattr(self, "event_thread", None) or getattr(self, "event_watcher", None):
            self.display_error("Event watcher is still active")
            return
        self.event_thread = QtCore.QThread()
        self.event_watcher = EventWatcher(
            ready_event_list,
            active_event_list,
            active_event_check=active_event_check,
            timeout=self.rattlesnake.timeout,
        )
        self.event_watcher.moveToThread(self.event_thread)
        self.event_thread.started.connect(self.event_watcher.run)

    def clean_up_event_watcher(self):
        if getattr(self, "event_thread", None):
            self.event_thread.quit()
            self.event_thread.wait()
            self.event_thread.deleteLater()
            self.event_thread = None
        if getattr(self, "event_watcher", None):
            self.event_watcher.deleteLater()
            self.event_watcher = None

    def display_error(self, error_message):
        self.log(f"ERROR\n\n {error_message}")
        self.gui_update_queue.put(
            (
                UICommands.ERROR,
                (f"{self.log_name} Error", f"ERROR:\n\n{error_message}"),
            )
        )
