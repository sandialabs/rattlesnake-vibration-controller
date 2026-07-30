import copy
from enum import Enum
from typing import List
import multiprocessing as mp
import multiprocessing.queues as mpqueue
import queue as thqueue

import netCDF4 as nc4
import openpyxl
import numpy as np

from rattlesnake.utilities import GlobalCommands, RattlesnakeError, VerboseMessageQueue
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.abstract_environment import (
    Environment,
    EnvironmentCommands,
    EnvironmentInstructions,
    EnvironmentMetadata,
)
from rattlesnake.user_interface.ui_utilities import UICommands

ENVIRONMENT_TYPE = EnvironmentType.READ


# region Commands
class ReadCommands(EnvironmentCommands):

    RUN_ENVIRONMENT = 0
    CHANGE_WINDOW_SIZE = 1

    VALID_PROFILE_COMMANDS = (CHANGE_WINDOW_SIZE,)

    VALID_DATA = {CHANGE_WINDOW_SIZE: float}


class ReadUICommands(Enum):
    TIME_DATA = 0
    SET_WINDOW_SIZE = 1


# endregion


# region Metadata
class ReadMetadata(EnvironmentMetadata):
    def __init__(
        self,
        environment_name: str,
        channel_list_bools: List[bool],
        sample_rate: float,
    ):
        super().__init__(
            ENVIRONMENT_TYPE,
            environment_name,
            channel_list_bools,
            sample_rate,
        )

    # endregion

    # region Validation
    def validate(self, hardware_metadata: HardwareMetadata):
        super().validate(hardware_metadata)

    # endregion

    # region Loading
    def save_metadata_to_netcdf(
        self,
        netcdf_group_handle: nc4._netCDF4.Group,
    ):
        return

    @classmethod
    def load_metadata_from_netcdf(
        cls,
        netcdf_group_handle: nc4._netCDF4.Group,
        environment_name: str,
        channel_list_bools: List[bool],
        hardware_metadata: HardwareMetadata,
    ):
        return cls(
            environment_name,
            channel_list_bools,
            hardware_metadata.sample_rate,
        )

    @classmethod
    def create_blank_worksheet_template(cls, worksheet):
        """
        Create a blank worksheet template for skeleton environment metadata.

        Parameters
        ----------
        worksheet : openpyxl.worksheet.worksheet.Worksheet
            Worksheet to populate with the template layout and field labels.

        Notes
        ----------
        Worksheet cell 1, 2 must be set to a string of the environment type.
        """
        super().create_blank_worksheet_template(worksheet)

        worksheet.cell(1, 2, ENVIRONMENT_TYPE.name.title())

    def save_metadata_to_worksheet(
        self,
        worksheet: openpyxl.worksheet.worksheet.Worksheet,
    ):
        """
        Save environment metadata to a worksheet.

        Parameters
        ----------
        worksheet : openpyxl.worksheet.worksheet.Worksheet
            Worksheet in which the environment metadata should be stored.
        """
        super().save_metadata_to_worksheet(worksheet)

    @classmethod
    def load_metadata_from_worksheet(
        cls,
        worksheet: openpyxl.worksheet.worksheet.Worksheet,
        environment_name: str,
        channel_list_bools: List[bool],
        hardware_metadata: HardwareMetadata,
    ):
        """
        Load environment metadata from a worksheet.

        Parameters
        ----------
        worksheet : openpyxl.worksheet.worksheet.Worksheet
            Worksheet containing the stored environment metadata.
        environment_name : str
            Name of the environment, used for logging and identification.
        channel_list_bools : list of bool
            Boolean mask mapping the global channel table to the channels
            enabled for this environment.
        hardware_metadata : HardwareMetadata
            Hardware metadata used to supply hardware-dependent values such as
            the sample rate.

        Returns
        -------
        ReadMetadata
            Metadata instance reconstructed from the worksheet.

        Raises
        ------
        RattlesnakeError
            If the worksheet contains an unexpected parameter name for this
            environment.
        """
        return cls(
            environment_name,
            channel_list_bools,
            hardware_metadata.sample_rate,
        )

    # endregion


# region Instructions
class ReadInstructions(EnvironmentInstructions):
    def __init__(self, environment_name: str, window_size: float = 0):
        super().__init__(ENVIRONMENT_TYPE, environment_name)

        self.window_size = window_size

    def validate(self):
        super().validate()

        if self.window_size <= 0:
            raise RattlesnakeError(
                "{self.environment_name} must have a window size greater than 0"
            )


# endregion


# region Queues
class ReadQueues:
    """Container for queues passed to the environment on process startup"""

    def __init__(
        self,
        environment_command_queue: VerboseMessageQueue,
        gui_update_queue: mp.queues.Queue,
        controller_communication_queue: VerboseMessageQueue,
        data_in_queue: mp.queues.Queue,
        data_out_queue: mp.queues.Queue,
        log_file_queue: VerboseMessageQueue,
    ):
        """
        Creates a namespace to store all the queues used by the Read Environment

        Parameters
        ----------
        environment_command_queue : VerboseMessageQueue
            Queue from which the environment will receive instructions.
        gui_update_queue : mp.queues.Queue
            Queue to which the environment will put GUI updates.
        controller_communication_queue : VerboseMessageQueue
            Queue to which the environment will put global contorller instructions.
        data_in_queue : mp.queues.Queue
            Queue from which the environment will receive data from acquisition.
        data_out_queue : mp.queues.Queue
            Queue to which the environment will write data for output.
        log_file_queue : VerboseMessageQueue
            Queue to which the environment will write log file messages.
        """
        self.environment_command_queue = environment_command_queue
        self.gui_update_queue = gui_update_queue
        self.controller_communication_queue = controller_communication_queue
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.log_file_queue = log_file_queue


# endregion


# region Environment
class ReadEnvironment(Environment):

    def __init__(
        self,
        environment_name: str,
        queue_name: str,
        queue_container: ReadQueues,
        acquisition_active_event: mp.synchronize.Event,
        output_active_event: mp.synchronize.Event,
        active_event: mp.synchronize.Event,
        ready_event: mp.synchronize.Event,
    ):
        """

        Parameters
        ----------
        environment_name : str
            Name of the environment.
        queue_container : ReadQueues
            Container of queues used by the Read Environment.
        acqusition_active_event: mp.Event
            Event that is set when the acqusition process is actively reading from the hardware
        output_active_event: mp.Event
            Event that is set when the output process is actively sending data to the hardware
        active_event: mp.Event
            Event that needs to be set when the environment is processing data from acqusition and sending data to output
        ready_event: mp.Event
            Event that is checked by the controller to make sure that the environment process recieved information correctly
            without any errors.
        """
        super().__init__(
            environment_name,
            queue_name,
            queue_container.environment_command_queue,
            queue_container.gui_update_queue,
            queue_container.controller_communication_queue,
            queue_container.log_file_queue,
            queue_container.data_in_queue,
            queue_container.data_out_queue,
            acquisition_active_event,
            output_active_event,
            active_event,
            ready_event,
        )

        # Define command map
        self.map_command(GlobalCommands.START_ENVIRONMENT, self.start_environment)
        self.map_command(ReadCommands.RUN_ENVIRONMENT, self.run_control)
        self.map_command(ReadCommands.CHANGE_WINDOW_SIZE, self.set_window_size)

        # Persistent data
        self.shutdown_flag = True
        self.last_acqusition = False
        self.control_channels = []
        self.output_signal = []

        # Tell controller that initialization was successful
        self.set_ready()

    # endregion

    # region State Sync
    def initialize_hardware(self, hardware_metadata: HardwareMetadata):
        super().initialize_hardware(hardware_metadata)

        self.control_channels = [
            index
            for index, channel in enumerate(hardware_metadata.channel_list)
            if channel.feedback_device is not None
        ]
        self.output_signal = np.zeros(
            (len(self.control_channels), self.hardware_metadata.samples_per_write)
        )
        self.set_ready()

    def initialize_environment(self, environment_metadata: ReadMetadata):
        super().initialize_environment(environment_metadata)
        self.set_ready()

    # endregion

    # region Commands
    def start_environment(self, data: ReadInstructions):
        if not self.active:
            # Store instructions
            if data is not None:
                window_size = data.window_size
                self.gui_update_queue.put(
                    (
                        self.environment_name,
                        (ReadUICommands.SET_WINDOW_SIZE, window_size),
                    )
                )

            # Set startup flags
            self.set_active()
            self.shutdown_flag = False
            self.last_acqusition = False
            self.gui_update_queue.put(
                (self.environment_name, (UICommands.ENVIRONMENT_STARTED, None))
            )

            # Start Run Environment loop
            self.environment_command_queue.put(
                self.environment_name, (ReadCommands.RUN_ENVIRONMENT, None)
            )

    def run_control(self, data: None):
        # Get data from data in queue and send it to user interface
        try:
            acqusition_data, self.last_acqusition = self.data_in_queue.get_nowait()
            self.gui_update_queue.put(
                (
                    self.environment_name,
                    (ReadUICommands.TIME_DATA, acqusition_data),
                ),
            )
        except (thqueue.Empty, mpqueue.Empty):
            pass

        # If required, put data to data out queue
        if self.data_out_queue.empty():
            self.data_out_queue.put(
                (copy.deepcopy(self.output_signal), self.shutdown_flag)
            )
            if self.shutdown_flag:
                self.shutdown_flag = False

        # If environment is shutting down, flush queue and update UI
        if self.last_acqusition:
            self.environment_command_queue.flush(self.environment_name)
            self.gui_update_queue.put(
                (self.environment_name, (UICommands.ENVIRONMENT_ENDED, None))
            )
            self.clear_active()
        # Run control if environment is not shutting down
        else:
            self.environment_command_queue.put(
                self.environment_name, (ReadCommands.RUN_ENVIRONMENT, None)
            )

    def stop_environment(self, data):
        # Set shutdown flag so the run_control knows to stop control loop
        self.shutdown_flag = True

    def set_window_size(self, data):
        self.gui_update_queue.put(
            (
                self.environment_name,
                (ReadUICommands.SET_WINDOW_SIZE, data),
            )
        )


def read_process(
    environment_name: str,
    queue_name: str,
    input_queue: VerboseMessageQueue,
    gui_update_queue: mp.Queue,
    controller_command_queue: VerboseMessageQueue,
    log_file_queue: mp.Queue,
    data_in_queue: mp.Queue,
    data_out_queue: mp.Queue,
    acquisition_active_event: mp.synchronize.Event,
    output_active_event: mp.synchronize.Event,
    active_event: mp.synchronize.Event,
    ready_event: mp.synchronize.Event,
    shutdown_event: mp.synchronize.Event,
    sysid_active_event: mp.synchronize.Event,
    sysid_stored_event: mp.synchronize.Event,
    ping_alive_event: mp.synchronize.Event,
    threaded: bool,
):
    queue_container = ReadQueues(
        input_queue,
        gui_update_queue,
        controller_command_queue,
        data_in_queue,
        data_out_queue,
        log_file_queue,
    )

    process_class = ReadEnvironment(
        environment_name,
        queue_name,
        queue_container,
        acquisition_active_event,
        output_active_event,
        active_event,
        ready_event,
    )
    process_class.run(shutdown_event)
