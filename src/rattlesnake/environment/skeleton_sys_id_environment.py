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

import copy
import multiprocessing as mp
import threading
import traceback
from enum import Enum
from typing import List
import multiprocessing as mp
import multiprocessing.queues as mpqueue
import queue as thqueue

import netCDF4 as nc4
import openpyxl
import numpy as np

from rattlesnake.utilities import VerboseMessageQueue, GlobalCommands, RattlesnakeError
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.abstract_environment import (
    EnvironmentCommands,
    EnvironmentInstructions,
)
from rattlesnake.environment.abstract_sysid_environment import (
    SysIdEnvironment,
    SysIdEnvironmentMetadata,
)
from rattlesnake.process.abstract_sysid_data_analysis import (
    SysIdMetadata,
    sysid_data_analysis_process,
)
from rattlesnake.process.data_collector import data_collector_process
from rattlesnake.process.signal_generation_process import signal_generation_process
from rattlesnake.process.spectral_processing import spectral_processing_process
from rattlesnake.user_interface.ui_utilities import UICommands

# Update this line to define the environment type, and add to the EnvironmentType
# enumeration in environment/environment_utilities.py
ENVIRONMENT_TYPE = EnvironmentType.SYSID_SKELETON


# region Commands
class SkeletonSysIdCommands(EnvironmentCommands):
    """Enumeration of commands that the controller can send to the environment.

    This enum defines command values intended for use in profile events,
    allowing the controller to issue environment-specific instructions at
    designated times.
    """

    EXAMPLE_RUN_ENVIRONMENT = 0
    EXAMPLE_SET_TEST_LEVEL = 1

    VALID_PROFILE_COMMANDS = (EXAMPLE_SET_TEST_LEVEL,)

    VALID_DATA = {
        EXAMPLE_SET_TEST_LEVEL: float,
    }


class SkeletonSysIdUICommands(Enum):
    EXAMPLE_UI_SHOW_DATA = 0
    EXAMPLE_UI_SET_TEST_LEVEL = 1


# endregion


# region Metadata
class SkeletonSysIdMetadata(SysIdEnvironmentMetadata):
    """Metadata required to define the system identification skeleton environment."""

    def __init__(
        self,
        environment_name: str,
        channel_list_bools: List[bool],
        sample_rate: float,
        example_window_size: float,
        control_channel_indices: List[int],
        output_channel_indices: List[int],
        response_transformation_matrix=None,
        reference_transformation_matrix=None,
        sysid_metadata: SysIdMetadata = None,
    ):

        super().__init__(
            ENVIRONMENT_TYPE,
            environment_name,
            channel_list_bools,
            sample_rate,
            sysid_metadata=sysid_metadata,
        )
        self.example_window_size = example_window_size
        self.control_channel_indices = control_channel_indices
        self.output_channel_indices = output_channel_indices
        self.response_transformation_matrix = response_transformation_matrix
        self.reference_transformation_matrix = reference_transformation_matrix

    @property
    def number_of_channels(self):
        return len(self.channel_indices)

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
    def response_transformation_matrix(self, value):
        self._response_transformation_matrix = value

    @property
    def reference_transformation_matrix(self):
        return self._reference_transformation_matrix

    @reference_transformation_matrix.setter
    def reference_transformation_matrix(self, value):
        self._reference_transformation_matrix = value

    # region Validation
    def validate(self, hardware_metadata: HardwareMetadata):
        super().validate(hardware_metadata)

        if self.example_window_size <= 0:
            raise RattlesnakeError("{self.environment_name} must have a window size greater than 0")
        

    # endregion

    # region Loading
    def save_metadata_to_netcdf(self, netcdf_group_handle: nc4._netCDF4.Group):
        netcdf_group_handle.environment_name = self.environment_name
        netcdf_group_handle.environment_type = str(self.environment_type)
        netcdf_group_handle.sample_rate = self.sample_rate
        return super().save_metadata_to_netcdf(netcdf_group_handle)

    @classmethod
    def load_metadata_from_netcdf(
        cls,
        netcdf_group_handle: nc4._netCDF4.Group,
        environment_name: str,
        channel_list_bools: List[bool],
        hardware_metadata: HardwareMetadata,
    ):
        sysid_metadata = super().load_metadata_from_netcdf(
            netcdf_group_handle, environment_name, channel_list_bools, hardware_metadata
        )

        return cls(
            environment_name=environment_name,
            channel_list_bools=channel_list_bools,
            sample_rate=hardware_metadata.sample_rate,
            sysid_metadata=sysid_metadata,
        )

    @classmethod
    def create_blank_worksheet_template(cls, worksheet):
        super().create_blank_worksheet_template(worksheet)

        worksheet.cell(1, 2, ENVIRONMENT_TYPE.name.title())

    def save_metadata_to_worksheet(self, worksheet):
        super().save_metadata_to_worksheet(worksheet)

    @classmethod
    def load_metadata_from_worksheet(
        cls,
        worksheet,
        environment_name: str,
        channel_list_bools: List[bool],
        hardware_metadata: HardwareMetadata,
    ):
        super().load_metadata_from_worksheet(
            worksheet, environment_name, channel_list_bools, hardware_metadata
        )

        return cls(
            environment_name=environment_name,
            channel_list_bools=channel_list_bools,
            sample_rate=hardware_metadata.sample_rate,
        )

    # endregion


# endregion
class SkeletonSysIdInstructions(EnvironmentInstructions):

    def __init__(self, environment_name: str, example_test_level: float):
        super().__init__(ENVIRONMENT_TYPE, environment_name)

        self.example_test_level = example_test_level

    def validate(self):
        return super().validate()


# region Queues
class SkeletonSysIdQueues:
    """A container class for the queues that this environment will manage."""

    def __init__(
        self,
        environment_name: str,
        environment_command_queue: VerboseMessageQueue,
        gui_update_queue: mp.Queue,
        controller_communication_queue: VerboseMessageQueue,
        data_in_queue: mp.Queue,
        data_out_queue: mp.Queue,
        log_file_queue: VerboseMessageQueue,
    ):
        """A container class for the queues that the skeleton environment will manage.

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
        self.data_analysis_command_queue = VerboseMessageQueue(
            log_file_queue,
            mp.Queue(),
            environment_name + " Data Analysis Command Queue",
        )
        self.signal_generation_command_queue = VerboseMessageQueue(
            log_file_queue,
            mp.Queue(),
            environment_name + " Signal Generation Command Queue",
        )
        self.spectral_command_queue = VerboseMessageQueue(
            log_file_queue,
            mp.Queue(),
            environment_name + " Spectral Computation Command Queue",
        )
        self.collector_command_queue = VerboseMessageQueue(
            log_file_queue,
            mp.Queue(),
            environment_name + " Data Collector Command Queue",
        )
        self.controller_communication_queue = controller_communication_queue
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.data_for_spectral_computation_queue = mp.Queue()
        self.updated_spectral_quantities_queue = mp.Queue()
        self.time_history_to_generate_queue = mp.Queue()
        self.log_file_queue = log_file_queue


# endregion


# region Environment
class SkeletonSysIdEnvironment(SysIdEnvironment):

    def __init__(
        self,
        environment_name: str,
        queue_name: str,
        queue_container: SkeletonSysIdQueues,
        acquisition_active_event: mp.synchronize.Event,
        output_active_event: mp.synchronize.Event,
        active_event: mp.synchronize.Event,
        ready_event: mp.synchronize.Event,
        sysid_active_event: mp.synchronize.Event,
        sysid_stored_event: mp.synchronize.Event,
    ):
        """
        Parameters
        ----------
        environment_name : str
            Name of the environment.
        queue_name : str
            Name of the queue assigned to the environment.
        queue_container : SkeletonQueues
            Container of queues used by the Skeleton System Identification Environment.
        acquisition_active_event : mp.Event
            Event that is set when the acquisition process is actively reading from the hardware
        output_active_event : mp.Event
            Event that is set when the output process is actively sending data to the hardware
        active_event : mp.Event
            Event that needs to be set when the environment is processing data from acquisition and sending data to output
        ready_event : mp.Event
            Event that is checked by the controller to make sure that the environment process received information correctly
            without any errors.
        sysid_active_event : mp.Event
            Event that is set when a system identification measurement is running
        sysid_stored_event : mp.Event
            Event that is set when a system identification result has been stored
        """
        super().__init__(
            environment_name,
            queue_name,
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
            acquisition_active_event,
            output_active_event,
            active_event,
            ready_event,
            sysid_active_event,
            sysid_stored_event,
        )

        self.map_command(GlobalCommands.START_ENVIRONMENT, self.start_environment)
        self.map_command(
            SkeletonSysIdCommands.EXAMPLE_RUN_ENVIRONMENT, self.run_control
        )
        self.map_command(
            SkeletonSysIdCommands.EXAMPLE_SET_TEST_LEVEL, self.set_test_level
        )

        # Persistent data
        self.test_level = 0
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

    def initialize_environment(self, environment_metadata: SkeletonSysIdMetadata):
        super().initialize_environment(environment_metadata)
        self.set_ready()

    def initialize_sysid(self, sysid_metadata: SysIdMetadata):
        super().initialize_sysid(sysid_metadata)
        self.set_ready()

    # endregion

    # region Commands
    def start_environment(self, data: SkeletonSysIdInstructions):
        if not self.active:
            # Store instructions
            if data is not None:
                test_level = data.example_test_level
                self.test_level = test_level
                self.gui_update_queue.put(
                    (
                        self.environment_name,
                        (SkeletonSysIdUICommands.EXAMPLE_UI_SET_TEST_LEVEL, test_level),
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
                self.environment_name,
                (SkeletonSysIdCommands.EXAMPLE_RUN_ENVIRONMENT, None),
            )

    def run_control(self, data: None):
        # Get data from data in queue and send it to user interface
        try:
            acqusition_data, self.last_acqusition = self.data_in_queue.get_nowait()
            self.gui_update_queue.put(
                (
                    self.environment_name,
                    (SkeletonSysIdUICommands.EXAMPLE_UI_SHOW_DATA, acqusition_data),
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
                self.environment_name,
                (SkeletonSysIdCommands.EXAMPLE_RUN_ENVIRONMENT, None),
            )

    def stop_environment(self, data):
        # Set shutdown flag so the run_control knows to stop control loop
        self.shutdown_flag = True

    def set_test_level(self, data):
        self.test_level = data
        print(f"Setting test level {self.test_level}")

    # endregion


# endregion


# region Process
def skeleton_sysid_process(
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
    """Skeleton system identification environment process function called by multiprocessing"""
    try:
        if threaded:
            new_process = threading.Thread
        else:
            new_process = mp.Process

        queue_container = SkeletonSysIdQueues(
            environment_name,
            input_queue,
            gui_update_queue,
            controller_command_queue,
            data_in_queue,
            data_out_queue,
            log_file_queue,
        )

        spectral_proc = new_process(
            target=spectral_processing_process,
            args=(
                environment_name,
                queue_container.spectral_command_queue,
                queue_container.data_for_spectral_computation_queue,
                queue_container.updated_spectral_quantities_queue,
                queue_container.environment_command_queue,
                queue_container.gui_update_queue,
                queue_container.log_file_queue,
                shutdown_event,
            ),
        )
        spectral_proc.start()
        analysis_proc = new_process(
            target=sysid_data_analysis_process,
            args=(
                environment_name,
                queue_container.data_analysis_command_queue,
                queue_container.updated_spectral_quantities_queue,
                queue_container.time_history_to_generate_queue,
                queue_container.environment_command_queue,
                queue_container.gui_update_queue,
                queue_container.log_file_queue,
                ping_alive_event,
                shutdown_event,
            ),
        )
        analysis_proc.start()
        siggen_proc = new_process(
            target=signal_generation_process,
            args=(
                environment_name,
                queue_container.signal_generation_command_queue,
                queue_container.time_history_to_generate_queue,
                queue_container.data_out_queue,
                queue_container.environment_command_queue,
                queue_container.log_file_queue,
                queue_container.gui_update_queue,
                shutdown_event,
            ),
        )
        siggen_proc.start()
        collection_proc = new_process(
            target=data_collector_process,
            args=(
                environment_name,
                queue_container.collector_command_queue,
                queue_container.data_in_queue,
                [queue_container.data_for_spectral_computation_queue],
                queue_container.environment_command_queue,
                queue_container.log_file_queue,
                queue_container.gui_update_queue,
                shutdown_event,
            ),
        )
        collection_proc.start()

        process_class = SkeletonSysIdEnvironment(
            environment_name,
            queue_name,
            queue_container,
            acquisition_active_event,
            output_active_event,
            active_event,
            ready_event,
            sysid_active_event,
            sysid_stored_event,
        )
        process_class.run(shutdown_event)

        # Rejoin all the processes
        process_class.log("Joining Subprocesses")
        process_class.log("Joining Spectral Computation")
        spectral_proc.join()
        process_class.log("Joining Data Analysis")
        analysis_proc.join()
        process_class.log("Joining Signal Generation")
        siggen_proc.join()
        process_class.log("Joining Data Collection")
        collection_proc.join()
    except Exception:  # pylint: disable = broad-exception-caught
        print(traceback.format_exc())


# endregion
