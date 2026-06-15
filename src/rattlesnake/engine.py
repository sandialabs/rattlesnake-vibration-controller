from datetime import datetime
from enum import Enum
import multiprocessing as mp
import os
import queue as thqueue
import threading
import time
from typing import Dict, List

import openpyxl
import netCDF4 as nc4

from rattlesnake.utilities import (
    log_file_task,
    flush_queue,
    EventContainer,
    GlobalCommands,
    QueueContainer,
    RattlesnakeError,
    VerboseMessageQueue,
)
from rattlesnake.environment_manager import EnvironmentManager
from rattlesnake.profile_manager import ProfileEvent, ProfileManager
from rattlesnake.load_utilities import (
    load_metadata_from_netcdf,
    load_metadata_from_workbook,
    save_rattlesnake_to_workbook,
)
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.hardware.hardware_registry import HARDWARE_METADATA
from rattlesnake.environment.abstract_environment import (
    EnvironmentInstructions,
    EnvironmentMetadata,
)
from rattlesnake.environment.environment_registry import ENVIRONMENT_METADATA
from rattlesnake.process.acquisition import acquisition_process
from rattlesnake.process.output import output_process
from rattlesnake.process.streaming import streaming_process, StreamMetadata, StreamType
from rattlesnake.process.controller import controller_process
from rattlesnake.process.abstract_sysid_data_analysis import SysIdMetadata
from rattlesnake.environment.environment_registry import SYSID_ENVIRONMENTS

# from rattlesnake.process.sysid_data_analysis import SysIdMetadata


TASK_NAME = "Rattlesnake"
CLOSE_TIMEOUT = 5  # Number of seconds to wait for process to join
THREADING = False


# region State
class RattlesnakeState(Enum):
    # We don't check for stored stream/profiles because they can be left blank
    INIT = 0  # Nothing is stored yet
    HARDWARE_STORE = 1  # Hardware has been stored
    ENVIRONMENT_STORE = 2  # Environments have been stored
    HARDWARE_ACTIVE = 3  # Acquisition is running
    ENVIRONMENT_ACTIVE = 4  # Environment output is running
    SYS_ID_ACTIVE = 5  # System identification is being performed


# region Rattlesnake
class RattlesnakeController:
    """Object responsible for setting up, sending data to, and running processes that
    make up the rattlesnake vibration controller."""

    def __init__(self, *, threaded: bool = THREADING, timeout: float = 10):
        """
        Initializes a blank rattlesnake controller object and spins up multiple processes
        required to run a vibration test.

        Running in the threaded mode results in much quicker data transfer rates as commands
        do not have to be copied in the RAM. This mode allows the control laws to respond quicker
        to changes in response data. The downside is that threaded mode can be overloaded if
        the processor is not fast enough which can result in a crash.

        The timeout is used for waiting for processes to confirm they have recieved a command
        which prevents rattlesnake from overloading the processes with new commands when it is
        running in headless mode.

        Parameters
        ----------
        threaded : bool
            Tells rattlesnake whether to run on a single core or multiple processing cores. A
            value of True corresponds to single threaded
        timeout : float
            The amount of time rattlesnake waits for a process to respond after giving it a
            command. A timeout error will roll back to the previous state and require you to
            perform the command again.
        """
        # Initialize values for checking state
        self._threaded = threaded
        self._blocking = True  # Wait for ready events?, True for IDE, False for UI
        self._timeout = timeout  # Timeout while waiting for ready_events
        self.has_gui = False

        if self.threaded:
            new_queue = thqueue.Queue  # threading-safe in-memory queue
            new_process = threading.Thread  # worker threads
            new_event = threading.Event  # optional stop flag
        else:
            new_queue = mp.Queue  # multiprocessing queue
            new_process = mp.Process  # worker processes
            new_event = mp.Event  # optional stop flag

        # Start up log file process
        log_file_queue = mp.Queue()
        log_close_event = mp.Event()
        self.log_file_process = mp.Process(
            target=log_file_task,
            args=(log_file_queue, log_close_event),
        )
        self.log_file_process.start()

        # Start up command queues and processes
        self.controller_queue_name_manager = (
            mp.Manager()
        )  # Adds minor overhead which is reasonable for COMMAND queues only
        controller_close_event = new_event()
        controller_ready_event = new_event()
        controller_command_queue = VerboseMessageQueue(
            log_file_queue,
            new_queue(),
            "Controller Command Queue",
            self.controller_queue_name_manager,
        )
        acquisition_close_event = new_event()
        acquisition_ready_event = new_event()
        acquisition_active_event = new_event()
        acquisition_active_event.clear()
        acquisition_command_queue = VerboseMessageQueue(
            log_file_queue,
            new_queue(),
            "Acquisition Command Queue",
            self.controller_queue_name_manager,
        )
        output_close_event = new_event()
        output_ready_event = new_event()
        output_active_event = new_event()
        output_active_event.clear()
        output_command_queue = VerboseMessageQueue(
            log_file_queue,
            mp.Queue(),
            "Output Command Queue",
            self.controller_queue_name_manager,
        )
        streaming_close_event = new_event()
        streaming_ready_event = new_event()
        streaming_active_event = new_event()
        streaming_active_event.clear()
        streaming_command_queue = VerboseMessageQueue(
            log_file_queue,
            new_queue(),
            "Streaming Command Queue",
            self.controller_queue_name_manager,
        )

        # Set up data queue
        input_output_sync_queue = new_queue()
        single_process_hardware_queue = new_queue()

        # Set up environment queues
        max_environments = 16
        environment_close_events = {}
        environment_ready_events = {}
        environment_active_events = {}
        environment_sysid_active_events = {}
        environment_sysid_stored_events = {}
        environment_command_queues = {}
        environment_data_in_queues = {}
        environment_data_out_queues = {}
        for env_idx in range(max_environments):
            queue_name = "Environment {:}".format(env_idx)
            environment_close_events[queue_name] = new_event()
            environment_ready_events[queue_name] = new_event()
            environment_active_events[queue_name] = new_event()
            environment_sysid_active_events[queue_name] = new_event()
            environment_sysid_stored_events[queue_name] = new_event()
            environment_active_events[queue_name].clear()
            environment_sysid_active_events[queue_name].clear()
            environment_command_queues[queue_name] = VerboseMessageQueue(
                log_file_queue,
                mp.Queue(),
                queue_name + " Command Queue",
                self.controller_queue_name_manager,
            )
            environment_data_in_queues[queue_name] = new_queue()
            environment_data_out_queues[queue_name] = new_queue()

        # Set up output queue
        gui_update_queue = new_queue()

        # Event for telling engine to restart timeout
        ping_alive_event = new_event()

        # Build queue container
        self.queue_container = QueueContainer(
            controller_command_queue,
            acquisition_command_queue,
            output_command_queue,
            streaming_command_queue,
            log_file_queue,
            input_output_sync_queue,
            single_process_hardware_queue,
            gui_update_queue,
            environment_command_queues,
            environment_data_in_queues,
            environment_data_out_queues,
        )
        self.event_container = EventContainer(
            controller_ready_event,
            acquisition_ready_event,
            output_ready_event,
            streaming_ready_event,
            environment_ready_events,
            log_close_event,
            controller_close_event,
            acquisition_close_event,
            output_close_event,
            streaming_close_event,
            environment_close_events,
            acquisition_active_event,
            output_active_event,
            streaming_active_event,
            environment_active_events,
            environment_sysid_active_events,
            environment_sysid_stored_events,
            ping_alive_event,
        )

        # Controller
        self.controller_proc = new_process(
            target=controller_process,
            args=(
                self.queue_container,
                self.event_container.acquisition_active_event,
                self.event_container.output_active_event,
                self.event_container.streaming_active_event,
                self.event_container.environment_active_events,
                self.event_container.environment_sysid_active_events,
                self.event_container.controller_ready_event,
                self.event_container.controller_close_event,
            ),
        )
        self.controller_proc.start()
        # Acquisition
        self.acquisition_proc = new_process(
            target=acquisition_process,
            args=(
                self.queue_container,
                self.event_container.acquisition_active_event,
                self.event_container.streaming_active_event,
                self.event_container.acquisition_ready_event,
                self.event_container.acquisition_close_event,
                self.event_container.ping_alive_event,
            ),
        )

        self.acquisition_proc.start()
        # Output
        self.output_proc = new_process(
            target=output_process,
            args=(
                self.queue_container,
                self.event_container.output_active_event,
                self.event_container.output_ready_event,
                self.event_container.output_close_event,
                self.event_container.ping_alive_event,
            ),
        )
        self.output_proc.start()
        # Streaming
        self.streaming_proc = new_process(
            target=streaming_process,
            args=(
                self.queue_container,
                self.event_container.streaming_ready_event,
                self.event_container.streaming_close_event,
            ),
        )
        self.streaming_proc.start()

        # Set up managers that will setup processes and store metadata
        self.environment_manager = (
            EnvironmentManager(  # Contains hardware/environment metadata
                self.queue_container,
                self.event_container,
                self.threaded,
            )
        )
        self.profile_manager = ProfileManager(
            self.queue_container
        )  # Contains instructions/profile events
        self._hardware_metadata = None
        # These are only used for UI to pull from if they have already been set to the controller. These
        # are not used for any logic in this controller
        self.last_stream_metadata = None
        self.last_profile_event_list = []

        if self.blocking:
            ready_event_list = [
                self.event_container.controller_ready_event,
                self.event_container.acquisition_ready_event,
                self.event_container.output_ready_event,
                self.event_container.streaming_ready_event,
                *self.environment_manager.ready_event_list,
            ]
            active_event_list = []
            self.wait_for_events(ready_event_list, active_event_list)

    @property
    def state(self) -> RattlesnakeState:
        hardware_store = self.hardware_metadata is not None
        environment_store = self.environment_metadata != {}
        acquisition_active = self.event_container.acquisition_active_event.is_set()
        output_active = self.event_container.output_active_event.is_set()
        environment_active = any(
            event.is_set()
            for event in self.event_container.environment_active_events.values()
        )
        sysid_active = any(
            event.is_set()
            for event in self.event_container.environment_sysid_active_events.values()
        )

        if (
            hardware_store
            and environment_store
            and acquisition_active
            and output_active
            and sysid_active
        ):
            return RattlesnakeState.SYS_ID_ACTIVE

        if (
            hardware_store
            and environment_store
            and acquisition_active
            and output_active
            and environment_active
        ):
            return RattlesnakeState.ENVIRONMENT_ACTIVE

        if (
            hardware_store
            and environment_store
            and acquisition_active
            and output_active
        ):
            return RattlesnakeState.HARDWARE_ACTIVE

        if hardware_store and environment_store:
            return RattlesnakeState.ENVIRONMENT_STORE

        if hardware_store:
            return RattlesnakeState.HARDWARE_STORE

        return RattlesnakeState.INIT

    @property
    def threaded(self):
        return self._threaded

    @property
    def blocking(self):
        return self._blocking

    @property
    def timeout(self):
        return self._timeout

    @property
    def is_alive(self):
        return self.event_container.ping_alive_event.is_set()

    def set_alive(self):
        self.event_container.ping_alive_event.set()

    def clear_alive(self):
        self.event_container.ping_alive_event.clear()

    def set_blocking(self):
        """
        Tells rattlesnake to wait for a response from a process after sending a command.
        """
        self._blocking = True

    def clear_blocking(self):
        """
        Tells rattlesnake to not check for response from a process after sending commands.
        """
        self._blocking = False

    def wait_for_events(
        self,
        ready_event_list: List[mp.synchronize.Event],
        active_event_list: List[mp.synchronize.Event],
        *,
        active_event_check: bool | None = None,
    ):
        """
        Checks for response from processes after rattlesnake has sent a command.

        The events in the ready_event_list are cleared after they are set as the
        ready state for that process has been confirmed. The events in the
        active_event_list are not cleared after being checked.

        Parameters
        ----------
        ready_event_list : List[mp.synchronize.Event]
            Events that processes use to confirm that they have recieved data. These
            events are cleared automatically after the confirmation
        active_event_list : List[mp.synchronize.Event]
            Events that processes use to confirm that they are ready to recieve data
            from other processes. These events are not cleared after being checked.
        active_event_check: bool
            This determines whether to check if the active_event_list processes are
            active or not active. True correspondes to active, False corresponds to not
            active. This is a required input when using active_event_list.
        """
        start_time = time.time()

        while True:
            ready_ok = all(event.is_set() for event in ready_event_list)
            active_ok = all(
                event.is_set() == active_event_check for event in active_event_list
            )

            if ready_ok and active_ok:
                return

            if self.is_alive:
                print("\nAlive Event Pinged\n")
                start_time = time.time()
                self.clear_alive()

            if self.timeout is not None and (time.time() - start_time) >= self.timeout:
                for event in ready_event_list:
                    event.set()
                raise RattlesnakeError("Timeout waiting for all events to be ready")

            time.sleep(0.25)

    # endregion

    # region Loading
    def setup_gui(self):
        self.clear_blocking()
        self.has_gui = True

    def load_rattlesnake_from_template(self, filepath: str):
        """
        Loads data from worksheet or netcdf4 file to the rattlesnake controller.

        Parameters
        ----------
        filepath : string
            Full path to template file.
        """
        filename, filetype = os.path.splitext(filepath)

        if not os.access(filepath, os.R_OK):
            raise RattlesnakeError("You do not have permissions to open {filepath}")

        # I force blocking on this
        initial_blocking = True
        if not self.blocking:
            initial_blocking = False
            self.set_blocking()
        try:
            match filetype:
                case ".nc4":
                    dataset = nc4.Dataset(filepath)
                    hardware_metadata, environment_metadata_list = (
                        load_metadata_from_netcdf(dataset)
                    )
                    self.initialize_hardware(hardware_metadata)
                    self.initialize_environments(environment_metadata_list)
                    # Initialize system identification if it exists
                    for environment_metadata in environment_metadata_list:
                        environment_type = environment_metadata.environment_type
                        if (
                            environment_type in SYSID_ENVIRONMENTS
                            and environment_metadata.sysid_metadata is not None
                        ):
                            self.initialize_system_id(
                                environment_metadata.sysid_metadata,
                                environment_metadata.environment_name,
                            )
                    self.initialize_profile_event_list([])
                    self.last_stream_metadata = None
                case ".xlsx":
                    workbook = openpyxl.load_workbook(filepath)
                    hardware_metadata, environment_metadata_list, profile_event_list = (
                        load_metadata_from_workbook(workbook)
                    )
                    self.initialize_hardware(hardware_metadata)
                    self.initialize_environments(environment_metadata_list)
                    # Initialize system identification if it exists
                    for environment_metadata in environment_metadata_list:
                        environment_type = environment_metadata.environment_type
                        if (
                            environment_type in SYSID_ENVIRONMENTS
                            and environment_metadata.sysid_metadata is not None
                        ):
                            self.initialize_system_id(
                                environment_metadata.sysid_metadata,
                                environment_metadata.environment_name,
                            )
                    self.initialize_profile_event_list(profile_event_list)
                    self.last_stream_metadata = None
        finally:
            if not initial_blocking:
                self.clear_blocking()

    def save_rattlesnake_to_template(self, filepath: str):
        """
        Saves excel template from current controller state.

        This function will fill out the current data from the rattlesnake class.
        If there is no data stored, it will store a blank template.

        Parameters
        ----------
        filepath : string
            Full path to template file.
        """
        filename, filetype = os.path.splitext(filepath)
        if filetype != ".xlsx":
            raise RattlesnakeError("Rattlesnake only saves .xlsx files as templates")
        workbook = openpyxl.Workbook()
        hardware_metadata = self.hardware_metadata
        environment_metadata_dict = self.environment_metadata.values()
        profile_event_list = self.last_profile_event_list

        save_rattlesnake_to_workbook(
            workbook,
            hardware_metadata,
            environment_metadata_dict,
            profile_event_list,
        )
        workbook.save(filepath)

    def save_system_id_to_file(self, environment_name, filepath):
        if self.state not in (
            RattlesnakeState.ENVIRONMENT_STORE,
            RattlesnakeState.HARDWARE_ACTIVE,
        ):
            raise RattlesnakeError(
                f"Invalid state for saving system identification: {self.state}"
            )

        try:
            queue_name = self.environment_manager.queue_names_dict[environment_name]
        except KeyError:
            raise RattlesnakeError(f"No environments exist for {environment_name} name")

        self.event_container.environment_ready_events[queue_name].clear()
        self.queue_container.environment_command_queues[queue_name].put(
            TASK_NAME, (GlobalCommands.SAVE_SYSTEM_ID, filepath)
        )

        # I dont care about blocking in this case as race issues are very possible
        ready_event_list = [self.event_container.environment_ready_events[queue_name]]
        active_event_list = []
        self.wait_for_events(ready_event_list, active_event_list)

    def load_system_id_from_package(self, environment_name, sysid_package):
        if self.state != RattlesnakeState.ENVIRONMENT_STORE:
            raise RattlesnakeError(
                f"Invalid state for loading system identification: {self.state}"
            )

        queue_name = self.environment_manager.validate_system_id_package(
            environment_name, sysid_package
        )

        self.event_container.environment_sysid_stored_events[queue_name].clear()
        self.queue_container.environment_command_queues[queue_name].put(
            TASK_NAME, (GlobalCommands.LOAD_SYSTEM_ID, sysid_package)
        )

        ready_event_list = [
            self.event_container.environment_sysid_stored_events[queue_name]
        ]
        active_event_list = []
        self.wait_for_events(ready_event_list, active_event_list)

    # endregion

    # region Hardware
    @property
    def hardware_metadata(self):
        return self._hardware_metadata

    @hardware_metadata.setter
    def hardware_metadata(self, value: HardwareMetadata):
        self._hardware_metadata = value

    def initialize_hardware(self, hardware_metadata: HardwareMetadata) -> None:
        """Validates hardware_metadata and sends data to relevant processes"""
        # Validate Rattlesnake State
        if self.state not in (
            RattlesnakeState.INIT,
            RattlesnakeState.HARDWARE_STORE,
            RattlesnakeState.ENVIRONMENT_STORE,
        ):
            raise RattlesnakeError(
                f"Invalid state for this setting hardware: {self.state}"
            )
        # Validate hardware
        if not isinstance(hardware_metadata, HardwareMetadata):
            raise RattlesnakeError(
                "Rattlesnake.set_hardware requires a valid HardwareMetadata class"
            )
        hardware_metadata.validate()

        # Send hardware metadata to the correct processes
        self.log("Setting Hardware")
        self.environment_manager.initialize_hardware(hardware_metadata)
        self.event_container.acquisition_ready_event.clear()
        self.queue_container.acquisition_command_queue.put(
            TASK_NAME, (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata)
        )
        self.event_container.output_ready_event.clear()
        self.queue_container.output_command_queue.put(
            TASK_NAME, (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata)
        )

        if self.blocking:
            ready_event_list = [
                self.event_container.acquisition_ready_event,
                self.event_container.output_ready_event,
                *self.environment_manager.ready_event_list,
            ]
            active_event_list = []
            self.wait_for_events(ready_event_list, active_event_list)

            # Update state
            self.hardware_metadata = hardware_metadata

    # endregion

    # region Environment
    @property
    def environment_metadata(self):
        return self.environment_manager.environment_metadata

    @environment_metadata.setter
    def environment_metadata(self, value: Dict[str, EnvironmentMetadata]):
        self.environment_manager.environment_metadata = value

    def initialize_environments(
        self, environment_metadata_list: List[EnvironmentMetadata]
    ):
        """Validates environment_metadata, starts up environment processes, assigns queues,
        and sends data to relevant processes"""
        # Validate Rattlesnake State
        if self.state not in (
            RattlesnakeState.HARDWARE_STORE,
            RattlesnakeState.ENVIRONMENT_STORE,
        ):
            raise RattlesnakeError(
                f"Invalid state for setting environment: {self.state}"
            )
        # Validate environment metadata list
        self.environment_manager.validate_environment_metadata(
            environment_metadata_list, self.hardware_metadata
        )

        # Send environment meetadata to correct processes
        self.log("Setting Environment")
        environment_metadata = self.environment_manager.initialize_environments(
            environment_metadata_list, self.hardware_metadata
        )
        self.event_container.acquisition_ready_event.clear()
        self.queue_container.acquisition_command_queue.put(
            TASK_NAME,
            (GlobalCommands.INITIALIZE_ENVIRONMENT, environment_metadata),
        )
        self.event_container.output_ready_event.clear()
        self.queue_container.output_command_queue.put(
            TASK_NAME,
            (GlobalCommands.INITIALIZE_ENVIRONMENT, environment_metadata),
        )

        if self.blocking:
            ready_event_list = [
                self.event_container.acquisition_ready_event,
                self.event_container.output_ready_event,
                *self.environment_manager.ready_event_list,
            ]
            active_event_list = []
            self.wait_for_events(ready_event_list, active_event_list)
            self.environment_metadata = environment_metadata

        return environment_metadata  # This is used for user_interface to set when events are confirmed

    # endregion

    # region System Id
    def initialize_system_id(self, sysid_metadata, environment_name):
        if self.state not in (
            RattlesnakeState.ENVIRONMENT_STORE,
            RattlesnakeState.HARDWARE_ACTIVE,
        ):
            raise RattlesnakeError(
                f"Invalid state for storing system identification metadata: {self.state}"
            )
        queue_name = self.environment_manager.validate_system_id_metadata(
            sysid_metadata, self.hardware_metadata, environment_name
        )

        self.event_container.environment_ready_events[queue_name].clear()
        environment_metadata = self.environment_manager.initialize_system_id(
            sysid_metadata, queue_name
        )

        if self.blocking:
            ready_event_list = [
                self.event_container.environment_ready_events[queue_name]
            ]
            active_event_list = []
            self.wait_for_events(
                ready_event_list, active_event_list, active_event_check=False
            )

        self.environment_metadata = environment_metadata

    def start_system_id_noise(self, environment_name):
        if self.state != RattlesnakeState.HARDWARE_ACTIVE:
            raise RattlesnakeError(
                f"Invalid state for starting system identification: {self.state}"
            )
        try:
            queue_name = self.environment_manager.queue_names_dict[environment_name]
        except KeyError:
            raise RattlesnakeError(f"No environments exist for {environment_name} name")

        self.queue_container.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.START_SYSTEM_ID_NOISE, queue_name)
        )

        if self.blocking:
            ready_event_list = []
            active_event_list = [
                self.event_container.environment_sysid_active_events[queue_name]
            ]
            self.wait_for_events(
                ready_event_list, active_event_list, active_event_check=True
            )

    def start_system_id_transfer_function(self, environment_name):
        if self.state != RattlesnakeState.HARDWARE_ACTIVE:
            raise RattlesnakeError(
                f"Invalid state for starting system identification: {self.state}"
            )
        try:
            queue_name = self.environment_manager.queue_names_dict[environment_name]
        except KeyError:
            raise RattlesnakeError(f"No environments exist for {environment_name} name")

        self.queue_container.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.START_SYSTEM_ID_TRANSFER, queue_name)
        )

        if self.blocking:
            ready_event_list = []
            active_event_list = [
                self.event_container.environment_sysid_active_events[queue_name]
            ]
            self.wait_for_events(
                ready_event_list, active_event_list, active_event_check=True
            )

    def stop_system_id(self, environment_name):
        if self.state != RattlesnakeState.SYS_ID_ACTIVE:
            raise RattlesnakeError(
                f"Invalid state for stopping system identification {self.state}"
            )
        try:
            queue_name = self.environment_manager.queue_names_dict[environment_name]
        except KeyError:
            raise RattlesnakeError(f"No environments exist for {environment_name} name")

        self.queue_container.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.STOP_SYSTEM_ID, queue_name)
        )

        if self.blocking:
            ready_event_list = []
            active_event_list = [
                self.event_container.environment_sysid_active_events[queue_name]
            ]
            self.wait_for_events(
                ready_event_list, active_event_list, active_event_check=False
            )

    def preview_system_id_noise(self, sysid_metadata: SysIdMetadata, environment_name):
        if self.state == RattlesnakeState.HARDWARE_ACTIVE:
            self.stop_acquisition()

        if self.state != RattlesnakeState.ENVIRONMENT_STORE:
            raise RattlesnakeError(
                f"Invalid state for previewing system identification noise: {self.state}"
            )

        sysid_metadata.auto_shutdown = False
        self.initialize_system_id(sysid_metadata, environment_name)

        stream_metadata = StreamMetadata(StreamType.NO_STREAM)

        self.start_acquisition(stream_metadata)
        self.start_system_id_noise(environment_name)

    def preview_system_id_transfer(
        self, sysid_metadata: SysIdMetadata, environment_name
    ):
        if self.state != RattlesnakeState.ENVIRONMENT_STORE:
            raise RattlesnakeError(
                f"Invalid state for previewing system identification transfer function: {self.state}"
            )

        sysid_metadata.auto_shutdown = False
        self.initialize_system_id(sysid_metadata, environment_name)

        stream_metadata = StreamMetadata(StreamType.NO_STREAM)

        self.start_acquisition(stream_metadata)
        self.start_system_id_transfer_function(environment_name)

    def run_system_id(self, sysid_metadata: SysIdMetadata, environment_name):
        if self.state != RattlesnakeState.ENVIRONMENT_STORE:
            raise RattlesnakeError(
                f"Invalid state for running system identification: {self.state}"
            )

        # Store metadata to environment
        sysid_metadata.auto_shutdown = True
        self.initialize_system_id(sysid_metadata, environment_name)
        queue_name = self.environment_manager.queue_names_dict[environment_name]

        # Check stream file attribute
        bool_stream = bool(getattr(sysid_metadata, "stream_file", False))
        if bool_stream:
            stream_metadata = StreamMetadata(
                StreamType.MANUAL, sysid_metadata.stream_file
            )
        else:
            stream_metadata = StreamMetadata(StreamType.NO_STREAM)

        self.start_acquisition(stream_metadata)
        if bool_stream:
            self.start_streaming()
        self.start_system_id_noise(environment_name)

        # Wait for automatic shutdown
        if self.blocking:
            ready_event_list = []
            active_event_list = [
                self.event_container.environment_sysid_active_events[queue_name]
            ]
            self.wait_for_events(
                ready_event_list, active_event_list, active_event_check=False
            )

        if bool_stream:
            self.stop_streaming()
            self.start_streaming()
        self.start_system_id_transfer_function(environment_name)

        # Wait for automatic shutdown
        if self.blocking:
            ready_event_list = []
            active_event_list = [
                self.event_container.environment_sysid_active_events[queue_name]
            ]
            self.wait_for_events(
                ready_event_list, active_event_list, active_event_check=False
            )

        if bool_stream:
            self.stop_streaming()
        self.stop_acquisition()

    def stop_system_id_run(self, environment_name):
        if self.state == RattlesnakeState.HARDWARE_ACTIVE:
            self.stop_acquisition()
            return
        self.stop_system_id(environment_name)
        self.stop_acquisition()

    # endregion

    # region Acquisition
    def set_stream_metadata(self, stream_metadata: StreamMetadata):
        """
        This is only used to load a stream_metadata to the controller for UI purposes. Start_acquisition
        still requirs a stream_metadata object so the metadata stored here will never be used.
        """
        if self.state != RattlesnakeState.ENVIRONMENT_STORE:
            raise RattlesnakeError(
                f"Invalid state for starting acquisition: {self.state}"
            )
        if not isinstance(stream_metadata, StreamMetadata):
            raise RattlesnakeError(
                "Rattlesnake.set_stream requires a valid StreamMetadata class"
            )
        stream_metadata.validate()

        self.last_stream_metadata = stream_metadata

    def start_acquisition(self, stream_metadata: StreamMetadata):
        # Validate Rattlesnake State
        if self.state != RattlesnakeState.ENVIRONMENT_STORE:
            raise RattlesnakeError(
                f"Invalid state for starting acquisition: {self.state}"
            )
        # Validate stream metadata
        if not isinstance(stream_metadata, StreamMetadata):
            raise RattlesnakeError(
                "Rattlesnake.set_stream requires a valid StreamMetadata class"
            )
        stream_metadata.validate()

        # Store streaming metadata to controller (side note: ControllerProcess decides when/why to stream not StreamingProcess)
        self.log("Setting Stream Metadata")
        self.event_container.streaming_ready_event.clear()
        self.queue_container.streaming_command_queue.put(
            TASK_NAME,
            (
                GlobalCommands.INITIALIZE_STREAMING,
                (stream_metadata, self.hardware_metadata, self.environment_metadata),
            ),
        )

        # Tell controller to start up the hardware, controller takes over logic from here
        self.log("Arming Test Hardware")
        self.queue_container.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.RUN_HARDWARE, stream_metadata)
        )

        if self.blocking:
            ready_event_list = [
                self.event_container.streaming_ready_event,
            ]
            active_event_list = [
                self.event_container.acquisition_active_event,
                self.event_container.output_active_event,
            ]
            self.wait_for_events(
                ready_event_list, active_event_list, active_event_check=True
            )

        # Update stream_metadata
        self.last_stream_metadata = stream_metadata

    def stop_acquisition(self):
        # Validate rattlesnake state (rattlesnake was acquiring data)
        if self.state not in (
            RattlesnakeState.HARDWARE_ACTIVE,
            RattlesnakeState.ENVIRONMENT_ACTIVE,
            RattlesnakeState.SYS_ID_ACTIVE,
        ):
            raise RattlesnakeError(
                f"Invalid state for stopping acquisition: {self.state}"
            )

        self.log("Disarming Test Hardware")
        self.event_container.acquisition_ready_event.clear()
        self.event_container.output_ready_event.clear()
        # Stop profile
        self.profile_manager.stop_profile()
        # Send stop to contoller -Stop Environment > Stop Streaming > Stop Hardware
        self.queue_container.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.STOP_HARDWARE, None)
        )

        if self.blocking:
            ready_event_list = [
                self.event_container.controller_ready_event,
            ]
            active_event_list = [
                self.event_container.acquisition_active_event,
                self.event_container.output_active_event,
                *self.environment_manager.active_event_list,
            ]
            self.wait_for_events(
                ready_event_list, active_event_list, active_event_check=False
            )

    # endregion

    # region Environment Active
    def start_environment(self, instructions):
        if self.state not in (
            RattlesnakeState.HARDWARE_ACTIVE,
            RattlesnakeState.ENVIRONMENT_ACTIVE,
        ):
            raise RattlesnakeError(
                f"Invalid state for starting environment: {self.state}"
            )
        if not isinstance(instructions, EnvironmentInstructions):
            raise RattlesnakeError(
                "Start_environment must be contain a valid EnvironmentInstructions object"
            )

        # Validate instructions
        queue_name = self.environment_manager.validate_environment_instructions(
            instructions
        )

        self.queue_container.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.START_ENVIRONMENT, (queue_name, instructions))
        )

        if self.blocking:
            ready_event_list = []
            active_event_list = [
                self.event_container.environment_active_events[queue_name]
            ]
            self.wait_for_events(
                ready_event_list, active_event_list, active_event_check=True
            )

    def stop_environment(self, environment_name: str):
        if self.state != RattlesnakeState.ENVIRONMENT_ACTIVE:
            raise RattlesnakeError(
                f"Invalid state for stopping environment: {self.state}"
            )
        try:
            queue_name = self.environment_manager.queue_names_dict[environment_name]
        except KeyError:
            raise RattlesnakeError(f"No environments exist for {environment_name} name")

        self.queue_container.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.STOP_ENVIRONMENT, queue_name)
        )

        if self.blocking:
            ready_event_list = []
            active_event_list = [
                self.event_container.environment_active_events[queue_name]
            ]
            self.wait_for_events(
                ready_event_list, active_event_list, active_event_check=False
            )

    # endregion

    # region Controller
    def environment_at_target_level(self, environment_name: str):
        try:
            self.environment_manager.queue_names_dict[environment_name]
        except KeyError:
            raise RattlesnakeError(f"No environments exist for {environment_name} name")

        self.queue_container.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.STREAM_AT_TARGET_LEVEL, environment_name)
        )

    def send_environment_command(self, environment_name, command, data):
        """
        This is a bypass environment ui's can use to request information from their environments. This
        should only be used for tasks that are almost certainly not going to throw an error and can be
        performed at any rattlesnake state.
        """
        self.log(f"Sending {command} to {environment_name}")
        try:
            queue_name = self.environment_manager.queue_names_dict[environment_name]
        except KeyError:
            raise RattlesnakeError(f"No environments exist for {environment_name} name")

        self.queue_container.environment_command_queues[queue_name].put(
            TASK_NAME, (command, data)
        )

    # endregion

    # region Streaming
    @property
    def streaming(self):
        return self.event_container.streaming_active_event.is_set()

    @property
    def has_streamed(self):
        if self.last_stream_metadata:
            return True
        return False

    def start_streaming(self):
        if self.state not in (
            RattlesnakeState.HARDWARE_ACTIVE,
            RattlesnakeState.ENVIRONMENT_ACTIVE,
            RattlesnakeState.SYS_ID_ACTIVE,
        ):
            raise RattlesnakeError(
                f"Invalid state for starting streaming: {self.state}"
            )
        if self.streaming:
            raise RattlesnakeError(f"Rattlesnake is currently streaming")

        self.queue_container.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.STREAM_MANUAL, None)
        )

        if self.blocking:
            ready_event_list = []
            active_event_list = [self.event_container.streaming_active_event]
            self.wait_for_events(
                ready_event_list, active_event_list, active_event_check=True
            )

    def stop_streaming(self):
        if not self.streaming:
            raise RattlesnakeError(f"Rattlesnake is not currently streaming")

        self.queue_container.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.STOP_STREAMING, None)
        )

        if self.blocking:
            ready_event_list = []
            active_event_list = [self.event_container.streaming_active_event]
            self.wait_for_events(
                ready_event_list, active_event_list, active_event_check=False
            )

    # endregion

    # region Profile
    @property
    def has_profile(self):
        if self.last_profile_event_list:
            return True
        return False

    def initialize_profile_event_list(self, profile_event_list: List[ProfileEvent]):
        """
        This is mainly to preload profile event list for UI purposes. You
        still have to give a profile event list to start_profile so this is
        not useful for headless opperation
        """
        self.log("Settting Profile Event List")

        if self.state not in (
            RattlesnakeState.ENVIRONMENT_STORE,
            RattlesnakeState.HARDWARE_ACTIVE,
        ):
            raise RattlesnakeError(f"Invalid state for storing profile: {self.state}")

        self.environment_manager.validate_profile_events(profile_event_list)
        self.last_profile_event_list = profile_event_list

    def start_profile(self, profile_event_list: List[ProfileEvent]):
        self.log("Starting Profile")
        if self.state != RattlesnakeState.HARDWARE_ACTIVE:
            raise RattlesnakeError(f"Invalid state to start profile: {self.state}")

        # Validate and assign queue_names to events
        self.environment_manager.validate_profile_events(profile_event_list)
        self.profile_manager.validate_profile_list(profile_event_list)

        # Start profile
        self.log("Starting Profile")
        self.event_container.controller_ready_event.clear()
        self.profile_manager.start_profile(profile_event_list)

        if self.blocking:
            ready_event_list = [self.event_container.controller_ready_event]
            active_event_list = []
            self.wait_for_events(ready_event_list, active_event_list)

        # Update profile event list
        self.last_profile_event_list = profile_event_list

    def stop_profile(self):
        self.log("Stopping Profile")
        self.event_container.controller_ready_event.clear()
        self.profile_manager.stop_profile()

        if self.blocking:
            ready_event_list = [self.event_container.controller_ready_event]
            active_event_list = []
            self.wait_for_events(ready_event_list, active_event_list)

    # endregion

    # region Shutdown
    def shutdown(self):
        if self.state in (
            RattlesnakeState.HARDWARE_ACTIVE,
            RattlesnakeState.ENVIRONMENT_ACTIVE,
            RattlesnakeState.SYS_ID_ACTIVE,
        ):
            self.stop_acquisition()
        # Close out of acquisition, output, streaming process
        self.queue_container.log_file_queue.put(
            f"{datetime.now()}: Joining Controller Process\n"
        )
        flush_queue(self.queue_container.gui_update_queue, timeout=CLOSE_TIMEOUT)
        self.queue_container.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.QUIT, None)
        )
        self.controller_proc.join(timeout=CLOSE_TIMEOUT)
        if self.controller_proc.is_alive():
            self.queue_container.log_file_queue.put(
                f"{datetime.now()}: Force Closing Controller Process\n"
            )
            self.event_container.controller_close_event.set()
            self.controller_proc.join(timeout=CLOSE_TIMEOUT)
            if self.controller_proc.is_alive() and not self.threaded:
                self.controller_proc.terminate()
                self.controller_proc.join()
        self.queue_container.log_file_queue.put(
            f"{datetime.now()}: Joining Acquisition Process\n"
        )
        self.queue_container.acquisition_command_queue.put(
            TASK_NAME, (GlobalCommands.QUIT, None)
        )
        self.acquisition_proc.join(timeout=CLOSE_TIMEOUT)
        if self.acquisition_proc.is_alive():
            self.queue_container.log_file_queue.put(
                f"{datetime.now()}: Force Closing Acquisition Process\n"
            )
            self.event_container.acquisition_close_event.set()
            self.acquisition_proc.join(timeout=CLOSE_TIMEOUT)
            if self.acquisition_proc.is_alive() and not self.threaded:
                self.acquisition_proc.terminate()
                self.acquisition_proc.join()
        self.queue_container.log_file_queue.put(
            f"{datetime.now()}: Joining Output Process\n"
        )
        self.queue_container.output_command_queue.put(
            TASK_NAME, (GlobalCommands.QUIT, None)
        )
        self.output_proc.join(timeout=CLOSE_TIMEOUT)
        if self.output_proc.is_alive():
            self.queue_container.log_file_queue.put(
                f"{datetime.now()}: Force Closing Output Process\n"
            )
            self.event_container.output_close_event.set()
            self.output_proc.join(timeout=CLOSE_TIMEOUT)
            if self.output_proc.is_alive() and not self.threaded:
                self.output_proc.terminate()
                self.output_proc.join()
        self.queue_container.log_file_queue.put(
            f"{datetime.now()}: Joining Streaming Process\n"
        )
        self.queue_container.streaming_command_queue.put(
            TASK_NAME, (GlobalCommands.QUIT, None)
        )
        self.streaming_proc.join(timeout=CLOSE_TIMEOUT)
        if self.streaming_proc.is_alive():
            self.queue_container.log_file_queue.put(
                f"{datetime.now()}: Force Closing Streaming Process\n"
            )
            self.event_container.streaming_close_event.set()
            self.streaming_proc.join(timeout=CLOSE_TIMEOUT)
            if self.streaming_proc.is_alive() and not self.threaded:
                self.streaming_proc.terminate()
                self.streaming_proc.join()

        # Close out of environment processes
        self.environment_manager.close_environments()

        # Close out log file process
        self.queue_container.log_file_queue.put(
            "{:}: Joining Log File Process\n".format(datetime.now())
        )
        self.queue_container.log_file_queue.put(GlobalCommands.QUIT)
        self.log_file_process.join()

    def log(self, string):
        """Pass a message to the log_file_queue along with date/time and task name

        Parameters
        ----------
        string : str
            Message that will be written to the queue

        """
        self.queue_container.log_file_queue.put(
            f"{datetime.now()}: {TASK_NAME} -- {string}\n"
        )

    # endregion
