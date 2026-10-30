"""
Main Rattlesnake controller engine.

This module defines the high-level controller object responsible for
constructing queues and events, starting controller subprocesses, managing
hardware and environment metadata, starting acquisition and output,
handling system identification workflows, managing streaming and profiles,
and shutting down the controller cleanly.

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

import atexit
import os
import threading
import time
import multiprocessing as mp
import queue as thqueue
from datetime import datetime
from enum import Enum
from typing import Dict, List

import openpyxl
import netCDF4 as nc4

from rattlesnake.environment_manager import EnvironmentManager
from rattlesnake.load_utilities import (
    load_metadata_from_netcdf,
    load_metadata_from_workbook,
    save_rattlesnake_to_workbook,
)
from rattlesnake.profile_manager import ProfileEvent, ProfileManager
from rattlesnake.typehint import EventType, PathType
from rattlesnake.utilities import (
    log_file_task,
    flush_queue,
    gui_queue_cleanup,
    EventContainer,
    GlobalCommands,
    QueueContainer,
    RattlesnakeError,
    VerboseMessageQueue,
)
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.hardware.hardware_registry import HARDWARE_METADATA
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.abstract_environment import (
    EnvironmentInstructions,
    EnvironmentMetadata,
)
from rattlesnake.environment.environment_registry import (
    ENVIRONMENT_METADATA,
    SYSID_ENVIRONMENTS,
)
from rattlesnake.process.abstract_sysid_data_analysis import (
    SysIdDataPackage,
    SysIdMetadata,
)
from rattlesnake.process.acquisition import acquisition_process
from rattlesnake.process.controller import controller_process
from rattlesnake.process.output import output_process
from rattlesnake.process.streaming import streaming_process, StreamMetadata, StreamType

TASK_NAME = "Rattlesnake"
CLOSE_TIMEOUT = 5  # Number of seconds to wait for process to join
THREADING = False  # Decides whether to spin up multiple processes or threads
GUI_QUEUE_MAX_SIZE = (
    500  # Max size of gui_update_queue before it is trimmed in headless mode
)
GUI_QUEUE_POLL_INTERVAL = 0.25  # Seconds between gui_update_queue size checks


# region State
class RattlesnakeState(Enum):
    """
    Enumeration of high-level controller states.

    The controller state is computed from the attributes of the controller and
    different event flags within the acquisition, output, and environment processes.

    Attributes
    ----------
    INIT : int
        No hardware or environment metadata has been stored.
    HARDWARE_STORE : int
        Hardware metadata has been stored.
    ENVIRONMENT_STORE : int
        Hardware and environment metadata have been stored.
    HARDWARE_ACTIVE : int
        Acquisition and output hardware are running.
    ENVIRONMENT_ACTIVE : int
        At least one environment is actively running while hardware is active.
    SYS_ID_ACTIVE : int
        System identification is active while hardware is active.

    Unit Tests
    ----------

    Notes
    -----
    We don't check for stored streaming/profiles because they can be left blank.
    """

    INIT = 0
    HARDWARE_STORE = 1
    ENVIRONMENT_STORE = 2
    HARDWARE_ACTIVE = 3
    ENVIRONMENT_ACTIVE = 4
    SYS_ID_ACTIVE = 5


# endregion


# region Controller
class RattlesnakeController:
    """
    The main controller that owns the process queues, synchronization events,
    subprocess handles, environment manager, profile manager, and stored
    metadata needed to run tests. It provides the public API
    for initializing hardware, initializing environments, starting and stopping
    acquisition, starting and stopping environments, running system
    identification, managing streaming, managing profiles, loading and saving
    templates, and shutting down all subprocesses.

    Parameters
    ----------
    threaded : bool, optional
        If ``True``, worker components are started as threads and use
        thread-compatible queues and events. If ``False``, worker components
        are started as multiprocessing processes. Defaults to ``THREADING``.
    timeout : float, optional
        Timeout in seconds used by blocking operations while waiting for
        subprocess ready or active events. Defaults to ``40``.

    Attributes
    ----------
    queue_container : QueueContainer
        Container holding controller, acquisition, output, streaming, GUI,
        sync, hardware, and environment queues.
    event_container : EventContainer
        Container holding ready, close, active, system-identification, and
        watchdog events.
    environment_manager : EnvironmentManager
        Manager responsible for environment metadata, environment processes,
        queue names, and environment validation.
    profile_manager : ProfileManager
        Manager responsible for validating, starting, and stopping profile
        events.
    hardware_metadata : HardwareMetadata or None
        Current hardware metadata stored by the controller.
    environment_metadata : dict of str to EnvironmentMetadata
        Current environment metadata stored by the environment manager.
    last_stream_metadata : StreamMetadata or None
        Most recently stored stream metadata for UI use.
    last_profile_event_list : list of ProfileEvent
        Most recently stored profile event list for UI use.
    has_gui : bool
        Whether the controller has been configured for GUI use.
    threaded : bool
        Whether the controller is running in threaded mode.
    blocking : bool
        Whether public API operations wait for process ready and active events.
    timeout : float
        Timeout used by blocking wait operations.

    Unit Tests
    ----------
    """

    def __init__(self, *, threaded: bool = THREADING, timeout: float = 40) -> None:
        """
        Initializes a blank rattlesnake controller object and spins up
        multiple processes required to run a vibration test.

        Creates the log process, queue container, event container, controller
        process, acquisition process, output process, streaming process,
        environment manager, and profile manager.

        Parameters
        ----------
        threaded : bool, optional
            If ``True``, use threads and thread-compatible queues and events. If
            ``False``, use multiprocessing processes and queues.
        timeout : float, optional
            Timeout in seconds to wait for processes to respond to commands before
            resorting to error handling.

        Unit Tests
        ----------
        """
        # Shutdown behavior
        self._shutdown_complete = False
        self._init_complete = False
        atexit.register(self.shutdown)

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

        # Bound the size of gui_update_queue while no GUI is attached
        self.gui_queue_cleanup_close_event = threading.Event()
        self.gui_queue_cleanup_thread = threading.Thread(
            target=gui_queue_cleanup,
            args=(
                gui_update_queue,
                self.gui_queue_cleanup_close_event,
                lambda: self.has_gui,
                GUI_QUEUE_MAX_SIZE,
                GUI_QUEUE_POLL_INTERVAL,
            ),
            daemon=True,
        )
        self.gui_queue_cleanup_thread.start()

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
        self.hardware_metadata = None
        # These are only used for UI to pull from if they have already been
        # set to the controller. These are not used for any logic in this
        # controller
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

        self._init_complete = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    @property
    def state(self) -> RattlesnakeState:
        """
        Return the current controller state.

        The state is computed from whether hardware metadata is stored, whether
        environment metadata is stored, whether acquisition and output are active,
        whether any environment is active, and whether any environment has active
        system identification.

        Returns
        -------
        RattlesnakeState
            Current controller state.

        Unit Tests
        ----------
        """
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
    def threaded(self) -> bool:
        """
        Return whether the controller is running in threaded mode.

        Returns
        -------
        bool
            ``True`` if worker components are threads, otherwise ``False``.

        Unit Tests
        ----------
        """
        return self._threaded

    @property
    def blocking(self) -> bool:
        """
        Return whether controller operations will block execution. This should be
        ``True`` for headless operation and ``False`` for user interface operations.

        Returns
        -------
        bool
            ``True`` if operations wait for ready and active events, otherwise
            ``False``.

        Unit Tests
        ----------
        """
        return self._blocking

    @property
    def timeout(self) -> float:
        """
        Return the controller event wait timeout.

        Returns
        -------
        float
            Timeout in seconds used by blocking wait operations.

        Unit Tests
        ----------
        """
        return self._timeout

    @property
    def is_alive(self) -> bool:
        """
        Returns whether the alive event is set. This event allows subprocesses to
        extend the timeout period when performing long operations.

        Returns
        -------
        bool
            ``True`` if the ping-alive event is set, otherwise ``False``.

        Unit Tests
        ----------
        """
        return self.event_container.ping_alive_event.is_set()

    def set_alive(self) -> None:
        """
        Set the alive event. Marks the controller as having received an alive
        ping from a worker process during a long blocking operation.

        Unit Tests
        ----------
        """
        self.event_container.ping_alive_event.set()

    def clear_alive(self) -> None:
        """
        Clear the alive event. This is used by this controller after receiving a
        ping alive signal and resetting the timeout timer.

        Unit Tests
        ----------
        """
        self.event_container.ping_alive_event.clear()

    def set_blocking(self) -> None:
        """
        Enables blocking controller operations. When this is enabled, the controller
        will pause execution to wait for subprocesses to complete when sending commands.

        Unit Tests
        ----------
        """
        self._blocking = True

    def clear_blocking(self):
        """
        Disables blocking controller operations. When this is enabled, the controller
        will not wait for subprocesses to complete when sending commands.

        Unit Tests
        ----------
        """
        self._blocking = False

    def wait_for_events(
        self,
        ready_event_list: List[EventType],
        active_event_list: List[EventType],
        *,
        active_event_check: bool | None = None,
    ) -> None:
        """
        Wait for process ready and active events.

        Blocks until all supplied ready events are set and all supplied active
        events match ``active_event_check``. If the controller receives an alive
        ping, the timeout timer is reset. If the timeout expires before all
        required event conditions are satisfied, all ready events are set and a
        ``RattlesnakeError`` is raised.

        Parameters
        ----------
        ready_event_list : list of threading.Event or multiprocessing.Event
            Events that subprocesses set to confirm they have completed an
            operation. These events will be set to ``False`` after completion.
        active_event_list : list of threading.Event or multiprocessing.Event
            Events that indicate whether subprocesses or environments are active.
            These events will be unchanged after completion.
        active_event_check : bool or None, optional
            Required state for each active event. Use ``True`` to wait for active
            events to be set, ``False`` to wait for them to be cleared, and
            ``None`` only when no active events are supplied.

        Raises
        ------
        RattlesnakeError
            If the timeout expires before all event conditions are satisfied.

        Unit Tests
        ----------
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
                # print("\nAlive Event Pinged\n")
                start_time = time.time()
                self.clear_alive()

            if self.timeout is not None and (time.time() - start_time) >= self.timeout:
                for event in ready_event_list:
                    event.set()
                raise RattlesnakeError("Timeout waiting for all events to be ready")

            time.sleep(0.25)

    # region Loading
    def setup_gui(self) -> None:
        """
        Configures the controller for GUI operation.

        Disables blocking mode and marks the controller as having a GUI attached.
        GUI-driven workflows typically handle asynchronous updates through gui signals
        to prevent locking the gui execution loop during operation.

        Unit Tests
        ----------
        """
        self.clear_blocking()
        self.has_gui = True

    def close_gui(self) -> None:
        """
        Configures the controller for headless operation.

        Enables blocking mode and removes the gui mark from the controller. The
        controller will now pause execution while subprocesses perform functions.

        Unit Tests
        ----------
        """
        self.set_blocking()
        self.has_gui = False

    def load_rattlesnake_from_template(self, filepath: PathType) -> None:
        """
        Stores metadata from a template file to the controller.

        Loads hardware metadata, environment metadata, optional system
        identification metadata, and optional profile events from a supported
        template file. Supported file types are netCDF ``.nc4`` and Excel
        ``.xlsx``. Blocking mode is temporarily enabled while loading so that
        hardware and environment initialization complete before returning.

        Parameters
        ----------
        filepath : str or os.PathLike
            Path to the template file.

        Raises
        ------
        RattlesnakeError
            If the file cannot be read.
        RattlesnakeError
            If the file type is unsupported by the loading logic.

        Unit Tests
        ----------
        """
        filename, filetype = os.path.splitext(filepath)

        if not os.access(filepath, os.R_OK):
            raise RattlesnakeError(f"You do not have permissions to open {filepath}")

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
                    if profile_event_list:
                        self.initialize_profile_event_list(profile_event_list)
                    self.last_stream_metadata = None
        finally:
            if not initial_blocking:
                self.clear_blocking()

    def save_rattlesnake_to_template(self, filepath: PathType) -> None:
        """
        Save the current controller configuration to an Excel template.

        Creates an Excel workbook and writes hardware metadata, environment
        metadata, and profile events to it. If no metadata is stored, a blank
        template is saved.

        Parameters
        ----------
        filepath : str or os.PathLike
            Output path for the Excel template.

        Raises
        ------
        RattlesnakeError
            If ``filepath`` does not have an ``.xlsx`` extension.

        Unit Tests
        ----------
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

    def save_system_id_to_file(self, environment_name: str, filepath: PathType) -> None:
        """
        Saves system identification data for an environment to a netCDF4, matlab,
        or numpy file.

        Sends a ``SAVE_SYSTEM_ID`` command to the environment and waits
        for its ready event, regardless of the blocking setting. Saving is only
        permitted in the ``ENVIRONMENT_STORE`` or ``HARDWARE_ACTIVE`` states.

        Parameters
        ----------
        environment_name : str
            Name of the environment whose system identification data is saved.
        filepath : str or os.PathLike
            Destination file path. The extension selects the format: ``.nc4``,
            ``.mat``, or ``.npz``. Other extensions are not saved.

        Raises
        ------
        RattlesnakeError
            If the controller state does not permit saving system identification,
            if no environment exists with ``environment_name``, or if the
            environment does not respond before the timeout.

        Unit Tests
        ----------
        """
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

        # I force blocking in this case as race issues are likely
        ready_event_list = [self.event_container.environment_ready_events[queue_name]]
        active_event_list = []
        self.wait_for_events(ready_event_list, active_event_list)

    def load_system_id_from_package(
        self,
        environment_name: str,
        sysid_package: SysIdDataPackage,
        ask_to_share: bool = False,
    ) -> None:
        """
        Loads a system identification package into an environment.

        Validates the package against the environment, sends it to the
        environment, and waits for the system identification stored event.
        Loading is only permitted in the ``ENVIRONMENT_STORE`` state.

        Parameters
        ----------
        environment_name : str
            Name of the environment that receives the package.
        sysid_package : SysIdDataPackage
            System identification package to load.
        ask_to_share : bool, optional
            Whether the user interface should offer to share the loaded system
            identification with other environments. Ignored when the controller
            has no GUI. Defaults to ``False``.

        Raises
        ------
        RattlesnakeError
            If the controller state is not ``ENVIRONMENT_STORE``.
        RattlesnakeError
            If the package does not pass the environment manager's validation.

        Unit Tests
        ----------
        """
        if self.state != RattlesnakeState.ENVIRONMENT_STORE:
            raise RattlesnakeError(
                f"Invalid state for loading system identification: {self.state}"
            )

        queue_name = self.environment_manager.validate_system_id_package(
            environment_name, sysid_package
        )

        if not self.has_gui:
            ask_to_share = False

        self.event_container.environment_sysid_stored_events[queue_name].clear()
        self.queue_container.environment_command_queues[queue_name].put(
            TASK_NAME, (GlobalCommands.LOAD_SYSTEM_ID, (sysid_package, ask_to_share))
        )

        ready_event_list = [
            self.event_container.environment_sysid_stored_events[queue_name]
        ]
        active_event_list = []
        self.wait_for_events(ready_event_list, active_event_list)

    # endregion

    # region Hardware
    def initialize_hardware(self, hardware_metadata: HardwareMetadata) -> None:
        """
        Validate and initialize hardware metadata.

        Validates the current controller state, verifies that the supplied object
        is a ``HardwareMetadata`` instance, calls its validation method, stores
        hardware information in the environment manager, sends
        ``INITIALIZE_HARDWARE`` commands to acquisition and output, and waits for
        ready events when blocking mode is enabled.

        Parameters
        ----------
        hardware_metadata : HardwareMetadata
            Hardware metadata used to initialize acquisition, output, and
            environment-related hardware state.

        Raises
        ------
        RattlesnakeError
            If the controller state does not permit hardware initialization.
        RattlesnakeError
            If ``hardware_metadata`` is not a ``HardwareMetadata`` instance.
        RattlesnakeError
            If hardware metadata validation fails.

        Unit Tests
        ----------
        """
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
    def environment_metadata(self) -> Dict[str, EnvironmentMetadata]:
        """
        Return a dict of environment metadata stored by the environment manager.

        Returns
        -------
        dict of str to EnvironmentMetadata
            Mapping from environment queue names to environment metadata.

        Unit Tests
        ----------
        """
        return self.environment_manager.environment_metadata

    @environment_metadata.setter
    def environment_metadata(self, value: Dict[str, EnvironmentMetadata]):
        """
        Store environment metadata through the environment manager.

        Parameters
        ----------
        value : dict of str to EnvironmentMetadata
            Mapping from environment queue names to environment metadata.
        """
        self.environment_manager.environment_metadata = value

    def initialize_environments(
        self, environment_metadata_list: List[EnvironmentMetadata]
    ):
        """
        Validate and initialize environments.

        Validates the current controller state, validates environment metadata
        against the stored hardware metadata, starts or updates environment
        processes through the environment manager, sends initialized environment
        metadata to acquisition and output, and waits for ready events when
        blocking mode is enabled.

        Parameters
        ----------
        environment_metadata_list : list of EnvironmentMetadata
            Environment metadata objects to initialize.

        Returns
        -------
        dict of str to EnvironmentMetadata
            Mapping from assigned queue names to initialized environment metadata.
            This is the main way to check which queue names were assigned to each
            environment.

        Raises
        ------
        RattlesnakeError
            If the controller state does not permit environment initialization.
        RattlesnakeError
            If environment metadata validation fails.

        Unit Tests
        ----------
        """
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

        # This is used for user_interface to set when events are confirmed
        return environment_metadata

    # endregion

    # region System Id
    def initialize_system_id(
        self, sysid_metadata: SysIdMetadata, environment_name: str
    ) -> None:
        """
        Initialize system identification metadata for an environment.

        Validates that system identification metadata can be stored in the current
        state, validates the metadata against hardware and environment
        configuration, sends initialization through the environment manager, waits
        for the environment ready event when blocking mode is enabled, and updates
        stored environment metadata.

        Parameters
        ----------
        sysid_metadata : SysIdMetadata
            System identification metadata to initialize.
        environment_name : str
            User-facing environment name associated with the metadata.

        Raises
        ------
        RattlesnakeError
            If the controller state does not permit system identification
            initialization.
        RattlesnakeError
            If system identification metadata validation fails.

        Unit Tests
        ----------
        """
        if self.state not in (
            RattlesnakeState.ENVIRONMENT_STORE,
            RattlesnakeState.HARDWARE_ACTIVE,
        ):
            raise RattlesnakeError(
                "Invalid state for storing system identification "
                f"metadata: {self.state}"
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

    def start_system_id_noise(self, environment_name: str) -> None:
        """
        Starts the noise phase for a system identification test.

        Validates that hardware is active, resolves the environment queue name,
        sends a ``START_SYSTEM_ID_NOISE`` command to the controller
        process, and waits for the environment system identification active event
        when blocking mode is enabled.

        Parameters
        ----------
        environment_name : str
            User-facing environment name whose noise measurement should start.

        Raises
        ------
        RattlesnakeError
            If the controller is not in ``HARDWARE_ACTIVE`` state.
        RattlesnakeError
            If no environment exists with ``environment_name``.

        Unit Tests
        ----------
        """

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

    def start_system_id_transfer_function(self, environment_name: str) -> None:
        """
        Starts the transfer-function phase of system identification.

        Validates that hardware is active, resolves the environment queue name,
        sends a ``START_SYSTEM_ID_TRANSFER`` command to the controller
        process, and waits for the environment system identification active event
        when blocking mode is enabled.

        Parameters
        ----------
        environment_name : str
            User-facing environment name whose transfer-function measurement should
            start.

        Raises
        ------
        RattlesnakeError
            If the controller is not in ``HARDWARE_ACTIVE`` state.
        RattlesnakeError
            If no environment exists with ``environment_name``.

        Unit Tests
        ----------
        """
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

    def stop_system_id(self, environment_name: str) -> None:
        """
        Stops active system identification.

        Validates that system identification is active, resolves the environment
        queue name, sends a ``STOP_SYSTEM_ID`` command to the controller
        process, and waits for the system identification active event to clear when
        blocking mode is enabled.

        Parameters
        ----------
        environment_name : str
            User-facing environment name whose system identification should stop.

        Raises
        ------
        RattlesnakeError
            If the controller is not in ``SYS_ID_ACTIVE`` state.
        RattlesnakeError
            If no environment exists with ``environment_name``.

        Unit Tests
        ----------
        """
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

    def preview_system_id_noise(
        self, sysid_metadata: SysIdMetadata, environment_name: str
    ) -> None:
        """
        Performs a preview of the noise for the system identification for the
        environment.

        This is used in headless mode as the preview button. It starts the
        acquisition, initializes the system identification metadata, and starts
        the system identification noise without streaming.

        Parameters
        ----------
        sysid_metadata : SysIdMetadata
            System identification metadata to preview.
        environment_name : str
            User-facing environment name.

        Raises
        ------
        RattlesnakeError
            If the controller is not in the ``ENVIRONMENT_STORE`` state.

        Unit Tests
        ----------
        """
        if self.state == RattlesnakeState.HARDWARE_ACTIVE:
            self.stop_acquisition()

        if self.state != RattlesnakeState.ENVIRONMENT_STORE:
            raise RattlesnakeError(
                "Invalid state for previewing system identification "
                f"noise: {self.state}"
            )

        sysid_metadata.auto_shutdown = False
        self.initialize_system_id(sysid_metadata, environment_name)

        stream_metadata = StreamMetadata(StreamType.NO_STREAM)

        self.start_acquisition(stream_metadata)
        self.start_system_id_noise(environment_name)

    def preview_system_id_transfer(
        self, sysid_metadata: SysIdMetadata, environment_name: str
    ) -> None:
        """
        Performs a preview of the transfer function for the system identification
        for the environment.

        This is used in headless mode as the preview button. It initializes the
        system identification metadata, starts the acquisition, and starts the
        system identification transfer function without streaming.

        Parameters
        ----------
        sysid_metadata : SysIdMetadata
            System identification metadata to preview.
        environment_name : str
            User-facing environment name.

        Raises
        ------
        RattlesnakeError
            If the controller is not in the ``ENVIRONMENT_STORE`` state.

        Unit Tests
        ----------
        """
        if self.state != RattlesnakeState.ENVIRONMENT_STORE:
            raise RattlesnakeError(
                "Invalid state for previewing system identification "
                f"transfer function: {self.state}"
            )

        sysid_metadata.auto_shutdown = False
        self.initialize_system_id(sysid_metadata, environment_name)

        stream_metadata = StreamMetadata(StreamType.NO_STREAM)

        self.start_acquisition(stream_metadata)
        self.start_system_id_transfer_function(environment_name)

    def run_system_id(
        self, sysid_metadata: SysIdMetadata, environment_name: str
    ) -> None:
        """
        Performs a full system identification for the environment.

        This is used in headless mode as the start button. It initializes the
        system identification metadata, starts the acquisition, and runs the
        system identification noise followed by the transfer function. Each phase
        shuts down automatically when complete, after which the acquisition is
        stopped. If the metadata has a ``stream_file``, each phase is streamed to
        that file.

        Parameters
        ----------
        sysid_metadata : SysIdMetadata
            System identification metadata to run.
        environment_name : str
            User-facing environment name.

        Raises
        ------
        RattlesnakeError
            If the controller is not in the ``ENVIRONMENT_STORE`` state.

        Unit Tests
        ----------
        """
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

    def stop_system_id_run(self, environment_name: str) -> None:
        """
        Stops a system identification preview or run early.

        Used within a headless script or Jupyter notebook to stop the system
        identification early. If the system identification is active, it is
        stopped before the acquisition is stopped. If only the hardware is
        active, the acquisition is stopped.

        Parameters
        ----------
        environment_name : str
            User-facing environment name.

        Raises
        ------
        RattlesnakeError
            If the controller is not in the ``HARDWARE_ACTIVE`` or
            ``SYS_ID_ACTIVE`` state, or if no environment exists with
            ``environment_name``.

        Unit Tests
        ----------
        """
        if self.state == RattlesnakeState.HARDWARE_ACTIVE:
            self.stop_acquisition()
            return
        self.stop_system_id(environment_name)
        self.stop_acquisition()

    # endregion

    # region Acquisition
    def set_stream_metadata(self, stream_metadata: StreamMetadata):
        """
        Store stream metadata for UI use.

        Stores stream metadata on the controller without starting acquisition.
        This is used to preload UI state; ``start_acquisition`` still requires an
        explicit ``StreamMetadata`` object.

        Parameters
        ----------
        stream_metadata : StreamMetadata
            Stream metadata to store.

        Raises
        ------
        RattlesnakeError
            If the controller is not in ``ENVIRONMENT_STORE`` state.
        RattlesnakeError
            If ``stream_metadata`` is not a ``StreamMetadata`` instance.
        RattlesnakeError
            If stream metadata validation fails.

        Unit Tests
        ----------
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

    def start_acquisition(self, stream_metadata: StreamMetadata) -> None:
        """
        Start acquisition and output hardware.

        Validates controller state and stream metadata, initializes streaming,
        sends a ``RUN_HARDWARE`` command to the controller process, waits for
        streaming ready event and acquisition/output active events when blocking
        mode is enabled, and stores the last stream metadata.

        Parameters
        ----------
        stream_metadata : StreamMetadata
            Streaming configuration to use for the acquisition run.

        Raises
        ------
        RattlesnakeError
            If the controller is not in ``ENVIRONMENT_STORE`` state.
        RattlesnakeError
            If ``stream_metadata`` is not a ``StreamMetadata`` instance.
        RattlesnakeError
            If stream metadata validation fails.

        Unit Tests
        ----------
        """
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

        # Store streaming metadata to controller (side note:
        # ControllerProcess decides when/why to stream not StreamingProcess)
        self.log("Setting Stream Metadata")
        self.event_container.streaming_ready_event.clear()
        self.queue_container.streaming_command_queue.put(
            TASK_NAME,
            (
                GlobalCommands.INITIALIZE_STREAMING,
                (stream_metadata, self.hardware_metadata, self.environment_metadata),
            ),
        )

        # Tell controller to start up the hardware, controller takes over
        # logic from here
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

    def stop_acquisition(self) -> None:
        """
        Stop acquisition and output hardware.

        Checks if the acquisition is active, stops the profile manager, sends
        a ``STOP_HARDWARE`` command to the controller process, and waits for the
        controller ready and acquisition/output active events to clear when
        blocking mode is enabled.

        Raises
        ------
        RattlesnakeError
            If the controller state does not permit stopping acquisition.

        Unit Tests
        ----------
        """
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
    def start_environment(self, instructions: EnvironmentInstructions) -> None:
        """
        Start an environment based on the environment name within the instructions.

        Validates the controller state and instruction type, validates environment
        instructions, sends a ``START_ENVIRONMENT`` command to the controller
        process, and waits for the environment active event when blocking mode is
        enabled.

        Parameters
        ----------
        instructions : EnvironmentInstructions
            Instructions used to start the environment.

        Raises
        ------
        RattlesnakeError
            If the controller state does not permit starting an environment.
        RattlesnakeError
            If ``instructions`` is not an ``EnvironmentInstructions`` instance.
        RattlesnakeError
            If instruction validation fails.

        Unit Tests
        ----------
        """
        if self.state not in (
            RattlesnakeState.HARDWARE_ACTIVE,
            RattlesnakeState.ENVIRONMENT_ACTIVE,
        ):
            raise RattlesnakeError(
                f"Invalid state for starting environment: {self.state}"
            )
        if not isinstance(instructions, EnvironmentInstructions):
            raise RattlesnakeError(
                "Start_environment must be contain a valid "
                "EnvironmentInstructions object"
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

    def stop_environment(self, environment_name: str) -> None:
        """
        Stops an active environment.

        Validates that an environment is active, resolves the environment queue
        name, sends a ``STOP_ENVIRONMENT`` command to the controller process, and
        waits for the environment active event to clear when blocking mode is
        enabled.

        Parameters
        ----------
        environment_name : str
            User-facing environment name to stop.

        Raises
        ------
        RattlesnakeError
            If the controller is not in ``ENVIRONMENT_ACTIVE`` state.
        RattlesnakeError
            If no environment exists with ``environment_name``.

        Unit Tests
        ----------
        """
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
    def environment_at_target_level(self, environment_name: str) -> None:
        """
        Starts streaming when stream metadata is set to ``TEST_LEVEL``.

        Sends a ``STREAM_AT_TARGET_LEVEL`` command to the controller process. Used
        internally by environment user interfaces to inform the controller that
        the environment has reached target level so streaming at target level can
        start.

        Parameters
        ----------
        environment_name : str
            User-facing environment name that reached target level.

        Raises
        ------
        RattlesnakeError
            If no environment exists with ``environment_name``.

        Unit Tests
        ----------
        """
        try:
            self.environment_manager.queue_names_dict[environment_name]
        except KeyError:
            raise RattlesnakeError(f"No environments exist for {environment_name} name")

        self.queue_container.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.STREAM_AT_TARGET_LEVEL, environment_name)
        )

    def send_environment_command(
        self, environment_name: str, command: any, data: any
    ) -> None:
        """
        Send an arbitrary command directly to an environment.

        This bypass is intended for UI requests that are safe to execute in any
        controller state. It resolves the environment name, logs the request, and
        forwards the command and payload to the environment command queue.

        Parameters
        ----------
        environment_name : str
            User-facing environment name.
        command : Any
            Environment command to send.
        data : Any
            Payload associated with the command.

        Raises
        ------
        RattlesnakeError
            If no environment exists with ``environment_name``.

        Unit Tests
        ----------
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
    def streaming(self) -> bool:
        """
        Return whether streaming is currently active.

        Returns
        -------
        bool
            ``True`` if the streaming active event is set, otherwise ``False``.

        Unit Tests
        ----------
        """
        return self.event_container.streaming_active_event.is_set()

    @property
    def has_streamed(self) -> bool:
        """
        Return whether stream metadata has been stored.

        Returns
        -------
        bool
            ``True`` if ``last_stream_metadata`` is not ``None``, otherwise
            ``False``.

        Unit Tests
        ----------
        """
        if self.last_stream_metadata:
            return True
        return False

    def start_streaming(self) -> None:
        """
        Start streaming when stream metadata is set to ``MANUAL``.

        Validates that acquisition, an environment, or system identification is
        active and that streaming is not already active. Sends a ``STREAM_MANUAL``
        command to the controller process and waits for the streaming active event
        when blocking mode is enabled.

        Raises
        ------
        RattlesnakeError
            If the controller state does not permit starting streaming.
        RattlesnakeError
            If streaming is already active.

        Unit Tests
        ----------
        """
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

    def stop_streaming(self) -> None:
        """
        Stops streaming.

        Validates that streaming is active, sends a ``STOP_STREAMING`` command to
        the controller process, and waits for the streaming active event to clear
        when blocking mode is enabled.

        Raises
        ------
        RattlesnakeError
            If streaming is not currently active.

        Unit Tests
        ----------
        """
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
    def has_profile(self) -> bool:
        """
        Return whether there is a current profile event list stored to the controller.

        Profile event lists are stored by either initializing them to the controller or
        by running a profile.

        Returns
        -------
        bool
            ``True`` if ``last_profile_event_list`` is not empty, otherwise
            ``False``.

        Unit Tests
        ----------
        """
        if self.last_profile_event_list:
            return True
        return False

    def initialize_profile_event_list(
        self, profile_event_list: List[ProfileEvent]
    ) -> None:
        """
        Store a profile event list to the controller.

        This is mainly used when launching the user interface from an existing
        controller as the user interface will mirror this profile event list.
        Validates that the controller state permits profile storage, validates
        profile events through the environment manager, and stores the list on the
        controller.

        Parameters
        ----------
        profile_event_list : list of ProfileEvent
            Profile events to store.

        Raises
        ------
        RattlesnakeError
            If the controller state does not permit storing a profile.
        RattlesnakeError
            If profile event validation fails.

        Unit Tests
        ----------
        """
        self.log("Settting Profile Event List")

        if self.state not in (
            RattlesnakeState.ENVIRONMENT_STORE,
            RattlesnakeState.HARDWARE_ACTIVE,
        ):
            raise RattlesnakeError(f"Invalid state for storing profile: {self.state}")

        self.environment_manager.validate_profile_events(profile_event_list)
        self.last_profile_event_list = profile_event_list

    def start_profile(self, profile_event_list: List[ProfileEvent]) -> None:
        """
        Start a profile.

        Validates that hardware is active, validates profile events through the
        environment manager and profile manager, clears the controller ready event,
        starts the profile manager, waits for controller ready event when blocking
        mode is enabled, and stores the profile event list.

        Parameters
        ----------
        profile_event_list : list of ProfileEvent
            Profile events to start.

        Raises
        ------
        RattlesnakeError
            If the controller state does not permit starting a profile.
        RattlesnakeError
            If profile event validation fails.

        Unit Tests
        ----------
        """
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

    def stop_profile(self) -> None:
        """
        Stop the active profile.

        Clears the controller ready event, commands the profile manager to stop,
        and waits for controller ready event when blocking mode is enabled.
        """
        self.log("Stopping Profile")
        self.event_container.controller_ready_event.clear()
        self.profile_manager.stop_profile()

        if self.blocking:
            ready_event_list = [self.event_container.controller_ready_event]
            active_event_list = []
            self.wait_for_events(ready_event_list, active_event_list)

    # endregion

    # region Shutdown
    def shutdown(self) -> None:
        """
        Shut down the controller and all subprocesses.

        Stops acquisition if hardware, environments, or system identification are
        active. Then sends ``QUIT`` commands to controller, acquisition, output, and
        streaming processes, joins each process, closes environment processes through
        the environment manager, and shuts down the log file process. Processes that
        do not shutdown within the timeout will be force closed.

        This function is not
        tied to the user interface shutdown and must be called after the execution loop
        is finished. It is recommended to use ``with RattlesnakeController as rattlesnake:``
        to automatically shutdown the controller within a python script.

        Unit Tests
        ----------
        """
        if self._shutdown_complete:
            return
        if not self._init_complete:
            # Weird edge case with jupyter notebook where sometimes shutdown can happen
            # before __init__ completes (computer dependent). Needs more investigation
            # but this patch works for now.
            self._shutdown_complete = True
            return
        print("Rattlesnake Shutting Down")
        if self.state in (
            RattlesnakeState.HARDWARE_ACTIVE,
            RattlesnakeState.ENVIRONMENT_ACTIVE,
            RattlesnakeState.SYS_ID_ACTIVE,
        ):
            try:
                self.set_blocking()
                self.stop_acquisition()
            except:
                print("Failed Acquisition Shutdown")
        # Close out of acquisition, output, streaming process
        self.queue_container.log_file_queue.put(
            f"{datetime.now()}: Joining Controller Process\n"
        )
        self.gui_queue_cleanup_close_event.set()
        self.gui_queue_cleanup_thread.join(timeout=CLOSE_TIMEOUT)
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
        self.log_file_process.join(timeout=CLOSE_TIMEOUT)
        if self.log_file_process.is_alive():
            self.log_file_process.terminate()
            self.log_file_process.join()

        try:
            self.controller_queue_name_manager.shutdown()
        except Exception:
            print("Failed Manager Shutdown")

        self._shutdown_complete = True

    def log(self, string) -> None:
        """
        Sends a message to the log file subprocess.

        Formats the supplied message with the current timestamp and controller task
        name, then places it on the log file queue.

        Parameters
        ----------
        string : str
            Message to write to the log file queue.
        """
        self.queue_container.log_file_queue.put(
            f"{datetime.now()}: {TASK_NAME} -- {string}\n"
        )

    # endregion


# endregion
