import copy
from datetime import datetime
import multiprocessing as mp
import multiprocessing.synchronize  # pylint: disable=unused-import
import threading
from typing import Dict, List

from rattlesnake.utilities import (
    EventContainer,
    GlobalCommands,
    QueueContainer,
    RattlesnakeError,
)
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.abstract_environment import (
    EnvironmentInstructions,
    EnvironmentMetadata,
)
from rattlesnake.environment.environment_registry import (
    ENVIRONMENT_PROCESS,
    SYSID_ENVIRONMENTS,
)
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.typehint import EventType
from rattlesnake.process.abstract_sysid_data_analysis import (
    SysIdMetadata,
    SysIdDataPackage,
)

TASK_NAME = "Environment Manager"
CLOSE_TIMEOUT = 5


# region Manager
class EnvironmentManager:
    """
    Stores the environment information and manages the environment processes.

    This class is responsible for spinning up and shutting down environment processes
    given a list of environments metadata objects. This class deals with putting
    commands into environment command queues while the ``RattlesnakeController`` is in
    the ``ENVIRONMENT_STORE`` state. This class also validates ``EnvironmentMetadata``,
    ``SysIdMetadata``, ``SysIdDataPackage``, ``EnvironmentInstructions``, and
    ``ProfileEvent`` objects.

    Each environment is assigned a queue name which is required since the number of
    environment queues is set at startup. The environments also have unique environment
    names which the user sets when defining them.

    Parameters
    ----------
    queue_container : QueueContainer
        Container holding the controller, GUI, log file, and environment queues.
    event_container : EventContainer
        Container holding the environment active, environment ready, acquisition, and
        output events.
    threaded : bool
        If ``True``, environments are started as threads. If ``False``, environments are
        started as multiprocessing processes.

    Attributes
    ----------
    queue_names : list of str
        Queue names currently assigned to an environment.
    environment_names : dict of str to str
        Mapping from queue names to environment names.
    environment_types : dict of str to EnvironmentType
        Mapping from queue names to environment types.
    environment_metadata : dict of str to EnvironmentMetadata
        Mapping from queue names to environment metadata. This is stored by the caller
        and is not updated by ``initialize_environments``.
    environment_processes : dict of str to threading.Thread or multiprocessing.Process
        Mapping from queue names to running environment processes.
    queue_container : QueueContainer
        Container holding the controller, GUI, log file, and environment queues.
    event_container : EventContainer
        Container holding the environment and acquisition/output events.
    environment_active_events : dict of str to EventType
        Mapping from queue names to events set while an environment is active.
    environment_ready_events : dict of str to EventType
        Mapping from queue names to events set when an environment is ready.
    environment_close_events : dict of str to EventType
        Mapping from queue names to events used to force close an environment.
    environment_sysid_active_events : dict of str to EventType
        Mapping from queue names to events set while system identification is active.
    environment_sysid_stored_events : dict of str to EventType
        Mapping from queue names to events set when system identification data is
        stored.
    ping_alive_event : EventType
        Watchdog event passed to the environment processes.
    new_process : threading.Thread callable or multiprocessing.Process callable
        Class used to start environments, selected by ``threaded``.
    new_event : threading.Event callable or multiprocessing.Event callable
        Factory that returns a new ``EventType``, selected by ``threaded``.
    threaded : bool
        Whether environments are started as threads.

    Unit Tests
    ----------
    """

    def __init__(
        self,
        queue_container: QueueContainer,
        event_container: EventContainer,
        threaded: bool,
    ) -> None:
        """
        Initializes a blank environment manager without environments.

        Stores the queues and events containers and initializes dicts used to sort
        environments.

        Parameters
        ----------
        queue_container : QueueContainer
            Container holding the controller, GUI, log file, and environment queues.
        event_container : EventContainer
            Container holding the environment and acquisition/output events.
        threaded : bool
            If ``True``, environments are started as threads. If ``False``, environments
            are started as multiprocessing processes.

        Unit Tests
        ----------
        """
        self.queue_names = []  # Static name for dictionary keys, process names, etc
        self.environment_names = {}  # Name of environment for Ui purposes
        self.environment_types = {}
        self.environment_metadata = {}
        self.environment_processes = {}
        self.queue_container = queue_container
        self.event_container = event_container
        self.environment_active_events = event_container.environment_active_events
        self.environment_ready_events = event_container.environment_ready_events
        self.environment_close_events = event_container.environment_close_events
        self.environment_sysid_active_events = (
            event_container.environment_sysid_active_events
        )
        self.environment_sysid_stored_events = (
            event_container.environment_sysid_stored_events
        )
        self.ping_alive_event = event_container.ping_alive_event
        self._threaded = threaded
        if threaded:
            self.new_process = threading.Thread
            self.new_event = threading.Event
        else:
            self.new_process = mp.Process
            self.new_event = mp.Event

    @property
    def threaded(self) -> bool:
        """
        Return whether environments should run on single threads.

        Returns
        -------
        bool
            ``True`` if environments are threads, otherwise ``False``.

        Unit Tests
        ----------
        """
        return self._threaded

    @property
    def available_queues(self) -> List[str]:
        """
        Return the queue names that are mapped to an existing environment.

        Returns
        -------
        list of str
            Queue names that can be assigned to a new environment.

        Unit Tests
        ----------
        """
        all_queue_names = list(self.queue_container.environment_command_queues.keys())
        return [
            queue_name
            for queue_name in all_queue_names
            if queue_name not in self.queue_names
        ]

    @property
    def num_queues(self) -> int:
        """
        Return the total number of environment queues.

        Returns
        -------
        int
            Number of assigned and available queues, which is the maximum number of
            environments.

        Unit Tests
        ----------
        """
        return len(self.queue_names) + len(self.available_queues)

    @property
    def queue_names_dict(self) -> Dict[str, str]:
        """
        Return a dict for looking up queue names from environment names.

        Returns
        -------
        dict of str to str
            Mapping from environment names to queue names.

        Unit Tests
        ----------
        """
        return {
            self.environment_names[queue_name]: queue_name
            for queue_name in self.queue_names
        }

    def log(self, message: str) -> None:
        """
        Write a message to the log file.

        This function puts a message onto the ``log_file_queue`` so it will eventually
        be written to the log file.

        Parameters
        ----------
        message : str
            A message that will be written to the log file.

        Unit Tests
        ----------
        """
        self.queue_container.log_file_queue.put(
            f"{datetime.now()}: {TASK_NAME} -- {message}\n"
        )

    # region State Sync
    @property
    def ready_event_list(self) -> List[EventType]:
        """
        Return the ready events of the stored environments.

        Returns
        -------
        list of EventType
            Ready event for each queue name that has an assigned environment.

        Unit Tests
        ----------
        """
        ready_event_list = [
            self.environment_ready_events[queue_name] for queue_name in self.queue_names
        ]
        return ready_event_list

    @property
    def active_event_list(self) -> List[EventType]:
        """
        Return the active events of the stored environments.

        Returns
        -------
        list of EventType
            Active event for each queue name that has an assigned environment.

        Unit Tests
        ----------
        """
        active_event_list = [
            self.environment_active_events[queue_name]
            for queue_name in self.queue_names
        ]
        return active_event_list

    @property
    def acquisition_ready_environments(self) -> Dict[str, bool]:
        """
        Return whether each environment is able to start.

        System identification environments are ready once their system identification
        data is stored. All other environments are always ready.

        Returns
        -------
        dict of str to bool
            Mapping from queue names to whether the environment is ready.

        Unit Tests
        ----------
        """
        acquisition_ready_environments = {}
        for queue_name, environment_type in self.environment_types.items():
            if environment_type in SYSID_ENVIRONMENTS:
                bool_ready = self.environment_sysid_stored_events[queue_name].is_set()
                acquisition_ready_environments[queue_name] = bool_ready
            else:
                acquisition_ready_environments[queue_name] = True

        return acquisition_ready_environments

    @property
    def sysid_active_environments(self) -> List[str]:
        """
        Return the environments that are running system identification.

        Returns
        -------
        list of str
            Environment names of the environments with system identification active.

        Unit Tests
        ----------
        """
        sysid_active_list = []
        for queue_name, environment_name in self.environment_names.items():
            if self.environment_sysid_active_events[queue_name].is_set():
                sysid_active_list.append(environment_name)

        return sysid_active_list

    @property
    def environment_active_environments(self) -> List[str]:
        """
        Return the environments that are currently active.

        Returns
        -------
        list of str
            Environment names of the active environments.

        Unit Tests
        ----------
        """
        environment_active_list = []
        for queue_name, environment_name in self.environment_names.items():
            if self.environment_active_events[queue_name].is_set():
                environment_active_list.append(environment_name)

        return environment_active_list

    def clear_sysid_events(self) -> None:
        """
        Clear the system identification stored event of every environment.

        Unit Tests
        ----------
        """
        for queue_name in self.queue_names:
            self.environment_sysid_stored_events[queue_name].clear()

    def initialize_hardware(self, hardware_metadata: HardwareMetadata) -> None:
        """
        Send hardware metadata to every environment.

        Clears the ready events and queues an ``INITIALIZE_HARDWARE`` command for each
        environment. The ``RattlesnakeController`` is responsible for waiting on
        ``ready_event_list``.

        Parameters
        ----------
        hardware_metadata : HardwareMetadata
            Hardware metadata to send to the environments.

        Unit Tests
        ----------
        """
        self.log("Initializing Hardware")
        for queue_name in self.queue_names:
            self.environment_ready_events[queue_name].clear()
            self.queue_container.environment_command_queues[queue_name].put(
                TASK_NAME, (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata)
            )

    def initialize_environments(
        self,
        metadata_list: List[EnvironmentMetadata],
        hardware_metadata: HardwareMetadata,
    ) -> Dict[str, EnvironmentMetadata]:
        """
        Spin up environment processes corresponding to environments within the metadata
        list.

        Existing environments with the same name (case-insensitive) and type keep their
        process and are sent the new metadata. Other existing environments are shut
        down, and a new process is started for each remaining metadata object. Once
        processes are spun up, they are first given hardware metadata and then
        environment metadata.

        Parameters
        ----------
        metadata_list : list of EnvironmentMetadata
            Environment metadata objects to initialize.
        hardware_metadata : HardwareMetadata
            Hardware metadata to send to the environments.

        Returns
        -------
        dict of str to EnvironmentMetadata
            Mapping from assigned queue names to environment metadata. This is used by
            the acquisition, output, and user interface to keep track of queue names.

        Raises
        ------
        RattlesnakeError
            If there are no available queues for a new environment.

        Unit Tests
        ----------
        """
        # TODO: Remove this, set an equivalency to a sysid data container to
        # see if this needs to be cleared
        self.clear_sysid_events()

        self.log("Initializing Environments")
        mapped_queue_names = set()
        extra_metadata = []
        environment_metadata_dict = {}

        # Check if there is an existing process with the same name
        # If there is, give it the new metadata
        existing_name_to_queue = {
            name.casefold(): queue_name
            for queue_name, name in self.environment_names.items()
        }

        for metadata in metadata_list:
            environment_type = metadata.environment_type
            environment_name = metadata.environment_name

            queue_name = existing_name_to_queue.get(environment_name.casefold())

            # Shut down the environment if the name matches but the type does not
            if (
                queue_name is not None
                and self.environment_types.get(queue_name) != environment_type
            ):
                queue_name = None

            if queue_name is None:
                extra_metadata.append(metadata)
                continue

            self.log(f"Assigning {environment_name} to {queue_name} Queue")
            self.environment_types[queue_name] = environment_type
            self.environment_names[queue_name] = environment_name
            environment_metadata_dict[queue_name] = metadata
            self.queue_container.environment_command_queues[
                queue_name
            ].assign_environment(environment_name)
            self.queue_container.environment_command_queues[queue_name].put(
                TASK_NAME, (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata)
            )
            self.environment_ready_events[queue_name].clear()
            self.queue_container.environment_command_queues[queue_name].put(
                TASK_NAME, (GlobalCommands.INITIALIZE_ENVIRONMENT, metadata)
            )

            mapped_queue_names.add(queue_name)

        # Close existing processes that dont have metadata associated with them
        unmapped_queue_names = set(self.environment_types.keys()) - mapped_queue_names
        for queue_name in unmapped_queue_names:
            self.remove_environment(queue_name)

        # Add process for metadata that needs a new process. Could do this in loop above
        # but I want to clear up queue_names before assigning new ones
        for metadata in extra_metadata:
            queue_name = self.add_environment(metadata, hardware_metadata)
            environment_metadata_dict[queue_name] = metadata

        return environment_metadata_dict

    def initialize_system_id(
        self, sysid_metadata: SysIdMetadata, queue_name: str
    ) -> Dict[str, EnvironmentMetadata]:
        """
        Send system identification metadata to an environment.

        Stores the system identification metadata to the environment and syncs the
        controller's metdata dict with the environment.

        Parameters
        ----------
        sysid_metadata : SysIdMetadata
            System identification metadata to send to the environment.
        queue_name : str
            Queue name of the environment.

        Returns
        -------
        dict of str to EnvironmentMetadata
            Copy of ``environment_metadata`` with the system identification metadata
            assigned to the environment.

        Unit Tests
        ----------
        """
        environment_metadata_dict = copy.deepcopy(self.environment_metadata)
        environment_metadata = environment_metadata_dict[queue_name]
        environment_metadata.sysid_metadata = sysid_metadata

        self.queue_container.environment_command_queues[queue_name].put(
            TASK_NAME, (GlobalCommands.INITIALIZE_SYSTEM_ID, sysid_metadata)
        )

        return environment_metadata_dict

    def find_sysid_compatible_environments(self, environment_name: str) -> List[str]:
        """
        Find the environments that can share system identification data.

        Environments are compatible if both are system identification environments with
        the same global response and reference channels and transformation matrices.

        Parameters
        ----------
        environment_name : str
            Name of the environment to find compatible environments for.

        Returns
        -------
        list of str
            Names of the compatible environments. The list is empty if
            ``environment_name`` is not a system identification environment.

        Raises
        ------
        RattlesnakeError
            If no environment exists with the given name.

        Unit Tests
        ----------
        """
        try:
            source_queue_name = self.queue_names_dict[environment_name]
        except KeyError:
            raise RattlesnakeError(f"No environments exist for {environment_name} name")

        if self.environment_types[source_queue_name] not in SYSID_ENVIRONMENTS:
            return []

        source_metadata = self.environment_metadata[source_queue_name]

        compatible_names = []
        for target_queue_name, name in self.environment_names.items():
            if target_queue_name == source_queue_name:
                continue
            if self.environment_types[target_queue_name] not in SYSID_ENVIRONMENTS:
                continue
            target_metadata = self.environment_metadata[target_queue_name]
            if target_metadata is None:
                continue
            if source_metadata.sysid_shared(target_metadata):
                compatible_names.append(name)

        return compatible_names

    # endregion

    # region Validation
    def validate_environment_metadata(
        self,
        metadata_list: List[EnvironmentMetadata],
        hardware_metadata: HardwareMetadata,
    ) -> None:
        """
        Validate environment metadata before it is initialized.

        Each metadata object also validates itself against the hardware metadata, which
        can raise additional errors.

        Parameters
        ----------
        metadata_list : list of EnvironmentMetadata
            Environment metadata objects to validate.
        hardware_metadata : HardwareMetadata
            Hardware metadata that the environment metadata is validated against.

        Raises
        ------
        RattlesnakeError
            If there are more environments than environment queues.
        RattlesnakeError
            If an object is not an ``EnvironmentMetadata`` object.
        RattlesnakeError
            If environment names are not unique (case-insensitive).

        Unit Tests
        ----------
        """
        # Check if there are available queues
        if len(metadata_list) > self.num_queues:
            raise RattlesnakeError(
                "Not enough environment command queues. Increase "
                "max_environments in rattlesnake.py"
            )

        # Validate individual environments
        environment_name_set = set()
        for metadata in metadata_list:
            # Check for valid class
            if not isinstance(metadata, EnvironmentMetadata):
                raise RattlesnakeError(
                    "Rattlesnake.set_environment was given an object that "
                    "is not an EnvironmentMetadata class"
                )
            # Check for unique name (case-insensitive)
            environment_name = metadata.environment_name
            normalized_name = environment_name.casefold()
            if normalized_name in environment_name_set:
                raise RattlesnakeError(
                    "Environment names must be unique (case-insensitive)"
                )
            environment_name_set.add(normalized_name)
            # Validate metadata
            metadata.validate(hardware_metadata)

    def validate_system_id_metadata(
        self,
        sysid_metadata: SysIdMetadata,
        hardware_metadata: HardwareMetadata,
        environment_name: str,
    ) -> str:
        """
        Validate system identification metadata for an environment.

        Parameters
        ----------
        sysid_metadata : SysIdMetadata
            System identification metadata to validate.
        hardware_metadata : HardwareMetadata
            Currently stored hardware metadata.
        environment_name : str
            Name of the environment the metadata is for.

        Returns
        -------
        str
            Queue name of the environment. This is used to send system identification
            metadata to the correct environment after validation.

        Raises
        ------
        RattlesnakeError
            If ``sysid_metadata`` is not a ``SysIdMetadata`` object.
        RattlesnakeError
            If no environment exists with the given name.
        RattlesnakeError
            If the environment is not a system identification environment.

        Unit Tests
        ----------
        """
        if not isinstance(sysid_metadata, SysIdMetadata):
            raise RattlesnakeError(
                "Rattlesnake.initialize_system_id was not given a SysIdMetadata object"
            )
        try:
            queue_name = self.queue_names_dict[environment_name]
        except KeyError:
            raise RattlesnakeError(
                f"No environments exist for {environment_name} instruction"
            )
        environment_type = self.environment_types[queue_name]
        if environment_type not in SYSID_ENVIRONMENTS:
            raise RattlesnakeError(
                f"{environment_name} is a {environment_type} environment "
                "which does not require system identification"
            )

        return queue_name

    def validate_system_id_package(
        self, environment_name: str, data_package: SysIdDataPackage
    ) -> str:
        """
        Validate a system identification data package for an environment.

        The package is only compared to the environment by its number of response and
        reference channels.

        Parameters
        ----------
        environment_name : str
            Name of the environment the data is for.
        data_package : SysIdDataPackage
            System identification data to validate.

        Returns
        -------
        str
            Queue name of the environment. This is used to send the system identification
            package to the environment after validation.

        Raises
        ------
        RattlesnakeError
            If no environment exists with the given name.
        RattlesnakeError
            If the environment is not a system identification environment.
        RattlesnakeError
            If the environment does not have system identification metadata stored.
        RattlesnakeError
            If ``data_package`` is not a ``SysIdDataPackage`` object.
        RattlesnakeError
            If the number of response or reference channels does not match the
            environment.

        Unit Tests
        ----------
        """
        try:
            queue_name = self.queue_names_dict[environment_name]
        except KeyError:
            raise RattlesnakeError(f"No environments exist for {environment_name} name")

        if self.environment_types[queue_name] not in SYSID_ENVIRONMENTS:
            raise RattlesnakeError(
                f"{environment_name} is not a system identification environment"
            )

        if not isinstance(
            self.environment_metadata[queue_name].sysid_metadata, SysIdMetadata
        ):
            raise RattlesnakeError(
                f"{environment_name} does not have a system identification metadata object loaded to it."
            )

        if not isinstance(data_package, SysIdDataPackage):
            raise RattlesnakeError(
                "Rattlesnake was provided an sysid_data_package that was "
                "not an SysIdDataPackage type"
            )

        data_package.validate()

        environment_metadata = self.environment_metadata[queue_name]
        if (
            environment_metadata.num_response_channels
            != data_package.num_response_channels
        ):
            raise RattlesnakeError(
                "The system identification data has "
                f"{data_package.num_response_channels} response channels "
                "and the environment has "
                f"{environment_metadata.num_response_channels} response "
                "channels"
            )

        if (
            environment_metadata.num_reference_channels
            != data_package.num_reference_channels
        ):
            raise RattlesnakeError(
                "The system identification data has "
                f"{data_package.num_reference_channels} reference "
                "channels and the environment has "
                f"{environment_metadata.num_reference_channels} reference "
                "channels"
            )

        return queue_name

    def validate_environment_instructions(
        self, instructions: EnvironmentInstructions
    ) -> str:
        """
        Validate environment instructions before an environment is started.

        The instructions object also validates itself, which can raise additional
        errors.

        Parameters
        ----------
        instructions : EnvironmentInstructions
            Instructions to validate.

        Returns
        -------
        str
            Queue name of the environment. This is used to send the instructions to the
            environment after validation.

        Raises
        ------
        RattlesnakeError
            If ``instructions`` is not an ``EnvironmentInstructions`` object.
        RattlesnakeError
            If no environment exists with the given name.
        RattlesnakeError
            If the instructions type does not match the environment type.
        RattlesnakeError
            If the environment requires system identification data that is not stored.

        Unit Tests
        ----------
        """
        # Validate class
        if not isinstance(instructions, EnvironmentInstructions):
            raise RattlesnakeError(
                "Rattlesnake was provided an environment_instruction that "
                "was not an EnvironmentInstructions type"
            )
        # Validate name
        environment_name = instructions.environment_name
        try:
            queue_name = self.queue_names_dict[environment_name]
        except KeyError:
            raise RattlesnakeError(
                f"No environments exist for {environment_name} instruction"
            )
        # Validate type
        environment_type = instructions.environment_type
        if environment_type != self.environment_types[queue_name]:
            raise RattlesnakeError(
                f"Instructions for {environment_name} is the wrong type "
                f"for {environment_type} vs "
                f"{self.environment_types[queue_name]}"
            )

        if environment_type in SYSID_ENVIRONMENTS:
            if not self.environment_sysid_stored_events[queue_name].is_set():
                raise RattlesnakeError(
                    f"{environment_name} requires a system identification "
                    "before starting"
                )
        # Validate instruction
        instructions.validate()

        return queue_name

    def validate_profile_events(self, profile_events_list: List[ProfileEvent]) -> None:
        """
        Validate profile events and assign their queue names and environment types.

        The profile events come from the UI/Terminal, which has no idea which queue was
        assigned to an environment, so the environment names are validated here. Each
        event is given a queue name. Events with the environment name ``"Global"`` are
        assigned ``"Global"`` as their queue name and environment type.

        Parameters
        ----------
        profile_events_list : list of ProfileEvent
            Profile events to validate.

        Raises
        ------
        RattlesnakeError
            If an object is not a ``ProfileEvent`` object.
        RattlesnakeError
            If no environment exists with the name given by an event.

        Unit Tests
        ----------
        """
        for profile_event in profile_events_list:
            # Validate class
            if not isinstance(profile_event, ProfileEvent):
                raise RattlesnakeError(
                    "The profile_events_list contains an object that is "
                    "not a ProfileEvent type"
                )
            # Validate environment_name and assign queue_name and type
            environment_name = profile_event.environment_name
            if environment_name == "Global":
                profile_event._queue_name = "Global"
                profile_event._environment_type = "Global"
            else:
                try:
                    queue_name = self.queue_names_dict[environment_name]
                except KeyError:
                    raise RattlesnakeError(
                        f"No environments exist for {environment_name} "
                        "when validating instruction"
                    )
                profile_event._queue_name = queue_name
                profile_event._environment_type = self.environment_types[queue_name]

    # endregion

    # region Environment
    def clear_environments(self):
        """
        Shut down all environments and clear their stored information.

        Unit Tests
        ----------
        """
        self.queue_names = []
        self.environment_types = {}
        self.environment_names = {}
        self.environment_metadata = {}
        self.close_environments()
        self.environment_processes = {}

    def add_environment(
        self, metadata: EnvironmentMetadata, hardware_metadata: HardwareMetadata
    ) -> str:
        """
        Start a process for a new environment.

        Assigns the first available queue to the environment, starts its process, and
        sends it the hardware and environment metadata.

        Parameters
        ----------
        metadata : EnvironmentMetadata
            Environment metadata of the new environment.
        hardware_metadata : HardwareMetadata
            Hardware metadata to send to the environment.

        Returns
        -------
        str
            Queue name assigned to the environment. This is used to build the
            environment metadata dict stored by the manager.

        Raises
        ------
        RattlesnakeError
            If there are no available queues.

        Unit Tests
        ----------
        """

        # Find the first available queue for the environment
        queue_name = None
        for queue_name in self.available_queues:
            if queue_name not in self.queue_names:
                queue_name = str(queue_name)
                break

        if queue_name is None:
            raise RattlesnakeError(
                "Not enough environment command queues. Increase "
                "max_environments in rattlesnake.py"
            )

        environment_type = metadata.environment_type
        environment_name = metadata.environment_name

        self.queue_container.environment_command_queues[queue_name].assign_environment(
            environment_name
        )
        self.environment_close_events[queue_name].clear()

        # Figure out what type of environment to add
        environment_process_function = ENVIRONMENT_PROCESS[environment_type]
        environment_process = self.new_process(
            target=environment_process_function,
            args=(
                environment_name,
                queue_name,
                self.queue_container.environment_command_queues[queue_name],
                self.queue_container.gui_update_queue,
                self.queue_container.controller_command_queue,
                self.queue_container.log_file_queue,
                self.queue_container.environment_data_in_queues[queue_name],
                self.queue_container.environment_data_out_queues[queue_name],
                self.event_container.acquisition_active_event,
                self.event_container.output_active_event,
                self.environment_active_events[queue_name],
                self.environment_ready_events[queue_name],
                self.environment_close_events[queue_name],
                self.environment_sysid_active_events[queue_name],
                self.environment_sysid_stored_events[queue_name],
                self.ping_alive_event,
                self.threaded,
            ),
        )
        environment_process.start()

        # Store the environment to the container
        self.log(f"Assigning {environment_name} to {queue_name} Queue")
        self.queue_names.append(queue_name)
        self.environment_names[queue_name] = environment_name
        self.environment_types[queue_name] = environment_type
        self.environment_processes[queue_name] = environment_process
        self.queue_container.environment_command_queues[queue_name].put(
            TASK_NAME, (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata)
        )
        self.environment_ready_events[
            queue_name
        ].clear()  # This looks weird, the event is set in the next line
        self.queue_container.environment_command_queues[queue_name].put(
            TASK_NAME, (GlobalCommands.INITIALIZE_ENVIRONMENT, metadata)
        )
        self.environment_active_events[queue_name].clear()
        self.environment_sysid_active_events[queue_name].clear()
        self.environment_sysid_stored_events[queue_name].clear()

        return queue_name

    # endregion

    # region Shutdown
    def remove_environment(self, queue_name: str) -> None:
        """
        Shut down an environment and clear its stored information.

        Sends a ``QUIT`` command and waits for the process to join. If it is still
        alive, its close event is set. A process that still does not join is terminated,
        while a thread is left running since threads cannot be terminated.

        Parameters
        ----------
        queue_name : str
            Queue name of the environment to remove.

        Raises
        ------
        RattlesnakeError
            If ``queue_name`` is not assigned to an environment.

        Unit Tests
        ----------
        """
        # Check if index corresponds to an existing environment
        if queue_name in self.queue_names:
            self.log(
                f"Removing {self.environment_names[queue_name]}, "
                f"Clearing up {queue_name} Queue"
            )
            self.queue_names.remove(queue_name)
            self.environment_names.pop(queue_name, None)
            self.environment_types.pop(queue_name, None)
            self.environment_metadata.pop(queue_name, None)

            # Join environment process
            self.queue_container.environment_command_queues[queue_name].put(
                TASK_NAME, (GlobalCommands.QUIT, None)
            )
            self.queue_container.log_file_queue.put(
                "{:}: Joining {:} Process\n".format(datetime.now(), queue_name)
            )
            self.environment_processes[queue_name].join(timeout=5)
            if self.environment_processes[queue_name].is_alive():
                self.queue_container.log_file_queue.put(
                    f"{datetime.now()}: Force Closing {queue_name} Process\n"
                )
                self.environment_close_events[queue_name].set()
                self.environment_processes[queue_name].join(timeout=CLOSE_TIMEOUT)
                if (
                    self.environment_processes[queue_name].is_alive()
                    and not self.threaded
                ):
                    self.environment_processes[queue_name].terminate()
                    self.environment_processes[queue_name].join()

            # Remove environment queue and process
            self.environment_processes.pop(queue_name, None)
        else:
            raise RattlesnakeError(
                f"Invalid control name: {queue_name}. Must be mapped to available queue"
            )

    def close_environments(self) -> None:
        """
        Shut down every environment process and flush its command queue.

        Sends a ``QUIT`` command to each process and waits for it to join. If it is
        still alive, its close event is set. A process that still does not join is
        terminated, while a thread is left running since threads cannot be terminated.

        This does not clear the stored environment information. Use
        ``clear_environments`` to also clear it.

        Unit Tests
        ----------
        """
        for queue_name, environment_process in self.environment_processes.items():
            self.queue_container.environment_command_queues[queue_name].put(
                TASK_NAME, (GlobalCommands.QUIT, None)
            )
            self.queue_container.log_file_queue.put(
                "{:}: Joining {:} Process\n".format(datetime.now(), queue_name)
            )
            environment_process.join(timeout=CLOSE_TIMEOUT)
            if environment_process.is_alive():
                self.queue_container.log_file_queue.put(
                    f"{datetime.now()}: Force Closing {queue_name} Process\n"
                )
                self.environment_close_events[queue_name].set()
                environment_process.join(timeout=CLOSE_TIMEOUT)
                if environment_process.is_alive() and not self.threaded:
                    environment_process.terminate()
                    environment_process.join()
            self.queue_container.environment_command_queues[queue_name].flush(TASK_NAME)

    # endregion


# endregion
