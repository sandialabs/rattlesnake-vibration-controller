# Global Files

# Main

## Engine
<!---
MARK: Engine
--->
    """
    Main Rattlesnake controller engine.

    This module defines the high-level controller object responsible for
    constructing queues and events, starting controller subprocesses, managing
    hardware and environment metadata, coordinating acquisition and output,
    handling system identification workflows, managing streaming and profiles,
    and shutting down the controller cleanly.
    """

### Rattlesnake State
<!---
MARK: Rattlesnake State
--->
    class RattlesnakeState
    """
    Enumeration of high-level controller states.

    The controller state is computed from stored metadata and process activity
    events. It is used to validate whether controller operations are allowed.

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
    test_rattlesnake_state
        Verifies that the controller reports the expected state for several
        combinations of stored hardware metadata, stored environment metadata,
        acquisition activity, output activity, and environment activity. This
        confirms that state priority and fallback behavior work as expected.
    """

### Rattlesnake Controller
<!---
MARK: Rattlesnake Controller
--->
    class RattlesnakeController
    """
    High-level controller for the Rattlesnake vibration controller.

    ``RattlesnakeController`` owns the process queues, synchronization events,
    subprocess handles, environment manager, profile manager, and stored
    metadata needed to run vibration control tests. It provides the public API
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
        subprocess ready or active events. Defaults to ``20``.

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
    test_rattlesnake_init
        Verifies that the controller initializes successfully in threaded and
        non-threaded modes. Also verifies that startup event waiting occurs
        when blocking mode is enabled.

    test_rattlesnake_properties
        Verifies that initialization stores the requested threaded mode,
        blocking state, timeout value, default hardware metadata, and default
        environment metadata.
    """

    def __init__
    """
    Initialize a Rattlesnake controller.

    Creates the log process, queue container, event container, controller
    process, acquisition process, output process, streaming process,
    environment manager, and profile manager. Depending on ``threaded``, worker
    components are created as either threads or multiprocessing processes. If
    blocking mode is enabled, initialization waits for process ready events.

    Parameters
    ----------
    threaded : bool, optional
        If ``True``, use threads and thread-compatible queues and events. If
        ``False``, use multiprocessing processes and queues.
    timeout : float, optional
        Timeout in seconds for blocking event waits.

    Unit Tests
    ----------
    test_rattlesnake_init
        Verifies that a controller object can be constructed in threaded and
        non-threaded modes, with blocking enabled or disabled. Confirms that
        blocking initialization calls ``wait_for_events``.
    """

    property
    def state
    """
    Return the current high-level controller state.

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
    test_rattlesnake_state
        Verifies that the controller returns ``INIT`` with no stored metadata,
        ``HARDWARE_STORE`` after hardware metadata is stored,
        ``ENVIRONMENT_STORE`` after environment metadata is stored,
        ``HARDWARE_ACTIVE`` when acquisition and output are active, and
        ``ENVIRONMENT_ACTIVE`` when an environment active event is also set.
    """

    property
    def threaded
    """
    Return whether the controller is running in threaded mode.

    Returns
    -------
    bool
        ``True`` if worker components are threads, otherwise ``False``.

    Unit Tests
    ----------
    test_rattlesnake_properties
        Verifies that this property matches the ``threaded`` argument supplied
        during controller initialization.
    """

    property
    def blocking
    """
    Return whether controller operations wait for subprocess events.

    Returns
    -------
    bool
        ``True`` if operations wait for ready and active events, otherwise
        ``False``.

    Unit Tests
    ----------
    test_rattlesnake_properties
        Verifies that this property reflects the controller blocking mode,
        including after ``clear_blocking`` has been called.
    """

    property
    def timeout
    """
    Return the controller event wait timeout.

    Returns
    -------
    float
        Timeout in seconds used by blocking wait operations.

    Unit Tests
    ----------
    test_rattlesnake_properties
        Verifies that this property matches the timeout supplied during
        controller initialization.
    """

    property
    def is_alive
    """
    Return whether the watchdog alive event is set.

    The alive event can be used by worker processes to reset the controller
    timeout during long operations.

    Returns
    -------
    bool
        ``True`` if the ping-alive event is set, otherwise ``False``.
    """

    def set_alive
    """
    Set the watchdog alive event.

    Marks the controller as having received a keep-alive ping from a worker
    process during a potentially long blocking operation.
    """

    def clear_alive
    """
    Clear the watchdog alive event.

    Clears the keep-alive signal after it has been observed by
    ``wait_for_events``.
    """

    def set_blocking
    """
    Enable blocking controller operations.

    When blocking mode is enabled, public API methods wait for the relevant
    ready and active events after sending commands to subprocesses.
    """

    def clear_blocking
    """
    Disable blocking controller operations.

    When blocking mode is disabled, public API methods send commands and
    return without waiting for subprocess confirmation.

    Unit Tests
    ----------
    test_rattlesnake_init
        Uses this method to configure non-blocking controller instances.

    test_rattlesnake_properties
        Verifies the resulting blocking state.
    """

    def wait_for_events
    """
    Wait for process ready and active events.

    Blocks until all supplied ready events are set and all supplied active
    events match ``active_event_check``. If the controller receives an alive
    ping, the timeout timer is reset. If the timeout expires before all
    required event conditions are satisfied, all ready events are set and a
    ``RattlesnakeError`` is raised.

    Parameters
    ----------
    ready_event_list : list of multiprocessing.synchronize.Event
        Events that subprocesses set to confirm they have completed an
        operation.
    active_event_list : list of multiprocessing.synchronize.Event
        Events that indicate whether subprocesses or environments are active.
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
    test_rattlesnake_wait_for_events
        Verifies successful waits when all ready and active event conditions
        are satisfied. Also verifies timeout behavior when ready events or
        active events do not reach the requested state. Confirms that ready
        events are set during timeout handling so the controller does not stay
        blocked indefinitely.
    """

### Loading and Saving
<!---
MARK: Engine Loading and Saving
--->
    def setup_gui
    """
    Configure the controller for GUI operation.

    Disables blocking mode and marks the controller as having a GUI attached.
    GUI-driven workflows typically handle asynchronous updates through queues
    rather than blocking on controller calls.
    """

    def load_rattlesnake_from_template
    """
    Load controller metadata from a template file.

    Loads hardware metadata, environment metadata, optional system
    identification metadata, and optional profile events from a supported
    template file. Supported file types are netCDF ``.nc4`` and Excel
    ``.xlsx``. Blocking mode is temporarily enabled while loading so that
    hardware and environment initialization complete before returning.

    Parameters
    ----------
    filepath : str
        Path to the template file.

    Raises
    ------
    RattlesnakeError
        If the file cannot be read.
    RattlesnakeError
        If the file type is unsupported by the loading logic.
    """

    def save_rattlesnake_to_template
    """
    Save the current controller configuration to an Excel template.

    Creates an Excel workbook and writes hardware metadata, environment
    metadata, and profile events to it. If no metadata is stored, a blank
    template is saved.

    Parameters
    ----------
    filepath : str
        Output path for the Excel template.

    Raises
    ------
    RattlesnakeError
        If ``filepath`` does not have an ``.xlsx`` extension.
    """

    def save_system_id_to_file
    """
    Save system identification data for an environment.

    Sends a save-system-identification command to the selected environment and
    waits for the environment ready event. Saving is permitted when the
    controller is in ``ENVIRONMENT_STORE`` or ``HARDWARE_ACTIVE`` state.

    Parameters
    ----------
    environment_name : str
        User-facing environment name whose system identification data should be
        saved.
    filepath : str or pathlib.Path
        Destination file path.

    Raises
    ------
    RattlesnakeError
        If the controller state does not permit saving system identification.
    RattlesnakeError
        If no environment exists with ``environment_name``.
    """

    def load_system_id_from_package
    """
    Load system identification data into an environment.

    Validates a system identification package with the environment manager,
    sends the package to the environment, and waits for the system
    identification stored event.

    Parameters
    ----------
    environment_name : str
        User-facing environment name that should receive the package.
    sysid_package : SysIdDataPackage
        System identification package to load.

    Raises
    ------
    RattlesnakeError
        If the controller state does not permit loading system identification.
    RattlesnakeError
        If package validation fails.
    """

### Hardware
<!---
MARK: Engine Hardware
--->
    property
    def hardware_metadata
    """
    Return the stored hardware metadata.

    Returns
    -------
    HardwareMetadata or None
        Hardware metadata currently stored on the controller.

    Unit Tests
    ----------
    test_rattlesnake_properties
        Verifies that hardware metadata is initially ``None``.
    """

    setter
    def hardware_metadata
    """
    Store hardware metadata on the controller.

    Parameters
    ----------
    value : HardwareMetadata
        Hardware metadata to store.
    """

    def initialize_hardware
    """
    Validate and initialize hardware metadata.

    Validates the current controller state, verifies that the supplied object
    is a ``HardwareMetadata`` instance, calls its validation method, stores
    hardware information in the environment manager, sends hardware
    initialization commands to acquisition and output, and waits for ready
    events when blocking mode is enabled.

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
    test_rattlesnake_initialize_hardware
        Verifies that hardware can be initialized from valid controller states
        and is rejected from active hardware or active environment states.
        Confirms that metadata validation is called, the environment manager
        receives the hardware metadata, acquisition and output initialization
        commands are queued, and blocking mode waits for ready events.
    """

### Environment
<!---
MARK: Engine Environment
--->
    property
    def environment_metadata
    """
    Return stored environment metadata.

    Returns
    -------
    dict of str to EnvironmentMetadata
        Mapping from environment queue names to environment metadata.

    Unit Tests
    ----------
    test_rattlesnake_properties
        Verifies that environment metadata is initially empty.
    """

    setter
    def environment_metadata
    """
    Store environment metadata through the environment manager.

    Parameters
    ----------
    value : dict of str to EnvironmentMetadata
        Environment metadata mapping to store.
    """

    def initialize_environments
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

    Raises
    ------
    RattlesnakeError
        If the controller state does not permit environment initialization.
    RattlesnakeError
        If environment metadata validation fails.

    Unit Tests
    ----------
    test_rattlesnake_initialize_environment
        Verifies that environments can be initialized only after hardware has
        been stored and before hardware is active. Confirms that environment
        metadata validation and initialization are delegated to the environment
        manager, initialized metadata is sent to acquisition and output, and
        blocking mode waits for ready events.

    test_rattlesnake_initialize_empty_environment
        Verifies that an empty environment metadata list can be initialized
        when the controller is already in an environment-store-compatible
        state. Confirms that empty initialized metadata is still sent to
        acquisition and output and returned to the caller.
    """

### System Identification
<!---
MARK: Engine System Identification
--->
    def initialize_system_id
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
    """

    def start_system_id_noise
    """
    Start the noise phase of system identification.

    Validates that hardware is active, resolves the environment queue name,
    sends a start-system-identification-noise command to the controller
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
    """

    def start_system_id_transfer_function
    """
    Start the transfer-function phase of system identification.

    Validates that hardware is active, resolves the environment queue name,
    sends a start-system-identification-transfer command to the controller
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
    """

    def stop_system_id
    """
    Stop active system identification.

    Validates that system identification is active, resolves the environment
    queue name, sends a stop-system-identification command to the controller
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
    """

    def preview_system_id_noise
    """
    Preview system identification noise.

    If hardware is already active, acquisition is stopped first. The method
    then initializes system identification metadata with automatic shutdown
    disabled, starts acquisition without streaming, and starts the noise phase.

    Parameters
    ----------
    sysid_metadata : SysIdMetadata
        System identification metadata to preview.
    environment_name : str
        User-facing environment name.

    Raises
    ------
    RattlesnakeError
        If the controller state does not permit previewing noise.
    """

    def preview_system_id_transfer
    """
    Preview system identification transfer function.

    Initializes system identification metadata with automatic shutdown
    disabled, starts acquisition without streaming, and starts the
    transfer-function phase.

    Parameters
    ----------
    sysid_metadata : SysIdMetadata
        System identification metadata to preview.
    environment_name : str
        User-facing environment name.

    Raises
    ------
    RattlesnakeError
        If the controller state does not permit previewing transfer function.
    """

    def run_system_id
    """
    Run a complete system identification workflow.

    Initializes system identification metadata with automatic shutdown enabled,
    starts acquisition, optionally starts streaming, runs the noise phase,
    optionally creates a new stream segment, runs the transfer-function phase,
    stops streaming if needed, and stops acquisition.

    Parameters
    ----------
    sysid_metadata : SysIdMetadata
        System identification metadata to run.
    environment_name : str
        User-facing environment name.

    Raises
    ------
    RattlesnakeError
        If the controller state does not permit running system identification.
    """

    def stop_system_id_run
    """
    Stop a system identification run.

    If hardware is active but system identification is not active, stops
    acquisition directly. Otherwise stops system identification for the
    environment and then stops acquisition.

    Parameters
    ----------
    environment_name : str
        User-facing environment name associated with the system identification
        run.
    """

### Acquisition
<!---
MARK: Engine Acquisition
--->
    def set_stream_metadata
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
    """

    def start_acquisition
    """
    Start acquisition and output hardware.

    Validates controller state and stream metadata, initializes streaming,
    sends a run-hardware command to the controller process, waits for streaming
    readiness and acquisition/output active events when blocking mode is
    enabled, and stores the last stream metadata.

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
    test_rattlesnake_start_acquisition
        Verifies that acquisition can only be started from
        ``ENVIRONMENT_STORE``. Confirms that valid stream metadata is
        validated, streaming initialization data is sent to the streaming
        process, the run-hardware command is sent to the controller process,
        and blocking mode waits for streaming readiness and acquisition/output
        active events.
    """

    def stop_acquisition
    """
    Stop acquisition and output hardware.

    Validates that hardware, an environment, or system identification is
    active; stops the profile manager; sends a stop-hardware command to the
    controller process; and waits for controller readiness and active events to
    clear when blocking mode is enabled.

    Raises
    ------
    RattlesnakeError
        If the controller state does not permit stopping acquisition.
    """

### Environment Active
<!---
MARK: Engine Environment Active
--->
    def start_environment
    """
    Start an initialized environment.

    Validates the controller state and instruction type, validates environment
    instructions through the environment manager, sends a start-environment
    command to the controller process, and waits for the environment active
    event when blocking mode is enabled.

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
    """

    def stop_environment
    """
    Stop an active environment.

    Validates that an environment is active, resolves the environment queue
    name, sends a stop-environment command to the controller process, and waits
    for the environment active event to clear when blocking mode is enabled.

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
    """

### Controller Commands
<!---
MARK: Engine Controller Commands
--->
    def environment_at_target_level
    """
    Notify the controller that an environment reached target level.

    Resolves the environment name and sends a stream-at-target-level command to
    the controller process. The controller process decides whether streaming
    should start based on current stream metadata.

    Parameters
    ----------
    environment_name : str
        User-facing environment name that reached target level.

    Raises
    ------
    RattlesnakeError
        If no environment exists with ``environment_name``.
    """

    def send_environment_command
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
    """

### Streaming
<!---
MARK: Engine Streaming
--->
    property
    def streaming
    """
    Return whether streaming is currently active.

    Returns
    -------
    bool
        ``True`` if the streaming active event is set, otherwise ``False``.
    """

    property
    def has_streamed
    """
    Return whether stream metadata has been stored.

    Returns
    -------
    bool
        ``True`` if ``last_stream_metadata`` is not ``None``, otherwise
        ``False``.
    """

    def start_streaming
    """
    Start manual streaming.

    Validates that acquisition, an environment, or system identification is
    active and that streaming is not already active. Sends a manual-stream
    command to the controller process and waits for the streaming active event
    when blocking mode is enabled.

    Raises
    ------
    RattlesnakeError
        If the controller state does not permit starting streaming.
    RattlesnakeError
        If streaming is already active.
    """

    def stop_streaming
    """
    Stop streaming.

    Validates that streaming is active, sends a stop-streaming command to the
    controller process, and waits for the streaming active event to clear when
    blocking mode is enabled.

    Raises
    ------
    RattlesnakeError
        If streaming is not currently active.
    """

### Profile
<!---
MARK: Engine Profile
--->
    property
    def has_profile
    """
    Return whether profile events have been stored.

    Returns
    -------
    bool
        ``True`` if ``last_profile_event_list`` is not empty, otherwise
        ``False``.
    """

    def initialize_profile_event_list
    """
    Store a profile event list for UI use.

    Validates that the controller state permits profile storage, validates
    profile events through the environment manager, and stores the list on the
    controller. This preloads profile state for the UI; ``start_profile`` still
    requires an explicit profile event list.

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
    """

    def start_profile
    """
    Start a profile.

    Validates that hardware is active, validates profile events through the
    environment manager and profile manager, clears the controller ready event,
    starts the profile manager, waits for controller readiness when blocking
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
    test_rattlesnake_start_profile
        Verifies that profiles can only be started while hardware is active.
        Confirms that profile events are validated by both the environment
        manager and profile manager, and that the profile manager is commanded
        to start the supplied profile event list.
    """

    def stop_profile
    """
    Stop the active profile.

    Clears the controller ready event, commands the profile manager to stop,
    and waits for controller readiness when blocking mode is enabled.
    """

### Shutdown and Logging
<!---
MARK: Engine Shutdown and Logging
--->
    def shutdown
    """
    Shut down the controller and all subprocesses.

    Stops acquisition if hardware, environments, or system identification are
    active. Then sends quit commands to controller, acquisition, output, and
    streaming processes; joins each process; force-closes processes that do not
    shut down within the close timeout; closes environment processes through
    the environment manager; and shuts down the log file process.

    Unit Tests
    ----------
    test_rattlesnake_shutdown
        Verifies orderly shutdown behavior for controller, acquisition, output,
        streaming, environment, and log processes. Confirms that active
        acquisition is stopped first, quit commands are sent to each process,
        processes are joined, close events are set for processes that remain
        alive after the first join attempt, multiprocessing workers are
        terminated if they remain alive after force-close signaling, environment
        processes are closed through the environment manager, and the log file
        process is joined.
    """

    def log
    """
    Queue a formatted controller log message.

    Formats the supplied message with the current timestamp and controller task
    name, then places it on the log file queue.

    Parameters
    ----------
    string : str
        Message to write to the log file queue.
    """

## Environment Manager
<!---
MARK: Environment Manager
--->
    """
    Manager for environment processes, metadata, queues, and events.

    This module defines the environment manager used by the Rattlesnake
    controller to validate environment metadata and instructions, assign
    environments to queue names, start and stop environment processes, route
    initialization commands, manage system-identification readiness state, and
    close environments during shutdown.
    """

### Environment Manager
<!---
MARK: Environment Manager Class
--->
    class EnvironmentManager
    """
    Container and process manager for controller environments.

    ``EnvironmentManager`` stores environment metadata, environment names,
    environment types, process handles, queue assignments, and event
    references. It is responsible for mapping user-facing environment names to
    internal queue names, starting new environment processes, reusing existing
    environment processes when possible, removing unused environment
    processes, validating metadata and instructions, and closing environments
    safely.

    Parameters
    ----------
    queue_container : QueueContainer
        Container holding environment command queues, environment data queues,
        controller queues, GUI queue, log queue, and hardware queues.
    event_container : EventContainer
        Container holding environment ready, active, close,
        system-identification active, and system-identification stored events.
    threaded : bool
        If ``True``, environments are started as threads. If ``False``,
        environments are started as multiprocessing processes.

    Attributes
    ----------
    queue_names : list of str
        Internal queue names currently assigned to active environment
        processes.
    environment_names : dict of str to str
        Mapping from queue names to user-facing environment names.
    environment_types : dict
        Mapping from queue names to environment types.
    environment_metadata : dict of str to EnvironmentMetadata
        Mapping from queue names to environment metadata.
    environment_processes : dict
        Mapping from queue names to environment process or thread handles.
    queue_container : QueueContainer
        Queue container used to communicate with environments and other
        controller processes.
    event_container : EventContainer
        Event container holding environment and controller synchronization
        events.
    environment_active_events : dict
        Mapping from queue names to environment active events.
    environment_ready_events : dict
        Mapping from queue names to environment ready events.
    environment_close_events : dict
        Mapping from queue names to environment close events.
    environment_sysid_active_events : dict
        Mapping from queue names to system-identification active events.
    environment_sysid_stored_events : dict
        Mapping from queue names to system-identification stored events.
    ping_alive_event : multiprocessing.synchronize.Event or threading.Event
        Event used to reset blocking timeouts during long-running operations.
    threaded : bool
        Whether environments are managed as threads rather than processes.

    Unit Tests
    ----------
    test_environment_manager_init
        Verifies that the environment manager initializes successfully and
        stores the requested threaded mode.
    """

    def __init__
    """
    Initialize the environment manager.

    Stores queue and event containers, initializes environment tracking
    dictionaries, stores event references, and chooses whether new environments
    should be started with ``threading.Thread`` or ``multiprocessing.Process``
    based on ``threaded``.

    Parameters
    ----------
    queue_container : QueueContainer
        Container holding queues used to communicate with environments and
        controller processes.
    event_container : EventContainer
        Container holding environment and controller synchronization events.
    threaded : bool
        If ``True``, create environment threads. If ``False``, create
        environment processes.

    Unit Tests
    ----------
    test_environment_manager_init
        Verifies that initialization creates an ``EnvironmentManager`` and that
        the ``threaded`` property reflects the supplied mode.
    """

    property
    def threaded
    """
    Return whether environments are managed in threaded mode.

    Returns
    -------
    bool
        ``True`` if new environments are created as threads, otherwise
        ``False``.

    Unit Tests
    ----------
    test_environment_manager_init
        Verifies that this property matches the initialization argument.
    """

    property
    def available_queues
    """
    Return environment queue names that are not currently assigned.

    Computes the set of environment command queues available in the queue
    container and removes any queue names already assigned to active
    environments.

    Returns
    -------
    list of str
        Available environment queue names.

    Unit Tests
    ----------
    test_environment_manager_available_queues
        Verifies that available queues exclude assigned queue names.
    """

    property
    def num_queues
    """
    Return the total number of environment queues.

    Returns the sum of assigned queue names and currently available queue
    names.

    Returns
    -------
    int
        Total number of environment queues managed by the controller.

    Unit Tests
    ----------
    test_environment_manager_available_queues
        Verifies that the total queue count includes assigned and available
        queues.
    """

    property
    def queue_names_dict
    """
    Return a mapping from user-facing environment names to queue names.

    Returns
    -------
    dict of str to str
        Mapping from environment names to internal queue names.

    Unit Tests
    ----------
    test_environment_manager_queue_names_dict
        Verifies that user-facing environment names map to the expected queue
        names.
    """

    def log
    """
    Queue a formatted environment manager log message.

    Formats the supplied message with the current timestamp and environment
    manager task name, then places it on the log file queue.

    Parameters
    ----------
    message : str
        Message to write to the log file queue.
    """

### Environment Manager State Sync
<!---
MARK: Environment Manager State Sync
--->
    property
    def ready_event_list
    """
    Return ready events for assigned environments.

    Returns
    -------
    list
        Ready events corresponding to currently assigned queue names.

    Unit Tests
    ----------
    test_environment_manager_ready_event_list
        Verifies that the returned list contains ready events for assigned
        environments in queue-name order.
    """

    property
    def active_event_list
    """
    Return active events for assigned environments.

    Returns
    -------
    list
        Active events corresponding to currently assigned queue names.
    """

    property
    def acquisition_ready_environments
    """
    Return whether each environment is ready for acquisition.

    Non-system-identification environments are always considered acquisition
    ready after initialization. System-identification environments are
    considered acquisition ready only after their system-identification stored
    event is set.

    Returns
    -------
    dict of str to bool
        Mapping from queue names to acquisition-readiness flags.
    """

    property
    def sysid_active_environments
    """
    Return user-facing names of environments currently running system
    identification.

    Returns
    -------
    list of str
        Environment names whose system-identification active events are set.
    """

    def clear_sysid_events
    """
    Clear system-identification stored events for assigned environments.

    This is used before reinitializing environments so stale system
    identification stored state does not incorrectly mark an environment as
    ready for acquisition.
    """

    def set_ready_events
    """
    Set all assigned environment ready events.

    This method is used by the main controller when a timeout occurs so that
    blocked waits can be released. Ready events are not cleared here because
    only the environment manager knows the currently mapped queue names.

    Unit Tests
    ----------
    test_environment_manager_set_ready_events
        Verifies that assigned environment ready events are set.
    """

    def initialize_hardware
    """
    Send hardware metadata to all assigned environments.

    Clears each environment ready event and sends an
    ``INITIALIZE_HARDWARE`` command containing the supplied hardware metadata
    to each assigned environment command queue.

    Parameters
    ----------
    hardware_metadata : HardwareMetadata
        Hardware metadata to send to environment processes.

    Unit Tests
    ----------
    test_environment_manager_initialize_hardware
        Verifies that hardware initialization commands are sent to assigned
        environment queues.
    """

    def initialize_environments
    """
    Initialize or update environment processes from metadata.

    Clears system-identification stored events, reuses existing environment
    processes when their environment type matches new metadata, removes
    existing processes that no longer have associated metadata, and starts new
    environment processes for remaining metadata. For reused environments, this
    method assigns the new environment name to the existing queue and sends
    hardware and environment initialization commands.

    Parameters
    ----------
    metadata_list : list of EnvironmentMetadata
        Environment metadata objects to initialize.
    hardware_metadata : HardwareMetadata
        Hardware metadata to send to initialized environments.

    Returns
    -------
    dict of str to EnvironmentMetadata
        Mapping from assigned queue names to environment metadata.

    Unit Tests
    ----------
    test_environment_manager_initialize_environment
        Verifies that matching existing environments receive initialization
        commands, new environments are added when needed, and unmapped
        environments are removed.
    """

    def initialize_system_id
    """
    Initialize system-identification metadata for an environment.

    Creates a deep copy of the stored environment metadata, updates the
    selected environment metadata with the supplied system-identification
    metadata, sends an ``INITIALIZE_SYSTEM_ID`` command to the environment, and
    returns the updated metadata dictionary.

    Parameters
    ----------
    sysid_metadata : SysIdMetadata
        System-identification metadata to store on the environment metadata.
    queue_name : str
        Internal queue name of the environment to update.

    Returns
    -------
    dict of str to EnvironmentMetadata
        Updated environment metadata dictionary.
    """

### Environment Manager Validation
<!---
MARK: Environment Manager Validation
--->
    def validate_environment_metadata
    """
    Validate a list of environment metadata objects.

    Ensures there are enough environment queues available, each item is an
    ``EnvironmentMetadata`` object, environment names are unique, and each
    metadata object validates against the supplied hardware metadata.

    Parameters
    ----------
    metadata_list : list of EnvironmentMetadata
        Environment metadata objects to validate.
    hardware_metadata : HardwareMetadata
        Hardware metadata used by metadata validation.

    Raises
    ------
    RattlesnakeError
        If there are more metadata objects than available queues.
    RattlesnakeError
        If any item is not an ``EnvironmentMetadata`` object.
    RattlesnakeError
        If environment names are not unique.
    RattlesnakeError
        If an environment metadata object fails validation.

    Unit Tests
    ----------
    test_environment_manager_validate_environment_metadata
        Verifies valid metadata passes validation and invalid object types,
        duplicate names, or too many environments raise ``RattlesnakeError``.
    """

    def validate_system_id_metadata
    """
    Validate system-identification metadata for an environment.

    Ensures the supplied metadata is a ``SysIdMetadata`` object, the requested
    environment exists, and the environment type supports system
    identification.

    Parameters
    ----------
    sysid_metadata : SysIdMetadata
        System-identification metadata to validate.
    hardware_metadata : HardwareMetadata
        Hardware metadata associated with the environment.
    environment_name : str
        User-facing environment name.

    Returns
    -------
    str
        Queue name associated with ``environment_name``.

    Raises
    ------
    RattlesnakeError
        If ``sysid_metadata`` is not a ``SysIdMetadata`` object.
    RattlesnakeError
        If no environment exists with ``environment_name``.
    RattlesnakeError
        If the environment type does not support system identification.
    """

    def validate_system_id_package
    """
    Validate a system-identification data package for an environment.

    Ensures the requested environment exists, the environment supports system
    identification, the supplied package is a ``SysIdDataPackage``, the package
    validates itself, and the package response and reference channel counts
    match the environment metadata.

    Parameters
    ----------
    environment_name : str
        User-facing environment name.
    data_package : SysIdDataPackage
        System-identification data package to validate.

    Returns
    -------
    str
        Queue name associated with ``environment_name``.

    Raises
    ------
    RattlesnakeError
        If no environment exists with ``environment_name``.
    RattlesnakeError
        If the environment does not support system identification.
    RattlesnakeError
        If ``data_package`` is not a ``SysIdDataPackage``.
    RattlesnakeError
        If package validation fails.
    RattlesnakeError
        If package channel counts do not match the environment metadata.
    """

    def validate_environment_instructions
    """
    Validate environment startup instructions.

    Ensures the supplied object is an ``EnvironmentInstructions`` object, the
    requested environment exists, the instruction type matches the environment
    type, required system-identification data has been stored for
    system-identification environments, and the instruction object validates
    itself.

    Parameters
    ----------
    instructions : EnvironmentInstructions
        Environment startup instructions to validate.

    Returns
    -------
    str
        Queue name associated with the instruction environment.

    Raises
    ------
    RattlesnakeError
        If ``instructions`` is not an ``EnvironmentInstructions`` object.
    RattlesnakeError
        If no environment exists for the instruction environment name.
    RattlesnakeError
        If the instruction environment type does not match the initialized
        environment type.
    RattlesnakeError
        If system-identification data is required but has not been stored.
    RattlesnakeError
        If instruction validation fails.

    Unit Tests
    ----------
    test_environment_manager_validate_instructions
        Verifies that valid instructions return a queue name and invalid
        environment names, environment types, or instruction objects raise
        ``RattlesnakeError``.
    """

    def validate_profile_events
    """
    Validate and annotate profile events.

    Ensures each profile event is a ``ProfileEvent`` object. For global
    profile events, assigns the queue name and environment type to ``"Global"``.
    For environment-specific profile events, resolves the environment name to a
    queue name and stores the queue name and environment type on the event.

    Parameters
    ----------
    profile_events_list : list of ProfileEvent
        Profile events to validate and annotate.

    Raises
    ------
    RattlesnakeError
        If any item is not a ``ProfileEvent`` object.
    RattlesnakeError
        If a profile event references an unknown environment.

    Unit Tests
    ----------
    test_environment_mananger_validate_profile_events
        Verifies that valid profile events are annotated with queue name and
        environment type, global profile events are annotated as global, and
        invalid event objects or environment names raise ``RattlesnakeError``.
    """

### Environment Manager Environment Operations
<!---
MARK: Environment Manager Environment Operations
--->
    def clear_environments
    """
    Clear all environment assignments and close environment processes.

    Resets assigned queue names, environment names, environment types, and
    stored environment metadata. Then closes all existing environment
    processes and clears the environment process dictionary.

    Unit Tests
    ----------
    test_environment_manager_clear_environment
        Verifies that environment tracking dictionaries are cleared and that
        ``close_environments`` is called.
    """

    def add_environment
    """
    Add a new environment process.

    Finds the first available environment queue, assigns the environment name
    to the queue, clears the close event, constructs the environment process or
    thread using the environment process registry, starts it, stores process
    and metadata mappings, sends hardware and environment initialization
    commands, and clears active and system-identification events.

    Parameters
    ----------
    metadata : EnvironmentMetadata
        Environment metadata defining the environment to add.
    hardware_metadata : HardwareMetadata
        Hardware metadata to send to the new environment.

    Returns
    -------
    str
        Queue name assigned to the new environment.

    Raises
    ------
    RattlesnakeError
        If no environment command queues are available.

    Unit Tests
    ----------
    test_environment_mananger_add_environment_error
        Verifies that adding an environment raises ``RattlesnakeError`` when no
        queues are available.
    """

    def remove_environment
    """
    Remove an assigned environment process.

    Sends a quit command to the environment, joins the environment process,
    force-closes it if necessary, terminates unresponsive multiprocessing
    processes, removes environment metadata and tracking entries, and frees the
    queue name for later reuse.

    Parameters
    ----------
    queue_name : str
        Internal queue name of the environment to remove.

    Raises
    ------
    RattlesnakeError
        If ``queue_name`` is not currently assigned to an environment.

    Unit Tests
    ----------
    test_environment_manager_remove_environment
        Verifies that removing an environment sends a quit command, joins the
        process, sets the close event if the process remains alive, and clears
        environment tracking dictionaries.

    test_environment_manager_remove_environment_invalid_queue_name
        Verifies that removing an unassigned queue name raises
        ``RattlesnakeError``.
    """

    def close_environments
    """
    Close all managed environment processes.

    Sends quit commands to all environment processes, joins each process,
    force-closes any process that remains alive, terminates unresponsive
    multiprocessing processes, and flushes each environment command queue.

    Unit Tests
    ----------
    test_environment_mananger_close_environment
        Verifies that closing environments sends quit commands, joins
        environment processes, and sets close events for processes that remain
        alive.
    """

## Profile Manager
<!---
MARK: Profile Manager
--->
    """
    Profile scheduling and command-routing utilities.

    This module defines profile events and the profile manager used by the
    Rattlesnake controller to validate timed command sequences, schedule those
    commands with timers, route profile commands to the controller, and send a
    closeout event after the final profile event has fired.
    """

### Profile Constants
<!---
MARK: Profile Constants
--->
    EXTRA_CLOSEOUT_TIME
    """
    Additional delay, in seconds, added after the final profile event.

    This delay allows the last scheduled profile event to be processed before
    the profile manager sends the profile closeout command to the controller.
    """

    TASK_NAME
    """
    Task name used in profile manager log messages and command queue writes.
    """

    VALID_COMMANDS
    """
    Mapping of valid profile commands by environment type.

    Global profile events may stop hardware or start and stop streaming.
    Environment-specific profile events may start or stop environments, update
    environment instructions, or send profile-safe environment commands
    defined by the environment command classes.
    """

    VALID_DATA
    """
    Mapping from profile commands to required data types.

    Used by ``ProfileEvent.validate`` to ensure each profile event carries data
    of the correct type for its command.
    """

### Profile Event
<!---
MARK: Profile Event
--->
    class ProfileEvent
    """
    Timed command scheduled as part of a profile.

    A profile event defines when a command should fire, which environment it
    applies to, what command should be sent, and what command data should be
    included. Environment manager validation assigns the internal queue name
    and environment type before the event is validated by the profile manager.

    Parameters
    ----------
    timestamp : float
        Time in seconds from profile start when this event should fire.
    environment_name : str
        User-facing environment name associated with the event. Use
        ``"Global"`` for controller-wide profile events.
    command : Enum
        Command to execute when the event fires.
    data : Any, optional
        Command data associated with the event. The required data type depends
        on ``command``.

    Attributes
    ----------
    timestamp : float
        Time in seconds from profile start when this event should fire.
    environment_name : str
        User-facing environment name associated with the event.
    command : Enum
        Command to execute.
    data : Any
        Command payload.
    environment_type : EnvironmentType or str or None
        Environment type assigned by the environment manager. ``"Global"`` is
        used for global profile events.
    queue_name : str or None
        Internal queue name assigned by the environment manager.

    Unit Tests
    ----------
    test_profile_event_init
        Verifies that a profile event can be initialized.

    test_profile_event_properties
        Verifies that the assigned environment type and queue name properties
        return expected values.
    """

    def __init__
    """
    Initialize a profile event.

    Stores the event timestamp, environment name, command, and command data.
    The environment type and queue name are initialized to ``None`` and are
    expected to be assigned by the environment manager before validation.

    Parameters
    ----------
    timestamp : float
        Time in seconds from profile start when this event should fire.
    environment_name : str
        User-facing environment name associated with the event.
    command : Enum
        Command to execute when the event fires.
    data : Any, optional
        Command data associated with the event.

    Unit Tests
    ----------
    test_profile_event_init
        Confirms that a ``ProfileEvent`` instance can be constructed.
    """

    property
    def environment_type
    """
    Return the environment type assigned to this profile event.

    Returns
    -------
    EnvironmentType or str or None
        Environment type assigned by the environment manager, ``"Global"`` for
        global events, or ``None`` if not yet assigned.

    Unit Tests
    ----------
    test_profile_event_properties
        Verifies that this property returns the assigned environment type.
    """

    property
    def queue_name
    """
    Return the internal queue name assigned to this profile event.

    Returns
    -------
    str or None
        Queue name assigned by the environment manager, ``"Global"`` for
        global events, or ``None`` if not yet assigned.

    Unit Tests
    ----------
    test_profile_event_properties
        Verifies that this property returns the assigned queue name.
    """

    def validate
    """
    Validate the profile event.

    Ensures the environment name is a string, the timestamp is a nonnegative
    number, the environment type supports the requested command, the command is
    valid for the assigned environment type, the queue name has been assigned,
    and the event data matches the command's required data type. For events
    carrying ``EnvironmentInstructions``, the instruction environment name and
    environment type must match the profile event.

    Raises
    ------
    RattlesnakeError
        If ``environment_name`` is not a string.
    RattlesnakeError
        If ``timestamp`` is not a nonnegative number.
    RattlesnakeError
        If ``environment_type`` is not valid.
    RattlesnakeError
        If ``command`` is not valid for the assigned environment type.
    RattlesnakeError
        If ``queue_name`` has not been assigned.
    RattlesnakeError
        If ``data`` does not match the required type for ``command``.
    RattlesnakeError
        If environment instruction data does not match the profile event
        environment name or environment type.

    Unit Tests
    ----------
    test_profile_event_validate
        Verifies that valid profile events pass validation and invalid
        timestamps, environment names, environment types, commands, queue names,
        and data types raise ``RattlesnakeError``.
    """

### Profile Manager
<!---
MARK: Profile Manager Class
--->
    class ProfileManager
    """
    Manager for validating, scheduling, and routing profile events.

    ``ProfileManager`` validates lists of ``ProfileEvent`` objects, schedules
    them with ``threading.Timer``, routes fired profile events to controller
    commands, cancels active profile timers when stopping a profile, and sends
    a closeout command after profile execution finishes.

    Parameters
    ----------
    queue_container : QueueContainer
        Container holding the log file queue and controller command queue used
        by the profile manager.

    Attributes
    ----------
    log_file_queue : multiprocessing.Queue
        Queue used to send profile manager log messages.
    controller_command_queue : VerboseMessageQueue
        Queue used to send profile-generated commands to the controller
        process.
    profile_timers : list of threading.Timer
        Timers created for scheduled profile events and closeout events.
    gui_timer : threading.Timer or None
        Timer reserved for GUI-related profile timing behavior.
    command_map : dict
        Mapping from profile commands to profile manager handler methods.

    Unit Tests
    ----------
    test_profile_manager_init
        Verifies that a profile manager can be initialized.

    test_profile_manager_properties
        Verifies that profile manager queue properties return expected queues.
    """

    def __init__
    """
    Initialize the profile manager.

    Stores log and controller command queues, initializes timer storage, and
    builds the command map used to route profile events. Global commands are
    mapped to controller-level handlers, while environment-specific profile
    commands are mapped to ``send_environment_command``.

    Parameters
    ----------
    queue_container : QueueContainer
        Container holding log and controller command queues.

    Unit Tests
    ----------
    test_profile_manager_init
        Confirms that a ``ProfileManager`` instance can be constructed.
    """

    property
    def log_file_queue
    """
    Return the profile manager log file queue.

    Returns
    -------
    multiprocessing.Queue
        Queue used to send formatted log messages.

    Unit Tests
    ----------
    test_profile_manager_properties
        Verifies that this property returns the stored log file queue.
    """

    property
    def controller_command_queue
    """
    Return the controller command queue.

    Returns
    -------
    VerboseMessageQueue
        Queue used to send profile-generated commands to the controller
        process.

    Unit Tests
    ----------
    test_profile_manager_properties
        Verifies that this property returns the stored controller command
        queue.
    """

    def validate_profile_list
    """
    Validate and sort a list of profile events.

    Ensures each item is a ``ProfileEvent``, validates each event, verifies
    that the profile manager has an implemented handler for each command, and
    sorts the list in ascending timestamp order.

    Parameters
    ----------
    profile_event_list : list of ProfileEvent
        Profile events to validate and sort.

    Raises
    ------
    RattlesnakeError
        If any item is not a ``ProfileEvent``.
    RattlesnakeError
        If any profile event fails validation.
    RattlesnakeError
        If no profile manager handler is implemented for an event command.

    Unit Tests
    ----------
    test_profile_manager_validate_profile_list
        Verifies that valid profile lists pass validation and invalid event
        types, invalid commands, or unimplemented profile commands raise
        ``RattlesnakeError``.
    """

    def start_profile
    """
    Start a profile by scheduling all profile events.

    Creates and starts one ``threading.Timer`` for each profile event. Each
    timer calls ``fire_profile_event`` with the event queue name, command, and
    data. Also schedules a final closeout timer at the last event timestamp
    plus ``EXTRA_CLOSEOUT_TIME``.

    Parameters
    ----------
    profile_event_list : list of ProfileEvent
        Validated profile events to schedule.

    Unit Tests
    ----------
    test_profile_manager_start_profile
        Verifies that timers are created for global, environment, and
        start-environment events and that an additional closeout timer is
        scheduled after the final event.
    """

    def fire_profile_event
    """
    Fire a scheduled profile event.

    Logs the event and dispatches it to the profile manager handler associated
    with ``command`` in ``command_map``.

    Parameters
    ----------
    queue_name : str
        Queue name associated with the profile event.
    command : Enum
        Command to execute.
    data : Any
        Command payload.

    Unit Tests
    ----------
    test_profile_manager_fire_profile_event
        Verifies that firing an event dispatches to the mapped command handler
        with queue name, command, and data.
    """

    def stop_profile
    """
    Stop the active profile.

    Cancels all existing profile timers and schedules a closeout timer so the
    controller is notified that profile execution has ended.

    Unit Tests
    ----------
    test_profile_manager_stop_profile
        Verifies that existing timers are canceled and that a closeout timer is
        created, started, and stored.
    """

    def stop_hardware
    """
    Route a profile stop-hardware event to the controller.

    Sends ``GlobalCommands.STOP_HARDWARE`` to the controller command queue.

    Parameters
    ----------
    queue_name : str
        Queue name associated with the event. This value is ignored for global
        stop-hardware events.
    command : GlobalCommands
        Profile command that triggered the handler.
    data : None
        Command data. This value is ignored.

    Unit Tests
    ----------
    test_profile_manager_stop_hardware
        Verifies that the stop-hardware command is sent to the controller.
    """

    def start_streaming
    """
    Route a profile start-streaming event to the controller.

    Sends ``GlobalCommands.START_STREAMING`` to the controller command queue
    with ``False`` as the override flag so normal profile streaming logic is
    used.

    Parameters
    ----------
    queue_name : str
        Queue name associated with the event. This value is ignored for global
        start-streaming events.
    command : GlobalCommands
        Profile command that triggered the handler.
    data : None
        Command data. This value is ignored.

    Unit Tests
    ----------
    test_profile_manager_start_streaming
        Verifies that the start-streaming command is sent to the controller.
    """

    def stop_streaming
    """
    Route a profile stop-streaming event to the controller.

    Sends ``GlobalCommands.STOP_STREAMING`` to the controller command queue.

    Parameters
    ----------
    queue_name : str
        Queue name associated with the event. This value is ignored for global
        stop-streaming events.
    command : GlobalCommands
        Profile command that triggered the handler.
    data : None
        Command data. This value is ignored.

    Unit Tests
    ----------
    test_profile_manager_stop_streaming
        Verifies that the stop-streaming command is sent to the controller.
    """

    def start_environment
    """
    Route a profile start-environment event to the controller.

    Sends ``GlobalCommands.START_ENVIRONMENT`` to the controller command queue
    with the environment queue name and supplied environment instructions.

    Parameters
    ----------
    queue_name : str
        Internal queue name of the environment to start.
    command : GlobalCommands
        Profile command that triggered the handler.
    data : EnvironmentInstructions
        Instructions used to start the environment.

    Unit Tests
    ----------
    test_start_environment
        Verifies that the start-environment command and instructions are sent
        to the controller.
    """

    def stop_environment
    """
    Route a profile stop-environment event to the controller.

    Sends ``GlobalCommands.STOP_ENVIRONMENT`` to the controller command queue
    with the environment queue name.

    Parameters
    ----------
    queue_name : str
        Internal queue name of the environment to stop.
    command : GlobalCommands
        Profile command that triggered the handler.
    data : None
        Command data. This value is ignored.

    Unit Tests
    ----------
    test_stop_environment
        Verifies that the stop-environment command is sent to the controller.
    """

    def send_environment_command
    """
    Route an environment-specific profile command to the controller.

    Sends ``GlobalCommands.SEND_ENVIRONMENT_COMMAND`` to the controller command
    queue with the destination environment queue name, original environment
    command, and command data.

    Parameters
    ----------
    queue_name : str
        Internal queue name of the target environment.
    command : Enum
        Environment-specific command to forward.
    data : Any
        Command data associated with ``command``.
    """

    def fire_closeout_event
    """
    Notify the controller that profile execution has completed.

    Sends ``GlobalCommands.PROFILE_CLOSEOUT`` to the controller command queue.

    Unit Tests
    ----------
    test_fire_closeout_event
        Verifies that the profile closeout command is sent to the controller.
    """

    def log
    """
    Queue a formatted profile manager log message.

    Formats the supplied message with the current timestamp and profile manager
    task name, then places it on the log file queue.

    Parameters
    ----------
    message : str
        Message to write to the log file queue.

    Unit Tests
    ----------
    test_profile_manager_log
        Verifies that calling this method writes the expected formatted message
        to the log file queue.
    """

## Utilities
<!---
MARK: Utilities
--->
    """
    General helper classes and functions used throughout the controller.

    This module contains shared controller utilities, including global command
    definitions, logging support, verbose queue wrappers, queue and event
    containers, queue flushing helpers, network discovery helpers, file loading
    utilities, signal-processing math utilities, signal alignment functions,
    and an overlap buffer implementation.
    """

### Global Utilities
<!---
MARK: Global Utilities
--->
    DIRECTORY
    """
    Base directory used by the application.

    When running from a frozen executable, this resolves to the PyInstaller
    temporary directory. Otherwise, it resolves to the directory containing this
    utilities module.
    """

    def log_file_task
    """
    Run the log file task.

    Continuously reads log messages from the supplied queue and writes them to
    ``Rattlesnake.log``. The task exits when either ``shutdown_event`` is set
    or a ``GlobalCommands.QUIT`` message is received. Multi-line log messages
    are compressed by replacing all but the final newline with ``"////"``.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Queue from which log messages are read.
    shutdown_event : multiprocessing.synchronize.Event
        Event used to signal that the logging task should stop.

    Unit Tests
    ----------
    test_log_file_process
        Verifies that queued log messages are written to the log file, that
        ``GlobalCommands.QUIT`` writes the shutdown message, and that the file
        is flushed.
    """

    class RattlesnakeError
    """
    Base exception type for Rattlesnake-specific errors.

    This exception is raised when controller metadata, configuration, process
    state, file loading, or runtime validation fails in a way that should be
    reported as a Rattlesnake controller error.
    """

### Global Commands
<!---
MARK: Global Commands
--->
    class GlobalCommands
    """
    Enumeration of global commands passed between controller processes.

    These commands define the common process-control interface for controller,
    acquisition, output, streaming, environment, profile, and system
    identification operations.

    Attributes
    ----------
    QUIT : int
        Stop an individual process.
    INITIALIZE_HARDWARE : int
        Send hardware metadata to a process.
    RUN_HARDWARE : int
        Start acquisition or output hardware.
    STOP_HARDWARE : int
        Stop acquisition or output hardware.
    INITIALIZE_ENVIRONMENT : int
        Send environment metadata to a process.
    START_ENVIRONMENT : int
        Start an environment.
    STOP_ENVIRONMENT : int
        Stop an environment.
    INITIALIZE_SYSTEM_ID : int
        Send system identification metadata to an environment or process.
    START_SYSTEM_ID_NOISE : int
        Start the noise phase of system identification.
    START_SYSTEM_ID_TRANSFER : int
        Start the transfer-function phase of system identification.
    STOP_SYSTEM_ID : int
        Stop system identification.
    INITIALIZE_STREAMING : int
        Initialize a streaming file.
    CREATE_NEW_STREAM : int
        Create a new stream variable in a streaming file.
    START_STREAMING : int
        Start sending acquisition data to streaming.
    STREAMING_DATA : int
        Send data to the streaming process.
    STOP_STREAMING : int
        Stop sending acquisition data to streaming.
    FINALIZE_STREAMING : int
        Close the active streaming file.
    INITIALIZE_PROFILE : int
        Initialize profile metadata.
    START_PROFILE : int
        Start a profile.
    STOP_PROFILE : int
        Stop a profile.
    PROFILE_CLOSEOUT : int
        Notify the controller that profile events are complete.
    STREAM_AT_TARGET_LEVEL : int
        Notify the controller that an environment reached target level.
    STREAM_MANUAL : int
        Notify the controller that manual streaming was requested.
    SEND_ENVIRONMENT_COMMAND : int
        Forward an environment-specific command to an environment.
    SAVE_SYSTEM_ID : int
        Save system identification data.
    LOAD_SYSTEM_ID : int
        Load system identification data.

    Unit Tests
    ----------
    test_verbose_message_queue_put
        Uses global command enum members as messages in verbose queue tests.

    test_verbose_message_queue_get
        Verifies that global command messages can be recovered from a verbose
        queue.
    """

    property
    def label
    """
    Return a user-friendly label for the command.

    Converts the enum member name by replacing underscores with spaces and
    applying title case.

    Returns
    -------
    str
        Human-readable command label.
    """

### Verbose Message Queue
<!---
MARK: Verbose Message Queue
--->
    class VerboseMessageQueue
    """
    Queue wrapper that logs put, get, and flush operations.

    ``VerboseMessageQueue`` wraps a standard multiprocessing or threading queue
    and adds structured log messages for command traffic. Messages are stored
    internally as ``(message_id, (message, data))`` tuples. The generated
    message ID allows put and get operations to be correlated in the log.

    Parameters
    ----------
    log_queue : multiprocessing.Queue
        Queue where verbose queue log messages are written.
    base_queue : multiprocessing.Queue or queue.Queue
        Underlying queue used to store message payloads.
    base_name : str, optional
        Base queue name used in log messages.
    name_manager : multiprocessing.Manager, optional
        Manager used to create a shared environment name value. If supplied,
        ``assign_environment`` can update the queue log name dynamically.

    Attributes
    ----------
    base_queue : multiprocessing.Queue or queue.Queue
        Underlying queue object.
    log_queue : multiprocessing.Queue
        Queue where log messages are written.
    base_name : str
        Base queue name used in log messages.
    environment_name : multiprocessing proxy value or None
        Optional shared environment name used to augment the log name.
    last_put_message : Enum or None
        Most recent message type logged during put.
    last_put_time : float
        Time of the most recent logged put.
    last_get_message : Enum or None
        Most recent message type logged during get.
    last_get_time : float
        Time of the most recent logged get.
    last_flush : float
        Time of the most recent logged flush.
    time_threshold : float
        Minimum time between repeated log messages of the same message type.

    Unit Tests
    ----------
    test_verbose_queue_init
        Verifies that the verbose queue initializes successfully.

    test_verbose_queue_name
        Verifies that the log name includes the assigned environment name.

    test_verbose_message_id
        Verifies deterministic message ID generation when the random seed is
        controlled.
    """

    def __init__
    """
    Initialize a verbose message queue.

    Stores the underlying queue, log queue, base log name, optional shared
    environment name, repeated-message logging state, and logging time
    threshold.

    Parameters
    ----------
    log_queue : multiprocessing.Queue
        Queue where verbose queue log messages are written.
    base_queue : multiprocessing.Queue or queue.Queue
        Underlying queue used to store message payloads.
    base_name : str, optional
        Base name used in queue log messages.
    name_manager : multiprocessing.Manager, optional
        Manager used to create shared environment-name storage.

    Unit Tests
    ----------
    test_verbose_queue_init
        Confirms that a ``VerboseMessageQueue`` object can be constructed.
    """

    property
    def log_name
    """
    Return the current name used in verbose queue log messages.

    If an environment name has been assigned, the returned name is formatted as
    ``"<base_name> | <environment_name>"``. Otherwise, only the base name is
    returned.

    Returns
    -------
    str
        Log name for this queue.

    Unit Tests
    ----------
    test_verbose_queue_name
        Verifies the base log name and environment-specific log name.
    """

    def assign_environment
    """
    Assign an environment name to this queue for logging.

    Parameters
    ----------
    env_name : str
        Environment name to include in queue log messages.

    Unit Tests
    ----------
    test_verbose_queue_name
        Verifies that assigning an environment updates the queue log name.
    """

    def generate_message_id
    """
    Generate a random message identifier.

    Parameters
    ----------
    size : int, optional
        Number of characters in the generated ID. Defaults to ``6``.
    chars : str, optional
        Character set used to generate the ID. Defaults to ASCII letters and
        digits.

    Returns
    -------
    str
        Random message identifier.

    Unit Tests
    ----------
    test_verbose_message_id
        Verifies that generated IDs match the expected random sequence when the
        random seed is fixed.
    """

    def put
    """
    Put a command payload on the verbose queue.

    Logs the put operation if the message type differs from the last logged put
    or if enough time has elapsed since the previous logged put. The payload is
    stored on the underlying queue as ``(message_id, message_data_tuple)``.

    Parameters
    ----------
    task_name : str
        Name of the task placing the message on the queue.
    message_data_tuple : tuple
        Tuple ``(message, data)`` containing the command and payload.
    *args
        Additional positional arguments passed to ``base_queue.put``.
    **kwargs
        Additional keyword arguments passed to ``base_queue.put``.

    Unit Tests
    ----------
    test_verbose_message_queue_put
        Verifies that the underlying queue receives the generated message ID
        and command payload.
    """

    def get
    """
    Get a command payload from the verbose queue.

    Retrieves a ``(message_id, message_data_tuple)`` pair from the underlying
    queue, logs the get operation if the message ID is not empty, and returns
    only the ``(message, data)`` payload.

    Parameters
    ----------
    task_name : str
        Name of the task retrieving the message.
    *args
        Additional positional arguments passed to ``base_queue.get``.
    **kwargs
        Additional keyword arguments passed to ``base_queue.get``.

    Returns
    -------
    tuple
        Tuple ``(message, data)`` retrieved from the queue.

    Unit Tests
    ----------
    test_verbose_message_queue_get
        Verifies that the payload returned from the verbose queue matches the
        payload stored on the underlying queue.

    test_verbose_queue_log
        Exercises put and get logging through a helper process or thread.
    """

    def flush
    """
    Flush all currently available messages from the verbose queue.

    Logs the flush operation, repeatedly retrieves messages from the underlying
    queue without blocking, logs flushed messages with nonempty IDs, and
    returns the collected payloads.

    Parameters
    ----------
    task_name : str
        Name of the task flushing the queue.

    Returns
    -------
    list
        List of ``(message, data)`` tuples removed from the queue.

    Unit Tests
    ----------
    test_verbose_message_queue_flush
        Verifies that flushed data matches the queued input payload.
    """

    def empty
    """
    Return whether the underlying queue is empty.

    Returns
    -------
    bool
        ``True`` if the underlying queue is empty, otherwise ``False``.

    Unit Tests
    ----------
    test_verbose_queue_close
        Calls this method after closing the queue to verify it is available.
    """

    def close
    """
    Close the underlying queue if it supports ``close``.

    Unit Tests
    ----------
    test_verbose_queue_close
        Verifies that calling this method does not raise.
    """

    def join_thread
    """
    Join the underlying queue's feeder thread if supported.

    Unit Tests
    ----------
    test_verbose_queue_close
        Verifies that calling this method does not raise.
    """

### Queue and Event Containers
<!---
MARK: Queue and Event Containers
--->
    class QueueContainer
    """
    Container for queues managed by the controller.

    This class groups the command, logging, synchronization, GUI, hardware, and
    per-environment data queues into a single namespace that can be passed to
    controller subprocesses and managers.

    Parameters
    ----------
    controller_command_queue : VerboseMessageQueue
        Queue used to send commands to the controller process.
    acquisition_command_queue : VerboseMessageQueue
        Queue used to send commands to the acquisition process.
    output_command_queue : VerboseMessageQueue
        Queue used to send commands to the output process.
    streaming_command_queue : VerboseMessageQueue
        Queue used to send commands to the streaming process.
    log_file_queue : multiprocessing.Queue
        Queue used to send messages to the log file task.
    input_output_sync_queue : multiprocessing.Queue or queue.Queue
        Queue used to synchronize output startup with acquisition.
    single_process_hardware_queue : multiprocessing.Queue or queue.Queue
        Queue used by hardware implementations that cannot split acquisition
        and output across processes.
    gui_update_queue : multiprocessing.Queue or queue.Queue
        Queue used to send GUI update messages.
    environment_command_queues : dict of str to VerboseMessageQueue
        Per-environment command queues.
    environment_data_in_queues : dict of str to multiprocessing.Queue
        Per-environment acquisition-data input queues.
    environment_data_out_queues : dict of str to multiprocessing.Queue
        Per-environment output-data queues.
    """

    class EventContainer
    """
    Container for events managed by the controller.

    This class groups ready, close, active, system-identification, storage, and
    liveness events into a single namespace that can be passed to controller
    subprocesses and managers.

    Parameters
    ----------
    controller_ready_event : multiprocessing.synchronize.Event
        Event set when the controller process is ready.
    acquisition_ready_event : multiprocessing.synchronize.Event
        Event set when the acquisition process is ready.
    output_ready_event : multiprocessing.synchronize.Event
        Event set when the output process is ready.
    streaming_ready_event : multiprocessing.synchronize.Event
        Event set when the streaming process is ready.
    environment_ready_events : dict of str to multiprocessing.synchronize.Event
        Per-environment ready events.
    log_close_event : multiprocessing.synchronize.Event
        Event used to close the logging task.
    controller_close_event : multiprocessing.synchronize.Event
        Event used to force-close the controller process.
    acquisition_close_event : multiprocessing.synchronize.Event
        Event used to force-close the acquisition process.
    output_close_event : multiprocessing.synchronize.Event
        Event used to force-close the output process.
    streaming_close_event : multiprocessing.synchronize.Event
        Event used to force-close the streaming process.
    environment_close_events : dict of str to multiprocessing.synchronize.Event
        Per-environment close events.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    streaming_active_event : multiprocessing.synchronize.Event
        Event indicating whether streaming is active.
    environment_active_events : dict of str to multiprocessing.synchronize.Event
        Per-environment active events.
    environment_sysid_active_events : dict of str to multiprocessing.synchronize.Event
        Per-environment system-identification active events.
    environment_sysid_stored_events : dict of str to multiprocessing.synchronize.Event
        Per-environment system-identification stored events.
    ping_alive_event : multiprocessing.synchronize.Event
        Event used to reset blocking wait timeouts during long operations.
    """

    def flush_queue
    """
    Flush all currently available items from a queue.

    Supports both standard queues and ``VerboseMessageQueue`` instances. When a
    timeout is supplied, queue reads are blocking with that timeout; otherwise,
    reads are nonblocking. Items are collected until the queue raises an empty
    exception.

    Parameters
    ----------
    queue : multiprocessing.Queue, queue.Queue, or VerboseMessageQueue
        Queue to flush.
    timeout : float, optional
        Optional timeout for blocking queue reads.

    Returns
    -------
    list
        Items removed from the queue.
    """

### LAN-XI Network Utilities
<!---
MARK: LAN-XI Network Utilities
--->
    def autofill_single_ip_address
    """
    Autofill one IP address record.

    Attempts to populate missing hostname, IPv4 address, and IPv6 address
    fields for one ``IPAddress`` object using reverse and forward lookup
    helpers. Existing valid IP records are returned unchanged.

    Parameters
    ----------
    ip_address : IPAddress
        IP address record to update.

    Returns
    -------
    IPAddress
        Updated IP address record.
    """

    def search_for_lanxi_devices
    """
    Search for LAN-XI devices for a fixed duration.

    Repeatedly calls ``find_lanxi_devices`` until ``timeout`` seconds have
    elapsed and returns unique devices by IPv4 address.

    Parameters
    ----------
    timeout : float
        Search duration in seconds.

    Returns
    -------
    list of IPAddress
        Unique discovered LAN-XI device addresses.
    """

    def find_lanxi_devices
    """
    Discover LAN-XI devices from ARP table candidates.

    Parses ``arp -a`` output for link-local IPv4 addresses, tests candidates in
    parallel using LAN-XI REST endpoints, and returns addresses that respond as
    valid devices.

    Returns
    -------
    list of IPAddress
        Discovered LAN-XI device addresses.
    """

    def test_lanxi_candidate
    """
    Test whether an IPv4 address corresponds to a LAN-XI device.

    Queries LAN-XI REST endpoints for module information and sync mode. If the
    device responds, a host name is constructed from the module type and serial
    number.

    Parameters
    ----------
    ipv4_address : str
        Candidate IPv4 address.

    Returns
    -------
    tuple
        Tuple ``(host_name, ipv4_address, info, sync, valid)``.
    """

    class IPAddress
    """
    Container for host name, IPv4 address, IPv6 address, and LAN-XI metadata.

    Parameters
    ----------
    host_name : str, optional
        Host name associated with the device.
    ipv4_address : str, optional
        IPv4 address.
    ipv6_address : str, optional
        IPv6 address.
    valid_ip : bool, optional
        Whether the address has been validated.

    Attributes
    ----------
    host_name : str or None
        Device host name.
    ipv4_address : str or None
        IPv4 address.
    ipv6_address : str or None
        IPv6 address.
    valid_ip : bool
        Whether the address has been validated.
    module_info : dict or None
        LAN-XI module information from REST API.
    sync_type : dict or None
        LAN-XI sync mode information from REST API.
    validation_timeout : float
        Timeout in seconds for validation REST requests.
    """

    def get_ip_from_host_name
    """
    Resolve IP addresses from the stored host name.

    Updates IPv4 and IPv6 address fields using ``socket.getaddrinfo``. If the
    host name is missing or resolution fails, ``valid_ip`` is set to ``False``.
    """

    def get_host_name_from_ip
    """
    Resolve LAN-XI host information from the stored IP address.

    Queries LAN-XI REST endpoints using IPv4 or IPv6 address, stores module and
    sync information, constructs the host name, and marks the address valid if
    successful.
    """

    def validate
    """
    Validate the stored IP address as a LAN-XI device.

    Queries LAN-XI REST endpoints and updates module information, sync type,
    and validity state.
    """

### Loading Utilities
<!---
MARK: Utility Loading Functions
--->
    def load_time_history
    """
    Load a time-history signal from disk.

    Supports NumPy ``.npy``, NumPy archive ``.npz``, and MATLAB ``.mat``
    files. For ``.npz`` and ``.mat`` files, the signal is read from the
    ``"signal"`` field and optional time data is read from the ``"t"`` field.
    If time data is present, the signal is interpolated to ``sample_rate``.
    Odd-length signals are truncated by one sample.

    Parameters
    ----------
    signal_path : str
        Path to the signal file.
    sample_rate : float
        Desired sample rate for interpolated output.

    Returns
    -------
    numpy.ndarray
        Loaded time-history signal.

    Raises
    ------
    ValueError
        If the file extension is not recognized.
    """

    def load_csv_matrix
    """
    Load a comma-separated matrix from a text file.

    Parameters
    ----------
    file : str
        Path to the CSV file.

    Returns
    -------
    list of list of str
        Matrix values read as stripped strings.
    """

    def save_csv_matrix
    """
    Save a two-dimensional string matrix to a CSV file.

    Parameters
    ----------
    data : iterable of iterable of str
        Matrix rows to write.
    file : str
        Output file path.
    """

    def load_python_module
    """
    Load a Python module from a file path at runtime.

    Parameters
    ----------
    module_path : str
        Path to the Python module file.

    Returns
    -------
    module
        Loaded Python module object.
    """

    def read_transformation_matrix_from_worksheet
    """
    Read a numeric transformation matrix from an Excel worksheet.

    Returns ``None`` if the first cell is blank or contains ``"None"``.
    Otherwise, reads ``num_rows`` rows starting at ``start_row`` and
    ``start_col`` until blank or comment cells are encountered.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet containing matrix values.
    start_row : int
        First row of the matrix.
    num_rows : int
        Number of rows to read.
    start_col : int
        First column of the matrix.

    Returns
    -------
    numpy.ndarray or None
        Loaded transformation matrix, or ``None``.
    """

### Math Operations
<!---
MARK: Utility Math Operations
--->
    def coherence
    """
    Compute coherence from a cross-power spectral density matrix.

    Parameters
    ----------
    cpsd_matrix : numpy.ndarray
        Complex CPSD array with shape ``(frequency_lines, rows, columns)``.
    row_column : tuple of int, optional
        Optional ``(row, column)`` pair. If supplied, coherence is computed
        only for that pair.

    Returns
    -------
    numpy.ndarray
        Coherence values.
    """

    def cpsd_to_time_history
    """
    Generate a time-history realization from a CPSD matrix.

    Uses singular value decomposition to synthesize random signals consistent
    with the supplied CPSD matrix, then transforms the frequency-domain
    realization to the time domain.

    Parameters
    ----------
    cpsd_matrix : numpy.ndarray
        Complex CPSD matrix with shape ``(frequency_lines, channels, channels)``.
    sample_rate : float
        Sample rate in samples per second.
    df : float
        Frequency spacing of the CPSD matrix.
    output_oversample : int, optional
        Output oversampling factor. Defaults to ``1``.

    Returns
    -------
    numpy.ndarray
        Time-history output array with shape ``(channels, samples)``.

    Notes
    -----
    Uses the process described by Schultz and Nelson for open-loop
    multiple-input/multiple-output input signal synthesis.
    """

    def reduce_array_by_coordinate
    """
    Reduce an array by matching coordinate strings.

    Selects and sign-corrects entries from an array according to requested
    control and excitation coordinates. Supports two-dimensional and
    three-dimensional coordinate layouts.

    Parameters
    ----------
    array : numpy.ndarray
        Data array to reduce.
    coordinate : numpy.ndarray
        Coordinates associated with ``array``.
    control_coordinate : numpy.ndarray
        Requested control coordinates.
    excitation_coordinate : numpy.ndarray, optional
        Requested excitation coordinates. If omitted, control coordinates are
        used.

    Returns
    -------
    numpy.ndarray
        Reduced and sign-corrected array.

    Raises
    ------
    ValueError
        If a requested coordinate is not found.
    """

    def db2scale
    """
    Convert decibels to linear scale.

    Parameters
    ----------
    decibel : float
        Value in decibels.

    Returns
    -------
    float
        Linear scale value.
    """

    def power2db
    """
    Convert power quantity to decibels.

    Parameters
    ----------
    power : float or numpy.ndarray
        Power value.

    Returns
    -------
    float or numpy.ndarray
        Power in decibels.
    """

    def scale2db
    """
    Convert linear scale quantity to decibels.

    Parameters
    ----------
    scale : float or numpy.ndarray
        Linear scale value.

    Returns
    -------
    float or numpy.ndarray
        Scale in decibels.
    """

    def rms_time
    """
    Compute root-mean-square value of a time signal.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal over which RMS is computed.
    axis : int, optional
        Axis over which to compute the mean.
    keepdims : bool, optional
        Whether to preserve reduced dimensions.

    Returns
    -------
    float or numpy.ndarray
        RMS value.
    """

    def rms_csd
    """
    Compute RMS from a cross-spectral density matrix.

    Parameters
    ----------
    csd : numpy.ndarray
        Complex CSD matrix with frequency as the first dimension.
    df : float
        Frequency spacing.

    Returns
    -------
    numpy.ndarray
        RMS values for each channel.
    """

    def trac
    """
    Compute the time response assurance criterion.

    Parameters
    ----------
    th_1 : numpy.ndarray
        First time-history signal array.
    th_2 : numpy.ndarray, optional
        Second time-history signal array. If omitted, ``th_1`` is compared to
        itself.

    Returns
    -------
    numpy.ndarray
        TRAC values for each signal or signal pair.
    """

    def moving_sum
    """
    Compute a moving sum along the final axis.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal array.
    n : int
        Window length.

    Returns
    -------
    numpy.ndarray
        Moving sum values.
    """

### Correlation and Alignment
<!---
MARK: Utility Correlation and Alignment
--->
    def corr_norm_signal_spec
    """
    Compute correlation normalized by signal and specification norms.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal being searched.
    specification : numpy.ndarray
        Reference signal.

    Returns
    -------
    numpy.ndarray
        Normalized correlation signal.
    """

    def corr_norm_spec2
    """
    Compute correlation normalized by the squared specification norm.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal being searched.
    specification : numpy.ndarray
        Reference signal.

    Returns
    -------
    numpy.ndarray
        Normalized correlation signal.
    """

    def norm_ratio
    """
    Compute a norm-ratio similarity metric.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal being searched.
    specification : numpy.ndarray
        Reference signal.

    Returns
    -------
    numpy.ndarray
        Norm-ratio metric values.
    """

    def correlation_norm_spec_ratio
    """
    Compute correlation weighted by specification norm and norm-ratio penalty.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal being searched.
    specification : numpy.ndarray
        Reference signal.

    Returns
    -------
    numpy.ndarray
        Weighted correlation metric.
    """

    def correlation_norm_signal_spec_ratio
    """
    Compute correlation normalized by signal and specification norms with a
    norm-ratio penalty.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal being searched.
    specification : numpy.ndarray
        Reference signal.

    Returns
    -------
    numpy.ndarray
        Weighted correlation metric.
    """

    def align_signals
    """
    Align a measurement buffer to a specification signal.

    Computes the best delay between a measured signal buffer and a reference
    specification using either simple correlation or a supplied correlation
    metric. Optionally performs subsample alignment using FFT phase slope.

    Parameters
    ----------
    measurement_buffer : numpy.ndarray
        Measurement signal buffer.
    specification : numpy.ndarray
        Reference signal to align against.
    correlation_threshold : float, optional
        Minimum acceptable correlation. Defaults to ``0.9``.
    perform_subsample : bool, optional
        If ``True``, perform phase-based subsample alignment.
    correlation_metric : callable, optional
        Custom metric used to compute correlation.

    Returns
    -------
    tuple
        Tuple ``(spec_portion_aligned, delay, mean_phase_slope,
        found_correlation)``. Returns ``(None, None, None, None)`` when the
        correlation threshold is not met.
    """

    def shift_signal
    """
    Shift a signal using sample delay and FFT phase slope.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal to shift.
    samples_to_keep : int
        Number of samples to retain after shifting.
    sample_delay : int
        Integer sample delay.
    phase_slope : float
        Phase slope used for subsample alignment.

    Returns
    -------
    numpy.ndarray
        Shifted signal.
    """

    def wrap
    """
    Wrap values into a symmetric interval.

    Parameters
    ----------
    data : numpy.ndarray or float
        Values to wrap.
    period : float, optional
        Wrapping period. Defaults to \(2\pi\).

    Returns
    -------
    numpy.ndarray or float
        Wrapped values.
    """

### Overlap Buffer
<!---
MARK: Overlap Buffer
--->
    class OverlapBuffer
    """
    Fixed-size buffer that supports overlapped reads.

    Data are stored in a NumPy array. New data can be appended while older data
    are shifted out. Reads may either update the internal buffer position or
    leave it unchanged, enabling overlapped processing.

    Parameters
    ----------
    shape : tuple
        Shape of the underlying buffer array.
    buffer_axis : int, optional
        Axis used as the buffer/time axis. Defaults to ``-1``.
    starting_value : scalar or array-like, optional
        Initial value used to fill the buffer. Defaults to ``0``.
    dtype : str or numpy.dtype, optional
        Data type of the buffer array. Defaults to ``"float64"``.

    Attributes
    ----------
    buffer_position : int
        Current number of available samples in the buffer.
    buffer_axis : int
        Positive axis index used as the buffer axis.
    buffer_data : numpy.ndarray
        Underlying buffer data.
    shape : tuple
        Shape of the buffer data.
    """

    def __init__
    """
    Initialize the overlap buffer.

    Creates the underlying NumPy buffer, fills it with the starting value,
    stores the buffer axis as a positive index, and initializes the buffer
    position to zero.

    Parameters
    ----------
    shape : tuple
        Shape of the buffer array.
    buffer_axis : int, optional
        Axis used as the buffer dimension.
    starting_value : scalar or array-like, optional
        Initial value used to fill the buffer.
    dtype : str or numpy.dtype, optional
        Buffer data type.
    """

    property
    def buffer_position
    """
    Return the current buffer position.

    Returns
    -------
    int
        Number of samples currently available for reading.
    """

    property
    def buffer_axis
    """
    Return the buffer axis.

    Returns
    -------
    int
        Positive axis index used as the buffer dimension.
    """

    property
    def buffer_data
    """
    Return the underlying buffer array.

    Returns
    -------
    numpy.ndarray
        Buffer data.
    """

    def add_data_noshift
    """
    Add data to the buffer without changing the buffer position.

    If the supplied data are longer than the buffer along the buffer axis, only
    the most recent samples that fit in the buffer are retained.

    Parameters
    ----------
    data : array-like
        Data to append to the buffer.
    """

    def add_data
    """
    Add data to the buffer and advance the buffer position.

    Parameters
    ----------
    data : array-like
        Data to append to the buffer.
    """

    def get_data_noshift
    """
    Get data from the buffer without changing the buffer position.

    Parameters
    ----------
    num_samples : int
        Number of samples to retrieve.

    Returns
    -------
    numpy.ndarray
        Requested buffer data.

    Raises
    ------
    ValueError
        If more samples are requested than are currently available.
    """

    def get_data
    """
    Get data from the buffer and update the buffer position.

    Parameters
    ----------
    num_samples : int
        Number of samples to retrieve.
    buffer_shift : int, optional
        Explicit buffer position shift. If omitted, the buffer position is
        reduced by ``num_samples``.

    Returns
    -------
    numpy.ndarray
        Requested buffer data.
    """

    def shift_buffer_position
    """
    Shift the current buffer position.

    The resulting position is clamped to the range from zero to the buffer
    length along the buffer axis.

    Parameters
    ----------
    samples : int
        Number of samples by which to shift the buffer position.
    """

    def set_buffer_position
    """
    Set the current buffer position.

    The supplied position is clamped to the range from zero to the buffer
    length along the buffer axis.

    Parameters
    ----------
    position : int, optional
        New buffer position. Defaults to ``0``.
    """

    def __getitem__
    """
    Return an item from the underlying buffer data.

    Parameters
    ----------
    key : Any
        NumPy indexing key.

    Returns
    -------
    Any
        Indexed buffer data.
    """

    property
    def shape
    """
    Return the shape of the buffer data.

    Returns
    -------
    tuple
        Shape of the underlying buffer array.
    """

## Load Utilities
<!---
MARK: Load Utilities
--->
    """
    Utilities for loading and saving Rattlesnake metadata.

    This module provides helper functions for serializing and deserializing
    hardware metadata, environment metadata, and profile events to and from
    supported Rattlesnake file formats. Supported formats include netCDF files
    and Excel workbooks.
    """

### netCDF Metadata Loading and Saving
<!---
MARK: netCDF Metadata Loading and Saving
--->
    def load_metadata_from_netcdf
    """
    Load hardware and environment metadata from a netCDF dataset.

    Reads the hardware type from the dataset, loads hardware metadata using the
    corresponding registered hardware metadata class, then loads each stored
    environment using the registered environment metadata class. For newer
    files, environment type information is read directly from the
    ``environment_types`` variable. For older files, the environment type is
    inferred from environment-group attributes.

    Parameters
    ----------
    dataset : nc4.Dataset
        Open netCDF dataset containing Rattlesnake hardware and environment
        metadata.

    Returns
    -------
    tuple
        Tuple ``(hardware_metadata, environment_metadata_list)`` where
        ``hardware_metadata`` is a ``HardwareMetadata`` subclass instance and
        ``environment_metadata_list`` is a list of ``EnvironmentMetadata``
        subclass instances.

    Raises
    ------
    RattlesnakeError
        If an older netCDF file does not contain recognizable environment
        metadata attributes.

    Unit Tests
    ----------
    test_load_metadata_from_netcdf
        Verifies that hardware metadata and environment metadata are loaded
        from a netCDF dataset using the registered metadata classes.
    """

    def discover_environment_type_in_old_netcdf
    """
    Infer an environment type from attributes in an older netCDF group.

    Older streaming files may not contain explicit environment type variables.
    This helper inspects known environment-specific attributes to infer the
    environment type.

    Parameters
    ----------
    environment_group : nc4._netCDF4.Group
        netCDF group containing environment metadata.

    Returns
    -------
    EnvironmentType
        Inferred environment type.

    Raises
    ------
    RattlesnakeError
        If the group does not contain recognizable attributes for any supported
        legacy environment type.

    Unit Tests
    ----------
    test_discover_environment_type_in_old_netcdf
        Verifies that known legacy attributes map to the expected environment
        types and unrecognized groups raise ``RattlesnakeError``.
    """

    def save_rattlesnake_to_netcdf
    """
    Save hardware and environment metadata to a netCDF dataset.

    Writes hardware metadata, environment names, environment types,
    environment active channel masks, and each environment's metadata group to
    the supplied netCDF dataset.

    Parameters
    ----------
    netcdf_dataset : nc4.Dataset
        Open netCDF dataset where metadata should be stored.
    hardware_metadata : HardwareMetadata
        Hardware metadata to write.
    environment_metadata_dict : dict of str to EnvironmentMetadata
        Mapping from environment queue names or environment names to
        environment metadata objects.

    Unit Tests
    ----------
    test_save_rattlesnake_to_netcdf
        Verifies that hardware metadata, environment names, environment types,
        active channel masks, and environment groups are written to the netCDF
        dataset.
    """

### Workbook Metadata Loading and Saving
<!---
MARK: Workbook Metadata Loading and Saving
--->
    def load_metadata_from_workbook
    """
    Load hardware metadata, environment metadata, and profile events from an
    Excel workbook.

    Reads the hardware type from the ``Hardware`` worksheet, loads hardware
    metadata using the corresponding registered metadata class, reads
    environment channel masks from the channel table worksheet, loads each
    environment worksheet using the registered environment metadata class, and
    loads profile events from the ``Test Profile`` worksheet.

    Parameters
    ----------
    workbook : openpyxl.workbook.workbook.Workbook
        Workbook containing Rattlesnake metadata worksheets.

    Returns
    -------
    tuple
        Tuple ``(hardware_metadata, environment_metadata_list,
        profile_event_list)``.

    Raises
    ------
    KeyError
        If required worksheets are missing.
    RattlesnakeError
        If profile loading encounters an invalid command.

    Unit Tests
    ----------
    test_load_metadata_from_workbook
        Verifies that hardware metadata, environment metadata, and profile
        events are loaded from a workbook.
    """

    def save_rattlesnake_to_workbook
    """
    Save hardware metadata, environment metadata, and profile events to an
    Excel workbook.

    Writes a blank hardware template, stores hardware metadata, annotates the
    channel table with environment membership, creates one worksheet per
    environment, writes environment metadata templates or values, creates the
    ``Test Profile`` worksheet, and writes profile events.

    Parameters
    ----------
    workbook : openpyxl.workbook.workbook.Workbook
        Workbook where metadata should be written.
    hardware_metadata : HardwareMetadata
        Hardware metadata to write.
    environment_metadata_dict : dict
        Mapping from environment names to either ``EnvironmentMetadata`` objects
        or ``EnvironmentType`` values. ``EnvironmentType`` values cause a blank
        environment worksheet template to be created.
    profile_event_list : list of ProfileEvent, optional
        Profile events to write to the workbook.

    Unit Tests
    ----------
    test_save_rattlesnake_to_workbook
        Verifies that hardware metadata, environment channel markers,
        environment worksheets, and profile events are written to the workbook.
    """

### Profile Workbook Loading and Saving
<!---
MARK: Profile Workbook Loading and Saving
--->
    def load_profile_from_workbook
    """
    Load profile events from an Excel workbook.

    Reads the ``Test Profile`` worksheet row by row starting at row 2. Each row
    is converted into a ``ProfileEvent`` by reading timestamp, environment
    name, command, and data. Command strings are converted to
    ``GlobalCommands`` or environment-specific command enum members. Rows that
    specify ``SET_ENVIRONMENT_INSTRUCTIONS`` are skipped because instruction
    objects cannot be reconstructed directly from the workbook.

    Parameters
    ----------
    workbook : openpyxl.workbook.workbook.Workbook
        Workbook containing a ``Test Profile`` worksheet.
    environment_types : dict
        Mapping from environment names to ``EnvironmentType`` values or
        ``"Global"`` for global events.

    Returns
    -------
    list of ProfileEvent
        Loaded profile event list.

    Raises
    ------
    RattlesnakeError
        If a command string cannot be resolved for the specified environment.

    Unit Tests
    ----------
    test_load_profile_from_workbook
        Verifies that global and environment-specific profile commands are
        loaded correctly from a worksheet and that invalid command strings
        raise ``RattlesnakeError``.
    """

    def save_profile_to_workbook
    """
    Save profile events to an Excel worksheet.

    Writes profile column headers and one row per profile event. Events using
    ``UICommands.SET_ENVIRONMENT_INSTRUCTIONS`` are skipped because instruction
    objects are not serialized to the profile worksheet.

    Parameters
    ----------
    profile_sheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet where profile events should be written.
    profile_event_list : list of ProfileEvent, optional
        Profile events to write.

    Unit Tests
    ----------
    test_save_profile_to_workbook
        Verifies that profile headers and event rows are written and that
        instruction-update events are skipped.
    """

# Hardware

## Abstract Hardware
<!---
MARK: Abstract Hardware
--->
    """
    Abstract hardware definition that can be used to implement new hardware
    devices.

    This module defines the common interfaces used by the controller to
    describe hardware metadata, acquire data from hardware, and write output
    data to hardware. Hardware implementations should subclass these abstract
    classes to provide device-specific metadata validation, acquisition setup,
    output setup, and file serialization behavior.
    """

### Hardware Metadata
<!---
MARK: Hardware Metadata
--->
    class HardwareMetadata
    """
    Abstract base class for metadata required to define a hardware setup.

    Hardware metadata stores the hardware type, channel table, sampling
    parameters, read and write timing, and output oversampling used by the
    acquisition, output, and streaming processes. Subclasses should extend this
    class with hardware-specific parameters and validation logic.

    Parameters
    ----------
    hardware_type : HardwareType
        Type of hardware represented by this metadata object.
    channel_list : list of Channel
        List of hardware channels used by the test.
    sample_rate : int
        Acquisition sample rate in samples per second.
    time_per_read : float
        Duration, in seconds, of each acquisition read frame.
    time_per_write : float
        Duration, in seconds, of each output write frame before output
        oversampling is applied.
    output_oversample : int, optional
        Output oversampling factor. Defaults to ``1``.

    Attributes
    ----------
    hardware_type : HardwareType
        Type of hardware represented by this metadata object.
    channel_list : list of Channel
        List of configured hardware channels.
    sample_rate : int
        Acquisition sample rate.
    time_per_read : float
        Time duration of each acquisition read frame.
    time_per_write : float
        Time duration of each output write frame.
    output_oversample : int
        Output oversampling factor.

    Unit Tests
    ----------
    test_hardware_metadata_init
        Verifies that mock hardware metadata initializes required attributes
        and is an instance of ``HardwareMetadata``.
    """

    def __init__
    """
    Initialize hardware metadata.

    Stores the common hardware metadata required by acquisition, output, and
    streaming processes. Hardware-specific metadata subclasses should call this
    constructor and then store any additional device-specific configuration
    parameters.

    Parameters
    ----------
    hardware_type : HardwareType
        Type of hardware represented by this metadata object.
    channel_list : list of Channel
        List of hardware channels used by the test.
    sample_rate : int
        Acquisition sample rate in samples per second.
    time_per_read : float
        Duration, in seconds, of each acquisition read frame.
    time_per_write : float
        Duration, in seconds, of each output write frame before output
        oversampling is applied.
    output_oversample : int, optional
        Output oversampling factor. Defaults to ``1``.

    Unit Tests
    ----------
    test_hardware_metadata_init
        Confirms that initialization stores the expected hardware metadata
        attributes.
    """

    property
    def channel_list
    """
    Return the configured hardware channel list.

    Returns
    -------
    list of Channel
        Hardware channels associated with this metadata object.

    Unit Tests
    ----------
    test_hardware_metadata_init
        Verifies that the hardware metadata object exposes a channel list.
    """

    setter
    def channel_list
    """
    Set the configured hardware channel list.

    Parameters
    ----------
    value : list of Channel
        Hardware channel list to store on the metadata object.

    Unit Tests
    ----------
    test_hardware_metadata_validate
        Sets the channel list before validating duplicate-channel behavior.
    """

    property
    def samples_per_read
    """
    Return the number of samples in each acquisition read frame.

    The value is computed from ``sample_rate`` and ``time_per_read`` as:

    \[
    \mathrm{samples\_per\_read} =
    \operatorname{round}(\mathrm{sample\_rate} \cdot \mathrm{time\_per\_read})
    \]

    Returns
    -------
    int
        Number of samples per read frame.

    Unit Tests
    ----------
    test_hardware_metadata_properties
        Verifies that this property returns the expected sample count.
    """

    property
    def samples_per_write
    """
    Return the number of samples in each output write frame.

    The value is computed from ``sample_rate``, ``time_per_write``, and
    ``output_oversample`` as:

    \[
    \mathrm{samples\_per\_write} =
    \operatorname{round}(
    \mathrm{sample\_rate}
    \cdot \mathrm{time\_per\_write}
    \cdot \mathrm{output\_oversample})
    \]

    Returns
    -------
    int
        Number of samples per output write frame.

    Unit Tests
    ----------
    test_hardware_metadata_properties
        Verifies that this property includes the output oversampling factor.
    """

    property
    def nyquist_frequency
    """
    Return the Nyquist frequency of the acquisition system.

    The Nyquist frequency is half of the acquisition sample rate.

    Returns
    -------
    float
        Nyquist frequency in hertz.

    Unit Tests
    ----------
    test_hardware_metadata_properties
        Verifies that this property returns half of the sample rate.
    """

    property
    def output_sample_rate
    """
    Return the output sample rate.

    The output sample rate is computed from the acquisition sample rate and
    output oversampling factor.

    Returns
    -------
    int or float
        Output sample rate in samples per second.

    Unit Tests
    ----------
    test_hardware_metadata_properties
        Verifies that this property returns the sample rate multiplied by the
        output oversampling factor.
    """

    def validate
    """
    Validate the hardware metadata.

    Performs common validation checks required by all hardware metadata
    subclasses. Subclasses should extend this method with device-specific
    checks such as connected-device validation, valid physical channels,
    supported sample rates, channel limits, coupling modes, excitation
    settings, and required hardware files.

    Raises
    ------
    RattlesnakeError
        If duplicate channels are found in ``channel_list``.
    RattlesnakeError
        If subclass-specific validation fails.

    Unit Tests
    ----------
    test_hardware_metadata_validate
        Verifies that valid channel lists pass validation and duplicate channel
        lists raise ``RattlesnakeError``.
    """

    def valid_channel_dict
    """
    Return valid values for each channel attribute.

    Creates a dictionary whose keys are channel attribute names and whose
    values are lists of valid entries for those attributes. Subclasses should
    override or extend the returned dictionary with hardware-specific valid
    values used by the UI and validation logic.

    Parameters
    ----------
    channel : Channel
        Channel whose valid values should be queried.

    Returns
    -------
    dict
        Mapping from channel attribute names to lists of valid values.

    Unit Tests
    ----------
    test_hardware_metadata_validate
        Exercises metadata validation using mock channel definitions.
    """

    property
    def assist_mode_modules
    """
    Return hardware assist modules for channel attributes.

    The returned dictionary maps each channel attribute name to the hardware
    assist module used to populate, validate, or assist with that field in the
    user interface. Subclasses should override values for attributes that have
    hardware-specific assist behavior.

    Returns
    -------
    dict
        Mapping from channel attribute names to ``HardwareAssistModules``
        values.

    Unit Tests
    ----------
    test_hardware_metadata_init
        Verifies that mock hardware metadata can be instantiated with required
        abstract properties implemented.
    """

    classmethod
    def load_channel_table_from_netcdf
    """
    Load a channel table from a netCDF dataset.

    Reads the ``channels`` group and ``response_channels`` dimension from a
    netCDF dataset and reconstructs a list of ``Channel`` objects. Blank string
    values are converted to ``None``. Empty channels are skipped.

    Parameters
    ----------
    netcdf_dataset : nc4.Dataset
        netCDF dataset containing a saved channel table.

    Returns
    -------
    list of Channel
        Channel list reconstructed from the netCDF dataset.

    Unit Tests
    ----------
    test_hardware_metadata_load_save_netcdf
        Saves hardware metadata to netCDF, reloads it, and verifies that the
        channel table is reconstructed correctly.
    """

    classmethod
    def save_channel_table_to_workbook
    """
    Save a hardware channel table to an Excel workbook.

    Writes the channel table to the active worksheet, including merged header
    regions, column labels, channel indices, and channel attribute values.
    The active worksheet is renamed ``"Channel Table"``.

    Parameters
    ----------
    channel_list : list of Channel
        Channels to write to the workbook.
    workbook : openpyxl.workbook.workbook.Workbook
        Workbook where the channel table should be stored.

    Unit Tests
    ----------
    test_hardware_metadata_load_save_workbook
        Saves hardware metadata to a workbook and verifies that channel data
        can be reloaded.
    """

    classmethod
    def load_channel_table_from_workbook
    """
    Load a hardware channel table from an Excel workbook.

    Finds the channel table worksheet, reads channel rows starting at row 3,
    and reconstructs ``Channel`` objects from worksheet cell values. Reading
    stops when an empty channel row is encountered.

    Parameters
    ----------
    workbook : openpyxl.workbook.workbook.Workbook
        Workbook containing a channel table worksheet.

    Returns
    -------
    list of Channel
        Channel list reconstructed from the workbook.

    Raises
    ------
    RattlesnakeError
        If multiple candidate channel table sheets are found.
    RattlesnakeError
        If no channel table sheet is found.

    Unit Tests
    ----------
    test_hardware_metadata_load_save_workbook
        Saves hardware metadata to a workbook, reloads it, and verifies that
        the channel table is reconstructed correctly.
    """

    classmethod
    def save_blank_hardware_to_workbook
    """
    Save a blank hardware metadata template to an Excel workbook.

    Creates or updates the ``"Hardware"`` worksheet with common hardware
    metadata fields, explanatory comments, and hardware-specific optional
    settings. This template is used to help users populate hardware metadata
    in spreadsheet form.

    Parameters
    ----------
    workbook : openpyxl.workbook.workbook.Workbook
        Workbook where the blank hardware template should be written.

    Unit Tests
    ----------
    test_hardware_metadata_save_blank_hardware_to_workbook
        Verifies that a blank hardware worksheet is created with expected
        labels and comments.
    """

    classmethod
    def load_metadata_from_workbook
    """
    Load hardware metadata values from an Excel workbook.

    Reads the channel table and common hardware metadata fields from a
    workbook. This method returns the parsed constructor values rather than
    constructing a concrete metadata object directly. Subclasses should call
    this method and use the returned values to instantiate the appropriate
    hardware metadata subclass.

    Parameters
    ----------
    workbook : openpyxl.workbook.workbook.Workbook
        Workbook containing hardware metadata and a channel table.

    Returns
    -------
    tuple
        Tuple containing ``hardware_type``, ``channel_list``, ``sample_rate``,
        ``time_per_read``, ``time_per_write``, and ``output_oversample``.

    Unit Tests
    ----------
    test_hardware_metadata_load_save_workbook
        Verifies that hardware metadata saved to a workbook can be parsed and
        reconstructed.
    """

    def save_metadata_to_netcdf
    """
    Save hardware metadata to a netCDF dataset.

    Creates dimensions, attributes, variables, and channel table entries needed
    to document the hardware setup in a streaming netCDF file. Subclasses
    should extend this method with hardware-specific metadata fields.

    Parameters
    ----------
    netcdf_dataset : nc4.Dataset
        netCDF dataset where hardware metadata should be stored.

    Unit Tests
    ----------
    test_hardware_metadata_load_save_netcdf
        Saves hardware metadata to netCDF, reloads it, and verifies that the
        metadata and channel table are reconstructed correctly.
    """

    classmethod
    def load_metadata_from_netcdf
    """
    Load hardware metadata values from a netCDF dataset.

    Reads common hardware metadata attributes and the channel table from a
    netCDF dataset. This method returns the parsed constructor values rather
    than constructing a concrete metadata object directly. Subclasses should
    call this method and use the returned values to instantiate the appropriate
    hardware metadata subclass.

    Parameters
    ----------
    netcdf_dataset : nc4.Dataset
        netCDF dataset containing saved hardware metadata.

    Returns
    -------
    tuple
        Tuple containing ``hardware_type``, ``channel_list``, ``sample_rate``,
        ``time_per_read``, ``time_per_write``, and ``output_oversample``.

    Unit Tests
    ----------
    test_hardware_metadata_load_save_netcdf
        Verifies that saved hardware metadata can be parsed and reconstructed
        from a netCDF dataset.
    """

    def save_metadata_to_workbook
    """
    Save hardware metadata to an Excel workbook.

    Writes the channel table and common hardware metadata fields to the
    supplied workbook. Subclasses should extend this method with
    hardware-specific fields such as device paths, trigger options, virtual
    hardware settings, or process limits.

    Parameters
    ----------
    workbook : openpyxl.workbook.workbook.Workbook
        Workbook where hardware metadata should be stored.

    Unit Tests
    ----------
    test_hardware_metadata_load_save_workbook
        Saves hardware metadata to a workbook, reloads it, and verifies that
        the metadata can be reconstructed.
    """

### Hardware Acquisition
<!---
MARK: Hardware Acquisition
--->
    class HardwareAcquisition
    """
    Abstract base class defining the interface between the controller and
    hardware acquisition.

    Hardware acquisition subclasses are responsible for configuring input
    channels, starting acquisition, reading frames of measured data, reading
    remaining buffered data, stopping acquisition, closing hardware resources,
    and reporting acquisition delay.

    Unit Tests
    ----------
    test_hardware_acquisition_init
        Verifies that the mock acquisition class is an instance of
        ``HardwareAcquisition``.

    test_hardware_acquisition_functions
        Calls all acquisition interface methods on a mock implementation and
        verifies expected return types.
    """

    def initialize_hardware
    """
    Initialize acquisition hardware.

    Configures hardware input channels, acquisition sample rates, timing, and
    any other device-specific acquisition state using the supplied metadata.

    Parameters
    ----------
    metadata : HardwareMetadata
        Hardware metadata defining channel configuration, sample rate, timing,
        and device-specific setup parameters.

    Unit Tests
    ----------
    test_hardware_acquisition_functions
        Verifies that the mock acquisition implementation accepts hardware
        metadata during initialization.
    """

    def start
    """
    Start acquiring data from the hardware.

    Implementations should arm or start the hardware acquisition task so that
    subsequent calls to ``read`` return measured data.

    Unit Tests
    ----------
    test_hardware_acquisition_functions
        Verifies that the mock acquisition implementation can be started.
    """

    def read
    """
    Read one frame of acquired data from the hardware.

    Returns
    -------
    numpy.ndarray
        Array containing one frame of acquired data. The expected shape is
        typically channels by samples.

    Unit Tests
    ----------
    test_hardware_acquisition_functions
        Verifies that the mock acquisition implementation returns a NumPy
        array.
    """

    def read_remaining
    """
    Read remaining buffered data from the hardware.

    This method is used during shutdown or flushing to retrieve any data that
    remains available after normal frame acquisition has stopped.

    Returns
    -------
    numpy.ndarray
        Array containing the remaining acquired data.

    Unit Tests
    ----------
    test_hardware_acquisition_functions
        Verifies that the mock acquisition implementation returns a NumPy
        array.
    """

    def stop
    """
    Stop acquisition.

    Implementations should stop the active acquisition task without releasing
    all hardware resources unless required by the hardware API.

    Unit Tests
    ----------
    test_hardware_acquisition_functions
        Verifies that the mock acquisition implementation can be stopped.
    """

    def close
    """
    Close acquisition hardware resources.

    Implementations should release hardware tasks, handles, sessions, and any
    other resources allocated during initialization or acquisition.

    Unit Tests
    ----------
    test_hardware_acquisition_functions
        Verifies that the mock acquisition implementation can be closed.
    """

    def get_acquisition_delay
    """
    Return the acquisition delay in samples.

    This delay represents the number of samples between output generation and
    acquisition measurement. It is used to account for buffering and hardware
    latency so output and acquisition data can be synchronized.

    Returns
    -------
    int
        Acquisition delay in samples.

    Unit Tests
    ----------
    test_hardware_acquisition_functions
        Verifies that the mock acquisition implementation returns an integer
        delay.
    """

### Hardware Output
<!---
MARK: Hardware Output
--->
    class HardwareOutput
    """
    Abstract base class defining the interface between the controller and
    hardware output.

    Hardware output subclasses are responsible for configuring output channels,
    starting output generation, writing output frames, stopping output,
    closing hardware resources, and reporting whether the hardware is ready to
    accept a new output frame.

    Unit Tests
    ----------
    test_hardware_output_init
        Verifies that the mock output class is an instance of
        ``HardwareOutput``.

    test_hardware_output_functions
        Calls all output interface methods on a mock implementation.
    """

    def initialize_hardware
    """
    Initialize output hardware.

    Configures hardware output channels, output sample rates, timing, and any
    other device-specific output state using the supplied metadata.

    Parameters
    ----------
    metadata : HardwareMetadata
        Hardware metadata defining channel configuration, sample rate, timing,
        output oversampling, and device-specific setup parameters.

    Unit Tests
    ----------
    test_hardware_output_functions
        Verifies that the mock output implementation accepts hardware metadata
        during initialization.
    """

    def start
    """
    Start output generation.

    Implementations should arm or start the hardware output task so that data
    written with ``write`` is sent to the configured output channels.

    Unit Tests
    ----------
    test_hardware_output_functions
        Verifies that the mock output implementation can be started.
    """

    def write
    """
    Write one frame of output data to the hardware.

    Parameters
    ----------
    data : numpy.ndarray
        Output data array to write to the hardware. The expected shape is
        typically output channels by samples.

    Unit Tests
    ----------
    test_hardware_output_functions
        Verifies that the mock output implementation accepts an output data
        array.
    """

    def stop
    """
    Stop output generation.

    Implementations should stop the active output task while avoiding abrupt
    output behavior that could damage hardware or test articles.

    Unit Tests
    ----------
    test_hardware_output_functions
        Verifies that the mock output implementation can be stopped.
    """

    def close
    """
    Close output hardware resources.

    Implementations should release hardware tasks, handles, sessions, and any
    other resources allocated during initialization or output generation.

    Unit Tests
    ----------
    test_hardware_output_functions
        Verifies that the mock output implementation can be closed.
    """

    def ready_for_new_output
    """
    Return whether the hardware is ready to accept new output data.

    This method allows the output process to determine whether another output
    frame should be written to the hardware.

    Returns
    -------
    bool
        ``True`` if the hardware can accept a new output frame, otherwise
        ``False``.

    Unit Tests
    ----------
    test_hardware_output_functions
        Verifies that the mock output implementation exposes this method.
    """

## Hardware Utilities
<!---
MARK: Hardware Utilities
--->
    """
    Utility classes and enumerations for hardware definitions.

    This module defines the supported hardware type enumeration and the
    ``Channel`` metadata container used throughout the controller to describe
    measurement, reference, response, output-feedback, and limit information
    for each hardware channel.
    """

### Hardware Type
<!---
MARK: Hardware Type
--->
    class HardwareType
    """
    Enumeration of supported hardware backends.

    Values are used in hardware metadata, hardware registries, template files,
    and netCDF files to identify which hardware implementation should be used.

    Attributes
    ----------
    NONE : int
        Placeholder or no hardware type.
    NI_DAQMX : int
        National Instruments DAQmx hardware.
    LAN_XI : int
        Brüel & Kjær LAN-XI hardware.
    DP_QUATTRO : int
        Data Physics Quattro hardware.
    DP_900 : int
        Data Physics 900 Series hardware.
    EXODUS : int
        Exodus modal-solution virtual hardware.
    STATE_SPACE : int
        State-space integration virtual hardware.
    SDYNPY_SYSTEM : int
        SDynPy system integration virtual hardware.
    SDYNPY_FRF : int
        SDynPy FRF-based virtual hardware.
    MOCK : int
        Mock hardware used for testing.

    Unit Tests
    ----------
    test_hardware_type
        Verifies that hardware type enum values construct valid
        ``HardwareType`` members.
    """

### Channel
<!---
MARK: Channel
--->
    class Channel
    """
    Metadata container for one hardware channel.

    ``Channel`` stores test-article, sensor, physical hardware, signal
    conditioning, output feedback, and limit metadata for a single controller
    channel. Channel objects are used in hardware metadata, templates, netCDF
    files, worksheet channel tables, validation, acquisition, output, and
    environment channel selection.

    Parameters
    ----------
    node_number : str, optional
        Test article node number.
    node_direction : str, optional
        Test article node direction.
    comment : str, optional
        Additional channel comments.
    serial_number : str, optional
        Sensor or instrument serial number.
    triax_dof : str, optional
        Degree of freedom for a triaxial sensor.
    sensitivity : str, optional
        Sensor sensitivity in mV per engineering unit.
    unit : str, optional
        Engineering unit of the sensor.
    make : str, optional
        Sensor make.
    model : str, optional
        Sensor model.
    expiration : str, optional
        Sensor calibration expiration date.
    physical_device : str, optional
        Physical hardware device name.
    physical_channel : str, optional
        Physical channel on the hardware device.
    channel_type : str, optional
        Channel type.
    minimum_value : str, optional
        Minimum channel value in volts.
    maximum_value : str, optional
        Maximum channel value in volts.
    coupling : str, optional
        Channel coupling type.
    excitation_source : str, optional
        Signal-conditioning excitation source.
    excitation : str, optional
        Signal-conditioning excitation level.
    feedback_device : str, optional
        Physical output device associated with this feedback channel.
    feedback_channel : str, optional
        Physical output channel associated with this feedback channel.
    warning_level : str, optional
        Engineering-unit warning threshold.
    abort_level : str, optional
        Engineering-unit abort threshold.

    Attributes
    ----------
    node_number : str or None
        Test article node number.
    node_direction : str or None
        Test article node direction.
    comment : str or None
        Additional channel comments.
    serial_number : str or None
        Sensor or instrument serial number.
    triax_dof : str or None
        Degree of freedom for a triaxial sensor.
    sensitivity : str or None
        Sensor sensitivity in mV per engineering unit.
    unit : str or None
        Engineering unit.
    make : str or None
        Sensor make.
    model : str or None
        Sensor model.
    expiration : str or None
        Calibration expiration date.
    physical_device : str or None
        Physical hardware device.
    physical_channel : str or None
        Physical hardware channel.
    channel_type : str or None
        Channel type.
    minimum_value : str or None
        Minimum channel value in volts.
    maximum_value : str or None
        Maximum channel value in volts.
    coupling : str or None
        Coupling type.
    excitation_source : str or None
        Excitation source.
    excitation : str or None
        Excitation value.
    feedback_device : str or None
        Associated output feedback device.
    feedback_channel : str or None
        Associated output feedback channel.
    warning_level : str or None
        Warning limit in engineering units.
    abort_level : str or None
        Abort limit in engineering units.

    Unit Tests
    ----------
    test_channel_init
        Verifies that a ``Channel`` object can be initialized with all channel
        fields.

    test_channel_attr_list
        Verifies that ``channel_attr_list`` contains every channel attribute
        stored on the object.

    test_channel_is_empty
        Verifies that a channel is considered empty only when all listed
        attributes are ``None``.

    test_channel_eq
        Verifies equality comparison between channels.

    test_channel_eq_foreign_type
        Verifies comparison behavior with non-channel objects.
    """

    def __init__
    """
    Initialize a channel metadata object.

    Stores all supplied test-article, sensor, hardware, signal-conditioning,
    feedback, warning, and abort metadata fields. All fields default to
    ``None``.

    Parameters
    ----------
    node_number : str, optional
        Test article node number.
    node_direction : str, optional
        Test article node direction.
    comment : str, optional
        Additional channel comments.
    serial_number : str, optional
        Sensor or instrument serial number.
    triax_dof : str, optional
        Degree of freedom for a triaxial sensor.
    sensitivity : str, optional
        Sensor sensitivity in mV per engineering unit.
    unit : str, optional
        Engineering unit of the sensor.
    make : str, optional
        Sensor make.
    model : str, optional
        Sensor model.
    expiration : str, optional
        Calibration expiration date.
    physical_device : str, optional
        Physical hardware device name.
    physical_channel : str, optional
        Physical channel on the hardware device.
    channel_type : str, optional
        Channel type.
    minimum_value : str, optional
        Minimum channel value in volts.
    maximum_value : str, optional
        Maximum channel value in volts.
    coupling : str, optional
        Channel coupling type.
    excitation_source : str, optional
        Signal-conditioning excitation source.
    excitation : str, optional
        Signal-conditioning excitation level.
    feedback_device : str, optional
        Physical output device associated with this feedback channel.
    feedback_channel : str, optional
        Physical output channel associated with this feedback channel.
    warning_level : str, optional
        Engineering-unit warning threshold.
    abort_level : str, optional
        Engineering-unit abort threshold.

    Unit Tests
    ----------
    test_channel_init
        Confirms that a ``Channel`` instance can be constructed.
    """

    property
    def channel_attr_list
    """
    Return channel attribute names in channel-table order.

    The returned order matches the order used by workbook channel tables and
    hardware metadata serialization.

    Returns
    -------
    list of str
        Channel attribute names.

    Unit Tests
    ----------
    test_channel_attr_list
        Verifies that all attributes listed by this property exist on the
        channel and that all channel instance attributes are included.
    """

    property
    def is_empty
    """
    Return whether all channel attributes are unset.

    Returns
    -------
    bool
        ``True`` if every attribute in ``channel_attr_list`` is ``None``,
        otherwise ``False``.

    Unit Tests
    ----------
    test_channel_is_empty
        Verifies empty and non-empty channel detection.
    """

    def is_output_channel
    """
    Return whether this channel is associated with output feedback.

    A channel is considered an output channel when ``feedback_device`` is not
    ``None``.

    Returns
    -------
    bool
        ``True`` if ``feedback_device`` is set, otherwise ``False``.
    """

    def __eq__
    """
    Compare two channel objects for equality.

    Channels are equal when the other object is also a ``Channel`` and all
    attributes in ``channel_attr_list`` compare equal.

    Parameters
    ----------
    other : object
        Object to compare against this channel.

    Returns
    -------
    bool or NotImplemented
        ``True`` if all channel attributes match, ``False`` if any differ, or
        ``NotImplemented`` for non-channel objects.

    Unit Tests
    ----------
    test_channel_eq
        Verifies equality and inequality for channel objects.

    test_channel_eq_foreign_type
        Verifies comparison behavior with a non-channel object.
    """

    def __hash__
    """
    Return a hash based on all channel attributes.

    Returns
    -------
    int
        Hash of the tuple of all channel attributes.

    Notes
    -----
    This allows channels to be used in sets and enables duplicate-channel
    detection in hardware metadata validation.
    """

# Environment
<!---
MARK: Environment
--->


## Abstract Environment
<!---
MARK: Abstract Environment
--->
    """
    Abstract environment that can be used to create new environment control strategies
    in the controller.
    """

### Environment Commands
<!---
MARK: Environment Commands
--->
    class EnvironmentCommands:
    """
    Abstract base enum for commands that a controller can send to an environment.

    This enum is intended to be subclassed and used as a common interface for
    environment command definitions. Subclasses should define
    ``VALID_PROFILE_COMMANDS`` and ``VALID_DATA`` as enum members so they can
    be converted into command-specific values by the class methods provided.
    This class should be added to the ``ENVIRONMENT_COMMANDS`` dictionary
    within the environment registry.

    Attributes
    ----------
    VALID_PROFILE_COMMANDS : tuple of int
        Tuple of command values permitted for use in profile events.
    VALID_DATA : dict of int to type
        Mapping from command values to their expected associated data types.
        Used to validate profile event definitions before they are sent to
        the environment.

    Unit Tests
    ----------
    test_environment_commands_have_unique_integer_values
        Iterates through each enum member to confirm unique integer values.
        Verifies that ``VALID_PROFILE_COMMANDS`` is a tuple of ints and
        ``VALID_DATA`` is a dict mapping ints to types.
    """

    property
    def label:
    """
    Return a user-friendly label for the command.

    Returns
    -------
    str
        User-friendly command label for the UI.

    Unit Tests
    ----------
    test_environment_commands_label
        Checks that labels replace underscores with spaces and use title
        case.
    """

    classmethod
    def valid_profile_commands:
    """
    Return the commands valid for use in profile events.

    Returns
    -------
    tuple of EnvironmentCommands
        Enum members permitted for profile events.

    Unit Tests
    ----------
    test_environment_commands_valid_profile_commands
        Ensures this method returns a tuple of EnvironmentCommands
        members corresponding to VALID_PROFILE_COMMANDS.
    """

    classmethod
    def valid_data:
    """
    Return the valid data types associated with each command.

    Converts the key-value pairs in ``VALID_DATA`` into enum-member keys
    mapped to their data types.

    Returns
    -------
    dict of EnvironmentCommands to type
        Mapping from command members to expected data types.

    Unit Tests
    ----------
    test_environment_commands_valid_data
        Verifies this method returns a dict mapping enum members to their
        predefined types.
    """

### Environment UI Commands
<!---
MARK: Environment UI Commands
--->
    class EnvironmentUICommands
    """
    Base enum for UI-specific commands associated with environments.

    This enum is intended to define commands that are needed by the user
    interface but are not part of the standard environment command set.
    Subclasses may extend this enum with additional UI-only commands used
    to update widgets, communicate UI state, or trigger UI-specific
    behavior.
    """

### Environment Metadata
<!---
MARK: Environment Metadata
--->
    class EnvironmentMetadata
    """
    Abstract base class for storing environment metadata.

    This class stores the parameters required to fully define an
    environment. Metadata objects are typically created by environment UI
    classes and passed to the controller during environment initialization.
    Subclasses must include enough information to reconstruct the
    environment configuration, validate it against hardware metadata, and
    serialize or deserialize it from supported file formats.
    This class should be added to the ``ENVIRONMENT_METADATA`` dictionary
    within the environment registry.

    Parameters
    ----------
    environment_type : EnvironmentType
        Type of environment represented by this metadata object.
    environment_name : str
        Environment name used for logging, UI display, and
        task identification.
    channel_list_bools : list of bool
        Boolean mask identifying which hardware channels are used by this
        environment. Each entry corresponds to an entry in the hardware
        channel list.
    sample_rate : int
        Sample rate associated with the environment.

    Attributes
    ----------
    environment_type : EnvironmentType
        Type of environment represented by this metadata object.
    environment_name : str
        Name used for logging, UI display, and task identification.
    sample_rate : int
        Environment sample rate.
    channel_list_bools : list of bool
        Boolean mask identifying channels assigned to the environment.
    queue_name : str
        Unique queue identifier assigned by the controller for routing
        environment-specific messages. This is assigned by the environment
        manager when spinning up the environment process.

    Unit Tests
    ----------
    test_environment_metadata
        Verifies that subclasses within ``ENVIRONMENT_METADATA`` in the
        registry initialize required metadata attributes and preserve the
        supplied environment name, channel mask, and sample rate.
    """

    def __init__
    """
    Initialize environment metadata.

    Stores the common metadata required by all environment types. The
    environment type should normally be fixed by subclasses when calling
    ``super().__init__`` so users do not need to provide it directly.

    Parameters
    ----------
    environment_type : EnvironmentType
        Type of environment represented by this metadata object.
    environment_name : str
        Environment name used for logging, UI display, and
        task identification.
    channel_list_bools : list of bool
        Boolean mask identifying which hardware channels belong to the
        environment.
    sample_rate : int
        Sample rate associated with the environment.

    Unit Tests
    ----------
    test_environment_metadata_init
        Confirms that initialization stores the environment type,
        environment name, sample rate, channel list bools, and
        initializes ``queue_name`` to ``None``.
    """

    property
    def channel_indices
    """
    Return indices of hardware channels assigned to the environment.

    The indices are computed from ``channel_list_bools`` by returning
    the positions where the mask value is true.

    Returns
    -------
    list of int
        Indices of channels selected for this environment.

    Unit Tests
    ----------
    test_environment_metadata_channel_indices
        Verifies that selected channel indices correspond to true
        entries in ``channel_list_bools``.
    """

    def environment_channel_list
    """
    Return the subset of channels assigned to the environment.

    Applies ``channel_list_bools`` as a Boolean mask to the supplied
    hardware channel list and returns only the channels assigned to
    this environment.

    Parameters
    ----------
    channel_list : list
        Full hardware channel list.

    Returns
    -------
    list of Channel
        Channels from ``channel_list`` whose corresponding
        ``channel_list_bools`` entries are true.

    Unit Tests
    ----------
    test_environment_metadata_environment_channel_list
        Confirms that the returned channel list contains only channels
        selected by ``channel_list_bools`` and preserves their original
        order.
    """

    def validate
    """
    Validate environment metadata against hardware metadata.

    Performs common metadata validation checks required by all
    environments. Subclasses should extend this method with
    environment-specific checks, such as verifying control channels,
    duplicate channel assignments, or environment-specific parameter
    bounds. This should throw an error if metadata fails to pass
    validation.

    Parameters
    ----------
    hardware_metadata : HardwareMetadata
        Hardware metadata containing the available hardware channel
        list and other hardware configuration parameters.

    Raises
    ------
    RattlesnakeError
        If ``environment_type`` is not an ``EnvironmentType``.
    RattlesnakeError
        If ``environment_name`` is not a string.
    RattlesnakeError
        If ``channel_list_bools`` is not the same length as the
        hardware channel list.

    Unit Tests
    ----------
    test_environment_metadata_validate_truth
        Verifies that valid mock metadata class passes the validation check.

    test_environment_metadata_validate_invalid_environment_type
        Verifies that an error is thrown with an invalid environment type object.

    test_environment_metadata_validate_invalid_environment_name
        Verifies that an error is thrown when the environment name is not a string.

    test_environment_metadata_validate_invalid_channel_list
        Verifies that an error is thrown when an invalid channel list is given to the metadata.
    """

    def save_metadata_to_netcdf
    """
    Save environment metadata to a netCDF group.

    Stores parameters for this environment in the supplied netCDF group.
    Subclasses should write all information required to reconstruct the
    environment metadata, using group attributes, dimensions, or
    variables as appropriate.

    Parameters
    ----------
    netcdf_group_handle : nc4._netCDF4.Group
        netCDF group where this environment's metadata should be
        stored.

    Unit Tests
    ----------
    test_environment_metadata_load_save_netcdf
        Saves a valid metadata subclass to a netcdf file and then loads
        it back into a metadata object. Verifies that the metadata object
        is valid and that the netcdf handle contains environment_name and
        environment_type.
    """

    classmethod
    def load_metadata_from_netcdf
    """
    Load environment metadata from a netCDF dataset or group.

    Retrieves metadata previously written by
    ``save_metadata_to_netcdf`` and constructs an environment metadata
    object. Subclasses should read parameters from the group associated
    with ``environment_name`` and use the supplied hardware metadata as
    needed to reconstruct or validate the environment configuration.

    Parameters
    ----------
    netcdf_handle : nc4._netCDF4.Group
        netCDF dataset or group containing stored environment metadata.
    environment_name : str
        Name of the environment whose metadata should be loaded.
    channel_list_bools : list of bool
        Boolean channel mask identifying channels assigned to the
        environment.
    hardware_metadata : HardwareMetadata
        Hardware metadata associated with the stored environment.

    Returns
    -------
    EnvironmentMetadata
        Instance of the metadata subclass populated from the netCDF
        data.

    Unit Tests
    ----------
    test_environment_metadata_load_save_netcdf
        Saves a valid metadata subclass to a netcdf file and then loads
        it back into a metadata object. Verifies that the metadata object
        is valid and that the netcdf handle contains environment_name and
        environment_type.
    """

    classmethod
    def create_blank_worksheet_template
    """
    Create a blank Excel worksheet template for environment metadata.

    Writes the common worksheet fields required by environment metadata
    exports. Subclasses should extend this template with
    environment-specific metadata fields.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet where the blank metadata template should be created.

    Unit Tests
    ----------
    test_environment_metadata_load_save_worksheet
        Saves a valid metadata subclass to a worksheet file and then loads
        it back into a metadata object. Verifies that the metadata object
        is valid and that the worksheet cell 1, 2 contains a string of the
        environment type. 

    Notes
    -----
    Worksheet cell 1, 2 must be set to a string of the environment type.
    """

    def save_metadata_to_worksheet
    """
    Save environment metadata to an Excel worksheet.

    Stores parameters for this environment in the supplied worksheet.
    This method should write all information required to reconstruct the
    metadata from the worksheet. Subclasses should call or extend
    ``create_blank_worksheet_template`` before writing
    environment-specific values.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet where this environment's metadata should be stored.

    Unit Tests
    ----------
    test_environment_metadata_load_save_worksheet
        Saves a valid metadata subclass to a worksheet file and then loads
        it back into a metadata object. Verifies that the metadata object
        is valid and that the worksheet cell 1, 2 contains a string of the
        environment type. 
    """

    classmethod
    def load_metadata_from_worksheeet
    """
    Load environment metadata from an Excel worksheet.

    Retrieves metadata previously written by
    ``save_metadata_to_worksheet`` and constructs an environment
    metadata object. Subclasses should read all required
    environment-specific parameters from the worksheet.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet containing stored environment metadata.
    environment_name : str
        Name of the environment whose metadata should be loaded.
    channel_list_bools : list of bool
        Boolean channel mask identifying channels assigned to the
        environment.
    hardware_metadata : HardwareMetadata
        Hardware metadata associated with the stored environment.

    Returns
    -------
    EnvironmentMetadata
        Instance of the metadata subclass populated from the worksheet.

    Unit Tests
    ----------
    test_environment_metadata_load_save_worksheet
        Saves a valid metadata subclass to a worksheet file and then loads
        it back into a metadata object. Verifies that the metadata object
        is valid and that the worksheet cell 1, 2 contains a string of the
        environment type. 
    """

### Environment Instructions
<!---
MARK: Environment Instructions
--->
    class EnvironmentInstructions
    """
    Abstract base class for environment startup instructions.

    Environment instructions define runtime parameters that are passed to an
    environment when control is started. These parameters are separate from
    environment metadata because they may change between runs and do not
    necessarily need to be stored as part of the environment definition.

    Instructions are commonly sent to the controller before starting an
    environment or when starting a profile. Subclasses should include any
    environment-specific startup values needed by the environment control
    strategy and should validate that those values are compatible with the
    target environment.

    Parameters
    ----------
    environment_type : EnvironmentType
        Type of environment that these instructions apply to.
    environment_name : str
        Name of the environment that these instructions apply to.

    Attributes
    ----------
    environment_type : EnvironmentType
        Type of environment that these instructions apply to. This is used
        to verify that the instructions are sent to the correct environment.
    environment_name : str
        Name of the environment that these instructions apply to.

    Unit Tests
    ----------
    test_environment_instructions
        Verifies that instruction subclasses initialize the required
        environment type and environment name attributes.
    """

    def __init__
    """
    Initialize environment instructions.

    Stores the common instruction attributes required by all environment
    instruction subclasses. The environment type should normally be fixed by
    subclasses when calling ``super().__init__`` so users do not need to
    provide it directly.

    Parameters
    ----------
    environment_type : EnvironmentType
        Type of environment that these instructions apply to.
    environment_name : str
        Name of the environment that these instructions apply to.

    Unit Tests
    ----------
    test_environment_instructions_init
        Confirms that initialization stores the environment type and
        environment name.
    """

    def validate
    """
    Validate environment startup instructions.

    Subclasses should implement this method to verify that all instruction
    values are valid for the corresponding environment. This should include
    checks for value ranges, compatible modes, required startup parameters,
    and any other environment-specific constraints.

    This method should throw an error if the instructions fail validation.

    Raises
    ------
    RattlesnakeError
        If any instruction value is invalid for the environment.

    Unit Tests
    ----------
    test_environment_instructions_validate_truth
        Verifies that a valid mock instruction subclass passes the validation
        check.
    """

### Environment
<!---
MARK: Environment
--->
    class Environment
    """
    Abstract base class defining the controller-side environment process.

    This class defines the common interface used by the controller to manage
    an environment. Environment subclasses receive commands through an
    environment command queue, map those commands to callable methods, and
    exchange data with acquisition, output, GUI, controller, and logging
    processes through queues and events.

    Subclasses must implement the hardware initialization, environment
    initialization, and graceful shutdown behavior required by the specific
    environment control strategy. Additional environment-specific commands
    can be registered by adding entries to ``command_map`` with
    ``map_command``. This class should be added to the ``ENVIRONMENT_CLASS``
    dictionary within the environment registry.

    Parameters
    ----------
    environment_name : str
        Environment name used for logging, UI display, and
        task identification.
    queue_name : str
        Unique queue identifier assigned by the environment manager for
        routing environment-specific messages.
    command_queue : VerboseMessageQueue
        Queue used to receive commands sent to this environment.
    gui_update_queue : multiprocessing.Queue or queue.Queue
        Queue used to send GUI update commands.
    controller_command_queue : VerboseMessageQueue
        Queue used to send commands back to the controller.
    log_file_queue : multiprocessing.Queue
        Queue used to send messages to the logging process.
    data_in_queue : multiprocessing.Queue or queue.Queue
        Queue used to receive acquired data from the acquisition process.
    data_out_queue : multiprocessing.Queue or queue.Queue
        Queue used to send output data to the output process.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    active_event : multiprocessing.synchronize.Event
        Event indicating whether this environment is active.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether this environment is ready.

    Attributes
    ----------
    environment_name : str
        Environment name used for logging, UI display, and
        task identification.
    hardware_metadata : HardwareMetadata
        Hardware metadata used by the environment after initialization.
    environment_metadata : EnvironmentMetadata
        Environment metadata used by the environment after initialization.

    Unit Tests
    ----------
    test_environment
        Verifies that instruction subclasses initialize the required
        environment name attribute.
    """

    def __init__
    """
    Initialize the environment process object.

    Stores queues, events, names, and default state needed by the environment.
    Also initializes the command map with global controller commands for
    quitting, hardware initialization, environment initialization, and
    environment shutdown.

    Parameters
    ----------
    environment_name : str
        Environment name used for logging, UI display, and
        task identification.
    queue_name : str
        Unique queue identifier assigned by the environment manager.
    command_queue : VerboseMessageQueue
        Queue used to receive environment commands.
    gui_update_queue : multiprocessing.Queue
        Queue used to send GUI updates.
    controller_command_queue : VerboseMessageQueue
        Queue used to send commands to the controller.
    log_file_queue : multiprocessing.Queue
        Queue used to send log messages.
    data_in_queue : multiprocessing.Queue
        Queue used to receive acquired data.
    data_out_queue : multiprocessing.Queue
        Queue used to send output data.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    active_event : multiprocessing.synchronize.Event
        Event indicating whether this environment is active.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether this environment is ready.

    Unit Tests
    ----------
    test_environment_init
        Confirms that initialization stores all queues and events, initializes
        metadata attributes to ``None``, and maps the default global commands.
    """

    property
    def command_map
    """
    Return the command-to-function mapping for this environment.

    The command map is used by ``run`` to determine which environment method
    should be called when a command is received from the command queue.

    Returns
    -------
    dict
        Mapping from command enum members to bound environment methods.

    Unit Tests
    ----------
    test_environment_command_map
        Verifies that the default command map contains expected global
        commands and maps them to callable methods.
    """

    def map_command
    """
    Map a command to an environment method.

    Adds or replaces an entry in ``command_map``. The mapped function must
    accept one input argument containing the command data, even if that data
    is ignored.

    Parameters
    ----------
    key : Enum
        Command key that will be received from the environment command queue.
    function : callable
        Function to call when ``key`` is received.

    Unit Tests
    ----------
    test_environment_map_command
        Confirms that a new command can be added to the command map and maps
        to the provided callable.
    """

    def set_ready
    """
    Set the environment ready event.

    Marks the environment as ready for controller operations.

    Unit Tests
    ----------
    test_environment_set_ready
        Verifies that calling this method sets the ready event.
    """

    def clear_ready
    """
    Clear the environment ready event.

    Marks the environment as not ready for controller operations.

    Unit Tests
    ----------
    test_environment_clear_ready
        Verifies that calling this method clears the ready event.
    """

    property
    def ready
    """
    Return whether the environment is ready.

    Returns
    -------
    bool
        ``True`` if the ready event is set, otherwise ``False``.

    Unit Tests
    ----------
    test_environment_set_ready
        Verifies that calling this method sets the ready event.
    test_environment_clear_ready
        Verifies that calling this method clears the ready event.
    """

    def set_active
    """
    Set the environment active event.

    Marks the environment as active.

    Unit Tests
    ----------
    test_environment_set_active
        Verifies that calling this method sets the active event.
    """

    def clear_active
    """
    Clear the environment active event.

    Marks the environment as inactive.

    Unit Tests
    ----------
    test_environment_clear_active
        Verifies that calling this method clears the active event.
    """

    property
    def active
    """
    Return whether the environment is active.

    Returns
    -------
    bool
        ``True`` if the active event is set, otherwise ``False``.

    Unit Tests
    ----------
    test_environment_set_active
        Verifies that calling this method sets the active event.
    test_environment_clear_active
        Verifies that calling this method clears the active event.
    """

    property
    def acquisition_active
    """
    Return whether acquisition is active.

    Returns
    -------
    bool
        ``True`` if the acquisition active event is set, otherwise ``False``.

    Unit Tests
    ----------
    test_environment_acquisition_active
        Verifies that this property reflects the state of the acquisition
        active event.
    """

    property
    def output_active
    """
    Return whether output is active.

    Returns
    -------
    bool
        ``True`` if the output active event is set, otherwise ``False``.

    Unit Tests
    ----------
    test_environment_output_active
        Verifies that this property reflects the state of the output active
        event.
    """

    def initialize_hardware
    """
    Initialize hardware metadata for the environment.

    Stores the hardware metadata received from the controller. Subclasses
    should extend this method to perform hardware-dependent setup required by
    the environment. Subclasses should call ``set_ready`` when initialization
    is complete if the environment is ready for operation.

    Parameters
    ----------
    hardware_metadata : HardwareMetadata
        Hardware metadata containing hardware configuration information
        needed by the environment.

    Unit Tests
    ----------
    test_environment_initialize_hardware
        Verifies that a mock environment subclass stores the supplied hardware
        metadata and sets itself as ready at the end of the function.
    """

    def initialize_environment
    """
    Initialize environment-specific metadata.

    Stores environment metadata received from the controller and updates the
    environment name from the metadata object. Subclasses should extend this
    method to perform environment-specific setup. Subclasses should call
    ``set_ready`` when initialization is complete if the environment is ready
    for operation.

    Parameters
    ----------
    environment_metadata : EnvironmentMetadata
        Metadata object containing the parameters defining this environment.

    Unit Tests
    ----------
    test_environment_initialize_environment
        Verifies that a mock environment subclass stores the supplied
        environment metadata and updates the environment name. Checks
        that subclasses set their ready event at the end of the function.
    """

    property
    def queue_name
    """
    Return the unique queue name assigned to the environment.

    Returns
    -------
    str
        Queue name used to route environment-specific messages.

    Unit Tests
    ----------
    test_environment_queue_name
        Verifies that this property returns the queue name supplied during
        initialization.
    """

    property
    def environment_command_queue
    """
    Return the environment command queue.

    Returns
    -------
    VerboseMessageQueue
        Queue used to receive commands sent to this environment.

    Unit Tests
    ----------
    test_environment_environment_command_queue
        Verifies that this property returns the command queue supplied during
        initialization.
    """

    property
    def data_in_queue
    """
    Return the data input queue.

    Returns
    -------
    queue.Queue or multiprocessing.Queue
        Queue used to receive acquired data from the acquisition process.

    Unit Tests
    ----------
    test_environment_data_in_queue
        Verifies that this property returns the data input queue supplied
        during initialization.
    """

    property
    def data_out_queue
    """
    The queue to which data is written that will be output to exciters.

    Returns
    -------
    queue.Queue or multiprocessing.Queue
        Queue used to send output data to the output process.

    Unit Tests
    ----------
    test_environment_data_out_queue
        Verifies that this property returns the data output queue supplied
        during initialization.
    """

    property
    def gui_update_queue
    """
    The queue that GUI update instructions are written to.

    Returns
    -------
    queue.Queue or multiprocessing.Queue
        Queue used to send update commands to the GUI.

    Unit Tests
    ----------
    test_environment_gui_update_queue
        Verifies that this property returns the GUI update queue supplied
        during initialization.
    """

    property
    def controller_command_queue
    """
    The queue that global controller updates are written to.

    Returns
    -------
    VerboseMessageQueue
        Queue used to send commands from the environment back to the
        controller.

    Unit Tests
    ----------
    test_environment_controller_command_queue
        Verifies that this property returns the controller command queue
        supplied during initialization.
    """

    property
    def log_file_queue
    """
    Return the log file queue.

    Returns
    -------
    multiprocessing.Queue
        Queue used to send log messages to the logging process.

    Unit Tests
    ----------
    test_environment_log_file_queue
        Verifies that this property returns the log file queue supplied during
        initialization.
    """

    def log
    """
    Queue a message for the log file.

    Formats the supplied message with the current timestamp and environment
    name, then places it on the log file queue.

    Parameters
    ----------
    message : str
        Message to write to the log file.

    Unit Tests
    ----------
    test_environment_log
        Verifies that calling this method places a formatted log message on
        the log file queue.
    """

    def run
    """
    Run the environment command loop.

    A function that is called by the environment's process function that
    sits in a while loop waiting for instructions on the command queue.

    When the instructions are recieved, they are separated into
    ``(message,data)`` pairs.  The ``message`` is used in conjuction with
    the ``command_map`` to identify which function should be called, and
    the ``data`` is passed to that function as the argument.  If the
    function returns a truthy value, it signals to the ``run`` function
    that it is time to stop the loop and exit.

    Parameters
    ----------
    shutdown_event : multiprocessing.synchronize.Event
        Event used to signal that the environment command loop should stop.

    Unit Tests
    ----------
    test_environment_run_quit
        Verifies that the command loop exits when a mapped command returns a
        truthy halt flag.

    test_environment_run_undefined_command
        Verifies that an undefined command is logged and does not halt the
        environment.

    test_environment_run_command_exception
        Verifies that an exception raised by a mapped command is logged and
        sent to the GUI update queue.
    """

    def stop_environment
    """
    Stop the environment gracefully.

    This function defines the operations to shut down the environment
    gracefully. This should include any operations needed to avoid 
    abrupt output changes, protect test equipment, stop background 
    activity, and clear environment state as appropriate.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This may be ignored
        by implementations that do not require additional shutdown data.

    Unit Tests
    ----------
    test_environment_stop_environment
        Verifies that a mock environment subclass performs graceful shutdown
        behavior and clears or updates expected state.
    """

    def quit
    """
    Signal the environment command loop to stop.

    Returns ``True`` so the ``run`` loop exits after processing the quit
    command.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is
        ignored.

    Returns
    -------
    bool
        Always returns ``True`` to indicate that the environment process
        should stop.

    Unit Tests
    ----------
    test_environment_quit
        Verifies that this method returns ``True``.
    """

### Process
<!---
MARK: Process
--->
    def process
    """
    Function executed by ``multiprocessing.Process`` to start an environment.

    This function serves as the entry point for an environment process. It
    constructs an ``Environment`` instance with the supplied communication
    queues and synchronization events, then runs the environment until
    ``shutdown_event`` is set. It is intended to be used as the target of a
    ``multiprocessing.Process`` and should not be called directly.

    Parameters
    ----------
    environment_name : str
        Name of the environment.

    queue_name : str
        Name used to identify the environment's communication queues.

    input_queue : VerboseMessageQueue
        Queue used to receive commands sent to the environment.

    gui_update_queue : queue.Queue or multiproccessing.Queue
        Queue used to send ``(message, data)`` pairs to the GUI.

    controller_command_queue : VerboseMessageQueue
        Queue used to send commands to the controller.

    log_file_queue : multiproccessing.Queue
        Queue used to send log messages to the logging process.

    data_in_queue : queue.Queue or multiproccessing.Queue
        Queue used to receive acquired data from the acquisition process.

    data_out_queue : queue.Queue or multiproccessing.Queue
        Queue used to send output data to the output process.

    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether the acquisition process is active.

    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether the output process is active.

    active_event : multiprocessing.synchronize.Event
        Event indicating whether the environment is running.

    ready_event : multiprocessing.synchronize.Event
        Event set when the environment has completed initialization and is
        ready to receive commands.

    shutdown_event : multiprocessing.synchronize.Event
        Event used to signal the environment to terminate.

    sysid_active_event : multiprocessing.synchronize.Event
        Event indicating whether system identification is active.

    sysid_stored_event : multiprocessing.synchronize.Event
        Event indicating that system identification data has been stored.

    ping_alive_event : multiprocessing.synchronize.Event
        Event used to restart the blocking timeout if environment is going to
        stall the main process for an extended time.

    threaded : bool
        Indicates whether the environment is running in threaded mode rather
        than multiprocessing mode.

    Unit Tests
    ----------
    test_process
        Verifies that valid environment process functions receive correct parameters and
        shutdown properly when shutdown event is set.
    """

## Abstract Sys Id Environment
<!---
MARK: Abstract Sysid Environment
--->
    """
    Abstract environment that can be used to create new environment control
    strategies that use system identification.

    This module extends the abstract environment interface with common
    metadata, commands, runtime state, startup, shutdown, signal generation,
    data collection, spectral processing, and data analysis behavior required
    for system identification workflows.
    """

### System Id Commands
<!---
MARK: System Id Commands
--->
    class SystemIdCommands
    """
    Enumeration of commands that can be sent to a system identification
    environment.

    These commands are used internally by system identification environments
    to coordinate environment-level behavior that is not covered by the
    global controller command set.

    Attributes
    ----------
    CHECK_FOR_COMPLETE_SHUTDOWN : int
        Command used to check whether all system identification subprocesses
        have completed shutdown.

    Unit Tests
    ----------
    test_sysid_commands
        Iterates through each enum member to confirm unique integer values.
    """

    class SysIdUICommands
    """
    Enumeration of commands sent from a system identification environment to
    the system identification user interface.

    These commands notify the UI when system identification activity starts
    or ends so that widgets, tabs, indicators, and controls can be updated.

    Attributes
    ----------
    SYSID_STARTED : int
        Indicates that a system identification measurement has started.
    SYSID_ENDED : int
        Indicates that a system identification measurement has ended.

    Unit Tests
    ----------
    test_sysid_ui_commands
        Iterates through each enum member to confirm unique integer values.
    """

### System Id Environment Metadata
<!---
MARK: System Id Environment Metadata
--->
    class SysIdEnvironmentMetadata
    """
    Abstract base class for metadata used by environments that support system
    identification.

    This class extends ``EnvironmentMetadata`` with a ``SysIdMetadata`` object
    defining excitation signal settings, frame sizes, averaging parameters,
    spectral processing settings, estimator selection, and other system
    identification parameters.

    Subclasses must define the physical channel mappings, transformed channel
    counts, and transformation matrices required by the specific environment.

    Parameters
    ----------
    environment_type : EnvironmentType
        Type of environment represented by this metadata object.
    environment_name : str
        Environment name used for logging, UI display, and task
        identification.
    channel_list_bools : list of bool
        Boolean mask identifying which hardware channels are used by this
        environment.
    sample_rate : int
        Sample rate associated with the environment.
    sysid_metadata : SysIdMetadata, optional
        System identification metadata. If omitted or invalid, default system
        identification metadata is created using ``sample_rate``.

    Attributes
    ----------
    sysid_metadata : SysIdMetadata
        Metadata defining excitation, acquisition, spectral processing, and
        averaging settings used for system identification.

    Unit Tests
    ----------
    test_sysid_environment_metadata
        Verifies that subclasses initialize the base metadata attributes and
        create or preserve a valid ``SysIdMetadata`` instance.
    """

    def __init__
    """
    Initialize system identification environment metadata.

    Initializes the base environment metadata fields and stores the supplied
    system identification metadata. If ``sysid_metadata`` is not a
    ``SysIdMetadata`` instance, default metadata is created using the supplied
    sample rate.

    Parameters
    ----------
    environment_type : EnvironmentType
        Type of environment represented by this metadata object.
    environment_name : str
        Environment name used for logging, UI display, and task
        identification.
    channel_list_bools : list of bool
        Boolean mask identifying which hardware channels belong to the
        environment.
    sample_rate : int
        Sample rate associated with the environment.
    sysid_metadata : SysIdMetadata, optional
        System identification metadata to store.

    Unit Tests
    ----------
    test_sysid_environment_metadata_init
        Verifies that supplied system identification metadata is stored and
        that default metadata is created when none is supplied.
    """

    property
    def number_of_channels
    """
    Return the number of physical channels in the environment.

    This property must be implemented by subclasses because the number of
    channels depends on the environment-specific channel definition.

    Returns
    -------
    int
        Number of physical channels assigned to the system identification
        environment.

    Unit Tests
    ----------
    test_sysid_environment_metadata_number_of_channels
        Verifies that subclasses return the correct number of channels.
    """

    property
    def response_channel_indices
    """
    Return indices of response channels in the environment channel list.

    Response channels are channels measured as outputs of the test article or
    system under test. These are commonly used as the numerator channels in
    transfer-function estimation.

    Returns
    -------
    list of int
        Indices corresponding to response channels.

    Unit Tests
    ----------
    test_sysid_environment_metadata_response_channel_indices
        Verifies that subclasses return valid response channel indices.
    """

    property
    def reference_channel_indices
    """
    Return indices of reference channels in the environment channel list.

    Reference channels are excitation or drive-related channels used as input
    references for system identification.

    Returns
    -------
    list of int
        Indices corresponding to reference channels.

    Unit Tests
    ----------
    test_sysid_environment_metadata_reference_channel_indices
        Verifies that subclasses return valid reference channel indices.
    """

    property
    def num_response_channels
    """
    Return the number of response channels after transformation.

    If no response transformation matrix is defined, this is the number of
    physical response channels. If a response transformation matrix is defined,
    this is the number of transformed response channels, equal to the number
    of rows in the response transformation matrix.

    Returns
    -------
    int
        Number of response channels used by system identification processing.

    Unit Tests
    ----------
    test_sysid_environment_metadata_num_response_channels
        Verifies that the returned count reflects either the physical response
        channels or the transformed response matrix row count.
    """

    property
    def num_reference_channels
    """
    Return the number of reference channels after transformation.

    If no reference transformation matrix is defined, this is the number of
    physical reference channels. If a reference transformation matrix is
    defined, this is the number of transformed reference channels, equal to
    the number of rows in the reference transformation matrix.

    Returns
    -------
    int
        Number of reference channels used by system identification processing.

    Unit Tests
    ----------
    test_sysid_environment_metadata_num_reference_channels
        Verifies that the returned count reflects either the physical
        reference channels or the transformed reference matrix row count.
    """

    property
    def response_transformation_matrix
    """
    Return the response transformation matrix.

    The response transformation matrix maps physical response channels to
    transformed response coordinates. Subclasses must implement this property.

    Returns
    -------
    numpy.ndarray or None
        Response transformation matrix, or ``None`` if no transformation is
        applied.

    Unit Tests
    ----------
    test_sysid_environment_metadata_response_transformation_matrix
        Verifies that subclasses return either ``None`` or a valid response
        transformation matrix.
    """

    property
    def reference_transformation_matrix
    """
    Return the reference transformation matrix.

    The reference transformation matrix maps physical reference channels to
    transformed reference coordinates. Subclasses must implement this property.

    Returns
    -------
    numpy.ndarray or None
        Reference transformation matrix, or ``None`` if no transformation is
        applied.

    Unit Tests
    ----------
    test_sysid_environment_metadata_reference_transformation_matrix
        Verifies that subclasses return either ``None`` or a valid reference
        transformation matrix.
    """

    def validate
    """
    Validate system identification environment metadata.

    Performs the common environment metadata validation checks from
    ``EnvironmentMetadata``. Subclasses should extend this method with
    system-identification-specific checks, such as validating response and
    reference channel assignments, transformation matrix dimensions, and
    system identification parameter compatibility.

    Parameters
    ----------
    hardware_metadata : HardwareMetadata
        Hardware metadata containing available channel definitions and hardware
        configuration parameters.

    Raises
    ------
    RattlesnakeError
        If any base environment metadata validation check fails.
    RattlesnakeError
        If subclass-specific system identification metadata validation fails.

    Unit Tests
    ----------
    test_sysid_environment_metadata_validate_truth
        Verifies that valid system identification metadata passes validation.
    """

    def __eq__
    """
    Compare two system identification metadata objects for equality.

    Compares all fields in this object's ``__dict__`` against the
    corresponding fields in another metadata object. Array-like values are
    compared using NumPy equality operations.

    Parameters
    ----------
    other : object
        Object to compare against this metadata object.

    Returns
    -------
    bool
        ``True`` if all metadata fields compare equal, otherwise ``False``.

    Unit Tests
    ----------
    test_sysid_environment_metadata_eq_truth/false
        Verifies that equivalent metadata objects compare equal and that
        mismatched or incompatible objects compare unequal.
    """

    def save_metadata_to_netcdf
    """
    Save system identification environment metadata to a netCDF group.

    Stores system identification metadata into the supplied netCDF group.
    Subclasses should extend this method to store all environment-specific
    metadata needed to reconstruct the system identification environment.

    Parameters
    ----------
    netcdf_group_handle : nc4._netCDF4.Group
        netCDF group where this environment's metadata should be stored.

    Unit Tests
    ----------
    test_sysid_environment_metadata_load_save_netcdf
        Saves metadata to a netCDF file, loads it back, and verifies that the
        loaded metadata is valid.
    """

    classmethod
    def load_metadata_from_netcdf
    """
    Load system identification metadata from a netCDF group.

    Retrieves system identification metadata previously written to a netCDF
    group. Subclasses should extend this method to read environment-specific
    metadata and construct the full metadata object.

    Parameters
    ----------
    netcdf_group_handle : nc4._netCDF4.Group
        netCDF group containing stored system identification metadata.
    environment_name : str
        Name of the environment whose metadata should be loaded.
    channel_list_bools : list of bool
        Boolean channel mask identifying channels assigned to the environment.
    hardware_metadata : HardwareMetadata
        Hardware metadata associated with the stored environment.

    Returns
    -------
    SysIdMetadata
        Loaded system identification metadata.

    Unit Tests
    ----------
    test_sysid_environment_metadata_load_save_netcdf
        Saves metadata to a netCDF file, loads it back, and verifies that the
        loaded metadata is valid.
    """

    classmethod
    def create_blank_worksheet_template
    """
    Create a blank Excel worksheet template for system identification
    environment metadata.

    Writes the common worksheet fields required by environment metadata.
    Subclasses should extend this template with system-identification-specific
    fields, including channel definitions, transformation matrices, and
    excitation settings.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet where the blank metadata template should be created.

    Unit Tests
    ----------
    test_sysid_environment_metadata_load_save_worksheet
        Verifies that a worksheet template can be created and used to save and
        reload valid metadata.
    """

    def save_metadata_to_worksheet
    """
    Save system identification environment metadata to an Excel worksheet.

    Writes common environment metadata fields to the supplied worksheet.
    Subclasses should extend this method to write system-identification-specific
    metadata fields and transformation matrices.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet where this environment's metadata should be stored.

    Unit Tests
    ----------
    test_sysid_environment_metadata_load_save_worksheet
        Saves metadata to a worksheet, loads it back, and verifies that the
        loaded metadata is valid.
    """

    classmethod
    def load_metadata_from_worksheet
    """
    Load system identification environment metadata from an Excel worksheet.

    Retrieves metadata previously written by ``save_metadata_to_worksheet``.
    Subclasses should read all required environment-specific system
    identification parameters from the worksheet and construct a metadata
    object.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet containing stored environment metadata.
    environment_name : str
        Name of the environment whose metadata should be loaded.
    channel_list_bools : list of bool
        Boolean channel mask identifying channels assigned to the environment.
    hardware_metadata : HardwareMetadata
        Hardware metadata associated with the stored environment.

    Returns
    -------
    SysIdEnvironmentMetadata
        Instance of the metadata subclass populated from the worksheet.

    Unit Tests
    ----------
    test_sysid_environment_metadata_load_save_worksheet
        Saves metadata to a worksheet, loads it back, and verifies that the
        loaded metadata is valid.
    """

    classmethod
    def save_sysid_matrix_to_worksheet
    """
    Save response and output transformation matrices to an Excel worksheet.

    Writes the response transformation matrix and output transformation matrix
    starting at the requested worksheet row. If a matrix is ``None``, the
    worksheet is populated with ``"None"`` for that matrix. If the response
    matrix is present, the output matrix section is shifted down so the two
    matrices do not overlap.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet where transformation matrices should be written.
    response_matrix : numpy.ndarray or None
        Response transformation matrix to write.
    output_matrix : numpy.ndarray or None
        Output or reference transformation matrix to write.
    start_row : int
        Starting row for the response transformation matrix section.

    Unit Tests
    ----------
    test_sysid_environment_metadata_save_sysid_matrix_to_worksheet
        Verifies that response and output transformation matrices, including
        ``None`` values, are written to the worksheet correctly.
    """

    classmethod
    def load_sysid_matrix_from_worksheet
    """
    Load response and output transformation matrices from an Excel worksheet.

    Reads transformation matrix definitions starting at the supplied row. A
    matrix is returned as ``None`` if the worksheet cell contains ``"None"`` or
    a template comment. Otherwise, numeric worksheet values are read into
    transformation matrices.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet containing transformation matrix definitions.
    start_row : int
        Starting row for the response transformation matrix section.

    Returns
    -------
    tuple
        Tuple ``(response_transformation_matrix, output_transformation_matrix)``.
        Each entry is either a NumPy array or ``None``.

    Unit Tests
    ----------
    test_sysid_environment_metadata_load_sysid_matrix_from_worksheet
        Verifies that response and output transformation matrices are loaded
        correctly from worksheet data.
    """

### System Id Environment
<!---
MARK: System Id Environment
--->
    class SysIdEnvironment
    """
    Abstract base class defining controller-side behavior for environments
    that support system identification.

    This class extends ``Environment`` with common system identification
    orchestration. It coordinates data collection, signal generation, spectral
    processing, data analysis, GUI updates, and shutdown state for system
    identification measurements.

    Subclasses must implement hardware initialization, environment
    initialization, system identification initialization, and graceful
    environment shutdown behavior specific to the control strategy.

    Parameters
    ----------
    environment_name : str
        Environment name used for logging, UI display, and task
        identification.
    queue_name : str
        Unique queue identifier assigned by the environment manager.
    command_queue : VerboseMessageQueue
        Queue used to receive commands sent to this environment.
    gui_update_queue : multiprocessing.Queue or queue.Queue
        Queue used to send GUI update commands.
    controller_command_queue : VerboseMessageQueue
        Queue used to send commands back to the controller.
    log_file_queue : multiprocessing.Queue
        Queue used to send messages to the logging process.
    collector_command_queue : VerboseMessageQueue
        Queue used to send commands to the data collector process.
    signal_generator_command_queue : VerboseMessageQueue
        Queue used to send commands to the signal generation process.
    spectral_processing_command_queue : VerboseMessageQueue
        Queue used to send commands to the spectral processing process.
    data_analysis_command_queue : VerboseMessageQueue
        Queue used to send commands to the system identification data analysis
        process.
    data_in_queue : multiprocessing.Queue or queue.Queue
        Queue used to receive acquired data.
    data_out_queue : multiprocessing.Queue or queue.Queue
        Queue used to send output data.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    active_event : multiprocessing.synchronize.Event
        Event indicating whether this environment is active.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether this environment is ready.
    sysid_active_event : multiprocessing.synchronize.Event
        Event indicating whether system identification is active.
    sysid_stored_event : multiprocessing.synchronize.Event
        Event indicating whether system identification data has been stored.

    Attributes
    ----------
    sysid_data : SysIdDataPackage
        Container for the most recently completed or loaded system
        identification results.
    collector_shutdown_achieved : bool
        Whether the collector process has reported shutdown completion.
    spectral_shutdown_achieved : bool
        Whether the spectral processing process has reported shutdown
        completion.
    siggen_shutdown_achieved : bool
        Whether the signal generation process has reported shutdown
        completion.
    analysis_shutdown_achieved : bool
        Whether the data analysis process has reported shutdown completion.

    Unit Tests
    ----------
    test_sysid_environment
        Verifies that system identification environment subclasses initialize
        required environment and system identification attributes.
    """

    def __init__
    """
    Initialize the system identification environment process object.

    Stores queues, events, metadata references, system identification state,
    and shutdown flags. Also maps global system identification commands and
    subsystem shutdown/completion commands to their associated environment
    methods.

    Parameters
    ----------
    environment_name : str
        Environment name used for logging, UI display, and task
        identification.
    queue_name : str
        Unique queue identifier assigned by the environment manager.
    command_queue : VerboseMessageQueue
        Queue used to receive environment commands.
    gui_update_queue : multiprocessing.Queue or queue.Queue
        Queue used to send GUI updates.
    controller_command_queue : VerboseMessageQueue
        Queue used to send commands to the controller.
    log_file_queue : multiprocessing.Queue
        Queue used to send log messages.
    collector_command_queue : VerboseMessageQueue
        Queue used to send commands to the data collector process.
    signal_generator_command_queue : VerboseMessageQueue
        Queue used to send commands to the signal generation process.
    spectral_processing_command_queue : VerboseMessageQueue
        Queue used to send commands to the spectral processing process.
    data_analysis_command_queue : VerboseMessageQueue
        Queue used to send commands to the system identification data analysis
        process.
    data_in_queue : multiprocessing.Queue or queue.Queue
        Queue used to receive acquired data.
    data_out_queue : multiprocessing.Queue or queue.Queue
        Queue used to send output data.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    active_event : multiprocessing.synchronize.Event
        Event indicating whether this environment is active.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether this environment is ready.
    sysid_active_event : multiprocessing.synchronize.Event
        Event indicating whether system identification is active.
    sysid_stored_event : multiprocessing.synchronize.Event
        Event indicating whether system identification data has been stored.

    Unit Tests
    ----------
    test_sysid_environment_init
        Confirms that initialization stores queues and events, initializes
        system identification state, and maps system identification commands.
    """

    property
    def sysid_active
    """
    Return whether system identification is active.

    Returns
    -------
    bool
        ``True`` if the system identification active event is set, otherwise
        ``False``.

    Unit Tests
    ----------
    test_sysid_environment_sysid_active
        Verifies that this property reflects the state of the system
        identification active event.
    """

    def set_sysid_active
    """
    Set the system identification active event.

    Marks system identification as active.

    Unit Tests
    ----------
    test_sysid_environment_set_sysid_active
        Verifies that calling this method sets the system identification active
        event.
    """

    def clear_sysid_active
    """
    Clear the system identification active event.

    Marks system identification as inactive.

    Unit Tests
    ----------
    test_sysid_environment_clear_sysid_active
        Verifies that calling this method clears the system identification
        active event.
    """

    property
    def sysid_stored
    """
    Return whether system identification data has been stored.

    Returns
    -------
    bool
        ``True`` if the system identification stored event is set, otherwise
        ``False``.

    Unit Tests
    ----------
    test_sysid_environment_sysid_stored
        Verifies that this property reflects the state of the system
        identification stored event.
    """

    def set_sysid_stored
    """
    Set the system identification stored event.

    Marks system identification data as stored.

    Unit Tests
    ----------
    test_sysid_environment_set_sysid_stored
        Verifies that calling this method sets the system identification stored
        event.
    """

    def clear_sysid_stored
    """
    Clear the system identification stored event.

    Marks system identification data as not stored.

    Unit Tests
    ----------
    test_sysid_environment_clear_sysid_stored
        Verifies that calling this method clears the system identification
        stored event.
    """

    def initialize_hardware
    """
    Initialize hardware metadata for the system identification environment.

    Stores hardware metadata received from the controller. Subclasses should
    extend this method to perform hardware-dependent setup required by the
    specific environment.

    Parameters
    ----------
    hardware_metadata : HardwareMetadata
        Hardware metadata containing hardware configuration information needed
        by the environment.

    Unit Tests
    ----------
    test_sysid_environment_initialize_hardware
        Verifies that hardware metadata is stored and that the environment is
        marked ready after initialization.
    """

    def initialize_environment
    """
    Initialize system identification environment metadata.

    Sends the environment name to the system identification data analysis
    process, then initializes the base environment metadata state. Subclasses
    should extend this method to perform environment-specific setup.

    Parameters
    ----------
    environment_metadata : SysIdEnvironmentMetadata
        Metadata object containing the parameters defining this system
        identification environment.

    Unit Tests
    ----------
    test_sysid_environment_initialize_environment
        Verifies that environment metadata is stored and that the data analysis
        process receives environment initialization information.
    """

    def initialize_sysid
    """
    Initialize system identification parameters.

    Stores the supplied ``SysIdMetadata`` object on the environment metadata
    and sends the parameters to the system identification data analysis
    process. Subclasses should extend this method to perform additional
    system-identification-specific initialization.

    Parameters
    ----------
    sysid_metadata : SysIdMetadata
        System identification metadata defining signal generation, acquisition,
        averaging, and spectral processing parameters.

    Unit Tests
    ----------
    test_sysid_environment_initialize_sysid
        Verifies that system identification metadata is stored and forwarded to
        the data analysis process.
    """

    def get_sysid_data_collector_metadata
    """
    Build data collector metadata for a system identification measurement.

    Creates a ``CollectorMetadata`` object using the environment metadata and
    system identification settings. The collector metadata defines channel
    counts, response and reference channel indices, acquisition type, trigger
    settings, frame size, overlap, windowing, transformation matrices, and
    kurtosis buffer settings.

    Returns
    -------
    CollectorMetadata
        Metadata used to initialize the data collector process for system
        identification.

    Unit Tests
    ----------
    test_sysid_environment_get_sysid_data_collector_metadata
        Verifies that collector metadata is populated correctly from system
        identification metadata.
    """

    def get_sysid_spectral_processing_metadata
    """
    Build spectral processing metadata for a system identification measurement.

    Creates a ``SpectralProcessingMetadata`` object using the environment
    metadata and system identification settings. This includes averaging type,
    number of averages, exponential averaging coefficient, FRF estimator,
    channel counts, frequency spacing, sample rate, and number of frequency
    lines.

    Parameters
    ----------
    is_noise : bool, optional
        If ``True``, use noise measurement averaging settings. If ``False``,
        use transfer-function measurement averaging settings.

    Returns
    -------
    SpectralProcessingMetadata
        Metadata used to initialize the spectral processing process.

    Raises
    ------
    ValueError
        If the configured FRF estimator is not recognized.

    Unit Tests
    ----------
    test_sysid_environment_get_sysid_spectral_processing_metadata
        Verifies that spectral processing metadata is populated correctly for
        noise and transfer-function measurements.
    """

    def get_sysid_signal_generation_metadata
    """
    Build signal generation metadata for system identification.

    Creates a ``SignalGenerationMetadata`` object using hardware write size,
    level ramp duration, sample rate, hardware output oversampling, and the
    reference transformation matrix.

    Returns
    -------
    SignalGenerationMetadata
        Metadata used to initialize the signal generation process.

    Unit Tests
    ----------
    test_sysid_environment_get_sysid_signal_generation_metadata
        Verifies that signal generation metadata is populated correctly.
    """

    def get_sysid_signal_generator
    """
    Create a signal generator for the configured system identification signal.

    Constructs and returns the appropriate signal generator based on
    ``sysid_signal_type``. Supported signal types include random,
    pseudorandom, burst random, and chirp.

    Returns
    -------
    SignalGenerator
        Signal generator configured for the current system identification
        settings.

    Unit Tests
    ----------
    test_sysid_environment_get_sysid_signal_generator
        Verifies that the correct signal generator type is created for each
        supported system identification signal type.
    """

    def load_noise
    """
    Send noise data to the system identification data analysis process.

    Parameters
    ----------
    data : Any
        Noise data package or payload to forward to data analysis.

    Unit Tests
    ----------
    test_sysid_environment_load_noise
        Verifies that noise data is forwarded to the data analysis command
        queue.
    """

    def load_transfer_function
    """
    Send transfer-function data to the system identification data analysis
    process.

    Parameters
    ----------
    data : Any
        Transfer-function data package or payload to forward to data analysis.

    Unit Tests
    ----------
    test_sysid_environment_load_transfer_function
        Verifies that transfer-function data is forwarded to the data analysis
        command queue.
    """

    def save_system_id_to_file
    """
    Save system identification data to a file.

    Saves the current ``SysIdDataPackage`` to the requested path. Supported
    file formats are netCDF ``.nc4``, MATLAB ``.mat``, and NumPy ``.npz``.
    For netCDF files, environment metadata is saved with the system
    identification data.

    Parameters
    ----------
    data : str or pathlib.Path
        Output file path. The file extension determines the save format.

    Unit Tests
    ----------
    test_sysid_environment_save_system_id_to_file
        Verifies that system identification data is saved for supported file
        extensions and that the environment is marked ready afterward.
    """

    def load_system_id_from_package
    """
    Load system identification data from an existing package.

    Sends the supplied ``SysIdDataPackage`` to the data analysis process,
    updates the GUI with the loaded data, and queues a system identification
    completion command so environment-specific completion behavior is reused.

    Parameters
    ----------
    data : SysIdDataPackage
        System identification data package to load.

    Unit Tests
    ----------
    test_sysid_environment_load_system_id_from_package
        Verifies that loaded system identification data is forwarded to data
        analysis and GUI update queues.
    """

    def start_noise
    """
    Start a system identification noise measurement.

    Initializes collector, signal generation, data analysis, and spectral
    processing processes for a noise measurement. The signal generator is
    muted, acquisition and signal generation are started, spectral processing
    is cleared and started, and the system identification active event is set.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is not
        used by the base implementation.

    Unit Tests
    ----------
    test_sysid_environment_start_noise
        Verifies that all required subprocess initialization and start commands
        are sent for a noise measurement.
    """

    def start_transfer_function
    """
    Start a system identification transfer-function measurement.

    Initializes collector, signal generation, data analysis, and spectral
    processing processes for a transfer-function measurement. The signal
    generator is initialized, muted, adjusted to full test level, acquisition
    and signal generation are started, spectral processing is cleared and
    started, and the system identification active event is set.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is not
        used by the base implementation.

    Unit Tests
    ----------
    test_sysid_environment_start_transfer_function
        Verifies that all required subprocess initialization and start commands
        are sent for a transfer-function measurement.
    """

    def stop_system_id
    """
    Start the shutdown process for system identification.

    Commands the collector, signal generator, spectral processing process, and
    optionally the data analysis process to stop. Then queues a shutdown check
    command so the environment can determine when all subprocesses have
    completed shutdown.

    Parameters
    ----------
    stop_tasks : bool
        If ``True``, command the data analysis process to stop. If ``False``,
        assume data analysis has initiated shutdown itself.

    Unit Tests
    ----------
    test_sysid_environment_stop_system_id
        Verifies that shutdown commands are sent to system identification
        subprocesses and that a shutdown check is queued.
    """

    def siggen_shutdown_achieved_fn
    """
    Mark signal generation shutdown as complete.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_sysid_environment_siggen_shutdown_achieved_fn
        Verifies that the signal generation shutdown flag is set.
    """

    def collector_shutdown_achieved_fn
    """
    Mark data collector shutdown as complete.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_sysid_environment_collector_shutdown_achieved_fn
        Verifies that the collector shutdown flag is set.
    """

    def spectral_shutdown_achieved_fn
    """
    Mark spectral processing shutdown as complete.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_sysid_environment_spectral_shutdown_achieved_fn
        Verifies that the spectral processing shutdown flag is set.
    """

    def analysis_shutdown_achieved_fn
    """
    Mark data analysis shutdown as complete.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_sysid_environment_analysis_shutdown_achieved_fn
        Verifies that the data analysis shutdown flag is set.
    """

    def check_for_sysid_shutdown
    """
    Check whether all system identification subprocesses have shut down.

    If signal generation, data collection, spectral processing, and data
    analysis have all reported shutdown completion, this method clears the
    system identification active event and notifies the GUI that system
    identification has ended. Otherwise, it logs which subprocesses are still
    active and requeues another shutdown check.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_sysid_environment_check_for_sysid_shutdown_complete
        Verifies that system identification is marked inactive when all
        shutdown flags are complete.

    test_sysid_environment_check_for_sysid_shutdown_incomplete
        Verifies that incomplete shutdown state is logged and another shutdown
        check is queued.
    """

    def system_id_noise_complete
    """
    Handle completion of the system identification noise measurement.

    Logs completion of the noise measurement and notifies the GUI that noise
    data collection has completed. The GUI may use this message to start the
    transfer-function portion of system identification.

    Parameters
    ----------
    data : Any
        Noise completion data from the data analysis process.

    Unit Tests
    ----------
    test_sysid_environment_system_id_noise_complete
        Verifies that noise completion is logged and forwarded to the GUI.
    """

    def system_id_complete
    """
    Handle completion of the system identification transfer-function
    measurement.

    Stores the completed system identification data package, notifies the GUI
    that transfer-function measurement is complete, notifies the controller UI
    that system identification has completed, and marks system identification
    data as stored.

    Parameters
    ----------
    data : tuple
        Tuple containing the system identification metadata and completed
        ``SysIdDataPackage``.

    Unit Tests
    ----------
    test_sysid_environment_system_id_complete
        Verifies that completed system identification data is stored and GUI
        completion messages are sent.
    """

    def stop_environment
    """
    Stop the environment gracefully.

    Subclasses must implement environment-specific shutdown behavior. This
    should include stopping active system identification or control activity,
    protecting hardware from abrupt output changes, and clearing or updating
    environment state as appropriate.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This may be ignored by
        implementations that do not require additional shutdown data.

    Unit Tests
    ----------
    test_sysid_environment_stop_environment
        Verifies that a system identification environment subclass performs
        graceful shutdown behavior.
    """

    def quit
    """
    Quit the system identification environment and associated subprocesses.

    Sends ``GlobalCommands.QUIT`` to the collector, signal generation,
    spectral processing, and data analysis command queues. Returns ``True`` so
    the environment command loop exits.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Returns
    -------
    bool
        Always returns ``True`` to indicate that the environment process
        should stop.

    Unit Tests
    ----------
    test_sysid_environment_quit
        Verifies that quit commands are sent to all system identification
        subprocesses and that this method returns ``True``.
    """

## Time Environment
<!---
MARK: Time Environment
--->
    """
    Time signal generation environment.

    This module defines a simple environment that plays a predefined time
    history signal directly to output hardware. It includes command enums,
    metadata, instructions, queue containers, environment runtime behavior, and
    a process entry point. The time environment is useful as a reference
    implementation for creating new environment control strategies.
    """

### Time Commands
<!---
MARK: Time Commands
--->
    class TimeCommands
    """
    Commands supported by the time environment.

    These commands can be sent to the time environment by the controller,
    profile manager, or user interface to change test level and repeat
    behavior.

    Attributes
    ----------
    SET_TEST_LEVEL : int
        Command used to set the time environment test level in decibels.
    SET_REPEAT : int
        Command used to enable repeating of the output time signal.
    SET_NO_REPEAT : int
        Command used to disable repeating of the output time signal.

    Unit Tests
    ----------
    test_environment_commands_have_unique_integer_values
        Verifies that registered environment command values are unique
        integers and that profile command metadata is well formed.

    test_profile_event_validate_environment_profile_command
        Verifies that an environment-specific profile command can validate
        with the correct data type.
    """

    class TimeUICommands
    """
    Commands sent from the time environment to the user interface.

    Attributes
    ----------
    TIME_DATA : int
        Command containing measured response data and output feedback data for
        time-history display.

    Unit Tests
    ----------
    test_time_environment_run_environment
        Verifies that time data is sent to the GUI update queue when
        acquisition data is received.
    """

### Time Metadata
<!---
MARK: Time Metadata
--->
    class TimeMetadata
    """
    Metadata container for the time environment.

    ``TimeMetadata`` stores the output signal, sampling parameters, output
    oversampling factor, selected hardware channels, and cancellation ramp-down
    time required to run a time-history output environment.

    Parameters
    ----------
    environment_name : str, optional
        Name of the time environment. Defaults to ``"Time"``.
    channel_list_bools : list of bool, optional
        Boolean mask identifying hardware channels assigned to this
        environment.
    sample_rate : int, optional
        Acquisition sample rate in samples per second.
    output_oversample : float, optional
        Output oversampling factor used to compute output timing.
    output_signal : numpy.ndarray, optional
        Two-dimensional output signal array with shape
        ``(output_channels, signal_samples)``.
    cancel_rampdown_time : float, optional
        Time in seconds used to ramp output to zero when the environment is
        stopped.

    Attributes
    ----------
    output_oversample : float
        Output oversampling factor.
    output_signal : numpy.ndarray
        Output signal to generate.
    cancel_rampdown_time : float
        Ramp-down time used during cancellation.
    signal_file : str or pathlib.Path or None
        Optional source file path used when saving metadata to worksheets.

    Unit Tests
    ----------
    test_time_metadata_init
        Verifies that ``TimeMetadata`` initializes and is an
        ``EnvironmentMetadata``.

    test_time_metadata_properties
        Verifies derived signal sample count, output channel count, signal
        duration, and ramp-down sample count.
    """

    def __init__
    """
    Initialize time environment metadata.

    Stores time-environment-specific metadata in addition to the base
    environment metadata. The signal file path is initialized to ``None`` and
    is only used for worksheet save/load workflows.

    Parameters
    ----------
    environment_name : str, optional
        Name of the time environment.
    channel_list_bools : list of bool, optional
        Boolean mask identifying hardware channels assigned to this
        environment.
    sample_rate : int, optional
        Acquisition sample rate in samples per second.
    output_oversample : float, optional
        Output oversampling factor.
    output_signal : numpy.ndarray, optional
        Output signal array with shape ``(output_channels, signal_samples)``.
    cancel_rampdown_time : float, optional
        Ramp-down time in seconds used when stopping the environment.

    Unit Tests
    ----------
    test_time_metadata_init
        Confirms that the object initializes as both ``TimeMetadata`` and
        ``EnvironmentMetadata``.
    """

    property
    def signal_samples
    """
    Return the number of samples in the output signal.

    Returns
    -------
    int
        Number of samples along the output signal time axis.

    Unit Tests
    ----------
    test_time_metadata_properties
        Verifies that this property returns the expected sample count.
    """

    property
    def output_channels
    """
    Return the number of output channels in the signal.

    Returns
    -------
    int
        Number of output signal channels.

    Unit Tests
    ----------
    test_time_metadata_properties
        Verifies that this property returns the expected output channel count.
    """

    property
    def signal_time
    """
    Return the duration of the output signal.

    The signal duration is computed from signal samples, acquisition sample
    rate, and output oversampling.

    Returns
    -------
    float
        Signal duration in seconds.

    Unit Tests
    ----------
    test_time_metadata_properties
        Verifies that this property returns the expected signal duration.
    """

    property
    def cancel_rampdown_samples
    """
    Return the number of samples used for cancellation ramp-down.

    The ramp-down sample count is computed from cancellation ramp-down time,
    sample rate, and output oversampling.

    Returns
    -------
    int
        Number of samples over which output is ramped toward zero.

    Unit Tests
    ----------
    test_time_metadata_properties
        Verifies that this property returns the expected sample count.
    """

    property
    def signal_file
    """
    Return the signal file associated with this metadata.

    Returns
    -------
    str or pathlib.Path or None
        Signal file path used for worksheet save/load workflows.
    """

    def set_file
    """
    Store the signal file path associated with this metadata.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the signal file used to create the output signal.
    """

    def validate
    """
    Validate time environment metadata against hardware metadata.

    Performs base environment metadata validation and then checks that
    cancellation ramp-down time is positive, sample rate is a positive integer,
    the output signal is a two-dimensional NumPy array, and the number of
    signal output channels matches the number of output hardware channels
    selected by the environment channel mask.

    Parameters
    ----------
    hardware_metadata : HardwareMetadata
        Hardware metadata containing the full hardware channel list and sample
        information.

    Returns
    -------
    bool
        ``True`` if validation succeeds.

    Raises
    ------
    RattlesnakeError
        If base environment metadata validation fails.
    RattlesnakeError
        If cancellation ramp-down time is missing or not positive.
    RattlesnakeError
        If sample rate is missing, not an integer, or not positive.
    RattlesnakeError
        If output signal is missing or is not a two-dimensional NumPy array.
    RattlesnakeError
        If the output signal channel count does not match selected output
        channels.

    Unit Tests
    ----------
    test_time_metadata_validate
        Verifies that valid metadata passes validation and invalid sample rate,
        ramp-down time, output signal type, output signal dimensions, or output
        channel counts raise ``RattlesnakeError``.
    """

    def save_metadata_to_netcdf
    """
    Save time environment metadata to a netCDF group.

    Stores cancellation ramp-down time, creates output channel and signal
    sample dimensions, creates the ``output_signal`` variable, and writes the
    output signal data.

    Parameters
    ----------
    netcdf_group_handle : nc4._netCDF4.Group
        netCDF group where time environment metadata should be stored.

    Unit Tests
    ----------
    test_environment_metadata_store_to_netcdf
        Verifies that time metadata can be written to a netCDF group.
    """

    classmethod
    def load_metadata_from_netcdf
    """
    Load time environment metadata from a netCDF group.

    Reads the saved output signal and cancellation ramp-down time from a
    netCDF group and constructs a ``TimeMetadata`` object using hardware sample
    rate and output oversampling values.

    Parameters
    ----------
    group : nc4._netCDF4.Group
        netCDF group containing saved time environment metadata.
    environment_name : str
        Name of the environment to reconstruct.
    channel_list_bools : list of bool
        Boolean channel mask identifying channels assigned to the environment.
    hardware_metadata : HardwareMetadata
        Hardware metadata associated with the saved environment.

    Returns
    -------
    TimeMetadata
        Reconstructed time environment metadata.
    """

    classmethod
    def create_blank_worksheet_template
    """
    Create a blank worksheet template for time environment metadata.

    Writes common environment worksheet fields and time-specific fields for
    signal file and cancellation ramp-down time.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet where the blank time metadata template should be written.
    """

    def save_metadata_to_worksheet
    """
    Save time environment metadata to an Excel worksheet.

    Writes the common environment worksheet template and stores the signal file
    path and cancellation ramp-down time when available.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet where metadata should be written.
    """

    classmethod
    def load_metadata_from_worksheet
    """
    Load time environment metadata from an Excel worksheet.

    Reads the signal file path and cancellation ramp-down time from the
    worksheet. The output signal is loaded from the signal file using the
    hardware sample rate. If signal loading fails, a placeholder signal is
    created.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet containing time environment metadata.
    environment_name : str
        Name of the environment to reconstruct.
    channel_list_bools : list of bool
        Boolean channel mask identifying channels assigned to the environment.
    hardware_metadata : HardwareMetadata
        Hardware metadata associated with the worksheet.

    Returns
    -------
    TimeMetadata
        Reconstructed time environment metadata.

    Raises
    ------
    RattlesnakeError
        If the worksheet contains a field that does not belong to the time
        environment.
    """

### Time Instructions
<!---
MARK: Time Instructions
--->
    class TimeInstructions
    """
    Runtime startup instructions for the time environment.

    These instructions define the initial test level and repeat state used when
    starting the time environment.

    Parameters
    ----------
    environment_name : str
        Name of the time environment.
    current_test_level : float
        Initial test level in decibels.
    repeat : bool
        Whether the output signal should repeat.

    Attributes
    ----------
    current_test_level : float
        Initial test level in decibels.
    repeat : bool
        Whether the signal repeats after reaching the end.

    Unit Tests
    ----------
    test_time_instructions_init
        Verifies that ``TimeInstructions`` initializes as an
        ``EnvironmentInstructions`` object and stores current test level and
        repeat attributes.
    """

    def __init__
    """
    Initialize time environment startup instructions.

    Parameters
    ----------
    environment_name : str
        Name of the time environment.
    current_test_level : float
        Initial test level in decibels.
    repeat : bool
        Whether the signal should repeat.

    Unit Tests
    ----------
    test_time_instructions_init
        Confirms that instruction fields are initialized.
    """

    def validate
    """
    Validate time environment startup instructions.

    Currently delegates to the base ``EnvironmentInstructions`` validation
    implementation.

    Unit Tests
    ----------
    test_time_instructions_init
        Verifies that instruction objects can be constructed for use by the
        environment.
    """

### Time Queues
<!---
MARK: Time Queues
--->
    class TimeQueues
    """
    Namespace containing queues used by the time environment.

    ``TimeQueues`` groups the command, GUI update, controller communication,
    data input, data output, and log queues needed by ``TimeEnvironment``.

    Parameters
    ----------
    environment_command_queue : VerboseMessageQueue
        Queue from which the environment receives commands.
    gui_update_queue : multiprocessing.Queue or queue.Queue
        Queue where GUI update messages are sent.
    controller_communication_queue : VerboseMessageQueue
        Queue where commands to the global controller are sent.
    data_in_queue : multiprocessing.Queue or queue.Queue
        Queue from which acquired data is received.
    data_out_queue : multiprocessing.Queue or queue.Queue
        Queue where output data is written.
    log_file_queue : multiprocessing.Queue
        Queue where log messages are written.

    Attributes
    ----------
    environment_command_queue : VerboseMessageQueue
        Environment command queue.
    gui_update_queue : multiprocessing.Queue or queue.Queue
        GUI update queue.
    controller_communication_queue : VerboseMessageQueue
        Controller command queue.
    data_in_queue : multiprocessing.Queue or queue.Queue
        Acquisition data queue.
    data_out_queue : multiprocessing.Queue or queue.Queue
        Output data queue.
    log_file_queue : multiprocessing.Queue
        Log file queue.

    Unit Tests
    ----------
    test_time_queues_init
        Verifies that a ``TimeQueues`` object can be initialized.
    """

    def __init__
    """
    Initialize the time environment queue namespace.

    Stores all queue references required by the time environment.

    Parameters
    ----------
    environment_command_queue : VerboseMessageQueue
        Queue from which the environment receives commands.
    gui_update_queue : multiprocessing.Queue or queue.Queue
        Queue where GUI update messages are sent.
    controller_communication_queue : VerboseMessageQueue
        Queue where controller commands are sent.
    data_in_queue : multiprocessing.Queue or queue.Queue
        Queue from which acquisition data is received.
    data_out_queue : multiprocessing.Queue or queue.Queue
        Queue where output data is written.
    log_file_queue : multiprocessing.Queue
        Queue where log messages are written.

    Unit Tests
    ----------
    test_time_queues_init
        Confirms that the queue namespace initializes successfully.
    """

### Time Environment
<!---
MARK: Time Environment Class
--->
    class TimeEnvironment
    """
    Environment that outputs a predefined time-history signal.

    ``TimeEnvironment`` reads a stored output signal from ``TimeMetadata`` and
    writes scaled chunks of that signal to the output queue. It supports
    starting with a specified test level, repeating or not repeating the
    signal, changing test level while running, ramping to zero during stop, and
    forwarding acquired measurement/output data to the GUI.

    Parameters
    ----------
    environment_name : str
        Name of the environment.
    queue_name : str
        Internal queue name assigned to the environment.
    queue_container : TimeQueues
        Container of queues used by the time environment.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    active_event : multiprocessing.synchronize.Event
        Event indicating whether this environment is active.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether this environment is ready.

    Attributes
    ----------
    queue_container : TimeQueues
        Queue namespace used by the environment.
    shutdown_flag : bool
        Flag indicating shutdown state.
    current_test_level : float
        Current linear test-level scale factor.
    target_test_level : float
        Target test-level scale factor.
    test_level_change : float
        Per-sample change used while ramping between test levels.
    repeat : bool
        Whether the output signal repeats.
    signal_remainder : numpy.ndarray or None
        Remaining signal samples waiting to be output.
    output_channels : list of int or None
        Hardware channel indices corresponding to output channels.
    measurement_channels : list of int or None
        Hardware channel indices corresponding to measurement channels.

    Unit Tests
    ----------
    test_time_environment_init
        Verifies that ``TimeEnvironment`` initializes and is an ``Environment``.
    """

    def __init__
    """
    Initialize the time environment.

    Initializes the base ``Environment``, stores the queue container, maps time
    commands to environment methods, initializes runtime state and metadata
    references, and marks the environment ready.

    Parameters
    ----------
    environment_name : str
        Name of the environment.
    queue_name : str
        Internal queue name assigned to the environment.
    queue_container : TimeQueues
        Queue namespace used by the environment.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    active_event : multiprocessing.synchronize.Event
        Event indicating whether this environment is active.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether this environment is ready.

    Unit Tests
    ----------
    test_time_environment_init
        Confirms that the environment initializes successfully.
    """

    def initialize_hardware
    """
    Initialize hardware information for the time environment.

    Stores hardware metadata, identifies measurement channels as channels
    without feedback devices, identifies output channels as channels with
    feedback devices, and marks the environment ready.

    Parameters
    ----------
    hardware_metadata : HardwareMetadata
        Hardware metadata containing channel configuration and sampling
        information.

    Unit Tests
    ----------
    test_time_environment_initialize_hardware
        Verifies that hardware metadata is stored and measurement/output
        channel indices are computed.
    """

    def initialize_environment
    """
    Initialize time environment metadata.

    Stores the supplied ``TimeMetadata`` through the base environment
    initializer and marks the environment ready.

    Parameters
    ----------
    environment_metadata : TimeMetadata
        Metadata defining the time-history output signal and ramp-down
        behavior.

    Unit Tests
    ----------
    test_time_environment_initialize_environment
        Verifies that environment metadata is stored.
    """

    def run_environment
    """
    Run one time-environment loop iteration.

    On startup, applies supplied instructions, initializes the signal remainder,
    marks the environment active, and notifies the GUI that the environment has
    started. During each iteration, it reads any available acquisition data and
    sends measurement/output data to the GUI. If the output queue is ready, it
    writes the next output chunk, optionally repeats the signal, and determines
    whether the signal is complete. When the final signal is written, it waits
    for the final acquisition data before shutting down. If not complete, it
    requeues itself to continue running.

    Parameters
    ----------
    data : TimeInstructions or None
        Startup instructions containing test level and repeat state. Subsequent
        loop iterations may pass ``None``.

    Unit Tests
    ----------
    test_time_environment_run_environment
        Verifies startup behavior, instruction GUI updates, environment-started
        GUI updates, acquisition data forwarding to the GUI, output chunk
        selection, output call arguments, command-loop requeueing, and that
        shutdown is not called during normal non-final output.
    """

    def output
    """
    Write scaled time signal data to the output queue.

    Scales the supplied signal by the current test level. If a test-level ramp
    is active, computes per-sample scaling, clamps the ramp at the target
    level once reached, updates the current test level, and clears the ramp
    when complete. The scaled output data and final-signal flag are then placed
    on the data output queue.

    Parameters
    ----------
    write_data : numpy.ndarray
        Output signal samples to write.
    last_signal : bool, optional
        Whether this is the final signal chunk for the current run. Defaults
        to ``False``.

    Unit Tests
    ----------
    test_time_environment_output
        Verifies constant test-level output, ramped test-level output, output
        queue data values, final-signal flag handling, and expected log
        messages.
    """

    def set_test_level
    """
    Set the target test level in decibels.

    Converts the supplied decibel level to a linear scale factor, begins a
    test-level ramp toward the new scale, and notifies the GUI of the command.

    Parameters
    ----------
    data : int
        Target test level in decibels.
    """

    def set_no_repeat
    """
    Disable signal repeat mode.

    Clears the repeat flag, logs the change, and notifies the GUI.

    Parameters
    ----------
    data : Any, optional
        Command data supplied through the command queue. This value is passed
        through to the GUI update.
    """

    def set_repeat
    """
    Enable signal repeat mode.

    Sets the repeat flag, logs the change, and notifies the GUI.

    Parameters
    ----------
    data : Any, optional
        Command data supplied through the command queue. This value is passed
        through to the GUI update.
    """

    def stop_environment
    """
    Stop the time environment by ramping the test level to zero.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_time_environment_stop_environment
        Verifies that stopping the environment requests a test-level adjustment
        to zero.
    """

    def adjust_test_level
    """
    Begin ramping toward a new target test level.

    Computes the per-sample test-level change needed to reach the supplied
    target level over the configured cancellation ramp-down sample count.

    Parameters
    ----------
    data : float
        Target linear test-level scale factor.

    Unit Tests
    ----------
    test_time_environment_adjust_test_level
        Verifies that the target test level and per-sample change are updated
        and that the change is logged.
    """

    def shutdown
    """
    Shut down the time environment after final acquisition data is received.

    Logs shutdown, flushes pending environment commands, clears the active
    event, and notifies the GUI that the environment has ended.

    Unit Tests
    ----------
    test_time_environment_shutdown
        Verifies that shutdown is logged, the command queue is flushed, the
        active event is cleared, and an environment-ended GUI update is sent.
    """

### Time Process
<!---
MARK: Time Process
--->
    def time_process
    """
    Entry point used to start the time environment process.

    Constructs a ``TimeQueues`` container and a ``TimeEnvironment`` instance
    using the supplied queues and events, then runs the environment command
    loop until the shutdown event is set or a quit command is received. This
    function is intended to be used as a multiprocessing or threaded process
    target.

    Parameters
    ----------
    environment_name : str
        Name of the environment.
    queue_name : str
        Internal queue name assigned to the environment.
    input_queue : VerboseMessageQueue
        Queue used to receive commands sent to the environment.
    gui_update_queue : multiprocessing.Queue or queue.Queue
        Queue used to send GUI updates.
    controller_command_queue : VerboseMessageQueue
        Queue used to send commands to the controller.
    log_file_queue : multiprocessing.Queue
        Queue used to send log messages.
    data_in_queue : multiprocessing.Queue or queue.Queue
        Queue used to receive acquired data.
    data_out_queue : multiprocessing.Queue or queue.Queue
        Queue used to send output data.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    active_event : multiprocessing.synchronize.Event
        Event indicating whether the time environment is active.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the time environment is ready.
    shutdown_event : multiprocessing.synchronize.Event
        Event used to signal process shutdown.
    sysid_active_event : multiprocessing.synchronize.Event
        Unused system-identification active event supplied for process
        signature compatibility.
    sysid_stored_event : multiprocessing.synchronize.Event
        Unused system-identification stored event supplied for process
        signature compatibility.
    ping_alive_event : multiprocessing.synchronize.Event
        Event used for process liveness/watchdog compatibility.
    threaded : bool
        Whether the environment is running in threaded mode.

    Unit Tests
    ----------
    test_time_process
        Verifies that the process function constructs a ``TimeEnvironment`` and
        calls its ``run`` method.
    """

## Modal Environment
<!---
MARK: Modal Environment
--->
    """
    Modal testing environment.

    This module defines an environment for hammer or shaker modal testing. It
    coordinates signal generation, data collection, and spectral processing
    subprocesses to acquire time frames, estimate frequency response functions,
    compute related spectral quantities, and send modal data updates to the
    user interface.
    """

### Modal Commands
<!---
MARK: Modal Commands
--->
    class ModalCommands
    """
    Commands supported by the modal environment.

    Attributes
    ----------
    ACCEPT_FRAME : int
        Accept or reject the most recent manually reviewed modal frame.
    RUN_CONTROL : int
        Continue the modal environment control loop.
    CHECK_FOR_COMPLETE_SHUTDOWN : int
        Check whether all modal subprocesses have completed shutdown.

    Unit Tests
    ----------
    test_modal_commands
        Verifies that modal command enum values construct valid
        ``ModalCommands`` members.
    """

    class ModalUICommands
    """
    Commands sent from the modal environment to the user interface.

    Attributes
    ----------
    SPECTRAL_UPDATE : int
        GUI update containing modal spectral quantities including frames,
        frequencies, FRFs, coherence, CPSD data, and condition information.

    Unit Tests
    ----------
    test_modal_environment_run_control
        Verifies that spectral data are forwarded to the GUI as a spectral
        update.
    """

### Modal Metadata
<!---
MARK: Modal Metadata
--->
    class ModalMetadata
    """
    Metadata container for the modal environment.

    ``ModalMetadata`` stores sampling, averaging, FRF estimation, triggering,
    acceptance, signal-generation, channel-selection, and windowing parameters
    required to configure modal testing.

    Parameters
    ----------
    environment_name : str
        Name of the modal environment.
    channel_list_bools : list of bool
        Boolean mask identifying hardware channels assigned to this
        environment.
    sample_rate : int
        Acquisition sample rate in samples per second.
    samples_per_frame : int
        Number of samples per measurement frame.
    averaging_type : str
        Averaging type, typically ``"Linear"`` or ``"Exponential"``.
    num_averages : int
        Number of averages used for FRF estimation.
    averaging_coefficient : float
        Exponential averaging coefficient.
    frf_technique : str
        FRF estimator name such as ``"H1"``, ``"H2"``, ``"H3"``, or ``"Hv"``.
    frf_window : str
        Window type used for FRF computation.
    overlap_percent : float
        Percent overlap between frames.
    trigger_type : str
        Triggering strategy, such as ``"Free Run"``, ``"First Frame"``, or
        ``"Every Frame"``.
    accept_type : str
        Frame acceptance strategy, such as ``"Accept All"``, ``"Manual"``, or
        ``"Autoreject..."``.
    wait_for_steady_state : float
        Time in seconds to wait before accepting steady-state frames.
    trigger_channel : int
        Zero-based channel index used for triggering.
    pretrigger_percent : float
        Percent of frame to use as pretrigger data.
    trigger_slope_positive : bool
        Whether the trigger slope is positive.
    trigger_level_percent : float
        Trigger level as a percent of channel range.
    hysteresis_level_percent : float
        Hysteresis reset level as a percent of channel range.
    hysteresis_frame_percent : float
        Fraction of frame required to satisfy hysteresis reset.
    signal_generator_type : str
        Signal generator type, such as ``"none"``, ``"random"``,
        ``"pseudorandom"``, ``"burst"``, ``"chirp"``, ``"square"``, or
        ``"sine"``.
    signal_generator_level : float
        Output level for generated excitation.
    signal_generator_min_frequency : float
        Minimum frequency, or sine/square frequency.
    signal_generator_max_frequency : float
        Maximum frequency for broadband signals.
    signal_generator_on_percent : float
        On fraction for burst or square signals.
    acceptance_function : tuple or None
        Optional ``(module_path, function_name)`` tuple for automatic frame
        rejection.
    reference_channel_indices : list of int
        Reference channel indices within the environment channel list.
    response_channel_indices : list of int
        Response channel indices within the environment channel list.
    output_channel_indices : list of int
        Output channel indices within the environment channel list.
    output_oversample : int
        Output oversampling factor.
    exponential_window_value_at_frame_end : float
        End value used when configuring exponential windows.

    Attributes
    ----------
    signal_generator : SignalGenerator or None
        Signal generator object constructed from the signal generator settings.

    Unit Tests
    ----------
    test_modal_metadata_init
        Verifies that modal metadata initializes and that derived properties
        return expected values.

    test_modal_metadata_get_trigger_levels
        Verifies conversion of trigger and hysteresis levels to volts and
        engineering units.

    test_modal_metadata_generate_signal
        Verifies generated signal behavior with and without a configured signal
        generator.
    """

    def __init__
    """
    Initialize modal environment metadata.

    Stores modal test settings, converts percent values to fractions, stores
    channel selections, stores output oversampling, and creates the configured
    signal generator.

    Unit Tests
    ----------
    test_modal_metadata_init
        Verifies initialization and derived modal metadata properties.
    """

    def get_signal_generator
    """
    Create the configured modal excitation signal generator.

    Constructs a signal generator based on ``signal_generator_type``. Supported
    types include no output, random, pseudorandom, burst random, chirp, square,
    and sine. If the type is not recognized, ``None`` is returned.

    Returns
    -------
    SignalGenerator or None
        Configured signal generator, or ``None`` for invalid generator types.

    Unit Tests
    ----------
    test_modal_metadata_get_signal_generator
        Verifies that supported signal generator types construct the expected
        signal generator classes.
    """

    property
    def samples_per_acquire
    """
    Return the number of new samples per acquisition frame.

    Returns
    -------
    int
        Samples per acquisition step after accounting for overlap.

    Unit Tests
    ----------
    test_modal_metadata_init
        Verifies this derived sample count.
    """

    property
    def frame_time
    """
    Return the duration of a measurement frame.

    Returns
    -------
    float
        Frame duration in seconds.

    Unit Tests
    ----------
    test_modal_metadata_init
        Verifies this derived frame duration.
    """

    property
    def nyquist_frequency
    """
    Return the Nyquist frequency.

    Returns
    -------
    float
        Half of the sample rate.

    Unit Tests
    ----------
    test_modal_metadata_init
        Verifies this derived frequency.
    """

    property
    def fft_lines
    """
    Return the number of FFT frequency lines.

    Returns
    -------
    int
        Number of one-sided FFT lines for the configured frame size.

    Unit Tests
    ----------
    test_modal_metadata_init
        Verifies this derived FFT line count.
    """

    property
    def skip_frames
    """
    Return the number of frames to skip while waiting for steady state.

    Returns
    -------
    int
        Number of frames to skip after output starts.

    Unit Tests
    ----------
    test_modal_metadata_init
        Verifies this derived skip-frame count.
    """

    property
    def frequency_spacing
    """
    Return the frequency spacing.

    Returns
    -------
    float
        Frequency resolution in hertz.

    Unit Tests
    ----------
    test_modal_metadata_init
        Verifies this derived frequency spacing.
    """

    def get_trigger_levels
    """
    Convert trigger and hysteresis levels to volts and engineering units.

    Uses the selected trigger channel's voltage range and sensitivity to
    convert fractional trigger and hysteresis levels to volts and engineering
    units. Defaults are used when channel values are missing or invalid.

    Parameters
    ----------
    channels : list of Channel
        Environment channel list.

    Returns
    -------
    tuple
        Tuple ``(trigger_level_v, trigger_level_eu, hysteresis_level_v,
        hysteresis_level_eu)``.

    Unit Tests
    ----------
    test_modal_metadata_get_trigger_levels
        Verifies trigger and hysteresis level conversion.
    """

    property
    def disabled_signals
    """
    Return indices of disabled output signals.

    Output signals are disabled when their output channel index is not present
    in either response or reference channel index lists.

    Returns
    -------
    list of int
        Disabled output signal indices.

    Unit Tests
    ----------
    test_modal_metadata_init
        Verifies disabled signal calculation.
    """

    property
    def hysteresis_samples
    """
    Return the number of hysteresis samples.

    Returns
    -------
    int
        Number of samples for which the trigger signal must satisfy hysteresis
        before another trigger is considered.

    Unit Tests
    ----------
    test_modal_metadata_init
        Verifies hysteresis sample calculation.
    """

    def generate_signal
    """
    Generate one frame of modal excitation data.

    If no signal generator is configured, returns zeros. Otherwise, returns one
    frame from the configured signal generator.

    Returns
    -------
    numpy.ndarray
        Generated output signal frame.

    Unit Tests
    ----------
    test_modal_metadata_generate_signal
        Verifies generated signal behavior for ``None`` and mocked signal
        generators.
    """

    def validate
    """
    Validate modal metadata against hardware metadata.

    Delegates common metadata validation to ``EnvironmentMetadata``.

    Parameters
    ----------
    hardware_metadata : HardwareMetadata
        Hardware metadata used for validation.

    Returns
    -------
    Any
        Result of base metadata validation.
    """

    def save_metadata_to_netcdf
    """
    Save modal metadata to a netCDF group.

    Writes modal configuration attributes, acceptance-function information,
    reference channel indices, and response channel indices to the supplied
    group.

    Parameters
    ----------
    netcdf_group_handle : nc4._netCDF4.Group
        netCDF group where modal metadata should be stored.

    Unit Tests
    ----------
    test_modal_metadata_store_to_netcdf
        Verifies that reference and response channel dimensions and variables
        are created.
    """

    classmethod
    def load_metadata_from_netcdf
    """
    Load modal metadata from a netCDF group.

    Reads modal configuration attributes, channel index variables, and
    environment output channel information from a netCDF group and constructs a
    ``ModalMetadata`` object.

    Parameters
    ----------
    netcdf_group_handle : nc4._netCDF4.Group
        netCDF group containing modal metadata.
    environment_name : str
        Environment name.
    channel_list_bools : list of bool
        Boolean channel mask for this environment.
    hardware_metadata : HardwareMetadata
        Hardware metadata associated with the saved file.

    Returns
    -------
    ModalMetadata
        Reconstructed modal metadata object.
    """

    classmethod
    def create_blank_worksheet_template
    """
    Create a blank Excel worksheet template for modal metadata.

    Writes common environment worksheet fields and modal-specific parameter
    rows for FRF estimation, triggering, acceptance, signal generation,
    steady-state wait time, autoreject function, reference channels, and
    disabled channels.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet where the modal template should be written.
    """

    def save_metadata_to_worksheet
    """
    Save modal metadata to an Excel worksheet.

    Writes modal metadata values to their corresponding worksheet rows,
    including frame settings, averaging settings, FRF settings, trigger
    settings, signal generator settings, acceptance function information,
    reference channels, and disabled channels.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet where modal metadata should be written.
    """

    classmethod
    def load_metadata_from_worksheet
    """
    Load modal metadata from an Excel worksheet.

    Reads modal worksheet values, reconstructs reference, response, disabled,
    and output channel lists, converts worksheet fractions to percent inputs,
    and constructs a ``ModalMetadata`` object.

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        Worksheet containing modal metadata.
    environment_name : str
        Environment name.
    channel_list_bools : list of bool
        Boolean channel mask for this environment.
    hardware_metadata : HardwareMetadata
        Hardware metadata associated with the worksheet.

    Returns
    -------
    ModalMetadata
        Reconstructed modal metadata object.
    """

### Modal Instructions
<!---
MARK: Modal Instructions
--->
    class ModalInstructions
    """
    Runtime startup instructions for the modal environment.

    Parameters
    ----------
    environment_name : str
        Name of the modal environment.

    Unit Tests
    ----------
    test_modal_instructions_init
        Verifies that modal instructions initialize as
        ``EnvironmentInstructions``.
    """

    def __init__
    """
    Initialize modal instructions.

    Parameters
    ----------
    environment_name : str
        Name of the modal environment.
    """

    def validate
    """
    Validate modal instructions.

    Delegates validation to the base ``EnvironmentInstructions`` implementation.
    """

### Modal Queues
<!---
MARK: Modal Queues
--->
    class ModalQueues
    """
    Namespace containing queues used by the modal environment.

    In addition to environment command, data, GUI, controller, and log queues,
    this container creates queues for spectral computation, signal generation,
    and data collection subprocesses.

    Parameters
    ----------
    environment_name : str
        Environment name used to label subprocess command queues.
    environment_command_queue : VerboseMessageQueue
        Queue from which the modal environment receives commands.
    gui_update_queue : multiprocessing.Queue or queue.Queue
        Queue where GUI updates are sent.
    controller_communication_queue : VerboseMessageQueue
        Queue where commands to the controller are sent.
    data_in_queue : multiprocessing.Queue or queue.Queue
        Queue from which acquired data are received.
    data_out_queue : multiprocessing.Queue or queue.Queue
        Queue where generated output data are written.
    log_file_queue : multiprocessing.Queue
        Queue where log messages are written.
    threaded : bool
        If ``True``, creates thread queues for subprocess communication;
        otherwise creates multiprocessing queues.

    Attributes
    ----------
    data_for_spectral_computation_queue : multiprocessing.Queue or queue.Queue
        Queue from collector to spectral processing.
    updated_spectral_quantities_queue : multiprocessing.Queue or queue.Queue
        Queue from spectral processing to modal environment.
    signal_generation_update_queue : multiprocessing.Queue or queue.Queue
        Queue used by signal generation.
    spectral_command_queue : VerboseMessageQueue
        Command queue for spectral processing.
    collector_command_queue : VerboseMessageQueue
        Command queue for data collection.
    signal_generation_command_queue : VerboseMessageQueue
        Command queue for signal generation.

    Unit Tests
    ----------
    test_modal_queues_init
        Verifies that a modal queue container can be initialized.
    """

### Modal Environment
<!---
MARK: Modal Environment Class
--->
    class ModalEnvironment
    """
    Controller-side environment for modal testing.

    ``ModalEnvironment`` coordinates data collection, signal generation, and
    spectral processing subprocesses for modal testing. It initializes
    subprocess metadata, starts modal acquisition and excitation, forwards
    spectral updates to the GUI, handles manual frame acceptance, and performs
    coordinated shutdown.

    Parameters
    ----------
    environment_name : str
        Name of the modal environment.
    queue_name : str
        Internal queue name assigned to the environment.
    queue_container : ModalQueues
        Queue namespace used by the modal environment.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    active_event : multiprocessing.synchronize.Event
        Event indicating whether the modal environment is active.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the modal environment is ready.

    Attributes
    ----------
    frame_number : int
        Modal frame counter.
    siggen_shutdown_achieved : bool
        Whether signal generation shutdown is complete.
    collector_shutdown_achieved : bool
        Whether collector shutdown is complete.
    spectral_shutdown_achieved : bool
        Whether spectral processing shutdown is complete.

    Unit Tests
    ----------
    test_modal_environment_init
        Verifies that the modal environment initializes successfully.
    """

    def __init__
    """
    Initialize the modal environment.

    Initializes the base environment, stores queues and state, maps modal and
    subprocess shutdown commands, initializes shutdown flags, and marks the
    environment ready.
    """

    def initialize_hardware
    """
    Store hardware metadata for modal testing.

    Parameters
    ----------
    hardware_metadata : HardwareMetadata
        Hardware metadata containing channel and sampling configuration.
    """

    def initialize_environment
    """
    Initialize modal environment metadata and subprocess metadata.

    Stores modal metadata, sends collector metadata to the data collector,
    sends signal generation metadata to the signal generation process, sends
    spectral processing metadata to spectral processing, and marks the
    environment ready.

    Parameters
    ----------
    environment_metadata : ModalMetadata
        Modal environment metadata.

    Unit Tests
    ----------
    test_modal_environment_initialize_environment_test_parameters
        Verifies that collector, signal generation, and spectral processing
        initialization commands are sent.
    """

    def get_data_collector_metadata
    """
    Build collector metadata for modal testing.

    Converts modal acquisition, trigger, acceptance, and window settings into a
    ``CollectorMetadata`` object used by the data collector.

    Returns
    -------
    CollectorMetadata
        Data collector metadata.

    Raises
    ------
    ValueError
        If trigger type, acceptance type, or window type is invalid.

    Unit Tests
    ----------
    test_modal_environment_get_data_collector_metadata
        Verifies that collector metadata is constructed from modal metadata.
    """

    def get_spectral_processing_metadata
    """
    Build spectral processing metadata for modal testing.

    Converts modal averaging and FRF estimator settings into a
    ``SpectralProcessingMetadata`` object.

    Returns
    -------
    SpectralProcessingMetadata
        Spectral processing metadata.

    Raises
    ------
    ValueError
        If the FRF estimator is invalid.

    Unit Tests
    ----------
    test_modal_environment_get_spectral_processing_metadata
        Verifies that spectral processing metadata is constructed from modal
        metadata.
    """

    def get_signal_generation_metadata
    """
    Build signal generation metadata for modal testing.

    Returns
    -------
    SignalGenerationMetadata
        Signal generation metadata containing write size, ramp samples,
        optional transformation matrix, and disabled signal indices.

    Unit Tests
    ----------
    test_modal_environment_get_signal_generator_metadata
        Verifies that signal generation metadata is constructed from modal
        metadata.
    """

    def get_signal_generator
    """
    Return the modal signal generator.

    Returns
    -------
    SignalGenerator or None
        Signal generator from the modal metadata.

    Unit Tests
    ----------
    test_modal_environment_get_signal_generator
        Verifies delegation to the metadata signal generator method.
    """

    def start_environment
    """
    Start modal testing.

    Reinitializes subprocesses, sets test level and skip frames, initializes and
    starts signal generation, starts data collection, initializes and starts
    spectral processing, queues the modal control loop, marks the environment
    active, and notifies the GUI that the environment has started.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_modal_environment_start_environment
        Verifies that modal startup commands are sent to collector, signal
        generation, spectral processing, and environment command queues and
        that startup is logged.
    """

    def run_control
    """
    Run one modal control-loop iteration.

    Flushes spectral updates from the spectral processing queue. If updates are
    available, sends the most recent update to the GUI. Otherwise waits briefly.
    Requeues itself so modal control continues.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_modal_environment_run_control
        Verifies spectral data forwarding to the GUI and requeueing of the
        modal control command.
    """

    def siggen_shutdown_achieved_fn
    """
    Mark signal generation shutdown as complete.

    Unit Tests
    ----------
    test_modal_environment_siggen_shutdown_achieved_fn
        Verifies that the signal generation shutdown flag is set.
    """

    def collector_shutdown_achieved_fn
    """
    Mark data collector shutdown as complete.

    Unit Tests
    ----------
    test_modal_environment_collector_shutdown_achieved_fn
        Verifies that the collector shutdown flag is set.
    """

    def spectral_shutdown_achieved_fn
    """
    Mark spectral processing shutdown as complete.

    Unit Tests
    ----------
    test_modal_environment_spectral_shutdown_achieved_fn
        Verifies that the spectral processing shutdown flag is set.
    """

    def check_for_shutdown
    """
    Check whether all modal subprocesses have completed shutdown.

    If signal generation, data collection, and spectral processing have all
    reported shutdown, clears the environment active event and notifies the GUI
    that the environment ended. Otherwise, waits briefly and requeues another
    shutdown check.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_modal_environment_check_for_shutdown
        Verifies complete-shutdown GUI notification and incomplete-shutdown
        requeue behavior.
    """

    def accept_frame
    """
    Accept or reject the previous modal frame.

    Forwards the acceptance decision to the data collector.

    Parameters
    ----------
    data : bool
        ``True`` to accept the frame, ``False`` to reject it.

    Unit Tests
    ----------
    test_modal_environment_accept_frame
        Verifies that the acceptance command is sent to the collector.
    """

    def stop_environment
    """
    Stop modal testing gracefully.

    Flushes pending environment commands, instructs the collector to skip
    frames while staying at level, starts signal generator shutdown, stops
    spectral processing, and queues a shutdown-completion check.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_modal_environment_stop_environment
        Verifies shutdown commands are sent to collector, signal generation,
        spectral processing, and environment command queues.
    """

    def quit
    """
    Quit the modal environment and subprocesses.

    Sends quit commands to spectral processing, signal generation, and data
    collector command queues, then returns ``True`` so the environment command
    loop exits.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Returns
    -------
    bool
        Always returns ``True``.

    Unit Tests
    ----------
    test_modal_environment_quit
        Verifies that quit commands are sent to all modal subprocess queues.
    """

### Modal Process
<!---
MARK: Modal Process
--->
    def modal_process
    """
    Entry point used to start the modal environment process.

    Creates modal queues, starts spectral processing, signal generation, and
    data collection subprocesses, constructs and runs the modal environment,
    then joins all modal subprocesses after the environment command loop exits.

    Parameters
    ----------
    environment_name : str
        Name of the environment.
    queue_name : str
        Internal queue name assigned to the environment.
    input_queue : VerboseMessageQueue
        Environment command queue.
    gui_update_queue : multiprocessing.Queue or queue.Queue
        GUI update queue.
    controller_command_queue : VerboseMessageQueue
        Controller command queue.
    log_file_queue : multiprocessing.Queue
        Log file queue.
    data_in_queue : multiprocessing.Queue or queue.Queue
        Acquisition data queue.
    data_out_queue : multiprocessing.Queue or queue.Queue
        Output data queue.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    active_event : multiprocessing.synchronize.Event
        Event indicating whether the modal environment is active.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the modal environment is ready.
    shutdown_event : multiprocessing.synchronize.Event
        Event used to stop the modal environment.
    sysid_active_event : multiprocessing.synchronize.Event
        Unused system-identification active event supplied for process
        signature compatibility.
    sysid_stored_event : multiprocessing.synchronize.Event
        Unused system-identification stored event supplied for process
        signature compatibility.
    ping_alive_event : multiprocessing.synchronize.Event
        Watchdog/liveness event.
    threaded : bool
        If ``True``, subprocesses are created as threads. Otherwise they are
        created as multiprocessing processes.

    Unit Tests
    ----------
    test_modal_process_function
        Verifies that modal subprocesses are started, the modal environment is
        run, and subprocesses are joined after completion.
    """

# Process

## Abstract Message Process
<!---
MARK: Abstract Message Process
--->
    """
    Defines abstract processes that can be used as subprocesses in the
    controller.

    This module provides a common message-driven process interface based on
    the ``(message, data)`` producer-consumer paradigm. Subclasses and
    concrete users map command messages to callable methods, receive commands
    from a ``VerboseMessageQueue``, execute the mapped methods, and communicate
    status, logs, and errors back to the controller or GUI.
    """

### Abstract Message Process
<!---
MARK: Abstract Message Process Class
--->
    class AbstractMessageProcess
    """
    Abstract base class for message-driven controller subprocesses.

    This class provides the common command-loop behavior used by controller
    subprocesses. Each process receives ``(message, data)`` pairs from a
    command queue, looks up the corresponding callable in ``command_map``, and
    calls that function with ``data``. If the mapped function returns a truthy
    value, the command loop exits.

    The class also provides common logging, GUI error reporting, ready-event
    management, and command registration behavior. It operates similarly to
    the abstract environment command loop but is intended for processes that
    are subordinate to an environment or controller task.

    Parameters
    ----------
    process_name : str
        Name of the process. Used for logging, queue access, and error
        messages.
    log_file_queue : multiprocessing.Queue
        Queue used to send formatted log messages to the logging process.
    command_queue : VerboseMessageQueue
        Queue from which this process receives command ``(message, data)``
        pairs.
    gui_update_queue : multiprocessing.Queue or queue.Queue
        Queue used to send GUI update commands and error notifications.
    ready_event : multiprocessing.synchronize.Event, optional
        Event indicating whether the process is initialized and ready. If
        supplied, the event is set during initialization.

    Attributes
    ----------
    process_name : str
        Name of the process.
    command_map : dict
        Mapping from command messages to callable process methods.
    gui_update_queue : multiprocessing.Queue or queue.Queue
        Queue used to send GUI update commands.
    command_queue : VerboseMessageQueue
        Queue used to receive process commands.
    log_file_queue : multiprocessing.Queue
        Queue used to send log messages.
    ready_event : multiprocessing.synchronize.Event
        Event indicating process readiness, if one was supplied.

    Unit Tests
    ----------
    test_message_process_init
        Verifies that an ``AbstractMessageProcess`` can be initialized in
        threaded and non-threaded configurations.

    test_message_process_properties
        Verifies that process properties return expected values and that the
        default command map contains ``GlobalCommands.QUIT``.
    """

    def __init__
    """
    Initialize the message-driven process.

    Stores the process name, queues, optional ready event, and initializes the
    command map with the global quit command. If a ready event is supplied, it
    is set during initialization.

    Parameters
    ----------
    process_name : str
        Name of the process used for logging and queue access.
    log_file_queue : multiprocessing.Queue
        Queue used to send formatted log messages.
    command_queue : VerboseMessageQueue
        Queue from which process commands are received.
    gui_update_queue : multiprocessing.Queue or queue.Queue
        Queue used to send GUI updates and error messages.
    ready_event : multiprocessing.synchronize.Event, optional
        Event indicating process readiness. If supplied, the event is set
        during initialization.

    Unit Tests
    ----------
    test_message_process_init
        Verifies that the process initializes successfully.

    test_message_process_properties
        Verifies that the default command map is initialized with
        ``GlobalCommands.QUIT``.
    """

    def log
    """
    Queue a formatted message for the log file.

    Formats the supplied message with the current timestamp and process name,
    then places it on the log file queue.

    Parameters
    ----------
    message : str
        Message to write to the log file.

    Unit Tests
    ----------
    test_message_process_log
        Verifies that calling this method places the expected formatted log
        message on the log file queue.
    """

    property
    def process_name
    """
    Return the process name.

    Returns
    -------
    str
        Name of the process used for logging, queue access, and error
        reporting.

    Unit Tests
    ----------
    test_message_process_properties
        Verifies that this property returns the name supplied during
        initialization.
    """

    property
    def command_map
    """
    Return the command-to-function mapping for this process.

    The command map is used by ``run`` to determine which function should be
    called when a command message is received from the command queue.

    Returns
    -------
    dict
        Mapping from command messages to callable process methods.

    Unit Tests
    ----------
    test_message_process_properties
        Verifies that this property returns the default command map containing
        ``GlobalCommands.QUIT``.

    test_abstract_message_process_map_command
        Verifies that commands can be added to this mapping.
    """

    property
    def gui_update_queue
    """
    Return the GUI update queue.

    Returns
    -------
    multiprocessing.Queue or queue.Queue
        Queue used to send GUI update commands and error notifications.

    Unit Tests
    ----------
    test_message_process_properties
        Accesses this property to verify it is available.
    """

    property
    def ready_event
    """
    Return the ready event associated with this process.

    Returns
    -------
    multiprocessing.synchronize.Event or threading.Event
        Event indicating whether this process is ready.

    Unit Tests
    ----------
    test_message_process_set_ready
        Verifies that the ready event can be set.

    test_message_process_clear_ready
        Verifies that the ready event can be cleared.
    """

    def set_ready
    """
    Set the process ready event.

    If a ready event was supplied during initialization, this method marks the
    process as ready.

    Unit Tests
    ----------
    test_message_process_set_ready
        Verifies that calling this method sets the ready event.
    """

    def clear_ready
    """
    Clear the process ready event.

    If a ready event was supplied during initialization, this method marks the
    process as not ready.

    Unit Tests
    ----------
    test_message_process_clear_ready
        Verifies that calling this method clears the ready event.
    """

    def map_command
    """
    Map a command message to a process method.

    Adds or replaces an entry in ``command_map``. The mapped function must
    accept one argument containing the command data, even if that data is
    ignored.

    Parameters
    ----------
    key : Any
        Command message pulled from the command queue.
    function : callable
        Function to call when ``key`` is received.

    Unit Tests
    ----------
    test_abstract_message_process_map_command
        Confirms that a custom command key can be added to the command map and
        maps to the provided callable.
    """

    property
    def command_queue
    """
    Return the process command queue.

    Returns
    -------
    VerboseMessageQueue
        Queue from which this process receives command ``(message, data)``
        pairs.

    Unit Tests
    ----------
    test_message_process_properties
        Accesses this property to verify it is available.
    """

    property
    def log_file_queue
    """
    Return the log file queue.

    Returns
    -------
    multiprocessing.Queue
        Queue used to send formatted log messages to the logging process.

    Unit Tests
    ----------
    test_message_process_properties
        Accesses this property to verify it is available.

    test_message_process_log
        Verifies that this queue receives formatted log messages.
    """

    def run
    """
    Run the process command loop.

    Continuously receives ``(message, data)`` pairs from ``command_queue`` and
    dispatches each message to the corresponding callable in ``command_map``.
    The command data is passed to the mapped function as its only argument.

    If a mapped function returns a truthy value, the loop logs that the process
    is stopping and exits. If ``shutdown_event`` is supplied and becomes set,
    the loop exits. Undefined commands are logged and ignored. Exceptions
    raised by mapped command functions are logged and sent to the GUI update
    queue as ``UICommands.ERROR`` messages.

    Parameters
    ----------
    shutdown_event : multiprocessing.synchronize.Event or threading.Event, optional
        Event used to signal that the command loop should stop.

    Unit Tests
    ----------
    test_abstract_message_process_run
        Verifies that the command loop dispatches mapped commands, handles
        undefined commands, handles command exceptions, and exits when a mapped
        command returns a truthy halt flag.
    """

    def quit
    """
    Signal the process command loop to stop.

    Returns ``True`` so the ``run`` loop exits after processing the quit
    command.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Returns
    -------
    bool
        Always returns ``True`` to indicate that the process should stop.

    Unit Tests
    ----------
    test_message_process_properties
        Verifies that ``GlobalCommands.QUIT`` is mapped to this method.

    test_abstract_message_process_run
        Verifies that the command loop stops after this method is invoked.
    """

## Acquisition
<!---
MARK: Acquisition
--->
    """
    Controller subsystem that handles reading data from acquisition hardware.

    This module defines the acquisition process used by the controller to
    initialize acquisition hardware, read measured data frames, monitor channel
    limits, synchronize acquired data with output startup data, forward data to
    environment processes, and optionally forward acquired data to the
    streaming process.
    """

### Acquisition Process
<!---
MARK: Acquisition Process
--->
    class AcquisitionProcess
    """
    Message-driven process that manages hardware acquisition.

    ``AcquisitionProcess`` extends ``AbstractMessageProcess`` and implements
    controller-side acquisition behavior. It receives controller commands
    through the acquisition command queue, initializes acquisition hardware,
    tracks per-environment acquisition state, reads data from hardware,
    forwards environment-specific channel subsets to environment data queues,
    sends monitoring data to the GUI, and sends streaming data to the
    streaming process when streaming is active.

    Parameters
    ----------
    process_name : str
        Name of the acquisition process. Used for logging and queue access.
    queue_container : QueueContainer
        Container holding controller, acquisition, streaming, GUI, sync, and
        environment communication queues.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition hardware is currently active.
    streaming_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquired data should be forwarded to the
        streaming process.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the acquisition process is ready.
    ping_alive_event : multiprocessing.synchronize.Event
        Event used to keep watchdog or blocking-timeout logic alive during
        potentially long hardware operations.

    Attributes
    ----------
    queue_container : QueueContainer
        Container of process communication queues.
    startup : bool
        Whether the next acquisition pass should perform hardware startup
        synchronization.
    shutdown_flag : bool
        Whether acquisition shutdown has been requested.
    any_environments_started : bool
        Whether any environment has supplied first-output synchronization data
        during the current acquisition run.
    ping_alive_event : multiprocessing.synchronize.Event
        Event used to keep long-running hardware operations alive.
    sample_rate : int or None
        Acquisition sample rate from hardware metadata.
    read_size : int or None
        Number of samples per acquisition read frame.
    environment_list : list of str
        Queue names of initialized environments.
    environment_acquisition_channels : dict
        Mapping from environment queue name to acquisition channel indices.
    environment_active_flags : dict
        Mapping from environment queue name to whether that environment is
        actively receiving acquisition data.
    environment_last_data : dict
        Mapping from environment queue name to whether final acquisition data
        is still being delivered to that environment.
    environment_samples_remaining_to_read : dict
        Mapping from environment queue name to number of samples remaining
        before final data delivery is complete.
    environment_first_data : dict
        Mapping from environment queue name to first output data used for
        acquisition-output synchronization.
    hardware : HardwareAcquisition or None
        Active hardware acquisition object.
    hardware_metadata : HardwareMetadata or None
        Metadata used to initialize the acquisition hardware.
    has_streamed : bool
        Whether streaming has occurred during the current acquisition session.
    read_data : numpy.ndarray or None
        Rolling acquisition data buffer.
    output_indices : list of int or None
        Channel indices corresponding to output feedback channels.
    abort_limits : numpy.ndarray or None
        Per-channel abort limits.
    warning_limits : numpy.ndarray or None
        Per-channel warning limits.

    Unit Tests
    ----------
    test_acquisition_init
        Verifies that ``AcquisitionProcess`` initializes successfully and is an
        ``AbstractMessageProcess``.

    test_acquisition_properties
        Verifies that acquisition active state can be set and cleared.
    """

    def __init__
    """
    Initialize the acquisition process.

    Initializes the base message process with acquisition queues and maps
    acquisition-related global commands to process methods. Also initializes
    hardware state, environment state dictionaries, streaming flags, persistent
    acquisition buffers, warning and abort limit storage, and acquisition and
    streaming events.

    Parameters
    ----------
    process_name : str
        Name of the acquisition process.
    queue_container : QueueContainer
        Container holding queues used to communicate between controller
        processes.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    streaming_active_event : multiprocessing.synchronize.Event
        Event indicating whether streaming is active.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the acquisition process is ready.
    ping_alive_event : multiprocessing.synchronize.Event
        Event used to keep long hardware operations from being interpreted as
        stalled.

    Unit Tests
    ----------
    test_acquisition_init
        Verifies that initialization creates an ``AcquisitionProcess`` and an
        ``AbstractMessageProcess``.
    """

    property
    def acquisition_active
    """
    Return whether acquisition is active.

    Returns
    -------
    bool
        ``True`` if the acquisition active event is set, otherwise ``False``.

    Unit Tests
    ----------
    test_acquisition_properties
        Verifies that this property reflects the acquisition active event.
    """

    property
    def streaming
    """
    Return whether acquisition streaming is active.

    Returns
    -------
    bool
        ``True`` if the streaming active event is set, otherwise ``False``.

    Unit Tests
    ----------
    test_acqusition_process_start_streaming
        Verifies that starting streaming sets the streaming state.

    test_acquisition_process_stop_streaming
        Verifies that stopping streaming clears the streaming state.
    """

    def set_active
    """
    Set the acquisition active event.

    Marks acquisition as active.

    Unit Tests
    ----------
    test_acquisition_properties
        Verifies that calling this method sets acquisition active state.
    """

    def clear_active
    """
    Clear the acquisition active event.

    Marks acquisition as inactive.

    Unit Tests
    ----------
    test_acquisition_properties
        Verifies that calling this method clears acquisition active state.
    """

    def set_streaming
    """
    Set the streaming active event.

    Marks acquisition streaming as active so acquired data will be forwarded to
    the streaming process.

    Unit Tests
    ----------
    test_acqusition_process_start_streaming
        Verifies that starting streaming sets the streaming state.
    """

    def clear_streaming
    """
    Clear the streaming active event.

    Marks acquisition streaming as inactive so acquired data will no longer be
    forwarded to the streaming process.

    Unit Tests
    ----------
    test_acquisition_process_stop_streaming
        Verifies that stopping streaming clears the streaming state.
    """

    def initialize_hardware
    """
    Initialize acquisition hardware.

    Stores sampling information from hardware metadata, closes any existing
    hardware acquisition object, constructs the appropriate hardware
    acquisition implementation from the hardware registry, initializes the
    hardware, extracts per-channel warning and abort limits, identifies output
    feedback channel indices, allocates the rolling read buffer, stores the
    hardware metadata, and marks the process ready.

    Parameters
    ----------
    metadata : HardwareMetadata
        Hardware metadata containing hardware type, channel list, sampling
        parameters, read and write frame sizes, output oversampling, and
        channel limit information.

    Unit Tests
    ----------
    test_acquisition_process_initialize_hardware
        Verifies that hardware initialization stores sampling parameters,
        creates hardware, initializes channel limit arrays, initializes the
        rolling read buffer, and sets the ready event.
    """

    def initialize_environment
    """
    Initialize per-environment acquisition routing state.

    Configures the acquisition process with environment metadata supplied by
    the controller. For each environment, this method records the environment
    queue name, selected acquisition channel indices, active flag, final-data
    flag, remaining final samples counter, and first-output synchronization
    data placeholder.

    Parameters
    ----------
    metadata_dict : dict of str to EnvironmentMetadata
        Mapping from environment queue names to environment metadata objects.

    Unit Tests
    ----------
    test_acquisition_process_initialize_environment
        Verifies that environment names, acquisition channel indices, active
        flags, final-data flags, remaining sample counters, and first-output
        placeholders are initialized and that the ready event is set.
    """

    def stop_environment
    """
    Deactivate an environment and prepare final data delivery.

    Marks the specified environment as inactive, indicates that final data
    should still be delivered to it, and initializes the number of remaining
    samples to read using the acquisition delay reported by the hardware.

    Parameters
    ----------
    data : str
        Environment queue name to deactivate.

    Unit Tests
    ----------
    test_acquisition_process_stop_environment
        Verifies that the environment is deactivated, final-data state is set,
        and hardware acquisition delay is requested.
    """

    def start_streaming
    """
    Start forwarding acquired data to the streaming process.

    Sets the streaming active event. If data has already been streamed in the
    current acquisition session, this method requests creation of a new stream
    before continuing. Otherwise, it marks that streaming has begun.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_acqusition_process_start_streaming
        Verifies that streaming state is set, ``has_streamed`` is updated, and
        a new stream is requested when streaming had previously occurred.
    """

    def stop_streaming
    """
    Stop forwarding acquired data to the streaming process.

    Clears the streaming active event.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_acquisition_process_stop_streaming
        Verifies that streaming state is cleared.
    """

    def acquire_signal
    """
    Run one acquisition-loop iteration.

    Handles startup synchronization with the output process, starts hardware
    acquisition, reads acquired data, updates the rolling read buffer, checks
    warning and abort limits, sends monitor updates to the GUI, aligns first
    acquired data with first output data for each environment, forwards
    environment-specific channel subsets to environment queues, sends acquired
    data to the streaming process when streaming is active, and schedules the
    next acquisition iteration.

    During shutdown, this method reads remaining hardware data, sends final
    streaming data when needed, stops hardware acquisition, clears active state,
    notifies the GUI that hardware ended, and resets startup and shutdown
    state.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Notes
    -----
    Unit-test documentation for this method is intentionally omitted.
    """

    def add_data_to_buffer
    """
    Add newly acquired data to the rolling read buffer.

    Shifts existing samples toward the beginning of ``read_data`` and writes
    the supplied data at the end of the buffer. Empty data arrays are ignored.

    Parameters
    ----------
    data : numpy.ndarray
        Acquired data array with shape ``(num_channels, num_samples)``.

    Unit Tests
    ----------
    test_add_data_to_buffer
        Verifies that newly acquired data is inserted into the rolling read
        buffer.
    """

    def get_first_output_data
    """
    Retrieve first-output synchronization data from the sync queue.

    Flushes the input-output synchronization queue and stores any
    ``(environment, data)`` pairs as first-output data for the corresponding
    environment. Also marks that at least one environment has started.

    Unit Tests
    ----------
    test_acquisition_process_get_first_output_data
        Verifies that first-output data is read from the sync queue, stored by
        environment name, and logged.
    """

    def stop_acquisition
    """
    Request acquisition shutdown.

    Sets ``shutdown_flag`` so the acquisition loop will begin shutdown once
    active environments have completed and final data delivery conditions are
    satisfied.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_acquisition_process_stop_acquisition
        Verifies that calling this method sets ``shutdown_flag``.
    """

    def quit
    """
    Quit the acquisition process.

    Flushes environment data queues and the acquisition command queue, logs how
    many queued items were removed, closes hardware if it exists, and returns
    ``True`` so the message-process command loop exits.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Returns
    -------
    bool
        Always returns ``True`` to indicate that the acquisition process should
        stop.

    Unit Tests
    ----------
    test_acquisition_process_quit
        Verifies that queues are flushed, hardware is closed, and a flush
        count is logged.
    """

### Acquisition Process Function
<!---
MARK: Acquisition Process Function
--->
    def acquisition_process
    """
    Entry point used to start the acquisition subprocess.

    Constructs an ``AcquisitionProcess`` with the supplied queues and events,
    then runs its message-processing command loop until the shutdown event is
    set or a quit command is received. This function is intended to be used as
    the target of a ``multiprocessing.Process`` or equivalent threaded process.

    Parameters
    ----------
    queue_container : QueueContainer
        Container holding queues used to communicate between controller
        processes.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    streaming_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition streaming is active.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the acquisition process is ready.
    shutdown_event : multiprocessing.synchronize.Event
        Event used to signal that the acquisition process should terminate.
    ping_alive_event : multiprocessing.synchronize.Event
        Event used to keep watchdog or blocking-timeout logic alive during
        long acquisition operations.

    Unit Tests
    ----------
    test_acquisition_process_func
        Verifies that this function constructs an ``AcquisitionProcess`` and
        calls its ``run`` method.
    """

## Output
<!---
MARK: Output
--->
    """
    Controller subsystem that handles output from the hardware to shaker
    amplifiers or other excitation devices.

    This module defines the output process used by the controller to initialize
    output hardware, collect output data from environment processes, combine
    environment output contributions into hardware write frames, synchronize
    output startup with acquisition, write data to hardware, and shut down
    output safely after environments have completed.
    """

### Output Process
<!---
MARK: Output Process
--->
    class OutputProcess
    """
    Message-driven process that manages hardware output.

    ``OutputProcess`` extends ``AbstractMessageProcess`` and implements
    controller-side output behavior. It receives controller commands through
    the output command queue, initializes output hardware, tracks
    per-environment output state, collects output data from environment data
    queues, sums each environment's contribution into the correct hardware
    output channels, writes completed frames to output hardware, and
    coordinates startup and shutdown with acquisition.

    Parameters
    ----------
    process_name : str
        Name of the output process. Used for logging and queue access.
    queue_container : QueueContainer
        Container holding controller, acquisition, output, GUI, sync,
        single-process hardware, and environment communication queues.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output hardware is currently active.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the output process is ready.
    ping_alive_event : multiprocessing.synchronize.Event
        Event used to keep watchdog or blocking-timeout logic alive during
        potentially long hardware operations.

    Attributes
    ----------
    queue_container : QueueContainer
        Container of process communication queues.
    startup : bool
        Whether the next output pass should start hardware output.
    shutdown_flag : bool
        Whether output shutdown has been requested.
    ping_alive_event : multiprocessing.synchronize.Event
        Event used to keep long-running hardware operations alive.
    sample_rate : int or None
        Output sample rate base value from hardware metadata.
    write_size : int or None
        Number of samples per hardware output write frame.
    num_outputs : int or None
        Number of physical output channels.
    output_oversample : int or None
        Output oversampling factor from hardware metadata.
    environment_list : list of str
        Queue names of initialized environments.
    environment_output_channels : dict or None
        Mapping from environment queue name to output channel indices within
        the hardware output frame.
    environment_active_flags : dict
        Mapping from environment queue name to whether that environment is
        actively contributing output data.
    environment_starting_up_flags : dict
        Mapping from environment queue name to whether that environment is
        waiting for enough initial output data to begin.
    environment_shutting_down_flags : dict
        Mapping from environment queue name to whether that environment is
        draining final output data.
    environment_data_out_remainders : dict or None
        Mapping from environment queue name to queued output data that has not
        yet been written to hardware.
    environment_first_data : dict
        Mapping from environment queue name to whether the next write should
        be sent to acquisition for startup synchronization.
    hardware : HardwareOutput or None
        Active hardware output object.
    hardware_metadata : HardwareMetadata or None
        Metadata used to initialize the output hardware.

    Unit Tests
    ----------
    test_output_init
        Verifies that ``OutputProcess`` initializes successfully and is an
        ``AbstractMessageProcess``.

    test_output_properties
        Verifies that output active state can be set and cleared.
    """

    def __init__
    """
    Initialize the output process.

    Initializes the base message process with output queues and maps
    output-related global commands to process methods. Also initializes
    hardware state, environment state dictionaries, output sampling fields,
    shutdown flags, startup flags, and the output active event.

    Parameters
    ----------
    process_name : str
        Name of the output process.
    queue_container : QueueContainer
        Container holding queues used to communicate between controller
        processes.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the output process is ready.
    ping_alive_event : multiprocessing.synchronize.Event
        Event used to keep long hardware operations from being interpreted as
        stalled.

    Unit Tests
    ----------
    test_output_init
        Verifies that initialization creates an ``OutputProcess`` and an
        ``AbstractMessageProcess``.
    """

    property
    def output_active
    """
    Return whether output is active.

    Returns
    -------
    bool
        ``True`` if the output active event is set, otherwise ``False``.

    Unit Tests
    ----------
    test_output_properties
        Verifies that this property reflects the output active event.
    """

    def set_active
    """
    Set the output active event.

    Marks output as active.

    Unit Tests
    ----------
    test_output_properties
        Verifies that calling this method sets output active state.
    """

    def clear_active
    """
    Clear the output active event.

    Marks output as inactive.

    Unit Tests
    ----------
    test_output_properties
        Verifies that calling this method clears output active state.
    """

    def initialize_hardware
    """
    Initialize output hardware.

    Stores sampling information from hardware metadata, closes any existing
    hardware output object, constructs the appropriate hardware output
    implementation from the hardware registry, initializes the hardware, counts
    physical output channels, stores the hardware metadata, and marks the
    process ready.

    Output channels are identified from hardware channels whose feedback device
    field is populated and is not a comment or blank string.

    Parameters
    ----------
    metadata : HardwareMetadata
        Hardware metadata containing hardware type, channel list, sampling
        parameters, write frame size, output oversampling, and output feedback
        channel information.

    Unit Tests
    ----------
    test_output_process_initialize_hardware
        Verifies that hardware initialization stores sampling parameters,
        creates hardware, initializes the hardware object, computes output
        channel count, stores metadata, and sets the ready event.
    """

    def initialize_environment
    """
    Initialize per-environment output routing state.

    Configures the output process with environment metadata supplied by the
    controller. For each environment, this method records the environment
    queue name, initializes active/startup/shutdown flags, initializes
    first-data synchronization flags, determines which hardware output channel
    indices are used by the environment, and creates an empty output remainder
    buffer.

    Parameters
    ----------
    metadata_dict : dict of str to EnvironmentMetadata
        Mapping from environment queue names to environment metadata objects.

    Unit Tests
    ----------
    test_output_process_initialize_environment
        Verifies that environment names, active flags, startup flags, shutdown
        flags, first-data flags, output channel mappings, output remainder
        buffers, and ready state are initialized.
    """

    def output_signal
    """
    Run one output-loop iteration.

    Collects output data from active or starting environments, appends new
    environment data to per-environment remainder buffers, determines when
    enough data is available to write a hardware frame, combines active
    environment outputs into a single hardware output array, writes output data
    to hardware, sends first output data to acquisition for synchronization,
    starts hardware output on the first write, and schedules the next output
    iteration.

    During shutdown, this method waits for environments to finish, drains
    remaining output data, notifies acquisition when environments are stopped,
    stops hardware output when all output has completed, clears active state,
    and resets startup and shutdown flags.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Notes
    -----
    Unit-test documentation for this method is intentionally omitted.
    """

    def stop_output
    """
    Request output shutdown.

    Logs the start of the shutdown procedure and sets ``shutdown_flag`` so the
    output loop will begin shutting down once all active environments have
    drained their remaining data.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_output_process_stop_output
        Verifies that calling this method logs the shutdown request and sets
        ``shutdown_flag``.
    """

    def start_environment
    """
    Mark an environment as starting output.

    Sets the specified environment's startup flag, clears its shutdown flag,
    and ensures it is not yet marked active. The output loop will activate the
    environment after enough initial output data has been received for a
    complete write frame.

    Parameters
    ----------
    data : str
        Environment queue name to activate.

    Unit Tests
    ----------
    test_output_process_start_environment
        Verifies that the environment startup flag is set, shutdown flag is
        cleared, active flag remains false, and the action is logged.
    """

    def quit
    """
    Quit the output process.

    Flushes environment output queues, the output command queue, the
    input-output synchronization queue, and the single-process hardware queue.
    Logs how many queued items were removed, closes hardware if it exists, and
    returns ``True`` so the message-process command loop exits.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Returns
    -------
    bool
        Always returns ``True`` to indicate that the output process should
        stop.

    Unit Tests
    ----------
    test_output_process_quit
        Verifies that queues are flushed, hardware is closed, and a flush
        count is logged.
    """

### Output Process Function
<!---
MARK: Output Process Function
--->
    def output_process
    """
    Entry point used to start the output subprocess.

    Constructs an ``OutputProcess`` with the supplied queues and events, then
    runs its message-processing command loop until the shutdown event is set or
    a quit command is received. This function is intended to be used as the
    target of a ``multiprocessing.Process`` or equivalent threaded process.

    Parameters
    ----------
    queue_container : QueueContainer
        Container holding queues used to communicate between controller
        processes.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the output process is ready.
    shutdown_event : multiprocessing.synchronize.Event
        Event used to signal that the output process should terminate.
    ping_alive_event : multiprocessing.synchronize.Event
        Event used to keep watchdog or blocking-timeout logic alive during
        long output operations.

    Unit Tests
    ----------
    test_output_process_func
        Verifies that this function constructs an ``OutputProcess`` and calls
        its ``run`` method.
    """
## Streaming
<!---
MARK: Streaming
--->
    """
    Controller subsystem that handles streaming data and metadata to NetCDF4
    files on disk.

    This module defines stream configuration metadata and the streaming process
    used by the controller to create netCDF streaming files, save controller
    metadata, append acquisition data to streaming variables, create additional
    stream variables within the same file, and close streaming files when data
    collection is complete.
    """

### Stream Type
<!---
MARK: Stream Type
--->
    class StreamType
    """
    Enumeration of supported streaming modes.

    Streaming modes determine when, or whether, acquisition data should be
    written to disk.

    Attributes
    ----------
    NO_STREAM : int
        Do not stream acquisition data to disk.
    IMMEDIATELY : int
        Begin streaming immediately when acquisition starts.
    PROFILE_INSTRUCTION : int
        Begin streaming based on a profile instruction.
    TEST_LEVEL : int
        Begin streaming when a selected environment reaches a test level
        condition.
    MANUAL : int
        Begin streaming when manually requested by the user.

    Unit Tests
    ----------
    test_stream_type
        Iterates through each enum member to confirm unique integer values.
    """

### Stream Metadata
<!---
MARK: Stream Metadata
--->
    class StreamMetadata
    """
    Metadata defining how acquisition data should be streamed to disk.

    ``StreamMetadata`` stores the selected streaming mode, the target netCDF
    file path, and the environment associated with test-level-triggered
    streaming. It is used by the controller and streaming process to validate
    streaming configuration and initialize streaming files.

    Parameters
    ----------
    stream_type : StreamType, optional
        Streaming mode. Defaults to ``StreamType.NO_STREAM``.
    stream_file : str or pathlib.Path, optional
        Path to the netCDF file used for streaming data. Required for all
        streaming modes except ``StreamType.NO_STREAM``.
    test_level_environment_name : str, optional
        Environment name used to trigger streaming for
        ``StreamType.TEST_LEVEL``.

    Attributes
    ----------
    stream_type : StreamType
        Selected streaming mode.
    stream_file : str or pathlib.Path or None
        Path to the streaming file.
    test_level_environment_name : str or None
        Environment name associated with test-level-triggered streaming.

    Unit Tests
    ----------
    test_stream_metadata_init
        Verifies that stream metadata initializes required attributes.

    test_stream_metadata_validate
        Verifies that valid streaming configurations pass validation and
        invalid file paths or test-level settings raise ``RattlesnakeError``.
    """

    def __init__
    """
    Initialize stream metadata.

    Stores the streaming mode, output file path, and optional test-level
    environment name.

    Parameters
    ----------
    stream_type : StreamType, optional
        Streaming mode. Defaults to ``StreamType.NO_STREAM``.
    stream_file : str or pathlib.Path, optional
        Path to the netCDF streaming file.
    test_level_environment_name : str, optional
        Environment name used for ``StreamType.TEST_LEVEL`` streaming.

    Unit Tests
    ----------
    test_stream_metadata_init
        Confirms that initialization stores ``stream_type``, ``stream_file``,
        and ``test_level_environment_name``.
    """

    def validate
    """
    Validate stream metadata.

    Ensures that streaming configurations are internally consistent. If
    streaming is enabled, a valid stream file path must be provided and the
    parent directory must exist. If streaming is configured to start based on
    test level, a valid environment name must be supplied.

    Raises
    ------
    RattlesnakeError
        If streaming is enabled but ``stream_file`` is missing or is not a
        string or ``Path``.
    RattlesnakeError
        If the parent directory of ``stream_file`` does not exist.
    RattlesnakeError
        If ``stream_type`` is ``StreamType.TEST_LEVEL`` and no valid test-level
        environment name is supplied.

    Unit Tests
    ----------
    test_stream_metadata_validate
        Verifies validation behavior for manual, no-stream, immediate, and
        test-level streaming configurations.
    """

### Streaming Process
<!---
MARK: Streaming Process
--->
    class StreamingProcess
    """
    Message-driven process that writes acquisition data to netCDF files.

    ``StreamingProcess`` extends ``AbstractMessageProcess`` and implements the
    controller-side streaming behavior. It receives streaming commands through
    the streaming command queue, creates streaming files, stores hardware and
    environment metadata, writes incoming acquisition data to the active netCDF
    variable, creates new stream variables when requested, and finalizes files
    when streaming is complete.

    Parameters
    ----------
    process_name : str
        Name of the streaming process. Used for logging and queue access.
    queue_container : QueueContainer
        Container holding controller, streaming, GUI, and logging queues.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the streaming process is ready.

    Attributes
    ----------
    netcdf_handle : nc.Dataset or None
        Open netCDF dataset used for streaming, or ``None`` when streaming is
        not initialized.
    stream_variable : str
        Name of the active netCDF variable receiving acquisition data.
    stream_dimension : str
        Name of the active unlimited time dimension.
    stream_index : int
        Index used to name additional streaming variables and dimensions.

    Unit Tests
    ----------
    test_streaming_init
        Verifies that ``StreamingProcess`` initializes successfully and is an
        ``AbstractMessageProcess``.
    """

    def __init__
    """
    Initialize the streaming process.

    Initializes the base message process with streaming queues and maps
    streaming-related global commands to process methods. Also initializes the
    netCDF handle and default active streaming variable, dimension, and stream
    index.

    Parameters
    ----------
    process_name : str
        Name of the streaming process.
    queue_container : QueueContainer
        Container holding queues used to communicate between controller
        processes.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the streaming process is ready.

    Unit Tests
    ----------
    test_streaming_init
        Verifies that initialization creates a ``StreamingProcess`` and an
        ``AbstractMessageProcess``.
    """

    def initialize
    """
    Initialize streaming output.

    Creates a netCDF streaming file and writes controller metadata to it. The
    input data must contain stream metadata, hardware metadata, and a mapping
    of environment names to environment metadata. If the stream type is
    ``StreamType.NO_STREAM``, no file is created and the process is simply
    marked ready.

    Parameters
    ----------
    data : tuple
        Tuple ``(stream_metadata, hardware_metadata, environment_metadata_dict)``
        where ``stream_metadata`` is a ``StreamMetadata`` object,
        ``hardware_metadata`` is a ``HardwareMetadata`` object, and
        ``environment_metadata_dict`` is a dictionary mapping environment names
        to ``EnvironmentMetadata`` objects.

    Unit Tests
    ----------
    test_streaming_process_initialize
        Verifies that streaming initialization creates a netCDF dataset and
        saves metadata for streaming modes other than ``NO_STREAM``, and does
        not create a file for ``NO_STREAM``.
    """

    def write_data
    """
    Write acquisition data to the active stream variable.

    Appends the supplied data to the currently active netCDF variable using the
    current unlimited time dimension. If streaming has not been initialized,
    the method returns without writing.

    Parameters
    ----------
    data : numpy.ndarray
        Acquisition data to write. Expected shape is typically
        ``(response_channels, time_samples)``.

    Unit Tests
    ----------
    test_streaming_process_write_data
        Verifies that data is written to the active netCDF variable at the end
        of the current time dimension.

    test_streaming_process_write_data_no_init
        Verifies that calling this method without an open netCDF handle returns
        without error.
    """

    def create_new_stream
    """
    Create a new stream variable in the current netCDF file.

    Increments the stream index and creates a new unlimited time dimension and
    corresponding ``time_data`` variable. This allows multiple stream segments
    to be stored in the same netCDF file. If streaming has not been
    initialized, the method returns without creating anything.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_streaming_process_create_new_stream
        Verifies that a new time dimension and data variable are created.

    test_streaming_process_create_new_stream_no_netcdf
        Verifies that calling this method without an open netCDF handle returns
        without error.
    """

    def finalize
    """
    Finalize streaming and close the netCDF file.

    Closes the active netCDF dataset if one is open and clears the stored
    netCDF handle.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_streaming_process_finalize
        Verifies that an open netCDF handle is closed and cleared.
    """

    def quit
    """
    Quit the streaming process.

    Finalizes any open netCDF file and returns ``True`` so the
    message-process command loop exits.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Returns
    -------
    bool
        Always returns ``True`` to indicate that the streaming process should
        stop.

    Unit Tests
    ----------
    test_streaming_process_quit
        Verifies that ``finalize`` is called and that this method returns
        ``True``.
    """

### Streaming Process Function
<!---
MARK: Streaming Process Function
--->
    def streaming_process
    """
    Entry point used to start the streaming subprocess.

    Constructs a ``StreamingProcess`` with the supplied queues and ready event,
    then runs its message-processing command loop until the shutdown event is
    set or a quit command is received. This function is intended to be used as
    the target of a ``multiprocessing.Process`` or equivalent threaded process.

    Parameters
    ----------
    queue_container : QueueContainer
        Container holding queues used to communicate between controller
        processes.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the streaming process is ready.
    shutdown_event : multiprocessing.synchronize.Event
        Event used to signal that the streaming process should terminate.

    Unit Tests
    ----------
    test_output_process_func
        Verifies that this function constructs a ``StreamingProcess`` and calls
        its ``run`` method.
    """

## Controller
<!---
MARK: Controller
--->
    """
    Controller subsystem that coordinates hardware, environments, system
    identification, and streaming.

    This module defines the controller process used to route high-level
    commands to acquisition, output, streaming, and environment processes. The
    controller receives commands from the controller command queue and performs
    coordinated actions such as starting and stopping hardware, starting and
    stopping environments, starting and stopping system identification, and
    triggering streaming modes.
    """

### Controller Process
<!---
MARK: Controller Process
--->
    class ControllerProcess
    """
    Message-driven process that coordinates controller-wide actions.

    ``ControllerProcess`` extends ``AbstractMessageProcess`` and implements
    the command-routing behavior for high-level Rattlesnake operations. It
    receives commands from the controller command queue and sends the
    appropriate commands to acquisition, output, streaming, and environment
    queues.

    This class is intended to centralize actions that would otherwise require
    multiple manual UI operations. For example, starting hardware requires
    commands to both acquisition and output; stopping hardware may require
    stopping active environments, system identification, and streaming before
    stopping acquisition and output.

    Parameters
    ----------
    process_name : str
        Name of the controller process. Used for logging and queue access.
    queue_container : QueueContainer
        Container holding controller, acquisition, output, streaming, GUI, and
        environment communication queues.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    streaming_active_event : multiprocessing.synchronize.Event
        Event indicating whether streaming is active.
    environment_active_event : dict of str to multiprocessing.synchronize.Event
        Mapping from environment queue names to events indicating whether each
        environment is active.
    environment_sysid_active_event : dict of str to multiprocessing.synchronize.Event
        Mapping from environment queue names to events indicating whether
        system identification is active for each environment.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the controller process is ready.

    Attributes
    ----------
    queue_container : QueueContainer
        Container of process communication queues.
    stream_metadata : StreamMetadata
        Current streaming configuration used when starting, stopping, or
        triggering streaming.
    acquisition_active : bool
        Whether acquisition is currently active.
    output_active : bool
        Whether output is currently active.
    streaming_active : bool
        Whether streaming is currently active.
    environments_active : list of str
        Environment queue names whose active events are set.
    environments_sysid_active : list of str
        Environment queue names whose system identification active events are
        set.

    Unit Tests
    ----------
    test_controller_init
        Verifies that ``ControllerProcess`` initializes successfully and is an
        ``AbstractMessageProcess``.
    """

    def __init__
    """
    Initialize the controller process.

    Initializes the base message process with controller queues, stores
    process activity events, initializes default stream metadata, and maps
    controller-level global commands to controller methods.

    Parameters
    ----------
    process_name : str
        Name of the controller process.
    queue_container : QueueContainer
        Container holding queues used to communicate between controller
        processes.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    streaming_active_event : multiprocessing.synchronize.Event
        Event indicating whether streaming is active.
    environment_active_event : dict of str to multiprocessing.synchronize.Event
        Mapping from environment queue names to environment active events.
    environment_sysid_active_event : dict of str to multiprocessing.synchronize.Event
        Mapping from environment queue names to system identification active
        events.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the controller process is ready.

    Unit Tests
    ----------
    test_controller_init
        Verifies that initialization creates a ``ControllerProcess`` and an
        ``AbstractMessageProcess``.
    """

    property
    def acquisition_active
    """
    Return whether acquisition is active.

    Returns
    -------
    bool
        ``True`` if the acquisition active event is set, otherwise ``False``.

    Unit Tests
    ----------
    test_controller_run_hardware
        Uses this property to verify that hardware cannot be started while
        acquisition is active.

    test_controller_stop_hardware
        Uses this property to verify that hardware cannot be stopped unless
        acquisition is active.
    """

    property
    def output_active
    """
    Return whether output is active.

    Returns
    -------
    bool
        ``True`` if the output active event is set, otherwise ``False``.

    Unit Tests
    ----------
    test_controller_run_hardware
        Uses this property to verify that hardware cannot be started while
        output is active.

    test_controller_stop_hardware
        Uses this property to verify that hardware cannot be stopped unless
        output is active.
    """

    property
    def streaming_active
    """
    Return whether streaming is active.

    Returns
    -------
    bool
        ``True`` if the streaming active event is set, otherwise ``False``.

    Unit Tests
    ----------
    test_controller_stop_hardware
        Uses this property to determine whether streaming should be stopped
        during hardware shutdown.
    """

    property
    def environments_active
    """
    Return names of environments that are currently active.

    Iterates through the environment active event dictionary and returns the
    queue names whose active events are set.

    Returns
    -------
    list of str
        Queue names of active environments.

    Unit Tests
    ----------
    test_controller_stop_hardware
        Uses this property to stop active environments during hardware
        shutdown.

    test_controller_start_environment
        Uses this property to reject attempts to start an already active
        environment.

    test_controller_stop_environment
        Uses this property to reject attempts to stop an inactive environment.
    """

    property
    def environments_sysid_active
    """
    Return names of environments with active system identification.

    Iterates through the system identification active event dictionary and
    returns the queue names whose system identification active events are set.

    Returns
    -------
    list of str
        Queue names of environments currently running system identification.

    Unit Tests
    ----------
    test_controller_stop_hardware
        Uses this property to stop active system identification during hardware
        shutdown.
    """

    def run_hardware
    """
    Start acquisition and output hardware.

    Stores the supplied streaming metadata, verifies that acquisition and
    output are not already active, sends run commands to acquisition and output
    processes, and starts streaming immediately when the stream mode is
    ``StreamType.IMMEDIATELY``.

    Parameters
    ----------
    data : StreamMetadata
        Streaming configuration to use while hardware is running.

    Raises
    ------
    RuntimeError
        If acquisition is already active.
    RuntimeError
        If output is already active.

    Unit Tests
    ----------
    test_controller_run_hardware
        Verifies that acquisition and output run commands are sent and that
        immediate streaming is started for ``StreamType.IMMEDIATELY``.

    test_controller_run_hardware_error
        Verifies that starting hardware raises errors when acquisition or
        output is already active.
    """

    def stop_hardware
    """
    Stop system identification, environments, streaming, acquisition, and
    output hardware.

    Stops active system identification tasks, stops active environments, stops
    streaming if streaming is active, finalizes the streaming file if
    streaming was configured, sends stop commands to acquisition and output,
    and verifies that acquisition and output were active when stop was
    requested.

    Parameters
    ----------
    data : None, optional
        Command data supplied through the command queue. This value is ignored.

    Raises
    ------
    RuntimeError
        If acquisition is not active when hardware stop is requested.
    RuntimeError
        If output is not active when hardware stop is requested.

    Unit Tests
    ----------
    test_controller_stop_hardware
        Verifies that active environments are stopped, acquisition and output
        stop commands are sent, and errors are raised when acquisition or
        output is inactive.
    """

    def start_environment
    """
    Start an environment.

    Sends a start-environment command to the output process so output data will
    be collected for the environment, then sends the start command and
    instructions to the environment command queue.

    Parameters
    ----------
    data : tuple
        Tuple ``(queue_name, instruction)`` where ``queue_name`` is the
        environment queue name and ``instruction`` is an
        ``EnvironmentInstructions`` object or environment-specific instruction
        payload.

    Raises
    ------
    RuntimeError
        If the requested environment is already active.

    Unit Tests
    ----------
    test_controller_start_environment
        Verifies that output and environment start commands are sent, and that
        starting an already active environment raises ``RuntimeError``.
    """

    def stop_environment
    """
    Stop an active environment.

    Sends a stop-environment command to the specified environment command
    queue. The output and acquisition processes are notified later as the
    environment and output drain remaining data.

    Parameters
    ----------
    data : str
        Environment queue name to stop.

    Raises
    ------
    RuntimeError
        If the requested environment is not active.

    Unit Tests
    ----------
    test_controller_stop_environment
        Verifies that the environment stop command is sent for active
        environments and that stopping an inactive environment raises
        ``RuntimeError``.
    """

    def send_environment_command
    """
    Send an arbitrary command to an environment.

    Forwards a command and associated command data to the specified
    environment command queue. This is used for environment-specific commands
    that are not otherwise represented by controller helper methods.

    Parameters
    ----------
    data : tuple
        Tuple ``(queue_name, command, command_data)`` where ``queue_name`` is
        the destination environment queue name, ``command`` is the command to
        send, and ``command_data`` is the payload associated with the command.

    Unit Tests
    ----------
    test_controller_send_environment_command
        Verifies that the command and payload are forwarded to the requested
        environment command queue.
    """

    def start_system_id_noise
    """
    Start the noise-measurement phase of system identification.

    Commands the output process to start routing output for the environment,
    then sends the system identification noise-start command to the
    environment.

    Parameters
    ----------
    data : str
        Environment queue name whose system identification noise measurement
        should start.

    Unit Tests
    ----------
    test_controller_start_system_id_noise
        Verifies that output receives a start-environment command and the
        environment receives a start-system-identification-noise command.
    """

    def start_system_id_transfer
    """
    Start the transfer-function phase of system identification.

    Commands the output process to start routing output for the environment,
    then sends the system identification transfer-start command to the
    environment.

    Parameters
    ----------
    data : str
        Environment queue name whose system identification transfer-function
        measurement should start.

    Unit Tests
    ----------
    test_controller_start_system_id_transfer
        Verifies that output receives a start-environment command and the
        environment receives a start-system-identification-transfer command.
    """

    def stop_system_id
    """
    Stop system identification for an environment.

    Sends a stop-system-identification command to the specified environment
    command queue. The command payload indicates that associated system
    identification tasks should also stop.

    Parameters
    ----------
    data : str
        Environment queue name whose system identification should stop.

    Unit Tests
    ----------
    test_controller_stop_system_id
        Verifies that the stop-system-identification command is forwarded to
        the requested environment.
    """

    def start_streaming
    """
    Start streaming acquisition data.

    Starts streaming by sending a start-streaming command to the acquisition
    process when either ``override`` is truthy or the configured stream type is
    ``StreamType.PROFILE_INSTRUCTION``. The override allows the controller to
    start streaming for modes such as immediate, manual, or test-level
    streaming without requiring the stream type to be profile-instruction
    based.

    Parameters
    ----------
    data : bool, optional
        Override flag. If truthy, streaming starts regardless of configured
        stream type. Defaults to ``False``.

    Unit Tests
    ----------
    test_controller_start_streaming
        Verifies that streaming starts when override is truthy or when stream
        type is ``StreamType.PROFILE_INSTRUCTION``.
    """

    def stop_streaming
    """
    Stop streaming acquisition data.

    Sends a stop-streaming command to the acquisition process.

    Parameters
    ----------
    data : None, optional
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_controller_stop_streaming
        Verifies that a stop-streaming command is sent to acquisition.
    """

    def at_target_level
    """
    Handle notification that an environment reached target level.

    Starts streaming if the configured stream type is ``StreamType.TEST_LEVEL``
    and the reported environment name matches the stream metadata's selected
    test-level environment.

    Parameters
    ----------
    data : str
        Environment name that reached target level.

    Unit Tests
    ----------
    test_controller_at_target_level_match
        Verifies that streaming starts only when stream type is
        ``StreamType.TEST_LEVEL`` and the environment name matches.
    """

    def manual_stream
    """
    Start streaming in response to a manual streaming request.

    Starts streaming if the configured stream type is ``StreamType.MANUAL``.

    Parameters
    ----------
    data : None, optional
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_controller_manual_stream
        Verifies that manual streaming starts only when stream type is
        ``StreamType.MANUAL``.
    """

    def profile_closeout
    """
    Mark the controller ready after profile closeout.

    This method is called when profile execution has completed closeout
    activities. It sets the controller ready event so additional operations can
    proceed.

    Parameters
    ----------
    data : None
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_controller_profile_closeout
        Verifies that calling this method sets the ready event.
    """

### Controller Process Function
<!---
MARK: Controller Process Function
--->
    def controller_process
    """
    Entry point used to start the controller subprocess.

    Constructs a ``ControllerProcess`` with the supplied queues and events,
    then runs its message-processing command loop until the shutdown event is
    set or a quit command is received. This function is intended to be used as
    the target of a ``multiprocessing.Process`` or equivalent threaded process.

    Parameters
    ----------
    queue_container : QueueContainer
        Container holding queues used to communicate between controller
        processes.
    acquisition_active_event : multiprocessing.synchronize.Event
        Event indicating whether acquisition is active.
    output_active_event : multiprocessing.synchronize.Event
        Event indicating whether output is active.
    streaming_active_event : multiprocessing.synchronize.Event
        Event indicating whether streaming is active.
    environment_active_event : dict of str to multiprocessing.synchronize.Event
        Mapping from environment queue names to environment active events.
    environment_sysid_active_event : dict of str to multiprocessing.synchronize.Event
        Mapping from environment queue names to system identification active
        events.
    ready_event : multiprocessing.synchronize.Event
        Event indicating whether the controller process is ready.
    shutdown_event : multiprocessing.synchronize.Event
        Event used to signal that the controller process should terminate.

    Unit Tests
    ----------
    test_controller_process_func
        Verifies that this function constructs a ``ControllerProcess`` and
        calls its ``run`` method.
    """

## Data Collector
<!---
MARK: Data Collector
--->
    """
    Data collection subprocess for frame-based signal processing.

    This module defines utilities and process classes that collect acquired
    time data, store it in a rolling frame buffer, identify measurement frames
    using free-run or trigger-based acquisition, apply acceptance logic,
    transform response and reference channels, apply window functions, compute
    FFTs, report time frames and kurtosis to the GUI, and forward spectral
    frames to downstream processing queues.
    """

### Frame Buffer
<!---
MARK: Frame Buffer
--->
    class FrameBuffer
    """
    Rolling acquisition buffer used to extract measurement frames.

    ``FrameBuffer`` stores recently acquired time data and identifies frame
    start locations using either fixed spacing or trigger logic. It supports
    pretrigger data, positive or negative trigger slopes, hysteresis reset,
    manual frame acceptance, first-trigger-only acquisition, waiting a fixed
    number of samples before frames are returned, and overlapping frames.

    Parameters
    ----------
    num_channels : int
        Number of channels stored in the buffer.
    trigger_index : int
        Channel index used for trigger detection.
    pretrigger : float
        Fraction of the frame to include before the trigger.
    positive_slope : bool
        If ``True``, detect triggers on positive-going threshold crossings. If
        ``False``, detect negative-going crossings.
    trigger_level : float
        Trigger threshold level.
    hysteresis_level : float
        Level that must be crossed to reset the trigger.
    hysteresis_samples : int
        Number of samples that must satisfy the hysteresis reset condition.
    samples_per_frame : int
        Number of samples in each measurement frame.
    maximum_overlap : float
        Fractional frame overlap. Determines spacing between free-run frames.
    manual_accept : bool
        If ``True``, return one frame and wait for explicit acceptance before
        returning another.
    trigger_enabled : bool
        If ``True``, frame extraction is trigger-based. If ``False``, frames
        are returned using fixed spacing.
    trigger_only_first : bool
        If ``True``, only the first frame requires a trigger and subsequent
        frames use fixed spacing.
    wait_samples : int
        Number of samples to wait before returning frames.
    dtype : str or numpy.dtype, optional
        Buffer data type. Defaults to ``"float64"``.
    starting_value : scalar, optional
        Initial buffer fill value. Defaults to ``numpy.nan``.
    buffer_size_frame_multiplier : int, optional
        Buffer length multiplier relative to frame size. Defaults to ``2``.

    Attributes
    ----------
    samples_per_frame : int
        Number of samples per returned frame.
    trigger_index : int
        Trigger channel index.
    pretrigger_samples : int
        Number of pretrigger samples.
    positive_slope : bool
        Trigger slope direction flag.
    trigger_level : float
        Trigger threshold.
    hysteresis_level : float
        Hysteresis reset threshold.
    hysteresis_samples : int
        Number of hysteresis samples required for reset.
    overlap_samples : int
        Minimum spacing between returned frame triggers.
    manual_accept : bool
        Whether manual frame acceptance is enabled.
    waiting_for_accept : bool
        Whether the buffer is waiting for manual acceptance.
    buffer_size_frame_multiplier : int
        Buffer length multiplier relative to frame size.
    wait_samples : int
        Number of samples to wait before returning frames.
    last_trigger : int
        Location of most recent accepted trigger relative to the rolling
        buffer.
    last_reset : int
        Location of most recent trigger reset relative to the rolling buffer.
    trigger_enabled : bool
        Whether trigger-based acquisition is enabled.
    trigger_only_first : bool
        Whether only the first frame requires a trigger.
    first_trigger : bool
        Whether the first trigger has not yet occurred.

    Unit Tests
    ----------
    test_frame_buffer_init
        Verifies that a frame buffer initializes with the expected shape and
        starting values.

    test_frame_buffer_add_data
        Verifies that added data shifts into the buffer correctly.
    """

    def __init__
    """
    Initialize the frame buffer.

    Creates the rolling acquisition buffer, computes pretrigger and overlap
    sample counts, initializes trigger and reset counters, stores trigger and
    acceptance configuration, and fills the buffer with the requested starting
    value.

    Unit Tests
    ----------
    test_frame_buffer_init
        Confirms that initialization creates a ``FrameBuffer`` with expected
        buffer contents.
    """

    property
    def buffer_data
    """
    Return the current rolling buffer data.

    Returns
    -------
    numpy.ndarray
        Buffer array with shape ``(num_channels, buffer_samples)``.

    Unit Tests
    ----------
    test_frame_buffer_init
        Verifies initial buffer contents.

    test_frame_buffer_add_data
        Verifies buffer contents after data are added.
    """

    def add_data
    """
    Add acquired data to the rolling buffer.

    Increments trigger and reset counters by the number of new samples,
    truncates input data to the buffer length if necessary, and shifts the new
    data into the end of the rolling buffer.

    Parameters
    ----------
    data : numpy.ndarray
        New acquired data with shape ``(num_channels, samples)``.

    Unit Tests
    ----------
    test_frame_buffer_add_data
        Verifies that added data replaces the rolling buffer contents as
        expected.
    """

    def find_triggers
    """
    Find frame trigger locations in the current buffer.

    Detects trigger locations using threshold crossings and hysteresis reset
    when trigger-based acquisition is enabled. When triggering is disabled, it
    returns frame locations based on fixed overlap spacing. If manual
    acceptance is enabled, only one trigger is returned and the buffer enters a
    waiting-for-accept state.

    Returns
    -------
    list of int
        Trigger offsets used to extract measurement frames.

    Unit Tests
    ----------
    test_frame_buffer_find_triggers
        Verifies positive and negative slope trigger detection, missing-trigger
        behavior, and manual acceptance waiting behavior.
    """

    def reset_trigger
    """
    Reset trigger tracking state.

    Restores ``last_trigger`` and ``last_reset`` to their initial positions
    based on overlap and wait samples.

    Unit Tests
    ----------
    test_frame_buffer_reset_trigger
        Verifies that trigger and reset counters are restored to expected
        values.
    """

    def accept
    """
    Accept the most recent manually reviewed frame.

    Resets trigger and reset counters and clears the
    ``waiting_for_accept`` flag.

    Unit Tests
    ----------
    test_frame_buffer_accept
        Verifies that acceptance resets counters and clears waiting state.
    """

    def add_data_get_frame
    """
    Add data and return newly available measurement frames.

    Adds new data to the buffer, finds trigger locations, extracts one frame
    per trigger, applies pretrigger offset, and returns frames with frame index
    as the first dimension.

    Parameters
    ----------
    data : numpy.ndarray
        New acquired data.

    Returns
    -------
    numpy.ndarray
        Array of extracted frames with shape
        ``(num_frames, num_channels, samples_per_frame)``.

    Unit Tests
    ----------
    test_frame_buffer_add_data_get_frame
        Verifies that data are added, triggers are queried, and expected frame
        data are extracted from the buffer.
    """

    def __getitem__
    """
    Return values from the underlying buffer.

    Parameters
    ----------
    key : Any
        NumPy indexing key.

    Returns
    -------
    Any
        Indexed buffer data.

    Unit Tests
    ----------
    test_frame_buffer_get_item
        Verifies indexed buffer access.
    """

    def __setitem__
    """
    Set values in the underlying buffer.

    Parameters
    ----------
    key : Any
        NumPy indexing key.
    val : Any
        Value to assign.

    Unit Tests
    ----------
    test_frame_buffer_set_item
        Verifies indexed buffer assignment.
    """

### Kurtosis Buffer
<!---
MARK: Kurtosis Buffer
--->
    class KurtosisBuffer
    """
    Running buffer for channel-wise kurtosis calculation.

    ``KurtosisBuffer`` stores moment sums for a fixed number of recent frames
    and computes kurtosis efficiently from accumulated raw moments.

    Parameters
    ----------
    n_channels : int
        Number of channels for which kurtosis is tracked.
    averages : int, optional
        Number of frames retained for the running kurtosis calculation.
        Defaults to ``100``.

    Attributes
    ----------
    idx : int
        Current circular buffer index.
    averages : int
        Number of frame statistics retained.
    g0 : numpy.ndarray
        Stored sample counts per frame.
    g1 : numpy.ndarray
        Stored first raw moment sums.
    g2 : numpy.ndarray
        Stored second raw moment sums.
    g3 : numpy.ndarray
        Stored third raw moment sums.
    g4 : numpy.ndarray
        Stored fourth raw moment sums.
    """

    def __init__
    """
    Initialize the kurtosis buffer.

    Allocates circular buffers for sample count and first through fourth raw
    moment sums for each channel.
    """

    def clear
    """
    Clear the kurtosis buffer.

    Resets the circular index and all stored moment sums to zero.
    """

    def add_data
    """
    Add a frame of data to the kurtosis buffer.

    Computes raw moment sums of the supplied data along the specified axis and
    stores them at the current circular buffer index.

    Parameters
    ----------
    arr : numpy.ndarray
        Frame data to add.
    axis : int, optional
        Axis along which moments are computed. Defaults to ``-1``.
    """

    def get_kurtosis
    """
    Return current channel-wise kurtosis values.

    Combines stored raw moments, converts them to central moments, and computes
    kurtosis for each channel.

    Parameters
    ----------
    fisher : bool, optional
        If ``True``, return excess kurtosis by subtracting 3. If ``False``,
        return standard kurtosis. Defaults to ``False``.

    Returns
    -------
    numpy.ndarray
        Kurtosis values for each channel.
    """

### Data Collector Commands
<!---
MARK: Data Collector Commands
--->
    class DataCollectorCommands
    """
    Commands accepted by the data collector process.

    Attributes
    ----------
    INITIALIZE_COLLECTOR : int
        Initialize collector if metadata has changed.
    FORCE_INITIALIZE_COLLECTOR : int
        Initialize collector even if metadata appears unchanged.
    ACQUIRE : int
        Acquire data and process available frames.
    STOP : int
        Stop data collection.
    ACCEPT : int
        Accept or reject a manually reviewed frame.
    SET_TEST_LEVEL : int
        Set current test level and skip-frame count.
    ACCEPTED : int
        Notify environment that manual frame acceptance completed.
    SHUTDOWN_ACHIEVED : int
        Notify environment that collector shutdown completed.
    CLEAR_KURTOSIS_BUFFER : int
        Clear the running kurtosis buffer.

    Unit Tests
    ----------
    test_data_collector_commands
        Verifies that command enum values construct valid
        ``DataCollectorCommands`` members.
    """

    class DataCollectorUICommands
    """
    Commands sent from the data collector to the GUI.

    Attributes
    ----------
    TIME_FRAME : int
        GUI update containing a time frame and acceptance state.
    KURTOSIS : int
        GUI update containing current kurtosis values.
    """

    class AcquisitionType
    """
    Enumeration of supported data acquisition trigger strategies.

    Attributes
    ----------
    FREE_RUN : int
        Frames are returned based on fixed spacing without trigger detection.
    TRIGGER_EVERY_FRAME : int
        Every returned frame requires a trigger.
    TRIGGER_FIRST_FRAME : int
        Only the first returned frame requires a trigger; subsequent frames use
        fixed spacing.

    Unit Tests
    ----------
    test_acqusition_type
        Verifies that enum values construct valid ``AcquisitionType`` members.
    """

    class Acceptance
    """
    Enumeration of frame acceptance strategies.

    Attributes
    ----------
    MANUAL : int
        Frames require explicit user or environment acceptance.
    AUTOMATIC : int
        Frames are accepted automatically according to the configured
        acceptance function.

    Unit Tests
    ----------
    test_acceptance
        Verifies that enum values construct valid ``Acceptance`` members.
    """

    class TriggerSlope
    """
    Enumeration of supported trigger slope directions.

    Attributes
    ----------
    POSITIVE : int
        Trigger on positive-going threshold crossings.
    NEGATIVE : int
        Trigger on negative-going threshold crossings.

    Unit Tests
    ----------
    test_trigger_slope
        Verifies that enum values construct valid ``TriggerSlope`` members.
    """

    class Window
    """
    Enumeration of supported data collector window functions.

    Attributes
    ----------
    RECTANGLE : int
        Rectangular window.
    HANN : int
        Hann window.
    HAMMING : int
        Hamming window.
    FLATTOP : int
        Flattop window.
    TUKEY : int
        Tukey window.
    BLACKMANHARRIS : int
        Blackman-Harris window.
    EXPONENTIAL : int
        Exponential response/reference window.
    EXPONENTIAL_FORCE : int
        Exponential force window with reference pulse truncation.

    Unit Tests
    ----------
    test_window
        Verifies that enum values construct valid ``Window`` members.
    """

### Collector Metadata
<!---
MARK: Collector Metadata
--->
    class CollectorMetadata
    """
    Metadata defining data collector behavior.

    ``CollectorMetadata`` stores channel definitions, acquisition trigger
    strategy, frame acceptance strategy, overlap, trigger settings, frame size,
    window settings, wait samples, optional kurtosis buffer length, and optional
    response/reference transformation matrices.

    Parameters
    ----------
    num_channels : int
        Total number of acquisition channels.
    response_channel_indices : list or numpy.ndarray
        Indices of response channels.
    reference_channel_indices : list or numpy.ndarray
        Indices of reference channels.
    acquisition_type : AcquisitionType
        Frame acquisition strategy.
    acceptance : Acceptance
        Frame acceptance strategy.
    acceptance_function : tuple or None
        Optional ``(module_path, function_name)`` tuple used for automatic
        frame acceptance.
    overlap_fraction : float
        Fractional overlap between successive frames.
    trigger_channel_index : int
        Channel index used for triggering.
    trigger_slope : TriggerSlope
        Trigger slope direction.
    trigger_level : float
        Trigger threshold.
    trigger_hysteresis : float
        Hysteresis reset threshold.
    trigger_hysteresis_samples : int
        Number of samples required for hysteresis reset.
    pretrigger_fraction : float
        Fraction of the frame to include before trigger.
    frame_size : int
        Number of samples per frame.
    window : Window
        Window function to apply before FFT.
    window_parameter_1 : float, optional
        First optional window parameter.
    window_parameter_2 : float, optional
        Second optional window parameter.
    window_parameter_3 : float, optional
        Third optional window parameter.
    wait_samples : int, optional
        Number of samples to wait before returning frames.
    kurtosis_buffer_length : int, optional
        Number of frames retained for kurtosis calculation.
    response_transformation_matrix : numpy.ndarray, optional
        Transformation applied to response channels.
    reference_transformation_matrix : numpy.ndarray, optional
        Transformation applied to reference channels.

    Unit Tests
    ----------
    test_data_collector_metadata_init
        Verifies that metadata initializes successfully.

    test_data_collector_metadata_eq
        Verifies equality comparison for equivalent metadata objects.
    """

    def __init__
    """
    Initialize collector metadata.

    Stores all metadata required to initialize frame buffering, trigger logic,
    acceptance logic, windowing, kurtosis tracking, and channel transformations.
    """

    def __eq__
    """
    Compare collector metadata objects for equality.

    Parameters
    ----------
    other : object
        Object to compare against.

    Returns
    -------
    bool
        ``True`` if all stored fields compare equal, otherwise ``False``.

    Unit Tests
    ----------
    test_data_collector_metadata_eq
        Verifies that a deep-copied metadata object compares equal to the
        original.
    """

### Data Collector Process
<!---
MARK: Data Collector Process
--->
    class DataCollectorProcess
    """
    Message-driven process that converts acquired time data into spectral
    frames.

    ``DataCollectorProcess`` receives acquisition data, extracts frames using a
    ``FrameBuffer``, applies frame acceptance logic, separates response and
    reference channels, applies optional transformation matrices, windows the
    frames, computes FFTs, sends accepted frames to output queues, sends time
    frames and kurtosis updates to the GUI, and notifies the environment when
    shutdown or manual acceptance completes.

    Parameters
    ----------
    process_name : str
        Name of the collector process used for logging and queue access.
    command_queue : VerboseMessageQueue
        Queue from which collector commands are received.
    data_in_queue : multiprocessing.Queue
        Queue from which acquired time data are received.
    data_out_queues : list of multiprocessing.Queue
        Queues where processed response/reference FFT frames are sent.
    environment_command_queue : VerboseMessageQueue
        Queue used to notify the environment of collector status.
    log_file_queue : multiprocessing.Queue
        Queue used to send log messages.
    gui_update_queue : multiprocessing.Queue
        Queue used to send GUI updates.
    environment_name : str
        Name of the environment associated with this collector.

    Attributes
    ----------
    environment_command_queue : VerboseMessageQueue
        Queue used to send status commands to the environment.
    environment_name : str
        Associated environment name.
    collector_metadata : CollectorMetadata or None
        Current collector metadata.
    frame_buffer : FrameBuffer or None
        Frame buffer used to extract measurement frames.
    kurtosis_buffer : KurtosisBuffer or None
        Running kurtosis buffer.
    reference_window : numpy.ndarray or int or None
        Window applied to reference channels.
    response_window : numpy.ndarray or int or None
        Window applied to response channels.
    acceptance_function : callable or None
        Function used for automatic frame acceptance.
    skip_frames : int
        Number of upcoming frames to skip after a test-level change.
    test_level : float or None
        Current test level used to normalize frames.
    data_in_queue : multiprocessing.Queue
        Queue containing acquired data.
    data_out_queues : list of multiprocessing.Queue
        Output queues for processed frames.
    last_frame : numpy.ndarray or None
        Most recent frame awaiting manual acceptance.
    window_correction : float or numpy.ndarray or None
        Correction factor applied after windowing.

    Unit Tests
    ----------
    test_data_collector_process_init
        Verifies that the data collector process initializes successfully.
    """

    def __init__
    """
    Initialize the data collector process.

    Initializes the base ``AbstractMessageProcess``, maps collector commands to
    process methods, stores queue references, initializes metadata and runtime
    state, and prepares optional debug state.

    Unit Tests
    ----------
    test_data_collector_process_init
        Confirms that a ``DataCollectorProcess`` instance can be constructed.
    """

    def initialize_collector
    """
    Initialize the collector if metadata has changed.

    Compares the supplied metadata against the current metadata and calls
    ``force_initialize_collector`` only when the metadata differs.

    Parameters
    ----------
    data : CollectorMetadata
        Metadata defining collector behavior.

    Unit Tests
    ----------
    test_data_collector_process_initialize_collector
        Verifies that changed metadata triggers force initialization.
    """

    def force_initialize_collector
    """
    Force collector initialization from metadata.

    Flushes output queues, stores collector metadata, creates a ``FrameBuffer``,
    optionally creates a ``KurtosisBuffer``, loads or defaults the acceptance
    function, creates response and reference windows, applies special force
    window behavior when requested, and computes the window correction factor.

    Parameters
    ----------
    data : CollectorMetadata
        Metadata defining collector behavior.

    Raises
    ------
    ValueError
        If the metadata specifies an invalid window type.

    Unit Tests
    ----------
    test_data_collector_process_force_initialize_collector
        Verifies initialization for supported window types and confirms that
        reference window data are created.
    """

    def acquire
    """
    Acquire data and process available frames.

    Reads acquired data from ``data_in_queue``, adds it to the frame buffer,
    processes any returned frames, skips frames when requested, applies
    acceptance logic, separates response/reference channels, applies optional
    transformations, applies windows and test-level normalization, computes
    FFTs, sends accepted data to output queues, sends GUI frame and kurtosis
    updates, requeues acquisition if not at the end of data, and stops when the
    final acquisition data have been processed.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_data_collector_process_acquire
        Verifies data acquisition, logging, frame processing, GUI update
        behavior, output queue writes, requeue behavior when more data remain,
        and stop behavior when final data are received.
    """

    def accept
    """
    Accept or reject the most recent manually reviewed frame.

    Signals the frame buffer that manual acceptance is complete. If
    ``keep_frame`` is true, sends the accepted time frame to the GUI, computes
    FFTs, separates response and reference channels, and sends the spectral
    frame to output queues. In all cases, clears ``last_frame`` and notifies
    the environment that acceptance completed.

    Parameters
    ----------
    keep_frame : bool
        If ``True``, accept and forward the frame. If ``False``, reject it.

    Unit Tests
    ----------
    test_data_collector_process_accept
        Verifies manual acceptance logging, GUI updates, FFT output, output
        queue writes, and environment acceptance notification.
    """

    def stop
    """
    Stop data collection.

    Waits briefly, logs shutdown, flushes output queues, flushes the command
    queue, resets the frame buffer trigger state, and notifies the environment
    that collector shutdown has been achieved.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_data_collector_process_stop
        Verifies shutdown logging, output queue flushing, command queue
        flushing, trigger reset, and environment shutdown notification.
    """

    def set_test_level
    """
    Set the current test level and skip-frame count.

    Parameters
    ----------
    data : tuple
        Tuple ``(skip_frames, test_level)`` where ``skip_frames`` is the number
        of upcoming frames to skip and ``test_level`` is the normalization level
        used for subsequent accepted frames.

    Unit Tests
    ----------
    test_data_collector_process_set_test_level
        Verifies that skip-frame count and test level are stored and logged.
    """

    def clear_kurtosis_buffer
    """
    Clear the running kurtosis buffer.

    If a kurtosis buffer exists, its stored data are cleared.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.
    """

### Data Collector Process Function
<!---
MARK: Data Collector Process Function
--->
    def data_collector_process
    """
    Entry point used to start the data collector subprocess.

    Constructs a ``DataCollectorProcess`` for the supplied environment and
    queues, then runs its message-processing command loop.

    Parameters
    ----------
    environment_name : str
        Name of the environment associated with the data collector.
    command_queue : VerboseMessageQueue
        Queue from which collector commands are received.
    data_in_queue : multiprocessing.Queue
        Queue from which acquired data are received.
    data_out_queues : list of multiprocessing.Queue
        Queues where processed frames are sent.
    environment_command_queue : VerboseMessageQueue
        Queue used to send status commands to the environment.
    log_file_queue : multiprocessing.Queue
        Queue used to send log messages.
    gui_update_queue : multiprocessing.Queue
        Queue used to send GUI updates.
    process_name : str, optional
        Explicit process name. If omitted, a name is generated from
        ``environment_name``.

    Unit Tests
    ----------
    test_data_collector_process_function
        Verifies that the process function constructs a data collector process
        and starts its command loop.
    """

## Signal Generation Process
<!---
MARK: Signal Generation Process
--->
    """
    Signal generation subprocess for environment excitation.

    This module defines commands, metadata, and a message-driven process for
    generating output time histories. The signal generation process receives
    signal generator objects and parameter updates, generates signal frames,
    applies output transformations, applies disabled-signal masks, ramps test
    level changes, writes output data to the environment output queue, and
    coordinates graceful shutdown.
    """

### Signal Generation Commands
<!---
MARK: Signal Generation Commands
--->
    class SignalGenerationCommands
    """
    Commands accepted by the signal generation process.

    Attributes
    ----------
    INITIALIZE_PARAMETERS : int
        Initialize signal generation parameters.
    INITIALIZE_SIGNAL_GENERATOR : int
        Store the signal generator object used to generate output frames.
    GENERATE_SIGNALS : int
        Generate and output signal data.
    START_SHUTDOWN : int
        Begin graceful signal generation shutdown.
    SHUTDOWN : int
        Complete signal generation shutdown.
    MUTE : int
        Immediately set test level state to zero.
    ADJUST_TEST_LEVEL : int
        Ramp from the current test level to a new target test level.
    SET_TEST_LEVEL : int
        Immediately set the current and target test level.
    SHUTDOWN_ACHIEVED : int
        Notify the environment that signal generation shutdown is complete.

    Unit Tests
    ----------
    test_signal_generation_commands
        Verifies that signal generation command enum values construct valid
        ``SignalGenerationCommands`` members.
    """

### Signal Generation Metadata
<!---
MARK: Signal Generation Metadata
--->
    class SignalGenerationMetadata
    """
    Metadata required to configure the signal generation process.

    Parameters
    ----------
    samples_per_write : int
        Number of samples written to the output queue each time output data are
        produced.
    level_ramp_samples : int
        Number of samples over which test-level changes are ramped.
    output_transformation_matrix : numpy.ndarray, optional
        Matrix used to transform generated output signals to physical output
        channels. The signal generation process stores the pseudoinverse of
        this matrix.
    new_signal_sample_threshold : int, optional
        Minimum remaining signal sample threshold below which a new signal
        frame should be generated. If omitted, defaults to ``samples_per_write``.
    disabled_signals : list of int, optional
        Output signal indices that should be zeroed before output.

    Attributes
    ----------
    ramp_samples : int
        Number of samples over which test-level changes are ramped.
    output_transformation_matrix : numpy.ndarray or None
        Optional output transformation matrix.
    samples_per_write : int
        Number of samples per output write.
    new_signal_sample_threshold : int
        Threshold below which additional signal data should be generated.
    disabled_signals : list of int
        Signal indices that should be disabled.

    Unit Tests
    ----------
    test_signal_generation_metadata_init
        Verifies that signal generation metadata initializes successfully.

    test_signal_generation_metadata_eq
        Verifies equality comparison for equivalent metadata objects.
    """

    def __init__
    """
    Initialize signal generation metadata.

    Stores write size, ramp length, optional output transformation matrix,
    optional generation threshold, and disabled signal indices.

    Parameters
    ----------
    samples_per_write : int
        Number of samples per output write.
    level_ramp_samples : int
        Number of samples used for test-level ramps.
    output_transformation_matrix : numpy.ndarray, optional
        Optional transformation matrix.
    new_signal_sample_threshold : int, optional
        Threshold used to decide when to generate new signal data.
    disabled_signals : list of int, optional
        Signal indices to disable.

    Unit Tests
    ----------
    test_signal_generation_metadata_init
        Confirms that a metadata instance can be constructed.
    """

    def __eq__
    """
    Compare signal generation metadata objects for equality.

    Parameters
    ----------
    other : object
        Object to compare against.

    Returns
    -------
    bool
        ``True`` if all stored metadata fields compare equal, otherwise
        ``False``.

    Unit Tests
    ----------
    test_signal_generation_metadata_eq
        Verifies that equivalent metadata objects compare equal.
    """

### Signal Generation Process
<!---
MARK: Signal Generation Process Class
--->
    class SignalGenerationProcess
    """
    Message-driven process that generates environment output signals.

    ``SignalGenerationProcess`` receives commands through a verbose command
    queue, stores signal generation metadata and a signal generator object,
    accepts parameter updates, generates signal frames when needed, applies
    test-level scaling and ramps, applies optional output transformations,
    writes output data to the output queue, and notifies the environment when
    graceful shutdown is complete.

    Parameters
    ----------
    process_name : str
        Name of the signal generation process used for logging and queue
        access.
    command_queue : VerboseMessageQueue
        Queue from which signal generation commands are received.
    data_in_queue : multiprocessing.Queue
        Queue from which parameter updates for the signal generator are
        received.
    data_out_queue : multiprocessing.Queue
        Queue where generated output time data are written.
    environment_command_queue : VerboseMessageQueue
        Queue used to notify the environment of signal generation status.
    log_file_queue : multiprocessing.Queue
        Queue used to send log messages.
    gui_update_queue : multiprocessing.Queue
        Queue used to send GUI error updates.
    environment_name : str
        Name of the associated environment.

    Attributes
    ----------
    environment_name : str
        Associated environment name.
    data_in_queue : multiprocessing.Queue
        Queue containing parameter updates.
    data_out_queue : multiprocessing.Queue
        Queue receiving generated output data.
    environment_command_queue : VerboseMessageQueue
        Queue used to notify the environment.
    ramp_samples : int or None
        Number of samples used for test-level ramps.
    output_transformation_matrix : numpy.ndarray or None
        Pseudoinverse output transformation matrix.
    samples_per_write : int or None
        Number of samples per output write.
    new_signal_sample_threshold : int or None
        Threshold below which new signal data are generated.
    test_level_target : float
        Target test level scale factor.
    current_test_level : float
        Current test level scale factor.
    test_level_change : float
        Per-sample test level change during ramps.
    signal_remainder : numpy.ndarray or None
        Generated signal samples not yet written to output.
    startup : bool
        Whether the process is waiting to complete startup behavior.
    shutdown_flag : bool
        Whether graceful shutdown has been requested.
    done_generating : bool
        Whether the signal generator has reported that no more data remain.
    signal_generator : SignalGenerator or None
        Signal generator object used to produce new frames.
    disabled_signals : list of int or None
        Signal indices that should be zeroed before output.

    Unit Tests
    ----------
    test_signal_generation_process_init
        Verifies that the signal generation process initializes successfully.
    """

    def __init__
    """
    Initialize the signal generation process.

    Initializes the base ``AbstractMessageProcess``, maps signal generation
    commands to methods, stores queue references and environment name, and
    initializes signal generation state.

    Unit Tests
    ----------
    test_signal_generation_process_init
        Confirms that a ``SignalGenerationProcess`` instance can be
        constructed.
    """

    def initialize_parameters
    """
    Store signal generation parameters.

    Stores ramp length, samples per write, new-signal threshold, disabled
    signals, and the pseudoinverse of the output transformation matrix if one
    is supplied.

    Parameters
    ----------
    data : SignalGenerationMetadata
        Metadata defining signal generation behavior.

    Unit Tests
    ----------
    test_signal_generation_process_initialize_parameters
        Verifies that metadata values are stored, the transformation matrix is
        pseudoinverted, and initialization is logged.
    """

    def initialize_signal_generator
    """
    Store the signal generator object.

    Stores the signal generator and clears any existing signal remainder so
    future generation starts from the new generator.

    Parameters
    ----------
    signal_generator : SignalGenerator
        Signal generator object used to generate output frames.

    Unit Tests
    ----------
    test_signal_generation_process_initialize_signal_generator
        Verifies that the signal generator is stored and signal remainder is
        cleared.
    """

    def generate_signals
    """
    Generate and output signal data.

    Handles startup parameter acquisition, updates signal generator parameters
    from available input queue data, generates additional signal frames when
    the remainder falls below the configured threshold, writes output chunks
    when the output queue is ready, requeues itself to continue generation, and
    initiates shutdown when the final output has been written.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Raises
    ------
    RuntimeError
        If no signal generator has been initialized.

    Unit Tests
    ----------
    test_signal_generation_process_generate_signals
        Verifies startup behavior, parameter updates, frame generation,
        logging, output calls, and command requeueing.
    """

    def output
    """
    Scale and write generated signal data to the output queue.

    Applies disabled-signal zeroing, optional output transformation, current or
    ramped test-level scaling, and writes ``(scaled_data, last_signal)`` to the
    output queue.

    Parameters
    ----------
    write_data : numpy.ndarray
        Generated signal data to output.
    last_signal : bool, optional
        Whether this output is the final signal chunk. Defaults to ``False``.

    Unit Tests
    ----------
    test_signal_generation_process_output
        Verifies disabled-signal handling, test-level ramping, logging, and
        output queue data.
    """

    def mute
    """
    Immediately mute signal generation.

    Sets current test level, target test level, and test-level change to zero.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_signal_generation_process_mute
        Verifies that test-level state is reset to zero.
    """

    def set_test_level
    """
    Immediately set the signal generation test level.

    Sets both current and target test level to the supplied value and clears
    any active test-level ramp.

    Parameters
    ----------
    data : float
        New test level scale factor.

    Unit Tests
    ----------
    test_signal_generation_process_set_test_level
        Verifies that current level, target level, and ramp state are updated.
    """

    def adjust_test_level
    """
    Begin ramping toward a new target test level.

    Computes the per-sample test-level change required to reach the supplied
    target level over the configured ramp length.

    Parameters
    ----------
    data : float
        Target test level scale factor.

    Unit Tests
    ----------
    test_signal_generation_process_adjust_test_level
        Verifies that target level and per-sample level change are updated and
        logged.
    """

    def start_shutdown
    """
    Begin graceful signal generation shutdown.

    If the process is not already shutting down or still in startup, sets the
    shutdown flag, ramps the test level toward zero, flushes pending commands,
    and requeues signal generation if a generate command was pending.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_signal_generation_process_start_shutdown
        Verifies shutdown flag behavior, command queue flushing, level ramp
        initiation, and regeneration command requeueing.
    """

    def shutdown
    """
    Complete signal generation shutdown.

    Logs shutdown, flushes the command queue, notifies the environment that
    shutdown was achieved, and resets startup, shutdown, and generation state.

    Unit Tests
    ----------
    test_signal_generation_process_shutdown
        Verifies shutdown logging, command queue flushing, environment
        notification, and state reset.
    """

### Signal Generation Process Function
<!---
MARK: Signal Generation Process Function
--->
    def signal_generation_process
    """
    Entry point used to start the signal generation subprocess.

    Constructs a ``SignalGenerationProcess`` with the supplied queues and
    environment name, then runs its message-processing command loop.

    Parameters
    ----------
    environment_name : str
        Name of the associated environment.
    command_queue : VerboseMessageQueue
        Queue from which signal generation commands are received.
    data_in_queue : multiprocessing.Queue
        Queue from which parameter updates are received.
    data_out_queue : multiprocessing.Queue
        Queue where generated output data are written.
    environment_command_queue : VerboseMessageQueue
        Queue used to notify the environment of status changes.
    log_file_queue : multiprocessing.Queue
        Queue used to send log messages.
    gui_update_queue : multiprocessing.Queue
        Queue used to send GUI updates.
    process_name : str, optional
        Explicit process name. If omitted, a process name is generated from
        ``environment_name``.

    Unit Tests
    ----------
    test_signal_generation_process_func
        Verifies that the process function constructs a signal generation
        process and starts its command loop.
    """

## Signal Generation
<!---
MARK: Signal Generation
--->
    """
    Signal generation utilities and generator classes.

    This module defines signal generator types and concrete signal generator
    implementations used by environments and signal generation processes. It
    includes random, pseudorandom, burst random, chirp, sine, square, CPSD,
    transient, and continuous transient signal generators.
    """

### Signal Types
<!---
MARK: Signal Types
--->
    class SignalTypes
    """
    Enumeration of supported signal generator types.

    Attributes
    ----------
    RANDOM : int
        Broadband random signal generated using COLA.
    PSEUDORANDOM : int
        Periodic signal with random phase spectrum.
    BURST_RANDOM : int
        Bandlimited random signal multiplied by a burst envelope.
    CHIRP : int
        Periodic sine sweep.
    SINE : int
        Stationary sine signal.
    SQUARE : int
        Stationary square wave.
    CPSD : int
        Random signal synthesized from a target CPSD matrix.
    TRANSIENT : int
        Predefined transient signal.
    CONTINUOUSTRANSIENT : int
        Continuously updated transient signal buffer.

    Unit Tests
    ----------
    test_signal_types
        Verifies that signal type enum values construct valid ``SignalTypes``
        members.
    """

### Signal Generation Utility Functions
<!---
MARK: Signal Generation Utility Functions
--->
    def cola
    """
    Combine two signal frames using constant overlap-add.

    Applies a window to the current signal and overlaps the end of the previous
    signal with the beginning of the current signal. This is commonly used to
    generate long random signals from individual random realizations while
    maintaining smooth transitions and approximately constant variance.

    Parameters
    ----------
    signal_samples : int
        Number of output samples from the current signal.
    end_samples : int
        Number of overlapped samples from the previous signal.
    signals : numpy.ndarray
        Array containing two signal frames with shape
        ``(2, num_signals, total_samples)``.
    window_name : str
        Name of the scipy window used for blending.
    window_exponent : float, optional
        Exponent applied to the window. Defaults to ``0.5``.

    Returns
    -------
    numpy.ndarray
        Combined overlap-add output signal.

    Unit Tests
    ----------
    test_cola
        Verifies that the COLA helper returns expected overlapped signal data.
    """

    def cpsd_to_time_history
    """
    Generate a time-history realization from a CPSD matrix.

    Synthesizes random frequency-domain signals consistent with a target CPSD
    matrix using singular value decomposition, then transforms them to the time
    domain with an inverse real FFT.

    Parameters
    ----------
    cpsd_matrix : numpy.ndarray
        Complex CPSD matrix with shape
        ``(frequency_lines, num_signals, num_signals)``.
    sample_rate : float
        Sample rate in samples per second.
    df : float
        Frequency spacing of the CPSD matrix.
    output_oversample : int, optional
        Output oversampling factor. Defaults to ``1``.

    Returns
    -------
    numpy.ndarray
        Time-history realization with shape ``(num_signals, samples)``.

    Unit Tests
    ----------
    test_cpsd_to_time_history
        Verifies that CPSD synthesis returns an output array with expected
        dimensions.
    """

### Signal Generator Base Class
<!---
MARK: Signal Generator Base Class
--->
    class SignalGenerator
    """
    Abstract base class for signal generator implementations.

    Signal generator subclasses must provide a ``generate_frame`` method and a
    ``ready_for_next_output`` property. They may also implement
    ``update_parameters`` when runtime parameter updates are supported.

    Unit Tests
    ----------
    test_signal_generator_init
        Verifies that a concrete dummy signal generator subclass can be
        constructed.
    """

    def generate_frame
    """
    Generate one frame of signal data.

    Returns
    -------
    tuple
        Tuple ``(signal, done)`` where ``signal`` is a NumPy array containing
        generated output data and ``done`` indicates whether generation is
        complete.
    """

    def update_parameters
    """
    Update generator parameters.

    Subclasses may override this method to accept runtime parameter updates.
    """

    property
    def ready_for_next_output
    """
    Return whether the generator can currently produce a frame.

    Returns
    -------
    bool
        ``True`` if ``generate_frame`` can be called successfully.
    """

### Random Signal Generator
<!---
MARK: Random Signal Generator
--->
    class RandomSignalGenerator
    """
    Generator for broadband random signals using constant overlap-add.

    Random frames are generated in the time domain, bandlimited in the
    frequency domain, stored in a two-frame COLA queue, and overlap-added to
    produce smooth output.

    Parameters
    ----------
    rms : float
        Desired RMS level.
    sample_rate : float
        Sample rate in samples per second.
    num_samples_per_frame : int
        Number of controller-frame samples before output oversampling.
    num_signals : int
        Number of independent signals to generate.
    low_frequency_cutoff : float or None
        Lower frequency cutoff. ``None`` uses zero hertz.
    high_frequency_cutoff : float or None
        Upper frequency cutoff. ``None`` uses Nyquist frequency.
    cola_overlap : float
        Fractional overlap between random frames.
    cola_window : str
        Window used for COLA blending.
    cola_exponent : float
        Exponent applied to the COLA window.
    output_oversample : int
        Output oversampling factor.

    Unit Tests
    ----------
    test_random_wave
        Verifies generated random signal data against an independently
        computed reference for fixed random seeds.

    test_random_wave_ready_output
        Verifies that random signal generators are always ready for output.
    """

    property
    def samples_per_output
    """
    Return the number of non-overlapped samples per output.

    Returns
    -------
    int
        Number of samples advanced per output frame.
    """

    property
    def overlapped_output_samples
    """
    Return the number of overlapped samples.

    Returns
    -------
    int
        Number of samples blended with the previous random frame.
    """

    property
    def ready_for_next_output
    """
    Return whether random output is ready.

    Random signal generators are always ready to generate another output frame.

    Returns
    -------
    bool
        Always ``True``.
    """

    def generate_frame
    """
    Generate one random output frame.

    Creates random Gaussian signals, bandlimits them, updates the COLA queue,
    overlap-adds the newest and previous frames, and returns the resulting
    output.

    Returns
    -------
    tuple
        Tuple ``(output_signal, False)``.

    Unit Tests
    ----------
    test_random_wave
        Verifies generated output for deterministic random seeds.
    """

### Pseudorandom Signal Generator
<!---
MARK: Pseudorandom Signal Generator
--->
    class PseudorandomSignalGenerator
    """
    Generator for periodic pseudorandom signals.

    A random phase spectrum is generated once during initialization and
    transformed to the time domain. Subsequent frames repeat the same signal.

    Parameters
    ----------
    rms : float
        Desired RMS level.
    sample_rate : float
        Sample rate in samples per second.
    num_samples_per_frame : int
        Number of controller-frame samples before output oversampling.
    num_signals : int
        Number of signals to generate.
    low_frequency_cutoff : float or None
        Lower frequency cutoff.
    high_frequency_cutoff : float or None
        Upper frequency cutoff.
    output_oversample : int
        Output oversampling factor.

    Unit Tests
    ----------
    test_pseudorandom_wave
        Verifies generated pseudorandom signal data against an independently
        computed reference for fixed random seeds.

    test_pseudorandom_wave_ready_output
        Verifies that pseudorandom signal generators are always ready for
        output.
    """

    def generate_frame
    """
    Return one pseudorandom signal frame.

    Returns
    -------
    tuple
        Tuple ``(signal_copy, False)``.
    """

    property
    def ready_for_next_output
    """
    Return whether pseudorandom output is ready.

    Returns
    -------
    bool
        Always ``True``.
    """

### Burst Random Signal Generator
<!---
MARK: Burst Random Signal Generator
--->
    class BurstRandomSignalGenerator
    """
    Generator for burst random excitation signals.

    A bandlimited random signal is multiplied by an envelope consisting of a
    ramp-up, constant-on portion, ramp-down, and trailing zeros.

    Parameters
    ----------
    rms : float
        Desired RMS level during the burst.
    sample_rate : float
        Sample rate in samples per second.
    num_samples_per_frame : int
        Number of controller-frame samples before output oversampling.
    num_signals : int
        Number of signals to generate.
    low_frequency_cutoff : float or None
        Lower frequency cutoff.
    high_frequency_cutoff : float or None
        Upper frequency cutoff.
    on_fraction : float
        Fraction of frame occupied by the burst.
    ramp_fraction : float
        Fraction of burst-on time used for each ramp.
    output_oversample : int
        Output oversampling factor.

    Raises
    ------
    ValueError
        If ``ramp_fraction`` is greater than ``0.5``.

    Unit Tests
    ----------
    test_burst_random_wave
        Verifies generated burst random signal data against an independently
        computed reference for fixed random seeds.

    test_burst_wave_ready_output
        Verifies that burst random signal generators are always ready for
        output.
    """

    property
    def ramp_samples
    """
    Return the number of ramp samples.

    Returns
    -------
    int
        Number of samples in each ramp-up and ramp-down section.
    """

    property
    def on_samples
    """
    Return the number of fully-on burst samples.

    Returns
    -------
    int
        Number of samples between ramp-up and ramp-down sections.
    """

    property
    def ready_for_next_output
    """
    Return whether burst random output is ready.

    Returns
    -------
    bool
        Always ``True``.
    """

    def generate_frame
    """
    Generate one burst random frame.

    Returns
    -------
    tuple
        Tuple ``(burst_random_signal, False)``.
    """

### Chirp Signal Generator
<!---
MARK: Chirp Signal Generator
--->
    class ChirpSignalGenerator
    """
    Generator for periodic fast sine sweeps.

    Parameters
    ----------
    level : float
        Peak chirp amplitude.
    sample_rate : float
        Sample rate in samples per second.
    num_samples_per_frame : int
        Number of controller-frame samples before output oversampling.
    num_signals : int
        Number of signals to generate.
    low_frequency_cutoff : float
        Starting frequency.
    high_frequency_cutoff : float
        Ending frequency.
    output_oversample : int
        Output oversampling factor.

    Unit Tests
    ----------
    test_chirp_wave
        Verifies generated chirp signal data against an independently computed
        reference.

    test_chirp_wave_ready_output
        Verifies that chirp signal generators are always ready for output.
    """

    def generate_frame
    """
    Return one chirp signal frame.

    Returns
    -------
    tuple
        Tuple ``(signal_copy, False)``.
    """

    property
    def ready_for_next_output
    """
    Return whether chirp output is ready.

    Returns
    -------
    bool
        Always ``True``.
    """

### Sine Signal Generator
<!---
MARK: Sine Signal Generator
--->
    class SineSignalGenerator
    """
    Generator for stationary sine signals with phase tracking.

    Parameters
    ----------
    level : float or array-like
        Sine amplitude.
    sample_rate : float
        Sample rate in samples per second.
    num_samples_per_frame : int
        Number of controller-frame samples before output oversampling.
    num_signals : int
        Number of signals to generate.
    frequency : float or array-like or None
        Sine frequency.
    phase : float or array-like or None
        Initial phase in radians.
    output_oversample : int
        Output oversampling factor.

    Unit Tests
    ----------
    test_sine_wave
        Verifies generated sine signal data against an independently computed
        reference.

    test_sine_wave_ready_output
        Verifies readiness when frequency and phase are defined.

    test_sine_wae_update_parameters
        Verifies frequency, level, and phase parameter updates.
    """

    property
    def phase_per_sample
    """
    Return phase change per sample.

    Returns
    -------
    numpy.ndarray
        Phase increment per sample.
    """

    property
    def phase_per_frame
    """
    Return phase change per generated frame.

    Returns
    -------
    numpy.ndarray
        Phase increment per frame.
    """

    property
    def ready_for_next_output
    """
    Return whether sine output is ready.

    Returns
    -------
    bool
        ``True`` when both frequency and phase are defined.
    """

    def update_parameters
    """
    Update sine frequency, level, and optionally phase.

    Parameters
    ----------
    frequency : numpy.ndarray or float
        New sine frequencies.
    level : numpy.ndarray or float
        New amplitudes.
    phase : numpy.ndarray or float, optional
        New phases. If omitted, phase is not updated.
    """

    def generate_frame
    """
    Generate one sine frame and advance phase.

    Returns
    -------
    tuple
        Tuple ``(signal, False)``.
    """

### Square Signal Generator
<!---
MARK: Square Signal Generator
--->
    class SquareSignalGenerator
    """
    Generator for square waves with phase tracking.

    Parameters
    ----------
    level : float or array-like
        Square wave amplitude.
    sample_rate : float
        Sample rate in samples per second.
    num_samples_per_frame : int
        Number of controller-frame samples before output oversampling.
    num_signals : int
        Number of signals to generate.
    frequency : float or array-like or None
        Square wave frequency.
    phase : float or array-like or None
        Initial phase in radians.
    on_fraction : float
        Fraction of each cycle spent at the positive level.
    output_oversample : int
        Output oversampling factor.

    Unit Tests
    ----------
    test_square_wave
        Verifies generated square wave data against an independently computed
        reference.

    test_square_wave_ready_output
        Verifies readiness when frequency and phase are defined.

    test_square_wave_update_parameters
        Verifies frequency and phase parameter updates.
    """

    property
    def phase_per_sample
    """
    Return phase change per sample.

    Returns
    -------
    numpy.ndarray
        Phase increment per sample.
    """

    property
    def phase_per_frame
    """
    Return phase change per generated frame.

    Returns
    -------
    numpy.ndarray
        Phase increment per frame.
    """

    property
    def ready_for_next_output
    """
    Return whether square output is ready.

    Returns
    -------
    bool
        ``True`` when both frequency and phase are defined.
    """

    def update_parameters
    """
    Update square wave frequency and optionally phase.

    Parameters
    ----------
    frequency : numpy.ndarray or float
        New square wave frequencies.
    phase : numpy.ndarray or float, optional
        New phases. If omitted, phase is not updated.
    """

    def generate_frame
    """
    Generate one square wave frame and advance phase.

    Returns
    -------
    tuple
        Tuple ``(signal, False)``.
    """

### CPSD Signal Generator
<!---
MARK: CPSD Signal Generator
--->
    class CPSDSignalGenerator
    """
    Generator for random signals satisfying a target CPSD matrix.

    Uses SVD-based synthesis to generate time histories consistent with a
    target CPSD matrix. Generated frames are blended using COLA. Optional sigma
    clipping can reduce extreme random samples.

    Parameters
    ----------
    sample_rate : float
        Sample rate in samples per second.
    num_samples_per_frame : int
        Number of controller-frame samples before output oversampling.
    num_signals : int
        Number of signals to generate.
    cpsd_matrix : numpy.ndarray or None
        Target CPSD matrix.
    cola_overlap : float
        Fractional COLA overlap.
    cola_window : str
        COLA window name.
    cola_exponent : float
        COLA window exponent.
    sigma_clip : float or numpy.ndarray or None
        Optional sigma clipping threshold. Values greater than or equal to
        ``5`` disable clipping.
    output_oversample : int
        Output oversampling factor.
    """

    property
    def samples_per_output
    """
    Return the number of non-overlapped samples per output.

    Returns
    -------
    int
        Samples advanced per generated output frame.
    """

    property
    def overlapped_output_samples
    """
    Return the number of overlapped samples.

    Returns
    -------
    int
        Samples blended with the previous CPSD realization.
    """

    property
    def frequency_spacing
    """
    Return frequency line spacing.

    Returns
    -------
    float
        Frequency resolution in hertz.
    """

    property
    def ready_for_next_output
    """
    Return whether CPSD output is ready.

    Returns
    -------
    bool
        ``True`` when a CPSD matrix is defined.
    """

    def update_parameters
    """
    Update the target CPSD matrix.

    Computes RMS values, sigma-clip correction factors, random vector shape,
    and the SVD-based spectral factor used to generate frames.

    Parameters
    ----------
    cpsd_matrix : numpy.ndarray or None
        Target CPSD matrix.
    """

    def rejection_sample
    """
    Generate Gaussian random samples using rejection sampling.

    Parameters
    ----------
    size : tuple
        Desired output shape.
    threshold : float or numpy.ndarray, optional
        Sigma clipping threshold.

    Returns
    -------
    numpy.ndarray or None
        Rejection-sampled Gaussian data, or ``None`` if threshold is ``None``.
    """

    def generate_frame
    """
    Generate one CPSD-consistent random output frame.

    Returns
    -------
    tuple
        Tuple ``(output_signal, False)``.
    """

### Continuous Transient Signal Generator
<!---
MARK: Continuous Transient Signal Generator
--->
    class ContinuousTransientSignalGenerator
    """
    Generator for continuously supplied transient signal data.

    Stores incoming transient data in a buffer and outputs fixed-size frames as
    enough data become available. The generator can also output a final partial
    frame when no more signal data are incoming.

    Parameters
    ----------
    num_samples_per_frame : int
        Number of samples per generated frame.
    num_signals : int
        Number of signals.
    signal : numpy.ndarray or None
        Initial signal data.
    last_signal : bool
        Whether no more signal data will be supplied.
    """

    property
    def ready_for_next_output
    """
    Return whether enough signal data are available for output.

    Returns
    -------
    bool
        ``True`` when the signal buffer contains at least one frame or when no
        more signal data are incoming.
    """

    def update_parameters
    """
    Append new signal data to the transient buffer.

    Parameters
    ----------
    signal : numpy.ndarray
        New signal data to append.
    last_signal : bool
        Whether this is the final supplied signal data.
    """

    def generate_frame
    """
    Generate the next transient signal frame.

    Returns
    -------
    tuple
        Tuple ``(output_signal, done)`` where ``done`` is ``True`` when the
        final available signal has been output.
    """

### Transient Signal Generator
<!---
MARK: Transient Signal Generator
--->
    class TransientSignalGenerator
    """
    Generator for a predefined transient signal.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal to output.
    repeat : bool
        Whether the signal should repeat.
    """

    property
    def ready_for_next_output
    """
    Return whether a transient signal is defined.

    Returns
    -------
    bool
        ``True`` if ``signal`` is not ``None``.
    """

    def update_parameters
    """
    Replace the transient signal and repeat flag.

    Parameters
    ----------
    signal : numpy.ndarray
        New signal data.
    repeat : bool
        Whether the signal should repeat.
    """

    def generate_frame
    """
    Return the transient signal.

    Returns
    -------
    tuple
        Tuple ``(signal, done)`` where ``done`` is ``True`` when the signal is
        not repeating.
    """

## Spectral Processing
<!---
MARK: Spectral Processing
--->
    """
    Spectral processing subprocess for FRFs, CPSDs, APSDs, coherence, and
    related spectral quantities.

    This module defines command enums, metadata, and a message-driven process
    that receives response/reference FFT frames, averages spectral matrices,
    computes frequency response functions using several estimators, computes
    coherence, normalizes CPSD/APSD outputs, and sends updated spectral data
    back to the owning environment.
    """

### Spectral Processing Commands
<!---
MARK: Spectral Processing Commands
--->
    class SpectralProcessingCommands
    """
    Commands accepted by the spectral processing process.

    Attributes
    ----------
    INITIALIZE_PARAMETERS : int
        Initialize spectral processing metadata and internal arrays.
    RUN_SPECTRAL_PROCESSING : int
        Process available response/reference FFT frames.
    CLEAR_SPECTRAL_PROCESSING : int
        Clear accumulated spectral state.
    STOP_SPECTRAL_PROCESSING : int
        Stop spectral processing and notify the environment.
    SENT_SPECTRAL_DATA : int
        Indicates spectral data were sent.
    SHUTDOWN_ACHIEVED : int
        Notify the environment that spectral processing shutdown is complete.

    Unit Tests
    ----------
    test_spectral_processing_commands
        Verifies that enum values construct valid
        ``SpectralProcessingCommands`` members.
    """

### Averaging Types
<!---
MARK: Averaging Types
--->
    class AveragingTypes
    """
    Enumeration of supported spectral averaging strategies.

    Attributes
    ----------
    LINEAR : int
        Linear averaging over a fixed number of frames.
    EXPONENTIAL : int
        Exponential averaging using an averaging coefficient.

    Unit Tests
    ----------
    test_averaging_types
        Verifies that enum values construct valid ``AveragingTypes`` members.
    """

### FRF Estimator
<!---
MARK: FRF Estimator
--->
    class Estimator
    """
    Enumeration of supported frequency response function estimators.

    Attributes
    ----------
    H1 : int
        H1 estimator.
    H2 : int
        H2 estimator.
    H3 : int
        Average of H1-like and H2-like estimates.
    HV : int
        Hv estimator based on an augmented spectral matrix.

    Unit Tests
    ----------
    test_averaging_types
        Verifies that enum values construct valid ``Estimator`` members.
    """

### Spectral Processing Metadata
<!---
MARK: Spectral Processing Metadata
--->
    class SpectralProcessingMetadata
    """
    Metadata required to configure spectral processing.

    This class stores averaging settings, FRF estimator settings, channel
    counts, frequency spacing, sample rate, number of frequency lines, and
    booleans controlling which spectral quantities should be computed.

    Parameters
    ----------
    averaging_type : AveragingTypes
        Spectral averaging strategy.
    averages : int
        Number of frames used for linear averaging or reported averaging count.
    exponential_averaging_coefficient : float
        Weighting coefficient used for exponential averaging.
    frf_estimator : Estimator
        FRF estimator to compute.
    num_response_channels : int
        Number of response channels.
    num_reference_channels : int
        Number of reference channels.
    frequency_spacing : float
        Frequency spacing in hertz.
    sample_rate : float
        Sample rate in samples per second.
    num_frequency_lines : int
        Number of frequency lines in each FFT frame.
    compute_cpsd : bool, optional
        Whether to compute full CPSD matrices. Defaults to ``True``.
    compute_frf : bool, optional
        Whether to compute FRFs. Defaults to ``True``.
    compute_coherence : bool, optional
        Whether to compute coherence. Defaults to ``True``.
    compute_apsd : bool, optional
        Whether to compute APSD outputs. Defaults to ``True``.

    Attributes
    ----------
    averaging_type : AveragingTypes
        Spectral averaging strategy.
    averages : int
        Averaging count.
    exponential_averaging_coefficient : float
        Exponential averaging coefficient.
    frf_estimator : Estimator
        FRF estimator.
    num_response_channels : int
        Number of response channels.
    num_reference_channels : int
        Number of reference channels.
    frequency_spacing : float
        Frequency spacing.
    sample_rate : float
        Sample rate.
    num_frequency_lines : int
        Number of FFT frequency lines.
    compute_cpsd : bool
        Whether full CPSD matrices should be computed.
    compute_frf : bool
        Whether FRFs should be computed.
    compute_coherence : bool
        Whether coherence should be computed.
    compute_apsd : bool
        Whether APSDs should be computed.

    Unit Tests
    ----------
    test_spectral_processing_metadata_init
        Verifies that metadata initializes and that spectral-requirement
        properties return expected values.

    test_spectral_processing_metadata_eq
        Verifies equality comparison for equivalent metadata objects.
    """

    def __init__
    """
    Initialize spectral processing metadata.

    Stores averaging, estimator, channel, frequency, sample-rate, and requested
    output settings.

    Unit Tests
    ----------
    test_spectral_processing_metadata_init
        Confirms that a ``SpectralProcessingMetadata`` instance can be
        constructed.
    """

    def __eq__
    """
    Compare spectral processing metadata objects for equality.

    Parameters
    ----------
    other : object
        Object to compare against.

    Returns
    -------
    bool
        ``True`` if all stored metadata fields compare equal, otherwise
        ``False``.

    Unit Tests
    ----------
    test_spectral_processing_metadata_eq
        Verifies that equivalent metadata objects compare equal.
    """

    property
    def requires_full_spectral_response
    """
    Return whether the full response spectral matrix is required.

    Full response spectra are required for H2 or H3 FRF computation and for
    full CPSD output.

    Returns
    -------
    bool
        ``True`` if full response spectral matrices are required.

    Unit Tests
    ----------
    test_spectral_processing_metadata_init
        Verifies this requirement flag for a representative metadata
        configuration.
    """

    property
    def requires_diagonal_spectral_response
    """
    Return whether response spectral diagonals are required.

    Response diagonal spectra are required for Hv FRF computation, APSD output,
    or coherence computation.

    Returns
    -------
    bool
        ``True`` if response diagonal spectra are required.

    Unit Tests
    ----------
    test_spectral_processing_metadata_init
        Verifies this requirement flag for a representative metadata
        configuration.
    """

    property
    def requires_full_spectral_reference
    """
    Return whether the full reference spectral matrix is required.

    Full reference spectra are required for H1, H3, or Hv FRF computation, full
    CPSD output, or coherence computation.

    Returns
    -------
    bool
        ``True`` if full reference spectral matrices are required.

    Unit Tests
    ----------
    test_spectral_processing_metadata_init
        Verifies this requirement flag for a representative metadata
        configuration.
    """

    property
    def requires_diagonal_spectral_reference
    """
    Return whether reference spectral diagonals are required.

    Reference diagonal spectra are required for APSD output.

    Returns
    -------
    bool
        ``True`` if reference diagonal spectra are required.

    Unit Tests
    ----------
    test_spectral_processing_metadata_init
        Verifies this requirement flag for a representative metadata
        configuration.
    """

    property
    def requires_spectral_reference_response
    """
    Return whether response/reference cross spectra are required.

    Cross spectra are required for FRF or coherence computation.

    Returns
    -------
    bool
        ``True`` if response/reference cross spectra are required.

    Unit Tests
    ----------
    test_spectral_processing_metadata_init
        Verifies this requirement flag for a representative metadata
        configuration.
    """

### Spectral Processing Process
<!---
MARK: Spectral Processing Process
--->
    class SpectralProcessingProcess
    """
    Message-driven process for spectral matrix and FRF computation.

    ``SpectralProcessingProcess`` receives response/reference FFT frames from a
    data collector, accumulates linear or exponential spectral averages,
    computes requested response, reference, and cross spectral matrices,
    computes FRFs, coherence, APSD/CPSD outputs, and sends updated spectral
    results to the owning environment.

    Parameters
    ----------
    process_name : str
        Name of the spectral processing process used for logging and queue
        access.
    command_queue : VerboseMessageQueue
        Queue from which spectral processing commands are received.
    data_in_queue : multiprocessing.Queue
        Queue containing response/reference FFT frame tuples.
    data_out_queue : multiprocessing.Queue
        Queue where updated spectral results are sent.
    environment_command_queue : VerboseMessageQueue
        Queue used to notify the owning environment of shutdown completion.
    gui_update_queue : multiprocessing.Queue
        Queue used for GUI updates and error reporting.
    log_file_queue : multiprocessing.Queue
        Queue used to send log messages.
    environment_name : str
        Name of the owning environment.

    Attributes
    ----------
    environment_name : str
        Owning environment name.
    data_in_queue : multiprocessing.Queue
        Input queue containing FFT frame data.
    data_out_queue : multiprocessing.Queue
        Output queue for updated spectral data.
    environment_command_queue : VerboseMessageQueue
        Queue used to notify the environment.
    response_spectral_matrix : numpy.ndarray or None
        Full response spectral matrix.
    reference_spectral_matrix : numpy.ndarray or None
        Full reference spectral matrix.
    response_reference_spectral_matrix : numpy.ndarray or None
        Cross spectral matrix from response to reference.
    reference_diagonal_matrix : numpy.ndarray or None
        Reference spectral diagonal values.
    response_diagonal_matrix : numpy.ndarray or None
        Response spectral diagonal values.
    response_fft : numpy.ndarray or None
        Linear averaging buffer for response FFT frames.
    reference_fft : numpy.ndarray or None
        Linear averaging buffer for reference FFT frames.
    spectral_processing_parameters : SpectralProcessingMetadata or None
        Current spectral processing metadata.
    frames_computed : int
        Number of frames processed for exponential averaging.

    Unit Tests
    ----------
    test_spectral_processing_init
        Verifies that the spectral processing process initializes successfully.
    """

    def __init__
    """
    Initialize the spectral processing process.

    Initializes the base ``AbstractMessageProcess``, maps spectral processing
    commands to process methods, stores queue references and environment name,
    and initializes spectral matrices and counters.

    Unit Tests
    ----------
    test_spectral_processing_init
        Confirms that a ``SpectralProcessingProcess`` can be constructed.
    """

    def initialize_parameters
    """
    Initialize or update spectral processing parameters.

    Stores the supplied metadata. If dimensions, averaging type, or averaging
    count have changed, internal spectral arrays are reset. For linear
    averaging, response and reference FFT buffers are allocated and initialized
    with NaNs. For exponential averaging, FFT buffers are cleared.

    Parameters
    ----------
    data : SpectralProcessingMetadata
        Spectral processing metadata.

    Unit Tests
    ----------
    test_spectral_processing_initialize_parameters
        Verifies that linear averaging arrays are initialized with expected
        shapes and NaN contents.
    """

    def run_spectral_processing
    """
    Process available FFT frames and compute spectral quantities.

    Flushes available FFT frame data from the input queue. If no data are
    available, waits briefly and requeues processing. For linear averaging,
    frame data are inserted into rolling FFT buffers and averaged after
    excluding NaN frames. For exponential averaging, spectral matrices are
    updated recursively. Depending on metadata flags, computes response,
    reference, and cross spectral matrices, FRFs, coherence, CPSDs or APSDs,
    frequency vector, FRF condition number, and sends updated results to the
    output queue. The command is requeued to continue processing.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.
    """

    def clear_spectral_processing
    """
    Clear accumulated spectral processing state.

    Resets frame count and spectral matrices. For linear averaging, FFT buffers
    are filled with NaNs. For exponential averaging, FFT buffers are set to
    ``None``.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_spectral_processing_clear_spectral_processing
        Verifies that spectral matrices are cleared and linear FFT buffers are
        reset to NaN.
    """

    def stop_spectral_processing
    """
    Stop spectral processing and notify the environment.

    Waits briefly, flushes the command queue, requeues any quit command pulled
    during flushing, flushes outgoing spectral data, and notifies the
    environment that shutdown has been achieved.

    Parameters
    ----------
    data : Any
        Command data supplied through the command queue. This value is ignored.

    Unit Tests
    ----------
    test_spectral_processing_stop_spectral_processing
        Verifies command queue flushing, quit command preservation, output
        queue flushing, and shutdown notification to the environment.
    """

### Spectral Processing Process Function
<!---
MARK: Spectral Processing Process Function
--->
    def spectral_processing_process
    """
    Entry point used to start the spectral processing subprocess.

    Constructs a ``SpectralProcessingProcess`` with the supplied queues and
    environment name, then runs its message-processing command loop.

    Parameters
    ----------
    environment_name : str
        Name of the owning environment.
    command_queue : VerboseMessageQueue
        Queue from which spectral processing commands are received.
    data_in_queue : multiprocessing.Queue
        Queue containing response/reference FFT frame tuples.
    data_out_queue : multiprocessing.Queue
        Queue where updated spectral results are sent.
    environment_command_queue : VerboseMessageQueue
        Queue used to notify the environment of shutdown completion.
    gui_update_queue : multiprocessing.Queue
        Queue used for GUI updates.
    log_file_queue : multiprocessing.Queue
        Queue used to send log messages.
    process_name : str, optional
        Explicit process name. If omitted, a process name is generated from
        ``environment_name``.

    Unit Tests
    ----------
    test_spectral_processing_process
        Verifies that the process function constructs a spectral processing
        process and starts its command loop.
    """

# User Interface

# Examples

# Testing

