# Rattlesnake Codebase Overview

(sec:codebase_overview)=
# Rattlesnake Codebase Overview

This document provides a high-level overview of the Rattlesnake codebase for new developers, advanced users, and maintainers. Its purpose is to explain the major subsystems of the software, how they interact, and where to begin when extending or debugging the framework.

Rattlesnake is a MIMO Vibration Controller that supports both graphical use through the Qt-based user interface, and headless/programmatic use through the Python API.

Because the software is heavily process-oriented and modular, it is useful to understand the architecture before attempting to add new hardware backends, environments, control laws, or headless workflows.

## High-Level Architecture

At a high level, Rattlesnake is composed of four major layers:

1. **Controller / Orchestration** — the `RattlesnakeController`, its helper managers, and the logic that coordinates startup, shutdown, acquisition, environments, and profiles.
2. **Hardware Abstraction** — a common interface for acquisition and output hardware, with concrete implementations for both physical and virtual systems.
3. **Environment Logic** — the test-specific logic for Random, Sine, Transient, SDS, Modal, Time History Generation, and Read Data environments.
4. **User Interface** — the Qt-based application, per-environment UI classes, dialogs, plots, and interaction workflows.

These layers communicate primarily through:

- **queues** for commands and data,
- **events** for synchronization and state tracking,
- **metadata objects** for persistent configuration, and
- **instruction objects** for per-run settings.

A simplified architectural view is shown below.

```{mermaid}
flowchart TD
    Caller["GUI / Headless Caller"]
    Controller["RattlesnakeController"]

    CtrlProc["Controller Process"]
    Acq["Acquisition Process"]
    Out["Output Process"]
    Stream["Streaming Process"]
    EnvMgr["Environment Manager"]
    ProfMgr["Profile Manager"]

    EnvProc["Environment Processes"]
    EnvSub["Environment Subprocesses
(signal generation, collector,
spectral processing, data analysis)"]

    Caller --> Controller
    Controller --> CtrlProc
    Controller --> Acq
    Controller --> Out
    Controller --> Stream
    Controller --> EnvMgr
    Controller --> ProfMgr
    EnvMgr --> EnvProc
    EnvProc --> EnvSub
```

## Main Entry Points

There are several important runtime entry points in the codebase.

- `rattlesnake.main:main` — the standard Python entry point for launching the application.
- `launch_rattlesnake_ui(...)` — wraps a `RattlesnakeController` in the Qt UI and enters the Qt event loop.
- `build_rattlesnake_app(...)` — constructs the QApplication and `RattlesnakeUI` without immediately entering `app.exec_()`, which is useful for scripted setup, tests, and automated documentation.
- `RattlesnakeController` — the main programmatic API object and the core orchestrator of the software.

For most headless workflows, `RattlesnakeController` is the primary object that user code will interact with.

## The `RattlesnakeController`

The `RattlesnakeController`, defined in `engine.py`, is the central coordination object of the software.

It is responsible for:

- spinning up and shutting down subprocesses,
- maintaining global controller state,
- storing hardware and environment metadata,
- relaying commands to acquisition, output, streaming, and environment processes,
- coordinating system identification and environment runs, and
- managing profiles and timed events.

### Controller state machine

Rattlesnake uses the `RattlesnakeState` enumeration to describe its current operating state.

- `INIT` — no hardware or environments have been initialized yet.
- `HARDWARE_STORE` — hardware metadata has been initialized and distributed.
- `ENVIRONMENT_STORE` — environment metadata has been initialized and distributed.
- `HARDWARE_ACTIVE` — acquisition and output are active, but no environment is currently running.
- `ENVIRONMENT_ACTIVE` — one or more environments are actively running.
- `SYS_ID_ACTIVE` — a system identification run is currently active.

These states are derived from:
- whether metadata has been initialized,
- whether acquisition/output are active,
- and whether environment or system ID events are active.

### Important collaborators

The controller owns or coordinates several major helper objects.

- `EnvironmentManager` — launches, tracks, validates, and shuts down environment processes.
- `ProfileManager` — validates and runs timed profile events.
- `QueueContainer` — stores the controller’s interprocess communication queues.
- `EventContainer` — stores readiness, active, shutdown, and heartbeat events.

### Important public methods

Some of the most important controller methods are:

- `initialize_hardware(...)` — validates and distributes hardware metadata to the acquisition, output, and environment processes.
- `initialize_environments(...)` — validates environment metadata, launches environment processes, and distributes environment configuration.
- `initialize_system_id(...)` — installs system identification metadata into a system-ID-capable environment.
- `start_acquisition(...)` — arms the acquisition and output processes and optionally initializes streaming.
- `stop_acquisition()` — gracefully stops acquisition, output, streaming, and active environments.
- `start_environment(...)` — starts an environment using an `EnvironmentInstructions` object.
- `stop_environment(...)` — requests graceful stop of a running environment.
- `start_system_id_noise(...)` — begins the noise phase of a system identification run.
- `start_system_id_transfer_function(...)` — begins the transfer-function phase of a system identification run.
- `save_system_id_to_file(...)` — saves the currently loaded system identification package to disk.
- `load_system_id_from_package(...)` — loads a `SysIdDataPackage` into an environment.
- `start_profile(...)` — schedules and begins a timed profile.
- `stop_profile()` — cancels pending profile timers and starts profile closeout.
- `shutdown()` — performs a full process shutdown of the controller and all child processes.

Together, these form the main headless API surface.

## Managers

### `EnvironmentManager`

`EnvironmentManager`, defined in `environment_manager.py`, manages the lifecycle of environment processes.

Its responsibilities include:

- assigning queue names to environments,
- launching environment processes,
- storing environment metadata by queue,
- managing environment ready/active/system-ID events,
- validating environment metadata and environment instructions, and
- checking compatibility when sharing system identification packages between environments.

Conceptually, `EnvironmentManager` maps user-facing environment names to internal queue/process identities.

### `ProfileManager`

`ProfileManager`, defined in `profile_manager.py`, manages timed test-profile execution.

Its responsibilities include:

- validating `ProfileEvent` objects,
- sorting events by timestamp,
- scheduling commands with `threading.Timer`,
- dispatching commands to the controller at the proper time, and
- issuing a profile closeout event when all scheduled events have fired.

A profile event is essentially a scheduled command with:
- a timestamp,
- an environment name,
- a command,
- and optional data.

## Interprocess Structure

Rattlesnake is deliberately split across multiple processes (or threads, in threaded mode) to separate concerns and keep time-sensitive operations responsive.

### Always-present controller processes

These processes exist independently of which environments are active.

- **Controller Process** — routes global commands to acquisition, output, streaming, or environment queues.
- **Acquisition Process** — reads data from the hardware layer and forwards it to active environments and the streaming process.
- **Output Process** — collects generated signals from active environments and writes them to output hardware.
- **Streaming Process** — writes streamed time-domain data and metadata to disk.
- **Log File Process** — serializes log messages from across the system into a single log file.

### Environment processes

Each initialized environment gets its own process, launched by the `EnvironmentManager`.

Examples include:

- `RandomVibrationEnvironment` — closed-loop random vibration control using system identification.
- `SineEnvironment` — swept sine environment with prediction and control logic.
- `TransientEnvironment` — transient waveform environment using system identification.
- `SDSEnvironment` — shock replication using sum-of-decayed-sines signals.
- `ModalEnvironment` — modal testing environment for burst/random/chirp/sine excitations and FRF estimation.
- `TimeEnvironment` — direct playback of a specified time history.
- `ReadEnvironment` — read-only environment for measuring incoming data without generating outputs.

### Environment-owned subprocesses

Many environments, especially system-ID-based ones, also start helper subprocesses such as:

- **signal generation** — converts control results into writeable time histories,
- **data collection** — breaks raw acquisition data into measurement frames,
- **spectral processing** — computes FRFs, CPSDs, coherence, and related frequency-domain quantities,
- **data analysis** — performs system identification updates or control-law calculations.

The exact set depends on the environment.

For example:
- Random, Sine, Transient, and SDS all use the `SysIdEnvironment` pattern and therefore own several subordinate subprocesses.
- Modal uses collector, signal generation, and spectral processing, but is not a `SysIdEnvironment`.
- Time and Read are simpler and do not need the same analysis stack.

## Queues and Events

Rattlesnake uses a queue-and-event architecture extensively.

### Queues

Queues carry:
- commands,
- time-domain data,
- spectral data,
- generated output data,
- GUI update instructions,
- and logging messages.

The main queue container is `QueueContainer`, defined in `utilities.py`.

It stores:

- `controller_command_queue` — commands routed to the controller process,
- `acquisition_command_queue` — commands routed to the acquisition process,
- `output_command_queue` — commands routed to the output process,
- `streaming_command_queue` — commands routed to the streaming process,
- `gui_update_queue` — GUI update messages sent back toward the UI,
- `environment_command_queues` — one command queue per environment,
- `environment_data_in_queues` — one acquisition-data queue per environment,
- `environment_data_out_queues` — one output-data queue per environment.

### `VerboseMessageQueue`

`VerboseMessageQueue`, also defined in `utilities.py`, is a queue wrapper that automatically logs queue operations. This makes command routing and process debugging much easier.

### Events

Events are used for synchronization and state tracking. These are stored in `EventContainer`, also in `utilities.py`.

Important examples include:

- `controller_ready_event` — controller process has acknowledged a command.
- `acquisition_ready_event` — acquisition process has acknowledged configuration.
- `output_ready_event` — output process has acknowledged configuration.
- `streaming_ready_event` — streaming process has acknowledged configuration.
- `environment_ready_events` — per-environment readiness events.
- `environment_active_events` — per-environment run-state events.
- `environment_sysid_active_events` — per-environment system identification active events.
- `environment_sysid_stored_events` — per-environment flags that a valid system identification package is available.
- `ping_alive_event` — heartbeat-like event used to avoid false timeout failures when a process is still working.

The UI, controller, and scenario/documentation tooling all rely on these events heavily.

## Hardware Layer

The hardware subsystem is built around abstract interfaces plus a registry.

### Core abstractions

Defined in `abstract_hardware.py`:

- `HardwareMetadata` — stores hardware configuration such as channel table, sample rate, read/write timing, and any backend-specific parameters.
- `HardwareAcquisition` — abstract interface for reading data from hardware.
- `HardwareOutput` — abstract interface for writing output data to hardware.

### Hardware registry

Defined in `hardware_registry.py`, this maps each `HardwareType` to:

- a metadata class,
- an acquisition class,
- and an output class.

### Hardware types

The `HardwareType` enumeration, defined in `hardware_utilities.py`, currently includes:

- `NI_DAQMX` — National Instruments DAQmx-backed hardware.
- `LAN_XI` — HBK LAN-XI hardware.
- `DP_QUATTRO` — Data Physics Quattro hardware.
- `DP_900` — Data Physics 900 series hardware.
- `EXODUS` — virtual hardware backed by an Exodus modal solution.
- `STATE_SPACE` — state-space-based virtual hardware.
- `SDYNPY_SYSTEM` — SDynPy-system-backed virtual hardware.
- `SDYNPY_FRF` — SDynPy-FRF-backed virtual hardware.
- plus skeleton and placeholder types used for development.

### Channel model

The `Channel` class defines the per-channel metadata used throughout the software, including:

- node number and direction,
- sensitivity,
- engineering unit,
- physical device and physical channel,
- feedback device and feedback channel,
- warning and abort levels.

Many parts of the system derive meaning from the channel table, so it is one of the most fundamental data structures in the codebase.

### Virtual hardware

Several virtual hardware implementations make it possible to exercise environments, UI flows, and documentation scenarios without real DAQ hardware.

Examples include:

- `state_space_virtual_hardware.py` — integrates linear state-space models.
- `sdynpy_system_virtual_hardware.py` — simulates response using SDynPy system data.
- `sdynpy_frf_virtual_hardware.py` — simulates response from FRFs.

These are especially useful for:
- examples,
- automated documentation,
- and offline control-law development.

## Environment Layer

Each environment encapsulates a specific testing workflow.

### Core abstractions

Defined in `abstract_environment.py`:

- `EnvironmentMetadata`
- `EnvironmentInstructions`
- `Environment`

### Metadata vs Instructions

This is an important conceptual separation in the codebase.

#### `EnvironmentMetadata`

Environment metadata describes the configured environment. It usually includes things like:

- enabled channels,
- sample rate assumptions,
- transformation matrices,
- specification data,
- environment definition settings,
- control-law configuration.

This data is relatively persistent and is typically associated with the Environment Definition tab in the UI.

#### `EnvironmentInstructions`

Environment instructions describe a specific run request. They usually include things like:

- test level,
- repeat flags,
- start/stop time selections,
- target hit counts,
- runtime behavior choices.

This data is often ephemeral and changes from run to run.

### Environment registry

Defined in `environment_registry.py`, this maps each `EnvironmentType` to:

- a command enum,
- a metadata class,
- an instructions class,
- an environment class,
- and a process function.

### Environment types

The `EnvironmentType` enumeration, defined in `environment_utilities.py`, currently includes:

- `RANDOM` — random vibration environment with control to a CPSD specification.
- `TRANSIENT` — transient waveform control environment using system identification.
- `SINE` — sine environment with specifications, prediction, and control.
- `SDS` — sum-of-decayed-sines shock environment.
- `MODAL` — modal testing environment for FRFs and related modal data products.
- `TIME` — direct playback of time histories.
- `READ` — passive read-only display and measurement environment.
- plus skeleton environments used for development and extension.

## System-ID Environment Layer

Many environments derive from `SysIdEnvironment`, defined in `abstract_sysid_environment.py`.

These environments share common system-identification machinery and therefore have a richer lifecycle than simpler environments like Time or Read.

### Key abstractions

- `SysIdEnvironmentMetadata` — an `EnvironmentMetadata` subclass with system-ID-related expectations.
- `SysIdEnvironment` — an `Environment` subclass that understands system identification startup, shutdown, save/load, and data flow.
- `SysIdMetadata` — metadata describing how a system identification run itself should be performed.
- `SysIdDataPackage` — a container for the results of a system identification run.

### `SysIdMetadata`

`SysIdMetadata` describes how the system identification run is performed. Important fields include:

- `sysid_frame_size` — the number of samples per frame used in spectral calculations.
- `sysid_averaging_type` — either linear or exponential averaging.
- `sysid_noise_averages` — number of frames used for the noise measurement.
- `sysid_averages` — number of frames used for transfer function estimation.
- `sysid_estimator` — FRF estimator choice such as H1, H2, H3, or Hv.
- `sysid_signal_type` — the signal type used for excitation, such as Random, Burst Random, Pseudorandom, or Chirp.
- `sysid_overlap` — overlap fraction used for system identification spectral processing.
- `sysid_low_frequency_cutoff` / `sysid_high_frequency_cutoff` — bandwidth limits for the system identification excitation.

### `SysIdDataPackage`

`SysIdDataPackage` stores the actual system identification results, including:

- `sysid_frf` — the measured frequency response function matrix.
- `sysid_coherence` — coherence or multiple coherence information.
- `sysid_response_cpsd` — response CPSD matrix from system identification.
- `sysid_reference_cpsd` — reference CPSD matrix from system identification.
- `sysid_response_noise` — response noise-floor CPSD.
- `sysid_reference_noise` — reference noise-floor CPSD.
- `frequencies` — frequency line vector associated with the data.

This package is what gets:
- saved to disk,
- loaded back in,
- and optionally shared across compatible environments.

### SysID-capable environments

Currently, the system-ID-capable environments include:

- `RandomVibrationEnvironment` — uses system identification to perform CPSD-based closed-loop random vibration control.
- `SineEnvironment` — uses system identification to predict and control drive signals for sine specifications.
- `TransientEnvironment` — uses system identification to derive drive signals that replicate a desired transient response.
- `SDSEnvironment` — uses system identification as the basis for SDS control and prediction.
- `SkeletonSysIdEnvironment` — a minimal template environment used for development.

## User Interface Layer

The UI is built with Qt and organized around one main application shell plus per-environment UI classes.

### Main UI

Defined in `user_interface.py`:

- `RattlesnakeUI` — the main application window.
- `build_rattlesnake_app(...)` — helper for constructing the QApplication and the main UI.
- `RattlesnakeAppHandle` — convenience wrapper for returning the controller, UI, and app together.

The main UI owns the top-level tabs and manages:

- hardware setup,
- environment definition,
- system identification,
- prediction,
- profile execution,
- run/test execution,
- and miscellaneous dialogs.

### Per-environment UIs

Each environment has a dedicated UI class. Examples include:

- `RandomVibrationUI` — user interface for MIMO random vibration.
- `TransientUI` — user interface for transient waveform control.
- `SineUI` — user interface for sine testing.
- `SDSUI` — user interface for sum-of-decayed-sines shock control.
- `ModalUI` — user interface for modal testing.
- `TimeUI` — user interface for direct time-history playback.
- `ReadUI` — user interface for read-only data viewing.

These are registered in `ui_registry.py`.

### Shared UI abstraction classes

Defined in:

- `abstract_user_interface.py`
- `abstract_sys_id_user_interface.py`

These provide shared logic for:

- hardware/environment initialization,
- starting and stopping environments,
- system ID start/stop flows,
- GUI update handling,
- event-watcher-based synchronization,
- and throttled plot updates.

### GUI update flow

Subprocesses send messages into the `gui_update_queue`. The `RattlesnakeUI` updater receives them and routes them either:

- to the main UI, or
- to the appropriate environment UI.

This is how:
- control results,
- system ID plots,
- run-time data,
- and dialog/UI state changes

flow back into the graphical interface.

## Profile / Timed Event System

Profiles are scheduled test procedures built from `ProfileEvent` objects.

Each `ProfileEvent` stores:

- a timestamp,
- an environment name,
- a command,
- and optional data.

Examples include:
- starting or stopping an environment,
- changing a test level,
- starting or stopping streaming,
- and issuing environment-specific commands at specific times.

### `ProfileManager`

The `ProfileManager`, defined in `profile_manager.py`, is responsible for:

- validating profile events,
- ensuring event data matches the expected command type,
- scheduling command execution using timers,
- firing commands into the controller command queue,
- and sending a final profile closeout event when the profile has completed.

This timed-event system is one of the main mechanisms that bridges:
- GUI-driven profile use, and
- headless automated operation.

## Streaming and File I/O

### Streaming

The `streaming.py` module writes time-domain data and metadata to netCDF files during acquisition.

It is responsible for:

- creating stream files,
- creating additional streams within a file when needed,
- writing incoming data blocks,
- and closing out the file cleanly.

### Metadata and template loading

The `load_utilities.py` module handles:

- saving controller state to workbook templates,
- loading hardware and environment metadata from workbook templates,
- loading metadata from netCDF,
- saving and loading profile event lists.

### Environment-specific persistence

Most environments implement methods such as:

- `save_metadata_to_netcdf(...)`
- `load_metadata_from_netcdf(...)`
- `create_blank_worksheet_template(...)`
- `save_metadata_to_worksheet(...)`
- `load_metadata_from_worksheet(...)`

These methods are important extension points when adding new environments.

### System identification persistence

System identification packages can be saved to and loaded from multiple formats, including:

- netCDF (`.nc4`)
- NumPy archives (`.npz`)
- MATLAB files (`.mat`)

This makes it possible to:
- reuse system identification across environments,
- archive test setup and results,
- and support offline/debug workflows.

## Data Flow Walkthroughs

This section sketches a few common controller flows.

### Hardware initialization flow

1. A UI or headless caller builds a `HardwareMetadata` object.
2. `RattlesnakeController.initialize_hardware(...)` validates it.
3. The metadata is sent to:
   - acquisition process,
   - output process,
   - and any existing environment processes.
4. Ready events confirm the configuration was received.

### Environment initialization flow

1. A UI or headless caller builds one `EnvironmentMetadata` object per environment.
2. `RattlesnakeController.initialize_environments(...)` validates them.
3. `EnvironmentManager` starts any needed environment processes.
4. Metadata is sent to:
   - acquisition,
   - output,
   - and the environment process itself.

### System identification flow

1. A caller builds `SysIdMetadata`.
2. The controller sends it to the target environment.
3. Acquisition starts.
4. Noise measurement runs.
5. Transfer-function measurement runs.
6. A `SysIdDataPackage` is produced.
7. The UI and/or environment use the resulting data for prediction or control.

### Environment run flow

1. A caller builds `EnvironmentInstructions`.
2. The controller sends `START_ENVIRONMENT`.
3. The output process activates that environment.
4. The environment process starts its signal/control loop.
5. Acquisition collects data and routes it back to the active environment.
6. GUI updates and control updates are produced.
7. Stop/shutdown eventually clears active events and drains queues.

### UI update flow

1. A subprocess sends a GUI update message to `gui_update_queue`.
2. `RattlesnakeUI` receives the message.
3. It routes the message either:
   - to itself, or
   - to the appropriate environment UI.
4. The receiving UI updates:
   - plots,
   - widgets,
   - dialogs,
   - or tables.

## Headless / API Usage

Headless use means interacting with Rattlesnake through Python code rather than through the GUI.

The same internal architecture is still used:
- controller,
- subprocesses,
- metadata classes,
- instruction classes,
- queues and events.

The main difference is that the caller is now responsible for:
- constructing metadata objects,
- invoking controller methods in the correct order,
- and optionally managing timing/wait behavior itself.

A typical headless flow looks like:

1. create a `RattlesnakeController`,
2. initialize hardware,
3. initialize environments,
4. initialize system identification if needed,
5. start acquisition,
6. start one or more environments,
7. wait for completion or stop them explicitly,
8. stop acquisition,
9. shut the controller down.

## Extending the Codebase

### Adding new hardware

Adding new hardware usually requires:

- a new `HardwareMetadata` subclass — to store configuration and validate it,
- a new `HardwareAcquisition` subclass — to read from the hardware,
- a new `HardwareOutput` subclass — to write to the hardware,
- a new `HardwareType` enum entry — to identify the backend,
- and registration in `hardware_registry.py` — so the controller can instantiate it.

If the hardware should be available through the GUI, it also needs associated UI registration in `ui_registry.py`.

See @sec:new_hardware for more information.

### Adding a new environment

Adding a new environment usually requires:

- a new `EnvironmentMetadata` subclass — to describe the configured environment,
- a new `EnvironmentInstructions` subclass — to describe a particular run request,
- a new `Environment` subclass — to implement environment behavior,
- a process entry function — to launch the environment process,
- a new `EnvironmentType` enum entry — to identify the environment,
- and registry updates in `environment_registry.py` — so the controller can create and manage it.

If the environment is user-facing, it also needs a UI class and registration in `ui_registry.py`.

See @sec:new_environment for more information.

### Adding a new system-ID environment

If the environment depends on system identification, it will generally be easier to derive from:

- `SysIdEnvironmentMetadata`
- and `SysIdEnvironment`

rather than starting from the plain environment base classes.

These environments typically also reuse:
- data collection,
- spectral processing,
- signal generation,
- and system ID data-analysis subprocesses.

### Adding a custom control law

Custom control laws depend on the environment:

- Random control laws operate on CPSD and FRF data to produce a drive CPSD matrix.
- Sine control laws operate on specifications, amplitudes, phases, and drive updates to produce sinusoidal drive signals.
- Transient control laws operate on transient response signals and FRF data to produce a drive time signal.
- SDS control laws operate on SRS and produce sum-of-decayed-sines drive signals.

Some environments also support **interactive control laws** via:

- `AbstractControlLawComputation`
- and related UI-side abstractions

which allow a custom UI component to exchange parameters and results with the environment process.

### Adding UI documentation scenarios

The UI documentation generation system relies on:
- example scenarios that create meaningful UI states,
- stateful page rendering,
- and dialog/page-state hooks.

If new UI pages or dialogs are added, documentation generation may need:
- a new example/scenario,
- a new page state,
- or an updated dialog-opening helper.

## Suggested Reading Order for New Developers

A good order to read the code is:

1. `main.py` — to see how the application starts.
2. `engine.py` — to understand the controller and lifecycle.
3. `controller.py` — to understand how global commands are routed.
4. `environment_manager.py` — to understand environment process ownership and validation.
5. `abstract_environment.py` — to understand environment metadata/instruction/process abstractions.
6. `abstract_sysid_environment.py` — to understand how system-ID-capable environments extend the base model.
7. `abstract_user_interface.py` — to understand the general UI/controller interface.
8. `abstract_sys_id_user_interface.py` — to understand shared system-ID UI flows.
9. One simple environment, such as:
   - `time_environment.py`, or
   - `read_environment.py`
10. One system-ID environment, such as:
   - `random_vibration_sys_id_environment.py`, or
   - `sine_sys_id_environment.py`
11. One virtual hardware backend, such as:
   - `state_space_virtual_hardware.py`

That reading order gives a good progression from orchestration to abstractions to concrete examples.

## Closing Remarks

Rattlesnake is a modular but process-oriented codebase. Understanding the queues, events, metadata objects, and environment lifecycle is more important than memorizing any single file.

A good mental model is:

- **metadata defines configuration**
- **instructions define a run**
- **the controller orchestrates**
- **processes perform work**
- **environments encapsulate test-specific behavior**
- **the UI is a client of the same controller model**
- **headless use follows the same architecture, just without the Qt layer**

This document is intended as a starting point for contributors and maintainers, and should evolve as the codebase evolves.