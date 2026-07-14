# Unit Test Architecture

Rules to follow
- Fixtures
    - Put the functions that initialize fixtures in mock_utilities
    - Use fixtures when needing default/valid base class, use functions
    when needing to mutate base class
    - When using fixtures, use type hints 
- Abstract Classes
    - For relevant functions, test the subclasses for required behavior (set_ready, store required attributes, etc.)
        - Use instantiate with mocks for building subclasses. Get the subclasses from registries
    - Split verification tests into multiple tests that describe what the test is checking for
- 

## Main

## Hardware

### Hardware Types

## Environment

### Environment File
Fixtures
    HardwareMetadata
    EnvironmentMetdata
    EnvironmentInstructions
    EnvironmentQueues
    Environment
    

## Process

## User Interface

## Examples


# Unit Test Completeness
Completed
    abstract_environment
Good Tests
    
Has Tests
    engine
    environment_manager
    profile_manager
    utilities
    load_utilities
    hardware_utilities
    abstract_hardware
    environment_utilities
    environment_registry
    time_environment
    modal_environment
    abstract_sys_id_environment
    sine_sys_id_environment
    transient_sys_id_environment
    abstract_message_process
    acquisition
    output
    controller
    data_collector
    signal_generation_process
    signal_generation
    spectral_processing
    streaming
No Tests
    main
    data_physics_dp900_hardware
    data_physics_dp900_interface
    data_physics_hardware
    data_physics_interface
    exodus_modal_solution_hardware
    lanxi_hardware_multiprocess
    lanxi_stream
    nidaqmx_hardware_multitask
    sdynpy_frf_virtual_hardware
    sdynpy_system_virtual_hardware
    state_space_virtual_hardware
    skeleton_sys_id_environment
    sine_sys_id_utilities
    abstract_control_law
    abstract_interactive_control_law
    abstract_sysid_data_analysis
    random_vibration_sys_id_data_analysis
    user_interface
    ui_utilities
    ui_registry
    abstract_user_interface
    time_ui
    modal_ui
    abstract_sys_id_user_interface
    skeleton_sys_id_ui
    sine_sys_id_ui
    sine_sys_id_ui_utilities
    transient_sys_id_ui
    random_vibration_sys_id_ui
    random_vibration_sys_id_ui_utilities