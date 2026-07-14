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

## Tests I need example for
    utilities
    engine
environment_manager
main

hardware_utilities/registry
    abstract_hardware
sdynpy_system_virtual_hardware
lanxi_hardware

environment_utilities/registry
    abstract_environment
modal_environment
abstract_control_law

abstract_message_process
controller
acqusition
abstract_sysid_data_analysis

ui_utilities
user_interface
abstract_user_interface
modal_ui

control_law
qualification


# Unit Test Completeness
Has Tests and Documentation
    abstract_environment
Has Tests I Like
    abstract_hardware
    utilities
    engine
Has Tests
    environment_manager
    profile_manager
    load_utilities
    hardware_utilities
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