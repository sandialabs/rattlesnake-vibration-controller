# Ranking bugs based on time/ease of fix
Stuff to do in order
[ ] Add in new ping alive events into lanxi intialize hardware
[ ] Make Unittests
[ ] Clean up documentation
[ ] Write guides

Stuff for the future
[ ] Clear jupyter notebooks automatically on push
[ ] Make Unittests/documentation
[ ] Build out notebook example
[ ] Write guide for building new environments

## Easy
[ ] Add in new ping alive events into lanxi intialize hardware

[ ] Make git repo clear all jupyter notebook outputs when pushing to repo
    [ ] There is a pip install nbstripout that works for this: https://github.com/kynan/nbstripout

[ ] Unittests
    [ ] Finish making standard structure for unittests.
    [ ] Go through and comment/verify unit tests.

[ ] Build out notebook example.

[ ] Write guide for rattlesnake architecture
    [ ] Compile list of need to knows for making environments
    [ ] Compile list of need to knows for making hardware
    [ ] Compile list of need to knows for processes

## Hard (Major issues are going to happen)
[ ] There is a potential problem with LanXi where the valid_physical_devices does not fill up on initialize_hardware

[ ] Figure out issue with changing font size in jupyter notebooks.
    [ ] A lot of the time this is an import order issue.
    [ ] Can be fixed by inserting font sizes in __init__.py of rattlesnake but I really dont want to do this.

[ ] Certain load_sysid files (forcefinder, etc.) do not contain all of the information that is usually sent to control laws.

## Impossible
[ ] This is an internal firmware issue with reconnecting generator sockets with IPv6. BNK would need to fix this in order for it to work.
    [ ] Should probably create a notebook to recreate this issue for BK issues tab

[ ] A dll in cvxpy import clashes with a dll in pyqt so if cvxpy must be imported before pyqt
    [ ] Has issues with some custom control laws

[ ] Profile events can error out when going to fast. (ex. Stop Environment then immediate Start Environment)
    [ ] Should probably just stop profile events from firing if this happens.
    [ ] Need to figure out what the desired logic is here. Should the profile manager check for startup/shutdown environments

[ ] Threaded Environments
    [ ] I need to swap queues within environments to threaded queues. This has caused major lag issues which need investigation
    [ ] This basically crashes the system identification process

[ ] Some hardware files need time.sleeps to work at certain points in headless mode (LanXi)

## Unit Tests/Documentation
Test [ ] Test Documented [ ] Code [ ] Code Documented [ ]

[ ] [ ] [ ] [ ] Main
    [ ] [ ] [ ] [ ] main
    [ ] [ ] [ ] [ ] engine
    [ ] [ ] [ ] [ ] environment_manager
    [ ] [ ] [ ] [ ] profile_manager
    [ ] [ ] [ ] [ ] utilities
    [ ] [ ] [ ] [ ] load_utilities
[ ] [ ] [ ] [ ] Hardware
    [ ] [ ] [ ] [ ] hardware_utilities
    [ ] [ ] [ ] [ ] abstract_hardware
    [ ] [ ] [ ] [ ] data_physics_dp900_hardware
    [ ] [ ] [ ] [ ] data_physics_dp900_interface
    [ ] [ ] [ ] [ ] data_physics_hardware
    [ ] [ ] [ ] [ ] data_physics_interface
    [ ] [ ] [ ] [ ] exodus_modal_solution_hardware
    [ ] [ ] [ ] [ ] lanxi_hardware_multiprocess
    [ ] [ ] [ ] [ ] lanxi_stream
    [ ] [ ] [ ] [ ] nidaqmx_hardware_multitask
    [ ] [ ] [ ] [ ] sdynpy_frf_virtual_hardware
    [ ] [ ] [ ] [ ] sdynpy_system_virtual_hardware
    [ ] [ ] [ ] [ ] state_space_virtual_hardware
[ ] [ ] [ ] [ ] Environment
    [ ] [ ] [ ] [ ] environment_utilities
    [ ] [ ] [ ] [ ] environment_registry
    [ ] [ ] [ ] [ ] abstract_environment
    [ ] [ ] [ ] [ ] time_environment
    [ ] [ ] [ ] [ ] modal_environment
    [ ] [ ] [ ] [ ] abstract_sys_id_environment
    [ ] [ ] [ ] [ ] skeleton_sys_id_environment
    [ ] [ ] [ ] [ ] sine_sys_id_environment
    [ ] [ ] [ ] [ ] sine_sys_id_utilities
    [ ] [ ] [ ] [ ] transient_sys_id_environment
    [ ] [ ] [ ] [ ] random_vibration_sys_id_environment
    [ ] [ ] [ ] [ ] abstract_control_law
    [ ] [ ] [ ] [ ] abstract_interactive_control_law
[ ] Process
    [ ] [ ] [ ] [ ] abstract_message_process
    [ ] [ ] [ ] [ ] abstract_sysid_data_analysis
    [ ] [ ] [ ] [ ] acquisition
    [ ] [ ] [ ] [ ] output
    [ ] [ ] [ ] [ ] controller
    [ ] [ ] [ ] [ ] data_collector
    [ ] [ ] [ ] [ ] random_vibration_sys_id_data_analysis
    [ ] [ ] [ ] [ ] signal_generation_process
    [ ] [ ] [ ] [ ] signal_generation
    [ ] [ ] [ ] [ ] spectral_processing
    [ ] [ ] [ ] [ ] streaming
[ ] [ ] [ ] [ ] User Interface
    [ ] [ ] [ ] [ ] user_interface
    [ ] [ ] [ ] [ ] ui_utilities
    [ ] [ ] [ ] [ ] ui_registry
    [ ] [ ] [ ] [ ] abstract_user_interface
    [ ] [ ] [ ] [ ] time_ui
    [ ] [ ] [ ] [ ] modal_ui
    [ ] [ ] [ ] [ ] abstract_sys_id_user_interface
    [ ] [ ] [ ] [ ] skeleton_sys_id_ui
    [ ] [ ] [ ] [ ] sine_sys_id_ui
    [ ] [ ] [ ] [ ] sine_sys_id_ui_utilities
    [ ] [ ] [ ] [ ] transient_sys_id_ui
    [ ] [ ] [ ] [ ] random_vibration_sys_id_ui
    [ ] [ ] [ ] [ ] random_vibration_sys_id_ui_utilities
