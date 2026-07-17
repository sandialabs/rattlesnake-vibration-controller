## Stuff I need to Create
[ ] Unittests
    [ ] Finish making standard structure for unittests
    [ ] Go through and comment/verify unit tests

[ ] Set the default environment control laws for random and transient environment
    [ ] Just do this in the user interface

[ ] Change event list in the example modal worksheet to follow new format with change_filename

[ ] Write guide for building out new environments

[ ] Add a load environment button to user interface

[ ] Add a save spec to sine environment

[ ] Allow for SysID data packages to be sent to other environments that share the same control/channel_table

[ ] Build out skeleton sysid environment so that it has a working user interface

[ ] Finish implementation of Read Environment

[ ] Migrate metadata objects to rattlesnake.headless.__init__.py

[ ] The modal FRF for the hammer hits only show after acceptance. Should probably show individual one before accepting then average

[ ] Remove Set Environment Instructions from the list of options in the profile event list user interface
    [ ] This is an internal event done to sync the environment user interface before performing the list

## Code I need to Change but not Bugs
[ ] Renaming Environments
    [ ] I want to just pop up a dialog box on add_environment with a name. Disable double click renaming stuff
    [ ] Spin up/shut down environments based off name, not type

[ ] Threaded Environments
    [ ] I need to swap queues within environments to threaded queues. This has caused major lag issues which need investigation

[ ] Memory Leak
    [ ] Headless mode needs to clear the gui_update_queue at a certain size
        [ ] This has had major issues with the sine environment example for some reason

## Bugs I can make scripts to recreate (easy fixes)
[ ] Profile events can error out when going to fast. (ex. Stop Environment then immediate Start Environment)
    [ ] Should probably just stop profile events from firing if this happens

[ ] Transient example problem results in zero division at the end which sends TRAC values to 0

[ ] Start Profile > Stop Acqusition disables both start and stop profile button
    [ ] Timers also keep moving in the user interface

[ ] If the process crashes during system identification, the hardware_acquisiton never shuts down and you get stuck
    [ ] Repeatable by running system id with 0 frame size
    
[ ] For some reason, skeleton sysid never enables sysid_tab

## Bugs I need to Investigate (I have little hope of fixing)
[ ] LanXI IP Manager
    [ ] This appends every channel to the ip address list instead of just the unique ones.
        [ ] I kinda want to just add a __eq__ to ip addresses and have it as a set

[ ] Figure out issue with closing generator sockets in lanxi hardware multiprocessing

[ ] If the Acqusition input/output never syncs, the program is unclosable/stop_acqusition does not stop this behavior.

[ ] Editable spinboxes tab out when inputting data

## Bugs I believe I have already fixed (cope)
[ ] If you change the samples per frame in the modal template, it shifts the trigger wait for signal over?