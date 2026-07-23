# Ranking bugs based on time/ease of fix

## Easy
[ ] Rattlesnake object and user interface need methods for when they are overwritten
    [ ] Currently just crashes

[ ] Add a load environment button to user interface

[ ] Modal environment throws a zero division error when there are no references selected
    [ ] Dont enable FRF/spectral stuff with no references/do a try/catch

[ ] Transient example problem results in zero division at the end which sends TRAC values to 0
    [ ] Look into behavior of having a 0 in the transient and random spec

[ ] LanXI IP Manager
    [ ] This appends every channel to the ip address list instead of just the unique ones.
        [ ] I kinda want to just add a __eq__ to ip addresses and have it as a set

[ ] Unittests
    [ ] Finish making standard structure for unittests
    [ ] Go through and comment/verify unit tests

[ ] Build out notebook example

[ ] Finish implementation of Read Environment

[ ] Write guide for building out new environments

## Medium (Possible but needs a lot of verification)
[ ] Figure out how to cleanly close out Rattlesnake object when overwriting an existing rattlesnake object

[ ] Renaming Environments
    [ ] I want to just pop up a dialog box on add_environment with a name. Disable double click renaming stuff
    [ ] Spin up/shut down environments based off name, not type

[ ] Add a save spec to sine environment

[ ] The modal FRF for the hammer hits only show after acceptance. Should probably show individual one before accepting then average.

[ ] Profile events can error out when going to fast. (ex. Stop Environment then immediate Start Environment)
    [ ] Should probably just stop profile events from firing if this happens.

[ ] If the Acqusition input/output never syncs, the program is unclosable/stop_acqusition does not stop this behavior.
    [ ] Stop Acqusition should be able to stop the input/output sync loop.

## Hard (Major issues are going to happen)
[ ] Editable spinboxes tab out when inputting data.

[ ] Remove Set Environment Instructions from the list of options in the profile event list user interface.
    [ ] This is an internal event done to sync the environment user interface before performing the list.

[ ] Threaded Environments
    [ ] I need to swap queues within environments to threaded queues. This has caused major lag issues which need investigation

[ ] Allow for SysID data packages to be sent to other environments that share the same control/channel_table

[ ] Memory Leak
    [ ] Headless mode needs to clear the gui_update_queue at a certain size
        [ ] This has had major issues with the sine environment example for some reason

## Impossible
[ ] Figure out issue with closing generator sockets in lanxi hardware multiprocessing
