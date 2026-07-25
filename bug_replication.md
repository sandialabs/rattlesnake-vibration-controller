# Ranking bugs based on time/ease of fix

Stuff to do in order
[ ] Have modal environment display current frf/timehistory/etc. before accepting frame
[ ] Add a save spec to the sine environment
[ ] Have some equality/function that checks if SysIdDataPackages can go with a certain environment
    [ ] When one sysid has completed, have the user interface ask if it wants to be stored to other environments
        [ ] Maybe have a dialog box to select other environments with a table and checkboxes for their names
    [ ] Make load_sys_id not require the environment to have the same name as the one that generated it
[ ] Add validation to environment/hardware metadata objects

Stuff for the future
[ ] Clear jupyter notebooks automatically on push
[ ] Make Unittests/documentation
[ ] Build out notebook example
[ ] Build out read environment
[ ] Write guide for building new environments
[ ] Deal with font size changing issues

## Easy
[ ] Make git repo clear all jupyter notebook outputs when pushing to repo
    [ ] There is a pip install nbstripout that works for this: https://github.com/kynan/nbstripout

[x] Add a load environment button to user interface

[x] Modal environment throws a zero division error when there are no references selected
    [x] Dont enable FRF/spectral stuff with no references/do a try/catch

[ ] Transient example problem results in zero division at the end which sends TRAC values to 0
    [ ] Look into behavior of having a 0 in the transient and random spec

[ ] LanXI IP Manager
    [ ] This appends every channel to the ip address list instead of just the unique ones.
        [ ] I kinda want to just add a __eq__ to ip addresses and have it as a set

[ ] Unittests
    [ ] Finish making standard structure for unittests
    [ ] Go through and comment/verify unit tests

[ ] Build out notebook example

[ ] Figure out validation for each metadata object.

[ ] Finish implementation of Read Environment

[ ] Write guide for building out new environments

[x] Prevent the log file task from writing to the same log file if two Controllers are open simultaneously

## Medium (Possible but needs a lot of verification)
[x] Figure out how to cleanly close out Rattlesnake object when overwriting an existing rattlesnake object

[x] System identification throws a streaming error since it is trying to start up after finishing a sysid run
    in headless mode

[x] Renaming Environments
    [x] I want to just pop up a dialog box on add_environment with a name. Disable double click renaming stuff
    [x] Spin up/shut down environments based off name, not type
    [x] Unique names need to be capitalization invariant as "Modal" and "MODAL" cannot be in the same workbook

[ ] Sysid environments need to write spec filenames to worksheet if they exists.

[ ] Add a save spec to sine environment

[ ] The modal FRF for the hammer hits only show after acceptance. Should probably show individual one before accepting then average.

[x] If the Acqusition input/output never syncs, the program is unclosable/stop_acqusition does not stop this behavior.
    [x] Stop Acqusition should be able to stop the input/output sync loop.

[ ] Figure out issue with changing font size in jupyter notebooks.

[x] Figure out how to keep the current light mode theme but change the dark theme to have
    dark plots

[ ] The first plot data command from the sine environment when starting from headless mode causes an error.

[ ] Every now and then, the sine environment does not show the spec on the run tab when launching form headless mode

## Hard (Major issues are going to happen)
[x] Editable spinboxes tab out when inputting data.

[ ] Remove Set Environment Instructions from the list of options in the profile event list user interface.
    [ ] This is an internal event done to sync the environment user interface before performing the list.

[ ] Allow for SysID data packages to be sent to other environments that share the same control/channel_table

[x] Memory Leak
    [x] Headless mode needs to clear the gui_update_queue at a certain size

## Impossible
[ ] Figure out issue with closing generator sockets in lanxi hardware multiprocessing

[ ] Certain load_sysid files (forcefinder, etc.) do not contain all of the information that is usually sent to
control laws

[ ] Profile events can error out when going to fast. (ex. Stop Environment then immediate Start Environment)
    [ ] Should probably just stop profile events from firing if this happens.
    [ ] Need to figure out what the desired logic is here. Should the profile manager check for startup/shutdown environments

[ ] Threaded Environments
    [ ] I need to swap queues within environments to threaded queues. This has caused major lag issues which need investigation
    [ ] This basically crashes the system identification process
