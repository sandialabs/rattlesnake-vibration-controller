# Adaptive Buffer Stuff
The goal for the adaptive buffer size is to get the Rattlesnake software to a point where we can
benchmark it and make meaningful changes to optimize the program. The goals for this refactor are:
[x] Clean up control loops to allow for easier human readability
[x] Add in buffers to the user interface so that it does not crash when updating a lot
[ ] Add in adaptive buffer sizes/read sizes so that the program runs as fast as possible
    [ ] The way I want this to work is that each environment predicts the output for the next ~0.25 seconds
    [ ] The ammount of data each environment gets is based off the time per read which will change adaptively
    [ ] Each environment will output data to the output process everytime they get data comming in
    [ ] When the output process gets this data, it will overide the stale prediction
        [ ] Output will sync with input to see which input sample each output sample corresponds to
        [ ] Output will assume this does not vary over time
        [ ] The logic for staleness is as follows
            [ ] If acqusition sends 0.1 data to the environment
            [ ] Output is assuming that the last 0.1 data 
[ ] Build out a benchmarking tool that gets runtime statistics for each read/write loop
    [ ] I want this to ONLY do statistics when the environment is active so I can see exactly
        where bottlenecks are arrising.
    [ ] A Gaant chart like plot would be nice that shows the 95% CI of each process within each
        read/write cycle so it would only be like 0.25 seconds long but average each thing

# Ranking bugs based on time/ease of fix

Stuff to do in order
[ ] Investigate viability of adding benchmarking to the program.

Stuff to check
[ ] Figure out how to do sharing sysid data packages
[ ] Figure out notebooks on git repo

Stuff for the future
[ ] Clear jupyter notebooks automatically on push
[ ] Make Unittests/documentation
[ ] Build out notebook example
[ ] Build out read environment
[ ] Write guide for building new environments
[ ] Deal with minor/harder bugs

## Warnings
[ ] Every now and then, the sine environment does not show the spec on the run tab when launching form headless mode.

[ ] Transient example problem results in zero division at the end which sends TRAC values to 0
    [ ] Look into behavior of having a 0 in the transient and random spec

## Easy
[ ] Make git repo clear all jupyter notebook outputs when pushing to repo
    [ ] There is a pip install nbstripout that works for this: https://github.com/kynan/nbstripout

[ ] LanXI IP Manager
    [ ] This appends every channel to the ip address list instead of just the unique ones.
        [ ] I kinda want to just add a __eq__ to ip addresses and have it as a set

[ ] Unittests
    [ ] Finish making standard structure for unittests
    [ ] Go through and comment/verify unit tests

[ ] Build out notebook example

[ ] Finish implementation of Read Environment.

[ ] Write guide for building out new environments

[ ] Add units to each of the plots.


## Medium (Possible but needs a lot of verification)
[ ] The first plot data command from the sine environment when starting from headless mode causes an error.

## Hard (Major issues are going to happen)
[ ] Figure out issue with changing font size in jupyter notebooks.
    [ ] A lot of the time this is an import order issue
    [ ] Can be fixed by inserting font sizes in __init__.py of rattlesnake

[ ] Remove Set Environment Instructions from the list of options in the profile event list user interface.
    [ ] This is an internal event done to sync the environment user interface before performing the list.

[ ] Allow for SysID data packages to be sent to other environments that share the same control/channel_table.
    [ ] Have some equality/function that checks if SysIdDataPackages can go with a certain environment
    [ ] When one sysid has completed, have the user interface ask if it wants to be stored to other environments
        [ ] Maybe have a dialog box to select other environments with a table and checkboxes for their names
    [ ] Make load_sys_id not require the environment to have the same name as the one that generated it

## Impossible
[ ] Figure out issue with closing generator sockets in lanxi hardware multiprocessing

[ ] Certain load_sysid files (forcefinder, etc.) do not contain all of the information that is usually sent to control laws

[ ] Profile events can error out when going to fast. (ex. Stop Environment then immediate Start Environment)
    [ ] Should probably just stop profile events from firing if this happens.
    [ ] Need to figure out what the desired logic is here. Should the profile manager check for startup/shutdown environments

[ ] Threaded Environments
    [ ] I need to swap queues within environments to threaded queues. This has caused major lag issues which need investigation
    [ ] This basically crashes the system identification process

## Done
[x] Add a load environment button to user interface
[x] Modal environment throws a zero division error when there are no references selected
    [x] Dont enable FRF/spectral stuff with no references/do a try/catch
[x] Memory Leak
    [x] Headless mode needs to clear the gui_update_queue at a certain size
[x] Editable spinboxes tab out when inputting data.
[x] Figure out how to keep the current light mode theme but change the dark theme to have
    dark plots
[x] Figure out how to cleanly close out Rattlesnake object when overwriting an existing rattlesnake object
[x] System identification throws a streaming error since it is trying to start up after finishing a sysid run in headless mode
[x] Renaming Environments
    [x] I want to just pop up a dialog box on add_environment with a name. Disable double click renaming stuff
    [x] Spin up/shut down environments based off name, not type
    [x] Unique names need to be capitalization invariant as "Modal" and "MODAL" cannot be in the same workbook
[x] Sysid environments need to write spec filenames to worksheet if they exists.
[x] Add a save spec to sine environment
[x] The modal FRF for the hammer hits only show after acceptance. Should probably show individual one before accepting then average.
[x] If the Acqusition input/output never syncs, the program is unclosable/stop_acqusition does not stop this behavior.
    [x] Stop Acqusition should be able to stop the input/output sync loop.
[x] Prevent the log file task from writing to the same log file if two Controllers are open simultaneously
[x] Add assist mode to virtual hardware
    [x] Exodus
    [x] SDynPyFRF
    [x] StateSpace
[x] Figure out validation for each metadata object.
    [x] NI
    [x] LanXI
    [x] Exodus
    [x] SDynPyFRF
    [x] StateSpace
    [x] Time
    [x] Modal
    [x] Sine
    [x] Random
    [x] Transient