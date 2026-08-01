# Ranking bugs based on time/ease of fix
Stuff to do in order
[ ] Figure out what the sysid sharing user interface should look like
[ ] Look for and document/recreate warnings

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
[ ] Transient example problem results in zero division at the end which sends TRAC values to 0
    [ ] Look into behavior of having a 0 in the transient and random spec

[ ] Run Sysid results in zero division with virtual hardware due to no noise signal

[ ] Reciprocity plots show warning in example. Not sure when reciprocity is meant to be shown

## Easy
[ ] Add units to each of the plots.
    [ ] Add units to example problems
    [ ] Check units in the plots

[ ] Make git repo clear all jupyter notebook outputs when pushing to repo
    [ ] There is a pip install nbstripout that works for this: https://github.com/kynan/nbstripout

[ ] Add in new ping alive events into lanxi intialize hardware

[ ] Unittests
    [ ] Finish making standard structure for unittests.
    [ ] Go through and comment/verify unit tests.

[ ] Build out notebook example.

[ ] Write guide for building out new environments.

## Hard (Major issues are going to happen)
[ ] Figure out issue with changing font size in jupyter notebooks.
    [ ] A lot of the time this is an import order issue.
    [ ] Can be fixed by inserting font sizes in __init__.py of rattlesnake but I really dont want to do this.

[ ] Remove Set Environment Instructions from the list of options in the profile event list user interface.
    [ ] This is an internal event done to sync the environment user interface before performing the list.

[ ] Allow for SysID data packages to be sent to other environments that share the same control/channel_table.
    [ ] Have some equality/function that checks if SysIdDataPackages can go with a certain environment
    [ ] When one sysid has completed, have the user interface ask if it wants to be stored to other environments
        [ ] Maybe have a dialog box to select other environments with a table and checkboxes for their names
    [ ] Make load_sys_id not require the environment to have the same name as the one that generated it

## Impossible
[ ] This is an internal firmware issue with reconnecting generator sockets with IPv6. BNK would need to fix this in order for it to work.
    [ ] Should probably create a notebook to recreate this issue for BK issues tab

[ ] Certain load_sysid files (forcefinder, etc.) do not contain all of the information that is usually sent to control laws.

[ ] Profile events can error out when going to fast. (ex. Stop Environment then immediate Start Environment)
    [ ] Should probably just stop profile events from firing if this happens.
    [ ] Need to figure out what the desired logic is here. Should the profile manager check for startup/shutdown environments

[ ] Threaded Environments
    [ ] I need to swap queues within environments to threaded queues. This has caused major lag issues which need investigation
    [ ] This basically crashes the system identification process

[ ] Some hardware files need time.sleeps to work at certain points in headless mode (LanXi)

## Done
[x] Testing suite requirements
    [x] Build out example rattlesnake objects/apps/uis with overrides where defaults are none/blank
    [x] Launch temporary UI's with a list of UIEvent objects?/dict with timestamps of buttons to hit
    [x] Build object/app/etc. should stay in headless __init__
        [x] Anything that would want to be imported into a jupyter notebook should be in that init
    [x] Save/load current rattlesnake states to a file.
    [x] Functions that compare metadata objects/netcdf files/worksheets.
[x] Skeleton/Read environments take a suspiciously long time to intialize enviornment
[x] Rattlesnake.log goes to the examples folder for headless_example
[x] Finish implementation of Read Environment.
    [x] Add callbacks to when the window spinbox value is changed
    [x] Edit the profile commands to block signals of that spinbox
    [x] Add logic to the plot enabled table
    [x] Add in enable all/disable all/enable/disable buttons
[x] Fix keyboard interrupt
[x] SDynPy System example throws an error when loaded from netcdf
[x] Weird BrokenPipeError when rattlesnake.shutdown with LanXi