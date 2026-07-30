# Ranking bugs based on time/ease of fix

Stuff to do in order
[ ] Add buffer to other user interface plots
[x] Add in patch to LanXi IP Manager
[x] Add in IPv4 override on LanXi Generator Sockets
[ ] Build out read environment

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

[ ] Headless example goes to run tab when sysid enviromnet does not have a sysid loaded to it
    [ ] Should swap this to use with ... as rattlesnake

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
[ ] This is an internal firmware issue with reconnecting generator sockets with IPv6. BNK would need to fix this in order for it to work

[ ] Certain load_sysid files (forcefinder, etc.) do not contain all of the information that is usually sent to control laws

[ ] Profile events can error out when going to fast. (ex. Stop Environment then immediate Start Environment)
    [ ] Should probably just stop profile events from firing if this happens.
    [ ] Need to figure out what the desired logic is here. Should the profile manager check for startup/shutdown environments

[ ] Threaded Environments
    [ ] I need to swap queues within environments to threaded queues. This has caused major lag issues which need investigation
    [ ] This basically crashes the system identification process

[ ] Some hardware files need time.sleeps to work at certain points in headless mode (LanXi)

## Done
[x] Figure out issue with closing generator sockets in lanxi hardware multiprocessing
[x] LanXI IP Manager
    [x] This appends every channel to the ip address list instead of just the unique ones.
        [x] I kinda want to just add a __eq__ to ip addresses and have it as a set