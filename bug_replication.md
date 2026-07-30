# Ranking bugs based on time/ease of fix

Stuff to do in order
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
[ ] Transient example problem results in zero division at the end which sends TRAC values to 0
    [ ] Look into behavior of having a 0 in the transient and random spec

## Easy
[ ] Make git repo clear all jupyter notebook outputs when pushing to repo
    [ ] There is a pip install nbstripout that works for this: https://github.com/kynan/nbstripout

[ ] Add in new ping alive events into lanxi intialize hardware

[ ] Unittests
    [ ] Finish making standard structure for unittests.
    [ ] Go through and comment/verify unit tests.

[ ] Build out notebook example.

[ ] Skeleton/Read environments take a suspiciously long time to intialize enviornment

[ ] Finish implementation of Read Environment.
    [ ] Add callbacks to when the window spinbox value is changed
    [ ] Edit the profile commands to block signals of that spinbox
    [ ] Add logic to the plot enabled table

[ ] Write guide for building out new environments.

[ ] Add units to each of the plots.

## Medium (Possible but needs a lot of verification)

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
[ ] Weird BrokenPipeError when rattlesnake.shutdown with LanXi

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
