# Ranking bugs based on time/ease of fix
Stuff to do in order
[ ] Look for and document/recreate warnings

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

[ ] Write guide for building out new environments.

## Hard (Major issues are going to happen)
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

## Done
[x] Figure out why some errors do not display on user interface (errors inside control laws)
[x] Reciprocity plots show warning in example. Not sure when reciprocity is meant to be shown
[x] Rework regions so they are nested
