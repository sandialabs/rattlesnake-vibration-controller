# -*- coding: utf-8 -*-
"""
Main Rattlesnake application entry point.

This module defines the functions used to launch the Rattlesnake graphical
user interface, either from the ``Rattlesnake`` console script or from a
Python script that has already constructed and configured a controller.

Rattlesnake Vibration Control Software
Copyright (C) 2021  National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import datetime
import multiprocessing as mp
import sys

from qtpy import QtWidgets, QtCore

# from rattlesnake.process.streaming import streaming_process
from rattlesnake.engine import RattlesnakeController
from rattlesnake.user_interface.user_interface import (
    build_rattlesnake_app,
)


def launch_rattlesnake_ui(rattlesnake: RattlesnakeController):
    """
    Launches the Rattlesnake user interface for an existing controller.

    This starts up an execution loop for the main QtApplication and should be used
    within python scripts and not jupyter notebooks. The user interface and
    controller will shut down automatically when the user interface closes.

    Parameters
    ----------
    rattlesnake : RattlesnakeController
        The controller that the user interface will represent. It may be blank
        or already configured with hardware and environment metadata.

    Unit Tests
    ----------
    """
    with build_rattlesnake_app(rattlesnake) as (rattlesnake, rattlesnake_ui, app):
        rattlesnake_ui.show()
        app.exec_()


def main():
    """
    Starts the Rattlesnake application.

    This is the target of the ``Rattlesnake`` command line entry point

    Unit Tests
    ----------
    """
    print("Loading Rattlesnake...")

    rattlesnake = RattlesnakeController()

    launch_rattlesnake_ui(rattlesnake)


if __name__ == "__main__":
    main()
