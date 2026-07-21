# -*- coding: utf-8 -*-
"""
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
from rattlesnake.user_interface.user_interface import RattlesnakeUI


def build_rattlesnake_app(rattlesnake: RattlesnakeController):
    # Configure High DPI for UI scaling
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):  # PyQt5 only
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):  # PyQt5 only
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Build app
    app = QtWidgets.QApplication(sys.argv)

    # Set font size. This can be commented out if it is extremely small
    font_size = 10  # pt size
    screen = app.primaryScreen()
    dpi = screen.logicalDotsPerInch()
    scale_factor = dpi / 96  # 96 DPI
    font = app.font()
    font.setPointSizeF(font_size * scale_factor)  # base font is 12pt
    app.setFont(font)

    app.rattlesnake_controller = rattlesnake
    app.rattlesnake_ui = RattlesnakeUI(rattlesnake)

    return app


def launch_rattlesnake_ui(rattlesnake: RattlesnakeController):
    """
    Function for launching rattlesnake ui with the default formatting
    that scales correcctly.

    Parameters
    ----------
    rattlesnake : RattlesnakeController
        The rattlesnake controller object that the UI is going to represent.
    """
    app = build_rattlesnake_app()
    app.exec_()
    rattlesnake.shutdown()


def main():
    """Main Rattlesnake Application Entry Point"""
    print("Loading Rattlesnake...")

    rattlesnake = RattlesnakeController()

    launch_rattlesnake_ui(rattlesnake)


if __name__ == "__main__":
    main()
