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
from rattlesnake.rattlesnake import RattlesnakeController
from rattlesnake.process.streaming import streaming_process
import datetime
import multiprocessing as mp
import sys
from qtpy import QtWidgets


def main():
    """Main Rattlesnake Application Entry Point"""

    rattlesnake = Rattlesnake(threaded=True, timeout=30)

    from rattlesnake.user_interface.example_files.metadata import (
        make_sdynpy_system_metadata,
        make_time_environment_metadata,
        make_time_environment_event_list,
        make_time_environment_stream_metadata,
        make_time_environment_instructions,
        make_modal_environment_metadata,
        make_sine_environment_metadata,
    )

    hardware_metadata = make_sdynpy_system_metadata()
    time_environment_metadata = make_time_environment_metadata(hardware_metadata)
    time_profile_event_list = make_time_environment_event_list()
    time_stream_metadata = make_time_environment_stream_metadata()
    time_environment_instructions = make_time_environment_instructions()
    modal_environment_metadata = make_modal_environment_metadata(hardware_metadata)
    sine_environment_metadata = make_sine_environment_metadata(hardware_metadata)

    rattlesnake.set_hardware(hardware_metadata)
    # Time Environment
    # rattlesnake.set_environments([time_environment_metadata])
    # rattlesnake.set_profile_event_list(time_profile_event_list)
    # rattlesnake.set_stream_metadata(time_stream_metadata)
    # rattlesnake.start_acquisition(time_stream_metadata)
    # rattlesnake.start_environment(time_environment_instructions)
    # Modal Environment
    # rattlesnake.set_environments([modal_environment_metadata])
    # # Sine Environment
    # rattlesnake.set_environments([sine_environment_metadata])

    # This is a fix for scaling Rattlesnake to different resolution monitors
    font_size = 10  # pt size
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QtWidgets.QApplication(sys.argv)
    screen = app.primaryScreen()
    dpi = screen.logicalDotsPerInch()
    scale_factor = dpi / 96  # 96 DPI = standard
    font = app.font()
    font.setPointSizeF(font_size * scale_factor)  # base font 12pt
    app.setFont(font)
    _ = RattlesnakeUI(rattlesnake)
    app.exec_()

    rattlesnake.shutdown()


if __name__ == "__main__":
    main()
