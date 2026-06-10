import sys

from qtpy import QtWidgets, QtCore

from rattlesnake.engine import RattlesnakeController
from rattlesnake.user_interface.user_interface import RattlesnakeUI


def launch_temporary_rattlesnake_ui_profile(
    rattlesnake: RattlesnakeController, closeout_time: float
):
    """
    Function for launching rattlesnake ui with the default formatting
    that scales correcctly.

    Parameters
    ----------
    rattlesnake : RattlesnakeController
        The rattlesnake controller object that the UI is going to represent.
    """
    # Fix to scale font for different size monitors
    font_size = 10  # pt size
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):  # PyQt5 only
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):  # PyQt5 only
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Build app
    app = QtWidgets.QApplication(sys.argv)

    # Scale app to current monitor
    screen = app.primaryScreen()
    dpi = screen.logicalDotsPerInch()
    scale_factor = dpi / 96  # 96 DPI
    font = app.font()
    font.setPointSizeF(font_size * scale_factor)  # base font is 12pt
    app.setFont(font)

    # Execute UI object
    ui = RattlesnakeUI(rattlesnake)
    QtCore.QTimer.singleShot(int(closeout_time * 1000), ui.close)
    profile_start = 20
    QtCore.QTimer.singleShot(int(profile_start * 1000), ui.start_profile)
    app.exec_()

    # Shutdown processes
    rattlesnake.shutdown()


def launch_temporary_rattlesnake_ui_environment(
    rattlesnake: RattlesnakeController, closeout_time: float
):
    """
    Function for launching rattlesnake ui with the default formatting
    that scales correcctly.

    Parameters
    ----------
    rattlesnake : RattlesnakeController
        The rattlesnake controller object that the UI is going to represent.
    """
    # Fix to scale font for different size monitors
    font_size = 10  # pt size
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):  # PyQt5 only
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):  # PyQt5 only
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Build app
    app = QtWidgets.QApplication(sys.argv)

    # Scale app to current monitor
    screen = app.primaryScreen()
    dpi = screen.logicalDotsPerInch()
    scale_factor = dpi / 96  # 96 DPI
    font = app.font()
    font.setPointSizeF(font_size * scale_factor)  # base font is 12pt
    app.setFont(font)

    # Execute UI object
    ui = RattlesnakeUI(rattlesnake)
    QtCore.QTimer.singleShot(int(closeout_time * 1000), ui.close)
    environment_end = 40
    QtCore.QTimer.singleShot(int(environment_end * 1000), ui.stop_acquisition)
    app.exec_()

    # Shutdown processes
    rattlesnake.shutdown()
