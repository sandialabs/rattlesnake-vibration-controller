from qtpy import QtCore, QtWidgets

# Set Qt Preferences
if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):  # PyQt5 only
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):  # PyQt5 only
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
    QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
)

# Enable Qt user interface within ipython kernels
try:
    from IPython import get_ipython

    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("gui", "qt")
except ImportError:
    pass
