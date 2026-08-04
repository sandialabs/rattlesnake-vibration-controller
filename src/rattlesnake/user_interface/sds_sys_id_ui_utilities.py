import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets
from qtpy.QtCore import Qt


class SDSPlotSRSWindow(QtWidgets.QDialog):
    def __init__(self, parent, ui, channel_index):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & Qt.Tool)

        self.ui = ui
        self.channel_index = channel_index

        layout = QtWidgets.QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showGrid(True, True, 0.25)
        self.plot_item.enableAutoRange()
        self.plot_item.getViewBox().enableAutoRange(enable=True)
        self.plot_item.setLogMode(True, True)
        self.plot_item.addLegend()

        self.spec_curve = self.plot_item.plot(
            np.array([0, 1]),
            np.nan * np.ones(2),
            pen={"color": "b", "width": 1},
            name="Specification",
        )
        self.lower_curve = self.plot_item.plot(
            np.array([0, 1]),
            np.nan * np.ones(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
            name="Lower Limit",
        )
        self.upper_curve = self.plot_item.plot(
            np.array([0, 1]),
            np.nan * np.ones(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
            name="Upper Limit",
        )
        self.measured_curve = self.plot_item.plot(
            np.array([0, 1]),
            np.nan * np.ones(2),
            pen={"color": (0, 180, 0), "width": 1},
            name="Measured",
        )

        self.setWindowTitle(f"SRS - {self.ui.initialized_control_names[channel_index]}")
        self.update_plot()
        self.show()

    def update_plot(self):
        spec = self.ui.environment_metadata.specification_data
        freqs = spec.frequencies
        idx = self.channel_index

        self.spec_curve.setData(freqs, spec.srs_spec[:, idx])
        self.lower_curve.setData(freqs, spec.srs_lower_limit[:, idx])
        self.upper_curve.setData(freqs, spec.srs_upper_limit[:, idx])

        if self.ui.run_table.measured_response_srs is not None:
            measured = self.ui.run_table.measured_response_srs[:, idx]
            self.measured_curve.setData(
                self.ui.environment_metadata.get_sds_frequencies(),
                measured,
            )
        else:
            self.measured_curve.setData(np.nan * np.ones(2), np.nan * np.ones(2))


class SDSPlotTimeHistoryWindow(QtWidgets.QDialog):
    def __init__(self, parent, ui, channel_index):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & Qt.Tool)

        self.ui = ui
        self.channel_index = channel_index

        layout = QtWidgets.QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showGrid(True, True, 0.25)
        self.plot_item.enableAutoRange()
        self.plot_item.getViewBox().enableAutoRange(enable=True)

        self.measured_curve = self.plot_item.plot(
            np.array([0, 1]),
            np.nan * np.ones(2),
            pen={"color": (0, 180, 0), "width": 1},
        )

        self.setWindowTitle(f"Time History - {self.ui.initialized_control_names[channel_index]}")
        self.update_plot()
        self.show()

    def update_plot(self):
        if self.ui.run_table.measured_response_time_history is not None:
            th = self.ui.run_table.measured_response_time_history[self.channel_index]
            times = np.arange(th.size) / self.ui.environment_metadata.sample_rate
            self.measured_curve.setData(times, th)
        else:
            self.measured_curve.setData(np.nan * np.ones(2), np.nan * np.ones(2))
