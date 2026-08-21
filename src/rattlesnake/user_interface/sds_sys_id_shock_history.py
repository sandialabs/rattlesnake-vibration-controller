import os
from collections import Counter

import numpy as np
from qtpy import QtWidgets, uic
from qtpy.QtCore import Qt
import pyqtgraph as pg

from rattlesnake.utilities import DIRECTORY
from rattlesnake.user_interface.ui_utilities import colororder


class SDSShockHistoryDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        ui_path = os.path.join(
            DIRECTORY,
            "user_interface",
            "ui_files",
            "srs_sds_shock_history.ui",
        )
        uic.loadUi(ui_path, self)

        self.history = []

        self._setup_plots()
        self._setup_table()
        self._connect_callbacks()

        self.history_table_groupbox.setVisible(self.show_history_table_checkbox.isChecked())

    def _connect_callbacks(self):
        self.close_button.clicked.connect(self.hide)
        self.show_history_table_checkbox.toggled.connect(self.history_table_groupbox.setVisible)

    def _setup_plots(self):
        for plot_widget in [self.hits_by_level_plot, self.shock_timeline_plot]:
            plot_item = plot_widget.getPlotItem()
            plot_item.showGrid(True, True, 0.25)
            plot_item.enableAutoRange()
            plot_item.getViewBox().enableAutoRange(enable=True)

        self.hits_by_level_plot.getPlotItem().setLabel("bottom", "Test Level (dB)")
        self.hits_by_level_plot.getPlotItem().setLabel("left", "Hit Count")

        self.shock_timeline_plot.getPlotItem().setLabel("bottom", "Hit Number")
        self.shock_timeline_plot.getPlotItem().setLabel("left", "Test Level (dB)")

    def _setup_table(self):
        headers = [
            "Hit #",
            "Timestamp",
            "Test Level (dB)",
            "Counted at Target?",
            "Total Hits",
            "Hits at Target",
            "Target Hits Requested",
        ]
        self.history_table.setColumnCount(len(headers))
        self.history_table.setHorizontalHeaderLabels(headers)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.history_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.history_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

    def update_history(
        self,
        hit_history,
        total_hits=None,
        hits_at_target=None,
        target_hits=None,
    ):
        self.history = [] if hit_history is None else list(hit_history)

        # Derive summary if not explicitly provided
        if total_hits is None:
            total_hits = len(self.history)

        if hits_at_target is None:
            hits_at_target = sum(
                1 for entry in self.history if entry.get("counted_at_target", False)
            )

        if target_hits is None:
            if len(self.history) > 0:
                target_hits = self.history[-1].get("target_hits_at_level", 0)
            else:
                target_hits = 0

        distinct_levels = len(
            set(round(entry.get("test_level_db", 0.0), 6) for entry in self.history)
        )

        self.total_hits_display.setValue(int(total_hits))
        self.hits_at_target_display.setValue(int(hits_at_target))
        self.target_hits_display.setValue(int(target_hits))
        self.distinct_levels_display.setValue(int(distinct_levels))

        self._update_level_plot()
        self._update_timeline_plot()
        self._update_table()

    def _update_level_plot(self):
        plot_item = self.hits_by_level_plot.getPlotItem()
        plot_item.clear()
        plot_item.showGrid(True, True, 0.25)

        if len(self.history) == 0:
            return

        levels = [round(entry.get("test_level_db", 0.0), 6) for entry in self.history]
        counts = Counter(levels)

        sorted_levels = sorted(counts.keys())
        x = np.arange(len(sorted_levels))
        heights = np.array([counts[level] for level in sorted_levels], dtype=float)

        brushes = []
        for level in sorted_levels:
            if np.isclose(level, 0.0):
                brushes.append(pg.mkBrush(0, 114, 189))
            else:
                brushes.append(pg.mkBrush(150, 150, 150))

        bar_item = pg.BarGraphItem(
            x=x,
            height=heights,
            width=0.8,
            brushes=brushes,
        )
        plot_item.addItem(bar_item)

        axis = plot_item.getAxis("bottom")
        axis.setTicks([[(xi, f"{level:g}") for xi, level in zip(x, sorted_levels)]])

    def _update_timeline_plot(self):
        plot_item = self.shock_timeline_plot.getPlotItem()
        plot_item.clear()
        plot_item.showGrid(True, True, 0.25)

        if len(self.history) == 0:
            return

        hit_indices = np.array(
            [entry.get("hit_index", i + 1) for i, entry in enumerate(self.history)],
            dtype=float,
        )
        levels = np.array(
            [entry.get("test_level_db", 0.0) for entry in self.history],
            dtype=float,
        )
        counted = np.array(
            [entry.get("counted_at_target", False) for entry in self.history],
            dtype=bool,
        )

        # Draw a connecting line
        plot_item.plot(hit_indices, levels, pen=pg.mkPen(color=(120, 120, 120), width=1))

        # Non-target hits
        if np.any(~counted):
            plot_item.plot(
                hit_indices[~counted],
                levels[~counted],
                pen=None,
                symbol="o",
                symbolSize=8,
                symbolBrush=pg.mkBrush(150, 150, 150),
                symbolPen=pg.mkPen(150, 150, 150),
            )

        # Target hits
        if np.any(counted):
            plot_item.plot(
                hit_indices[counted],
                levels[counted],
                pen=None,
                symbol="o",
                symbolSize=9,
                symbolBrush=pg.mkBrush(0, 114, 189),
                symbolPen=pg.mkPen(0, 114, 189),
            )

        # Reference line at 0 dB
        zero_line = pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen((200, 0, 0), style=Qt.DashLine))
        plot_item.addItem(zero_line)

    def _update_table(self):
        self.history_table.setRowCount(len(self.history))

        for row, entry in enumerate(self.history):
            values = [
                entry.get("hit_index", ""),
                entry.get("timestamp", ""),
                entry.get("test_level_db", ""),
                "Yes" if entry.get("counted_at_target", False) else "No",
                entry.get("total_hits", ""),
                entry.get("hits_at_target", ""),
                entry.get("target_hits_at_level", ""),
            ]

            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.history_table.setItem(row, col, item)
