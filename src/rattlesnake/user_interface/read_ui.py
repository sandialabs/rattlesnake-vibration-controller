import os

import numpy as np
from qtpy import QtCore, QtWidgets, uic

from rattlesnake.utilities import DIRECTORY
from rattlesnake.engine import RattlesnakeController
from rattlesnake.hardware.hardware_utilities import Channel
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.read_environment import (
    ReadCommands,
    ReadUICommands,
    ReadMetadata,
    ReadInstructions,
)
from rattlesnake.user_interface.ui_utilities import multiline_plotter
from rattlesnake.user_interface.abstract_user_interface import EnvironmentUI

ENVIRONMENT_TYPE = EnvironmentType.READ


# region User Interface
class ReadUI(EnvironmentUI):
    def __init__(
        self,
        environment_name: str,
        rattlesnake: RattlesnakeController,
    ):
        super().__init__(ENVIRONMENT_TYPE, environment_name, rattlesnake)

        self.definition_widget = QtWidgets.QWidget()

        self.run_widget = QtWidgets.QWidget()
        skeleton_ui_run_path = os.path.join(
            DIRECTORY, "user_interface", "ui_files", "read_run.ui"
        )
        uic.loadUi(skeleton_ui_run_path, self.run_widget)

        self.plot_data_item = None
        self.channel_enable_checkboxes = []

        self.complete_ui()
        self.connect_callbacks()

    def complete_ui(self):
        plot_item = self.run_widget.response_signal_plot.getPlotItem()
        plot_item.showGrid(True, True, 0.25)
        plot_item.enableAutoRange()
        plot_item.getViewBox().enableAutoRange(enable=True)

        self.run_widget.channel_enable_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )

    def connect_callbacks(self):
        self.run_widget.start_test_button.clicked.connect(self.start_environment)
        self.run_widget.stop_test_button.clicked.connect(self.stop_environment)
        self.run_widget.window_size_spinbox.editingFinished.connect(
            self.change_window_size
        )

    # region State Sync
    def initialize_hardware(self, hardware_metadata: HardwareMetadata):
        self.run_widget.response_signal_plot.getPlotItem().clear()

        plot_names = [
            f"{'' if channel.channel_type is None else channel.channel_type} "
            f"{channel.node_number}{channel.node_direction}"
            for channel in hardware_metadata.channel_list
        ]

        self.plot_data_item = multiline_plotter(
            np.arange(2),
            np.zeros((len(hardware_metadata.channel_list), 2)),
            widget=self.run_widget.response_signal_plot,
            other_pen_options={"width": 1},
            names=plot_names,
        )

        self.run_widget.channel_enable_table.setRowCount(len(plot_names))
        self.channel_enable_checkboxes = []
        for row, name in enumerate(plot_names):
            item = QtWidgets.QTableWidgetItem(name)
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.run_widget.channel_enable_table.setItem(row, 0, item)

            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_channel_visibility)
            checkbox_container = QtWidgets.QWidget()
            checkbox_layout = QtWidgets.QHBoxLayout(checkbox_container)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(QtCore.Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.run_widget.channel_enable_table.setCellWidget(
                row, 1, checkbox_container
            )
            self.channel_enable_checkboxes.append(checkbox)

        return super().initialize_hardware(hardware_metadata)

    def initialize_environment(self, environment_metadata: ReadMetadata):
        num_samples = int(
            self.run_widget.window_size_spinbox.value()
            * self.hardware_metadata.sample_rate
        )
        x = np.arange(num_samples) / self.hardware_metadata.sample_rate
        y = np.zeros(num_samples)
        for curve in self.plot_data_item:
            curve.setData(x, y)

        for checkbox in self.channel_enable_checkboxes:
            checkbox.setChecked(True)
        self.update_channel_visibility()

        return super().initialize_environment(environment_metadata)

    def update_channel_visibility(self):
        if self.plot_data_item is None:
            return
        for curve, checkbox in zip(self.plot_data_item, self.channel_enable_checkboxes):
            curve.setVisible(checkbox.isChecked())

    def get_environment_metadata(self, global_channel_list: list[Channel]):
        if self.hardware_metadata and global_channel_list:
            channel_list_bools = self.get_channel_list_bools(global_channel_list)
        else:
            channel_list_bools = []

        return ReadMetadata(
            self.environment_name,
            channel_list_bools,
            self.hardware_metadata.sample_rate,
        )

    def set_environment_metadata(self, metadata: ReadMetadata):
        return

    def get_environment_instructions(self):
        window_size = self.run_widget.window_size_spinbox.value()
        return ReadInstructions(self.environment_name, window_size)

    def set_environment_instructions(self, instructions: ReadInstructions):
        self.run_widget.window_size_spinbox.setValue(instructions.window_size)

    # endregion

    # region Run
    def start_environment(self):
        self.run_widget.start_test_button.setEnabled(False)
        return super().start_environment()

    def start_environment_ready(self):
        super().start_environment_ready()

    def start_environment_error(self, error):
        super().start_environment_error(error)

    def stop_environment(self):
        self.run_widget.stop_test_button.setEnabled(False)
        return super().stop_environment()

    def stop_environment_ready(self):
        super().stop_environment_ready()

    def stop_environment_error(self, error):
        super().stop_environment_error(error)

    def display_environment_started(self):
        self.run_widget.start_test_button.setEnabled(False)
        self.run_widget.stop_test_button.setEnabled(True)
        return super().display_environment_started()

    def display_environment_ended(self):
        self.run_widget.start_test_button.setEnabled(True)
        self.run_widget.stop_test_button.setEnabled(False)
        return super().display_environment_ended()

    def change_window_size(self):
        new_window_size = self.run_widget.window_size_spinbox.value()
        try:
            self.rattlesnake.send_environment_command(
                self.environment_name,
                ReadCommands.CHANGE_WINDOW_SIZE,
                new_window_size,
            )
        except Exception as e:
            self.display_error(e)

    # endregion

    # region Commands
    def set_window_size(self, data):
        self.run_widget.window_size_spinbox.blockSignals(True)
        self.run_widget.window_size_spinbox.setValue(data)
        self.run_widget.window_size_spinbox.blockSignals(False)

        if self.hardware_metadata is None or self.plot_data_item is None:
            return

        num_samples = int(data * self.hardware_metadata.sample_rate)
        if num_samples <= 0:
            return

        new_x = np.arange(num_samples) / self.hardware_metadata.sample_rate
        for curve in self.plot_data_item:
            _, old_y = self.throttled_curves.get(curve)
            new_y = np.zeros(num_samples)
            copy_size = min(num_samples, old_y.size)
            new_y[-copy_size:] = old_y[-copy_size:]
            self.throttled_curves.set(curve, new_x, new_y)

    def plot_time_data(self, data: np.array):
        response_data = data
        for curve, this_data in zip(self.plot_data_item, response_data):
            self.throttled_curves.roll(curve, this_data)

    def update_gui(self, queue_data):
        if super().update_gui(queue_data):
            return
        command, data = queue_data
        match command:
            case ReadCommands.CHANGE_WINDOW_SIZE:
                self.set_window_size(data)
            case ReadUICommands.SET_WINDOW_SIZE:
                self.set_window_size(data)
            case ReadUICommands.TIME_DATA:
                self.plot_time_data(data)
            case _:
                print(f"Unknown Read UI Command {command}")
