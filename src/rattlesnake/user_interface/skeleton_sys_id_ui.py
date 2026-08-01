import os

import numpy as np
from qtpy import QtWidgets, uic
from multiprocessing.queues import Queue

from rattlesnake.utilities import DIRECTORY
from rattlesnake.engine import RattlesnakeController
from rattlesnake.hardware.abstract_hardware import Channel
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.skeleton_sys_id_environment import (
    SkeletonSysIdCommands,
    SkeletonSysIdMetadata,
    SkeletonSysIdUICommands,
    SkeletonSysIdInstructions,
)
from rattlesnake.user_interface.ui_utilities import (
    axis_label,
    channel_unit_label,
    multiline_plotter,
)
from rattlesnake.user_interface.abstract_sys_id_user_interface import SysIdEnvironmentUI

CONTROL_TYPE = EnvironmentType.SYSID_SKELETON  # noqa pylint: disable=no-member
MAXIMUM_NAME_LENGTH = 50


# region User Interface
class SkeletonSysIdUI(SysIdEnvironmentUI):
    def __init__(
        self,
        environment_name: str,
        rattlesnake: RattlesnakeController,
    ):
        super().__init__(
            CONTROL_TYPE,
            environment_name,
            rattlesnake,
        )
        # Add the page to the control definition tabwidget
        self.definition_widget = QtWidgets.QWidget()
        skeleton_ui_definition_path = os.path.join(
            DIRECTORY, "user_interface", "ui_files", "skeleton_definition.ui"
        )
        uic.loadUi(skeleton_ui_definition_path, self.definition_widget)

        self.run_widget = QtWidgets.QWidget()
        skeleton_ui_run_path = os.path.join(
            DIRECTORY, "user_interface", "ui_files", "skeleton_run.ui"
        )
        uic.loadUi(skeleton_ui_run_path, self.run_widget)

        self.plot_data_item = None

        self.complete_ui()
        self.connect_callbacks()

    def complete_ui(self):
        plot_item = self.run_widget.response_signal_plot.getPlotItem()
        plot_item.showGrid(True, True, 0.25)
        plot_item.enableAutoRange()
        plot_item.getViewBox().enableAutoRange(enable=True)
        plot_item.setLabel("bottom", "Time (s)")
        plot_item.setLabel("left", "Amplitude")

    def connect_callbacks(self):
        self.run_widget.start_test_button.clicked.connect(self.start_environment)
        self.run_widget.stop_test_button.clicked.connect(self.stop_environment)

    @property
    def physical_output_names(self):
        """Names of the physical output channels"""
        return [self.physical_channel_names[i] for i in self.physical_output_indices]

    @property
    def physical_control_indices(self):
        """Indices corresponding to the physical channels that are used as responses (not outputs)"""
        return [
            i
            for i, channel in enumerate(self.hardware_metadata.channel_list)
            if not channel.feedback_device
        ]

    @property
    def physical_control_names(self):
        """Names of the physical control channels"""
        return [self.physical_channel_names[i] for i in self.physical_control_indices]

    @property
    def initialized_control_names(self):
        if self.environment_metadata.response_transformation_matrix is None:
            return [
                self.physical_channel_names[i]
                for i in self.environment_metadata.control_channel_indices
            ]
        else:
            return [
                f"Transformed Response {i + 1}"
                for i in range(
                    self.environment_metadata.response_transformation_matrix.shape[0]
                )
            ]

    @property
    def initialized_output_names(self):
        if self.environment_metadata.reference_transformation_matrix is None:
            return self.physical_output_names
        else:
            return [
                f"Transformed Drive {i + 1}"
                for i in range(
                    self.environment_metadata.reference_transformation_matrix.shape[0]
                )
            ]

    # region State Sync
    def initialize_hardware(self, hardware_metadata: HardwareMetadata):
        super().initialize_hardware(hardware_metadata)

        self.run_widget.response_signal_plot.getPlotItem().clear()

        plot_names = [
            f"{'' if channel.channel_type is None else channel.channel_type} "
            f"{channel.node_number}{channel.node_direction}"
            for channel in hardware_metadata.channel_list
        ]

        self.physical_channel_names = [
            (
                f"{'' if channel.channel_type is None else channel.channel_type} "
                f"{channel.node_number} "
                f"{'' if channel.node_direction is None else channel.node_direction}"
            )[:MAXIMUM_NAME_LENGTH]
            for channel in hardware_metadata.channel_list
        ]
        self.physical_output_indices = [
            i
            for i, channel in enumerate(hardware_metadata.channel_list)
            if channel.feedback_device
        ]

        self.plot_data_item = multiline_plotter(
            np.arange(2),
            np.zeros((len(hardware_metadata.channel_list), 2)),
            widget=self.run_widget.response_signal_plot,
            other_pen_options={"width": 1},
            names=plot_names,
        )
        self.run_widget.response_signal_plot.getPlotItem().setLabel(
            "left",
            axis_label(
                "amplitude",
                "Amplitude",
                channel_unit_label(hardware_metadata.channel_list),
            ),
        )

    def initialize_environment(self, environment_metadata: SkeletonSysIdMetadata):
        num_samples = int(
            environment_metadata.example_window_size
            * self.hardware_metadata.sample_rate
        )
        x = np.arange(num_samples) / self.hardware_metadata.sample_rate
        y = np.zeros(num_samples)
        for curve in self.plot_data_item:
            curve.setData(x, y)

        return super().initialize_environment(environment_metadata)

    def get_environment_metadata(self, global_channel_list: list[Channel]):
        if self.hardware_metadata and global_channel_list:
            channel_list_bools = self.get_channel_list_bools(global_channel_list)
        else:
            channel_list_bools = []

        window_size = self.definition_widget.window_size_spinbox.value()

        environment_channel_list = [
            channel
            for channel, channel_bool in zip(
                self.hardware_metadata.channel_list, channel_list_bools
            )
            if channel_bool
        ]

        control_channel_indices = [
            index
            for index, channel in enumerate(environment_channel_list)
            if channel.feedback_device is None
        ]

        output_channel_indices = [
            index
            for index, channel in enumerate(environment_channel_list)
            if channel.feedback_device is not None
        ]

        return SkeletonSysIdMetadata(
            self.environment_name,
            channel_list_bools,
            self.hardware_metadata.sample_rate,
            window_size,
            control_channel_indices,
            output_channel_indices,
        )

    def set_environment_metadata(self, metadata: SkeletonSysIdMetadata):
        self.definition_widget.window_size_spinbox.setValue(
            metadata.example_window_size
        )

    def get_environment_instructions(self):
        test_level = self.run_widget.test_level_spinbox.value()

        return SkeletonSysIdInstructions(self.environment_name, test_level)

    def set_environment_instructions(self, instructions: SkeletonSysIdInstructions):
        self.run_widget.test_level_spinbox.setValue(instructions.example_test_level)

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

    # endregion

    # region Commands
    def plot_time_data(self, data: np.array):
        response_data = data
        for curve, this_data in zip(self.plot_data_item, response_data):
            self.throttled_curves.roll(curve, this_data)

    def set_test_level(self, data: float):
        test_level = float(data)
        self.run_widget.test_level_spinbox.setValue(test_level)

    def update_gui(self, queue_data):
        if super().update_gui(queue_data):
            return
        command, data = queue_data
        match command:
            case SkeletonSysIdCommands.EXAMPLE_SET_TEST_LEVEL:
                # The normal commands here are supposed to perform the action on the user_interface
                # that would affect the instructions. These are here so the profile event list can simulate the state
                # of the user_interface when collecting environment instructions before start_environment
                # profile commands. These are only needed if the command will do something to the instructions.
                self.set_test_level(data)
            case SkeletonSysIdUICommands.EXAMPLE_UI_SET_TEST_LEVEL:
                self.set_test_level(data)
            case SkeletonSysIdUICommands.EXAMPLE_UI_SHOW_DATA:
                self.plot_time_data(data)
            case _:
                print(f"Unknown Skeleton UI Command {command}")

    # endregion


# endregion
