# -*- coding: utf-8 -*-
"""
This file defines a shock environment that utilizes system
identification.

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

import importlib
import inspect
import multiprocessing as mp
import multiprocessing.sharedctypes  # pylint: disable=unused-import
import os
import traceback
from enum import Enum
from multiprocessing.queues import Queue

import netCDF4 as nc4
import numpy as np
import scipy.signal as sig
from qtpy import QtCore, QtWidgets, uic
from qtpy.QtCore import Qt

from .abstract_sysid_environment import (
    AbstractSysIdEnvironment,
    AbstractSysIdUI,
)
from .environments import (
    ControlTypes,
    environment_definition_ui_paths,
    environment_prediction_ui_paths,
    environment_run_ui_paths,
)
from .shock_sys_id_metadata import (
    ShockMetadata,
)
from .ui_utilities import (
    PlotTimeWindow,
    TransformationMatrixWindow,
    colororder,
    load_time_history,
    multiline_plotter,
    AdaptiveNoWheelSpinBox,
)
from .utilities import (
    DataAcquisitionParameters,
    GlobalCommands,
    VerboseMessageQueue,
    db2scale,
    load_python_module,
)

# %% Global Variables
CONTROL_TYPE = ControlTypes.SDS
MAXIMUM_NAME_LENGTH = 50
BUFFER_SIZE_SAMPLES_PER_READ_MULTIPLIER = 2


# %% Commands
class ShockCommands(Enum):
    """Valid commands for the Shock environment"""

    START_CONTROL = 0
    STOP_CONTROL = 1
    PERFORM_CONTROL_PREDICTION = 3
    # UPDATE_INTERACTIVE_CONTROL_PARAMETERS = 4


# %% Queues


class ShockQueues:
    """A container class for the queues that this environment will manage."""

    def __init__(
        self,
        environment_name: str,
        environment_command_queue: VerboseMessageQueue,
        gui_update_queue: Queue,
        controller_communication_queue: VerboseMessageQueue,
        data_in_queue: Queue,
        data_out_queue: Queue,
        log_file_queue: VerboseMessageQueue,
    ):
        """A container class for the queues that Shock will manage.

        The environment uses many queues to pass data between the various pieces.
        This class organizes those queues into one common namespace.

        Parameters
        ----------
        environment_name : str
            Name of the environment
        environment_command_queue : VerboseMessageQueue
            Queue that is read by the environment for environment commands
        gui_update_queue : mp.queues.Queue
            Queue where various subtasks put instructions for updating the
            widgets in the user interface
        controller_communication_queue : VerboseMessageQueue
            Queue that is read by the controller for global controller commands
        data_in_queue : mp.queues.Queue
            Multiprocessing queue that connects the acquisition subtask to the
            environment subtask.  Each environment will retrieve acquired data
            from this queue.
        data_out_queue : mp.queues.Queue
            Multiprocessing queue that connects the output subtask to the
            environment subtask.  Each environment will put data that it wants
            the controller to generate in this queue.
        log_file_queue : VerboseMessageQueue
            Queue for putting logging messages that will be read by the logging
            subtask and written to a file.
        """
        self.environment_command_queue = environment_command_queue
        self.gui_update_queue = gui_update_queue
        self.data_analysis_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Data Analysis Command Queue"
        )
        self.signal_generation_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Signal Generation Command Queue"
        )
        self.spectral_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Spectral Computation Command Queue"
        )
        self.collector_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Data Collector Command Queue"
        )
        self.controller_communication_queue = controller_communication_queue
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.data_for_spectral_computation_queue = mp.Queue()
        self.updated_spectral_quantities_queue = mp.Queue()
        self.time_history_to_generate_queue = mp.Queue()
        self.log_file_queue = log_file_queue


# %% UI

from .abstract_interactive_control_law import (  # noqa: E402 pylint: disable=wrong-import-position
    AbstractControlLawComputation,
)
from .abstract_sysid_data_analysis import (  # noqa: E402 pylint: disable=wrong-import-position
    sysid_data_analysis_process,
)
from .data_collector import (  # noqa: E402 pylint: disable=wrong-import-position
    FrameBuffer,
    data_collector_process,
)
from .signal_generation import (  # noqa: E402 pylint: disable=wrong-import-position
    TransientSignalGenerator,
)
from .signal_generation_process import (  # noqa: E402 pylint: disable=wrong-import-position
    SignalGenerationCommands,
    SignalGenerationMetadata,
    signal_generation_process,
)
from .spectral_processing import (  # noqa: E402 pylint: disable=wrong-import-position
    spectral_processing_process,
)


class ShockUI(AbstractSysIdUI):
    """Class defining the user interface for the Shock environment"""

    def __init__(
        self,
        environment_name: str,
        definition_tabwidget: QtWidgets.QTabWidget,
        system_id_tabwidget: QtWidgets.QTabWidget,
        test_predictions_tabwidget: QtWidgets.QTabWidget,
        run_tabwidget: QtWidgets.QTabWidget,
        environment_command_queue: VerboseMessageQueue,
        controller_communication_queue: VerboseMessageQueue,
        log_file_queue: Queue,
    ):
        super().__init__(
            environment_name,
            environment_command_queue,
            controller_communication_queue,
            log_file_queue,
            system_id_tabwidget,
        )
        # Add the page to the control definition tabwidget
        self.definition_widget = QtWidgets.QWidget()
        uic.loadUi(environment_definition_ui_paths[CONTROL_TYPE], self.definition_widget)
        definition_tabwidget.addTab(self.definition_widget, self.environment_name)
        # Add the page to the control prediction tabwidget
        self.prediction_widget = QtWidgets.QWidget()
        uic.loadUi(environment_prediction_ui_paths[CONTROL_TYPE], self.prediction_widget)
        test_predictions_tabwidget.addTab(self.prediction_widget, self.environment_name)
        # Add the page to the run tabwidget
        self.run_widget = QtWidgets.QWidget()
        uic.loadUi(environment_run_ui_paths[CONTROL_TYPE], self.run_widget)
        run_tabwidget.addTab(self.run_widget, self.environment_name)

        # Initialize persistent data
        self.plot_data_items = {}
        self.physical_channel_names = None
        self.physical_output_indices = None
        self.physical_unit_names = None
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.python_control_module = None

        self.control_selector_widgets = [self.definition_widget.specification_plot_selector]

        self.output_selector_widgets = []

        self.plotwidgets = [
            self.definition_widget.specification_plot,
            self.run_widget.global_test_performance_plot,
        ]

        for plotwidget in self.plotwidgets:
            plot_item = plotwidget.getPlotItem()
            plot_item.showGrid(True, True, 0.25)
            plot_item.enableAutoRange()
            plot_item.getViewBox().enableAutoRange(enable=True)

    def connect_callbacks(self):
        """Connects the callbacks to the Shock UI widgets"""
        # Definition
        self.definition_widget.add_breakpoint_button.clicked.connect(self.add_breakpoint)
        self.definition_widget.remove_breakpoint_button.clicked.connect(self.remove_breakpoint)
        self.definition_widget.from_spec_button.toggled.connect(self.update_tone_table)
        self.definition_widget.octave_button.toggled.connect(self.update_tone_table)
        self.definition_widget.manual_button.toggled.connect(self.update_tone_table)
        self.definition_widget.common_decay_checkbox.toggled.connect(self.update_decay_table)
        self.definition_widget.decay_value_selector.valueChanged.connect(self.update_decay_table)
        self.definition_widget.add_tone_button.clicked.connect(self.add_tone)
        self.definition_widget.remove_tone_button.clicked.connect(self.remove_tone)
        self.definition_widget.transformation_matrices_button.clicked.connect(
            self.define_transformation_matrices
        )
        self.definition_widget.control_channels_selector.itemChanged.connect(
            self.update_control_channels
        )
        self.definition_widget.check_selected_button.clicked.connect(
            self.check_selected_control_channels
        )
        self.definition_widget.uncheck_selected_button.clicked.connect(
            self.uncheck_selected_control_channels
        )
        self.definition_widget.specification_plot_selector.currentIndexChanged.connect(
            self.update_specification
        )
        self.definition_widget.synthesize_button.clicked.connect(self.synthesize_sds)
        # Prediction
        # Run Test

    # %% Data Acquisition

    def initialize_data_acquisition(self, data_acquisition_parameters: DataAcquisitionParameters):
        super().initialize_data_acquisition(data_acquisition_parameters)
        # Initialize and clear plots
        for plotwidget in self.plotwidgets:
            plotwidget.clear()
        self.plot_data_items[
            "specification_srs"
        ] = self.definition_widget.specification_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.zeros(2),
            pen={"color": "b", "width": 1},
            name="Amplitude",
        )
        self.plot_data_items[
            "specification_lower_limit"
        ] = self.definition_widget.specification_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.zeros(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
            name="Warning",
        )
        self.plot_data_items[
            "specification_upper_limit"
        ] = self.definition_widget.specification_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.zeros(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
        )
        self.definition_widget.specification_plot.getPlotItem().addLegend()

        # Set up channel names
        # Set up channel names
        self.physical_channel_names = [
            (
                f"{'' if channel.channel_type is None else channel.channel_type} "
                f"{channel.node_number} "
                f"{'' if channel.node_direction is None else channel.node_direction}"
            )[:MAXIMUM_NAME_LENGTH]
            for channel in data_acquisition_parameters.channel_list
        ]
        self.physical_unit_names = [
            f"{'-' if channel.unit is None else channel.unit}"
            for channel in data_acquisition_parameters.channel_list
        ]
        self.physical_output_indices = [
            i
            for i, channel in enumerate(data_acquisition_parameters.channel_list)
            if channel.feedback_device
        ]

        # Set default values for various widgets
        # Sampling parameters
        self.definition_widget.sample_rate_display.setValue(data_acquisition_parameters.sample_rate)
        self.system_id_widget.samplesPerFrameSpinBox.setValue(
            data_acquisition_parameters.sample_rate
        )
        # By default set the block time length to 1 second
        self.definition_widget.block_size_selector.setValue(data_acquisition_parameters.sample_rate)
        # Set up control channel list
        self.definition_widget.control_channels_selector.clear()
        for channel_name in self.physical_channel_names:
            item = QtWidgets.QListWidgetItem()
            item.setText(channel_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.definition_widget.control_channels_selector.addItem(item)
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.define_transformation_matrices(None, False)
        self.definition_widget.input_channels_display.setValue(len(self.physical_channel_names))
        self.definition_widget.output_channels_display.setValue(len(self.physical_output_indices))
        self.definition_widget.control_channels_display.setValue(0)

        # Clear and update specification table
        self.clear_and_update_specification_table()

    @property
    def physical_output_names(self):
        """Names of the physical drive channels"""
        return [self.physical_channel_names[i] for i in self.physical_output_indices]

    # %% Environment

    @property
    def physical_control_indices(self):
        """Indices of the control channels"""
        return [
            i
            for i in range(self.definition_widget.control_channels_selector.count())
            if self.definition_widget.control_channels_selector.item(i).checkState() == Qt.Checked
        ]

    @property
    def physical_control_names(self):
        """Names of the selected control channels"""
        return [self.physical_channel_names[i] for i in self.physical_control_indices]

    @property
    def control_names(self):
        return (
            self.physical_control_names
            if self.response_transformation_matrix is None
            else [
                f"Transformed Response {i+1}"
                for i in range(self.response_transformation_matrix.shape[0])
            ]
        )

    @property
    def physical_control_units(self):
        """Gets the unit for the control channels currently checked"""
        return [self.physical_unit_names[i] for i in self.physical_control_indices]

    @property
    def initialized_control_names(self):
        """Names of the control channels that have been initialized"""
        if self.environment_parameters.response_transformation_matrix is None:
            return [
                self.physical_channel_names[i]
                for i in self.environment_parameters.control_channel_indices
            ]
        else:
            return [
                f"Transformed Response {i + 1}"
                for i in range(self.environment_parameters.response_transformation_matrix.shape[0])
            ]

    @property
    def initialized_output_names(self):
        """Names of the drive channels that have been initialized"""
        if self.environment_parameters.reference_transformation_matrix is None:
            return self.physical_output_names
        else:
            return [
                f"Transformed Drive {i + 1}"
                for i in range(self.environment_parameters.reference_transformation_matrix.shape[0])
            ]

    def load_specification(self, clicked, filename=None):  # pylint: disable=unused-argument
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select Specification File",
                filter="Numpy or Mat (*.npy *.npz *.mat)",
            )
            if filename == "":
                return
        spec_data = np.load(filename)
        self.clear_and_update_specification_table(
            spec_data["f"],
            spec_data["srs"],
            spec_data["lower_limit"] if "lower_limit" in spec_data else None,
            spec_data["upper_limit"] if "upper_limit" in spec_data else None,
        )
        self.definition_widget.num_hits_spinbox.setValue(spec_data["num_hits"])
        self.update_specification()

    def clear_and_update_specification_table(
        self, frequencies=None, srs=None, lower_limit=None, upper_limit=None
    ):
        control_names = self.control_names
        if frequencies is None:
            num_rows = 2
        else:
            num_rows = frequencies.size
        self.definition_widget.breakpoint_table.clear()
        self.definition_widget.breakpoint_table.setRowCount(num_rows)
        self.definition_widget.breakpoint_table.setColumnCount(1 + len(control_names))
        self.definition_widget.lower_limit_table.setRowCount(num_rows)
        self.definition_widget.lower_limit_table.setColumnCount(1 + len(control_names))
        self.definition_widget.upper_limit_table.setRowCount(num_rows)
        self.definition_widget.upper_limit_table.setColumnCount(1 + len(control_names))
        header_labels = ["Frequency"] + list(control_names)
        self.definition_widget.breakpoint_table.setHorizontalHeaderLabels(header_labels)
        self.definition_widget.lower_limit_table.setHorizontalHeaderLabels(header_labels)
        self.definition_widget.upper_limit_table.setHorizontalHeaderLabels(header_labels)
        for row in range(num_rows):
            # Frequency breakpoint
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
            spinbox.setSingleStep(1)
            if frequencies is None:
                spinbox.setValue(0)
            else:
                spinbox.setValue(frequencies[row])
            spinbox.setKeyboardTracking(False)
            spinbox.setDecimals(4)
            spinbox.valueChanged.connect(self.update_specification)
            self.definition_widget.breakpoint_table.setCellWidget(row, 0, spinbox)
            # Frequency breakpoint, lower limit
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
            spinbox.setSingleStep(1)
            if frequencies is None:
                spinbox.setValue(0)
            else:
                spinbox.setValue(frequencies[row])
            spinbox.setKeyboardTracking(False)
            spinbox.setDecimals(4)
            spinbox.setReadOnly(True)
            spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            self.definition_widget.lower_limit_table.setCellWidget(row, 0, spinbox)
            # Frequency breakpoint, upper limit
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
            spinbox.setSingleStep(1)
            if frequencies is None:
                spinbox.setValue(0)
            else:
                spinbox.setValue(frequencies[row])
            spinbox.setKeyboardTracking(False)
            spinbox.setDecimals(4)
            spinbox.setReadOnly(True)
            spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            self.definition_widget.upper_limit_table.setCellWidget(row, 0, spinbox)
            for j in range(len(control_names)):
                spinbox = AdaptiveNoWheelSpinBox()
                spinbox.setRange(0, 1000000)
                spinbox.setSingleStep(1)
                if srs is None:
                    spinbox.setValue(1)
                else:
                    spinbox.setValue(0 if np.isnan(srs[row, j]) else srs[row, j])
                spinbox.setKeyboardTracking(False)
                spinbox.setSpecialValueText("No Control")
                spinbox.valueChanged.connect(self.update_specification)
                self.definition_widget.breakpoint_table.setCellWidget(row, 1 + j, spinbox)
                spinbox = AdaptiveNoWheelSpinBox()
                spinbox.setRange(0, 1000000)
                spinbox.setSingleStep(1)
                if lower_limit is None:
                    spinbox.setValue(0)
                else:
                    spinbox.setValue(0 if np.isnan(lower_limit[row, j]) else lower_limit[row, j])
                spinbox.setKeyboardTracking(False)
                spinbox.setSpecialValueText("No Limit")
                spinbox.valueChanged.connect(self.update_specification)
                self.definition_widget.lower_limit_table.setCellWidget(row, 1 + j, spinbox)
                spinbox = AdaptiveNoWheelSpinBox()
                spinbox.setRange(0, 1000000)
                spinbox.setSingleStep(1)
                if upper_limit is None:
                    spinbox.setValue(0)
                else:
                    spinbox.setValue(0 if np.isnan(upper_limit[row, j]) else upper_limit[row, j])
                spinbox.setKeyboardTracking(False)
                spinbox.setSpecialValueText("No Limit")
                spinbox.valueChanged.connect(self.update_specification)
                self.definition_widget.upper_limit_table.setCellWidget(row, 1 + j, spinbox)

    def add_breakpoint(self):
        selected_indices = self.definition_widget.breakpoint_table.selectedIndexes()
        if selected_indices:
            selected_row = selected_indices[0].row()
        else:
            # If no row is selected, add the row at the start
            selected_row = 0
        control_names = self.control_names
        self.definition_widget.breakpoint_table.insertRow(selected_row)
        self.definition_widget.lower_limit_table.insertRow(selected_row)
        self.definition_widget.upper_limit_table.insertRow(selected_row)
        # Frequency breakpoint
        spinbox = AdaptiveNoWheelSpinBox()
        spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
        spinbox.setSingleStep(1)
        spinbox.setValue(0)
        spinbox.setKeyboardTracking(False)
        spinbox.setDecimals(4)
        spinbox.valueChanged.connect(self.update_specification)
        self.definition_widget.breakpoint_table.setCellWidget(selected_row, 0, spinbox)
        # Frequency breakpoint, lower limit
        spinbox = AdaptiveNoWheelSpinBox()
        spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
        spinbox.setSingleStep(1)
        spinbox.setValue(0)
        spinbox.setKeyboardTracking(False)
        spinbox.setDecimals(4)
        spinbox.setReadOnly(True)
        spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
        self.definition_widget.lower_limit_table.setCellWidget(selected_row, 0, spinbox)
        # Frequency breakpoint, upper limit
        spinbox = AdaptiveNoWheelSpinBox()
        spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
        spinbox.setSingleStep(1)
        spinbox.setValue(0)
        spinbox.setKeyboardTracking(False)
        spinbox.setDecimals(4)
        spinbox.setReadOnly(True)
        spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
        self.definition_widget.upper_limit_table.setCellWidget(selected_row, 0, spinbox)
        for j in range(len(control_names)):
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(0, 1000000)
            spinbox.setSingleStep(1)
            spinbox.setValue(1)
            spinbox.setKeyboardTracking(False)
            spinbox.setSpecialValueText("No Control")
            spinbox.valueChanged.connect(self.update_specification)
            self.definition_widget.breakpoint_table.setCellWidget(selected_row, 1 + j, spinbox)
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(0, 1000000)
            spinbox.setSingleStep(1)
            spinbox.setValue(0)
            spinbox.setKeyboardTracking(False)
            spinbox.setSpecialValueText("No Limit")
            spinbox.valueChanged.connect(self.update_specification)
            self.definition_widget.lower_limit_table.setCellWidget(selected_row, 1 + j, spinbox)
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(0, 1000000)
            spinbox.setSingleStep(1)
            spinbox.setValue(0)
            spinbox.setKeyboardTracking(False)
            spinbox.setSpecialValueText("No Limit")
            spinbox.valueChanged.connect(self.update_specification)
            self.definition_widget.upper_limit_table.setCellWidget(selected_row, 1 + j, spinbox)
        self.update_specification()

    def remove_breakpoint(self):
        selected_indices = self.definition_widget.breakpoint_table.selectedIndexes()
        if selected_indices:
            selected_row = selected_indices[0].row()
        else:
            # If no row is selected, remove the last row
            selected_row = self.definition_widget.breakpoint_table.rowCount() - 1
        self.definition_widget.breakpoint_table.removeRow(selected_row)
        self.definition_widget.lower_limit_table.removeRow(selected_row)
        self.definition_widget.upper_limit_table.removeRow(selected_row)
        self.update_specification()

    def select_python_module(self, clicked, filename=None):  # pylint: disable=unused-argument
        """Loads a Python module using a dialog or the specified filename

        Parameters
        ----------
        clicked :
            The clicked event that triggered the callback.
        filename :
            File name defining the Python module for bypassing the callback when
            loading from a file (Default value = None).

        """
        if filename is None or not os.path.isfile(filename):
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.definition_widget,
                "Select Python Module",
                filter="Python Modules (*.py)",
            )
            if filename == "":
                return
        self.python_control_module = load_python_module(filename)
        functions = [
            function
            for function in inspect.getmembers(self.python_control_module)
            if (
                inspect.isfunction(function[1])
                and len(inspect.signature(function[1]).parameters)
                >= 6  # TODO: Change proper number of arguments
            )
            or inspect.isgeneratorfunction(function[1])
            or (
                inspect.isclass(function[1])
                and all(
                    [
                        (
                            method in function[1].__dict__
                            and not (
                                hasattr(function[1].__dict__[method], "__isabstractmethod__")
                                and function[1].__dict__[method].__isabstractmethod__
                            )
                        )
                        for method in ["system_id_update", "control"]
                    ]
                )
            )
        ]
        self.log(
            f"Loaded module {self.python_control_module.__name__} with "
            f"functions {[function[0] for function in functions]}"
        )
        self.definition_widget.control_function_input.clear()
        self.definition_widget.control_script_file_path_input.setText(filename)
        for function in functions:
            self.definition_widget.control_function_input.addItem(function[0])

    def update_specification(self):
        channel_index = self.definition_widget.specification_plot_selector.currentIndex()
        num_freqs = self.definition_widget.breakpoint_table.rowCount()
        freqs = np.empty(num_freqs, "float")
        srss = np.empty(num_freqs, "float")
        lower_limits = np.empty(num_freqs, "float")
        upper_limits = np.empty(num_freqs, "float")
        if self.definition_widget.from_spec_button.isChecked():
            self.update_tone_table()
        for row in range(num_freqs):
            freqs[row] = self.definition_widget.breakpoint_table.cellWidget(row, 0).value()
            srss[row] = (
                np.nan
                if self.definition_widget.breakpoint_table.cellWidget(
                    row, 1 + channel_index
                ).value()
                == 0
                else self.definition_widget.breakpoint_table.cellWidget(
                    row, 1 + channel_index
                ).value()
            )
            lower_limits[row] = (
                np.nan
                if self.definition_widget.lower_limit_table.cellWidget(
                    row, 1 + channel_index
                ).value()
                == 0
                else self.definition_widget.lower_limit_table.cellWidget(
                    row, 1 + channel_index
                ).value()
            )
            upper_limits[row] = (
                np.nan
                if self.definition_widget.upper_limit_table.cellWidget(
                    row, 1 + channel_index
                ).value()
                == 0
                else self.definition_widget.upper_limit_table.cellWidget(
                    row, 1 + channel_index
                ).value()
            )
        self.plot_data_items["specification_srs"].setData(freqs, srss)
        self.plot_data_items["specification_lower_limit"].setData(freqs, lower_limits)
        self.plot_data_items["specification_upper_limit"].setData(freqs, upper_limits)

    def collect_environment_definition_parameters(self):
        """Collects the metadata defining the environment from the UI widgets"""
        if self.python_control_module is None:
            control_module = None
            control_function = None
            control_function_type = None
            control_function_parameters = None
        else:
            control_module = self.definition_widget.control_script_file_path_input.text()
            control_function = self.definition_widget.control_function_input.itemText(
                self.definition_widget.control_function_input.currentIndex()
            )
            control_function_type = (
                self.definition_widget.control_function_generator_selector.currentIndex()
            )
            control_function_parameters = (
                self.definition_widget.control_parameters_text_input.toPlainText()
            )
        return ShockMetadata()

    def initialize_environment(self):
        super().initialize_environment()
        return self.environment_parameters

    def define_transformation_matrices(
        self, clicked, dialog=True
    ):  # pylint: disable=unused-argument
        """Defines the transformation matrices using the dialog box"""
        if dialog:
            (response_transformation, output_transformation, result) = (
                TransformationMatrixWindow.define_transformation_matrices(
                    self.response_transformation_matrix,
                    self.definition_widget.control_channels_display.value(),
                    self.output_transformation_matrix,
                    self.definition_widget.output_channels_display.value(),
                    self.definition_widget,
                )
            )
        else:
            response_transformation = self.response_transformation_matrix
            output_transformation = self.output_transformation_matrix
            result = True
        if result:
            # Update the control names
            for widget in self.control_selector_widgets:
                widget.blockSignals(True)
                widget.clear()
            if response_transformation is None:
                for i, control_name in enumerate(self.physical_control_names):
                    for widget in self.control_selector_widgets:
                        widget.addItem(f"{i + 1}: {control_name}")
                self.definition_widget.transform_channels_display.setValue(
                    len(self.physical_control_names)
                )
            else:
                for i in range(response_transformation.shape[0]):
                    for widget in self.control_selector_widgets:
                        widget.addItem(f"{i + 1}: Virtual Response")
                self.definition_widget.transform_channels_display.setValue(
                    response_transformation.shape[0]
                )
            for widget in self.control_selector_widgets:
                widget.blockSignals(False)
            # Update the output names
            for widget in self.output_selector_widgets:
                widget.blockSignals(True)
                widget.clear()
            if output_transformation is None:
                for i, drive_name in enumerate(self.physical_output_names):
                    for widget in self.output_selector_widgets:
                        widget.addItem(f"{i + 1}: {drive_name}")
                self.definition_widget.transform_outputs_display.setValue(
                    len(self.physical_output_names)
                )
            else:
                for i in range(output_transformation.shape[0]):
                    for widget in self.output_selector_widgets:
                        widget.addItem(f"{i + 1}: Virtual Drive")
                self.definition_widget.transform_outputs_display.setValue(
                    output_transformation.shape[0]
                )
            for widget in self.output_selector_widgets:
                widget.blockSignals(False)
            self.response_transformation_matrix = response_transformation
            self.output_transformation_matrix = output_transformation
            self.clear_and_update_specification_table()

    def set_sine_tone_values(self, frequencies=None, decays=None):
        if frequencies is None and decays is None:
            self.definition_widget.tone_table.setRowCount(0)
            return
        if frequencies is not None:
            self.definition_widget.tone_table.setRowCount(len(frequencies))
            for row, frequency in enumerate(frequencies):
                # Check to see if there is a widget
                freq_spinbox = self.definition_widget.tone_table.getCellWidget(row, 0)
                decay_spinbox = self.definition_widget.tone_table.getCellWidget(row, 1)
                # If there isn't, make one
                if freq_spinbox is None:
                    freq_spinbox = AdaptiveNoWheelSpinBox()
                    freq_spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
                    freq_spinbox.setSingleStep(1)
                    freq_spinbox.setKeyboardTracking(False)
                    freq_spinbox.setDecimals(4)
                    self.definition_widget.tone_table.setCellWidget(row, 0, freq_spinbox)
                if decay_spinbox is None:
                    decay_spinbox = AdaptiveNoWheelSpinBox()
                    decay_spinbox.setRange(0, 1000000)
                    decay_spinbox.setSingleStep(1)
                    decay_spinbox.setKeyboardTracking(False)
                    if decays is None:
                        decay_spinbox.setValue(self.definition_widget.decay_value_selector.value())
                    self.definition_widget.tone_table.setCellWidget(row, 1, decay_spinbox)
                freq_spinbox.setValue(frequency)
                if decays is not None:
                    decay_spinbox.setValue(decays[row])
        elif decays is not None:
            self.definition_widget.tone_table.setRowCount(len(decays))
            for row, decay in enumerate(decays):
                # Check to see if there is a widget
                freq_spinbox = self.definition_widget.tone_table.getCellWidget(row, 0)
                decay_spinbox = self.definition_widget.tone_table.getCellWidget(row, 1)
                # If there isn't, make one
                if freq_spinbox is None:
                    freq_spinbox = AdaptiveNoWheelSpinBox()
                    freq_spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
                    freq_spinbox.setSingleStep(1)
                    freq_spinbox.setKeyboardTracking(False)
                    freq_spinbox.setDecimals(4)
                    freq_spinbox.setValue(0)
                    self.definition_widget.tone_table.setCellWidget(row, 0, freq_spinbox)
                if decay_spinbox is None:
                    decay_spinbox = AdaptiveNoWheelSpinBox()
                    decay_spinbox.setRange(0, 1000000)
                    decay_spinbox.setSingleStep(1)
                    decay_spinbox.setKeyboardTracking(False)
                    self.definition_widget.tone_table.setCellWidget(row, 1, decay_spinbox)
                decay_spinbox.setValue(decay)
        self.update_tone_table()

    def enable_sine_tone_modifications(self, enabled=True):
        pass

    def enable_sine_decay_modifications(self, enabled=True):
        pass

    def update_tone_table(self):
        if self.definition_widget.from_spec_button.isChecked():
            # Get frequencies from the specification
            num_freqs = self.definition_widget.breakpoint_table.rowCount()
            freqs = np.empty(num_freqs, "float")
            for row in range(num_freqs):
                freqs[row] = self.definition_widget.breakpoint_table.cellWidget(row, 0).value()
            self.set_sine_tone_values(freqs)
            self.enable_sine_tone_modifications(False)
        elif self.definition_widget.octave_button.isChecked():
            # Get frequencies from octave spacing
            


    def update_decay_table(self):
        pass

    def add_tone(self):
        pass

    def remove_tone(self):
        pass

    def update_control_channels(self):
        pass

    def check_selected_control_channels(self):
        pass

    def uncheck_selected_control_channels(self):
        pass

    def synthesize_sds(self):
        pass

    # %% Predictions

    def show_max_voltage_prediction(self):
        """Callback to find and plot the time history showing the maximum drive voltage required"""
        widget = self.prediction_widget.excitation_voltage_list
        index = np.argmax([float(widget.item(v).text()) for v in range(widget.count())])
        self.prediction_widget.excitation_selector.setCurrentIndex(index)

    def show_min_voltage_prediction(self):
        """Callback to find and plot the time history showing the minimum drive voltage required"""
        widget = self.prediction_widget.excitation_voltage_list
        index = np.argmin([float(widget.item(v).text()) for v in range(widget.count())])
        self.prediction_widget.excitation_selector.setCurrentIndex(index)

    def show_max_error_prediction(self):
        """Callback to find and plot the time history with the largest error compared to spec"""
        widget = self.prediction_widget.response_error_list
        index = np.argmax([float(widget.item(v).text()) for v in range(widget.count())])
        self.prediction_widget.response_selector.setCurrentIndex(index)

    def show_min_error_prediction(self):
        """Callback to find and plot the time history with the smallest error compared to spec"""
        widget = self.prediction_widget.response_error_list
        index = np.argmin([float(widget.item(v).text()) for v in range(widget.count())])
        self.prediction_widget.response_selector.setCurrentIndex(index)

    def update_response_error_prediction_selector(self, item):
        """Callback to update the response prediction selector when an item is doubleclicked"""
        index = self.prediction_widget.response_error_list.row(item)
        self.prediction_widget.response_selector.setCurrentIndex(index)

    def update_excitation_prediction_selector(self, item):
        """Callback to update the drive predition selector when an item is doubleclicked"""
        index = self.prediction_widget.excitation_voltage_list.row(item)
        self.prediction_widget.excitation_selector.setCurrentIndex(index)

    def recompute_predictions(self):
        """Recomputes the control predictions"""
        self.environment_command_queue.put(
            self.log_name, (ShockCommands.PERFORM_CONTROL_PREDICTION, False)
        )

    # %% Control

    def start_control(self):
        """Starts the chain of events to start the environment"""

    def stop_control(self):
        """Starts the sequence of events to stop the controller prematurely"""
        self.environment_command_queue.put(self.log_name, (ShockCommands.STOP_CONTROL, None))

    def enable_control(self, enabled):
        """Enables or disables the buttons to start control if it's already running"""

    def change_test_level_from_profile(self, test_level):
        """Updates the test level based on a profile event"""
        self.run_widget.test_level_selector.setValue(int(test_level))

    # %% Misc

    def retrieve_metadata(self, netcdf_handle: nc4.Dataset = None, environment_name=None):
        group = super().retrieve_metadata(netcdf_handle, environment_name)

    def update_gui(self, queue_data):
        if super().update_gui(queue_data):
            return

    def set_parameters_from_template(self, worksheet):
        self.definition_widget.block_size_selector.setValue(int(worksheet.cell(2, 2).value))

        # Sine tones
        if worksheet.cell(3, 2).value.lower() in [
            "from specification",
            "from spec",
            "from_specification",
            "from_spec",
        ]:
            self.definition_widget.from_spec_button.setChecked(True)
        elif worksheet.cell(3, 2).value.lower() in ["octave", "oct"]:
            self.definition_widget.octave_button.setChecked(True)
            self.definition_widget.min_frequency_selector.setValue(
                float(worksheet.cell(3, 3).value)
            )
            self.definition_widget.max_frequency_selector.setValue(
                float(worksheet.cell(3, 4).value)
            )
            self.definition_widget.tones_per_octave_selector.setValue(
                float(worksheet.cell(3, 5).value)
            )
        elif worksheet.cell(3, 2).value.lower() in ["manual"]:
            self.definition_widget.manual_button.setChecked(True)
            freqs = []
            column = 3
            while True:
                try:
                    freqs.append(float(worksheet.cell(3, column).value))
                except (TypeError, ValueError):
                    break
                column += 1
            self.definition_widget.tone_table.setRowCount(len(freqs))
            # TODO: Set all tone values in table
        else:
            raise ValueError(
                f"Unknown Sine Tone Strategy {worksheet.cell(3, 2).value}.  "
                'Should be one of "From Specification", "Octave", or "Manual"'
            )

        # Compensation Pulse
        if worksheet.cell(4, 2).value.upper() == "Y":
            self.definition_widget.use_compensation_pulse_checkbox.setChecked(True)
            try:
                if worksheet.cell(4, 3).value.lower() == "auto":
                    self.definition_widget.autoselect_comp_frequency_checkbox.setChecked(True)
                else:
                    self.definition_widget.autoselect_comp_frequency_checkbox.setChecked(False)
                    self.definition_widget.compensation_frequency_selector.setValue(
                        float(worksheet.cell(4, 3).value)
                    )
            except AttributeError:
                self.definition_widget.autoselect_comp_frequency_checkbox.setChecked(False)
                self.definition_widget.compensation_frequency_selector.setValue(
                    float(worksheet.cell(4, 3).value)
                )
            self.definition_widget.compensation_decay_selector.setValue(
                float(worksheet.cell(4, 4).value) * 100
            )
        else:
            self.definition_widget.use_compensation_pulse_checkbox.setChecked(False)

        # Decay Values
        if worksheet.cell(5, 2).value.lower().replace("_", " ") in ["damping", "zeta"]:
            self.definition_widget.damping_zeta_button.setChecked(True)
        elif worksheet.cell(5, 2).value.lower().replace("_", " ") in [
            "time constant",
            "tc",
            "time constant",
            "tau",
            "time const",
            "time const",
        ]:
            self.definition_widget.time_constant_tau_button.setChecked(True)
        elif worksheet.cell(5, 2).value.lower().replace("_", " ") in [
            "ntc",
            "num time const",
            "num time constants",
            "number of time constants",
        ]:
            self.definition_widget.num_time_constants_button.setChecked(True)
        else:
            raise ValueError(
                f"Unknown Decay Strategy {worksheet.cell(5, 2).value}.  "
                'Should be one of "Damping", "Time Constant", or "Num Time Constants"'
            )
        decay_values = []
        column = 3
        while True:
            try:
                decay_values.append(float(worksheet.cell(5, column).value))
            except (TypeError, ValueError):
                break
            column += 1
        if len(decay_values) == 1:
            self.definition_widget.common_decay_checkbox.setChecked(True)
            self.definition_widget.decay_value_selector.setValue(decay_values[0])
        else:
            self.definition_widget.common_decay_checkbox.setChecked(False)
            # TODO: Set set all comboboxes in the sine tone table to these values.

        # SRS
        srs_type_options = [
            self.definition_widget.srs_type_setter.itemText(i).lower()
            for i in range(self.definition_widget.srs_type_setter.count())
        ]
        srs_displacement_options = [
            self.definition_widget.srs_displacement_setter.itemText(i).lower()
            for i in range(self.definition_widget.srs_displacement_setter.count())
        ]
        try:
            srs_type_index = srs_type_options.index(
                worksheet.cell(6, 2).value.lower().replace("_", " ")
            )
        except ValueError as exc:
            raise ValueError(
                f"Unknown SRS Type {worksheet.cell(6, 2).value}.  "
                f'Should be one of {", ".join(srs_type_options)}'
            ) from exc
        self.definition_widget.srs_type_setter.setCurrentIndex(srs_type_index)
        try:
            srs_displacement_index = srs_displacement_options.index(
                worksheet.cell(6, 3).value.lower().replace("_", " ")
            )
        except ValueError as exc:
            raise ValueError(
                f"Unknown SRS Displacement Type {worksheet.cell(6, 3).value}.  "
                f'Should be one of {", ".join(srs_displacement_options)}'
            ) from exc
        self.definition_widget.srs_displacement_setter.setCurrentIndex(srs_displacement_index)
        self.definition_widget.srs_damping_setter.setValue(100 * float(worksheet.cell(6, 4).value))

        # Control Script and Parameters
        if worksheet.cell(8, 2).value is not None and worksheet.cell(8, 2).value != "":
            self.select_python_module(None, worksheet.cell(8, 2).value)
            self.definition_widget.python_class_input.setCurrentIndex(
                self.definition_widget.python_class_input.findText(worksheet.cell(9, 2).value)
            )
        self.definition_widget.control_parameters_text_input.setText(
            "" if worksheet.cell(10, 2).value is None else str(worksheet.cell(10, 2).value)
        )

        # Control channels
        column_index = 2
        while True:
            value = worksheet.cell(11, column_index).value
            if value is None or (isinstance(value, str) and value.strip() == ""):
                break
            item = self.definition_widget.control_channels_selector.item(int(value) - 1)
            item.setCheckState(Qt.Checked)
            column_index += 1

        # System identification
        self.system_id_widget.samplesPerFrameSpinBox.setValue(int(worksheet.cell(12, 2).value))
        self.system_id_widget.averagingTypeComboBox.setCurrentIndex(
            self.system_id_widget.averagingTypeComboBox.findText(worksheet.cell(13, 2).value)
        )
        self.system_id_widget.noiseAveragesSpinBox.setValue(int(worksheet.cell(14, 2).value))
        self.system_id_widget.systemIDAveragesSpinBox.setValue(int(worksheet.cell(15, 2).value))
        self.system_id_widget.averagingCoefficientDoubleSpinBox.setValue(
            float(worksheet.cell(16, 2).value)
        )
        self.system_id_widget.estimatorComboBox.setCurrentIndex(
            self.system_id_widget.estimatorComboBox.findText(worksheet.cell(17, 2).value)
        )
        self.system_id_widget.levelDoubleSpinBox.setValue(float(worksheet.cell(18, 2).value))
        self.system_id_widget.levelRampTimeDoubleSpinBox.setValue(
            float(worksheet.cell(19, 2).value)
        )
        self.system_id_widget.signalTypeComboBox.setCurrentIndex(
            self.system_id_widget.signalTypeComboBox.findText(worksheet.cell(20, 2).value)
        )
        self.system_id_widget.windowComboBox.setCurrentIndex(
            self.system_id_widget.windowComboBox.findText(worksheet.cell(21, 2).value)
        )
        self.system_id_widget.overlapDoubleSpinBox.setValue(float(worksheet.cell(22, 2).value))
        self.system_id_widget.onFractionDoubleSpinBox.setValue(float(worksheet.cell(23, 2).value))
        self.system_id_widget.pretriggerDoubleSpinBox.setValue(float(worksheet.cell(24, 2).value))
        self.system_id_widget.rampFractionDoubleSpinBox.setValue(float(worksheet.cell(25, 2).value))

        # Transformation matrices
        response_channels = self.definition_widget.control_channels_display.value()
        output_channels = self.definition_widget.output_channels_display.value()
        output_transform_row = 27
        if (
            isinstance(worksheet.cell(26, 2).value, str)
            and worksheet.cell(26, 2).value.lower() == "none"
        ):
            self.response_transformation_matrix = None
        else:
            while True:
                if worksheet.cell(output_transform_row, 1).value == "Output Transformation Matrix:":
                    break
                output_transform_row += 1
            response_size = output_transform_row - 26
            response_transformation = []
            for i in range(response_size):
                response_transformation.append([])
                for j in range(response_channels):
                    response_transformation[-1].append(float(worksheet.cell(26 + i, 2 + j).value))
            self.response_transformation_matrix = np.array(response_transformation)
        if (
            isinstance(worksheet.cell(output_transform_row, 2).value, str)
            and worksheet.cell(output_transform_row, 2).value.lower() == "none"
        ):
            self.output_transformation_matrix = None
        else:
            output_transformation = []
            i = 0
            while True:
                if worksheet.cell(output_transform_row + i, 2).value is None or (
                    isinstance(worksheet.cell(output_transform_row + i, 2).value, str)
                    and worksheet.cell(output_transform_row + i, 2).value.strip() == ""
                ):
                    break
                output_transformation.append([])
                for j in range(output_channels):
                    output_transformation[-1].append(
                        float(worksheet.cell(output_transform_row + i, 2 + j).value)
                    )
                i += 1
            self.output_transformation_matrix = np.array(output_transformation)
        self.define_transformation_matrices(None, dialog=False)

        # Load in the specification
        self.load_specification(None, worksheet.cell(7, 2).value)

    @staticmethod
    def create_environment_template(environment_name, workbook):

        worksheet.cell(22, 1, "Response Transformation Matrix:")
        worksheet.cell(
            22,
            2,
            "# Transformation matrix to apply to the response channels.  Type None if there "
            "is none.  Otherwise, make this a 2D array in the spreadsheet and move the Output "
            "Transformation Matrix line down so it will fit.  The number of columns should be "
            "the number of physical control channels.",
        )
        worksheet.cell(23, 1, "Output Transformation Matrix:")
        worksheet.cell(
            23,
            2,
            "# Transformation matrix to apply to the outputs.  Type None if there is none.  "
            "Otherwise, make this a 2D array in the spreadsheet.  The number of columns should "
            "be the number of physical output channels in the environment.",
        )


# %% Environment


class ShockEnvironment(AbstractSysIdEnvironment):
    """Class defining calculations for the Shock environment"""

    def __init__(
        self,
        environment_name: str,
        queue_container: ShockQueues,
        acquisition_active: mp.sharedctypes.Synchronized,
        output_active: mp.sharedctypes.Synchronized,
    ):
        super().__init__(
            environment_name,
            queue_container.environment_command_queue,
            queue_container.gui_update_queue,
            queue_container.controller_communication_queue,
            queue_container.log_file_queue,
            queue_container.collector_command_queue,
            queue_container.signal_generation_command_queue,
            queue_container.spectral_command_queue,
            queue_container.data_analysis_command_queue,
            queue_container.data_in_queue,
            queue_container.data_out_queue,
            acquisition_active,
            output_active,
        )
        self.map_command(
            ShockCommands.PERFORM_CONTROL_PREDICTION,
            self.perform_control_prediction,
        )
        self.map_command(ShockCommands.START_CONTROL, self.start_control)
        self.map_command(ShockCommands.STOP_CONTROL, self.stop_environment)
        # self.map_command(
        #     GlobalCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS,
        #     self.update_interactive_control_parameters,
        # )
        # self.map_command(GlobalCommands.SEND_INTERACTIVE_COMMAND, self.send_interactive_command)

    def initialize_environment_test_parameters(self, environment_parameters: ShockMetadata):
        super().initialize_environment_test_parameters(environment_parameters)
        self.environment_parameters: ShockMetadata

    def system_id_complete(self, data):
        """Sends the message that system identification is complete and control calculations
        should be performed"""
        super().system_id_complete(data)
        (
            self.frames,
            _,  # avg,
            self.frequencies,
            self.frf,
            self.sysid_coherence,
            self.sysid_response_cpsd,
            self.sysid_reference_cpsd,
            self.sysid_condition,
            self.sysid_response_noise,
            self.sysid_reference_noise,
        ) = data
        # Perform the control prediction
        self.perform_control_prediction(True)

    def perform_control_prediction(self, sysid_update):
        """Performs the control prediction based on system identification information"""
        self.show_test_prediction()

    def show_test_prediction(self):
        """Sends the test predictions to the UI"""

    def get_signal_generation_metadata(self):
        """Collects the metadata required to define the signal generation process"""
        return SignalGenerationMetadata(
            samples_per_write=self.data_acquisition_parameters.samples_per_write,
            level_ramp_samples=self.environment_parameters.test_level_ramp_time
            * self.environment_parameters.sample_rate
            * self.data_acquisition_parameters.output_oversample,
            output_transformation_matrix=self.environment_parameters.reference_transformation_matrix,
        )

    def start_control(self, data):
        """Starts up the control to generate the signal"""
        if self.startup:
            pass

    def shutdown(self):
        """Let the UI know that this environment has completely shut down"""
        self.log("Environment Shut Down")
        self.gui_update_queue.put((self.environment_name, ("enable_control", None)))
        self.startup = True

    def stop_environment(self, data):
        """Starts the shutdown sequence based on commands from the UI"""
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.START_SHUTDOWN, None)
        )


# %% Process


def shock_process(
    environment_name: str,
    input_queue: VerboseMessageQueue,
    gui_update_queue: Queue,
    controller_communication_queue: VerboseMessageQueue,
    log_file_queue: Queue,
    data_in_queue: Queue,
    data_out_queue: Queue,
    acquisition_active: mp.sharedctypes.Synchronized,
    output_active: mp.sharedctypes.Synchronized,
):
    """
    Shock environment process function called by multiprocessing

    This function defines the Shock Environment process that
    gets run by the multiprocessing module when it creates a new process.  It
    creates a ShockEnvironment object and runs it.

    Parameters
    ----------
    environment_name : str :
        Name of the environment
    input_queue : VerboseMessageQueue :
        Queue containing instructions for the environment
    gui_update_queue : Queue :
        Queue where GUI updates are put
    controller_communication_queue : Queue :
        Queue for global communications with the controller
    log_file_queue : Queue :
        Queue for writing log file messages
    data_in_queue : Queue :
        Queue from which data will be read by the environment
    data_out_queue : Queue :
        Queue to which data will be written that will be output by the hardware.
    acquisition_active : mp.sharedctypes.Synchronized
        A synchronized value that indicates when the acquisition is active
    output_active : mp.sharedctypes.Synchronized
        A synchronized value that indicates when the output is active
    """
    try:
        # Create vibration queues
        queue_container = ShockQueues(
            environment_name,
            input_queue,
            gui_update_queue,
            controller_communication_queue,
            data_in_queue,
            data_out_queue,
            log_file_queue,
        )

        spectral_proc = mp.Process(
            target=spectral_processing_process,
            args=(
                environment_name,
                queue_container.spectral_command_queue,
                queue_container.data_for_spectral_computation_queue,
                queue_container.updated_spectral_quantities_queue,
                queue_container.environment_command_queue,
                queue_container.gui_update_queue,
                queue_container.log_file_queue,
            ),
        )
        spectral_proc.start()
        analysis_proc = mp.Process(
            target=sysid_data_analysis_process,
            args=(
                environment_name,
                queue_container.data_analysis_command_queue,
                queue_container.updated_spectral_quantities_queue,
                queue_container.time_history_to_generate_queue,
                queue_container.environment_command_queue,
                queue_container.gui_update_queue,
                queue_container.log_file_queue,
            ),
        )
        analysis_proc.start()
        siggen_proc = mp.Process(
            target=signal_generation_process,
            args=(
                environment_name,
                queue_container.signal_generation_command_queue,
                queue_container.time_history_to_generate_queue,
                queue_container.data_out_queue,
                queue_container.environment_command_queue,
                queue_container.log_file_queue,
                queue_container.gui_update_queue,
            ),
        )
        siggen_proc.start()
        collection_proc = mp.Process(
            target=data_collector_process,
            args=(
                environment_name,
                queue_container.collector_command_queue,
                queue_container.data_in_queue,
                [queue_container.data_for_spectral_computation_queue],
                queue_container.environment_command_queue,
                queue_container.log_file_queue,
                queue_container.gui_update_queue,
            ),
        )
        collection_proc.start()

        process_class = ShockEnvironment(
            environment_name, queue_container, acquisition_active, output_active
        )
        process_class.run()

        # Rejoin all the processes
        process_class.log("Joining Subprocesses")
        process_class.log("Joining Spectral Computation")
        spectral_proc.join()
        process_class.log("Joining Data Analysis")
        analysis_proc.join()
        process_class.log("Joining Signal Generation")
        siggen_proc.join()
        process_class.log("Joining Data Collection")
        collection_proc.join()
    except Exception:  # pylint: disable = broad-exception-caught
        print(traceback.format_exc())
