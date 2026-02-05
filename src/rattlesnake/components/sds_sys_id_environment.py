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
from multiprocessing.queues import Queue
from enum import Enum

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
from .sds_sys_id_metadata import (
    SDSMetadata,
    ToneStrategy,
    ToneParameters,
    CompPulseParameters,
    DecayStrategy,
    DecayParameters,
    SRSType,
    SRSParameters,
    SRSDisplacementType,
    SDSParameters,
    SpecParameters,
    ControlLawType,
    ControlParameters,
)
from .sds_sys_id_utilities import (
    octspace,
    convert_damping_strategy,
    SDSQueues,
    SDSCommands,
    sum_decayed_sines_reconstruction,
    srs as srs_function,
)
from .sds_sys_id_prediction_table import SDSPredictionTable
from .sds_sys_id_synthesize_dialog import SDSSynthesizeDialog
from .ui_utilities import (
    PlotTimeWindow,
    TransformationMatrixWindow,
    colororder,
    load_time_history,
    multiline_plotter,
    AdaptiveNoWheelSpinBox,
    ScientificDoubleSpinBox,
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


class SDSUI(AbstractSysIdUI):
    """Class defining the user interface for the SDS environment"""

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
        self.python_function_extra_arguments = []
        self.python_function_extra_argument_widgets = {}
        self.decay_values_current_strategy = DecayStrategy.NUM_TIME_CONSTANTS

        self.control_selector_widgets = [self.definition_widget.specification_plot_selector]

        self.output_selector_widgets = []

        self.prediction_table = SDSPredictionTable(
            self.prediction_widget.prediction_table_placeholder,
            environment_command_queue,
            self.log_name,
        )
        self.prediction_table.lock_table()

        self.plotwidgets = [
            self.definition_widget.specification_plot,
            self.run_widget.global_test_performance_plot,
        ]

        for plotwidget in self.plotwidgets:
            plot_item = plotwidget.getPlotItem()
            plot_item.showGrid(True, True, 0.25)
            plot_item.enableAutoRange()
            plot_item.getViewBox().enableAutoRange(enable=True)
            plot_item.setLogMode(True, True)

        self.connect_callbacks()

        self.select_python_module(default=True)

    def connect_callbacks(self):
        """Connects the callbacks to the SDS UI widgets"""
        # Definition
        self.definition_widget.add_breakpoint_button.clicked.connect(self.add_breakpoint)
        self.definition_widget.remove_breakpoint_button.clicked.connect(self.remove_breakpoint)
        self.definition_widget.from_spec_button.toggled.connect(self.update_tone_table)
        self.definition_widget.octave_button.toggled.connect(self.update_tone_table)
        self.definition_widget.manual_button.toggled.connect(self.update_tone_table)
        self.definition_widget.min_frequency_selector.valueChanged.connect(self.update_tone_table)
        self.definition_widget.max_frequency_selector.valueChanged.connect(self.update_tone_table)
        self.definition_widget.tones_per_octave_selector.valueChanged.connect(
            self.update_tone_table
        )
        self.definition_widget.common_decay_checkbox.toggled.connect(self.update_decay_table)
        self.definition_widget.decay_value_selector.valueChanged.connect(self.update_decay_table)
        self.definition_widget.damping_zeta_button.toggled.connect(self.update_decay_table)
        self.definition_widget.time_constant_tau_button.toggled.connect(self.update_decay_table)
        self.definition_widget.num_time_constants_button.toggled.connect(self.update_decay_table)
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
        self.definition_widget.use_compensation_pulse_checkbox.toggled.connect(
            self.update_compensation_pulse
        )
        self.definition_widget.autoselect_comp_frequency_checkbox.toggled.connect(
            self.update_compensation_pulse
        )
        self.definition_widget.control_function_input.currentIndexChanged.connect(
            self.set_up_widgets
        )
        # Prediction
        # Run Test

    # %% Data Acquisition

    def initialize_data_acquisition(self, data_acquisition_parameters: DataAcquisitionParameters):
        super().initialize_data_acquisition(data_acquisition_parameters)
        # Initialize and clear plots
        for plotwidget in self.plotwidgets:
            plotwidget.clear()
        self.definition_widget.specification_plot.getPlotItem().addLegend()
        self.plot_data_items[
            "specification_srs"
        ] = self.definition_widget.specification_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.zeros(2),
            pen={"color": "b", "width": 1},
            name="Control SRS",
        )
        self.plot_data_items[
            "specification_lower_limit"
        ] = self.definition_widget.specification_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.zeros(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
            name="Limit",
        )
        self.plot_data_items[
            "specification_upper_limit"
        ] = self.definition_widget.specification_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.zeros(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
        )

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

    @staticmethod
    def valid_annotation(annotation):
        if annotation == int:
            return True
        if annotation == float:
            return True
        if annotation == str:
            return True
        if issubclass(annotation, Enum):
            return True
        return False

    @staticmethod
    def get_valid_control_laws(module):
        required_control_law_arguments = {
            "environment_parameters",
            "transfer_function_frequencies",
            "transfer_function",
            "noise_response_cpsd",
            "noise_reference_cpsd",
            "sysid_response_cpsd",
            "sysid_reference_cpsd",
            "multiple_coherence",
            "frames",
            "last_response_srs",
            "last_drive_amplitudes",
            "last_drive_decays",
            "last_drive_delays",
        }
        valid_control_laws = []
        members = inspect.getmembers(module)
        for objname, obj in members:
            print(f"Analyzing member {objname}")
            valid_control_law = True
            # Check if it is a function
            if inspect.isfunction(obj):
                signature = inspect.signature(obj)
                parameters = signature.parameters
                # Check if is a valid object
                if not all(arg in parameters for arg in required_control_law_arguments):
                    print(f"Member {objname} does not have all required arguments")
                    continue
                # Get extra arguments
                extra_arguments = []
                print(f"{signature=}")
                print(f"{parameters=}")
                for name, parameter in parameters.items():
                    # Extra arguments are not required arguments
                    if name in required_control_law_arguments:
                        print(f"  Argument {name} is a required argument")
                        continue
                    # Extra arguments must be be able to be set by keyword
                    elif parameter.kind != inspect.Parameter.POSITIONAL_ONLY:
                        annotation = parameter.annotation
                        default = parameter.default
                        if (
                            not SDSUI.valid_annotation(annotation)
                            and default == inspect.Parameter.empty
                        ):
                            # If it doesn't have an annotation, we can't automatically create a
                            # widget for it, so if it also doesn't have a default argument, we can't
                            # use it.
                            print(
                                f"  Argument {name} has no valid annotation or default "
                                "value, invalid control law"
                            )
                            valid_control_law = False
                            break
                        if (
                            not SDSUI.valid_annotation(annotation)
                            and default != inspect.Parameter.empty
                        ):
                            print(
                                f"  Argument {name} has an invalid annotation but contains a "
                                "default value which will be used, and will therefore not be "
                                "treated as an extra parameter"
                            )
                            continue
                        extra_arguments.append([name, annotation, default])
                    elif (
                        parameter.kind == inspect.Parameter.POSITIONAL_ONLY
                        and default == inspect.Parameter.empty
                    ):
                        print(
                            f"  Argument {name} is positional only without a "
                            "default and therefore cannot be specified."
                        )
                        valid_control_law = False
                        break
                    else:
                        print(
                            f"  Argument {name} is positional only but has a default argument"
                            "which will be used, and will therefore not be treated as an "
                            "extra parameter"
                        )
                        continue
                if not valid_control_law:
                    continue
                valid_control_laws.append([objname, extra_arguments])
            else:
                print(f"Member {objname} is not a valid type of object to be a control law")
        print(valid_control_laws)
        return valid_control_laws

    def select_python_module(
        self, clicked=None, filename=None, default=False
    ):  # pylint: disable=unused-argument
        """Loads a Python module using a dialog or the specified filename

        Parameters
        ----------
        clicked :
            The clicked event that triggered the callback.
        filename :
            File name defining the Python module for bypassing the callback when
            loading from a file (Default value = None).

        """
        if default:
            self.python_control_module = importlib.import_module(
                "rattlesnake.components.sds_sys_id_control_law"
            )
            filename = "rattlesnake.components.sds_sys_id_control_law"
        else:
            if filename is None or not os.path.isfile(filename):
                filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self.definition_widget,
                    "Select Python Module",
                    filter="Python Modules (*.py)",
                )
                if filename == "":
                    return
            self.python_control_module = load_python_module(filename)

        # Any valid control law must have the required arguments

        # Analyze the functions and classes in the module

        self.python_function_extra_arguments = SDSUI.get_valid_control_laws(
            self.python_control_module
        )
        self.log(
            f"Loaded module {self.python_control_module.__name__} with "
            f"functions {[function[0] for function in self.python_function_extra_arguments]}"
        )
        self.definition_widget.control_function_input.clear()
        self.definition_widget.control_script_file_path_input.setText(filename)
        for function in self.python_function_extra_arguments:
            self.definition_widget.control_function_input.addItem(function[0])
        self.set_up_widgets()

    @staticmethod
    def create_widget_for_type(arg_type, default_value):
        if arg_type == int:
            widget = QtWidgets.QSpinBox()
            widget.setMinimum(1000000)
            widget.setMaximum(1000000)
            widget.setValue(default_value)
        elif arg_type == float:
            widget = ScientificDoubleSpinBox()
            widget.setValue(default_value)
        elif arg_type == str:
            widget = QtWidgets.QTextEdit()
            widget.setText(default_value)
        elif issubclass(arg_type, Enum):
            widget = QtWidgets.QComboBox()
            for e in arg_type:
                widget.addItem(e.name, e)
            widget.setCurrentText(default_value.name)
        else:
            raise ValueError(f"Unsupported argument type: {arg_type}")
        return widget

    def set_up_widgets(self):
        # Clear out existing widgets
        layout = self.definition_widget.control_parameters_widget_layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.python_function_extra_argument_widgets.clear()
        index = self.definition_widget.control_function_input.currentIndex()
        _, extra_arguments = self.python_function_extra_arguments[index]
        for arg_name, arg_type, arg_default in extra_arguments:
            widget = SDSUI.create_widget_for_type(arg_type, arg_default)
            layout.addRow(arg_name, widget)
            self.python_function_extra_argument_widgets[arg_name] = widget

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

    def collect_specification(self):
        num_freqs = self.definition_widget.breakpoint_table.rowCount()
        num_control = self.definition_widget.breakpoint_table.columnCount() - 1
        freqs = np.empty(num_freqs, "float")
        srss = np.empty((num_freqs, num_control), "float")
        lower_limits = np.empty((num_freqs, num_control), "float")
        upper_limits = np.empty((num_freqs, num_control), "float")
        if self.definition_widget.from_spec_button.isChecked():
            self.update_tone_table()
        for row in range(num_freqs):
            freqs[row] = self.definition_widget.breakpoint_table.cellWidget(row, 0).value()
            for col in range(num_control):
                srss[row, col] = (
                    np.nan
                    if self.definition_widget.breakpoint_table.cellWidget(row, 1 + col).value() == 0
                    else self.definition_widget.breakpoint_table.cellWidget(row, 1 + col).value()
                )
                lower_limits[row, col] = (
                    np.nan
                    if self.definition_widget.lower_limit_table.cellWidget(row, 1 + col).value()
                    == 0
                    else self.definition_widget.lower_limit_table.cellWidget(row, 1 + col).value()
                )
                upper_limits[row, col] = (
                    np.nan
                    if self.definition_widget.upper_limit_table.cellWidget(row, 1 + col).value()
                    == 0
                    else self.definition_widget.upper_limit_table.cellWidget(row, 1 + col).value()
                )
        spec_data = SpecParameters(
            freqs, srss, lower_limits, upper_limits, self.definition_widget.num_hits_spinbox.value()
        )
        return spec_data

    def collect_control_extra_parameters(self):
        kwargs = {}
        for arg_name, widget in self.python_function_extra_argument_widgets.items():
            if isinstance(widget, QtWidgets.QSpinBox):
                kwargs[arg_name] = widget.value()
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                kwargs[arg_name] = widget.value()
            elif isinstance(widget, QtWidgets.QTextEdit):
                kwargs[arg_name] = widget.toPlainText()
            elif isinstance(widget, QtWidgets.QComboBox):
                kwargs[arg_name] = widget.currentData()
            else:
                raise ValueError(f"Unsupported widget type: {type(widget)}")
        print(f"Got Arguments {kwargs=}")
        return kwargs

    def collect_control_data(self):
        control_module = self.definition_widget.control_script_file_path_input.text()
        control_function = self.definition_widget.control_function_input.itemText(
            self.definition_widget.control_function_input.currentIndex()
        )
        control_function_type = ControlLawType(
            self.definition_widget.control_function_generator_selector.currentIndex()
        )
        control_function_parameters = self.collect_control_extra_parameters()
        control_data = ControlParameters(
            control_module, control_function, control_function_type, control_function_parameters
        )
        return control_data

    def collect_tone_data(self):
        if self.definition_widget.from_spec_button.isChecked():
            tone_data = ToneParameters(ToneStrategy.FROM_SPEC, None)
        elif self.definition_widget.octave_button.isChecked():
            tone_data = ToneParameters(
                ToneStrategy.OCTAVE,
                np.array(
                    [
                        self.definition_widget.min_frequency_selector.value(),
                        self.definition_widget.max_frequency_selector.value(),
                        self.definition_widget.tones_per_octave_selector.value(),
                    ]
                ),
            )
        elif self.definition_widget.manual_button.isChecked():
            num_rows = self.definition_widget.tone_table.rowCount()
            freq = np.empty(num_rows, "float")
            for row in num_rows:
                freq[row] = self.definition_widget.tone_table.cellWidget(row, 0).value()
            tone_data = ToneParameters(ToneStrategy.MANUAL, freq)
        else:
            raise ValueError("Invalid Tone Strategy (how did you get here?!)")
        return tone_data

    def collect_compensation_pulse_data(self):
        compensation_pulse_data = CompPulseParameters(
            self.definition_widget.use_compensation_pulse_checkbox.isChecked(),
            (
                None
                if self.definition_widget.autoselect_comp_frequency_checkbox.isChecked()
                else self.definition_widget.compensation_frequency_selector.value()
            ),
            self.definition_widget.compensation_decay_selector.value() / 100,
        )
        return compensation_pulse_data

    def collect_decay_data(self):
        if self.definition_widget.damping_zeta_button.isChecked():
            decay_strategy = DecayStrategy.DAMPING
        elif self.definition_widget.time_constant_tau_button.isChecked():
            decay_strategy = DecayStrategy.TIME_CONSTANT
        elif self.definition_widget.num_time_constants_button.isChecked():
            decay_strategy = DecayStrategy.NUM_TIME_CONSTANTS
        else:
            raise ValueError("Invalid Decay Strategy (how did you get here?!)")
        common_decay = self.definition_widget.common_decay_checkbox.isChecked()
        if common_decay:
            decay_data = self.definition_widget.decay_value_selector.value()
        else:
            num_rows = self.definition_widget.tone_table.rowCount()
            decay_data = np.empty(num_rows, "float")
            for row in num_rows:
                decay_data[row] = self.definition_widget.tone_table.cellWidget(row, 1).value()
        decay_parameters = DecayParameters(decay_strategy, common_decay, decay_data)
        return decay_parameters

    def collect_srs_data(self):
        srs_data = SRSParameters(
            SRSType(self.definition_widget.srs_type_setter.currentIndex() + 1),
            (
                SRSDisplacementType.ABSOLUTE
                if self.definition_widget.srs_displacement_setter.currentIndex() == 0
                else SRSDisplacementType.RELATIVE
            ),
            self.definition_widget.srs_damping_setter.value() / 100,
        )
        return srs_data

    def collect_sds_data(self):
        sds_data = SDSParameters(
            self.definition_widget.sds_iterations_selector.value(),
            self.definition_widget.sds_convergence_selector.value() / 100,
            self.definition_widget.sds_scale_factor_selector.value() / 100,
            self.definition_widget.error_tolerance_selector.value() / 100,
        )
        return sds_data

    def collect_environment_definition_parameters(self):
        """Collects the metadata defining the environment from the UI widgets"""
        control_data = self.collect_control_data()
        tone_data = self.collect_tone_data()
        compensation_pulse_data = self.collect_compensation_pulse_data()
        decay_parameters = self.collect_decay_data()
        srs_data = self.collect_srs_data()
        sds_data = self.collect_sds_data()
        spec_data = self.collect_specification()
        return SDSMetadata(
            sample_rate=self.data_acquisition_parameters.sample_rate,
            num_channels=len(self.data_acquisition_parameters.channel_list),
            block_size=self.definition_widget.block_size_selector.value(),
            tone_data=tone_data,
            compensation_pulse_data=compensation_pulse_data,
            decay_data=decay_parameters,
            srs_data=srs_data,
            sds_data=sds_data,
            control_script_data=control_data,
            control_channel_indices=self.physical_control_indices,
            output_channel_indices=self.physical_output_indices,
            response_transformation_matrix=self.response_transformation_matrix,
            excitation_transformation_matrix=self.output_transformation_matrix,
            specification_data=spec_data,
        )

    def initialize_environment(self):
        super().initialize_environment()
        self.prediction_table.update_names(
            self.initialized_output_names, self.initialized_control_names
        )
        self.prediction_table.update_parameters(self.environment_parameters)
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
                freq_spinbox = self.definition_widget.tone_table.cellWidget(row, 0)
                decay_spinbox = self.definition_widget.tone_table.cellWidget(row, 1)
                # If there isn't, make one
                if freq_spinbox is None:
                    freq_spinbox = AdaptiveNoWheelSpinBox()
                    freq_spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
                    freq_spinbox.setSingleStep(1)
                    freq_spinbox.setKeyboardTracking(False)
                    freq_spinbox.setDecimals(4)
                    freq_spinbox.valueChanged.connect(self.update_sine_table)
                    self.definition_widget.tone_table.setCellWidget(row, 0, freq_spinbox)
                if decay_spinbox is None:
                    decay_spinbox = AdaptiveNoWheelSpinBox()
                    decay_spinbox.setRange(0, 1000000)
                    decay_spinbox.setSingleStep(1)
                    decay_spinbox.setKeyboardTracking(False)
                    decay_spinbox.valueChanged.connect(self.update_sine_table)
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
                freq_spinbox = self.definition_widget.tone_table.cellWidget(row, 0)
                decay_spinbox = self.definition_widget.tone_table.cellWidget(row, 1)
                # If there isn't, make one
                if freq_spinbox is None:
                    freq_spinbox = AdaptiveNoWheelSpinBox()
                    freq_spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
                    freq_spinbox.setSingleStep(1)
                    freq_spinbox.setKeyboardTracking(False)
                    freq_spinbox.setDecimals(4)
                    freq_spinbox.setValue(0)
                    freq_spinbox.valueChanged.connect(self.update_sine_table)
                    self.definition_widget.tone_table.setCellWidget(row, 0, freq_spinbox)
                if decay_spinbox is None:
                    decay_spinbox = AdaptiveNoWheelSpinBox()
                    decay_spinbox.setRange(0, 1000000)
                    decay_spinbox.setSingleStep(1)
                    decay_spinbox.setKeyboardTracking(False)
                    decay_spinbox.valueChanged.connect(self.update_sine_table)
                    self.definition_widget.tone_table.setCellWidget(row, 1, decay_spinbox)
                decay_spinbox.setValue(decay)
        if decays is None:
            self.update_decay_table()

    def enable_sine_tone_modifications(self, enabled=True):
        self.definition_widget.add_tone_button.setVisible(enabled)
        self.definition_widget.remove_tone_button.setVisible(enabled)
        for row in range(self.definition_widget.tone_table.rowCount()):
            widget = self.definition_widget.tone_table.cellWidget(row, 0)
            widget.setReadOnly(not enabled)
            widget.setButtonSymbols(
                AdaptiveNoWheelSpinBox.UpDownArrows if enabled else AdaptiveNoWheelSpinBox.NoButtons
            )

    def enable_sine_decay_modifications(self, enabled=True):
        for row in range(self.definition_widget.tone_table.rowCount()):
            widget = self.definition_widget.tone_table.cellWidget(row, 1)
            widget.setReadOnly(not enabled)
            widget.setButtonSymbols(
                AdaptiveNoWheelSpinBox.UpDownArrows if enabled else AdaptiveNoWheelSpinBox.NoButtons
            )
        self.definition_widget.decay_value_selector.setVisible(not enabled)

    def enable_octave_buttons(self, enabled=True):
        for widget in [
            self.definition_widget.min_frequency_label,
            self.definition_widget.min_frequency_selector,
            self.definition_widget.max_frequency_label,
            self.definition_widget.max_frequency_selector,
            self.definition_widget.tones_per_octave_label,
            self.definition_widget.tones_per_octave_selector,
        ]:
            widget.setVisible(enabled)

    def update_tone_table(self):
        if self.definition_widget.from_spec_button.isChecked():
            # Get frequencies from the specification
            num_freqs = self.definition_widget.breakpoint_table.rowCount()
            freqs = np.empty(num_freqs, "float")
            for row in range(num_freqs):
                freqs[row] = self.definition_widget.breakpoint_table.cellWidget(row, 0).value()
            self.set_sine_tone_values(freqs)
            self.enable_sine_tone_modifications(False)
            self.enable_octave_buttons(False)
        elif self.definition_widget.octave_button.isChecked():
            # Get frequencies from octave spacing
            freqs = octspace(
                self.definition_widget.min_frequency_selector.value(),
                self.definition_widget.max_frequency_selector.value(),
                self.definition_widget.tones_per_octave_selector.value(),
            )
            self.set_sine_tone_values(freqs)
            self.enable_sine_tone_modifications(False)
            self.enable_octave_buttons(True)
        elif self.definition_widget.manual_button.isChecked():
            # Allow manual specification of frequencies
            self.enable_octave_buttons(False)
            self.enable_sine_tone_modifications(True)
        self.update_decay_table()

    def update_decay_table(self):
        if self.definition_widget.common_decay_checkbox.isChecked():
            num_decays = self.definition_widget.tone_table.rowCount()
            decay_values = self.definition_widget.decay_value_selector.value() * np.ones(num_decays)
            self.set_sine_tone_values(decays=decay_values)
            if self.definition_widget.damping_zeta_button.isChecked():
                self.decay_values_current_strategy = DecayStrategy.DAMPING
            elif self.definition_widget.time_constant_tau_button.isChecked():
                self.decay_values_current_strategy = DecayStrategy.TIME_CONSTANT
            elif self.definition_widget.num_time_constants_button.isChecked():
                self.decay_values_current_strategy = DecayStrategy.NUM_TIME_CONSTANTS
            self.enable_sine_decay_modifications(False)
        else:
            # Get old version
            current_decay_strategy = self.decay_values_current_strategy
            if self.definition_widget.damping_zeta_button.isChecked():
                self.decay_values_current_strategy = DecayStrategy.DAMPING
            elif self.definition_widget.time_constant_tau_button.isChecked():
                self.decay_values_current_strategy = DecayStrategy.TIME_CONSTANT
            elif self.definition_widget.num_time_constants_button.isChecked():
                self.decay_values_current_strategy = DecayStrategy.NUM_TIME_CONSTANTS
            num_decays = self.definition_widget.tone_table.rowCount()
            current_decay_values = np.empty(num_decays, "float")
            frequency_values = np.empty(num_decays, "float")
            for row in range(num_decays):
                current_decay_values[row] = self.definition_widget.tone_table.cellWidget(
                    row, 1
                ).value()
                frequency_values[row] = self.definition_widget.tone_table.cellWidget(row, 0).value()
            new_decay_values = convert_damping_strategy(
                current_decay_values,
                frequency_values,
                self.definition_widget.block_size_selector.value()
                / self.definition_widget.sample_rate_display.value(),
                current_decay_strategy,
                self.decay_values_current_strategy,
            )
            self.set_sine_tone_values(decays=new_decay_values)
            self.enable_sine_decay_modifications(True)

    def add_tone(self):
        selected_indices = self.definition_widget.tone_table.selectedIndexes()
        if selected_indices:
            selected_row = selected_indices[0].row()
        else:
            # If no row is selected, add the row at the end
            selected_row = self.definition_widget.tone_table.rowCount()
        self.definition_widget.tone_table.insertRow(selected_row)
        freq_spinbox = AdaptiveNoWheelSpinBox()
        freq_spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
        freq_spinbox.setSingleStep(1)
        freq_spinbox.setKeyboardTracking(False)
        freq_spinbox.setDecimals(4)
        freq_spinbox.valueChanged.connect(self.update_sine_table)
        self.definition_widget.tone_table.setCellWidget(selected_row, 0, freq_spinbox)
        decay_spinbox = AdaptiveNoWheelSpinBox()
        decay_spinbox.setRange(0, 1000000)
        decay_spinbox.setSingleStep(1)
        decay_spinbox.setKeyboardTracking(False)
        decay_spinbox.valueChanged.connect(self.update_sine_table)
        self.definition_widget.tone_table.setCellWidget(selected_row, 1, decay_spinbox)
        self.enable_sine_tone_modifications(True)
        self.update_decay_table()

    def remove_tone(self):
        selected_indices = self.definition_widget.tone_table.selectedIndexes()
        if selected_indices:
            selected_row = selected_indices[0].row()
        else:
            # If no row is selected, remove the row at the end
            selected_row = self.definition_widget.tone_table.rowCount() - 1
        self.definition_widget.tone_table.removeRow(selected_row)

    def update_compensation_pulse(self):
        visible = self.definition_widget.use_compensation_pulse_checkbox.isChecked()
        for widget in [
            self.definition_widget.autoselect_comp_frequency_checkbox,
            self.definition_widget.comp_frequency_label,
            self.definition_widget.compensation_frequency_selector,
            self.definition_widget.comp_decay_label,
            self.definition_widget.compensation_decay_selector,
        ]:
            widget.setVisible(visible)
        auto = self.definition_widget.autoselect_comp_frequency_checkbox.isChecked()
        self.definition_widget.compensation_frequency_selector.setReadOnly(auto)
        self.definition_widget.compensation_frequency_selector.setButtonSymbols(
            QtWidgets.QSpinBox.NoButtons if auto else QtWidgets.QSpinBox.UpDownArrows
        )
        if auto:
            min_freq = np.inf
            for row in range(self.definition_widget.tone_table.rowCount()):
                widget = self.definition_widget.tone_table.cellWidget(row, 0)
                if widget is not None:
                    val = widget.value()
                    min_freq = min(min_freq, val)
            if min_freq == np.inf:
                min_freq = 3
            self.definition_widget.compensation_frequency_selector.setValue(min_freq / 3)

    def update_control_channels(self):
        """Updates the control channels due to selection changes"""
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.definition_widget.control_channels_display.setValue(len(self.physical_control_indices))
        self.define_transformation_matrices(None, False)
        self.clear_and_update_specification_table()

    def check_selected_control_channels(self):
        """Checks the selected control channels on the UI"""
        for item in self.definition_widget.control_channels_selector.selectedItems():
            item.setCheckState(Qt.Checked)

    def uncheck_selected_control_channels(self):
        """Unchecks the selected control channels on the UI"""
        for item in self.definition_widget.control_channels_selector.selectedItems():
            item.setCheckState(Qt.Unchecked)

    def update_sine_table(self):
        self.update_compensation_pulse()

    def synthesize_sds(self):
        SDSSynthesizeDialog.show_dialog(self)

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
            self.log_name, (SDSCommands.PERFORM_CONTROL_PREDICTION, False)
        )

    # %% Control

    def start_control(self):
        """Starts the chain of events to start the environment"""

    def stop_control(self):
        """Starts the sequence of events to stop the controller prematurely"""
        self.environment_command_queue.put(self.log_name, (SDSCommands.STOP_CONTROL, None))

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
        message, data = queue_data
        if message == "control_predictions":
            (
                predicted_amplitudes,
                predicted_delays,
                predicted_decays,
                predicted_drive_time_history,
                predicted_response_time_history,
                predicted_response_srs,
            ) = data
            self.prediction_table.update_prediction_information(
                predicted_response_time_history,
                predicted_response_srs,
                predicted_amplitudes,
                predicted_delays,
                predicted_decays,
                predicted_drive_time_history,
            )

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
            # TODO: Set set all spinboxes in the sine tone table to these values.

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

        # SDS
        self.definition_widget.sds_iterations_selector.setValue(int(worksheet.cell(7, 2).value))
        self.definition_widget.sds_convergence_selector.setValue(
            100 * float(worksheet.cell(8, 2).value)
        )
        self.definition_widget.sds_scale_factor_selector.setValue(
            100 * float(worksheet.cell(9, 2).value)
        )
        self.definition_widget.error_tolerance_selector.setValue(
            100 * float(worksheet.cell(10, 2).value)
        )

        # Control Script and Parameters
        if worksheet.cell(12, 2).value is not None and worksheet.cell(12, 2).value != "":
            self.select_python_module(None, worksheet.cell(12, 2).value)
            self.definition_widget.python_class_input.setCurrentIndex(
                self.definition_widget.python_class_input.findText(worksheet.cell(13, 2).value)
            )

        # Control channels
        column_index = 2
        while True:
            value = worksheet.cell(15, column_index).value
            if value is None or (isinstance(value, str) and value.strip() == ""):
                break
            item = self.definition_widget.control_channels_selector.item(int(value) - 1)
            item.setCheckState(Qt.Checked)
            column_index += 1

        # System identification
        self.system_id_widget.samplesPerFrameSpinBox.setValue(int(worksheet.cell(16, 2).value))
        self.system_id_widget.averagingTypeComboBox.setCurrentIndex(
            self.system_id_widget.averagingTypeComboBox.findText(worksheet.cell(17, 2).value)
        )
        self.system_id_widget.noiseAveragesSpinBox.setValue(int(worksheet.cell(18, 2).value))
        self.system_id_widget.systemIDAveragesSpinBox.setValue(int(worksheet.cell(19, 2).value))
        self.system_id_widget.averagingCoefficientDoubleSpinBox.setValue(
            float(worksheet.cell(20, 2).value)
        )
        self.system_id_widget.estimatorComboBox.setCurrentIndex(
            self.system_id_widget.estimatorComboBox.findText(worksheet.cell(21, 2).value)
        )
        self.system_id_widget.levelDoubleSpinBox.setValue(float(worksheet.cell(22, 2).value))
        self.system_id_widget.levelRampTimeDoubleSpinBox.setValue(
            float(worksheet.cell(23, 2).value)
        )
        self.system_id_widget.signalTypeComboBox.setCurrentIndex(
            self.system_id_widget.signalTypeComboBox.findText(worksheet.cell(24, 2).value)
        )
        self.system_id_widget.windowComboBox.setCurrentIndex(
            self.system_id_widget.windowComboBox.findText(worksheet.cell(25, 2).value)
        )
        self.system_id_widget.overlapDoubleSpinBox.setValue(float(worksheet.cell(26, 2).value))
        self.system_id_widget.onFractionDoubleSpinBox.setValue(float(worksheet.cell(27, 2).value))
        self.system_id_widget.pretriggerDoubleSpinBox.setValue(float(worksheet.cell(28, 2).value))
        self.system_id_widget.rampFractionDoubleSpinBox.setValue(float(worksheet.cell(29, 2).value))

        # Transformation matrices
        response_channels = self.definition_widget.control_channels_display.value()
        output_channels = self.definition_widget.output_channels_display.value()
        output_transform_row = 31
        if (
            isinstance(worksheet.cell(30, 2).value, str)
            and worksheet.cell(30, 2).value.lower() == "none"
        ):
            self.response_transformation_matrix = None
        else:
            while True:
                if worksheet.cell(output_transform_row, 1).value == "Output Transformation Matrix:":
                    break
                output_transform_row += 1
            response_size = output_transform_row - 30
            response_transformation = []
            for i in range(response_size):
                response_transformation.append([])
                for j in range(response_channels):
                    response_transformation[-1].append(float(worksheet.cell(30 + i, 2 + j).value))
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
        self.load_specification(None, worksheet.cell(11, 2).value)

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


class SDSEnvironment(AbstractSysIdEnvironment):
    """Class defining calculations for the SDS environment"""

    def __init__(
        self,
        environment_name: str,
        queue_container: SDSQueues,
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
            SDSCommands.PERFORM_CONTROL_PREDICTION,
            self.perform_control_prediction,
        )
        self.map_command(SDSCommands.START_CONTROL, self.start_control)
        self.map_command(SDSCommands.STOP_CONTROL, self.stop_environment)
        self.map_command(
            GlobalCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS,
            self.update_interactive_control_parameters,
        )
        self.map_command(GlobalCommands.SEND_INTERACTIVE_COMMAND, self.send_interactive_command)

        # Persistent Data
        self.data_acquisition_parameters = None
        self.environment_parameters = None
        self.queue_container = queue_container
        # System ID information
        self.sysid_frames = None
        self.sysid_frequencies = None
        self.sysid_frf = None
        self.sysid_coherence = None
        self.sysid_response_cpsd = None
        self.sysid_reference_cpsd = None
        self.sysid_condition = None
        self.sysid_response_noise = None
        self.sysid_reference_noise = None
        # Control information
        self.control_module = None
        self.control_law = None
        self.control_last_interactive_parameters = None
        self.control_has_sent_interactive_control_transfer_function_results = False
        self.last_response_srs = None
        self.last_response_time_history = None
        self.last_drive_amplitudes = None
        self.last_drive_decays = None
        self.last_drive_delays = None
        # Prediction information
        self.predicted_response_srs = None
        self.predicted_response_time_history = None
        self.predicted_amplitudes = None
        self.predicted_decays = None
        self.predicted_delays = None
        self.predicted_drive_time_history = None

    def initialize_environment_test_parameters(self, environment_parameters: SDSMetadata):
        # Check if things need to be reset
        if self.environment_parameters is None or not np.array_equal(
            self.environment_parameters.control_channel_indices,
            environment_parameters.control_channel_indices,
        ):
            # System ID information
            self.sysid_frames = None
            self.sysid_frequencies = None
            self.sysid_frf = None
            self.sysid_coherence = None
            self.sysid_response_cpsd = None
            self.sysid_reference_cpsd = None
            self.sysid_condition = None
            self.sysid_response_noise = None
            self.sysid_reference_noise = None
            self.control_last_interactive_parameters = None
            self.control_has_sent_interactive_control_transfer_function_results = False
            self.last_response_srs = None
            self.last_drive_amplitudes = None
            self.last_drive_decays = None
            self.last_drive_delays = None
        super().initialize_environment_test_parameters(environment_parameters)
        self.environment_parameters: SDSMetadata
        # Load in the control law
        if (
            self.environment_parameters.control_script_data.control_script
            == "rattlesnake.components.sds_sys_id_control_law"
        ):
            self.control_module = importlib.import_module(
                "rattlesnake.components.sds_sys_id_control_law"
            )
        else:
            self.control_module = load_python_module(
                self.environment_parameters.control_script_data.control_script
            )
        # Depending on the type, initialize the control law
        if self.environment_parameters.control_script_data.control_type == ControlLawType.FUNCTION:
            self.control_law = getattr(
                self.control_module, self.environment_parameters.control_script_data.control_object
            )
        elif self.environment_parameters.control_script_data.control_type == ControlLawType.CLASS:
            self.control_law = getattr(
                self.control_module, self.environment_parameters.control_script_data.control_object
            )(
                environment_parameters=self.environment_parameters,
                transfer_function_frequencies=self.sysid_frequencies,
                transfer_function=self.sysid_frf,
                noise_response_cpsd=self.sysid_response_noise,
                noise_reference_cpsd=self.sysid_reference_noise,
                sysid_response_cpsd=self.sysid_response_cpsd,
                sysid_reference_cpsd=self.sysid_reference_cpsd,
                multiple_coherence=self.sysid_coherence,
                frames=self.sysid_frames,
                last_response_srs=self.last_response_srs,
                last_drive_amplitudes=self.last_drive_amplitudes,
                last_drive_decays=self.last_drive_decays,
                last_drive_delays=self.last_drive_delays,
                **self.environment_parameters.control_script_data.control_parameters,
            )
        elif (
            self.environment_parameters.control_script_data.control_type
            == ControlLawType.INTERACTIVE_CLASS
        ):
            self.control_law = getattr(
                self.control_module, self.environment_parameters.control_script_data.control_object
            )(
                environment_parameters=self.environment_parameters,
                transfer_function_frequencies=self.sysid_frequencies,
                transfer_function=self.sysid_frf,
                noise_response_cpsd=self.sysid_response_noise,
                noise_reference_cpsd=self.sysid_reference_noise,
                sysid_response_cpsd=self.sysid_response_cpsd,
                sysid_reference_cpsd=self.sysid_reference_cpsd,
                multiple_coherence=self.sysid_coherence,
                frames=self.sysid_frames,
                last_response_srs=self.last_response_srs,
                last_drive_amplitudes=self.last_drive_amplitudes,
                last_drive_decays=self.last_drive_decays,
                last_drive_delays=self.last_drive_delays,
                **self.environment_parameters.control_script_data.control_parameters,
            )
            self.control_last_interactive_parameters = None
            self.control_has_sent_interactive_control_transfer_function_results = False
        else:
            raise ValueError(
                f"Invalid type {self.environment_parameters.control_script_data.control_type}. "
                "How did you get here?!"
            )

    def update_interactive_control_parameters(self, interactive_control_parameters):
        """Updates the interactive control law based on received parameters"""
        if (
            self.environment_parameters.control_script_data.control_type
            == ControlLawType.INTERACTIVE_CLASS
        ):
            self.control_law.update_parameters(interactive_control_parameters)
            self.control_last_interactive_parameters = interactive_control_parameters
        else:
            raise ValueError(
                "Received an UPDATE_INTERACTIVE_CONTROL_PARAMETERS signal without an "
                "interactive control law.  How did this happen?"
            )

    def send_interactive_command(self, command):
        """General method that can be used by an interactive UI object to pass commands
        and data to its corresponding computation object"""
        if (
            self.environment_parameters.control_script_data.control_type
            == ControlLawType.INTERACTIVE_CLASS
        ):
            self.control_law.send_command(command)
        else:
            raise ValueError(
                "Received an SEND_INTERACTIVE_COMMAND signal without an interactive "
                "control law.  How did this happen?"
            )

    def system_id_complete(self, data):
        """Sends the message that system identification is complete and control calculations
        should be performed"""
        super().system_id_complete(data)
        (
            self.sysid_frames,
            _,  # avg,
            self.sysid_frequencies,
            self.sysid_frf,
            self.sysid_coherence,
            self.sysid_response_cpsd,
            self.sysid_reference_cpsd,
            self.sysid_condition,
            self.sysid_response_noise,
            self.sysid_reference_noise,
        ) = data
        self.perform_control_prediction(True)

    def perform_control_prediction(self, sysid_update):
        """Performs the control prediction based on system identification information"""
        if self.sysid_frf is None:
            self.gui_update_queue.put(
                (
                    "error",
                    (
                        "Perform System Identification",
                        "Perform System ID before performing test predictions",
                    ),
                )
            )
            return
        # Perform the control prediction
        # Depending on the type, initialize the control law
        if self.environment_parameters.control_script_data.control_type == ControlLawType.FUNCTION:
            output_amplitudes, output_decays, output_delays = self.control_law(
                environment_parameters=self.environment_parameters,
                transfer_function_frequencies=self.sysid_frequencies,
                transfer_function=self.sysid_frf,
                noise_response_cpsd=self.sysid_response_noise,
                noise_reference_cpsd=self.sysid_reference_noise,
                sysid_response_cpsd=self.sysid_response_cpsd,
                sysid_reference_cpsd=self.sysid_reference_cpsd,
                multiple_coherence=self.sysid_coherence,
                frames=self.sysid_frames,
                last_response_srs=self.last_response_srs,
                last_drive_amplitudes=self.last_drive_amplitudes,
                last_drive_decays=self.last_drive_decays,
                last_drive_delays=self.last_drive_delays,
                **self.environment_parameters.control_script_data.control_parameters,
            )
        elif (
            self.environment_parameters.control_script_data.control_type == ControlLawType.CLASS
            or self.environment_parameters.control_script_data.control_type
            == ControlLawType.INTERACTIVE_CLASS
        ):
            if sysid_update:
                self.control_law.system_id_update(
                    transfer_function_frequencies=self.sysid_frequencies,
                    transfer_function=self.sysid_frf,
                    noise_response_cpsd=self.sysid_response_noise,
                    noise_reference_cpsd=self.sysid_reference_noise,
                    sysid_response_cpsd=self.sysid_response_cpsd,
                    sysid_reference_cpsd=self.sysid_reference_cpsd,
                    multiple_coherence=self.sysid_coherence,
                    frames=self.sysid_frames,
                )
                if (
                    self.environment_parameters.control_script_data.control_type
                    == ControlLawType.INTERACTIVE_CLASS
                ):
                    self.gui_update_queue.put(
                        (
                            self.environment_name,
                            (
                                "interactive_control_sysid_update",
                                (
                                    self.sysid_frequencies,
                                    self.sysid_frf,
                                    self.sysid_response_noise,
                                    self.sysid_reference_noise,
                                    self.sysid_response_cpsd,
                                    self.sysid_reference_cpsd,
                                    self.sysid_coherence,
                                    self.sysid_frames,
                                ),
                            ),
                        )
                    )
                    self.control_has_sent_interactive_control_transfer_function_results = True
            if (
                self.environment_parameters.control_script_data.control_type == ControlLawType.CLASS
                or self.control_last_interactive_parameters is not None
            ):
                output_amplitudes, output_decays, output_delays = self.control_law.control(
                    last_response_srs=self.last_response_srs,
                    last_drive_amplitudes=self.last_drive_amplitudes,
                    last_drive_decays=self.last_drive_decays,
                    last_drive_delays=self.last_drive_delays,
                )
            else:
                self.log("Have not yet received control parameters from interactive control law!")
                return
        else:
            raise ValueError(
                f"Invalid type {self.environment_parameters.control_script_data.control_type}. "
                "How did you get here?!"
            )

        (
            self.predicted_drive_time_history,
            self.predicted_response_time_history,
            self.predicted_response_srs,
        ) = self.simulate_response((output_amplitudes, output_decays, output_delays))

        self.show_test_prediction()

    def simulate_response(self, data):
        # Reconstruct drive signals
        amplitudes, decays, delays = data
        frequencies = self.environment_parameters.get_sds_frequencies()
        drive_signals = sum_decayed_sines_reconstruction(
            frequencies,
            amplitudes[:, np.newaxis, :].T,
            decays[:, np.newaxis, :].T,
            delays[:, np.newaxis, :].T,
            self.environment_parameters.sample_rate,
            self.environment_parameters.block_size,
        )
        # Simulate responses to those drive signals
        impulse_responses = np.moveaxis(np.fft.irfft(self.sysid_frf, axis=0), 0, -1)

        predicted_response_time_history = np.zeros(
            (impulse_responses.shape[0], drive_signals.shape[-1])
        )

        for i, impulse_response_row in enumerate(impulse_responses):
            for impulse, drive in zip(impulse_response_row, drive_signals):
                # print('Convolving {:},{:}'.format(i,j))
                predicted_response_time_history[i, :] += sig.convolve(drive, impulse, "full")[
                    : drive_signals.shape[-1]
                ]

        srss = []
        for signal in predicted_response_time_history:
            srss.append(
                srs_function(
                    signal,
                    1 / self.environment_parameters.sample_rate,
                    frequencies,
                    self.environment_parameters.srs_data.srs_damping,
                    self.environment_parameters.srs_data.srs_type.value
                    * self.environment_parameters.srs_data.srs_displacement.value,
                )
            )
        return drive_signals, predicted_response_time_history, srss

    def show_test_prediction(self):
        """Sends the test predictions to the UI"""
        self.gui_update_queue.put(
            (
                self.environment_name,
                (
                    "control_predictions",
                    (
                        self.predicted_amplitudes,
                        self.predicted_delays,
                        self.predicted_decays,
                        self.predicted_drive_time_history,
                        self.predicted_response_time_history,
                        self.predicted_response_srs,
                    ),
                ),
            )
        )

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


def sds_process(
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
    SDS environment process function called by multiprocessing

    This function defines the SDS Environment process that
    gets run by the multiprocessing module when it creates a new process.  It
    creates a SDSEnvironment object and runs it.

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
        queue_container = SDSQueues(
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

        process_class = SDSEnvironment(
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
