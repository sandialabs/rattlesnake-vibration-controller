from qtpy import QtCore, QtWidgets, uic
from qtpy.QtCore import Qt, QTime
from rattlesnake.user_interface.abstract_sys_id_user_interface import SysIdEnvironmentUI
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.sds_sys_id_metadata import (
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
    convert_decay_strategy,
)
from rattlesnake.engine import RattlesnakeController
from rattlesnake.environment.sds_sys_id_utilities import (
    octspace,
    SDSQueues,
    SDSCommands,
    SDSUICommands,
    SDSInstructions,
    sum_decayed_sines_reconstruction,
    srs as srs_function,
)
from rattlesnake.user_interface.ui_utilities import (
    PlotTimeWindow,
    TransformationMatrixWindow,
    colororder,
    multiline_plotter,
    AdaptiveNoWheelSpinBox,
    ScientificDoubleSpinBox,
)
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.utilities import (
    GlobalCommands,
    VerboseMessageQueue,
    load_time_history,
    db2scale,
    load_python_module,
    DIRECTORY,
)
from multiprocessing.queues import Queue
from rattlesnake.user_interface.sds_sys_id_prediction_table import SDSPredictionTable
from rattlesnake.user_interface.sds_sys_id_synthesize_dialog import SDSSynthesizeDialog
from rattlesnake.user_interface.sds_sys_id_run_table import SDSRunTableDialog
from rattlesnake.user_interface.sds_sys_id_shock_history import SDSShockHistoryDialog
import inspect
import numpy as np
import importlib
from enum import Enum
import os
import netCDF4 as nc4

CONTROL_TYPE = EnvironmentType.SDS
MAXIMUM_NAME_LENGTH = 50


class SDSUI(SysIdEnvironmentUI):
    """Class defining the user interface for the SDS environment"""

    def __init__(
        self,
        environment_name: str,
        rattlesnake: RattlesnakeController,
    ):
        # print(f"Building UI for environment {environment_name}")
        super().__init__(
            CONTROL_TYPE,
            environment_name,
            rattlesnake,
        )
        # Add the page to the control definition tabwidget
        self.definition_widget = QtWidgets.QWidget()
        environment_definition_ui_path = os.path.join(
            DIRECTORY, "user_interface", "ui_files", "srs_sds_definition.ui"
        )
        uic.loadUi(environment_definition_ui_path, self.definition_widget)
        # Add the page to the control prediction tabwidget
        self.prediction_widget = QtWidgets.QWidget()
        environment_prediction_ui_path = os.path.join(
            DIRECTORY, "user_interface", "ui_files", "srs_sds_prediction.ui"
        )
        uic.loadUi(environment_prediction_ui_path, self.prediction_widget)
        # Add the page to the run tabwidget
        self.run_widget = QtWidgets.QWidget()
        environment_run_ui_path = os.path.join(
            DIRECTORY, "user_interface", "ui_files", "srs_sds_run.ui"
        )
        uic.loadUi(environment_run_ui_path, self.run_widget)

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
        self.shock_history = None

        self.control_selector_widgets = [self.definition_widget.specification_plot_selector]

        self.output_selector_widgets = []

        self.prediction_table = SDSPredictionTable(
            self.prediction_widget.prediction_table_placeholder,
            self.rattlesnake,
            self.environment_name,
            prediction_mode=True,
        )
        self.prediction_table.lock_table(all_data=True)

        self.run_table_dialog = SDSRunTableDialog(
            self.rattlesnake,
            self.environment_name,
            prediction_mode=False,
            parent=self.run_widget,
            other_voltage_lists=[self.run_widget.test_output_voltage_list],
            other_error_lists=[self.run_widget.test_response_error_list],
        )
        self.run_table = self.run_table_dialog.run_table
        self.run_table.lock_table(frequencies=True)

        self.shock_history_dialog = SDSShockHistoryDialog(self.run_widget)

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
        self.definition_widget.load_breakpoints_button.clicked.connect(self.load_specification)
        # Prediction
        self.prediction_widget.recompute_prediction_button.clicked.connect(
            self.recompute_prediction
        )
        # Run Test
        self.run_widget.sds_table_button.clicked.connect(self.show_run_table)
        self.run_widget.start_test_button.clicked.connect(self.start_environment)
        self.run_widget.shock_history_button.clicked.connect(self.show_shock_history)
        self.run_widget.stop_test_button.clicked.connect(self.stop_environment)

    # region UI Data Acquisition

    def initialize_hardware(self, hardware_metadata: HardwareMetadata):
        super().initialize_hardware(hardware_metadata)
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
            for channel in hardware_metadata.channel_list
        ]
        self.physical_unit_names = [
            f"{'-' if channel.unit is None else channel.unit}"
            for channel in hardware_metadata.channel_list
        ]
        self.physical_output_indices = [
            i for i, channel in enumerate(hardware_metadata.channel_list) if channel.feedback_device
        ]

        # Set default values for various widgets
        # Sampling parameters
        self.definition_widget.sample_rate_display.setValue(hardware_metadata.sample_rate)
        self.system_id_widget.samplesPerFrameSpinBox.setValue(hardware_metadata.sample_rate)
        # By default set the block time length to 1 second
        self.definition_widget.block_size_selector.setValue(hardware_metadata.sample_rate)
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

    # region UI Environment

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
        if self.environment_metadata.response_transformation_matrix is None:
            return [
                self.physical_channel_names[i]
                for i in self.environment_metadata.control_channel_indices
            ]
        else:
            return [
                f"Transformed Response {i + 1}"
                for i in range(self.environment_metadata.response_transformation_matrix.shape[0])
            ]

    @property
    def initialized_output_names(self):
        """Names of the drive channels that have been initialized"""
        if self.environment_metadata.reference_transformation_matrix is None:
            return self.physical_output_names
        else:
            return [
                f"Transformed Drive {i + 1}"
                for i in range(self.environment_metadata.reference_transformation_matrix.shape[0])
            ]

    # region UI Specification
    def load_specification(self, clicked, filename=None):  # pylint: disable=unused-argument
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.definition_widget,
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
            spinbox.setRange(0, self.hardware_metadata.sample_rate / 2)
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
            spinbox.setRange(0, self.hardware_metadata.sample_rate / 2)
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
            spinbox.setRange(0, self.hardware_metadata.sample_rate / 2)
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
        spinbox.setRange(0, self.hardware_metadata.sample_rate / 2)
        spinbox.setSingleStep(1)
        spinbox.setValue(0)
        spinbox.setKeyboardTracking(False)
        spinbox.setDecimals(4)
        spinbox.valueChanged.connect(self.update_specification)
        self.definition_widget.breakpoint_table.setCellWidget(selected_row, 0, spinbox)
        # Frequency breakpoint, lower limit
        spinbox = AdaptiveNoWheelSpinBox()
        spinbox.setRange(0, self.hardware_metadata.sample_rate / 2)
        spinbox.setSingleStep(1)
        spinbox.setValue(0)
        spinbox.setKeyboardTracking(False)
        spinbox.setDecimals(4)
        spinbox.setReadOnly(True)
        spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
        self.definition_widget.lower_limit_table.setCellWidget(selected_row, 0, spinbox)
        # Frequency breakpoint, upper limit
        spinbox = AdaptiveNoWheelSpinBox()
        spinbox.setRange(0, self.hardware_metadata.sample_rate / 2)
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

    def set_specification_data(self, specification_data: SpecParameters):
        self.clear_and_update_specification_table(
            specification_data.frequencies,
            specification_data.srs_spec,
            specification_data.srs_lower_limit,
            specification_data.srs_upper_limit,
        )
        self.definition_widget.num_hits_spinbox.setValue(specification_data.num_hits)
        self.update_specification()

    # region UI Control Laws
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
            "environment_metadata",
            "sysid_data",
            "last_response_srs",
            "last_response_signals",
            "last_drive_amplitudes",
            "last_drive_decays",
            "last_drive_delays",
            "last_drive_signals",
        }
        valid_control_laws = []
        members = inspect.getmembers(module)
        for objname, obj in members:
            # print(f"Analyzing member {objname}")
            valid_control_law = True
            # Check if it is a function
            if inspect.isfunction(obj):
                signature = inspect.signature(obj)
                parameters = signature.parameters
                # Check if is a valid object
                if not all(arg in parameters for arg in required_control_law_arguments):
                    # print(f"Member {objname} does not have all required arguments")
                    continue
                # Get extra arguments
                extra_arguments = []
                # print(f"{signature=}")
                # print(f"{parameters=}")
                for name, parameter in parameters.items():
                    # Extra arguments are not required arguments
                    if name in required_control_law_arguments:
                        # print(f"  Argument {name} is a required argument")
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
                            # print(
                            #     f"  Argument {name} has no valid annotation or default "
                            #     "value, invalid control law"
                            # )
                            valid_control_law = False
                            break
                        if (
                            not SDSUI.valid_annotation(annotation)
                            and default != inspect.Parameter.empty
                        ):
                            # print(
                            #     f"  Argument {name} has an invalid annotation but contains a "
                            #     "default value which will be used, and will therefore not be "
                            #     "treated as an extra parameter"
                            # )
                            continue
                        extra_arguments.append([name, annotation, default])
                    elif (
                        parameter.kind == inspect.Parameter.POSITIONAL_ONLY
                        and default == inspect.Parameter.empty
                    ):
                        # print(
                        #     f"  Argument {name} is positional only without a "
                        #     "default and therefore cannot be specified."
                        # )
                        valid_control_law = False
                        break
                    else:
                        # print(
                        #     f"  Argument {name} is positional only but has a default argument"
                        #     "which will be used, and will therefore not be treated as an "
                        #     "extra parameter"
                        # )
                        continue
                if not valid_control_law:
                    continue
                valid_control_laws.append([objname, extra_arguments])
            # else:
            # print(f"Member {objname} is not a valid type of object to be a control law")
        # print(valid_control_laws)
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
        # print(f"Select Python Spec: {filename=}, {default=}, {clicked=}")
        if default or filename == "rattlesnake.environment.sds_sys_id_control_law":
            self.python_control_module = importlib.import_module(
                "rattlesnake.environment.sds_sys_id_control_law"
            )
            filename = "rattlesnake.environment.sds_sys_id_control_law"
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
        # print(f"Got Arguments {kwargs=}")
        return kwargs

    def set_control_extra_parameters(self, kwargs: dict):
        for key, value in kwargs.items():
            try:
                widget = self.python_function_extra_argument_widgets[key]
            except KeyError:
                raise ValueError(f"Parameter {key} not found in extra parameter widget list.")
            if isinstance(widget, QtWidgets.QSpinBox):
                widget.setValue(value)
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.setValue(value)
            elif isinstance(widget, QtWidgets.QTextEdit):
                widget.setPlainText(value)
            elif isinstance(widget, QtWidgets.QComboBox):
                index = widget.findData(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
                else:
                    raise ValueError(f"Data {value} not found in combobox for parameter {key}")
            else:
                raise ValueError(f"Unsupported widget type: {type(widget)}")

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

    def set_control_data(self, control_data: ControlParameters):
        self.select_python_module(filename=control_data.control_script)
        index = self.definition_widget.control_function_input.findText(control_data.control_object)
        if index != -1:
            self.definition_widget.control_function_input.setCurrentIndex(index)
        else:
            raise ValueError("Control function/class was not found.")
        self.definition_widget.control_function_generator_selector.setCurrentIndex(
            control_data.control_type.value
        )
        self.set_control_extra_parameters(control_data.control_parameters)

    # region UI Collect Metadata
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

    def set_tone_data(self, tone_data: ToneParameters):
        if tone_data.tone_strategy == ToneStrategy.FROM_SPEC:
            self.definition_widget.from_spec_button.setChecked(True)
        elif tone_data.tone_strategy == ToneStrategy.OCTAVE:
            self.definition_widget.octave_button.setChecked(True)
            min_freq, max_freq, tones_per_octave = tone_data.tone_data
            self.definition_widget.min_frequency_selector.setValue(min_freq)
            self.definition_widget.max_frequency_selector.setValue(max_freq)
            self.definition_widget.tones_per_octave_selector.setValue(tones_per_octave)
        elif tone_data.tone_strategy == ToneStrategy.MANUAL:
            self.definition_widget.manual_button.setChecked(True)
            self.set_sine_tone_values(frequencies=tone_data.tone_data)

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

    def set_compensation_pulse_data(self, compensation_pulse_data: CompPulseParameters):
        self.definition_widget.use_compensation_pulse_checkbox.setChecked(
            compensation_pulse_data.use_compensation_pulse
        )
        if compensation_pulse_data.use_compensation_pulse:
            self.definition_widget.autoselect_comp_frequency_checkbox.setChecked(
                compensation_pulse_data.compensation_frequency is None
            )
            if compensation_pulse_data.compensation_frequency is not None:
                self.definition_widget.compensation_frequency_selector.setValue(
                    compensation_pulse_data.compensation_frequency
                )
            self.definition_widget.compensation_decay_selector.setValue(
                compensation_pulse_data.compensation_decay * 100
            )

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

    def set_decay_data(self, decay_parameters: DecayParameters):
        if decay_parameters.decay_strategy == DecayStrategy.DAMPING:
            self.definition_widget.damping_zeta_button.setChecked(True)
        elif decay_parameters.decay_strategy == DecayStrategy.TIME_CONSTANT:
            self.definition_widget.time_constant_tau_button.setChecked(True)
        elif decay_parameters.decay_strategy == DecayStrategy.NUM_TIME_CONSTANTS:
            self.definition_widget.num_time_constants_button.setChecked(True)
        self.definition_widget.common_decay_checkbox.setChecked(decay_parameters.common_decay)
        if decay_parameters.common_decay:
            self.definition_widget.decay_value_selector.setValue(decay_parameters.decay_data[0])
        else:
            self.set_sine_tone_values(decays=decay_parameters.decay_data)

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

    def set_srs_data(self, srs_data: SRSParameters):
        srs_type_index = srs_data.srs_type.value - 1
        self.definition_widget.srs_type_setter.setCurrentIndex(srs_type_index)
        if srs_data.srs_displacement == SRSDisplacementType.ABSOLUTE:
            self.definition_widget.srs_displacement_setter.setCurrentIndex(0)
        elif srs_data.srs_displacement == SRSDisplacementType.RELATIVE:
            self.definition_widget.srs_displacement_setter.setCurrentIndex(1)
        else:
            raise ValueError(f"Invalid value for srs_displacement {srs_data.srs_displacement}")
        self.definition_widget.srs_damping_setter.setValue(srs_data.srs_damping * 100)

    def collect_sds_data(self):
        sds_data = SDSParameters(
            self.definition_widget.sds_iterations_selector.value(),
            self.definition_widget.sds_convergence_selector.value() / 100,
            self.definition_widget.sds_scale_factor_selector.value() / 100,
            self.definition_widget.error_tolerance_selector.value() / 100,
        )
        return sds_data

    def set_sds_data(self, sds_data: SDSParameters):
        self.definition_widget.sds_iterations_selector.setValue(sds_data.iterations)
        self.definition_widget.sds_convergence_selector.setValue(sds_data.convergence * 100)
        self.definition_widget.sds_scale_factor_selector.setValue(sds_data.scale_factor * 100)
        self.definition_widget.error_tolerance_selector.setValue(sds_data.error_tolerance * 100)

    def get_environment_metadata(self, global_channel_list):
        """Collects the metadata defining the environment from the UI widgets"""
        if self.hardware_metadata and global_channel_list:
            channel_list_bools = self.get_channel_list_bools(global_channel_list)
        else:
            channel_list_bools = []
        control_data = self.collect_control_data()
        tone_data = self.collect_tone_data()
        compensation_pulse_data = self.collect_compensation_pulse_data()
        decay_parameters = self.collect_decay_data()
        srs_data = self.collect_srs_data()
        sds_data = self.collect_sds_data()
        spec_data = self.collect_specification()
        return SDSMetadata(
            environment_name=self.environment_name,
            channel_list_bools=channel_list_bools,
            sample_rate=self.hardware_metadata.sample_rate,
            num_channels=len(self.hardware_metadata.channel_list),
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

    def set_environment_metadata(self, metadata: SDSMetadata):
        """Sets the UI State given a metadata object"""
        self.definition_widget.block_size_selector.setValue(metadata.block_size)
        # First we need to set control channels correctly so the specification has the right
        # number of columns
        for index in range(self.definition_widget.control_channels_selector.count()):
            self.definition_widget.control_channels_selector.item(index).setCheckState(Qt.Unchecked)
        for index in metadata.control_channel_indices:
            self.definition_widget.control_channels_selector.item(index).setCheckState(Qt.Checked)
        # Now we need to set up any transformation matrices again so the specification has the
        # correct number of channels
        self.response_transformation_matrix = metadata.response_transformation_matrix
        self.output_transformation_matrix = metadata.reference_transformation_matrix
        self.define_transformation_matrices(None, dialog=False)
        # Now we need to set up the specification so the default tone and decay values are
        # inherited correctly from the specification
        self.set_specification_data(metadata.specification_data)
        # Now we need to set up the sine tone frequencies
        self.set_tone_data(metadata.tone_data)
        # Now we need to set the compensation pulse data
        self.set_compensation_pulse_data(metadata.compensation_pulse_data)
        # Now we need to set up the decay parameters
        self.set_decay_data(metadata.decay_data)
        # Now the rest of the parameters shouldn't matter which order
        self.set_srs_data(metadata.srs_data)
        self.set_sds_data(metadata.sds_data)
        self.set_control_data(metadata.control_script_data)

    def initialize_environment(self, environment_metadata: SDSMetadata):
        super().initialize_environment(environment_metadata)
        self.prediction_table.update_names(
            self.initialized_output_names, self.initialized_control_names
        )
        self.run_table.update_names(self.initialized_output_names, self.initialized_control_names)
        self.prediction_table.update_parameters(environment_metadata)
        self.run_table.update_parameters(environment_metadata)
        return self.environment_metadata

    def define_transformation_matrices(
        self, clicked, dialog=True
    ):  # pylint: disable=unused-argument
        """Defines the transformation matrices using the dialog box"""
        if dialog:
            response_transformation, output_transformation, result = (
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
                    freq_spinbox.setRange(0, self.hardware_metadata.sample_rate / 2)
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
                    freq_spinbox.setRange(0, self.hardware_metadata.sample_rate / 2)
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
            new_decay_values = convert_decay_strategy(
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
        freq_spinbox.setRange(0, self.hardware_metadata.sample_rate / 2)
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

    # region UI Prediction

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

    def recompute_prediction(self):
        """Recomputes the control predictions"""
        self.rattlesnake.send_environment_command(
            self.environment_name, SDSCommands.PERFORM_CONTROL_PREDICTION, False
        )

    # region UI Control

    def ask_yes_no(self, parent=None, title="Confirm", text="Proceed?") -> bool:
        btn = QtWidgets.QMessageBox.question(
            parent,
            title,
            text,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,  # default selection (optional)
        )
        return btn == QtWidgets.QMessageBox.Yes

    def show_run_table(self):
        self.run_table_dialog.show()

    def show_shock_history(self):
        self.shock_history_dialog.show()
        self.shock_history_dialog.raise_()
        self.shock_history_dialog.activateWindow()

    # def start_control(self):
    #     """Starts the chain of events to start the environment"""
    #     if (
    #         self.run_widget.current_hits_at_level_display.value()
    #         >= self.run_widget.target_hits_at_level_selector.value()
    #     ):
    #         confirm = self.ask_yes_no(
    #             self.run_widget,
    #             "Target Hits Reached, Are you Sure?",
    #             "The number of target hits at this level has been reached.\n\n"
    #             "Are you sure you wish to continue?",
    #         )
    #         if not confirm:
    #             return
    #     metadata = self.collect_run_metadata()
    #     self.enable_control(False)
    #     self.rattlesnake.send_environment_command(
    #         self.environment_name, GlobalCommands.START_ENVIRONMENT, self.environment_name
    #     )
    #     self.rattlesnake.send_environment_command(
    #         self.environment_name, SDSCommands.START_CONTROL, metadata
    #     )
    #     if self.run_widget.current_test_level_selector.value >= 0:
    #         self.rattlesnake.environment_at_target_level(self.environment_name)

    # def stop_control(self):
    #     """Starts the sequence of events to stop the controller prematurely"""
    #     self.rattlesnake.send_environment_command(
    #         self.environment_name, SDSCommands.STOP_CONTROL, None
    #     )

    def enable_control(self, enabled):
        """Enables or disables the buttons to start control if it's already running"""
        for widget in [
            self.run_widget.current_test_level_selector,
            self.run_widget.target_hits_at_level_selector,
            self.run_widget.manual_hits_checkbox,
            self.run_widget.auto_hits_checkbox,
            self.run_widget.auto_hit_interval_selector,
            self.run_widget.start_test_button,
        ]:
            widget.setEnabled(enabled)
        for widget in [self.run_widget.stop_test_button]:
            widget.setEnabled(not enabled)

    def change_test_level_from_profile(self, test_level):
        """Updates the test level based on a profile event"""
        self.run_widget.current_test_level_selector.setValue(int(test_level))

    def display_environment_ended(self):
        """Enables or disables the buttons to start control if it's already running"""
        self.enable_control(True)

    def display_environment_started(self):
        self.enable_control(False)

    def start_environment(self):
        """Sets itself up to start controlling and sends a signal to the environment to start"""
        if (
            self.run_widget.current_hits_at_level_display.value()
            >= self.run_widget.target_hits_at_level_selector.value()
        ):
            confirm = self.ask_yes_no(
                self.run_widget,
                "Target Hits Reached, Are you Sure?",
                "The number of target hits at this level has been reached.\n\n"
                "Are you sure you wish to continue?",
            )
            if not confirm:
                return
        for widget in [
            self.run_widget.start_test_button,
            self.run_widget.current_test_level_selector,
            self.run_widget.target_hits_at_level_selector,
            self.run_widget.manual_hits_checkbox,
            self.run_widget.auto_hits_checkbox,
            self.run_widget.auto_hit_interval_selector,
        ]:
            widget.setEnabled(False)

        super().start_environment()
        # TODO: This needs to come from the environment, not the UI.
        if self.run_widget.current_test_level_selector.value() >= 0:
            self.rattlesnake.environment_at_target_level(self.environment_name)

    def start_environment_ready(self):
        return super().start_environment_ready()

    def start_environment_error(self, error):
        return super().start_environment_error(error)

    def stop_environment(self):
        """Sends a signal to shut down the control"""
        for widget in [self.run_widget.stop_test_button]:
            widget.setEnabled(False)

        super().stop_environment()

    def stop_environment_error(self, error):
        return super().stop_environment_error(error)

    def stop_environment_ready(self):
        return super().stop_environment_ready()

    def get_environment_instructions(self):
        automatic_hits = self.run_widget.auto_hits_checkbox.isChecked()
        automatic_interval = None
        if automatic_hits:
            automatic_interval = QTime(0, 0).secsTo(
                self.run_widget.auto_hit_interval_selector.time()
            )

        instructions = SDSInstructions(
            environment_name=self.environment_name,
            control_test_level=self.run_widget.current_test_level_selector.value(),
            target_hits_at_level=self.run_widget.target_hits_at_level_selector.value(),
            automatic_hits=automatic_hits,
            automatic_interval=automatic_interval,
            sds_table=None if self.run_table is None else self.run_table.sds_table,
            allow_automatic_updates=self.run_table_dialog.allow_automatic_updates_checkbox.isChecked(),
        )
        return instructions

    def set_environment_instructions(self, instructions):
        self.run_widget.current_test_level_selector.setValue(instructions.control_test_level)
        self.run_widget.target_hits_at_level_selector.setValue(instructions.target_hits_at_level)
        self.run_widget.auto_hits_checkbox.setChecked(instructions.automatic_hits)
        self.run_widget.manual_hits_checkbox.setChecked(not instructions.automatic_hits)
        self.run_table_dialog.allow_automatic_updates_checkbox.setChecked(
            instructions.allow_automatic_updates
        )

        if instructions.automatic_interval is not None:
            self.run_widget.auto_hit_interval_selector.setTime(
                QTime(0, 0).addSecs(int(instructions.automatic_interval))
            )

        if instructions.sds_table is not None and self.run_table is not None:
            self.run_table.sds_table = instructions.sds_table
            self.run_table.update_ui()

        super().set_environment_instructions(instructions)

    def update_global_srs_plot(self, measured_response_srs):
        plot_item = self.run_widget.global_test_performance_plot.getPlotItem()
        plot_item.clear()
        plot_item.showGrid(True, True, 0.25)
        plot_item.enableAutoRange()
        plot_item.getViewBox().enableAutoRange(enable=True)
        plot_item.setLogMode(True, True)

        if measured_response_srs is None:
            return

        freqs = self.environment_metadata.get_sds_frequencies()
        for i in range(measured_response_srs.shape[1]):
            pen = {"color": colororder[i % len(colororder)], "width": 1}
            plot_item.plot(freqs, measured_response_srs[:, i], pen=pen)

    # region UI Update and Templates

    def update_gui(self, queue_data):
        self.log(f"Updating GUI: {queue_data[0]}")
        if super().update_gui(queue_data):
            return
        message, data = queue_data
        if message == SDSUICommands.CONTROL_PREDICTIONS:
            self.log("Updating Prediction Table Predictions")
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
        elif message == SDSUICommands.RUN_CONTROL_PREDICTIONS:
            self.log("Updating Run Table Predictions")
            (
                predicted_amplitudes,
                predicted_delays,
                predicted_decays,
                predicted_drive_time_history,
                predicted_response_time_history,
                predicted_response_srs,
            ) = data
            self.run_table.update_prediction_information(
                predicted_response_time_history,
                predicted_response_srs,
                predicted_amplitudes,
                predicted_delays,
                predicted_decays,
                predicted_drive_time_history,
            )
        elif message == SDSUICommands.CONTROL_UPDATE:
            self.log("Updating UI with control data")
            self.run_widget.current_total_hits_display.setValue(data["total_hits"])
            self.run_widget.current_hits_at_level_display.setValue(data["hits_at_target"])

            target_hits = data["target_hits_at_level"]
            if target_hits is None:
                target_hits = 1
            progress = int(100 * data["hits_at_target"] / max(1, target_hits))
            self.run_widget.current_hits_progress.setValue(min(progress, 100))

            if self.run_table is not None:
                self.run_table.update_control_information(
                    measured_response_time_history=data["measured_response_time_history"],
                    measured_response_srs=data["measured_response_srs"],
                    run_sds_table=data["run_sds_table"],
                )

            # update global plot
            if data["measured_response_srs"] is not None:
                self.update_global_srs_plot(data["measured_response_srs"])

            self.shock_history = data["hit_history"]

            self.shock_history_dialog.update_history(
                hit_history=self.shock_history,
                total_hits=data["total_hits"],
                hits_at_target=data["hits_at_target"],
                target_hits=data["target_hits_at_level"],
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

    # region Templates
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
