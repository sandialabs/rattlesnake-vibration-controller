import datetime
import inspect
import time
import os

import numpy as np
from qtpy import QtWidgets, uic
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor

from rattlesnake.engine import RattlesnakeController
from rattlesnake.environment.abstract_interactive_control_law import (
    AbstractControlLawComputation,
    ControlLawUICommands,
)
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.random_vibration_sys_id_environment import (
    RandomVibrationCommands,
    RandomVibrationMetadata,
    RandomVibrationUICommands,
    RandomVibrationInstructions,
)
from rattlesnake.process.random_vibration_sys_id_data_analysis import (
    RandomVibrationDataAnalysisUICommands,
)
from rattlesnake.user_interface.abstract_sys_id_user_interface import SysIdEnvironmentUI
from rattlesnake.environment.random_vibration_sys_id_utilities import (
    load_specification,
)
from rattlesnake.user_interface.ui_utilities import (
    PlotWindow,
    TransformationMatrixWindow,
    UICommands,
    error_message_qt,
    multiline_plotter,
)
from rattlesnake.utilities import (
    _direction_map,
    DIRECTORY,
    load_python_module,
)

CONTROL_TYPE = EnvironmentType.RANDOM
MAXIMUM_NAME_LENGTH = 50


# region User Interface
class RandomVibrationUI(SysIdEnvironmentUI):
    """Class defining the user interface for a Random Vibration environment.

    This class will contain four main UIs, the environment definition,
    system identification, test prediction, and run.  The widgets corresponding
    to these interfaces are stored in TabWidgets in the main UI.

    This class defines all the call backs and user interface operations required
    for the Random Vibration environment."""

    def __init__(
        self,
        environment_name: str,
        rattlesnake: RattlesnakeController,
    ):
        """
        Constructs a Random Vibration User Interface

        Given the tab widgets from the main interface as well as communication
        queues, this class assembles the user interface components specific to
        the Random Vibration Environment

        Parameters
        ----------
        definition_tabwidget : QtWidgets.QTabWidget
            QTabWidget containing the environment subtabs on the Control
            Definition main tab
        system_id_tabwidget : QtWidgets.QTabWidget
            QTabWidget containing the environment subtabs on the System
            Identification main tab
        test_predictions_tabwidget : QtWidgets.QTabWidget
            QTabWidget containing the environment subtabs on the Test Predictions
            main tab
        run_tabwidget : QtWidgets.QTabWidget
            QTabWidget containing the environment subtabs on the Run
            main tab.
        environment_command_queue : VerboseMessageQueue
            Queue for sending commands to the Random Vibration Environment
        controller_communication_queue : VerboseMessageQueue
            Queue for sending global commands to the controller
        log_file_queue : Queue
            Queue where log file messages can be written.

        """
        super().__init__(
            CONTROL_TYPE,
            environment_name,
            rattlesnake,
        )
        # Add the page to the control definition tabwidget
        self.definition_widget = QtWidgets.QWidget()
        random_definition_ui_path = os.path.join(
            DIRECTORY, "user_interface", "ui_files", "random_vibration_definition.ui"
        )
        uic.loadUi(random_definition_ui_path, self.definition_widget)
        # Add the page to the control prediction tabwidget
        self.prediction_widget = QtWidgets.QWidget()
        random_prediction_ui_path = os.path.join(
            DIRECTORY, "user_interface", "ui_files", "random_vibration_prediction.ui"
        )
        uic.loadUi(random_prediction_ui_path, self.prediction_widget)
        # Add the page to the run tabwidget
        self.run_widget = QtWidgets.QWidget()
        random_run_ui_path = os.path.join(
            DIRECTORY, "user_interface", "ui_files", "random_vibration_run.ui"
        )
        uic.loadUi(random_run_ui_path, self.run_widget)

        self.plot_data_items = {}
        self.plot_windows = []
        self.run_start_time = None
        self.run_level_start_time = None
        self.run_timer = QTimer()
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.python_control_module = None
        self.specification_frequency_lines = None
        self.specification_cpsd_matrix = None
        self.specification_warning_matrix = None
        self.specification_abort_matrix = None
        self.physical_channel_names = None
        self.physical_output_indices = None
        self.excitation_prediction = None
        self.response_prediction = None
        self.rms_voltage_prediction = None
        self.rms_db_error_prediction = None
        self.interactive_control_law_widget = None
        self.interactive_control_law_window = None
        self.control_selector_widgets = [
            self.definition_widget.specification_row_selector,
            self.definition_widget.specification_column_selector,
            self.prediction_widget.response_row_selector,
            self.prediction_widget.response_column_selector,
            self.run_widget.control_channel_1_selector,
            self.run_widget.control_channel_2_selector,
        ]
        self.output_selector_widgets = [
            self.prediction_widget.excitation_row_selector,
            self.prediction_widget.excitation_column_selector,
        ]
        self.system_id_widget.samplesPerFrameSpinBox.setReadOnly(True)
        self.system_id_widget.samplesPerFrameSpinBox.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.system_id_widget.levelRampTimeDoubleSpinBox.setReadOnly(True)
        self.system_id_widget.levelRampTimeDoubleSpinBox.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )

        # Set common look and feel for plots
        plot_widgets = [
            self.definition_widget.specification_single_plot,
            self.definition_widget.specification_sum_asds_plot,
            self.prediction_widget.excitation_display_plot,
            self.prediction_widget.response_display_plot,
            self.run_widget.global_test_performance_plot,
        ]
        for plot_widget in plot_widgets:
            plot_item = plot_widget.getPlotItem()
            plot_item.showGrid(True, True, 0.25)
            plot_item.enableAutoRange()
            plot_item.getViewBox().enableAutoRange(enable=True)
        logscale_plot_widgets = [
            self.definition_widget.specification_single_plot,
            self.definition_widget.specification_sum_asds_plot,
            self.prediction_widget.excitation_display_plot,
            self.prediction_widget.response_display_plot,
            self.run_widget.global_test_performance_plot,
        ]
        for plot_widget in logscale_plot_widgets:
            plot_item = plot_widget.getPlotItem()
            plot_item.setLogMode(False, True)

        self.connect_callbacks()

        # Complete the profile commands
        # self.command_map["Set Test Level"] = self.change_test_level_from_profile
        # self.command_map["Change Specification"] = (
        #     self.change_specification_from_profile
        # )
        # self.command_map["Save Control Data"] = self.save_control_data_from_profile

    def connect_callbacks(self):
        """Connects callback functions to the UI Widgets"""
        # Definition
        self.definition_widget.samples_per_frame_selector.valueChanged.connect(
            self.update_parameters_and_clear_spec
        )
        self.definition_widget.cpsd_overlap_selector.valueChanged.connect(
            self.update_parameters
        )
        self.definition_widget.cola_overlap_percentage_selector.valueChanged.connect(
            self.update_parameters
        )
        self.definition_widget.transformation_matrices_button.clicked.connect(
            self.define_transformation_matrices
        )
        self.definition_widget.control_script_load_file_button.clicked.connect(
            self.select_python_module
        )
        self.definition_widget.control_function_input.currentIndexChanged.connect(
            self.update_generator_selector
        )
        self.definition_widget.load_spec_button.clicked.connect(self.select_spec_file)
        self.definition_widget.specification_row_selector.currentIndexChanged.connect(
            self.show_specification
        )
        self.definition_widget.specification_column_selector.currentIndexChanged.connect(
            self.show_specification
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
        # Prediction
        self.prediction_widget.excitation_row_selector.currentIndexChanged.connect(
            self.update_control_predictions
        )
        self.prediction_widget.excitation_column_selector.currentIndexChanged.connect(
            self.update_control_predictions
        )
        self.prediction_widget.response_row_selector.currentIndexChanged.connect(
            self.update_control_predictions
        )
        self.prediction_widget.response_column_selector.currentIndexChanged.connect(
            self.update_control_predictions
        )
        self.prediction_widget.maximum_voltage_button.clicked.connect(
            self.show_max_voltage_prediction
        )
        self.prediction_widget.minimum_voltage_button.clicked.connect(
            self.show_min_voltage_prediction
        )
        self.prediction_widget.maximum_error_button.clicked.connect(
            self.show_max_error_prediction
        )
        self.prediction_widget.minimum_error_button.clicked.connect(
            self.show_min_error_prediction
        )
        self.prediction_widget.response_error_list.itemClicked.connect(
            self.update_response_error_prediction_selector
        )
        self.prediction_widget.excitation_voltage_list.itemClicked.connect(
            self.update_excitation_prediction_selector
        )
        self.prediction_widget.recompute_prediction_button.clicked.connect(
            self.recompute_prediction
        )
        # Run Test
        self.run_widget.current_test_level_selector.valueChanged.connect(
            self.change_control_test_level
        )
        self.run_widget.start_test_button.clicked.connect(self.start_environment)
        self.run_widget.stop_test_button.clicked.connect(self.stop_environment)
        self.run_widget.create_window_button.clicked.connect(self.create_window)
        self.run_widget.show_all_asds_button.clicked.connect(self.show_all_asds)
        self.run_widget.show_all_csds_phscoh_button.clicked.connect(
            self.show_all_csds_phscoh
        )
        self.run_widget.show_all_csds_realimag_button.clicked.connect(
            self.show_all_csds_realimag
        )
        self.run_widget.tile_windows_button.clicked.connect(self.tile_windows)
        self.run_widget.close_windows_button.clicked.connect(self.close_windows)
        self.run_timer.timeout.connect(self.update_run_time)
        self.run_widget.test_response_error_list.itemDoubleClicked.connect(
            self.show_magnitude_window
        )
        self.run_widget.save_current_spectral_data_button.clicked.connect(
            self.save_spectral_data
        )

    @property
    def physical_output_names(self):
        """Names of the physical output channels"""
        return [self.physical_channel_names[i] for i in self.physical_output_indices]

    @property
    def physical_control_indices(self):
        """Indices corresponding to the physical channels that are used as outputs"""
        return [
            i
            for i in range(self.definition_widget.control_channels_selector.count())
            if self.definition_widget.control_channels_selector.item(i).checkState()
            == Qt.Checked
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

    # endregion

    # region State Sync
    def initialize_hardware(self, hardware_metadata):
        super().initialize_hardware(hardware_metadata)

        self.definition_widget.specification_single_plot.getPlotItem().clear()
        self.definition_widget.specification_sum_asds_plot.getPlotItem().clear()
        self.run_widget.global_test_performance_plot.getPlotItem().clear()

        # Now add initial lines that we can update later
        self.definition_widget.specification_single_plot.getPlotItem().addLegend()
        self.plot_data_items[
            "specification_real"
        ] = self.definition_widget.specification_single_plot.getPlotItem().plot(
            np.array([0, hardware_metadata.sample_rate / 2]),
            np.zeros(2),
            pen={"color": "b", "width": 1},
            name="Real Part",
        )
        self.plot_data_items[
            "specification_imag"
        ] = self.definition_widget.specification_single_plot.getPlotItem().plot(
            np.array([0, hardware_metadata.sample_rate / 2]),
            np.zeros(2),
            pen={"color": "r", "width": 1},
            name="Imaginary Part",
        )
        self.plot_data_items[
            "specification_warning_upper"
        ] = self.definition_widget.specification_single_plot.getPlotItem().plot(
            np.array([0, hardware_metadata.sample_rate / 2]),
            np.zeros(2),
            pen={"color": PlotWindow.WARNING_COLOR, "width": 0.25},
            name="Warning",
        )
        self.plot_data_items[
            "specification_warning_lower"
        ] = self.definition_widget.specification_single_plot.getPlotItem().plot(
            np.array([0, hardware_metadata.sample_rate / 2]),
            np.zeros(2),
            pen={"color": PlotWindow.WARNING_COLOR, "width": 0.25},
        )
        self.plot_data_items[
            "specification_abort_upper"
        ] = self.definition_widget.specification_single_plot.getPlotItem().plot(
            np.array([0, hardware_metadata.sample_rate / 2]),
            np.zeros(2),
            pen={"color": PlotWindow.ABORT_COLOR, "width": 0.25},
            name="Abort",
        )
        self.plot_data_items[
            "specification_abort_lower"
        ] = self.definition_widget.specification_single_plot.getPlotItem().plot(
            np.array([0, hardware_metadata.sample_rate / 2]),
            np.zeros(2),
            pen={"color": PlotWindow.ABORT_COLOR, "width": 0.25},
        )
        self.plot_data_items[
            "specification_sum"
        ] = self.definition_widget.specification_sum_asds_plot.getPlotItem().plot(
            np.array([0, hardware_metadata.sample_rate / 2]),
            np.zeros(2),
            pen={"color": "b", "width": 1},
        )
        self.run_widget.global_test_performance_plot.getPlotItem().addLegend()
        self.plot_data_items[
            "specification_sum_control"
        ] = self.run_widget.global_test_performance_plot.getPlotItem().plot(
            np.array([0, hardware_metadata.sample_rate / 2]),
            np.zeros(2),
            pen={"color": "b", "width": 1},
            name="Specification",
        )
        self.plot_data_items[
            "sum_asds_control"
        ] = self.run_widget.global_test_performance_plot.getPlotItem().plot(
            np.array([0, hardware_metadata.sample_rate / 2]),
            np.zeros(2),
            pen={"color": "r", "width": 1},
            name="Response",
        )

        # Set up channel names
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
        # Set up widgets
        self.definition_widget.sample_rate_display.setValue(
            hardware_metadata.sample_rate
        )
        self.definition_widget.samples_per_frame_selector.setValue(
            hardware_metadata.sample_rate
        )
        self.definition_widget.control_channels_selector.clear()
        for channel_name in self.physical_channel_names:
            item = QtWidgets.QListWidgetItem()
            item.setText(channel_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.definition_widget.control_channels_selector.addItem(item)
        self.definition_widget.input_channels_display.setValue(
            len(self.physical_channel_names)
        )
        self.definition_widget.output_channels_display.setValue(
            len(self.physical_output_indices)
        )
        self.definition_widget.control_channels_display.setValue(0)
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.define_transformation_matrices(None, False)

    def initialize_environment(self, environment_metadata):
        super().initialize_environment(environment_metadata)
        self.system_id_widget.samplesPerFrameSpinBox.setMaximum(
            self.definition_widget.samples_per_frame_selector.value()
        )
        self.system_id_widget.samplesPerFrameSpinBox.setValue(
            self.definition_widget.samples_per_frame_selector.value()
        )
        self.system_id_widget.levelRampTimeDoubleSpinBox.setValue(
            self.definition_widget.ramp_time_spinbox.value()
        )

        for widget in [
            self.prediction_widget.response_row_selector,
            self.prediction_widget.response_column_selector,
            self.run_widget.control_channel_1_selector,
            self.run_widget.control_channel_2_selector,
        ]:
            widget.blockSignals(True)
            widget.clear()
            for i, control_name in enumerate(self.initialized_control_names):
                widget.addItem(f"{i + 1}: {control_name}")
            widget.blockSignals(False)
        for widget in [
            self.prediction_widget.excitation_row_selector,
            self.prediction_widget.excitation_column_selector,
        ]:
            widget.blockSignals(True)
            widget.clear()
            for i, drive_name in enumerate(self.initialized_output_names):
                widget.addItem(f"{i + 1}: {drive_name}")
            widget.blockSignals(False)
        # Set up the prediction plots
        self.prediction_widget.excitation_display_plot.getPlotItem().clear()
        self.prediction_widget.response_display_plot.getPlotItem().clear()
        self.prediction_widget.excitation_display_plot.getPlotItem().addLegend()
        self.prediction_widget.response_display_plot.getPlotItem().addLegend()
        self.plot_data_items["response_prediction"] = multiline_plotter(
            np.arange(self.environment_metadata.fft_lines)
            * self.environment_metadata.frequency_spacing,
            np.zeros((4, self.environment_metadata.fft_lines)),
            widget=self.prediction_widget.response_display_plot,
            other_pen_options={"width": 2},
            names=["Real Prediction", "Real Spec", "Imag Prediction", "Imag Spec"],
        )
        self.plot_data_items[
            "prediction_warning_upper"
        ] = self.prediction_widget.response_display_plot.getPlotItem().plot(
            np.array([0, self.hardware_metadata.sample_rate / 2]),
            np.zeros(2),
            pen={
                "color": PlotWindow.WARNING_COLOR,
                "width": PlotWindow.WARNING_LINEWIDTH,
                "style": PlotWindow.WARNING_LINESTYLE,
            },
            name="Warning",
        )
        self.plot_data_items[
            "prediction_warning_lower"
        ] = self.prediction_widget.response_display_plot.getPlotItem().plot(
            np.array([0, self.hardware_metadata.sample_rate / 2]),
            np.zeros(2),
            pen={
                "color": PlotWindow.WARNING_COLOR,
                "width": PlotWindow.WARNING_LINEWIDTH,
                "style": PlotWindow.WARNING_LINESTYLE,
            },
        )
        self.plot_data_items[
            "prediction_abort_upper"
        ] = self.prediction_widget.response_display_plot.getPlotItem().plot(
            np.array([0, self.hardware_metadata.sample_rate / 2]),
            np.zeros(2),
            pen={
                "color": PlotWindow.ABORT_COLOR,
                "width": PlotWindow.ABORT_LINEWIDTH,
                "style": PlotWindow.ABORT_LINESTYLE,
            },
            name="Abort",
        )
        self.plot_data_items[
            "prediction_abort_lower"
        ] = self.prediction_widget.response_display_plot.getPlotItem().plot(
            np.array([0, self.hardware_metadata.sample_rate / 2]),
            np.zeros(2),
            pen={
                "color": PlotWindow.ABORT_COLOR,
                "width": PlotWindow.ABORT_LINEWIDTH,
                "style": PlotWindow.ABORT_LINESTYLE,
            },
        )
        self.plot_data_items["excitation_prediction"] = multiline_plotter(
            np.arange(self.environment_metadata.fft_lines)
            * self.environment_metadata.frequency_spacing,
            np.zeros((2, self.environment_metadata.fft_lines)),
            widget=self.prediction_widget.excitation_display_plot,
            other_pen_options={"width": 1},
            names=["Real Prediction", "Imag Prediction"],
        )
        # Create the interactive control law if necessary
        if (
            self.definition_widget.control_function_generator_selector.currentIndex()
            == 3
        ):
            control_class = getattr(
                self.python_control_module,
                self.definition_widget.control_function_input.itemText(
                    self.definition_widget.control_function_input.currentIndex()
                ),
            )
            self.log(f"Building Interactive UI for class {control_class.__name__}")
            ui_class = control_class.get_ui_class()
            if ui_class == self.interactive_control_law_widget.__class__:
                print("initializing data acquisition and environment parameters")
                self.interactive_control_law_widget.initialize_parameters(
                    self.hardware_metadata, self.environment_metadata
                )
            else:
                if self.interactive_control_law_widget is not None:
                    self.interactive_control_law_widget.close()
                self.interactive_control_law_window = QtWidgets.QDialog(
                    self.definition_widget
                )
                self.interactive_control_law_widget = ui_class(
                    self.log_name,
                    self.environment_command_queue,
                    self.interactive_control_law_window,
                    self,
                    self.hardware_metadata,
                    self.environment_metadata,
                )
            self.interactive_control_law_window.show()

    def get_environment_metadata(self, global_channel_list=None):
        if self.hardware_metadata and global_channel_list:
            channel_list_bools = self.get_channel_list_bools(global_channel_list)
        else:
            channel_list_bools = []

        if self.python_control_module is None:
            control_module = None
            control_function = None
            control_function_type = None
            control_function_parameters = None
        else:
            control_module = (
                self.definition_widget.control_script_file_path_input.text()
            )
            control_function = self.definition_widget.control_function_input.itemText(
                self.definition_widget.control_function_input.currentIndex()
            )
            control_function_type = (
                self.definition_widget.control_function_generator_selector.currentIndex()
            )
            control_function_parameters = (
                self.definition_widget.control_parameters_text_input.toPlainText()
            )
        return RandomVibrationMetadata(
            environment_name=self.environment_name,
            channel_list_bools=channel_list_bools,
            sample_rate=self.definition_widget.sample_rate_display.value(),
            number_of_channels=len(self.hardware_metadata.channel_list),
            samples_per_frame=self.definition_widget.samples_per_frame_selector.value(),
            test_level_ramp_time=self.definition_widget.ramp_time_spinbox.value(),
            cola_window=self.definition_widget.cola_window_selector.itemText(
                self.definition_widget.cola_window_selector.currentIndex()
            ),
            cola_overlap=self.definition_widget.cola_overlap_percentage_selector.value()
            / 100,
            cola_window_exponent=self.definition_widget.cola_exponent_selector.value(),
            sigma_clip=self.definition_widget.sigma_clipping_selector.value(),
            update_tf_during_control=self.definition_widget.update_transfer_function_during_control_selector.isChecked(),
            frames_in_cpsd=self.definition_widget.cpsd_frames_selector.value(),
            cpsd_window=self.definition_widget.cpsd_computation_window_selector.itemText(
                self.definition_widget.cpsd_computation_window_selector.currentIndex()
            ),
            cpsd_overlap=self.definition_widget.cpsd_overlap_selector.value() / 100,
            response_transformation_matrix=self.response_transformation_matrix,
            output_transformation_matrix=self.output_transformation_matrix,
            control_python_script=control_module,
            control_python_function=control_function,
            control_python_function_type=control_function_type,
            control_python_function_parameters=control_function_parameters,
            control_channel_indices=self.physical_control_indices,
            output_channel_indices=self.physical_output_indices,
            specification_frequency_lines=self.specification_frequency_lines,
            specification_cpsd_matrix=self.specification_cpsd_matrix,
            specification_warning_matrix=self.specification_warning_matrix,
            specification_abort_matrix=self.specification_abort_matrix,
            percent_lines_out=self.definition_widget.frequency_lines_out_spinbox.value(),
            allow_automatic_aborts=self.definition_widget.auto_abort_checkbox.isChecked(),
        )

    def set_environment_metadata(self, metadata: RandomVibrationMetadata):
        self.definition_widget.sample_rate_display.setValue(metadata.sample_rate)
        self.definition_widget.samples_per_frame_selector.setValue(
            metadata.samples_per_frame
        )
        self.definition_widget.ramp_time_spinbox.setValue(metadata.test_level_ramp_time)
        index = self.definition_widget.cola_window_selector.findText(
            metadata.cola_window
        )
        if index >= 0:
            self.definition_widget.cola_window_selector.setCurrentIndex(index)

        self.definition_widget.cola_overlap_percentage_selector.setValue(
            metadata.cola_overlap * 100
        )
        self.definition_widget.cola_exponent_selector.setValue(
            metadata.cola_window_exponent
        )
        self.definition_widget.sigma_clipping_selector.setValue(metadata.sigma_clip)
        self.definition_widget.update_transfer_function_during_control_selector.setChecked(
            metadata.update_tf_during_control
        )
        self.definition_widget.cpsd_frames_selector.setValue(metadata.frames_in_cpsd)
        index = self.definition_widget.cpsd_computation_window_selector.findText(
            metadata.cpsd_window
        )
        if index >= 0:
            self.definition_widget.cpsd_computation_window_selector.setCurrentIndex(
                index
            )
        self.definition_widget.cpsd_overlap_selector.setValue(
            metadata.cpsd_overlap * 100
        )
        if metadata.control_python_script:
            self.select_python_module(None, metadata.control_python_script)
            function_index = self.definition_widget.control_function_input.findText(
                metadata.control_python_function
            )
            if function_index >= 0:
                self.definition_widget.control_function_input.setCurrentIndex(
                    function_index
                )
            self.definition_widget.control_function_generator_selector.setCurrentIndex(
                metadata.control_python_function_type
            )
        self.definition_widget.control_parameters_text_input.setPlainText(
            metadata.control_python_function_parameters or ""
        )
        self.definition_widget.frequency_lines_out_spinbox.setValue(
            metadata.percent_lines_out
        )
        self.definition_widget.auto_abort_checkbox.setChecked(
            metadata.allow_automatic_aborts
        )
        for i in range(self.definition_widget.control_channels_selector.count()):
            state = (
                Qt.Checked if i in metadata.control_channel_indices else Qt.Unchecked
            )
            self.definition_widget.control_channels_selector.item(i).setCheckState(
                state
            )

        self.response_transformation_matrix = metadata.response_transformation_matrix
        self.output_transformation_matrix = metadata.reference_transformation_matrix
        self.specification_frequency_lines = metadata.specification_frequency_lines
        self.specification_cpsd_matrix = metadata.specification_cpsd_matrix
        self.specification_warning_matrix = metadata.specification_warning_matrix
        self.specification_abort_matrix = metadata.specification_abort_matrix

        if np.all(np.isnan(self.specification_abort_matrix)):
            self.definition_widget.auto_abort_checkbox.setChecked(False)
            self.definition_widget.auto_abort_checkbox.setEnabled(False)
        else:
            self.definition_widget.auto_abort_checkbox.setEnabled(True)
        self.show_specification()

    def get_environment_instructions(self):
        control_test_level = self.run_widget.current_test_level_selector.value()
        return RandomVibrationInstructions(self.environment_name, control_test_level)

    def set_environment_instructions(self, instructions):
        self.run_widget.current_test_level_selector.setValue(
            instructions.control_test_level
        )
        super().set_environment_instructions(instructions)

    # endregion

    # region Definition
    def select_spec_file(
        self, clicked, filename=None
    ):  # pylint: disable=unused-argument
        """Loads a specification using a dialog or the specified filename

        Parameters
        ----------
        clicked :
            The clicked event that triggered the callback.
        filename :
            File name defining the specification for bypassing the callback when
            loading from a file (Default value = None).

        """
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.definition_widget,
                "Select Specification File",
                filter="Numpyz or Mat (*.npz *.mat)",
            )
            if filename == "":
                return
        self.definition_widget.specification_file_name_display.setText(filename)
        coord_dtype = np.dtype([("node", "<u8"), ("direction", "i1")])
        if self.response_transformation_matrix is not None:
            control_coordinate = None
        else:
            control_coordinate = np.array(
                [
                    (
                        self.hardware_metadata.channel_list[i].node_number,
                        _direction_map[
                            self.hardware_metadata.channel_list[i].node_direction
                        ],
                    )
                    for i in self.physical_control_indices
                ],
                dtype=coord_dtype,
            )
        try:
            frequency_spacing = (
                self.definition_widget.sample_rate_display.value()
                / self.definition_widget.samples_per_frame_selector.value()
            )
            (
                self.specification_frequency_lines,
                self.specification_cpsd_matrix,
                self.specification_warning_matrix,
                self.specification_abort_matrix,
            ) = load_specification(
                filename,
                self.definition_widget.fft_lines_display.value(),
                frequency_spacing,
                control_coordinate,
            )
        except ValueError as e:
            error_message_qt(type(e).__name__, str(e))
            return

        if np.all(np.isnan(self.specification_abort_matrix)):
            self.definition_widget.auto_abort_checkbox.setChecked(False)
            self.definition_widget.auto_abort_checkbox.setEnabled(False)
        else:
            self.definition_widget.auto_abort_checkbox.setEnabled(True)
        self.show_specification()

    def select_python_module(
        self, clicked, filename=None
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
        if filename is None:
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
                and len(inspect.signature(function[1]).parameters) >= 12
            )
            or inspect.isgeneratorfunction(function[1])
            or (
                inspect.isclass(function[1])
                and all(
                    [
                        (
                            method in function[1].__dict__
                            and not (
                                hasattr(
                                    function[1].__dict__[method], "__isabstractmethod__"
                                )
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

    def update_generator_selector(self):
        """Updates the function/generator selector based on the function selected"""
        if self.python_control_module is None:
            return
        try:
            function = getattr(
                self.python_control_module,
                self.definition_widget.control_function_input.itemText(
                    self.definition_widget.control_function_input.currentIndex()
                ),
            )
        except AttributeError:
            return
        if inspect.isgeneratorfunction(function):
            self.definition_widget.control_function_generator_selector.setCurrentIndex(
                1
            )
        elif inspect.isclass(function) and issubclass(
            function, AbstractControlLawComputation
        ):
            self.definition_widget.control_function_generator_selector.setCurrentIndex(
                3
            )
        elif inspect.isclass(function):
            self.definition_widget.control_function_generator_selector.setCurrentIndex(
                2
            )
        else:
            self.definition_widget.control_function_generator_selector.setCurrentIndex(
                0
            )

    def show_specification(self):
        """Show the specification on the GUI"""
        if self.specification_cpsd_matrix is None:
            self.plot_data_items["specification_real"].setData(
                np.array([0, self.definition_widget.sample_rate_display.value() / 2]),
                np.zeros(2),
            )
            self.plot_data_items["specification_imag"].setData(
                np.array([0, self.definition_widget.sample_rate_display.value() / 2]),
                np.zeros(2),
            )
            self.plot_data_items["specification_sum"].setData(
                np.array([0, self.definition_widget.sample_rate_display.value() / 2]),
                np.zeros(2),
            )
            self.plot_data_items["specification_warning_upper"].setData(
                np.array([0, self.definition_widget.sample_rate_display.value() / 2]),
                np.zeros(2),
            )
            self.plot_data_items["specification_warning_lower"].setData(
                np.array([0, self.definition_widget.sample_rate_display.value() / 2]),
                np.zeros(2),
            )
            self.plot_data_items["specification_abort_upper"].setData(
                np.array([0, self.definition_widget.sample_rate_display.value() / 2]),
                np.zeros(2),
            )
            self.plot_data_items["specification_abort_lower"].setData(
                np.array([0, self.definition_widget.sample_rate_display.value() / 2]),
                np.zeros(2),
            )
            # enabled_state = self.run_widget.isEnabled()
            # self.run_widget.setEnabled(True)
            self.plot_data_items["specification_sum_control"].setData(
                np.array([0, self.definition_widget.sample_rate_display.value() / 2]),
                np.zeros(2),
            )
            # self.run_widget.setEnabled(enabled_state)
        else:
            row = self.definition_widget.specification_row_selector.currentIndex()
            column = self.definition_widget.specification_column_selector.currentIndex()
            spec_real = abs(self.specification_cpsd_matrix[:, row, column].real)
            spec_imag = abs(self.specification_cpsd_matrix[:, row, column].imag)
            spec_sum = abs(
                np.nansum(
                    self.specification_cpsd_matrix[
                        :,
                        np.arange(self.specification_cpsd_matrix.shape[-1]),
                        np.arange(self.specification_cpsd_matrix.shape[-1]),
                    ],
                    axis=-1,
                )
            )
            self.plot_data_items["specification_real"].setData(
                self.specification_frequency_lines[spec_real > 0.0],
                spec_real[spec_real > 0.0],
            )
            self.plot_data_items["specification_imag"].setData(
                self.specification_frequency_lines[spec_imag > 0.0],
                spec_imag[spec_imag > 0.0],
            )
            if row == column:
                warning_upper = abs(self.specification_warning_matrix[1, :, row])
                warning_lower = abs(self.specification_warning_matrix[0, :, row])
                abort_upper = abs(self.specification_abort_matrix[1, :, row])
                abort_lower = abs(self.specification_abort_matrix[0, :, row])
                self.plot_data_items["specification_warning_upper"].setData(
                    self.specification_frequency_lines, warning_upper
                )
                self.plot_data_items["specification_warning_lower"].setData(
                    self.specification_frequency_lines, warning_lower
                )
                self.plot_data_items["specification_abort_upper"].setData(
                    self.specification_frequency_lines, abort_upper
                )
                self.plot_data_items["specification_abort_lower"].setData(
                    self.specification_frequency_lines, abort_lower
                )
            else:
                self.plot_data_items["specification_warning_upper"].setData(
                    np.array(
                        [0, self.definition_widget.sample_rate_display.value() / 2]
                    ),
                    np.zeros(2),
                )
                self.plot_data_items["specification_warning_lower"].setData(
                    np.array(
                        [0, self.definition_widget.sample_rate_display.value() / 2]
                    ),
                    np.zeros(2),
                )
                self.plot_data_items["specification_abort_upper"].setData(
                    np.array(
                        [0, self.definition_widget.sample_rate_display.value() / 2]
                    ),
                    np.zeros(2),
                )
                self.plot_data_items["specification_abort_lower"].setData(
                    np.array(
                        [0, self.definition_widget.sample_rate_display.value() / 2]
                    ),
                    np.zeros(2),
                )
            self.plot_data_items["specification_sum"].setData(
                self.specification_frequency_lines[spec_sum > 0.0],
                spec_sum[spec_sum > 0.0],
            )
            # enabled_state = self.run_widget.isEnabled()
            # self.run_widget.setEnabled(True)
            self.plot_data_items["specification_sum_control"].setData(
                self.specification_frequency_lines[spec_sum > 0.0],
                spec_sum[spec_sum > 0.0],
            )
            # self.run_widget.setEnabled(enabled_state)

    def check_selected_control_channels(self):
        """Checks the selected channels to make them control channels"""
        for item in self.definition_widget.control_channels_selector.selectedItems():
            item.setCheckState(Qt.Checked)

    def uncheck_selected_control_channels(self):
        """Unchecks the selected channels to make them no longer control channels"""
        for item in self.definition_widget.control_channels_selector.selectedItems():
            item.setCheckState(Qt.Unchecked)

    def update_control_channels(self):
        """Resets the definition UI when the number of control channels has changed"""
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.specification_abort_matrix = None
        self.specification_warning_matrix = None
        self.specification_cpsd_matrix = None
        self.specification_frequency_lines = None
        self.definition_widget.control_channels_display.setValue(
            len(self.physical_control_indices)
        )
        self.definition_widget.specification_row_selector.blockSignals(True)
        self.definition_widget.specification_column_selector.blockSignals(True)
        self.definition_widget.specification_row_selector.clear()
        self.definition_widget.specification_column_selector.clear()
        for i, control_name in enumerate(self.physical_control_names):
            self.definition_widget.specification_row_selector.addItem(
                f"{i + 1}: {control_name}"
            )
            self.definition_widget.specification_column_selector.addItem(
                f"{i + 1}: {control_name}"
            )
        self.definition_widget.specification_row_selector.blockSignals(False)
        self.definition_widget.specification_column_selector.blockSignals(False)
        self.define_transformation_matrices(None, False)
        self.show_specification()

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
                        widget.addItem(f"{i + 1}: Transformed Response")
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
                        widget.addItem(f"{i + 1}: Transformed Drive")
                self.definition_widget.transform_outputs_display.setValue(
                    output_transformation.shape[0]
                )
            for widget in self.output_selector_widgets:
                widget.blockSignals(False)

            self.response_transformation_matrix = response_transformation
            self.output_transformation_matrix = output_transformation
            self.update_parameters_and_clear_spec()

    def update_parameters(self):
        """Recompute derived parameters from updated sampling parameters"""
        data = self.get_environment_metadata()
        self.definition_widget.samples_per_acquire_display.setValue(
            data.samples_per_acquire
        )
        self.definition_widget.frame_time_display.setValue(data.frame_time)
        self.definition_widget.nyquist_frequency_display.setValue(
            data.nyquist_frequency
        )
        self.definition_widget.fft_lines_display.setValue(data.fft_lines)
        self.definition_widget.frequency_spacing_display.setValue(
            data.frequency_spacing
        )
        self.definition_widget.samples_per_write_display.setValue(
            data.samples_per_output
        )

    def update_parameters_and_clear_spec(self):
        """Clears the specification data and updates parameters"""
        samples_per_frame = self.definition_widget.samples_per_frame_selector.value()
        if samples_per_frame % 2 != 0:
            self.definition_widget.samples_per_frame_selector.blockSignals(True)
            self.definition_widget.samples_per_frame_selector.setValue(
                samples_per_frame + 1
            )
            self.definition_widget.samples_per_frame_selector.blockSignals(False)
        self.specification_frequency_lines = None
        self.specification_cpsd_matrix = None
        self.specification_warning_matrix = None
        self.specification_abort_matrix = None
        self.definition_widget.specification_file_name_display.setText("")
        self.show_specification()
        self.update_parameters()

    # endregion

    # region Prediction
    def show_max_voltage_prediction(self):
        """Shows the prediction with the largest RMS voltage"""
        widget = self.prediction_widget.excitation_voltage_list
        index = np.argmax([float(widget.item(v).text()) for v in range(widget.count())])
        self.prediction_widget.excitation_row_selector.setCurrentIndex(index)
        self.prediction_widget.excitation_column_selector.setCurrentIndex(index)

    def show_min_voltage_prediction(self):
        """Shows the prediction with the smallest RMS voltage"""
        widget = self.prediction_widget.excitation_voltage_list
        index = np.argmin([float(widget.item(v).text()) for v in range(widget.count())])
        self.prediction_widget.excitation_row_selector.setCurrentIndex(index)
        self.prediction_widget.excitation_column_selector.setCurrentIndex(index)

    def show_max_error_prediction(self):
        """Shows the prediction with the largest error"""
        widget = self.prediction_widget.response_error_list
        index = np.argmax([float(widget.item(v).text()) for v in range(widget.count())])
        self.prediction_widget.response_row_selector.setCurrentIndex(index)
        self.prediction_widget.response_column_selector.setCurrentIndex(index)

    def show_min_error_prediction(self):
        """Shows the prediction with the smallest error"""
        widget = self.prediction_widget.response_error_list
        index = np.argmin([float(widget.item(v).text()) for v in range(widget.count())])
        self.prediction_widget.response_row_selector.setCurrentIndex(index)
        self.prediction_widget.response_column_selector.setCurrentIndex(index)

    def update_response_error_prediction_selector(self, item):
        """Updates the selection when an item is double-clicked"""
        index = self.prediction_widget.response_error_list.row(item)
        self.prediction_widget.response_row_selector.setCurrentIndex(index)
        self.prediction_widget.response_column_selector.setCurrentIndex(index)

    def update_excitation_prediction_selector(self, item):
        """Updates the selection when an item is double-clicked"""
        index = self.prediction_widget.excitation_voltage_list.row(item)
        self.prediction_widget.excitation_row_selector.setCurrentIndex(index)
        self.prediction_widget.excitation_column_selector.setCurrentIndex(index)

    def update_control_predictions(self):
        """Updates the control prediction with new data"""
        excite_row_index = self.prediction_widget.excitation_row_selector.currentIndex()
        excite_column_index = (
            self.prediction_widget.excitation_column_selector.currentIndex()
        )
        self.plot_data_items["excitation_prediction"][0].setData(
            self.sysid_data.frequencies,
            np.abs(
                np.real(
                    self.excitation_prediction[:, excite_row_index, excite_column_index]
                )
            ),
        )
        row_index = self.prediction_widget.response_row_selector.currentIndex()
        column_index = self.prediction_widget.response_column_selector.currentIndex()
        self.plot_data_items["response_prediction"][0].setData(
            self.sysid_data.frequencies,
            np.abs(np.real(self.response_prediction[:, row_index, column_index])),
        )
        if row_index == column_index:
            warning_upper = abs(
                self.environment_metadata.specification_warning_matrix[1, :, row_index]
            )
            warning_lower = abs(
                self.environment_metadata.specification_warning_matrix[0, :, row_index]
            )
            abort_upper = abs(
                self.environment_metadata.specification_abort_matrix[1, :, row_index]
            )
            abort_lower = abs(
                self.environment_metadata.specification_abort_matrix[0, :, row_index]
            )
            self.plot_data_items["prediction_warning_upper"].setData(
                self.specification_frequency_lines, warning_upper
            )
            self.plot_data_items["prediction_warning_lower"].setData(
                self.specification_frequency_lines, warning_lower
            )
            self.plot_data_items["prediction_abort_upper"].setData(
                self.specification_frequency_lines, abort_upper
            )
            self.plot_data_items["prediction_abort_lower"].setData(
                self.specification_frequency_lines, abort_lower
            )
            self.plot_data_items["excitation_prediction"][1].setData(
                self.sysid_data.frequencies, np.zeros(self.sysid_data.frequencies.shape)
            )
            self.plot_data_items["response_prediction"][2].setData(
                self.sysid_data.frequencies, np.zeros(self.sysid_data.frequencies.shape)
            )
            self.plot_data_items["response_prediction"][3].setData(
                self.sysid_data.frequencies, np.zeros(self.sysid_data.frequencies.shape)
            )
        else:
            self.plot_data_items["prediction_warning_upper"].setData(
                np.array([0, self.definition_widget.sample_rate_display.value() / 2]),
                np.zeros(2),
            )
            self.plot_data_items["prediction_warning_lower"].setData(
                np.array([0, self.definition_widget.sample_rate_display.value() / 2]),
                np.zeros(2),
            )
            self.plot_data_items["prediction_abort_upper"].setData(
                np.array([0, self.definition_widget.sample_rate_display.value() / 2]),
                np.zeros(2),
            )
            self.plot_data_items["prediction_abort_lower"].setData(
                np.array([0, self.definition_widget.sample_rate_display.value() / 2]),
                np.zeros(2),
            )
            self.plot_data_items["excitation_prediction"][1].setData(
                self.sysid_data.frequencies,
                np.abs(
                    np.imag(
                        self.excitation_prediction[
                            :, excite_row_index, excite_column_index
                        ]
                    )
                ),
            )
            self.plot_data_items["response_prediction"][2].setData(
                self.sysid_data.frequencies,
                np.abs(np.imag(self.response_prediction[:, row_index, column_index])),
            )
            self.plot_data_items["response_prediction"][3].setData(
                self.sysid_data.frequencies,
                np.abs(
                    np.imag(
                        self.environment_metadata.specification_cpsd_matrix[
                            :, row_index, column_index
                        ]
                    )
                ),
            )
        self.plot_data_items["response_prediction"][1].setData(
            self.sysid_data.frequencies,
            np.abs(
                np.real(
                    self.environment_metadata.specification_cpsd_matrix[
                        :, row_index, column_index
                    ]
                )
            ),
        )

    def recompute_prediction(self):
        """Sends a message to the environment process to recompute the prediction"""
        self.rattlesnake.send_environment_command(
            self.environment_name, RandomVibrationCommands.RECOMPUTE_PREDICTION, None
        )

    # endregion

    # region Run
    def enable_control(self, enabled):
        """Enables or disables widgets to start or stop control if the control is running or not"""
        for widget in [
            self.run_widget.test_time_selector,
            self.run_widget.time_test_at_target_level_checkbox,
            self.run_widget.timed_test_radiobutton,
            self.run_widget.continuous_test_radiobutton,
            self.run_widget.target_test_level_selector,
            self.run_widget.start_test_button,
        ]:
            widget.setEnabled(enabled)
        for widget in [self.run_widget.stop_test_button]:
            widget.setEnabled(not enabled)
        if enabled:
            self.run_timer.stop()

    def update_run_time(self):
        """Updates the time that the control has been running on the GUI"""
        # Update the total run time
        current_time = time.time()
        time_elapsed = current_time - self.run_start_time
        time_at_level_elapsed = current_time - self.run_level_start_time
        self.run_widget.total_test_time_display.setText(
            str(datetime.timedelta(seconds=time_elapsed)).split(".", maxsplit=1)[0]
        )
        self.run_widget.time_at_level_display.setText(
            str(datetime.timedelta(seconds=time_at_level_elapsed)).split(
                ".", maxsplit=1
            )[0]
        )
        # Check if we need to stop the test due to timeout
        if self.run_widget.timed_test_radiobutton.isChecked():
            check_time = self.run_widget.test_time_selector.time()
            check_time_seconds = (
                check_time.hour() * 3600
                + check_time.minute() * 60
                + check_time.second()
            )
            if self.run_widget.time_test_at_target_level_checkbox.isChecked():
                if (
                    self.run_widget.current_test_level_selector.value()
                    >= self.run_widget.target_test_level_selector.value()
                ):
                    self.run_widget.test_progress_bar.setValue(
                        int(time_at_level_elapsed / check_time_seconds * 100)
                    )
                    if time_at_level_elapsed > check_time_seconds:
                        self.run_widget.test_progress_bar.setValue(100)
                        self.stop_environment()
                else:
                    self.run_widget.test_progress_bar.setValue(0)
            else:
                self.run_widget.test_progress_bar.setValue(
                    int(time_elapsed / check_time_seconds * 100)
                )
                if time_elapsed > check_time_seconds:
                    self.stop_environment()

    def change_control_test_level(self):
        """Updates the test level of the control."""
        self.rattlesnake.send_environment_command(
            self.environment_name,
            RandomVibrationCommands.ADJUST_TEST_LEVEL,
            self.run_widget.current_test_level_selector.value(),
        )
        self.run_level_start_time = time.time()
        # Check and see if we need to start streaming data
        if (
            self.run_widget.current_test_level_selector.value()
            >= self.run_widget.target_test_level_selector.value()
        ):
            self.rattlesnake.environment_at_target_level(self.environment_name)

    def change_test_level_from_profile(self, test_level):
        """Sets the test level from a profile instruction

        Parameters
        ----------
        test_level :
            Value to set the test level to.
        """
        self.run_widget.current_test_level_selector.setValue(float(test_level))

    def show_magnitude_window(self, item):
        """Creates a window showing the magnitude of a signal when an item is double-clicked"""
        index = self.run_widget.test_response_error_list.row(item)
        self.create_window(None, index, index, 0)

    def create_window(
        self, event, row_index=None, column_index=None, datatype_index=None
    ):  # pylint: disable=unused-argument
        """Creates a subwindow to show a specific channel information

        Parameters
        ----------
        event :

        row_index :
            Row index in the CPSD matrix to display (Default value = None)
        column_index :
            Column index in the CPSD matrix to display (Default value = None)
        datatype_index :
            Data type to display (real,imag,mag,phase,etc) (Default value = None)

        """
        if row_index is None:
            row_index = self.run_widget.control_channel_1_selector.currentIndex()
        if column_index is None:
            column_index = self.run_widget.control_channel_2_selector.currentIndex()
        if datatype_index is None:
            datatype_index = self.run_widget.data_type_selector.currentIndex()
        self.plot_windows.append(
            PlotWindow(
                None,
                row_index,
                column_index,
                datatype_index,
                (self.specification_frequency_lines, self.specification_cpsd_matrix),
                self.run_widget.control_channel_1_selector.itemText(row_index),
                self.run_widget.control_channel_2_selector.itemText(column_index),
                self.run_widget.data_type_selector.itemText(datatype_index),
                (
                    self.specification_warning_matrix
                    if row_index == column_index and datatype_index == 0
                    else None
                ),
                (
                    self.specification_abort_matrix
                    if row_index == column_index and datatype_index == 0
                    else None
                ),
            )
        )

    def show_all_asds(self):
        """Creates a subwindow for each ASD in the CPSD matrix"""
        for i in range(self.specification_cpsd_matrix.shape[-1]):
            self.create_window(None, i, i, 0)
        self.tile_windows()

    def show_all_csds_phscoh(self):
        """Creates a subwindow for each entry in the CPSD matrix showing phase and coherence"""
        for i in range(self.specification_cpsd_matrix.shape[-1]):
            for j in range(self.specification_cpsd_matrix.shape[-1]):
                if i == j:
                    datatype_index = 0
                elif i < j:
                    datatype_index = 1
                elif i > j:
                    datatype_index = 2
                else:
                    raise ValueError("Invalid situation.  How did you get here?!")
                self.create_window(None, i, j, datatype_index)
        self.tile_windows()

    def show_all_csds_realimag(self):
        """Creates a subwindow for each entry in the CPSD matrix showing real and imaginary"""
        for i in range(self.specification_cpsd_matrix.shape[-1]):
            for j in range(self.specification_cpsd_matrix.shape[-1]):
                if i == j:
                    datatype_index = 0
                elif i < j:
                    datatype_index = 3
                elif i > j:
                    datatype_index = 4
                else:
                    raise ValueError("Invalid situation.  How did you get here?!")
                self.create_window(None, i, j, datatype_index)
        self.tile_windows()

    def tile_windows(self):
        """Tile subwindow equally across the screen"""
        screen_rect = QtWidgets.QApplication.desktop().screenGeometry()
        # Go through and remove any closed windows
        self.plot_windows = [
            window for window in self.plot_windows if window.isVisible()
        ]
        num_windows = len(self.plot_windows)
        ncols = int(np.ceil(np.sqrt(num_windows)))
        nrows = int(np.ceil(num_windows / ncols))
        window_width = int(screen_rect.width() / ncols)
        window_height = int(screen_rect.height() / nrows)
        for index, window in enumerate(self.plot_windows):
            window.resize(window_width, window_height)
            row_ind = index // ncols
            col_ind = index % ncols
            window.move(col_ind * window_width, row_ind * window_height)

    def close_windows(self):
        """Close all subwindows"""
        for window in self.plot_windows:
            window.close()

    def save_spectral_data(self, clicked, filename=None):
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self.definition_widget,
                "Select File to Save Spectral Data",
                filter="NetCDF File (*.nc4)",
            )
        if filename == "":
            return

        self.rattlesnake.send_environment_command(
            self.environment_name, RandomVibrationCommands.SAVE_CONTROL_DATA, filename
        )

    # endregion

    # region Commands
    def display_environment_ended(self):
        self.enable_control(True)

    def display_environment_started(self):
        self.enable_control(False)

    def start_environment(self):
        """Sets itself up to start controlling and sends a signal to the environment to start"""
        for widget in [
            self.run_widget.test_time_selector,
            self.run_widget.time_test_at_target_level_checkbox,
            self.run_widget.timed_test_radiobutton,
            self.run_widget.continuous_test_radiobutton,
            self.run_widget.target_test_level_selector,
            self.run_widget.start_test_button,
        ]:
            widget.setEnabled(False)
        return super().start_environment()

    def start_environment_ready(self):
        self.run_timer.start(250)
        self.run_start_time = time.time()
        self.run_level_start_time = self.run_start_time
        self.run_widget.test_progress_bar.setValue(0)
        return super().start_environment_ready()

    def start_environment_error(self, error):
        return super().start_environment_error(error)

    def stop_environment(self):
        """Sends a signal to shut down the control"""
        self.run_widget.stop_test_button.setEnabled(True)

        super().stop_environment()

    def stop_environment_error(self, error):
        return super().stop_environment_error(error)

    def stop_environment_ready(self):
        return super().stop_environment_ready()

    def update_gui(self, queue_data: tuple):
        """Update the environment's graphical user interface

        This function will receive data from the gui_update_queue that
        specifies how the user interface should be updated.  Data will usually
        be received as ``(instruction,data)`` pairs, where the ``instruction`` notes
        what operation should be taken or which widget should be modified, and
        the ``data`` notes what data should be used in the update.

        Parameters
        ----------
        queue_data : tuple
            A tuple containing ``(instruction,data)`` pairs where ``instruction``
            defines and operation or widget to be modified and ``data`` contains
            the data used to perform the operation.
        """
        if super().update_gui(queue_data):
            return

        command, data = queue_data
        match command:
            case RandomVibrationDataAnalysisUICommands.CONTROL_PREDICTIONS:
                (
                    _,
                    self.excitation_prediction,
                    self.response_prediction,
                    _,
                    rms_voltage_prediction,
                    rms_db_error_prediction,
                ) = data
                self.update_control_predictions()
                for widget, widget_data in zip(
                    [
                        self.prediction_widget.excitation_voltage_list,
                        self.prediction_widget.response_error_list,
                    ],
                    [rms_voltage_prediction, rms_db_error_prediction],
                ):
                    widget.clear()
                    widget.addItems([f"{d:.3f}" for d in widget_data])
                # Now compute if any channels are erroring or not
                with np.errstate(invalid="ignore"):
                    lines_out = (
                        self.environment_metadata.percent_lines_out / 100
                    ) * self.environment_metadata.fft_lines
                    for i in range(self.prediction_widget.response_error_list.count()):
                        item = self.prediction_widget.response_error_list.item(i)
                        if (
                            sum(
                                self.response_prediction[:, i, i]
                                > self.environment_metadata.specification_abort_matrix[
                                    1, :, i
                                ]
                            )
                            > lines_out
                        ):
                            item.setBackground(QColor(255, 125, 125))
                        elif (
                            sum(
                                self.response_prediction[:, i, i]
                                < self.environment_metadata.specification_abort_matrix[
                                    0, :, i
                                ]
                            )
                            > lines_out
                        ):
                            item.setBackground(QColor(255, 125, 125))
                        elif (
                            sum(
                                self.response_prediction[:, i, i]
                                > self.environment_metadata.specification_warning_matrix[
                                    1, :, i
                                ]
                            )
                            > lines_out
                        ):
                            item.setBackground(QColor(255, 255, 125))
                        elif (
                            sum(
                                self.response_prediction[:, i, i]
                                < self.environment_metadata.specification_warning_matrix[
                                    0, :, i
                                ]
                            )
                            > lines_out
                        ):
                            item.setBackground(QColor(255, 255, 125))
                        else:
                            item.setBackground(QColor(255, 255, 255))
            case RandomVibrationDataAnalysisUICommands.CONTROL_UPDATE:
                (
                    frames,
                    total_frames,
                    self.sysid_data.frequencies,
                    self.sysid_data.sysid_frf,
                    self.sysid_data.sysid_coherence,
                    self.sysid_data.sysid_response_cpsd,
                    self.sysid_data.sysid_reference_cpsd,
                    self.sysid_data.sysid_condition,
                ) = data
                self.update_sysid_plots(
                    update_time=False, update_transfer_function=True, update_noise=True
                )
                self.system_id_widget.current_frames_spinbox.setValue(frames)
                self.system_id_widget.total_frames_spinbox.setValue(total_frames)
                self.system_id_widget.progressBar.setValue(
                    int(frames / total_frames * 100)
                )
                self.plot_data_items["sum_asds_control"].setData(
                    self.sysid_data.frequencies,
                    np.einsum("ijj", self.sysid_data.sysid_response_cpsd).real,
                )
                # Go through and remove any closed windows
                self.plot_windows = [
                    window for window in self.plot_windows if window.isVisible()
                ]
                for window in self.plot_windows:
                    window.update_plot(self.sysid_data.sysid_response_cpsd)
            case RandomVibrationDataAnalysisUICommands.INTERACTIVE_CONTROL_SYSID_UPDATE:
                if self.interactive_control_law_widget is not None:
                    self.interactive_control_law_widget.update_ui_sysid(*data)
            case ControlLawUICommands.INTERACTIVE_CONTROL_UPDATE:
                if self.interactive_control_law_widget is not None:
                    self.interactive_control_law_widget.update_ui_control(data)
            case RandomVibrationDataAnalysisUICommands.UPDATE_TEST_RESPONSE_ERROR_LIST:
                rms_db_error, warning_channels, abort_channels = data
                self.run_widget.test_response_error_list.clear()
                self.run_widget.test_response_error_list.addItems(
                    [f"{d:.3f}" for d in rms_db_error]
                )
                for index in warning_channels:
                    item = self.run_widget.test_response_error_list.item(index)
                    item.setBackground(QColor(255, 255, 125))
                for index in abort_channels:
                    item = self.run_widget.test_response_error_list.item(index)
                    item.setBackground(QColor(255, 125, 125))
            case RandomVibrationUICommands.ENABLE_CONTROL:
                self.enable_control(True)
            case RandomVibrationUICommands.CHANGE_SPECIFICATION:
                filename, environment_metadata = data
                self.select_spec_file(None, filename)
                self.initialize_environment(environment_metadata)
            case RandomVibrationUICommands.ADJUST_TEST_LEVEL:
                self.run_widget.current_test_level_selector.blockSignals(True)
                self.run_widget.current_test_level_selector.setValue(data)
                self.run_widget.current_test_level_selector.blockSignals(False)
            case RandomVibrationCommands.ADJUST_TEST_LEVEL:
                self.run_widget.current_test_level_selector.blockSignals(True)
                self.run_widget.current_test_level_selector.setValue(data)
                self.run_widget.current_test_level_selector.blockSignals(False)
            case RandomVibrationCommands.SAVE_CONTROL_DATA:
                pass
            case RandomVibrationCommands.CHANGE_SPECIFICATION:
                pass
            case UICommands.ENABLE:
                widget = None
                for parent in [
                    self.definition_widget,
                    self.system_id_widget,
                    self.prediction_widget,
                    self.run_widget,
                ]:
                    try:
                        widget = getattr(parent, data)
                        break
                    except AttributeError:
                        continue
                if widget is None:
                    raise ValueError(f"Cannot Enable Widget {data}: not found in UI")
                widget.setEnabled(True)
            case UICommands.DISABLE:
                widget = None
                for parent in [
                    self.definition_widget,
                    self.system_id_widget,
                    self.prediction_widget,
                    self.run_widget,
                ]:
                    try:
                        widget = getattr(parent, data)
                        break
                    except AttributeError:
                        continue
                if widget is None:
                    raise ValueError(f"Cannot Disable Widget {data}: not found in UI")
                widget.setEnabled(False)
            case _:
                print(f"Unknown Random UI Command {command}")
