import inspect
import os
from multiprocessing.queues import Queue

import netCDF4 as nc4
import numpy as np
from qtpy import QtCore, QtWidgets, uic
from qtpy.QtCore import Qt

from rattlesnake.engine import RattlesnakeController
from rattlesnake.utilities import (
    DIRECTORY,
    GlobalCommands,
    db2scale,
    load_python_module,
    rms_time,
    load_time_history,
)
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.transient_sys_id_environment import (
    TransientCommands,
    TransientUICommands,
    TransientMetadata,
    TransientInstructions,
)
from rattlesnake.environment.abstract_interactive_control_law import (  # noqa: E402 pylint: disable=wrong-import-position
    AbstractControlLawComputation,
    ControlLawUICommands,
)

from rattlesnake.user_interface.abstract_sys_id_user_interface import SysIdEnvironmentUI
from rattlesnake.user_interface.ui_utilities import (
    UICommands,
    PlotTimeWindow,
    TransformationMatrixWindow,
    axis_label,
    channel_unit_label,
    colororder,
    multiline_plotter,
)

CONTROL_TYPE = EnvironmentType.TRANSIENT
MAXIMUM_NAME_LENGTH = 50


# region User Interface
class TransientUI(SysIdEnvironmentUI):
    """Class defining the user interface for the transient environment"""

    def __init__(
        self,
        environment_name: str,
        rattlesnake: RattlesnakeController,
    ):
        super().__init__(CONTROL_TYPE, environment_name, rattlesnake)
        # Add the page to the control definition tabwidget
        self.definition_widget = QtWidgets.QWidget()
        transient_definition_ui_path = os.path.join(
            DIRECTORY, "user_interface", "ui_files", "transient_definition.ui"
        )
        uic.loadUi(transient_definition_ui_path, self.definition_widget)
        # Add the page to the control prediction tabwidget
        self.prediction_widget = QtWidgets.QWidget()
        transient_prediction_ui_path = os.path.join(
            DIRECTORY, "user_interface", "ui_files", "transient_prediction.ui"
        )
        uic.loadUi(transient_prediction_ui_path, self.prediction_widget)
        # Add the page to the run tabwidget
        self.run_widget = QtWidgets.QWidget()
        transient_run_ui_path = os.path.join(
            DIRECTORY, "user_interface", "ui_files", "transient_run.ui"
        )
        uic.loadUi(transient_run_ui_path, self.run_widget)

        self.specification_signal = None
        self.show_signal_checkboxes = None
        self.plot_data_items = {}
        self.plot_windows = []
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.python_control_module = None
        self.physical_channel_names = None
        self.physical_output_indices = None
        self.excitation_prediction = None
        self.response_prediction = None
        self.last_control_data = None
        self.last_output_data = None
        self.interactive_control_law_widget = None
        self.interactive_control_law_window = None
        self.max_plot_samples = None

        self.control_selector_widgets = [
            self.prediction_widget.response_selector,
            self.run_widget.control_channel_selector,
        ]
        self.output_selector_widgets = [
            self.prediction_widget.excitation_selector,
        ]

        # Set common look and feel for plots
        plot_widgets = [
            self.definition_widget.signal_display_plot,
            self.prediction_widget.excitation_display_plot,
            self.prediction_widget.response_display_plot,
            self.run_widget.output_signal_plot,
            self.run_widget.response_signal_plot,
        ]
        for plot_widget in plot_widgets:
            plot_item = plot_widget.getPlotItem()
            plot_item.showGrid(True, True, 0.25)
            plot_item.enableAutoRange()
            plot_item.getViewBox().enableAutoRange(enable=True)
            plot_item.setLabel("bottom", "Time (s)")
            plot_item.setLabel("left", "Amplitude")

        self.connect_callbacks()

        # Complete the profile commands

    def connect_callbacks(self):
        """Connects the callbacks to the transient UI widgets"""
        # Definition
        self.definition_widget.load_signal_button.clicked.connect(self.load_signal)
        self.definition_widget.transformation_matrices_button.clicked.connect(
            self.define_transformation_matrices
        )
        self.definition_widget.show_all_button.clicked.connect(self.show_all_signals)
        self.definition_widget.show_none_button.clicked.connect(self.show_no_signals)
        self.definition_widget.control_channels_selector.itemChanged.connect(
            self.update_control_channels
        )
        self.definition_widget.control_script_load_file_button.clicked.connect(
            self.select_python_module
        )
        self.definition_widget.control_function_input.currentIndexChanged.connect(
            self.update_generator_selector
        )
        self.definition_widget.check_selected_button.clicked.connect(
            self.check_selected_control_channels
        )
        self.definition_widget.uncheck_selected_button.clicked.connect(
            self.uncheck_selected_control_channels
        )
        # Prediction
        self.prediction_widget.excitation_selector.currentIndexChanged.connect(
            self.plot_predictions
        )
        self.prediction_widget.response_selector.currentIndexChanged.connect(
            self.plot_predictions
        )
        self.prediction_widget.response_error_list.itemClicked.connect(
            self.update_response_error_prediction_selector
        )
        self.prediction_widget.excitation_voltage_list.itemClicked.connect(
            self.update_excitation_prediction_selector
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
        self.prediction_widget.recompute_predictions_button.clicked.connect(
            self.recompute_predictions
        )
        # Run Test
        self.run_widget.start_test_button.clicked.connect(self.start_environment)
        self.run_widget.stop_test_button.clicked.connect(self.stop_environment)
        self.run_widget.create_window_button.clicked.connect(self.create_window)
        self.run_widget.show_all_channels_button.clicked.connect(self.show_all_channels)
        self.run_widget.tile_windows_button.clicked.connect(self.tile_windows)
        self.run_widget.close_windows_button.clicked.connect(self.close_windows)
        self.run_widget.control_response_error_list.itemDoubleClicked.connect(
            self.show_window
        )
        self.run_widget.save_current_control_data_button.clicked.connect(
            self.save_control_data
        )
        self.run_widget.display_duration_spinbox.valueChanged.connect(
            self.set_display_duration
        )

    @property
    def physical_output_names(self):
        """Names of the physical drive channels"""
        return [self.physical_channel_names[i] for i in self.physical_output_indices]

    @property
    def physical_control_indices(self):
        """Indices of the control channels"""
        return [
            i
            for i in range(self.definition_widget.control_channels_selector.count())
            if self.definition_widget.control_channels_selector.item(i).checkState()
            == Qt.Checked
        ]

    @property
    def physical_control_names(self):
        """Names of the selected control channels"""
        return [self.physical_channel_names[i] for i in self.physical_control_indices]

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
                for i in range(
                    self.environment_metadata.response_transformation_matrix.shape[0]
                )
            ]

    @property
    def initialized_output_names(self):
        """Names of the drive channels that have been initialized"""
        if self.environment_metadata.reference_transformation_matrix is None:
            return self.physical_output_names
        else:
            return [
                f"Transformed Drive {i + 1}"
                for i in range(
                    self.environment_metadata.reference_transformation_matrix.shape[0]
                )
            ]

    def initialized_control_unit(self, control_index):
        """Engineering unit of an initialized control channel, or None if unknown

        Returns None for transformed control channels, since those don't
        correspond to a single physical channel with a single unit.
        """
        if self.environment_metadata.response_transformation_matrix is not None:
            return None
        try:
            channel_index = self.environment_metadata.control_channel_indices[
                control_index
            ]
            return self.hardware_metadata.channel_list[channel_index].unit
        except (IndexError, TypeError):
            return None

    def initialized_output_unit(self, output_index):
        """Engineering unit of an initialized drive channel, or None if unknown

        Returns None for transformed drive channels, since those don't
        correspond to a single physical channel with a single unit.
        """
        if self.environment_metadata.reference_transformation_matrix is not None:
            return None
        try:
            return self.hardware_metadata.channel_list[
                self.physical_output_indices[output_index]
            ].unit
        except (IndexError, TypeError):
            return None

    # region State Sync
    def initialize_hardware(self, data_acquisition_parameters):
        super().initialize_hardware(data_acquisition_parameters)
        # Initialize the plots
        for plot in [
            self.definition_widget.signal_display_plot,
            self.prediction_widget.excitation_display_plot,
            self.prediction_widget.response_display_plot,
            self.run_widget.output_signal_plot,
            self.run_widget.response_signal_plot,
        ]:
            plot.getPlotItem().clear()

        # Set up channel names
        self.physical_channel_names = [
            (
                f"{'' if channel.channel_type is None else channel.channel_type} "
                f"{channel.node_number} "
                f"{'' if channel.node_direction is None else channel.node_direction}"
            )[:MAXIMUM_NAME_LENGTH]
            for channel in data_acquisition_parameters.channel_list
        ]
        self.physical_output_indices = [
            i
            for i, channel in enumerate(data_acquisition_parameters.channel_list)
            if channel.feedback_device
        ]
        # Set up widgets
        self.definition_widget.sample_rate_display.setValue(
            data_acquisition_parameters.sample_rate
        )
        self.system_id_widget.samplesPerFrameSpinBox.setValue(
            data_acquisition_parameters.sample_rate
        )
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
        self.definition_widget.input_channels_display.setValue(
            len(self.physical_channel_names)
        )
        self.definition_widget.output_channels_display.setValue(
            len(self.physical_output_indices)
        )
        self.definition_widget.control_channels_display.setValue(0)

        if not self.definition_widget.control_script_file_path_input.text():
            control_law_path = os.path.join(
                DIRECTORY, "examples", "control_laws", "transient_control_laws.py"
            )
            self.select_python_module(clicked=False, filename=control_law_path)

    def initialize_environment(self, environment_metadata):
        super().initialize_environment(environment_metadata)
        # Make sure everything is defined
        if self.environment_metadata.control_signal is None:
            raise ValueError(
                f"Control Signal is not defined for {self.environment_name}!"
            )
        if self.environment_metadata.control_python_script is None:
            raise ValueError(
                f"Control function has not been loaded for {self.environment_name}"
            )
        self.system_id_widget.samplesPerFrameSpinBox.setMaximum(
            self.environment_metadata.control_signal.shape[-1]
        )
        for widget in [
            self.prediction_widget.response_selector,
            self.run_widget.control_channel_selector,
        ]:
            widget.blockSignals(True)
            widget.clear()
            for i, control_name in enumerate(self.initialized_control_names):
                widget.addItem(f"{i + 1}: {control_name}")
            widget.blockSignals(False)
        for widget in [self.prediction_widget.excitation_selector]:
            widget.blockSignals(True)
            widget.clear()
            for i, drive_name in enumerate(self.initialized_output_names):
                widget.addItem(f"{i + 1}: {drive_name}")
            widget.blockSignals(False)
        # Set up the prediction plots
        self.prediction_widget.excitation_display_plot.getPlotItem().clear()
        self.prediction_widget.response_display_plot.getPlotItem().clear()
        self.plot_data_items["response_prediction"] = multiline_plotter(
            np.arange(self.environment_metadata.control_signal.shape[-1])
            / self.environment_metadata.sample_rate,
            np.zeros((2, self.environment_metadata.control_signal.shape[-1])),
            widget=self.prediction_widget.response_display_plot,
            other_pen_options={"width": 1},
            names=["Prediction", "Spec"],
            downsample={"auto": True},
            clip_to_view=True,
        )
        self.plot_data_items["excitation_prediction"] = multiline_plotter(
            np.arange(self.environment_metadata.control_signal.shape[-1])
            / self.environment_metadata.sample_rate,
            np.zeros((1, self.environment_metadata.control_signal.shape[-1])),
            widget=self.prediction_widget.excitation_display_plot,
            other_pen_options={"width": 1},
            names=["Prediction"],
            downsample={"auto": True},
            clip_to_view=True,
        )
        # Set up the run plots
        self.run_widget.output_signal_plot.getPlotItem().clear()
        self.run_widget.response_signal_plot.getPlotItem().clear()
        self.max_plot_samples = (
            self.hardware_metadata.sample_rate
            * self.run_widget.display_duration_spinbox.value()
        )
        self.plot_data_items["output_signal_measurement"] = multiline_plotter(
            (np.array([])),
            np.zeros((len(self.initialized_control_names), 0)),
            widget=self.run_widget.output_signal_plot,
            other_pen_options={"width": 1},
            names=self.initialized_control_names,
            downsample={"auto": True},
            clip_to_view=True,
        )
        self.plot_data_items[
            "signal_range"
        ] = self.run_widget.response_signal_plot.getPlotItem().plot(
            np.zeros(5),
            np.zeros(5),
            pen={"color": "k", "width": 1},
            name="Signal Lower Bound",
        )
        self.plot_data_items["control_signal_measurement"] = multiline_plotter(
            (np.array([])),
            np.zeros((len(self.initialized_output_names), 0)),
            widget=self.run_widget.response_signal_plot,
            other_pen_options={"width": 1},
            names=self.initialized_output_names,
            downsample={"auto": True},
            clip_to_view=True,
        )
        control_unit = (
            channel_unit_label(
                self.hardware_metadata.channel_list[i]
                for i in self.environment_metadata.control_channel_indices
            )
            if self.environment_metadata.response_transformation_matrix is None
            else None
        )
        output_unit = channel_unit_label(
            self.hardware_metadata.channel_list[i] for i in self.physical_output_indices
        )
        self.run_widget.output_signal_plot.getPlotItem().setLabel(
            "left", axis_label("amplitude", "Response", control_unit)
        )
        self.run_widget.response_signal_plot.getPlotItem().setLabel(
            "left", axis_label("amplitude", "Drive", output_unit)
        )
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
        return self.environment_metadata

    def get_environment_metadata(self, global_channel_list):
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
        metadata = TransientMetadata(
            environment_name=self.environment_name,
            channel_list_bools=channel_list_bools,
            sample_rate=self.definition_widget.sample_rate_display.value(),
            number_of_channels=len(self.hardware_metadata.channel_list),
            control_signal=self.specification_signal,
            ramp_time=self.definition_widget.ramp_selector.value(),
            control_python_script=control_module,
            control_python_function=control_function,
            control_python_function_type=control_function_type,
            control_python_function_parameters=control_function_parameters,
            control_channel_indices=self.physical_control_indices,
            output_channel_indices=self.physical_output_indices,
            response_transformation_matrix=self.response_transformation_matrix,
            output_transformation_matrix=self.output_transformation_matrix,
        )

        signal_file = self.definition_widget.signal_file_name_display.text()
        if signal_file:
            metadata.set_file(signal_file)

        return metadata

    def set_environment_metadata(self, metadata):
        """Sets the UI widgets and internal state from a metadata object."""

        # Basic numeric and UI values
        self.definition_widget.sample_rate_display.setValue(metadata.sample_rate)
        self.definition_widget.ramp_selector.setValue(metadata.test_level_ramp_time)

        # Python Control Module Logic
        if metadata.control_python_script:
            # Assuming select_python_module triggers the logic to load functions into the UI
            self.select_python_module(None, metadata.control_python_script)

            # Set the function dropdown
            func_index = self.definition_widget.control_function_input.findText(
                metadata.control_python_function
            )
            if func_index != -1:
                self.definition_widget.control_function_input.setCurrentIndex(
                    func_index
                )

            # Set the generator type selector
            self.definition_widget.control_function_generator_selector.setCurrentIndex(
                metadata.control_python_function_type
            )

            # Set the parameters text
            self.definition_widget.control_parameters_text_input.setPlainText(
                ""
                if metadata.control_python_function_parameters is None
                else str(metadata.control_python_function_parameters)
            )

        # Control Channel Selection (ListWidget/Selector)
        # First, clear all existing checks
        for i in range(self.definition_widget.control_channels_selector.count()):
            self.definition_widget.control_channels_selector.item(i).setCheckState(
                Qt.Unchecked
            )

        # Check the indices provided in metadata
        for control_channel in metadata.control_channel_indices:
            item = self.definition_widget.control_channels_selector.item(
                control_channel
            )
            if item:
                item.setCheckState(Qt.Checked)

        self.specification_signal = metadata.control_signal
        self.definition_widget.signal_file_name_display.setText(
            metadata.spec_filename or ""
        )
        self.setup_specification_table()
        self.show_signal()

    def get_environment_instructions(self):
        test_level = self.run_widget.test_level_selector.value()
        repeat = self.run_widget.repeat_signal_checkbox.isChecked()
        instruction = TransientInstructions(self.environment_name, test_level, repeat)

        return instruction

    def set_environment_instructions(self, instructions):
        self.run_widget.test_level_selector.setValue(instructions.test_level)
        self.run_widget.repeat_signal_checkbox.setChecked(instructions.repeat)
        super().set_environment_instructions(instructions)

    # endregion

    # region Definition
    def update_control_channels(self):
        """Callback called when control channels are updated in the UI"""
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.specification_signal = None
        self.definition_widget.control_channels_display.setValue(
            len(self.physical_control_indices)
        )
        self.define_transformation_matrices(None, False)
        self.show_signal()

    def load_signal(self, clicked, filename=None):  # pylint: disable=unused-argument
        """Loads a time signal using a dialog or the specified filename

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
                "Select Signal File",
                filter="Numpy or Mat (*.npy *.npz *.mat)",
            )
            if filename == "":
                return
        self.definition_widget.signal_file_name_display.setText(filename)
        self.specification_signal = load_time_history(
            filename, self.definition_widget.sample_rate_display.value()
        )
        self.setup_specification_table()
        self.show_signal()

    def setup_specification_table(self):
        """Sets up the specification table for the Transient Environment

        This function computes the RMS and max values for the signals and then
        creates entries in the table for each signal"""
        self.definition_widget.signal_samples_display.setValue(
            self.specification_signal.shape[-1]
        )
        self.definition_widget.signal_time_display.setValue(
            self.specification_signal.shape[-1]
            / self.definition_widget.sample_rate_display.value()
        )
        maxs = np.max(np.abs(self.specification_signal), axis=-1)
        rmss = rms_time(self.specification_signal, axis=-1)
        # Add rows to the signal table
        self.definition_widget.signal_information_table.setRowCount(
            self.specification_signal.shape[0]
        )
        self.show_signal_checkboxes = []
        for i, (name, mx, rms) in enumerate(
            zip(self.physical_control_names, maxs, rmss)
        ):
            item = QtWidgets.QTableWidgetItem()
            item.setText(name)
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.definition_widget.signal_information_table.setItem(i, 1, item)
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.show_signal)
            self.show_signal_checkboxes.append(checkbox)
            self.definition_widget.signal_information_table.setCellWidget(
                i, 0, checkbox
            )
            item = QtWidgets.QTableWidgetItem()
            item.setText(f"{mx:0.2f}")
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.definition_widget.signal_information_table.setItem(i, 2, item)
            item = QtWidgets.QTableWidgetItem()
            item.setText(f"{rms:0.2f}")
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.definition_widget.signal_information_table.setItem(i, 3, item)

    def show_signal(self):
        """Shows the signal on the user interface"""
        pi = self.definition_widget.signal_display_plot.getPlotItem()
        pi.clear()
        control_unit = channel_unit_label(
            self.hardware_metadata.channel_list[i]
            for i in self.physical_control_indices
        )
        pi.setLabel("left", axis_label("amplitude", "Amplitude", control_unit))
        if self.specification_signal is None:
            self.definition_widget.signal_information_table.setRowCount(0)
            return
        abscissa = (
            np.arange(self.specification_signal.shape[-1])
            / self.definition_widget.sample_rate_display.value()
        )
        for i, (curve, checkbox) in enumerate(
            zip(self.specification_signal, self.show_signal_checkboxes)
        ):
            pen = {"color": colororder[i % len(colororder)]}
            if checkbox.isChecked():
                pi.plot(abscissa, curve, pen=pen)
            else:
                pi.plot((0, 0), (0, 0), pen=pen)

    def show_all_signals(self):
        """Callback to show all signals in the specification"""
        # print('Showing All Signals')
        for checkbox in self.show_signal_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
        self.show_signal()

    def show_no_signals(self):
        """Callback to hide all signals in the specification"""
        # print('Showing No Signals')
        for checkbox in self.show_signal_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        self.show_signal()

    def define_transformation_matrices(  # pylint: disable=unused-argument
        self, clicked, dialog=True
    ):
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
            # Clear the signals
            self.definition_widget.signal_information_table.clear()
            self.definition_widget.signal_display_plot.clear()
            self.definition_widget.signal_file_name_display.clear()
            self.definition_widget.signal_information_table.setRowCount(0)
            self.show_signal_checkboxes = None
            self.response_transformation_matrix = response_transformation
            self.output_transformation_matrix = output_transformation

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
        if filename is None or not os.path.isfile(filename):
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.definition_widget,
                "Select Python Module",
                filter="Python Modules (*.py)",
            )
            if filename == "":
                return
        try:
            self.python_control_module = load_python_module(filename)
            functions = [
                function
                for function in inspect.getmembers(self.python_control_module)
                if (
                    inspect.isfunction(function[1])
                    and len(inspect.signature(function[1]).parameters) >= 6
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
                                        function[1].__dict__[method],
                                        "__isabstractmethod__",
                                    )
                                    and function[1]
                                    .__dict__[method]
                                    .__isabstractmethod__
                                )
                            )
                            for method in ["system_id_update", "control"]
                        ]
                    )
                )
            ]
        except Exception as e:
            self.display_error(e)
            return
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

    def check_selected_control_channels(self):
        """Callback to check control channels that are selected"""
        for item in self.definition_widget.control_channels_selector.selectedItems():
            item.setCheckState(Qt.Checked)

    def uncheck_selected_control_channels(self):
        """Callback to uncheck control channels that are selected"""
        for item in self.definition_widget.control_channels_selector.selectedItems():
            item.setCheckState(Qt.Unchecked)

    # endregion

    # region Predictions
    def plot_predictions(self):
        """Plots the control predictions based on the currently selected item"""
        times = (
            np.arange(self.specification_signal.shape[-1])
            / self.hardware_metadata.sample_rate
        )
        index = self.prediction_widget.excitation_selector.currentIndex()
        self.plot_data_items["excitation_prediction"][0].setData(
            times, self.excitation_prediction[index]
        )
        self.prediction_widget.excitation_display_plot.getPlotItem().setLabel(
            "left",
            axis_label("amplitude", "Amplitude", self.initialized_output_unit(index)),
        )
        index = self.prediction_widget.response_selector.currentIndex()
        self.plot_data_items["response_prediction"][0].setData(
            times, self.response_prediction[index]
        )
        self.plot_data_items["response_prediction"][1].setData(
            times, self.specification_signal[index]
        )
        self.prediction_widget.response_display_plot.getPlotItem().setLabel(
            "left",
            axis_label("amplitude", "Amplitude", self.initialized_control_unit(index)),
        )

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
        self.rattlesnake.send_environment_command(
            self.environment_name, (TransientCommands.PERFORM_CONTROL_PREDICTION, False)
        )

    # endregion

    # region Run
    def set_display_duration(self, value):
        """Updates the display duration in the UI"""
        self.max_plot_samples = int(self.hardware_metadata.sample_rate * value)

    def create_window(
        self, event, control_index=None
    ):  # pylint: disable=unused-argument
        """Creates a subwindow to show a specific channel information

        Parameters
        ----------
        event :

        control_index :
            Row index in the specification matrix to display (Default value = None)

        """
        if control_index is None:
            control_index = self.run_widget.control_channel_selector.currentIndex()
        self.plot_windows.append(
            PlotTimeWindow(
                None,
                control_index,
                self.environment_metadata.control_signal,
                self.hardware_metadata.sample_rate,
                self.run_widget.control_channel_selector.itemText(control_index),
                self.initialized_control_unit(control_index),
            )
        )
        if self.last_control_data is not None:
            self.plot_windows[-1].update_plot(self.last_control_data)

    def show_all_channels(self):
        """Creates a subwindow for each ASD in the CPSD matrix"""
        for i in range(self.environment_metadata.control_signal.shape[0]):
            self.create_window(None, i)
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

    def show_window(self, item):
        """Shows the currently selected control channel in a new subwindow"""
        index = self.run_widget.control_response_error_list.row(item)
        self.create_window(None, index)

    def close_windows(self):
        """Close all subwindows"""
        for window in self.plot_windows:
            window.close()

    def update_control_plots(self):
        """Updates plots in all of the existing subwindows"""
        # Go through and remove any closed windows
        self.plot_windows = [
            window for window in self.plot_windows if window.isVisible()
        ]
        for window in self.plot_windows:
            window.update_plot(self.last_control_data)

    def save_control_data(self):
        """Save Time-aligned Control Data from the Controller"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.definition_widget,
            "Select File to Save Spectral Data",
            filter="NetCDF File (*.nc4)",
        )
        if filename == "":
            return
        labels = [
            ["node_number", str],
            ["node_direction", str],
            ["comment", str],
            ["serial_number", str],
            ["triax_dof", str],
            ["sensitivity", str],
            ["unit", str],
            ["make", str],
            ["model", str],
            ["expiration", str],
            ["physical_device", str],
            ["physical_channel", str],
            ["channel_type", str],
            ["minimum_value", str],
            ["maximum_value", str],
            ["coupling", str],
            ["excitation_source", str],
            ["excitation", str],
            ["feedback_device", str],
            ["feedback_channel", str],
            ["warning_level", str],
            ["abort_level", str],
        ]
        global_data_parameters: HardwareMetadata
        global_data_parameters = self.hardware_metadata
        netcdf_handle = nc4.Dataset(  # pylint: disable=no-member
            filename, "w", format="NETCDF4", clobber=True
        )
        # Create dimensions
        netcdf_handle.createDimension(
            "response_channels", len(global_data_parameters.channel_list)
        )
        netcdf_handle.createDimension(
            "output_channels",
            len(
                [
                    channel
                    for channel in global_data_parameters.channel_list
                    if channel.feedback_device is not None
                ]
            ),
        )
        netcdf_handle.createDimension("time_samples", None)
        netcdf_handle.createDimension(
            "num_environments", len(global_data_parameters.environment_names)
        )
        # Create attributes
        netcdf_handle.file_version = "3.0.0"
        netcdf_handle.sample_rate = global_data_parameters.sample_rate
        netcdf_handle.time_per_write = (
            global_data_parameters.samples_per_write
            / global_data_parameters.output_sample_rate
        )
        netcdf_handle.time_per_read = (
            global_data_parameters.samples_per_read / global_data_parameters.sample_rate
        )
        netcdf_handle.hardware = global_data_parameters.hardware
        netcdf_handle.hardware_file = (
            "None"
            if global_data_parameters.hardware_file is None
            else global_data_parameters.hardware_file
        )
        netcdf_handle.output_oversample = global_data_parameters.output_oversample
        for key, value in global_data_parameters.extra_parameters.items():
            setattr(netcdf_handle, key, value)
        # Create Variables
        var = netcdf_handle.createVariable(
            "environment_names", str, ("num_environments",)
        )
        this_environment_index = None
        for i, name in enumerate(global_data_parameters.environment_names):
            var[i] = name
            if name == self.environment_name:
                this_environment_index = i
        var = netcdf_handle.createVariable(
            "environment_active_channels",
            "i1",
            ("response_channels", "num_environments"),
        )
        var[...] = global_data_parameters.environment_active_channels.astype("int8")[
            global_data_parameters.environment_active_channels[
                :, this_environment_index
            ],
            :,
        ]
        # Create channel table variables

        for label, netcdf_datatype in labels:
            var = netcdf_handle.createVariable(
                "/channels/" + label, netcdf_datatype, ("response_channels",)
            )
            channel_data = [
                getattr(channel, label)
                for channel in global_data_parameters.channel_list
            ]
            if netcdf_datatype == "i1":
                channel_data = np.array([1 if val else 0 for val in channel_data])
            else:
                channel_data = ["" if val is None else val for val in channel_data]
            for i, cd in enumerate(channel_data):
                var[i] = cd
        # Save the environment to the file
        group_handle = netcdf_handle.createGroup(self.environment_name)
        self.environment_metadata.store_to_netcdf(group_handle)
        # Create Variables for Spectral Data
        group_handle.createDimension(
            "drive_channels", self.last_transfer_function.shape[2]
        )
        group_handle.createDimension(
            "fft_lines", self.environment_metadata.sysid_frame_size // 2 + 1
        )
        var = group_handle.createVariable(
            "frf_data_real",
            "f8",
            ("fft_lines", "specification_channels", "drive_channels"),
        )
        var[...] = self.last_transfer_function.real
        var = group_handle.createVariable(
            "frf_data_imag",
            "f8",
            ("fft_lines", "specification_channels", "drive_channels"),
        )
        var[...] = self.last_transfer_function.imag
        var = group_handle.createVariable(
            "frf_coherence", "f8", ("fft_lines", "specification_channels")
        )
        var[...] = self.last_coherence.real
        var = group_handle.createVariable(
            "response_cpsd_real",
            "f8",
            ("fft_lines", "specification_channels", "specification_channels"),
        )
        var[...] = self.last_response_cpsd.real
        var = group_handle.createVariable(
            "response_cpsd_imag",
            "f8",
            ("fft_lines", "specification_channels", "specification_channels"),
        )
        var[...] = self.last_response_cpsd.imag
        var = group_handle.createVariable(
            "drive_cpsd_real", "f8", ("fft_lines", "drive_channels", "drive_channels")
        )
        var[...] = self.last_reference_cpsd.real
        var = group_handle.createVariable(
            "drive_cpsd_imag", "f8", ("fft_lines", "drive_channels", "drive_channels")
        )
        var[...] = self.last_reference_cpsd.imag
        var = group_handle.createVariable(
            "response_noise_cpsd_real",
            "f8",
            ("fft_lines", "specification_channels", "specification_channels"),
        )
        var[...] = self.last_response_noise.real
        var = group_handle.createVariable(
            "response_noise_cpsd_imag",
            "f8",
            ("fft_lines", "specification_channels", "specification_channels"),
        )
        var[...] = self.last_response_noise.imag
        var = group_handle.createVariable(
            "drive_noise_cpsd_real",
            "f8",
            ("fft_lines", "drive_channels", "drive_channels"),
        )
        var[...] = self.last_reference_noise.real
        var = group_handle.createVariable(
            "drive_noise_cpsd_imag",
            "f8",
            ("fft_lines", "drive_channels", "drive_channels"),
        )
        var[...] = self.last_reference_noise.imag
        var = group_handle.createVariable(
            "control_response", "f8", ("specification_channels", "signal_samples")
        )
        var[...] = self.last_control_data
        var = group_handle.createVariable(
            "control_drives", "f8", ("drive_channels", "signal_samples")
        )
        var[...] = self.last_output_data
        netcdf_handle.close()

    # endregion

    # region Commands
    def display_environment_ended(self):
        """Enables or disables the buttons to start control if it's already running"""
        for widget in [
            self.run_widget.test_level_selector,
            self.run_widget.repeat_signal_checkbox,
            self.run_widget.start_test_button,
        ]:
            widget.setEnabled(True)
        for widget in [self.run_widget.stop_test_button]:
            widget.setEnabled(False)

    def display_environment_started(self):
        for widget in [
            self.run_widget.test_level_selector,
            self.run_widget.repeat_signal_checkbox,
            self.run_widget.start_test_button,
        ]:
            widget.setEnabled(False)
        for widget in [self.run_widget.stop_test_button]:
            widget.setEnabled(True)

    def start_environment(self):
        """Sets itself up to start controlling and sends a signal to the environment to start"""
        for widget in [
            self.run_widget.test_level_selector,
            self.run_widget.repeat_signal_checkbox,
            self.run_widget.start_test_button,
        ]:
            widget.setEnabled(False)

        for item in self.plot_data_items["control_signal_measurement"]:
            item.clear()
        for item in self.plot_data_items["output_signal_measurement"]:
            item.clear()

        super().start_environment()
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

    def change_test_level_from_profile(self, test_level):
        """Updates the test level based on a profile event"""
        self.run_widget.test_level_selector.setValue(int(test_level))

    def set_repeat_from_profile(self, data=None):  # pylint: disable=unused-argument
        """Sets whether or not to repeat the signal based on profile events"""
        self.run_widget.repeat_signal_checkbox.setChecked(True)

    def set_norepeat_from_profile(self, data=None):  # pylint: disable=unused-argument
        """Sets whether or not to repeat the signal based on profile events"""
        self.run_widget.repeat_signal_checkbox.setChecked(False)

    def update_gui(self, queue_data):
        if super().update_gui(queue_data):
            return

        command, data = queue_data
        match command:
            case TransientUICommands.TIME_DATA:
                response_data, output_data, signal_delay = data
                max_y = -1e15
                min_y = 1e15
                for curve, this_data in zip(
                    self.plot_data_items["control_signal_measurement"], response_data
                ):
                    x, y = self.throttled_curves.get(curve)
                    if y is not None:
                        if np.max(y) > max_y:
                            max_y = np.max(y)
                        if np.min(y) < min_y:
                            min_y = np.min(y)
                        if self.max_plot_samples == x.size:
                            x += (this_data.size) / self.hardware_metadata.sample_rate
                            y = np.roll(y, -this_data.size)
                            y[-this_data.size :] = this_data
                        else:
                            x = np.concatenate(
                                (
                                    x,
                                    x[-1]
                                    + (
                                        (1 + np.arange(this_data.size))
                                        / self.hardware_metadata.sample_rate
                                    ),
                                ),
                                axis=0,
                            )
                            y = np.concatenate((y, this_data), axis=0)
                    else:
                        x = (
                            np.arange(this_data.size)
                            / self.hardware_metadata.sample_rate
                        )
                        y = this_data
                    self.throttled_curves.set(
                        curve,
                        x[-self.max_plot_samples :],
                        y[-self.max_plot_samples :],
                    )
                # Display the data
                for curve, this_output in zip(
                    self.plot_data_items["output_signal_measurement"], output_data
                ):
                    x, y = self.throttled_curves.get(curve)
                    if y is not None:
                        if self.max_plot_samples == x.size:
                            x += (this_output.size) / self.hardware_metadata.sample_rate
                            y = np.roll(y, -this_output.size)
                            y[-this_output.size :] = this_output
                        else:
                            x = np.concatenate(
                                (
                                    x,
                                    x[-1]
                                    + (
                                        (1 + np.arange(this_output.size))
                                        / self.hardware_metadata.sample_rate
                                    ),
                                ),
                                axis=0,
                            )
                            y = np.concatenate((y, this_output), axis=0)
                    else:
                        x = (
                            np.arange(this_output.size)
                            / self.hardware_metadata.sample_rate
                        )
                        y = this_output
                    self.throttled_curves.set(
                        curve,
                        x[-self.max_plot_samples :],
                        y[-self.max_plot_samples :],
                    )
                if signal_delay is None:
                    self.throttled_curves.set(
                        self.plot_data_items["signal_range"],
                        np.ones(5) * x[-1],
                        np.zeros(5),
                    )
            case TransientUICommands.CONTROL_DATA:
                self.last_control_data, self.last_output_data = data
                self.update_control_plots()
                max_y = np.max(self.last_control_data)
                min_y = np.min(self.last_control_data)
                for curve, this_data in zip(
                    self.plot_data_items["control_signal_measurement"],
                    self.last_control_data,
                ):
                    x = np.arange(this_data.size) / self.hardware_metadata.sample_rate
                    y = this_data
                    self.throttled_curves.set(curve, x, y)
                # Display the data
                for curve, this_output in zip(
                    self.plot_data_items["output_signal_measurement"],
                    self.last_output_data,
                ):
                    x = np.arange(this_output.size) / self.hardware_metadata.sample_rate
                    y = this_output
                    self.throttled_curves.set(curve, x, y)
                sr = self.hardware_metadata.sample_rate
                self.throttled_curves.set(
                    self.plot_data_items["signal_range"],
                    np.array(
                        (
                            0,
                            0,
                            (self.environment_metadata.control_signal.shape[-1] - 1)
                            / sr,
                            (self.environment_metadata.control_signal.shape[-1] - 1)
                            / sr,
                            0,
                        )
                    ),
                    1.05 * np.array((min_y, max_y, max_y, min_y, min_y)),
                )
            case TransientUICommands.CONTROL_PREDICTIONS:
                (
                    _,  # times,
                    self.excitation_prediction,
                    self.response_prediction,
                    _,  # prediction,
                ) = data
                self.plot_predictions()
            case TransientUICommands.INTERACTIVE_CONTROL_SYSID_UPDATE:
                if self.interactive_control_law_widget is not None:
                    self.interactive_control_law_widget.update_ui_sysid(*data)
            case ControlLawUICommands.INTERACTIVE_CONTROL_UPDATE:
                if self.interactive_control_law_widget is not None:
                    self.interactive_control_law_widget.update_ui_control(data)
            case TransientCommands.SET_TEST_LEVEL:
                self.change_test_level_from_profile(data)
            case TransientCommands.SET_REPEAT:
                self.set_repeat_from_profile()
            case TransientCommands.SET_NO_REPEAT:
                self.set_norepeat_from_profile()
            case UICommands.ENABLE:
                widget = None
                for parent in [
                    self.definition_widget,
                    self.run_widget,
                    self.system_id_widget,
                    self.prediction_widget,
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
                    self.run_widget,
                    self.system_id_widget,
                    self.prediction_widget,
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
                print(f"Unknown Sine UI Command {command}")

    # endregion


# endregion
