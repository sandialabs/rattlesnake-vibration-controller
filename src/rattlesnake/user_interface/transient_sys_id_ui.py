from rattlesnake.environment.transient_sys_id_environment import (
    TransientCommands,
    TransientUICommands,
    TransientMetadata,
)
from rattlesnake.user_interface.abstract_sys_id_user_interface import AbstractSysIdUI
from rattlesnake.environment.abstract_interactive_control_law import (  # noqa: E402 pylint: disable=wrong-import-position
    AbstractControlLawComputation,
    ControlLawUICommands,
)
from rattlesnake.utilities import GlobalCommands, VerboseMessageQueue
from rattlesnake.environment.environment_utilities import ControlTypes
from rattlesnake.user_interface.ui_utilities import (
    environment_definition_ui_paths,
    environment_prediction_ui_paths,
    environment_run_ui_paths,
)
from rattlesnake.user_interface.ui_utilities import (
    UICommands,
    PlotTimeWindow,
    TransformationMatrixWindow,
    colororder,
    load_time_history,
    multiline_plotter,
)
from rattlesnake.utilities import DataAcquisitionParameters, db2scale, load_python_module, rms_time
from qtpy import QtCore, QtWidgets, uic
from qtpy.QtCore import Qt
from multiprocessing.queues import Queue
import inspect
import os
import numpy as np
import netCDF4 as nc4

CONTROL_TYPE = ControlTypes.TRANSIENT
MAXIMUM_NAME_LENGTH = 50


# region: User Interface
class TransientUI(AbstractSysIdUI):
    """Class defining the user interface for the transient environment"""

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

        self.connect_callbacks()

        # Complete the profile commands
        self.command_map["Set Test Level"] = self.change_test_level_from_profile
        self.command_map["Set Repeat"] = self.set_repeat_from_profile
        self.command_map["Set No Repeat"] = self.set_norepeat_from_profile

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
        self.prediction_widget.response_selector.currentIndexChanged.connect(self.plot_predictions)
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
        self.prediction_widget.maximum_error_button.clicked.connect(self.show_max_error_prediction)
        self.prediction_widget.minimum_error_button.clicked.connect(self.show_min_error_prediction)
        self.prediction_widget.recompute_predictions_button.clicked.connect(
            self.recompute_predictions
        )
        # Run Test
        self.run_widget.start_test_button.clicked.connect(self.start_control)
        self.run_widget.stop_test_button.clicked.connect(self.stop_control)
        self.run_widget.create_window_button.clicked.connect(self.create_window)
        self.run_widget.show_all_channels_button.clicked.connect(self.show_all_channels)
        self.run_widget.tile_windows_button.clicked.connect(self.tile_windows)
        self.run_widget.close_windows_button.clicked.connect(self.close_windows)
        self.run_widget.control_response_error_list.itemDoubleClicked.connect(self.show_window)
        self.run_widget.save_current_control_data_button.clicked.connect(self.save_control_data)
        self.run_widget.display_duration_spinbox.valueChanged.connect(self.set_display_duration)

    # %% Data Acquisition

    def initialize_data_acquisition(self, data_acquisition_parameters):
        super().initialize_data_acquisition(data_acquisition_parameters)
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
        self.definition_widget.sample_rate_display.setValue(data_acquisition_parameters.sample_rate)
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
        self.definition_widget.input_channels_display.setValue(len(self.physical_channel_names))
        self.definition_widget.output_channels_display.setValue(len(self.physical_output_indices))
        self.definition_widget.control_channels_display.setValue(0)

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

    def update_control_channels(self):
        """Callback called when control channels are updated in the UI"""
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.specification_signal = None
        self.definition_widget.control_channels_display.setValue(len(self.physical_control_indices))
        self.define_transformation_matrices(None, False)
        self.show_signal()

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
        return TransientMetadata(
            len(self.data_acquisition_parameters.channel_list),
            self.definition_widget.sample_rate_display.value(),
            self.specification_signal,
            self.definition_widget.ramp_selector.value(),
            control_module,
            control_function,
            control_function_type,
            control_function_parameters,
            self.physical_control_indices,
            self.physical_output_indices,
            self.response_transformation_matrix,
            self.output_transformation_matrix,
        )

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
        self.definition_widget.signal_samples_display.setValue(self.specification_signal.shape[-1])
        self.definition_widget.signal_time_display.setValue(
            self.specification_signal.shape[-1] / self.definition_widget.sample_rate_display.value()
        )
        maxs = np.max(np.abs(self.specification_signal), axis=-1)
        rmss = rms_time(self.specification_signal, axis=-1)
        # Add rows to the signal table
        self.definition_widget.signal_information_table.setRowCount(
            self.specification_signal.shape[0]
        )
        self.show_signal_checkboxes = []
        for i, (name, mx, rms) in enumerate(zip(self.physical_control_names, maxs, rmss)):
            item = QtWidgets.QTableWidgetItem()
            item.setText(name)
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.definition_widget.signal_information_table.setItem(i, 1, item)
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.show_signal)
            self.show_signal_checkboxes.append(checkbox)
            self.definition_widget.signal_information_table.setCellWidget(i, 0, checkbox)
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
            # Clear the signals
            self.definition_widget.signal_information_table.clear()
            self.definition_widget.signal_display_plot.clear()
            self.definition_widget.signal_file_name_display.clear()
            self.definition_widget.signal_information_table.setRowCount(0)
            self.show_signal_checkboxes = None
            self.response_transformation_matrix = response_transformation
            self.output_transformation_matrix = output_transformation

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
            self.definition_widget.control_function_generator_selector.setCurrentIndex(1)
        elif inspect.isclass(function) and issubclass(function, AbstractControlLawComputation):
            self.definition_widget.control_function_generator_selector.setCurrentIndex(3)
        elif inspect.isclass(function):
            self.definition_widget.control_function_generator_selector.setCurrentIndex(2)
        else:
            self.definition_widget.control_function_generator_selector.setCurrentIndex(0)

    def initialize_environment(self):
        super().initialize_environment()
        # Make sure everything is defined
        if self.environment_parameters.control_signal is None:
            raise ValueError(f"Control Signal is not defined for {self.environment_name}!")
        if self.environment_parameters.control_python_script is None:
            raise ValueError(f"Control function has not been loaded for {self.environment_name}")
        self.system_id_widget.samplesPerFrameSpinBox.setMaximum(self.specification_signal.shape[-1])
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
            np.arange(self.environment_parameters.control_signal.shape[-1])
            / self.environment_parameters.sample_rate,
            np.zeros((2, self.environment_parameters.control_signal.shape[-1])),
            widget=self.prediction_widget.response_display_plot,
            other_pen_options={"width": 1},
            names=["Prediction", "Spec"],
            downsample={"auto": True},
            clip_to_view=True,
        )
        self.plot_data_items["excitation_prediction"] = multiline_plotter(
            np.arange(self.environment_parameters.control_signal.shape[-1])
            / self.environment_parameters.sample_rate,
            np.zeros((1, self.environment_parameters.control_signal.shape[-1])),
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
            self.data_acquisition_parameters.sample_rate
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
        if self.definition_widget.control_function_generator_selector.currentIndex() == 3:
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
                    self.data_acquisition_parameters, self.environment_parameters
                )
            else:
                if self.interactive_control_law_widget is not None:
                    self.interactive_control_law_widget.close()
                self.interactive_control_law_window = QtWidgets.QDialog(self.definition_widget)
                self.interactive_control_law_widget = ui_class(
                    self.log_name,
                    self.environment_command_queue,
                    self.interactive_control_law_window,
                    self,
                    self.data_acquisition_parameters,
                    self.environment_parameters,
                )
            self.interactive_control_law_window.show()
        return self.environment_parameters

    def check_selected_control_channels(self):
        """Callback to check control channels that are selected"""
        for item in self.definition_widget.control_channels_selector.selectedItems():
            item.setCheckState(Qt.Checked)

    def uncheck_selected_control_channels(self):
        """Callback to uncheck control channels that are selected"""
        for item in self.definition_widget.control_channels_selector.selectedItems():
            item.setCheckState(Qt.Unchecked)

    # %% Predictions
    def plot_predictions(self):
        """Plots the control predictions based on the currently selected item"""
        times = (
            np.arange(self.specification_signal.shape[-1])
            / self.data_acquisition_parameters.sample_rate
        )
        index = self.prediction_widget.excitation_selector.currentIndex()
        self.plot_data_items["excitation_prediction"][0].setData(
            times, self.excitation_prediction[index]
        )
        index = self.prediction_widget.response_selector.currentIndex()
        self.plot_data_items["response_prediction"][0].setData(
            times, self.response_prediction[index]
        )
        self.plot_data_items["response_prediction"][1].setData(
            times, self.specification_signal[index]
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
        self.environment_command_queue.put(
            self.log_name, (TransientCommands.PERFORM_CONTROL_PREDICTION, False)
        )

    # %% Control

    def start_control(self):
        """Starts the chain of events to start the environment"""
        self.enable_control(False)
        self.controller_communication_queue.put(
            self.log_name, (GlobalCommands.START_ENVIRONMENT, self.environment_name)
        )
        self.environment_command_queue.put(
            self.log_name,
            (
                TransientCommands.START_CONTROL,
                (
                    db2scale(self.run_widget.test_level_selector.value()),
                    self.run_widget.repeat_signal_checkbox.isChecked(),
                ),
            ),
        )
        if self.run_widget.test_level_selector.value() >= 0:
            self.controller_communication_queue.put(
                self.log_name, (GlobalCommands.AT_TARGET_LEVEL, self.environment_name)
            )
        for item in self.plot_data_items["control_signal_measurement"]:
            item.clear()
        for item in self.plot_data_items["output_signal_measurement"]:
            item.clear()

    def stop_control(self):
        """Starts the sequence of events to stop the controller prematurely"""
        self.environment_command_queue.put(self.log_name, (TransientCommands.STOP_CONTROL, None))

    def enable_control(self, enabled):
        """Enables or disables the buttons to start control if it's already running"""
        for widget in [
            self.run_widget.test_level_selector,
            self.run_widget.repeat_signal_checkbox,
            self.run_widget.start_test_button,
        ]:
            widget.setEnabled(enabled)
        for widget in [self.run_widget.stop_test_button]:
            widget.setEnabled(not enabled)

    def change_test_level_from_profile(self, test_level):
        """Updates the test level based on a profile event"""
        self.run_widget.test_level_selector.setValue(int(test_level))

    def set_repeat_from_profile(self, data):  # pylint: disable=unused-argument
        """Sets whether or not to repeat the signal based on profile events"""
        self.run_widget.repeat_signal_checkbox.setChecked(True)

    def set_norepeat_from_profile(self, data):  # pylint: disable=unused-argument
        """Sets whether or not to repeat the signal based on profile events"""
        self.run_widget.repeat_signal_checkbox.setChecked(False)

    def set_display_duration(self, value):
        """Updates the display duration in the UI"""
        self.max_plot_samples = int(self.data_acquisition_parameters.sample_rate * value)

    def create_window(self, event, control_index=None):  # pylint: disable=unused-argument
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
                self.environment_parameters.control_signal,
                self.data_acquisition_parameters.sample_rate,
                self.run_widget.control_channel_selector.itemText(control_index),
            )
        )
        if self.last_control_data is not None:
            self.plot_windows[-1].update_plot(self.last_control_data)

    def show_all_channels(self):
        """Creates a subwindow for each ASD in the CPSD matrix"""
        for i in range(self.environment_parameters.control_signal.shape[0]):
            self.create_window(None, i)
        self.tile_windows()

    def tile_windows(self):
        """Tile subwindow equally across the screen"""
        screen_rect = QtWidgets.QApplication.desktop().screenGeometry()
        # Go through and remove any closed windows
        self.plot_windows = [window for window in self.plot_windows if window.isVisible()]
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
        self.plot_windows = [window for window in self.plot_windows if window.isVisible()]
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
        global_data_parameters: DataAcquisitionParameters
        global_data_parameters = self.data_acquisition_parameters
        netcdf_handle = nc4.Dataset(  # pylint: disable=no-member
            filename, "w", format="NETCDF4", clobber=True
        )
        # Create dimensions
        netcdf_handle.createDimension("response_channels", len(global_data_parameters.channel_list))
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
            global_data_parameters.samples_per_write / global_data_parameters.output_sample_rate
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
        var = netcdf_handle.createVariable("environment_names", str, ("num_environments",))
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
            global_data_parameters.environment_active_channels[:, this_environment_index],
            :,
        ]
        # Create channel table variables

        for label, netcdf_datatype in labels:
            var = netcdf_handle.createVariable(
                "/channels/" + label, netcdf_datatype, ("response_channels",)
            )
            channel_data = [
                getattr(channel, label) for channel in global_data_parameters.channel_list
            ]
            if netcdf_datatype == "i1":
                channel_data = np.array([1 if val else 0 for val in channel_data])
            else:
                channel_data = ["" if val is None else val for val in channel_data]
            for i, cd in enumerate(channel_data):
                var[i] = cd
        # Save the environment to the file
        group_handle = netcdf_handle.createGroup(self.environment_name)
        self.environment_parameters.store_to_netcdf(group_handle)
        # Create Variables for Spectral Data
        group_handle.createDimension("drive_channels", self.last_transfer_function.shape[2])
        group_handle.createDimension(
            "fft_lines", self.environment_parameters.sysid_frame_size // 2 + 1
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

    # %% Misc

    def retrieve_metadata(self, netcdf_handle=None, environment_name=None):
        group = super().retrieve_metadata(netcdf_handle, environment_name)

        # Control channels
        try:
            for i in group.variables["control_channel_indices"][...]:
                item = self.definition_widget.control_channels_selector.item(i)
                item.setCheckState(Qt.Checked)
        except KeyError:
            print("no variable control_channel_indices, please select control channels manually")
        # Other Data
        try:
            self.response_transformation_matrix = group.variables["response_transformation_matrix"][
                ...
            ].data
        except KeyError:
            self.response_transformation_matrix = None
        try:
            self.output_transformation_matrix = group.variables["reference_transformation_matrix"][
                ...
            ].data
        except KeyError:
            self.output_transformation_matrix = None
        self.define_transformation_matrices(None, dialog=False)

        if (
            environment_name is None
        ):  # environment_name is passed when the saved environment doesn't
            # match the current environment
            self.definition_widget.ramp_selector.setValue(group.test_level_ramp_time)
            self.specification_signal = group.variables["control_signal"][...].data
            self.select_python_module(None, group.control_python_script)
            index = self.definition_widget.control_function_input.findText(
                group.control_python_function
            )
            if index == -1:
                index = 0
                default = self.definition_widget.control_function_input.itemText(index)
                print(
                    f'Warning: control function "{group.control_python_function}" '
                    f'not found, defaulting to "{default}"'
                )
            self.definition_widget.control_function_input.setCurrentIndex(index)
            self.definition_widget.control_parameters_text_input.setText(
                group.control_python_function_parameters
            )
            self.setup_specification_table()
            self.show_signal()

    def update_gui(self, queue_data):
        if super().update_gui(queue_data):
            return
        message, data = queue_data
        if message == TransientUICommands.TIME_DATA:
            response_data, output_data, signal_delay = data
            max_y = -1e15
            min_y = 1e15
            for curve, this_data in zip(
                self.plot_data_items["control_signal_measurement"], response_data
            ):
                x, y = curve.getOriginalDataset()
                if y is not None:
                    if np.max(y) > max_y:
                        max_y = np.max(y)
                    if np.min(y) < min_y:
                        min_y = np.min(y)
                    if self.max_plot_samples == x.size:
                        x += (this_data.size) / self.data_acquisition_parameters.sample_rate
                        y = np.roll(y, -this_data.size)
                        y[-this_data.size :] = this_data
                    else:
                        x = np.concatenate(
                            (
                                x,
                                x[-1]
                                + (
                                    (1 + np.arange(this_data.size))
                                    / self.data_acquisition_parameters.sample_rate
                                ),
                            ),
                            axis=0,
                        )
                        y = np.concatenate((y, this_data), axis=0)
                else:
                    x = np.arange(this_data.size) / self.data_acquisition_parameters.sample_rate
                    y = this_data
                curve.setData(x[-self.max_plot_samples :], y[-self.max_plot_samples :])
            # Display the data
            for curve, this_output in zip(
                self.plot_data_items["output_signal_measurement"], output_data
            ):
                x, y = curve.getOriginalDataset()
                if y is not None:
                    if self.max_plot_samples == x.size:
                        x += (this_output.size) / self.data_acquisition_parameters.sample_rate
                        y = np.roll(y, -this_output.size)
                        y[-this_output.size :] = this_output
                    else:
                        x = np.concatenate(
                            (
                                x,
                                x[-1]
                                + (
                                    (1 + np.arange(this_output.size))
                                    / self.data_acquisition_parameters.sample_rate
                                ),
                            ),
                            axis=0,
                        )
                        y = np.concatenate((y, this_output), axis=0)
                else:
                    x = np.arange(this_output.size) / self.data_acquisition_parameters.sample_rate
                    y = this_output
                curve.setData(x[-self.max_plot_samples :], y[-self.max_plot_samples :])
            if signal_delay is None:
                self.plot_data_items["signal_range"].setData(np.ones(5) * x[-1], np.zeros(5))
        elif message == TransientUICommands.CONTROL_DATA:
            self.last_control_data, self.last_output_data = data
            self.update_control_plots()
            max_y = np.max(self.last_control_data)
            min_y = np.min(self.last_control_data)
            for curve, this_data in zip(
                self.plot_data_items["control_signal_measurement"],
                self.last_control_data,
            ):
                x, y = curve.getOriginalDataset()
                x = np.arange(this_data.size) / self.data_acquisition_parameters.sample_rate
                y = this_data
                curve.setData(x, y)
            # Display the data
            for curve, this_output in zip(
                self.plot_data_items["output_signal_measurement"], self.last_output_data
            ):
                x, y = curve.getOriginalDataset()
                x = np.arange(this_output.size) / self.data_acquisition_parameters.sample_rate
                y = this_output
                curve.setData(x, y)
            sr = self.data_acquisition_parameters.sample_rate
            self.plot_data_items["signal_range"].setData(
                np.array(
                    (
                        0,
                        0,
                        (self.environment_parameters.control_signal.shape[-1] - 1) / sr,
                        (self.environment_parameters.control_signal.shape[-1] - 1) / sr,
                        0,
                    )
                ),
                1.05 * np.array((min_y, max_y, max_y, min_y, min_y)),
            )
        elif message == TransientUICommands.CONTROL_PREDICTIONS:
            (
                _,  # times,
                self.excitation_prediction,
                self.response_prediction,
                _,  # prediction,
            ) = data
            self.plot_predictions()
        elif message == TransientUICommands.INTERACTIVE_CONTROL_SYSID_UPDATE:
            if self.interactive_control_law_widget is not None:
                self.interactive_control_law_widget.update_ui_sysid(*data)
        elif message == ControlLawUICommands.INTERACTIVE_CONTROL_UPDATE:
            if self.interactive_control_law_widget is not None:
                self.interactive_control_law_widget.update_ui_control(data)
        elif message == TransientUICommands.ENABLE_CONTROL:
            self.enable_control(True)
        elif message == UICommands.ENABLE:
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
        elif message == UICommands.DISABLE:
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
        else:
            widget = None
            for parent in [
                self.definition_widget,
                self.run_widget,
                self.system_id_widget,
                self.prediction_widget,
            ]:
                try:
                    widget = getattr(parent, message)
                    break
                except AttributeError:
                    continue
            if widget is None:
                raise ValueError(f"Cannot Update Widget {message}: not found in UI")
            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.setValue(data)
            elif isinstance(widget, QtWidgets.QSpinBox):
                widget.setValue(data)
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(data)
            elif isinstance(widget, QtWidgets.QListWidget):
                widget.clear()
                widget.addItems([f"{d:.3f}" for d in data])

    def set_parameters_from_template(self, worksheet):
        self.definition_widget.ramp_selector.setValue(float(worksheet.cell(3, 2).value))
        self.select_python_module(None, worksheet.cell(4, 2).value)
        self.definition_widget.control_function_input.setCurrentIndex(
            self.definition_widget.control_function_input.findText(worksheet.cell(5, 2).value)
        )
        self.definition_widget.control_parameters_text_input.setText(
            "" if worksheet.cell(6, 2).value is None else str(worksheet.cell(6, 2).value)
        )
        column_index = 2
        while True:
            value = worksheet.cell(7, column_index).value
            if value is None or (isinstance(value, str) and value.strip() == ""):
                break
            item = self.definition_widget.control_channels_selector.item(int(value) - 1)
            item.setCheckState(Qt.Checked)
            column_index += 1
        self.system_id_widget.samplesPerFrameSpinBox.setValue(int(worksheet.cell(8, 2).value))
        self.system_id_widget.averagingTypeComboBox.setCurrentIndex(
            self.system_id_widget.averagingTypeComboBox.findText(worksheet.cell(9, 2).value)
        )
        self.system_id_widget.noiseAveragesSpinBox.setValue(int(worksheet.cell(10, 2).value))
        self.system_id_widget.systemIDAveragesSpinBox.setValue(int(worksheet.cell(11, 2).value))
        self.system_id_widget.averagingCoefficientDoubleSpinBox.setValue(
            float(worksheet.cell(12, 2).value)
        )
        self.system_id_widget.estimatorComboBox.setCurrentIndex(
            self.system_id_widget.estimatorComboBox.findText(worksheet.cell(13, 2).value)
        )
        self.system_id_widget.levelDoubleSpinBox.setValue(float(worksheet.cell(14, 2).value))
        # this should be a temporary solution - template file rework needed
        low, high = worksheet.cell(14, 3).value, worksheet.cell(14, 4).value
        if low is not None:
            self.system_id_widget.lowFreqCutoffSpinBox.setValue(int(low))
        if high is not None:
            self.system_id_widget.highFreqCutoffSpinBox.setValue(int(high))
        self.system_id_widget.levelRampTimeDoubleSpinBox.setValue(
            float(worksheet.cell(15, 2).value)
        )
        self.system_id_widget.signalTypeComboBox.setCurrentIndex(
            self.system_id_widget.signalTypeComboBox.findText(worksheet.cell(16, 2).value)
        )
        self.system_id_widget.windowComboBox.setCurrentIndex(
            self.system_id_widget.windowComboBox.findText(worksheet.cell(17, 2).value)
        )
        self.system_id_widget.overlapDoubleSpinBox.setValue(float(worksheet.cell(18, 2).value))
        self.system_id_widget.onFractionDoubleSpinBox.setValue(float(worksheet.cell(19, 2).value))
        self.system_id_widget.pretriggerDoubleSpinBox.setValue(float(worksheet.cell(20, 2).value))
        self.system_id_widget.rampFractionDoubleSpinBox.setValue(float(worksheet.cell(21, 2).value))

        # Now we need to find the transformation matrices' sizes
        response_channels = self.definition_widget.control_channels_display.value()
        output_channels = self.definition_widget.output_channels_display.value()
        output_transform_row = 23
        if (
            isinstance(worksheet.cell(22, 2).value, str)
            and worksheet.cell(22, 2).value.lower() == "none"
        ):
            self.response_transformation_matrix = None
        else:
            while True:
                if worksheet.cell(output_transform_row, 1).value == "Output Transformation Matrix:":
                    break
                output_transform_row += 1
            response_size = output_transform_row - 22
            response_transformation = []
            for i in range(response_size):
                response_transformation.append([])
                for j in range(response_channels):
                    response_transformation[-1].append(float(worksheet.cell(22 + i, 2 + j).value))
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
        self.load_signal(None, worksheet.cell(2, 2).value)

    @staticmethod
    def create_environment_template(environment_name, workbook):
        worksheet = workbook.create_sheet(environment_name)
        worksheet.cell(1, 1, "Control Type")
        worksheet.cell(1, 2, "Transient")
        worksheet.cell(
            1,
            4,
            "Note: Replace cells with hash marks (#) to provide the requested parameters.",
        )
        worksheet.cell(2, 1, "Signal File")
        worksheet.cell(2, 2, "# Path to the file that contains the time signal that will be output")
        worksheet.cell(3, 1, "Ramp Time")
        worksheet.cell(
            3,
            2,
            "# Time for the environment to ramp between levels or from start or to stop.",
        )
        worksheet.cell(4, 1, "Control Python Script:")
        worksheet.cell(4, 2, "# Path to the Python script containing the control law")
        worksheet.cell(5, 1, "Control Python Function:")
        worksheet.cell(
            5,
            2,
            "# Function name within the Python Script that will serve as the control law",
        )
        worksheet.cell(6, 1, "Control Parameters:")
        worksheet.cell(6, 2, "# Extra parameters used in the control law")
        worksheet.cell(7, 1, "Control Channels (1-based):")
        worksheet.cell(7, 2, "# List of channels, one per cell on this row")
        worksheet.cell(8, 1, "System ID Samples per Frame")
        worksheet.cell(
            8,
            2,
            "# Number of Samples per Measurement Frame in the System Identification",
        )
        worksheet.cell(9, 1, "System ID Averaging:")
        worksheet.cell(9, 2, "# Averaging Type, should be Linear or Exponential")
        worksheet.cell(10, 1, "Noise Averages:")
        worksheet.cell(10, 2, "# Number of Averages used when characterizing noise")
        worksheet.cell(11, 1, "System ID Averages:")
        worksheet.cell(11, 2, "# Number of Averages used when computing the FRF")
        worksheet.cell(12, 1, "Exponential Averaging Coefficient:")
        worksheet.cell(12, 2, "# Averaging Coefficient for Exponential Averaging (if used)")
        worksheet.cell(13, 1, "System ID Estimator:")
        worksheet.cell(
            13,
            2,
            "# Technique used to compute system ID.  Should be one of H1, H2, H3, or Hv.",
        )
        worksheet.cell(14, 1, "System ID Level (V RMS):")
        worksheet.cell(
            14,
            2,
            "# RMS Value of Flat Voltage Spectrum used for System Identification.",
        )
        worksheet.cell(15, 1, "System ID Ramp Time")
        worksheet.cell(
            15,
            2,
            "# Time for the system identification to ramp between levels or from start or to stop.",
        )
        worksheet.cell(16, 1, "System ID Signal Type:")
        worksheet.cell(16, 2, "# Signal to use for the system identification")
        worksheet.cell(17, 1, "System ID Window:")
        worksheet.cell(
            17,
            2,
            "# Window used to compute FRFs during system ID.  Should be one of Hann or None",
        )
        worksheet.cell(18, 1, "System ID Overlap %:")
        worksheet.cell(18, 2, "# Overlap to use in the system identification")
        worksheet.cell(19, 1, "System ID Burst On %:")
        worksheet.cell(19, 2, "# Percentage of a frame that the burst random is on for")
        worksheet.cell(20, 1, "System ID Burst Pretrigger %:")
        worksheet.cell(
            20,
            2,
            "# Percentage of a frame that occurs before the burst starts in a burst random signal",
        )
        worksheet.cell(21, 1, "System ID Ramp Fraction %:")
        worksheet.cell(
            21,
            2,
            '# Percentage of the "System ID Burst On %" that will be used to ramp up to full level',
        )
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
