import multiprocessing as mp
import traceback

import netCDF4 as nc4
import numpy as np
import openpyxl
from qtpy import QtCore, QtWidgets, uic

from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.time_environment import (
    TimeCommands,
    TimeInstructions,
    TimeMetadata,
    TimeUICommands,
)
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.engine import RattlesnakeController
from rattlesnake.user_interface.abstract_user_interface import AbstractUI
from rattlesnake.user_interface.ui_utilities import (
    environment_definition_ui_paths,
    environment_run_ui_paths,
    multiline_plotter,
)
from rattlesnake.utilities import (
    RattlesnakeError,
    load_time_history,
    rms_time,
)

ENVIRONMENT_TYPE = EnvironmentType.TIME
MAX_RESPONSES_TO_PLOT = 20
MAX_SAMPLES_TO_PLOT = 100000


# region: User Interface
class TimeUI(AbstractUI):
    """Class defining the user interface for a Random Vibration environment.

    This class will contain two main UIs, the environment definition and run.
    The widgets corresponding to these interfaces are stored in TabWidgets in
    the main UI.

    This class defines all the call backs and user interface operations required
    for the Time environment."""

    def __init__(
        self,
        environment_name: str,
        rattlesnake: RattlesnakeController,
    ):
        """
        Constructs a Time User Interfae

        Given the tab widgets from the main interface as well as communication
        queues, this class assembles the user interface components specific to
        the Time Environment

        Parameters
        ----------
        definition_tabwidget : QtWidgets.QTabWidget
            QTabWidget containing the environment subtabs on the Control
            Definition main tab
        system_id_tabwidget : QtWidgets.QTabWidget
            QTabWidget containing the environment subtabs on the System
            Identification main tab.  The Time Environment has no system
            identification step, so this is not used.
        test_predictions_tabwidget : QtWidgets.QTabWidget
            QTabWidget containing the environment subtabs on the Test Predictions
            main tab.    The Time Environment has no system identification
            step, so this is not used.
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
        super().__init__(ENVIRONMENT_TYPE, environment_name, rattlesnake)
        # Add the page to the control definition tabwidget
        self.definition_widget = QtWidgets.QWidget()
        uic.loadUi(
            environment_definition_ui_paths[ENVIRONMENT_TYPE], self.definition_widget
        )
        # Add the page to the run tabwidget
        self.run_widget = QtWidgets.QWidget()
        uic.loadUi(environment_run_ui_paths[ENVIRONMENT_TYPE], self.run_widget)

        # Set up some persistent data
        self.signal = None
        self.physical_output_names = None
        self.physical_measurement_names = None
        self.show_signal_checkboxes = None
        self.plot_data_items = {}

        # Map commands

        self.complete_ui()
        self.connect_callbacks()

    def complete_ui(self):
        """Helper Function to continue setting up the user interface"""
        # Set common look and feel for plots
        plot_widgets = [
            self.definition_widget.signal_display_plot,
            self.run_widget.output_signal_plot,
            self.run_widget.response_signal_plot,
        ]
        for plot_widget in plot_widgets:
            plot_item = plot_widget.getPlotItem()
            plot_item.showGrid(True, True, 0.25)
            plot_item.enableAutoRange()
            plot_item.getViewBox().enableAutoRange(enable=True)

    def connect_callbacks(self):
        """Helper function to connect callbacks to functions in the class"""
        self.definition_widget.load_signal_button.clicked.connect(self.load_signal)
        self.run_widget.start_test_button.clicked.connect(self.start_environment)
        self.run_widget.stop_test_button.clicked.connect(self.stop_environment)

    # region: Metadata
    def initialize_hardware(self, hardware_metadata: HardwareMetadata):
        """Update the user interface with data acquisition parameters

        This function is called when the Data Acquisition parameters are
        initialized.  This function should set up the environment user interface
        accordingly.

        Parameters
        ----------
        hardware_metadata : DataAcquisitionParameters :
            Container containing the data acquisition parameters, including
            channel table and sampling information.

        """
        self.log("Initializing Data Acquisition")
        self.signal = None
        # Get channel information
        channels = hardware_metadata.channel_list
        num_measurements = len(
            [channel for channel in channels if channel.feedback_device is None]
        )
        num_output = len(
            [channel for channel in channels if channel.feedback_device is not None]
        )
        self.physical_output_names = [
            f"{'' if channel.channel_type is None else channel.channel_type} "
            f"{channel.node_number}{channel.node_direction}"
            for channel in channels
            if channel.feedback_device
        ]
        self.physical_measurement_names = [
            f"{'' if channel.channel_type is None else channel.channel_type} "
            "{channel.node_number}{channel.node_direction}"
            for channel in channels
            if channel.feedback_device is None
        ]
        # Add rows to the signal table
        self.definition_widget.signal_information_table.setRowCount(num_output)
        self.show_signal_checkboxes = []
        for i, name in enumerate(self.physical_output_names):
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
            item.setText("0.0")
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.definition_widget.signal_information_table.setItem(i, 2, item)
            item = QtWidgets.QTableWidgetItem()
            item.setText("0.0")
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.definition_widget.signal_information_table.setItem(i, 3, item)
        # Fill in the info at the bottom
        self.definition_widget.sample_rate_display.setValue(
            hardware_metadata.sample_rate
        )
        self.definition_widget.output_sample_rate_display.setValue(
            hardware_metadata.sample_rate * hardware_metadata.output_oversample
        )
        self.definition_widget.output_channels_display.setValue(num_output)

        # Clear the signal plot
        self.definition_widget.signal_display_plot.getPlotItem().clear()
        self.run_widget.output_signal_plot.getPlotItem().clear()
        self.run_widget.response_signal_plot.getPlotItem().clear()

        # Set initial lines
        self.plot_data_items["output_signal_definition"] = multiline_plotter(
            np.arange(2),
            np.zeros((num_output, 2)),
            widget=self.definition_widget.signal_display_plot,
            other_pen_options={"width": 1},
            names=self.physical_output_names,
        )
        self.plot_data_items["output_signal_measurement"] = multiline_plotter(
            np.arange(2),
            np.zeros((num_output, 2)),
            widget=self.run_widget.output_signal_plot,
            other_pen_options={"width": 1},
            names=self.physical_output_names,
        )
        self.plot_data_items["response_signal_measurement"] = multiline_plotter(
            np.arange(2),
            np.zeros(
                (
                    (
                        num_measurements
                        if num_measurements < MAX_RESPONSES_TO_PLOT
                        else MAX_RESPONSES_TO_PLOT
                    ),
                    2,
                )
            ),
            widget=self.run_widget.response_signal_plot,
            other_pen_options={"width": 1},
            names=self.physical_measurement_names,
        )

        self.hardware_metadata = hardware_metadata

    def initialize_environment(self, environment_metadata):
        return

    def get_environment_metadata(self, global_channel_list=None) -> TimeMetadata:
        """Collect the parameters from the user interface defining the environment

        Returns
        -------
        TimeParameters
            A metadata or parameters object containing the parameters defining
            the corresponding environment.
        """
        if self.hardware_metadata and global_channel_list:
            channel_list_bools = self.get_channel_list_bools(global_channel_list)
        else:
            channel_list_bools = []

        sample_rate = self.definition_widget.output_sample_rate_display.value()
        output_signal = self.signal
        cancel_rampdown_time = self.definition_widget.cancel_rampdown_selector.value()
        metadata = TimeMetadata(
            self.environment_name,
            channel_list_bools,
            sample_rate,
            output_signal,
            cancel_rampdown_time,
        )

        signal_filepath = self.definition_widget.signal_file_name_display.text()
        if signal_filepath:
            metadata.set_file(signal_filepath)

        return metadata

    def set_environment_metadata(self, metadata: TimeMetadata):
        """Update the user interface with environment parameters

        This function is called when the Environment parameters are initialized.
        This function should set up the user interface accordingly.  It must
        return the parameters class of the environment that inherits from
        AbstractMetadata.

        Returns
        -------
        environment_parameters : TimeParameters
            A TimeParameters object that contains the parameters
            defining the environment.
        """
        self.log("Initializing Environment Parameters")
        # Make sure everything is defined
        if metadata.output_signal is None:
            raise RattlesnakeError("Output Signal is not defined!")

        self.signal = metadata.output_signal
        if metadata.signal_file:
            self.definition_widget.signal_file_name_display.setText(
                metadata.signal_file
            )

        self.show_signal()

    def get_environment_instructions(self):
        current_test_level = self.run_widget.test_level_selector.value()
        repeat = self.run_widget.repeat_signal_checkbox.isChecked()
        instruction = TimeInstructions(
            self.environment_name, current_test_level, repeat
        )

        return instruction

    def set_environment_instructions(self, data: TimeInstructions):
        instructions = data
        self.run_widget.test_level_selector.setValue(instructions.current_test_level)
        self.run_widget.repeat_signal_checkbox.setChecked(instructions.repeat)

    def update_gui(self, queue_data):
        """Update the graphical interface for the environment

        Parameters
        ----------
        queue_data :
            A 2-tuple consisting of ``(message,data)`` pairs where the message
            denotes what to change and the data contains the information needed
            to be displayed.
        """
        if super().update_gui(queue_data):
            return
        command, data = queue_data
        match command:
            case TimeUICommands.TIME_DATA:
                self.plot_time_data(data)
            case TimeCommands.SET_TEST_LEVEL:
                self.set_test_level(data)
            case TimeCommands.SET_NO_REPEAT:
                self.set_no_repeat(data)
            case TimeCommands.SET_REPEAT:
                self.set_repeat(data)
            case _:
                print(f"Unknown Modal UI Command {command}")

    # region: Commands
    def display_environment_started(self):
        self.run_widget.stop_test_button.setEnabled(True)
        self.run_widget.start_test_button.setEnabled(False)
        self.run_widget.test_level_selector.setEnabled(False)
        self.run_widget.repeat_signal_checkbox.setEnabled(False)

    def display_environment_ended(self):
        self.run_widget.stop_test_button.setEnabled(False)
        self.run_widget.start_test_button.setEnabled(True)
        self.run_widget.test_level_selector.setEnabled(True)
        self.run_widget.repeat_signal_checkbox.setEnabled(True)

    def set_test_level(self, data):
        test_level = int(data)
        self.run_widget.test_level_selector.setValue(test_level)

    def set_repeat(self, data):
        self.run_widget.repeat_signal_checkbox.setChecked(True)

    def set_no_repeat(self, data):
        self.run_widget.repeat_signal_checkbox.setChecked(False)

    def plot_time_data(self, data):
        response_data, output_data = data
        for curve, this_data in zip(
            self.plot_data_items["response_signal_measurement"], response_data
        ):
            x, y = curve.getData()
            y = np.concatenate((y[this_data.size :], this_data[-x.size :]), axis=0)
            curve.setData(x, y)
        # Display the data
        for curve, this_output in zip(
            self.plot_data_items["output_signal_measurement"], output_data
        ):
            x, y = curve.getData()
            y = np.concatenate((y[this_output.size :], this_output[-x.size :]), axis=0)
            curve.setData(x, y)

    # region: Callbacks
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
        self.signal = load_time_history(
            filename, self.definition_widget.output_sample_rate_display.value()
        )
        self.definition_widget.signal_samples_display.setValue(self.signal.shape[-1])
        self.definition_widget.signal_time_display.setValue(
            self.signal.shape[-1]
            / self.definition_widget.output_sample_rate_display.value()
        )
        maxs = np.max(np.abs(self.signal), axis=-1)
        rmss = rms_time(self.signal, axis=-1)
        for i, (mx, rms) in enumerate(zip(maxs, rmss)):
            self.definition_widget.signal_information_table.item(i, 2).setText(
                f"{mx:0.2f}"
            )
            self.definition_widget.signal_information_table.item(i, 3).setText(
                f"{rms:0.2f}"
            )
        self.show_signal()

    def show_signal(self):
        """Shows the signal on the user interface"""
        # Update run tab x-axis length
        for plot_items in [
            self.plot_data_items["output_signal_measurement"],
            self.plot_data_items["response_signal_measurement"],
        ]:
            for curve in plot_items:
                curve.setData(
                    np.arange(
                        (
                            self.signal.shape[-1]
                            // self.hardware_metadata.output_oversample
                            * 2
                            if self.signal.shape[-1]
                            // self.hardware_metadata.output_oversample
                            * 2
                            < MAX_SAMPLES_TO_PLOT
                            else MAX_SAMPLES_TO_PLOT
                        )
                    )
                    / self.hardware_metadata.sample_rate,
                    np.zeros(
                        (
                            self.signal.shape[-1]
                            // self.hardware_metadata.output_oversample
                            * 2
                            if self.signal.shape[-1]
                            // self.hardware_metadata.output_oversample
                            * 2
                            < MAX_SAMPLES_TO_PLOT
                            else MAX_SAMPLES_TO_PLOT
                        )
                    ),
                )

        for curve, signal, check_box in zip(
            self.plot_data_items["output_signal_definition"],
            self.signal,
            self.show_signal_checkboxes,
        ):
            if check_box.isChecked():
                x = (
                    np.arange(signal.shape[-1])
                    / self.definition_widget.output_sample_rate_display.value()
                )
                curve.setData(x, signal)
            else:
                curve.setData((0, 0), (0, 0))

    def start_environment(self):
        """Starts running the environment"""
        self.run_widget.start_test_button.setEnabled(False)
        self.run_widget.test_level_selector.setEnabled(False)
        self.run_widget.repeat_signal_checkbox.setEnabled(False)
        super().start_environment()
        self.rattlesnake.environment_at_target_level(self.environment_name)

    def start_environment_ready(self):
        super().start_environment_ready()

    def start_environment_error(self, error):
        super().start_environment_error(error)

    def stop_environment(self):
        """Stops running the environment"""
        self.run_widget.stop_test_button.setEnabled(False)
        super().stop_environment()

    def stop_environment_ready(self):
        super().stop_environment_ready()

    def stop_environment_error(self, error):
        super().stop_environment_error(error)
