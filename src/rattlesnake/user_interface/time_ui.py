from rattlesnake.user_interface.abstract_user_interface import AbstractUI
from rattlesnake.environment.abstract_environment import AbstractMetadata
from rattlesnake.environment.time_environment import TimeParameters, TimeUICommands
from rattlesnake.utilities import (
    GlobalCommands,
    VerboseMessageQueue,
    DataAcquisitionParameters,
    rms_time,
    db2scale,
)
from rattlesnake.user_interface.ui_utilities import UICommands, multiline_plotter, load_time_history
from rattlesnake.environment.environment_utilities import ControlTypes
from rattlesnake.user_interface.ui_utilities import (
    environment_definition_ui_paths,
    environment_run_ui_paths,
)
from qtpy import QtWidgets, QtCore, uic
from multiprocessing.queues import Queue
import numpy as np
import openpyxl
import netCDF4 as nc4

CONTROL_TYPE = ControlTypes.TIME
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
        definition_tabwidget: QtWidgets.QTabWidget,
        system_id_tabwidget: QtWidgets.QTabWidget,  # pylint: disable=unused-argument
        test_predictions_tabwidget: QtWidgets.QTabWidget,  # pylint: disable=unused-argument
        run_tabwidget: QtWidgets.QTabWidget,
        environment_command_queue: VerboseMessageQueue,
        controller_communication_queue: VerboseMessageQueue,
        log_file_queue: Queue,
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
        super().__init__(
            environment_name,
            environment_command_queue,
            controller_communication_queue,
            log_file_queue,
        )
        # Add the page to the control definition tabwidget
        self.definition_widget = QtWidgets.QWidget()
        uic.loadUi(environment_definition_ui_paths[CONTROL_TYPE], self.definition_widget)
        definition_tabwidget.addTab(self.definition_widget, self.environment_name)
        # Add the page to the run tabwidget
        self.run_widget = QtWidgets.QWidget()
        uic.loadUi(environment_run_ui_paths[CONTROL_TYPE], self.run_widget)
        run_tabwidget.addTab(self.run_widget, self.environment_name)

        # Set up some persistent data
        self.data_acquisition_parameters = None
        self.environment_parameters = None
        self.signal = None
        self.physical_output_names = None
        self.physical_measurement_names = None
        self.show_signal_checkboxes = None
        self.plot_data_items = {}

        self.complete_ui()
        self.connect_callbacks()

        # Complete the profile commands
        self.command_map["Set Test Level"] = self.change_test_level_from_profile
        self.command_map["Set Repeat"] = self.set_repeat_from_profile
        self.command_map["Set No Repeat"] = self.set_norepeat_from_profile

    def collect_environment_definition_parameters(self) -> TimeParameters:
        """Collect the parameters from the user interface defining the environment

        Returns
        -------
        TimeParameters
            A metadata or parameters object containing the parameters defining
            the corresponding environment.
        """
        return TimeParameters.from_ui(self)

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
        self.run_widget.start_test_button.clicked.connect(self.start_control)
        self.run_widget.stop_test_button.clicked.connect(self.stop_control)

    def initialize_data_acquisition(self, data_acquisition_parameters: DataAcquisitionParameters):
        """Update the user interface with data acquisition parameters

        This function is called when the Data Acquisition parameters are
        initialized.  This function should set up the environment user interface
        accordingly.

        Parameters
        ----------
        data_acquisition_parameters : DataAcquisitionParameters :
            Container containing the data acquisition parameters, including
            channel table and sampling information.

        """
        self.log("Initializing Data Acquisition")
        self.signal = None
        # Get channel information
        channels = data_acquisition_parameters.channel_list
        num_measurements = len([channel for channel in channels if channel.feedback_device is None])
        num_output = len([channel for channel in channels if channel.feedback_device is not None])
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
            self.definition_widget.signal_information_table.setCellWidget(i, 0, checkbox)
            item = QtWidgets.QTableWidgetItem()
            item.setText("0.0")
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.definition_widget.signal_information_table.setItem(i, 2, item)
            item = QtWidgets.QTableWidgetItem()
            item.setText("0.0")
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.definition_widget.signal_information_table.setItem(i, 3, item)
        # Fill in the info at the bottom
        self.definition_widget.sample_rate_display.setValue(data_acquisition_parameters.sample_rate)
        self.definition_widget.output_sample_rate_display.setValue(
            data_acquisition_parameters.sample_rate * data_acquisition_parameters.output_oversample
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

        self.data_acquisition_parameters = data_acquisition_parameters

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
            self.signal.shape[-1] / self.definition_widget.output_sample_rate_display.value()
        )
        maxs = np.max(np.abs(self.signal), axis=-1)
        rmss = rms_time(self.signal, axis=-1)
        for i, (mx, rms) in enumerate(zip(maxs, rmss)):
            self.definition_widget.signal_information_table.item(i, 2).setText(f"{mx:0.2f}")
            self.definition_widget.signal_information_table.item(i, 3).setText(f"{rms:0.2f}")
        self.show_signal()

    def show_signal(self):
        """Shows the signal on the user interface"""
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

    def initialize_environment(self) -> AbstractMetadata:
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
        data = self.collect_environment_definition_parameters()
        # Make sure everything is defined
        if data.output_signal is None:
            raise ValueError("Output Signal is not defined!")
        # Initialize the correct sizes of the arrays
        for plot_items in [
            self.plot_data_items["output_signal_measurement"],
            self.plot_data_items["response_signal_measurement"],
        ]:
            for curve in plot_items:
                curve.setData(
                    np.arange(
                        (
                            data.output_signal.shape[-1]
                            // self.data_acquisition_parameters.output_oversample
                            * 2
                            if data.output_signal.shape[-1]
                            // self.data_acquisition_parameters.output_oversample
                            * 2
                            < MAX_SAMPLES_TO_PLOT
                            else MAX_SAMPLES_TO_PLOT
                        )
                    )
                    / self.data_acquisition_parameters.sample_rate,
                    np.zeros(
                        (
                            data.output_signal.shape[-1]
                            // self.data_acquisition_parameters.output_oversample
                            * 2
                            if data.output_signal.shape[-1]
                            // self.data_acquisition_parameters.output_oversample
                            * 2
                            < MAX_SAMPLES_TO_PLOT
                            else MAX_SAMPLES_TO_PLOT
                        )
                    ),
                )
        self.environment_parameters = data
        return data

    def retrieve_metadata(
        self, netcdf_handle: nc4._netCDF4.Dataset  # pylint: disable=c-extension-no-member
    ):
        """Collects environment parameters from a netCDF dataset.

        This function retrieves parameters from a netCDF dataset that was written
        by the controller during streaming.  It must populate the widgets
        in the user interface with the proper information.

        This function is the "read" counterpart to the store_to_netcdf
        function in the TimeParameters class, which will write
        parameters to the netCDF file to document the metadata.

        Note that the entire dataset is passed to this function, so the function
        should collect parameters pertaining to the environment from a Group
        in the dataset sharing the environment's name, e.g.

        ``group = netcdf_handle.groups[self.environment_name]``
        ``self.definition_widget.parameter_selector.setValue(group.parameter)``

        Parameters
        ----------
        netcdf_handle : nc4._netCDF4.Dataset :
            The netCDF dataset from which the data will be read.  It should have
            a group name with the enviroment's name.
        """
        group = netcdf_handle.groups[self.environment_name]
        self.signal = group.variables["output_signal"][...].data
        self.definition_widget.cancel_rampdown_selector.setValue(group.cancel_rampdown_time)
        maxs = np.max(np.abs(self.signal), axis=-1)
        rmss = rms_time(self.signal, axis=-1)
        for i, (mx, rms) in enumerate(zip(maxs, rmss)):
            self.definition_widget.signal_information_table.item(i, 2).setText(f"{mx:0.2f}")
            self.definition_widget.signal_information_table.item(i, 3).setText(f"{rms:0.2f}")
        self.show_signal()

    def start_control(self):
        """Starts running the environment"""
        self.run_widget.stop_test_button.setEnabled(True)
        self.run_widget.start_test_button.setEnabled(False)
        self.run_widget.test_level_selector.setEnabled(False)
        self.run_widget.repeat_signal_checkbox.setEnabled(False)
        self.controller_communication_queue.put(
            self.log_name, (GlobalCommands.START_ENVIRONMENT, self.environment_name)
        )
        self.environment_command_queue.put(
            self.log_name,
            (
                GlobalCommands.START_ENVIRONMENT,
                (
                    db2scale(self.run_widget.test_level_selector.value()),
                    self.run_widget.repeat_signal_checkbox.isChecked(),
                ),
            ),
        )
        self.controller_communication_queue.put(
            self.log_name, (GlobalCommands.AT_TARGET_LEVEL, self.environment_name)
        )

    def stop_control(self):
        """Stops running the environment"""
        self.environment_command_queue.put(self.log_name, (GlobalCommands.STOP_ENVIRONMENT, None))

    def change_test_level_from_profile(self, test_level):
        """Sets the test level from a profile instruction

        Parameters
        ----------
        test_level :
            Value to set the test level to.
        """
        self.run_widget.test_level_selector.setValue(int(test_level))

    def set_repeat_from_profile(self, data):  # pylint: disable=unused-argument
        """Sets the the signal to repeat from a profile instruction

        Parameters
        ----------
        data : Ignored
            Parameter is ignored but required by the ``command_map``

        """
        self.run_widget.repeat_signal_checkbox.setChecked(True)

    def set_norepeat_from_profile(self, data):  # pylint: disable=unused-argument
        """Sets the the signal to not repeat from a profile instruction

        Parameters
        ----------
        data : Ignored
            Parameter is ignored but required by the ``command_map``

        """
        self.run_widget.repeat_signal_checkbox.setChecked(False)

    def update_gui(self, queue_data):
        """Update the graphical interface for the environment

        Parameters
        ----------
        queue_data :
            A 2-tuple consisting of ``(message,data)`` pairs where the message
            denotes what to change and the data contains the information needed
            to be displayed.
        """
        message, data = queue_data
        if message == TimeUICommands.TIME_DATA:
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
        elif message == UICommands.ENABLE:
            widget = None
            for parent in [self.definition_widget, self.run_widget]:
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
            for parent in [self.definition_widget, self.run_widget]:
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
            for parent in [self.definition_widget, self.run_widget]:
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

    @staticmethod
    def create_environment_template(
        environment_name: str, workbook: openpyxl.workbook.workbook.Workbook
    ):
        """Creates a template worksheet in an Excel workbook defining the
        environment.

        This function creates a template worksheet in an Excel workbook that
        when filled out could be read by the controller to re-create the
        environment.

        This function is the "write" counterpart to the
        ``set_parameters_from_template`` function in the ``TimeUI`` class,
        which reads the values from the template file to populate the user
        interface.

        Parameters
        ----------
        environment_name : str :
            The name of the environment that will specify the worksheet's name
        workbook : openpyxl.workbook.workbook.Workbook :
            A reference to an ``openpyxl`` workbook.

        """
        worksheet = workbook.create_sheet(environment_name)
        worksheet.cell(1, 1, "Control Type")
        worksheet.cell(1, 2, "Time")
        worksheet.cell(
            1,
            4,
            "Note: Replace cells with hash marks (#) to provide the requested parameters.",
        )
        worksheet.cell(2, 1, "Signal File")
        worksheet.cell(2, 2, "# Path to the file that contains the time signal that will be output")
        worksheet.cell(3, 1, "Cancel Rampdown Time")
        worksheet.cell(
            3,
            2,
            "# Time for the environment to ramp to zero if the environment is cancelled.",
        )

    def set_parameters_from_template(self, worksheet: openpyxl.worksheet.worksheet.Worksheet):
        """
        Collects parameters for the user interface from the Excel template file

        This function reads a filled out template worksheet to create an
        environment.  Cells on this worksheet contain parameters needed to
        specify the environment, so this function should read those cells and
        update the UI widgets with those parameters.

        This function is the "read" counterpart to the
        ``create_environment_template`` function in the ``TimeUI`` class,
        which writes a template file that can be filled out by a user.


        Parameters
        ----------
        worksheet : openpyxl.worksheet.worksheet.Worksheet
            An openpyxl worksheet that contains the environment template.
            Cells on this worksheet should contain the parameters needed for the
            user interface.

        """
        self.load_signal(None, worksheet.cell(2, 2).value)
        self.definition_widget.cancel_rampdown_selector.setValue(float(worksheet.cell(3, 2).value))
