from rattlesnake.user_interface.abstract_user_interface import AbstractUI
from rattlesnake.utilities import VerboseMessageQueue, DataAcquisitionParameters, error_message_qt
from rattlesnake.environment.abstract_environment import AbstractMetadata
from rattlesnake.environment.abstract_sysid_environment import (
    AbstractSysIdMetadata,
    SystemIdCommands,
    SystemIdUICommands,
)
from rattlesnake.user_interface.ui_utilities import system_identification_ui_path
from rattlesnake.process.abstract_sysid_data_analysis import SysIDDataAnalysisCommands
from rattlesnake.process.data_collector import DataCollectorUICommands
from rattlesnake.process.abstract_sysid_data_analysis import SysIDDataAnalysisUICommands
from abc import ABC, abstractmethod
from multiprocessing.queues import Queue
import netCDF4 as nc4
import numpy as np
import pyqtgraph as pg
import openpyxl
from qtpy import QtWidgets, uic
from scipy.io import loadmat, savemat


class RotatedAxisItem(pg.AxisItem):  # pylint: disable=abstract-method
    """Plot axis labels that can be rotated by some value"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_height = self.height()
        self._angle = None

    def setAngle(self, angle):  # pylint: disable=invalid-name
        """Sets the angle and ensures it's between -180 and 180"""
        self._angle = angle
        self._angle = (self._angle + 180) % 360 - 180

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        """UPdated draw picture method that includes the rotation of the text"""
        profiler = pg.debug.Profiler()
        max_width = 0

        # draw long line along axis
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        # draw ticks
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)
        profiler("draw ticks")

        for rect, flags, text in textSpecs:
            p.save()  # save the painter state

            p.translate(rect.center())  # move coordinate system to center of text rect
            p.rotate(self._angle)  # rotate text
            p.translate(-rect.center())  # revert coordinate system

            x_offset = np.ceil(np.fabs(np.sin(np.radians(self._angle)) * rect.width()))
            if self._angle < 0:
                x_offset = -x_offset
            p.translate(x_offset / 2, 0)  # Move the coordinate system (relatively) downwards

            p.drawText(rect, flags, text)
            p.restore()  # restore the painter state
            offset = np.fabs(x_offset)
            max_width = offset if max_width < offset else max_width

        profiler("draw text")
        #  Adjust the height
        self.setHeight(self._original_height + max_width)

    def boundingRect(self):
        """Sets the bounding rectangle of the item to give more space at the bottom"""
        rect = super().boundingRect()
        rect.adjust(0, 0, 0, 20)  # Add 20 pixels to bottom
        return rect


# region: User Interface
class AbstractSysIdUI(AbstractUI):
    """Abstract User Interface class defining the interface with the controller

    This class is used to define the interface between the User Interface of a
    environment in the controller and the main controller."""

    @abstractmethod
    def __init__(
        self,
        environment_name: str,
        environment_command_queue: VerboseMessageQueue,
        controller_communication_queue: VerboseMessageQueue,
        log_file_queue: Queue,
        system_id_tabwidget: QtWidgets.QTabWidget,
    ):
        """
        Stores data required by the controller to interact with the UI

        This class stores data required by the controller to interact with the
        user interface for a given environment.  This includes the environment
        name and queues to pass information between the controller and
        environment.  It additionally initializes the ``command_map`` which is
        used by the Test Profile functionality to map profile instructions to
        operations on the user interface.


        Parameters
        ----------
        environment_name : str
            The name of the environment
        environment_command_queue : VerboseMessageQueue
            A queue that will provide instructions to the corresponding
            environment
        controller_communication_queue : VerboseMessageQueue
            The queue that relays global communication messages to the controller
        log_file_queue : Queue
            The queue that will be used to put messages to the log file.


        """
        super().__init__(
            environment_name,
            environment_command_queue,
            controller_communication_queue,
            log_file_queue,
        )
        # Add the page to the system id tabwidget
        self.system_id_widget = QtWidgets.QWidget()
        uic.loadUi(system_identification_ui_path, self.system_id_widget)
        system_id_tabwidget.addTab(self.system_id_widget, self.environment_name)
        self.connect_sysid_callbacks()

        self.data_acquisition_parameters = None
        self.environment_parameters = None
        self.frequencies = None
        self.last_time_response = None
        self.last_transfer_function = None
        self.last_response_noise = None
        self.last_reference_noise = None
        self.last_response_cpsd = None
        self.last_reference_cpsd = None
        self.last_coherence = None
        self.last_condition = None
        self.last_kurtosis = None

        self.time_response_plot = self.system_id_widget.time_data_graphicslayout.addPlot(
            row=0, column=0
        )
        self.time_response_plot.setLabel("left", "Response")
        self.time_response_plot.setLabel("bottom", "Time (s)")
        self.time_reference_plot = self.system_id_widget.time_data_graphicslayout.addPlot(
            row=0, column=1
        )
        self.time_reference_plot.setLabel("left", "Reference")
        self.time_reference_plot.setLabel("bottom", "Time (s)")
        self.level_response_plot = self.system_id_widget.levels_graphicslayout.addPlot(
            row=0, column=0
        )
        self.level_response_plot.setLabel("left", "Response PSD")
        self.level_response_plot.setLabel("bottom", "Frequency (Hz)")
        self.level_reference_plot = self.system_id_widget.levels_graphicslayout.addPlot(
            row=0, column=1
        )
        self.level_reference_plot.setLabel("left", "Reference PSD")
        self.level_reference_plot.setLabel("bottom", "Frequency (Hz)")
        self.transfer_function_phase_plot = (
            self.system_id_widget.transfer_function_graphics_layout.addPlot(row=0, column=0)
        )
        self.transfer_function_phase_plot.setLabel("left", "Phase")
        self.transfer_function_phase_plot.setLabel("bottom", "Frequency (Hz)")
        self.transfer_function_magnitude_plot = (
            self.system_id_widget.transfer_function_graphics_layout.addPlot(row=0, column=1)
        )
        self.transfer_function_magnitude_plot.setLabel("left", "Amplitude")
        self.transfer_function_magnitude_plot.setLabel("bottom", "Frequency (Hz)")
        self.impulse_response_plot = self.system_id_widget.impulse_graphicslayout.addPlot(
            row=0, column=0
        )
        self.impulse_response_plot.setLabel("left", "Impulse Response")
        self.impulse_response_plot.setLabel("bottom", "Time (s)")
        self.coherence_plot = self.system_id_widget.coherence_graphicslayout.addPlot(
            row=0, column=0
        )
        self.coherence_plot.setLabel("left", "Multiple Coherence")
        self.coherence_plot.setLabel("bottom", "Frequency (Hz)")
        self.condition_plot = self.system_id_widget.coherence_graphicslayout.addPlot(
            row=0, column=1
        )
        self.condition_plot.setLabel("left", "Condition Number")
        self.condition_plot.setLabel("bottom", "Frequency (Hz)")
        self.coherence_plot.vb.setLimits(yMin=0, yMax=1)
        self.coherence_plot.vb.disableAutoRange(axis="y")
        # Set up kurtosis plots
        self.response_nodes = []
        self.reference_nodes = []
        self.all_response_indices = []
        self.all_reference_indices = []
        self.kurtosis_response_plot = self.system_id_widget.kurtosis_graphicslayout.addPlot(
            row=0, column=0
        )
        self.kurtosis_reference_plot = self.system_id_widget.kurtosis_graphicslayout.addPlot(
            row=0, column=1
        )
        self.kurtosis_response_plot.setLabel("left", "Response")
        self.kurtosis_reference_plot.setLabel("left", "Reference")
        response_axis = RotatedAxisItem("bottom")
        reference_axis = RotatedAxisItem("bottom")
        response_axis.setAngle(-60)
        reference_axis.setAngle(-60)
        self.kurtosis_response_plot.setAxisItems({"bottom": response_axis})
        self.kurtosis_reference_plot.setAxisItems({"bottom": reference_axis})
        for plot in [
            self.level_response_plot,
            self.level_reference_plot,
            self.transfer_function_magnitude_plot,
            self.condition_plot,
        ]:
            plot.setLogMode(False, True)
        self.show_hide_coherence()
        self.show_hide_impulse()
        self.show_hide_levels()
        self.show_hide_time_data()
        self.show_hide_transfer_function()
        self.show_hide_kurtosis()

    def connect_sysid_callbacks(self):
        """Connects the callback functions to the system identification widgets"""
        self.system_id_widget.preview_noise_button.clicked.connect(self.preview_noise)
        self.system_id_widget.preview_system_id_button.clicked.connect(
            self.preview_transfer_function
        )
        self.system_id_widget.start_button.clicked.connect(self.acquire_transfer_function)
        self.system_id_widget.stop_button.clicked.connect(self.stop_system_id)
        self.system_id_widget.select_transfer_function_stream_file_button.clicked.connect(
            self.select_transfer_function_stream_file
        )
        self.system_id_widget.response_selector.itemSelectionChanged.connect(
            self.update_sysid_plots
        )
        self.system_id_widget.reference_selector.itemSelectionChanged.connect(
            self.update_sysid_plots
        )
        self.system_id_widget.coherence_checkbox.stateChanged.connect(self.show_hide_coherence)
        self.system_id_widget.levels_checkbox.stateChanged.connect(self.show_hide_levels)
        self.system_id_widget.time_data_checkbox.stateChanged.connect(self.show_hide_time_data)
        self.system_id_widget.impulse_checkbox.stateChanged.connect(self.show_hide_impulse)
        self.system_id_widget.transfer_function_checkbox.stateChanged.connect(
            self.show_hide_transfer_function
        )
        self.system_id_widget.kurtosis_checkbox.stateChanged.connect(self.show_hide_kurtosis)
        self.system_id_widget.signalTypeComboBox.currentIndexChanged.connect(
            self.update_signal_type
        )
        self.system_id_widget.save_system_id_matrices_button.clicked.connect(
            self.save_sysid_matrix_file
        )
        self.system_id_widget.load_system_id_matrices_button.clicked.connect(
            self.load_sysid_matrix_file
        )

    @abstractmethod
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
        # Store for later
        self.data_acquisition_parameters = data_acquisition_parameters
        self.system_id_widget.highFreqCutoffSpinBox.setMaximum(
            data_acquisition_parameters.sample_rate // 2
        )
        # finish setting up kurtosis plots using node number + direction
        for i, channel in enumerate(self.data_acquisition_parameters.channel_list):
            node = channel.node_number + (
                "" if channel.node_direction is None else channel.node_direction
            )
            if channel.feedback_device is None:
                self.response_nodes.append(node)
                self.all_response_indices.append(i)
            else:
                self.reference_nodes.append(node)
                self.all_reference_indices.append(i)
        response_ax = self.kurtosis_response_plot.getAxis("bottom")
        reference_ax = self.kurtosis_reference_plot.getAxis("bottom")
        response_ax.setTicks([list(enumerate(self.response_nodes))])
        reference_ax.setTicks([list(enumerate(self.reference_nodes))])
        self.system_id_widget.kurtosis_graphicslayout.ci.layout.setColumnStretchFactor(
            0, len(self.all_response_indices) * 2 + len(self.all_reference_indices)
        )
        self.system_id_widget.kurtosis_graphicslayout.ci.layout.setColumnStretchFactor(
            1, len(self.all_reference_indices) * 2 + len(self.all_response_indices)
        )

    @abstractmethod
    def collect_environment_definition_parameters(self) -> AbstractSysIdMetadata:
        """
        Collect the parameters from the user interface defining the environment

        Returns
        -------
        AbstractSysIdMetadata
            A metadata or parameters object containing the parameters defining
            the corresponding environment.

        """

    def update_sysid_metadata(self, metadata: AbstractSysIdMetadata):
        """Updates the provided system identification metadata based on current UI widget values"""
        metadata.sysid_frame_size = self.system_id_widget.samplesPerFrameSpinBox.value()
        metadata.sysid_averaging_type = self.system_id_widget.averagingTypeComboBox.itemText(
            self.system_id_widget.averagingTypeComboBox.currentIndex()
        )
        metadata.sysid_noise_averages = self.system_id_widget.noiseAveragesSpinBox.value()
        metadata.sysid_averages = self.system_id_widget.systemIDAveragesSpinBox.value()
        metadata.sysid_exponential_averaging_coefficient = (
            self.system_id_widget.averagingCoefficientDoubleSpinBox.value()
        )
        metadata.sysid_estimator = self.system_id_widget.estimatorComboBox.itemText(
            self.system_id_widget.estimatorComboBox.currentIndex()
        )
        metadata.sysid_level = self.system_id_widget.levelDoubleSpinBox.value()
        metadata.sysid_level_ramp_time = self.system_id_widget.levelRampTimeDoubleSpinBox.value()
        metadata.sysid_signal_type = self.system_id_widget.signalTypeComboBox.itemText(
            self.system_id_widget.signalTypeComboBox.currentIndex()
        )
        metadata.sysid_window = self.system_id_widget.windowComboBox.itemText(
            self.system_id_widget.windowComboBox.currentIndex()
        )
        metadata.sysid_overlap = (
            self.system_id_widget.overlapDoubleSpinBox.value() / 100
            if metadata.sysid_signal_type == "Random"
            else 0.0
        )
        metadata.sysid_burst_on = self.system_id_widget.onFractionDoubleSpinBox.value() / 100
        metadata.sysid_pretrigger = self.system_id_widget.pretriggerDoubleSpinBox.value() / 100
        metadata.sysid_burst_ramp_fraction = (
            self.system_id_widget.rampFractionDoubleSpinBox.value() / 100
        )
        metadata.sysid_low_frequency_cutoff = self.system_id_widget.lowFreqCutoffSpinBox.value()
        metadata.sysid_high_frequency_cutoff = self.system_id_widget.highFreqCutoffSpinBox.value()
        # for key in dir(metadata):
        #     if '__' == key[:2]:
        #         continue
        #     print('Key: {:}'.format(key))
        #     print('Value: {:}'.format(getattr(metadata,key)))

    @property
    @abstractmethod
    def initialized_control_names(self):
        """Names of control channels that have been initialized and will be used in displays"""

    @property
    @abstractmethod
    def initialized_output_names(self):
        """Names of output channels that have been initialized and will be used in displays"""

    @abstractmethod
    def initialize_environment(self) -> AbstractMetadata:
        """
        Update the user interface with environment parameters

        This function is called when the Environment parameters are initialized.
        This function should set up the user interface accordingly.  It must
        return the parameters class of the environment that inherits from
        AbstractMetadata.

        Returns
        -------
        AbstractMetadata
            An AbstractMetadata-inheriting object that contains the parameters
            defining the environment.

        """
        self.environment_parameters = self.collect_environment_definition_parameters()
        self.update_sysid_metadata(self.environment_parameters)
        self.system_id_widget.reference_selector.blockSignals(True)
        self.system_id_widget.response_selector.blockSignals(True)
        self.system_id_widget.reference_selector.clear()
        self.system_id_widget.response_selector.clear()
        for i, control_name in enumerate(self.initialized_control_names):
            self.system_id_widget.response_selector.addItem(f"{i + 1}: {control_name}")
        for i, drive_name in enumerate(self.initialized_output_names):
            self.system_id_widget.reference_selector.addItem(f"{i + 1}: {drive_name}")
        self.system_id_widget.reference_selector.blockSignals(False)
        self.system_id_widget.response_selector.blockSignals(False)
        self.system_id_widget.reference_selector.setCurrentRow(0)
        self.system_id_widget.response_selector.setCurrentRow(0)
        self.update_signal_type()
        return self.environment_parameters

    def preview_noise(self):
        """Starts the noise preview"""
        self.log("Starting Noise Preview")
        self.update_sysid_metadata(self.environment_parameters)
        for widget in [
            self.system_id_widget.preview_noise_button,
            self.system_id_widget.preview_system_id_button,
            self.system_id_widget.start_button,
            self.system_id_widget.samplesPerFrameSpinBox,
            self.system_id_widget.averagingTypeComboBox,
            self.system_id_widget.noiseAveragesSpinBox,
            self.system_id_widget.systemIDAveragesSpinBox,
            self.system_id_widget.averagingCoefficientDoubleSpinBox,
            self.system_id_widget.estimatorComboBox,
            self.system_id_widget.levelDoubleSpinBox,
            self.system_id_widget.signalTypeComboBox,
            self.system_id_widget.windowComboBox,
            self.system_id_widget.overlapDoubleSpinBox,
            self.system_id_widget.onFractionDoubleSpinBox,
            self.system_id_widget.pretriggerDoubleSpinBox,
            self.system_id_widget.rampFractionDoubleSpinBox,
            self.system_id_widget.stream_transfer_function_data_checkbox,
            self.system_id_widget.select_transfer_function_stream_file_button,
            self.system_id_widget.transfer_function_stream_file_display,
            self.system_id_widget.levelRampTimeDoubleSpinBox,
            self.system_id_widget.save_system_id_matrices_button,
            self.system_id_widget.load_system_id_matrices_button,
            self.system_id_widget.lowFreqCutoffSpinBox,
            self.system_id_widget.highFreqCutoffSpinBox,
        ]:
            widget.setEnabled(False)
        for widget in [self.system_id_widget.stop_button]:
            widget.setEnabled(True)
        self.environment_command_queue.put(
            self.log_name, (SystemIdCommands.PREVIEW_NOISE, self.environment_parameters)
        )

    def preview_transfer_function(self):
        """Starts previewing the system identification transfer function calculation"""
        self.log("Starting System ID Preview")
        self.update_sysid_metadata(self.environment_parameters)
        for widget in [
            self.system_id_widget.preview_noise_button,
            self.system_id_widget.preview_system_id_button,
            self.system_id_widget.start_button,
            self.system_id_widget.samplesPerFrameSpinBox,
            self.system_id_widget.averagingTypeComboBox,
            self.system_id_widget.noiseAveragesSpinBox,
            self.system_id_widget.systemIDAveragesSpinBox,
            self.system_id_widget.averagingCoefficientDoubleSpinBox,
            self.system_id_widget.estimatorComboBox,
            self.system_id_widget.levelDoubleSpinBox,
            self.system_id_widget.signalTypeComboBox,
            self.system_id_widget.windowComboBox,
            self.system_id_widget.overlapDoubleSpinBox,
            self.system_id_widget.onFractionDoubleSpinBox,
            self.system_id_widget.pretriggerDoubleSpinBox,
            self.system_id_widget.rampFractionDoubleSpinBox,
            self.system_id_widget.stream_transfer_function_data_checkbox,
            self.system_id_widget.select_transfer_function_stream_file_button,
            self.system_id_widget.transfer_function_stream_file_display,
            self.system_id_widget.levelRampTimeDoubleSpinBox,
            self.system_id_widget.save_system_id_matrices_button,
            self.system_id_widget.load_system_id_matrices_button,
            self.system_id_widget.lowFreqCutoffSpinBox,
            self.system_id_widget.highFreqCutoffSpinBox,
        ]:
            widget.setEnabled(False)
        for widget in [self.system_id_widget.stop_button]:
            widget.setEnabled(True)
        self.environment_command_queue.put(
            self.log_name,
            (SystemIdCommands.PREVIEW_TRANSFER_FUNCTION, (self.environment_parameters)),
        )

    def acquire_transfer_function(self):
        """Starts the acquisition phase of the controller"""
        self.log("Starting System ID")
        self.update_sysid_metadata(self.environment_parameters)
        for widget in [
            self.system_id_widget.preview_noise_button,
            self.system_id_widget.preview_system_id_button,
            self.system_id_widget.start_button,
            self.system_id_widget.samplesPerFrameSpinBox,
            self.system_id_widget.averagingTypeComboBox,
            self.system_id_widget.noiseAveragesSpinBox,
            self.system_id_widget.systemIDAveragesSpinBox,
            self.system_id_widget.averagingCoefficientDoubleSpinBox,
            self.system_id_widget.estimatorComboBox,
            self.system_id_widget.levelDoubleSpinBox,
            self.system_id_widget.signalTypeComboBox,
            self.system_id_widget.windowComboBox,
            self.system_id_widget.overlapDoubleSpinBox,
            self.system_id_widget.onFractionDoubleSpinBox,
            self.system_id_widget.pretriggerDoubleSpinBox,
            self.system_id_widget.rampFractionDoubleSpinBox,
            self.system_id_widget.stream_transfer_function_data_checkbox,
            self.system_id_widget.select_transfer_function_stream_file_button,
            self.system_id_widget.transfer_function_stream_file_display,
            self.system_id_widget.levelRampTimeDoubleSpinBox,
            self.system_id_widget.save_system_id_matrices_button,
            self.system_id_widget.load_system_id_matrices_button,
            self.system_id_widget.lowFreqCutoffSpinBox,
            self.system_id_widget.highFreqCutoffSpinBox,
        ]:
            widget.setEnabled(False)
        for widget in [self.system_id_widget.stop_button]:
            widget.setEnabled(True)
        if self.system_id_widget.stream_transfer_function_data_checkbox.isChecked():
            stream_name = self.system_id_widget.transfer_function_stream_file_display.text()
        else:
            stream_name = None
        self.environment_command_queue.put(
            self.log_name,
            (
                SystemIdCommands.START_SYSTEM_ID,
                (self.environment_parameters, stream_name),
            ),
        )

    def stop_system_id(self):
        """Stops the system identification"""
        self.log("Stopping System ID")
        self.system_id_widget.stop_button.setEnabled(False)
        self.environment_command_queue.put(
            self.log_name, (SystemIdCommands.STOP_SYSTEM_ID, (True, True))
        )

    def select_transfer_function_stream_file(self):
        """Select a file to save transfer function data to"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.system_id_widget,
            "Select NetCDF File to Save Transfer Function Data",
            filter="NetCDF File (*.nc4)",
        )
        if filename == "":
            return
        self.system_id_widget.transfer_function_stream_file_display.setText(filename)
        self.system_id_widget.stream_transfer_function_data_checkbox.setChecked(True)

    def update_sysid_plots(
        self,
        update_time=True,
        update_transfer_function=True,
        update_noise=True,
        update_kurtosis=True,
    ):
        """Updates the plots on the system identification window

        Parameters
        ----------
        update_time : bool, optional
            If True, updates the time hitory plots, by default True
        update_transfer_function : bool, optional
            If True, updates the transfer function plots, by default True
        update_noise : bool, optional
            If True, updates the noise plots, by default True
        update_kurtosis : bool, optional
            If True, updates the kurtosis bar graph, by default True
        """
        # Figure out the selected entries
        response_indices = [
            i
            for i in range(self.system_id_widget.response_selector.count())
            if self.system_id_widget.response_selector.item(i).isSelected()
        ]
        reference_indices = [
            i
            for i in range(self.system_id_widget.reference_selector.count())
            if self.system_id_widget.reference_selector.item(i).isSelected()
        ]
        # print(response_indices)
        # print(reference_indices)
        if update_time:
            self.time_response_plot.clear()
            self.time_reference_plot.clear()
            if self.last_time_response is not None:
                response_frame_indices = np.array(
                    self.environment_parameters.response_channel_indices
                )[response_indices]
                reference_frame_indices = np.array(
                    self.environment_parameters.reference_channel_indices
                )[reference_indices]
                response_time_data = self.last_time_response[response_frame_indices]
                reference_time_data = self.last_time_response[reference_frame_indices]
                times = (
                    np.arange(response_time_data.shape[-1])
                    / self.data_acquisition_parameters.sample_rate
                )
                for i, time_data in enumerate(response_time_data):
                    self.time_response_plot.plot(times, time_data, pen=i)
                for i, time_data in enumerate(reference_time_data):
                    self.time_reference_plot.plot(times, time_data, pen=i)
        if update_transfer_function:
            self.transfer_function_phase_plot.clear()
            self.transfer_function_magnitude_plot.clear()
            self.condition_plot.clear()
            self.coherence_plot.clear()
            self.impulse_response_plot.clear()
            if (
                self.last_transfer_function is not None
                and len(response_indices) > 0
                and len(reference_indices) > 0
            ):
                # print(self.last_transfer_function)
                # print(np.array(response_indices)[:,np.newaxis])
                # print(np.array(reference_indices))
                frf_section = np.reshape(
                    self.last_transfer_function[
                        ...,
                        np.array(response_indices)[:, np.newaxis],
                        np.array(reference_indices),
                    ],
                    (self.frequencies.size, -1),
                ).T
                impulse_response = np.fft.irfft(frf_section, axis=-1)
                for i, (frf, imp) in enumerate(zip(frf_section, impulse_response)):
                    self.transfer_function_phase_plot.plot(
                        self.frequencies, np.angle(frf) * 180 / np.pi, pen=i
                    )
                    self.transfer_function_magnitude_plot.plot(self.frequencies, np.abs(frf), pen=i)
                    self.impulse_response_plot.plot(
                        np.arange(imp.size) / self.environment_parameters.sample_rate,
                        imp,
                        pen=i,
                    )
                for i, coherence in enumerate(self.last_coherence[..., response_indices].T):
                    self.coherence_plot.plot(self.frequencies, coherence, pen=i)
            if self.last_condition is not None:
                self.condition_plot.plot(self.frequencies, self.last_condition, pen=0)
        if update_noise:
            reference_noise = (
                None
                if self.last_reference_noise is None or len(reference_indices) == 0
                else self.last_reference_noise[..., reference_indices, reference_indices].real
            )
            response_noise = (
                None
                if self.last_response_noise is None or len(response_indices) == 0
                else self.last_response_noise[..., response_indices, response_indices].real
            )
            reference_level = (
                None
                if self.last_reference_cpsd is None or len(reference_indices) == 0
                else self.last_reference_cpsd[..., reference_indices, reference_indices].real
            )
            response_level = (
                None
                if self.last_response_cpsd is None or len(response_indices) == 0
                else self.last_response_cpsd[..., response_indices, response_indices].real
            )
            self.level_reference_plot.clear()
            self.level_response_plot.clear()
            for i in range(len(reference_indices)):
                if reference_noise is not None:
                    self.level_reference_plot.plot(self.frequencies, reference_noise[:, i], pen=i)
                if reference_level is not None:
                    try:
                        self.level_reference_plot.plot(
                            self.frequencies, reference_level[:, i], pen=i
                        )
                    except Exception:
                        pass
            for i in range(len(response_indices)):
                if response_noise is not None:
                    self.level_response_plot.plot(self.frequencies, response_noise[:, i], pen=i)
                if response_level is not None:
                    try:
                        self.level_response_plot.plot(self.frequencies, response_level[:, i], pen=i)
                    except Exception:
                        pass

        if update_kurtosis:
            self.kurtosis_response_plot.clear()
            self.kurtosis_reference_plot.clear()
            if self.last_kurtosis is not None:
                response_kurtosis = self.last_kurtosis[self.all_response_indices]
                reference_kurtosis = self.last_kurtosis[self.all_reference_indices]
                response_bar = pg.BarGraphItem(
                    x=range(len(self.response_nodes)),
                    height=response_kurtosis,
                    width=0.5,
                    pen="r",
                    brush="r",
                )
                reference_bar = pg.BarGraphItem(
                    x=range(len(self.reference_nodes)),
                    height=reference_kurtosis,
                    width=0.5,
                    pen="r",
                    brush="r",
                )
                self.kurtosis_response_plot.addItem(response_bar)
                self.kurtosis_reference_plot.addItem(reference_bar)

    def show_hide_coherence(self):
        """Sets the visibility of the coherence plots"""
        if self.system_id_widget.coherence_checkbox.isChecked():
            self.system_id_widget.coherence_groupbox.show()
        else:
            self.system_id_widget.coherence_groupbox.hide()

    def show_hide_levels(self):
        """Sets the visibility of the level plots"""
        if self.system_id_widget.levels_checkbox.isChecked():
            self.system_id_widget.levels_groupbox.show()
        else:
            self.system_id_widget.levels_groupbox.hide()

    def show_hide_time_data(self):
        """Sets the visibility of the time data plots"""
        if self.system_id_widget.time_data_checkbox.isChecked():
            self.system_id_widget.time_data_groupbox.show()
        else:
            self.system_id_widget.time_data_groupbox.hide()

    def show_hide_transfer_function(self):
        """Sets the visibility of the transfer function plots"""
        if self.system_id_widget.transfer_function_checkbox.isChecked():
            self.system_id_widget.transfer_function_groupbox.show()
        else:
            self.system_id_widget.transfer_function_groupbox.hide()

    def show_hide_impulse(self):
        """Sets the visibility of the impulse response plots"""
        if self.system_id_widget.impulse_checkbox.isChecked():
            self.system_id_widget.impulse_groupbox.show()
        else:
            self.system_id_widget.impulse_groupbox.hide()

    def show_hide_kurtosis(self):
        """Sets the visibility of the kurtosis plots"""
        if self.system_id_widget.kurtosis_checkbox.isChecked():
            self.system_id_widget.kurtosis_groupbox.show()
        else:
            self.system_id_widget.kurtosis_groupbox.hide()

    def update_signal_type(self):
        """Updates the UI widgets based on the type of signal that has been selected"""
        if self.system_id_widget.signalTypeComboBox.currentIndex() == 0:  # Random
            self.system_id_widget.windowComboBox.setCurrentIndex(0)
            self.system_id_widget.overlapDoubleSpinBox.show()
            self.system_id_widget.overlapLabel.show()
            self.system_id_widget.onFractionLabel.hide()
            self.system_id_widget.onFractionDoubleSpinBox.hide()
            self.system_id_widget.pretriggerLabel.hide()
            self.system_id_widget.pretriggerDoubleSpinBox.hide()
            self.system_id_widget.rampFractionLabel.hide()
            self.system_id_widget.rampFractionDoubleSpinBox.hide()
            self.system_id_widget.bandwidthLabel.show()
            self.system_id_widget.lowFreqCutoffSpinBox.show()
            self.system_id_widget.highFreqCutoffSpinBox.show()
        elif self.system_id_widget.signalTypeComboBox.currentIndex() == 1:  # Pseudorandom
            self.system_id_widget.windowComboBox.setCurrentIndex(1)
            self.system_id_widget.overlapDoubleSpinBox.hide()
            self.system_id_widget.overlapLabel.hide()
            self.system_id_widget.onFractionLabel.hide()
            self.system_id_widget.onFractionDoubleSpinBox.hide()
            self.system_id_widget.pretriggerLabel.hide()
            self.system_id_widget.pretriggerDoubleSpinBox.hide()
            self.system_id_widget.rampFractionLabel.hide()
            self.system_id_widget.rampFractionDoubleSpinBox.hide()
            self.system_id_widget.bandwidthLabel.show()
            self.system_id_widget.lowFreqCutoffSpinBox.show()
            self.system_id_widget.highFreqCutoffSpinBox.show()
        elif self.system_id_widget.signalTypeComboBox.currentIndex() == 2:  # Burst
            self.system_id_widget.windowComboBox.setCurrentIndex(1)
            self.system_id_widget.overlapDoubleSpinBox.hide()
            self.system_id_widget.overlapLabel.hide()
            self.system_id_widget.onFractionLabel.show()
            self.system_id_widget.onFractionDoubleSpinBox.show()
            self.system_id_widget.pretriggerLabel.show()
            self.system_id_widget.pretriggerDoubleSpinBox.show()
            self.system_id_widget.rampFractionLabel.show()
            self.system_id_widget.rampFractionDoubleSpinBox.show()
            self.system_id_widget.bandwidthLabel.show()
            self.system_id_widget.lowFreqCutoffSpinBox.show()
            self.system_id_widget.highFreqCutoffSpinBox.show()
        elif self.system_id_widget.signalTypeComboBox.currentIndex() == 3:  # Chirp
            self.system_id_widget.windowComboBox.setCurrentIndex(1)
            self.system_id_widget.overlapDoubleSpinBox.hide()
            self.system_id_widget.overlapLabel.hide()
            self.system_id_widget.onFractionLabel.hide()
            self.system_id_widget.onFractionDoubleSpinBox.hide()
            self.system_id_widget.pretriggerLabel.hide()
            self.system_id_widget.pretriggerDoubleSpinBox.hide()
            self.system_id_widget.rampFractionLabel.hide()
            self.system_id_widget.rampFractionDoubleSpinBox.hide()
            self.system_id_widget.bandwidthLabel.hide()
            self.system_id_widget.lowFreqCutoffSpinBox.hide()
            self.system_id_widget.highFreqCutoffSpinBox.hide()

    @abstractmethod
    def retrieve_metadata(
        self,
        netcdf_handle: nc4._netCDF4.Dataset,  # pylint: disable=c-extension-no-member
        environment_name: str = None,
    ) -> nc4._netCDF4.Group:  # pylint: disable=c-extension-no-member
        """Collects environment parameters from a netCDF dataset.

        This function retrieves parameters from a netCDF dataset that was written
        by the controller during streaming.  It must populate the widgets
        in the user interface with the proper information.

        This function is the "read" counterpart to the store_to_netcdf
        function in the AbstractMetadata class, which will write parameters to
        the netCDF file to document the metadata.

        Note that the entire dataset is passed to this function, so the function
        should collect parameters pertaining to the environment from a Group
        in the dataset sharing the environment's name, e.g.

        Parameters
        ----------
        netcdf_handle : nc4._netCDF4.Dataset :
            The netCDF dataset from which the data will be read.  It should have
            a group name with the enviroment's name.

        environment_name : str : (optional)
            The netCDF group name from which the data will be read. This will override
            the current environment's name if given.

        Returns
        -------
        group : nc4._netCDF4.Group
            The netCDF group that was used to set the system ID parameters
        """
        # Get the group
        group = netcdf_handle.groups[
            self.environment_name if environment_name is None else environment_name
        ]
        self.system_id_widget.samplesPerFrameSpinBox.setValue(group.sysid_frame_size)
        self.system_id_widget.averagingTypeComboBox.setCurrentIndex(
            self.system_id_widget.averagingTypeComboBox.findText(group.sysid_averaging_type)
        )
        self.system_id_widget.noiseAveragesSpinBox.setValue(group.sysid_noise_averages)
        self.system_id_widget.systemIDAveragesSpinBox.setValue(group.sysid_averages)
        self.system_id_widget.averagingCoefficientDoubleSpinBox.setValue(
            group.sysid_exponential_averaging_coefficient
        )
        self.system_id_widget.estimatorComboBox.setCurrentIndex(
            self.system_id_widget.estimatorComboBox.findText(group.sysid_estimator)
        )
        self.system_id_widget.levelDoubleSpinBox.setValue(group.sysid_level)
        self.system_id_widget.levelRampTimeDoubleSpinBox.setValue(group.sysid_level_ramp_time)
        self.system_id_widget.signalTypeComboBox.setCurrentIndex(
            self.system_id_widget.signalTypeComboBox.findText(group.sysid_signal_type)
        )
        self.system_id_widget.windowComboBox.setCurrentIndex(
            self.system_id_widget.windowComboBox.findText(group.sysid_window)
        )
        self.system_id_widget.overlapDoubleSpinBox.setValue(group.sysid_overlap * 100)
        self.system_id_widget.onFractionDoubleSpinBox.setValue(group.sysid_burst_on * 100)
        self.system_id_widget.pretriggerDoubleSpinBox.setValue(group.sysid_pretrigger * 100)
        self.system_id_widget.rampFractionDoubleSpinBox.setValue(
            group.sysid_burst_ramp_fraction * 100
        )
        if hasattr(group, "sysid_low_frequency_cutoff"):
            self.system_id_widget.lowFreqCutoffSpinBox.setValue(group.sysid_low_frequency_cutoff)
        if hasattr(group, "sysid_high_frequency_cutoff"):
            self.system_id_widget.highFreqCutoffSpinBox.setValue(group.sysid_high_frequency_cutoff)
        return group

    @abstractmethod
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
        message, data = queue_data
        self.log(f"Got GUI Message {message}")
        # print('Update GUI Got {:}'.format(message))
        if message == DataCollectorUICommands.TIME_FRAME:
            self.last_time_response, accept = data
            self.update_sysid_plots(
                update_time=True,
                update_transfer_function=False,
                update_noise=False,
                update_kurtosis=False,
            )
        elif message == DataCollectorUICommands.KURTOSIS:
            self.last_kurtosis = data
            self.update_sysid_plots(
                update_time=False,
                update_transfer_function=False,
                update_noise=False,
                update_kurtosis=True,
            )
        elif message == SysIDDataAnalysisUICommands.NOISE_UPDATE:
            (
                frames,
                total_frames,
                self.frequencies,
                self.last_response_noise,
                self.last_reference_noise,
            ) = data
            self.update_sysid_plots(
                update_time=False,
                update_transfer_function=False,
                update_noise=True,
                update_kurtosis=False,
            )
            self.system_id_widget.current_frames_spinbox.setValue(frames)
            self.system_id_widget.total_frames_spinbox.setValue(total_frames)
            self.system_id_widget.progressBar.setValue(int(frames / total_frames * 100))
        elif message == SysIDDataAnalysisUICommands.SYS_ID_UPDATE:
            (
                frames,
                total_frames,
                self.frequencies,
                self.last_transfer_function,
                self.last_coherence,
                self.last_response_cpsd,
                self.last_reference_cpsd,
                self.last_condition,
            ) = data
            # print(self.last_transfer_function.shape)
            # print(self.last_coherence.shape)
            # print(self.last_response_cpsd.shape)
            # print(self.last_reference_cpsd.shape)
            self.update_sysid_plots(
                update_time=False,
                update_transfer_function=True,
                update_noise=True,
                update_kurtosis=False,
            )
            self.system_id_widget.current_frames_spinbox.setValue(frames)
            self.system_id_widget.total_frames_spinbox.setValue(total_frames)
            self.system_id_widget.progressBar.setValue(int(frames / total_frames * 100))
        elif message == SystemIdUICommands.ENABLE_SYSTEM_ID:
            for widget in [
                self.system_id_widget.preview_noise_button,
                self.system_id_widget.preview_system_id_button,
                self.system_id_widget.start_button,
                self.system_id_widget.samplesPerFrameSpinBox,
                self.system_id_widget.averagingTypeComboBox,
                self.system_id_widget.noiseAveragesSpinBox,
                self.system_id_widget.systemIDAveragesSpinBox,
                self.system_id_widget.averagingCoefficientDoubleSpinBox,
                self.system_id_widget.estimatorComboBox,
                self.system_id_widget.levelDoubleSpinBox,
                self.system_id_widget.signalTypeComboBox,
                self.system_id_widget.windowComboBox,
                self.system_id_widget.overlapDoubleSpinBox,
                self.system_id_widget.onFractionDoubleSpinBox,
                self.system_id_widget.pretriggerDoubleSpinBox,
                self.system_id_widget.rampFractionDoubleSpinBox,
                self.system_id_widget.stream_transfer_function_data_checkbox,
                self.system_id_widget.select_transfer_function_stream_file_button,
                self.system_id_widget.transfer_function_stream_file_display,
                self.system_id_widget.levelRampTimeDoubleSpinBox,
                self.system_id_widget.save_system_id_matrices_button,
                self.system_id_widget.load_system_id_matrices_button,
                self.system_id_widget.lowFreqCutoffSpinBox,
                self.system_id_widget.highFreqCutoffSpinBox,
            ]:
                widget.setEnabled(True)
            for widget in [self.system_id_widget.stop_button]:
                widget.setEnabled(False)
        elif message == SystemIdUICommands.DISABLE_SYSTEM_ID:
            for widget in [
                self.system_id_widget.preview_noise_button,
                self.system_id_widget.preview_system_id_button,
                self.system_id_widget.start_button,
                self.system_id_widget.samplesPerFrameSpinBox,
                self.system_id_widget.averagingTypeComboBox,
                self.system_id_widget.noiseAveragesSpinBox,
                self.system_id_widget.systemIDAveragesSpinBox,
                self.system_id_widget.averagingCoefficientDoubleSpinBox,
                self.system_id_widget.estimatorComboBox,
                self.system_id_widget.levelDoubleSpinBox,
                self.system_id_widget.signalTypeComboBox,
                self.system_id_widget.windowComboBox,
                self.system_id_widget.overlapDoubleSpinBox,
                self.system_id_widget.onFractionDoubleSpinBox,
                self.system_id_widget.pretriggerDoubleSpinBox,
                self.system_id_widget.rampFractionDoubleSpinBox,
                self.system_id_widget.stream_transfer_function_data_checkbox,
                self.system_id_widget.select_transfer_function_stream_file_button,
                self.system_id_widget.transfer_function_stream_file_display,
                self.system_id_widget.levelRampTimeDoubleSpinBox,
                self.system_id_widget.save_system_id_matrices_button,
                self.system_id_widget.load_system_id_matrices_button,
                self.system_id_widget.lowFreqCutoffSpinBox,
                self.system_id_widget.highFreqCutoffSpinBox,
            ]:
                widget.setEnabled(False)
            for widget in [self.system_id_widget.stop_button]:
                widget.setEnabled(True)
        else:
            return False
        return True

    @staticmethod
    @abstractmethod
    def create_environment_template(
        environment_name: str, workbook: openpyxl.workbook.workbook.Workbook
    ):
        """Creates a template worksheet in an Excel workbook defining the
        environment.

        This function creates a template worksheet in an Excel workbook that
        when filled out could be read by the controller to re-create the
        environment.

        This function is the "write" counterpart to the
        ``set_parameters_from_template`` function in the ``AbstractUI`` class,
        which reads the values from the template file to populate the user
        interface.

        Parameters
        ----------
        environment_name : str :
            The name of the environment that will specify the worksheet's name
        workbook : openpyxl.workbook.workbook.Workbook :
            A reference to an ``openpyxl`` workbook.

        """

    @abstractmethod
    def set_parameters_from_template(self, worksheet: openpyxl.worksheet.worksheet.Worksheet):
        """
        Collects parameters for the user interface from the Excel template file

        This function reads a filled out template worksheet to create an
        environment.  Cells on this worksheet contain parameters needed to
        specify the environment, so this function should read those cells and
        update the UI widgets with those parameters.

        This function is the "read" counterpart to the
        ``create_environment_template`` function in the ``AbstractUI`` class,
        which writes a template file that can be filled out by a user.


        Parameters
        ----------
        worksheet : openpyxl.worksheet.worksheet.Worksheet
            An openpyxl worksheet that contains the environment template.
            Cells on this worksheet should contain the parameters needed for the
            user interface.

        """

    def save_sysid_matrix_file(self):
        """Saves out system identification data to a file"""
        if self.last_transfer_function is None or self.last_response_noise is None:
            error_message_qt(
                "Run System Identification First!",
                "System Identification Matrices not yet created.\n\n"
                "Run System Identification First!",
            )
            return
        filename, file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self.system_id_widget,
            "Select File to Save Transfer Function Matrices",
            filter="NetCDF File (*.nc4);;MatLab File (*.mat);;Numpy File (*.npz)",
        )
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
        if file_filter == "NetCDF File (*.nc4)":
            netcdf_handle = nc4.Dataset(  # pylint: disable=no-member
                filename, "w", format="NETCDF4", clobber=True
            )
            # Create dimensions
            netcdf_handle.createDimension(
                "response_channels", len(self.data_acquisition_parameters.channel_list)
            )

            netcdf_handle.createDimension(
                "num_environments",
                len(self.data_acquisition_parameters.environment_names),
            )
            # Create attributes
            netcdf_handle.file_version = "3.0.0"
            netcdf_handle.sample_rate = self.data_acquisition_parameters.sample_rate
            netcdf_handle.time_per_write = (
                self.data_acquisition_parameters.samples_per_write
                / self.data_acquisition_parameters.output_sample_rate
            )
            netcdf_handle.time_per_read = (
                self.data_acquisition_parameters.samples_per_read
                / self.data_acquisition_parameters.sample_rate
            )
            netcdf_handle.hardware = self.data_acquisition_parameters.hardware
            netcdf_handle.hardware_file = (
                "None"
                if self.data_acquisition_parameters.hardware_file is None
                else self.data_acquisition_parameters.hardware_file
            )
            netcdf_handle.output_oversample = self.data_acquisition_parameters.output_oversample
            for (
                name,
                value,
            ) in self.data_acquisition_parameters.extra_parameters.items():
                setattr(netcdf_handle, name, value)
            # Create Variables
            var = netcdf_handle.createVariable("environment_names", str, ("num_environments",))
            this_environment_index = None
            for i, name in enumerate(self.data_acquisition_parameters.environment_names):
                var[i] = name
                if name == self.environment_name:
                    this_environment_index = i
            var = netcdf_handle.createVariable(
                "environment_active_channels",
                "i1",
                ("response_channels", "num_environments"),
            )
            var[...] = self.data_acquisition_parameters.environment_active_channels.astype("int8")[
                self.data_acquisition_parameters.environment_active_channels[
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
                    for channel in self.data_acquisition_parameters.channel_list
                ]
                if netcdf_datatype == "i1":
                    channel_data = np.array([1 if val else 0 for val in channel_data])
                else:
                    channel_data = ["" if val is None else val for val in channel_data]
                for i, cd in enumerate(channel_data):
                    var[i] = cd
            group_handle = netcdf_handle.createGroup(self.environment_name)
            self.environment_parameters.store_to_netcdf(group_handle)
            try:
                group_handle.createDimension(
                    "sysid_control_channels", self.last_transfer_function.shape[1]
                )
            except RuntimeError:
                pass
            try:
                group_handle.createDimension(
                    "sysid_output_channels", self.last_transfer_function.shape[2]
                )
            except RuntimeError:
                pass
            try:
                group_handle.createDimension(
                    "sysid_fft_lines", self.last_transfer_function.shape[0]
                )
            except RuntimeError:
                pass
            var = group_handle.createVariable(
                "frf_data_real",
                "f8",
                ("sysid_fft_lines", "sysid_control_channels", "sysid_output_channels"),
            )
            var[...] = self.last_transfer_function.real
            var = group_handle.createVariable(
                "frf_data_imag",
                "f8",
                ("sysid_fft_lines", "sysid_control_channels", "sysid_output_channels"),
            )
            var[...] = self.last_transfer_function.imag
            var = group_handle.createVariable(
                "frf_coherence", "f8", ("sysid_fft_lines", "sysid_control_channels")
            )
            var[...] = self.last_coherence.real
            var = group_handle.createVariable(
                "response_cpsd_real",
                "f8",
                ("sysid_fft_lines", "sysid_control_channels", "sysid_control_channels"),
            )
            var[...] = self.last_response_cpsd.real
            var = group_handle.createVariable(
                "response_cpsd_imag",
                "f8",
                ("sysid_fft_lines", "sysid_control_channels", "sysid_control_channels"),
            )
            var[...] = self.last_response_cpsd.imag
            var = group_handle.createVariable(
                "reference_cpsd_real",
                "f8",
                ("sysid_fft_lines", "sysid_output_channels", "sysid_output_channels"),
            )
            var[...] = self.last_reference_cpsd.real
            var = group_handle.createVariable(
                "reference_cpsd_imag",
                "f8",
                ("sysid_fft_lines", "sysid_output_channels", "sysid_output_channels"),
            )
            var[...] = self.last_reference_cpsd.imag
            var = group_handle.createVariable(
                "response_noise_cpsd_real",
                "f8",
                ("sysid_fft_lines", "sysid_control_channels", "sysid_control_channels"),
            )
            var[...] = self.last_response_noise.real
            var = group_handle.createVariable(
                "response_noise_cpsd_imag",
                "f8",
                ("sysid_fft_lines", "sysid_control_channels", "sysid_control_channels"),
            )
            var[...] = self.last_response_noise.imag
            var = group_handle.createVariable(
                "reference_noise_cpsd_real",
                "f8",
                ("sysid_fft_lines", "sysid_output_channels", "sysid_output_channels"),
            )
            var[...] = self.last_reference_noise.real
            var = group_handle.createVariable(
                "reference_noise_cpsd_imag",
                "f8",
                ("sysid_fft_lines", "sysid_output_channels", "sysid_output_channels"),
            )
            var[...] = self.last_reference_noise.imag
        else:
            field_dict = {}
            field_dict["version"] = "3.0.0"
            field_dict["sample_rate"] = self.data_acquisition_parameters.sample_rate
            field_dict["time_per_write"] = (
                self.data_acquisition_parameters.samples_per_write
                / self.data_acquisition_parameters.output_sample_rate
            )
            field_dict["time_per_read"] = (
                self.data_acquisition_parameters.samples_per_read
                / self.data_acquisition_parameters.sample_rate
            )
            field_dict["hardware"] = self.data_acquisition_parameters.hardware
            field_dict["hardware_file"] = (
                "None"
                if self.data_acquisition_parameters.hardware_file is None
                else self.data_acquisition_parameters.hardware_file
            )
            field_dict["output_oversample"] = self.data_acquisition_parameters.output_oversample
            field_dict["frf_data"] = self.last_transfer_function
            field_dict["response_cpsd"] = self.last_response_cpsd
            field_dict["reference_cpsd"] = self.last_reference_cpsd
            field_dict["coherence"] = self.last_coherence
            field_dict["response_noise_cpsd"] = self.last_response_noise
            field_dict["reference_noise_cpsd"] = self.last_reference_noise
            field_dict["response_indices"] = self.environment_parameters.response_channel_indices
            field_dict["reference_indices"] = self.environment_parameters.reference_channel_indices
            field_dict["response_transformation_matrix"] = (
                np.nan
                if self.environment_parameters.response_transformation_matrix is None
                else self.environment_parameters.response_transformation_matrix
            )
            field_dict["reference_transformation_matrix"] = (
                np.nan
                if self.environment_parameters.reference_transformation_matrix is None
                else self.environment_parameters.reference_transformation_matrix
            )
            field_dict["sysid_frequency_spacing"] = (
                self.environment_parameters.sysid_frequency_spacing
            )
            field_dict.update(self.data_acquisition_parameters.extra_parameters)
            for key, value in self.environment_parameters.__dict__.items():
                try:
                    if "sysid_" in key:
                        field_dict[key] = np.array(value)
                except TypeError:
                    continue
            for label, _ in labels:
                field_dict["channel_" + label] = np.array(
                    [
                        ("" if getattr(channel, label) is None else getattr(channel, label))
                        for channel in self.data_acquisition_parameters.channel_list
                    ]
                )
            # print(field_dict)
            if file_filter == "MatLab File (*.mat)":
                for field in [
                    "frf_data",
                    "response_cpsd",
                    "reference_cpsd",
                    "coherence",
                    "response_noise_cpsd",
                    "reference_noise_cpsd",
                ]:
                    field_dict[field] = np.moveaxis(field_dict[field], 0, -1)
                savemat(filename, field_dict)
            elif file_filter == "Numpy File (*.npz)":
                np.savez(filename, **field_dict)

    def load_sysid_matrix_file(self, filename, popup=True):
        """Loads a system identification dataset from previous analysis or testing

        Parameters
        ----------
        filename : str
            The filename of the system identification file to load
        popup : bool, optional
            If True, bring up a file selection dialog box instead of using filename, by default True

        Raises
        ------
        ValueError
            If the wrong type of file is loaded
        """
        if popup:
            filename, file_filter = QtWidgets.QFileDialog.getOpenFileName(
                self.system_id_widget,
                "Select File to Load Transfer Function Matrices",
                filter="NetCDF File (*.nc4);;MatLab File (*.mat);;Numpy File (*.npz);;"
                "SDynPy FRF (*.npz);;Forcefinder SPR (*.npz)",
            )
        else:
            file_filter = None
        if filename is None or filename == "":
            return
        elif file_filter == "NetCDF File (*.nc4)" or (
            file_filter is None and filename.endswith(".nc4")
        ):
            netcdf_handle = nc4.Dataset(  # pylint: disable=no-member
                filename, "r", format="NETCDF4"
            )
            # TODO: error checking to make sure relevant info matches current controller state
            group_handle = netcdf_handle[self.environment_name]
            sample_rate = netcdf_handle.sample_rate
            frame_size = group_handle.sysid_frame_size
            fft_lines = group_handle.dimensions["fft_lines"].size
            variables = group_handle.variables
            combine = np.vectorize(complex)
            try:
                self.last_transfer_function = np.array(
                    combine(variables["frf_data_real"][:], variables["frf_data_imag"][:])
                )
                self.last_coherence = np.array(variables["frf_coherence"][:])
                self.last_response_cpsd = np.array(
                    combine(
                        variables["response_cpsd_real"][:],
                        variables["response_cpsd_imag"][:],
                    )
                )
                self.last_reference_cpsd = np.array(
                    combine(
                        variables["reference_cpsd_real"][:],
                        variables["reference_cpsd_imag"][:],
                    )
                )
                self.last_response_noise = np.array(
                    combine(
                        variables["response_noise_cpsd_real"][:],
                        variables["response_noise_cpsd_imag"][:],
                    )
                )
                self.last_reference_noise = np.array(
                    combine(
                        variables["reference_noise_cpsd_real"][:],
                        variables["reference_noise_cpsd_imag"][:],
                    )
                )
                self.last_condition = np.linalg.cond(self.last_transfer_function)
                self.frequencies = np.arange(fft_lines) * sample_rate / frame_size
            except KeyError:
                # TODO: in the case that a time history file was chosen, should FRF be
                # auto-computed? could work on environment run or sysid (environment run just
                # may have poor FRF)
                # could we use the data analysis process to do the computation? so we don't
                # lock up the UI
                # could we also pass the FRF to any virtual hardware?
                return
        elif file_filter == "SDynPy FRF (*.npz)":
            sdynpy_dict = np.load(filename)
            if sdynpy_dict["function_type"].item() != 4:
                raise ValueError("File must contain a Sdynpy FrequencyResponseFunctionArray")
            self.last_transfer_function = np.moveaxis(
                np.array(sdynpy_dict["data"]["ordinate"]), -1, 0
            )
            self.last_condition = np.linalg.cond(self.last_transfer_function)
            self.frequencies = np.array(sdynpy_dict["data"]["abscissa"][0][0])
            self.last_coherence = np.zeros((0, self.last_transfer_function.shape[1]))
            # TODO: pull coordinate out to verify matching info
        elif file_filter == "Forcefinder SPR (*.npz)":
            forcefinder_dict = np.load(filename)
            self.last_transfer_function = np.array(
                forcefinder_dict["training_frf"]
            )  # training frf will generally be the one used for testing
            self.last_condition = np.linalg.cond(self.last_transfer_function)
            self.frequencies = np.array(forcefinder_dict["abscissa"])
            self.last_coherence = np.zeros((0, self.last_transfer_function.shape[1]))
            if "buzz_cpsd" in forcefinder_dict:
                self.last_response_cpsd = np.array(forcefinder_dict["buzz_cpsd"])
        else:
            if file_filter == "MatLab File (*.mat)":
                field_dict = loadmat(filename)
                for field in [
                    "frf_data",
                    "response_cpsd",
                    "reference_cpsd",
                    "coherence",
                    "response_noise_cpsd",
                    "reference_noise_cpsd",
                ]:
                    field_dict[field] = np.moveaxis(field_dict[field], -1, 0)
            elif file_filter == "Numpy File (*.npz)":
                field_dict = np.load(filename)
            self.last_transfer_function = np.array(field_dict["frf_data"])
            self.last_response_cpsd = np.array(field_dict["response_cpsd"])
            self.last_reference_cpsd = np.array(field_dict["reference_cpsd"])
            self.last_coherence = np.array(field_dict["coherence"])
            self.last_response_noise = np.array(field_dict["response_noise_cpsd"])
            self.last_reference_noise = np.array(field_dict["reference_noise_cpsd"])
            self.last_condition = np.linalg.cond(self.last_transfer_function)
            self.frequencies = (
                np.arange(self.last_transfer_function.shape[0])
                * field_dict["sysid_frequency_spacing"].squeeze()
            )
        # Send values to data analysis process (through the
        # environment queue, environment then passes to data analysis)
        self.environment_command_queue.put(
            self.log_name,
            (
                SysIDDataAnalysisCommands.LOAD_NOISE,
                (
                    0,
                    self.frequencies,
                    None,
                    None,
                    self.last_response_noise,
                    self.last_reference_noise,
                    None,
                ),
            ),
        )
        self.environment_command_queue.put(
            self.log_name,
            (
                SysIDDataAnalysisCommands.LOAD_TRANSFER_FUNCTION,
                (
                    0,
                    self.frequencies,
                    self.last_transfer_function,
                    self.last_coherence,
                    self.last_response_cpsd,
                    self.last_reference_cpsd,
                    self.last_condition,
                ),
            ),
        )
        self.update_sysid_plots(
            update_time=False,
            update_transfer_function=True,
            update_noise=True,
            update_kurtosis=False,
        )
        self.system_id_widget.current_frames_spinbox.setValue(0)
        self.system_id_widget.total_frames_spinbox.setValue(0)
        self.system_id_widget.progressBar.setValue(100)

    def disable_system_id_daq_armed(self):
        """Disables widget on the UI due to the data acquisition being in use"""
        for widget in [
            self.system_id_widget.preview_noise_button,
            self.system_id_widget.preview_system_id_button,
            self.system_id_widget.start_button,
            self.system_id_widget.samplesPerFrameSpinBox,
            self.system_id_widget.averagingTypeComboBox,
            self.system_id_widget.noiseAveragesSpinBox,
            self.system_id_widget.systemIDAveragesSpinBox,
            self.system_id_widget.averagingCoefficientDoubleSpinBox,
            self.system_id_widget.estimatorComboBox,
            self.system_id_widget.levelDoubleSpinBox,
            self.system_id_widget.signalTypeComboBox,
            self.system_id_widget.windowComboBox,
            self.system_id_widget.overlapDoubleSpinBox,
            self.system_id_widget.onFractionDoubleSpinBox,
            self.system_id_widget.pretriggerDoubleSpinBox,
            self.system_id_widget.rampFractionDoubleSpinBox,
            self.system_id_widget.stream_transfer_function_data_checkbox,
            self.system_id_widget.select_transfer_function_stream_file_button,
            self.system_id_widget.transfer_function_stream_file_display,
            self.system_id_widget.levelRampTimeDoubleSpinBox,
            self.system_id_widget.save_system_id_matrices_button,
            self.system_id_widget.load_system_id_matrices_button,
            self.system_id_widget.lowFreqCutoffSpinBox,
            self.system_id_widget.highFreqCutoffSpinBox,
        ]:
            widget.setEnabled(False)
        for widget in [self.system_id_widget.stop_button]:
            widget.setEnabled(False)

    def enable_system_id_daq_disarmed(self):
        """Enables widgets on the UI due to the data acquisition being no longer in use"""
        for widget in [
            self.system_id_widget.preview_noise_button,
            self.system_id_widget.preview_system_id_button,
            self.system_id_widget.start_button,
            self.system_id_widget.samplesPerFrameSpinBox,
            self.system_id_widget.averagingTypeComboBox,
            self.system_id_widget.noiseAveragesSpinBox,
            self.system_id_widget.systemIDAveragesSpinBox,
            self.system_id_widget.averagingCoefficientDoubleSpinBox,
            self.system_id_widget.estimatorComboBox,
            self.system_id_widget.levelDoubleSpinBox,
            self.system_id_widget.signalTypeComboBox,
            self.system_id_widget.windowComboBox,
            self.system_id_widget.overlapDoubleSpinBox,
            self.system_id_widget.onFractionDoubleSpinBox,
            self.system_id_widget.pretriggerDoubleSpinBox,
            self.system_id_widget.rampFractionDoubleSpinBox,
            self.system_id_widget.stream_transfer_function_data_checkbox,
            self.system_id_widget.select_transfer_function_stream_file_button,
            self.system_id_widget.transfer_function_stream_file_display,
            self.system_id_widget.levelRampTimeDoubleSpinBox,
            self.system_id_widget.save_system_id_matrices_button,
            self.system_id_widget.load_system_id_matrices_button,
            self.system_id_widget.lowFreqCutoffSpinBox,
            self.system_id_widget.highFreqCutoffSpinBox,
        ]:
            widget.setEnabled(True)
        for widget in [self.system_id_widget.stop_button]:
            widget.setEnabled(False)
