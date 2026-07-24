import os
from abc import abstractmethod

import netCDF4 as nc4
import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, uic
from scipy.io import loadmat, savemat

from rattlesnake.utilities import DIRECTORY
from rattlesnake.engine import RattlesnakeController, RattlesnakeState
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.abstract_environment import EnvironmentMetadata
from rattlesnake.environment.abstract_sysid_environment import (
    SysIdEnvironmentMetadata,
    SysIdUICommands,
)

from rattlesnake.process.streaming import StreamType, StreamMetadata
from rattlesnake.process.data_collector import DataCollectorUICommands
from rattlesnake.process.abstract_sysid_data_analysis import (
    SysIdMetadata,
    SysIdDataAnalysisCommands,
    SysIdDataAnalysisUICommands,
    SysIdDataPackage,
)
from rattlesnake.user_interface.abstract_user_interface import EnvironmentUI
from rattlesnake.user_interface.ui_utilities import (
    error_message_qt,
    RotatedAxisItem,
    SysIdSelector,
)


# region User Interface
class SysIdEnvironmentUI(EnvironmentUI):
    """Abstract User Interface class defining the interface with the controller

    This class is used to define the interface between the User Interface of a
    environment in the controller and the main controller."""

    @abstractmethod
    def __init__(
        self,
        environment_type: EnvironmentType,
        environment_name: str,
        rattlesnake: RattlesnakeController,
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
        super().__init__(environment_type, environment_name, rattlesnake)
        # Add the page to the system id tabwidget
        self.system_id_widget = QtWidgets.QWidget()
        system_identification_ui_path = os.path.join(
            DIRECTORY, "user_interface", "ui_files", "system_identification.ui"
        )
        uic.loadUi(system_identification_ui_path, self.system_id_widget)
        self.connect_sysid_callbacks()
        self.complete_sysid_ui()

        self.sysid_data = SysIdDataPackage()
        self.last_time_response = None
        self.last_kurtosis = None

    def connect_sysid_callbacks(self):
        """Connects the callback functions to the system identification widgets"""
        self.system_id_widget.preview_noise_button.clicked.connect(
            lambda: self.run_system_id(bool_preview=True, type="noise")
        )
        self.system_id_widget.preview_system_id_button.clicked.connect(
            lambda: self.run_system_id(bool_preview=True, type="transfer")
        )
        self.system_id_widget.start_button.clicked.connect(
            lambda: self.run_system_id(bool_preview=False, type="noise")
        )
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
        self.system_id_widget.coherence_checkbox.stateChanged.connect(
            self.show_hide_coherence
        )
        self.system_id_widget.levels_checkbox.stateChanged.connect(
            self.show_hide_levels
        )
        self.system_id_widget.time_data_checkbox.stateChanged.connect(
            self.show_hide_time_data
        )
        self.system_id_widget.impulse_checkbox.stateChanged.connect(
            self.show_hide_impulse
        )
        self.system_id_widget.transfer_function_checkbox.stateChanged.connect(
            self.show_hide_transfer_function
        )
        self.system_id_widget.kurtosis_checkbox.stateChanged.connect(
            self.show_hide_kurtosis
        )
        self.system_id_widget.signalTypeComboBox.currentIndexChanged.connect(
            self.update_signal_type
        )
        self.system_id_widget.save_system_id_matrices_button.clicked.connect(
            self.save_sysid_matrix_file
        )
        self.system_id_widget.load_system_id_matrices_button.clicked.connect(
            self.load_sysid_matrix_file
        )

    def complete_sysid_ui(self):
        self.time_response_plot = (
            self.system_id_widget.time_data_graphicslayout.addPlot(row=0, column=0)
        )
        self.time_response_plot.setLabel("left", "Response")
        self.time_response_plot.setLabel("bottom", "Time (s)")
        self.time_reference_plot = (
            self.system_id_widget.time_data_graphicslayout.addPlot(row=0, column=1)
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
            self.system_id_widget.transfer_function_graphics_layout.addPlot(
                row=0, column=0
            )
        )
        self.transfer_function_phase_plot.setLabel("left", "Phase")
        self.transfer_function_phase_plot.setLabel("bottom", "Frequency (Hz)")
        self.transfer_function_magnitude_plot = (
            self.system_id_widget.transfer_function_graphics_layout.addPlot(
                row=0, column=1
            )
        )
        self.transfer_function_magnitude_plot.setLabel("left", "Amplitude")
        self.transfer_function_magnitude_plot.setLabel("bottom", "Frequency (Hz)")
        self.impulse_response_plot = (
            self.system_id_widget.impulse_graphicslayout.addPlot(row=0, column=0)
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
        self.kurtosis_response_plot = (
            self.system_id_widget.kurtosis_graphicslayout.addPlot(row=0, column=0)
        )
        self.kurtosis_reference_plot = (
            self.system_id_widget.kurtosis_graphicslayout.addPlot(row=0, column=1)
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

    @property
    @abstractmethod
    def initialized_control_names(self):
        """Names of control channels that have been initialized and will be used in displays"""

    @property
    @abstractmethod
    def initialized_output_names(self):
        """Names of output channels that have been initialized and will be used in displays"""

    @property
    def sysid_active(self):
        try:
            queue_name = self.rattlesnake.environment_manager.queue_names_dict[
                self.environment_name
            ]
            return self.rattlesnake.environment_manager.event_container.environment_sysid_active_events[
                queue_name
            ].is_set()
        except:
            return False

    # endregion

    # region State Sync
    @abstractmethod
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
        # Store for later
        super().initialize_hardware(hardware_metadata)
        self.system_id_widget.highFreqCutoffSpinBox.setMaximum(
            hardware_metadata.sample_rate // 2
        )
        # finish setting up kurtosis plots using node number + direction
        for i, channel in enumerate(self.hardware_metadata.channel_list):
            node = str(channel.node_number) + (
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
    def initialize_environment(self, environment_metadata):
        super().initialize_environment(environment_metadata)
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

    @abstractmethod
    def get_environment_metadata(self, global_channel_list) -> SysIdEnvironmentMetadata:
        """
        Collect the parameters from the user interface defining the environment

        Returns
        -------
        EnvironmentMetadata
            An EnvironmentMetadata-inheriting object that contains the parameters
            defining the environment.
        """

    @abstractmethod
    def set_environment_metadata(self, metadata: SysIdEnvironmentMetadata):
        """
        Update the user interface from environment metadata

        This function is called when the Environment parameters are initialized.
        This function should set up the user interface accordingly.
        """

    def get_sysid_metadata(self, hardware_metadata: HardwareMetadata):
        """Updates the provided system identification metadata based on current UI widget values"""
        sysid_frame_size = self.system_id_widget.samplesPerFrameSpinBox.value()
        sysid_averaging_type = self.system_id_widget.averagingTypeComboBox.itemText(
            self.system_id_widget.averagingTypeComboBox.currentIndex()
        )
        sysid_noise_averages = self.system_id_widget.noiseAveragesSpinBox.value()
        sysid_averages = self.system_id_widget.systemIDAveragesSpinBox.value()
        sysid_exponential_averaging_coefficient = (
            self.system_id_widget.averagingCoefficientDoubleSpinBox.value()
        )
        sysid_estimator = self.system_id_widget.estimatorComboBox.itemText(
            self.system_id_widget.estimatorComboBox.currentIndex()
        )
        sysid_level = self.system_id_widget.levelDoubleSpinBox.value()
        sysid_level_ramp_time = self.system_id_widget.levelRampTimeDoubleSpinBox.value()
        sysid_signal_type = self.system_id_widget.signalTypeComboBox.itemText(
            self.system_id_widget.signalTypeComboBox.currentIndex()
        )
        sysid_window = self.system_id_widget.windowComboBox.itemText(
            self.system_id_widget.windowComboBox.currentIndex()
        )
        sysid_overlap = (
            self.system_id_widget.overlapDoubleSpinBox.value() / 100
            if sysid_signal_type == "Random"
            else 0.0
        )
        sysid_burst_on = self.system_id_widget.onFractionDoubleSpinBox.value() / 100
        sysid_pretrigger = self.system_id_widget.pretriggerDoubleSpinBox.value() / 100
        sysid_burst_ramp_fraction = (
            self.system_id_widget.rampFractionDoubleSpinBox.value() / 100
        )
        sysid_low_frequency_cutoff = self.system_id_widget.lowFreqCutoffSpinBox.value()
        sysid_high_frequency_cutoff = (
            self.system_id_widget.highFreqCutoffSpinBox.value()
        )
        stream_file = self.system_id_widget.transfer_function_stream_file_display.text()

        sysid_metadata = SysIdMetadata(
            hardware_metadata.sample_rate,
            sysid_frame_size,
            sysid_averaging_type,
            sysid_noise_averages,
            sysid_averages,
            sysid_exponential_averaging_coefficient,
            sysid_estimator,
            sysid_level,
            sysid_level_ramp_time,
            sysid_signal_type,
            sysid_window,
            sysid_overlap,
            sysid_burst_on,
            sysid_pretrigger,
            sysid_burst_ramp_fraction,
            sysid_low_frequency_cutoff,
            sysid_high_frequency_cutoff,
            stream_file,
        )
        return sysid_metadata

    def set_sysid_metadata(self, sysid_metadata: SysIdMetadata):
        """
        Update the user interface with sysid parameters.

        Parameters
        ----------
        sysid_metadata : SysIdMetadata
            Metadata object containing the system identification parameters.
        """
        self.system_id_widget.samplesPerFrameSpinBox.setValue(
            sysid_metadata.sysid_frame_size
        )

        averaging_index = self.system_id_widget.averagingTypeComboBox.findText(
            sysid_metadata.sysid_averaging_type
        )
        if averaging_index >= 0:
            self.system_id_widget.averagingTypeComboBox.setCurrentIndex(averaging_index)

        self.system_id_widget.noiseAveragesSpinBox.setValue(
            sysid_metadata.sysid_noise_averages
        )
        self.system_id_widget.systemIDAveragesSpinBox.setValue(
            sysid_metadata.sysid_averages
        )
        self.system_id_widget.averagingCoefficientDoubleSpinBox.setValue(
            sysid_metadata.sysid_exponential_averaging_coefficient
        )

        estimator_index = self.system_id_widget.estimatorComboBox.findText(
            sysid_metadata.sysid_estimator
        )
        if estimator_index >= 0:
            self.system_id_widget.estimatorComboBox.setCurrentIndex(estimator_index)

        self.system_id_widget.levelDoubleSpinBox.setValue(sysid_metadata.sysid_level)
        self.system_id_widget.levelRampTimeDoubleSpinBox.setValue(
            sysid_metadata.sysid_level_ramp_time
        )

        signal_type_index = self.system_id_widget.signalTypeComboBox.findText(
            sysid_metadata.sysid_signal_type
        )
        if signal_type_index >= 0:
            self.system_id_widget.signalTypeComboBox.setCurrentIndex(signal_type_index)

        window_index = self.system_id_widget.windowComboBox.findText(
            sysid_metadata.sysid_window
        )
        if window_index >= 0:
            self.system_id_widget.windowComboBox.setCurrentIndex(window_index)

        self.system_id_widget.overlapDoubleSpinBox.setValue(
            sysid_metadata.sysid_overlap * 100
        )
        self.system_id_widget.onFractionDoubleSpinBox.setValue(
            sysid_metadata.sysid_burst_on * 100
        )
        self.system_id_widget.pretriggerDoubleSpinBox.setValue(
            sysid_metadata.sysid_pretrigger * 100
        )
        self.system_id_widget.rampFractionDoubleSpinBox.setValue(
            sysid_metadata.sysid_burst_ramp_fraction * 100
        )

        self.system_id_widget.lowFreqCutoffSpinBox.setValue(
            sysid_metadata.sysid_low_frequency_cutoff
        )
        self.system_id_widget.highFreqCutoffSpinBox.setValue(
            sysid_metadata.sysid_high_frequency_cutoff
        )

        self.system_id_widget.transfer_function_stream_file_display.setText(
            sysid_metadata.stream_file if sysid_metadata.stream_file is not None else ""
        )

    @abstractmethod
    def get_environment_instructions(self):
        return

    @abstractmethod
    def set_environment_instructions(self, instructions):
        super().set_environment_instructions(instructions)

    # endregion

    # region Callbacks
    def display_system_id_started(self):
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

    def display_system_id_ended(self):
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

    """
    To run the noise, the steps are
    1. Initialize system id metadata
    2. Start up hardware acquisition
    3. Start up streaming and noise
    4. If the full test is happening, wait for sysid
    analysis process to send the NOISE_COMPLETED command
    """

    def run_system_id(self, bool_preview, type):
        """Starts the acquisition phase of the controller"""
        self.log("Starting System ID")

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

        try:
            queue_name = self.rattlesnake.environment_manager.queue_names_dict[
                self.environment_name
            ]
            sysid_metadata = self.get_sysid_metadata(self.hardware_metadata)
            if bool_preview:
                sysid_metadata.auto_shutdown = False
            else:
                sysid_metadata.auto_shutdown = True
            self.rattlesnake.initialize_system_id(sysid_metadata, self.environment_name)
        except Exception as e:
            self.run_system_id_error(e)
            return

        ready_event_list = [
            self.rattlesnake.event_container.environment_ready_events[queue_name]
        ]
        active_event_list = []
        self.create_event_watcher(
            ready_event_list, active_event_list, active_event_check=True
        )
        self.event_watcher.ready.connect(
            lambda: self.run_system_id_acquisition_noise(
                bool_preview=bool_preview, type=type
            )
        )
        self.event_watcher.error.connect(self.run_system_id_error)
        self.event_thread.start()

    def run_system_id_acquisition_noise(self, bool_preview, type):
        self.clean_up_event_watcher()

        # Start Acqusition
        try:
            sysid_metadata = self.get_sysid_metadata(self.hardware_metadata)
            if bool_preview:
                stream_metadata = StreamMetadata(StreamType.NO_STREAM)
            elif sysid_metadata.stream_file:
                stream_metadata = StreamMetadata(
                    StreamType.MANUAL, sysid_metadata.stream_file
                )
            else:
                stream_metadata = StreamMetadata(StreamType.NO_STREAM)
            self.rattlesnake.start_acquisition(stream_metadata)
        except Exception as e:
            self.run_system_id_error(e)
            return

        ready_event_list = [
            self.rattlesnake.event_container.streaming_ready_event,
        ]
        active_event_list = [
            self.rattlesnake.event_container.acquisition_active_event,
            self.rattlesnake.event_container.output_active_event,
        ]
        self.create_event_watcher(
            ready_event_list, active_event_list, active_event_check=True
        )
        if str(type).lower() == "noise":
            self.event_watcher.ready.connect(self.run_system_id_noise)
        else:
            self.event_watcher.ready.connect(self.run_system_id_transfer)
        self.event_watcher.error.connect(self.run_system_id_error)
        self.event_thread.start()

    def run_system_id_noise(self):
        self.clean_up_event_watcher()

        try:
            queue_name = self.rattlesnake.environment_manager.queue_names_dict[
                self.environment_name
            ]
            self.rattlesnake.start_streaming()
            self.rattlesnake.start_system_id_noise(self.environment_name)
        except Exception as e:
            self.run_system_id_error(e)
            return

        ready_event_list = []
        active_event_list = [
            self.rattlesnake.event_container.environment_sysid_active_events[queue_name]
        ]
        self.create_event_watcher(
            ready_event_list, active_event_list, active_event_check=True
        )
        self.event_watcher.ready.connect(self.run_system_id_ready)
        self.event_watcher.error.connect(self.run_system_id_error)
        self.event_thread.start()

    def run_system_id_validate_noise_closeout(self):
        """This is used to refresh streaming when the sys_id_data_analysis process
        tells the UI to start up transfer function after noise has automatically
        shutdown"""
        if self.rattlesnake.state not in (
            RattlesnakeState.HARDWARE_ACTIVE,
            RattlesnakeState.ENVIRONMENT_ACTIVE,
            RattlesnakeState.SYS_ID_ACTIVE,
        ):
            # This is to prevent this from running in the case that the sysid was
            # run headlessly.
            self.display_system_id_ended()
            return

        try:
            queue_name = self.rattlesnake.environment_manager.queue_names_dict[
                self.environment_name
            ]
            if self.rattlesnake.streaming:
                self.rattlesnake.stop_streaming()
        except Exception as e:
            self.run_system_id_error(e)
            return

        ready_event_list = []
        active_event_list = [
            self.rattlesnake.event_container.streaming_active_event,
            self.rattlesnake.event_container.environment_sysid_active_events[
                queue_name
            ],
        ]
        self.create_event_watcher(
            ready_event_list, active_event_list, active_event_check=False
        )
        self.event_watcher.ready.connect(self.run_system_id_transfer)
        self.event_watcher.error.connect(self.run_system_id_error)
        self.event_thread.start()

    def run_system_id_transfer(self):
        self.clean_up_event_watcher()

        if self.rattlesnake.state not in (
            RattlesnakeState.HARDWARE_ACTIVE,
            RattlesnakeState.ENVIRONMENT_ACTIVE,
            RattlesnakeState.SYS_ID_ACTIVE,
        ):
            # Prevent from running in the case that the sysid was completed
            # in headless mode
            self.display_system_id_ended()
            return

        try:
            queue_name = self.rattlesnake.environment_manager.queue_names_dict[
                self.environment_name
            ]
            self.rattlesnake.start_streaming()
            self.rattlesnake.start_system_id_transfer_function(self.environment_name)
        except Exception as e:
            self.run_system_id_error(e)
            return

        ready_event_list = []
        active_event_list = [
            self.rattlesnake.event_container.environment_sysid_active_events[queue_name]
        ]
        self.create_event_watcher(
            ready_event_list, active_event_list, active_event_check=True
        )
        self.event_watcher.ready.connect(self.run_system_id_ready)
        self.event_watcher.error.connect(self.run_system_id_error)
        self.event_thread.start()

    def run_system_id_validate_transfer_closeout(self):
        try:
            queue_name = self.rattlesnake.environment_manager.queue_names_dict[
                self.environment_name
            ]
            if self.rattlesnake.streaming:
                self.rattlesnake.stop_streaming()
            # Sometimes this function is run when loading sysid so the hardware
            # is not active
            if self.rattlesnake.state in (
                RattlesnakeState.HARDWARE_ACTIVE,
                RattlesnakeState.ENVIRONMENT_ACTIVE,
                RattlesnakeState.SYS_ID_ACTIVE,
            ):
                self.rattlesnake.stop_acquisition()
        except Exception as e:
            self.run_system_id_error(e)
            return

        ready_event_list = [self.rattlesnake.event_container.controller_ready_event]
        active_event_list = [
            self.rattlesnake.event_container.streaming_active_event,
            self.rattlesnake.event_container.acquisition_active_event,
            self.rattlesnake.event_container.output_active_event,
            self.rattlesnake.event_container.environment_sysid_active_events[
                queue_name
            ],
        ]
        self.create_event_watcher(
            ready_event_list, active_event_list, active_event_check=False
        )
        self.event_watcher.ready.connect(self.run_system_id_ready)
        self.event_watcher.error.connect(self.run_system_id_error)
        self.event_thread.start()

    def run_system_id_ready(self):
        if self.sysid_active:
            self.display_system_id_started()
        else:
            self.display_system_id_ended()

        self.clean_up_event_watcher()

    def run_system_id_error(self, error):
        if self.sysid_active:
            self.display_system_id_started()
        else:
            self.display_system_id_ended()

        if self.rattlesnake.streaming:
            self.rattlesnake.stop_streaming()

        self.display_error(error)
        self.clean_up_event_watcher()

    def stop_system_id(self):
        """Stops the system identification"""
        self.log("Stopping System ID")

        for widget in [self.system_id_widget.stop_button]:
            widget.setEnabled(True)

        try:
            self.rattlesnake.stop_acquisition()
        except Exception as e:
            self.run_system_id_error(e)
            return

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

    def open_sysid_selector(self, source_environments, target_environments):
        dialog = SysIdSelector(
            source_environments,
            target_environments,
            parent=self,
        )
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            load_from, load_to = dialog.get_selection()

            if not load_from or not load_to:
                return

            return (load_from, load_to)

    def save_sysid_matrix_file(self):
        """Saves out system identification data to a file"""
        if (
            self.sysid_data.sysid_frf is None
            or self.sysid_data.sysid_response_noise is None
        ):
            self.display_error("System Identification Matrices not yet created.")
            return
        filepath, file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self.system_id_widget,
            "Select File to Save Transfer Function Matrices",
            filter="NetCDF File (*.nc4);;MatLab File (*.mat);;Numpy File (*.npz)",
        )
        if filepath == "":
            return

        try:
            # os.remove(filepath)  # The sysid_save only appends to files
            self.rattlesnake.save_system_id_to_file(self.environment_name, filepath)
        except Exception as e:
            self.display_error(e)

    def load_sysid_matrix_file(self):
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

        filename, file_filter = QtWidgets.QFileDialog.getOpenFileName(
            self.system_id_widget,
            "Select File to Load Transfer Function Matrices",
            filter="NetCDF File (*.nc4);;MatLab File (*.mat);;Numpy File (*.npz);;"
            "SDynPy FRF (*.npz);;Forcefinder SPR (*.npz)",
        )
        if filename is None or filename == "":
            return

        match file_filter:
            case "NetCDF File (*.nc4)":
                netcdf_dataset = nc4.Dataset(  # pylint: disable=no-member
                    filename, "r", format="NETCDF4"
                )
                # the world is not ready for this right now
                # source_environments = netcdf_dataset.variables["environment_names"][...]
                # target_environments = (
                #     self.rattlesnake.environment_manager.environment_names.values()
                # )
                # load_environment, save_environments = self.open_sysid_selector(
                #     source_environments, target_environments
                # )
                # if not load_environment or not save_environments:
                #     return

                netcdf_handle = netcdf_dataset.groups[self.environment_name]
                sysid_metadata = SysIdMetadata().load_metadata_from_netcdf(
                    netcdf_handle, self.hardware_metadata
                )
                sysid_data = SysIdDataPackage().load_package_from_netcdf(netcdf_handle)
            case "SDynPy FRF (*.npz)":
                sdynpy_dict = np.load(filename)
                sysid_metadata = SysIdMetadata().default_metadata(
                    self.hardware_metadata.sample_rate
                )
                sysid_data = SysIdDataPackage().load_package_from_sdynpy_frf(
                    sdynpy_dict
                )
            case "Forcefinder SPR (*.npz)":
                forcefinder_dict = np.load(filename)
                sysid_metadata = SysIdMetadata().default_metadata(
                    self.hardware_metadata.sample_rate
                )
                sysid_data = SysIdDataPackage().load_package_from_forcefinder_spr(
                    forcefinder_dict
                )
            case "MatLab File (*.mat)":
                field_dict = loadmat(filename)
                sysid_metadata = SysIdMetadata().default_metadata(
                    self.hardware_metadata.sample_rate
                )
                sysid_data = SysIdDataPackage().load_package_from_mat_field(field_dict)
            case "Numpy File (*.npz)":
                field_dict = np.load(filename)
                sysid_metadata = SysIdMetadata().default_metadata(
                    self.hardware_metadata.sample_rate
                )
                sysid_data = SysIdDataPackage().load_package_from_numpy_field(
                    field_dict
                )
            case _:
                self.display_error(
                    f"Invalid system identification data filetype: {file_filter}"
                )
                return

        try:
            self.rattlesnake.initialize_system_id(sysid_metadata, self.environment_name)
            self.rattlesnake.load_system_id_from_package(
                self.environment_name, sysid_data
            )
        except Exception as e:
            self.display_error(e)
            return

    @abstractmethod
    def display_environment_ended(self):
        return

    @abstractmethod
    def display_environment_started(self):
        return

    @abstractmethod
    def start_environment(self):
        return super().start_environment()

    @abstractmethod
    def start_environment_ready(self):
        return super().start_environment_ready()

    @abstractmethod
    def start_environment_error(self, error):
        return super().start_environment_error(error)

    @abstractmethod
    def stop_environment(self):
        return super().stop_environment()

    @abstractmethod
    def stop_environment_error(self, error):
        return super().stop_environment_error(error)

    @abstractmethod
    def stop_environment_ready(self):
        return super().stop_environment_ready()

    # endregion

    # region Commands
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
                    self.environment_metadata.response_channel_indices
                )[response_indices]
                reference_frame_indices = np.array(
                    self.environment_metadata.reference_channel_indices
                )[reference_indices]
                response_time_data = self.last_time_response[response_frame_indices]
                reference_time_data = self.last_time_response[reference_frame_indices]
                times = (
                    np.arange(response_time_data.shape[-1])
                    / self.hardware_metadata.sample_rate
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
                self.sysid_data.sysid_frf is not None
                and len(response_indices) > 0
                and len(reference_indices) > 0
            ):
                # print(self.sysid_data.sysid_frf)
                # print(np.array(response_indices)[:,np.newaxis])
                # print(np.array(reference_indices))
                frf_section = np.reshape(
                    self.sysid_data.sysid_frf[
                        ...,
                        np.array(response_indices)[:, np.newaxis],
                        np.array(reference_indices),
                    ],
                    (self.sysid_data.frequencies.size, -1),
                ).T
                impulse_response = np.fft.irfft(frf_section, axis=-1)
                for i, (frf, imp) in enumerate(zip(frf_section, impulse_response)):
                    self.transfer_function_phase_plot.plot(
                        self.sysid_data.frequencies, np.angle(frf) * 180 / np.pi, pen=i
                    )
                    self.transfer_function_magnitude_plot.plot(
                        self.sysid_data.frequencies, np.abs(frf), pen=i
                    )
                    self.impulse_response_plot.plot(
                        np.arange(imp.size) / self.environment_metadata.sample_rate,
                        imp,
                        pen=i,
                    )
                for i, coherence in enumerate(
                    self.sysid_data.sysid_coherence[..., response_indices].T
                ):
                    self.coherence_plot.plot(
                        self.sysid_data.frequencies, coherence, pen=i
                    )
            if self.sysid_data.sysid_condition is not None:
                self.condition_plot.plot(
                    self.sysid_data.frequencies, self.sysid_data.sysid_condition, pen=0
                )
        if update_noise:
            reference_noise = (
                None
                if self.sysid_data.sysid_reference_noise is None
                or len(reference_indices) == 0
                else self.sysid_data.sysid_reference_noise[
                    ..., reference_indices, reference_indices
                ].real
            )
            response_noise = (
                None
                if self.sysid_data.sysid_response_noise is None
                or len(response_indices) == 0
                else self.sysid_data.sysid_response_noise[
                    ..., response_indices, response_indices
                ].real
            )
            reference_level = (
                None
                if self.sysid_data.sysid_reference_cpsd is None
                or len(reference_indices) == 0
                else self.sysid_data.sysid_reference_cpsd[
                    ..., reference_indices, reference_indices
                ].real
            )
            response_level = (
                None
                if self.sysid_data.sysid_response_cpsd is None
                or len(response_indices) == 0
                else self.sysid_data.sysid_response_cpsd[
                    ..., response_indices, response_indices
                ].real
            )
            self.level_reference_plot.clear()
            self.level_response_plot.clear()
            for i in range(len(reference_indices)):
                if reference_noise is not None:
                    self.level_reference_plot.plot(
                        self.sysid_data.frequencies, reference_noise[:, i], pen=i
                    )
                if reference_level is not None:
                    try:
                        self.level_reference_plot.plot(
                            self.sysid_data.frequencies, reference_level[:, i], pen=i
                        )
                    except Exception:
                        pass
            for i in range(len(response_indices)):
                if response_noise is not None:
                    self.level_response_plot.plot(
                        self.sysid_data.frequencies, response_noise[:, i], pen=i
                    )
                if response_level is not None:
                    try:
                        self.level_response_plot.plot(
                            self.sysid_data.frequencies, response_level[:, i], pen=i
                        )
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
        elif (
            self.system_id_widget.signalTypeComboBox.currentIndex() == 1
        ):  # Pseudorandom
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
            return True
        command, data = queue_data
        self.log(f"Got GUI Message {command}")
        # print('Update GUI Got {:}'.format(message))
        match command:
            case SysIdUICommands.SYSID_STARTED:
                self.display_system_id_started()
            case SysIdUICommands.SYSID_ENDED:
                self.display_system_id_ended()
            case DataCollectorUICommands.TIME_FRAME:
                self.last_time_response, accept = data
                self.update_sysid_plots(
                    update_time=True,
                    update_transfer_function=False,
                    update_noise=False,
                    update_kurtosis=False,
                )
            case DataCollectorUICommands.KURTOSIS:
                self.last_kurtosis = data
                self.update_sysid_plots(
                    update_time=False,
                    update_transfer_function=False,
                    update_noise=False,
                    update_kurtosis=True,
                )
            case SysIdDataAnalysisUICommands.NOISE_COMPLETED:
                if self.rattlesnake.has_gui:
                    self.run_system_id_validate_noise_closeout()
            case SysIdDataAnalysisUICommands.TRANSFER_COMPLETED:
                if self.rattlesnake.has_gui:
                    self.run_system_id_validate_transfer_closeout()
            case SysIdDataAnalysisUICommands.NOISE_UPDATE:
                (
                    frames,
                    total_frames,
                    self.sysid_data.frequencies,
                    self.sysid_data.sysid_response_noise,
                    self.sysid_data.sysid_reference_noise,
                ) = data
                self.update_sysid_plots(
                    update_time=False,
                    update_transfer_function=False,
                    update_noise=True,
                    update_kurtosis=False,
                )
                self.system_id_widget.current_frames_spinbox.setValue(frames)
                self.system_id_widget.total_frames_spinbox.setValue(total_frames)
                self.system_id_widget.progressBar.setValue(
                    int(frames / total_frames * 100)
                )
            case SysIdDataAnalysisUICommands.SYSID_UPDATE:
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
                # print(self.sysid_data.sysid_frf.shape)
                # print(self.sysid_data.sysid_coherence.shape)
                # print(self.sysid_data.sysid_response_cpsd.shape)
                # print(self.sysid_data.sysid_reference_cpsd.shape)
                self.update_sysid_plots(
                    update_time=False,
                    update_transfer_function=True,
                    update_noise=True,
                    update_kurtosis=False,
                )
                self.system_id_widget.current_frames_spinbox.setValue(frames)
                self.system_id_widget.total_frames_spinbox.setValue(total_frames)
                self.system_id_widget.progressBar.setValue(
                    int(frames / total_frames * 100)
                )
            case _:
                return False
        return True

    # endregion
