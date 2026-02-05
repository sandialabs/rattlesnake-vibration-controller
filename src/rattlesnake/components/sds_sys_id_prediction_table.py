import numpy as np
from qtpy import uic, QtWidgets

from .environments import sds_prediction_table_ui_path
from .sds_sys_id_utilities import DecayedSineTable, decayed_sine_table
from .sds_sys_id_metadata import SRSParameters, SpecParameters, SDSMetadata
from .sds_sys_id_utilities import SDSCommands
from .utilities import VerboseMessageQueue
from .ui_utilities import AdaptiveNoWheelSpinBox


class SDSPredictionTable:

    def __init__(
        self,
        parent_widget: QtWidgets.QWidget,
        environment_command_queue: VerboseMessageQueue,
        log_name: str,
        sds_table: None | DecayedSineTable = None,
        drive_names: None | np.ndarray = None,
        response_names: None | np.ndarray = None,
        sds_parameters: None | SDSMetadata = None,
    ):
        uic.loadUi(sds_prediction_table_ui_path, parent_widget)
        # Processing data
        self.parent_widget = parent_widget
        self.environment_command_queue = environment_command_queue
        self.log_name = log_name
        self.sds_table = sds_table
        self.drive_names = drive_names
        self.response_names = response_names
        self.sds_parameters = sds_parameters
        self.frequency_locked = False
        self.amplitude_locked = False
        self.delay_locked = False
        self.decay_locked = False
        # Keep track of tables and tabs
        self.sds_table_widgets = []
        # Persistent calculated data
        self.response_time_history = None
        self.response_srs = None
        self.drive_time_history = None

        # Connect callbacks
        self.parent_widget.excitation_selector.currentIndexChanged.connect(
            self.update_drive_plot_ui
        )
        self.parent_widget.response_selector.currentIndexChanged.connect(
            self.update_response_plot_ui
        )
        self.parent_widget.response_error_list.itemClicked.connect(self.update_response_selector)
        self.parent_widget.excitation_voltage_list.itemClicked.connect(
            self.update_excitation_selector
        )

        # Update information
        self.update_ui()

    def update_names(
        self, drive_names: None | np.ndarray = None, response_names: None | np.ndarray = None
    ):
        self.drive_names = drive_names
        self.response_names = response_names
        self.update_names_ui()

    def update_parameters(self, parameters: SDSMetadata):
        self.sds_parameters = parameters
        if self.sds_parameters is not None:
            self.update_frequencies_ui()

    def update_prediction_information(
        self,
        response_time_history: np.ndarray,
        response_srs: np.ndarray,
        drive_amplitudes: np.ndarray = None,
        drive_delays: np.ndarray = None,
        drive_decays: np.ndarray = None,
        drive_time_histories: np.ndarray = None,
    ):
        self.response_time_history = response_time_history
        self.response_srs = response_srs
        if drive_amplitudes is not None:
            self.sds_table["amplitude"] = drive_amplitudes
        if drive_decays is not None:
            self.sds_table["decay"] = drive_decays
        if drive_delays is not None:
            self.sds_table["delay"] = drive_delays
        self.drive_time_history = drive_time_histories
        self.update_table_ui()
        self.update_drive_plot_ui()
        self.update_response_plot_ui()

    def perform_prediction(self):
        self.environment_command_queue.put(
            self.log_name, (SDSCommands.SDS_TABLE_PREDICTION, self.sds_table)
        )

    def synchronize_sds_table(self):
        """This function is called when a widget is modified in the table to update the internal
        representation of the sds_table"""

    def lock_table(
        self, frequencies=None, amplitudes=None, delays=None, decays=None, all_data=None
    ):
        """This function allows various columns of the table to be locked out"""
        if all_data is not None:
            self.lock_frequency(all_data)
            self.lock_amplitude(all_data)
            self.lock_delay(all_data)
            self.lock_decay(all_data)
        if frequencies is not None:
            self.lock_frequency(frequencies)
        if amplitudes is not None:
            self.lock_amplitude(amplitudes)
        if decays is not None:
            self.lock_decay(decays)
        if delays is not None:
            self.lock_delay(delays)

    def lock_frequency(self, locked=True):
        self.frequency_locked = locked
        index = 0
        for row in range(self.sds_table.rowCount()):
            widget = self.sds_table.cellWidget(row, index)
            if locked:
                widget.setReadOnly(True)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            else:
                widget.setReadOnly(False)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownButtons)

    def lock_amplitude(self, locked=True):
        self.amplitude_locked = locked
        index = 1
        for row in range(self.sds_table.rowCount()):
            widget = self.sds_table.cellWidget(row, index)
            if locked:
                widget.setReadOnly(True)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            else:
                widget.setReadOnly(False)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownButtons)

    def lock_delay(self, locked=True):
        self.delay_locked = locked
        index = 2
        for row in range(self.sds_table.rowCount()):
            widget = self.sds_table.cellWidget(row, index)
            if locked:
                widget.setReadOnly(True)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            else:
                widget.setReadOnly(False)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownButtons)

    def lock_decay(self, locked=True):
        self.decay_locked = locked
        index = 3
        for row in range(self.sds_table.rowCount()):
            widget = self.sds_table.cellWidget(row, index)
            if locked:
                widget.setReadOnly(True)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            else:
                widget.setReadOnly(False)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownButtons)

    def update_ui(self):
        self.update_names_ui()
        self.update_frequencies_ui()
        self.update_table_ui()
        self.update_response_plot_ui()
        self.update_drive_plot_ui()

    def update_names_ui(self):
        # Update the drives if there are names
        if self.drive_names is not None:
            self.sds_table_widgets = []
            self.parent_widget.excitation_selector.clear()
            for name in self.drive_names:
                self.parent_widget.excitation_selector.addItem(name)
        if self.response_names is not None:
            self.parent_widget.response_selector.clear()
            for name in self.response_names:
                self.parent_widget.response_selector.addItem(name)

    def update_frequencies_ui(self):
        frequencies = self.sds_parameters.get_sds_frequencies()
        self.sds_table = decayed_sine_table(
            frequency=frequencies,
            amplitude=np.zeros((len(frequencies), len(self.drive_names))),
            decay=np.zeros((len(frequencies), len(self.drive_names))),
            delay=np.zeros((len(frequencies), len(self.drive_names))),
        )
        self.parent_widget.sds_table.clearContent()
        self.parent_widget.sds_table.setRowCount(len(frequencies))
        self.sds_table_widgets = []
        num_rows = len(frequencies)
        self.sds_table.setRowCount(num_rows)
        for row in range(num_rows):
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(0, self.sds_parameters.sample_rate / 2)
            spinbox.setSingleStep(1)
            spinbox.setValue(frequencies[row])
            spinbox.setKeyboardTracking(False)
            spinbox.setDecimals(4)
            if self.frequency_locked:
                spinbox.setReadOnly(True)
                spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            else:
                spinbox.setReadOnly(False)
                spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownButtons)
            self.sds_table.setCellWidget(row, 0, spinbox)
            self.sds_table_widgets.append(spinbox)
            # Amplitude
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(-1000000, 1000000)
            spinbox.setSingleStep(1)
            spinbox.setValue(0)
            spinbox.setKeyboardTracking(False)
            if self.amplitude_locked:
                spinbox.setReadOnly(True)
                spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            else:
                spinbox.setReadOnly(False)
                spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownButtons)
            spinbox.valueChanged.connect(self.synchronize_sds_table)
            self.sds_table.setCellWidget(row, 1, spinbox)
            self.sds_table_widgets.append(spinbox)
            # Delay
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(
                -self.sds_parameters.block_size / self.sds_parameters.sample_rate,
                self.sds_parameters.block_size / self.sds_parameters.sample_rate,
            )
            spinbox.setSingleStep(0.1)
            spinbox.setValue(0)
            spinbox.setKeyboardTracking(False)
            if self.delay_locked:
                spinbox.setReadOnly(True)
                spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            else:
                spinbox.setReadOnly(False)
                spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownButtons)
            spinbox.valueChanged.connect(self.synchronize_sds_table)
            self.sds_table.setCellWidget(row, 2, spinbox)
            self.sds_table_widgets.append(spinbox)
            # Decay
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(0, 10)
            spinbox.setSingleStep(0.01)
            spinbox.setValue(0)
            spinbox.setKeyboardTracking(False)
            if self.decay_locked:
                spinbox.setReadOnly(True)
                spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            else:
                spinbox.setReadOnly(False)
                spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownButtons)
            spinbox.valueChanged.connect(self.synchronize_sds_table)
            self.sds_table.setCellWidget(row, 3, spinbox)
            self.sds_table_widgets.append(spinbox)

    def update_table_ui(self):
        """This function is called to update the table values based on changes to the internal
        sds array of from changing the active drive channel."""

    def update_response_plot_ui(self):
        """This function is called to update the response plots"""

    def update_drive_plot_ui(self):
        """This function is called to update the drive plots"""

    def update_response_selector(self, item):
        index = self.parent_widget.response_error_list.row(item)
        self.parent_widget.response_selector.setCurrentIndex(index)

    def update_excitation_selector(self, item):
        index = self.parent_widget.excitation_voltage_list.row(item)
        self.parent_widget.excitation_selector.setCurrentIndex(index)
