import numpy as np
from qtpy import uic, QtWidgets
from qtpy.QtCore import Qt
import os
from rattlesnake.environment.sds_sys_id_metadata import SRSParameters, SpecParameters, SDSMetadata
from rattlesnake.engine import RattlesnakeController
from rattlesnake.environment.sds_sys_id_utilities import (
    SDSCommands,
    sum_decayed_sines_reconstruction,
    DecayedSineTable,
    decayed_sine_table,
)
from rattlesnake.utilities import DIRECTORY
from rattlesnake.user_interface.ui_utilities import AdaptiveNoWheelSpinBox


class SDSPredictionTable:

    def __init__(
        self,
        parent_widget: QtWidgets.QWidget,
        rattlesnake: RattlesnakeController,
        environment_name: str,
        prediction_mode: bool,
        sds_table: None | DecayedSineTable = None,
        drive_names: None | np.ndarray = None,
        response_names: None | np.ndarray = None,
        sds_parameters: None | SDSMetadata = None,
    ):
        uic.loadUi(
            os.path.join(DIRECTORY, "user_interface", "ui_files", "srs_sds_prediction_table.ui"),
            parent_widget,
        )
        # Utility Information
        self.parent_widget = parent_widget
        self.rattlesnake = rattlesnake
        self.environment_name = environment_name
        self.prediction_mode = prediction_mode
        # Processing data
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
        self.predicted_response_time_history = None
        self.predicted_response_srs = None
        self.measured_response_time_history = None
        self.measured_response_srs = None
        self.drive_time_history = None

        # Connect callbacks
        self.parent_widget.excitation_selector.currentIndexChanged.connect(self.update_table_ui)
        self.parent_widget.response_selector.currentIndexChanged.connect(
            self.update_response_plot_ui
        )
        self.parent_widget.response_error_list.itemClicked.connect(self.update_response_selector)
        self.parent_widget.excitation_voltage_list.itemClicked.connect(
            self.update_excitation_selector
        )
        self.parent_widget.sds_table.itemSelectionChanged.connect(self.update_tone_selection_ui)

        # Initialize Plots
        self.plot_data_items = {}
        plot_item = self.parent_widget.excitation_display_plot.getPlotItem()
        plot_item.showGrid(True, True, 0.25)
        plot_item.enableAutoRange()
        plot_item.getViewBox().enableAutoRange(enable=True)
        plot_item.addLegend()

        self.plot_data_items = {}
        plot_item = self.parent_widget.response_display_plot.getPlotItem()
        plot_item.showGrid(True, True, 0.25)
        plot_item.enableAutoRange()
        plot_item.getViewBox().enableAutoRange(enable=True)
        plot_item.addLegend()

        plot_item = self.parent_widget.response_srs_plot.getPlotItem()
        plot_item.showGrid(True, True, 0.25)
        plot_item.enableAutoRange()
        plot_item.getViewBox().enableAutoRange(enable=True)
        plot_item.setLogMode(True, True)
        plot_item.addLegend()

        self.plot_data_items[
            "full_time_history_excitation"
        ] = self.parent_widget.excitation_display_plot.getPlotItem().plot(
            np.array(
                [
                    0,
                    (
                        self.sds_parameters.block_size / self.sds_parameters.sample_rate
                        if self.sds_parameters is not None
                        else 1
                    ),
                ]
            ),
            np.nan * np.ones(2),
            pen={"color": "b", "width": 1},
            name="Time History",
        )
        self.plot_data_items[
            "single_tone_time_history_excitation"
        ] = self.parent_widget.excitation_display_plot.getPlotItem().plot(
            np.array(
                [
                    0,
                    (
                        self.sds_parameters.block_size / self.sds_parameters.sample_rate
                        if self.sds_parameters is not None
                        else 1
                    ),
                ]
            ),
            np.nan * np.ones(2),
            pen={"color": "r", "width": 1},
            name="Single Tone",
        )
        self.plot_data_items[
            "full_time_history_response"
        ] = self.parent_widget.response_display_plot.getPlotItem().plot(
            np.array(
                [
                    0,
                    (
                        self.sds_parameters.block_size / self.sds_parameters.sample_rate
                        if self.sds_parameters is not None
                        else 1
                    ),
                ]
            ),
            np.nan * np.ones(2),
            pen={"color": "b", "width": 1},
            name="Time History",
        )

        self.plot_data_items[
            "specification_srs"
        ] = self.parent_widget.response_srs_plot.getPlotItem().plot(
            np.nan * np.array([0, 1]),
            np.nan * np.ones(2),
            pen={"color": "b", "width": 1},
            name="Control SRS",
        )
        self.plot_data_items[
            "specification_lower_limit"
        ] = self.parent_widget.response_srs_plot.getPlotItem().plot(
            np.nan * np.array([0, 1]),
            np.nan * np.ones(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
            name="Limit",
        )
        self.plot_data_items[
            "specification_upper_limit"
        ] = self.parent_widget.response_srs_plot.getPlotItem().plot(
            np.nan * np.array([0, 1]),
            np.nan * np.zeros(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
        )
        self.plot_data_items[
            "srs_predicted"
        ] = self.parent_widget.response_srs_plot.getPlotItem().plot(
            np.nan * np.array([0, 1]),
            np.nan * np.zeros(2),
            pen={"color": "r", "width": 1},
            name="SRS Predicted",
        )
        self.plot_data_items[
            "srs_measured"
        ] = self.parent_widget.response_srs_plot.getPlotItem().plot(
            np.nan * np.array([0, 1]),
            np.nan * np.zeros(2),
            pen={"color": (0, 180, 0), "width": 1},
            name="SRS Measured",
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
        for widget in self.sds_table_widgets:
            widget.blockSignals(True)
        self.predicted_response_time_history = response_time_history
        self.predicted_response_srs = response_srs
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
        self.update_all_voltages_ui()
        for widget in self.sds_table_widgets:
            widget.blockSignals(False)

    def perform_prediction(self):
        print("Performing Prediction!")
        if self.prediction_mode:
            self.rattlesnake.send_environment_command(
                self.environment_name, SDSCommands.SDS_TABLE_PREDICTION, self.sds_table
            )
        else:
            self.rattlesnake.send_environment_command(
                self.environment_name, SDSCommands.SDS_RUN_TABLE_PREDICTION, self.sds_table
            )

    def synchronize_sds_table(self):
        """This function is called when a widget is modified in the table to update the internal
        representation of the sds_table"""
        index = self.parent_widget.excitation_selector.currentIndex()
        for col_index, name in enumerate(["frequency", "amplitude", "delay", "decay"]):
            for row_index in range(self.parent_widget.sds_table.rowCount()):
                value = self.parent_widget.sds_table.cellWidget(row_index, col_index).value()
                if col_index == 0:
                    self.sds_table[name][row_index] = value
                else:
                    self.sds_table[name][row_index, index] = value
        self.update_voltage_ui(index)
        self.perform_prediction()

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
        for row in range(self.parent_widget.sds_table.rowCount()):
            widget = self.parent_widget.sds_table.cellWidget(row, index)
            if locked:
                widget.setReadOnly(True)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            else:
                widget.setReadOnly(False)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownArrows)

    def lock_amplitude(self, locked=True):
        self.amplitude_locked = locked
        index = 1
        for row in range(self.parent_widget.sds_table.rowCount()):
            widget = self.parent_widget.sds_table.cellWidget(row, index)
            if locked:
                widget.setReadOnly(True)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            else:
                widget.setReadOnly(False)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownArrows)

    def lock_delay(self, locked=True):
        self.delay_locked = locked
        index = 2
        for row in range(self.parent_widget.sds_table.rowCount()):
            widget = self.parent_widget.sds_table.cellWidget(row, index)
            if locked:
                widget.setReadOnly(True)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            else:
                widget.setReadOnly(False)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownArrows)

    def lock_decay(self, locked=True):
        self.decay_locked = locked
        index = 3
        for row in range(self.parent_widget.sds_table.rowCount()):
            widget = self.parent_widget.sds_table.cellWidget(row, index)
            if locked:
                widget.setReadOnly(True)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            else:
                widget.setReadOnly(False)
                widget.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownArrows)

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
        if self.sds_parameters is None:
            return
        frequencies = self.sds_parameters.get_sds_frequencies_w_compensation_pulse()
        self.sds_table = decayed_sine_table(
            frequency=frequencies,
            amplitude=np.zeros((len(frequencies), len(self.drive_names))),
            decay=np.zeros((len(frequencies), len(self.drive_names))),
            delay=np.zeros((len(frequencies), len(self.drive_names))),
        )
        self.parent_widget.sds_table.clearContents()
        self.parent_widget.sds_table.setRowCount(len(frequencies))
        self.sds_table_widgets = []
        num_rows = len(frequencies)
        self.parent_widget.sds_table.setRowCount(num_rows)
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
                spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownArrows)
            spinbox.valueChanged.connect(self.synchronize_sds_table)
            self.parent_widget.sds_table.setCellWidget(row, 0, spinbox)
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
                spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownArrows)
            spinbox.valueChanged.connect(self.synchronize_sds_table)
            self.parent_widget.sds_table.setCellWidget(row, 1, spinbox)
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
                spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownArrows)
            spinbox.valueChanged.connect(self.synchronize_sds_table)
            self.parent_widget.sds_table.setCellWidget(row, 2, spinbox)
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
                spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.UpDownArrows)
            spinbox.valueChanged.connect(self.synchronize_sds_table)
            self.parent_widget.sds_table.setCellWidget(row, 3, spinbox)
            self.sds_table_widgets.append(spinbox)

    def update_table_ui(self):
        """This function is called to update the table values based on changes to the internal
        sds array of from changing the active drive channel."""
        if self.sds_table is None or self.drive_names is None:
            return
        for widget in self.sds_table_widgets:
            widget.blockSignals(True)
        index = self.parent_widget.excitation_selector.currentIndex()
        for col_index, name in enumerate(["frequency", "amplitude", "delay", "decay"]):
            for row_index in range(self.parent_widget.sds_table.rowCount()):
                widget = self.parent_widget.sds_table.cellWidget(row_index, col_index)
                if col_index == 0:
                    widget.setValue(self.sds_table[name][row_index])
                else:
                    widget.setValue(self.sds_table[name][row_index, index])
        for widget in self.sds_table_widgets:
            widget.blockSignals(False)
        self.update_drive_plot_ui()

    def update_response_plot_ui(self):
        """This function is called to update the response plots"""
        if self.response_names is None:
            return
        index = self.parent_widget.response_selector.currentIndex()
        if self.sds_parameters is not None:
            # Get the specification
            abscissa = self.sds_parameters.specification_data.frequencies
            srs = self.sds_parameters.specification_data.srs_spec[:, index]
            lower = self.sds_parameters.specification_data.srs_lower_limit[:, index]
            upper = self.sds_parameters.specification_data.srs_upper_limit[:, index]
            self.plot_data_items["specification_srs"].setData(abscissa, srs)
            self.plot_data_items["specification_lower_limit"].setData(abscissa, lower)
            self.plot_data_items["specification_upper_limit"].setData(abscissa, upper)
        if self.predicted_response_srs is not None:
            abscissa = self.sds_parameters.get_sds_frequencies()
            srs = self.predicted_response_srs[:, index]
            self.plot_data_items["srs_predicted"].setData(abscissa, srs)
        if self.predicted_response_time_history is not None:
            th = self.predicted_response_time_history[index, :]
            abscissa = np.arange(th.size) / self.sds_parameters.sample_rate
            self.plot_data_items["full_time_history_response"].setData(abscissa, th)

    def update_drive_plot_ui(self):
        """This function is called to update the drive plots"""
        if self.sds_table is None or self.drive_names is None:
            return
        index = self.parent_widget.excitation_selector.currentIndex()
        signal = sum_decayed_sines_reconstruction(
            self.sds_table["frequency"],
            self.sds_table["amplitude"][:, index],
            self.sds_table["decay"][:, index],
            self.sds_table["delay"][:, index],
            self.sds_parameters.sample_rate,
            self.sds_parameters.block_size,
        )
        self.plot_data_items["full_time_history_excitation"].setData(
            np.arange(self.sds_parameters.block_size) / self.sds_parameters.sample_rate, signal
        )
        self.update_tone_selection_ui()

    def compute_max_voltage(self, index=None):
        if self.sds_table is None:
            return
        if index is None:
            voltages = []
            for index in range(len(self.drive_names)):
                signal = sum_decayed_sines_reconstruction(
                    self.sds_table["frequency"],
                    self.sds_table["amplitude"][:, index],
                    self.sds_table["decay"][:, index],
                    self.sds_table["delay"][:, index],
                    self.sds_parameters.sample_rate,
                    self.sds_parameters.block_size,
                )
                voltages.append(max(abs(signal)))
            return voltages
        else:
            signal = sum_decayed_sines_reconstruction(
                self.sds_table["frequency"],
                self.sds_table["amplitude"][:, index],
                self.sds_table["decay"][:, index],
                self.sds_table["delay"][:, index],
                self.sds_parameters.sample_rate,
                self.sds_parameters.block_size,
            )
            return max(abs(signal))

    def update_all_voltages_ui(self):
        voltages = self.compute_max_voltage()
        if voltages is None:
            return
        self.parent_widget.excitation_voltage_list.clear()
        self.parent_widget.excitation_voltage_list.addItems([f"{volt:0.2f}" for volt in voltages])

    def update_voltage_ui(self, index):
        volt = self.compute_max_voltage(index)
        if volt is None:
            return
        self.parent_widget.excitation_voltage_list.item(index).setText(f"{volt:0.2f}")

    def update_tone_selection_ui(self):
        """This gets called when a different row of the table is selected."""
        if self.sds_table is None or self.drive_names is None:
            return
        index = self.parent_widget.excitation_selector.currentIndex()
        tone = self.parent_widget.sds_table.currentRow()
        if self.sds_table is not None:
            signal = sum_decayed_sines_reconstruction(
                self.sds_table["frequency"][tone],
                self.sds_table["amplitude"][tone, index],
                self.sds_table["decay"][tone, index],
                self.sds_table["delay"][tone, index],
                self.sds_parameters.sample_rate,
                self.sds_parameters.block_size,
            )
            self.plot_data_items["single_tone_time_history_excitation"].setData(
                np.arange(self.sds_parameters.block_size) / self.sds_parameters.sample_rate, signal
            )
        else:
            self.plot_data_items["single_tone_time_history_excitation"].setData(
                np.nan * np.ones(2),
                np.nan * np.ones(2),
            )

    def update_response_selector(self, item):
        index = self.parent_widget.response_error_list.row(item)
        self.parent_widget.response_selector.setCurrentIndex(index)

    def update_excitation_selector(self, item):
        index = self.parent_widget.excitation_voltage_list.row(item)
        self.parent_widget.excitation_selector.setCurrentIndex(index)
