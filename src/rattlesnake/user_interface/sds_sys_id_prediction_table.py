import time
import numpy as np
from qtpy import uic, QtWidgets, QtGui
from qtpy.QtCore import Qt
import os
from rattlesnake.environment.sds_sys_id_metadata import (
    SRSParameters,
    SpecParameters,
    SDSMetadata,
)
from rattlesnake.engine import RattlesnakeController
from rattlesnake.environment.sds_sys_id_utilities import (
    SDSCommands,
    sum_decayed_sines_reconstruction,
    DecayedSineTable,
    decayed_sine_table,
    normalize_delays_for_synthesis,
    normalized_sds_table_for_synthesis,
)
from rattlesnake.utilities import DIRECTORY
from rattlesnake.user_interface.ui_utilities import AdaptiveNoWheelSpinBox, axis_label

DEBUG = False
DEBUG_DIRECTORY = "debug_data"


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
        other_voltage_lists=None,
        other_error_lists=None,
    ):
        if DEBUG:
            print("Calling SDSPredictionTable.__init__")
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
        self.drive_units = None
        self.response_units = None
        self.sds_parameters = sds_parameters
        self.frequency_locked = False
        self.amplitude_locked = False
        self.delay_locked = False
        self.decay_locked = False
        # Keep track of tables and tabs
        self.sds_table_widgets = []
        self.other_voltage_lists = [] if other_voltage_lists is None else other_voltage_lists
        self.other_error_lists = [] if other_error_lists is None else other_error_lists
        # Persistent calculated data
        self.predicted_response_time_history = None
        self.predicted_response_srs = None
        self.measured_response_time_history = None
        self.measured_response_srs = None
        self.drive_time_history = None

        # Guard against widgets firing while rebuilding the table
        self.rebuilding_table = False

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
        plot_item.setLabel("bottom", "Time (s)")

        self.plot_data_items = {}
        plot_item = self.parent_widget.response_display_plot.getPlotItem()
        plot_item.showGrid(True, True, 0.25)
        plot_item.enableAutoRange()
        plot_item.getViewBox().enableAutoRange(enable=True)
        plot_item.addLegend()
        plot_item.setLabel("bottom", "Time (s)")

        plot_item = self.parent_widget.response_srs_plot.getPlotItem()
        plot_item.showGrid(True, True, 0.25)
        plot_item.enableAutoRange()
        plot_item.getViewBox().enableAutoRange(enable=True)
        plot_item.setLogMode(True, True)
        plot_item.setLabel("bottom", "Frequency (Hz)")
        plot_item.addLegend()

        self.plot_data_items["full_time_history_excitation"] = (
            self.parent_widget.excitation_display_plot.getPlotItem().plot(
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
        )
        self.plot_data_items["single_tone_time_history_excitation"] = (
            self.parent_widget.excitation_display_plot.getPlotItem().plot(
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
        )

        self.plot_data_items["full_time_history_response_predicted"] = (
            self.parent_widget.response_display_plot.getPlotItem().plot(
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
                name="Predicted Time History",
            )
        )

        self.plot_data_items["full_time_history_response_measured"] = (
            self.parent_widget.response_display_plot.getPlotItem().plot(
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
                pen={"color": (0, 180, 0), "width": 1},
                name="Measured Time History",
            )
        )

        self.plot_data_items["specification_srs"] = (
            self.parent_widget.response_srs_plot.getPlotItem().plot(
                np.nan * np.array([0, 1]),
                np.nan * np.ones(2),
                pen={"color": "b", "width": 1},
                name="Control SRS",
            )
        )
        self.plot_data_items["specification_lower_limit"] = (
            self.parent_widget.response_srs_plot.getPlotItem().plot(
                np.nan * np.array([0, 1]),
                np.nan * np.ones(2),
                pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
                name="Limit",
            )
        )
        self.plot_data_items["specification_upper_limit"] = (
            self.parent_widget.response_srs_plot.getPlotItem().plot(
                np.nan * np.array([0, 1]),
                np.nan * np.zeros(2),
                pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
            )
        )
        self.plot_data_items["srs_predicted"] = (
            self.parent_widget.response_srs_plot.getPlotItem().plot(
                np.nan * np.array([0, 1]),
                np.nan * np.zeros(2),
                pen={"color": "r", "width": 1},
                name="SRS Predicted",
            )
        )
        self.plot_data_items["srs_measured"] = (
            self.parent_widget.response_srs_plot.getPlotItem().plot(
                np.nan * np.array([0, 1]),
                np.nan * np.zeros(2),
                pen={"color": (0, 180, 0), "width": 1},
                name="SRS Measured",
            )
        )

        # Update information
        self.update_ui()

    def update_names(
        self,
        drive_names: None | np.ndarray = None,
        response_names: None | np.ndarray = None,
        drive_units: None | np.ndarray = None,
        response_units: None | np.ndarray = None,
    ):
        if DEBUG:
            print("Calling SDSPredictionTable.update_names")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        self.drive_names = drive_names
        self.response_names = response_names
        self.drive_units = drive_units
        self.response_units = response_units
        self.update_names_ui()

    def update_parameters(self, parameters: SDSMetadata):
        if DEBUG:
            print("Calling SDSPredictionTable.update_parameters")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        self.sds_parameters = parameters
        self.reset_state()

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
        if DEBUG:
            print("Calling SDSPredictionTable.update_prediction_information")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
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
        self.update_all_response_errors_ui(
            other_error_lists=self.other_error_lists,
            use_measured=False,
        )
        for widget in self.sds_table_widgets:
            widget.blockSignals(False)

    def update_control_information(
        self,
        measured_response_time_history=None,
        measured_response_srs=None,
        run_sds_table=None,
    ):
        """
        Update the table and plots using measured post-hit control data.

        Parameters
        ----------
        measured_response_time_history : np.ndarray | None
            Measured control response time histories with shape
            (num_control_channels, num_samples)
        measured_response_srs : np.ndarray | None
            Measured control response SRS with shape
            (num_frequencies, num_control_channels)
        run_sds_table : DecayedSineTable | None
            Updated SDS table. If None, the current table is left unchanged.
        """
        if DEBUG:
            print("Calling SDSPredictionTable.update_control_information")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        for widget in self.sds_table_widgets:
            widget.blockSignals(True)

        if measured_response_time_history is not None:
            self.measured_response_time_history = measured_response_time_history
        if measured_response_srs is not None:
            self.measured_response_srs = measured_response_srs
        if run_sds_table is not None:
            self.sds_table = run_sds_table

        self.update_table_ui()
        self.update_drive_plot_ui()
        self.update_response_plot_ui()
        self.update_all_voltages_ui()
        self.update_all_response_errors_ui(
            other_error_lists=self.other_error_lists,
            use_measured=True,
        )

        for widget in self.sds_table_widgets:
            widget.blockSignals(False)

    def perform_prediction(self):
        if DEBUG:
            print("Calling SDSPredictionTable.perform_prediction")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        if self.rebuilding_table:
            return
        print("Performing Prediction!")
        if self.prediction_mode:
            self.rattlesnake.send_environment_command(
                self.environment_name, SDSCommands.SDS_TABLE_PREDICTION, self.sds_table
            )
        else:
            self.rattlesnake.send_environment_command(
                self.environment_name,
                SDSCommands.SDS_RUN_TABLE_PREDICTION,
                self.sds_table,
            )

    def synchronize_sds_table(self):
        """This function is called when a widget is modified in the table to update the internal
        representation of the sds_table"""
        if DEBUG:
            sender = self.parent_widget.sender()
            print("Calling SDSPredictionTable.synchronize_sds_table")
            print("  sender:", sender)
            if sender is not None:
                try:
                    print("  sender class:", sender.__class__.__name__)
                except Exception:
                    pass
                try:
                    print("  sender objectName:", sender.objectName())
                except Exception:
                    pass
                table = self.parent_widget.sds_table
                for row in range(table.rowCount()):
                    for col in range(table.columnCount()):
                        widget = table.cellWidget(row, col)
                        if widget is sender:
                            print(f"  sender at {row}, {col} in table.")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        if self.rebuilding_table:
            return
        index = self.parent_widget.excitation_selector.currentIndex()
        if DEBUG:
            print(f"Current Excitation Index Selector: {index}")
        if index < 0:
            return
        for col_index, name in enumerate(["frequency", "amplitude", "delay", "decay"]):
            for row_index in range(self.parent_widget.sds_table.rowCount()):
                if DEBUG:
                    print(
                        f"  At column index {col_index} ({name}) and row index {row_index} and tone index {index}"
                    )
                value = self.parent_widget.sds_table.cellWidget(row_index, col_index).value()
                if DEBUG:
                    print(f"    Widget Value: {value}")
                if col_index == 0:
                    if DEBUG:
                        print(f"  Previous self.sds_table value: {self.sds_table[name][row_index]}")
                    self.sds_table[name][row_index] = value
                else:
                    if DEBUG:
                        print(
                            f"    Previous self.sds_table value: {self.sds_table[name][row_index, index]}"
                        )
                    self.sds_table[name][row_index, index] = value
        self.update_voltage_ui(index)
        self.perform_prediction()

    def lock_table(
        self, frequencies=None, amplitudes=None, delays=None, decays=None, all_data=None
    ):
        """This function allows various columns of the table to be locked out"""
        if DEBUG:
            print("Calling SDSPredictionTable.lock_table")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
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
        if DEBUG:
            print("Calling SDSPredictionTable.lock_frequency")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
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
        if DEBUG:
            print("Calling SDSPredictionTable.lock_amplitude")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
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
        if DEBUG:
            print("Calling SDSPredictionTable.lock_delay")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
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
        if DEBUG:
            print("Calling SDSPredictionTable.lock_decay")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
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
        if DEBUG:
            print("Calling SDSPredictionTable.update_ui")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        self.update_names_ui()
        self.update_frequencies_ui()
        self.update_table_ui()
        self.update_response_plot_ui()
        self.update_drive_plot_ui()

    def update_names_ui(self):
        if DEBUG:
            print("Calling SDSPredictionTable.update_names_ui")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )

        # Update the drive names if there are names
        if self.drive_names is not None:
            self.parent_widget.excitation_selector.blockSignals(True)
            try:
                self.parent_widget.excitation_selector.clear()
                for name in self.drive_names:
                    self.parent_widget.excitation_selector.addItem(name)
            finally:
                self.parent_widget.excitation_selector.blockSignals(False)

        if self.response_names is not None:
            self.parent_widget.response_selector.blockSignals(True)
            try:
                self.parent_widget.response_selector.clear()
                for name in self.response_names:
                    self.parent_widget.response_selector.addItem(name)
            finally:
                self.parent_widget.response_selector.blockSignals(False)

    def update_frequencies_ui(self):
        if DEBUG:
            print("Calling SDSPredictionTable.update_frequencies_ui")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        if self.sds_parameters is None:
            return

        # Guard against widgets firing while rebuilding.
        self.rebuilding_table = True
        try:
            # Clear plots tied to prior SDS grid
            self.plot_data_items["full_time_history_excitation"].setData(
                np.nan * np.ones(2), np.nan * np.ones(2)
            )
            self.plot_data_items["single_tone_time_history_excitation"].setData(
                np.nan * np.ones(2), np.nan * np.ones(2)
            )
            self.plot_data_items["full_time_history_response_predicted"].setData(
                np.nan * np.ones(2), np.nan * np.ones(2)
            )
            self.plot_data_items["full_time_history_response_measured"].setData(
                np.nan * np.ones(2), np.nan * np.ones(2)
            )
            self.plot_data_items["srs_predicted"].setData(np.nan * np.ones(2), np.nan * np.ones(2))
            self.plot_data_items["srs_measured"].setData(np.nan * np.ones(2), np.nan * np.ones(2))

            frequencies = self.sds_parameters.get_sds_frequencies_w_compensation_pulse()
            expected_num_rows = len(frequencies)
            expected_num_drives = len(self.drive_names)

            # Preserve an existing SDS table if it already matches the current
            # environment definition. Only create a new zero-filled table if the
            # current one is missing or has the wrong shape.
            rebuild_data = (
                self.sds_table is None
                or self.sds_table.shape[0] != expected_num_rows
                or self.sds_table["amplitude"].shape[1] != expected_num_drives
            )

            if rebuild_data:
                self.sds_table = decayed_sine_table(
                    frequency=frequencies,
                    amplitude=np.zeros((expected_num_rows, expected_num_drives)),
                    decay=np.zeros((expected_num_rows, expected_num_drives)),
                    delay=np.zeros((expected_num_rows, expected_num_drives)),
                )
            else:
                # Keep existing amplitude / delay / decay data, but refresh the
                # frequency column from metadata in case it is authoritative.
                self.sds_table["frequency"] = frequencies

            self.sds_table_widgets = []
            self.parent_widget.sds_table.clearContents()
            num_rows = expected_num_rows
            self.parent_widget.sds_table.setRowCount(num_rows)
            self.sds_table_widgets = []
            self.parent_widget.sds_table.clearContents()
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

            self.update_drive_plot_ui()
            self.update_response_plot_ui()
        finally:
            self.rebuilding_table = False

    def update_table_ui(self):
        """This function is called to update the table values based on changes to the internal
        sds array of from changing the active drive channel."""
        if DEBUG:
            print("Calling SDSPredictionTable.update_table_ui")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        if self.sds_table is None or self.drive_names is None:
            return
        for widget in self.sds_table_widgets:
            widget.blockSignals(True)
        index = self.parent_widget.excitation_selector.currentIndex()
        if index < 0:
            return
        for col_index, name in enumerate(["frequency", "amplitude", "delay", "decay"]):
            for row_index in range(self.parent_widget.sds_table.rowCount()):
                if DEBUG:
                    print(
                        f"  Updating Table UI for widget at row {row_index} and column {col_index} ({name}) and tone {index}"
                    )
                widget = self.parent_widget.sds_table.cellWidget(row_index, col_index)
                if DEBUG:
                    print(
                        f"    widget {widget} {widget in self.sds_table_widgets=} {widget.signalsBlocked()=}"
                    )
                if col_index == 0:
                    print(f"    self.sds_table value: {self.sds_table[name][row_index]}")
                    widget.setValue(self.sds_table[name][row_index])
                else:
                    print(f"    self.sds_table value: {self.sds_table[name][row_index, index]}")
                    widget.setValue(self.sds_table[name][row_index, index])
        for widget in self.sds_table_widgets:
            widget.blockSignals(False)
        self.update_drive_plot_ui()

    def update_response_plot_ui(self):
        """This function is called to update the response plots"""
        if DEBUG:
            print("Calling SDSPredictionTable.update_response_plot_ui")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        if self.rebuilding_table:
            return
        if self.response_names is None or self.sds_parameters is None:
            return

        index = self.parent_widget.response_selector.currentIndex()

        unit = self.response_units[index] if self.response_units is not None else None
        self.parent_widget.response_display_plot.getPlotItem().setLabel(
            "left", axis_label("amplitude", "Amplitude", unit)
        )
        self.parent_widget.response_srs_plot.getPlotItem().setLabel(
            "left", axis_label("amplitude", "SRS", unit)
        )

        # Specification SRS
        abscissa = self.sds_parameters.specification_data.frequencies
        srs = self.sds_parameters.specification_data.srs_spec[:, index]
        lower = self.sds_parameters.specification_data.srs_lower_limit[:, index]
        upper = self.sds_parameters.specification_data.srs_upper_limit[:, index]
        self.plot_data_items["specification_srs"].setData(abscissa, srs)
        self.plot_data_items["specification_lower_limit"].setData(abscissa, lower)
        self.plot_data_items["specification_upper_limit"].setData(abscissa, upper)

        # Predicted SRS
        if self.predicted_response_srs is not None:
            abscissa = self.sds_parameters.get_truncated_specification_frequencies()
            srs = self.predicted_response_srs[:, index]
            self.plot_data_items["srs_predicted"].setData(abscissa, srs)
        else:
            self.plot_data_items["srs_predicted"].setData(
                np.nan * np.ones(2), np.nan * np.ones(2)
            )

        # Measured SRS
        if self.measured_response_srs is not None:
            abscissa = self.sds_parameters.get_truncated_specification_frequencies()
            srs = self.measured_response_srs[:, index]
            self.plot_data_items["srs_measured"].setData(abscissa, srs)
        else:
            self.plot_data_items["srs_measured"].setData(
                np.nan * np.ones(2), np.nan * np.ones(2)
            )

        # Predicted response time history
        if self.predicted_response_time_history is not None:
            th = self.predicted_response_time_history[index, :]
            abscissa = np.arange(th.size) / self.sds_parameters.sample_rate
            self.plot_data_items["full_time_history_response_predicted"].setData(
                abscissa, th
            )
        else:
            self.plot_data_items["full_time_history_response_predicted"].setData(
                np.nan * np.ones(2), np.nan * np.ones(2)
            )

        # Measured response time history
        if self.measured_response_time_history is not None:
            th = self.measured_response_time_history[index, :]
            abscissa = np.arange(th.size) / self.sds_parameters.sample_rate
            self.plot_data_items["full_time_history_response_measured"].setData(
                abscissa, th
            )
        else:
            self.plot_data_items["full_time_history_response_measured"].setData(
                np.nan * np.ones(2), np.nan * np.ones(2)
            )
        self.save_debug_snapshot("sds_prediction_table_response")

    def update_drive_plot_ui(self):
        """This function is called to update the drive plots"""
        if DEBUG:
            print("Calling SDSPredictionTable.update_drive_plot_ui")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        if self.rebuilding_table:
            return
        if self.sds_table is None or self.drive_names is None:
            return
        index = self.parent_widget.excitation_selector.currentIndex()
        signal = sum_decayed_sines_reconstruction(
            self.sds_table["frequency"],
            self.sds_table["amplitude"][:, index],
            self.sds_table["decay"][:, index],
            normalize_delays_for_synthesis(self.sds_table["delay"])[:, index],
            self.sds_parameters.sample_rate,
            self.sds_parameters.block_size,
        )
        self.plot_data_items["full_time_history_excitation"].setData(
            np.arange(self.sds_parameters.block_size) / self.sds_parameters.sample_rate,
            signal,
        )
        unit = self.drive_units[index] if self.drive_units is not None else None
        self.parent_widget.excitation_display_plot.getPlotItem().setLabel(
            "left", axis_label("amplitude", "Amplitude", unit)
        )
        self.update_tone_selection_ui()
        self.save_debug_snapshot("sds_prediction_table_drive")

    def compute_max_voltage(self, index=None):
        if DEBUG:
            print("Calling SDSPredictionTable.compute_max_voltage")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        if self.sds_table is None:
            return
        if index is None:
            voltages = []
            for index in range(len(self.drive_names)):
                signal = sum_decayed_sines_reconstruction(
                    self.sds_table["frequency"],
                    self.sds_table["amplitude"][:, index],
                    self.sds_table["decay"][:, index],
                    normalize_delays_for_synthesis(self.sds_table["delay"])[:, index],
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
                normalize_delays_for_synthesis(self.sds_table["delay"])[:, index],
                self.sds_parameters.sample_rate,
                self.sds_parameters.block_size,
            )
            return max(abs(signal))

    def update_all_voltages_ui(self):
        if DEBUG:
            print("Calling SDSPredictionTable.update_all_voltages_ui")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        voltages = self.compute_max_voltage()
        if voltages is None:
            return

        voltage_strings = [f"{volt:0.2f}" for volt in voltages]

        self.parent_widget.excitation_voltage_list.clear()
        self.parent_widget.excitation_voltage_list.addItems(voltage_strings)

        for voltage_list in self.other_voltage_lists:
            voltage_list.clear()
            voltage_list.addItems(voltage_strings)

    def update_voltage_ui(self, index):
        if DEBUG:
            print("Calling SDSPredictionTable.update_voltage_ui")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        volt = self.compute_max_voltage(index)
        if volt is None:
            return

        this_text = f"{volt:0.2f}"

        # Local list
        local_item = self.parent_widget.excitation_voltage_list.item(index)
        if local_item is None:
            # If the list hasn't been built yet, rebuild all rows
            self.update_all_voltages_ui()
        else:
            local_item.setText(this_text)

            # Mirror to any linked lists
            for voltage_list in self.other_voltage_lists:
                item = voltage_list.item(index)
                if item is None:
                    # fall back to full rebuild for that list set
                    self.update_all_voltages_ui()
                    break
                item.setText(this_text)

    def update_tone_selection_ui(self):
        """This gets called when a different row of the table is selected."""
        if DEBUG:
            print("Calling SDSPredictionTable.update_tone_selection_ui")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        if self.rebuilding_table:
            return
        if self.sds_table is None or self.drive_names is None:
            return
        index = self.parent_widget.excitation_selector.currentIndex()
        tone = self.parent_widget.sds_table.currentRow()
        if self.sds_table is not None:
            normalized_delays = normalize_delays_for_synthesis(self.sds_table["delay"])
            signal = sum_decayed_sines_reconstruction(
                self.sds_table["frequency"][tone],
                self.sds_table["amplitude"][tone, index],
                self.sds_table["decay"][tone, index],
                normalized_delays[tone, index],
                self.sds_parameters.sample_rate,
                self.sds_parameters.block_size,
            )
            self.plot_data_items["single_tone_time_history_excitation"].setData(
                np.arange(self.sds_parameters.block_size)
                / self.sds_parameters.sample_rate,
                signal,
            )
        else:
            self.plot_data_items["single_tone_time_history_excitation"].setData(
                np.nan * np.ones(2),
                np.nan * np.ones(2),
            )

    def compute_peak_response_error(self, index=None, use_measured=True):
        """
        Compute the worst-case dB error for each response channel relative to the
        truncated specification. Assumes the environment already computed SRS on
        the truncated specification frequencies.
        """
        if DEBUG:
            print("Calling SDSPredictionTable.compute_peak_response_error")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        if self.sds_parameters is None:
            return None, None

        srs_data = (
            self.measured_response_srs if use_measured else self.predicted_response_srs
        )
        if srs_data is None:
            return None, None

        target_srs = self.sds_parameters.get_truncated_specification_srs()
        lower_limit = self.sds_parameters.get_truncated_specification_lower_limit()
        upper_limit = self.sds_parameters.get_truncated_specification_upper_limit()

        def _compute_one(channel_index):
            measured = srs_data[:, channel_index]
            target = target_srs[:, channel_index]
            lower = lower_limit[:, channel_index]
            upper = upper_limit[:, channel_index]

            valid = (~np.isnan(measured)) & (~np.isnan(target)) & (target > 0)
            if np.any(valid):
                error_db = np.max(
                    np.abs(20 * np.log10(measured[valid] / target[valid]))
                )
            else:
                error_db = np.nan

            lower_valid = (~np.isnan(lower)) & (~np.isnan(measured))
            upper_valid = (~np.isnan(upper)) & (~np.isnan(measured))

            lower_exceeded = np.any(measured[lower_valid] < lower[lower_valid])
            upper_exceeded = np.any(measured[upper_valid] > upper[upper_valid])

            warning_flag = lower_exceeded or upper_exceeded
            return error_db, warning_flag

        if index is not None:
            return _compute_one(index)

        errors = []
        warnings = []
        for channel_index in range(srs_data.shape[1]):
            err, warn = _compute_one(channel_index)
            errors.append(err)
            warnings.append(warn)

        return errors, warnings

    def reset_state(self):
        """
        Reset all cached state that depends on the previous SDS/environment definition.
        """
        if DEBUG:
            print("Calling SDSPredictionTable.reset_state")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        self.sds_table = None
        self.predicted_response_time_history = None
        self.predicted_response_srs = None
        self.measured_response_time_history = None
        self.measured_response_srs = None
        self.drive_time_history = None

        self.parent_widget.sds_table.clearContents()
        self.parent_widget.sds_table.setRowCount(0)
        self.parent_widget.sds_table.clearSelection()

        for key in [
            "full_time_history_excitation",
            "single_tone_time_history_excitation",
            "full_time_history_response_predicted",
            "full_time_history_response_measured",
            "specification_srs",
            "specification_lower_limit",
            "specification_upper_limit",
            "srs_predicted",
            "srs_measured",
        ]:
            if key in self.plot_data_items:
                self.plot_data_items[key].setData(
                    np.nan * np.ones(2),
                    np.nan * np.ones(2),
                )

        self.parent_widget.excitation_voltage_list.clear()
        self.parent_widget.response_error_list.clear()

    def save_debug_snapshot(self, tag="prediction_table"):
        if DEBUG:
            print("Calling SDSPredictionTable.save_debug_snapshot")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        if not DEBUG:
            return

        os.makedirs(DEBUG_DIRECTORY, exist_ok=True)
        filename = os.path.join(
            DEBUG_DIRECTORY,
            f"{tag}_{self.environment_name}_{int(time.time() * 1000)}.npz",
        )

        output_dict = {
            "timestamp": np.array(time.time()),
            "environment_name": np.array(self.environment_name),
            "prediction_mode": np.array(self.prediction_mode),
            "selected_excitation_index": np.array(
                self.parent_widget.excitation_selector.currentIndex()
            ),
            "selected_response_index": np.array(
                self.parent_widget.response_selector.currentIndex()
            ),
        }

        if self.sds_parameters is not None:
            output_dict["sds_frequencies"] = np.array(self.sds_parameters.get_sds_frequencies())
            output_dict["sds_frequencies_with_comp"] = np.array(
                self.sds_parameters.get_sds_frequencies_w_compensation_pulse()
            )
            output_dict["truncated_specification_frequencies"] = np.array(
                self.sds_parameters.get_truncated_specification_frequencies()
            )
            output_dict["specification_frequencies"] = np.array(
                self.sds_parameters.specification_data.frequencies
            )
            output_dict["specification_srs"] = np.array(
                self.sds_parameters.specification_data.srs_spec
            )
            output_dict["specification_lower_limit"] = np.array(
                self.sds_parameters.specification_data.srs_lower_limit
            )
            output_dict["specification_upper_limit"] = np.array(
                self.sds_parameters.specification_data.srs_upper_limit
            )
            output_dict["sample_rate"] = np.array(self.sds_parameters.sample_rate)
            output_dict["block_size"] = np.array(self.sds_parameters.block_size)

        if self.sds_table is not None:
            output_dict["table_frequency"] = np.array(self.sds_table["frequency"])
            output_dict["table_amplitude"] = np.array(self.sds_table["amplitude"])
            output_dict["table_decay"] = np.array(self.sds_table["decay"])
            output_dict["table_delay"] = np.array(self.sds_table["delay"])
            output_dict["table_delay_normalized"] = np.array(
                normalize_delays_for_synthesis(self.sds_table["delay"])
            )

        if self.drive_time_history is not None:
            output_dict["predicted_drive_time_history"] = np.array(self.drive_time_history)

        if self.predicted_response_time_history is not None:
            output_dict["predicted_response_time_history"] = np.array(
                self.predicted_response_time_history
            )

        if self.measured_response_time_history is not None:
            output_dict["measured_response_time_history"] = np.array(
                self.measured_response_time_history
            )

        if self.predicted_response_srs is not None:
            output_dict["predicted_response_srs"] = np.array(self.predicted_response_srs)

        if self.measured_response_srs is not None:
            output_dict["measured_response_srs"] = np.array(self.measured_response_srs)

        np.savez(filename, **output_dict)

    def update_all_response_errors_ui(self, other_error_lists=None, use_measured=True):
        """
        Update the response error list(s) using the current measured or predicted SRS.

        Parameters
        ----------
        other_error_lists : list[QListWidget] | None
            Additional list widgets to mirror the same values into.
        use_measured : bool
            If True, use measured_response_srs. Otherwise use predicted_response_srs.
        """
        if DEBUG:
            print("Calling SDSPredictionTable.update_all_response_errors_ui")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        if other_error_lists is None:
            other_error_lists = []

        errors, warnings = self.compute_peak_response_error(use_measured=use_measured)
        if errors is None:
            return

        all_lists = [self.parent_widget.response_error_list] + list(other_error_lists)

        # Styling depending on whether these are measured or predicted values
        if use_measured:
            text_brush = None
            normal_background = None
            warning_background = Qt.yellow
        else:
            text_brush = Qt.gray
            normal_background = None
            warning_background = QtGui.QColor(255, 255, 200)  # pale yellow

        for error_list in all_lists:
            error_list.clear()
            for err, warn in zip(errors, warnings):
                item = QtWidgets.QListWidgetItem(
                    "nan" if np.isnan(err) else f"{err:0.3f}"
                )

                if text_brush is not None:
                    item.setForeground(text_brush)

                if warn:
                    item.setBackground(warning_background)

                error_list.addItem(item)

    def update_response_error_ui(
        self, index, other_error_lists=None, use_measured=True
    ):
        if DEBUG:
            print("Calling SDSPredictionTable.update_response_error_ui")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        if other_error_lists is None:
            other_error_lists = []

        err, warn = self.compute_peak_response_error(
            index=index, use_measured=use_measured
        )
        if err is None:
            return

        all_lists = [self.parent_widget.response_error_list] + list(other_error_lists)
        text = "nan" if np.isnan(err) else f"{err:0.3f}"

        if use_measured:
            text_brush = None
            warning_background = Qt.yellow
        else:
            text_brush = Qt.gray
            warning_background = QtGui.QColor(255, 255, 200)

        for error_list in all_lists:
            item = error_list.item(index)
            if item is None:
                continue
            item.setText(text)

            if text_brush is not None:
                item.setForeground(text_brush)

            if warn:
                item.setBackground(warning_background)

    def update_response_selector(self, item):
        if DEBUG:
            print("Calling SDSPredictionTable.update_response_selector")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        index = self.parent_widget.response_error_list.row(item)
        self.parent_widget.response_selector.setCurrentIndex(index)

    def update_excitation_selector(self, item):
        if DEBUG:
            print("Calling SDSPredictionTable.update_excitation_selector")
            print(
                "  self.sds_table is None"
                if self.sds_table is None
                else f"  {self.sds_table['delay'][0, :]}"
            )
        index = self.parent_widget.excitation_voltage_list.row(item)
        self.parent_widget.excitation_selector.setCurrentIndex(index)
