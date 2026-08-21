from qtpy import uic, QtWidgets
from qtpy.QtCore import Qt
import os
import numpy as np
from typing import TYPE_CHECKING
from rattlesnake.user_interface.ui_utilities import AdaptiveNoWheelSpinBox
from rattlesnake.environment.sds_sys_id_utilities import (
    sum_decayed_sines,
    decayed_sine_table,
    srs,
    sum_decayed_sines_reconstruction,
)
from rattlesnake.utilities import DIRECTORY
from rattlesnake.environment.sds_sys_id_metadata import DecayStrategy

if TYPE_CHECKING:
    from .sds_sys_id_environment import SDSUI


class SDSSynthesizeDialog(QtWidgets.QDialog):
    """Dialog box for visualizing SDS synthesis for response channels"""

    def __init__(self, parent: "SDSUI"):
        """Initializes a dialog box for visualizing sum-of-decayed-sines for control channels

        Parameters
        ----------
        parent : SDSUI
            The SDSUI object that generated this dialog box and contains the parameters and SRS
            targets used to compute the sum of decayed sines.
        """
        super().__init__(parent.definition_widget)
        uic.loadUi(
            os.path.join(DIRECTORY, "user_interface", "ui_files", "srs_sds_synthesize_dialog.ui"),
            self,
        )
        self.parent_widget = parent
        self.metadata = parent.get_environment_metadata(parent.hardware_metadata.channel_list)
        self.setWindowTitle("Sum-of-Decayed-Sines Synthesis")
        self.plot_data_items = {}
        self.sds_tables = [None for i in parent.initialized_control_names]
        self.sds_signals = [None for i in parent.initialized_control_names]
        self.sds_srss = [None for i in parent.initialized_control_names]
        self.response_selector.addItems(parent.initialized_control_names)
        self.all_table_widgets = []

        plot_item = self.time_history_plot.getPlotItem()
        plot_item.showGrid(True, True, 0.25)
        plot_item.enableAutoRange()
        plot_item.getViewBox().enableAutoRange(enable=True)
        plot_item.addLegend()

        plot_item = self.srs_plot.getPlotItem()
        plot_item.showGrid(True, True, 0.25)
        plot_item.enableAutoRange()
        plot_item.getViewBox().enableAutoRange(enable=True)
        plot_item.setLogMode(True, True)
        plot_item.addLegend()

        self.plot_data_items["full_time_history"] = self.time_history_plot.getPlotItem().plot(
            np.array([0, self.metadata.block_size / self.metadata.sample_rate]),
            np.nan * np.ones(2),
            pen={"color": "b", "width": 1},
            name="Time History",
        )
        self.plot_data_items[
            "single_tone_time_history"
        ] = self.time_history_plot.getPlotItem().plot(
            np.array([0, self.metadata.block_size / self.metadata.sample_rate]),
            np.nan * np.ones(2),
            pen={"color": "r", "width": 1},
            name="Single Tone",
        )
        self.plot_data_items["specification_srs"] = self.srs_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.nan * np.ones(2),
            pen={"color": "b", "width": 1},
            name="Control SRS",
        )
        self.plot_data_items["specification_lower_limit"] = self.srs_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.nan * np.ones(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
            name="Limit",
        )
        self.plot_data_items["specification_upper_limit"] = self.srs_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.zeros(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
        )
        self.plot_data_items["sds_srs"] = self.srs_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.zeros(2),
            pen={"color": "r", "width": 1},
            name="SRS from SDS",
        )

        # Set up the table
        frequencies = self.metadata.get_sds_frequencies()
        num_rows = len(frequencies)
        self.sds_table.setRowCount(num_rows)
        for row in range(num_rows):
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(0, self.metadata.sample_rate / 2)
            spinbox.setSingleStep(1)
            spinbox.setValue(frequencies[row])
            spinbox.setKeyboardTracking(False)
            spinbox.setDecimals(4)
            spinbox.setReadOnly(True)
            spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            self.sds_table.setCellWidget(row, 0, spinbox)
            self.all_table_widgets.append(spinbox)
            # Amplitude
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(-1000000, 1000000)
            spinbox.setSingleStep(1)
            spinbox.setValue(0)
            spinbox.setKeyboardTracking(False)
            spinbox.valueChanged.connect(self.update_response)
            self.sds_table.setCellWidget(row, 1, spinbox)
            self.all_table_widgets.append(spinbox)
            # Delay
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(
                -self.metadata.block_size / self.metadata.sample_rate,
                self.metadata.block_size / self.metadata.sample_rate,
            )
            spinbox.setSingleStep(0.1)
            spinbox.setValue(0)
            spinbox.setKeyboardTracking(False)
            spinbox.valueChanged.connect(self.update_response)
            self.sds_table.setCellWidget(row, 2, spinbox)
            self.all_table_widgets.append(spinbox)
            # Decay
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(0, 10)
            spinbox.setSingleStep(0.01)
            spinbox.setValue(0)
            spinbox.setKeyboardTracking(False)
            spinbox.valueChanged.connect(self.update_response)
            self.sds_table.setCellWidget(row, 3, spinbox)
            self.all_table_widgets.append(spinbox)

        # Callbacks
        self.response_selector.currentIndexChanged.connect(self.update_response_channel)
        self.synthesize_current_button.clicked.connect(self.compute_current_sds)
        self.synthesize_all_button.clicked.connect(self.compute_all_sds)
        self.sds_table.itemSelectionChanged.connect(self.update_tone_selection)

        self.update_response_channel()

    def update_response(self):
        index = self.response_selector.currentIndex()
        delays = []
        decays = []
        amplitudes = []
        for row in range(self.sds_table.rowCount()):
            amplitudes.append(self.sds_table.cellWidget(row, 1).value())
            delays.append(self.sds_table.cellWidget(row, 2).value())
            decays.append(self.sds_table.cellWidget(row, 3).value())
        amplitudes = np.array(amplitudes)
        delays = np.array(delays)
        decays = np.array(decays)
        frequencies = self.metadata.get_sds_frequencies()
        self.sds_tables[index] = decayed_sine_table(
            frequencies, amplitudes[:, np.newaxis], decays[:, np.newaxis], delays[:, np.newaxis]
        )
        self.sds_signals[index] = sum_decayed_sines_reconstruction(
            self.sds_tables[index]["frequency"][:],
            self.sds_tables[index]["amplitude"][:, 0],
            self.sds_tables[index]["decay"][:, 0],
            self.sds_tables[index]["delay"][:, 0],
            self.metadata.sample_rate,
            self.metadata.block_size,
        )
        self.sds_srss[index] = srs(
            signal=self.sds_signals[index],
            dt=1 / self.metadata.sample_rate,
            frequencies=frequencies,
            damping=self.metadata.srs_data.srs_damping,
            spectrum_type=self.metadata.srs_data.srs_type.value
            * self.metadata.srs_data.srs_displacement.value,
        )
        self.update_response_channel()

    def update_response_channel(self):
        index = self.response_selector.currentIndex()
        self.plot_data_items["specification_srs"].setData(
            self.metadata.specification_data.frequencies,
            self.metadata.specification_data.srs_spec[:, index],
        )
        self.plot_data_items["specification_lower_limit"].setData(
            self.metadata.specification_data.frequencies,
            self.metadata.specification_data.srs_lower_limit[:, index],
        )
        self.plot_data_items["specification_upper_limit"].setData(
            self.metadata.specification_data.frequencies,
            self.metadata.specification_data.srs_upper_limit[:, index],
        )
        if self.sds_srss[index] is not None:
            self.plot_data_items["sds_srs"].setData(
                self.sds_srss[index][1], self.sds_srss[index][0]
            )
        else:
            self.plot_data_items["sds_srs"].setData(np.nan * np.ones(2), np.nan * np.ones(2))
        if self.sds_signals[index] is not None:
            self.plot_data_items["full_time_history"].setData(
                np.arange(self.sds_signals[index].size) / self.metadata.sample_rate,
                self.sds_signals[index],
            )
        else:
            self.plot_data_items["full_time_history"].setData(
                np.nan * np.ones(2), np.nan * np.ones(2)
            )
        if self.sds_tables[index] is not None:
            # Block all signals
            for widget in self.all_table_widgets:
                widget.blockSignals(True)
            for row, amplitude in enumerate(self.sds_tables[index]["amplitude"][:, 0]):
                self.sds_table.cellWidget(row, 1).setValue(amplitude)
            for row, delay in enumerate(self.sds_tables[index]["delay"][:, 0]):
                self.sds_table.cellWidget(row, 2).setValue(delay)
            for row, decay in enumerate(self.sds_tables[index]["decay"][:, 0]):
                self.sds_table.cellWidget(row, 3).setValue(decay)
            for widget in self.all_table_widgets:
                widget.blockSignals(False)
        else:
            # Block all signals
            for widget in self.all_table_widgets:
                widget.blockSignals(True)
            for row in range(self.sds_table.rowCount()):
                self.sds_table.cellWidget(row, 1).setValue(0)
                self.sds_table.cellWidget(row, 2).setValue(0)
                self.sds_table.cellWidget(row, 3).setValue(0)
            for widget in self.all_table_widgets:
                widget.blockSignals(False)

        self.update_tone_selection()

    def compute_sds(self, index: int):
        # print(f"In compute_sds: {self.metadata.get_sds_decays()=}")
        (
            time_signal,
            _,
            _,
            freq_w_comp,
            amp_w_comp,
            decay_w_comp,
            delay_w_comp,
        ) = sum_decayed_sines(
            sample_rate=self.metadata.sample_rate,
            block_size=self.metadata.block_size,
            sine_frequencies=self.metadata.get_sds_frequencies(),
            sine_decays=self.metadata.get_sds_decays(),
            srs_breakpoints=np.array(
                (
                    self.metadata.specification_data.frequencies,
                    self.metadata.specification_data.srs_spec[:, index],
                )
            ).T,
            srs_damping=self.metadata.srs_data.srs_damping,
            # We've defined the enumeration values to be consistent with the SRS functions,
            # so we can simply multiply the values together.
            srs_type=self.metadata.srs_data.srs_type.value
            * self.metadata.srs_data.srs_displacement.value,
            compensation_frequency=self.metadata.compensation_pulse_data.compensation_frequency,
            compensation_decay=self.metadata.compensation_pulse_data.compensation_decay,
            number_of_iterations=self.metadata.sds_data.iterations,
            convergence=self.metadata.sds_data.convergence,
            error_tolerance=self.metadata.sds_data.error_tolerance,
            scale_factor=self.metadata.sds_data.scale_factor,
            ignore_compensation_pulse=not self.metadata.compensation_pulse_data.use_compensation_pulse,
            verbose=True,
        )
        print(freq_w_comp)
        self.sds_tables[index] = decayed_sine_table(
            freq_w_comp[:-1],
            amp_w_comp[:-1, np.newaxis],
            decay_w_comp[:-1, np.newaxis],
            delay_w_comp[:-1, np.newaxis],
        )
        self.sds_signals[index] = time_signal
        self.sds_srss[index] = srs(
            signal=time_signal,
            dt=1 / self.metadata.sample_rate,
            frequencies=freq_w_comp[:-1],
            damping=self.metadata.srs_data.srs_damping,
            spectrum_type=self.metadata.srs_data.srs_type.value
            * self.metadata.srs_data.srs_displacement.value,
        )

    def compute_current_sds(self):
        index = self.response_selector.currentIndex()
        self.compute_sds(index)
        self.update_response_channel()

    def compute_all_sds(self):
        for index in range(len(self.sds_tables)):
            self.compute_sds(index)
        self.update_response_channel()

    def update_tone_selection(self):
        index = self.response_selector.currentIndex()
        tone = self.sds_table.currentRow()
        sds = self.sds_tables[index]
        if sds is not None:
            signal = sum_decayed_sines_reconstruction(
                sds["frequency"][tone],
                sds["amplitude"][tone, 0],
                sds["decay"][tone, 0],
                sds["delay"][tone, 0],
                self.metadata.sample_rate,
                self.metadata.block_size,
            )
            self.plot_data_items["single_tone_time_history"].setData(
                np.arange(self.metadata.block_size) / self.metadata.sample_rate, signal
            )
        else:
            self.plot_data_items["single_tone_time_history"].setData(
                np.nan * np.ones(2),
                np.nan * np.ones(2),
            )

    @staticmethod
    def show_dialog(parent: "SDSUI"):
        """Shows the SDS Synthesis Dialog Box

        Parameters
        ----------
        parent : SDSUI
            Sum of Decayed Sines Environment user interface from which parameters will be
            taken.
        """
        dialog = SDSSynthesizeDialog(parent)
        dialog.exec_()
