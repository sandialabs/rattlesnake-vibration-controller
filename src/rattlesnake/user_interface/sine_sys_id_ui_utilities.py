import pyqtgraph as pqtg
from qtpy import QtWidgets, uic
from qtpy.QtCore import Qt, QLocale  # pylint: disable=no-name-in-module

from rattlesnake.user_interface.ui_utilities import (
    sine_sweep_table_ui_path,
    filter_explorer_ui_path,
)
from rattlesnake.user_interface.ui_utilities import VaryingNumberOfLinePlot
from rattlesnake.environment.sine_sys_id_utilities import (
    load_specification,
    SineSpecification,
    digital_tracking_filter_generator,
    vold_kalman_filter_generator,
)
from rattlesnake.utilities import wrap
import numpy as np


class NoWheelSpinBox(QtWidgets.QDoubleSpinBox):
    """A simple class to remove the scroll wheel capability from a spin box"""

    def wheelEvent(self, event):  # pylint: disable=invalid-name
        """Capture the wheel event but ignore it"""
        event.ignore()


class AdaptiveNoWheelSpinBox(NoWheelSpinBox):
    """A spinbox that changes number of decimals based on the value provided"""

    localization = QLocale(QLocale.English, QLocale.UnitedStates)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setDecimals(10)

    def textFromValue(self, value):  # pylint: disable=invalid-name
        """Gets the text to show in the spinbox based on the value stored in the spinbox"""
        return AdaptiveNoWheelSpinBox.localization.toString(value, "g", self.decimals())


class NoWheelComboBox(QtWidgets.QComboBox):
    """A simple class to remove the scroll wheel capability from a combo box"""

    def wheelEvent(self, event):  # pylint: disable=invalid-name
        """Capture the wheel event but ignore it"""
        event.ignore()


class SineSweepTable:
    """A class representing a breakpoint table defining a sine sweep"""

    def __init__(
        self,
        parent_tabwidget: QtWidgets.QTabWidget,
        update_specification_function,
        remove_function,
        control_names,
        data_acquisition_parameters,
    ):
        """Initializes a sine sweep table to represent the breakpoints of a sine tone

        Parameters
        ----------
        parent_tabwidget : QtWidgets.QTabWidget
            The parent tabwidget in the sine ui class, which is needed to
            propogate changes in this widget back up to the main UI class
        update_specification_function : function
            The function to call to update the specification when the values
            in this table have changed
        remove_function : function
            The function to call when we remove a table from the tab widget
        control_names : list of str
            A list of strings to be used as the control channel names in the
            table
        data_acquisition_parameters : DataAcquisitionParameters
            The data acquisition parameters, including sample rate
        """
        self.parent_tabwidget = parent_tabwidget
        self.update_specification_function = update_specification_function
        self.remove_function = remove_function
        self.control_names = control_names
        self.data_acquisition_parameters = data_acquisition_parameters
        self.widget = QtWidgets.QWidget()
        uic.loadUi(sine_sweep_table_ui_path, self.widget)
        self.index = self.parent_tabwidget.count() - 1
        self.parent_tabwidget.insertTab(self.index, self.widget, f"Sine {self.index+1}")
        self.widget.name_editor.setText(f"Sine {self.index+1}")
        self.parent_tabwidget.setCurrentIndex(self.index)
        self.connect_callbacks()
        self.clear_and_update_specification_table()

    def connect_callbacks(self):
        """Connects the widgets in the UI to methods of the object"""
        self.widget.add_breakpoint_button.clicked.connect(self.add_breakpoint)
        self.widget.remove_breakpoint_button.clicked.connect(self.remove_breakpoint)
        self.widget.load_breakpoints_button.clicked.connect(self.load_specification)
        self.widget.name_editor.editingFinished.connect(self.update_name)
        self.widget.start_time_selector.valueChanged.connect(self.update_specification_function)
        self.widget.remove_tone_button.clicked.connect(self.remove_tone)

    def add_breakpoint(self):
        """Adds a breakpoint to the table"""
        selected_indices = self.widget.breakpoint_table.selectedIndexes()
        if selected_indices:
            selected_row = selected_indices[0].row()
        else:
            # If no row is selected, add the row at the start
            selected_row = 0
        control_names = self.control_names
        self.widget.breakpoint_table.insertRow(selected_row)
        self.widget.warning_table.insertRow(selected_row)
        self.widget.abort_table.insertRow(selected_row)
        # Frequency display, Breakpoint Table
        spinbox = AdaptiveNoWheelSpinBox()
        spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
        spinbox.setSingleStep(1)
        spinbox.setValue(0)
        spinbox.setKeyboardTracking(False)
        spinbox.valueChanged.connect(self.update_specification_function)
        self.widget.breakpoint_table.setCellWidget(selected_row, 0, spinbox)
        # Frequency display, warning table
        spinbox = AdaptiveNoWheelSpinBox()
        spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
        spinbox.setSingleStep(1)
        spinbox.setValue(0)
        spinbox.setKeyboardTracking(False)
        spinbox.setReadOnly(True)
        spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
        self.widget.warning_table.setCellWidget(selected_row, 0, spinbox)
        # Frequency display, abort table
        spinbox = AdaptiveNoWheelSpinBox()
        spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
        spinbox.setSingleStep(1)
        spinbox.setValue(0)
        spinbox.setKeyboardTracking(False)
        spinbox.setReadOnly(True)
        self.widget.abort_table.setCellWidget(selected_row, 0, spinbox)
        # Linear or logarithmic selector
        combobox = NoWheelComboBox()
        combobox.addItems(["Linear", "Logarithmic"])
        combobox.setCurrentIndex(0)
        combobox.currentIndexChanged.connect(self.update_specification_function)
        self.widget.breakpoint_table.setCellWidget(selected_row, 1, combobox)
        # Rate selector
        spinbox = AdaptiveNoWheelSpinBox()
        spinbox.setRange(-1000000, 1000000)
        spinbox.setSingleStep(1)
        spinbox.setValue(1)
        spinbox.setSuffix(" Hz/s")
        spinbox.setKeyboardTracking(False)
        spinbox.valueChanged.connect(self.update_specification_function)
        self.widget.breakpoint_table.setCellWidget(selected_row, 2, spinbox)
        # All of the individual values
        for j in range(len(control_names)):
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(0, 1000000)
            spinbox.setSingleStep(1)
            spinbox.setValue(1)
            spinbox.setKeyboardTracking(False)
            spinbox.valueChanged.connect(self.update_specification_function)
            self.widget.breakpoint_table.setCellWidget(selected_row, 3 + j * 2, spinbox)
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(-1000000, 1000000)
            spinbox.setSingleStep(1)
            spinbox.setValue(0)
            spinbox.setKeyboardTracking(False)
            spinbox.valueChanged.connect(self.update_specification_function)
            self.widget.breakpoint_table.setCellWidget(selected_row, 4 + j * 2, spinbox)
            for k in range(4):
                if selected_row == 0 and k in (0, 2):
                    item = self.widget.warning_table.item(selected_row, 1 + k + j * 4)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        self.widget.warning_table.setItem(selected_row, 1 + k + j * 4, item)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item = self.widget.abort_table.item(selected_row, 1 + k + j * 4)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        self.widget.abort_table.setItem(selected_row, 1 + k + j * 4, item)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                spinbox = AdaptiveNoWheelSpinBox()
                spinbox.setRange(0, 1000000)
                spinbox.setSingleStep(1)
                spinbox.setValue(0)
                spinbox.setKeyboardTracking(False)
                spinbox.setSpecialValueText("Disabled")
                spinbox.valueChanged.connect(self.update_specification_function)
                self.widget.warning_table.setCellWidget(
                    selected_row + (1 if selected_row == 0 and k in (0, 2) else 0),
                    1 + k + j * 4,
                    spinbox,
                )
                spinbox = AdaptiveNoWheelSpinBox()
                spinbox.setRange(0, 1000000)
                spinbox.setSingleStep(1)
                spinbox.setValue(0)
                spinbox.setKeyboardTracking(False)
                spinbox.setSpecialValueText("Disabled")
                spinbox.valueChanged.connect(self.update_specification_function)
                self.widget.abort_table.setCellWidget(
                    selected_row + (1 if selected_row == 0 and k in (0, 2) else 0),
                    1 + k + j * 4,
                    spinbox,
                )
        self.update_specification_function()

    def remove_breakpoint(self):
        """Removes a breakpoint from the table"""
        selected_indices = self.widget.breakpoint_table.selectedIndexes()
        if selected_indices:
            selected_row = selected_indices[0].row()
        else:
            # If no row is selected, remove the last row
            selected_row = self.widget.breakpoint_table.rowCount() - 1
        if selected_row == self.widget.breakpoint_table.rowCount() - 1:
            last_row = True
        else:
            last_row = False
        if selected_row == 0:
            first_row = True
        else:
            first_row = False
        self.widget.breakpoint_table.removeRow(selected_row)
        self.widget.warning_table.removeRow(selected_row)
        self.widget.abort_table.removeRow(selected_row)
        if last_row:
            new_last_row_index = self.widget.breakpoint_table.rowCount() - 1
            for column in [1, 2]:
                widget = self.widget.breakpoint_table.cellWidget(new_last_row_index, column)
                if widget:
                    # Remove the widget from the cell
                    self.widget.breakpoint_table.removeCellWidget(new_last_row_index, column)
                    widget.deleteLater()
                item = self.widget.breakpoint_table.item(new_last_row_index, column)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    self.widget.breakpoint_table.setItem(new_last_row_index, column, item)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            for column in np.arange(2, self.widget.warning_table.columnCount(), 2):
                for table in [self.widget.warning_table, self.widget.abort_table]:
                    widget = table.cellWidget(new_last_row_index, column)
                    if widget:
                        # Remove the widget from the cell
                        table.removeCellWidget(new_last_row_index, column)
                        widget.deleteLater()
                    item = table.item(new_last_row_index, column)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        table.setItem(new_last_row_index, column, item)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        if first_row:
            for column in np.arange(1, self.widget.warning_table.columnCount(), 2):
                for table in [self.widget.warning_table, self.widget.abort_table]:
                    widget = table.cellWidget(0, column)
                    if widget:
                        # Remove the widget from the cell
                        table.removeCellWidget(0, column)
                        widget.deleteLater()
                    item = table.item(0, column)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        table.setItem(0, column, item)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.update_specification_function()

    def load_specification(self, clicked, filename=None):  # pylint: disable=unused-argument
        """Loads a breakpoint table using a dialog or the specified filename

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
                self.widget,
                "Select Specification File",
                filter="Numpy or Mat (*.npy *.npz *.mat)",
            )
            if filename == "":
                return
        (
            frequencies,
            amplitudes,
            phases,  # Degrees
            sweep_types,
            sweep_rates,
            warnings,
            aborts,
            start_time,
            name,
        ) = load_specification(filename)
        self.clear_and_update_specification_table(
            frequencies,
            amplitudes,
            phases,  # Degrees
            sweep_types,
            sweep_rates,
            warnings,
            aborts,
            start_time,
            name,
        )
        self.update_specification_function()

    def clear_and_update_specification_table(
        self,
        frequencies=None,
        amplitudes=None,
        phases=None,
        sweep_types=None,
        sweep_rates=None,
        warning_amplitudes=None,
        abort_amplitudes=None,
        start_time=None,
        sine_name=None,
        control_names=None,
    ):
        """Clears the table and updates it with the optional parameters supplied.

        Parameters
        ----------
        frequencies : ndarray, optional
            A 1D array containing the frequencies to use as the breakpoints. By
            default, a table consisting of two breakpoints will be specified.
        amplitudes : ndarray, optional
            A 2D array consisting of amplitudes for (channel, frequency) pairs.
            If not specified, an amplitude of zero will be used.
        phases : ndarray, optional
            A 2D array consisting of phases for (channel, frequency) pairs.
            If not specified, an amplitude of zero will be used.  Phases
            are specified in degrees.
        sweep_types : ndarray or list of strings, optional
            A 1D array of strings to use as the sweep type.  They should
            be one of lin, log, linear, or logarithmic.  Linear is used
            if not specified.
        sweep_rates : ndarray, optional
            A 1D array of values to use as the sweep rate.  They should
            be in Hz/s for linear sweeps or octave per minute for
            logarithmic sweeps.
        warning_amplitudes : ndarray, optional
            A 4D ndarray with shape 2, 2, num_channels, num_frequencies.
            The first dimension specifies upper and lower limits, the
            second dimension specifies frequencies greater or lower than
            the frequency breakpoint.  If a value is not desired, it
            should be set to nan.
        abort_amplitudes : ndarray, optional
            A 4D ndarray with shape 2, 2, num_channels, num_frequencies.
            The first dimension specifies upper and lower limits, the
            second dimension specifies frequencies greater or lower than
            the frequency breakpoint.  If a value is not desired, it
            should be set to nan.
        start_time : float, optional
            The starting time for the specified sine tone.  If not specified,
            it will be set to 0
        sine_name : str, optional
            The name of the sine tone used in the software.
        control_names : array of str, optional
            Channel names to use in the table.  If not specified, the
            existing channel names will be used.
        """
        # print(f'Called clear_and_update_specification with {control_names=}')
        # print(f'Called clear_and_update_specification with {start_time=}')
        # print(f'Called clear_and_update_specification with {sine_name=}')
        if control_names is not None:
            self.control_names = control_names
        control_names = self.control_names
        if frequencies is None:
            num_rows = 2
        else:
            num_rows = frequencies.size
        self.widget.breakpoint_table.clear()
        self.widget.breakpoint_table.setRowCount(num_rows)
        self.widget.breakpoint_table.setColumnCount(3 + 2 * len(control_names))
        self.widget.warning_table.setRowCount(num_rows)
        self.widget.warning_table.setColumnCount(1 + 4 * len(control_names))
        self.widget.abort_table.setRowCount(num_rows)
        self.widget.abort_table.setColumnCount(1 + 4 * len(control_names))
        breakpoint_header_labels = ["Frequency", "Sweep Type", "Sweep Rate"]
        other_header_labels = ["Frequency"]
        for name in control_names:
            breakpoint_header_labels.append(name + " Amplitude")
            breakpoint_header_labels.append(name + " Phase")
            other_header_labels.append(name + " Lower Left")
            other_header_labels.append(name + " Lower Right")
            other_header_labels.append(name + " Upper Left")
            other_header_labels.append(name + " Upper Right")
        self.widget.breakpoint_table.setHorizontalHeaderLabels(breakpoint_header_labels)
        self.widget.warning_table.setHorizontalHeaderLabels(other_header_labels)
        self.widget.abort_table.setHorizontalHeaderLabels(other_header_labels)
        # Set up widgets in the table
        for row in range(num_rows):
            # Frequency Breakpoint
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
            spinbox.setSingleStep(1)
            if frequencies is None:
                spinbox.setValue(0)
            else:
                spinbox.setValue(frequencies[row])
            spinbox.setKeyboardTracking(False)
            spinbox.setDecimals(4)
            spinbox.valueChanged.connect(self.update_specification_function)
            self.widget.breakpoint_table.setCellWidget(row, 0, spinbox)
            # Frequency display, warning table
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
            spinbox.setSingleStep(1)
            if frequencies is None:
                spinbox.setValue(0)
            else:
                spinbox.setValue(frequencies[row])
            spinbox.setKeyboardTracking(False)
            spinbox.setReadOnly(True)
            spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            self.widget.warning_table.setCellWidget(row, 0, spinbox)
            # Frequency display, abort table
            spinbox = AdaptiveNoWheelSpinBox()
            spinbox.setRange(0, self.data_acquisition_parameters.sample_rate / 2)
            spinbox.setSingleStep(1)
            if frequencies is None:
                spinbox.setValue(0)
            else:
                spinbox.setValue(frequencies[row])
            spinbox.setKeyboardTracking(False)
            spinbox.setReadOnly(True)
            spinbox.setButtonSymbols(AdaptiveNoWheelSpinBox.NoButtons)
            self.widget.abort_table.setCellWidget(row, 0, spinbox)
            # Rate and type
            if row < num_rows - 1:
                combobox = NoWheelComboBox()
                combobox.addItems(["Linear", "Logarithmic"])
                if sweep_types is not None:
                    if str(sweep_types[row]).lower() in ["lin", "linear"]:
                        combobox.setCurrentIndex(0)
                    else:
                        combobox.setCurrentIndex(1)
                combobox.currentIndexChanged.connect(self.update_specification_function)
                self.widget.breakpoint_table.setCellWidget(row, 1, combobox)
                spinbox = AdaptiveNoWheelSpinBox()
                spinbox.setRange(-1000000, 1000000)
                spinbox.setSingleStep(1)
                if sweep_rates is not None:
                    spinbox.setValue(sweep_rates[row])
                else:
                    spinbox.setValue(1)
                if combobox.currentIndex() == 0:
                    spinbox.setSuffix(" Hz/s")
                else:
                    spinbox.setSuffix(" oct/min")
                spinbox.setKeyboardTracking(False)
                spinbox.valueChanged.connect(self.update_specification_function)
                self.widget.breakpoint_table.setCellWidget(row, 2, spinbox)
            else:
                item = self.widget.breakpoint_table.item(row, 1)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    self.widget.breakpoint_table.setItem(row, 1, item)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item = self.widget.breakpoint_table.item(row, 2)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    self.widget.breakpoint_table.setItem(row, 2, item)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            # Amplitude and Phases
            for j in range(len(control_names)):
                spinbox = AdaptiveNoWheelSpinBox()
                spinbox.setRange(0, 1000000)
                spinbox.setSingleStep(1)
                if amplitudes is None:
                    spinbox.setValue(1)
                else:
                    spinbox.setValue(amplitudes[j, row])
                spinbox.setKeyboardTracking(False)
                spinbox.valueChanged.connect(self.update_specification_function)
                self.widget.breakpoint_table.setCellWidget(row, 3 + j * 2, spinbox)
                spinbox = AdaptiveNoWheelSpinBox()
                spinbox.setRange(-1000000, 1000000)
                spinbox.setSingleStep(1)
                if phases is None:
                    spinbox.setValue(0)
                else:
                    spinbox.setValue(phases[j, row])
                spinbox.valueChanged.connect(self.update_specification_function)
                spinbox.setKeyboardTracking(False)
                self.widget.breakpoint_table.setCellWidget(row, 4 + j * 2, spinbox)
                for k in range(4):
                    spinbox = AdaptiveNoWheelSpinBox()
                    spinbox.setRange(0, 1000000)
                    spinbox.setSingleStep(1)
                    if (
                        row == 0 and k in (0, 2)
                    ) or (  # If first frequency and looking at left side
                        row == num_rows - 1 and k in (1, 3)
                    ):  # or if last frequency and looking at right side
                        item = self.widget.warning_table.item(row, 1 + k + j * 4)
                        if item is None:
                            item = QtWidgets.QTableWidgetItem()
                            self.widget.warning_table.setItem(row, 1 + k + j * 4, item)
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        item = self.widget.abort_table.item(row, 1 + k + j * 4)
                        if item is None:
                            item = QtWidgets.QTableWidgetItem()
                            self.widget.abort_table.setItem(row, 1 + k + j * 4, item)
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    else:
                        if warning_amplitudes is None:
                            spinbox.setValue(0)
                        else:
                            val = warning_amplitudes[np.unravel_index(k, (2, 2)) + (j, row)]
                            spinbox.setValue(0 if np.isnan(val) else val)
                        spinbox.setKeyboardTracking(False)
                        spinbox.setSpecialValueText("Disabled")
                        spinbox.valueChanged.connect(self.update_specification_function)
                        self.widget.warning_table.setCellWidget(row, 1 + k + j * 4, spinbox)
                        spinbox = AdaptiveNoWheelSpinBox()
                        spinbox.setRange(0, 1000000)
                        spinbox.setSingleStep(1)
                        if abort_amplitudes is None:
                            spinbox.setValue(0)
                        else:
                            val = abort_amplitudes[np.unravel_index(k, (2, 2)) + (j, row)]
                            spinbox.setValue(0 if np.isnan(val) else val)
                        spinbox.setKeyboardTracking(False)
                        spinbox.setSpecialValueText("Disabled")
                        spinbox.valueChanged.connect(self.update_specification_function)
                        self.widget.abort_table.setCellWidget(row, 1 + k + j * 4, spinbox)
        if sine_name is not None:
            self.widget.name_editor.setText(sine_name)
            self.update_name()
        if start_time is not None:
            self.widget.start_time_selector.setValue(start_time)

    def update_name(self):
        """Called when the name of the sine tone is changed"""
        self.parent_tabwidget.setTabText(self.index, self.widget.name_editor.text())

    def remove_tone(self):
        """Called when the remove button is pressed"""
        self.remove_function(self.index)

    def get_specification(self):
        """Computes a sine sweep specification from the table"""
        num_control = (self.widget.breakpoint_table.columnCount() - 3) // 2
        spec = SineSpecification(
            self.widget.name_editor.text(),
            self.widget.start_time_selector.value(),
            num_control,
            self.widget.breakpoint_table.rowCount(),
        )
        for row, spec_row in enumerate(spec.breakpoint_table):
            spec_row["frequency"] = self.widget.breakpoint_table.cellWidget(row, 0).value()
            if row < len(spec.breakpoint_table) - 1:
                spec_row["sweep_type"] = self.widget.breakpoint_table.cellWidget(
                    row, 1
                ).currentIndex()
                spec_row["sweep_rate"] = self.widget.breakpoint_table.cellWidget(row, 2).value()
            for i in range(num_control):
                spec_row["amplitude"][i] = self.widget.breakpoint_table.cellWidget(
                    row, 3 + 2 * i
                ).value()
                spec_row["phase"][i] = (
                    self.widget.breakpoint_table.cellWidget(row, 4 + 2 * i).value() * np.pi / 180
                )  # Convert degrees to radians for all calculations
                for k in range(4):
                    ind = np.unravel_index(k, (2, 2))
                    if (row == 0 and k in (0, 2)) or (
                        row == len(spec.breakpoint_table) - 1 and k in (1, 3)
                    ):
                        spec_row["warning"][ind + (i,)] = np.nan
                        spec_row["abort"][ind + (i,)] = np.nan
                    else:
                        val = self.widget.warning_table.cellWidget(row, 1 + k + i * 4).value()
                        spec_row["warning"][ind + (i,)] = np.nan if val == 0 else val
                        val = self.widget.abort_table.cellWidget(row, 1 + k + i * 4).value()
                        spec_row["abort"][ind + (i,)] = np.nan if val == 0 else val
        return spec


class FilterExplorer(QtWidgets.QDialog):
    """Dialog box for exploring the Vold-Kalman Filter Settings"""

    @staticmethod
    def explore_filter_settings(
        channel_names,
        order_names,
        specifications,
        current_filter_type,
        current_tracking_filter_cutoff,
        current_tracking_filter_order,
        current_filter_order,
        current_bandwidth,
        current_block_size,
        current_overlap,
        sample_rate,
        ramp_time,
        acquire_size,
        parent=None,
    ):
        """
        Brings up the explore filter dialog box

        Parameters
        ----------
        channel_names : list of str
            Channel names to use in the dialog box
        order_names : list of st
            Tone names to use in the dialog box
        specifications : list of SineSpecification
            Sine specifications used to compute the signals
        current_filter_type : int
            Choose the starting filter type, 0-DTF, 1-VK
        current_tracking_filter_cutoff : float
            The cutoff for the tracking filter
        current_tracking_filter_order : int
            The filter order for the tracking filter.
        current_filter_order : int
            The filter order for the Vold Kalman filter
        current_bandwidth : float
            The bandwidth for the Vold Kalman filter.
        current_block_size : int
            The number of samples to use when computing the Vold Kalman filter.
        current_overlap : float
            The percentage overlap of the frame size (in percent, so 15, not
            0.15).
        sample_rate : float
            The sample rate of the signals to generate.
        ramp_time : float
            The ramp time added to the signal to ramp to full level.
        acquire_size : int
            The acquisition size in number of samples.
        parent : QWidget, optional
            A parent widget to the dialog box. The default is None.

        Returns
        -------
        result : bool
            True if the dialog box was accepted, false if not.
        filter_type : int
            0 if DTF, 1 if VK.
        filter_cutoff : float
            The cutoff value for the DTF.
        tracking_filter_order : int
            The filter order for the DTF.
        filter_order : int
            The filter order for the VK filter.
        filter_bandwidth : float
            The bandwidth for the VK filter.
        filter_blocksize : int
            The number of samples in the analysis block for the VK filter.
        filter_overlap : float
            The overlap percentage (in percent not fraction) of the VK filter.

        """
        dialog = FilterExplorer(
            parent,
            channel_names,
            order_names,
            specifications,
            current_filter_type,
            current_tracking_filter_cutoff,
            current_tracking_filter_order,
            current_filter_order,
            current_bandwidth,
            current_block_size,
            current_overlap,
            sample_rate,
            ramp_time,
            acquire_size,
        )
        result = dialog.exec_() == QtWidgets.QDialog.Accepted
        filter_type = dialog.filter_type_selector.currentIndex()
        filter_order = dialog.filter_order_selector.currentIndex() + 1
        filter_bandwidth = dialog.filter_bandwidth_selector.value()
        filter_blocksize = dialog.filter_block_size_selector.value()
        filter_overlap = dialog.filter_block_overlap_selector.value()
        filter_cutoff = dialog.tracking_filter_cutoff_selector.value()
        tracking_filter_order = dialog.tracking_filter_order_selector.value()
        return (
            result,
            filter_type,
            filter_cutoff,
            tracking_filter_order,
            filter_order,
            filter_bandwidth,
            filter_blocksize,
            filter_overlap,
        )

    def __init__(
        self,
        parent,
        channel_names,
        order_names,
        specifications,
        current_filter_type,
        current_tracking_filter_cutoff,
        current_tracking_filter_order,
        current_filter_order,
        current_bandwidth,
        current_block_size,
        current_overlap,
        sample_rate,
        ramp_time,
        acquire_size,
    ):
        super().__init__(parent)
        uic.loadUi(filter_explorer_ui_path, self)

        for channel_name in channel_names:
            self.channel_selector.addItem(channel_name)

        self.order_selector.setSelectionMode(QtWidgets.QListWidget.SingleSelection)
        for order_name in order_names:
            self.order_selector.addItem(order_name)

        self.full_time_history_plotter = VaryingNumberOfLinePlot(
            self.full_time_history_plot.getPlotItem()
        )
        self.order_time_history_plotter = VaryingNumberOfLinePlot(
            self.order_time_history_plot.getPlotItem()
        )
        self.order_phase_plotter = VaryingNumberOfLinePlot(self.order_phase_plot.getPlotItem())
        self.order_amplitude_plotter = VaryingNumberOfLinePlot(
            self.order_amplitude_plot.getPlotItem()
        )

        self.filter_type_selector.setCurrentIndex(current_filter_type)
        self.tracking_filter_order_selector.setValue(current_tracking_filter_order)
        self.tracking_filter_cutoff_selector.setValue(current_tracking_filter_cutoff)
        self.filter_order_selector.setCurrentIndex(current_filter_order - 1)
        self.filter_bandwidth_selector.setValue(current_bandwidth)
        self.filter_block_overlap_selector.setValue(current_overlap)
        self.filter_block_size_selector.setValue(current_block_size)

        self.ramp_time = ramp_time
        self.sample_rate = sample_rate
        self.specifications = specifications
        self.acquire_size = acquire_size

        self.signal = None
        self.order_signals = None
        self.order_frequencies = None
        self.order_arguments = None
        self.order_amplitudes = None
        self.order_phases = None
        self.reconstructed_signal = None
        self.reconstructed_order_signals = None
        self.reconstructed_order_amplitudes = None
        self.reconstructed_order_phases = None
        self.setWindowTitle("Sine Filter Explorer")

        self.change_filter_setting_visibility()

        self.create_signals()

        self.update_plots()

        self.connect_callbacks()

    def connect_callbacks(self):
        """
        Connects callback functions to the filter widgets
        """
        self.accept_button.clicked.connect(self.accept)
        self.reject_button.clicked.connect(self.reject)
        self.order_selector.itemSelectionChanged.connect(self.update_plots)
        self.channel_selector.currentIndexChanged.connect(self.create_and_plot_signals)
        self.filter_type_selector.currentIndexChanged.connect(self.remove_filter_data_and_replot)
        self.filter_order_selector.currentIndexChanged.connect(self.remove_filter_data_and_replot)
        self.filter_bandwidth_selector.valueChanged.connect(self.remove_filter_data_and_replot)
        self.filter_block_size_selector.valueChanged.connect(self.remove_filter_data_and_replot)
        self.filter_block_overlap_selector.valueChanged.connect(self.remove_filter_data_and_replot)
        self.tracking_filter_cutoff_selector.valueChanged.connect(
            self.remove_filter_data_and_replot
        )
        self.tracking_filter_order_selector.valueChanged.connect(self.remove_filter_data_and_replot)
        self.noise_selector.valueChanged.connect(self.remove_filter_data_and_replot)
        self.compute_button.clicked.connect(self.compute_filter)
        self.filter_type_selector.currentIndexChanged.connect(self.change_filter_setting_visibility)

    @property
    def ramp_samples(self):
        """Number of ramp samples computed from sample rate and ramp time"""
        return int(self.ramp_time * self.sample_rate)

    @property
    def channel_index(self):
        """Currently selected channel index"""
        return self.channel_selector.currentIndex()

    def change_filter_setting_visibility(self):
        """Updates the visible widgets based on which filter type is selected"""
        isdtf = self.filter_type_selector.currentIndex() == 0
        for widget in [
            self.filter_order_label,
            self.filter_order_selector,
            self.filter_block_overlap_label,
            self.filter_block_overlap_selector,
            self.filter_bandwidth_label,
            self.filter_bandwidth_selector,
            self.filter_block_size_label,
            self.filter_block_size_selector,
        ]:
            widget.setVisible(not isdtf)
        for widget in [
            self.tracking_filter_cutoff_label,
            self.tracking_filter_cutoff_selector,
            self.tracking_filter_order_label,
            self.tracking_filter_order_selector,
        ]:
            widget.setVisible(isdtf)

    def create_signals(self):
        """
        Creates signals from the specification to plot
        """
        (
            self.signal,
            self.order_signals,
            self.order_frequencies,
            self.order_arguments,
            self.order_amplitudes,
            self.order_phases,
            _,
            _,
        ) = SineSpecification.create_combined_signals(
            self.specifications, self.sample_rate, self.ramp_samples, self.channel_index
        )

        self.reconstructed_signal = None
        self.reconstructed_order_signals = None
        self.reconstructed_order_amplitudes = None
        self.reconstructed_order_phases = None

    def remove_filter_data_and_replot(self):
        """Removes existing filter data and updates the plots"""
        self.reconstructed_signal = None
        self.reconstructed_order_signals = None
        self.reconstructed_order_amplitudes = None
        self.reconstructed_order_phases = None
        self.update_plots()

    def compute_filter(self):
        """Performs the filtering operations"""
        if self.filter_type_selector.currentIndex() == 0:
            block_size = self.acquire_size
            generator = [
                digital_tracking_filter_generator(
                    dt=1 / self.sample_rate,
                    cutoff_frequency_ratio=self.tracking_filter_cutoff_selector.value() / 100,
                    filter_order=self.tracking_filter_order_selector.value(),
                )
                for tone in self.order_signals
            ]
            for gen in generator:
                gen.send(None)
        else:
            block_size = self.filter_block_size_selector.value()
            generator = vold_kalman_filter_generator(
                sample_rate=self.sample_rate,
                num_orders=self.order_signals.shape[0],
                block_size=block_size,
                overlap=self.filter_block_overlap_selector.value(),
                bandwidth=self.filter_bandwidth_selector.value(),
                filter_order=self.filter_order_selector.currentIndex() + 1,
            )
            generator.send(None)

        # print(f"{self.signal.shape=}")
        start_index = 0
        reconstructed_signals = []
        reconstructed_amplitudes = []
        reconstructed_phases = []

        last_data = False
        while not last_data:
            end_index = start_index + block_size
            block = self.signal[start_index:end_index]
            block = block + self.noise_selector.value() * np.random.randn(block.size)
            block_arguments = self.order_arguments[:, start_index:end_index]
            block_frequencies = self.order_frequencies[:, start_index:end_index]
            last_data = end_index >= self.signal.size
            if self.filter_type_selector.currentIndex() == 0:
                amps = []
                phss = []
                for arg, freq, gen in zip(block_arguments, block_frequencies, generator):
                    amp, phs = gen.send((block, freq, arg))
                    amps.append(amp)
                    phss.append(phs)
                reconstructed_amplitudes.append(np.array(amps))
                reconstructed_phases.append(np.array(phss))
                reconstructed_signals.append(
                    np.array(amps) * np.cos(block_arguments + np.array(phss))
                )
            else:
                vk_signals, vk_amplitudes, vk_phases = generator.send(
                    (block, block_arguments, last_data)
                )
                if vk_signals is not None:
                    reconstructed_signals.append(vk_signals)
                    reconstructed_amplitudes.append(vk_amplitudes)
                    reconstructed_phases.append(vk_phases)
            start_index += block_size

        self.reconstructed_order_signals = reconstructed_signals
        self.reconstructed_order_amplitudes = reconstructed_amplitudes
        self.reconstructed_order_phases = reconstructed_phases
        self.reconstructed_signal = [
            np.sum(value, axis=0) for value in self.reconstructed_order_signals
        ]
        # print(f"{[sig.shape for sig in self.reconstructed_order_signals]=}")

        self.update_plots()

    def update_plots(self):
        """Updates the plots"""
        abscissa_plot_vals = []
        signal_plot_vals = []
        frequency_plot_vals = []
        amplitude_plot_vals = []
        phase_plot_vals = []
        full_signal_plot_vals = []
        abscissa = np.arange(self.signal.size) / self.sample_rate

        try:
            selected_index = [
                idx.row() for idx in self.order_selector.selectionModel().selectedIndexes()
            ][0]
        except IndexError:
            selected_index = 0
        full_signal_plot_vals.append(self.signal)
        abscissa_plot_vals.append(abscissa)
        signal_plot_vals.append(self.order_signals[selected_index])
        frequency_plot_vals.append(self.order_frequencies[selected_index])
        amplitude_plot_vals.append(self.order_amplitudes[selected_index])
        phase_plot_vals.append(self.order_phases[selected_index] * 180 / np.pi)

        if self.reconstructed_order_signals is not None:
            if self.plot_separate_frames_selector.isChecked():
                start_index = 0
                for (
                    block_order_signal,
                    block_order_amplitude,
                    block_order_phase,
                    block_signal,
                ) in zip(
                    self.reconstructed_order_signals,
                    self.reconstructed_order_amplitudes,
                    self.reconstructed_order_phases,
                    self.reconstructed_signal,
                ):
                    end_index = start_index + block_signal.shape[-1]
                    block_abscissa = abscissa[start_index:end_index]
                    block_frequency = self.order_frequencies[selected_index, start_index:end_index]
                    abscissa_plot_vals.append(block_abscissa)
                    full_signal_plot_vals.append(block_signal)
                    signal_plot_vals.append(block_order_signal[selected_index])
                    frequency_plot_vals.append(block_frequency)
                    amplitude_plot_vals.append(block_order_amplitude[selected_index])
                    phase_plot_vals.append(block_order_phase[selected_index] * 180 / np.pi)
                    start_index = end_index
            else:
                abscissa_plot_vals.append(abscissa)
                full_signal_plot_vals.append(np.concatenate(self.reconstructed_signal, axis=-1))
                signal_plot_vals.append(
                    np.concatenate(self.reconstructed_order_signals, axis=-1)[selected_index]
                )
                # print(f"{[v.shape for v in self.reconstructed_order_amplitudes]}")
                amplitude_plot_vals.append(
                    np.concatenate(self.reconstructed_order_amplitudes, axis=-1)[selected_index]
                )
                phase_plot_vals.append(
                    np.concatenate(self.reconstructed_order_phases, axis=-1)[selected_index]
                    * 180
                    / np.pi
                )
                frequency_plot_vals.append(self.order_frequencies[selected_index])

        self.full_time_history_plotter.set_data(abscissa_plot_vals, full_signal_plot_vals)
        self.order_time_history_plotter.set_data(abscissa_plot_vals, signal_plot_vals)
        self.order_amplitude_plotter.set_data(frequency_plot_vals, amplitude_plot_vals)
        self.order_phase_plotter.set_data(frequency_plot_vals, phase_plot_vals)

    def create_and_plot_signals(self):
        """Creates signals then plots the signals"""
        self.create_signals()
        self.update_plots()


class PlotSineWindow(QtWidgets.QDialog):
    """Class defining a subwindow that displays specific channel information"""

    def __init__(self, parent, ui, tone_index, channel_index):
        """
        Creates a window showing amplitude and phase information.

        Parameters
        ----------
        parent : QWidget
            Parent of the window.
        ui : SineUI
            The User Interface of the Sine Controller
        tone_index : int
            Index specifying the tone to plot
        channel_index : int
            Index specifying the channel to plot
        """
        super(QtWidgets.QDialog, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() & Qt.Tool)
        self.tone_index = tone_index
        self.channel_index = channel_index
        spec_frequency = ui.specification_frequencies[tone_index]
        spec_amplitude = ui.specification_amplitudes[tone_index, channel_index]
        spec_phase = wrap(ui.specification_phases[tone_index, channel_index])
        spec = ui.environment_parameters.specifications[tone_index]
        warn_freq = np.repeat(spec.breakpoint_table["frequency"], 2)
        warn_low = spec.breakpoint_table["warning"][:, 0, :, channel_index].flatten()
        warn_high = spec.breakpoint_table["warning"][:, 1, :, channel_index].flatten()
        abort_low = spec.breakpoint_table["abort"][:, 0, :, channel_index].flatten()
        abort_high = spec.breakpoint_table["abort"][:, 1, :, channel_index].flatten()
        tone_name = spec.name
        channel_name = ui.initialized_control_names[channel_index]
        # Now plot the data
        layout = QtWidgets.QVBoxLayout()
        amp_plotwidget = pqtg.PlotWidget()
        layout.addWidget(amp_plotwidget)
        phs_plotwidget = pqtg.PlotWidget()
        layout.addWidget(phs_plotwidget)
        self.setLayout(layout)
        amp_plot_item = amp_plotwidget.getPlotItem()
        phs_plot_item = phs_plotwidget.getPlotItem()
        for plot_item in [amp_plot_item, phs_plot_item]:
            plot_item.showGrid(True, True, 0.25)
            plot_item.enableAutoRange()
            plot_item.getViewBox().enableAutoRange(enable=True)
        amp_plot_item.plot(spec_frequency, spec_amplitude, pen={"color": "b", "width": 1})
        phs_plot_item.plot(spec_frequency, spec_phase, pen={"color": "b", "width": 1})
        amp_plot_item.plot(
            warn_freq,
            warn_low,
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
        )
        amp_plot_item.plot(
            warn_freq,
            warn_high,
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
        )
        amp_plot_item.plot(
            warn_freq,
            abort_low,
            pen={"color": (153, 0, 0), "width": 1, "style": Qt.DashLine},
        )
        amp_plot_item.plot(
            warn_freq,
            abort_high,
            pen={"color": (153, 0, 0), "width": 1, "style": Qt.DashLine},
        )
        if ui.achieved_excitation_frequencies is not None:
            achieved_frequency = np.concatenate(
                [fh[tone_index] for fh in ui.achieved_excitation_frequencies]
            )
            achieved_amplitude = np.concatenate(
                [ah[tone_index, channel_index] for ah in ui.achieved_response_amplitudes]
            )
            achieved_phase = np.concatenate(
                [ph[tone_index, channel_index] for ph in ui.achieved_response_phases]
            )
        else:
            achieved_frequency = np.array([0, 1])
            achieved_amplitude = np.nan * np.ones(2)
            achieved_phase = np.nan * np.ones(2)
        self.amp_curve = amp_plot_item.plot(
            achieved_frequency, achieved_amplitude, pen={"color": "r", "width": 1}
        )
        self.phs_curve = phs_plot_item.plot(
            achieved_frequency, achieved_phase, pen={"color": "r", "width": 1}
        )
        self.setWindowTitle(f"{tone_name} {channel_name}")
        self.ui = ui
        self.show()

    def update_plot(self):
        """Updates the plots with new data"""
        if self.ui.achieved_excitation_frequencies is not None:
            achieved_frequency = np.concatenate(
                [fh[self.tone_index] for fh in self.ui.achieved_excitation_frequencies]
            )
            achieved_amplitude = np.concatenate(
                [
                    ah[self.tone_index, self.channel_index]
                    for ah in self.ui.achieved_response_amplitudes
                ]
            )
            achieved_phase = np.concatenate(
                [ph[self.tone_index, self.channel_index] for ph in self.ui.achieved_response_phases]
            )
        else:
            achieved_frequency = np.array([0, 1])
            achieved_amplitude = np.nan * np.ones(2)
            achieved_phase = np.nan * np.ones(2)
        self.amp_curve.setData(achieved_frequency, achieved_amplitude)
        self.phs_curve.setData(achieved_frequency, achieved_phase)
