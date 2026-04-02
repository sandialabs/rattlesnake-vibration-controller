# -*- coding: utf-8 -*-
"""
User interface-specific utilities that might be used in multiple environments

Rattlesnake Vibration Control Software
Copyright (C) 2021  National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import socket
import sys
import numpy as np
import pyqtgraph
import requests
from qtpy import QtCore, QtGui, QtWidgets, uic
from qtpy.QtCore import Qt, QTimer
from scipy.interpolate import interp1d
from scipy.io import loadmat
from enum import Enum

from rattlesnake.utilities import (
    DIRECTORY,
    coherence,
    load_csv_matrix,
    save_csv_matrix,
    trac,
    DataAcquisitionParameters,
)


# region Global
class UICommands(Enum):
    ERROR = -1
    ENABLE = 0
    DISABLE = 1
    MONITOR = 2
    ENABLE_TAB = 3
    DISABLE_TAB = 4
    SET_ATTR = 5
    STOP = 6
    HARDWARE_STARTED = 7
    HARDWARE_ENDED = 8
    SET_ENVIRONMENT_INSTRUCTIONS = 9
    COMPLETED_SYSTEM_ID = 10
    ENVIRONMENT_STARTED = 11
    ENVIRONMENT_ENDED = 12

    @property
    def label(self):
        """Used by UI as names for"""
        return self.name.replace("_", " ").title()


def error_message_qt(title, message):
    """Helper class to create an error dialog.

    Parameters
    ----------
    title : str :
        Title of the window that the error message will appear in.
    message : str :
        Error message that will be displayed.

    """
    QtWidgets.QMessageBox.critical(None, title, message)


colororder = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def multiline_plotter(
    x,
    y,
    widget=None,
    curve_list=None,
    names=None,
    other_pen_options=None,
    legend=False,
    downsample=None,
    clip_to_view=False,
):
    """Helper function for PyQtGraph to deal with plots with multiple curves

    Parameters
    ----------
    x : np.ndarray
        Abscissa for the data that will be plotted, 1D array with shape n_samples
    y : np.ndarray
        Ordinates for the data that will be plotted.  2D array with shape
        n_curves x n_samples
    widget :
        The plot widget on which the curves will be drawn. (Default value = None)
    curve_list :
        Alternatively to specifying the widget, a curve list can be specified
        directly.  (Default value = None)
    names :
        Names of the curves that will appear in the legend. (Default value = None)
    other_pen_options : dict
        Additional options besides color that will be applied to the curves.
        (Default value = {'width':1})
    legend :
        Whether or not to draw a legend (Default value = False)

    Returns
    -------

    """
    if other_pen_options is None:
        other_pen_options = {"width": 1}
    if downsample is None:
        downsample = {"ds": 1, "auto": False, "mode": "peak"}
    if widget is not None:
        plot_item = widget.getPlotItem()
        plot_item.setDownsampling(**downsample)
        plot_item.setClipToView(clip_to_view)
        if legend:
            plot_item.addLegend(colCount=len(y) // 10)
        handles = []
        for i, this_y in enumerate(y):
            pen = {"color": colororder[i % len(colororder)]}
            pen.update(other_pen_options)
            handles.append(
                plot_item.plot(
                    x, this_y, pen=pen, name=None if names is None else names[i]
                )
            )
        return handles
    elif curve_list is not None:
        for this_y, curve in zip(y, curve_list):
            curve.setData(x, y)
        return curve_list
    else:
        raise ValueError("Either Widget or list of curves must be specified")


def blended_scatter_plot(xy, widget=None, curve_list=None, names=None, symbol="o"):
    """Creates a scatter plot with the specified symbols"""
    if widget is not None:
        plot_item = widget.getPlotItem()
        handles = []
        for index, (x, y) in enumerate(xy):
            c = (1 - (index + 1) / len(xy)) * 255
            handles.append(
                plot_item.plot(
                    [x],
                    [y],
                    symbolBrush=(c, c, c),
                    name=None if names is None else names[index],
                    symbol=symbol,
                )
            )
        return handles
    elif curve_list is not None:
        for (x, y), curve in zip(xy, curve_list):
            curve.setData([x], [y])
        return curve_list
    else:
        raise ValueError("Either Widget or list of curves must be specified")


class PlotWindow(QtWidgets.QDialog):
    """Class defining a subwindow that displays specific channel information"""

    WARNING_COLOR = (225, 190, 0)
    WARNING_LINEWIDTH = 0.3
    WARNING_LINESTYLE = Qt.SolidLine
    ABORT_COLOR = (200, 0, 0)
    ABORT_LINEWIDTH = 0.3
    ABORT_LINESTYLE = Qt.SolidLine

    def __init__(
        self,
        parent,
        row,
        column,
        datatype,
        specification,
        row_name,
        column_name,
        datatype_name,
        warning_matrix=None,
        abort_matrix=None,
    ):
        """
        Creates a window showing CPSD matrix information for a single channel.

        Parameters
        ----------
        parent : QWidget
            Parent of the window.
        row : int
            Row of the CPSD matrix to plot.
        column : int
            Column of the CPSD matrix to plot.
        datatype : int
            Type of data to plot: 0 - Magnitude, 1 - Coherence, 2 - Phase, 3 -
            Real, 4 - Imaginary.
        specification : np.ndarray
            The specification against which data will be compared.
        row_name : str
            Channel name for the row.
        column_name : str
            Channel name for the column.
        datatype_name : str
            Name for the datatype.


        """
        super(QtWidgets.QDialog, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() & Qt.Tool)
        self.row = row
        self.column = column
        self.datatype = datatype
        self.frequencies = specification[0]
        self.spec_data = self.reduce_matrix(specification[1])
        self.data = np.zeros(self.spec_data.shape)
        # Now plot the data
        layout = QtWidgets.QVBoxLayout()
        plotwidget = pyqtgraph.PlotWidget()
        layout.addWidget(plotwidget)
        self.setLayout(layout)
        plot_item = plotwidget.getPlotItem()
        plot_item.showGrid(True, True, 0.25)
        plot_item.enableAutoRange()
        plot_item.getViewBox().enableAutoRange(enable=True)
        if self.datatype == 0:
            plot_item.setLogMode(False, True)
        plot_item.plot(self.frequencies, self.spec_data, pen={"color": "b", "width": 1})
        if warning_matrix is not None:
            plot_item.plot(
                self.frequencies,
                warning_matrix[0, :, row],
                pen={
                    "color": PlotWindow.WARNING_COLOR,
                    "width": PlotWindow.WARNING_LINEWIDTH,
                    "style": PlotWindow.WARNING_LINESTYLE,
                },
            )
            plot_item.plot(
                self.frequencies,
                warning_matrix[1, :, row],
                pen={
                    "color": PlotWindow.WARNING_COLOR,
                    "width": PlotWindow.WARNING_LINEWIDTH,
                    "style": PlotWindow.WARNING_LINESTYLE,
                },
            )
        if abort_matrix is not None:
            plot_item.plot(
                self.frequencies,
                abort_matrix[0, :, row],
                pen={
                    "color": PlotWindow.ABORT_COLOR,
                    "width": PlotWindow.ABORT_LINEWIDTH,
                    "style": PlotWindow.ABORT_LINESTYLE,
                },
            )
            plot_item.plot(
                self.frequencies,
                abort_matrix[1, :, row],
                pen={
                    "color": PlotWindow.ABORT_COLOR,
                    "width": PlotWindow.ABORT_LINEWIDTH,
                    "style": PlotWindow.ABORT_LINESTYLE,
                },
            )
        self.curve = plot_item.plot(
            self.frequencies, self.data, pen={"color": "r", "width": 1}
        )
        self.setWindowTitle(f"{datatype_name} {row_name} / {column_name}")
        self.show()

    def reduce_matrix(self, matrix):
        """Collects the data specific to the row and column and datatype

        Parameters
        ----------
        matrix : np.ndarray
            The 3D CPSD data that will be reduced

        Returns
        -------
        plot_data : np.ndarray
            The data that will be plotted

        """
        if self.datatype == 0:  # Magnitude
            return np.abs(matrix[..., self.row, self.column])
        elif self.datatype == 1:  # Coherence
            return coherence(matrix, (self.row, self.column))
        elif self.datatype == 2:  # Phase
            return np.angle(matrix[..., self.row, self.column])
        elif self.datatype == 3:  # Real
            return np.real(matrix[..., self.row, self.column])
        elif self.datatype == 4:  # Imag
            return np.imag(matrix[..., self.row, self.column])
        else:
            raise ValueError(f"{self.datatype} is not a valid datatype!")

    def update_plot(self, cpsd_matrix):
        """Updates the plot with the given CPSD matrix data

        Parameters
        ----------
        cpsd_matrix : np.ndarray
            3D CPSD matrix that will be reduced for plotting

        """
        self.curve.setData(self.frequencies, self.reduce_matrix(cpsd_matrix))


class PlotTimeWindow(QtWidgets.QDialog):
    """Class defining a subwindow that displays specific channel information"""

    def __init__(self, parent, index, specification, sample_rate, index_name):
        """
        Creates a window showing time history information for a single channel.

        Parameters
        ----------
        parent : QWidget
            Parent of the window.
        index : int
            Row of the time history matrix to plot
        specification : np.ndarray
            The specification against which data will be compared.
        sample_rate : int
            The sample rate of the time signal
        index_name : str
            Channel name for the row.
        """
        super(QtWidgets.QDialog, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() & Qt.Tool)
        self.index = index
        self.times = np.arange(specification.shape[-1]) / sample_rate
        self.spec_data = self.reduce_matrix(specification)
        self.data = np.zeros(self.spec_data.shape)
        # Now plot the data
        layout = QtWidgets.QVBoxLayout()
        plotwidget = pyqtgraph.PlotWidget()
        layout.addWidget(plotwidget)
        self.setLayout(layout)
        plot_item = plotwidget.getPlotItem()
        plot_item.showGrid(True, True, 0.25)
        plot_item.enableAutoRange()
        plot_item.getViewBox().enableAutoRange(enable=True)
        plot_item.plot(self.times, self.spec_data, pen={"color": "b", "width": 1})
        plot_item.setLabel("left", "TRAC: 0.0")
        self.plot_item = plot_item
        self.curve = plot_item.plot(
            self.times, self.data, pen={"color": "r", "width": 1}
        )
        self.setWindowTitle(f"{index_name}")
        self.show()

    def reduce_matrix(self, matrix):
        """Collects the data specific to the row and column and datatype

        Parameters
        ----------
        matrix : np.ndarray
            The 3D CPSD data that will be reduced

        Returns
        -------
        plot_data : np.ndarray
            The data that will be plotted

        """
        return matrix[self.index]

    def update_plot(self, data):
        """Updates the plot with the given CPSD matrix data

        Parameters
        ----------
        cpsd_matrix : np.ndarray
            3D CPSD matrix that will be reduced for plotting

        """
        data = self.reduce_matrix(data)
        self.curve.setData(self.times, data)
        self.plot_item.setLabel(
            "left", f"TRAC: {trac(data, self.spec_data).squeeze():0.2f}"
        )


class TransformationMatrixWindow(QtWidgets.QDialog):
    """Dialog box for specifying transformation matrices"""

    def __init__(
        self,
        parent,
        current_response_transformation_matrix,
        num_responses,
        current_output_transformation_matrix,
        num_outputs,
    ):
        """
        Creates a dialog box for specifying response and output transformations

        Parameters
        ----------
        parent : QWidget
            Parent to the dialog box.
        current_response_transformation_matrix : np.ndarray
            The current value of the transformation matrix that will be used to
            populate the entries in the table.
        num_responses : int
            Number of physical responses in the transformation.
        current_output_transformation_matrix : np.ndarray
            The current value of the transformation matrix that will be used to
            populate the entries in the table.
        num_outputs : int
            Number of physical outputs in the transformation.

        """
        super().__init__(parent)
        uic.loadUi(transformation_matrices_ui_path, self)
        self.setWindowTitle("Transformation Matrix Definition")

        self.response_transformation_matrix.setColumnCount(num_responses)
        self.output_transformation_matrix.setColumnCount(num_outputs)

        if current_response_transformation_matrix is None:
            self.set_response_transformation_identity()
        else:
            self.response_transformation_matrix.setRowCount(
                current_response_transformation_matrix.shape[0]
            )
            for row_idx, row in enumerate(current_response_transformation_matrix):
                for col_idx, col in enumerate(row):
                    try:
                        self.response_transformation_matrix.item(
                            row_idx, col_idx
                        ).setText(str(col))
                    except AttributeError:
                        item = QtWidgets.QTableWidgetItem(str(col))
                        self.response_transformation_matrix.setItem(
                            row_idx, col_idx, item
                        )
        if current_output_transformation_matrix is None:
            self.set_output_transformation_identity()
        else:
            self.output_transformation_matrix.setRowCount(
                current_output_transformation_matrix.shape[0]
            )
            for row_idx, row in enumerate(current_output_transformation_matrix):
                for col_idx, col in enumerate(row):
                    try:
                        self.output_transformation_matrix.item(
                            row_idx, col_idx
                        ).setText(str(col))
                    except AttributeError:
                        item = QtWidgets.QTableWidgetItem(str(col))
                        self.output_transformation_matrix.setItem(
                            row_idx, col_idx, item
                        )

        # Callbacks
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        self.response_transformation_add_row_button.clicked.connect(
            self.response_transformation_add_row
        )
        self.response_transformation_remove_row_button.clicked.connect(
            self.response_transformation_remove_row
        )
        self.response_transformation_save_matrix_button.clicked.connect(
            self.save_response_transformation_matrix
        )
        self.response_transformation_load_matrix_button.clicked.connect(
            self.load_response_transformation_matrix
        )
        self.response_transformation_identity_button.clicked.connect(
            self.set_response_transformation_identity
        )
        self.response_transformation_6dof_kinematic_button.clicked.connect(
            self.set_response_transformation_6dof
        )
        self.response_transformation_reversed_6dof_kinematic_button.clicked.connect(
            self.set_response_transformation_6dof_reversed
        )

        self.output_transformation_add_row_button.clicked.connect(
            self.output_transformation_add_row
        )
        self.output_transformation_remove_row_button.clicked.connect(
            self.output_transformation_remove_row
        )
        self.output_transformation_save_matrix_button.clicked.connect(
            self.save_output_transformation_matrix
        )
        self.output_transformation_load_matrix_button.clicked.connect(
            self.load_output_transformation_matrix
        )
        self.output_transformation_identity_button.clicked.connect(
            self.set_output_transformation_identity
        )
        self.output_transformation_6dof_kinematic_button.clicked.connect(
            self.set_output_transformation_6dof
        )
        self.output_transformation_reversed_6dof_kinematic_button.clicked.connect(
            self.set_output_transformation_6dof_reversed
        )

    @staticmethod
    def define_transformation_matrices(
        current_response_transformation_matrix,
        num_responses,
        current_output_transformation_matrix,
        num_outputs,
        parent=None,
    ):
        """
        Shows the dialog and returns the transformation matrices

        Parameters
        ----------
        current_response_transformation_matrix : np.ndarray
            The current value of the transformation matrix that will be used to
            populate the entries in the table.
        num_responses : int
            Number of physical responses in the transformation.
        current_output_transformation_matrix : np.ndarray
            The current value of the transformation matrix that will be used to
            populate the entries in the table.
        num_outputs : int
            Number of physical outputs in the transformation.
        parent : QWidget
            Parent to the dialog box. (Default value = None)

        Returns
        -------
        response_transformation : np.ndarray
            Response transformation (or None if Identity)
        output_transformation : np.ndarray
            Output transformation (or None if Identity)
        result : bool
            True if dialog was accepted, false if cancelled.
        """
        dialog = TransformationMatrixWindow(
            parent,
            current_response_transformation_matrix,
            num_responses,
            current_output_transformation_matrix,
            num_outputs,
        )
        result = dialog.exec_() == QtWidgets.QDialog.Accepted
        response_transformation = np.array(
            [
                [float(val) for val in row]
                for row in get_table_strings(dialog.response_transformation_matrix)
            ]
        )
        if all(
            val == response_transformation.shape[0]
            for val in response_transformation.shape
        ) and np.allclose(
            response_transformation, np.eye(response_transformation.shape[0])
        ):
            response_transformation = None
        output_transformation = np.array(
            [
                [float(val) for val in row]
                for row in get_table_strings(dialog.output_transformation_matrix)
            ]
        )
        if all(
            val == output_transformation.shape[0] for val in output_transformation.shape
        ) and np.allclose(
            output_transformation, np.eye(output_transformation.shape[0])
        ):
            output_transformation = None
        return (response_transformation, output_transformation, result)

    def response_transformation_add_row(self):
        """Adds a row to the response transformation"""
        num_rows = self.response_transformation_matrix.rowCount()
        self.response_transformation_matrix.insertRow(num_rows)
        for col_idx in range(self.response_transformation_matrix.columnCount()):
            item = QtWidgets.QTableWidgetItem("0.0")
            self.response_transformation_matrix.setItem(num_rows, col_idx, item)

    def response_transformation_remove_row(self):
        """Removes a row from the response transformation"""
        num_rows = self.response_transformation_matrix.rowCount()
        self.response_transformation_matrix.removeRow(num_rows - 1)

    def save_response_transformation_matrix(self):
        """Saves the response transformation matrix to a csv file"""
        string_array = self.get_table_strings(self.response_transformation_matrix)
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Response Transformation",
            filter="Comma-separated Values (*.csv)",
        )
        if filename == "":
            return
        save_csv_matrix(string_array, filename)

    def load_response_transformation_matrix(self):
        """Loads the response transformation from a csv file"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Response Transformation",
            filter="Comma-separated values (*.csv *.txt);;"
            "Numpy Files (*.npy *.npz);;Matlab Files (*.mat)",
        )
        if filename == "":
            return
        _, extension = os.path.splitext(filename)
        string_array = None
        if extension.lower() == ".npy":
            string_array = np.load(filename).astype("U")
        elif extension.lower() == ".npz":
            data = np.load(filename)
            for key, array in data.items():
                string_array = array.astype("U")
                break
        elif extension.lower() == ".mat":
            data = loadmat(filename)
            for key, array in data.items():
                if "__" in key:
                    continue
                string_array = array.astype("U")
                break
        else:
            string_array = load_csv_matrix(filename)
        if string_array is None:
            return
        # Set the number of rows
        self.response_transformation_matrix.setRowCount(len(string_array))
        num_rows = self.response_transformation_matrix.rowCount()
        num_cols = self.response_transformation_matrix.columnCount()
        for row_idx, row in enumerate(string_array):
            if row_idx == num_rows:
                break
            for col_idx, value in enumerate(row):
                if col_idx == num_cols:
                    break
                try:
                    self.response_transformation_matrix.item(row_idx, col_idx).setText(
                        value
                    )
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(value)
                    self.response_transformation_matrix.setItem(row_idx, col_idx, item)

    def set_response_transformation_identity(self):
        """Sets the response transformation to identity matrix (no transform)"""
        num_columns = self.response_transformation_matrix.columnCount()
        self.response_transformation_matrix.setRowCount(num_columns)
        for row_idx in range(num_columns):
            for col_idx in range(num_columns):
                if row_idx == col_idx:
                    value = 1.0
                else:
                    value = 0.0
                try:
                    self.response_transformation_matrix.item(row_idx, col_idx).setText(
                        str(value)
                    )
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.response_transformation_matrix.setItem(row_idx, col_idx, item)

    def set_response_transformation_6dof(self):
        """Sets the response transformation matrix to the 6DoF table"""
        num_columns = self.response_transformation_matrix.columnCount()
        if num_columns != 12:
            error_message_qt(
                "Invalid Number of Control Channels.",
                "Invalid Number of Control Channels.  "
                "6DoF Transform assumes 12 control accelerometer channels.",
            )
            return
        self.response_transformation_matrix.setRowCount(6)
        matrix = [
            [0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0],
            [0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25],
            [0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, -0.25, 0.0, 0.0, -0.25],
            [0.0, 0.0, -0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, -0.25],
            [
                -0.125,
                0.125,
                0.0,
                -0.125,
                -0.125,
                0.0,
                0.125,
                -0.125,
                0.0,
                0.125,
                0.125,
                0.0,
            ],
        ]
        for row_idx, row in enumerate(matrix):
            for col_idx, value in enumerate(row):
                try:
                    self.response_transformation_matrix.item(row_idx, col_idx).setText(
                        str(value)
                    )
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.response_transformation_matrix.setItem(row_idx, col_idx, item)

    def set_response_transformation_6dof_reversed(self):
        """Sets the response transformation matrix to the 6DoF table"""
        num_columns = self.response_transformation_matrix.columnCount()
        if num_columns != 12:
            error_message_qt(
                "Invalid Number of Control Channels.",
                "Invalid Number of Control Channels.  "
                "6DoF Transform assumes 12 control accelerometer channels.",
            )
            return
        self.response_transformation_matrix.setRowCount(6)
        matrix = [
            [0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0],
            [0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25],
            [0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, -0.25, 0.0, 0.0, -0.25],
            [0.0, 0.0, -0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, -0.25],
            [
                -0.125,
                0.125,
                0.0,
                -0.125,
                -0.125,
                0.0,
                0.125,
                -0.125,
                0.0,
                0.125,
                0.125,
                0.0,
            ],
        ]
        for row_idx, row in enumerate(matrix):
            for col_idx, value in enumerate(row):
                try:
                    self.response_transformation_matrix.item(row_idx, col_idx).setText(
                        str(value)
                    )
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.response_transformation_matrix.setItem(row_idx, col_idx, item)

    def output_transformation_add_row(self):
        """Adds a row to the output transformation"""
        num_rows = self.output_transformation_matrix.rowCount()
        self.output_transformation_matrix.insertRow(num_rows)
        for col_idx in range(self.output_transformation_matrix.columnCount()):
            item = QtWidgets.QTableWidgetItem("0.0")
            self.output_transformation_matrix.setItem(num_rows, col_idx, item)

    def output_transformation_remove_row(self):
        """Removes a row from the output tranformation"""
        num_rows = self.output_transformation_matrix.rowCount()
        self.output_transformation_matrix.removeRow(num_rows - 1)

    def save_output_transformation_matrix(self):
        """Saves output transformation matrix to a CSV file"""
        string_array = self.get_table_strings(self.output_transformation_matrix)
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Output Transformation", filter="Comma-separated Values (*.csv)"
        )
        if filename == "":
            return
        save_csv_matrix(string_array, filename)

    def load_output_transformation_matrix(self):
        """Loads the output transformation from a CSV file"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Output Transformation",
            filter="Comma-separated values (*.csv *.txt);;"
            "Numpy Files (*.npy *.npz);;Matlab Files (*.mat)",
        )
        if filename == "":
            return
        _, extension = os.path.splitext(filename)
        string_array = None
        if extension.lower() == ".npy":
            string_array = np.load(filename).astype("U")
        elif extension.lower() == ".npz":
            data = np.load(filename)
            for key, array in data.items():
                string_array = array.astype("U")
                break
        elif extension.lower() == ".mat":
            data = loadmat(filename)
            for key, array in data.items():
                if "__" in key:
                    continue
                string_array = array.astype("U")
                break
        else:
            string_array = load_csv_matrix(filename)
        if string_array is None:
            return
        # Set the number of rows
        self.output_transformation_matrix.setRowCount(len(string_array))
        num_rows = self.output_transformation_matrix.rowCount()
        num_cols = self.output_transformation_matrix.columnCount()
        for row_idx, row in enumerate(string_array):
            if row_idx == num_rows:
                break
            for col_idx, value in enumerate(row):
                if col_idx == num_cols:
                    break
                try:
                    self.output_transformation_matrix.item(row_idx, col_idx).setText(
                        value
                    )
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(value)
                    self.output_transformation_matrix.setItem(row_idx, col_idx, item)

    def set_output_transformation_identity(self):
        """Sets the output transformation to identity (no transform)"""
        num_columns = self.output_transformation_matrix.columnCount()
        self.output_transformation_matrix.setRowCount(num_columns)
        for row_idx in range(num_columns):
            for col_idx in range(num_columns):
                if row_idx == col_idx:
                    value = 1.0
                else:
                    value = 0.0
                try:
                    self.output_transformation_matrix.item(row_idx, col_idx).setText(
                        str(value)
                    )
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.output_transformation_matrix.setItem(row_idx, col_idx, item)

    def set_output_transformation_6dof(self):
        """Sets the output transformation matrix to the 6DoF table"""
        num_columns = self.output_transformation_matrix.columnCount()
        if num_columns != 12:
            error_message_qt(
                "Invalid Number of Output Signals.",
                "Invalid Number of Output Signals.  6DoF Transform assumes 12 drive channels.",
            )
            return
        self.output_transformation_matrix.setRowCount(6)
        matrix = [
            [0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0],
            [0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25],
            [0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, -0.25, 0.0, 0.0, -0.25],
            [0.0, 0.0, -0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, -0.25],
            [
                -0.125,
                0.125,
                0.0,
                -0.125,
                -0.125,
                0.0,
                0.125,
                -0.125,
                0.0,
                0.125,
                0.125,
                0.0,
            ],
        ]
        for row_idx, row in enumerate(matrix):
            for col_idx, value in enumerate(row):
                try:
                    self.output_transformation_matrix.item(row_idx, col_idx).setText(
                        str(value)
                    )
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.output_transformation_matrix.setItem(row_idx, col_idx, item)

    def set_output_transformation_6dof_reversed(self):
        """Sets the output transformation matrix to the 6DoF table"""
        num_columns = self.output_transformation_matrix.columnCount()
        if num_columns != 12:
            error_message_qt(
                "Invalid Number of Output Signals.",
                "Invalid Number of Output Signals.  6DoF Transform assumes 12 drive channels.",
            )
            return
        self.output_transformation_matrix.setRowCount(6)
        matrix = [
            [0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0],
            [0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25],
            [0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, -0.25, 0.0, 0.0, -0.25],
            [0.0, 0.0, -0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, -0.25],
            [
                -0.125,
                0.125,
                0.0,
                -0.125,
                -0.125,
                0.0,
                0.125,
                -0.125,
                0.0,
                0.125,
                0.125,
                0.0,
            ],
        ]
        for row_idx, row in enumerate(matrix):
            for col_idx, value in enumerate(row):
                try:
                    self.output_transformation_matrix.item(row_idx, col_idx).setText(
                        str(value)
                    )
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.output_transformation_matrix.setItem(row_idx, col_idx, item)


def get_table_strings(tablewidget: QtWidgets.QTableWidget):
    """Collect a table of strings from a QTableWidget

    Parameters
    ----------
    tablewidget : QtWidgets.QTableWidget
        A table widget to pull the strings from

    Returns
    -------
    string_array : list[list[str]]
        A nested list of strings from the table items

    """
    string_array = []
    for row_idx in range(tablewidget.rowCount()):
        string_array.append([])
        for col_idx in range(tablewidget.columnCount()):
            value = tablewidget.item(row_idx, col_idx).text()
            string_array[-1].append(value)
    return string_array


class ChannelMonitor(QtWidgets.QDialog):
    """Class defining a subwindow that displays specific channel information"""

    def __init__(self, parent, daq_settings: DataAcquisitionParameters):
        """
        Creates a window showing CPSD matrix information for a single channel.

        Parameters
        ----------
        parent : QWidget
            Parent of the window.
        """
        super(QtWidgets.QDialog, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() & Qt.Tool)
        self.channels = daq_settings.channel_list
        # Set up the window
        self.graphics_layout_widget = pyqtgraph.GraphicsLayoutWidget(self)
        self.push_button = QtWidgets.QPushButton("Clear Alerts", self)
        self.channels_per_row_label = QtWidgets.QLabel("Channels per Row: ", self)
        self.channels_per_row_selector = QtWidgets.QSpinBox(self)
        self.channels_per_row_selector.setMinimum(2)
        self.channels_per_row_selector.setMaximum(100)
        self.channels_per_row_selector.setValue(20)
        self.channels_per_row_selector.setKeyboardTracking(False)
        layout = QtWidgets.QVBoxLayout()
        control_layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.graphics_layout_widget)
        control_layout.addWidget(self.channels_per_row_label)
        control_layout.addWidget(self.channels_per_row_selector)
        control_layout.addStretch()
        control_layout.addWidget(self.push_button)
        layout.addLayout(control_layout)
        self.setLayout(layout)
        # Set up defaults for the channel ranges
        self.channel_ranges = None
        self.channel_warning_limits = None
        self.channel_abort_limits = None
        self.background_bars = None
        self.history_bars = None
        self.level_bars = None
        self.history_last_update = None
        self.history_hold_frames = int(
            np.ceil(10 * daq_settings.sample_rate / daq_settings.samples_per_read)
        )
        self.aborted_channels = None
        # Set up defaults for the plot
        self.plots = None
        self.bar_channel_indices = None
        self.pen = pyqtgraph.mkPen(color=(0, 0, 0, 255), width=1)
        self.background_brush = pyqtgraph.mkBrush((255, 255, 255))
        self.history_brush = pyqtgraph.mkBrush((124, 124, 255))
        self.current_brush = pyqtgraph.mkBrush((34, 139, 34))
        self.limit_brush = pyqtgraph.mkBrush((145, 197, 17))
        self.abort_brush = pyqtgraph.mkBrush((145, 70, 17))
        self.limit_background_brush = pyqtgraph.mkBrush(
            (
                255,
                255,
                0,
            )
        )
        self.abort_background_brush = pyqtgraph.mkBrush((255, 0, 0))
        self.limit_history_brush = pyqtgraph.mkBrush((190, 190, 128))
        self.abort_history_brush = pyqtgraph.mkBrush((190, 62, 128))
        # Connect everything and do final builds
        self.connect_callbacks()
        self.build_plot()
        self.setWindowTitle("Channel Monitor")
        self.resize(400, 300)
        self.show()

    def connect_callbacks(self):
        """Connects callback functions to the respective widgets"""
        self.channels_per_row_selector.valueChanged.connect(self.build_plot)
        self.push_button.clicked.connect(self.clear_alerts)

    def update_channel_list(self, daq_settings):
        """Updates the channel list in the test"""
        self.channels = daq_settings.channel_list
        self.history_hold_frames = int(
            np.ceil(10 * daq_settings.sample_rate / daq_settings.samples_per_read)
        )
        self.build_plot()

    def clear_alerts(self):
        """Clears any alerts that have been triggered by high values"""
        self.aborted_channels = [False for val in self.aborted_channels]
        for current_bar in self.level_bars:
            current_bar.setOpts(brushes=[self.current_brush])
        for history_bar in self.history_bars:
            history_bar.setOpts(brushes=[self.history_brush])
        for background_bar in self.background_bars:
            background_bar.setOpts(brushes=[self.background_brush])

    def build_plot(self):
        """Builds the channel monitor window and plots"""
        # TODO Need to get the values from the bars before deleting them so we
        # can maintain the levels from before the value was changed
        self.graphics_layout_widget.clear()
        num_channels = len(self.channels)
        num_bars = int(np.ceil(num_channels / self.channels_per_row_selector.value()))
        # Compute number of channels per bar
        channels_per_bar = [0 for i in range(num_bars)]
        for i in range(num_channels):
            channels_per_bar[i % num_bars] += 1

        # print('Channels per Bar {:}'.format(channels_per_bar))
        # Now let's actually make the plots
        self.plots = [
            self.graphics_layout_widget.addPlot(i, 0) for i in range(num_bars)
        ]

        # Now parse the channel ranges
        self.channel_ranges = []
        self.channel_warning_limits = []
        self.channel_abort_limits = []
        for channel in self.channels:
            try:
                max_abs_volt = np.min(
                    np.abs([float(channel.maximum_value), float(channel.minimum_value)])
                )
            except (ValueError, TypeError):
                max_abs_volt = 10  # Assume 10 V range on DAQ
            try:
                sensitivity = float(channel.sensitivity) / 1000  # mV -> V
            except (ValueError, TypeError):
                sensitivity = 0.01  # Assume 10 mV/EU
            max_abs_eu = max_abs_volt / sensitivity
            try:
                warning_limit = float(channel.warning_level)
            except (ValueError, TypeError):
                warning_limit = max_abs_eu * 0.9  # Put out warning at 90% the max range
            try:
                abort_limit = float(channel.abort_level)
            except (ValueError, TypeError):
                abort_limit = max_abs_eu  # Never abort on this channel if not specified
            self.channel_ranges.append(max_abs_eu)
            self.channel_warning_limits.append(warning_limit)
            self.channel_abort_limits.append(abort_limit)
        self.channel_ranges = np.array(self.channel_ranges)
        self.channel_warning_limits = np.array(self.channel_warning_limits)
        self.channel_abort_limits = np.array(self.channel_abort_limits)
        # Display abort limit as range rather than channel if it is lower
        abort_lower = self.channel_ranges > self.channel_abort_limits
        self.channel_ranges[abort_lower] = self.channel_abort_limits[abort_lower]

        # Now build the plots
        self.bar_channel_indices = []
        for i, num_channels in enumerate(channels_per_bar):
            try:
                next_starting_index = self.bar_channel_indices[-1][-1] + 1
            except IndexError:
                next_starting_index = 0
            self.bar_channel_indices.append(
                next_starting_index + np.arange(num_channels)
            )
        # print(self.bar_channel_indices)
        self.background_bars = []
        self.history_bars = []
        self.level_bars = []
        self.history_last_update = []
        self.aborted_channels = []
        for indices, plot in zip(self.bar_channel_indices, self.plots):
            plot.hideAxis("left")
            for _, index in enumerate(indices):
                background_bar = pyqtgraph.BarGraphItem(
                    x=[index + 1],
                    height=1.0,
                    width=0.9,
                    pen=self.pen,
                    brush=self.background_brush,
                )
                plot.addItem(background_bar)
                self.background_bars.append(background_bar)
                history_bar = pyqtgraph.BarGraphItem(
                    x=[index + 1],
                    height=0,
                    width=0.9,
                    pen=self.pen,
                    brush=self.history_brush,
                )
                plot.addItem(history_bar)
                self.history_bars.append(history_bar)
                current_bar = pyqtgraph.BarGraphItem(
                    x=[index + 1],
                    height=0,
                    width=0.9,
                    pen=self.pen,
                    brush=self.current_brush,
                )
                plot.addItem(current_bar)
                self.level_bars.append(current_bar)
                self.history_last_update.append(0)
                self.aborted_channels.append(False)

    def update(self, channel_levels):
        """Updates the level data in each bar"""
        # print('Data {:}'.format(channel_levels.shape))
        # print(channel_levels)
        for index, (
            level,
            current_bar,
            history_bar,
            background_bar,
            history_last_update,
            warning,
            abort,
            value_range,
            aborted,
        ) in enumerate(
            zip(
                channel_levels,
                self.level_bars,
                self.history_bars,
                self.background_bars,
                self.history_last_update,
                self.channel_warning_limits,
                self.channel_abort_limits,
                self.channel_ranges,
                self.aborted_channels,
            )
        ):
            # Set the current bar height
            current_height = level / value_range
            current_bar.setOpts(height=current_height if current_height < 1 else 1)
            # Now look at the history bar
            last_history_height = history_bar.opts.get("height")
            # print(last_history_height)
            if history_last_update > self.history_hold_frames:
                desired_history_height = (
                    last_history_height - 1 / self.history_hold_frames
                )
            else:
                desired_history_height = last_history_height
            if desired_history_height < current_height:
                desired_history_height = current_height
                self.history_last_update[index] = 0
            else:
                self.history_last_update[index] += 1
            history_bar.setOpts(
                height=1 if desired_history_height > 1 else desired_history_height
            )
            # Now look at the pen color
            if level > abort or aborted:
                current_bar.setOpts(brushes=[self.abort_brush])
                background_bar.setOpts(brushes=[self.abort_background_brush])
                history_bar.setOpts(brushes=[self.abort_history_brush])
                self.aborted_channels[index] = True
            elif level > warning:
                current_bar.setOpts(brushes=[self.limit_brush])
                background_bar.setOpts(brushes=[self.limit_background_brush])
                history_bar.setOpts(brushes=[self.limit_history_brush])


class VaryingNumberOfLinePlot:
    """A plot that can have a dynamic number of lines assigned,
    adding or removing lines as necessary"""

    def __init__(self, plot_item, initial_abscissa=None, initial_ordinate=None):
        self.plot_item = plot_item
        self.lines = []
        if initial_abscissa is not None and initial_ordinate is not None:
            self.set_data(initial_abscissa, initial_ordinate)

    def set_data(self, abscissa, ordinate):
        """Sets the data of the plot

        Parameters
        ----------
        abscissa : np.ndarray
            A 2D dataset where each row is a different plot and the columns are the abscissa values
            of each curve
        ordinate : np.ndarray
            A 2D dataset where each row is a different plot and the columns are the ordinate values
            of each curve
        """
        for i, (this_ordinate, this_abscissa) in enumerate(zip(ordinate, abscissa)):
            try:
                self.lines[i].setData(this_abscissa, this_ordinate)
            except IndexError:
                pen = {"color": colororder[i % len(colororder)]}
                self.lines.append(
                    self.plot_item.plot(this_abscissa, this_ordinate, pen=pen)
                )

        # Remove extra lines
        extra_lines = len(self.lines) - len(ordinate)
        for i in range(extra_lines):
            line = self.lines.pop()
            self.plot_item.removeItem(line)

    def clear(self):
        """Clears all data from the plots"""
        self.lines = []
        self.plot_item.clear()


# endregion

# region Hardware
ip_manager_ui_path = os.path.join(
    DIRECTORY, "user_interface", "ui_files", "ip_manager.ui"
)


class IPAddress:
    """Container for information about IPAddress, mainly used to make
    sure each address has a values for relevant information"""

    def __init__(
        self, host_name=None, ipv4_address=None, ipv6_address=None, valid_ip=False
    ):
        self.host_name = host_name
        self.ipv4_address = ipv4_address
        self.ipv6_address = ipv6_address
        self.valid_ip = valid_ip


class IPAddressManager(QtWidgets.QDialog):
    """A class to manage IP addresses"""

    def __init__(self, ip_addresses: list[IPAddress] = None, parent=None):
        if ip_addresses is None:
            ip_addresses = []
        super().__init__(parent)
        uic.loadUi(ip_manager_ui_path, self)

        self.ip_address_table.setColumnWidth(0, 200)
        self.ip_address_table.setColumnWidth(1, 200)
        self.ip_address_table.setColumnWidth(2, 250)

        self.ip_addresses = []
        self.unique_indices = []
        for ind, address in enumerate(ip_addresses):
            self.add_ip_address()
            self.ip_addresses[ind] = address

        self.validation_timeout = 0.5
        self.selected_index = -1

        self.refresh_ip_table()
        self.loading_bar.hide()

        self.connect_callbacks()

        self.setWindowIcon(QtGui.QIcon("logo/Rattlesnake_Icon.png"))

    def connect_callbacks(self):
        """Connects callbacks to the widgets"""
        self.add_ip_address_button.clicked.connect(self.add_ip_address)
        self.remove_ip_address_button.clicked.connect(self.remove_ip_address)
        self.validate_ip_address_button.clicked.connect(self.validate_button_pressed)

        self.button_box.accepted.disconnect()
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def set_row_count(self, row_count):
        """Sets the number of rows in the table"""
        while self.ip_address_table.rowCount() < row_count:
            clicked = False
            self.add_ip_address(clicked)

    def add_ip_address(
        self, clicked=None, append_list=True
    ):  # pylint: disable=unused-argument
        """Adds a new IP address to the manager"""
        if append_list:
            new_ip = IPAddress()
            self.ip_addresses.append(new_ip)

            unique_index = 0
            while unique_index in self.unique_indices:
                unique_index += 1
            self.unique_indices.append(unique_index)

        # Add new row to list
        current_row = self.ip_address_table.rowCount()
        self.ip_address_table.setRowCount(len(self.unique_indices))

        # Add a host name line edit with move up and move down buttons
        host_name_widget = QtWidgets.QWidget()
        host_name_layout = QtWidgets.QHBoxLayout(host_name_widget)
        host_name_layout.setContentsMargins(0, 0, 0, 0)
        host_name_layout.setSpacing(0)

        move_button_layout = QtWidgets.QVBoxLayout()
        move_button_layout.setContentsMargins(0, 0, 0, 0)
        move_button_layout.setSpacing(0)

        up_button = QtWidgets.QToolButton()
        up_button.setArrowType(QtCore.Qt.UpArrow)
        up_button.setFixedSize(30, 15)
        up_button.clicked.connect(lambda: self.move_address_up(unique_index))

        down_button = QtWidgets.QToolButton()
        down_button.setArrowType(QtCore.Qt.DownArrow)
        down_button.setFixedSize(30, 15)
        down_button.clicked.connect(lambda: self.move_address_down(unique_index))

        move_button_layout.addWidget(up_button)
        move_button_layout.addWidget(down_button)

        host_name_input = QtWidgets.QLineEdit()
        host_name_input.setFixedHeight(45)
        host_name_input.setPlaceholderText("BK<Type>-<Serial>")
        host_name_input.textChanged.connect(
            lambda text: self.host_name_changed(text, unique_index)
        )
        host_name_input.focusInEvent = lambda event: self.ip_input_focused(
            event, unique_index
        )

        host_name_layout.addLayout(move_button_layout)
        host_name_layout.addWidget(host_name_input)

        self.ip_address_table.setCellWidget(current_row, 0, host_name_widget)

        ipv4_input = QtWidgets.QLineEdit()
        ipv4_input.setFixedHeight(45)
        ipv4_input.setPlaceholderText("169.254.001.001")
        ipv4_input.textChanged.connect(
            lambda text: self.ipv4_address_changed(text, unique_index)
        )
        ipv4_input.focusInEvent = lambda event: self.ip_input_focused(
            event, unique_index
        )

        self.ip_address_table.setCellWidget(current_row, 1, ipv4_input)

        ipv6_input = QtWidgets.QLineEdit()
        ipv6_input.setFixedHeight(45)
        ipv6_input.setPlaceholderText("[<Unicast>%<Network>]")
        ipv6_input.textChanged.connect(
            lambda text: self.ipv6_address_changed(text, unique_index)
        )
        ipv6_input.focusInEvent = lambda event: self.ip_input_focused(
            event, unique_index
        )

        self.ip_address_table.setCellWidget(current_row, 2, ipv6_input)

    def host_name_changed(self, text: str, unique_index: int):
        """Updates the host name"""
        try:
            current_row = self.unique_indices.index(unique_index)
        except ValueError:
            return
        self.ip_addresses[current_row].host_name = text
        self.ip_addresses[current_row].valid_ip = False

    def ipv4_address_changed(self, text: str, unique_index: int):
        """Updates the IPv4 Address"""
        try:
            current_row = self.unique_indices.index(unique_index)
        except ValueError:
            return
        self.ip_addresses[current_row].ipv4_address = text
        self.ip_addresses[current_row].valid_ip = False

    def ipv6_address_changed(self, text: str, unique_index: int):
        """Updates the IPv6 Address"""
        try:
            current_row = self.unique_indices.index(unique_index)
        except ValueError:
            return
        self.ip_addresses[current_row].ipv6_address = text
        self.ip_addresses[current_row].valid_ip = False

    def ip_input_focused(
        self, event, unique_index: int
    ):  # pylint: disable=unused-argument
        """Updates the selected index based on the window focus"""
        self.selected_index = unique_index

    def remove_ip_address(self):
        """Removes the currently selected IP Address"""
        try:
            current_row = self.unique_indices.index(self.selected_index)
        except ValueError:
            return
        if 0 <= current_row < len(self.unique_indices):
            self.ip_address_table.removeRow(
                current_row,
            )
            self.ip_addresses.pop(current_row)
            self.unique_indices.pop(current_row)
            self.selected_index = -1

    def move_address_up(self, ip_index):
        """Shifts values up one line edit"""
        # Just shifts text values up one LineEdit, unique_indices correspond
        # to LineEdit objects which dont shift therefore the unique_indices dont change
        try:
            current_row = self.unique_indices.index(ip_index)
        except ValueError:
            return
        if current_row > 0:
            move_ip = self.ip_addresses.pop(current_row)
            self.ip_addresses.insert(current_row - 1, move_ip)
            self.refresh_ip_table([current_row - 1, current_row])

    def move_address_down(self, ip_index):
        """Shifts the addresses down one line edit"""
        # Just shifts text values down one LineEdit, unique_indices correspond
        # to LineEdit objects which dont shift therefore the unique_indices dont change
        try:
            current_row = self.unique_indices.index(ip_index)
        except ValueError:
            return
        if current_row < len(self.unique_indices) - 1:
            move_ip = self.ip_addresses.pop(current_row)
            self.ip_addresses.insert(current_row + 1, move_ip)
            self.refresh_ip_table([current_row, current_row + 1])

    def refresh_ip_table(self, rows: list[int] = None):
        """Refreshes the IP address table"""
        if rows is None:
            rows = range(len(self.unique_indices))

        # This is slower than just deleting widgets and refreshing them but I do
        # this to keep a consistent unique indice corresponding to rows
        for row_idx in rows:
            if row_idx >= self.ip_address_table.rowCount():
                return

            host_name = (
                str(self.ip_addresses[row_idx].host_name)
                if self.ip_addresses[row_idx].host_name is not None
                else ""
            )
            host_name_widget = self.ip_address_table.cellWidget(row_idx, 0)
            host_name_input = host_name_widget.findChild(QtWidgets.QLineEdit)
            host_name_input.blockSignals(True)
            host_name_input.setText(host_name)
            host_name_input.blockSignals(False)

            ipv4_address = (
                str(self.ip_addresses[row_idx].ipv4_address)
                if self.ip_addresses[row_idx].ipv4_address is not None
                else ""
            )
            ipv4_input = self.ip_address_table.cellWidget(row_idx, 1)
            ipv4_input.blockSignals(True)
            ipv4_input.setText(ipv4_address)
            ipv4_input.blockSignals(False)

            ipv6_address = (
                str(self.ip_addresses[row_idx].ipv6_address)
                if self.ip_addresses[row_idx].ipv6_address is not None
                else ""
            )
            ipv6_input = self.ip_address_table.cellWidget(row_idx, 2)
            ipv6_input.blockSignals(True)
            ipv6_input.setText(ipv6_address)
            ipv6_input.blockSignals(False)

    def get_ip_addresses(self, host_name: str = None):
        """Gets valid IP Addresses given the host name"""
        valid_host_name = False
        ipv4_address = None
        ipv6_address = None
        try:
            # Get the address info for the hostname
            ipv4_info = socket.getaddrinfo(host_name, None, socket.AF_INET)
            ipv6_info = socket.getaddrinfo(host_name, None, socket.AF_INET6)
            ipv4 = ipv4_info[0]
            ipv4_address = ipv4[4][0]
            ipv6 = ipv6_info[0]
            ipv6_address = f"[{ipv6[4][0]}%{ipv6[4][3]}]"

            valid_host_name = self.validate_ip_address(ipv6_address)
        except (socket.gaierror, IndexError):
            # print(f'Error retrieving info')
            pass

        return (valid_host_name, ipv4_address, ipv6_address)

    def get_host_name(self, ip_address: str = None):
        """Gets the host name from an IP address"""
        host_name = None
        host = "http://" + ip_address
        valid_ip = self.validate_ip_address(ip_address)
        if valid_ip:
            try:
                response = requests.get(host + "/rest/rec/module/info", timeout=1)
                info = response.json()
                host_name = (
                    f"BK{info['module']['type']['number']}-{info['module']['serial']}"
                )
            except Exception:
                valid_ip = False
                host_name = None

        return (valid_ip, host_name)

    def validate_ip_address(self, ip_address: str = None):
        """Checks if IP addresses are valid"""
        valid_ip = False
        host = "http://" + ip_address
        try:
            response = requests.put(
                host + "/rest/rec/open", timeout=self.validation_timeout
            )
            if response.status_code == 200:
                valid_ip = True
        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.RequestException:
            pass

        return valid_ip

    def autofill_ip_addresses(self):
        """This function validates the ip address and autofills the other values.
        If multiple inputs are valid but correspond to different devices, the
        priority is host_name > ipv4 > ipv6
        Note: Having 2 of the same host names may not validate correctly due to weird
        socket waiting requirements
        """
        self.loading_bar.setValue(0)
        self.loading_bar.show()
        num_rows = len(self.unique_indices)
        for row_idx in range(num_rows):
            valid_row = self.ip_addresses[row_idx].valid_ip
            percent_complete = round((row_idx + 1) / num_rows * 100)
            self.loading_bar.setValue(percent_complete)

            # Check if you can pull information from hostname
            host_name = (
                str(self.ip_addresses[row_idx].host_name)
                if self.ip_addresses[row_idx].host_name is not None
                else ""
            )
            if not valid_row and host_name != "":
                valid_row, ipv4_address, ipv6_address = self.get_ip_addresses(host_name)

                if valid_row:
                    self.ip_addresses[row_idx].ipv4_address = ipv4_address
                    self.ip_addresses[row_idx].ipv6_address = ipv6_address
                    self.ip_addresses[row_idx].valid_ip = valid_row
                    continue

            ipv4_address = (
                str(self.ip_addresses[row_idx].ipv4_address)
                if self.ip_addresses[row_idx].ipv4_address is not None
                else ""
            )
            if not valid_row and ipv4_address is not None:
                valid_row, host_name = self.get_host_name(ipv4_address)

                if valid_row:
                    self.ip_addresses[row_idx].host_name = host_name
                    (valid_row, _, ipv6_address) = self.get_ip_addresses(host_name)
                    self.ip_addresses[row_idx].ipv6_address = ipv6_address
                    self.ip_addresses[row_idx].valid_ip = valid_row
                    continue

            ipv6_address = (
                str(self.ip_addresses[row_idx].ipv6_address)
                if self.ip_addresses[row_idx].ipv6_address is not None
                else ""
            )
            if not valid_row and ipv6_address is not None:
                valid_row, host_name = self.get_host_name(ipv6_address)

                if valid_row:
                    self.ip_addresses[row_idx].host_name = host_name
                    (valid_row, ipv4_address, _) = self.get_ip_addresses(host_name)
                    self.ip_addresses[row_idx].ipv4_address = ipv4_address
                    self.ip_addresses[row_idx].valid_ip = valid_row
                    continue

        self.loading_bar.hide()

    def validate_button_pressed(self):
        """Validates the IP Addresses"""
        self.autofill_ip_addresses()
        self.refresh_ip_table()

        valid_ip_list = [ip.valid_ip for ip in self.ip_addresses]
        if not all(valid_ip_list):
            invalid_ip_rows = [
                row for row, valid_bool in enumerate(valid_ip_list) if not valid_bool
            ]
            message = (
                f"Invalid IP address at rows: {invalid_ip_rows}.\n\n  "
                f"If IPv4 connection is unstable, try inputting host name."
            )
            reply = QtWidgets.QMessageBox.question(
                self,
                "Invalid IP Addresses",
                message,
                QtWidgets.QMessageBox.Ok,
                QtWidgets.QMessageBox.Ok,
            )

    def closeEvent(self, a0):  # pylint: disable=unused-argument,invalid-name
        """Returns the IP addresses"""
        return self.ip_addresses


# endregion


# region Profile
class ProfileTimer(QTimer):
    """A timer class that allows storage of controller instruction information"""

    def __init__(self, environment: str, operation: str, data: str):
        """
        A timer class that allows storage of controller instruction information

        When the timer times out, the environment, operation, and any data can
        be collected by the callback by accessing the self.sender().environment,
        .operation, or .data attributes.

        Parameters
        ----------
        environment : str
            The name of the environment (or 'Global') that the instruction will
            be sent to
        operation : str
            The operation that the environment will be instructed to perform
        data : str
            Any data corresponding to that operation that is required


        """
        super().__init__()
        self.environment = environment
        self.operation = operation
        self.data = data


# endregion


# region Modal
class ModalMDISubWindow(QtWidgets.QWidget):
    """A window that shows modal data"""

    def __init__(self, parent):
        super().__init__(parent)
        uic.loadUi(modal_mdi_ui_path, self)

        self.parent = parent
        self.channel_names = self.parent.channel_names
        self.reference_names = np.array(
            [
                self.parent.channel_names[i]
                for i in self.parent.reference_channel_indices
            ]
        )
        self.response_names = np.array(
            [self.parent.channel_names[i] for i in self.parent.response_channel_indices]
        )
        self.reciprocal_responses = self.parent.reciprocal_responses

        self.signal_selector.currentIndexChanged.connect(self.update_ui)
        self.data_type_selector.currentIndexChanged.connect(self.update_ui_no_clear)
        self.response_coordinate_selector.currentIndexChanged.connect(self.update_data)
        self.reference_coordinate_selector.currentIndexChanged.connect(self.update_data)

        self.primary_plotitem = self.primary_plot.getPlotItem()
        self.secondary_plotitem = self.secondary_plot.getPlotItem()
        self.primary_viewbox = self.primary_plotitem.getViewBox()
        self.secondary_viewbox = self.secondary_plotitem.getViewBox()
        self.primary_axis = self.primary_plotitem.getAxis("left")
        self.secondary_axis = self.secondary_plotitem.getAxis("left")

        self.secondary_plotitem.setXLink(self.primary_plotitem)

        self.primary_plotdataitem = pyqtgraph.PlotDataItem(
            np.arange(2), np.zeros(2), pen={"color": "r", "width": 1}
        )
        self.secondary_plotdataitem = pyqtgraph.PlotDataItem(
            np.arange(2), np.zeros(2), pen={"color": "r", "width": 1}
        )

        self.primary_viewbox.addItem(self.primary_plotdataitem)
        self.secondary_viewbox.addItem(self.secondary_plotdataitem)

        self.twinx_viewbox = None
        self.twinx_axis = None
        self.twinx_original_plotitem = None
        self.twinx_plotdataitem = None

        self.is_comparing = False
        self.primary_plotdataitem_compare = pyqtgraph.PlotDataItem(
            np.arange(2), np.zeros(2), pen={"color": "b", "width": 1}
        )
        self.secondary_plotdataitem_compare = pyqtgraph.PlotDataItem(
            np.arange(2), np.zeros(2), pen={"color": "b", "width": 1}
        )

        self.update_ui()

    def remove_twinx(self):
        """Removes the overlaid plot"""
        if self.twinx_viewbox is None:
            return
        self.twinx_original_plotitem.layout.removeItem(self.twinx_axis)
        self.twinx_original_plotitem.scene().removeItem(self.twinx_viewbox)
        self.twinx_original_plotitem.scene().removeItem(self.twinx_axis)
        self.twinx_viewbox = None
        self.twinx_axis = None
        self.twinx_original_plotitem = None

    def add_twinx(self, existing_plot_item: pyqtgraph.PlotItem):
        """Adds an overlaid plot"""
        # Create a viewbox
        self.twinx_original_plotitem = existing_plot_item
        self.twinx_viewbox = pyqtgraph.ViewBox()
        self.twinx_original_plotitem.scene().addItem(self.twinx_viewbox)
        self.twinx_axis = pyqtgraph.AxisItem("right")
        self.twinx_axis.setLogMode(False)
        self.twinx_axis.linkToView(self.twinx_viewbox)
        self.twinx_original_plotitem.layout.addItem(self.twinx_axis, 2, 3)
        self.updateTwinXViews()
        self.twinx_viewbox.setXLink(self.twinx_original_plotitem)
        self.twinx_original_plotitem.vb.sigResized.connect(self.updateTwinXViews)
        self.twinx_plotdataitem = pyqtgraph.PlotDataItem(
            np.arange(2), np.zeros(2), pen={"color": "b", "width": 1}
        )
        self.twinx_viewbox.addItem(self.twinx_plotdataitem)

    def add_compare(self):
        """Adds a second function for comparison for reciprocal plots"""
        self.is_comparing = True
        self.primary_viewbox.addItem(self.primary_plotdataitem_compare)
        self.secondary_viewbox.addItem(self.secondary_plotdataitem_compare)

    def remove_compare(self):
        """Removes the second function that was used for comparison"""
        if self.is_comparing:
            self.primary_viewbox.removeItem(self.primary_plotdataitem_compare)
            self.secondary_viewbox.removeItem(self.secondary_plotdataitem_compare)
            self.is_comparing = False

    def updateTwinXViews(self):  # pylint: disable=invalid-name
        """Updates the second view box based on the view from the first box"""
        if self.twinx_viewbox is None:
            return
        self.twinx_viewbox.setGeometry(
            self.twinx_original_plotitem.vb.sceneBoundingRect()
        )
        # self.twinx_viewbox.linkedViewChanged(
        #     self.twinx_original_plotitem.vb, self.twinx_viewbox.XAxis)

    def update_ui_no_clear(self):
        """Updates the UI without clearing the data"""
        self.update_ui(False)

    def update_ui(self, clear_channels=True):
        """Updates the UI based on which function type is selected"""
        self.response_coordinate_selector.blockSignals(True)
        self.reference_coordinate_selector.blockSignals(True)
        self.remove_twinx()
        self.remove_compare()
        if self.signal_selector.currentIndex() in [
            0,
            1,
            2,
            3,
        ]:  # Time or Windowed Time or Spectrum or Autospectrum
            self.reference_coordinate_selector.hide()
            self.data_type_selector.hide()
            self.secondary_plot.hide()
            if clear_channels:
                self.response_coordinate_selector.clear()
                self.reference_coordinate_selector.clear()
                for channel_name in self.channel_names:
                    self.response_coordinate_selector.addItem(channel_name)
            if self.signal_selector.currentIndex() in [0, 1]:
                self.primary_axis.setLogMode(False)
                self.primary_plotdataitem.setLogMode(False, False)
            else:
                self.primary_axis.setLogMode(True)
                self.primary_plotdataitem.setLogMode(False, True)
        elif self.signal_selector.currentIndex() in [
            4,
            6,
            7,
        ]:  # FRF or FRF Coherence or Reciprocity
            self.reference_coordinate_selector.show()
            self.data_type_selector.show()
            if self.data_type_selector.currentIndex() in [1, 4]:
                self.secondary_plot.show()
                if self.signal_selector.currentIndex() == 6:
                    self.add_twinx(self.secondary_plotitem)
            else:
                self.secondary_plot.hide()
                if self.signal_selector.currentIndex() == 6:
                    self.add_twinx(self.primary_plotitem)
            if self.signal_selector.currentIndex() == 7:
                if any([val is None for val in self.reciprocal_responses]):
                    error_message_qt(
                        "Invalid Reciprocal Channels",
                        "Could not deterimine reciprocal channels for this test",
                    )
                    self.signal_selector.setCurrentIndex(4)
                    return
                self.add_compare()
            if clear_channels:
                self.response_coordinate_selector.clear()
                self.reference_coordinate_selector.clear()
                if self.signal_selector.currentIndex() == 7:
                    for channel_name in self.response_names[self.reciprocal_responses]:
                        self.response_coordinate_selector.addItem(channel_name)
                else:
                    for channel_name in self.response_names:
                        self.response_coordinate_selector.addItem(channel_name)
                for channel_name in self.reference_names:
                    self.reference_coordinate_selector.addItem(channel_name)
            if self.data_type_selector.currentIndex() == 0:
                self.primary_axis.setLogMode(True)
                self.primary_plotdataitem.setLogMode(False, True)
                self.primary_plotdataitem_compare.setLogMode(False, True)
            elif self.data_type_selector.currentIndex() == 1:
                self.primary_axis.setLogMode(False)
                self.primary_plotdataitem.setLogMode(False, False)
                self.primary_plotdataitem_compare.setLogMode(False, False)
                self.secondary_axis.setLogMode(True)
                self.secondary_plotdataitem.setLogMode(False, True)
                self.secondary_plotdataitem_compare.setLogMode(False, True)
            elif self.data_type_selector.currentIndex() in [2, 3]:
                self.primary_axis.setLogMode(False)
                self.primary_plotdataitem.setLogMode(False, False)
                self.primary_plotdataitem_compare.setLogMode(False, False)
            elif self.data_type_selector.currentIndex() == 4:
                self.primary_axis.setLogMode(False)
                self.primary_plotdataitem.setLogMode(False, False)
                self.primary_plotdataitem_compare.setLogMode(False, False)
                self.secondary_axis.setLogMode(False)
                self.secondary_plotdataitem.setLogMode(False, False)
                self.secondary_plotdataitem_compare.setLogMode(False, False)
            if self.signal_selector.currentIndex() == 6:
                self.twinx_axis.setLogMode(False)
                self.twinx_plotdataitem.setLogMode(False, False)
        elif self.signal_selector.currentIndex() in [5]:  # Coherence
            self.reference_coordinate_selector.hide()
            self.data_type_selector.hide()
            self.secondary_plot.hide()
            if clear_channels:
                self.response_coordinate_selector.clear()
                self.reference_coordinate_selector.clear()
                for channel_name in self.response_names:
                    self.response_coordinate_selector.addItem(channel_name)
            self.primary_axis.setLogMode(False)
            self.primary_plotdataitem.setLogMode(False, False)
        self.update_data()
        self.response_coordinate_selector.blockSignals(False)
        self.reference_coordinate_selector.blockSignals(False)

    def set_window_title(self):
        """Sets the window title"""
        signal_name = self.signal_selector.itemText(self.signal_selector.currentIndex())
        response_name = self.response_coordinate_selector.itemText(
            self.response_coordinate_selector.currentIndex()
        )
        reference_name = (
            self.reference_coordinate_selector.itemText(
                self.reference_coordinate_selector.currentIndex()
            )
            if self.signal_selector.currentIndex() == 4
            else ""
        )
        self.setWindowTitle(f"{signal_name} {response_name} {reference_name}")

    def update_data(self):
        """Updates the data in the plot"""
        self.set_window_title()
        current_index = self.signal_selector.currentIndex()
        if current_index in [0, 1]:  # Time history
            if self.parent.last_frame is None:
                return
            data = self.parent.last_frame[
                self.response_coordinate_selector.currentIndex()
            ]
            if current_index == 1:
                data = data * self.parent.window_function
            self.primary_plotdataitem.setData(self.parent.time_abscissa, data)
        elif current_index == 2:  # Spectrum
            if self.parent.last_spectrum is None:
                return
            data = self.parent.last_spectrum[
                self.response_coordinate_selector.currentIndex()
            ]
            self.primary_plotdataitem.setData(self.parent.frequency_abscissa, data)
        elif current_index == 3:  # Autospectrum
            if self.parent.last_autospectrum is None:
                return
            data = self.parent.last_autospectrum[
                self.response_coordinate_selector.currentIndex()
            ]
            self.primary_plotdataitem.setData(self.parent.frequency_abscissa, data)
        elif current_index == 4 or current_index == 6:  # FRF or FRF Coherence
            if self.parent.last_frf is None:
                return
            data = self.parent.last_frf[
                :,
                self.response_coordinate_selector.currentIndex(),
                self.reference_coordinate_selector.currentIndex(),
            ]
            if self.data_type_selector.currentIndex() == 0:  # Magnitude
                self.primary_plotdataitem.setData(
                    self.parent.frequency_abscissa, np.abs(data)
                )
            elif self.data_type_selector.currentIndex() == 1:  # Magnitude/Phase
                self.primary_plotdataitem.setData(
                    self.parent.frequency_abscissa, np.angle(data)
                )
                self.secondary_plotdataitem.setData(
                    self.parent.frequency_abscissa, np.abs(data)
                )
            elif self.data_type_selector.currentIndex() == 2:  # Real
                self.primary_plotdataitem.setData(
                    self.parent.frequency_abscissa, np.real(data)
                )
            elif self.data_type_selector.currentIndex() == 3:  # Imag
                self.primary_plotdataitem.setData(
                    self.parent.frequency_abscissa, np.imag(data)
                )
            elif self.data_type_selector.currentIndex() == 4:  # Real/Imag
                self.primary_plotdataitem.setData(
                    self.parent.frequency_abscissa, np.real(data)
                )
                self.secondary_plotdataitem.setData(
                    self.parent.frequency_abscissa, np.imag(data)
                )
            if current_index == 6:
                data = self.parent.last_coh[
                    self.response_coordinate_selector.currentIndex()
                ]
                self.twinx_plotdataitem.setData(self.parent.frequency_abscissa, data)
        elif current_index == 5:  # Coherence
            if self.parent.last_coh is None:
                return
            data = self.parent.last_coh[
                self.response_coordinate_selector.currentIndex()
            ]
            self.primary_plotdataitem.setData(self.parent.frequency_abscissa, data)
        elif current_index == 7:  # FRF or FRF Coherence
            if self.parent.last_frf is None:
                return
            resp_ind = self.response_coordinate_selector.currentIndex()
            ref_ind = self.reference_coordinate_selector.currentIndex()
            data = self.parent.last_frf[:, self.reciprocal_responses[resp_ind], ref_ind]
            compare_data = self.parent.last_frf[
                :, self.reciprocal_responses[ref_ind], resp_ind
            ]
            if self.data_type_selector.currentIndex() == 0:  # Magnitude
                self.primary_plotdataitem.setData(
                    self.parent.frequency_abscissa, np.abs(data)
                )
                self.primary_plotdataitem_compare.setData(
                    self.parent.frequency_abscissa, np.abs(compare_data)
                )
            elif self.data_type_selector.currentIndex() == 1:  # Magnitude/Phase
                self.primary_plotdataitem.setData(
                    self.parent.frequency_abscissa, np.angle(data)
                )
                self.secondary_plotdataitem.setData(
                    self.parent.frequency_abscissa, np.abs(data)
                )
                self.primary_plotdataitem_compare.setData(
                    self.parent.frequency_abscissa, np.angle(compare_data)
                )
                self.secondary_plotdataitem_compare.setData(
                    self.parent.frequency_abscissa, np.abs(compare_data)
                )
            elif self.data_type_selector.currentIndex() == 2:  # Real
                self.primary_plotdataitem.setData(
                    self.parent.frequency_abscissa, np.real(data)
                )
                self.primary_plotdataitem_compare.setData(
                    self.parent.frequency_abscissa, np.real(compare_data)
                )
            elif self.data_type_selector.currentIndex() == 3:  # Imag
                self.primary_plotdataitem.setData(
                    self.parent.frequency_abscissa, np.imag(data)
                )
                self.primary_plotdataitem_compare.setData(
                    self.parent.frequency_abscissa, np.imag(compare_data)
                )
            elif self.data_type_selector.currentIndex() == 4:  # Real/Imag
                self.primary_plotdataitem.setData(
                    self.parent.frequency_abscissa, np.real(data)
                )
                self.secondary_plotdataitem.setData(
                    self.parent.frequency_abscissa, np.imag(data)
                )
                self.primary_plotdataitem_compare.setData(
                    self.parent.frequency_abscissa, np.real(compare_data)
                )
                self.secondary_plotdataitem_compare.setData(
                    self.parent.frequency_abscissa, np.imag(compare_data)
                )

    def increment_channel(self, increment=1):
        """Increments the channel number by the specified amount"""
        if not self.lock_response_checkbox.isChecked():
            num_channels = self.response_coordinate_selector.count()
            current_index = self.response_coordinate_selector.currentIndex()
            new_index = (current_index + increment) % num_channels
            self.response_coordinate_selector.setCurrentIndex(new_index)


# endregion

# region Deteriorated

# Define paths to the User Interface UI Files
this_path = os.path.split(__file__)[0]
environment_definition_ui_paths = {}
environment_prediction_ui_paths = {}
environment_run_ui_paths = {}
# This is true if running from an executable and the UI is embedded in the executable
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    directory = sys._MEIPASS  # pylint: disable=protected-access
else:
    directory = this_path

# Base Controller UI
directory = os.path.join(directory, "ui_files")
ui_path = os.path.join(directory, "combined_environments_controller.ui")
environment_select_ui_path = os.path.join(directory, "environment_selector.ui")
control_select_ui_path = os.path.join(directory, "control_select.ui")
# Random Vibration Environment
environment_definition_ui_paths[ControlTypes.RANDOM] = os.path.join(
    directory, "random_vibration_definition.ui"
)
environment_prediction_ui_paths[ControlTypes.RANDOM] = os.path.join(
    directory, "random_vibration_prediction.ui"
)
environment_run_ui_paths[ControlTypes.RANDOM] = os.path.join(
    directory, "random_vibration_run.ui"
)
system_identification_ui_path = os.path.join(directory, "system_identification.ui")
transformation_matrices_ui_path = os.path.join(directory, "transformation_matrices.ui")
# Time Environment
environment_definition_ui_paths[ControlTypes.TIME] = os.path.join(
    directory, "time_definition.ui"
)
environment_run_ui_paths[ControlTypes.TIME] = os.path.join(directory, "time_run.ui")
# Transient Environment
environment_definition_ui_paths[ControlTypes.TRANSIENT] = os.path.join(
    directory, "transient_definition.ui"
)
environment_prediction_ui_paths[ControlTypes.TRANSIENT] = os.path.join(
    directory, "transient_prediction.ui"
)
environment_run_ui_paths[ControlTypes.TRANSIENT] = os.path.join(
    directory, "transient_run.ui"
)
# Sine Environment
environment_definition_ui_paths[ControlTypes.SINE] = os.path.join(
    directory, "sine_definition.ui"
)
environment_prediction_ui_paths[ControlTypes.SINE] = os.path.join(
    directory, "sine_prediction.ui"
)
environment_run_ui_paths[ControlTypes.SINE] = os.path.join(directory, "sine_run.ui")
sine_sweep_table_ui_path = os.path.join(directory, "sine_sweep_table.ui")
filter_explorer_ui_path = os.path.join(directory, "sine_filter_explorer.ui")
# Modal Environments
environment_definition_ui_paths[ControlTypes.MODAL] = os.path.join(
    directory, "modal_definition.ui"
)
environment_run_ui_paths[ControlTypes.MODAL] = os.path.join(directory, "modal_run.ui")
modal_mdi_ui_path = os.path.join(directory, "modal_acquisition_window.ui")


def get_table_bools(tablewidget: QtWidgets.QTableWidget):
    """Collect a table of booleans from a QTableWidget full of QCheckBoxes

    Parameters
    ----------
    tablewidget : QtWidgets.QTableWidget
        A table widget to pull the strings from

    Returns
    -------
    bool_array : list[list[bool]]
        A nested list of booleans from the table widgets

    """
    bool_array = []
    for row_idx in range(tablewidget.rowCount()):
        bool_array.append([])
        for col_idx in range(tablewidget.columnCount()):
            value = tablewidget.cellWidget(row_idx, col_idx).isChecked()
            bool_array[-1].append(value)
    return bool_array


def load_time_history(signal_path, sample_rate):
    """Loads a time history from a given file

    The signal can be loaded from numpy files (.npz, .npy) or matlab files (.mat).
    For .mat and .npz files, the time data can be included in the file in the
    't' field, or it can be excluded and the sample_rate input argument will
    be used.  If time data is specified, it will be linearly interpolated to the
    sample rate of the controller.
    For these file types, the signal should be stored in the 'signal'
    field.  For .npy files, only one array is stored, so it is treated as the
    signal, and the sample_rate input argument is used to construct the time
    data.

    Parameters
    ----------
    signal_path : str:
        Path to the file from which to load the time history

    sample_rate : str:
        The sample rate of the loaded signal.

    Returns
    -------
    signal : np.ndarray:
        A signal loaded from the file

    """
    _, extension = os.path.splitext(signal_path)
    if extension.lower() == ".npy":
        signal = np.load(signal_path)
    elif extension.lower() == ".npz":
        data = np.load(signal_path)
        signal = data["signal"]
        try:
            times = data["t"].squeeze()
            fn = interp1d(times, signal)
            abscissa = np.arange(
                0, max(times) + 1 / sample_rate - 1e-10, 1 / sample_rate
            )
            abscissa = abscissa[abscissa <= max(times)]
            signal = fn(abscissa)
        except KeyError:
            pass
    elif extension.lower() == ".mat":
        data = loadmat(signal_path)
        signal = data["signal"]
        try:
            times = data["t"].squeeze()
            fn = interp1d(times, signal)
            abscissa = np.arange(
                0, max(times) + 1 / sample_rate - 1e-10, 1 / sample_rate
            )
            abscissa = abscissa[abscissa <= max(times)]
            signal = fn(abscissa)
        except KeyError:
            pass
    else:
        raise ValueError(
            f"Could Not Determine the file type from the filename {signal_path}: {extension}"
        )
    if signal.shape[-1] % 2 == 1:
        signal = signal[..., :-1]
    return signal


class ControlSelect(QtWidgets.QDialog):
    """Environment selector dialog box to select the control type for the test"""

    def __init__(self, parent=None):
        """
        Selects the environment type that gets used for the test.

        This function reads from the environment control types to populate the
        radiobuttons on the dialog.

        Parameters
        ----------
        parent : QWidget, optional
            Parent of the dialog box. The default is None.

        """
        super(QtWidgets.QDialog, self).__init__(parent)
        uic.loadUi(control_select_ui_path, self)
        self.setWindowIcon(QtGui.QIcon("logo/Rattlesnake_Icon.png"))

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.control_select_buttongroup = QtWidgets.QButtonGroup()

        # Go through and create radiobuttons for each control type
        control_types_sorted = sorted(
            [(control_type.value, control_type) for control_type in ControlTypes]
        )

        for value, control_type in control_types_sorted[1:] + control_types_sorted[:1]:
            radiobutton = QtWidgets.QRadioButton(environment_long_names[control_type])
            self.control_select_buttongroup.addButton(radiobutton, value)
            if value == ControlTypes.RANDOM.value:
                radiobutton.setChecked(True)
            self.environment_radiobutton_layout.addWidget(radiobutton)

    @staticmethod
    def select_control(parent=None):
        """Create the dialog box and parse the output

        Parameters
        ----------
        parent : QWidget
            Parent of the dialog box (Default value = None)

        Returns
        -------
        button_id : int
            The index of the button that was pressed
        result : bool
            True if dialog was accepted, otherwise false if cancelled.
        """
        dialog = ControlSelect(parent)
        result = dialog.exec_() == QtWidgets.QDialog.Accepted
        index = dialog.control_select_buttongroup.checkedId()
        button_id = ControlTypes(index)
        # print(button_id)
        return (button_id, result)


# endregion
