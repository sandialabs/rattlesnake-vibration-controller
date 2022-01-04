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
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtCore import Qt,QTimer
import numpy as np
import pyqtgraph
from scipy.io import loadmat
from scipy.interpolate import interp1d
import importlib.util
import openpyxl

from .utilities import (coherence,error_message_qt,save_csv_matrix,
                        load_csv_matrix,trac)
from .environments import (ControlTypes,environment_long_names,
                           combined_environments_capable,control_select_ui_path,
                           environment_select_ui_path,environment_UIs,
                           transformation_matrices_ui_path
                           )

ACQUISITION_FRAMES_TO_DISPLAY = 4

class ProfileTimer(QTimer):
    """A timer class that allows storage of controller instruction information"""
    def __init__(self,environment : str,operation : str,data : str):
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

def get_table_strings(tablewidget : QtWidgets.QTableWidget):
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
            value = tablewidget.item(row_idx,col_idx).text()
            string_array[-1].append(value)
    return string_array

def get_table_bools(tablewidget : QtWidgets.QTableWidget):
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
            value = tablewidget.cellWidget(row_idx,col_idx).isChecked()
            bool_array[-1].append(value)
    return bool_array
    
def load_python_module(module_path):
    """Loads in the Python file at the specified path as a module at runtime

    Parameters
    ----------
    module_path : str:
        Path to the module to be loaded
        

    Returns
    -------
    module : module:
        A reference to the loaded module
    """
    path,file = os.path.split(module_path)
    file,ext = os.path.splitext(file)
    spec = importlib.util.spec_from_file_location(file, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_time_history(signal_path,sample_rate):
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
    file_base,extension = os.path.splitext(signal_path)
    if extension.lower() == '.npy':
        signal = np.load(signal_path)
    elif extension.lower() == '.npz':
        data = np.load(signal_path)
        signal = data['signal']
        try:
            times = data['t'].squeeze()
            fn = interp1d(times,signal)
            signal = fn(np.arange(0,max(times),1/sample_rate))
        except KeyError:
            pass
    elif extension.lower() == '.mat':
        data = loadmat(signal_path)
        signal = data['signal']
        try:
            times = data['t'].squeeze()
            fn = interp1d(times,signal)
            signal = fn(np.arange(0,max(times),1/sample_rate))
        except KeyError:
            pass
    return signal

def load_specification(spec_path,n_freq_lines,df):
    """Loads a specification CPSD matrix from a file.

    Parameters
    ----------
    spec_path : str
        Loads the specification contained in this file
    n_freq_lines : int
        The number of frequency lines 
    df : float
        The frequency spacing

    Returns
    -------
    frequency_lines : np.ndarray
        The frequency lines ``df*np.arange(n_freq_lines)``
    cpsd_matrix : np.ndarray
        3D numpy array consisting of a CPSD matrix at each frequency line
    """
    file_base,extension = os.path.splitext(spec_path)
    if extension.lower() == '.mat':
        data = loadmat(spec_path)
        frequencies = data['f'].squeeze()
        cpsd = data['cpsd'].transpose(2,0,1)
    elif extension.lower() == '.npz':
        data = np.load(spec_path)
        frequencies = data['f'].squeeze()
        cpsd = data['cpsd']
    
    # Create the full CPSD matrix
    frequency_lines = df*np.arange(n_freq_lines)
    cpsd_matrix = np.zeros((n_freq_lines,)+cpsd.shape[1:],dtype='complex128')
    for frequency,cpsd_line in zip(frequencies,cpsd):
        index = np.argmin(np.abs(frequency-frequency_lines))
        if abs(frequency-frequency_lines[index]) > 1e-5:
            #raise ValueError('Frequency {:} not a valid frequency ({:} closest)'.format(frequency,frequency_lines[index]))
            continue
        cpsd_matrix[index,...] = cpsd_line
    # Deliever specification to data analysis
    return frequency_lines,cpsd_matrix

colororder = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
def multiline_plotter(x,y,widget = None,curve_list = None,names=None,other_pen_options = {'width':1},legend=False):
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
    if not widget is None:
        plot_item = widget.getPlotItem()
        if legend:
            plot_item.addLegend(colCount = len(y)//10)
        handles = []
        for i,this_y in enumerate(y):
            pen = {'color':colororder[i%len(colororder)]}
            pen.update(other_pen_options)
            handles.append(plot_item.plot(x,this_y,pen=pen,name=None if names is None else names[i]))
        return handles
    elif not curve_list is None:
        for this_y,curve in zip(y,curve_list):
            curve.setData(x,y)
        return curve_list
    else:
        raise ValueError('Either Widget or list of curves must be specified')

class EnvironmentSelect(QtWidgets.QDialog):
    """QDialog for selecting the environments in a combined environments run"""
    def __init__(self,parent=None):
        """
        Constructor for the EnvironmentSelect dialog box.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget to the dialog. The default is None.

        """
        super(QtWidgets.QDialog,self).__init__(parent)
        uic.loadUi(environment_select_ui_path, self)
        self.setWindowIcon(QtGui.QIcon('logo/Rattlesnake_Icon.png'))
        
        self.add_environment_button.clicked.connect(self.add_environment)
        self.remove_environment_button.clicked.connect(self.remove_environment)
        self.load_profile_button.clicked.connect(self.load_profile)
        self.save_profile_template_button.clicked.connect(self.save_profile_template)
        self.loaded_profile = None
        
    def add_environment(self):
        """Adds a row to the environment table"""
        selected_row = self.environment_display_table.rowCount()
        self.environment_display_table.insertRow(selected_row)
        combobox = QtWidgets.QComboBox()
        for control_type in combined_environments_capable:
            combobox.addItem(control_type.name.title(),control_type.value)
        self.environment_display_table.setCellWidget(selected_row,0,combobox)
    
    def remove_environment(self):
        """Removes a row from the environment table"""
        selected_row = self.environment_display_table.currentRow()
        if selected_row >= 0:
            self.environment_display_table.removeRow(selected_row)
    
    def save_profile_template(self):
        """Saves a template for the given environments table
        
        This template can be filled out by a user and then loaded."""
        filename,file_filter = QtWidgets.QFileDialog.getSaveFileName(self,'Save Combined Environments Template',filter='Excel File (*.xlsx)')
        if filename == '':
            return
        file_base,file_type = os.path.splitext(filename)
        # Create the header
        workbook = openpyxl.Workbook()
        worksheet = workbook.active
        worksheet.title = "Channel Table"
        hardware_worksheet = workbook.create_sheet('Hardware')
        # Create the header
        worksheet.cell(row=1,column=3,value='Test Article Definition')
        worksheet.merge_cells(start_row=1, start_column=3, end_row=1, end_column=5)
        worksheet.cell(row=1,column=6,value='Instrument Definition')
        worksheet.merge_cells(start_row=1, start_column=6, end_row=1, end_column=12)
        worksheet.cell(row=1,column=13,value='Channel Definition')
        worksheet.merge_cells(start_row=1, start_column=13, end_row=1, end_column=20)
        worksheet.cell(row=1,column=21,value='Output Feedback')
        worksheet.merge_cells(start_row=1, start_column=21, end_row=1, end_column=22)
        for col_idx,val in enumerate(['Channel Index',
                                      'Control?',
                                      'Node Number',
                                      'Node Direction',
                                      'Comment',
                                      'Serial Number',
                                      'Triax DoF',
                                      'Sensitivity  (mV/EU)',
                                      'Engineering Unit',
                                      'Make',
                                      'Model',
                                      'Calibration Exp Date',
                                      'Physical Device',
                                      'Physical Channel',
                                      'Type',
                                      'Minimum Value (V)',
                                      'Maximum Value (V)',
                                      'Coupling',
                                      'Current Excitation Source',
                                      'Current Excitation Value',
                                      'Physical Device',
                                      'Physical Channel']):
            worksheet.cell(row=2,column=1+col_idx,value=val)
        # Fill out the hardware worksheet
        hardware_worksheet.cell(1,1,'Hardware Type')
        hardware_worksheet.cell(1,2,'# Enter hardware index here')
        hardware_worksheet.cell(1,3,'Hardware Indices: 0 - NI DAQmx; 1 - LAN XI; 2 - Exodus Modal Solution; 3 - State Space Integration')
        hardware_worksheet.cell(2,1,'Hardware File')
        hardware_worksheet.cell(2,2,'# Path to Hardware File (Exodus File path for index 2)')
        hardware_worksheet.cell(3,1,'Sample Rate')
        hardware_worksheet.cell(3,2,'# Sample Rate of Data Acquisition System')
        hardware_worksheet.cell(4,1,'Time Per Read')
        hardware_worksheet.cell(4,2,'# Number of seconds per Read from the Data Acquisition System')
        hardware_worksheet.cell(5,1,'Time Per Write')
        hardware_worksheet.cell(5,2,'# Number of seconds per Write to the Data Acquisition System')
        hardware_worksheet.cell(6,1,'Acquisition Processes')
        hardware_worksheet.cell(6,2,'# Maximum Number of Acquisition Processes to start to pull data from hardware')
        hardware_worksheet.cell(7,1,'Integration Oversampling')
        hardware_worksheet.cell(7,2,'# For virual control, an integration oversampling can be specified')
        # Now do the environment
        worksheet.cell(row=1,column=23,value='Environments')
        # Now do the environments
        for row in range(self.environment_display_table.rowCount()):
            combobox = self.environment_display_table.cellWidget(row,0)
            value = ControlTypes(combobox.currentData())
            name = self.environment_display_table.item(row,1).text()
            environment_UIs[value].create_environment_template(name, workbook)
            worksheet.cell(row=2,column=23+row,value=name)
        # Now create a profile page
        profile_sheet = workbook.create_sheet('Test Profile')
        profile_sheet.cell(1,1,'Time (s)')
        profile_sheet.cell(1,2,'Environment')
        profile_sheet.cell(1,3,'Operation')
        profile_sheet.cell(1,4,'Data')
        
        workbook.save(filename)
        
    def load_profile(self):
        """Loads a profile from an excel spreadsheet."""
        filename,file_filter = QtWidgets.QFileDialog.getOpenFileName(self,'Load Combined Environments Profile',filter='Excel File (*.xlsx)')
        if filename == '':
            return
        else:
            self.loaded_profile = filename
            self.accept()
    
    @staticmethod
    def select_environment(parent=None):
        """Creates the dialog box and then parses the output.
        
        Note that there are variable numbers of outputs for this function

        Parameters
        ----------
        parent : QWidget
            Parent to the dialog box (Default value = None)

        Returns
        -------
        result : int
            A flag specifying the outcome of the dialog box.  Will be 1 if the
            dialog was accepted, zero if cancelled, and -1 if a profile was
            loaded instead.
        environment_table : list of lists
            A list of environment type, environment name pairs that will be
            used to specify the environments in a test.
        loaded_profile : str
            File name to the profile file that needs to be loaded.  Only
            output if result == -1
        """
        dialog = EnvironmentSelect(parent)
        result = 1 if (dialog.exec_()==QtWidgets.QDialog.Accepted) else 0
        if dialog.loaded_profile is None:
            environment_table = []
            if result:
                for row in range(dialog.environment_display_table.rowCount()):
                    combobox = dialog.environment_display_table.cellWidget(row,0)
                    value = ControlTypes(combobox.currentData())
                    name = dialog.environment_display_table.item(row,1).text()
                    environment_table.append([value,name])
            print(environment_table)
            return result,environment_table
        else:
            result = -1
            workbook = openpyxl.load_workbook(dialog.loaded_profile)
            environment_sheets = [sheet for sheet in workbook if (not sheet.title in ['Channel Table','Hardware','Test Profile']) and sheet.cell(1,1).value == 'Control Type']
            environment_table = [(ControlTypes[sheet.cell(1,2).value.upper()],sheet.title) for sheet in environment_sheets]
            workbook.close()
            return result,environment_table,dialog.loaded_profile
            
        
class ControlSelect(QtWidgets.QDialog):
    """Environment selector dialog box to select the control type for the test"""
    def __init__(self,parent=None):
        """
        Selects the environment type that gets used for the test.
        
        This function reads from the environment control types to populate the
        radiobuttons on the dialog.

        Parameters
        ----------
        parent : QWidget, optional
            Parent of the dialog box. The default is None.

        """
        super(QtWidgets.QDialog,self).__init__(parent)
        uic.loadUi(control_select_ui_path, self)
        self.setWindowIcon(QtGui.QIcon('logo/Rattlesnake_Icon.png'))
    
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.control_select_buttongroup = QtWidgets.QButtonGroup()
        
        # Go through and create radiobuttons for each control type
        control_types_sorted = sorted([(control_type.value,control_type) for control_type in ControlTypes])
        
        for value,control_type in control_types_sorted[1:]+control_types_sorted[:1]:
            radiobutton = QtWidgets.QRadioButton(environment_long_names[control_type])
            self.control_select_buttongroup.addButton(radiobutton,value)
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
        result = dialog.exec_()==QtWidgets.QDialog.Accepted
        index = dialog.control_select_buttongroup.checkedId()
        button_id = ControlTypes(index)
        print(button_id)
        return (button_id,result)


class PlotWindow(QtWidgets.QDialog):
    """Class defining a subwindow that displays specific channel information"""
    def __init__(self,parent,row,column,datatype,specification,row_name,column_name,datatype_name):
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
        super(QtWidgets.QDialog,self).__init__(parent)
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
        plot_item.showGrid(True,True,0.25)
        plot_item.enableAutoRange()
        plot_item.getViewBox().enableAutoRange(enable=True)
        if self.datatype==0:
            plot_item.setLogMode(False,True)
        plot_item.plot(self.frequencies,self.spec_data,pen = {'color': "b", 'width': 1})
        self.curve = plot_item.plot(self.frequencies,self.data,pen = {'color': "r", 'width': 1})
        self.setWindowTitle('{:} {:} / {:}'.format(datatype_name,row_name,column_name))
        self.show()
    
    def reduce_matrix(self,matrix):
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
        if self.datatype == 0: # Magnitude
            return np.abs(matrix[...,self.row,self.column])
        elif self.datatype == 1: # Coherence
            return coherence(matrix,(self.row,self.column))
        elif self.datatype == 2: # Phase
            return np.angle(matrix[...,self.row,self.column])    
        elif self.datatype == 3: # Real
            return np.real(matrix[...,self.row,self.column])     
        elif self.datatype == 4: # Imag
            return np.imag(matrix[...,self.row,self.column])
        else:
            raise ValueError('{:} is not a valid datatype!'.format(self.datatype))
        
    def update_plot(self,cpsd_matrix):
        """Updates the plot with the given CPSD matrix data

        Parameters
        ----------
        cpsd_matrix : np.ndarray
            3D CPSD matrix that will be reduced for plotting

        """
        self.curve.setData(self.frequencies,self.reduce_matrix(cpsd_matrix))
        
        
class PlotTimeWindow(QtWidgets.QDialog):
    """Class defining a subwindow that displays specific channel information"""
    def __init__(self,parent,index,specification,sample_rate,index_name):
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
        super(QtWidgets.QDialog,self).__init__(parent)
        self.setWindowFlags(self.windowFlags() & Qt.Tool)
        self.index = index
        self.times = np.arange(specification.shape[-1])/sample_rate
        self.spec_data = self.reduce_matrix(specification)
        self.data = np.zeros(self.spec_data.shape)
        # Now plot the data
        layout = QtWidgets.QVBoxLayout()
        plotwidget = pyqtgraph.PlotWidget()
        layout.addWidget(plotwidget)
        self.setLayout(layout)
        plot_item = plotwidget.getPlotItem()
        plot_item.showGrid(True,True,0.25)
        plot_item.enableAutoRange()
        plot_item.getViewBox().enableAutoRange(enable=True)
        plot_item.plot(self.times,self.spec_data,pen = {'color': "b", 'width': 1})
        plot_item.setLabel('left','TRAC: 0.0')
        self.plot_item = plot_item
        self.curve = plot_item.plot(self.times,self.data,pen = {'color': "r", 'width': 1})
        self.setWindowTitle('{:}'.format(index_name))
        self.show()
    
    def reduce_matrix(self,matrix):
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
        
    def update_plot(self,data):
        """Updates the plot with the given CPSD matrix data

        Parameters
        ----------
        cpsd_matrix : np.ndarray
            3D CPSD matrix that will be reduced for plotting

        """
        data = self.reduce_matrix(data)
        self.curve.setData(self.times,data)
        self.plot_item.setLabel('left','TRAC: {:0.2f}'.format(trac(data,self.spec_data).squeeze()))
        
        
class TransformationMatrixWindow(QtWidgets.QDialog):
    """Dialog box for specifying transformation matrices"""
    def __init__(self,parent,current_response_transformation_matrix,num_responses,
                 current_output_transformation_matrix,num_outputs):
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
        
        self.response_transformation_matrix.setColumnCount(num_responses)
        self.output_transformation_matrix.setColumnCount(num_outputs)
        
        if current_response_transformation_matrix is None:
            self.set_response_transformation_identity()
        else:
            self.response_transformation_matrix.setRowCount(current_response_transformation_matrix.shape[0])
            for row_idx,row in enumerate(current_response_transformation_matrix):
                for col_idx,col in enumerate(row):
                    try:
                        self.response_transformation_matrix.item(row_idx,col_idx).setText(str(col))
                    except AttributeError:
                        item = QtWidgets.QTableWidgetItem(str(col))
                        self.response_transformation_matrix.setItem(row_idx,col_idx,item)
        if current_output_transformation_matrix is None:
            self.set_output_transformation_identity()
        else:
            self.output_transformation_matrix.setRowCount(current_output_transformation_matrix.shape[0])
            for row_idx,row in enumerate(current_output_transformation_matrix):
                for col_idx,col in enumerate(row):
                    try:
                        self.output_transformation_matrix.item(row_idx,col_idx).setText(str(col))
                    except AttributeError:
                        item = QtWidgets.QTableWidgetItem(str(col))
                        self.output_transformation_matrix.setItem(row_idx,col_idx,item)
            
        # Callbacks
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        self.response_transformation_add_row_button.clicked.connect(self.response_transformation_add_row)
        self.response_transformation_remove_row_button.clicked.connect(self.response_transformation_remove_row)
        self.response_transformation_save_matrix_button.clicked.connect(self.save_response_transformation_matrix)
        self.response_transformation_load_matrix_button.clicked.connect(self.load_response_transformation_matrix)
        self.response_transformation_identity_button.clicked.connect(self.set_response_transformation_identity)
        self.response_transformation_6dof_kinematic_button.clicked.connect(self.set_response_transformation_6dof)
        self.response_transformation_reversed_6dof_kinematic_button.clicked.connect(self.set_response_transformation_6dof_reversed)
        
        self.output_transformation_add_row_button.clicked.connect(self.output_transformation_add_row)
        self.output_transformation_remove_row_button.clicked.connect(self.output_transformation_remove_row)
        self.output_transformation_save_matrix_button.clicked.connect(self.save_output_transformation_matrix)
        self.output_transformation_load_matrix_button.clicked.connect(self.load_output_transformation_matrix)
        self.output_transformation_identity_button.clicked.connect(self.set_output_transformation_identity)
        self.output_transformation_6dof_kinematic_button.clicked.connect(self.set_output_transformation_6dof)
        self.output_transformation_reversed_6dof_kinematic_button.clicked.connect(self.set_output_transformation_6dof_reversed)
        
    @staticmethod
    def define_transformation_matrices(current_response_transformation_matrix,num_responses,
                 current_output_transformation_matrix,num_outputs,parent=None):
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
        dialog = TransformationMatrixWindow(parent,
                 current_response_transformation_matrix,num_responses,
                 current_output_transformation_matrix,num_outputs)
        result = dialog.exec_()==QtWidgets.QDialog.Accepted
        response_transformation = np.array([[float(val) for val in row] 
            for row in get_table_strings(dialog.response_transformation_matrix)])
        if all(val == response_transformation.shape[0] for val in response_transformation.shape) and np.allclose(response_transformation,np.eye(response_transformation.shape[0])):
            response_transformation = None
        output_transformation = np.array([[float(val) for val in row]
            for row in get_table_strings(dialog.output_transformation_matrix)])
        if all(val == output_transformation.shape[0] for val in output_transformation.shape) and np.allclose(output_transformation,np.eye(output_transformation.shape[0])):
            output_transformation = None
        return (response_transformation,output_transformation,result)

    def response_transformation_add_row(self):
        """Adds a row to the response transformation"""
        num_rows = self.response_transformation_matrix.rowCount()
        self.response_transformation_matrix.insertRow(num_rows)
        for col_idx in range(self.response_transformation_matrix.columnCount()):
            item = QtWidgets.QTableWidgetItem('0.0')
            self.response_transformation_matrix.setItem(num_rows,col_idx,item)
    def response_transformation_remove_row(self):
        """Removes a row from the response transformation"""
        num_rows = self.response_transformation_matrix.rowCount()
        self.response_transformation_matrix.removeRow(num_rows-1)
    def save_response_transformation_matrix(self):
        """Saves the response transformation matrix to a csv file"""
        string_array = self.get_table_strings(self.response_transformation_matrix)
        filename,file_filter = QtWidgets.QFileDialog.getSaveFileName(self,'Save Response Transformation',filter='Comma-separated Values (*.csv)')
        if filename == '':
            return
        save_csv_matrix(string_array,filename)
    def load_response_transformation_matrix(self):
        """Loads the response transformation from a csv file"""
        filename,file_filter = QtWidgets.QFileDialog.getOpenFileName(self,'Load Response Transformation',filter='Comma-separated values (*.csv *.txt);;Numpy Files (*.npy *.npz);;Matlab Files (*.mat)')
        if filename == '':
            return
        file_base,extension = os.path.splitext(filename)
        string_array = None
        if extension.lower() == '.npy':
            string_array = np.load(filename).astype('U')
        elif extension.lower() == '.npz':
            data = np.load(filename)
            for key,array in data.items():
                string_array = array.astype('U')
                break
        elif extension.lower() == '.mat':
            data = loadmat(filename)
            for key,array in data.items():
                if '__' in key:
                    continue
                string_array = array.astype('U')
                break
        else:
            string_array = load_csv_matrix(filename)
        if string_array is None:
            return
        # Set the number of rows
        self.response_transformation_matrix.setRowCount(len(string_array))
        num_rows = self.response_transformation_matrix.rowCount()
        num_cols = self.response_transformation_matrix.columnCount()
        for row_idx,row in enumerate(string_array):
            if row_idx == num_rows:
                break
            for col_idx,value in enumerate(row):
                if col_idx == num_cols:
                    break
                try:
                    self.response_transformation_matrix.item(row_idx,col_idx).setText(value)
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(value)
                    self.response_transformation_matrix.setItem(row_idx,col_idx,item)
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
                    self.response_transformation_matrix.item(row_idx,col_idx).setText(str(value))
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.response_transformation_matrix.setItem(row_idx,col_idx,item)
    def set_response_transformation_6dof(self):
        """Sets the response transformation matrix to the 6DoF table"""
        num_columns = self.response_transformation_matrix.columnCount()
        if num_columns != 12:
            error_message_qt('Invalid Number of Control Channels.','Invalid Number of Control Channels.  6DoF Transform assumes 12 control accelerometer channels.')
            return
        self.response_transformation_matrix.setRowCount(6)
        matrix = [[0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0],
                   [ 0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0],
                   [ 0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25],
                   [ 0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,-0.25,0.0,0.0,-0.25],
                   [ 0.0,0.0,-0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,-0.25],
                   [ -0.125,0.125,0.0,-0.125,-0.125,0.0,0.125,-0.125,0.0,0.125,0.125,0.0]]
        for row_idx,row in enumerate(matrix):
            for col_idx,value in enumerate(row):
                try:
                    self.response_transformation_matrix.item(row_idx,col_idx).setText(str(value))
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.response_transformation_matrix.setItem(row_idx,col_idx,item)           
    def set_response_transformation_6dof_reversed(self):
        """Sets the response transformation matrix to the 6DoF table"""
        num_columns = self.response_transformation_matrix.columnCount()
        if num_columns != 12:
            error_message_qt('Invalid Number of Control Channels.','Invalid Number of Control Channels.  6DoF Transform assumes 12 control accelerometer channels.')
            return  
        self.response_transformation_matrix.setRowCount(6)
        matrix = [[0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0],
                   [ 0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0],
                   [ 0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25],
                   [ 0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,-0.25,0.0,0.0,-0.25],
                   [ 0.0,0.0,-0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,-0.25],
                   [ -0.125,0.125,0.0,-0.125,-0.125,0.0,0.125,-0.125,0.0,0.125,0.125,0.0]]
        for row_idx,row in enumerate(matrix):
            for col_idx,value in enumerate(row):
                try:
                    self.response_transformation_matrix.item(row_idx,col_idx).setText(str(value))
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.response_transformation_matrix.setItem(row_idx,col_idx,item)
        
    def output_transformation_add_row(self):
        """Adds a row to the output transformation"""
        num_rows = self.output_transformation_matrix.rowCount()
        self.output_transformation_matrix.insertRow(num_rows)
        for col_idx in range(self.output_transformation_matrix.columnCount()):
            item = QtWidgets.QTableWidgetItem('0.0')
            self.output_transformation_matrix.setItem(num_rows,col_idx,item)
    def output_transformation_remove_row(self):
        """Removes a row from the output tranformation"""
        num_rows = self.output_transformation_matrix.rowCount()
        self.output_transformation_matrix.removeRow(num_rows-1)
    def save_output_transformation_matrix(self):
        """Saves output transformation matrix to a CSV file"""
        string_array = self.get_table_strings(self.output_transformation_matrix)
        filename,file_filter = QtWidgets.QFileDialog.getSaveFileName(self,'Save Output Transformation',filter='Comma-separated Values (*.csv)')
        if filename == '':
            return
        save_csv_matrix(string_array,filename)
    def load_output_transformation_matrix(self):
        """Loads the output transformation from a CSV file"""
        filename,file_filter = QtWidgets.QFileDialog.getOpenFileName(self,'Load Output Transformation',filter='Comma-separated values (*.csv *.txt);;Numpy Files (*.npy *.npz);;Matlab Files (*.mat)')
        if filename == '':
            return
        file_base,extension = os.path.splitext(filename)
        string_array = None
        if extension.lower() == '.npy':
            string_array = np.load(filename).astype('U')
        elif extension.lower() == '.npz':
            data = np.load(filename)
            for key,array in data.items():
                string_array = array.astype('U')
                break
        elif extension.lower() == '.mat':
            data = loadmat(filename)
            for key,array in data.items():
                if '__' in key:
                    continue
                string_array = array.astype('U')
                break
        else:
            string_array = load_csv_matrix(filename)
        if string_array is None:
            return
        # Set the number of rows
        self.output_transformation_matrix.setRowCount(len(string_array))
        num_rows = self.output_transformation_matrix.rowCount()
        num_cols = self.output_transformation_matrix.columnCount()
        for row_idx,row in enumerate(string_array):
            if row_idx == num_rows:
                break
            for col_idx,value in enumerate(row):
                if col_idx == num_cols:
                    break
                try:
                    self.output_transformation_matrix.item(row_idx,col_idx).setText(value)
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(value)
                    self.output_transformation_matrix.setItem(row_idx,col_idx,item)
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
                    self.output_transformation_matrix.item(row_idx,col_idx).setText(str(value))
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.output_transformation_matrix.setItem(row_idx,col_idx,item)
    def set_output_transformation_6dof(self):
        """Sets the output transformation matrix to the 6DoF table"""
        num_columns = self.output_transformation_matrix.columnCount()
        if num_columns != 12:
            error_message_qt('Invalid Number of Output Signals.','Invalid Number of Output Signals.  6DoF Transform assumes 12 drive channels.')
            return  
        self.output_transformation_matrix.setRowCount(6)
        matrix = [[0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0],
                   [ 0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0],
                   [ 0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25],
                   [ 0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,-0.25,0.0,0.0,-0.25],
                   [ 0.0,0.0,-0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,-0.25],
                   [ -0.125,0.125,0.0,-0.125,-0.125,0.0,0.125,-0.125,0.0,0.125,0.125,0.0]]
        for row_idx,row in enumerate(matrix):
            for col_idx,value in enumerate(row):
                try:
                    self.output_transformation_matrix.item(row_idx,col_idx).setText(str(value))
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.output_transformation_matrix.setItem(row_idx,col_idx,item)
                    
    def set_output_transformation_6dof_reversed(self):
        """Sets the output transformation matrix to the 6DoF table"""
        num_columns = self.output_transformation_matrix.columnCount()
        if num_columns != 12:
            error_message_qt('Invalid Number of Output Signals.','Invalid Number of Output Signals.  6DoF Transform assumes 12 drive channels.')
            return
        self.output_transformation_matrix.setRowCount(6)
        matrix = [[0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0],
                   [ 0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0],
                   [ 0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,0.25],
                   [ 0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,-0.25,0.0,0.0,-0.25],
                   [ 0.0,0.0,-0.25,0.0,0.0,0.25,0.0,0.0,0.25,0.0,0.0,-0.25],
                   [ -0.125,0.125,0.0,-0.125,-0.125,0.0,0.125,-0.125,0.0,0.125,0.125,0.0]]
        for row_idx,row in enumerate(matrix):
            for col_idx,value in enumerate(row):
                try:
                    self.output_transformation_matrix.item(row_idx,col_idx).setText(str(value))
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.output_transformation_matrix.setItem(row_idx,col_idx,item)