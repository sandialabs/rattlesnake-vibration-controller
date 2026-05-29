"""
Rattlesnake Selectors Testing

This code unit tests the Control Selector and Combined Environments Selector
which are the first user interface components that Rattlesnake runs. The tests
assume Rattlesnake is being called without arguments else the main will skip
over these elements

The following code is tested:
environments.py
- ControlTypes
ui_utilities.py
- ControlSelect.__init__
- ControlSelect.select_control
- EnvironmentSelect.__init__
- EnvironmentSelect.add_environment
- EnvironmentSelect.remove_environment
- EnvironmentSelect.save_profile_template
- EnvironmentSelect.select_environment (Manual and Args)
- save_combined_environments_profile_template
"""

import sys
from unittest import mock

# import numpy as np  # unused import
# import pyqtgraph as pg  # unused import
import pytest
from functions.common_functions import DummyMainWindow
from qtpy import QtWidgets

# from rattlesnake.components.environments import ControlTypes  # unused import
from rattlesnake.user_interface.ui_utilities import (
    ChannelMonitor,
    ControlSelect,
    PlotWindow,
    ProfileTimer,
    multiline_plotter,
)
from rattlesnake.user_interface.ui_registry import (
    EnvironmentSelect,
    save_combined_environments_profile_template,
)
from rattlesnake.utilities import Channel, DataAcquisitionParameters


# Initialize an QApplication which is required for all other widgets
@pytest.fixture
def app(qtbot):
    return qtbot


@pytest.fixture
def main_window(app):
    main_window = DummyMainWindow()
    app.addWidget(main_window)

    return main_window


# Create a ControlSelect object
@pytest.fixture
def control_select(app):
    control_select = ControlSelect()
    app.addWidget(control_select)

    return control_select


# Create an EnvironmentSelect object
@pytest.fixture
def environment_select(app):
    environment_select = EnvironmentSelect()
    app.addWidget(environment_select)

    return environment_select


@pytest.fixture
def channel_list():
    return [
        Channel.from_channel_table_row(
            [
                "221",
                "Y+",
                "",
                "19644",
                "X+",
                "",
                "",
                "",
                "",
                "",
                "Virtual",
                "",
                "Accel",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        ),
        Channel.from_channel_table_row(
            [
                "221",
                "Y+",
                "",
                "19644",
                "X+",
                "",
                "",
                "",
                "",
                "",
                "Virtual",
                "",
                "Force",
                "",
                "",
                "",
                "",
                "",
                "Phys_dev",
                "",
                "5",
                "10",
            ]
        ),
    ]


@pytest.fixture()
def data_acquisition_parameters(channel_list):
    sample_rate = 2000
    time_per_read = 0.25
    time_per_write = 0.25
    output_oversample = 2
    hardware_selector_idx = 6
    hardware_file = "ExampleFile.nc4"
    environments = ["Modal"]
    environment_booleans = [[True]]
    acquisition_processes = 1
    data_acquisition_parameters = DataAcquisitionParameters(
        channel_list,
        sample_rate,
        round(sample_rate * time_per_read),
        round(sample_rate * time_per_write * output_oversample),
        hardware_selector_idx,
        hardware_file,
        environments,
        environment_booleans,
        output_oversample,
        acquisition_processes,
    )

    return data_acquisition_parameters


@pytest.fixture
def channel_monitor(app, data_acquisition_parameters):
    with mock.patch("rattlesnake.user_interface.ui_utilities.QtWidgets.QDialog.show") as mock_show:
        channel_monitor = ChannelMonitor(None, data_acquisition_parameters)
        app.addWidget(channel_monitor)

    return channel_monitor


def test_profile_timer_init():
    profile_timer = ProfileTimer("Environment", "Operation", "Data")

    assert isinstance(profile_timer, ProfileTimer)


# def test_multiline_plotter_init(main_window):
#     data = multiline_plotter((0,1),
#                 np.zeros((len([0]),2)),
#                 widget = main_window.plot_widget,
#                 other_pen_options = {'width':1},
#                 names = ['Output {:}'.format(i+1) for i in range(len([0]))])

#     assert isinstance(data[0],pg.graphicsItems.PlotDataItem.PlotDataItem)


# Test if environment_select is an EnvironmentSelect object
# def test_environment_select_init(app):
#     environment_select = EnvironmentSelect()
#     app.add_widget(environment_select)

#     assert isinstance(environment_select, EnvironmentSelect)


# # Test the EnvrionmentSelect add environment button
# def test_add_environment(environment_select):
#     environment_select.add_environment()
#     data = environment_select.environment_display_table.cellWidget(0, 0)

#     # Test if added row contains combobox
#     assert isinstance(data, QtWidgets.QComboBox)


# # Test the EnvrionmentSelect remove environment button
# # Loop through removed rows
# @pytest.mark.parametrize("remove_row", [2])
# # Prevent table from removing a row
# @mock.patch("rattlesnake.components.ui_utilities.QtWidgets.QTableWidget.removeRow")
# # Mock the return value of the current selected row
# @mock.patch("rattlesnake.components.ui_utilities.QtWidgets.QTableWidget.currentRow")
# def test_remove_environment(mock_row, mock_remove, remove_row, environment_select):
#     # Mock the current selected row
#     mock_row.return_value = remove_row

#     environment_select.remove_environment()

#     # Test if row was removed
#     mock_remove.assert_called_with(remove_row)


# # Test the EnvrionmentSelect save profile button
# # Loop through chosen file names
# @pytest.mark.parametrize("filename, file_filter",
#                          [
#                              ("EnvironmentSelect.xlsx", "Excel File (*.xlsx)"),
#                              ("", "")
#                          ])
# # Prevent saving file to directory
# @mock.patch("rattlesnake.components.ui_utilities.save_combined_environments_profile_template")
# # Prevent windows from opening the save file dialog
# @mock.patch("rattlesnake.components.ui_utilities.QtWidgets.QFileDialog.getSaveFileName")
# def test_save_profile(mock_file, mock_save,  filename, file_filter, environment_select):
#     # Automatically give windows a filename and filter to save file
#     mock_file.return_value = (filename, file_filter)
#     environment_select.save_profile_template()

#     # Test if file was not saved when cancel was clicked
#     if filename == '':
#         mock_save.assert_not_called()
#     # Test if file was saved with correct filename
#     else:
#         mock_save.assert_called_with(filename, [])


# # Test the EnvironmentSelect save file function
# # Prevent openpyxl from saving a file
# @mock.patch("openpyxl.Workbook.save")
# def test_save_combined_environments_profile_template(mock_save):
#     filename = "EnvironmentSelect.xlsx"
#     environment_data = [(ControlTypes(1), 'Random'), (ControlTypes(4), 'Time')]
#     save_combined_environments_profile_template(filename, environment_data)

#     # Test if saved correct filename
#     mock_save.assert_called_with(filename)


# # Test the EnvironmentSelect load profile button
# # Loop through different filenames to open
# @pytest.mark.parametrize("filename, file_filter",
#                          [
#                              ("EnvironmentSelect.xlsx", "Excel File (*.xlsx)"),
#                              ("", "")
#                          ])
# # Mock to see if the program closed
# @mock.patch("rattlesnake.components.ui_utilities.EnvironmentSelect.accept")
# # Prevent windows from opening the get file dialog
# @mock.patch("rattlesnake.components.ui_utilities.QtWidgets.QFileDialog.getOpenFileName")
# def test_load_profile(mock_file, mock_accept, filename, file_filter, environment_select):
#     # Automatically give windows a filename and filter for the get file dialog
#     mock_file.return_value = (filename, file_filter)
#     environment_select.load_profile()

#     # If there is no file, make sure program doesnt accept file
#     if filename == '':
#         mock_accept.assert_not_called()
#     # If there is a file, make sure program accepts it
#     else:
#         mock_accept.assert_called()


# # Test EnvironmentSelect main function
# # Loop through added control types, ('Ok','Cancel') button, and environment names
# @pytest.mark.parametrize("exit_option, control_type, environment_name",
#                          [
#                              (QtWidgets.QDialog.Accepted, 0, "Name"),
#                              (QtWidgets.QDialog.Accepted, 1, "Name"),
#                              (QtWidgets.QDialog.Accepted, 2, ""),
#                              (QtWidgets.QDialog.Accepted, 4, ""),
#                              (QtWidgets.QDialog.Accepted, 6, "Name"),
#                              (QtWidgets.QDialog.Rejected, 1, "Name"),
#                              (QtWidgets.QDialog.Rejected, 2, "Name")
#                          ])
# # Mock response to pulling environment name
# @mock.patch("rattlesnake.components.ui_utilities.QtWidgets.QTableWidget.item")
# # Mock response to pulling control type
# @mock.patch("rattlesnake.components.ui_utilities.QtWidgets.QTableWidget.cellWidget")
# # Mock response to pulling number of rows
# @mock.patch("rattlesnake.components.ui_utilities.QtWidgets.QTableWidget.rowCount")
# # Prevent dialog box from executing
# @mock.patch("rattlesnake.components.ui_utilities.QtWidgets.QDialog.exec_")
# def test_environment_select(mock_exit, mock_row, mock_combobox, mock_name, exit_option, control_type, environment_name, environment_select):
#     # Mock response to pulling control type with a combobox
#     combobox = QtWidgets.QComboBox()
#     combobox.addItem("Control_Name", control_type)
#     mock_combobox.return_value = combobox
#     # Mock response to pulling environment name with table item
#     environment_item = QtWidgets.QTableWidgetItem(environment_name)
#     mock_name.return_value = environment_item
#     mock_row.return_value = 1
#     # Mock response to exit button
#     mock_exit.return_value = exit_option

#     result, environment_table = environment_select.select_environment()

#     # Test if correct exit option was used
#     assert result == exit_option
#     # If 'Cancel' was clicked make sure table is empty
#     if exit_option == 0:
#         assert environment_table == []
#     # If 'Ok' was clicked make sure environment was stored
#     else:
#         assert environment_table[0][0] == ControlTypes(control_type)
#         assert environment_table[0][1] == environment_name


# # Test load profile button
# # Mock environment object so that correct filename can be used
# @mock.patch("rattlesnake.components.ui_utilities.EnvironmentSelect",)
# # Prevent environment from executing
# @mock.patch("rattlesnake.components.ui_utilities.QtWidgets.QDialog.exec_")
# def test_load_environment_select(mock_exit, mock_environment, environment_select):
#     # Store filename to mock EnvironmentSelect object
#     filename = "tests\\TemplateFiles\\EnvironmentSelectTemplate.xlsx"
#     environment_select.loaded_profile = filename
#     mock_environment.return_value = environment_select
#     # 'Ok' button was clicked
#     mock_exit.return_value = QtWidgets.QDialog.Accepted

#     result, environment_table, loaded_profile = environment_select.select_environment()

#     # Test if profile was loaded
#     assert result == -1
#     # Test if environment was stored
#     assert environment_table[0][0] == ControlTypes(1)
#     assert environment_table[0][1] == "RandomEnv"
#     # Test if filename was stored to Ui
#     assert loaded_profile == filename


# # Test the control_select dialog box
# # Loop through exit options (Ok,Cancel) and selected control types
# @pytest.mark.parametrize("exit_option, control_value",
#                          [
#                              (QtWidgets.QDialog.Accepted, 0),
#                              (QtWidgets.QDialog.Accepted, 1),
#                              (QtWidgets.QDialog.Accepted, 2),
#                              (QtWidgets.QDialog.Accepted, 4),
#                              (QtWidgets.QDialog.Accepted, 6),
#                              (QtWidgets.QDialog.Rejected, 1),
#                              (QtWidgets.QDialog.Rejected, 2)
#                          ])
# # Prevent the selector from reading selected control type
# @mock.patch("rattlesnake.components.ui_utilities.QtWidgets.QButtonGroup.checkedId")
# # Prevent the dialog box from rendering
# @mock.patch("rattlesnake.components.ui_utilities.QtWidgets.QDialog.exec_")
# def test_control_select(mock_exit, mock_sel, exit_option, control_value, control_select):
#     # Mock response to exit button and selected control type
#     mock_exit.return_value = exit_option
#     mock_sel.return_value = control_value

#     control_type, close_flag = control_select.select_control()

#     # Test if correct control_type was returned
#     assert control_type.value == control_value
#     # Test if correct exit option was returned
#     assert_exit = [False, True]
#     assert close_flag == assert_exit[exit_option]


# @mock.patch("rattlesnake.components.ui_utilities.QtWidgets.QDialog.show")
# def test_channel_monitor_init(mock_show, app, data_acquisition_parameters):
#     channel_monitor = ChannelMonitor(None,data_acquisition_parameters)
#     app.addWidget(channel_monitor)

#     assert isinstance(channel_monitor, ChannelMonitor)


# @mock.patch("rattlesnake.components.ui_utilities.pyqtgraph.graphicsItems.PlotItem.PlotItem.addItem")
# def test_channel_monitor_build_plot(mock_add, channel_monitor):
#     channel_monitor.build_plot()

#     mock_add.assert_called()


# @mock.patch("rattlesnake.components.ui_utilities.ChannelMonitor.build_plot")
# def test_channel_monitor_update_channel_list(mock_plot, channel_monitor, data_acquisition_parameters):
#     channel_list = [Channel.from_channel_table_row(['230', 'Y+', '', '19644', 'X+', '', '',
#                                                     '', '', '', 'Virtual', '', 'Accel', '',
#                                                     '', '', '', '', '', '', '', ''])]
#     daq_settings = data_acquisition_parameters
#     daq_settings.sample_rate = 100
#     daq_settings.samples_per_read = 5
#     daq_settings.channel_list = channel_list

#     channel_monitor.update_channel_list(daq_settings)

#     assert channel_monitor.channels == channel_list
#     assert channel_monitor.history_hold_frames == 200


# @mock.patch("rattlesnake.components.ui_utilities.pyqtgraph.BarGraphItem.setOpts")
# def test_channel_monitor_clear_alerts(mock_opts, channel_monitor):
#     channel_monitor.clear_alerts()

#     mock_opts.assert_called()


# @mock.patch("rattlesnake.components.ui_utilities.pyqtgraph.BarGraphItem.setOpts")
# def test_channel_monitor_update(mock_opts, channel_monitor):
#     channel_levels = [20, 30]
#     channel_monitor.update(channel_levels)

#     mock_opts.assert_called()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    environment_select = EnvironmentSelect()
    # environment_select_template = EnvironmentSelect()
    # environment_select_template.loaded_profile = "EnvironmentSelectTemplate.xlsx"
    # test_control_selector(
    #     exit_option=QtWidgets.QDialog.Accepted, control_value=2,app=app)
    # test_save_profile(app=app)
    # test_environment_select(exit_option=QtWidgets.QDialog.Rejected,control_type=1,environment_name="a",environment_select=environment_select)
    # test_load_environment_select(environment_select=environment_select)

    channel_list = [
        Channel.from_channel_table_row(
            [
                "221",
                "Y+",
                "",
                "19644",
                "X+",
                "",
                "",
                "",
                "",
                "",
                "Virtual",
                "",
                "Accel",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        ),
        Channel.from_channel_table_row(
            [
                "221",
                "Y+",
                "",
                "19644",
                "X+",
                "",
                "",
                "",
                "",
                "",
                "Virtual",
                "",
                "Force",
                "",
                "",
                "",
                "",
                "",
                "Phys_dev",
                "",
                "5",
                "10",
            ]
        ),
    ]

    sample_rate = 2000
    time_per_read = 0.25
    time_per_write = 0.25
    output_oversample = 2
    hardware_selector_idx = 6
    hardware_file = "ExampleFile.nc4"
    environments = ["Modal"]
    environment_booleans = [[True]]
    acquisition_processes = 1
    data_acquisition_parameters = DataAcquisitionParameters(
        channel_list,
        sample_rate,
        round(sample_rate * time_per_read),
        round(sample_rate * time_per_write * output_oversample),
        hardware_selector_idx,
        hardware_file,
        environments,
        environment_booleans,
        output_oversample,
        acquisition_processes,
    )

    channel_monitor = ChannelMonitor(None, data_acquisition_parameters)
    # test_channel_monitor_build_plot(channel_monitor=channel_monitor)
    main_window = DummyMainWindow()
    test_multiline_plotter_init(main_window)
    # test_load_profile(filename="EnvironmentSelect.xlsx",
    #                   file_filter="Excel File (*.xlsx)", environment_select=environment_select)
    pass
