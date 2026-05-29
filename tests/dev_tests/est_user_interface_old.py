# """
# Data Acquisition Setup Testing

# This code unit tests the Data Acquisition Setup tab in Rattlesnake. The
# callbacks for this tab are located in the user_interface.py file. These
# unit tests are ran in a Modal environment.

# The tests include:
# user_interface.py
# - Ui.__init__
# - Ui.stop_program
# - Ui.load_channel_table
# - Ui.save_channel_table
# - Ui.hardware_update
# - Ui.sample_rate_update
# - Ui.initialize_data_acquisition
# - Ui.get_channel_table_strings
# - DataAcquisitionParameters.__init__

# ui_utilities.py
# - get_table_bools
# - get_table_strings

# utilities.py
# - Channel.__init__
# - Channel.from_channel_row
# """
# from rattlesnake.components.ui_utilities import get_table_bools, get_table_strings
# from rattlesnake.components.user_interface import Ui
# from rattlesnake.components.utilities import VerboseMessageQueue, QueueContainer, Channel, DataAcquisitionParameters, GlobalCommands
# from rattlesnake.components.environments import ControlTypes
# from functions.common_functions import fake_time
# from unittest import mock
# from PyQt5 import QtWidgets
# import multiprocessing as mp
# import pytest
# import sys


# # Create QApplication, required to generate a QMainWindow
# @pytest.fixture
# def app(qtbot):
#     return qtbot


# # Define a Modal environment for the Ui
# @pytest.fixture
# def environments():
#     control_type = ControlTypes(6)
#     return [[control_type, control_type.name.title()]]


# # Create log_file_queue for Ui initialization
# @pytest.fixture
# def log_file_queue():
#     return mp.Queue()


# # Create a QueueContainer for the Ui
# @pytest.fixture
# def queue_container(log_file_queue):
#     queue_container = QueueContainer(VerboseMessageQueue(log_file_queue, 'Controller Communication Queue'),
#                                      VerboseMessageQueue(
#                                          log_file_queue, 'Acquisition Command Queue'),
#                                      VerboseMessageQueue(
#                                          log_file_queue, 'Output Command Queue'),
#                                      VerboseMessageQueue(
#                                          log_file_queue, 'Streaming Command Queue'),
#                                      log_file_queue,
#                                      mp.Queue(),
#                                      mp.Queue(),
#                                      mp.Queue(),
#                                      {'Modal': VerboseMessageQueue(
#                                          log_file_queue, 'Environment Command Queue')},
#                                      {'Modal': mp.Queue()},
#                                      {'Modal': mp.Queue()})
#     return queue_container


# # Generate a multiprocessing value for the Ui
# @pytest.fixture
# def output_active():
#     return mp.Value('i', 0)


# # Create an Ui object
# @pytest.fixture
# def window(app, environments, queue_container):
#     # Stop the window from rendering
#     with mock.patch("rattlesnake.components.user_interface.QtWidgets.QMainWindow.show") as mock_show:
#         window = Ui(environments, queue_container, None)
#         window.close()
#     return window


# # Generate an example channel_table_string
# @pytest.fixture
# def channel_table_strings():
#     channel_table_strings = [['221', 'Y+', '', '19644', 'X+', '', '', '', '', '', 'Virtual', '', 'Accel', '', '', '', '', '', '', '', '', ''],
#                              ['221', 'X-', '', '19644', 'Y+', '', '', '', '', '',
#                                  'Virtual', '', 'Accel', '', '', '', '', '', '', '', '', ''],
#                              ['221', 'Z+', '', '19644', 'Z+', '', '', '', '', '', 'Virtual', '', 'Accel', '', '', '', '', '', '', '', '', '']]
#     return channel_table_strings


# # Create a Channel object
# @pytest.fixture
# def channel(channel_table_strings):
#     channel_table_row = channel_table_strings[0]
#     return Channel.from_channel_table_row(channel_table_row)


# # Generate an example environment_channels()
# @pytest.fixture
# def environment_channels():
#     return [[True], [True], [True]]


# # Ui initialization test
# # Prevent the window from rendering
# @mock.patch("rattlesnake.components.user_interface.QtWidgets.QMainWindow.show")
# def test_ui_initialization(mock_show, app, environments, queue_container):
#     # Create a window and close it
#     window = Ui(environments, queue_container)
#     app.addWidget(window)

#     # Test if the user interface tried to render
#     mock_show.assert_called()
#     # Test if the window was created
#     assert isinstance(window, Ui)


# # Ui log test
# # Prevent from writing to the log_file_queue
# @mock.patch("rattlesnake.components.user_interface.mp.queues.Queue.put")
# # Change the current date and time
# @mock.patch('rattlesnake.components.user_interface.datetime.datetime')
# def test_ui_log(mock_time, mock_put, window):
#     message = "Test Message"
#     mock_time.now = fake_time

#     window.log(message)

#     # Test if correct string was written to log_file_queue
#     mock_put.assert_called_with(
#         '{:}: {:} -- {:}\n'.format("Datetime", "UI", message))


# # Ui stop_program test
# # Prevent the window from closing a second time
# @mock.patch("rattlesnake.components.user_interface.QtWidgets.QMainWindow.close")
# def test_ui_stop_program(mock_close, app, window):
#     window.stop_program()

#     # Test if the QMainWindow.close() function was run
#     mock_close.assert_called()


# # Test reading from channel table
# def test_ui_get_channel_table_strings(channel_table_strings, app, window):
#     # Set the channel table to a set template
#     for row_idx, row_data in enumerate(channel_table_strings):
#         for col_idx, cell_data in enumerate(row_data):
#             window.channel_table.item(row_idx, col_idx).setText(cell_data)

#     # Store the first 3 rows of channel tale
#     get_channel_table_result = window.get_channel_table_strings()
#     get_channel_table_result = get_channel_table_result[:][0:3]

#     # Test if the read channel table matches the template
#     assert get_channel_table_result == channel_table_strings


# # Test the load channel button on example template
# # Test both terminal and non-terminal load file
# @pytest.mark.parametrize("load_file", [True, False])
# # Prevent file from logging
# @mock.patch("rattlesnake.components.user_interface.Ui.log")
# # Prevent Windows from opening get file dialog
# @mock.patch("rattlesnake.components.user_interface.QtWidgets.QFileDialog.getOpenFileName")
# def test_ui_load_channel_table(mock_file, mock_log, load_file, app, window):
#     # Automatically fill out get file dialog
#     filename = "Tests\\TemplateFiles\\ChannelTableTemplate.xlsx"
#     file_filter = "Excel File (*.xlsx)"
#     mock_file.return_value = (filename, file_filter)

#     # Give a file name if args was given in terminal
#     if load_file:
#         window.load_channel_table(clicked=True, filename=filename)
#     else:
#         window.load_channel_table(clicked=True)

#     # Sample multiple rows and columns to confirm correct inputs
#     assert window.channel_table.item(0, 0).text() == '221'
#     assert window.channel_table.item(1, 1).text() == 'X-'
#     assert window.channel_table.item(2, 3).text() == '19644'
#     assert window.channel_table.item(2, 12).text() == 'Accel'


# # Test the save channel button
# # Prevent Windows from opening save file dialog
# @mock.patch("rattlesnake.components.user_interface.QtWidgets.QFileDialog.getSaveFileName")
# # Prevent openpyxl from saving a workbook
# @mock.patch("openpyxl.Workbook.save")
# def test_ui_save_channel_table(mock_save, mock_file, app, window):
#     # Automatically fill out save file dialog
#     filename = "ExampleFile.xlsx"
#     file_filter = "Excel File (*.xlsx)"
#     mock_file.return_value = (filename, file_filter)

#     window.save_channel_table()

#     # Test workbook was saved under the filename
#     mock_save.assert_called_with(filename)


# # Test the hardware selector dropdown
# # Loop through different hardware selector optoins
# @pytest.mark.parametrize("assert_idx", [0, 1, 2, 4, 5, 6])
# # Prevent Ui.sample_rate_update from being called
# @mock.patch("rattlesnake.components.user_interface.Ui.sample_rate_update")
# # Prevent Windows from opening get file dialog
# @mock.patch("rattlesnake.components.user_interface.QtWidgets.QFileDialog.getOpenFileName")
# def test_ui_hardware_update(mock_file, mock_rate, assert_idx, app, window):
#     # Automatically fill out get file dialog
#     filename = "ExampleFile.xlsx"
#     file_filter = "Excel File (*.xlsx)"
#     mock_file.return_value = (filename, file_filter)

#     window.hardware_selector.setCurrentIndex(assert_idx)
#     hardware_idx = window.hardware_selector.currentIndex()

#     # If hardware selector required file, make sure it was stored
#     if mock_file.called:
#         assert window.hardware_file == filename
#     # Test if hardware index is correct
#     assert hardware_idx == assert_idx


# # Test the sample rate update function for Lan-Xi hardware
# # Loop through sample rates
# @pytest.mark.parametrize("sample_rate",
#                          [
#                              (1500, 1600),
#                              (9000, 8192)
#                          ])
# # Prevent Ui from reading hardware selector
# @mock.patch("rattlesnake.components.user_interface.QtWidgets.QComboBox.currentIndex")
# def test_ui_sample_rate_update(mock_idx, sample_rate, app, window):
#     rate, corrected_rate = sample_rate
#     # Force hardware selector to be Lan-Xi
#     mock_idx.return_value = 2

#     window.sample_rate_selector.setValue(rate)
#     window.sample_rate_update()
#     round_rate = window.sample_rate_selector.value()

#     # Test if sample rate was rounded to its correct value
#     assert round_rate == corrected_rate


# # Test the initialize data acquisition button
# @mock.patch("rattlesnake.components.user_interface.Ui.log")
# # Prevent window from changing to environments tab
# @mock.patch("rattlesnake.components.user_interface.QtWidgets.QTabWidget.setCurrentIndex")
# # Prevent window from reading off environments table
# @mock.patch("rattlesnake.components.ui_utilities.get_table_bools")
# # Prevent window from reading off channel table
# @mock.patch("rattlesnake.components.user_interface.Ui.get_channel_table_strings")
# def test_ui_initialize_data_acquisition(mock_string, mock_bool, mock_tab, mock_log, channel_table_strings, environment_channels, app, window):
#     # Force channel table and environment table to a set template
#     mock_string.return_value = channel_table_strings
#     mock_bool.return_value = environment_channels

#     window.initialize_data_acquisition()

#     # Test if the program tried to change to the environments tab
#     mock_tab.assert_called()


# def test_ui_initialize_environment_parameters(window):
#     mock_ui = mock.MagicMock()
#     mock_ui.initialize_environment.return_value = "Environment Parameters"
#     window.environment_UIs['Modal'] = mock_ui
#     mock_queue = mock.MagicMock()
#     window.queue_container.environment_command_queues['Modal'] = mock_queue
#     window.has_test_predictions = True

#     window.initialize_environment_parameters()

#     mock_ui.initialize_environment.assert_called()
#     assert window.environment_metadata["Modal"] == "Environment Parameters"
#     mock_queue.put.assert_called_with("UI", (GlobalCommands.INITIALIZE_ENVIRONMENT_PARAMETERS, "Environment Parameters"))
#     assert window.rattlesnake_tabs.isTabEnabled(2)
#     assert window.rattlesnake_tabs.isTabEnabled(3)


# @mock.patch("rattlesnake.components.user_interface.QtWidgets.QFileDialog.getSaveFileName")
# def test_ui_select_control_streaming_file(mock_save, window):
#     mock_save.return_value = ("Filename", "File Filter")

#     window.select_control_streaming_file()

#     assert window.streaming_file_display.text() == "Filename"


# @mock.patch("rattlesnake.components.user_interface.Ui.start_streaming")
# @mock.patch("rattlesnake.components.user_interface.Ui.log")
# def test_ui_arm_test(mock_log, mock_start, window):
#     mock_control = mock.MagicMock()
#     window.queue_container.controller_communication_queue = mock_control
#     mock_stream = mock.MagicMock()
#     window.queue_container.streaming_command_queue = mock_stream
#     mock_environment = mock.MagicMock()
#     window.environment_UIs = {"Environment Ui" : mock_environment}
#     window.immediate_streaming_radiobutton.setChecked(True)
#     window.streaming_file_display.setText("Filename")
#     window.global_daq_parameters = "Global Daq Parameters"
#     window.environment_metadata = "Environment Metadata"

#     window.arm_test()

#     mock_log.assert_called_with("Arming Test Hardware")
#     mock_stream.put.assert_called_with("UI", (GlobalCommands.INITIALIZE_STREAMING, ("Filename", "Global Daq Parameters", "Environment Metadata")))
#     mock_control.put.assert_called_with("UI", (GlobalCommands.RUN_HARDWARE, None))
#     mock_environment.disable_system_id_daq_armed.assert_called()
#     mock_start.assert_called()


# @mock.patch("rattlesnake.components.user_interface.Ui.log")
# def test_ui_disarm_test(mock_log, window):
#     mock_control = mock.MagicMock()
#     window.queue_container.controller_communication_queue = mock_control
#     mock_environment = mock.MagicMock()
#     window.environment_UIs = {"Environment Ui" : mock_environment}

#     window.disarm_test()

#     mock_log.assert_called_with('Disarming Test Hardware')
#     mock_environment.stop_control.assert_called()
#     mock_environment.enable_system_id_daq_disarmed.assert_called()
#     assert not window.start_profile_button.isEnabled()


# # @mock.patch("rattlesnake.components.ui_utilities.ProfileTimer")
# # @mock.patch("rattlesnake.components.user_interface.Ui.log")
# # def test_ui_start_profile(mock_log, mock_timer, window):
# #     window.profile_events = (1, "Environment Name", "Operation", "Data")

# #     window.start_profile()

# #     mock_log.assert_called_with('Running Profile')
# #     mock_timer.assert_called_with("Environment Name", "Operation", "Data")
    

# # Test reading from environments table
# # Loop through different environment templates
# @pytest.mark.parametrize("table_bools",
#                          [
#                              [[True], [True], [True], [True]],
#                              [[False], [False], [False], [False]],
#                              [[True], [False], [True], [False]],
#                              [[True], [True], [False], [False]]
#                          ])
# def test_get_table_bools(table_bools, app, window):
#     # Find column with booleans
#     environment_name = 'Modal'
#     environment_table_column = window.environments.index(environment_name)
#     # Store template to column
#     for row_idx, row_data in enumerate(table_bools):
#         window.environment_channels_table.cellWidget(
#             row_idx, environment_table_column).setChecked(row_data[0])

#     # Read first 4 rows from environments table
#     table_bools_result = get_table_bools(window.environment_channels_table)
#     table_bools_result = table_bools_result[0:4]

#     # Test if read environment table matches template
#     assert table_bools_result == table_bools


# # Test the universal read table strings method
# def test_get_table_strings(channel_table_strings, app, window):
#     # Store template to channel table
#     for row_idx, row_data in enumerate(channel_table_strings):
#         for col_idx, cell_data in enumerate(row_data):
#             window.channel_table.item(row_idx, col_idx).setText(cell_data)

#     # Read first 3 rows from channel table
#     table_strings_result = get_table_strings(window.channel_table)
#     table_strings_result = table_strings_result[0:3]

#     # Test if read table matches template
#     assert table_strings_result == channel_table_strings


# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     log_file_queue = mp.Queue()
#     verbose_queue = VerboseMessageQueue(log_file_queue, "VerboseQueue")

#     control_type = ControlTypes(6)
#     environments = [[control_type, control_type.name.title()]]

#     acquisition_command_queue = VerboseMessageQueue(
#         log_file_queue, 'Acquisition Command Queue')
#     output_command_queue = VerboseMessageQueue(
#         log_file_queue, 'Output Command Queue')
#     streaming_command_queue = VerboseMessageQueue(
#         log_file_queue, 'Streaming Command Queue')
#     input_output_sync_queue = mp.Queue()
#     single_process_hardware_queue = mp.Queue()
#     gui_update_queue = mp.Queue()
#     controller_communication_queue = VerboseMessageQueue(
#         log_file_queue, 'Controller Communication Queue')

#     for environment_type, environment_name in environments:
#         environment_command_queues = {}
#         environment_data_in_queues = {}
#         environment_data_out_queues = {}
#         environment_command_queues[environment_name] = VerboseMessageQueue(
#             log_file_queue, environment_name+' Command Queue')
#         environment_data_in_queues[environment_name] = mp.Queue()
#         environment_data_out_queues[environment_name] = mp.Queue()

#     queue_container = QueueContainer(controller_communication_queue,
#                                      acquisition_command_queue,
#                                      output_command_queue,
#                                      streaming_command_queue,
#                                      log_file_queue,
#                                      input_output_sync_queue,
#                                      #                                     environment_sync_queue,
#                                      single_process_hardware_queue,
#                                      gui_update_queue,
#                                      environment_command_queues,
#                                      environment_data_in_queues,
#                                      environment_data_out_queues)

#     output_active = mp.Value('i', 0)
#     with mock.patch("rattlesnake.components.user_interface.QtWidgets.QMainWindow.show") as mock_show:
#         window = Ui(environments, queue_container, None)
#         window.close()
#     # test_save_channel_table(app=app,window=window)
#     # test_hardware_update(app=app,window=window,assert_idx=6)
#     # test_sample_rate_update(app=app, window=window, sample_rate=(9000, 8192))

#     environment_channels = [[True], [True], [True]]
#     channel_table_strings = [['221', 'Y+', '', '19644', 'X+', '', '', '', '', '', 'Virtual', '', 'Accel', '', '', '', '', '', '', '', '', ''],
#                              ['221', 'X-', '', '19644', 'Y+', '', '', '', '', '',
#                                  'Virtual', '', 'Accel', '', '', '', '', '', '', '', '', ''],
#                              ['221', 'Z+', '', '19644', 'Z+', '', '', '', '', '', 'Virtual', '', 'Accel', '', '', '', '', '', '', '', '', '']]
#     channel_table_row = channel_table_strings[0]
#     channel = Channel.from_channel_table_row(channel_table_row)
#     # test_initialize_data_acquisition(environment_channels=environment_channels,channel_table_strings=channel_table_strings,app=app,window=window)
#     # test_get_channel_table_strings(channel_table_strings=channel_table_strings,app=app,window=window)
#     # test_get_table_bools(table_bools=[[True], [False], [True], [False]], app=app, window=window)
#     # test_get_table_strings(channel_table_strings=channel_table_strings,app=app,window=window)
#     # test_channel_initialization(channel_table_strings=channel_table_strings)
#     # test_data_acquisition_parameters_initialization(channel=channel)
#     # test_Ui_log(window=window)
#     test_ui_arm_test(window=window)
    
