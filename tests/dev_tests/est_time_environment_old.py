import multiprocessing as mp

# import os  # unused import
import sys
from unittest import mock

import numpy as np

# import openpyxl  # unused import
import pytest
from functions.common_functions import DummyMainWindow
from qtpy import QtWidgets

from rattlesnake.environment.time_environment import (
    TimeEnvironment,
    TimeParameters,
    TimeQueues,
    time_process,
)
from rattlesnake.user_interface.time_ui import TimeUI
from rattlesnake.user_interface.ui_utilities import UICommands
from rattlesnake.utilities import (
    Channel,
    DataAcquisitionParameters,
    GlobalCommands,
    VerboseMessageQueue,
)


@pytest.fixture
def log_file_queue():
    return mp.Queue()


@pytest.fixture
def channel():
    channel_table_row = [
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
        "Feedback device",
        "",
        "5",
        "10",
    ]
    return Channel.from_channel_table_row(channel_table_row)


@pytest.fixture()
def data_acquisition_parameters(channel):
    channel_list = [channel]
    sample_rate = 2000
    time_per_read = 0.025
    time_per_write = 0.025
    output_oversample = 10
    hardware_selector_idx = 6
    hardware_file = "ExampleFile.nc4"
    environments = ["Environment Name"]
    environment_booleans = np.array([[True]])
    acquisition_processes = 1
    task_trigger = 0
    task_trigger_output_channel = ""
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
        maximum_acquisition_processes=acquisition_processes,
        task_trigger=task_trigger,
        task_trigger_output_channel=task_trigger_output_channel,
    )

    return data_acquisition_parameters


@pytest.fixture
def time_queue(log_file_queue):
    time_queue = TimeQueues(
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Controller Communication Queue"),
        mp.Queue(),
        mp.Queue(),
        log_file_queue,
    )

    return time_queue


@pytest.fixture
def time_parameters():
    return TimeParameters(2000, np.ones((1, 1000)), 0.2)


@pytest.fixture
def app(qtbot):
    return qtbot


@pytest.fixture
def main_window(app):
    main_window = DummyMainWindow()
    app.addWidget(main_window)

    return main_window


@pytest.fixture
def time_ui(app, main_window, log_file_queue):
    time_ui = TimeUI(
        "Environment Name",
        main_window.definition_tabwidget,
        main_window.system_id_tabwidget,
        main_window.test_predictions_tabwidget,
        main_window.test_predictions_tabwidget,
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        VerboseMessageQueue(log_file_queue, "Controller Command Queue"),
        log_file_queue,
    )

    return time_ui


@pytest.fixture
def time_ui_init_data(time_ui, data_acquisition_parameters):
    time_ui.initialize_data_acquisition(data_acquisition_parameters)

    return time_ui


@pytest.fixture
def time_environment(time_queue):
    time_environment = TimeEnvironment(
        "Environment Name", time_queue, mp.Value("i", 0), mp.Value("i", 0)
    )

    return time_environment


def test_time_queues_init(log_file_queue):
    time_queue = TimeQueues(
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Controller Communication Queue"),
        mp.Queue(),
        mp.Queue(),
        log_file_queue,
    )

    assert isinstance(time_queue, TimeQueues)


def test_time_parameters_init():
    time_parameters = TimeParameters(2000, np.ones((1, 1000)), 0.2)

    assert isinstance(time_parameters, TimeParameters)
    assert time_parameters.signal_samples == 1000
    assert time_parameters.output_channels == 1
    assert time_parameters.signal_time == 0.5
    assert time_parameters.cancel_rampdown_samples == 400


def test_time_parameters_store_to_netcdf(time_parameters):
    mock_netcdf = mock.MagicMock()

    time_parameters.store_to_netcdf(mock_netcdf)

    mock_netcdf.createDimension.assert_called_with("signal_samples", 1000)
    mock_netcdf.createVariable.assert_called_with(
        "output_signal", "f8", ("output_channels", "signal_samples")
    )


# def test_time_parameters_from_ui(time_ui, data_acquisition_parameters):
#     time_ui.definition_widget.output_sample_rate_display.setValue(2000)
#     time_ui.definition_widget.cancel_rampdown_selector.setValue(0.2)

#     time_parameters = TimeParameters.from_ui(time_ui)

#     assert time_parameters.sample_rate == 2000
#     assert time_parameters.cancel_rampdown_time == 0.2


# def test_time_ui_init(app, log_file_queue):
#     time_ui = TimeUI("Environment Name",
#                      QtWidgets.QTabWidget(), QtWidgets.QTabWidget(),
#                      QtWidgets.QTabWidget(), QtWidgets.QTabWidget(),
#                      VerboseMessageQueue(log_file_queue, 'Environment Command Queue'),
#                      VerboseMessageQueue(log_file_queue, 'Controller Communication Queue'),
#                      log_file_queue)

#     assert isinstance(time_ui, TimeUI)


# @mock.patch("rattlesnake.environment.time_environment.TimeParameters")
# def test_time_ui_collect_environment_definition_parameters(mock_metadata, time_ui):
#     time_ui.collect_environment_definition_parameters()

#     mock_metadata.from_ui.assert_called_with(time_ui)


# @mock.patch("rattlesnake.environment.time_environment.multiline_plotter")
# @mock.patch("rattlesnake.environment.time_environment.TimeUI.log")
# def test_time_ui_initialize_data_acquisition(mock_log, mock_plot, time_ui, data_acquisition_parameters):
#     time_ui.initialize_data_acquisition(data_acquisition_parameters)

#     mock_log.assert_called_with("Initializing Data Acquisition")
#     assert time_ui.definition_widget.output_channels_display.value() == 1
#     assert time_ui.data_acquisition_parameters == data_acquisition_parameters


# @pytest.mark.parametrize("filename", ["Filename.npy", None])
# @mock.patch("rattlesnake.environment.time_environment.TimeUI.show_signal")
# @mock.patch("rattlesnake.environment.time_environment.rms_time")
# @mock.patch("rattlesnake.environment.time_environment.load_time_history")
# @mock.patch("rattlesnake.environment.time_environment.QtWidgets.QFileDialog.getOpenFileName")
# def test_time_ui_load_signal(mock_file, mock_signal, mock_rms, mock_show, time_ui_init_data, filename):
#     time_ui_init_data.definition_widget.output_sample_rate_display.setValue(2000)
#     mock_file.return_value = (filename, "File filter")
#     mock_signal.return_value = np.ones((1, 1000))
#     mock_rms.return_value = [0.5]

#     time_ui_init_data.load_signal(True, filename)

#     mock_signal.assert_called_with(filename, 2000)
#     mock_rms.assert_called()
#     mock_show.assert_called()


# @pytest.mark.parametrize("checkbox", [True, False])
# def test_time_ui_show_signal(time_ui_init_data, checkbox):
#     mock_plot = mock.MagicMock()
#     mock_checkbox = mock.MagicMock()
#     mock_checkbox.isChecked.return_value = checkbox
#     time_ui_init_data.show_signal_checkboxes = [mock_checkbox]
#     time_ui_init_data.plot_data_items['output_signal_definition'] = [mock_plot]
#     time_ui_init_data.signal = [np.ones((1, 100))]
#     x = np.arange(100)/20000

#     time_ui_init_data.show_signal()

#     if checkbox:
#         np.testing.assert_array_equal(x, mock_plot.setData.call_args[0][0])
#         np.testing.assert_array_equal(np.ones((1, 100)), mock_plot.setData.call_args[0][1])
#     else:
#         np.testing.assert_array_equal((0, 0), mock_plot.setData.call_args[0][0])
#         np.testing.assert_array_equal((0, 0), mock_plot.setData.call_args[0][1])


# @mock.patch("rattlesnake.environment.time_environment.TimeUI.collect_environment_definition_parameters")
# @mock.patch("rattlesnake.environment.time_environment.TimeUI.log")
# def test__time_ui_initialize_enviornment(mock_log, mock_metadata, time_ui_init_data, time_parameters):
#     mock_out_plot = mock.MagicMock()
#     mock_res_plot = mock.MagicMock()
#     time_ui_init_data.plot_data_items['output_signal_measurement'] = [mock_out_plot]
#     time_ui_init_data.plot_data_items['response_signal_measurement'] = [mock_res_plot]
#     mock_metadata.return_value = time_parameters

#     metadata = time_ui_init_data.initialize_environment()

#     mock_log.assert_called_with('Initializing Environment Parameters')
#     np.testing.assert_array_equal(np.arange(200)/2000, mock_out_plot.setData.call_args[0][0])
#     np.testing.assert_array_equal(np.zeros((200,)), mock_out_plot.setData.call_args[0][1])
#     np.testing.assert_array_equal(np.arange(200)/2000, mock_res_plot.setData.call_args[0][0])
#     np.testing.assert_array_equal(np.zeros((200,)), mock_res_plot.setData.call_args[0][1])
#     assert metadata == time_parameters


# @mock.patch("rattlesnake.environment.time_environment.TimeUI.show_signal")
# def test_time_ui_retrieve_metadata(mock_show, time_ui_init_data):
#     mock_netcdf = mock.MagicMock()
#     mock_netcdf.cancel_rampdown_time = 0.2
#     mock_netcdf.variables['output_signal'][...].data = np.ones((1, 100))
#     mock_group_dataset = mock.MagicMock()
#     mock_group_dataset.groups = {"Environment Name" : mock_netcdf}

#     time_ui_init_data.retrieve_metadata(mock_group_dataset)

#     assert time_ui_init_data.definition_widget.cancel_rampdown_selector.value() == 0.2
#     mock_show.assert_called()


# @mock.patch("rattlesnake.components.utilities.VerboseMessageQueue.put")
# def test_time_ui_start_control(mock_put, time_ui):
#     time_ui.start_control()

#     put_calls = [mock.call("Environment Name UI", (GlobalCommands.START_ENVIRONMENT, "Environment Name")),
#                  mock.call("Environment Name UI", (GlobalCommands.START_ENVIRONMENT, (1.0, False))),
#                  mock.call("Environment Name UI", (GlobalCommands.AT_TARGET_LEVEL, "Environment Name"))]
#     mock_put.assert_has_calls(put_calls)
#     assert time_ui.run_widget.stop_test_button.isEnabled()

# @mock.patch("rattlesnake.components.utilities.VerboseMessageQueue.put")
# def test_time_ui_stop_control(mock_put, time_ui):
#     time_ui.stop_control()

#     mock_put.assert_called_with("Environment Name UI", (GlobalCommands.STOP_ENVIRONMENT, None))


# def test_time_ui_change_test_level_from_profile(time_ui):
#     time_ui.change_test_level_from_profile(2)

#     assert time_ui.run_widget.test_level_selector.value() == 2


# def test_time_ui_set_repeat_from_profile(time_ui):
#     time_ui.set_repeat_from_profile(None)

#     assert time_ui.run_widget.repeat_signal_checkbox.isChecked()


# def test_time_ui_set_norepeat_from_profile(time_ui):
#     time_ui.set_norepeat_from_profile(None)

#     assert not time_ui.run_widget.repeat_signal_checkbox.isChecked()


# @pytest.mark.parametrize("message, data", [("time_data", ([np.ones((10))], [np.ones((10))])),
#                                            ("enable", 'test_level_selector'),
#                                            ("disable", 'test_level_selector')])
# def test_time_ui_update_gui(time_ui, message, data):
#     mock_res_data = mock.MagicMock()
#     mock_res_data.getData.return_value = (np.zeros((2)), np.zeros((2)))
#     time_ui.plot_data_items['response_signal_measurement'] = [mock_res_data]
#     mock_out_data = mock.MagicMock()
#     mock_out_data.getData.return_value = (np.zeros((2)), np.zeros((2)))
#     time_ui.plot_data_items['output_signal_measurement'] = [mock_out_data]

#     time_ui.update_gui((message, data))

#     if message == "time_data":
#         np.testing.assert_array_equal(np.zeros((2)),mock_res_data.setData.call_args_list[0][0][0])
#         np.testing.assert_array_equal(np.ones((2)),mock_out_data.setData.call_args_list[0][0][1])
#     elif message == "enable":
#         assert time_ui.run_widget.test_level_selector.isEnabled()
#     elif message == "disable":
#         assert not time_ui.run_widget.test_level_selector.isEnabled()


# def test_time_ui_create_environment_template(time_ui):
#     mock_worksheet = mock.MagicMock()
#     mock_workbook = mock.MagicMock()
#     mock_workbook.create_sheet.return_value = mock_worksheet

#     time_ui.create_environment_template("Environment Name", mock_workbook)

#     worksheet_calls = [mock.call(1,1,'Control Type'),
#                        mock.call(1,2,'Time'),
#                        mock.call(1,4,'Note: Replace cells with hash marks (#) to provide the requested parameters.'),
#                        mock.call(2,1,'Signal File'),
#                        mock.call(2,2,'# Path to the file that contains the time signal that will be output'),
#                        mock.call(3,1,'Cancel Rampdown Time'),
#                        mock.call(3,2,'# Time for the environment to ramp to zero if the environment is cancelled.')]
#     mock_worksheet.cell.assert_has_calls(worksheet_calls)


# @mock.patch("rattlesnake.environment.time_environment.TimeUI.load_signal")
# def test_time_ui_set_parameters_from_template(mock_load, time_ui):
#     filename = os.path.join("tests","TemplateFiles","TimeEnvironmentTemplate.xlsx")
#     workbook = openpyxl.load_workbook(filename)
#     worksheet = workbook['Time']

#     time_ui.set_parameters_from_template(worksheet)

#     mock_load.assert_called_with(None, "Filename.np")
#     time_ui.definition_widget.cancel_rampdown_selector.value() == 0.5


def test_time_environment_init(time_queue):
    time_environment = TimeEnvironment(
        "Environment Name", time_queue, mp.Value("i", 0), mp.Value("i", 0)
    )

    assert isinstance(time_environment, TimeEnvironment)


@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.log")
def test_time_environment_initialize_data_acquisition_parameters(
    mock_log, time_environment, data_acquisition_parameters
):
    time_environment.initialize_data_acquisition_parameters(data_acquisition_parameters)

    mock_log.assert_called_with("Initializing Data Acquisition Parameters")
    assert time_environment.data_acquisition_parameters == data_acquisition_parameters


@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.log")
def test_time_environment_initialize_environment_test_parameters(
    mock_log, time_environment, time_parameters
):
    time_environment.initialize_environment_test_parameters(time_parameters)

    mock_log.assert_called_with("Initializing Environment Parameters")
    assert time_environment.environment_parameters == time_parameters


@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.shutdown")
@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.output")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
@mock.patch("rattlesnake.environment.time_environment.mp.queues.Queue.put")
@mock.patch("rattlesnake.environment.time_environment.mp.queues.Queue.empty")
@mock.patch("rattlesnake.environment.time_environment.mp.queues.Queue.get")
@mock.patch("rattlesnake.environment.time_environment.mp.queues.Queue.get_nowait")
@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.log")
def test_time_environment_run_environment(
    mock_log,
    mock_get_no,
    mock_get,
    mock_empty,
    mock_put,
    mock_vput,
    mock_output,
    mock_shutdown,
    time_environment,
    data_acquisition_parameters,
    time_parameters,
):
    mock_get_no.side_effect = [(np.ones((1, 1200)), False), (np.ones((1, 1200)), False)]
    mock_get.return_value = (np.ones((1, 1200)), True)
    mock_empty.side_effect = [True, True]

    time_environment.data_acquisition_parameters = data_acquisition_parameters
    time_environment.environment_parameters = time_parameters
    time_environment.run_environment((1, True))
    time_environment.current_test_level = 0
    time_environment.run_environment((0, False))

    log_calls = [
        mock.call("Test Level set to 1"),
        mock.call("Waiting for Last Acquisition"),
    ]
    mock_log.assert_has_calls(log_calls)
    mock_shutdown.assert_called()
    np.testing.assert_array_equal(np.ones((1, 500)), mock_output.call_args_list[0][0][0])
    assert mock_output.call_args_list[0][0][1] == False
    np.testing.assert_array_equal(np.ones((1, 1, 1200)), mock_put.call_args_list[0][0][0][1][1][0])
    np.testing.assert_array_equal(np.ones((1, 1, 1200)), mock_put.call_args_list[2][0][0][1][1][0])
    mock_vput.assert_called_with("Environment Name", (GlobalCommands.START_ENVIRONMENT, None))


@pytest.mark.parametrize("test_level_change", [0, -0.001])
@mock.patch("rattlesnake.environment.time_environment.mp.queues.Queue.put")
@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.log")
def test_time_environment_output(mock_log, mock_put, time_environment, test_level_change):
    time_environment.test_level_change = test_level_change
    time_environment.current_test_level = 1
    time_environment.test_level_target = 0.8

    time_environment.output(np.ones((1, 1000)), False)

    if test_level_change == 0:
        log_calls = [
            mock.call("Test Level at 1"),
            mock.call("Sending data to data_out queue"),
        ]
    else:
        log_calls = [
            mock.call("Test level from 0.999 to 0.8"),
            mock.call("Sending data to data_out queue"),
        ]
    mock_log.assert_has_calls(log_calls)
    output_array = 1 + test_level_change + np.arange(1000) * test_level_change
    target_indices = np.where(output_array <= 0.8)
    output_array[target_indices] = 0.8
    output_array = output_array.reshape(1, -1)
    np.testing.assert_array_almost_equal(output_array, mock_put.call_args_list[0][0][0][0])
    assert mock_put.call_args_list[0][0][0][1] == False


@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.adjust_test_level")
def test_time_environment_stop_environment(mock_adjust, time_environment):
    time_environment.stop_environment(None)

    mock_adjust.assert_called_with(0.0)


@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.log")
def test_time_environment_adjust_test_level(mock_log, time_environment, time_parameters):
    time_environment.current_test_level = 1
    time_environment.test_level_target = 0.1
    time_environment.environment_parameters = time_parameters

    time_environment.adjust_test_level(0.8)

    assert time_environment.test_level_target == 0.8
    mock_log.assert_called_with(
        "Changed test level to 0.8 from 1, {:} change per sample".format(
            time_environment.test_level_change
        )
    )


@mock.patch("rattlesnake.environment.time_environment.mp.queues.Queue.put")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.flush")
@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.log")
def test_time_environment_shutdown(mock_log, mock_flush, mock_put, time_environment):
    time_environment.shutdown()

    mock_log.assert_called_with("Shutting Down Time History Generation")
    mock_flush.assert_called_with("Environment Name")
    put_calls = [
        mock.call(("Environment Name", (UICommands.ENABLE, "test_level_selector"))),
        mock.call(("Environment Name", (UICommands.ENABLE, "repeat_signal_checkbox"))),
        mock.call(("Environment Name", (UICommands.ENABLE, "start_test_button"))),
        mock.call(("Environment Name", (UICommands.DISABLE, "stop_test_button"))),
    ]
    mock_put.assert_has_calls(put_calls)
    assert time_environment.startup == True


@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.run")
def test_time_process_function(mock_run, log_file_queue):
    time_process(
        "Environment Name",
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Controller Communication Queue"),
        log_file_queue,
        mp.Queue(),
        mp.Queue(),
        mp.Value("i", 0),
        mp.Value("i", 0),
    )

    mock_run.assert_called()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    log_file_queue = mp.Queue()

    time_queue = TimeQueues(
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Controller Communication Queue"),
        mp.Queue(),
        mp.Queue(),
        log_file_queue,
    )

    channel_table_row = [
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
        "Feedback device",
        "",
        "5",
        "10",
    ]
    channel = Channel.from_channel_table_row(channel_table_row)
    channel_list = [channel]
    sample_rate = 2000
    time_per_read = 0.025
    time_per_write = 0.025
    output_oversample = 10
    hardware_selector_idx = 6
    hardware_file = "ExampleFile.nc4"
    environments = ["Environment Name"]
    environment_booleans = np.array([[True]])
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
    time_parameters = TimeParameters(2000, np.ones((1, 1000)), 0.2)

    main_window = DummyMainWindow()
    time_ui = TimeUI(
        "Environment Name",
        main_window.definition_tabwidget,
        main_window.system_id_tabwidget,
        main_window.test_predictions_tabwidget,
        main_window.test_predictions_tabwidget,
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        VerboseMessageQueue(log_file_queue, "Controller Command Queue"),
        log_file_queue,
    )
    time_ui.initialize_data_acquisition(data_acquisition_parameters)

    time_environment = TimeEnvironment(
        "Environment Name", time_queue, mp.Value("i", 0), mp.Value("i", 0)
    )

    test_time_ui_update_gui(
        time_ui=time_ui, message="time_data", data=([np.ones((10))], [np.ones((10))])
    )
    # test_time_environment_run_environment(time_environment=time_environment, data_acquisition_parameters=data_acquisition_parameters, time_parameters=time_parameters)
    # test_time_environment_output(time_environment=time_environment, test_level_change = -0.001)
