import multiprocessing as mp

# import os  # unused import
import sys
from unittest import mock

import numpy as np

# import openpyxl  # usused import
import pytest

# from functions.common_functions import DummyMainWindow, fake_time  # unused import
from functions.common_functions import DummyMainWindow
from functions.modal_environment_functions import (  # create_environment_template_calls,  # unused import
    create_environment_collector_metadata_calls,
    create_environment_signal_metadata_calls,
    create_environment_spectral_metadata_calls,
    create_environment_start_calls,
    create_modal_commands_dict,
    create_signal_generator_dict,
)
from qtpy import QtWidgets

from rattlesnake.process.data_collector import DataCollectorCommands

# from rattlesnake.components.environments import ControlTypes  # unused import
from rattlesnake.environment.modal_environment import (
    ModalCommands,
    ModalUICommands,
    ModalEnvironment,
    ModalMetadata,
    ModalQueues,
    modal_process,
)
from rattlesnake.user_interface.modal_ui import ModalUI
from rattlesnake.process.signal_generation_process import SignalGenerationCommands
from rattlesnake.process.spectral_processing import SpectralProcessingCommands

# from rattlesnake.components.user_interface import Ui   # unused import
from rattlesnake.utilities import (
    Channel,
    DataAcquisitionParameters,
    GlobalCommands,
    QueueContainer,
    VerboseMessageQueue,
)


@pytest.fixture()
def log_file_queue():
    return mp.Queue


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
        "",
        "",
        "5",
        "10",
    ]
    return Channel.from_channel_table_row(channel_table_row)


@pytest.fixture()
def data_acquisition_parameters(channel):
    channel_list = [channel]
    sample_rate = 2048
    time_per_read = 5
    time_per_write = 5
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
def modal_queue(log_file_queue):
    modal_queue = ModalQueues(
        "Environment Name",
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Controller Communication Queue"),
        mp.Queue(),
        mp.Queue(),
        log_file_queue,
    )

    return modal_queue


@pytest.fixture
def modal_environment(modal_queue):
    modal_environment = ModalEnvironment(
        "Environment_name", modal_queue, mp.Value("i", 0), mp.Value("i", 0)
    )

    return modal_environment


@pytest.fixture
def modal_metadata(data_acquisition_parameters):
    modal_metadata = ModalMetadata(
        2048,
        10,
        "Linear",
        30,
        0.1,
        "H1",
        "rectangle",
        0.0,
        "Free Run",
        "Accept All",
        0.0,
        0,
        0.0,
        True,
        0.1,
        0.2,
        0.4,
        "none",
        0,
        0,
        0,
        0,
        None,
        [],
        [1, 2],
        [3],
        data_acquisition_parameters,
        0.25,
    )

    return modal_metadata


@pytest.fixture
def app(qtbot):
    return qtbot


@pytest.fixture
def main_window(app):
    main_window = DummyMainWindow()
    app.addWidget(main_window)

    return main_window


@pytest.fixture
def modal_ui(app, main_window, log_file_queue):
    modal_ui = ModalUI(
        "Environment Name",
        main_window.definition_tabwidget,
        main_window.system_id_tabwidget,
        main_window.test_predictions_tabwidget,
        main_window.test_predictions_tabwidget,
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        VerboseMessageQueue(log_file_queue, "Controller Command Queue"),
        log_file_queue,
    )

    return modal_ui


@pytest.fixture
def modal_ui_init_data(app, modal_ui, data_acquisition_parameters):
    modal_ui.initialize_data_acquisition(data_acquisition_parameters)

    return modal_ui


@pytest.fixture
def modal_ui_init_env(app, modal_ui_init_data):
    modal_ui_init_data.initialize_environment()

    return modal_ui_init_data


@pytest.fixture
def dummy_netcdf(modal_metadata):
    dummy_netcdf = mock.MagicMock()
    modal_metadata.store_to_netcdf(dummy_netcdf)

    return dummy_netcdf


@pytest.mark.parametrize("modal_idx", [0, 1, 2, 3, 4])
def test_modal_commands(modal_idx):
    modal_command = ModalCommands(modal_idx)
    modal_commands_dict = create_modal_commands_dict()[modal_idx]

    assert isinstance(modal_command, ModalCommands)
    assert modal_command == modal_commands_dict


def test_modal_queues_init(log_file_queue):
    modal_queue = ModalQueues(
        "Environment Name",
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Controller Communication Queue"),
        mp.Queue(),
        mp.Queue(),
        log_file_queue,
    )

    assert isinstance(modal_queue, ModalQueues)


def test_modal_metadata_init(data_acquisition_parameters):
    modal_metadata = ModalMetadata(
        2000,
        10,
        "Linear",
        30,
        0.0,
        "H1",
        "rectangle",
        0.0,
        "Free Run",
        "Accept All",
        0.2,
        0,
        0.1,
        True,
        0.0,
        0.0,
        0.0,
        "none",
        0.0,
        0.0,
        0.0,
        0.0,
        None,
        [],
        [1, 2],
        [3],
        data_acquisition_parameters,
        0.25,
    )

    assert isinstance(modal_metadata, ModalMetadata)
    assert modal_metadata.samples_per_acquire == 10
    assert modal_metadata.frame_time == 0.005
    assert modal_metadata.nyquist_frequency == 1000
    assert modal_metadata.fft_lines == 6
    assert modal_metadata.skip_frames == 40
    assert modal_metadata.frequency_spacing == 200
    modal_metadata.samples_per_output = 2
    assert modal_metadata.disabled_signals == [0]
    assert modal_metadata.hysteresis_samples == 0


# def test_modal_metadata_from_ui(modal_ui, data_acquisition_parameters):
#     modal_ui.initialize_data_acquisition(data_acquisition_parameters)

#     modal_metadata = ModalMetadata.from_ui(modal_ui)

#     assert modal_metadata.output_oversample == modal_ui.data_acquisition_parameters.output_oversample


def test_modal_metadata_get_trigger_levels(modal_metadata, channel):
    channel_list = [channel]
    (t_v, t_eu, h_v, h_eu) = modal_metadata.get_trigger_levels(channel_list)

    assert t_v == 0.01
    assert t_eu == 0.01
    assert h_v == 0.02
    assert h_eu == 0.02


@pytest.mark.parametrize("signal_generator", [None, mock.MagicMock()])
def test_modal_metadata_generate_signal(modal_metadata, signal_generator):
    modal_metadata.signal_generator = signal_generator
    data = modal_metadata.generate_signal()

    if signal_generator == None:
        np.testing.assert_array_equal(data, np.zeros((1, 100)))
    else:
        signal_generator.generate_frame.assert_called()


def test_modal_metadata_store_to_netcdf(modal_metadata):
    mock_netcdf = mock.MagicMock()

    modal_metadata.store_to_netcdf(mock_netcdf)

    mock_netcdf.createDimension.assert_called_with("response_channels", 2)
    mock_netcdf.createVariable.assert_called_with(
        "response_channel_indices", "i4", ("response_channels")
    )


@pytest.mark.parametrize("signal_idx", [1, 2, 3, 4, 5, 6])
def test_modal_metadata_get_signal_generator(modal_metadata, signal_idx):
    signal_generator_dict = create_signal_generator_dict()
    modal_metadata.signal_generator_type = signal_generator_dict[signal_idx][0]

    with mock.patch(signal_generator_dict[signal_idx][1]) as mock_signal:
        modal_metadata.get_signal_generator()

        mock_signal.assert_called()


# def test_modal_ui_init(app, log_file_queue):
#     modal_ui = ModalUI("Environment Name",
#                        QtWidgets.QTabWidget(), QtWidgets.QTabWidget(),
#                        QtWidgets.QTabWidget(), QtWidgets.QTabWidget(),
#                        VerboseMessageQueue(log_file_queue, 'Environment Command Queue'),
#                        VerboseMessageQueue(log_file_queue, 'Controller Communication Queue'),
#                        log_file_queue)

#     assert isinstance(modal_ui, ModalUI)


# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.generate_signal")
# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.collect_environment_definition_parameters")
# def test_modal_ui_update_parameters(mock_collect, mock_signal, modal_ui):
#     mock_metadata = mock.MagicMock()
#     mock_metadata.samples_per_acquire = 100
#     mock_metadata.frame_time = 20
#     mock_metadata.nyquist_frequency = 1000
#     mock_metadata.fft_lines = 4
#     mock_metadata.frequency_spacing = 10
#     mock_collect.return_value = mock_metadata

#     modal_ui.update_parameters()

#     mock_collect.assert_called()
#     mock_signal.assert_called()
#     assert modal_ui.definition_widget.samples_per_acquire_display.value() == 100
#     assert modal_ui.definition_widget.frequency_spacing_display.value() == 10


# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.generate_signal")
# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.output_channel_indices",new_callable=mock.PropertyMock)
# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.reference_indices",new_callable=mock.PropertyMock)
# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.response_indices",new_callable=mock.PropertyMock)
# def test_modal_ui_update_reference_channels(mock_response, mock_reference, mock_output, mock_signal, modal_ui):
#     mock_response.return_value = [0]
#     mock_reference.return_value = [1, 2]
#     mock_output.return_value = [3, 4, 5]

#     modal_ui.update_reference_channels()

#     mock_signal.assert_called()
#     assert modal_ui.definition_widget.response_channels_display.value() == 1
#     assert modal_ui.definition_widget.reference_channels_display.value() == 2
#     assert modal_ui.definition_widget.output_channels_display.value() == 3


# def test_modal_ui_check_selected_reference_channels(modal_ui_init_data):
#     modal_ui_init_data.definition_widget.reference_channels_selector.cellWidget(0,1).setChecked(False)
#     modal_ui_init_data.definition_widget.reference_channels_selector.selectRow(0)

#     modal_ui_init_data.check_selected_reference_channels()

#     assert modal_ui_init_data.definition_widget.reference_channels_selector.cellWidget(0,1).isChecked()


# def test_modal_ui_uncheck_selected_reference_channels(modal_ui_init_data):
#     modal_ui_init_data.definition_widget.reference_channels_selector.cellWidget(0,1).setChecked(True)
#     modal_ui_init_data.definition_widget.reference_channels_selector.selectRow(0)

#     modal_ui_init_data.uncheck_selected_reference_channels()

#     assert not modal_ui_init_data.definition_widget.reference_channels_selector.cellWidget(0,1).isChecked()


# def test_modal_ui_enable_selected_channels(modal_ui_init_data):
#     modal_ui_init_data.definition_widget.reference_channels_selector.cellWidget(0,0).setChecked(False)
#     modal_ui_init_data.definition_widget.reference_channels_selector.selectRow(0)

#     modal_ui_init_data.enable_selected_channels()

#     assert modal_ui_init_data.definition_widget.reference_channels_selector.cellWidget(0,0).isChecked()


# def test_modal_ui_disable_selected_channels(modal_ui_init_data):
#     modal_ui_init_data.definition_widget.reference_channels_selector.cellWidget(0,0).setChecked(True)
#     modal_ui_init_data.definition_widget.reference_channels_selector.selectRow(0)

#     modal_ui_init_data.disable_selected_channels()

#     assert not modal_ui_init_data.definition_widget.reference_channels_selector.cellWidget(0,0).isChecked()


# @pytest.mark.parametrize('trigger_idx', [0, 1])
# def test_modal_ui_actiate_trigger_options(modal_ui, trigger_idx):
#     mock_idx = mock.MagicMock()
#     mock_idx.currentIndex.return_value = trigger_idx
#     modal_ui.definition_widget.triggering_type_selector = mock_idx
#     modal_ui.activate_trigger_options()

#     if trigger_idx == 0:
#         assert not modal_ui.definition_widget.hysteresis_eu_display.isEnabled()
#     else:
#         assert modal_ui.definition_widget.hysteresis_eu_display.isEnabled()


# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.collect_environment_definition_parameters")
# def test_modal_ui_update_trigger_levels(mock_collect, modal_ui, data_acquisition_parameters):
#     t_v = 0.8
#     t_eu = 0.2
#     h_v = 0.1
#     h_eu = 0.5
#     mock_metadata = mock.MagicMock()
#     mock_metadata.get_trigger_levels.return_value = (t_v, t_eu, h_v, h_eu)
#     mock_metadata.trigger_channel = 0
#     mock_collect.return_value = mock_metadata
#     modal_ui.data_acquisition_parameters = data_acquisition_parameters

#     modal_ui.update_trigger_levels()

#     assert modal_ui.definition_widget.trigger_level_voltage_display.value() == t_v
#     assert modal_ui.definition_widget.trigger_level_eu_display.value() == t_eu
#     assert modal_ui.definition_widget.hysteresis_voltage_display.value() == h_v
#     assert modal_ui.definition_widget.hysteresis_eu_display.value() == h_eu


# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.collect_environment_definition_parameters")
# def test_modal_ui_update_hysteresis_length(mock_collect, modal_ui):
#     mock_metadata = mock.MagicMock()
#     mock_metadata.hysteresis_samples = 100
#     mock_metadata.sample_rate = 2000
#     mock_collect.return_value = mock_metadata

#     modal_ui.update_hysteresis_length()

#     assert modal_ui.definition_widget.hysteresis_samples_display.value() == 100
#     assert modal_ui.definition_widget.hysteresis_time_display.value() == 0.05


# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.generate_signal")
# def test_modal_ui_update_signal(mock_signal, modal_ui):
#     modal_ui.update_signal()

#     mock_signal.assert_called()


# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.collect_environment_definition_parameters")
# def test_modal_ui_generate_signal(mock_collect, modal_ui, data_acquisition_parameters):
#     mock_metadata = mock.MagicMock()
#     mock_metadata.samples_per_frame = 100
#     mock_metadata.disabled_signals = []
#     mock_metadata.generate_signal.return_value = np.zeros((1,mock_metadata.samples_per_frame))
#     mock_collect.return_value = mock_metadata
#     mock_plot = mock.MagicMock()
#     modal_ui.plot_data_items['signal_representation'] = [mock_plot]
#     modal_ui.data_acquisition_parameters = data_acquisition_parameters

#     modal_ui.generate_signal()

#     mock_metadata.generate_signal.assert_called()
#     mock_plot.setData.assert_called()


# @pytest.mark.parametrize('sysid_idx', [0, 1])
# def test_modal_ui_update_averaging_type(modal_ui, sysid_idx):
#     mock_idx = mock.MagicMock()
#     mock_idx.currentIndex.return_value = sysid_idx
#     modal_ui.definition_widget.system_id_averaging_scheme_selector = mock_idx

#     modal_ui.update_averaging_type()

#     if sysid_idx == 0:
#         assert not modal_ui.definition_widget.system_id_averaging_coefficient_selector.isEnabled()
#     else:
#         assert modal_ui.definition_widget.system_id_averaging_coefficient_selector.isEnabled()


# @pytest.mark.parametrize('sysid_idx', [0, 2])
# def test_modal_ui_update_window(modal_ui, sysid_idx):
#     mock_idx = mock.MagicMock()
#     mock_idx.currentIndex.return_value = sysid_idx
#     modal_ui.definition_widget.system_id_transfer_function_computation_window_selector = mock_idx
#     mock_widget = mock.MagicMock()
#     modal_ui.window_parameter_widgets = [mock_widget]

#     modal_ui.update_window()

#     if sysid_idx == 2:
#         mock_widget.show.assert_called()
#     else:
#         mock_widget.hide.assert_called()


# @mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
# def test_modal_ui_preview_acquisition(mock_put, modal_ui):
#     modal_ui.preview_acquisition()

#     put_calls = [mock.call('Environment Name UI', (GlobalCommands.START_ENVIRONMENT, "Environment Name")),
#                  mock.call('Environment Name UI', (ModalCommands.START_CONTROL, None))]
#     mock_put.assert_has_calls(put_calls)
#     assert modal_ui.run_widget.stop_test_button.isEnabled()


# @mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
# def test_modal_ui_stop_control(mock_put, modal_ui):
#     modal_ui.stop_control()

#     mock_put.assert_called_with('Environment Name UI', (ModalCommands.STOP_CONTROL, None))


# @mock.patch("rattlesnake.environment.modal_environment.QtWidgets.QFileDialog.getSaveFileName")
# def test_modal_ui_select_file(mock_file, modal_ui):
#     mock_file.return_value = ("Filename", "File Filter")

#     modal_ui.select_file()

#     assert modal_ui.run_widget.data_file_selector.text() == "Filename"


# @mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
# def test_modal_ui_accept_frame(mock_put, modal_ui):
#     modal_ui.accept_frame()

#     mock_put.assert_called_with('Environment Name UI', (ModalCommands.ACCEPT_FRAME, True))
#     assert not modal_ui.run_widget.accept_average_button.isEnabled()
#     assert not modal_ui.run_widget.reject_average_button.isEnabled()


# @mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
# def test_modal_ui_reject_frame(mock_put, modal_ui):
#     modal_ui.reject_frame()

#     mock_put.assert_called_with('Environment Name UI', (ModalCommands.ACCEPT_FRAME, False))
#     assert not modal_ui.run_widget.accept_average_button.isEnabled()
#     assert not modal_ui.run_widget.reject_average_button.isEnabled()


# @mock.patch("rattlesnake.environment.modal_environment.ModalMDISubWindow")
# def test_modal_ui_new_window(mock_window, modal_ui):
#     mock_display = mock.MagicMock()
#     modal_ui.run_widget.channel_display_area = mock_display

#     widget = modal_ui.new_window()

#     mock_display.addSubWindow.assert_called_with(mock_window())
#     mock_window().show.assert_called()
#     assert widget == mock_window()


# def test_modal_ui_close_window(modal_ui):
#     mock_window = mock.MagicMock()
#     mock_window_list = mock.MagicMock()
#     mock_window_list.subWindowList.return_value = [mock_window]
#     modal_ui.run_widget.channel_display_area = mock_window_list

#     modal_ui.close_windows()

#     mock_window.close.assert_called()


# def test_modal_ui_decrement_channels(modal_ui):
#     mock_window = mock.MagicMock()
#     mock_window_list = mock.MagicMock()
#     mock_window_list.subWindowList.return_value = [mock_window]
#     modal_ui.run_widget.channel_display_area = mock_window_list
#     modal_ui.run_widget.increment_channels_number.setValue(2)

#     modal_ui.decrement_channels()

#     mock_window.widget().increment_channel.assert_called_with(-2)


# def test_modal_ui_increment_channels(modal_ui):
#     mock_window = mock.MagicMock()
#     mock_window_list = mock.MagicMock()
#     mock_window_list.subWindowList.return_value = [mock_window]
#     modal_ui.run_widget.channel_display_area = mock_window_list
#     modal_ui.run_widget.increment_channels_number.setValue(2)

#     modal_ui.increment_channels()

#     mock_window.widget().increment_channel.assert_called_with(2)


# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.update_override_table")
# def test_modal_ui_add_override_channels(mock_update, modal_ui):
#     modal_ui.channel_names = ["Channel 1", "Channel 2", "Channel 3"]
#     modal_ui.add_override_channel()

#     mock_update.assert_called()
#     assert isinstance(modal_ui.run_widget.dof_override_table.cellWidget(0,0), QtWidgets.QComboBox)
#     assert modal_ui.run_widget.dof_override_table.item(0,1).text() == '1'
#     assert modal_ui.run_widget.dof_override_table.item(0,2).text() == 'X+'


# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.update_override_table")
# def test_modal_ui_remove_override_channels(mock_update, modal_ui):
#     mock_table = mock.MagicMock()
#     mock_table.currentRow.return_value = 0
#     modal_ui.run_widget.dof_override_table = mock_table

#     modal_ui.remove_override_channel()

#     mock_table.removeRow.assert_called_with(0)
#     mock_update.assert_called()


# @mock.patch("rattlesnake.environment.modal_environment.ModalMDISubWindow.show")
# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.update_channel_names")
# def test_modal_ui_update_override_table(mock_update, mock_show, main_window, modal_ui_init_env, data_acquisition_parameters):
#     modal_ui_init_env.data_acquisition_parameters = data_acquisition_parameters
#     mock_table = mock.MagicMock()
#     mock_table.rowCount.return_value = 1
#     combobox = QtWidgets.QComboBox()
#     combobox.addItem("Item 1")
#     mock_table.cellWidget(0,0).currentIndex.return_value = combobox
#     mock_table.item(0,1).text.return_value = 'Item 2'
#     mock_table.item(0,2).text.return_value = 'Item 3'
#     modal_ui_init_env.run_widget.dof_override_table = mock_table
#     modal_ui_init_env.new_window()

#     modal_ui_init_env.update_override_table()

#     assert modal_ui_init_env.run_widget.channel_display_area.reciprocal_responses == []


# @mock.patch("rattlesnake.environment.modal_environment.ModalMetadata.store_to_netcdf")
# @mock.patch("rattlesnake.environment.modal_environment.nc4.Dataset")
# def test_modal_ui_create_netcdf_file(mock_dataset, mock_store, modal_ui_init_env):
#     modal_ui_init_env.create_netcdf_file("filename")

#     mock_store.assert_called()


# def test_modal_ui_update_channel_names(modal_ui, data_acquisition_parameters):
#     modal_ui.data_acquisition_parameters = data_acquisition_parameters
#     modal_ui.update_channel_names()


# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.generate_signal")
# @mock.patch("rattlesnake.environment.modal_environment.multiline_plotter")
# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.update_trigger_levels")
# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.update_channel_names")
# def test_modal_ui_initialize_data_acquisition(mock_channel, mock_trigger, mock_plot, mock_signal, modal_ui, data_acquisition_parameters):
#     modal_ui.channel_names = ['Accel 221 Y+']
#     modal_ui.initialize_data_acquisition(data_acquisition_parameters)

#     mock_channel.assert_called()
#     mock_trigger.assert_called()
#     mock_plot.assert_called()
#     mock_signal.assert_called()
#     assert modal_ui.reference_indices == []
#     assert modal_ui.response_indices == [0]
#     assert modal_ui. output_channel_indices == []


# @pytest.mark.parametrize("frf_window", ["rectangle", "exponential", "hann"])
# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.get_reciprocal_measurements")
# @mock.patch("rattlesnake.environment.modal_environment.ModalUI.collect_environment_definition_parameters")
# def test_modal_ui_initialize_environment(mock_metadata, mock_reciprocal, modal_ui_init_data, modal_metadata, frf_window):
#     modal_metadata.frf_window = frf_window
#     mock_metadata.return_value = modal_metadata

#     metadata = modal_ui_init_data.initialize_environment()

#     assert metadata == modal_metadata
#     assert modal_ui_init_data.initialized_response_names == ['Accel 221 Y+']
#     assert modal_ui_init_data.initialized_reference_names == []


# def test_modal_ui_retrieve_metadata(modal_ui_init_data, dummy_netcdf, modal_metadata):
#     mock_group_dataset = mock.MagicMock()
#     mock_group_dataset = {"Environment Name" : dummy_netcdf}

#     modal_ui_init_data.retrieve_metadata(mock_group_dataset)

#     np.testing.assert_almost_equal(modal_ui_init_data.definition_widget.system_id_averaging_coefficient_selector.value(), dummy_netcdf.averaging_coefficient, decimal=5)
#     np.testing.assert_almost_equal(modal_ui_init_data.definition_widget.square_frequency_selector.value(), dummy_netcdf.signal_generator_min_frequency, decimal=5)


# @pytest.mark.parametrize("message, data", [("finished", None)])
# def test_modal_ui_update_gui(modal_ui, dummy_netcdf, message, data):
#     modal_ui.netcdf_handle = dummy_netcdf
#     modal_ui.update_gui((message, data))

#     if message == "finished":
#         dummy_netcdf.close.assert_called()


# @mock.patch("rattlesnake.environment.modal_environment.ModalMetadata")
# def test_modal_ui_collect_environment_definition_parameters(mock_metadata, modal_ui):
#     modal_ui.collect_environment_definition_parameters()

#     mock_metadata.from_ui.assert_called_with(modal_ui)


# def test_modal_ui_create_environment_template(modal_ui):
#     mock_worksheet = mock.MagicMock()
#     mock_workbook = mock.MagicMock()
#     mock_workbook.create_sheet.return_value = mock_worksheet

#     modal_ui.create_environment_template("Environment Name", mock_workbook)

#     environment_template_calls = create_environment_template_calls()
#     mock_worksheet.cell.assert_has_calls(environment_template_calls)


# def test_modal_ui_set_parameters_from_template(modal_ui_init_data, data_acquisition_parameters):
#     filename = os.path.join("tests","TemplateFiles","ModalEnvironmentTemplate.xlsx")
#     workbook = openpyxl.load_workbook(filename)
#     worksheet = workbook['Modal']

#     modal_ui_init_data.set_parameters_from_template(worksheet)

#     assert modal_ui_init_data.definition_widget.samples_per_frame_selector.value() == 15000
#     assert modal_ui_init_data.definition_widget.pretrigger_selector.value() == 1


def test_modal_environment_init(modal_queue):
    modal_environment = ModalEnvironment(
        "Environment_name", modal_queue, mp.Value("i", 0), mp.Value("i", 0)
    )

    assert isinstance(modal_environment, ModalEnvironment)


def test_modal_environment_initialize_data_acquisition_parameters(modal_environment):
    assert_value = "Test Value"
    modal_environment.initialize_data_acquisition_parameters(assert_value)

    assert modal_environment.data_acquisition_parameters == assert_value


@mock.patch(
    "rattlesnake.environment.modal_environment.ModalEnvironment.get_spectral_processing_metadata"
)
@mock.patch(
    "rattlesnake.environment.modal_environment.ModalEnvironment.get_signal_generation_metadata"
)
@mock.patch(
    "rattlesnake.environment.modal_environment.ModalEnvironment.get_data_collector_metadata"
)
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
def test_modal_environment_initialize_environment_test_parameters(
    mock_put,
    mock_collect,
    mock_signal,
    mock_spectral,
    modal_environment,
    modal_metadata,
):
    mock_collect.return_value = "Collect Metadata"
    mock_signal.return_value = "Signal Metadata"
    mock_spectral.return_value = "Spectral Metadata"
    modal_environment.initialize_environment_test_parameters(modal_metadata)

    put_calls = [
        mock.call(
            "Environment_name",
            (DataCollectorCommands.INITIALIZE_COLLECTOR, mock_collect()),
        ),
        mock.call(
            "Environment_name",
            (SignalGenerationCommands.INITIALIZE_PARAMETERS, mock_signal()),
        ),
        mock.call(
            "Environment_name",
            (SpectralProcessingCommands.INITIALIZE_PARAMETERS, mock_spectral()),
        ),
    ]

    mock_put.assert_has_calls(put_calls)


@mock.patch("rattlesnake.environment.modal_environment.CollectorMetadata")
def test_modal_environment_get_data_collector_metadata(
    mock_metadata, modal_environment, data_acquisition_parameters, modal_metadata
):
    modal_environment.data_acquisition_parameters = data_acquisition_parameters
    modal_environment.environment_parameters = modal_metadata
    mock_metadata.return_value = "metadata"

    metadata = modal_environment.get_data_collector_metadata()

    assert metadata == "metadata"
    metadata_call = create_environment_collector_metadata_calls()
    mock_metadata.assert_has_calls(metadata_call)


@mock.patch("rattlesnake.environment.modal_environment.SpectralProcessingMetadata")
def test_modal_environment_get_spectral_processing_metadata(
    mock_metadata, modal_environment, data_acquisition_parameters, modal_metadata
):
    modal_environment.data_acquisition_parameters = data_acquisition_parameters
    modal_environment.environment_parameters = modal_metadata
    mock_metadata.return_value = "metadata"

    metadata = modal_environment.get_spectral_processing_metadata()

    assert metadata == "metadata"
    metadata_call = create_environment_spectral_metadata_calls()
    mock_metadata.assert_has_calls(metadata_call)


@mock.patch("rattlesnake.environment.modal_environment.SignalGenerationMetadata")
def test_modal_environment_get_signal_generator_metadata(
    mock_metadata, modal_environment, data_acquisition_parameters, modal_metadata
):
    modal_environment.data_acquisition_parameters = data_acquisition_parameters
    modal_environment.environment_parameters = modal_metadata
    mock_metadata.return_value = "metadata"

    metadata = modal_environment.get_signal_generation_metadata()

    assert metadata == "metadata"
    metadata_call = create_environment_signal_metadata_calls()
    mock_metadata.assert_has_calls(metadata_call)


def test_modal_environment_get_signal_generator(modal_environment):
    mock_parameters = mock.MagicMock()
    modal_environment.environment_parameters = mock_parameters

    modal_environment.get_signal_generator()

    mock_parameters.get_signal_generator.assert_called()


@mock.patch(
    "rattlesnake.environment.modal_environment.ModalEnvironment.get_spectral_processing_metadata"
)
@mock.patch("rattlesnake.environment.modal_environment.ModalEnvironment.get_signal_generator")
@mock.patch(
    "rattlesnake.environment.modal_environment.ModalEnvironment.get_signal_generation_metadata"
)
@mock.patch(
    "rattlesnake.environment.modal_environment.ModalEnvironment.get_data_collector_metadata"
)
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
@mock.patch("rattlesnake.environment.modal_environment.ModalEnvironment.log")
def test_modal_environment_start_environment(
    mock_log,
    mock_put,
    mock_collector,
    mock_signal,
    mock_siggen,
    mock_spectral,
    modal_environment,
    modal_metadata,
):
    modal_environment.environment_parameters = modal_metadata

    modal_environment.start_environment(None)

    start_calls = create_environment_start_calls(
        mock_collector, mock_signal, mock_siggen, mock_spectral
    )
    mock_put.assert_has_calls(start_calls)
    mock_log.assert_called_with("Starting Modal")


@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
@mock.patch("rattlesnake.environment.modal_environment.mp.queues.Queue.put")
@mock.patch("rattlesnake.environment.modal_environment.ModalEnvironment.log")
@mock.patch("rattlesnake.environment.modal_environment.flush_queue")
def test_modal_environment_run_control(
    mock_flush, mock_log, mock_put, mock_vput, modal_environment, modal_metadata
):
    modal_environment.environment_parameters = modal_metadata
    spectral_data = (
        "frames",
        "frequencies",
        "frf",
        "coherence",
        "response_cpsd",
        "reference_cpsd",
        "condition",
    )
    mock_flush.return_value = [spectral_data]

    modal_environment.run_control(None)

    mock_log.assert_called_with("Received Data")
    put_data = (
        "frames",
        30,
        "frequencies",
        "frf",
        "coherence",
        "response_cpsd",
        "reference_cpsd",
        "condition",
    )
    mock_put.assert_called_with(("Environment_name", (ModalUICommands.SPECTRAL_UPDATE, put_data)))
    mock_vput.assert_called_with("Environment_name", (ModalCommands.RUN_CONTROL, None))


def test_modal_environment_siggen_shutdown_achieved_fn(modal_environment):
    modal_environment.siggen_shutdown_achieved_fn("data")

    assert modal_environment.siggen_shutdown_achieved == True


def test_modal_environment_collector_shutdown_achieved_fn(modal_environment):
    modal_environment.collector_shutdown_achieved_fn("data")

    assert modal_environment.collector_shutdown_achieved == True


def test_modal_environment_spectral_shutdown_achieved_fn(modal_environment):
    modal_environment.spectral_shutdown_achieved_fn("data")

    assert modal_environment.spectral_shutdown_achieved == True


@pytest.mark.parametrize("shutdown", [True, False])
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
@mock.patch("rattlesnake.environment.modal_environment.mp.queues.Queue.put")
@mock.patch("rattlesnake.environment.modal_environment.ModalEnvironment.log")
def test_modal_environment_check_for_shutdown(
    mock_log, mock_put, mock_vput, modal_environment, shutdown
):
    modal_environment.siggen_shutdown_achieved = shutdown
    modal_environment.collector_shutdown_achieved = shutdown
    modal_environment.spectral_shutdown_achieved = shutdown

    modal_environment.check_for_shutdown("data")

    if shutdown:
        mock_log.assert_called_with("Shutdown Achieved")
        mock_put.assert_called_with(("Environment_name", (ModalUICommands.FINISHED, None)))
    else:
        mock_vput.assert_called_with(
            "Environment_name", (ModalCommands.CHECK_FOR_COMPLETE_SHUTDOWN, None)
        )


@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
def test_modal_environment_accept_frame(mock_put, modal_environment):
    modal_environment.accept_frame("data")

    mock_put.assert_called_with("Environment_name", (DataCollectorCommands.ACCEPT, "data"))


@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
@mock.patch("rattlesnake.environment.modal_environment.flush_queue")
@mock.patch("rattlesnake.environment.modal_environment.ModalEnvironment.log")
def test_modal_environment_stop_environment(mock_log, mock_flush, mock_put, modal_environment):
    modal_environment.stop_environment("data")

    mock_log.assert_called_with("Stopping Control")
    mock_flush.assert_called_with(modal_environment.queue_container.environment_command_queue)
    put_calls = [
        mock.call("Environment_name", (DataCollectorCommands.SET_TEST_LEVEL, (1000, 1))),
        mock.call("Environment_name", (SignalGenerationCommands.START_SHUTDOWN, None)),
        mock.call(
            "Environment_name",
            (SpectralProcessingCommands.STOP_SPECTRAL_PROCESSING, None),
        ),
        mock.call("Environment_name", (ModalCommands.CHECK_FOR_COMPLETE_SHUTDOWN, None)),
    ]
    mock_put.assert_has_calls(put_calls)


@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
def test_modal_environment_quit(mock_put, modal_environment):
    data = modal_environment.quit("data")

    put_calls = [
        mock.call("Environment_name", (GlobalCommands.QUIT, None)),
        mock.call("Environment_name", (GlobalCommands.QUIT, None)),
        mock.call("Environment_name", (GlobalCommands.QUIT, None)),
    ]
    mock_put.assert_has_calls(put_calls)
    assert data == True


@mock.patch("rattlesnake.environment.modal_environment.ModalEnvironment.log")
@mock.patch("rattlesnake.environment.modal_environment.mp.Process.join")
@mock.patch("rattlesnake.environment.modal_environment.ModalEnvironment.run")
@mock.patch("rattlesnake.environment.modal_environment.mp.Process.start")
def test_modal_process_function(mock_start, mock_run, mock_join, mock_log, log_file_queue):
    modal_process(
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

    mock_start.assert_called()
    mock_run.assert_called()
    mock_join.assert_called()
    mock_log.assert_called_with("Joining Data Collection")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    log_file_queue = mp.Queue()
    queue_container = QueueContainer(
        VerboseMessageQueue(log_file_queue, "Controller Communication Queue"),
        VerboseMessageQueue(log_file_queue, "Acquisition Command Queue"),
        VerboseMessageQueue(log_file_queue, "Output Command Queue"),
        VerboseMessageQueue(log_file_queue, "Streaming Command Queue"),
        log_file_queue,
        mp.Queue(),
        mp.Queue(),
        mp.Queue(),
        {"Modal": VerboseMessageQueue(log_file_queue, "Environment Command Queue")},
        {"Modal": mp.Queue()},
        {"Modal": mp.Queue()},
    )

    modal_queue = ModalQueues(
        "Environment Name",
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Controller Communication Queue"),
        mp.Queue(),
        mp.Queue(),
        log_file_queue,
    )

    modal_environment = ModalEnvironment(
        "Environment_name", modal_queue, mp.Value("i", 0), mp.Value("i", 0)
    )

    acquisition_active = mp.Value("i", 0)
    output_active = mp.Value("i", 0)

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
        "",
        "",
        "5",
        "10",
    ]
    channel = Channel.from_channel_table_row(channel_table_row)

    channel_list = [channel]
    sample_rate = 2048
    time_per_read = 5
    time_per_write = 5
    output_oversample = 10
    hardware_selector_idx = 6
    hardware_file = "ExampleFile.nc4"
    environments = ["Environment Name"]
    environment_booleans = [[True]]
    environment_booleans = np.array(environment_booleans)
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

    modal_metadata = ModalMetadata(
        2048,
        10,
        "Linear",
        30,
        0.1,
        "H1",
        "rectangle",
        0.0,
        "Free Run",
        "Accept All",
        0.0,
        0,
        0.0,
        True,
        0.1,
        0.2,
        0.4,
        "none",
        0,
        0,
        0,
        0,
        None,
        [],
        [1, 2],
        [3],
        data_acquisition_parameters,
        0.25,
    )

    main_window = DummyMainWindow()
    modal_ui = ModalUI(
        "Environment Name",
        main_window.definition_tabwidget,
        main_window.system_id_tabwidget,
        main_window.test_predictions_tabwidget,
        main_window.test_predictions_tabwidget,
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        VerboseMessageQueue(log_file_queue, "Controller Command Queue"),
        log_file_queue,
    )

    # modal_ui.initialize_data_acquisition(data_acquisition_parameters)
    # modal_ui.initialize_environment()

    # test_modal_ui_update_window(modal_ui=modal_ui, sysid_idx=1)
    # test_modal_ui_set_parameters_from_template(modal_ui=modal_ui)
    # test_modal_ui_generate_signal(modal_ui = modal_ui, data_acquisition_parameters=data_acquisition_parameters)

    test_modal_ui_init(app=app, log_file_queue=log_file_queue)
    # test_modal_metadata_get_trigger_levels(modal_metadata=modal_metadata, channel_list=channel_list)
    # test_modal_environment_get_signal_generator_metadata(modal_environment=modal_environment, data_acquisition_parameters=data_acquisition_parameters, modal_metadata=modal_metadata)
