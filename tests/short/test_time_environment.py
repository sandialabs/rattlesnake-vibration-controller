import multiprocessing as mp
from unittest import mock

import netCDF4 as nc4
import numpy as np
import pytest

from rattlesnake.testing.mock_hardware import MockHardwareMetadata
from rattlesnake.testing.mock_utilities import (
    mock_channel_list_bools,
    mock_event_container,
    mock_queue_container,
)
from rattlesnake.environment.abstract_environment import (
    Environment,
    EnvironmentInstructions,
    EnvironmentMetadata,
)
from rattlesnake.environment.time_environment import (
    TimeCommands,
    TimeEnvironment,
    TimeInstructions,
    TimeMetadata,
    TimeQueues,
    TimeUICommands,
    time_process,
)
from rattlesnake.hardware.hardware_utilities import Channel
from rattlesnake.utilities import GlobalCommands, RattlesnakeError
from rattlesnake.user_interface.ui_utilities import UICommands


# region Fixtures
@pytest.fixture
def time_metadata():
    time_metadata = TimeMetadata(
        "Time Environment",
        channel_list_bools=mock_channel_list_bools,
        sample_rate=1000,
        output_signal=np.zeros((1, 2000)),
        cancel_rampdown_time=0.1,
    )
    time_metadata.queue_name = "Environment 0"

    return time_metadata


@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def time_environment(request):
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    time_queues = TimeQueues(
        queue_container.environment_command_queues["Environment 0"],
        queue_container.gui_update_queue,
        queue_container.controller_command_queue,
        queue_container.environment_data_in_queues["Environment 0"],
        queue_container.environment_data_out_queues["Environment 0"],
        queue_container.log_file_queue,
    )
    acquisition_active = mp.Value("i", 0)
    output_active = mp.Value("i", 0)
    time_environment = TimeEnvironment(
        "Time Environment",
        "Environment 0",
        time_queues,
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.environment_active_events["Environment 0"],
        event_container.environment_ready_events["Environment 0"],
    )
    return time_environment


# region Metadata
def test_time_metadata_init():
    time_metadata = TimeMetadata("Time Environment")

    assert isinstance(time_metadata, TimeMetadata)
    assert isinstance(time_metadata, EnvironmentMetadata)


def test_time_metadata_properties(time_metadata):
    assert time_metadata.signal_samples == 2000
    assert time_metadata.output_channels == 1
    assert time_metadata.signal_time == 2
    assert time_metadata.cancel_rampdown_samples == 100


@pytest.mark.parametrize(
    "channel_list_bools, sample_rate, cancel_rampdown_time, output_signal, expected",
    [
        (mock_channel_list_bools(), 1000, 0.5, np.zeros((1, 2000)), True),
        (mock_channel_list_bools(), -10, 0.5, np.zeros((1, 2000)), RattlesnakeError),
        (mock_channel_list_bools(), 1000, None, np.zeros((1, 2000)), RattlesnakeError),
        (mock_channel_list_bools(), None, 0.5, np.zeros((1, 2000)), RattlesnakeError),
        (mock_channel_list_bools(), 1000, 0.5, None, RattlesnakeError),
        (
            mock_channel_list_bools(),
            1000,
            0.5,
            np.zeros((1, 2000, 3)),
            RattlesnakeError,
        ),
        (mock_channel_list_bools(), 1000, 0.5, np.zeros((0, 2000)), RattlesnakeError),
    ],
)
def test_time_metadata_validate(
    channel_list_bools,
    sample_rate,
    cancel_rampdown_time,
    output_signal,
    expected,
    time_metadata,
):
    hardware_metadata = MockHardwareMetadata()
    time_metadata.channel_list_bools = channel_list_bools
    time_metadata.sample_rate = sample_rate
    time_metadata.cancel_rampdown_time = cancel_rampdown_time
    time_metadata.output_signal = output_signal

    if expected is RattlesnakeError:
        with pytest.raises(RattlesnakeError):
            time_metadata.validate(hardware_metadata)
    elif expected:
        valid_metadata = time_metadata.validate(hardware_metadata)
        assert valid_metadata


def test_environment_metadata_store_to_netcdf(time_metadata):
    dataset = nc4.Dataset("temp.nc", mode="w", diskless=True, persist=False)
    netcdf_group = dataset.createGroup("temp_group")

    time_metadata.save_metadata_to_netcdf(netcdf_group)

    assert True


# region Instructions
def test_time_instructions_init():
    time_instructions = TimeInstructions(
        environment_name="Time Environment",
        current_test_level=1,
        repeat=False,
    )

    assert isinstance(time_instructions, TimeInstructions)
    assert isinstance(time_instructions, EnvironmentInstructions)
    assert hasattr(time_instructions, "current_test_level")
    assert hasattr(time_instructions, "repeat")


# region Queues
@pytest.mark.parametrize("use_thread", [True, False])
def test_time_queues_init(use_thread):
    queue_container = mock_queue_container(use_thread)
    time_queues = TimeQueues(
        queue_container.environment_command_queues["Environment 0"],
        queue_container.gui_update_queue,
        queue_container.controller_command_queue,
        queue_container.environment_data_in_queues["Environment 0"],
        queue_container.environment_data_out_queues["Environment 0"],
        queue_container.log_file_queue,
    )

    assert isinstance(time_queues, TimeQueues)


# region Environment
@pytest.mark.parametrize("use_thread", [True, False])
def test_time_environment_init(use_thread):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    time_queues = TimeQueues(
        queue_container.environment_command_queues["Environment 0"],
        queue_container.gui_update_queue,
        queue_container.controller_command_queue,
        queue_container.environment_data_in_queues["Environment 0"],
        queue_container.environment_data_out_queues["Environment 0"],
        queue_container.log_file_queue,
    )
    time_environment = TimeEnvironment(
        "Time Environment",
        "Environment 0",
        time_queues,
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.environment_active_events["Environment 0"],
        event_container.environment_ready_events["Environment 0"],
    )
    assert isinstance(time_environment, TimeEnvironment)
    assert isinstance(time_environment, Environment)


def test_time_environment_initialize_hardware(time_environment):
    hardware_metadata = MockHardwareMetadata()
    time_environment.initialize_hardware(hardware_metadata)

    assert time_environment.hardware_metadata == hardware_metadata
    assert time_environment.measurement_channels == [0]
    assert time_environment.output_channels == [1]


def test_time_environment_initialize_environment(time_metadata, time_environment):
    time_environment.initialize_environment(time_metadata)

    assert time_environment.environment_metadata == time_metadata


@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.shutdown")
@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.output")
@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.log")
def test_time_environment_run_environment(
    mock_log,
    mock_output,
    mock_shutdown,
    time_environment,
    time_metadata,
):
    mock_gui_queue = mock.MagicMock()
    time_environment.queue_container.gui_update_queue = mock_gui_queue

    mock_data_in_queue = mock.MagicMock()
    time_environment.queue_container.data_in_queue = mock_data_in_queue

    mock_data_out_queue = mock.MagicMock()
    time_environment.queue_container.data_out_queue = mock_data_out_queue

    mock_command_queue = mock.MagicMock()
    time_environment.queue_container.environment_command_queue = mock_command_queue

    mock_data_in_queue.get_nowait.return_value = (np.ones((2, 2000)), False)
    mock_data_out_queue.empty.return_value = True

    hardware_metadata = MockHardwareMetadata()
    time_environment.initialize_hardware(hardware_metadata)
    time_environment.environment_metadata = time_metadata

    time_instructions = TimeInstructions(
        environment_name="Time Environment",
        current_test_level=1,
        repeat=True,
    )

    time_environment.run_environment(time_instructions)

    assert time_environment.active

    mock_gui_queue.put.assert_any_call(
        (
            "Time Environment",
            (UICommands.SET_ENVIRONMENT_INSTRUCTIONS, time_instructions),
        )
    )
    mock_gui_queue.put.assert_any_call(
        ("Time Environment", (UICommands.ENVIRONMENT_STARTED, None))
    )

    time_data_call = None
    for call in mock_gui_queue.put.call_args_list:
        args = call[0][0]
        if args[0] == "Time Environment" and args[1][0] == TimeUICommands.TIME_DATA:
            time_data_call = args
            break

    assert time_data_call is not None

    measurement_data, output_data = time_data_call[1][1]
    np.testing.assert_array_equal(measurement_data, np.ones((1, 2000)))
    np.testing.assert_array_equal(output_data, np.ones((1, 2000)))

    np.testing.assert_array_equal(
        time_metadata.output_signal[:, : hardware_metadata.samples_per_write],
        mock_output.call_args[0][0],
    )
    assert mock_output.call_args[0][1] is False

    mock_command_queue.put.assert_called_with(
        "Time Environment", (GlobalCommands.START_ENVIRONMENT, None)
    )

    mock_shutdown.assert_not_called()


@pytest.mark.parametrize("test_level_change", [0, -0.001])
@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.log")
def test_time_environment_output(mock_log, test_level_change, time_environment):
    mock_data_out_queue = mock.MagicMock()
    time_environment.queue_container.data_out_queue = mock_data_out_queue
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
    np.testing.assert_array_almost_equal(
        output_array, mock_data_out_queue.put.call_args_list[0][0][0][0]
    )
    assert mock_data_out_queue.put.call_args_list[0][0][0][1] == False


@mock.patch(
    "rattlesnake.environment.time_environment.TimeEnvironment.adjust_test_level"
)
def test_time_environment_stop_environment(mock_adjust, time_environment):
    time_environment.stop_environment(None)

    mock_adjust.assert_called_with(0.0)


@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.log")
def test_time_environment_adjust_test_level(mock_log, time_environment, time_metadata):
    time_environment.current_test_level = 1
    time_environment.test_level_target = 0.1
    time_environment.environment_metadata = time_metadata

    time_environment.adjust_test_level(0.8)

    assert time_environment.test_level_target == 0.8
    mock_log.assert_called_with(
        "Changed test level to 0.8 from 1, {:} change per sample".format(
            time_environment.test_level_change
        )
    )


@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment.log")
def test_time_environment_shutdown(mock_log, time_environment):
    mock_gui_queue = mock.MagicMock()
    time_environment.queue_container.gui_update_queue = mock_gui_queue
    mock_command_queue = mock.MagicMock()
    time_environment.queue_container.environment_command_queue = mock_command_queue
    time_environment.shutdown()

    mock_log.assert_called_with("Shutting Down Time History Generation")
    mock_command_queue.flush.assert_called_with("Time Environment")
    put_calls = [
        mock.call(("Time Environment", (UICommands.ENVIRONMENT_ENDED, None))),
    ]
    mock_gui_queue.put.assert_has_calls(put_calls)
    assert not time_environment._active_event.is_set()


# region: time_process
@mock.patch("rattlesnake.environment.time_environment.TimeEnvironment")
@pytest.mark.parametrize("use_thread", [True, False])
def test_time_process(mock_process_class, use_thread):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    time_process(
        "Environment Name",
        "Environment 0",
        queue_container.environment_command_queues["Environment 0"],
        queue_container.gui_update_queue,
        queue_container.controller_command_queue,
        queue_container.log_file_queue,
        queue_container.environment_data_in_queues["Environment 0"],
        queue_container.environment_data_out_queues["Environment 0"],
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.environment_active_events["Environment 0"],
        event_container.environment_ready_events["Environment 0"],
        event_container.environment_close_events["Environment 0"],
        event_container.environment_sysid_active_events["Environment 0"],
        event_container.environment_sysid_stored_events["Environment 0"],
        use_thread,
    )

    mock_instance = mock_process_class.return_value
    mock_instance.run.assert_called()
