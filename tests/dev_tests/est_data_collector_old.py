import copy
import multiprocessing as mp
from unittest import mock

import numpy as np
import pytest
from functions.data_collector_functions import create_acquire_log_calls

# from PyQt5 import QtWidgets  # unused import

from rattlesnake.process.data_collector import (
    Acceptance,
    AcquisitionType,
    CollectorMetadata,
    DataCollectorCommands,
    DataCollectorProcess,
    FrameBuffer,
    TriggerSlope,
    Window,
    data_collector_process,
)
from rattlesnake.utilities import VerboseMessageQueue


# Create log_file_queue
@pytest.fixture
def log_file_queue():
    return mp.Queue()


@pytest.fixture
def frame_buffer():
    frame_buffer = FrameBuffer(
        2, 0, 0, False, 0.5, 0.1, 10, 100, 0, False, False, False, 0, starting_value=1
    )
    return frame_buffer


@pytest.fixture
def collector_metadata():
    collector_metadata = CollectorMetadata(
        2,
        [1, 2],
        [],
        AcquisitionType.FREE_RUN,
        Acceptance.AUTOMATIC,
        None,
        0.2,
        0,
        TriggerSlope.POSITIVE,
        0.1,
        0.0,
        0.0,
        0.1,
        100,
        Window.RECTANGLE,
        window_parameter_1=1,
        window_parameter_2=1,
        window_parameter_3=1,
    )

    return collector_metadata


@pytest.fixture
def data_collector_process_obj(log_file_queue):
    data_collector_process = DataCollectorProcess(
        "Process Name",
        VerboseMessageQueue(log_file_queue, "Spectral Command Queue"),
        mp.Queue(),
        [mp.Queue()],
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        log_file_queue,
        mp.Queue(),
        "Environment Name",
    )

    return data_collector_process


def test_frame_buffer_init():
    frame_buffer = FrameBuffer(
        2, 0, 0, False, 0, 0, 0, 100, 0, False, False, False, 0, starting_value=1
    )

    assert isinstance(frame_buffer, FrameBuffer)
    np.testing.assert_array_equal(frame_buffer.buffer_data, np.ones((2, 200)))


def test_frame_buffer_add_data(frame_buffer):
    frame_buffer.add_data(np.zeros((2, 200)))

    np.testing.assert_array_equal(frame_buffer.buffer_data, np.zeros((2, 200)))


@pytest.mark.parametrize("manual_accept", [True, False])
@pytest.mark.parametrize(
    "data_idx, buffer_data, positive_slope",
    [
        (
            0,
            np.concatenate((np.zeros((2, 25)), np.ones((2, 25)), np.zeros((2, 150))), axis=1),
            False,
        ),
        (
            1,
            np.concatenate((np.zeros((2, 25)), np.ones((2, 25)), np.zeros((2, 150))), axis=1),
            True,
        ),
        (2, np.concatenate((np.zeros((2, 100)), np.ones((2, 100))), axis=1), False),
    ],
)
def test_frame_buffer_find_triggers(
    frame_buffer, data_idx, buffer_data, positive_slope, manual_accept
):
    frame_buffer.trigger_enabled = True
    frame_buffer.trigger_only_first = False
    frame_buffer.last_trigger = 300
    frame_buffer.manual_accept = manual_accept
    frame_buffer.positive_slope = positive_slope
    frame_buffer._buffer = buffer_data
    triggers = frame_buffer.find_triggers()

    if manual_accept and len(triggers) > 0:
        assert frame_buffer.waiting_for_accept == True
    else:
        assert frame_buffer.waiting_for_accept == False

    match data_idx:
        case 0:
            assert triggers[0] == np.int64(150)
        case 1:
            assert triggers[0] == np.int64(175)
        case 2:
            assert triggers == []


def test_frame_buffer_reset_trigger(frame_buffer):
    frame_buffer.reset_trigger()

    assert frame_buffer.last_trigger == 100
    assert frame_buffer.last_reset == 99


def test_frame_buffer_accept(frame_buffer):
    frame_buffer.accept()

    assert frame_buffer.last_trigger == 100
    assert frame_buffer.last_reset == 99
    assert frame_buffer.waiting_for_accept == False


@mock.patch("rattlesnake.process.data_collector.FrameBuffer.find_triggers")
@mock.patch("rattlesnake.process.data_collector.FrameBuffer.add_data")
def test_frame_buffer_add_data_get_frame(mock_add, mock_find, frame_buffer):
    mock_find.return_value = [np.int64(125)]
    buffer_data = np.concatenate((np.zeros((2, 75)), np.ones((2, 25)), np.zeros((2, 100))), axis=1)
    frame_buffer._buffer = buffer_data

    data = frame_buffer.add_data_get_frame(buffer_data)

    mock_add.assert_called_with(buffer_data)
    assert_data = np.concatenate((np.ones((1, 2, 25)), np.zeros((1, 2, 75))), axis=2)
    np.testing.assert_array_equal(data, assert_data)


def test_frame_buffer_get_item(frame_buffer):
    key = 1
    item = 123
    frame_buffer._buffer = np.zeros((2, 1))
    frame_buffer._buffer[key] = item

    assert frame_buffer[key] == item


def test_frame_buffer_set_item(frame_buffer):
    key = 1
    item = 123
    frame_buffer._buffer = np.zeros((2, 1))
    frame_buffer[key] = item

    assert frame_buffer._buffer[key] == item


@pytest.mark.parametrize("command_idx", [1, 2, 3, 4, 5, 6, 7, 8])
def test_data_collector_commands(command_idx):
    data_collector_command = DataCollectorCommands(command_idx)

    assert isinstance(data_collector_command, DataCollectorCommands)


@pytest.mark.parametrize("type_idx", [0, 1, 2])
def test_acqusition_type(type_idx):
    acqusition_type = AcquisitionType(type_idx)

    assert isinstance(acqusition_type, AcquisitionType)


@pytest.mark.parametrize("acceptance_idx", [0, 1])
def test_acceptance(acceptance_idx):
    acceptance = Acceptance(acceptance_idx)

    assert isinstance(acceptance, Acceptance)


@pytest.mark.parametrize("trigger_idx", [0, 1])
def test_trigger_slope(trigger_idx):
    trigger_slope = TriggerSlope(trigger_idx)

    assert isinstance(trigger_slope, TriggerSlope)


@pytest.mark.parametrize("window_idx", [0, 1, 2, 3, 4, 5, 6, 7])
def test_window(window_idx):
    window = Window(window_idx)

    assert isinstance(window, Window)


def test_data_collector_metadata_init():
    collector_metadata = CollectorMetadata(
        2,
        [1, 2],
        [],
        AcquisitionType.FREE_RUN,
        Acceptance.AUTOMATIC,
        None,
        0.2,
        0,
        TriggerSlope.POSITIVE,
        0.1,
        0.0,
        0.0,
        0.1,
        100,
        Window.RECTANGLE,
    )

    assert isinstance(collector_metadata, CollectorMetadata)


def test_data_collector_metadata_eq(collector_metadata):
    collector_metadata_assert = copy.deepcopy(collector_metadata)

    assert collector_metadata == collector_metadata_assert


# Test DataCollectorProcess intialization
def test_data_collector_process_init(log_file_queue):
    data_collector_process = DataCollectorProcess(
        "Process Name",
        VerboseMessageQueue(log_file_queue, "Spectral Command Queue"),
        mp.Queue(),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        log_file_queue,
        mp.Queue(),
        "Environment Name",
    )

    # Test if object is correct class
    assert isinstance(data_collector_process, DataCollectorProcess)


@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess.force_initialize_collector")
def test_data_collector_process_initialize_collector(
    mock_init, data_collector_process_obj, collector_metadata
):
    data_collector_process_obj.initialize_collector(collector_metadata)

    mock_init.assert_called_with(collector_metadata)


@pytest.mark.parametrize(
    "window",
    [
        Window.RECTANGLE,
        Window.HANN,
        Window.HAMMING,
        Window.FLATTOP,
        Window.TUKEY,
        Window.BLACKMANHARRIS,
        Window.BLACKMANHARRIS,
        Window.EXPONENTIAL,
        Window.EXPONENTIAL_FORCE,
    ],
)
@mock.patch("rattlesnake.process.data_collector.flush_queue")
def test_data_collector_process_force_initialize_collector(
    mock_flush, data_collector_process_obj, collector_metadata, window
):
    collector_metadata.window = window
    data_collector_process_obj.force_initialize_collector(collector_metadata)

    assert not data_collector_process_obj.reference_window is None


@pytest.mark.parametrize("last_data", [True, False])
@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess.stop")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
@mock.patch("rattlesnake.process.data_collector.mp.queues.Queue.put")
@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess.log")
@mock.patch("rattlesnake.process.data_collector.mp.queues.Queue.get")
def test_data_collector_process_acquire(
    mock_get,
    mock_log,
    mock_put,
    mock_vput,
    mock_stop,
    data_collector_process_obj,
    last_data,
):
    data = np.zeros((1, 2, 100))
    mock_get.return_value = (data, last_data)
    mock_buffer = mock.MagicMock()
    mock_buffer.add_data_get_frame.return_value = data
    mock_buffer.manual_accept = False
    data_collector_process_obj.frame_buffer = mock_buffer
    data_collector_process_obj.acceptance_function = lambda x: True
    mock_metadata = mock.MagicMock()
    mock_metadata.reference_channel_indices = [0]
    mock_metadata.response_channel_indices = [1]
    mock_metadata.response_transformation_matrix = None
    mock_metadata.reference_transformation_matrix = None
    mock_metadata.kurtosis_buffer_length = None
    data_collector_process_obj.collector_metadata = mock_metadata
    data_collector_process_obj.response_window = 1
    data_collector_process_obj.reference_window = 1
    data_collector_process_obj.test_level = 1
    data_collector_process_obj.window_correction = 1

    data_collector_process_obj.acquire(None)

    log_calls = create_acquire_log_calls(last_data)
    mock_log.assert_has_calls(log_calls)
    put_calls = [
        mock.call("Environment Name", ("time_frame", (data, True))),
        mock.call(np.array([[0.0 + 0.0j] * 100, [0.0 + 0.0j] * 100])),
    ]
    np.testing.assert_array_almost_equal(
        mock_put.call_args_list[1][0][0], np.zeros((2, 1, 51), dtype=complex)
    )
    if last_data:
        mock_stop.assert_called()
    else:
        mock_vput.assert_called_with("Process Name", (DataCollectorCommands.ACQUIRE, None))


@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
@mock.patch("rattlesnake.process.data_collector.mp.queues.Queue.put")
@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess.log")
def test_data_collector_process_accept(mock_log, mock_put, mock_vput, data_collector_process_obj):
    mock_buffer = mock.MagicMock()
    data_collector_process_obj.frame_buffer = mock_buffer
    accept_frame = np.ones((2, 100))
    data_collector_process_obj.last_frame = accept_frame
    data_collector_process_obj.window_correction = np.ones((2, 51))
    mock_metadata = mock.MagicMock()
    mock_metadata.reference_channel_indices = [0]
    mock_metadata.response_channel_indices = [1]
    data_collector_process_obj.collector_metadata = mock_metadata

    data_collector_process_obj.accept(True)

    log_calls = [
        mock.call("Received Accept Signal True"),
        mock.call("Sending data manually"),
        mock.call("Sent Data"),
    ]
    mock_log.assert_has_calls(log_calls)
    np.testing.assert_array_equal(mock_put.call_args_list[0][0][0][1][1][0], accept_frame)
    np.testing.assert_array_equal(
        mock_put.call_args_list[1][0][0][0],
        np.array([[100.0 + 0.0j] + [0.0 + 0.0j] * 50]),
    )
    mock_vput.assert_called_with("Process Name", (DataCollectorCommands.ACCEPTED, True))


@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.flush")
@mock.patch("rattlesnake.process.data_collector.flush_queue")
@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess.log")
def test_data_collector_process_stop(
    mock_log, mock_flush, mock_vflush, mock_put, data_collector_process_obj
):
    mock_buffer = mock.MagicMock()
    data_collector_process_obj.frame_buffer = mock_buffer

    data_collector_process_obj.stop(None)

    mock_log.assert_called_with("Stopping Data Collection")
    mock_flush.assert_called()
    mock_vflush.assert_called_with("Process Name")
    mock_buffer.reset_trigger.assert_called()
    mock_put.assert_called_with("Process Name", (DataCollectorCommands.SHUTDOWN_ACHIEVED, None))


@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess.log")
def test_data_collector_process_set_test_level(mock_log, data_collector_process_obj):
    data_collector_process_obj.set_test_level((10, 0.1))

    assert data_collector_process_obj.skip_frames == 10
    assert data_collector_process_obj.test_level == 0.1
    mock_log.assert_called_with("Setting Test Level to 0.1, skipping next 10 frames")


# Test the data_collector_process
# Prevent the run while loop from starting
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.run")
def test_data_collector_process_function(mock_run, log_file_queue):
    data_collector_process(
        "Environment Name",
        VerboseMessageQueue(log_file_queue, "Spectral Command Queue"),
        mp.Queue(),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        log_file_queue,
        mp.Queue(),
        "Process Name",
    )

    # Make sure the run function was called
    mock_run.assert_called()


if __name__ == "__main__":
    frame_buffer = FrameBuffer(
        2, 0, 0, False, 0.5, 0.1, 10, 100, 0, False, False, False, 0, starting_value=1
    )

    # test_frame_buffer_find_triggers(frame_buffer, data_idx = 2, buffer_data=np.concatenate((np.zeros((2, 100)), np.ones((2, 100))),axis=1), positive_slope = False, manual_accept = True)
    # test_frame_buffer_get_item(frame_buffer=frame_buffer)

    log_file_queue = mp.Queue()
    data_collector_process_obj = DataCollectorProcess(
        "Process Name",
        VerboseMessageQueue(log_file_queue, "Spectral Command Queue"),
        mp.Queue(),
        [mp.Queue()],
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        log_file_queue,
        mp.Queue(),
        "Environment Name",
    )

    test_data_collector_process_acquire(
        data_collector_process_obj=data_collector_process_obj, last_data=True
    )
