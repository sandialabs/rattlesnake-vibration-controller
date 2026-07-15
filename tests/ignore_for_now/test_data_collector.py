import copy
import multiprocessing as mp
from unittest import mock

import numpy as np
import pytest

from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.process.data_collector import (
    Acceptance,
    AcquisitionType,
    CollectorMetadata,
    DataCollectorCommands,
    DataCollectorProcess,
    DataCollectorUICommands,
    FrameBuffer,
    KurtosisBuffer,
    TriggerSlope,
    Window,
    data_collector_process,
)
from rattlesnake.utilities import GlobalCommands, VerboseMessageQueue


# region Fixtures
@pytest.fixture
def log_file_queue():
    return mp.Queue()


@pytest.fixture
def command_queue(log_file_queue):
    return VerboseMessageQueue(
        log_file_queue,
        mp.Queue(),
        "Data Collector Command Queue",
    )


@pytest.fixture
def environment_command_queue(log_file_queue):
    return VerboseMessageQueue(
        log_file_queue,
        mp.Queue(),
        "Environment Command Queue",
    )


@pytest.fixture
def frame_buffer():
    return FrameBuffer(
        num_channels=2,
        trigger_index=0,
        pretrigger=0,
        positive_slope=False,
        trigger_level=0.5,
        hysteresis_level=0.1,
        hysteresis_samples=10,
        samples_per_frame=100,
        maximum_overlap=0,
        manual_accept=False,
        trigger_enabled=False,
        trigger_only_first=False,
        wait_samples=0,
        starting_value=1,
    )


@pytest.fixture
def collector_metadata():
    return CollectorMetadata(
        num_channels=2,
        response_channel_indices=[1],
        reference_channel_indices=[0],
        acquisition_type=AcquisitionType.FREE_RUN,
        acceptance=Acceptance.AUTOMATIC,
        acceptance_function=None,
        overlap_fraction=0.2,
        trigger_channel_index=0,
        trigger_slope=TriggerSlope.POSITIVE,
        trigger_level=0.1,
        trigger_hysteresis=0.0,
        trigger_hysteresis_samples=0,
        pretrigger_fraction=0.1,
        frame_size=100,
        window=Window.RECTANGLE,
        window_parameter_1=1,
        window_parameter_2=1,
        window_parameter_3=1,
    )


@pytest.fixture
def data_collector(
    command_queue,
    environment_command_queue,
    log_file_queue,
):
    return DataCollectorProcess(
        "Process Name",
        command_queue,
        mp.Queue(),
        [mp.Queue()],
        environment_command_queue,
        log_file_queue,
        mp.Queue(),
        "Environment Name",
    )


# endregion


# region FrameBuffer
def test_frame_buffer_init():
    """
    Verifies that a frame buffer initializes with the expected shape and
    starting values.
    """
    frame_buffer = FrameBuffer(
        num_channels=2,
        trigger_index=0,
        pretrigger=0,
        positive_slope=False,
        trigger_level=0,
        hysteresis_level=0,
        hysteresis_samples=0,
        samples_per_frame=100,
        maximum_overlap=0,
        manual_accept=False,
        trigger_enabled=False,
        trigger_only_first=False,
        wait_samples=0,
        starting_value=1,
    )

    assert isinstance(frame_buffer, FrameBuffer)
    assert frame_buffer.samples_per_frame == 100
    assert frame_buffer.trigger_index == 0
    assert frame_buffer.pretrigger_samples == 0
    assert frame_buffer.overlap_samples == 100
    assert frame_buffer.waiting_for_accept is False
    assert frame_buffer.trigger_enabled is False
    assert frame_buffer.trigger_only_first is False
    assert frame_buffer.first_trigger is True
    np.testing.assert_array_equal(frame_buffer.buffer_data, np.ones((2, 200)))


def test_frame_buffer_add_data(frame_buffer):
    """
    Verifies that added data shifts into the buffer correctly.
    """
    frame_buffer.add_data(np.zeros((2, 200)))

    np.testing.assert_array_equal(frame_buffer.buffer_data, np.zeros((2, 200)))


def test_frame_buffer_add_data_truncates_to_buffer_length(frame_buffer):
    """
    Verifies that data longer than the buffer is truncated to the most recent
    samples.
    """
    data = np.arange(500).reshape(1, 500)
    data = np.vstack([data, data])

    frame_buffer.add_data(data)

    np.testing.assert_array_equal(frame_buffer.buffer_data, data[:, -200:])


@pytest.mark.parametrize("manual_accept", [True, False])
@pytest.mark.parametrize(
    "data_idx, buffer_data, positive_slope",
    [
        (
            0,
            np.concatenate(
                (np.zeros((2, 25)), np.ones((2, 25)), np.zeros((2, 150))),
                axis=1,
            ),
            False,
        ),
        (
            1,
            np.concatenate(
                (np.zeros((2, 25)), np.ones((2, 25)), np.zeros((2, 150))),
                axis=1,
            ),
            True,
        ),
        (
            2,
            np.concatenate((np.zeros((2, 100)), np.ones((2, 100))), axis=1),
            False,
        ),
    ],
)
def test_frame_buffer_find_triggers(
    frame_buffer,
    data_idx,
    buffer_data,
    positive_slope,
    manual_accept,
):
    """
    Verifies positive and negative slope trigger detection, no-trigger
    behavior, and manual-accept waiting behavior.
    """
    frame_buffer.trigger_enabled = True
    frame_buffer.trigger_only_first = False
    frame_buffer.last_trigger = 300
    frame_buffer.manual_accept = manual_accept
    frame_buffer.positive_slope = positive_slope
    frame_buffer._buffer = buffer_data

    triggers = frame_buffer.find_triggers()

    if manual_accept and len(triggers) > 0:
        assert frame_buffer.waiting_for_accept is True
    else:
        assert frame_buffer.waiting_for_accept is False

    match data_idx:
        case 0:
            assert triggers[0] == np.int64(150)
        case 1:
            assert triggers[0] == np.int64(175)
        case 2:
            assert triggers == []


def test_frame_buffer_find_triggers_free_run(frame_buffer):
    """
    Verifies that free-run acquisition returns fixed-spacing frame triggers.
    """
    frame_buffer.trigger_enabled = False
    frame_buffer.last_trigger = 200

    triggers = frame_buffer.find_triggers()

    assert triggers == [100]
    assert frame_buffer.last_trigger == 100


def test_frame_buffer_trigger_only_first(frame_buffer):
    """
    Verifies that trigger-only-first mode does not return subsequent trigger
    events after the first trigger has occurred.
    """
    frame_buffer.trigger_enabled = True
    frame_buffer.trigger_only_first = True
    frame_buffer.first_trigger = False
    frame_buffer.last_trigger = 300
    frame_buffer._buffer = np.concatenate(
        (np.zeros((2, 25)), np.ones((2, 25)), np.zeros((2, 150))),
        axis=1,
    )

    triggers = frame_buffer.find_triggers()

    assert triggers == []


def test_frame_buffer_reset_trigger(frame_buffer):
    """
    Verifies that trigger and reset counters are restored to expected values.
    """
    frame_buffer.last_trigger = 0
    frame_buffer.last_reset = 0

    frame_buffer.reset_trigger()

    assert frame_buffer.last_trigger == 100
    assert frame_buffer.last_reset == 99


def test_frame_buffer_accept(frame_buffer):
    """
    Verifies that acceptance resets counters and clears waiting state.
    """
    frame_buffer.waiting_for_accept = True
    frame_buffer.last_trigger = 0
    frame_buffer.last_reset = 0

    frame_buffer.accept()

    assert frame_buffer.last_trigger == 100
    assert frame_buffer.last_reset == 99
    assert frame_buffer.waiting_for_accept is False


@mock.patch("rattlesnake.process.data_collector.FrameBuffer.find_triggers")
@mock.patch("rattlesnake.process.data_collector.FrameBuffer.add_data")
def test_frame_buffer_add_data_get_frame(
    mock_add_data,
    mock_find_triggers,
    frame_buffer,
):
    """
    Verifies that data are added, triggers are queried, and expected frame data
    are extracted from the buffer.
    """
    mock_find_triggers.return_value = [np.int64(125)]
    buffer_data = np.concatenate(
        (np.zeros((2, 75)), np.ones((2, 25)), np.zeros((2, 100))),
        axis=1,
    )
    frame_buffer._buffer = buffer_data

    data = frame_buffer.add_data_get_frame(buffer_data)

    mock_add_data.assert_called_once_with(buffer_data)
    expected = np.concatenate(
        (np.ones((1, 2, 25)), np.zeros((1, 2, 75))),
        axis=2,
    )
    np.testing.assert_array_equal(data, expected)


def test_frame_buffer_get_item(frame_buffer):
    """
    Verifies indexed buffer access.
    """
    frame_buffer._buffer = np.zeros((2, 1))
    frame_buffer._buffer[1] = 123

    assert frame_buffer[1] == 123


def test_frame_buffer_set_item(frame_buffer):
    """
    Verifies indexed buffer assignment.
    """
    frame_buffer._buffer = np.zeros((2, 1))

    frame_buffer[1] = 123

    assert frame_buffer._buffer[1] == 123


# endregion


# region KurtosisBuffer
def test_kurtosis_buffer_init():
    """
    Verifies that the kurtosis buffer initializes moment arrays.
    """
    kurtosis_buffer = KurtosisBuffer(n_channels=2, averages=4)

    assert kurtosis_buffer.idx == 0
    assert kurtosis_buffer.averages == 4
    assert kurtosis_buffer.g0.shape == (2, 4)
    assert kurtosis_buffer.g1.shape == (2, 4)
    assert kurtosis_buffer.g2.shape == (2, 4)
    assert kurtosis_buffer.g3.shape == (2, 4)
    assert kurtosis_buffer.g4.shape == (2, 4)


def test_kurtosis_buffer_clear():
    """
    Verifies that clearing the kurtosis buffer resets all stored moment sums.
    """
    kurtosis_buffer = KurtosisBuffer(n_channels=2, averages=4)
    kurtosis_buffer.add_data(np.ones((2, 10)))

    kurtosis_buffer.clear()

    assert kurtosis_buffer.idx == 0
    np.testing.assert_array_equal(kurtosis_buffer.g0, np.zeros((2, 4)))
    np.testing.assert_array_equal(kurtosis_buffer.g1, np.zeros((2, 4)))
    np.testing.assert_array_equal(kurtosis_buffer.g2, np.zeros((2, 4)))
    np.testing.assert_array_equal(kurtosis_buffer.g3, np.zeros((2, 4)))
    np.testing.assert_array_equal(kurtosis_buffer.g4, np.zeros((2, 4)))


def test_kurtosis_buffer_add_data_and_get_kurtosis():
    """
    Verifies that data can be added and kurtosis values are returned for each
    channel.
    """
    kurtosis_buffer = KurtosisBuffer(n_channels=2, averages=2)
    data = np.array(
        [
            [-1.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0, 3.0],
        ]
    )

    kurtosis_buffer.add_data(data)

    kurtosis = kurtosis_buffer.get_kurtosis()
    fisher_kurtosis = kurtosis_buffer.get_kurtosis(fisher=True)

    assert kurtosis.shape == (2,)
    np.testing.assert_allclose(fisher_kurtosis, kurtosis - 3)


def test_kurtosis_buffer_wraps_index():
    """
    Verifies that the kurtosis buffer circular index wraps.
    """
    kurtosis_buffer = KurtosisBuffer(n_channels=1, averages=2)

    kurtosis_buffer.add_data(np.ones((1, 4)))
    kurtosis_buffer.add_data(np.ones((1, 4)))
    kurtosis_buffer.add_data(np.ones((1, 4)))

    assert kurtosis_buffer.idx == 1


# endregion


# region Enums
@pytest.mark.parametrize("command_idx", range(1, 10))
def test_data_collector_commands(command_idx):
    """
    Verifies that data collector command enum values construct valid members.
    """
    command = DataCollectorCommands(command_idx)

    assert isinstance(command, DataCollectorCommands)


def test_data_collector_commands_unique_integer_values():
    """
    Verifies that data collector command values are unique integers.
    """
    values = [command.value for command in DataCollectorCommands]

    assert all(isinstance(value, int) for value in values)
    assert len(values) == len(set(values))


def test_data_collector_ui_commands_unique_integer_values():
    """
    Verifies that data collector UI command values are unique integers.
    """
    values = [command.value for command in DataCollectorUICommands]

    assert all(isinstance(value, int) for value in values)
    assert len(values) == len(set(values))


@pytest.mark.parametrize("type_idx", range(3))
def test_acquisition_type(type_idx):
    """
    Verifies that acquisition type enum values construct valid members.
    """
    acquisition_type = AcquisitionType(type_idx)

    assert isinstance(acquisition_type, AcquisitionType)


@pytest.mark.parametrize("acceptance_idx", range(2))
def test_acceptance(acceptance_idx):
    """
    Verifies that acceptance enum values construct valid members.
    """
    acceptance = Acceptance(acceptance_idx)

    assert isinstance(acceptance, Acceptance)


@pytest.mark.parametrize("trigger_idx", range(2))
def test_trigger_slope(trigger_idx):
    """
    Verifies that trigger slope enum values construct valid members.
    """
    trigger_slope = TriggerSlope(trigger_idx)

    assert isinstance(trigger_slope, TriggerSlope)


@pytest.mark.parametrize("window_idx", range(8))
def test_window(window_idx):
    """
    Verifies that window enum values construct valid members.
    """
    window = Window(window_idx)

    assert isinstance(window, Window)


# endregion


# region CollectorMetadata
def test_data_collector_metadata_init(collector_metadata):
    """
    Verifies that collector metadata initializes all expected fields.
    """
    assert isinstance(collector_metadata, CollectorMetadata)
    assert collector_metadata.num_channels == 2
    assert collector_metadata.response_channel_indices == [1]
    assert collector_metadata.reference_channel_indices == [0]
    assert collector_metadata.acquisition_type == AcquisitionType.FREE_RUN
    assert collector_metadata.acceptance == Acceptance.AUTOMATIC
    assert collector_metadata.window == Window.RECTANGLE


def test_data_collector_metadata_eq(collector_metadata):
    """
    Verifies equality comparison for equivalent metadata objects.
    """
    collector_metadata_copy = copy.deepcopy(collector_metadata)

    assert collector_metadata == collector_metadata_copy


def test_data_collector_metadata_eq_false(collector_metadata):
    """
    Verifies equality comparison returns false for non-equivalent metadata.
    """
    other = copy.deepcopy(collector_metadata)
    other.frame_size = collector_metadata.frame_size + 1

    assert collector_metadata != other


def test_data_collector_metadata_eq_incompatible(collector_metadata):
    """
    Verifies equality comparison returns false for incompatible objects.
    """
    assert collector_metadata != object()


# endregion


# region DataCollectorProcess Initialization
def test_data_collector_process_init(
    command_queue,
    environment_command_queue,
    log_file_queue,
):
    """
    Verifies that the data collector process initializes successfully.
    """
    data_in_queue = mp.Queue()
    data_out_queues = [mp.Queue()]
    gui_update_queue = mp.Queue()

    process = DataCollectorProcess(
        "Process Name",
        command_queue,
        data_in_queue,
        data_out_queues,
        environment_command_queue,
        log_file_queue,
        gui_update_queue,
        "Environment Name",
    )

    assert isinstance(process, DataCollectorProcess)
    assert isinstance(process, AbstractMessageProcess)
    assert process.process_name == "Process Name"
    assert process.environment_name == "Environment Name"
    assert process.data_in_queue is data_in_queue
    assert process.data_out_queues is data_out_queues
    assert process.environment_command_queue is environment_command_queue

    assert process.collector_metadata is None
    assert process.frame_buffer is None
    assert process.kurtosis_buffer is None
    assert process.reference_window is None
    assert process.response_window is None
    assert process.acceptance_function is None
    assert process.skip_frames == 0
    assert process.test_level is None
    assert process.last_frame is None
    assert process.window_correction is None


def test_data_collector_process_command_map(data_collector):
    """
    Verifies that data collector commands are mapped to process methods.
    """
    assert data_collector.command_map[DataCollectorCommands.INITIALIZE_COLLECTOR] == (
        data_collector.initialize_collector
    )
    assert (
        data_collector.command_map[DataCollectorCommands.FORCE_INITIALIZE_COLLECTOR]
        == data_collector.force_initialize_collector
    )
    assert data_collector.command_map[DataCollectorCommands.ACQUIRE] == (
        data_collector.acquire
    )
    assert data_collector.command_map[DataCollectorCommands.STOP] == data_collector.stop
    assert data_collector.command_map[DataCollectorCommands.ACCEPT] == (
        data_collector.accept
    )
    assert data_collector.command_map[DataCollectorCommands.SET_TEST_LEVEL] == (
        data_collector.set_test_level
    )
    assert data_collector.command_map[DataCollectorCommands.CLEAR_KURTOSIS_BUFFER] == (
        data_collector.clear_kurtosis_buffer
    )


# endregion


# region Collector Initialization
@mock.patch(
    "rattlesnake.process.data_collector.DataCollectorProcess.force_initialize_collector"
)
def test_data_collector_process_initialize_collector_changed_metadata(
    mock_force_initialize,
    data_collector,
    collector_metadata,
):
    """
    Verifies that changed metadata triggers force initialization.
    """
    data_collector.collector_metadata = None

    data_collector.initialize_collector(collector_metadata)

    mock_force_initialize.assert_called_once_with(collector_metadata)


@mock.patch(
    "rattlesnake.process.data_collector.DataCollectorProcess.force_initialize_collector"
)
def test_data_collector_process_initialize_collector_same_metadata(
    mock_force_initialize,
    data_collector,
    collector_metadata,
):
    """
    Verifies that equivalent metadata does not trigger force initialization.
    """
    data_collector.collector_metadata = copy.deepcopy(collector_metadata)

    data_collector.initialize_collector(collector_metadata)

    mock_force_initialize.assert_not_called()


@pytest.mark.parametrize(
    "window",
    [
        Window.RECTANGLE,
        Window.HANN,
        Window.HAMMING,
        Window.FLATTOP,
        Window.TUKEY,
        Window.BLACKMANHARRIS,
        Window.EXPONENTIAL,
        Window.EXPONENTIAL_FORCE,
    ],
)
@mock.patch("rattlesnake.process.data_collector.flush_queue")
def test_data_collector_process_force_initialize_collector_windows(
    mock_flush_queue,
    data_collector,
    collector_metadata,
    window,
):
    """
    Verifies force initialization for supported window types.
    """
    collector_metadata.window = window
    collector_metadata.frame_size = 32
    collector_metadata.window_parameter_1 = 0.5
    collector_metadata.window_parameter_2 = 8
    collector_metadata.window_parameter_3 = 4
    collector_metadata.kurtosis_buffer_length = 3

    data_collector.force_initialize_collector(collector_metadata)

    assert data_collector.collector_metadata is collector_metadata
    assert isinstance(data_collector.frame_buffer, FrameBuffer)
    assert isinstance(data_collector.kurtosis_buffer, KurtosisBuffer)
    assert data_collector.acceptance_function(np.zeros((2, 32))) is True
    assert data_collector.reference_window is not None
    assert data_collector.response_window is not None
    assert data_collector.window_correction is not None
    mock_flush_queue.assert_called()


def test_data_collector_process_force_initialize_collector_invalid_window(
    data_collector,
    collector_metadata,
):
    """
    Verifies that invalid window types raise ``ValueError``.
    """
    collector_metadata.window = object()

    with pytest.raises(ValueError):
        data_collector.force_initialize_collector(collector_metadata)


@mock.patch("rattlesnake.process.data_collector.load_python_module")
def test_data_collector_process_force_initialize_collector_acceptance_function(
    mock_load_python_module,
    data_collector,
    collector_metadata,
):
    """
    Verifies that an external acceptance function can be loaded.
    """
    acceptance_function = mock.MagicMock(return_value=True)
    module = mock.MagicMock()
    module.accept_frame = acceptance_function
    mock_load_python_module.return_value = module

    collector_metadata.acceptance_function = ("acceptance_module.py", "accept_frame")

    data_collector.force_initialize_collector(collector_metadata)

    mock_load_python_module.assert_called_once_with("acceptance_module.py")
    assert data_collector.acceptance_function is acceptance_function


# endregion


# region Acquire
def configure_collector_for_acquire(data_collector):
    """
    Configure a data collector with mocked runtime state for acquire tests.
    """
    mock_buffer = mock.MagicMock()
    mock_buffer.manual_accept = False
    data_collector.frame_buffer = mock_buffer
    data_collector.acceptance_function = lambda frame: True

    metadata = mock.MagicMock()
    metadata.response_channel_indices = [1]
    metadata.reference_channel_indices = [0]
    metadata.response_transformation_matrix = None
    metadata.reference_transformation_matrix = None
    metadata.kurtosis_buffer_length = None
    data_collector.collector_metadata = metadata

    data_collector.response_window = 1
    data_collector.reference_window = 1
    data_collector.test_level = 1
    data_collector.window_correction = 1
    data_collector.skip_frames = 0

    data_collector._command_queue = mock.MagicMock()
    data_collector._gui_update_queue = mock.MagicMock()


@pytest.mark.parametrize("last_data", [True, False])
@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess.stop")
@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess.log")
def test_data_collector_process_acquire(
    mock_log,
    mock_stop,
    data_collector,
    last_data,
):
    """
    Verifies acquisition, frame processing, GUI updates, output writes, requeue
    behavior, and stop behavior.
    """
    configure_collector_for_acquire(data_collector)

    frame = np.zeros((2, 8))
    frames = frame[np.newaxis, ...]
    data_collector.frame_buffer.add_data_get_frame.return_value = frames

    data_collector.data_in_queue = mock.MagicMock()
    data_collector.data_in_queue.get.return_value = (frame, last_data)

    data_out_queue = mock.MagicMock()
    data_collector.data_out_queues = [data_out_queue]

    data_collector.acquire(None)

    data_collector.frame_buffer.add_data_get_frame.assert_called_once_with(frame)

    matching_call = None
    for call in data_collector.gui_update_queue.put.call_args_list:
        queued_environment_name, queued_payload = call.args[0]
        queued_command, queued_data = queued_payload

        if (
            queued_environment_name == "Environment Name"
            and queued_command == DataCollectorUICommands.TIME_FRAME
        ):
            matching_call = call
            break

    assert matching_call is not None

    queued_environment_name, queued_payload = matching_call.args[0]
    queued_command, queued_data = queued_payload
    queued_frame, accepted = queued_data

    assert queued_environment_name == "Environment Name"
    assert queued_command == DataCollectorUICommands.TIME_FRAME
    np.testing.assert_array_equal(queued_frame, frame)
    assert accepted is True

    data_out_queue.put.assert_called_once()
    response_fft, reference_fft = data_out_queue.put.call_args.args[0]
    np.testing.assert_array_equal(response_fft, np.zeros((1, 5), dtype=complex))
    np.testing.assert_array_equal(reference_fft, np.zeros((1, 5), dtype=complex))

    if last_data:
        mock_stop.assert_called_once_with(None)
        data_collector.command_queue.put.assert_not_called()
    else:
        data_collector.command_queue.put.assert_called_once_with(
            "Process Name",
            (DataCollectorCommands.ACQUIRE, None),
        )

    assert any(
        "Acquired Data with shape" in call.args[0] for call in mock_log.call_args_list
    )
    assert any("Sent Data" in call.args[0] for call in mock_log.call_args_list)


def test_data_collector_process_acquire_empty_queue_requeues(data_collector):
    """
    Verifies that an empty input queue requeues acquisition.
    """
    data_collector.data_in_queue = mock.MagicMock()
    data_collector.data_in_queue.get.side_effect = mp.queues.Empty
    data_collector._command_queue = mock.MagicMock()

    data_collector.acquire(None)

    data_collector.command_queue.put.assert_called_once_with(
        "Process Name",
        (DataCollectorCommands.ACQUIRE, None),
    )


@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess.log")
def test_data_collector_process_acquire_skips_frames(
    mock_log,
    data_collector,
):
    """
    Verifies that configured skip frames are skipped and not sent downstream.
    """
    configure_collector_for_acquire(data_collector)

    frame = np.ones((2, 8))
    frames = frame[np.newaxis, ...]
    data_collector.frame_buffer.add_data_get_frame.return_value = frames
    data_collector.skip_frames = 1

    data_collector.data_in_queue = mock.MagicMock()
    data_collector.data_in_queue.get.return_value = (frame, False)

    data_out_queue = mock.MagicMock()
    data_collector.data_out_queues = [data_out_queue]

    data_collector.acquire(None)

    assert data_collector.skip_frames == 0
    data_out_queue.put.assert_not_called()
    assert any("Skipped Frame" in call.args[0] for call in mock_log.call_args_list)


def test_data_collector_process_acquire_manual_accept(data_collector):
    """
    Verifies that manual acceptance stores the frame and sends an unaccepted GUI
    frame update.
    """
    configure_collector_for_acquire(data_collector)

    frame = np.ones((2, 8))
    frames = frame[np.newaxis, ...]
    data_collector.frame_buffer.add_data_get_frame.return_value = frames
    data_collector.frame_buffer.manual_accept = True

    data_collector.data_in_queue = mock.MagicMock()
    data_collector.data_in_queue.get.return_value = (frame, False)

    data_out_queue = mock.MagicMock()
    data_collector.data_out_queues = [data_out_queue]

    data_collector.acquire(None)

    np.testing.assert_array_equal(data_collector.last_frame, frame)

    matching_call = None
    for call in data_collector.gui_update_queue.put.call_args_list:
        queued_environment_name, queued_payload = call.args[0]
        queued_command, queued_data = queued_payload

        if (
            queued_environment_name == "Environment Name"
            and queued_command == DataCollectorUICommands.TIME_FRAME
        ):
            matching_call = call
            break

    assert matching_call is not None

    queued_environment_name, queued_payload = matching_call.args[0]
    queued_command, queued_data = queued_payload
    queued_frame, accepted = queued_data

    assert queued_environment_name == "Environment Name"
    assert queued_command == DataCollectorUICommands.TIME_FRAME
    np.testing.assert_array_equal(queued_frame, frame)
    assert accepted is False

    data_out_queue.put.assert_not_called()


def test_data_collector_process_acquire_rejected_frame(data_collector):
    """
    Verifies that rejected automatic frames are sent to the GUI as rejected and
    not sent downstream.
    """
    configure_collector_for_acquire(data_collector)
    data_collector.acceptance_function = lambda frame: False

    frame = np.ones((2, 8))
    data_collector.frame_buffer.add_data_get_frame.return_value = frame[np.newaxis, ...]

    data_collector.data_in_queue = mock.MagicMock()
    data_collector.data_in_queue.get.return_value = (frame, False)

    data_out_queue = mock.MagicMock()
    data_collector.data_out_queues = [data_out_queue]

    data_collector.acquire(None)

    matching_call = None
    for call in data_collector.gui_update_queue.put.call_args_list:
        queued_environment_name, queued_payload = call.args[0]
        queued_command, queued_data = queued_payload

        if (
            queued_environment_name == "Environment Name"
            and queued_command == DataCollectorUICommands.TIME_FRAME
        ):
            matching_call = call
            break

    assert matching_call is not None

    queued_environment_name, queued_payload = matching_call.args[0]
    queued_command, queued_data = queued_payload
    queued_frame, accepted = queued_data

    assert queued_environment_name == "Environment Name"
    assert queued_command == DataCollectorUICommands.TIME_FRAME
    np.testing.assert_array_equal(queued_frame, frame)
    assert accepted is False

    data_out_queue.put.assert_not_called()


def test_data_collector_process_acquire_with_transformations(data_collector):
    """
    Verifies that response and reference transformation matrices are applied
    before FFT output.
    """
    configure_collector_for_acquire(data_collector)

    frame = np.vstack([np.ones(8), 2 * np.ones(8)])
    data_collector.frame_buffer.add_data_get_frame.return_value = frame[np.newaxis, ...]

    metadata = data_collector.collector_metadata
    metadata.response_channel_indices = [1]
    metadata.reference_channel_indices = [0]
    metadata.response_transformation_matrix = np.array([[2.0]])
    metadata.reference_transformation_matrix = np.array([[3.0]])

    data_collector.data_in_queue = mock.MagicMock()
    data_collector.data_in_queue.get.return_value = (frame, True)

    data_out_queue = mock.MagicMock()
    data_collector.data_out_queues = [data_out_queue]

    with mock.patch.object(data_collector, "stop"):
        data_collector.acquire(None)

    response_fft, reference_fft = data_out_queue.put.call_args.args[0]

    np.testing.assert_allclose(response_fft[0, 0], 32.0 + 0j)
    np.testing.assert_allclose(reference_fft[0, 0], 24.0 + 0j)


def test_data_collector_process_acquire_with_kurtosis_buffer(data_collector):
    """
    Verifies that accepted frames update the kurtosis buffer and send kurtosis
    GUI updates.
    """
    configure_collector_for_acquire(data_collector)

    frame = np.vstack([np.arange(8.0), np.arange(8.0) + 1])
    data_collector.frame_buffer.add_data_get_frame.return_value = frame[np.newaxis, ...]

    data_collector.collector_metadata.kurtosis_buffer_length = 2
    data_collector.kurtosis_buffer = KurtosisBuffer(2, averages=2)

    data_collector.data_in_queue = mock.MagicMock()
    data_collector.data_in_queue.get.return_value = (frame, False)
    data_collector.data_out_queues = [mock.MagicMock()]

    data_collector.acquire(None)

    assert any(
        call.args[0][1][0] == DataCollectorUICommands.KURTOSIS
        for call in data_collector.gui_update_queue.put.call_args_list
    )


# endregion


# region Accept, Stop, Test Level, Kurtosis Clear
@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess.log")
def test_data_collector_process_accept_true(mock_log, data_collector):
    """
    Verifies manual acceptance logging, GUI updates, FFT output, output queue
    writes, and environment notification.
    """
    frame = np.ones((2, 8))

    data_collector.frame_buffer = mock.MagicMock()
    data_collector.last_frame = frame
    data_collector.window_correction = 1

    metadata = mock.MagicMock()
    metadata.reference_channel_indices = [0]
    metadata.response_channel_indices = [1]
    data_collector.collector_metadata = metadata

    data_collector._gui_update_queue = mock.MagicMock()
    data_out_queue = mock.MagicMock()
    data_collector.data_out_queues = [data_out_queue]
    data_collector.environment_command_queue = mock.MagicMock()

    data_collector.accept(True)

    data_collector.frame_buffer.accept.assert_called_once_with()

    data_collector.gui_update_queue.put.assert_called_once()
    queued_environment_name, queued_payload = (
        data_collector.gui_update_queue.put.call_args.args[0]
    )
    queued_command, queued_data = queued_payload
    queued_frame, accepted = queued_data

    assert queued_environment_name == "Environment Name"
    assert queued_command == DataCollectorUICommands.TIME_FRAME
    np.testing.assert_array_equal(queued_frame, frame)
    assert accepted is True

    response_fft, reference_fft = data_out_queue.put.call_args.args[0]
    np.testing.assert_allclose(response_fft[0, 0], 8.0 + 0j)
    np.testing.assert_allclose(reference_fft[0, 0], 8.0 + 0j)

    data_collector.environment_command_queue.put.assert_called_once_with(
        "Process Name",
        (DataCollectorCommands.ACCEPTED, True),
    )
    assert data_collector.last_frame is None
    mock_log.assert_has_calls(
        [
            mock.call("Received Accept Signal True"),
            mock.call("Sending data manually"),
            mock.call("Sent Data"),
        ]
    )


@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess.log")
def test_data_collector_process_accept_false(mock_log, data_collector):
    """
    Verifies that rejected manual frames are not sent downstream but still
    notify the environment.
    """
    data_collector.frame_buffer = mock.MagicMock()
    data_collector.last_frame = np.ones((2, 8))
    data_collector._gui_update_queue = mock.MagicMock()

    data_out_queue = mock.MagicMock()
    data_collector.data_out_queues = [data_out_queue]
    data_collector.environment_command_queue = mock.MagicMock()

    data_collector.accept(False)

    data_collector.frame_buffer.accept.assert_called_once_with()

    data_out_queue.put.assert_not_called()
    data_collector.gui_update_queue.put.assert_not_called()

    data_collector.environment_command_queue.put.assert_called_once_with(
        "Process Name",
        (DataCollectorCommands.ACCEPTED, False),
    )

    assert data_collector.last_frame is None

    mock_log.assert_called_once_with("Received Accept Signal False")


@mock.patch("rattlesnake.process.data_collector.sleep")
@mock.patch("rattlesnake.process.data_collector.flush_queue")
@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess.log")
def test_data_collector_process_stop(
    mock_log,
    mock_flush_queue,
    mock_sleep,
    data_collector,
):
    """
    Verifies shutdown logging, output queue flushing, command queue flushing,
    trigger reset, and environment shutdown notification.
    """
    data_out_queue = mock.MagicMock()
    data_collector.data_out_queues = [data_out_queue]
    data_collector._command_queue = mock.MagicMock()
    data_collector.frame_buffer = mock.MagicMock()
    data_collector.environment_command_queue = mock.MagicMock()

    data_collector.stop(None)

    mock_sleep.assert_called_once_with(0.05)
    mock_log.assert_called_once_with("Stopping Data Collection")
    mock_flush_queue.assert_called_once_with(data_out_queue)
    data_collector.command_queue.flush.assert_called_once_with("Process Name")
    data_collector.frame_buffer.reset_trigger.assert_called_once_with()
    data_collector.environment_command_queue.put.assert_called_once_with(
        "Process Name",
        (DataCollectorCommands.SHUTDOWN_ACHIEVED, None),
    )


@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess.log")
def test_data_collector_process_set_test_level(mock_log, data_collector):
    """
    Verifies that skip-frame count and test level are stored and logged.
    """
    data_collector.set_test_level((10, 0.1))

    assert data_collector.skip_frames == 10
    assert data_collector.test_level == 0.1
    mock_log.assert_called_once_with(
        "Setting Test Level to 0.1, skipping next 10 frames"
    )


def test_data_collector_process_clear_kurtosis_buffer(data_collector):
    """
    Verifies that the kurtosis buffer is cleared when present.
    """
    data_collector.kurtosis_buffer = mock.MagicMock()

    data_collector.clear_kurtosis_buffer(None)

    data_collector.kurtosis_buffer.clear.assert_called_once_with()


def test_data_collector_process_clear_kurtosis_buffer_none(data_collector):
    """
    Verifies that clearing a missing kurtosis buffer does not raise.
    """
    data_collector.kurtosis_buffer = None

    data_collector.clear_kurtosis_buffer(None)

    assert True


# endregion


# region data_collector_process
@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess")
def test_data_collector_process_function(
    mock_data_collector_process_class,
    command_queue,
    environment_command_queue,
    log_file_queue,
):
    """
    Verifies that the process function constructs a data collector process and
    starts its command loop.
    """
    data_in_queue = mp.Queue()
    data_out_queues = [mp.Queue()]
    gui_update_queue = mp.Queue()

    data_collector_process(
        "Environment Name",
        command_queue,
        data_in_queue,
        data_out_queues,
        environment_command_queue,
        log_file_queue,
        gui_update_queue,
        "Process Name",
    )

    mock_data_collector_process_class.assert_called_once_with(
        "Process Name",
        command_queue,
        data_in_queue,
        data_out_queues,
        environment_command_queue,
        log_file_queue,
        gui_update_queue,
        "Environment Name",
    )
    mock_instance = mock_data_collector_process_class.return_value
    mock_instance.run.assert_called_once_with()


@mock.patch("rattlesnake.process.data_collector.DataCollectorProcess")
def test_data_collector_process_function_default_process_name(
    mock_data_collector_process_class,
    command_queue,
    environment_command_queue,
    log_file_queue,
):
    """
    Verifies that the default process name is generated from the environment
    name.
    """
    data_collector_process(
        "Environment Name",
        command_queue,
        mp.Queue(),
        [mp.Queue()],
        environment_command_queue,
        log_file_queue,
        mp.Queue(),
    )

    assert mock_data_collector_process_class.call_args.args[0] == (
        "Environment Name Data Collector"
    )


# endregion
