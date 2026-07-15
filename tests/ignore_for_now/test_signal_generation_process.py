import multiprocessing as mp
from unittest import mock

import numpy as np
import pytest

from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.process.signal_generation import SignalGenerator
from rattlesnake.process.signal_generation_process import (
    SignalGenerationCommands,
    SignalGenerationMetadata,
    SignalGenerationProcess,
    signal_generation_process,
)
from rattlesnake.user_interface.ui_utilities import UICommands
from rattlesnake.utilities import VerboseMessageQueue


# region Fixtures
@pytest.fixture
def log_file_queue():
    return mp.Queue()


@pytest.fixture
def command_queue(log_file_queue):
    return VerboseMessageQueue(
        log_file_queue,
        mp.Queue(),
        "Signal Generation Command Queue",
    )


@pytest.fixture
def environment_command_queue(log_file_queue):
    return VerboseMessageQueue(
        log_file_queue,
        mp.Queue(),
        "Environment Command Queue",
    )


@pytest.fixture
def signal_generation_metadata():
    return SignalGenerationMetadata(
        samples_per_write=100,
        level_ramp_samples=20,
        output_transformation_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )


@pytest.fixture
def signal_generation_process_obj(
    command_queue,
    environment_command_queue,
    log_file_queue,
):
    return SignalGenerationProcess(
        "Process Name",
        command_queue,
        mp.Queue(),
        mp.Queue(),
        environment_command_queue,
        log_file_queue,
        mp.Queue(),
        "Environment Name",
    )


# endregion


# region Commands
@pytest.mark.parametrize("command_idx", range(9))
def test_signal_generation_commands(command_idx):
    """
    Verifies that signal generation command enum values construct valid
    ``SignalGenerationCommands`` members.
    """
    signal_command = SignalGenerationCommands(command_idx)

    assert isinstance(signal_command, SignalGenerationCommands)


def test_signal_generation_commands_unique_integer_values():
    """
    Verifies that signal generation command values are unique integers.
    """
    values = [command.value for command in SignalGenerationCommands]

    assert all(isinstance(value, int) for value in values)
    assert len(values) == len(set(values))


# endregion


# region Metadata
def test_signal_generation_metadata_init():
    """
    Verifies that signal generation metadata initializes successfully.
    """
    metadata = SignalGenerationMetadata(
        samples_per_write=100,
        level_ramp_samples=20,
    )

    assert isinstance(metadata, SignalGenerationMetadata)
    assert metadata.samples_per_write == 100
    assert metadata.ramp_samples == 20
    assert metadata.output_transformation_matrix is None
    assert metadata.new_signal_sample_threshold == 100
    assert metadata.disabled_signals == []


def test_signal_generation_metadata_init_with_optional_values():
    """
    Verifies that optional metadata values are stored.
    """
    transformation_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

    metadata = SignalGenerationMetadata(
        samples_per_write=100,
        level_ramp_samples=20,
        output_transformation_matrix=transformation_matrix,
        new_signal_sample_threshold=50,
        disabled_signals=[1],
    )

    assert metadata.samples_per_write == 100
    assert metadata.ramp_samples == 20
    assert metadata.output_transformation_matrix is transformation_matrix
    assert metadata.new_signal_sample_threshold == 50
    assert metadata.disabled_signals == [1]


def test_signal_generation_metadata_eq(signal_generation_metadata):
    """
    Verifies equality comparison for equivalent metadata objects.
    """
    expected_metadata = SignalGenerationMetadata(
        samples_per_write=100,
        level_ramp_samples=20,
        output_transformation_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    assert signal_generation_metadata == expected_metadata


def test_signal_generation_metadata_eq_false(signal_generation_metadata):
    """
    Verifies equality comparison returns false for different metadata.
    """
    other = SignalGenerationMetadata(
        samples_per_write=200,
        level_ramp_samples=20,
        output_transformation_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    assert signal_generation_metadata != other


def test_signal_generation_metadata_eq_incompatible(signal_generation_metadata):
    """
    Verifies equality comparison returns false for incompatible objects.
    """
    assert signal_generation_metadata != object()


# endregion


# region Process Initialization
def test_signal_generation_process_init(
    command_queue,
    environment_command_queue,
    log_file_queue,
):
    """
    Verifies that the signal generation process initializes successfully.
    """
    data_in_queue = mp.Queue()
    data_out_queue = mp.Queue()
    gui_update_queue = mp.Queue()

    process = SignalGenerationProcess(
        "Process Name",
        command_queue,
        data_in_queue,
        data_out_queue,
        environment_command_queue,
        log_file_queue,
        gui_update_queue,
        "Environment Name",
    )

    assert isinstance(process, SignalGenerationProcess)
    assert isinstance(process, AbstractMessageProcess)

    assert process.process_name == "Process Name"
    assert process.environment_name == "Environment Name"
    assert process.data_in_queue is data_in_queue
    assert process.data_out_queue is data_out_queue
    assert process.environment_command_queue is environment_command_queue

    assert process.ramp_samples is None
    assert process.output_transformation_matrix is None
    assert process.samples_per_write is None
    assert process.new_signal_sample_threshold is None
    assert process.test_level_target == 1.0
    assert process.current_test_level == 0.0
    assert process.test_level_change == 0.0
    assert process.signal_remainder is None
    assert process.startup is True
    assert process.shutdown_flag is False
    assert process.done_generating is False
    assert process.signal_generator is None
    assert process.disabled_signals is None


def test_signal_generation_process_command_map(signal_generation_process_obj):
    """
    Verifies that signal generation commands are mapped to process methods.
    """
    assert (
        signal_generation_process_obj.command_map[
            SignalGenerationCommands.INITIALIZE_PARAMETERS
        ]
        == signal_generation_process_obj.initialize_parameters
    )
    assert (
        signal_generation_process_obj.command_map[
            SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR
        ]
        == signal_generation_process_obj.initialize_signal_generator
    )
    assert (
        signal_generation_process_obj.command_map[
            SignalGenerationCommands.GENERATE_SIGNALS
        ]
        == signal_generation_process_obj.generate_signals
    )
    assert (
        signal_generation_process_obj.command_map[
            SignalGenerationCommands.START_SHUTDOWN
        ]
        == signal_generation_process_obj.start_shutdown
    )
    assert (
        signal_generation_process_obj.command_map[SignalGenerationCommands.SHUTDOWN]
        == signal_generation_process_obj.shutdown
    )
    assert (
        signal_generation_process_obj.command_map[SignalGenerationCommands.MUTE]
        == signal_generation_process_obj.mute
    )
    assert (
        signal_generation_process_obj.command_map[
            SignalGenerationCommands.ADJUST_TEST_LEVEL
        ]
        == signal_generation_process_obj.adjust_test_level
    )
    assert (
        signal_generation_process_obj.command_map[
            SignalGenerationCommands.SET_TEST_LEVEL
        ]
        == signal_generation_process_obj.set_test_level
    )


# endregion


# region Parameter and Signal Generator Initialization
@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.log")
def test_signal_generation_process_initialize_parameters(
    mock_log,
    signal_generation_process_obj,
    signal_generation_metadata,
):
    """
    Verifies that metadata values are stored and transformation matrix is
    pseudoinverted.
    """
    signal_generation_process_obj.initialize_parameters(signal_generation_metadata)

    mock_log.assert_called_once_with("Initializing Test Parameters")
    assert signal_generation_process_obj.ramp_samples == 20
    assert signal_generation_process_obj.samples_per_write == 100
    assert signal_generation_process_obj.new_signal_sample_threshold == 100
    assert signal_generation_process_obj.disabled_signals == []
    np.testing.assert_array_almost_equal(
        signal_generation_process_obj.output_transformation_matrix,
        np.linalg.pinv(np.array([[1.0, 2.0], [3.0, 4.0]])),
    )


@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.log")
def test_signal_generation_process_initialize_parameters_no_transformation(
    mock_log,
    signal_generation_process_obj,
):
    """
    Verifies parameter initialization without a transformation matrix.
    """
    metadata = SignalGenerationMetadata(
        samples_per_write=100,
        level_ramp_samples=20,
        output_transformation_matrix=None,
        new_signal_sample_threshold=30,
        disabled_signals=[1],
    )

    signal_generation_process_obj.initialize_parameters(metadata)

    assert signal_generation_process_obj.output_transformation_matrix is None
    assert signal_generation_process_obj.new_signal_sample_threshold == 30
    assert signal_generation_process_obj.disabled_signals == [1]


def test_signal_generation_process_initialize_signal_generator(
    signal_generation_process_obj,
):
    """
    Verifies that the signal generator is stored and signal remainder is
    cleared.
    """
    signal_generation_process_obj.signal_remainder = np.ones((1, 10))
    signal_generator = mock.MagicMock(spec=SignalGenerator)

    signal_generation_process_obj.initialize_signal_generator(signal_generator)

    assert signal_generation_process_obj.signal_generator is signal_generator
    assert signal_generation_process_obj.signal_remainder is None


# endregion


# region Generate Signals
def test_signal_generation_process_generate_signals_no_generator(
    signal_generation_process_obj,
):
    """
    Verifies that generation raises when no signal generator has been
    initialized.
    """
    signal_generation_process_obj.signal_generator = None

    with pytest.raises(RuntimeError):
        signal_generation_process_obj.generate_signals(None)


@mock.patch("rattlesnake.process.signal_generation_process.flush_queue")
@mock.patch(
    "rattlesnake.process.signal_generation_process.SignalGenerationProcess.output"
)
@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.log")
def test_signal_generation_process_generate_signals(
    mock_log,
    mock_output,
    mock_flush_queue,
    signal_generation_process_obj,
):
    """
    Verifies startup parameter acquisition, update handling, frame generation,
    output, and command requeueing.
    """
    signal_generation_process_obj.startup = True
    signal_generation_process_obj.shutdown_flag = False
    signal_generation_process_obj.samples_per_write = 100
    signal_generation_process_obj.new_signal_sample_threshold = 100
    signal_generation_process_obj.current_test_level = 1.0

    mock_signal_generator = mock.MagicMock()
    mock_signal_generator.ready_for_next_output = False

    def update_parameters_side_effect(*args):
        mock_signal_generator.ready_for_next_output = True

    mock_signal_generator.update_parameters.side_effect = update_parameters_side_effect
    mock_signal_generator.generate_frame.return_value = (np.ones((2, 100)), False)

    signal_generation_process_obj.signal_generator = mock_signal_generator

    mock_data_in_queue = mock.MagicMock()
    mock_data_in_queue.get.return_value = ("Params Data",)
    signal_generation_process_obj.data_in_queue = mock_data_in_queue

    mock_data_out_queue = mock.MagicMock()
    mock_data_out_queue.empty.return_value = True
    signal_generation_process_obj.data_out_queue = mock_data_out_queue

    signal_generation_process_obj._command_queue = mock.MagicMock()

    mock_flush_queue.return_value = [("Update Data",)]

    signal_generation_process_obj.generate_signals(None)

    mock_signal_generator.update_parameters.assert_has_calls(
        [
            mock.call("Params Data"),
            mock.call("Update Data"),
        ]
    )
    mock_signal_generator.generate_frame.assert_called_once_with()

    mock_output.assert_called_once()
    np.testing.assert_array_equal(mock_output.call_args.args[0], np.ones((2, 100)))
    assert mock_output.call_args.args[1] is False

    signal_generation_process_obj.command_queue.put.assert_called_once_with(
        "Process Name",
        (SignalGenerationCommands.GENERATE_SIGNALS, None),
    )

    mock_log.assert_has_calls(
        [
            mock.call("Starting up output"),
            mock.call("Waiting for Input Data"),
            mock.call("Got Updated Parameters"),
            mock.call("Generating Frame of Data"),
            mock.call("Generated Signal with RMS \n  [1. 1.]"),
            mock.call("Outputting Data"),
        ]
    )


def test_signal_generation_process_generate_signals_startup_timeout(
    signal_generation_process_obj,
):
    """
    Verifies that startup timeout while waiting for first parameters sends a GUI
    error and returns.
    """
    signal_generation_process_obj.startup = True

    mock_signal_generator = mock.MagicMock()
    mock_signal_generator.ready_for_next_output = False
    signal_generation_process_obj.signal_generator = mock_signal_generator

    mock_data_in_queue = mock.MagicMock()
    mock_data_in_queue.get.side_effect = mp.queues.Empty
    signal_generation_process_obj.data_in_queue = mock_data_in_queue

    signal_generation_process_obj._gui_update_queue = mock.MagicMock()

    signal_generation_process_obj.generate_signals(None)

    signal_generation_process_obj.gui_update_queue.put.assert_called_once()
    gui_message, gui_data = (
        signal_generation_process_obj.gui_update_queue.put.call_args.args[0]
    )

    assert gui_message == UICommands.ERROR
    assert gui_data[0] == "Process Name Error"
    assert "timed out while waiting for first set of parameters" in gui_data[1]


@mock.patch(
    "rattlesnake.process.signal_generation_process.SignalGenerationProcess.shutdown"
)
@mock.patch(
    "rattlesnake.process.signal_generation_process.SignalGenerationProcess.output"
)
def test_signal_generation_process_generate_signals_last_run_shutdown(
    mock_output,
    mock_shutdown,
    signal_generation_process_obj,
):
    """
    Verifies that a final output chunk triggers shutdown and is not requeued.
    """
    signal_generation_process_obj.startup = False
    signal_generation_process_obj.shutdown_flag = True
    signal_generation_process_obj.current_test_level = 0.0
    signal_generation_process_obj.samples_per_write = 10
    signal_generation_process_obj.new_signal_sample_threshold = 10
    signal_generation_process_obj.signal_remainder = np.ones((1, 10))
    signal_generation_process_obj.done_generating = False

    mock_signal_generator = mock.MagicMock()
    mock_signal_generator.ready_for_next_output = False
    signal_generation_process_obj.signal_generator = mock_signal_generator

    mock_data_out_queue = mock.MagicMock()
    mock_data_out_queue.empty.return_value = True
    signal_generation_process_obj.data_out_queue = mock_data_out_queue

    signal_generation_process_obj._command_queue = mock.MagicMock()

    signal_generation_process_obj.generate_signals(None)

    mock_output.assert_called_once()
    assert mock_output.call_args.args[1] is True
    mock_shutdown.assert_called_once_with()
    signal_generation_process_obj.command_queue.put.assert_not_called()


@mock.patch("rattlesnake.process.signal_generation_process.flush_queue")
@mock.patch(
    "rattlesnake.process.signal_generation_process.SignalGenerationProcess.output"
)
def test_signal_generation_process_generate_signals_appends_remainder(
    mock_output,
    mock_flush_queue,
    signal_generation_process_obj,
):
    """
    Verifies that newly generated data are concatenated onto existing signal
    remainder before output.
    """
    signal_generation_process_obj.startup = False
    signal_generation_process_obj.shutdown_flag = False
    signal_generation_process_obj.samples_per_write = 5
    signal_generation_process_obj.new_signal_sample_threshold = 10
    signal_generation_process_obj.signal_remainder = np.ones((1, 4))
    signal_generation_process_obj.done_generating = False

    mock_signal_generator = mock.MagicMock()
    mock_signal_generator.ready_for_next_output = True
    mock_signal_generator.generate_frame.return_value = (2 * np.ones((1, 6)), False)
    signal_generation_process_obj.signal_generator = mock_signal_generator

    mock_data_out_queue = mock.MagicMock()
    mock_data_out_queue.empty.return_value = True
    signal_generation_process_obj.data_out_queue = mock_data_out_queue

    signal_generation_process_obj._command_queue = mock.MagicMock()

    mock_flush_queue.return_value = []

    signal_generation_process_obj.generate_signals(None)

    mock_signal_generator.generate_frame.assert_called_once_with()
    np.testing.assert_array_equal(
        mock_output.call_args.args[0],
        np.concatenate((np.ones((1, 4)), 2 * np.ones((1, 1))), axis=-1),
    )


# endregion


# region Output and Test Level Control
@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.log")
def test_signal_generation_process_output_ramp_and_disabled_signals(
    mock_log,
    signal_generation_process_obj,
):
    """
    Verifies disabled-signal handling, test-level ramping, and output queue
    data.
    """
    write_data = np.ones((2, 50))

    expected_enabled = np.concatenate(
        (
            np.arange(0.95, -0.05, -0.05).reshape(1, -1),
            np.zeros((1, 30)),
        ),
        axis=1,
    )
    expected_disabled = np.zeros((1, 50))
    expected_data = np.concatenate((expected_enabled, expected_disabled), axis=0)

    signal_generation_process_obj.disabled_signals = [1]
    signal_generation_process_obj.output_transformation_matrix = None
    signal_generation_process_obj.current_test_level = 1.0
    signal_generation_process_obj.test_level_target = 0.0
    signal_generation_process_obj.test_level_change = -0.05

    mock_data_out_queue = mock.MagicMock()
    signal_generation_process_obj.data_out_queue = mock_data_out_queue

    signal_generation_process_obj.output(write_data, last_signal=False)

    mock_log.assert_has_calls(
        [
            mock.call("Test level from 0.95 to 0.0"),
            mock.call("Sending data to data_out queue"),
        ]
    )

    queued_data, last_signal = mock_data_out_queue.put.call_args.args[0]
    np.testing.assert_array_almost_equal(queued_data, expected_data)
    assert last_signal is False
    assert signal_generation_process_obj.current_test_level == 0.0
    assert signal_generation_process_obj.test_level_change == 0.0


@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.log")
def test_signal_generation_process_output_constant_level(
    mock_log,
    signal_generation_process_obj,
):
    """
    Verifies constant test-level output.
    """
    write_data = np.ones((2, 10))

    signal_generation_process_obj.disabled_signals = []
    signal_generation_process_obj.output_transformation_matrix = None
    signal_generation_process_obj.current_test_level = 0.5
    signal_generation_process_obj.test_level_change = 0.0

    mock_data_out_queue = mock.MagicMock()
    signal_generation_process_obj.data_out_queue = mock_data_out_queue

    signal_generation_process_obj.output(write_data, last_signal=True)

    queued_data, last_signal = mock_data_out_queue.put.call_args.args[0]

    np.testing.assert_array_equal(queued_data, 0.5 * np.ones((2, 10)))
    assert last_signal is True
    mock_log.assert_any_call("Test Level at 0.5")


@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.log")
def test_signal_generation_process_output_with_transformation(
    mock_log,
    signal_generation_process_obj,
):
    """
    Verifies that output transformation is applied before test-level scaling.
    """
    write_data = np.ones((2, 4))

    signal_generation_process_obj.disabled_signals = []
    signal_generation_process_obj.output_transformation_matrix = np.array([[1.0, 2.0]])
    signal_generation_process_obj.current_test_level = 1.0
    signal_generation_process_obj.test_level_change = 0.0

    mock_data_out_queue = mock.MagicMock()
    signal_generation_process_obj.data_out_queue = mock_data_out_queue

    signal_generation_process_obj.output(write_data)

    queued_data, _ = mock_data_out_queue.put.call_args.args[0]
    np.testing.assert_array_equal(queued_data, 3 * np.ones((1, 4)))
    mock_log.assert_any_call("Applying Transformation")


def test_signal_generation_process_mute(signal_generation_process_obj):
    """
    Verifies that muting resets test-level state to zero.
    """
    signal_generation_process_obj.current_test_level = 1.0
    signal_generation_process_obj.test_level_target = 1.0
    signal_generation_process_obj.test_level_change = 1.0

    signal_generation_process_obj.mute(None)

    assert signal_generation_process_obj.current_test_level == 0.0
    assert signal_generation_process_obj.test_level_target == 0.0
    assert signal_generation_process_obj.test_level_change == 0.0


def test_signal_generation_process_set_test_level(signal_generation_process_obj):
    """
    Verifies that setting test level updates current and target values.
    """
    signal_generation_process_obj.set_test_level(1.5)

    assert signal_generation_process_obj.current_test_level == 1.5
    assert signal_generation_process_obj.test_level_target == 1.5
    assert signal_generation_process_obj.test_level_change == 0.0


@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.log")
def test_signal_generation_process_adjust_test_level(
    mock_log,
    signal_generation_process_obj,
):
    """
    Verifies that adjusting test level computes per-sample ramp change and
    logs it.
    """
    signal_generation_process_obj.current_test_level = 2.0
    signal_generation_process_obj.ramp_samples = 20

    signal_generation_process_obj.adjust_test_level(1.0)

    assert signal_generation_process_obj.test_level_target == 1.0
    assert signal_generation_process_obj.test_level_change == -0.05
    mock_log.assert_called_once_with(
        "Changed test level from 2.0 to 1.0, -0.05 change per sample"
    )


def test_signal_generation_process_adjust_test_level_no_change(
    signal_generation_process_obj,
):
    """
    Verifies that adjusting to the current test level does not log.
    """
    signal_generation_process_obj.current_test_level = 1.0
    signal_generation_process_obj.ramp_samples = 20

    with mock.patch.object(signal_generation_process_obj, "log") as mock_log:
        signal_generation_process_obj.adjust_test_level(1.0)

    assert signal_generation_process_obj.test_level_target == 1.0
    assert signal_generation_process_obj.test_level_change == 0.0
    mock_log.assert_not_called()


# endregion


# region Shutdown
@mock.patch(
    "rattlesnake.process.signal_generation_process.SignalGenerationProcess.adjust_test_level"
)
def test_signal_generation_process_start_shutdown(
    mock_adjust_test_level,
    signal_generation_process_obj,
):
    """
    Verifies shutdown flag behavior, command flushing, level ramp initiation,
    and generation command requeueing.
    """
    signal_generation_process_obj.shutdown_flag = False
    signal_generation_process_obj.startup = False
    signal_generation_process_obj._command_queue = mock.MagicMock()
    signal_generation_process_obj.command_queue.flush.return_value = [
        (SignalGenerationCommands.GENERATE_SIGNALS, None)
    ]

    signal_generation_process_obj.start_shutdown(None)

    assert signal_generation_process_obj.shutdown_flag is True
    mock_adjust_test_level.assert_called_once_with(0.0)
    signal_generation_process_obj.command_queue.flush.assert_called_once_with(
        "Process Name"
    )
    signal_generation_process_obj.command_queue.put.assert_called_once_with(
        "Process Name",
        (SignalGenerationCommands.GENERATE_SIGNALS, None),
    )


@mock.patch(
    "rattlesnake.process.signal_generation_process.SignalGenerationProcess.adjust_test_level"
)
def test_signal_generation_process_start_shutdown_ignored_when_startup(
    mock_adjust_test_level,
    signal_generation_process_obj,
):
    """
    Verifies shutdown request is ignored while still in startup.
    """
    signal_generation_process_obj.shutdown_flag = False
    signal_generation_process_obj.startup = True
    signal_generation_process_obj._command_queue = mock.MagicMock()

    signal_generation_process_obj.start_shutdown(None)

    assert signal_generation_process_obj.shutdown_flag is False
    mock_adjust_test_level.assert_not_called()
    signal_generation_process_obj.command_queue.flush.assert_not_called()


@mock.patch(
    "rattlesnake.process.signal_generation_process.SignalGenerationProcess.adjust_test_level"
)
def test_signal_generation_process_start_shutdown_ignored_when_already_shutdown(
    mock_adjust_test_level,
    signal_generation_process_obj,
):
    """
    Verifies shutdown request is ignored when shutdown is already in progress.
    """
    signal_generation_process_obj.shutdown_flag = True
    signal_generation_process_obj.startup = False
    signal_generation_process_obj._command_queue = mock.MagicMock()

    signal_generation_process_obj.start_shutdown(None)

    mock_adjust_test_level.assert_not_called()
    signal_generation_process_obj.command_queue.flush.assert_not_called()


@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.log")
def test_signal_generation_process_shutdown(
    mock_log,
    signal_generation_process_obj,
):
    """
    Verifies shutdown logging, command queue flushing, environment
    notification, and state reset.
    """
    signal_generation_process_obj._command_queue = mock.MagicMock()
    signal_generation_process_obj.environment_command_queue = mock.MagicMock()

    signal_generation_process_obj.startup = False
    signal_generation_process_obj.shutdown_flag = True
    signal_generation_process_obj.done_generating = True

    signal_generation_process_obj.shutdown()

    mock_log.assert_called_once_with("Shutting Down Signal Generation")
    signal_generation_process_obj.command_queue.flush.assert_called_once_with(
        "Process Name"
    )
    signal_generation_process_obj.environment_command_queue.put.assert_called_once_with(
        "Process Name",
        (SignalGenerationCommands.SHUTDOWN_ACHIEVED, None),
    )
    assert signal_generation_process_obj.startup is True
    assert signal_generation_process_obj.shutdown_flag is False
    assert signal_generation_process_obj.done_generating is False


# endregion


# region signal_generation_process
@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess")
def test_signal_generation_process_func(
    mock_signal_generation_process_class,
    command_queue,
    environment_command_queue,
    log_file_queue,
):
    """
    Verifies that the process function constructs a signal generation process
    and starts its command loop.
    """
    data_in_queue = mp.Queue()
    data_out_queue = mp.Queue()
    gui_update_queue = mp.Queue()

    signal_generation_process(
        "Environment Name",
        command_queue,
        data_in_queue,
        data_out_queue,
        environment_command_queue,
        log_file_queue,
        gui_update_queue,
        "Process Name",
    )

    mock_signal_generation_process_class.assert_called_once_with(
        "Process Name",
        command_queue,
        data_in_queue,
        data_out_queue,
        environment_command_queue,
        log_file_queue,
        gui_update_queue,
        "Environment Name",
    )
    mock_signal_generation_process_class.return_value.run.assert_called_once_with()


@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess")
def test_signal_generation_process_func_default_process_name(
    mock_signal_generation_process_class,
    command_queue,
    environment_command_queue,
    log_file_queue,
):
    """
    Verifies that the default process name is generated from the environment
    name.
    """
    signal_generation_process(
        "Environment Name",
        command_queue,
        mp.Queue(),
        mp.Queue(),
        environment_command_queue,
        log_file_queue,
        mp.Queue(),
    )

    assert mock_signal_generation_process_class.call_args.args[0] == (
        "Environment Name Signal Generation"
    )


# endregion
