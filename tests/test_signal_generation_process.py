import multiprocessing as mp
from unittest import mock

import numpy as np
import pytest

# from PyQt5 import QtWidgets  # unused import

from rattlesnake.process.signal_generation_process import (
    SignalGenerationCommands,
    SignalGenerationMetadata,
    SignalGenerationProcess,
    signal_generation_process,
)
from rattlesnake.utilities import VerboseMessageQueue


# Create log_file_queue
@pytest.fixture()
def log_file_queue():
    return mp.Queue()


@pytest.fixture()
def signal_generation_metadata():
    return SignalGenerationMetadata(100, 20, np.array([[1, 2], [3, 4]]))


@pytest.fixture()
def signal_generation_process_obj():
    signal_generation_process_obj = SignalGenerationProcess(
        "Process Name",
        VerboseMessageQueue(log_file_queue, "Spectral Command Queue"),
        mp.Queue(),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        log_file_queue,
        mp.Queue(),
        "Environment Name",
    )
    return signal_generation_process_obj


@pytest.mark.parametrize("signal_idx", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_signal_generation_commands(signal_idx):
    signal_command = SignalGenerationCommands(signal_idx)

    assert isinstance(signal_command, SignalGenerationCommands)


def test_signal_generation_metadata_init():
    signal_generation_metadata = SignalGenerationMetadata(100, 20)

    assert isinstance(signal_generation_metadata, SignalGenerationMetadata)


def test_signal_generation_metadata_eq(signal_generation_metadata):
    assert_metadata = SignalGenerationMetadata(100, 20, np.array([[1, 2], [3, 4]]))

    assert signal_generation_metadata == assert_metadata


# Test the SignalGeneratorProcess intialization
def test_signal_generation_process_init(log_file_queue):
    signal_generation_process = SignalGenerationProcess(
        "Process Name",
        VerboseMessageQueue(log_file_queue, "Spectral Command Queue"),
        mp.Queue(),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        log_file_queue,
        mp.Queue(),
        "Environment Name",
    )

    # Test if object is the correct class
    assert isinstance(signal_generation_process, SignalGenerationProcess)


@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.log")
def test_signal_generation_process_initialize_parameters(
    mock_log, signal_generation_process_obj, signal_generation_metadata
):
    signal_generation_process_obj.initialize_parameters(signal_generation_metadata)

    mock_log.assert_called_with("Initializing Test Parameters")
    assert signal_generation_process_obj.ramp_samples == 20
    np.testing.assert_array_almost_equal(
        signal_generation_process_obj.output_transformation_matrix,
        np.linalg.pinv(np.array([[1, 2], [3, 4]])),
    )
    assert signal_generation_process_obj.samples_per_write == 100


def test_signal_generation_process_initialize_signal_generator(
    signal_generation_process_obj,
):
    signal_generation_process_obj.initialize_signal_generator("Signal Generator")

    assert signal_generation_process_obj.signal_generator == "Signal Generator"
    assert signal_generation_process_obj.signal_remainder == None


@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.shutdown")
@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.output")
@mock.patch("rattlesnake.process.signal_generation_process.flush_queue")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.log")
def test_signal_generation_process_generate_signals(
    mock_log,
    mock_vput,
    mock_flush,
    mock_output,
    mock_shutdown,
    signal_generation_process_obj,
):
    params_data = ("Params Data",)
    signal_generation_process_obj.startup = True
    mock_siggen = mock.MagicMock()
    mock_siggen.ready_for_next_output = False
    mock_siggen.generate_frame.return_value = (np.ones((2, 100)), True)
    signal_generation_process_obj.signal_generator = mock_siggen
    mock_data_in = mock.MagicMock()
    mock_data_in.get.return_value = params_data
    signal_generation_process_obj.data_in_queue = mock_data_in
    mock_data_out = mock.MagicMock()
    mock_data_out.empty.return_value = True
    signal_generation_process_obj.data_out_queue = mock_data_out
    mock_flush.return_value = [("Update Data",)]
    signal_generation_process_obj.shutdown_flag = False

    signal_generation_process_obj.generate_signals(None)
    mock_siggen.generate_frame.assert_not_called()
    assert not signal_generation_process_obj.startup
    mock_siggen.ready_for_next_output = (
        True  # Assume that reading in the data now allows a frame to be generated.
    )
    signal_generation_process_obj.generate_signals(None)

    mock_siggen.generate_frame.assert_called()
    param_calls = [mock.call("Params Data"), mock.call("Update Data")]
    mock_siggen.update_parameters.assert_has_calls(param_calls)
    log_calls = [
        mock.call("Starting up output"),
        mock.call("Waiting for Input Data"),
        mock.call("Got Updated Parameters"),
        mock.call("Got Updated Parameters"),
        mock.call("Generating Frame of Data"),
        mock.call("Generated Signal with RMS \n  [1. 1.]"),
        mock.call("Outputting Data"),
    ]
    mock_log.assert_has_calls(log_calls)
    np.testing.assert_array_equal(mock_output.call_args_list[0][0][0], np.ones((2, 100)))
    mock_vput.assert_called_with("Process Name", (SignalGenerationCommands.GENERATE_SIGNALS, None))


@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.log")
def test_signal_generation_process_output(mock_log, signal_generation_process_obj):
    data = np.ones((2, 50))
    testing_data_enabled = np.concatenate(
        (np.arange(0.95, -0.05, -0.05).reshape(1, -1), np.zeros((1, 30))), axis=1
    )
    testing_data_disabled = np.zeros((1, 50))
    testing_data = np.concatenate((testing_data_enabled, testing_data_disabled), axis=0)
    signal_generation_process_obj.disabled_signals = [1]
    signal_generation_process_obj.current_test_level = 1
    signal_generation_process_obj.test_level_target = 0
    signal_generation_process_obj.test_level_change = -0.05
    mock_data_out = mock.MagicMock()
    signal_generation_process_obj.data_out_queue = mock_data_out

    signal_generation_process_obj.output(data)

    log_calls = [
        mock.call("Test level from 0.95 to 0.0"),
        mock.call("Sending data to data_out queue"),
    ]
    mock_log.assert_has_calls(log_calls)
    np.testing.assert_array_almost_equal(mock_data_out.put.call_args_list[0][0][0][0], testing_data)


def test_signal_generation_process_mute(signal_generation_process_obj):
    signal_generation_process_obj.current_test_level = 1
    signal_generation_process_obj.test_level_target = 1
    signal_generation_process_obj.test_level_change = 1

    signal_generation_process_obj.mute(None)

    assert signal_generation_process_obj.current_test_level == 0
    assert signal_generation_process_obj.test_level_target == 0
    assert signal_generation_process_obj.test_level_change == 0


def test_signal_generation_process_set_test_level(signal_generation_process_obj):
    data = 1

    signal_generation_process_obj.set_test_level(data)

    assert signal_generation_process_obj.current_test_level == data
    assert signal_generation_process_obj.test_level_target == data
    assert signal_generation_process_obj.test_level_change == 0


@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.log")
def test_signal_generation_process_adjust_test_level(mock_log, signal_generation_process_obj):
    signal_generation_process_obj.current_test_level = 2
    signal_generation_process_obj.ramp_samples = 20
    data = 1

    signal_generation_process_obj.adjust_test_level(data)

    assert signal_generation_process_obj.test_level_target == data
    assert signal_generation_process_obj.test_level_change
    mock_log.assert_called_with("Changed test level from 2 to 1, -0.05 change per sample")


@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.flush")
@mock.patch(
    "rattlesnake.process.signal_generation_process.SignalGenerationProcess.adjust_test_level"
)
def test_signal_generation_process_start_shutdown(
    mock_adjust, mock_flush, mock_put, signal_generation_process_obj
):
    signal_generation_process_obj.shutdown_flag = False
    signal_generation_process_obj.startup = False
    mock_flush.return_value = [[SignalGenerationCommands.GENERATE_SIGNALS]]

    signal_generation_process_obj.start_shutdown(None)

    assert signal_generation_process_obj.shutdown_flag == True
    mock_flush.assert_called_with("Process Name")
    mock_put.assert_called_with("Process Name", (SignalGenerationCommands.GENERATE_SIGNALS, None))


@mock.patch("rattlesnake.utilities.VerboseMessageQueue.put")
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.flush")
@mock.patch("rattlesnake.process.signal_generation_process.SignalGenerationProcess.log")
def test_signal_generation_process_shutdown(
    mock_log, mock_flush, mock_put, signal_generation_process_obj
):
    signal_generation_process_obj.shutdown()

    mock_log.assert_called_with("Shutting Down Signal Generation")
    mock_flush.assert_called_with("Process Name")
    mock_put.assert_called_with("Process Name", (SignalGenerationCommands.SHUTDOWN_ACHIEVED, None))
    assert signal_generation_process_obj.startup == True
    assert signal_generation_process_obj.shutdown_flag == False
    assert signal_generation_process_obj.done_generating == False


# Test the signal_generation_process function
# Prevent run while loop from starting
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.run")
def test_signal_generation_process_func(mock_run, log_file_queue):
    signal_generation_process(
        "Environment Name",
        VerboseMessageQueue(log_file_queue, "Spectral Command Queue"),
        mp.Queue(),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        log_file_queue,
        mp.Queue(),
        "Process Name",
    )

    # Test if run function was called
    mock_run.assert_called()


if __name__ == "__main__":
    log_file_queue = mp.Queue()
    signal_generation_process_obj = SignalGenerationProcess(
        "Process Name",
        VerboseMessageQueue(log_file_queue, "Spectral Command Queue"),
        mp.Queue(),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        log_file_queue,
        mp.Queue(),
        "Environment Name",
    )

    test_signal_generation_process_output(
        signal_generation_process_obj=signal_generation_process_obj
    )
