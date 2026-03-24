# from PyQt5 import QtWidgets  # unused import
import multiprocessing as mp
from unittest import mock

import numpy as np
import pytest

from rattlesnake.process.spectral_processing import (
    AveragingTypes,
    Estimator,
    SpectralProcessingCommands,
    SpectralProcessingMetadata,
    SpectralProcessingProcess,
    spectral_processing_process,
)
from rattlesnake.utilities import GlobalCommands, VerboseMessageQueue


# Create log_file_queue
@pytest.fixture()
def log_file_queue():
    return mp.Queue()


@pytest.fixture()
def spectral_processing_metadata():
    spectral_processing_metadata = SpectralProcessingMetadata(
        AveragingTypes(0),
        2,
        0.1,
        Estimator(0),
        2,
        2,
        10,
        2000,
        200,
        compute_cpsd=False,
        compute_apsd=True,
    )
    return spectral_processing_metadata


# Create command queue
@pytest.fixture()
def spectral_command_queue(log_file_queue):
    return VerboseMessageQueue(log_file_queue, "Spectral Command Queue")


# Create a SpectralProcessingProcess object
@pytest.fixture()
def spectral_processing_process_obj(log_file_queue, spectral_command_queue):
    spectral_processing_process_obj = SpectralProcessingProcess(
        "Process Name",
        spectral_command_queue,
        mp.Queue(),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        mp.Queue(),
        mp.Queue(),
        "Environment Name",
    )
    return spectral_processing_process_obj


@pytest.mark.parametrize("spectral_idx", [0, 1, 2, 3, 4, 5])
def test_spectral_processing_commands(spectral_idx):
    spectral_processing_command = SpectralProcessingCommands(spectral_idx)

    assert isinstance(spectral_processing_command, SpectralProcessingCommands)


@pytest.mark.parametrize("average_idx", [0, 1])
def test_averaging_types(average_idx):
    averaging_type = AveragingTypes(average_idx)

    assert isinstance(averaging_type, AveragingTypes)


@pytest.mark.parametrize("estimator_idx", [0, 1, 2, 3])
def test_averaging_types(estimator_idx):
    estimator = Estimator(estimator_idx)

    assert isinstance(estimator, Estimator)


def test_spectral_processing_metadata_init():
    spectral_processing_metadata = SpectralProcessingMetadata(
        AveragingTypes(0),
        2,
        0.1,
        Estimator(0),
        2,
        2,
        10,
        2000,
        200,
        compute_cpsd=False,
        compute_apsd=True,
    )

    assert isinstance(spectral_processing_metadata, SpectralProcessingMetadata)
    assert spectral_processing_metadata.requires_full_spectral_response is False
    assert spectral_processing_metadata.requires_diagonal_spectral_response is True
    assert spectral_processing_metadata.requires_full_spectral_reference is True
    assert spectral_processing_metadata.requires_diagonal_spectral_reference is True
    assert spectral_processing_metadata.requires_spectral_reference_response is True


def test_spectral_processing_metadata_eq(spectral_processing_metadata):
    assert_metadata = SpectralProcessingMetadata(
        AveragingTypes(0),
        2,
        0.1,
        Estimator(0),
        2,
        2,
        10,
        2000,
        200,
        compute_cpsd=False,
        compute_apsd=True,
    )

    assert assert_metadata == spectral_processing_metadata


# Test SpectralProcessingProcess intialization
def test_spectral_processing_init(log_file_queue):
    spectral_processing_process = SpectralProcessingProcess(
        "Process Name",
        VerboseMessageQueue(log_file_queue, "Spectral Command Queue"),
        mp.Queue(),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        log_file_queue,
        mp.Queue(),
        "Environment Name",
    )

    # Check if object is the correct class
    assert isinstance(spectral_processing_process, SpectralProcessingProcess)


@mock.patch("rattlesnake.process.spectral_processing.SpectralProcessingProcess.log")
def test_spectral_processing_initialize_parameters(
    mock_log, spectral_processing_process_obj, spectral_processing_metadata
):
    spectral_processing_process_obj.initialize_parameters(spectral_processing_metadata)

    mock_log.assert_called_with("Initializing Empty Arrays")
    np.testing.assert_array_equal(
        spectral_processing_process_obj.response_fft, np.nan * np.ones((2, 2, 200))
    )
    np.testing.assert_array_equal(
        spectral_processing_process_obj.reference_fft, np.nan * np.ones((2, 2, 200))
    )


def test_spectral_processing_clear_spectral_processing(spectral_processing_process_obj):
    mock_metadata = mock.MagicMock()
    mock_metadata.averaging_type = AveragingTypes.LINEAR
    spectral_processing_process_obj.spectral_processing_parameters = mock_metadata
    spectral_processing_process_obj.response_fft = np.ones((2, 2, 200))
    spectral_processing_process_obj.reference_fft = np.ones((2, 2, 200))

    spectral_processing_process_obj.clear_spectral_processing(None)

    assert spectral_processing_process_obj.frames_computed == 0
    assert spectral_processing_process_obj.reference_spectral_matrix == None
    np.testing.assert_array_equal(
        spectral_processing_process_obj.response_fft, np.nan * np.ones((2, 2, 200))
    )
    np.testing.assert_array_equal(
        spectral_processing_process_obj.reference_fft, np.nan * np.ones((2, 2, 200))
    )


@mock.patch("rattlesnake.process.spectral_processing.SpectralProcessingProcess.command_queue")
@mock.patch("rattlesnake.process.spectral_processing.flush_queue")
def test_spectral_processing_stop_spectral_processing(
    mock_flush, mock_command, spectral_processing_process_obj
):
    mock_command.flush.return_value = [(GlobalCommands.QUIT, None)]
    spectral_processing_process_obj.command_queue = mock_command
    mock_environment = mock.MagicMock()
    spectral_processing_process_obj.environment_command_queue = mock_environment
    mock_data_out = mock.MagicMock()
    spectral_processing_process_obj.data_out_queue = mock_data_out

    spectral_processing_process_obj.stop_spectral_processing(None)

    mock_command.flush.assert_called_with("Process Name")
    mock_command.put.assert_called_with("Process Name", (GlobalCommands.QUIT, None))
    mock_flush.assert_called_with(mock_data_out)
    mock_environment.put.assert_called_with(
        "Process Name", (SpectralProcessingCommands.SHUTDOWN_ACHIEVED, None)
    )


# Test the spectral_processing_process function
# Prevent run while loop from starting
@mock.patch("rattlesnake.process.abstract_message_process.AbstractMessageProcess.run")
def test_spectral_processing_process(mock_run, log_file_queue):
    spectral_processing_process(
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

    spectral_processing_process_obj = SpectralProcessingProcess(
        "Process Name",
        VerboseMessageQueue(log_file_queue, "Spectral Command Queue"),
        mp.Queue(),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        mp.Queue(),
        mp.Queue(),
        "Environment Name",
    )

    test_spectral_processing_stop_spectral_processing(
        spectral_processing_process_obj=spectral_processing_process_obj
    )
