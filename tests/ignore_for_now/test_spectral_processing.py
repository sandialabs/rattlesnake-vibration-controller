import multiprocessing as mp
from unittest import mock

import numpy as np
import pytest

from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.process.spectral_processing import (
    AveragingTypes,
    Estimator,
    SpectralProcessingCommands,
    SpectralProcessingMetadata,
    SpectralProcessingProcess,
    spectral_processing_process,
)
from rattlesnake.utilities import GlobalCommands, VerboseMessageQueue


# region Fixtures
@pytest.fixture
def log_file_queue():
    return mp.Queue()


@pytest.fixture
def spectral_command_queue(log_file_queue):
    return VerboseMessageQueue(
        log_file_queue,
        mp.Queue(),
        "Spectral Command Queue",
    )


@pytest.fixture
def environment_command_queue(log_file_queue):
    return VerboseMessageQueue(
        log_file_queue,
        mp.Queue(),
        "Environment Command Queue",
    )


@pytest.fixture
def spectral_processing_metadata():
    return SpectralProcessingMetadata(
        averaging_type=AveragingTypes.LINEAR,
        averages=2,
        exponential_averaging_coefficient=0.1,
        frf_estimator=Estimator.H1,
        num_response_channels=2,
        num_reference_channels=2,
        frequency_spacing=10,
        sample_rate=2000,
        num_frequency_lines=200,
        compute_cpsd=False,
        compute_apsd=True,
    )


@pytest.fixture
def exponential_spectral_processing_metadata():
    return SpectralProcessingMetadata(
        averaging_type=AveragingTypes.EXPONENTIAL,
        averages=2,
        exponential_averaging_coefficient=0.25,
        frf_estimator=Estimator.H1,
        num_response_channels=1,
        num_reference_channels=1,
        frequency_spacing=1.0,
        sample_rate=100.0,
        num_frequency_lines=3,
        compute_cpsd=False,
        compute_frf=True,
        compute_coherence=True,
        compute_apsd=True,
    )


@pytest.fixture
def spectral_processing_process_obj(
    log_file_queue,
    spectral_command_queue,
    environment_command_queue,
):
    return SpectralProcessingProcess(
        "Process Name",
        spectral_command_queue,
        mp.Queue(),
        mp.Queue(),
        environment_command_queue,
        mp.Queue(),
        log_file_queue,
        "Environment Name",
    )


# endregion


# region Commands and Enums
@pytest.mark.parametrize("spectral_idx", range(6))
def test_spectral_processing_commands(spectral_idx):
    """
    Verifies that enum values construct valid
    ``SpectralProcessingCommands`` members.
    """
    command = SpectralProcessingCommands(spectral_idx)

    assert isinstance(command, SpectralProcessingCommands)


def test_spectral_processing_commands_unique_integer_values():
    """
    Verifies that spectral processing command values are unique integers.
    """
    values = [command.value for command in SpectralProcessingCommands]

    assert all(isinstance(value, int) for value in values)
    assert len(values) == len(set(values))


@pytest.mark.parametrize("average_idx", range(2))
def test_averaging_types(average_idx):
    """
    Verifies that enum values construct valid ``AveragingTypes`` members.
    """
    averaging_type = AveragingTypes(average_idx)

    assert isinstance(averaging_type, AveragingTypes)


def test_averaging_types_unique_integer_values():
    """
    Verifies that averaging type values are unique integers.
    """
    values = [averaging_type.value for averaging_type in AveragingTypes]

    assert all(isinstance(value, int) for value in values)
    assert len(values) == len(set(values))


@pytest.mark.parametrize("estimator_idx", range(4))
def test_estimators(estimator_idx):
    """
    Verifies that enum values construct valid ``Estimator`` members.
    """
    estimator = Estimator(estimator_idx)

    assert isinstance(estimator, Estimator)


def test_estimators_unique_integer_values():
    """
    Verifies that estimator values are unique integers.
    """
    values = [estimator.value for estimator in Estimator]

    assert all(isinstance(value, int) for value in values)
    assert len(values) == len(set(values))


# endregion


# region SpectralProcessingMetadata
def test_spectral_processing_metadata_init(spectral_processing_metadata):
    """
    Verifies that metadata initializes and that spectral-requirement properties
    return expected values.
    """
    metadata = spectral_processing_metadata

    assert isinstance(metadata, SpectralProcessingMetadata)
    assert metadata.averaging_type == AveragingTypes.LINEAR
    assert metadata.averages == 2
    assert metadata.exponential_averaging_coefficient == 0.1
    assert metadata.frf_estimator == Estimator.H1
    assert metadata.num_response_channels == 2
    assert metadata.num_reference_channels == 2
    assert metadata.frequency_spacing == 10
    assert metadata.sample_rate == 2000
    assert metadata.num_frequency_lines == 200
    assert metadata.compute_cpsd is False
    assert metadata.compute_frf is True
    assert metadata.compute_coherence is True
    assert metadata.compute_apsd is True

    assert metadata.requires_full_spectral_response is False
    assert metadata.requires_diagonal_spectral_response is True
    assert metadata.requires_full_spectral_reference is True
    assert metadata.requires_diagonal_spectral_reference is True
    assert metadata.requires_spectral_reference_response is True


def test_spectral_processing_metadata_eq(spectral_processing_metadata):
    """
    Verifies equality comparison for equivalent metadata objects.
    """
    expected_metadata = SpectralProcessingMetadata(
        averaging_type=AveragingTypes.LINEAR,
        averages=2,
        exponential_averaging_coefficient=0.1,
        frf_estimator=Estimator.H1,
        num_response_channels=2,
        num_reference_channels=2,
        frequency_spacing=10,
        sample_rate=2000,
        num_frequency_lines=200,
        compute_cpsd=False,
        compute_apsd=True,
    )

    assert expected_metadata == spectral_processing_metadata


def test_spectral_processing_metadata_eq_false(spectral_processing_metadata):
    """
    Verifies equality comparison returns false for different metadata.
    """
    other = SpectralProcessingMetadata(
        averaging_type=AveragingTypes.LINEAR,
        averages=3,
        exponential_averaging_coefficient=0.1,
        frf_estimator=Estimator.H1,
        num_response_channels=2,
        num_reference_channels=2,
        frequency_spacing=10,
        sample_rate=2000,
        num_frequency_lines=200,
        compute_cpsd=False,
        compute_apsd=True,
    )

    assert spectral_processing_metadata != other


def test_spectral_processing_metadata_eq_incompatible(spectral_processing_metadata):
    """
    Verifies equality comparison returns false for incompatible objects.
    """
    assert spectral_processing_metadata != object()


@pytest.mark.parametrize(
    "frf_estimator, compute_cpsd, expected",
    [
        (Estimator.H1, False, False),
        (Estimator.H2, False, True),
        (Estimator.H3, False, True),
        (Estimator.HV, False, False),
        (Estimator.H1, True, True),
    ],
)
def test_requires_full_spectral_response(frf_estimator, compute_cpsd, expected):
    """
    Verifies the full response spectral matrix requirement flag.
    """
    metadata = SpectralProcessingMetadata(
        AveragingTypes.LINEAR,
        2,
        0.1,
        frf_estimator,
        2,
        2,
        10,
        2000,
        200,
        compute_cpsd=compute_cpsd,
        compute_frf=True,
        compute_coherence=False,
        compute_apsd=False,
    )

    assert metadata.requires_full_spectral_response is expected


@pytest.mark.parametrize(
    "frf_estimator, compute_apsd, compute_coherence, expected",
    [
        (Estimator.H1, False, False, False),
        (Estimator.HV, False, False, True),
        (Estimator.H1, True, False, True),
        (Estimator.H1, False, True, True),
    ],
)
def test_requires_diagonal_spectral_response(
    frf_estimator,
    compute_apsd,
    compute_coherence,
    expected,
):
    """
    Verifies the diagonal response spectral matrix requirement flag.
    """
    metadata = SpectralProcessingMetadata(
        AveragingTypes.LINEAR,
        2,
        0.1,
        frf_estimator,
        2,
        2,
        10,
        2000,
        200,
        compute_cpsd=False,
        compute_frf=True,
        compute_coherence=compute_coherence,
        compute_apsd=compute_apsd,
    )

    assert metadata.requires_diagonal_spectral_response is expected


@pytest.mark.parametrize(
    "frf_estimator, compute_cpsd, compute_coherence, expected",
    [
        (Estimator.H1, False, False, True),
        (Estimator.H2, False, False, False),
        (Estimator.H3, False, False, True),
        (Estimator.HV, False, False, True),
        (Estimator.H2, True, False, True),
        (Estimator.H2, False, True, True),
    ],
)
def test_requires_full_spectral_reference(
    frf_estimator,
    compute_cpsd,
    compute_coherence,
    expected,
):
    """
    Verifies the full reference spectral matrix requirement flag.
    """
    metadata = SpectralProcessingMetadata(
        AveragingTypes.LINEAR,
        2,
        0.1,
        frf_estimator,
        2,
        2,
        10,
        2000,
        200,
        compute_cpsd=compute_cpsd,
        compute_frf=True,
        compute_coherence=compute_coherence,
        compute_apsd=False,
    )

    assert metadata.requires_full_spectral_reference is expected


@pytest.mark.parametrize("compute_apsd, expected", [(True, True), (False, False)])
def test_requires_diagonal_spectral_reference(compute_apsd, expected):
    """
    Verifies the diagonal reference spectral matrix requirement flag.
    """
    metadata = SpectralProcessingMetadata(
        AveragingTypes.LINEAR,
        2,
        0.1,
        Estimator.H1,
        2,
        2,
        10,
        2000,
        200,
        compute_apsd=compute_apsd,
    )

    assert metadata.requires_diagonal_spectral_reference is expected


@pytest.mark.parametrize(
    "compute_frf, compute_coherence, expected",
    [
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_requires_spectral_reference_response(
    compute_frf,
    compute_coherence,
    expected,
):
    """
    Verifies the cross spectral matrix requirement flag.
    """
    metadata = SpectralProcessingMetadata(
        AveragingTypes.LINEAR,
        2,
        0.1,
        Estimator.H1,
        2,
        2,
        10,
        2000,
        200,
        compute_frf=compute_frf,
        compute_coherence=compute_coherence,
    )

    assert metadata.requires_spectral_reference_response is expected


# endregion


# region SpectralProcessingProcess initialization
def test_spectral_processing_init(
    log_file_queue,
    spectral_command_queue,
    environment_command_queue,
):
    """
    Verifies that the spectral processing process initializes successfully.
    """
    data_in_queue = mp.Queue()
    data_out_queue = mp.Queue()
    gui_update_queue = mp.Queue()

    process = SpectralProcessingProcess(
        "Process Name",
        spectral_command_queue,
        data_in_queue,
        data_out_queue,
        environment_command_queue,
        gui_update_queue,
        log_file_queue,
        "Environment Name",
    )

    assert isinstance(process, SpectralProcessingProcess)
    assert isinstance(process, AbstractMessageProcess)

    assert process.process_name == "Process Name"
    assert process.environment_name == "Environment Name"
    assert process.data_in_queue is data_in_queue
    assert process.data_out_queue is data_out_queue
    assert process.environment_command_queue is environment_command_queue

    assert process.response_spectral_matrix is None
    assert process.reference_spectral_matrix is None
    assert process.response_reference_spectral_matrix is None
    assert process.reference_diagonal_matrix is None
    assert process.response_diagonal_matrix is None
    assert process.response_fft is None
    assert process.reference_fft is None
    assert process.spectral_processing_parameters is None
    assert process.frames_computed == 0


def test_spectral_processing_command_map(spectral_processing_process_obj):
    """
    Verifies that spectral processing commands are mapped to process methods.
    """
    assert (
        spectral_processing_process_obj.command_map[
            SpectralProcessingCommands.INITIALIZE_PARAMETERS
        ]
        == spectral_processing_process_obj.initialize_parameters
    )
    assert (
        spectral_processing_process_obj.command_map[
            SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING
        ]
        == spectral_processing_process_obj.run_spectral_processing
    )
    assert (
        spectral_processing_process_obj.command_map[
            SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING
        ]
        == spectral_processing_process_obj.clear_spectral_processing
    )
    assert (
        spectral_processing_process_obj.command_map[
            SpectralProcessingCommands.STOP_SPECTRAL_PROCESSING
        ]
        == spectral_processing_process_obj.stop_spectral_processing
    )


# endregion


# region initialize_parameters
@mock.patch("rattlesnake.process.spectral_processing.SpectralProcessingProcess.log")
def test_spectral_processing_initialize_parameters_linear(
    mock_log,
    spectral_processing_process_obj,
    spectral_processing_metadata,
):
    """
    Verifies that linear averaging arrays are initialized with expected shapes
    and NaN contents.
    """
    spectral_processing_process_obj.initialize_parameters(spectral_processing_metadata)

    mock_log.assert_called_once_with("Initializing Empty Arrays")

    assert spectral_processing_process_obj.spectral_processing_parameters is (
        spectral_processing_metadata
    )
    assert spectral_processing_process_obj.frames_computed == 0
    assert spectral_processing_process_obj.response_spectral_matrix is None
    assert spectral_processing_process_obj.reference_spectral_matrix is None
    assert spectral_processing_process_obj.response_reference_spectral_matrix is None

    assert spectral_processing_process_obj.response_fft.shape == (2, 2, 200)
    assert spectral_processing_process_obj.reference_fft.shape == (2, 2, 200)
    assert np.all(np.isnan(spectral_processing_process_obj.response_fft))
    assert np.all(np.isnan(spectral_processing_process_obj.reference_fft))


@mock.patch("rattlesnake.process.spectral_processing.SpectralProcessingProcess.log")
def test_spectral_processing_initialize_parameters_exponential(
    mock_log,
    spectral_processing_process_obj,
    exponential_spectral_processing_metadata,
):
    """
    Verifies that exponential averaging initialization clears FFT buffers.
    """
    spectral_processing_process_obj.initialize_parameters(
        exponential_spectral_processing_metadata
    )

    mock_log.assert_called_once_with("Initializing Empty Arrays")

    assert spectral_processing_process_obj.spectral_processing_parameters is (
        exponential_spectral_processing_metadata
    )
    assert spectral_processing_process_obj.response_fft is None
    assert spectral_processing_process_obj.reference_fft is None
    assert spectral_processing_process_obj.response_spectral_matrix is None
    assert spectral_processing_process_obj.reference_spectral_matrix is None


@mock.patch("rattlesnake.process.spectral_processing.SpectralProcessingProcess.log")
def test_spectral_processing_initialize_parameters_same_shape_no_reset(
    mock_log,
    spectral_processing_process_obj,
    spectral_processing_metadata,
):
    """
    Verifies that arrays are not reset when compatible metadata are supplied.
    """
    spectral_processing_process_obj.initialize_parameters(spectral_processing_metadata)
    response_fft = spectral_processing_process_obj.response_fft
    reference_fft = spectral_processing_process_obj.reference_fft

    mock_log.reset_mock()

    equivalent_metadata = SpectralProcessingMetadata(
        averaging_type=AveragingTypes.LINEAR,
        averages=2,
        exponential_averaging_coefficient=0.5,
        frf_estimator=Estimator.H2,
        num_response_channels=2,
        num_reference_channels=2,
        frequency_spacing=20,
        sample_rate=4000,
        num_frequency_lines=200,
    )

    spectral_processing_process_obj.initialize_parameters(equivalent_metadata)

    mock_log.assert_not_called()
    assert spectral_processing_process_obj.response_fft is response_fft
    assert spectral_processing_process_obj.reference_fft is reference_fft
    assert spectral_processing_process_obj.spectral_processing_parameters is (
        equivalent_metadata
    )


@mock.patch("rattlesnake.process.spectral_processing.SpectralProcessingProcess.log")
def test_spectral_processing_initialize_parameters_shape_change_resets(
    mock_log,
    spectral_processing_process_obj,
    spectral_processing_metadata,
):
    """
    Verifies that arrays are reset when frequency lines or channel counts
    change.
    """
    spectral_processing_process_obj.initialize_parameters(spectral_processing_metadata)

    changed_metadata = SpectralProcessingMetadata(
        averaging_type=AveragingTypes.LINEAR,
        averages=2,
        exponential_averaging_coefficient=0.1,
        frf_estimator=Estimator.H1,
        num_response_channels=1,
        num_reference_channels=2,
        frequency_spacing=10,
        sample_rate=2000,
        num_frequency_lines=100,
    )

    mock_log.reset_mock()

    spectral_processing_process_obj.initialize_parameters(changed_metadata)

    mock_log.assert_called_once_with("Initializing Empty Arrays")
    assert spectral_processing_process_obj.response_fft.shape == (2, 1, 100)
    assert spectral_processing_process_obj.reference_fft.shape == (2, 2, 100)


# endregion


# region run_spectral_processing
@mock.patch("rattlesnake.process.spectral_processing.time.sleep")
@mock.patch("rattlesnake.process.spectral_processing.flush_queue")
def test_spectral_processing_run_spectral_processing_no_data(
    mock_flush_queue,
    mock_sleep,
    spectral_processing_process_obj,
):
    """
    Verifies that no input data causes the process to sleep and requeue itself.
    """
    spectral_processing_process_obj._command_queue = mock.MagicMock()
    mock_flush_queue.return_value = []

    spectral_processing_process_obj.run_spectral_processing(None)

    mock_flush_queue.assert_called_once_with(
        spectral_processing_process_obj.data_in_queue,
        timeout=0.05,
    )
    mock_sleep.assert_called_once_with(0.05)
    spectral_processing_process_obj.command_queue.put.assert_called_once_with(
        "Process Name",
        (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING, None),
    )


@mock.patch("rattlesnake.process.spectral_processing.flush_queue")
@mock.patch("rattlesnake.process.spectral_processing.SpectralProcessingProcess.log")
def test_spectral_processing_run_spectral_processing_exponential_h1(
    mock_log,
    mock_flush_queue,
    spectral_processing_process_obj,
    exponential_spectral_processing_metadata,
):
    """
    Verifies exponential averaging computes spectral outputs and sends updated
    spectral data.
    """
    spectral_processing_process_obj.initialize_parameters(
        exponential_spectral_processing_metadata
    )

    response_fft = np.array([[1.0 + 0j, 2.0 + 0j, 3.0 + 0j]])
    reference_fft = np.array([[2.0 + 0j, 2.0 + 0j, 2.0 + 0j]])
    mock_flush_queue.return_value = [(response_fft, reference_fft)]

    spectral_processing_process_obj._command_queue = mock.MagicMock()
    spectral_processing_process_obj.data_out_queue = mock.MagicMock()

    spectral_processing_process_obj.run_spectral_processing(None)

    spectral_processing_process_obj.data_out_queue.put.assert_called_once()
    (
        frames,
        frequencies,
        frf,
        coherence,
        response_spectral_matrix,
        reference_spectral_matrix,
        frf_condition,
    ) = spectral_processing_process_obj.data_out_queue.put.call_args.args[0]

    assert frames == 1
    np.testing.assert_array_equal(frequencies, np.array([0.0, 1.0, 2.0]))

    assert frf.shape == (3, 1, 1)
    assert coherence.shape == (3, 1)
    assert response_spectral_matrix.shape == (3, 1)
    assert reference_spectral_matrix.shape == (3, 1)
    assert frf_condition.shape == (3,)

    spectral_processing_process_obj.command_queue.put.assert_called_once_with(
        "Process Name",
        (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING, None),
    )

    assert any("Received 1 Frames" in call.args[0] for call in mock_log.call_args_list)
    assert any(
        "Sending Updated Spectral Data" in call.args[0]
        for call in mock_log.call_args_list
    )


@mock.patch("rattlesnake.process.spectral_processing.flush_queue")
def test_spectral_processing_run_spectral_processing_exponential_without_outputs(
    mock_flush_queue,
    spectral_processing_process_obj,
):
    """
    Verifies spectral processing can run when optional outputs are disabled.
    """
    metadata = SpectralProcessingMetadata(
        averaging_type=AveragingTypes.EXPONENTIAL,
        averages=2,
        exponential_averaging_coefficient=0.25,
        frf_estimator=Estimator.H1,
        num_response_channels=1,
        num_reference_channels=1,
        frequency_spacing=1.0,
        sample_rate=100.0,
        num_frequency_lines=3,
        compute_cpsd=False,
        compute_frf=False,
        compute_coherence=False,
        compute_apsd=False,
    )
    spectral_processing_process_obj.initialize_parameters(metadata)

    response_fft = np.array([[1.0 + 0j, 2.0 + 0j, 3.0 + 0j]])
    reference_fft = np.array([[2.0 + 0j, 2.0 + 0j, 2.0 + 0j]])
    mock_flush_queue.return_value = [(response_fft, reference_fft)]

    spectral_processing_process_obj._command_queue = mock.MagicMock()
    spectral_processing_process_obj.data_out_queue = mock.MagicMock()

    spectral_processing_process_obj.run_spectral_processing(None)

    (
        frames,
        frequencies,
        frf,
        coherence,
        response_spectral_matrix,
        reference_spectral_matrix,
        frf_condition,
    ) = spectral_processing_process_obj.data_out_queue.put.call_args.args[0]

    assert frames == 1
    np.testing.assert_array_equal(frequencies, np.array([0.0, 1.0, 2.0]))
    assert frf is None
    assert coherence is None
    assert response_spectral_matrix is None
    assert reference_spectral_matrix is None
    assert frf_condition is None


@pytest.mark.parametrize(
    "estimator",
    [Estimator.H1, Estimator.H2, Estimator.H3, Estimator.HV],
)
@mock.patch("rattlesnake.process.spectral_processing.flush_queue")
def test_spectral_processing_run_spectral_processing_estimators(
    mock_flush_queue,
    spectral_processing_process_obj,
    estimator,
):
    """
    Verifies all FRF estimators produce an FRF output for a simple single-input,
    single-output case.
    """
    metadata = SpectralProcessingMetadata(
        averaging_type=AveragingTypes.EXPONENTIAL,
        averages=2,
        exponential_averaging_coefficient=0.5,
        frf_estimator=estimator,
        num_response_channels=1,
        num_reference_channels=1,
        frequency_spacing=1.0,
        sample_rate=100.0,
        num_frequency_lines=3,
        compute_cpsd=True,
        compute_frf=True,
        compute_coherence=True,
        compute_apsd=True,
    )
    spectral_processing_process_obj.initialize_parameters(metadata)

    response_fft = np.array([[1.0 + 0j, 2.0 + 0j, 3.0 + 0j]])
    reference_fft = np.array([[2.0 + 0j, 2.0 + 0j, 2.0 + 0j]])
    mock_flush_queue.return_value = [(response_fft, reference_fft)]

    spectral_processing_process_obj._command_queue = mock.MagicMock()
    spectral_processing_process_obj.data_out_queue = mock.MagicMock()

    spectral_processing_process_obj.run_spectral_processing(None)

    _, _, frf, coherence, response_cpsd, reference_cpsd, condition = (
        spectral_processing_process_obj.data_out_queue.put.call_args.args[0]
    )

    assert frf is not None
    assert coherence is not None
    assert response_cpsd is not None
    assert reference_cpsd is not None
    assert condition is not None


# endregion


# region clear and stop
@pytest.mark.parametrize(
    "averaging_type", [AveragingTypes.LINEAR, AveragingTypes.EXPONENTIAL]
)
def test_spectral_processing_clear_spectral_processing(
    spectral_processing_process_obj,
    averaging_type,
):
    """
    Verifies that accumulated spectral state is cleared for linear and
    exponential averaging modes.
    """
    metadata = mock.MagicMock()
    metadata.averaging_type = averaging_type
    spectral_processing_process_obj.spectral_processing_parameters = metadata

    spectral_processing_process_obj.frames_computed = 10
    spectral_processing_process_obj.response_spectral_matrix = np.ones((2, 2, 2))
    spectral_processing_process_obj.reference_spectral_matrix = np.ones((2, 2, 2))
    spectral_processing_process_obj.response_reference_spectral_matrix = np.ones(
        (2, 2, 2)
    )

    if averaging_type == AveragingTypes.LINEAR:
        spectral_processing_process_obj.response_fft = np.ones((2, 2, 4))
        spectral_processing_process_obj.reference_fft = np.ones((2, 2, 4))
    else:
        spectral_processing_process_obj.response_fft = np.ones((2, 2, 4))
        spectral_processing_process_obj.reference_fft = np.ones((2, 2, 4))

    spectral_processing_process_obj.clear_spectral_processing(None)

    assert spectral_processing_process_obj.frames_computed == 0
    assert spectral_processing_process_obj.response_spectral_matrix is None
    assert spectral_processing_process_obj.reference_spectral_matrix is None
    assert spectral_processing_process_obj.response_reference_spectral_matrix is None

    if averaging_type == AveragingTypes.LINEAR:
        assert np.all(np.isnan(spectral_processing_process_obj.response_fft))
        assert np.all(np.isnan(spectral_processing_process_obj.reference_fft))
    else:
        assert spectral_processing_process_obj.response_fft is None
        assert spectral_processing_process_obj.reference_fft is None


@mock.patch("rattlesnake.process.spectral_processing.time.sleep")
@mock.patch("rattlesnake.process.spectral_processing.flush_queue")
def test_spectral_processing_stop_spectral_processing_preserves_quit(
    mock_flush_queue,
    mock_sleep,
    spectral_processing_process_obj,
):
    """
    Verifies command queue flushing, quit command preservation, output queue
    flushing, and shutdown notification.
    """
    spectral_processing_process_obj._command_queue = mock.MagicMock()
    spectral_processing_process_obj.command_queue.flush.return_value = [
        (GlobalCommands.QUIT, None),
        (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING, None),
    ]

    spectral_processing_process_obj.environment_command_queue = mock.MagicMock()
    spectral_processing_process_obj.data_out_queue = mock.MagicMock()

    spectral_processing_process_obj.stop_spectral_processing(None)

    mock_sleep.assert_called_once_with(0.05)
    spectral_processing_process_obj.command_queue.flush.assert_called_once_with(
        "Process Name"
    )
    spectral_processing_process_obj.command_queue.put.assert_called_once_with(
        "Process Name",
        (GlobalCommands.QUIT, None),
    )
    mock_flush_queue.assert_called_once_with(
        spectral_processing_process_obj.data_out_queue
    )
    spectral_processing_process_obj.environment_command_queue.put.assert_called_once_with(
        "Process Name",
        (SpectralProcessingCommands.SHUTDOWN_ACHIEVED, None),
    )


@mock.patch("rattlesnake.process.spectral_processing.time.sleep")
@mock.patch("rattlesnake.process.spectral_processing.flush_queue")
def test_spectral_processing_stop_spectral_processing_without_quit(
    mock_flush_queue,
    mock_sleep,
    spectral_processing_process_obj,
):
    """
    Verifies stop behavior when no quit command was flushed.
    """
    spectral_processing_process_obj._command_queue = mock.MagicMock()
    spectral_processing_process_obj.command_queue.flush.return_value = [
        (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING, None),
    ]

    spectral_processing_process_obj.environment_command_queue = mock.MagicMock()
    spectral_processing_process_obj.data_out_queue = mock.MagicMock()

    spectral_processing_process_obj.stop_spectral_processing(None)

    spectral_processing_process_obj.command_queue.put.assert_not_called()
    mock_flush_queue.assert_called_once_with(
        spectral_processing_process_obj.data_out_queue
    )
    spectral_processing_process_obj.environment_command_queue.put.assert_called_once_with(
        "Process Name",
        (SpectralProcessingCommands.SHUTDOWN_ACHIEVED, None),
    )


# endregion


# region Process function
@mock.patch("rattlesnake.process.spectral_processing.SpectralProcessingProcess")
def test_spectral_processing_process_func(
    mock_spectral_processing_process_class,
    spectral_command_queue,
    environment_command_queue,
    log_file_queue,
):
    """
    Verifies that the process function constructs a spectral processing process
    and starts its command loop.
    """
    data_in_queue = mp.Queue()
    data_out_queue = mp.Queue()
    gui_update_queue = mp.Queue()

    spectral_processing_process(
        "Environment Name",
        spectral_command_queue,
        data_in_queue,
        data_out_queue,
        environment_command_queue,
        gui_update_queue,
        log_file_queue,
        "Process Name",
    )

    mock_spectral_processing_process_class.assert_called_once_with(
        "Process Name",
        spectral_command_queue,
        data_in_queue,
        data_out_queue,
        environment_command_queue,
        gui_update_queue,
        log_file_queue,
        "Environment Name",
    )
    mock_spectral_processing_process_class.return_value.run.assert_called_once_with()


@mock.patch("rattlesnake.process.spectral_processing.SpectralProcessingProcess")
def test_spectral_processing_process_func_default_process_name(
    mock_spectral_processing_process_class,
    spectral_command_queue,
    environment_command_queue,
    log_file_queue,
):
    """
    Verifies that the default process name is generated from the environment
    name.
    """
    spectral_processing_process(
        "Environment Name",
        spectral_command_queue,
        mp.Queue(),
        mp.Queue(),
        environment_command_queue,
        mp.Queue(),
        log_file_queue,
    )

    assert mock_spectral_processing_process_class.call_args.args[0] == (
        "Environment Name Spectral Processing Computation"
    )


# endregion
