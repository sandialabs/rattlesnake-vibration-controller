"""Tests for the transient sys-id environment.

These tests drive the metadata file flows (worksheet, netCDF) and the
environment lifecycle the controller uses in a real transient test
(initialize hardware/environment/sysid -> transfer function ->
SYSTEM_ID_COMPLETE -> control prediction with a real type-0 control law ->
start/stop control -> QUIT), using hand-fed queues with no spawned
sub-workers.

The multi-pulse repeat/alignment steady state requires live signal
generation and acquisition workers and is exercised end-to-end by
tests/long/test_qualification.py, so it is not covered here.
"""

import time
from unittest import mock

import netCDF4 as nc4
import numpy as np
import openpyxl
import pytest

from rattlesnake.environment.abstract_environment import (
    EnvironmentInstructions,
    EnvironmentMetadata,
)
from rattlesnake.environment.abstract_sysid_environment import (
    SysIdEnvironment,
    SysIdUICommands,
    SystemIdCommands,
)
from rattlesnake.environment.transient_sys_id_environment import (
    TransientCommands,
    TransientEnvironment,
    TransientInstructions,
    TransientMetadata,
    TransientQueues,
    TransientUICommands,
    transient_process,
)
from rattlesnake.process.abstract_sysid_data_analysis import (
    SysIdDataAnalysisCommands,
    SysIdMetadata,
)
from rattlesnake.process.data_collector import (
    DataCollectorCommands,
    FrameBuffer,
)
from rattlesnake.process.signal_generation import TransientSignalGenerator
from rattlesnake.process.signal_generation_process import SignalGenerationCommands
from rattlesnake.process.spectral_processing import SpectralProcessingCommands
from rattlesnake.testing.mock_utilities import (
    drain_queue_commands,
    get_queue_messages,
    sysid_measurement_data_package,
    sysid_measurement_metadata,
    numeric_hardware_metadata,
    write_transient_control_script,
    mock_event_container,
    mock_queue_container,
)
from rattlesnake.user_interface.ui_utilities import UICommands
from rattlesnake.utilities import GlobalCommands, RattlesnakeError, flush_queue

ENVIRONMENT_NAME = "Transient Environment"
QUEUE_NAME = "Environment 0"
SIGNAL_SAMPLES = 500


def flush_environment_queues(environment):
    """Drains every queue so no multiprocessing feeder thread is left
    blocked on a full pipe, which would hang the interpreter at exit."""
    queues = environment.queue_container
    for queue in (
        queues.gui_update_queue,
        queues.environment_command_queue,
        queues.data_analysis_command_queue,
        queues.collector_command_queue,
        queues.signal_generation_command_queue,
        queues.spectral_command_queue,
    ):
        flush_queue(queue, timeout=0.05)


# region Fixtures
def make_control_signal():
    # A half-sine pulse as the transient specification
    return np.sin(np.linspace(0, np.pi, SIGNAL_SAMPLES))[np.newaxis, :]


@pytest.fixture
def transient_metadata(tmp_path):
    script_path = str(tmp_path / "transient_control.py")
    function_name = write_transient_control_script(script_path)
    transient_metadata = TransientMetadata(
        environment_name=ENVIRONMENT_NAME,
        channel_list_bools=[True, True],
        sample_rate=1000,
        number_of_channels=2,
        control_signal=make_control_signal(),
        ramp_time=0.05,
        control_python_script=script_path,
        control_python_function=function_name,
        control_python_function_type=0,
        control_python_function_parameters="",
        control_channel_indices=[0],
        output_channel_indices=[1],
        response_transformation_matrix=None,
        output_transformation_matrix=None,
        sysid_metadata=sysid_measurement_metadata(),
    )
    transient_metadata.queue_name = QUEUE_NAME

    return transient_metadata


@pytest.fixture(params=[True, False], ids=["threaded", "non_threaded"])
def transient_environment(request):
    use_thread = request.param
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)
    transient_queues = TransientQueues(
        ENVIRONMENT_NAME,
        queue_container.environment_command_queues[QUEUE_NAME],
        queue_container.gui_update_queue,
        queue_container.controller_command_queue,
        queue_container.environment_data_in_queues[QUEUE_NAME],
        queue_container.environment_data_out_queues[QUEUE_NAME],
        queue_container.log_file_queue,
        use_thread,
    )
    transient_environment = TransientEnvironment(
        ENVIRONMENT_NAME,
        QUEUE_NAME,
        transient_queues,
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.environment_active_events[QUEUE_NAME],
        event_container.environment_ready_events[QUEUE_NAME],
        event_container.environment_sysid_active_events[QUEUE_NAME],
        event_container.environment_sysid_stored_events[QUEUE_NAME],
    )
    yield transient_environment
    flush_environment_queues(transient_environment)


def initialized_environment(transient_environment, transient_metadata):
    """Runs the environment through the real hardware/environment init flow."""
    transient_environment.initialize_hardware(numeric_hardware_metadata())
    transient_environment.initialize_environment(transient_metadata)
    # Discard the initialization queue traffic so tests can assert on
    # the traffic of the method under test only
    for queue in (
        transient_environment.queue_container.data_analysis_command_queue,
        transient_environment.queue_container.collector_command_queue,
        transient_environment.queue_container.signal_generation_command_queue,
        transient_environment.queue_container.spectral_command_queue,
    ):
        drain_queue_commands(queue, ENVIRONMENT_NAME)
    return transient_environment


def completed_system_id(transient_environment, transient_metadata):
    """Feeds the environment a completed system identification package."""
    data_package = sysid_measurement_data_package(transient_metadata.sysid_metadata)
    transient_environment.system_id_complete(
        (transient_metadata.sysid_metadata, data_package)
    )
    return data_package


# region Metadata
def test_transient_metadata_init(transient_metadata):
    assert isinstance(transient_metadata, TransientMetadata)
    assert isinstance(transient_metadata, EnvironmentMetadata)
    assert isinstance(transient_metadata.sysid_metadata, SysIdMetadata)


def test_transient_metadata_properties(transient_metadata):
    assert transient_metadata.ramp_samples == 50
    assert transient_metadata.signal_samples == SIGNAL_SAMPLES
    assert transient_metadata.response_channel_indices == [0]
    assert transient_metadata.reference_channel_indices == [1]
    assert transient_metadata.num_response_channels == 1
    assert transient_metadata.num_reference_channels == 1


def test_transient_metadata_transformed_channel_counts(transient_metadata):
    transient_metadata.response_transformation_matrix = np.ones((3, 1))
    transient_metadata.reference_transformation_matrix = np.ones((2, 1))

    assert transient_metadata.num_response_channels == 3
    assert transient_metadata.num_reference_channels == 2


@pytest.mark.parametrize(
    "channel_list_bools, environment_name, expected",
    [
        ([True, True], ENVIRONMENT_NAME, True),
        ([True], ENVIRONMENT_NAME, RattlesnakeError),
        ([True, True], 17, RattlesnakeError),
    ],
)
def test_transient_metadata_validate(
    channel_list_bools, environment_name, expected, transient_metadata
):
    hardware_metadata = numeric_hardware_metadata()
    transient_metadata.channel_list_bools = channel_list_bools
    transient_metadata.environment_name = environment_name

    if expected is RattlesnakeError:
        with pytest.raises(RattlesnakeError):
            transient_metadata.validate(hardware_metadata)
    else:
        transient_metadata.validate(hardware_metadata)


@pytest.mark.parametrize("with_transformations", [False, True])
def test_transient_metadata_netcdf_round_trip(transient_metadata, with_transformations):
    if with_transformations:
        transient_metadata.response_transformation_matrix = np.arange(2.0).reshape(2, 1)
        transient_metadata.reference_transformation_matrix = np.arange(3.0).reshape(
            3, 1
        )
    dataset = nc4.Dataset("temp.nc", mode="w", diskless=True, persist=False)
    netcdf_group = dataset.createGroup(ENVIRONMENT_NAME)

    transient_metadata.save_metadata_to_netcdf(netcdf_group)
    loaded_metadata = TransientMetadata.load_metadata_from_netcdf(
        netcdf_group,
        ENVIRONMENT_NAME,
        [True, True],
        numeric_hardware_metadata(),
    )

    assert loaded_metadata.test_level_ramp_time == 0.05
    assert (
        loaded_metadata.control_python_script
        == transient_metadata.control_python_script
    )
    assert loaded_metadata.control_python_function == "transient_control"
    assert loaded_metadata.control_python_function_type == 0
    np.testing.assert_allclose(
        np.asarray(loaded_metadata.control_signal),
        transient_metadata.control_signal,
    )
    np.testing.assert_array_equal(
        np.asarray(loaded_metadata.control_channel_indices), [0]
    )
    assert loaded_metadata.output_channel_indices == [1]
    if with_transformations:
        np.testing.assert_allclose(
            np.asarray(loaded_metadata.response_transformation_matrix),
            transient_metadata.response_transformation_matrix,
        )
        np.testing.assert_allclose(
            np.asarray(loaded_metadata.reference_transformation_matrix),
            transient_metadata.reference_transformation_matrix,
        )
    else:
        assert loaded_metadata.response_transformation_matrix is None
        assert loaded_metadata.reference_transformation_matrix is None
    assert loaded_metadata.sysid_metadata == transient_metadata.sysid_metadata

    dataset.close()


def test_transient_metadata_worksheet_round_trip(transient_metadata):
    # This is the profile .xlsx flow.  The control signal comes from a
    # separate signal file which the saver does not write, so the loaded
    # control_signal is None when that cell is left empty.
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    transient_metadata.save_metadata_to_worksheet(worksheet)

    loaded_metadata = TransientMetadata.load_metadata_from_worksheet(
        worksheet,
        ENVIRONMENT_NAME,
        [True, True],
        numeric_hardware_metadata(),
    )

    assert loaded_metadata.test_level_ramp_time == 0.05
    assert (
        loaded_metadata.control_python_script
        == transient_metadata.control_python_script
    )
    assert loaded_metadata.control_python_function == "transient_control"
    # The loader re-detects the function type by importing the script
    assert loaded_metadata.control_python_function_type == 0
    assert loaded_metadata.control_python_function_parameters == ""
    assert loaded_metadata.control_channel_indices == [0]
    assert loaded_metadata.output_channel_indices == [1]
    assert loaded_metadata.control_signal is None
    assert loaded_metadata.response_transformation_matrix is None
    assert loaded_metadata.reference_transformation_matrix is None
    assert loaded_metadata.sysid_metadata == transient_metadata.sysid_metadata


# region Instructions
def test_transient_instructions_init():
    transient_instructions = TransientInstructions(
        ENVIRONMENT_NAME, test_level=0.0, repeat=False
    )

    assert isinstance(transient_instructions, TransientInstructions)
    assert isinstance(transient_instructions, EnvironmentInstructions)
    assert transient_instructions.test_level == 0.0
    assert transient_instructions.repeat is False
    transient_instructions.validate()


# region Queues
@pytest.mark.parametrize("use_thread", [True, False])
def test_transient_queues_init(use_thread):
    queue_container = mock_queue_container(use_thread)
    transient_queues = TransientQueues(
        ENVIRONMENT_NAME,
        queue_container.environment_command_queues[QUEUE_NAME],
        queue_container.gui_update_queue,
        queue_container.controller_command_queue,
        queue_container.environment_data_in_queues[QUEUE_NAME],
        queue_container.environment_data_out_queues[QUEUE_NAME],
        queue_container.log_file_queue,
        use_thread,
    )

    assert isinstance(transient_queues, TransientQueues)
    for queue_name in (
        "data_analysis_command_queue",
        "signal_generation_command_queue",
        "spectral_command_queue",
        "collector_command_queue",
        "time_history_to_generate_queue",
    ):
        assert hasattr(transient_queues, queue_name)


# region Environment
def test_transient_environment_init(transient_environment):
    assert isinstance(transient_environment, TransientEnvironment)
    assert isinstance(transient_environment, SysIdEnvironment)
    assert transient_environment.ready
    assert transient_environment.startup
    assert transient_environment.next_drive is None
    for command in (
        GlobalCommands.QUIT,
        GlobalCommands.INITIALIZE_HARDWARE,
        GlobalCommands.INITIALIZE_ENVIRONMENT,
        GlobalCommands.INITIALIZE_SYSTEM_ID,
        GlobalCommands.START_SYSTEM_ID_TRANSFER,
        GlobalCommands.START_ENVIRONMENT,
        TransientCommands.START_CONTROL,
        TransientCommands.STOP_CONTROL,
        TransientCommands.PERFORM_CONTROL_PREDICTION,
        SysIdDataAnalysisCommands.SYSTEM_ID_COMPLETE,
    ):
        assert command in transient_environment.command_map


def test_transient_environment_initialize_hardware(transient_environment):
    hardware_metadata = numeric_hardware_metadata()
    transient_environment.initialize_hardware(hardware_metadata)

    assert transient_environment.hardware_metadata == hardware_metadata
    assert transient_environment.ready


def test_transient_environment_initialize_environment(
    transient_environment, transient_metadata
):
    transient_environment.initialize_hardware(numeric_hardware_metadata())
    transient_environment.initialize_environment(transient_metadata)

    assert transient_environment.environment_metadata is transient_metadata
    # The type-0 control law was loaded from the script file
    assert transient_environment.control_function_type == 0
    assert callable(transient_environment.control_function)
    assert transient_environment.control_function.__name__ == "transient_control"

    (analysis_message,) = get_queue_messages(
        transient_environment.queue_container.data_analysis_command_queue,
        ENVIRONMENT_NAME,
        1,
    )
    assert analysis_message == (
        GlobalCommands.INITIALIZE_ENVIRONMENT,
        ENVIRONMENT_NAME,
    )
    assert transient_environment.ready


def test_transient_environment_initialize_sysid(
    transient_environment, transient_metadata
):
    initialized_environment(transient_environment, transient_metadata)
    new_sysid_metadata = sysid_measurement_metadata(frame_size=100)

    transient_environment.initialize_sysid(new_sysid_metadata)

    assert (
        transient_environment.environment_metadata.sysid_metadata == new_sysid_metadata
    )
    (analysis_message,) = get_queue_messages(
        transient_environment.queue_container.data_analysis_command_queue,
        ENVIRONMENT_NAME,
        1,
    )
    assert analysis_message == (
        SysIdDataAnalysisCommands.INITIALIZE_PARAMETERS,
        new_sysid_metadata,
    )
    assert transient_environment.ready


def test_transient_environment_start_transfer_function(
    transient_environment, transient_metadata
):
    initialized_environment(transient_environment, transient_metadata)

    transient_environment.start_transfer_function(None)

    queues = transient_environment.queue_container
    collector_messages = get_queue_messages(
        queues.collector_command_queue, ENVIRONMENT_NAME, 4
    )
    assert [message for message, _ in collector_messages] == [
        DataCollectorCommands.FORCE_INITIALIZE_COLLECTOR,
        DataCollectorCommands.SET_TEST_LEVEL,
        DataCollectorCommands.ACQUIRE,
        DataCollectorCommands.CLEAR_KURTOSIS_BUFFER,
    ]
    siggen_messages = get_queue_messages(
        queues.signal_generation_command_queue, ENVIRONMENT_NAME, 5
    )
    assert [message for message, _ in siggen_messages] == [
        SignalGenerationCommands.INITIALIZE_PARAMETERS,
        SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,
        SignalGenerationCommands.MUTE,
        SignalGenerationCommands.ADJUST_TEST_LEVEL,
        SignalGenerationCommands.GENERATE_SIGNALS,
    ]
    (analysis_message,) = get_queue_messages(
        queues.data_analysis_command_queue, ENVIRONMENT_NAME, 1
    )
    assert analysis_message[0] == SysIdDataAnalysisCommands.RUN_TRANSFER_FUNCTION
    spectral_messages = get_queue_messages(
        queues.spectral_command_queue, ENVIRONMENT_NAME, 3
    )
    assert [message for message, _ in spectral_messages] == [
        SpectralProcessingCommands.INITIALIZE_PARAMETERS,
        SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING,
        SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,
    ]
    assert transient_environment.sysid_active
    gui_message = queues.gui_update_queue.get(timeout=10)
    assert gui_message == (ENVIRONMENT_NAME, (SysIdUICommands.SYSID_STARTED, None))


# region Control prediction
def test_transient_environment_system_id_complete_runs_prediction(
    transient_environment, transient_metadata
):
    """The real use case: system id completion drives the control law and
    the resulting drive/response prediction math."""
    initialized_environment(transient_environment, transient_metadata)

    data_package = completed_system_id(transient_environment, transient_metadata)

    assert transient_environment.sysid_data == data_package
    assert transient_environment.sysid_stored
    # The type-0 control law returned control_signal * 0.5
    np.testing.assert_allclose(
        transient_environment.next_drive,
        transient_metadata.control_signal * 0.5,
    )
    # The predicted response was computed by convolving the drive with the
    # impulse response of the identified FRF
    assert transient_environment.predicted_response.shape == (1, SIGNAL_SAMPLES)
    assert np.any(transient_environment.predicted_response != 0)

    gui_queue = transient_environment.gui_update_queue
    gui_commands = []
    for _ in range(5):
        gui_message = gui_queue.get(timeout=10)
        # Most messages are (environment_name, (command, data)), but
        # COMPLETED_SYSTEM_ID is (command, (environment_name, data))
        gui_commands.append(
            gui_message[1][0] if gui_message[0] == ENVIRONMENT_NAME else gui_message[0]
        )
    assert UICommands.COMPLETED_SYSTEM_ID in gui_commands
    assert UICommands.SET_ATTR in gui_commands
    assert TransientUICommands.CONTROL_PREDICTIONS in gui_commands


def test_transient_environment_prediction_without_sysid_errors(
    transient_environment, transient_metadata
):
    initialized_environment(transient_environment, transient_metadata)

    transient_environment.perform_control_prediction(False)

    assert transient_environment.next_drive is None
    gui_message = transient_environment.gui_update_queue.get(timeout=10)
    assert gui_message[0] == UICommands.ERROR


# region Control
def test_transient_environment_start_control_startup(
    transient_environment, transient_metadata
):
    initialized_environment(transient_environment, transient_metadata)
    completed_system_id(transient_environment, transient_metadata)
    drain_queue_commands(
        transient_environment.queue_container.data_analysis_command_queue,
        ENVIRONMENT_NAME,
    )
    transient_instructions = TransientInstructions(
        ENVIRONMENT_NAME, test_level=0.0, repeat=False
    )

    transient_environment.start_control(transient_instructions)

    assert not transient_environment.startup
    assert transient_environment.active
    assert transient_environment.test_level == 1.0
    assert isinstance(transient_environment.control_buffer, FrameBuffer)
    assert isinstance(transient_environment.output_buffer, FrameBuffer)

    siggen_messages = get_queue_messages(
        transient_environment.queue_container.signal_generation_command_queue,
        ENVIRONMENT_NAME,
        4,
    )
    assert [message for message, _ in siggen_messages] == [
        SignalGenerationCommands.INITIALIZE_PARAMETERS,
        SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,
        SignalGenerationCommands.SET_TEST_LEVEL,
        SignalGenerationCommands.GENERATE_SIGNALS,
    ]
    assert isinstance(siggen_messages[1][1], TransientSignalGenerator)
    assert siggen_messages[2][1] == 1.0

    # With no acquired data yet, the environment re-enqueues its own
    # START_CONTROL command to keep pumping the acquisition loop
    (environment_message,) = get_queue_messages(
        transient_environment.environment_command_queue, ENVIRONMENT_NAME, 1
    )
    assert environment_message == (TransientCommands.START_CONTROL, None)


def test_transient_environment_control_shutdown_after_last_acquisition(
    transient_environment, transient_metadata
):
    initialized_environment(transient_environment, transient_metadata)
    completed_system_id(transient_environment, transient_metadata)
    transient_instructions = TransientInstructions(
        ENVIRONMENT_NAME, test_level=0.0, repeat=False
    )
    transient_environment.start_control(transient_instructions)

    # Feed the last acquisition (no alignment found in zero data) with the
    # signal generation already shut down: the environment must shut down
    transient_environment.siggen_shutdown_achieved = True
    data_in_queue = transient_environment.queue_container.data_in_queue
    data_in_queue.put((np.zeros((2, SIGNAL_SAMPLES)), True))
    deadline = time.time() + 10
    while data_in_queue.empty() and time.time() < deadline:
        time.sleep(0.01)

    transient_environment.start_control(None)

    assert not transient_environment.active
    assert transient_environment.startup
    gui_queue = transient_environment.gui_update_queue
    gui_commands = []
    while True:
        gui_commands.append(gui_queue.get(timeout=0.5)[1][0])
        if gui_commands[-1] == UICommands.ENVIRONMENT_ENDED:
            break
    assert TransientUICommands.TIME_DATA in gui_commands


def test_transient_environment_stop_environment(
    transient_environment, transient_metadata
):
    initialized_environment(transient_environment, transient_metadata)

    transient_environment.stop_environment(None)

    (siggen_message,) = get_queue_messages(
        transient_environment.queue_container.signal_generation_command_queue,
        ENVIRONMENT_NAME,
        1,
    )
    assert siggen_message == (SignalGenerationCommands.START_SHUTDOWN, None)


def test_transient_environment_quit(transient_environment):
    halt_flag = transient_environment.quit(None)

    assert halt_flag is True
    queues = transient_environment.queue_container
    for queue in (
        queues.data_analysis_command_queue,
        queues.collector_command_queue,
        queues.signal_generation_command_queue,
        queues.spectral_command_queue,
    ):
        (message,) = get_queue_messages(queue, ENVIRONMENT_NAME, 1)
        assert message == (GlobalCommands.QUIT, None)


# region Run loop
@mock.patch(
    "rattlesnake.environment.abstract_environment.Environment.log",
    new=mock.MagicMock(),
)
def test_transient_environment_run_full_lifecycle(
    transient_environment, transient_metadata
):
    """Replays the controller's real message sequence through run().

    VerboseMessageQueue.get is mocked with a finite side_effect so
    exhaustion raises instead of hanging if the loop misbehaves.
    """
    hardware_metadata = numeric_hardware_metadata()
    sysid_metadata = transient_metadata.sysid_metadata
    data_package = sysid_measurement_data_package(sysid_metadata)
    transient_instructions = TransientInstructions(
        ENVIRONMENT_NAME, test_level=0.0, repeat=False
    )

    controller_sequence = [
        (GlobalCommands.INITIALIZE_HARDWARE, hardware_metadata),
        (GlobalCommands.INITIALIZE_ENVIRONMENT, transient_metadata),
        (GlobalCommands.INITIALIZE_SYSTEM_ID, sysid_metadata),
        (GlobalCommands.START_SYSTEM_ID_TRANSFER, None),
        (
            SysIdDataAnalysisCommands.SYSTEM_ID_COMPLETE,
            (sysid_metadata, data_package),
        ),
        (GlobalCommands.STOP_SYSTEM_ID, False),
        (SignalGenerationCommands.SHUTDOWN_ACHIEVED, None),
        (DataCollectorCommands.SHUTDOWN_ACHIEVED, None),
        (SpectralProcessingCommands.SHUTDOWN_ACHIEVED, None),
        (SysIdDataAnalysisCommands.SHUTDOWN_ACHIEVED, None),
        (SystemIdCommands.CHECK_FOR_COMPLETE_SHUTDOWN, None),
        (GlobalCommands.START_ENVIRONMENT, transient_instructions),
        (TransientCommands.STOP_CONTROL, None),
        (GlobalCommands.QUIT, None),
    ]

    mock_shutdown = mock.MagicMock()
    mock_shutdown.is_set.return_value = False
    with mock.patch(
        "rattlesnake.utilities.VerboseMessageQueue.get",
        side_effect=controller_sequence,
    ):
        transient_environment.run(mock_shutdown)

    # The lifecycle ran: the control law computed the drive signal, the
    # sysid handshake completed, and QUIT fanned out to the sub-workers
    np.testing.assert_allclose(
        transient_environment.next_drive,
        transient_metadata.control_signal * 0.5,
    )
    assert transient_environment.sysid_stored
    assert not transient_environment.sysid_active

    queues = transient_environment.queue_container
    for queue in (
        queues.data_analysis_command_queue,
        queues.collector_command_queue,
        queues.signal_generation_command_queue,
        queues.spectral_command_queue,
    ):
        commands = drain_queue_commands(queue, ENVIRONMENT_NAME)
        assert commands[-1] == GlobalCommands.QUIT


# region transient_process
@pytest.mark.parametrize("use_thread", [True, False])
@mock.patch("rattlesnake.environment.transient_sys_id_environment.mp.Process")
@mock.patch("rattlesnake.environment.transient_sys_id_environment.threading.Thread")
@mock.patch("rattlesnake.environment.transient_sys_id_environment.TransientEnvironment")
def test_transient_process(
    mock_environment_class, mock_thread, mock_process, use_thread
):
    queue_container = mock_queue_container(use_thread)
    event_container = mock_event_container(use_thread)

    transient_process(
        ENVIRONMENT_NAME,
        QUEUE_NAME,
        queue_container.environment_command_queues[QUEUE_NAME],
        queue_container.gui_update_queue,
        queue_container.controller_command_queue,
        queue_container.log_file_queue,
        queue_container.environment_data_in_queues[QUEUE_NAME],
        queue_container.environment_data_out_queues[QUEUE_NAME],
        event_container.acquisition_active_event,
        event_container.output_active_event,
        event_container.environment_active_events[QUEUE_NAME],
        event_container.environment_ready_events[QUEUE_NAME],
        event_container.environment_close_events[QUEUE_NAME],
        event_container.environment_sysid_active_events[QUEUE_NAME],
        event_container.environment_sysid_stored_events[QUEUE_NAME],
        event_container.ping_alive_event,
        use_thread,
    )

    mock_environment_class.return_value.run.assert_called()
    worker_class = mock_thread if use_thread else mock_process
    assert worker_class.call_count == 4
