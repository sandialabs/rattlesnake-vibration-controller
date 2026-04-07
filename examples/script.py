import numpy as np

from rattlesnake.engine import RattlesnakeController
from rattlesnake.main import launch_rattlesnake_ui
from rattlesnake.utilities import GlobalCommands
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.hardware.hardware_utilities import Channel
from rattlesnake.hardware.sdynpy_system_virtual_hardware import SDynPySystemMetadata
from rattlesnake.process.streaming import StreamMetadata, StreamType
from rattlesnake.environment.time_environment import (
    TimeMetadata,
    TimeInstructions,
    TimeCommands,
)
from rattlesnake.environment.modal_environment import (
    ModalMetadata,
    ModalInstructions,
    ModalCommands,
)

BUFFER_SIZE = 0.05
TIME_ENVIRONMENT_NAME = "My Time"
MODAL_ENVIRONMENT_NAME = "My Modal"
SINE_ENVIRONMENT_NAME = "My Sine"


# region Hardware
def make_sdynpy_system_metadata():
    excitation_1 = Channel()
    excitation_1.node_number = 2000038
    excitation_1.node_direction = "X+"
    excitation_1.comment = "2000038X+"
    excitation_1.physical_device = "Virtual"
    excitation_1.channel_type = "Acceleration"
    excitation_2 = Channel()
    excitation_2.node_number = 2000038
    excitation_2.node_direction = "Y+"
    excitation_2.comment = "2000038Y+"
    excitation_2.physical_device = "Virtual"
    excitation_2.channel_type = "Acceleration"
    excitation_3 = Channel()
    excitation_3.node_number = 2000038
    excitation_3.node_direction = "Z+"
    excitation_3.comment = "2000038Z+"
    excitation_3.physical_device = "Virtual"
    excitation_3.channel_type = "Acceleration"
    force_1 = Channel()
    force_1.node_number = 201
    force_1.node_direction = "X+"
    force_1.comment = "Force"
    force_1.physical_device = "Virtual"
    force_1.channel_type = "Force"
    force_1.feedback_device = "Virtual"
    force_2 = Channel()
    force_2.node_number = 201
    force_2.node_direction = "Y+"
    force_2.comment = "Force"
    force_2.physical_device = "Virtual"
    force_2.channel_type = "Force"
    force_2.feedback_device = "Virtual"
    force_3 = Channel()
    force_3.node_number = 201
    force_3.node_direction = "Z+"
    force_3.comment = "Force"
    force_3.physical_device = "Virtual"
    force_3.channel_type = "Force"
    force_3.feedback_device = "Virtual"
    channel_list = [excitation_1, excitation_2, excitation_3, force_1, force_2, force_3]

    hardware_metadata = SDynPySystemMetadata(
        channel_list,
        sample_rate=1000,
        time_per_read=BUFFER_SIZE,
        time_per_write=BUFFER_SIZE,
        output_oversample=1,
        hardware_file="E:/Rattlesnake/SampleData/sample_system.npz",
    )
    return hardware_metadata


def make_stream_metadata(environment_name=TIME_ENVIRONMENT_NAME):
    stream_type = StreamType.IMMEDIATELY
    stream_file = "E:/Rattlesnake/SampleData/streaming4.nc4"
    test_level_environment_name = environment_name
    stream_metadata = StreamMetadata(
        stream_type, stream_file, test_level_environment_name
    )

    return stream_metadata


# endregion


# region Time
def make_time_environment_metadata(
    hardware_metadata, environment_name=TIME_ENVIRONMENT_NAME
):
    num_rows = 3
    num_samples = 10000
    sample_rate = hardware_metadata.sample_rate  # Hz
    frequency = 2  # Hz sine wave

    # Create time vector
    t = np.arange(num_samples) / sample_rate

    # Create signal array
    signal = np.zeros((num_rows, num_samples))
    signal[0, :] = np.sin(2 * np.pi * frequency * t)  # sine wave in first row

    channel_list_bools = [True, True, True, True, True, True]
    cancel_rampdown_time = 0.5
    time_metadata = TimeMetadata(
        environment_name, channel_list_bools, sample_rate, signal, cancel_rampdown_time
    )

    return time_metadata


def make_time_environment_event_list(environment_name=TIME_ENVIRONMENT_NAME):
    timestamp = 0
    command = GlobalCommands.START_STREAMING
    start_stream_event = ProfileEvent(timestamp, "Global", command)

    timestamp = 2
    command = GlobalCommands.START_ENVIRONMENT
    time_instructions = make_time_environment_instructions(environment_name)
    start_environment_event = ProfileEvent(
        timestamp, environment_name, command, time_instructions
    )

    timestamp = 4
    command = TimeCommands.SET_TEST_LEVEL
    data = 2
    set_level_event = ProfileEvent(timestamp, environment_name, command, data)

    timestamp = 6
    command = GlobalCommands.STOP_ENVIRONMENT
    stop_environment_event = ProfileEvent(timestamp, environment_name, command)

    timestamp = 8
    command = GlobalCommands.STOP_STREAMING
    stop_stream_event = ProfileEvent(timestamp, "Global", command)

    profile_event_list = [
        start_stream_event,
        start_environment_event,
        set_level_event,
        stop_environment_event,
        stop_stream_event,
    ]
    return profile_event_list


def make_time_environment_instructions(environment_name=TIME_ENVIRONMENT_NAME):
    current_test_level = 1
    repeat = True
    time_instructions = TimeInstructions(environment_name, current_test_level, repeat)

    return time_instructions


# endregion


# region Modal
def make_modal_environment_metadata(
    hardware_metadata, environment_name=MODAL_ENVIRONMENT_NAME
):
    channel_list_bools = [True, True, True, True, True, True]
    sample_rate = hardware_metadata.sample_rate
    samples_per_frame = 1000
    averaging_type = "Linear"
    num_averages = 30
    averaging_coefficient = 0.1
    frf_technique = "H1"
    frf_window = "rectangle"
    overlap_percent = 0
    trigger_type = "Free Run"
    accept_type = "Accept All"
    wait_for_steady_state = 0
    trigger_channel = 0
    pretrigger_percent = 0
    trigger_slope_positive = True
    trigger_level_percent = 0
    hysteresis_level_percent = 0
    hysteresis_frame_percent = 0
    signal_generator_type = "random"
    signal_generator_level = 0.01
    signal_generator_min_frequency = 0
    signal_generator_max_frequency = 500
    signal_generator_on_percent = 0
    acceptance_function = None
    reference_channel_indices = [3, 4]
    response_channel_indices = [0, 1, 2, 5]
    output_channel_indices = [3, 4, 5]
    output_oversample = hardware_metadata.output_oversample
    exponential_window_value_at_frame_end = 0.25

    return ModalMetadata(
        environment_name,
        channel_list_bools,
        sample_rate,
        samples_per_frame,
        averaging_type,
        num_averages,
        averaging_coefficient,
        frf_technique,
        frf_window,
        overlap_percent,
        trigger_type,
        accept_type,
        wait_for_steady_state,
        trigger_channel,
        pretrigger_percent,
        trigger_slope_positive,
        trigger_level_percent,
        hysteresis_level_percent,
        hysteresis_frame_percent,
        signal_generator_type,
        signal_generator_level,
        signal_generator_min_frequency,
        signal_generator_max_frequency,
        signal_generator_on_percent,
        acceptance_function,
        reference_channel_indices,
        response_channel_indices,
        output_channel_indices,
        output_oversample,
        exponential_window_value_at_frame_end,
    )


def make_modal_environment_instructions(environment_name=MODAL_ENVIRONMENT_NAME):
    modal_instructions = ModalInstructions(environment_name)
    return modal_instructions


# endregion


# region Environments
def build_time_environment():
    rattlesnake = RattlesnakeController(threaded=True, timeout=30)
    hardware_metadata = make_sdynpy_system_metadata()
    time_environment_metadata = make_time_environment_metadata(hardware_metadata)
    time_profile_event_list = make_time_environment_event_list()
    time_stream_metadata = make_stream_metadata(TIME_ENVIRONMENT_NAME)
    time_environment_instructions = make_time_environment_instructions()

    rattlesnake.initialize_hardware(hardware_metadata)
    rattlesnake.initialize_environments([time_environment_metadata])
    rattlesnake.initialize_profile_event_list(time_profile_event_list)
    rattlesnake.set_stream_metadata(time_stream_metadata)
    rattlesnake.start_acquisition(time_stream_metadata)
    rattlesnake.start_environment(time_environment_instructions)

    return rattlesnake


def build_modal_environment():
    rattlesnake = RattlesnakeController(threaded=True, timeout=30)
    hardware_metadata = make_sdynpy_system_metadata()
    modal_environment_metadata = make_modal_environment_metadata(hardware_metadata)
    modal_stream_metadata = make_stream_metadata(MODAL_ENVIRONMENT_NAME)
    modal_environment_instructions = make_modal_environment_instructions()

    rattlesnake.initialize_hardware(hardware_metadata)
    rattlesnake.initialize_environments([modal_environment_metadata])
    # rattlesnake.start_acquisition(modal_stream_metadata)
    # rattlesnake.start_environment(modal_environment_instructions)

    return rattlesnake


# endregion

# region Startup
if __name__ == "__main__":
    # rattlesnake = build_time_environment()
    rattlesnake = build_modal_environment()

    launch_rattlesnake_ui(rattlesnake)
# endregion
