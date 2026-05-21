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
from rattlesnake.environment.sine_sys_id_environment import (
    SineMetadata,
    SineInstructions,
)
from rattlesnake.environment.random_vibration_sys_id_environment import (
    RandomVibrationMetadata,
    RandomVibrationInstructions
)
from rattlesnake.environment.sine_sys_id_utilities import SineSpecification
from rattlesnake.process.abstract_sysid_data_analysis import SysIdMetadata

BASE_DIR = "C:/Users/cmlangs/Desktop/ExampleFiles/"
THREADED = True
BUFFER_SIZE = 0.15
TIME_ENVIRONMENT_NAME = "My Time"
MODAL_ENVIRONMENT_NAME = "My Modal"
SINE_ENVIRONMENT_NAME = "My Sine"
TRANSIENT_ENVIRONMENT_NAME = "My Transient"
RANDOM_ENVIRONMENT_NAME = "My Random"


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
    hardware_file = BASE_DIR + "sample_system.npz"

    hardware_metadata = SDynPySystemMetadata(
        channel_list,
        sample_rate=1000,
        time_per_read=BUFFER_SIZE,
        time_per_write=BUFFER_SIZE,
        output_oversample=1,
        hardware_file=hardware_file,
    )
    return hardware_metadata


def make_stream_metadata(environment_name=TIME_ENVIRONMENT_NAME):
    stream_type = StreamType.IMMEDIATELY
    stream_file = BASE_DIR + "streaming4.nc4"
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


# region Sine
def make_sine_environment_metadata(
    hardware_metadata, environment_name=SINE_ENVIRONMENT_NAME
):
    channel_list_bools = [True, True, True, True, True, True]
    sample_rate = hardware_metadata.sample_rate
    samples_per_frame = 50
    number_of_channels = 6
    specification = SineSpecification(
        name="Sine Tone 1",
        start_time=0,
        num_control=1,
        num_breakpoints=2,
    )

    table = specification.breakpoint_table

    table[0]["frequency"] = 1
    table[0]["sweep_type"] = 0  # 0 = linear
    table[0]["sweep_rate"] = 1
    table[0]["amplitude"][0] = 1
    table[0]["phase"][0] = 0  # radians

    table[1]["frequency"] = 10  # you must set frequency
    table[1]["sweep_type"] = 0
    table[1]["sweep_rate"] = 1
    table[1]["amplitude"][0] = 1
    table[1]["phase"][0] = 0

    table["warning"][:] = np.nan
    table["abort"][:] = np.nan

    specifications = [specification]
    ramp_time = 0.5
    buffer_blocks = 2
    control_convergence = 0.15
    update_drives_after_environment = False
    phase_fit = False
    allow_automatic_aborts = False
    tracking_filter_type = 0
    tracking_filter_cutoff = 0.15
    tracking_filter_order = 2
    vk_filter_order = 2
    vk_filter_bandwidth = 2
    vk_filter_blocksize = 1000
    vk_filter_overlap = 0.15
    control_python_script = None
    control_python_class = None
    control_python_parameters = ""
    control_channel_indices = [1]
    output_channel_indices = [3, 4, 5]
    response_transformation_matrix = None
    output_transformation_matrix = None

    return SineMetadata(
        environment_name=environment_name,
        channel_list_bools=channel_list_bools,
        sample_rate=sample_rate,
        samples_per_frame=samples_per_frame,
        number_of_channels=number_of_channels,
        specifications=specifications,
        ramp_time=ramp_time,
        buffer_blocks=buffer_blocks,
        control_convergence=control_convergence,
        update_drives_after_environment=update_drives_after_environment,
        phase_fit=phase_fit,
        allow_automatic_aborts=allow_automatic_aborts,
        tracking_filter_type=tracking_filter_type,
        tracking_filter_cutoff=tracking_filter_cutoff,
        tracking_filter_order=tracking_filter_order,
        vk_filter_order=vk_filter_order,
        vk_filter_bandwidth=vk_filter_bandwidth,
        vk_filter_blocksize=vk_filter_blocksize,
        vk_filter_overlap=vk_filter_overlap,
        control_python_script=control_python_script,
        control_python_class=control_python_class,
        control_python_parameters=control_python_parameters,
        control_channel_indices=control_channel_indices,
        output_channel_indices=output_channel_indices,
        response_transformation_matrix=response_transformation_matrix,
        output_transformation_matrix=output_transformation_matrix,
    )


def make_sys_id_metadata():
    sample_rate = 1000
    sysid_frame_size = 1000
    sysid_averaging_type = "Linear"
    sysid_noise_averages = 1
    sysid_averages = 1
    sysid_exponential_averaging_coefficient = 0.01
    sysid_estimator = "H1"
    sysid_level = 0.01
    sysid_level_ramp_time = 0.5
    sysid_signal_type = "Random"
    sysid_window = "Hann"
    sysid_overlap = 0.5
    sysid_burst_on = 0.5
    sysid_pretrigger = 0.05
    sysid_burst_ramp_fraction = 0.05
    sysid_low_frequency_cutoff = 0
    sysid_high_frequency_cutoff = int(sample_rate / 2)
    stream_file = None
    auto_shutdown = False
    return SysIdMetadata(
        sample_rate=sample_rate,
        sysid_frame_size=sysid_frame_size,
        sysid_averaging_type=sysid_averaging_type,
        sysid_noise_averages=sysid_noise_averages,
        sysid_averages=sysid_averages,
        sysid_exponential_averaging_coefficient=sysid_exponential_averaging_coefficient,
        sysid_estimator=sysid_estimator,
        sysid_level=sysid_level,
        sysid_level_ramp_time=sysid_level_ramp_time,
        sysid_signal_type=sysid_signal_type,
        sysid_window=sysid_window,
        sysid_overlap=sysid_overlap,
        sysid_burst_on=sysid_burst_on,
        sysid_pretrigger=sysid_pretrigger,
        sysid_burst_ramp_fraction=sysid_burst_ramp_fraction,
        sysid_low_frequency_cutoff=sysid_low_frequency_cutoff,
        sysid_high_frequency_cutoff=sysid_high_frequency_cutoff,
        stream_file=stream_file,
        auto_shutdown=auto_shutdown,
    )


# endregion

# region Random
def make_random_environment_metadata(hardware_metadata, environment_name=RANDOM_ENVIRONMENT_NAME):
    channel_list_bools = [True, True, True, True, True, True]
    sample_rate = hardware_metadata.sample_rate
    number_of_channels = 6
    samples_per_frame = 1000
    test_level_ramp_time = 0.5
    cola_window = "Tukey"
    cola_overlap = 0.5
    cola_window_exponent = 0.5
    sigma_clip = 5.0
    update_tf_during_control = False
    frames_in_cpsd = 20
    cpsd_window = "Hann"
    cpsd_overlap = 0.5
    percent_lines_out = 10.0
    allow_automatic_aborts = False
    control_python_script = BASE_DIR + "/random_environment/control_laws.py"
    control_python_function = "buzz_control"
    control_python_function_type = 0
    control_python_function_parameters = ""
    control_channel_indices = [0, 1, 2]
    output_channel_indices = [3, 4, 5]

    
    specification_frequency_lines = np.arange(0, 501, 1)
    specification_cpsd_matrix = np.zeros((501, 3, 3))
    for i in range(len(specification_frequency_lines)):
        specification_cpsd_matrix[i] = np.eye(3)
    specification_warning_matrix = np.full((2, 501, 3), np.nan)
    specification_abort_matrix = np.full((2, 501, 3), np.nan)
    response_transformation_matrix = None
    output_transformation_matrix = None

    return RandomVibrationMetadata(
        environment_name = environment_name,
        channel_list_bools = channel_list_bools,
        sample_rate = sample_rate,
        number_of_channels = number_of_channels,
        samples_per_frame = samples_per_frame,
        test_level_ramp_time = test_level_ramp_time,
        cola_window = cola_window,
        cola_overlap = cola_overlap,
        cola_window_exponent = cola_window_exponent,
        sigma_clip = sigma_clip,
        update_tf_during_control = update_tf_during_control,
        frames_in_cpsd = frames_in_cpsd,
        cpsd_window = cpsd_window,
        cpsd_overlap = cpsd_overlap,
        percent_lines_out = percent_lines_out,
        allow_automatic_aborts = allow_automatic_aborts,
        control_python_script = control_python_script,
        control_python_function = control_python_function,
        control_python_function_type = control_python_function_type,
        control_python_function_parameters = control_python_function_parameters,
        control_channel_indices = control_channel_indices,
        output_channel_indices = output_channel_indices,
        specification_frequency_lines = specification_frequency_lines,
        specification_cpsd_matrix = specification_cpsd_matrix,
        specification_warning_matrix = specification_warning_matrix,
        specification_abort_matrix = specification_abort_matrix,
        response_transformation_matrix = response_transformation_matrix,
        output_transformation_matrix = output_transformation_matrix,
        sysid_metadata = None,
    )


# region Environments
def build_time_environment():
    rattlesnake = RattlesnakeController(threaded=THREADED, timeout=30)
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
    rattlesnake = RattlesnakeController(threaded=THREADED, timeout=30)
    hardware_metadata = make_sdynpy_system_metadata()
    modal_environment_metadata = make_modal_environment_metadata(hardware_metadata)
    modal_stream_metadata = make_stream_metadata(MODAL_ENVIRONMENT_NAME)
    modal_environment_instructions = make_modal_environment_instructions()

    rattlesnake.initialize_hardware(hardware_metadata)
    rattlesnake.initialize_environments([modal_environment_metadata])
    rattlesnake.start_acquisition(modal_stream_metadata)
    rattlesnake.start_environment(modal_environment_instructions)

    return rattlesnake


def build_sine_environment():
    rattlesnake = RattlesnakeController(threaded=THREADED, timeout=30)
    hardware_metadata = make_sdynpy_system_metadata()
    sine_environment_metadata = make_sine_environment_metadata(hardware_metadata)
    sine_sys_id_metadata = make_sys_id_metadata()
    sine_stream_metadata = make_stream_metadata(SINE_ENVIRONMENT_NAME)

    rattlesnake.initialize_hardware(hardware_metadata)
    rattlesnake.initialize_environments([sine_environment_metadata])
    # rattlesnake.preview_system_id_noise(sine_sys_id_metadata, SINE_ENVIRONMENT_NAME)
    # rattlesnake.preview_system_id_transfer(sine_sys_id_metadata, SINE_ENVIRONMENT_NAME)
    # rattlesnake.run_system_id(sine_sys_id_metadata, SINE_ENVIRONMENT_NAME)
    # rattlesnake.start_acquisition(sine_stream_metadata)

    return rattlesnake

def build_transient_environment():
    rattlesnake = RattlesnakeController(threaded=THREADED, timeout=30)
    hardware_metadata = make_sdynpy_system_metadata()

    rattlesnake.initialize_hardware(hardware_metadata)

    return rattlesnake

def build_random_environment():
    rattlesnake = RattlesnakeController(threaded=THREADED, timeout=30)
    hardware_metadata = make_sdynpy_system_metadata()
    random_environment_metadata = make_random_environment_metadata(hardware_metadata)
    random_sys_id_metadata = make_sys_id_metadata()
    random_stream_metadata = make_stream_metadata(RANDOM_ENVIRONMENT_NAME)

    rattlesnake.initialize_hardware(hardware_metadata)
    rattlesnake.initialize_environments([random_environment_metadata])
    # rattlesnake.run_system_id(random_sys_id_metadata, RANDOM_ENVIRONMENT_NAME)
    # rattlesnake.start_acquisition(random_stream_metadata)

    return rattlesnake



# endregion

# region Startup
if __name__ == "__main__":
    # rattlesnake = RattlesnakeController(threaded=True, timeout=30)
    # rattlesnake = build_time_environment()
    # rattlesnake = build_modal_environment()
    # rattlesnake = build_sine_environment()
    rattlesnake = build_transient_environment()
    # rattlesnake = build_random_environment()

    launch_rattlesnake_ui(rattlesnake)
# endregion
