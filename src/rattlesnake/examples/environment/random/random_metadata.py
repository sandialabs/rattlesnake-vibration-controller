import numpy as np
import netCDF4 as nc4
import openpyxl

import rattlesnake.examples.defaults as defaults
from rattlesnake.utilities import GlobalCommands
from rattlesnake.environment.random_vibration_sys_id_environment import (
    RandomVibrationCommands,
    RandomVibrationMetadata,
    RandomVibrationInstructions,
)
from rattlesnake.profile_manager import ProfileEvent

ENVIRONMENT_NAME = "Random 0"


def worksheet_random_metadata(hardware_metadata):

    worksheet_dir = defaults.DIRECTORY + "/environment/random/random_v4.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    worksheet = workbook[ENVIRONMENT_NAME]

    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    metadata = RandomVibrationMetadata.load_metadata_from_worksheet(
        worksheet, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata
    )
    metadata.control_python_script = defaults.DIRECTORY + "/control_laws/control_laws.py"
    metadata.control_python_function = "buzz_control"
    metadata.control_python_function_type = 0
    metadata.control_python_function_parameters = ""
    (
        metadata.specification_frequency_lines,
        metadata.specification_cpsd_matrix,
        metadata.specification_warning_matrix,
        metadata.specification_abort_matrix,
    ) = create_sine_specification(hardware_metadata.sample_rate)

    return metadata


def netcdf_random_metadata(hardware_metadata):
    netcdf_dir = defaults.DIRECTORY + "/environment/random/random.nc4"
    netcdf_dataset = nc4.Dataset(netcdf_dir)
    netcdf_group = netcdf_dataset.groups[ENVIRONMENT_NAME]

    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    metadata = RandomVibrationMetadata.load_metadata_from_netcdf(
        netcdf_group, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata
    )

    return metadata


def manual_random_metadata(hardware_metadata):
    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    sample_rate = hardware_metadata.sample_rate
    number_of_channels = 21
    samples_per_frame = 2048
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
    control_python_script = defaults.DIRECTORY + "/control_laws/control_laws.py"
    control_python_function = "buzz_control"
    control_python_function_type = 0
    control_python_function_parameters = ""
    control_channel_indices = [0, 1, 2]
    output_channel_indices = [12, 13, 14, 15, 16, 17, 18, 19, 20]

    (
        specification_frequency_lines,
        specification_cpsd_matrix,
        specification_warning_matrix,
        specification_abort_matrix,
    ) = create_sine_specification(hardware_metadata.sample_rate)
    response_transformation_matrix = None
    output_transformation_matrix = None

    metadata = RandomVibrationMetadata(
        environment_name=ENVIRONMENT_NAME,
        channel_list_bools=channel_list_bools,
        sample_rate=sample_rate,
        number_of_channels=number_of_channels,
        samples_per_frame=samples_per_frame,
        test_level_ramp_time=test_level_ramp_time,
        cola_window=cola_window,
        cola_overlap=cola_overlap,
        cola_window_exponent=cola_window_exponent,
        sigma_clip=sigma_clip,
        update_tf_during_control=update_tf_during_control,
        frames_in_cpsd=frames_in_cpsd,
        cpsd_window=cpsd_window,
        cpsd_overlap=cpsd_overlap,
        percent_lines_out=percent_lines_out,
        allow_automatic_aborts=allow_automatic_aborts,
        control_python_script=control_python_script,
        control_python_function=control_python_function,
        control_python_function_type=control_python_function_type,
        control_python_function_parameters=control_python_function_parameters,
        control_channel_indices=control_channel_indices,
        output_channel_indices=output_channel_indices,
        specification_frequency_lines=specification_frequency_lines,
        specification_cpsd_matrix=specification_cpsd_matrix,
        specification_warning_matrix=specification_warning_matrix,
        specification_abort_matrix=specification_abort_matrix,
        response_transformation_matrix=response_transformation_matrix,
        output_transformation_matrix=output_transformation_matrix,
        sysid_metadata=None,
    )

    return metadata


def create_sine_specification(sample_rate):
    n_freq = int(sample_rate / 2) + 1
    specification_frequency_lines = np.arange(0, n_freq, 1)
    specification_cpsd_matrix = np.zeros((n_freq, 3, 3))
    for i in range(len(specification_frequency_lines)):
        specification_cpsd_matrix[i] = np.eye(3)
    specification_warning_matrix = np.full((2, n_freq, 3), np.nan)
    specification_abort_matrix = np.full((2, n_freq, 3), np.nan)

    return (
        specification_frequency_lines,
        specification_cpsd_matrix,
        specification_warning_matrix,
        specification_abort_matrix,
    )

def random_instructions():
    control_test_level = 1
    instructions = RandomVibrationInstructions(ENVIRONMENT_NAME, control_test_level)

    return instructions

def random_event_list():
    timestamp = 0
    command = GlobalCommands.START_STREAMING
    start_stream_event = ProfileEvent(timestamp, "Global", command)

    timestamp = 0
    command = GlobalCommands.START_ENVIRONMENT
    instructions = random_instructions()
    start_environment_event = ProfileEvent(
        timestamp, ENVIRONMENT_NAME, command, instructions
    )
    
    timestamp = 5
    command = RandomVibrationCommands.ADJUST_TEST_LEVEL
    data = 5
    adjust_level_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command, data)

    timestamp = 15
    command = GlobalCommands.STOP_ENVIRONMENT
    stop_environment_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command)

    timestamp = 16
    command = RandomVibrationCommands.SAVE_CONTROL_DATA
    data = defaults.DIRECTORY + "/environment/random/random_control_data.nc4"
    save_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command, data)

    timestamp = 18
    command = RandomVibrationCommands.CHANGE_SPECIFICATION
    data = defaults.DIRECTORY + "/environment/random/random_spec.npz"
    change_spec_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command, data)

    timestamp = 20
    command = GlobalCommands.START_ENVIRONMENT
    instructions = random_instructions()
    start_environment_event_2 = ProfileEvent(
        timestamp, ENVIRONMENT_NAME, command, instructions
    )

    timestamp = 40
    command = GlobalCommands.STOP_ENVIRONMENT
    stop_environment_event_2 = ProfileEvent(timestamp, ENVIRONMENT_NAME, command)

    timestamp = 40
    command = GlobalCommands.STOP_STREAMING
    stop_stream_event = ProfileEvent(timestamp, "Global", command)

    timestamp = 40
    command = GlobalCommands.STOP_HARDWARE
    stop_hardware_event = ProfileEvent(timestamp, "Global", command)

    event_list = [
        start_stream_event,
        start_environment_event,
        adjust_level_event,
        stop_environment_event,
        save_event,
        change_spec_event,
        start_environment_event_2,
        stop_environment_event_2,
        stop_stream_event,
        stop_hardware_event,
    ]

    return event_list