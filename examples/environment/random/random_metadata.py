import numpy as np
import openpyxl

import defaults

from rattlesnake.environment.random_vibration_sys_id_environment import RandomVibrationCommands, RandomVibrationMetadata, RandomVibrationInstructions

ENVIRONMENT_NAME = "Random 0"

def random_instructions():
    control_test_level = 1
    instructions = RandomVibrationInstructions(ENVIRONMENT_NAME, control_test_level)

    return instructions

def worksheet_random_metadata(hardware_metadata):

    pass

def netcdf_random_metadata(hardware_metadata):
    pass

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
    control_channel_indices = [9, 10, 11]
    output_channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    
    n_freq = int(hardware_metadata.sample_rate/2) + 1
    specification_frequency_lines = np.arange(0, n_freq, 1)
    specification_cpsd_matrix = np.zeros((n_freq, 3, 3))
    for i in range(len(specification_frequency_lines)):
        specification_cpsd_matrix[i] = np.eye(3)
    specification_warning_matrix = np.full((2, n_freq, 3), np.nan)
    specification_abort_matrix = np.full((2, n_freq, 3), np.nan)
    response_transformation_matrix = None
    output_transformation_matrix = None

    metadata =  RandomVibrationMetadata(
        environment_name = ENVIRONMENT_NAME,
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

    return metadata