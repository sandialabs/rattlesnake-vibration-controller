import numpy as np
import openpyxl
import netCDF4 as nc4

import rattlesnake.examples.defaults as defaults

from rattlesnake.environment.transient_sys_id_environment import (
    TransientCommands,
    TransientMetadata,
    TransientInstructions,
)

ENVIRONMENT_NAME = "Transient 0"


def transient_instructions():
    pass


def netcdf_transient_metadata(hardware_metadata):
    pass


def worksheet_transient_metadata(hardware_metadata):
    pass


def manual_transient_metadata(hardware_metadata):
    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    sample_rate = hardware_metadata.sample_rate
    number_of_channels = 21
    control_signal = create_control_signal()
    ramp_time = 0.5
    control_python_script = (
        defaults.DIRECTORY + "/control_laws/transient_control_laws.py"
    )
    control_python_function = "pseudoinverse_control"
    control_python_function_type = 0
    control_python_function_parameters = ""
    control_channel_indices = [9, 10, 11]
    output_channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    response_transformation_matrix = None
    output_transformation_matrix = None

    metadata = TransientMetadata(
        environment_name=ENVIRONMENT_NAME,
        channel_list_bools=channel_list_bools,
        sample_rate=sample_rate,
        number_of_channels=number_of_channels,
        control_signal=control_signal,
        ramp_time=ramp_time,
        control_python_script=control_python_script,
        control_python_function=control_python_function,
        control_python_function_type=control_python_function_type,
        control_python_function_parameters=control_python_function_parameters,
        control_channel_indices=control_channel_indices,
        output_channel_indices=output_channel_indices,
        response_transformation_matrix=response_transformation_matrix,
        output_transformation_matrix=output_transformation_matrix,
    )

    return metadata


def create_control_signal():
    num_samples = defaults.SAMPLE_RATE * 5
    frequency = 2  # Hz sine wave
    t = np.arange(num_samples) / defaults.SAMPLE_RATE
    signal = np.zeros((3, num_samples))
    signal[0, :] = np.sin(2 * np.pi * frequency * t)  # sine wave in first row

    return signal
