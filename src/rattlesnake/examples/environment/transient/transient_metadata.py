import numpy as np
import openpyxl
import netCDF4 as nc4

import rattlesnake.examples.defaults as defaults

from rattlesnake.utilities import GlobalCommands
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.environment.transient_sys_id_environment import (
    TransientCommands,
    TransientMetadata,
    TransientInstructions,
)

ENVIRONMENT_NAME = "Transient 0"


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
    control_channel_indices = [0, 1, 2]
    output_channel_indices = [12, 13, 14, 15, 16, 17, 18, 19, 20]
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

def transient_instructions():
    instructions = TransientInstructions(ENVIRONMENT_NAME, test_level=0, repeat=True)
    return instructions

def transient_event_list():
    timestamp = 0
    command = GlobalCommands.START_STREAMING
    start_stream_event = ProfileEvent(timestamp, "Global", command)

    timestamp = 0
    command = TransientCommands.SET_REPEAT
    repeat_event = ProfileEvent(
        timestamp, ENVIRONMENT_NAME, command
    )

    timestamp = 0
    command = GlobalCommands.START_ENVIRONMENT
    instructions = transient_instructions()
    start_environment_event = ProfileEvent(
        timestamp, ENVIRONMENT_NAME, command, instructions
    )

    timestamp = 5
    command = GlobalCommands.STOP_ENVIRONMENT
    stop_environment_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command)

    timestamp = 5
    command = TransientCommands.SET_NO_REPEAT
    no_repeat_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command)

    timestamp = 5
    command = TransientCommands.SET_TEST_LEVEL
    data = 5
    set_level_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command, data)

    timestamp = 6.5
    command = GlobalCommands.START_ENVIRONMENT
    instructions = transient_instructions()
    instructions.test_level = 5
    instructions.repeat = False
    start_environment_event_2 = ProfileEvent(
        timestamp, ENVIRONMENT_NAME, command, instructions
    )

    timestamp = 20
    command = GlobalCommands.STOP_STREAMING
    stop_stream_event = ProfileEvent(timestamp, "Global", command)

    timestamp = 20
    command = GlobalCommands.STOP_HARDWARE
    stop_hardware_event = ProfileEvent(timestamp, "Global", command)

    event_list = [
        start_stream_event,
        repeat_event,
        start_environment_event,
        stop_environment_event,
        no_repeat_event,
        set_level_event,
        start_environment_event_2,
        stop_stream_event,
        stop_hardware_event,
    ]

    return event_list