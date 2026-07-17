import numpy as np
import openpyxl
import netCDF4 as nc4

import rattlesnake.examples.defaults as defaults

from rattlesnake.utilities import GlobalCommands
from rattlesnake.load_utilities import load_profile_from_workbook
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.time_environment import (
    TimeCommands,
    TimeMetadata,
    TimeInstructions,
)

ENVIRONMENT_NAME = "Time 0"


def create_time_signal(hardware_metadata):
    num_samples = hardware_metadata.sample_rate * 3
    frequency = 2  # Hz sine wave
    t = np.arange(num_samples * hardware_metadata.output_oversample) / (
        hardware_metadata.sample_rate * hardware_metadata.output_oversample
    )
    signal = np.zeros(
        (defaults.NUM_FORCES, num_samples * hardware_metadata.output_oversample)
    )
    signal[0, :] = np.sin(2 * np.pi * frequency * t)  # sine wave in first row

    return signal


def worksheet_time_metadata(hardware_metadata):
    worksheet_dir = defaults.DIRECTORY + "/environment/time/time_v4.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    worksheet = workbook[ENVIRONMENT_NAME]

    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    metadata = TimeMetadata.load_metadata_from_worksheet(
        worksheet, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata
    )

    signal = create_time_signal(hardware_metadata)
    metadata.output_signal = signal
    metadata._signal_file = defaults.DIRECTORY + "/environment/time/time_signal.npy"
    return metadata


def netcdf_time_metadata(hardware_metadata):
    netcdf_dir = defaults.DIRECTORY + "/environment/time/time_v4.nc4"
    netcdf_dataset = nc4.Dataset(netcdf_dir)
    netcdf_group = netcdf_dataset.groups[ENVIRONMENT_NAME]

    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    metadata = TimeMetadata.load_metadata_from_netcdf(
        netcdf_group, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata
    )
    metadata._signal_file = defaults.DIRECTORY + "/environment/time/time_signal.npy"

    return metadata


def manual_time_metadata(hardware_metadata):
    # Create signal array
    signal = create_time_signal(hardware_metadata)
    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    cancel_rampdown_time = 0.5

    metadata = TimeMetadata(
        environment_name=ENVIRONMENT_NAME,
        channel_list_bools=channel_list_bools,
        sample_rate=hardware_metadata.sample_rate,
        output_oversample=hardware_metadata.output_oversample,
        output_signal=signal,
        cancel_rampdown_time=cancel_rampdown_time,
    )
    metadata._signal_file = defaults.DIRECTORY + "/environment/time/time_signal.npy"

    return metadata


def time_instructions():
    current_test_level = 1
    repeat = True
    instructions = TimeInstructions(ENVIRONMENT_NAME, current_test_level, repeat)

    return instructions


def time_event_list():
    timestamp = 0
    command = GlobalCommands.START_STREAMING
    start_stream_event = ProfileEvent(timestamp, "Global", command)

    # This is purely for testing purposes, does not do anything since repeat is set
    # by headless instructions and not the user interface
    timestamp = 0
    command = TimeCommands.SET_REPEAT
    repeat_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command)

    timestamp = 0
    command = GlobalCommands.START_ENVIRONMENT
    instructions = time_instructions()
    start_environment_event = ProfileEvent(
        timestamp, ENVIRONMENT_NAME, command, instructions
    )

    timestamp = 5
    command = GlobalCommands.STOP_ENVIRONMENT
    stop_environment_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command)

    timestamp = 5
    command = TimeCommands.SET_NO_REPEAT
    no_repeat_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command)

    timestamp = 6.5
    command = GlobalCommands.START_ENVIRONMENT
    instructions = time_instructions()
    instructions.repeat = False
    start_environment_event_2 = ProfileEvent(
        timestamp, ENVIRONMENT_NAME, command, instructions
    )

    timestamp = 7
    command = TimeCommands.SET_TEST_LEVEL
    data = 5
    set_level_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command, data)

    timestamp = 10
    command = GlobalCommands.STOP_STREAMING
    stop_stream_event = ProfileEvent(timestamp, "Global", command)

    timestamp = 10
    command = GlobalCommands.STOP_HARDWARE
    stop_hardware_event = ProfileEvent(timestamp, "Global", command)

    event_list = [
        start_stream_event,
        repeat_event,
        start_environment_event,
        stop_environment_event,
        no_repeat_event,
        start_environment_event_2,
        set_level_event,
        stop_stream_event,
        stop_hardware_event,
    ]

    return event_list


def worksheet_time_event_list():
    worksheet_dir = defaults.DIRECTORY + "/environment/time/time_v4.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    environment_types = {
        "Global": "Global",
        ENVIRONMENT_NAME: EnvironmentType.TIME,
    }
    event_list = load_profile_from_workbook(workbook, environment_types)
    return event_list
