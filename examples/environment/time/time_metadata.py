import numpy as np
import openpyxl
import netCDF4 as nc4

import defaults

from rattlesnake.utilities import GlobalCommands
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.environment.time_environment import TimeCommands, TimeMetadata, TimeInstructions

ENVIRONMENT_NAME = "Time 0"

def create_time_signal():
    num_samples = defaults.SAMPLE_RATE * 5
    frequency = 2  # Hz sine wave
    t = np.arange(num_samples) / defaults.SAMPLE_RATE
    signal = np.zeros((defaults.NUM_FORCES, num_samples))
    signal[0, :] = np.sin(2 * np.pi * frequency * t)  # sine wave in first row

    return signal

def worksheet_time_metadata(hardware_metadata):
    worksheet_dir = defaults.DIRECTORY + "/environment/time/time.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    worksheet = workbook[ENVIRONMENT_NAME]

    channel_list_bools = [True] * defaults.NUM_CHANNELS
    metadata = TimeMetadata.load_metadata_from_worksheet(worksheet, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata)

    signal = create_time_signal()
    metadata.output_signal = signal
    return metadata

def netcdf_time_metadata(hardware_metadata):
    netcdf_dir = defaults.DIRECTORY + "/environment/time/time.nc4"
    netcdf_dataset = nc4.Dataset(netcdf_dir)
    netcdf_group = netcdf_dataset.groups[ENVIRONMENT_NAME]

    channel_list_bools = [True] * defaults.NUM_CHANNELS
    metadata = TimeMetadata.load_metadata_from_netcdf(netcdf_group, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata)

    return metadata

def manual_time_metadata(hardware_metadata):
    # Create signal array
    signal = create_time_signal()
    channel_list_bools = [True] * defaults.NUM_CHANNELS
    cancel_rampdown_time = 0.5

    metadata = TimeMetadata(
        environment_name=ENVIRONMENT_NAME,
        channel_list_bools=channel_list_bools,
        sample_rate=defaults.SAMPLE_RATE,
        output_signal=signal,
        cancel_rampdown_time=cancel_rampdown_time,
    )

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

    timestamp = 2
    command = GlobalCommands.START_ENVIRONMENT
    instructions = time_instructions()
    start_environment_event = ProfileEvent(
        timestamp, ENVIRONMENT_NAME, command, instructions
    )

    timestamp = 4
    command = TimeCommands.SET_TEST_LEVEL
    data = 2
    set_level_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command, data)

    timestamp = 6
    command = GlobalCommands.STOP_ENVIRONMENT
    stop_environment_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command)

    timestamp = 8
    command = GlobalCommands.STOP_STREAMING
    stop_stream_event = ProfileEvent(timestamp, "Global", command)

    event_list = [
        start_stream_event,
        start_environment_event,
        set_level_event,
        stop_environment_event,
        stop_stream_event,
    ]

    return event_list