import openpyxl
import netCDF4 as nc4

import rattlesnake.examples.defaults as defaults

from rattlesnake.utilities import GlobalCommands
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.load_utilities import load_profile_from_workbook
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.read_environment import (
    ReadCommands,
    ReadMetadata,
    ReadInstructions,
)

ENVIRONMENT_NAME = "Read 0"


def worksheet_read_metadata(hardware_metadata):
    worksheet_dir = defaults.DIRECTORY + "/environment/read/read_v4.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    worksheet = workbook[ENVIRONMENT_NAME]

    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    metadata = ReadMetadata.load_metadata_from_worksheet(
        worksheet, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata
    )
    return metadata


def netcdf_read_metadata(hardware_metadata):
    netcdf_dir = defaults.DIRECTORY + "/environment/read/read_v4.nc4"
    netcdf_dataset = nc4.Dataset(netcdf_dir)
    netcdf_group = netcdf_dataset.groups[ENVIRONMENT_NAME]

    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    metadata = ReadMetadata.load_metadata_from_netcdf(
        netcdf_group, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata
    )

    return metadata


def manual_read_metadata(hardware_metadata, **overrides):
    channel_list_bools = [True] * len(hardware_metadata.channel_list)

    kwargs = dict(
        environment_name=ENVIRONMENT_NAME,
        channel_list_bools=channel_list_bools,
        sample_rate=hardware_metadata.sample_rate,
    )
    kwargs.update(overrides)
    return ReadMetadata(**kwargs)


def read_instructions():
    read_instructions = ReadInstructions(ENVIRONMENT_NAME, window_size=5)
    return read_instructions


def read_event_list():
    timestamp = 0
    command = GlobalCommands.START_STREAMING
    start_stream_event = ProfileEvent(timestamp, "Global", command)

    timestamp = 0
    command = GlobalCommands.START_ENVIRONMENT
    instructions = read_instructions()
    start_environment_event = ProfileEvent(
        timestamp, ENVIRONMENT_NAME, command, instructions
    )

    timestamp = 5
    command = ReadCommands.CHANGE_WINDOW_SIZE
    data = 2
    set_window_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command, data)

    timestamp = 10
    command = GlobalCommands.STOP_ENVIRONMENT
    stop_environment_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command)

    timestamp = 10
    command = GlobalCommands.STOP_HARDWARE
    stop_hardware_event = ProfileEvent(timestamp, "Global", command)

    return [
        start_stream_event,
        start_environment_event,
        set_window_event,
        stop_environment_event,
        stop_hardware_event,
    ]


def worksheet_read_event_list():
    worksheet_dir = defaults.DIRECTORY + "/environment/read/read_v4.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    environment_types = {
        "Global": "Global",
        ENVIRONMENT_NAME: EnvironmentType.READ,
    }
    event_list = load_profile_from_workbook(workbook, environment_types)
    return event_list
