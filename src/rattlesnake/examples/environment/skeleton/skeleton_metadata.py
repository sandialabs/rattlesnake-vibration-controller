import openpyxl
import netCDF4 as nc4

import rattlesnake.examples.defaults as defaults

from rattlesnake.utilities import GlobalCommands
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.load_utilities import load_profile_from_workbook
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.skeleton_environment import (
    SkeletonCommands,
    SkeletonMetadata,
    SkeletonInstructions,
)

ENVIRONMENT_NAME = "Skeleton 0"


def worksheet_skeleton_metadata(hardware_metadata):
    worksheet_dir = defaults.DIRECTORY + "/environment/skeleton/skeleton_v4.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    worksheet = workbook[ENVIRONMENT_NAME]

    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    metadata = SkeletonMetadata.load_metadata_from_worksheet(
        worksheet, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata
    )
    return metadata


def netcdf_skeleton_metadata(hardware_metadata):
    netcdf_dir = defaults.DIRECTORY + "/environment/skeleton/skeleton_v4.nc4"
    netcdf_dataset = nc4.Dataset(netcdf_dir)
    netcdf_group = netcdf_dataset.groups[ENVIRONMENT_NAME]

    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    metadata = SkeletonMetadata.load_metadata_from_netcdf(
        netcdf_group, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata
    )

    return metadata


def manual_skeleton_metadata(hardware_metadata):
    # Create signal array
    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    example_window_size = 5

    metadata = SkeletonMetadata(
        environment_name=ENVIRONMENT_NAME,
        channel_list_bools=channel_list_bools,
        sample_rate=hardware_metadata.sample_rate,
        example_window_size=example_window_size,
    )

    return metadata


def skeleton_instructions():
    skeleton_instructions = SkeletonInstructions(ENVIRONMENT_NAME, example_test_level=1)
    return skeleton_instructions


def skeleton_event_list():
    timestamp = 0
    command = GlobalCommands.START_STREAMING
    start_stream_event = ProfileEvent(timestamp, "Global", command)

    timestamp = 0
    command = GlobalCommands.START_ENVIRONMENT
    instructions = skeleton_instructions()
    start_environment_event = ProfileEvent(
        timestamp, ENVIRONMENT_NAME, command, instructions
    )

    timestamp = 1
    command = SkeletonCommands.EXAMPLE_SET_TEST_LEVEL
    data = 2
    set_level_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command, data)

    timestamp = 1
    command = SkeletonCommands.EXAMPLE_FLOAT_COMMAND
    data = 3.5
    example_float_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command, data)

    timestamp = 10
    command = GlobalCommands.STOP_ENVIRONMENT
    stop_environment_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command)

    timestamp = 10
    command = GlobalCommands.STOP_HARDWARE
    stop_hardware_event = ProfileEvent(timestamp, "Global", command)

    return [
        start_stream_event,
        start_environment_event,
        set_level_event,
        example_float_event,
        stop_environment_event,
        stop_hardware_event,
    ]


def worksheet_skeleton_event_list():
    worksheet_dir = defaults.DIRECTORY + "/environment/skeleton/skeleton_v4.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    environment_types = {
        "Global": "Global",
        ENVIRONMENT_NAME: EnvironmentType.SKELETON,
    }
    event_list = load_profile_from_workbook(workbook, environment_types)
    return event_list
