import openpyxl
import netCDF4 as nc4

import rattlesnake.examples.defaults as defaults

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
