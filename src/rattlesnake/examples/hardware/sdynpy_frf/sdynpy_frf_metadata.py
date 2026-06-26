import openpyxl
import netCDF4 as nc4

import rattlesnake.examples.defaults as defaults

from rattlesnake.hardware.hardware_utilities import Channel
from rattlesnake.hardware.sdynpy_frf_virtual_hardware import (
    SDynPyFRFMetadata,
)

HARDWARE_FILE = defaults.DIRECTORY + "/hardware/sdynpy_frf/sdynpy_frf.npz"


def worksheet_sdynpy_frf_metadata():
    worksheet_dir = defaults.DIRECTORY + "/hardware/sdynpy_frf/sdynpy_frf_v4.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    metadata = SDynPyFRFMetadata.load_metadata_from_workbook(workbook)
    metadata.hardware_file = HARDWARE_FILE
    workbook.close()
    return metadata


def netcdf_sdynpy_frf_metadata():
    netcdf_dir = defaults.DIRECTORY + "/hardware/sdynpy_frf/sdynpy_frf_v4.nc4"
    netcdf_dataset = nc4.Dataset(netcdf_dir)
    metadata = SDynPyFRFMetadata.load_metadata_from_netcdf(netcdf_dataset)
    metadata.hardware_file = HARDWARE_FILE
    netcdf_dataset.close()
    return metadata


def manual_sdynpy_frf_metadata():
    directions = ["X+", "Y+", "Z+"]
    force_nodes = [13, 131, 135]
    excitation_nodes = [1004, 1020, 1065, 1049]

    channel_list = []
    for node in excitation_nodes:
        for direction in directions:
            channel = Channel(
                node_number=node,
                node_direction=direction,
                comment=f"{node}{direction}",
                physical_device="Virtual",
                channel_type="Acceleration",
            )
            channel_list.append(channel)

    for node in force_nodes:
        for direction in directions:
            channel = Channel(
                node_number=node,
                node_direction=direction,
                comment="Force",
                physical_device="Virtual",
                channel_type="Force",
                feedback_device="Input",
            )
            channel_list.append(channel)
    metadata = SDynPyFRFMetadata(
        channel_list=channel_list,
        sample_rate=defaults.SAMPLE_RATE,
        time_per_read=defaults.BUFFER_SIZE,
        time_per_write=defaults.BUFFER_SIZE,
        hardware_file=HARDWARE_FILE,
    )
    return metadata
