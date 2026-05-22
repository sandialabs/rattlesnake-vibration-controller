import openpyxl

import hardware.defaults as defaults

from rattlesnake.hardware.hardware_utilities import Channel
from rattlesnake.hardware.sdynpy_system_virtual_hardware import SDynPySystemMetadata

HARDWARE_FILE = defaults.DIRECTORY + "/sdynpy_system.npz"

def worksheet_sdynpy_system_metadata():
    worksheet_dir = defaults.DIRECTORY + "/sdynpy_system.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    channel_list = SDynPySystemMetadata.load_channel_table_from_workbook(workbook)
    metadata = SDynPySystemMetadata(
        channel_list=channel_list,
        sample_rate=defaults.SAMPLE_RATE,
        time_per_read=defaults.BUFFER_SIZE,
        time_per_write=defaults.BUFFER_SIZE,
        output_oversample=defaults.OUTPUT_OVERSAMPLE,
        hardware_file=HARDWARE_FILE,
    )
    return metadata

def template_sdynpy_system_metadata():
    pass

def manual_sdynpy_system_metadata():
    directions = ["X+", "Y+", "Z+"]
    force_nodes = [11, 15, 131, 135]
    excitation_nodes = [1004, 1012, 1020, 1065, 1057, 1049]

    channel_list = []
    for node in force_nodes:
        for direction in directions:
            channel = Channel(
                node_number=node,
                node_direction=direction,
                comment="Force",
                physical_device="Virtual",
                channel_type="Force",
                feedback_device="Virtual",
            )
            channel_list.append(channel)

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

    metadata = SDynPySystemMetadata(
        channel_list=channel_list,
        sample_rate=hardware_defaults.SAMPLE_RATE,
        time_per_read=hardware_defaults.BUFFER_SIZE,
        time_per_write=hardware_defaults.BUFFER_SIZE,
        output_oversample=hardware_defaults.OUTPUT_OVERSAMPLE,
        hardware_file=HARDWARE_FILE,
    )
    return metadata