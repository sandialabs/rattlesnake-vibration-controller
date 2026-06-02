import numpy as np
import openpyxl
import netCDF4 as nc4

import rattlesnake.examples.defaults as defaults

from rattlesnake.environment.sine_sys_id_environment import (
    SineCommands,
    SineMetadata,
    SineInstructions,
)
from rattlesnake.environment.sine_sys_id_utilities import SineSpecification

ENVIRONMENT_NAME = "Sine 0"
# ENVIRONMENT_NAME = "sysid"


def sine_instructions():
    control_test_level = 0
    control_tones = None
    control_end_time = None
    control_start_time = None

    instructions = SineInstructions(
        ENVIRONMENT_NAME,
        control_test_level,
        control_tones,
        control_start_time,
        control_end_time,
    )
    return instructions


def worksheet_sine_metadata(hardware_metadata):
    worksheet_dir = defaults.DIRECTORY + "/environment/sine/sine.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    worksheet = workbook[ENVIRONMENT_NAME]

    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    metadata = SineMetadata.load_metadata_from_worksheet(
        worksheet, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata
    )
    specifications = create_sine_specification()
    metadata.specifications = specifications

    return metadata


def netcdf_sine_metadata(hardware_metadata):
    netcdf_dir = defaults.DIRECTORY + "/environment/sine/sine.nc4"
    netcdf_dataset = nc4.Dataset(netcdf_dir)
    netcdf_group = netcdf_dataset.groups[ENVIRONMENT_NAME]

    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    metadata = SineMetadata.load_metadata_from_netcdf(
        netcdf_group, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata
    )

    return metadata


def manual_sine_metadata(hardware_metadata):
    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    sample_rate = hardware_metadata.sample_rate
    samples_per_frame = 50
    number_of_channels = 21
    specifications = create_sine_specification()
    ramp_time = 0.5
    buffer_blocks = 2
    control_convergence = 0.15
    update_drives_after_environment = False
    phase_fit = False
    allow_automatic_aborts = False
    tracking_filter_type = 0
    tracking_filter_cutoff = 0.15
    tracking_filter_order = 2
    vk_filter_order = 2
    vk_filter_bandwidth = 2
    vk_filter_blocksize = 1000
    vk_filter_overlap = 0.15
    control_python_script = None
    control_python_class = None
    control_python_parameters = ""
    control_channel_indices = [0]
    output_channel_indices = [12, 13, 14, 15, 16, 17, 18, 19, 20]
    response_transformation_matrix = None
    output_transformation_matrix = None

    return SineMetadata(
        environment_name=ENVIRONMENT_NAME,
        channel_list_bools=channel_list_bools,
        sample_rate=sample_rate,
        samples_per_frame=samples_per_frame,
        number_of_channels=number_of_channels,
        specifications=specifications,
        ramp_time=ramp_time,
        buffer_blocks=buffer_blocks,
        control_convergence=control_convergence,
        update_drives_after_environment=update_drives_after_environment,
        phase_fit=phase_fit,
        allow_automatic_aborts=allow_automatic_aborts,
        tracking_filter_type=tracking_filter_type,
        tracking_filter_cutoff=tracking_filter_cutoff,
        tracking_filter_order=tracking_filter_order,
        vk_filter_order=vk_filter_order,
        vk_filter_bandwidth=vk_filter_bandwidth,
        vk_filter_blocksize=vk_filter_blocksize,
        vk_filter_overlap=vk_filter_overlap,
        control_python_script=control_python_script,
        control_python_class=control_python_class,
        control_python_parameters=control_python_parameters,
        control_channel_indices=control_channel_indices,
        output_channel_indices=output_channel_indices,
        response_transformation_matrix=response_transformation_matrix,
        output_transformation_matrix=output_transformation_matrix,
    )


def create_sine_specification():
    specification = SineSpecification(
        name="Sine Tone 1",
        start_time=0,
        num_control=1,
        num_breakpoints=4,
    )

    table = specification.breakpoint_table

    table[0]["frequency"] = 1
    table[0]["sweep_type"] = 0  # 0 = linear
    table[0]["sweep_rate"] = 1
    table[0]["amplitude"][0] = 0
    table[0]["phase"][0] = 0  # radians

    table[1]["frequency"] = 10
    table[1]["sweep_type"] = 0
    table[1]["sweep_rate"] = 1
    table[1]["amplitude"][0] = 1
    table[1]["phase"][0] = 0

    table[2]["frequency"] = 15
    table[2]["sweep_type"] = 0
    table[2]["sweep_rate"] = 1
    table[2]["amplitude"][0] = 1
    table[2]["phase"][0] = 0

    table[3]["frequency"] = 20
    table[3]["sweep_type"] = 0
    table[3]["sweep_rate"] = 1
    table[3]["amplitude"][0] = 0.5
    table[3]["phase"][0] = 0

    table["warning"][:] = np.nan
    table["abort"][:] = np.nan

    specifications = [specification]
    return specifications
