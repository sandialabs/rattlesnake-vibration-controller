import openpyxl
import netCDF4 as nc4

import rattlesnake.examples.defaults as defaults

from rattlesnake.environment.modal_environment import (
    ModalCommands,
    ModalMetadata,
    ModalInstructions,
)

ENVIRONMENT_NAME = "Modal 0"


def modal_instructions():
    modal_instructions = ModalInstructions(ENVIRONMENT_NAME)
    return modal_instructions


def worksheet_modal_metadata(hardware_metadata):
    worksheet_dir = defaults.DIRECTORY + "/environment/modal/modal.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    worksheet = workbook[ENVIRONMENT_NAME]

    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    metadata = ModalMetadata.load_metadata_from_worksheet(
        worksheet, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata
    )

    return metadata


def netcdf_modal_metadata(hardware_metadata):
    netcdf_dir = defaults.DIRECTORY + "/environment/modal/modal.nc4"
    netcdf_dataset = nc4.Dataset(netcdf_dir)
    netcdf_group = netcdf_dataset.groups[ENVIRONMENT_NAME]

    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    metadata = ModalMetadata.load_metadata_from_netcdf(
        netcdf_group, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata
    )

    return metadata


def manual_modal_metadata(hardware_metadata):
    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    sample_rate = hardware_metadata.sample_rate
    samples_per_frame = 1000
    averaging_type = "Linear"
    num_averages = 30
    averaging_coefficient = 0.1
    frf_technique = "H1"
    frf_window = "rectangle"
    overlap_percent = 0
    trigger_type = "Free Run"
    accept_type = "Accept All"
    wait_for_steady_state = 0
    trigger_channel = 0
    pretrigger_percent = 0
    trigger_slope_positive = True
    trigger_level_percent = 0
    hysteresis_level_percent = 0
    hysteresis_frame_percent = 0
    signal_generator_type = "random"
    signal_generator_level = 0.01
    signal_generator_min_frequency = 0
    signal_generator_max_frequency = 500
    signal_generator_on_percent = 0
    acceptance_function = None
    reference_channel_indices = [12, 13, 14, 15, 16, 17, 18, 19, 20]
    response_channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    output_channel_indices = [12, 13, 14]
    output_oversample = hardware_metadata.output_oversample
    exponential_window_value_at_frame_end = 0.25

    return ModalMetadata(
        ENVIRONMENT_NAME,
        channel_list_bools,
        sample_rate,
        samples_per_frame,
        averaging_type,
        num_averages,
        averaging_coefficient,
        frf_technique,
        frf_window,
        overlap_percent,
        trigger_type,
        accept_type,
        wait_for_steady_state,
        trigger_channel,
        pretrigger_percent,
        trigger_slope_positive,
        trigger_level_percent,
        hysteresis_level_percent,
        hysteresis_frame_percent,
        signal_generator_type,
        signal_generator_level,
        signal_generator_min_frequency,
        signal_generator_max_frequency,
        signal_generator_on_percent,
        acceptance_function,
        reference_channel_indices,
        response_channel_indices,
        output_channel_indices,
        output_oversample,
        exponential_window_value_at_frame_end,
    )
