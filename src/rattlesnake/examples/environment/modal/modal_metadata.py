import openpyxl
import netCDF4 as nc4

import rattlesnake.examples.defaults as defaults

from rattlesnake.utilities import GlobalCommands
from rattlesnake.load_utilities import load_profile_from_workbook
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.modal_environment import (
    ModalCommands,
    ModalMetadata,
    ModalInstructions,
)

ENVIRONMENT_NAME = "Modal 0"


def worksheet_modal_metadata(hardware_metadata):
    worksheet_dir = defaults.DIRECTORY + "/environment/modal/modal_v4.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    worksheet = workbook[ENVIRONMENT_NAME]

    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    metadata = ModalMetadata.load_metadata_from_worksheet(
        worksheet, ENVIRONMENT_NAME, channel_list_bools, hardware_metadata
    )

    return metadata


def netcdf_modal_metadata(hardware_metadata):
    netcdf_dir = defaults.DIRECTORY + "/environment/modal/modal_v4.nc4"
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
    reference_channel_indices = [12, 13, 14]
    response_channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    output_channel_indices = [12, 13, 14, 15, 16, 17, 18, 19, 20]
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


def modal_instructions():
    modal_instructions = ModalInstructions(ENVIRONMENT_NAME)
    return modal_instructions


def modal_event_list():
    timestamp = 0
    command = GlobalCommands.START_STREAMING
    start_stream_event = ProfileEvent(timestamp, "Global", command)

    timestamp = 0
    command = GlobalCommands.START_ENVIRONMENT
    instructions = modal_instructions()
    start_environment_event = ProfileEvent(
        timestamp, ENVIRONMENT_NAME, command, instructions
    )

    timestamp = 5
    command = GlobalCommands.STOP_ENVIRONMENT
    stop_environment_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command)

    timestamp = 6
    command = ModalCommands.CHANGE_SAVEFILE
    data = defaults.DIRECTORY + "/environment/modal/modal_profile_example.nc4"
    change_savefile_event = ProfileEvent(timestamp, ENVIRONMENT_NAME, command, data)

    timestamp = 8
    command = GlobalCommands.START_ENVIRONMENT
    instructions = modal_instructions()
    start_environment_event_2 = ProfileEvent(
        timestamp, ENVIRONMENT_NAME, command, instructions
    )

    timestamp = 15
    command = GlobalCommands.STOP_ENVIRONMENT
    stop_environment_event_2 = ProfileEvent(timestamp, ENVIRONMENT_NAME, command)

    timestamp = 15
    command = GlobalCommands.STOP_STREAMING
    stop_stream_event = ProfileEvent(timestamp, "Global", command)

    timestamp = 15
    command = GlobalCommands.STOP_HARDWARE
    stop_hardware_event = ProfileEvent(timestamp, "Global", command)

    event_list = [
        start_stream_event,
        start_environment_event,
        stop_environment_event,
        change_savefile_event,
        start_environment_event_2,
        stop_environment_event_2,
        stop_stream_event,
        stop_hardware_event,
    ]

    return event_list


def worksheet_modal_event_list():
    worksheet_dir = defaults.DIRECTORY + "/environment/modal/modal_v4.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    environment_types = {
        "Global": "Global",
        ENVIRONMENT_NAME: EnvironmentType.MODAL,
    }
    event_list = load_profile_from_workbook(workbook, environment_types)
    save_event = event_list[3]
    save_event.data = defaults.DIRECTORY + r"\environment\modal\example_save_file.nc4"
    return event_list
