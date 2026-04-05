import numpy as np
import netCDF4 as nc4
import openpyxl

from rattlesnake.utilities import RattlesnakeError, GlobalCommands
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.hardware.hardware_registry import HARDWARE_METADATA
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.environment_registry import (
    ENVIRONMENT_METADATA,
    ENVIRONMENT_COMMANDS,
)


def load_metadata_from_netcdf(dataset):
    """Loads a test file using a file dialog"""
    hardware_type = HardwareType(dataset.hardware)
    hardware_metadata_class = HARDWARE_METADATA[hardware_type]
    hardware_metadata = hardware_metadata_class.load_metadata_from_netcdf(dataset)

    # Environments
    environment_metadata_list = []
    for environment_index, environment_name in enumerate(
        dataset.variables["environment_names"][...],
    ):
        # Discover environment type
        environment_group = dataset.groups[environment_name]
        try:
            environment_type_int = dataset.variables["environment_types"][
                environment_index
            ]
            environment_type = EnvironmentType(environment_type_int)
        except:
            environment_type = discover_environment_type_in_old_netcdf(
                environment_group
            )

        # Create metadata class and append to list
        environment_metadata_class = ENVIRONMENT_METADATA[environment_type]
        channel_list_bools = dataset.variables["environment_active_channels"][
            :, environment_index
        ]
        environment_metadata = environment_metadata_class.load_metadata_from_netcdf(
            environment_group,
            environment_name,
            channel_list_bools,
            hardware_metadata,
        )
        environment_metadata_list.append(environment_metadata)


def discover_environment_type_in_old_netcdf(environment_group):
    if hasattr(environment_group, "cancel_rampdown_time"):
        return EnvironmentType.TIME
    else:
        raise RattlesnakeError("Invalid netcdf4 file")


def load_metadata_from_workbook(workbook):
    hardware_sheet = workbook["Hardware"]
    for row in hardware_sheet.rows:
        name = str(row[0].value).lower().strip().replace(" ", "_")
        value = row[1].value
        if value is None or value == "":
            continue
        match name:
            case "hardware_type":
                hardware_type_int = int(value)
                break
    hardware_type = HardwareType(hardware_type_int)

    hardware_metadata_class = HARDWARE_METADATA[hardware_type]
    hardware_metadata = hardware_metadata_class.load_metadata_from_workbook(workbook)

    environment_names = []
    environment_channel_list_bools = {}
    sheets = workbook.sheetnames
    if len(sheets) > 1:
        sheets = [sheet for sheet in sheets if "channel" in sheet.lower()]
    channel_sheet = workbook[sheets[0]]
    col = 24
    num_channels = len(hardware_metadata.channel_list)
    while True:
        environment_name = channel_sheet.cell(row=2, column=col).value

        # Stop if empty or None
        if environment_name is None or str(environment_name).strip() == "":
            break

        # Build environment channel list
        environment_active_channels = [False] * num_channels
        for i in range(num_channels):
            row = 3 + i
            value = channel_sheet.cell(row=row, column=col).value

            if value is not None and str(value).strip() != "":
                environment_active_channels[i] = True

        environment_names.append(environment_name)
        environment_channel_list_bools[environment_name] = environment_active_channels
        col += 1

    environment_metadata_list = []
    environment_types = {"Global": "Global"}
    for environment_name in environment_names:
        environment_sheet = workbook[environment_name]
        environment_type_name = environment_sheet.cell(row=1, column=2).value
        environment_type_name = str(environment_type_name).upper()
        environment_type = EnvironmentType[environment_type_name]
        environment_types[environment_name] = environment_type

        environment_metadata_class = ENVIRONMENT_METADATA[environment_type]
        channel_list_bools = environment_channel_list_bools[environment_name]
        environment_metadata = environment_metadata_class.load_metadata_from_worksheet(
            environment_sheet,
            environment_name,
            channel_list_bools,
            hardware_metadata,
        )
        environment_metadata_list.append(environment_metadata)

    profile_event_list = load_profile_from_workbook(workbook, environment_types)

    return (hardware_metadata, environment_metadata_list, profile_event_list)


def save_rattlesnake_to_workbook(
    workbook,
    hardware_metadata=None,
    environment_metadata_list=None,
    profile_event_list=None,
):
    # Open workbook and save to blank template
    channel_worksheet = workbook.active
    HardwareMetadata.save_blank_hardware_to_workbook(workbook)

    # Save hardware metadata values
    if hardware_metadata is not None and hardware_metadata.hardware_type != "Select":
        hardware_metadata.save_metadata_to_workbook(workbook)

    # Save environment metadata values
    channel_worksheet.cell(row=1, column=24, value="Environments")
    if environment_metadata_list:
        for col, environment_metadata in enumerate(environment_metadata_list):
            col_idx = col + 24
            environment_name = environment_metadata.environment_name
            channel_worksheet.cell(row=2, column=col_idx, value=environment_name)
            bool_indices = environment_metadata.map_channel_indices()
            for row in bool_indices:
                row_idx = row + 3
                channel_worksheet.cell(row=row_idx, column=col_idx, value="x")
            environment_worksheet = workbook.create_sheet(environment_name)
            environment_metadata.store_to_worksheet(environment_worksheet)

    # Save profile event list
    profile_sheet = workbook.create_sheet("Test Profile")
    profile_sheet.cell(1, 1, "Time (s)")
    profile_sheet.cell(1, 2, "Environment")
    profile_sheet.cell(1, 3, "Operation")
    profile_sheet.cell(1, 4, "Data")
    # Fill out values
    if profile_event_list:
        for row, event in enumerate(profile_event_list):
            row_idx = row + 2
            profile_sheet.cell(row_idx, 1, str(event.timestamp))
            profile_sheet.cell(row_idx, 2, event.environment_name)
            profile_sheet.cell(row_idx, 3, event.command.label)
            profile_sheet.cell(row_idx, 4, str(event.data))


def load_profile_from_workbook(workbook, environment_types):
    profile_sheet = workbook["Test Profile"]
    index = 2
    profile_event_list = []
    while True:
        timestamp = profile_sheet.cell(index, 1).value
        if timestamp is None or (
            isinstance(timestamp, str) and timestamp.strip() == ""
        ):
            break
        timestamp = float(timestamp)

        environment_name = profile_sheet.cell(index, 2).value
        environment_type = environment_types[environment_name]

        # I have to conver the command string to an actual command
        command = profile_sheet.cell(index, 3).value
        command = str(command).upper().strip().replace(" ", "_")
        if command in GlobalCommands.__members__:
            command = GlobalCommands[command]
        elif command in ENVIRONMENT_COMMANDS[environment_type].__members__:
            command = ENVIRONMENT_COMMANDS[environment_type][command]
        else:
            raise RattlesnakeError(
                f"Invalid command: {command} for {environment_name} | {environment_type}"
            )

        data = profile_sheet.cell(index, 4).value
        data = None if isinstance(data, str) and not data.strip() else data

        event = ProfileEvent(timestamp, environment_name, command, data)
        profile_event_list.append(event)
        index += 1
    workbook.close()

    return profile_event_list


def save_profile_to_workbook(workbook, profile_event_list):
    profile_sheet = workbook.active
    profile_sheet.title = "Test Profile"
    profile_sheet.cell(1, 1, "Time (s)")
    profile_sheet.cell(1, 2, "Environment")
    profile_sheet.cell(1, 3, "Operation")
    profile_sheet.cell(1, 4, "Data")
    # Fill out values
    if profile_event_list:
        for row, event in enumerate(profile_event_list):
            row_idx = row + 2
            profile_sheet.cell(row_idx, 1, str(event.timestamp))
            profile_sheet.cell(row_idx, 2, event.environment_name)
            profile_sheet.cell(row_idx, 3, event.command.label)
            profile_sheet.cell(row_idx, 4, str(event.data))
