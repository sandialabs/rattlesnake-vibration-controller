"""
Functions for loading and saving Rattlesnake hardware, environment, and profile
metadata from netCDF4 files and Excel workbooks.

Rattlesnake Vibration Control Software
Copyright (C) 2021  National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple

import netCDF4
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from rattlesnake.utilities import GlobalCommands, RattlesnakeError
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.user_interface.ui_utilities import UICommands
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.hardware.hardware_registry import HARDWARE_METADATA
from rattlesnake.environment.environment_registry import (
    ENVIRONMENT_COMMANDS,
    ENVIRONMENT_METADATA,
)
from rattlesnake.profile_manager import ProfileEvent

if TYPE_CHECKING:
    from rattlesnake.environment.abstract_environment import EnvironmentMetadata


# TODO: The discover_environment_type logic will need to be removed eventually. It is
# still here to allow certain netcdf to be loaded from version 3.0
def load_metadata_from_netcdf(
    dataset: netCDF4.Dataset,
) -> Tuple[HardwareMetadata, List[EnvironmentMetadata]]:
    """
    Loads hardware and environment metadata from an open netCDF4 dataset.

    The hardware type is read from the dataset's ``hardware`` attribute and loads the
    metadata for that specific hardware class.

    Parameters
    ----------
    dataset : netCDF4.Dataset
        An open netCDF4 dataset containing Rattlesnake hardware and environment
        metadata.

    Returns
    -------
    hardware_metadata : HardwareMetadata
        The hardware metadata loaded from the dataset.
    environment_metadata_list : List[EnvironmentMetadata]
        A list of environment metadata objects, one per environment stored in the
        dataset.

    Raises
    ------
    RattlesnakeError
        If an environment's type cannot be determined.

    Unit Tests
    ----------
    """
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
        except (KeyError, ValueError):
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

    return (hardware_metadata, environment_metadata_list)


# Maps an attribute that only one environment type writes to its netCDF4 group
# to that environment type. Used to identify environments in netCDF4 files saved
# before the environment type was stored explicitly.
_LEGACY_ENVIRONMENT_TYPE_ATTRIBUTES: Dict[str, EnvironmentType] = {
    "tracking_filter_type": EnvironmentType.SINE,
    "update_tf_during_control": EnvironmentType.RANDOM,
    "num_averages": EnvironmentType.MODAL,
    "test_level_ramp_time": EnvironmentType.TRANSIENT,
    "cancel_rampdown_time": EnvironmentType.TIME,
}


def discover_environment_type_in_old_netcdf(
    environment_group: netCDF4.Group,
) -> EnvironmentType:
    """
    Infers the environment type from attributes in a legacy netCDF4 file.

    Version 3.0 netCDF4 files do not store an explicit environment type, so this
    function infers it from the presence of environment-specific attributes listed in
    ``_LEGACY_ENVIRONMENT_TYPE_ATTRIBUTES``. The first matching attribute determines the
    type.

    Parameters
    ----------
    environment_group : netCDF4.Group
        The netCDF4 group corresponding to a single environment.

    Returns
    -------
    EnvironmentType
        The inferred environment type.

    Raises
    ------
    RattlesnakeError
        If the environment type cannot be determined from the group's attributes.

    Notes
    -----
    Only used by ``load_metadata_from_netcdf``.

    Unit Tests
    ----------
    """
    for attribute, environment_type in _LEGACY_ENVIRONMENT_TYPE_ATTRIBUTES.items():
        if hasattr(environment_group, attribute):
            return environment_type
    raise RattlesnakeError("Invalid netcdf4 file")


def load_metadata_from_workbook(
    workbook: Workbook,
) -> Tuple[HardwareMetadata, List[EnvironmentMetadata], List[ProfileEvent]]:
    """
    Loads hardware, environment, and profile metadata from an Excel workbook.

    The workbook is read as follows:

    - **Hardware type** comes from the ``Hardware Type`` row of the ``"Hardware"``
      sheet.
    - **Environment names** are read from row 2 of the channel sheet, starting at column
      24 and stopping at the first empty cell. A non-empty cell in rows 3 onward of that
      column marks the channel as active in that environment.
    - **Environment metadata** is read from the sheet with each environment's name. Its
      type comes from cell ``B1`` of that sheet.
    - **Profile events** are read from the ``"Test Profile"`` sheet with
      ``load_profile_from_workbook``.

    If the workbook has more than one sheet, the first sheet with ``"channel"`` in its
    name is used as the channel sheet.

    The workbook is closed before returning.

    Parameters
    ----------
    workbook : openpyxl.workbook.workbook.Workbook
        An open Excel workbook containing a "Hardware" sheet, a channel sheet, one sheet
        per environment, and a "Test Profile" sheet.

    Returns
    -------
    hardware_metadata : HardwareMetadata
        The hardware metadata loaded from the workbook.
    environment_metadata_list : List[EnvironmentMetadata]
        A list of environment metadata objects, one per environment sheet.
    profile_event_list : List[ProfileEvent]
        The list of profile events loaded from the "Test Profile" sheet.

    Raises
    ------
    RattlesnakeError
        If a profile event has an invalid command.

    Notes
    -----
    Only used by ``Rattlesnake`` in ``engine.py``, the main UI, and
    ``testing/loading.py``.

    Unit Tests
    ----------
    """
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

    workbook.close()

    return (hardware_metadata, environment_metadata_list, profile_event_list)


def save_rattlesnake_to_workbook(
    workbook: Workbook,
    hardware_metadata: HardwareMetadata | None = None,
    environment_metadata_dict: (
        Dict[str, EnvironmentType | EnvironmentMetadata] | None
    ) = None,
    profile_event_list: List[ProfileEvent] | None = None,
) -> None:
    """
    Saves hardware, environment, and profile metadata to an Excel workbook.

    A blank hardware template is always written and filled in if ``hardware_metadata``
    is provided. Environments are added based off the key in the metadata dictionary.
    Mapping this key to an ``EnvironmentType`` will write a blank template and
    ``EnvironmentMetadata`` will fill out the template. The profile sheet is always
    created and filled in if a ``profile_event_list`` is provided.

    The workbook is not saved to disk or closed. The caller is responsible for that.

    Parameters
    ----------
    workbook : openpyxl.workbook.workbook.Workbook
        The workbook to save the metadata to. The active sheet is used as the channel
        sheet and additional sheets are created for each environment and for the test
        profile.
    hardware_metadata : HardwareMetadata, optional
        The hardware metadata to save.
    environment_metadata_dict : Dict[str, EnvironmentType or EnvironmentMetadata],
    optional
        A dictionary where the keys are environment names and the values are either an
        EnvironmentType (to create a blank template) or an environment metadata object
        to save.
    profile_event_list : List[ProfileEvent], optional
        The list of profile events to save to the "Test Profile" sheet.

    Unit Tests
    ----------
    """
    # Open workbook and save to blank template
    channel_worksheet = workbook.active
    HardwareMetadata.save_blank_hardware_to_workbook(workbook)
    # Save hardware metadata values
    if hardware_metadata:
        hardware_metadata.save_metadata_to_workbook(workbook)

    channel_worksheet.cell(row=1, column=24, value="Environments")
    # Save environment metadata values
    if environment_metadata_dict:
        for col, (environment_name, environment_metadata) in enumerate(
            environment_metadata_dict.items()
        ):
            col_idx = col + 24
            channel_worksheet.cell(row=2, column=col_idx, value=environment_name)
            environment_worksheet = workbook.create_sheet(environment_name)
            if isinstance(environment_metadata, EnvironmentType):
                ENVIRONMENT_METADATA[
                    environment_metadata
                ].create_blank_worksheet_template(environment_worksheet)
            else:
                for row in environment_metadata.channel_indices:
                    row_idx = row + 3
                    channel_worksheet.cell(row=row_idx, column=col_idx, value="x")
                environment_metadata.save_metadata_to_worksheet(environment_worksheet)

    # Save profile event list
    profile_sheet = workbook.create_sheet("Test Profile")
    save_profile_to_workbook(profile_sheet, profile_event_list)


def load_profile_from_workbook(
    workbook: Workbook,
    environment_types: Dict[str, EnvironmentType | str],
) -> List[ProfileEvent]:
    """
    Loads the test profile event list from an Excel workbook.

    Rows are read from the "Test Profile" sheet starting at row 2, until the
    first row with an empty timestamp. Each row holds the timestamp, the
    environment name, the command, and the data.

    Command names are converted to upper case with spaces replaced by
    underscores (e.g. ``"Start Environment"`` becomes ``START_ENVIRONMENT``),
    then looked up in ``GlobalCommands`` and then the environment's command
    enum. ``SET_ENVIRONMENT_INSTRUCTIONS`` rows are skipped, since those
    instructions cannot be set from a workbook. Empty data cells become
    ``None``.

    The workbook is closed before returning.

    Parameters
    ----------
    workbook : openpyxl.workbook.workbook.Workbook
        An open Excel workbook containing a "Test Profile" sheet.
    environment_types : Dict[str, EnvironmentType or str]
        A dictionary mapping environment names to their ``EnvironmentType``,
        used to resolve environment-specific commands. Should include
        ``{"Global": "Global"}`` for controller-wide commands.

    Returns
    -------
    List[ProfileEvent]
        The list of profile events loaded from the "Test Profile" sheet.

    Raises
    ------
    RattlesnakeError
        If a command is not a valid global or environment command.

    Unit Tests
    ----------
    """
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
        elif (
            command == "SET_ENVIRONMENT_INSTRUCTIONS"
        ):  # This should not be set by workbook.
            index += 1
            continue
        else:
            raise RattlesnakeError(
                f"Invalid command: {command} for {environment_name} | "
                f"{environment_type}"
            )

        data = profile_sheet.cell(index, 4).value
        data = None if isinstance(data, str) and not data.strip() else data

        event = ProfileEvent(timestamp, environment_name, command, data)
        profile_event_list.append(event)
        index += 1
    workbook.close()

    return profile_event_list


def save_profile_to_workbook(
    profile_sheet: Worksheet,
    profile_event_list: List[ProfileEvent] | None,
) -> None:
    """
    Saves a test profile event list to an Excel worksheet.

    A header row (``Time (s)``, ``Environment``, ``Operation``, ``Data``) is
    always written to row 1. Events are written from row 2 onward, with each
    command written as its ``label`` (e.g. ``"Start Environment"``) so that
    ``load_profile_from_workbook`` can read it back.

    Parameters
    ----------
    profile_sheet : openpyxl.worksheet.worksheet.Worksheet
        The worksheet to write the profile event list to.
    profile_event_list : List[ProfileEvent] or None
        The list of profile events to save. Events with the
        SET_ENVIRONMENT_INSTRUCTIONS command are skipped. If ``None`` or
        empty, only the header row is written.

    Unit Tests
    ----------
    """
    profile_sheet.cell(1, 1, "Time (s)")
    profile_sheet.cell(1, 2, "Environment")
    profile_sheet.cell(1, 3, "Operation")
    profile_sheet.cell(1, 4, "Data")
    # Fill out values
    if profile_event_list:
        row_idx = 2
        for event in profile_event_list:
            if event.command == UICommands.SET_ENVIRONMENT_INSTRUCTIONS:
                continue
            profile_sheet.cell(row_idx, 1, str(event.timestamp))
            profile_sheet.cell(row_idx, 2, event.environment_name)
            profile_sheet.cell(row_idx, 3, event.command.label)
            profile_sheet.cell(row_idx, 4, str(event.data))
            row_idx = row_idx + 1
