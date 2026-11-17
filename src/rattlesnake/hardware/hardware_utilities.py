"""
Common data objects used by all hardware devices.

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

from enum import Enum
from typing import List


class HardwareType(Enum):
    """
    Enumeration of the hardware devices Rattlesnake can control.

    The integer values are stored in netCDF4 files and Excel workbooks to identify the
    hardware, so existing values must not be changed. Each member maps to a set of
    metadata, acquisition, and output classes in the hardware registry.

    Unit Tests
    ----------
    """

    NONE = -1  # No hardware selected
    NI_DAQMX = 0  # National Instruments devices using the NI-DAQmx driver
    LAN_XI = 1  # HBK LAN-XI data acquisition modules
    DP_QUATTRO = 2  # Data Physics Quattro devices
    DP_900 = 3  # Data Physics 900 series devices
    EXODUS = 4  # Virtual hardware driven by an Exodus modal solution
    STATE_SPACE = 5  # Virtual hardware driven by a state space model
    SDYNPY_SYSTEM = 6  # Virtual hardware driven by a SDynPy system
    SDYNPY_FRF = 7  # Virtual hardware driven by SDynPy frequency response functions
    SKELETON = 8  # Template for new hardware implementations


class Channel:
    """
    Property container for a single channel in the controller.

    Each attribute corresponds to a column of the channel table. Every attribute
    defaults to ``None``, and a channel with every attribute equal to ``None`` is
    considered empty.

    Unit Tests
    ----------
    """

    def __init__(
        self,
        node_number: str | None = None,
        node_direction: str | None = None,
        comment: str | None = None,
        serial_number: str | None = None,
        triax_dof: str | None = None,
        sensitivity: str | None = None,
        unit: str | None = None,
        make: str | None = None,
        model: str | None = None,
        expiration: str | None = None,
        physical_device: str | None = None,
        physical_channel: str | None = None,
        channel_type: str | None = None,
        minimum_value: str | None = None,
        maximum_value: str | None = None,
        coupling: str | None = None,
        excitation_source: str | None = None,
        excitation: str | None = None,
        feedback_device: str | None = None,
        feedback_channel: str | None = None,
        warning_level: str | None = None,
        abort_level: str | None = None,
    ) -> None:
        """
        Initializes a channel from its channel table values.

        Parameters
        ----------
        node_number : str, optional
            Metadata specifying the node number.
        node_direction : str, optional
            Metadata specifying the direction at a node.
        comment : str, optional
            Metadata specifying any additional comments on the channel.
        serial_number : str, optional
            Metadata specifying the serial number of the instrument.
        triax_dof : str, optional
            Metadata specifying the degree of freedom on a triaxial sensor.
        sensitivity : str, optional
            Sensitivity value of the sensor in mV/engineering unit.
        unit : str, optional
            The engineering unit of the sensor.
        make : str, optional
            Metadata specifying the make of the sensor.
        model : str, optional
            Metadata specifying the model of the sensor.
        expiration : str, optional
            Metadata specifying the expiration date of the sensor.
        physical_device : str, optional
            Physical hardware that the instrument is connected to.
        physical_channel : str, optional
            Channel in the physical hardware that the instrument is connected to.
        channel_type : str, optional
            Type of channel.
        minimum_value : str, optional
            Minimum value of the channel in volts.
        maximum_value : str, optional
            Maximum value of the channel in volts.
        coupling : str, optional
            Coupling type for the channel.
        excitation_source : str, optional
            Source for the signal conditioning for the sensor.
        excitation : str, optional
            Level of excitation for the signal conditioning for the sensor.
        feedback_device : str, optional
            Physical hardware that the source output teed into this channel originates
            from.
        feedback_channel : str, optional
            Channel on the physical hardware that is teed into this channel.
        warning_level : str, optional
            Level at which warnings will be flagged on the monitor.
        abort_level : str, optional
            Level at which the system will shut down.

        Unit Tests
        ----------
        """
        self.node_number = node_number
        self.node_direction = node_direction
        self.comment = comment
        self.serial_number = serial_number
        self.triax_dof = triax_dof
        self.sensitivity = sensitivity
        self.make = make
        self.model = model
        self.expiration = expiration
        self.physical_device = physical_device
        self.physical_channel = physical_channel
        self.channel_type = channel_type
        self.unit = unit
        self.minimum_value = minimum_value
        self.maximum_value = maximum_value
        self.coupling = coupling
        self.excitation_source = excitation_source
        self.excitation = excitation
        self.feedback_device = feedback_device
        self.feedback_channel = feedback_channel
        self.warning_level = warning_level
        self.abort_level = abort_level

    @property
    def channel_attr_list(self) -> List[str]:
        """
        Return the channel attribute names in the order of the channel table headers.

        This order is used to read and write channel tables in netCDF4 files and Excel
        workbooks, so it must match the column order of the channel table.

        Returns
        -------
        List[str]
            Attribute names of the channel, ordered by channel table column.

        Unit Tests
        ----------
        """
        channel_attr_list = [
            "node_number",
            "node_direction",
            "comment",
            "serial_number",
            "triax_dof",
            "sensitivity",
            "unit",
            "make",
            "model",
            "expiration",
            "physical_device",
            "physical_channel",
            "channel_type",
            "minimum_value",
            "maximum_value",
            "coupling",
            "excitation_source",
            "excitation",
            "feedback_device",
            "feedback_channel",
            "warning_level",
            "abort_level",
        ]
        return channel_attr_list

    @property
    def is_empty(self) -> bool:
        """
        Return whether every attribute of the channel is ``None``.

        Used to detect the end of a channel table, where rows no longer contain
        channel values.

        Returns
        -------
        bool
            ``True`` if every attribute in ``channel_attr_list`` is ``None``,
            otherwise ``False``.

        Unit Tests
        ----------
        """
        return all(
            getattr(self, attr_name) is None for attr_name in self.channel_attr_list
        )

    def is_output_channel(self) -> bool:
        """
        Return whether the channel measures a source output.

        A channel is an output channel when a source output is teed into it, which is
        indicated by a ``feedback_device`` being assigned.

        Returns
        -------
        bool
            ``True`` if ``feedback_device`` is not ``None``, otherwise ``False``.

        Unit Tests
        ----------
        """
        return self.feedback_device is not None

    def __eq__(self, other: object) -> bool:
        """
        Compare two channels by the values of their channel table attributes.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` if ``other`` is a ``Channel`` and every attribute in
            ``channel_attr_list`` is equal, otherwise ``False``. Returns
            ``NotImplemented`` if ``other`` is not a ``Channel``.

        Unit Tests
        ----------
        """
        if not isinstance(other, Channel):
            return NotImplemented

        return all(
            getattr(self, attr_name) == getattr(other, attr_name)
            for attr_name in self.channel_attr_list
        )

    def __hash__(self) -> int:
        """
        Hash the channel by the values of its channel table attributes.

        Channels that compare equal produce the same hash.

        Returns
        -------
        int
            Hash of the tuple of attribute values in ``channel_attr_list`` order.

        Unit Tests
        ----------
        """
        return hash(
            tuple(getattr(self, attr_name) for attr_name in self.channel_attr_list)
        )
