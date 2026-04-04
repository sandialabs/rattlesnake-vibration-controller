from enum import Enum


class HardwareType(Enum):
    NONE = -1
    NI_DAQMX = 0
    LAN_XI = 1
    DP_QUATTRO = 2
    DP_900 = 3
    EXODUS = 4
    STATE_SPACE = 5
    SDYNPY_SYSTEM = 6
    SDYNPY_FRF = 7


class Channel:
    """Property container for a single channel in the controller."""

    def __init__(
        self,
        node_number,
        node_direction,
        comment,
        serial_number,
        triax_dof,
        sensitivity,
        unit,
        make,
        model,
        expiration,
        physical_device,
        physical_channel,
        channel_type,
        minimum_value,
        maximum_value,
        coupling,
        excitation_source,
        excitation,
        feedback_device,
        feedback_channel,
        warning_level,
        abort_level,
    ):
        """Property container for a single channel in the controller.

        Parameters
        ----------
        node_number : str :
            Metadata specifying the node number
        node_direction : str :
            Metadata specifying the direction at a node
        comment : str :
            Metadata specifying any additional comments on the channel
        serial_number : str :
            Metadata specifying the serial number of the instrument
        triax_dof : str :
            Metadata specifying the degree of freedom on a triaxial sensor
        sensitivity : str :
            Sensitivity value of the sensor in mV/engineering unit
        unit : str :
            The engineering unit of the sensor
        make : str :
            Metadata specifying the make of the sensor
        model : str :
            Metadata specifying the model of the sensor
        expiration : str :
            Metadata specifying the expiration date of the sensor
        physical_device : str :
            Physical hardware that the instrument is connected to
        physical_channel : str :
            Channel in the physical hardware that the instrument is connected to
        channel_type : str :
            Type of channel
        minimum_value : str :
            Minimum value of the channel in volts
        maximum_value : str :
            Maximum value of the channel in volts
        coupling : str :
            Coupling type for the channel
        excitation_source : str :
            Source for the signal conditioning for the sensor
        excitation : str :
            Level of excitation for the signal conditioning for the sensor
        feedback_device : str :
            Physical hardware that the source output teed into this channel
            originates from
        feedback_channel : str :
            Channel on the physical hardware that is teed into this channel
        warning_level : str :
            Level at which warnings will be flagged on the monitor
        abort_level : str :
            Level at which the system will shut down
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

    @classmethod
    def from_channel_table_row(cls, row: tuple[str]):
        """Creates a Channel object from a row in the channel table

        Parameters
        ----------
        row : iterable :
            Iterable of strings from a single row of the channel table


        Returns
        -------
        channel : Channel
            A channel object containing the data in the given row of the
            channel table.

        """
        new_row = [None if val.strip() == "" else val for val in row]
        physical_device = new_row[10]
        if physical_device is None:
            return None
        node_number = new_row[0]
        node_direction = new_row[1]
        comment = new_row[2]
        serial_number = new_row[3]
        triax_dof = new_row[4]
        sensitivity = new_row[5]
        unit = new_row[6]
        make = new_row[7]
        model = new_row[8]
        expiration = new_row[9]
        physical_channel = new_row[11]
        channel_type = new_row[12]
        minimum_value = new_row[13]
        maximum_value = new_row[14]
        coupling = new_row[15]
        excitation_source = new_row[16]
        excitation = new_row[17]
        feedback_device = new_row[18]
        feedback_channel = new_row[19]
        warning_level = new_row[20]
        abort_level = new_row[21]
        return cls(
            node_number,
            node_direction,
            comment,
            serial_number,
            triax_dof,
            sensitivity,
            unit,
            make,
            model,
            expiration,
            physical_device,
            physical_channel,
            channel_type,
            minimum_value,
            maximum_value,
            coupling,
            excitation_source,
            excitation,
            feedback_device,
            feedback_channel,
            warning_level,
            abort_level,
        )
