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
    SKELETON = 8


class Channel:
    """Property container for a single channel in the controller."""

    def __init__(
        self,
        node_number=None,
        node_direction=None,
        comment=None,
        serial_number=None,
        triax_dof=None,
        sensitivity=None,
        unit=None,
        make=None,
        model=None,
        expiration=None,
        physical_device=None,
        physical_channel=None,
        channel_type=None,
        minimum_value=None,
        maximum_value=None,
        coupling=None,
        excitation_source=None,
        excitation=None,
        feedback_device=None,
        feedback_channel=None,
        warning_level=None,
        abort_level=None,
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

    @property
    def channel_attr_list(self):
        """Returns a list of channel attributes in the orderr of the channel table headers"""
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
    def is_empty(self):
        """Returns True if every attribute in channel is equalt to None"""
        return all(
            getattr(self, attr_name) is None for attr_name in self.channel_attr_list
        )

    def is_output_channel(self):
        return self.feedback_device is not None

    def __eq__(self, other):
        if not isinstance(other, Channel):
            return NotImplemented

        return all(
            getattr(self, attr_name) == getattr(other, attr_name)
            for attr_name in self.channel_attr_list
        )

    def __hash__(self):
        return hash(
            tuple(getattr(self, attr_name) for attr_name in self.channel_attr_list)
        )
