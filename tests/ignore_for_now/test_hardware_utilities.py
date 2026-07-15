import pytest

from rattlesnake.hardware.hardware_utilities import Channel, HardwareType


# region HardwareType
def test_hardware_type_unique_integer_values():
    """
    Verifies that hardware type enum values are unique integers.
    """
    values = [hardware_type.value for hardware_type in HardwareType]

    assert all(isinstance(value, int) for value in values)
    assert len(values) == len(set(values))


@pytest.mark.parametrize(
    "hardware_type, expected_value",
    [
        (HardwareType.NONE, -1),
        (HardwareType.NI_DAQMX, 0),
        (HardwareType.LAN_XI, 1),
        (HardwareType.DP_QUATTRO, 2),
        (HardwareType.DP_900, 3),
        (HardwareType.EXODUS, 4),
        (HardwareType.STATE_SPACE, 5),
        (HardwareType.SDYNPY_SYSTEM, 6),
        (HardwareType.SDYNPY_FRF, 7),
        (HardwareType.SKELETON, 8),
    ],
)
def test_hardware_type_expected_values(hardware_type, expected_value):
    """
    Verifies that hardware type enum members have expected values.
    """
    assert hardware_type.value == expected_value
    assert HardwareType(expected_value) is hardware_type


# endregion


# region Channel
def test_channel_init():
    """
    Verifies that a ``Channel`` object can be initialized with all channel
    fields.
    """
    channel = Channel(
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
    )

    assert isinstance(channel, Channel)

    assert channel.node_number == "node_number"
    assert channel.node_direction == "node_direction"
    assert channel.comment == "comment"
    assert channel.serial_number == "serial_number"
    assert channel.triax_dof == "triax_dof"
    assert channel.sensitivity == "sensitivity"
    assert channel.unit == "unit"
    assert channel.make == "make"
    assert channel.model == "model"
    assert channel.expiration == "expiration"
    assert channel.physical_device == "physical_device"
    assert channel.physical_channel == "physical_channel"
    assert channel.channel_type == "channel_type"
    assert channel.minimum_value == "minimum_value"
    assert channel.maximum_value == "maximum_value"
    assert channel.coupling == "coupling"
    assert channel.excitation_source == "excitation_source"
    assert channel.excitation == "excitation"
    assert channel.feedback_device == "feedback_device"
    assert channel.feedback_channel == "feedback_channel"
    assert channel.warning_level == "warning_level"
    assert channel.abort_level == "abort_level"


def test_channel_init_defaults():
    """
    Verifies that channel fields default to ``None``.
    """
    channel = Channel()

    for attr in channel.channel_attr_list:
        assert getattr(channel, attr) is None


def test_channel_attr_list():
    """
    Verifies that ``channel_attr_list`` contains all instance attributes.
    """
    channel = Channel()
    attr_list = channel.channel_attr_list

    assert attr_list == [
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

    for attr in attr_list:
        assert hasattr(channel, attr)

    for attr in vars(channel):
        assert attr in attr_list


@pytest.mark.parametrize(
    "attribute_name",
    [
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
    ],
)
def test_channel_is_empty_false_for_any_set_attribute(attribute_name):
    """
    Verifies that setting any listed channel attribute makes the channel
    non-empty.
    """
    channel = Channel()

    assert channel.is_empty is True

    setattr(channel, attribute_name, "value")

    assert channel.is_empty is False


def test_channel_is_empty_true():
    """
    Verifies that a default channel is empty.
    """
    channel = Channel()

    assert channel.is_empty is True


@pytest.mark.parametrize(
    "feedback_device, expected",
    [
        (None, False),
        ("", True),
        ("Device", True),
    ],
)
def test_channel_is_output_channel(feedback_device, expected):
    """
    Verifies that output-channel detection is based on ``feedback_device`` not
    being ``None``.
    """
    channel = Channel(feedback_device=feedback_device)

    assert channel.is_output_channel() is expected


@pytest.mark.parametrize(
    "node_number_1, node_number_2, expected",
    [
        (None, None, True),
        ("node_number", None, False),
        ("node_number", "node_number", True),
    ],
)
def test_channel_eq(node_number_1, node_number_2, expected):
    """
    Verifies equality and inequality for channel objects.
    """
    channel_1 = Channel(node_number=node_number_1)
    channel_2 = Channel(node_number=node_number_2)

    assert (channel_1 == channel_2) is expected


def test_channel_eq_all_attributes():
    """
    Verifies equality when all attributes match and inequality when one differs.
    """
    kwargs = {
        "node_number": "node_number",
        "node_direction": "node_direction",
        "comment": "comment",
        "serial_number": "serial_number",
        "triax_dof": "triax_dof",
        "sensitivity": "sensitivity",
        "unit": "unit",
        "make": "make",
        "model": "model",
        "expiration": "expiration",
        "physical_device": "physical_device",
        "physical_channel": "physical_channel",
        "channel_type": "channel_type",
        "minimum_value": "minimum_value",
        "maximum_value": "maximum_value",
        "coupling": "coupling",
        "excitation_source": "excitation_source",
        "excitation": "excitation",
        "feedback_device": "feedback_device",
        "feedback_channel": "feedback_channel",
        "warning_level": "warning_level",
        "abort_level": "abort_level",
    }

    channel_1 = Channel(**kwargs)
    channel_2 = Channel(**kwargs)

    assert channel_1 == channel_2

    channel_2.abort_level = "different_abort_level"

    assert channel_1 != channel_2


def test_channel_eq_foreign_type():
    """
    Verifies comparison behavior with a non-channel object.
    """
    channel = Channel()

    assert (channel == 0) is False


def test_channel_hash_equal_channels():
    """
    Verifies that equal channels produce equal hashes.
    """
    channel_1 = Channel(node_number="node")
    channel_2 = Channel(node_number="node")

    assert channel_1 == channel_2
    assert hash(channel_1) == hash(channel_2)


def test_channel_hash_allows_set_duplicate_detection():
    """
    Verifies that channels can be used in sets for duplicate detection.
    """
    channel_1 = Channel(node_number="node")
    channel_2 = Channel(node_number="node")
    channel_3 = Channel(node_number="other")

    channel_set = {channel_1, channel_2, channel_3}

    assert len(channel_set) == 2


def test_channel_hash_changes_with_attributes():
    """
    Verifies that changing channel attributes changes the hash.
    """
    channel_1 = Channel(node_number="node")
    channel_2 = Channel(node_number="other")

    assert hash(channel_1) != hash(channel_2)


# endregion
