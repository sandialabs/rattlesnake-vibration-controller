from rattlesnake.hardware.hardware_utilities import Channel
from rattlesnake.testing.mock_utilities import mock_channel_list
import pytest


# region: Channel
# Test Channel initialization
def test_channel_init():
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

    # Test if object is a Channel
    assert isinstance(channel, Channel)


def test_channel_attr_list():
    channel = Channel()
    attr_list = channel.channel_attr_list

    for attr in attr_list:
        assert hasattr(channel, attr)

    for attr in vars(channel).keys():
        assert attr in attr_list


@pytest.mark.parametrize("node_number, expected", [(None, True), ("node", False)])
def test_channel_is_empty(node_number, expected):
    channel = Channel()
    channel.node_number = node_number

    assert channel.is_empty == expected


@pytest.mark.parametrize(
    "node_number_1, node_number_2, expected",
    [
        (None, None, True),
        ("node_number", None, False),
        ("node_number", "node_number", True),
    ],
)
def test_channel_eq(node_number_1, node_number_2, expected):
    channel_1 = Channel()
    channel_1.node_number = node_number_1
    channel_2 = Channel()
    channel_2.node_number = node_number_2

    assert (channel_1 == channel_2) == expected


def test_channel_eq_foreign_type():
    channel = Channel()
    assert (channel == 0) is False
