import numpy as np
import pytest

from mock_objects.mock_hardware import (
    MockHardwareAcquisition,
    MockHardwareMetadata,
    MockHardwareOutput,
)
from mock_objects.mock_utilities import mock_channel_list
from rattlesnake.utilities import RattlesnakeError
from rattlesnake.hardware.abstract_hardware import (
    HardwareAcquisition,
    HardwareMetadata,
    HardwareOutput,
)

channel_list = mock_channel_list()


@pytest.fixture
def hardware_metadata():
    return MockHardwareMetadata()


# region: Hardware Metadata
def test_hardware_metadata_init():
    hardware_metadata = MockHardwareMetadata()

    assert isinstance(hardware_metadata, HardwareMetadata)
    assert hasattr(hardware_metadata, "hardware_type")
    assert hasattr(hardware_metadata, "channel_list")
    assert hasattr(hardware_metadata, "sample_rate")
    assert hasattr(hardware_metadata, "time_per_read")
    assert hasattr(hardware_metadata, "time_per_write")
    assert hasattr(hardware_metadata, "output_oversample")


def test_hardware_metadata_properties(hardware_metadata):
    hardware_metadata.sample_rate = 1000
    hardware_metadata.time_per_read = 0.25
    hardware_metadata.time_per_write = 0.3
    hardware_metadata.output_oversample = 10

    assert hardware_metadata.samples_per_read == 250
    assert hardware_metadata.samples_per_write == 3000
    assert hardware_metadata.nyquist_frequency == 500
    assert hardware_metadata.output_sample_rate == 10000


@pytest.mark.parametrize(
    "channel_list, expected",
    [
        (channel_list, True),
        (channel_list + [channel_list[0]], RattlesnakeError),
    ],
)
def test_hardware_metadata_validate(channel_list, expected, hardware_metadata):
    hardware_metadata.channel_list = channel_list

    if expected is RattlesnakeError:
        with pytest.raises(RattlesnakeError):
            hardware_metadata.validate()
    else:
        hardware_metadata.validate()
        assert True


# region: Hardware Acquisition
def test_hardware_acquisition_init():
    hardware_acquistion = MockHardwareAcquisition()

    assert isinstance(hardware_acquistion, HardwareAcquisition)


# Just doing all the functions since the abstract class does nothing
def test_hardware_acquisition_functions(hardware_metadata):
    hardware_acquisition = MockHardwareAcquisition()

    hardware_acquisition.initialize_hardware(hardware_metadata)
    hardware_acquisition.start()
    array_1 = hardware_acquisition.read()
    array_2 = hardware_acquisition.read_remaining()
    hardware_acquisition.stop()
    hardware_acquisition.close()
    int_1 = hardware_acquisition.get_acquisition_delay()

    assert isinstance(array_1, np.ndarray)
    assert isinstance(array_2, np.ndarray)
    assert isinstance(int_1, int)


# region: Hardware Output
def test_hardware_output_init():
    hardware_output = MockHardwareOutput()

    assert isinstance(hardware_output, HardwareOutput)


def test_hardware_output_functions(hardware_metadata):
    hardware_output = MockHardwareOutput()

    hardware_output.initialize_hardware(hardware_metadata)
    hardware_output.start()
    hardware_output.write(np.zeros((1, 100)))
    hardware_output.stop()
    hardware_output.close()
    hardware_output.ready_for_new_output()

    assert True
