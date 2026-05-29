"""
Tests for Abstract Hardware

This module contains tests for the HardwareAcquisition and HardwareOutput classes,
using dummy implementations to verify their basic behavior and initialization.
"""

from rattlesnake.hardware.abstract_hardware import HardwareAcquisition, HardwareOutput


class DummyHardwareAcquisition(HardwareAcquisition):
    """
    Dummy implementation of HardwareAcquisition for testing.
    """

    def __init__(self):
        """
        Initialize the DummyHardwareAcquisition.
        """
        super().__init__()

    def set_up_data_acquisition_parameters_and_channels(self, test_data, channel_data):
        """
        Set up data acquisition parameters and channels.

        Args:
            test_data: Data associated with the test.
            channel_data: Data associated with the channels.

        Returns:
            The result of the superclass method.
        """
        return super().set_up_data_acquisition_parameters_and_channels(test_data, channel_data)

    def start(self):
        """
        Start hardware acquisition.

        Returns:
            The result of the superclass start method.
        """
        return super().start()

    def read(self):
        """
        Read data from the hardware.

        Returns:
            The result of the superclass read method.
        """
        return super().read()

    def read_remaining(self):
        """
        Read remaining data from the hardware.

        Returns:
            The result of the superclass read_remaining method.
        """
        return super().read_remaining()

    def stop(self):
        """
        Stop hardware acquisition.

        Returns:
            The result of the superclass stop method.
        """
        return super().stop()

    def close(self):
        """
        Close the hardware connection.

        Returns:
            The result of the superclass close method.
        """
        return super().close()

    def get_acquisition_delay(self):
        """
        Get the acquisition delay.

        Returns:
            The result of the superclass get_acquisition_delay method.
        """
        return super().get_acquisition_delay()


class DummyHardwareOutput(HardwareOutput):
    """
    Dummy implementation of HardwareOutput for testing.
    """

    def __init__(self):
        """
        Initialize the DummyHardwareOutput.
        """
        super().__init__()

    def set_up_data_output_parameters_and_channels(self, test_data, channel_data):
        """
        Set up data output parameters and channels.

        Args:
            test_data: Data associated with the test.
            channel_data: Data associated with the channels.

        Returns:
            The result of the superclass method.
        """
        return super().set_up_data_output_parameters_and_channels(test_data, channel_data)

    def start(self):
        """
        Start hardware output.

        Returns:
            The result of the superclass start method.
        """
        return super().start()

    def write(self, data):
        """
        Write data to the hardware.

        Args:
            data: The data to write.

        Returns:
            The result of the superclass write method.
        """
        return super().write(data)

    def stop(self):
        """
        Stop hardware output.

        Returns:
            The result of the superclass stop method.
        """
        return super().stop()

    def close(self):
        """
        Close the hardware connection.

        Returns:
            The result of the superclass close method.
        """
        return super().close()

    def ready_for_new_output(self):
        """
        Check if the hardware is ready for new output.

        Returns:
            The result of the superclass ready_for_new_output method.
        """
        return super().ready_for_new_output()


def test_hardware_acquisition_init():
    """
    Test the initialization of HardwareAcquisition via DummyHardwareAcquisition.
    """
    hardware_acquistion = DummyHardwareAcquisition()

    assert isinstance(hardware_acquistion, DummyHardwareAcquisition)


def test_hardware_output_init():
    """
    Test the initialization of HardwareOutput via DummyHardwareOutput.
    """
    hardware_output = DummyHardwareOutput()

    assert isinstance(hardware_output, HardwareOutput)
