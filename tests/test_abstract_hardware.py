from rattlesnake.hardware.abstract_hardware import HardwareAcquisition, HardwareOutput
from unittest import mock
import numpy as np
import pytest


class DummyHardwareAcquisition(HardwareAcquisition):
    def __init__(self):
        super().__init__()

    def set_up_data_acquisition_parameters_and_channels(self, test_data, channel_data):
        return super().set_up_data_acquisition_parameters_and_channels(test_data, channel_data)

    def start(self):
        return super().start()

    def read(self):
        return super().read()

    def read_remaining(self):
        return super().read_remaining()

    def stop(self):
        return super().stop()

    def close(self):
        return super().close()

    def get_acquisition_delay(self):
        return super().get_acquisition_delay()


class DummyHardwareOutput(HardwareOutput):
    def __init__(self):
        super().__init__()

    def set_up_data_output_parameters_and_channels(self, test_data, channel_data):
        return super().set_up_data_output_parameters_and_channels(test_data, channel_data)

    def start(self):
        return super().start()

    def write(self, data):
        return super().write(data)

    def stop(self):
        return super().stop()

    def close(self):
        return super().close()

    def ready_for_new_output(self):
        return super().ready_for_new_output()


def test_hardware_acquisition_init():
    hardware_acquistion = DummyHardwareAcquisition()

    assert isinstance(hardware_acquistion, DummyHardwareAcquisition)


def test_hardware_output_init():
    hardware_output = DummyHardwareOutput()

    assert isinstance(hardware_output, HardwareOutput)
