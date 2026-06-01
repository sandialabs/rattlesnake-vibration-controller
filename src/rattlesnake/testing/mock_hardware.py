from enum import Enum
from unittest import mock

import numpy as np

from rattlesnake.testing.mock_utilities import mock_channel_list
from rattlesnake.hardware.abstract_hardware import (
    HardwareAcquisition,
    HardwareMetadata,
    HardwareOutput,
)
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.hardware.hardware_registry import (
    UNIMPLEMENTED_HARDWARE,
    HARDWARE_METADATA,
    HARDWARE_ACQUISITION,
    HARDWARE_OUTPUT,
)

# IMPLEMENTED_HARDWARE = [
#     hardware for hardware in HardwareType if hardware not in UNIMPLEMENTED_HARDWARE
# ]
IMPLEMENTED_HARDWARE = [HardwareType.NONE]


def hardware_metadata_dict():
    hardware_metadata_dict = HARDWARE_METADATA
    hardware_metadata_dict[HardwareType.NONE] = MockHardwareMetadata
    return hardware_metadata_dict


def hardware_acqusition_dict():
    hardware_acqusition_dict = HARDWARE_ACQUISITION
    hardware_acqusition_dict[HardwareType.NONE] = MockHardwareAcquisition
    return hardware_acqusition_dict


def hardware_output_dict():
    hardware_output_dict = HARDWARE_OUTPUT
    hardware_output_dict[HardwareType.NONE] = MockHardwareOutput
    return hardware_output_dict


# region Import
def acquisition_dict():
    acquisition_dict = {
        HardwareType.NONE: "rattlesnake.testing.mock_hardware.MockHardwareAcqusition",
        HardwareType.NI_DAQMX: "rattlesnake.hardware.nidaqmx_hardware_multitask.NIDAQmxAcquisition",
        HardwareType.LAN_XI: "rattlesnake.hardware.lanxi_hardware_multiprocessing.LanXIAcquisition",
        HardwareType.DP_QUATTRO: "rattlesnake.hardware.data_physics_hardware.DataPhysicsAcquisition",
        HardwareType.DP_900: "rattlesnake.hardware.data_physics_dp900_hardware.DataPhysicsDP900Acquisition",
        HardwareType.EXODUS: "rattlesnake.hardwaare.exodus_modal_solution_hardware.ExodusAcquisition",
        HardwareType.STATE_SPACE: "rattlesnake.hardware.state_space_virtual_hardware.StateSpaceAcquisition",
        HardwareType.SDYNPY_SYSTEM: "rattlesnake.hardware.sdynpy_system_virtual_hardware.SDynPySystemAcquisition",
        HardwareType.SDYNPY_FRF: "rattlesnake.hardware.sdynpy_frf_virtual_hardware.SDynPyFRFAcquisition",
    }
    return acquisition_dict


def output_dict():
    output_dict = {
        HardwareType.NI_DAQMX: "rattlesnake.hardware.nidaqmx_hardware_multitask.NIDAQmxOutput",
        HardwareType.LAN_XI: "rattlesnake.hardware.lanxi_hardware_multiprocessing.LanXIOutput",
        HardwareType.DP_QUATTRO: "rattlesnake.hardware.data_physics_hardware.DataPhysicsOutput",
        HardwareType.DP_900: "rattlesnake.hardware.data_physics_dp900_hardware.DataPhysicsDP900Output",
        HardwareType.EXODUS: "rattlesnake.hardwaare.exodus_modal_solution_hardware.ExodusOutput",
        HardwareType.STATE_SPACE: "rattlesnake.hardware.state_space_virtual_hardware.StateSpaceOutput",
        HardwareType.SDYNPY_SYSTEM: "rattlesnake.hardware.sdynpy_system_virtual_hardware.SDynPySystemOutput",
        HardwareType.SDYNPY_FRF: "rattlesnake.hardware.sdynpy_frf_virtual_hardware.SDynPyFRFOutput",
    }
    return output_dict


# region Metadata
class MockHardwareMetadata(HardwareMetadata):
    def __init__(self):
        super().__init__(HardwareType.NONE, [], 1000, 0.5, 0.5)
        self.channel_list = mock_channel_list()
        self.sample_rate = 1000
        self.time_per_read = 0.25
        self.time_per_write = 0.25
        self.output_oversample = 1
        self.extra_attr = "attr"

    @property
    def extra_attr_list(self):
        super().extra_attr_list
        return ["extra_attr"]

    def validate(self):
        super().validate()
        return True

    def valid_channel_dict(self, channel):
        return super().valid_channel_dict(channel)

    @property
    def assist_mode_modules(self):
        return super().assist_mode_modules


# region Acquisition
class MockHardwareAcquisition(HardwareAcquisition):
    def __init__(self):
        super().__init__()

    def initialize_hardware(self, metadata):
        super().initialize_hardware(metadata)
        return None

    def start(self):
        super().start()
        return None

    def read(self):
        super().read()
        return np.zeros((2, 100))

    def read_remaining(self):
        super().read_remaining()
        return np.zeros((2, 100))

    def stop(self):
        super().stop()
        return None

    def close(self):
        super().close()
        return None

    def get_acquisition_delay(self):
        super().get_acquisition_delay()
        return 0


# region Output
class MockHardwareOutput(HardwareOutput):
    def __init__(self):
        super().__init__()

    def initialize_hardware(self, metadata):
        super().initialize_hardware(metadata)
        return None

    def start(self):
        super().start()
        return None

    def write(self, data):
        super().write(data)
        return None

    def stop(self):
        super().stop()
        return None

    def close(self):
        super().close()
        return None

    def ready_for_new_output(self):
        super().ready_for_new_output()
        return True
