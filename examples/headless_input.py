from example_registry import HARDWARE_DICT

from rattlesnake.engine import RattlesnakeController
from rattlesnake.main import launch_rattlesnake_ui
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.environment_utilities import EnvironmentType

THREADED = True
IMPORT_METHOD = "worksheet" # worksheet, netcdf, manual
HARDWARE_TYPE = HardwareType.SDYNPY_SYSTEM
ENVIRONMENT_TYPE = EnvironmentType.TIME


def build_rattlesnake_object():
    rattlesnake = RattlesnakeController(threaded=THREADED, timeout=10)

    # Initialize hardware metadata
    hardware_metadata_function = HARDWARE_DICT[HARDWARE_TYPE][IMPORT_METHOD]
    hardware_metadata = hardware_metadata_function()
    rattlesnake.initialize_hardware(hardware_metadata)

    return rattlesnake


if __name__ == "__main__":
    rattlesnake = build_rattlesnake_object()

    launch_rattlesnake_ui(rattlesnake)