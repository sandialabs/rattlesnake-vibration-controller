from example_registry import HARDWARE_DICT, ENVIRONMENT_DICT, STREAM_DICT, INSTRUCTIONS_DICT, EVENT_DICT

from rattlesnake.engine import RattlesnakeController
from rattlesnake.main import launch_rattlesnake_ui
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.process.streaming import StreamType

"""USER INPUTS"""
THREADED = False
IMPORT_METHOD = "manual" # worksheet, netcdf, manual
HARDWARE_TYPE = HardwareType.NONE
ENVIRONMENT_TYPE = EnvironmentType.TIME
STREAM_TYPE = StreamType.NO_STREAM # None means dont start streaming
START_ENVIRONMENT = False
RUN_PROFILE = False


def build_rattlesnake_object():
    rattlesnake = RattlesnakeController(threaded=THREADED, timeout=10)

    # Initialize hardware
    if HARDWARE_TYPE is HardwareType.NONE:
        return rattlesnake
    hardware_metadata = HARDWARE_DICT[HARDWARE_TYPE][IMPORT_METHOD]()
    rattlesnake.initialize_hardware(hardware_metadata)

    # Initialize environment
    if ENVIRONMENT_TYPE is EnvironmentType.NONE:
        return rattlesnake
    environment_metadata = ENVIRONMENT_DICT[ENVIRONMENT_TYPE][IMPORT_METHOD]()
    rattlesnake.initialize_environments([environment_metadata])

    # Start Acquisition
    if STREAM_TYPE is None:
        return rattlesnake
    if STREAM_TYPE == StreamType.TEST_LEVEL:
        stream_metadata = STREAM_DICT[STREAM_TYPE](environment_metadata.environment_name)
    else:
        stream_metadata = STREAM_DICT[STREAM_TYPE]()
    rattlesnake.start_acquisition(stream_metadata)

    # Start Environment
    if not START_ENVIRONMENT:
        return rattlesnake
    if not RUN_PROFILE:
        instructions = INSTRUCTIONS_DICT[ENVIRONMENT_TYPE]()
        rattlesnake.start_environment(instructions=instructions)
    else:
        event_list = EVENT_DICT[ENVIRONMENT_TYPE]()
        rattlesnake.start_profile(event_list)

    return rattlesnake


if __name__ == "__main__":
    print("Loading Rattlesnake...")

    rattlesnake = build_rattlesnake_object()

    launch_rattlesnake_ui(rattlesnake)