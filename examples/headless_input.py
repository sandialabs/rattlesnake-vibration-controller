from example_registry import (
    HARDWARE_DICT,
    ENVIRONMENT_DICT,
    SYSID_DICT,
    SYSID_LOAD_DICT,
    STREAM_DICT,
    INSTRUCTIONS_DICT,
    EVENT_DICT,
)

from rattlesnake.engine import RattlesnakeController
from rattlesnake.main import launch_rattlesnake_ui
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.environment_registry import SYSID_ENVIRONMENTS
from rattlesnake.process.streaming import StreamType

"""USER INPUTS"""
THREADED = False
IMPORT_METHOD = "manual"  # worksheet, netcdf, manual
HARDWARE_TYPE = HardwareType.SDYNPY_SYSTEM
ENVIRONMENT_TYPE = EnvironmentType.NONE
STREAM_TYPE = StreamType.NO_STREAM
LOAD_SYSID = False
RUN_SYSID = False
START_HARDWARE = False
START_ENVIRONMENT = False
RUN_PROFILE = False


def build_rattlesnake_object():
    rattlesnake = RattlesnakeController(threaded=THREADED, timeout=120)

    hardware_metadata = HARDWARE_DICT[HARDWARE_TYPE][IMPORT_METHOD]()
    environment_metadata = ENVIRONMENT_DICT[ENVIRONMENT_TYPE][IMPORT_METHOD](
        hardware_metadata
    )
    environment_name = getattr(environment_metadata, "environment_name", None)
    sysid_metadata = SYSID_DICT[IMPORT_METHOD](hardware_metadata)
    sysid_package = SYSID_LOAD_DICT["netcdf"]()
    stream_metadata = STREAM_DICT[STREAM_TYPE](environment_name)
    event_list = EVENT_DICT[ENVIRONMENT_TYPE]()
    instructions = INSTRUCTIONS_DICT[ENVIRONMENT_TYPE]()

    # Initialize hardware
    if HARDWARE_TYPE is HardwareType.NONE:
        return rattlesnake
    rattlesnake.initialize_hardware(hardware_metadata)

    # Initialize environment
    if ENVIRONMENT_TYPE is EnvironmentType.NONE:
        return rattlesnake
    rattlesnake.initialize_environments([environment_metadata])

    # These are purely to set the UI
    rattlesnake.initialize_profile_event_list(event_list)
    rattlesnake.set_stream_metadata(stream_metadata)

    if ENVIRONMENT_TYPE in SYSID_ENVIRONMENTS:
        rattlesnake.initialize_system_id(sysid_metadata, environment_name)
        if RUN_SYSID:
            rattlesnake.run_system_id(sysid_metadata, environment_name)
        if LOAD_SYSID:
            rattlesnake.load_system_id_from_package(environment_name, sysid_package)

    # Start Acquisition
    if not START_HARDWARE:
        return rattlesnake
    rattlesnake.start_acquisition(stream_metadata)

    # Start Environment
    if not START_ENVIRONMENT or RUN_PROFILE:
        return rattlesnake
    if RUN_PROFILE:
        rattlesnake.start_profile(event_list)
    else:
        rattlesnake.start_environment(instructions=instructions)
    return rattlesnake


def test_rattlesnake_objects():
    hardware_metadata = HARDWARE_DICT[HARDWARE_TYPE]["manual"]()
    hardware_metadata2 = HARDWARE_DICT[HARDWARE_TYPE]["worksheet"]()
    pass


if __name__ == "__main__":
    print("Loading Rattlesnake...")

    # test_rattlesnake_objects()
    rattlesnake = build_rattlesnake_object()

    launch_rattlesnake_ui(rattlesnake)
