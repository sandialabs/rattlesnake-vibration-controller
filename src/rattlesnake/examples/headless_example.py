from rattlesnake.examples.example_registry import (
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


def build_rattlesnake_object(
    threaded=THREADED,
    import_method=IMPORT_METHOD,
    hardware_type=HARDWARE_TYPE,
    environment_type=ENVIRONMENT_TYPE,
    stream_type=STREAM_TYPE,
    load_sysid=LOAD_SYSID,
    run_sysid=RUN_SYSID,
    start_hardware=START_HARDWARE,
    start_environment=START_ENVIRONMENT,
    run_profile=RUN_PROFILE,
):
    rattlesnake = RattlesnakeController(threaded=threaded, timeout=120)

    hardware_metadata = HARDWARE_DICT[hardware_type][import_method]()
    environment_metadata = ENVIRONMENT_DICT[environment_type][import_method](
        hardware_metadata
    )
    environment_name = getattr(environment_metadata, "environment_name", None)
    sysid_metadata = SYSID_DICT[import_method](hardware_metadata)
    sysid_package = SYSID_LOAD_DICT["netcdf"]()
    stream_metadata = STREAM_DICT[stream_type](environment_name)
    event_list = EVENT_DICT[environment_type]()
    instructions = INSTRUCTIONS_DICT[environment_type]()

    # Initialize hardware
    if hardware_type is HardwareType.NONE:
        return rattlesnake
    rattlesnake.initialize_hardware(hardware_metadata)

    # Initialize environment
    if environment_type is EnvironmentType.NONE:
        return rattlesnake
    rattlesnake.initialize_environments([environment_metadata])

    # These are purely to set the UI
    rattlesnake.initialize_profile_event_list(event_list)
    rattlesnake.set_stream_metadata(stream_metadata)

    if environment_type in SYSID_ENVIRONMENTS:
        rattlesnake.initialize_system_id(sysid_metadata, environment_name)
        if run_sysid:
            rattlesnake.run_system_id(sysid_metadata, environment_name)
        if load_sysid:
            rattlesnake.load_system_id_from_package(environment_name, sysid_package)

    # Start Acquisition
    if not start_hardware:
        return rattlesnake
    rattlesnake.start_acquisition(stream_metadata)

    # Start Environment
    if not start_environment or run_profile:
        return rattlesnake
    if run_profile:
        rattlesnake.start_profile(event_list)
    else:
        rattlesnake.start_environment(instructions=instructions)
    return rattlesnake


if __name__ == "__main__":
    print("Loading Rattlesnake...")

    # test_rattlesnake_objects()
    rattlesnake = build_rattlesnake_object()

    launch_rattlesnake_ui(rattlesnake)
