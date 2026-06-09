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

from rattlesnake.testing.mock_user_interface import launch_temporary_rattlesnake_ui_environment

"""USER INPUTS"""
THREADED = True
IMPORT_METHOD = "worksheet"  # worksheet, netcdf, manual
HARDWARE_TYPE = HardwareType.SDYNPY_SYSTEM
ENVIRONMENT_TYPE = EnvironmentType.RANDOM
STREAM_TYPE = StreamType.NO_STREAM
LOAD_SYSID = True
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

    # Initialize hardware
    if hardware_type is HardwareType.NONE:
        return rattlesnake
    hardware_metadata = HARDWARE_DICT[hardware_type][import_method]()
    rattlesnake.initialize_hardware(hardware_metadata)

    # Initialize environment
    if environment_type is EnvironmentType.NONE:
        return rattlesnake
    environment_metadata = ENVIRONMENT_DICT[environment_type][import_method](
        hardware_metadata
    )
    environment_name = getattr(environment_metadata, "environment_name", None)
    rattlesnake.initialize_environments([environment_metadata])

    # Run System Identification
    if environment_type in SYSID_ENVIRONMENTS:
        sysid_metadata = SYSID_DICT["manual"](hardware_metadata)
        rattlesnake.initialize_system_id(sysid_metadata, environment_name)
        if run_sysid:
            rattlesnake.run_system_id(sysid_metadata, environment_name)
        if load_sysid:
            sysid_package = SYSID_LOAD_DICT["netcdf"]()
            rattlesnake.load_system_id_from_package(environment_name, sysid_package)

    # Initialize profile event list and stream metadata in the UI (Don't normally need this for headless)
    event_list = EVENT_DICT[environment_type]()
    stream_metadata = STREAM_DICT[stream_type](environment_name)
    rattlesnake.initialize_profile_event_list(event_list)
    rattlesnake.set_stream_metadata(stream_metadata)

    # Start Hardware
    if not start_hardware:
        return rattlesnake
    rattlesnake.start_acquisition(stream_metadata)

    # Start Environment
    if run_profile:
        rattlesnake.start_profile(event_list)
    elif start_environment:
        instructions = INSTRUCTIONS_DICT[environment_type]()
        rattlesnake.start_environment(instructions=instructions)
    return rattlesnake


if __name__ == "__main__":
    print("Loading Rattlesnake...")

    # test_rattlesnake_objects()
    rattlesnake = build_rattlesnake_object()

    # launch_temporary_rattlesnake_ui_environment(rattlesnake, 60)
    launch_rattlesnake_ui(rattlesnake)
