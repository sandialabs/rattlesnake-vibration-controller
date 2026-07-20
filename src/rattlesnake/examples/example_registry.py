from rattlesnake.examples.hardware.sdynpy_system.sdynpy_system_metadata import (
    manual_sdynpy_system_metadata,
    netcdf_sdynpy_system_metadata,
    worksheet_sdynpy_system_metadata,
)
from rattlesnake.examples.hardware.sdynpy_frf.sdynpy_frf_metadata import (
    manual_sdynpy_frf_metadata,
    netcdf_sdynpy_frf_metadata,
    worksheet_sdynpy_frf_metadata,
)
from rattlesnake.examples.hardware.state_space.state_space_metadata import (
    manual_state_space_metadata,
    netcdf_state_space_metadata,
    worksheet_state_space_metadata,
)
from rattlesnake.examples.hardware.exodus.exodus_metadata import (
    manual_exodus_metadata,
    netcdf_exodus_metadata,
    worksheet_exodus_metadata,
)
from rattlesnake.examples.hardware.stream_metadata import (
    stream_metadata_no,
    stream_metadata_immediate,
    stream_metadata_manual,
    stream_metadata_profile,
    stream_metadata_test_level,
)
from rattlesnake.examples.environment.skeleton.skeleton_metadata import (
    manual_skeleton_metadata,
    netcdf_skeleton_metadata,
    worksheet_skeleton_metadata,
    skeleton_event_list,
    worksheet_skeleton_event_list,
    skeleton_instructions,
)
from rattlesnake.examples.environment.skeleton_sysid.skeleton_sysid_metadata import (
    manual_skeleton_sysid_metadata,
    netcdf_skeleton_sysid_metadata,
    worksheet_skeleton_sysid_metadata,
    skeleton_sysid_instructions,
)
from rattlesnake.examples.environment.time.time_metadata import (
    manual_time_metadata,
    netcdf_time_metadata,
    worksheet_time_metadata,
    time_event_list,
    worksheet_time_event_list,
    time_instructions,
)
from rattlesnake.examples.environment.modal.modal_metadata import (
    manual_modal_metadata,
    netcdf_modal_metadata,
    worksheet_modal_metadata,
    modal_instructions,
    modal_event_list,
    worksheet_modal_event_list,
)
from rattlesnake.examples.environment.sine.sine_metadata import (
    manual_sine_metadata,
    worksheet_sine_metadata,
    netcdf_sine_metadata,
    sine_instructions,
    sine_event_list,
    worksheet_sine_event_list,
)
from rattlesnake.examples.environment.sysid.sysid_metadata import (
    manual_sysid_metadata,
    worksheet_sysid_metadata,
    netcdf_sysid_metadata,
    netcdf_sysid_data_package,
)
from rattlesnake.examples.environment.random.random_metadata import (
    manual_random_metadata,
    netcdf_random_metadata,
    worksheet_random_metadata,
    random_instructions,
    random_event_list,
    worksheet_random_event_list,
)
from rattlesnake.examples.environment.transient.transient_metadata import (
    netcdf_transient_metadata,
    manual_transient_metadata,
    worksheet_transient_metadata,
    transient_instructions,
    transient_event_list,
    worksheet_transient_event_list,
)

from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.process.streaming import StreamType
import rattlesnake.examples.defaults as defaults

# Hardware
HARDWARE_DICT = {}

BLANK_HARDWARE_DICT = {
    "worksheet": lambda: None,
    "netcdf": lambda: None,
    "manual": lambda: None,
}
SDYNPY_SYSTEM_DICT = {
    "worksheet": worksheet_sdynpy_system_metadata,
    "netcdf": netcdf_sdynpy_system_metadata,
    "manual": manual_sdynpy_system_metadata,
}
SDYNPY_FRF_DICT = {
    "worksheet": worksheet_sdynpy_frf_metadata,
    "netcdf": netcdf_sdynpy_frf_metadata,
    "manual": manual_sdynpy_frf_metadata,
}
STATE_SPACE_DICT = {
    "worksheet": worksheet_state_space_metadata,
    "netcdf": netcdf_state_space_metadata,
    "manual": manual_state_space_metadata,
}
EXODUS_DICT = {
    "worksheet": worksheet_exodus_metadata,
    "netcdf": netcdf_exodus_metadata,
    "manual": manual_exodus_metadata,
}
HARDWARE_DICT[HardwareType.NONE] = BLANK_HARDWARE_DICT
HARDWARE_DICT[HardwareType.SDYNPY_SYSTEM] = SDYNPY_SYSTEM_DICT
HARDWARE_DICT[HardwareType.SDYNPY_FRF] = SDYNPY_FRF_DICT
HARDWARE_DICT[HardwareType.STATE_SPACE] = STATE_SPACE_DICT
HARDWARE_DICT[HardwareType.EXODUS] = EXODUS_DICT

# Environment
ENVIRONMENT_DICT = {}

BLANK_ENVIRONMENT_DICT = {
    "worksheet": lambda x: None,
    "netcdf": lambda x: None,
    "manual": lambda x: None,
}
SKELETON_DICT = {
    "worksheet": worksheet_skeleton_metadata,
    "netcdf": netcdf_skeleton_metadata,
    "manual": manual_skeleton_metadata,
}
SKELETON_SYSID_DICT = {
    "worksheet": worksheet_skeleton_sysid_metadata,
    "netcdf": netcdf_skeleton_sysid_metadata,
    "manual": manual_skeleton_sysid_metadata,
}
TIME_DICT = {
    "worksheet": worksheet_time_metadata,
    "netcdf": netcdf_time_metadata,
    "manual": manual_time_metadata,
}
MODAL_DICT = {
    "manual": manual_modal_metadata,
    "worksheet": worksheet_modal_metadata,
    "netcdf": netcdf_modal_metadata,
}
SINE_DICT = {
    "manual": manual_sine_metadata,
    "worksheet": worksheet_sine_metadata,
    "netcdf": netcdf_sine_metadata,
}
RANDOM_DICT = {
    "manual": manual_random_metadata,
    "worksheet": worksheet_random_metadata,
    "netcdf": netcdf_random_metadata,
}
TRANSIENT_DICT = {
    "manual": manual_transient_metadata,
    "worksheet": worksheet_transient_metadata,
    "netcdf": netcdf_transient_metadata,
}
ENVIRONMENT_DICT[EnvironmentType.NONE] = BLANK_ENVIRONMENT_DICT
ENVIRONMENT_DICT[EnvironmentType.SKELETON] = SKELETON_DICT
ENVIRONMENT_DICT[EnvironmentType.SYSID_SKELETON] = SKELETON_SYSID_DICT
ENVIRONMENT_DICT[EnvironmentType.TIME] = TIME_DICT
ENVIRONMENT_DICT[EnvironmentType.MODAL] = MODAL_DICT
ENVIRONMENT_DICT[EnvironmentType.SINE] = SINE_DICT
ENVIRONMENT_DICT[EnvironmentType.RANDOM] = RANDOM_DICT
ENVIRONMENT_DICT[EnvironmentType.TRANSIENT] = TRANSIENT_DICT

# Example Files for Testing Suite
EXAMPLE_NETCDF = {
    EnvironmentType.SKELETON: defaults.DIRECTORY
    + "/environment/skeleton/skeleton_v4.nc4",
    EnvironmentType.SYSID_SKELETON: defaults.DIRECTORY
    + "/environment/skeleton_sysid/skeleton_sysid_v4.nc4",
    EnvironmentType.TIME: defaults.DIRECTORY + "/environment/time/time_v4.nc4",
    EnvironmentType.MODAL: defaults.DIRECTORY + "/environment/modal/modal_v4.nc4",
    EnvironmentType.SINE: defaults.DIRECTORY + "/environment/sine/sine_v4.nc4",
    EnvironmentType.RANDOM: defaults.DIRECTORY + "/environment/random/random_v4.nc4",
    EnvironmentType.TRANSIENT: defaults.DIRECTORY
    + "/environment/transient/transient_v4.nc4",
}
EXAMPLE_WORKSHEET = {
    EnvironmentType.SKELETON: defaults.DIRECTORY
    + "/environment/skeleton/skeleton_v4.xlsx",
    EnvironmentType.SYSID_SKELETON: defaults.DIRECTORY
    + "/environment/skeleton_sysid/skeleton_sysid_v4.xlsx",
    EnvironmentType.TIME: defaults.DIRECTORY + "/environment/time/time_v4.xlsx",
    EnvironmentType.MODAL: defaults.DIRECTORY + "/environment/modal/modal_v4.xlsx",
    EnvironmentType.SINE: defaults.DIRECTORY + "/environment/sine/sine_v4.xlsx",
    EnvironmentType.RANDOM: defaults.DIRECTORY + "/environment/random/random_v4.xlsx",
    EnvironmentType.TRANSIENT: defaults.DIRECTORY
    + "/environment/transient/transient_v4.xlsx",
}

# System Identification
SYSID_DICT = {
    "manual": manual_sysid_metadata,
    "worksheet": worksheet_sysid_metadata,
    "netcdf": netcdf_sysid_metadata,
}
SYSID_LOAD_DICT = {"netcdf": netcdf_sysid_data_package}

# Streaming
STREAM_DICT = {}
STREAM_DICT[StreamType.NO_STREAM] = stream_metadata_no
STREAM_DICT[StreamType.IMMEDIATELY] = stream_metadata_immediate
STREAM_DICT[StreamType.MANUAL] = stream_metadata_manual
STREAM_DICT[StreamType.PROFILE_INSTRUCTION] = stream_metadata_profile
STREAM_DICT[StreamType.TEST_LEVEL] = stream_metadata_test_level

# Event list
EVENT_DICT = {}
BLANK_EVENT_DICT = {
    "worksheet": lambda: [],
    "netcdf": lambda: [],
    "manual": lambda: [],
}
SKELETON_EVENT_DICT = {
    "manual": skeleton_event_list,
    "netcdf": skeleton_event_list,
    "worksheet": worksheet_skeleton_event_list,
}
SKELETON_SYSID_EVENT_DICT = {
    "worksheet": lambda: [],
    "netcdf": lambda: [],
    "manual": lambda: [],
}
TIME_EVENT_DICT = {
    "manual": time_event_list,
    "netcdf": time_event_list,
    "worksheet": worksheet_time_event_list,
}
MODAL_EVENT_DICT = {
    "manual": modal_event_list,
    "netcdf": modal_event_list,
    "worksheet": worksheet_modal_event_list,
}
SINE_EVENT_DICT = {
    "manual": sine_event_list,
    "netcdf": sine_event_list,
    "worksheet": worksheet_sine_event_list,
}
RANDOM_EVENT_DICT = {
    "manual": random_event_list,
    "netcdf": random_event_list,
    "worksheet": worksheet_random_event_list,
}
TRANSIENT_EVENT_DICT = {
    "manual": transient_event_list,
    "netcdf": transient_event_list,
    "worksheet": worksheet_transient_event_list,
}
EVENT_DICT[EnvironmentType.NONE] = BLANK_EVENT_DICT
EVENT_DICT[EnvironmentType.SKELETON] = SKELETON_EVENT_DICT
EVENT_DICT[EnvironmentType.SYSID_SKELETON] = SKELETON_SYSID_EVENT_DICT
EVENT_DICT[EnvironmentType.TIME] = TIME_EVENT_DICT
EVENT_DICT[EnvironmentType.MODAL] = MODAL_EVENT_DICT
EVENT_DICT[EnvironmentType.SINE] = SINE_EVENT_DICT
EVENT_DICT[EnvironmentType.RANDOM] = RANDOM_EVENT_DICT
EVENT_DICT[EnvironmentType.TRANSIENT] = TRANSIENT_EVENT_DICT

# Instructions
INSTRUCTIONS_DICT = {}
INSTRUCTIONS_DICT[EnvironmentType.NONE] = lambda: None
INSTRUCTIONS_DICT[EnvironmentType.SKELETON] = skeleton_instructions
INSTRUCTIONS_DICT[EnvironmentType.SYSID_SKELETON] = skeleton_sysid_instructions
INSTRUCTIONS_DICT[EnvironmentType.TIME] = time_instructions
INSTRUCTIONS_DICT[EnvironmentType.MODAL] = modal_instructions
INSTRUCTIONS_DICT[EnvironmentType.SINE] = sine_instructions
INSTRUCTIONS_DICT[EnvironmentType.RANDOM] = random_instructions
INSTRUCTIONS_DICT[EnvironmentType.TRANSIENT] = transient_instructions
