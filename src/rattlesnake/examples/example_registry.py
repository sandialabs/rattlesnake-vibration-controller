from rattlesnake.examples.hardware.sdynpy_system.sdynpy_system_metadata import (
    manual_sdynpy_system_metadata,
    netcdf_sdynpy_system_metadata,
    worksheet_sdynpy_system_metadata,
)
from rattlesnake.examples.hardware.stream_metadata import (
    stream_metadata_no,
    stream_metadata_immediate,
    stream_metadata_manual,
    stream_metadata_profile,
    stream_metadata_test_level,
)
from rattlesnake.examples.environment.time.time_metadata import (
    manual_time_metadata,
    netcdf_time_metadata,
    worksheet_time_metadata,
    time_event_list,
    time_instructions,
)
from rattlesnake.examples.environment.modal.modal_metadata import (
    manual_modal_metadata,
    netcdf_modal_metadata,
    worksheet_modal_metadata,
    modal_instructions,
)
from rattlesnake.examples.environment.sine.sine_metadata import (
    manual_sine_metadata,
    worksheet_sine_metadata,
    netcdf_sine_metadata,
    sine_instructions,
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
)
from rattlesnake.examples.environment.transient.transient_metadata import (
    netcdf_transient_metadata,
    manual_transient_metadata,
    worksheet_transient_metadata,
    transient_instructions,
)

from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.process.streaming import StreamType

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
HARDWARE_DICT[HardwareType.NONE] = BLANK_HARDWARE_DICT
HARDWARE_DICT[HardwareType.SDYNPY_SYSTEM] = SDYNPY_SYSTEM_DICT


# Environment
ENVIRONMENT_DICT = {}

BLANK_ENVIRONMENT_DICT = {
    "worksheet": lambda x: None,
    "netcdf": lambda x: None,
    "manual": lambda x: None,
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
ENVIRONMENT_DICT[EnvironmentType.TIME] = TIME_DICT
ENVIRONMENT_DICT[EnvironmentType.MODAL] = MODAL_DICT
ENVIRONMENT_DICT[EnvironmentType.SINE] = SINE_DICT
ENVIRONMENT_DICT[EnvironmentType.RANDOM] = RANDOM_DICT
ENVIRONMENT_DICT[EnvironmentType.TRANSIENT] = TRANSIENT_DICT

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
EVENT_DICT[EnvironmentType.NONE] = lambda: []
EVENT_DICT[EnvironmentType.TIME] = time_event_list
EVENT_DICT[EnvironmentType.MODAL] = lambda: []
EVENT_DICT[EnvironmentType.SINE] = lambda: []
EVENT_DICT[EnvironmentType.RANDOM] = lambda: []
EVENT_DICT[EnvironmentType.TRANSIENT] = lambda: []

# Instructions
INSTRUCTIONS_DICT = {}
INSTRUCTIONS_DICT[EnvironmentType.NONE] = lambda: None
INSTRUCTIONS_DICT[EnvironmentType.TIME] = time_instructions
INSTRUCTIONS_DICT[EnvironmentType.MODAL] = modal_instructions
INSTRUCTIONS_DICT[EnvironmentType.SINE] = sine_instructions
INSTRUCTIONS_DICT[EnvironmentType.RANDOM] = random_instructions
INSTRUCTIONS_DICT[EnvironmentType.TRANSIENT] = transient_instructions
