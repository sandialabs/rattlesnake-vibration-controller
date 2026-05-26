from hardware.sdynpy_system.sdynpy_system_metadata import manual_sdynpy_system_metadata, netcdf_sdynpy_system_metadata, worksheet_sdynpy_system_metadata
from hardware.stream_metadata import (stream_metadata_no, stream_metadata_immediate,
                                      stream_metadata_manual, stream_metadata_profile, stream_metadata_test_level,)
from environment.time.time_metadata import (manual_time_metadata, netcdf_time_metadata, worksheet_time_metadata, 
time_event_list, time_instructions)

from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.process.streaming import StreamType

# Hardware
HARDWARE_DICT = {}

BLANK_DICT = {
    "worksheet": lambda: None,
    "netcdf": lambda: None,
    "manual": lambda: None,
}
SDYNPY_SYSTEM_DICT = {
    "worksheet": worksheet_sdynpy_system_metadata,
    "netcdf": netcdf_sdynpy_system_metadata,
    "manual": manual_sdynpy_system_metadata,
    }
HARDWARE_DICT[HardwareType.NONE] = BLANK_DICT
HARDWARE_DICT[HardwareType.SDYNPY_SYSTEM] = SDYNPY_SYSTEM_DICT


# Environment
ENVIRONMENT_DICT = {}

TIME_DICT = {
    "worksheet": worksheet_time_metadata,
    "netcdf": netcdf_time_metadata,
    "manual": manual_time_metadata,
    }
ENVIRONMENT_DICT[EnvironmentType.NONE] = BLANK_DICT
ENVIRONMENT_DICT[EnvironmentType.TIME] = TIME_DICT

# Streaming
STREAM_DICT = {}
STREAM_DICT[StreamType.NO_STREAM] = stream_metadata_no
STREAM_DICT[StreamType.IMMEDIATELY] = stream_metadata_immediate
STREAM_DICT[StreamType.MANUAL] = stream_metadata_manual
STREAM_DICT[StreamType.PROFILE_INSTRUCTION] = stream_metadata_profile
STREAM_DICT[StreamType.TEST_LEVEL] = stream_metadata_test_level

# Instructions
INSTRUCTIONS_DICT = {}
INSTRUCTIONS_DICT[EnvironmentType.NONE] = lambda: None
INSTRUCTIONS_DICT[EnvironmentType.TIME] = time_instructions

# Event list
EVENT_DICT = {}
EVENT_DICT[EnvironmentType.NONE] = lambda: None
EVENT_DICT[EnvironmentType.TIME] = time_event_list