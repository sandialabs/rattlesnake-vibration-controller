from hardware.sdynpy_metadata import manual_sdynpy_system_metadata, template_sdynpy_system_metadata, worksheet_sdynpy_system_metadata

from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.environment_utilities import EnvironmentType


SDYNPY_SYSTEM_DICT = {
    "worksheet": worksheet_sdynpy_system_metadata,
    "netcdf": template_sdynpy_system_metadata,
    "manual": manual_sdynpy_system_metadata,
    }

HARDWARE_DICT = {HardwareType.SDYNPY_SYSTEM: SDYNPY_SYSTEM_DICT}