from enum import Enum


class HardwareType(Enum):
    NI_DAQMX = 0
    LAN_XI = 1
    DP_QUATTRO = 2
    DP_900 = 3
    EXODUS = 4
    STATE_SPACE = 5
    SDYNPY_SYSTEM = 6
    SDYNPY_FRF = 7
