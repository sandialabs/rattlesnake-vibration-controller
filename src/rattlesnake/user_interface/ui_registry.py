from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.hardware.hardware_registry import UNIMPLEMENTED_HARDWARE
from rattlesnake.environment.environment_utilities import EnvironmentType
from qtpy import QtWidgets, uic, QtGui
from rattlesnake.environment.environment_registry import UNIMPLEMENTED_ENVIRONMENT

# region Hardware
UI_HARDWARE_OPTIONS = {"Select Hardware": HardwareType.NONE}
# See user_interface.py => self.hardware_widgets for options
UI_HARDWARE_WIDGETS = {HardwareType.NONE: []}
UI_ASK_FOR_FILE = []

for hardware_type in HardwareType:
    if hardware_type in UNIMPLEMENTED_HARDWARE:
        continue

    match hardware_type:
        case HardwareType.NI_DAQMX:
            UI_HARDWARE_OPTIONS["NI DAQmx"] = HardwareType.NI_DAQMX
            UI_HARDWARE_WIDGETS[HardwareType.NI_DAQMX] = [
                "sample_rate",
                "buffer_size",
                "task_trigger",
                "trigger_output",
            ]
        case HardwareType.LAN_XI:
            UI_HARDWARE_OPTIONS["HBK LAN-XI"] = HardwareType.LAN_XI
            UI_HARDWARE_WIDGETS[HardwareType.LAN_XI] = [
                "lanxi_sample_rate",
                "buffer_size",
                "lanxi_processes",
                "lanxi_ip",
            ]
        case HardwareType.DP_QUATTRO:
            UI_HARDWARE_OPTIONS["Data Physics Quattro"] = HardwareType.DP_QUATTRO
            UI_HARDWARE_WIDGETS[HardwareType.DP_QUATTRO] = [
                "sample_rate",
                "buffer_size",
                "integration_oversample",
                "select_file",
            ]
            UI_ASK_FOR_FILE.append(HardwareType.DP_QUATTRO)
        case HardwareType.DP_900:
            UI_HARDWARE_OPTIONS["Data Physics 900 Series"] = HardwareType.DP_900
            UI_HARDWARE_WIDGETS[HardwareType.DP_900] = [
                "sample_rate",
                "buffer_size",
                "integration_oversample",
                "select_file",
            ]
            UI_ASK_FOR_FILE.append(HardwareType.DP_900)
        case HardwareType.EXODUS:
            UI_HARDWARE_OPTIONS["Exodus Modal Solution..."] = HardwareType.EXODUS
            UI_HARDWARE_WIDGETS[HardwareType.EXODUS] = [
                "sample_rate",
                "buffer_size",
                "integration_oversample",
                "damping_ratio",
                "select_file",
            ]
            UI_ASK_FOR_FILE.append(HardwareType.EXODUS)
        case HardwareType.STATE_SPACE:
            UI_HARDWARE_OPTIONS["State Space Integration..."] = HardwareType.STATE_SPACE
            UI_HARDWARE_WIDGETS[HardwareType.STATE_SPACE] = [
                "sample_rate",
                "buffer_size",
                "integration_oversample",
                "select_file",
            ]
            UI_ASK_FOR_FILE.append(HardwareType.STATE_SPACE)
        case HardwareType.SDYNPY_SYSTEM:
            UI_HARDWARE_OPTIONS["SDynPy System Integration..."] = (
                HardwareType.SDYNPY_SYSTEM
            )
            UI_HARDWARE_WIDGETS[HardwareType.SDYNPY_SYSTEM] = [
                "sample_rate",
                "buffer_size",
                "integration_oversample",
                "select_file",
            ]
            UI_ASK_FOR_FILE.append(HardwareType.SDYNPY_SYSTEM)
        case HardwareType.SDYNPY_FRF:
            UI_HARDWARE_OPTIONS["SDynPy FRF Convolution..."] = HardwareType.SDYNPY_FRF
            UI_HARDWARE_WIDGETS[HardwareType.SDYNPY_FRF] = [
                "sample_rate",
                "buffer_size",
                "select_file",
            ]
            UI_ASK_FOR_FILE.append(HardwareType.SDYNPY_FRF)


# region Environments
ENVIRONMENT_UIS = {}
UI_ENVIRONMENT_OPTIONS = {"Add Environment": None}

for environment_type in EnvironmentType:
    if environment_type in UNIMPLEMENTED_ENVIRONMENT:
        continue
    match environment_type:
        case EnvironmentType.SKELETON:
            from rattlesnake.user_interface.skeleton_ui import SkeletonUI

            ENVIRONMENT_UIS[EnvironmentType.SKELETON] = SkeletonUI
            # UI_ENVIRONMENT_OPTIONS["Skeleton Environment"] = EnvironmentType.SKELETON
        case EnvironmentType.SYSID_SKELETON:
            from rattlesnake.user_interface.skeleton_sys_id_ui import SkeletonSysIdUI

            ENVIRONMENT_UIS[EnvironmentType.SYSID_SKELETON] = SkeletonSysIdUI
            # UI_ENVIRONMENT_OPTIONS["Skeleton SysId Environment"] = (
            #     EnvironmentType.SYSID_SKELETON
            # )
        case EnvironmentType.READ:
            from rattlesnake.user_interface.read_ui import ReadUI

            ENVIRONMENT_UIS[EnvironmentType.READ] = ReadUI
            UI_ENVIRONMENT_OPTIONS["Read Data"] = EnvironmentType.READ
        case EnvironmentType.TIME:
            from rattlesnake.user_interface.time_ui import TimeUI

            ENVIRONMENT_UIS[EnvironmentType.TIME] = TimeUI
            UI_ENVIRONMENT_OPTIONS["Time Signal Generation"] = EnvironmentType.TIME
        case EnvironmentType.MODAL:
            from rattlesnake.user_interface.modal_ui import ModalUI

            ENVIRONMENT_UIS[EnvironmentType.MODAL] = ModalUI
            UI_ENVIRONMENT_OPTIONS["Modal Testing"] = EnvironmentType.MODAL
        case EnvironmentType.SINE:
            from rattlesnake.user_interface.sine_sys_id_ui import SineUI

            ENVIRONMENT_UIS[EnvironmentType.SINE] = SineUI
            UI_ENVIRONMENT_OPTIONS["MIMO Sine Vibration"] = EnvironmentType.SINE
        case EnvironmentType.TRANSIENT:
            from rattlesnake.user_interface.transient_sys_id_ui import TransientUI

            ENVIRONMENT_UIS[EnvironmentType.TRANSIENT] = TransientUI
            UI_ENVIRONMENT_OPTIONS["MIMO Transient"] = EnvironmentType.TRANSIENT
        case EnvironmentType.RANDOM:
            from rattlesnake.user_interface.random_vibration_sys_id_ui import (
                RandomVibrationUI,
            )

            ENVIRONMENT_UIS[EnvironmentType.RANDOM] = RandomVibrationUI
            UI_ENVIRONMENT_OPTIONS["MIMO Random Vibration"] = EnvironmentType.RANDOM
        case _:
            continue
# endregion
