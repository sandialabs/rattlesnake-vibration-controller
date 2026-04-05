from rattlesnake.environment.environment_utilities import EnvironmentType
from qtpy import QtWidgets, uic, QtGui
from rattlesnake.environment.environment_registry import UNIMPLEMENTED_ENVIRONMENT

ENVIRONMENT_UIS = {}
UI_ENVIRONMENT_OPTIONS = {"Add Environment": None}

for environment_type in EnvironmentType:
    if environment_type in UNIMPLEMENTED_ENVIRONMENT:
        continue
    match environment_type:
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
