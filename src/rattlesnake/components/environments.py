# -*- coding: utf-8 -*-
"""
This file contains the interfaces to the individual environments, and should be
modified when adding new environment control strategies.

Rattlesnake Vibration Control Software
Copyright (C) 2021  National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from enum import Enum
import sys
import os

### Here is where the code needs to be modified to create a new environment.

class ControlTypes(Enum):
    """Enumeration of the possible control types"""
    COMBINED = 0
    RANDOM = 1
    TRANSIENT = 2
    # SINE = 3
    TIME = 4
    # NONLINEAR = 5
    MODAL = 6
    # Add new environment types here

# Name for each environment
environment_long_names = {}
environment_long_names[ControlTypes.RANDOM] = 'MIMO Random Vibration'
environment_long_names[ControlTypes.TRANSIENT] = 'MIMO Transient'
# environment_long_names[ControlTypes.SINE] = 'Sine Vibration'
environment_long_names[ControlTypes.TIME] = 'Time Signal Generation'
# environment_long_names[ControlTypes.NONLINEAR] = 'Nonlinear Normal Modes'
environment_long_names[ControlTypes.MODAL] = 'Modal Testing'
environment_long_names[ControlTypes.COMBINED] = 'Combined Environments...'

# Add the environment here if it can be used for combined environments
combined_environments_capable = [
    ControlTypes.RANDOM,
    ControlTypes.TRANSIENT,
    ControlTypes.TIME,
    ControlTypes.MODAL
    ]

# Define paths to the User Interface UI Files
environment_definition_ui_paths = {}
environment_prediction_ui_paths = {}
environment_run_ui_paths = {}
# This is true if running from an executable and the UI is embedded in the executable
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    directory = sys._MEIPASS
else:
    directory = 'components'
    
# Base Controller UI
ui_path = os.path.join(directory,'combined_environments_controller.ui')
environment_select_ui_path = os.path.join(directory,'environment_selector.ui')
control_select_ui_path = os.path.join(directory,'control_select.ui')
# Random Vibration Environment
environment_definition_ui_paths[ControlTypes.RANDOM] = os.path.join(directory,'random_vibration_definition.ui')
environment_prediction_ui_paths[ControlTypes.RANDOM] = os.path.join(directory,'random_vibration_prediction.ui')
environment_run_ui_paths[ControlTypes.RANDOM] = os.path.join(directory,'random_vibration_run.ui')
system_identification_ui_path = os.path.join(directory,'system_identification.ui')
transformation_matrices_ui_path = os.path.join(directory,'transformation_matrices.ui')
# Time Environment
environment_definition_ui_paths[ControlTypes.TIME] = os.path.join(directory,'time_definition.ui')
environment_run_ui_paths[ControlTypes.TIME] = os.path.join(directory,'time_run.ui')
# Transient Environment
environment_definition_ui_paths[ControlTypes.TRANSIENT] = os.path.join(directory,'transient_definition.ui')
environment_prediction_ui_paths[ControlTypes.TRANSIENT] = os.path.join(directory,'transient_prediction.ui')
environment_run_ui_paths[ControlTypes.TRANSIENT] = os.path.join(directory,'transient_run.ui')
# Modal Environments
environment_definition_ui_paths[ControlTypes.MODAL] = os.path.join(directory,'modal_definition.ui')
environment_run_ui_paths[ControlTypes.MODAL] = os.path.join(directory,'modal_run.ui')
modal_mdi_ui_path = os.path.join(directory,'modal_acquisition_window.ui')

# Import the process function and the UI from the module and add them to the
# respective dictionaries
environment_processes = {}
environment_UIs = {}
# Random Vibration
from .random_vibration_sys_id_environment import random_vibration_process,RandomVibrationUI
environment_processes[ControlTypes.RANDOM] = random_vibration_process
environment_UIs[ControlTypes.RANDOM] = RandomVibrationUI
# Time Signal Generation
from .time_environment import time_process,TimeUI
environment_processes[ControlTypes.TIME] = time_process
environment_UIs[ControlTypes.TIME] = TimeUI
# Transient
from .transient_sys_id_environment import transient_process,TransientUI
environment_processes[ControlTypes.TRANSIENT] = transient_process
environment_UIs[ControlTypes.TRANSIENT] = TransientUI
# Modal
from .modal_environment import modal_process,ModalUI
environment_processes[ControlTypes.MODAL] = modal_process
environment_UIs[ControlTypes.MODAL] = ModalUI

### End of code needed to be modified to create a new environment