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


class ControlTypes(Enum):
    """Enumeration of the possible control types"""

    COMBINED = 0
    RANDOM = 1
    TRANSIENT = 2
    SINE = 3
    SDS = 4
    TIME = 5
    # NONLINEAR = 6
    MODAL = 7
    # Add new environment types here


# Name for each environment
environment_long_names = {}
environment_long_names[ControlTypes.RANDOM] = "MIMO Random Vibration"
environment_long_names[ControlTypes.TRANSIENT] = "MIMO Transient"
environment_long_names[ControlTypes.SINE] = "MIMO Sine Vibration"
environment_long_names[ControlTypes.SDS] = "MIMO Sum-Decayed-Sines Shock"
environment_long_names[ControlTypes.TIME] = "Time Signal Generation"
# environment_long_names[ControlTypes.NONLINEAR] = 'Nonlinear Normal Modes'
environment_long_names[ControlTypes.MODAL] = "Modal Testing"
environment_long_names[ControlTypes.COMBINED] = "Combined Environments..."

# Add the environment here if it can be used for combined environments
combined_environments_capable = [
    ControlTypes.RANDOM,
    ControlTypes.TRANSIENT,
    ControlTypes.SINE,
    ControlTypes.SDS,
    ControlTypes.TIME,
    ControlTypes.MODAL,
]
