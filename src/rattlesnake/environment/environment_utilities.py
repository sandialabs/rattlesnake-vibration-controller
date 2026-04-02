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
    pass


class EnvironmentType(Enum):
    """Enumeration of the possible control types"""

    RANDOM = 1
    TRANSIENT = 2
    SINE = 3
    TIME = 4
    # NONLINEAR = 5
    MODAL = 6
    # Add new environment types here


# Name for each environment
environment_long_names = {}
environment_long_names[EnvironmentType.RANDOM] = "MIMO Random Vibration"
environment_long_names[EnvironmentType.TRANSIENT] = "MIMO Transient"
environment_long_names[EnvironmentType.SINE] = "MIMO Sine Vibration"
environment_long_names[EnvironmentType.TIME] = "Time Signal Generation"
# environment_long_names[ControlTypes.NONLINEAR] = 'Nonlinear Normal Modes'
environment_long_names[EnvironmentType.MODAL] = "Modal Testing"

# Add the environment here if it can be used for combined environments
combined_environments_capable = [
    EnvironmentType.RANDOM,
    EnvironmentType.TRANSIENT,
    EnvironmentType.SINE,
    EnvironmentType.TIME,
    EnvironmentType.MODAL,
]
