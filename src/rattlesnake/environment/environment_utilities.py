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


class EnvironmentType(Enum):
    """Enumeration of the possible control types"""

    NONE = 0
    RANDOM = 1
    TRANSIENT = 2
    SINE = 3
    TIME = 4
    # NONLINEAR = 5
    MODAL = 6
    SKELETON = 7
    SYSID_SKELETON = 8
    READ = 9
    # Add new environment types here
