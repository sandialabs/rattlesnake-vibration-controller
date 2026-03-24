# -*- coding: utf-8 -*-
"""
Random Vibration utilities

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
import os
import numpy as np
from scipy.io import loadmat
from rattlesnake.utilities import reduce_array_by_coordinate

_direction_map = {
    "X+": 1,
    "X": 1,
    "+X": 1,
    "Y+": 2,
    "Y": 2,
    "+Y": 2,
    "Z+": 3,
    "Z": 3,
    "+Z": 3,
    "RX+": 4,
    "RX": 4,
    "+RX": 4,
    "RY+": 5,
    "RY": 5,
    "+RY": 5,
    "RZ+": 6,
    "RZ": 6,
    "+RZ": 6,
    "X-": -1,
    "-X": -1,
    "Y-": -2,
    "-Y": -2,
    "Z-": -3,
    "-Z": -3,
    "RX-": -4,
    "-RX": -4,
    "RY-": -5,
    "-RY": -5,
    "RZ-": -6,
    "-RZ": -6,
    "": 0,
}


def load_specification(spec_path, n_freq_lines, df, control_dofs=None):
    """Loads a specification CPSD matrix from a file.

    Parameters
    ----------
    spec_path : str
        Loads the specification contained in this file
    n_freq_lines : int
        The number of frequency lines
    df : float
        The frequency spacing
    control_dofs : list | np.ndarray | None
        1-D array of DOFs to use for control. If coordinates are provided in spec file,
        the cpsd matrix will be indexed to match the provided control DOFs.

    Returns
    -------
    frequency_lines : np.ndarray
        The frequency lines ``df*np.arange(n_freq_lines)``
    cpsd_matrix : np.ndarray
        3D numpy array consisting of a CPSD matrix at each frequency line
    """
    _, extension = os.path.splitext(spec_path)
    if extension.lower() == ".mat":
        data = loadmat(spec_path)
        frequencies = data["f"].squeeze()
        cpsd = data["cpsd"].transpose(2, 0, 1)
        warning_upper = data["warning_upper"].transpose(1, 0) if "warning_upper" in data else None
        warning_lower = data["warning_lower"].transpose(1, 0) if "warning_lower" in data else None
        abort_upper = data["abort_upper"].transpose(1, 0) if "abort_upper" in data else None
        abort_lower = data["abort_lower"].transpose(1, 0) if "abort_lower" in data else None
    elif extension.lower() == ".npz":
        data = np.load(spec_path)
        frequencies = data["f"].squeeze()
        cpsd = data["cpsd"]
        warning_upper = data["warning_upper"] if "warning_upper" in data else None
        warning_lower = data["warning_lower"] if "warning_lower" in data else None
        abort_upper = data["abort_upper"] if "abort_upper" in data else None
        abort_lower = data["abort_lower"] if "abort_lower" in data else None
        coordinate = data["coordinate"] if "coordinate" in data else None
        if coordinate is not None and control_dofs is not None:
            cpsd = reduce_array_by_coordinate(cpsd, coordinate, control_dofs)
            limit_coordinate = coordinate.diagonal().T
            warning_upper = (
                reduce_array_by_coordinate(warning_upper, limit_coordinate, control_dofs)
                if warning_upper is not None
                else None
            )
            warning_lower = (
                reduce_array_by_coordinate(warning_lower, limit_coordinate, control_dofs)
                if warning_lower is not None
                else None
            )
            abort_upper = (
                reduce_array_by_coordinate(abort_upper, limit_coordinate, control_dofs)
                if abort_upper is not None
                else None
            )
            abort_lower = (
                reduce_array_by_coordinate(abort_lower, limit_coordinate, control_dofs)
                if abort_lower is not None
                else None
            )
    else:
        raise ValueError("Invalid File Format")
    # Create the full CPSD matrix
    frequency_lines = df * np.arange(n_freq_lines)
    cpsd_matrix = np.zeros((n_freq_lines,) + cpsd.shape[1:], dtype="complex128")
    warning_matrix = np.empty((2, n_freq_lines, cpsd.shape[-1]), dtype="float64")
    warning_matrix[:] = np.nan
    abort_matrix = np.empty((2, n_freq_lines, cpsd.shape[-1]), dtype="float64")
    abort_matrix[:] = np.nan
    for i, (frequency, cpsd_line) in enumerate(zip(frequencies, cpsd)):
        index = np.argmin(np.abs(frequency - frequency_lines))
        if abs(frequency - frequency_lines[index]) > 1e-5:
            continue
        cpsd_matrix[index, ...] = cpsd_line
        if warning_lower is not None:
            warning_matrix[0, index] = warning_lower[i]
        if warning_upper is not None:
            warning_matrix[1, index] = warning_upper[i]
        if abort_lower is not None:
            abort_matrix[0, index] = abort_lower[i]
        if abort_upper is not None:
            abort_matrix[1, index] = abort_upper[i]
    # Deliever specification to data analysis
    return frequency_lines, cpsd_matrix, warning_matrix, abort_matrix
