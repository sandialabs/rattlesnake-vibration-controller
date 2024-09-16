# -*- coding: utf-8 -*-
"""
Utilities for the random vibration environment

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

def load_specification(spec_path,n_freq_lines,df):
    """Loads a specification CPSD matrix from a file.

    Parameters
    ----------
    spec_path : str
        Loads the specification contained in this file
    n_freq_lines : int
        The number of frequency lines 
    df : float
        The frequency spacing

    Returns
    -------
    frequency_lines : np.ndarray
        The frequency lines ``df*np.arange(n_freq_lines)``
    cpsd_matrix : np.ndarray
        3D numpy array consisting of a CPSD matrix at each frequency line
    """
    file_base,extension = os.path.splitext(spec_path)
    if extension.lower() == '.mat':
        data = loadmat(spec_path)
        frequencies = data['f'].squeeze()
        cpsd = data['cpsd'].transpose(2,0,1)
        warning_upper = data['warning_upper'].transpose(1,0) if 'warning_upper' in data else None
        warning_lower = data['warning_lower'].transpose(1,0) if 'warning_lower' in data else None
        abort_upper = data['abort_upper'].transpose(1,0) if 'abort_upper' in data else None
        abort_lower = data['abort_lower'].transpose(1,0) if 'abort_lower' in data else None
    elif extension.lower() == '.npz':
        data = np.load(spec_path)
        frequencies = data['f'].squeeze()
        cpsd = data['cpsd']
        warning_upper = data['warning_upper'] if 'warning_upper' in data else None
        warning_lower = data['warning_lower'] if 'warning_lower' in data else None
        abort_upper = data['abort_upper'] if 'abort_upper' in data else None
        abort_lower = data['abort_lower'] if 'abort_lower' in data else None
    
    # Create the full CPSD matrix
    frequency_lines = df*np.arange(n_freq_lines)
    cpsd_matrix = np.zeros((n_freq_lines,)+cpsd.shape[1:],dtype='complex128')
    warning_matrix = np.empty((2,n_freq_lines,cpsd.shape[-1]),dtype='float64')
    warning_matrix[:] = np.nan
    abort_matrix = np.empty((2,n_freq_lines,cpsd.shape[-1]),dtype='float64')
    abort_matrix[:] = np.nan
    for i,(frequency,cpsd_line) in enumerate(zip(frequencies,cpsd)):
        index = np.argmin(np.abs(frequency-frequency_lines))
        if abs(frequency-frequency_lines[index]) > 1e-5:
            #raise ValueError('Frequency {:} not a valid frequency ({:} closest)'.format(frequency,frequency_lines[index]))
            continue
        cpsd_matrix[index,...] = cpsd_line
        if not warning_lower is None:
            warning_matrix[0,index] = warning_lower[i]
        if not warning_upper is None:
            warning_matrix[1,index] = warning_upper[i]
        if not abort_lower is None:
            abort_matrix[0,index] = abort_lower[i]
        if not abort_upper is None:
            abort_matrix[1,index] = abort_upper[i]
    # Deliever specification to data analysis
    return frequency_lines,cpsd_matrix,warning_matrix,abort_matrix