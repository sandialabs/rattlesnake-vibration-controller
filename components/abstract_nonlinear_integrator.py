# -*- coding: utf-8 -*-
"""
Synthetic "hardware" that allows the responses to be simulated by integrating
nonlinear equations of motion.

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

from abc import ABC,abstractmethod

class AbstractNonlinearIntegrator(ABC):
    """Integrator that itegrates general functions of state and force
    
    This class defines a custom nonlinear integrator that will be integrated
    by Rattlesnake.  For linear systems, see the virtual hardware defined by
    state space equations of motion instead.
    """
    
    @abstractmethod
    def eom(self,x,t,f):
        """
        This will define the equations of motion that will be integrated.
        
        Parameters
        ----------
        x : np.ndarray
            The state at a given time
        t : float
            The time step at which the excitation is defined
        f : function f = f(t)
            Function that defines the force f at time t
    
        Returns
        -------
        dx : np.ndarray
            The change in state over time dx/dt
        """
        pass
        
    def ivp_options(self):
        """
        Returns a dictionary that will be passed into the solve_ivp function
        as arguments solve_ivp(..., **option_dict).  This dictionary specifies
        options for the solver.
        
        If this is not defined by inheriting classes, the default parameters
        will be used.
        
        If any parameters defined in this function conflict with those defined
        by Rattlesnake, Rattlesnake will overwrite the user-specified parameters.

        Returns
        -------
        option_dict : dict
            A dictionary of argument, value pairs for the solve_ivp function.

        """
        return dict()
    
    @abstractmethod
    def state_transform(self,state,excitation):
        """
        This function transforms state and excitation values at each time step
        into output values desired by Rattlesnake.  This is the general version
        of the output equation in linear state space matrices
        
        y = Cx + Du
        
        so the results are instead computed by
        
        y = state_transform(x,u)

        Parameters
        ----------
        state : np.ndarray
            A 2D array with number of rows equivalent to the number of state
            degrees of freedom and number of columns equal to the number of
            time steps.
        excitation : np.ndarray
            A 2D array with number of rows equivalent to the number of excitation
            degrees of freedom and number of columns equal to the number of
            time steps.
            
        Returns
        -------
        responses : np.ndarray
            The responses desired by Rattlesnake.

        """
        pass
    
    