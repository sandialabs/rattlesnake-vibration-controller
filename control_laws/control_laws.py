# -*- coding: utf-8 -*-
"""
A collection of several control laws written for the Rattlesnake controller.

Created on Mon Dec 21 16:35:33 2020

@author: dprohe
"""

import numpy as np
from scipy.optimize import nnls

# Helper functions
def cpsd_coherence(cpsd):
    num = np.abs(cpsd)**2
    den = (cpsd[:,np.newaxis,np.arange(cpsd.shape[1]),np.arange(cpsd.shape[2])]*
           cpsd[:,np.arange(cpsd.shape[1]),np.arange(cpsd.shape[2]),np.newaxis])
    den[den==0.0] = 1 # Set to 1
    return np.real(num/
                   den)

def cpsd_phase(cpsd):
    return np.angle(cpsd)

def cpsd_from_coh_phs(asd,coh,phs):
    return np.exp(phs*1j)*np.sqrt(coh*asd[:,:,np.newaxis]*asd[:,np.newaxis,:])

def cpsd_autospectra(cpsd):
    return np.einsum('ijj->ij',cpsd)

def trace(cpsd):
    return np.einsum('ijj->i',cpsd)

def match_coherence_phase(cpsd_original,cpsd_to_match):
    coh = cpsd_coherence(cpsd_to_match)
    phs = cpsd_phase(cpsd_to_match)
    asd = cpsd_autospectra(cpsd_original)
    return cpsd_from_coh_phs(asd,coh,phs)

def uncorrelated(transfer_function,  # Transfer Functions
                 specification, # Specifications
                 buzz_cpsd, # Buzz test in case cross terms are to be computed
                 extra_parameters = '',
                 last_response_cpsd = None, # Last Response for Error Correction
                 last_output_cpsd = None, # Last output for Drive-based control
                 ):
    # Get the Spec from the Autospectrum
    Sxx = cpsd_autospectra(specification).real
    Svv = np.array([nnls(np.abs(this_H)**2,this_sxx)[0] for this_H,this_sxx in zip(transfer_function,Sxx)])
    return cpsd_from_coh_phs(Svv,np.eye(transfer_function.shape[2]),0)

def pseudoinverse_control(transfer_function,  # Transfer Functions
                          specification, # Specifications
                          buzz_cpsd, # Buzz test in case cross terms are to be computed
                          extra_parameters = '',
                          last_response_cpsd = None, # Last Response for Error Correction
                          last_output_cpsd = None, # Last output for Drive-based control
                          ):
    # Invert the transfer function using the pseudoinverse
    tf_pinv = np.linalg.pinv(transfer_function)
    # Return the least squares solution for the new output CPSD
    return tf_pinv@specification@tf_pinv.conjugate().transpose(0,2,1)

def pseudoinverse_control_generator():
    output_cpsd = None
    while True:
        (transfer_function,specification,buzz_cpsd,extra_parameters,
            last_response_cpsd,last_output_cpsd) = yield output_cpsd
         # Invert the transfer function using the pseudoinverse
        tf_pinv = np.linalg.pinv(transfer_function)
        # Assign the output_cpsd so it is yielded next time through the loop
        output_cpsd = tf_pinv@specification@tf_pinv.conjugate().transpose(0,2,1)

class pseudoinverse_control_class:
    def __init__(self,specification : np.ndarray,extra_control_parameters : str,
                 transfer_function : np.ndarray = None,  # Transfer Functions
                 buzz_cpsd : np.ndarray = None, # Buzz test in case cross terms are to be computed
                 last_response_cpsd : np.ndarray = None, # Last Response for Error Correction
                 last_output_cpsd : np.ndarray = None, # Last output for Drive-based control
                 ):
        """
        Initializes the control law

        Parameters
        ----------
        specification : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x control channels representing the specification defined
            by a CPSD matrix
        extra_control_parameters : str
            A string containing any extra parameters that might need to be
            passed to the control law.  This should be parsed by the __init__
            function.
        transfer_function : np.ndarray, optional
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x excitation sources representing the FRF matrix measured
            by the system identification process between drive voltages and
            control response. Will only be passed if the control is switched
            mid-run.  The default is None.
        buzz_cpsd : np.ndarray, optional
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x control channels representing the CPSD matrix measured
            by the system identification process.  Will only be passed if the
            control is switched mid-run.  The default is None.
        last_response_cpsd : np.ndarray, optional
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x control channels representing the last CPSD matrix of 
            control channel responses.  Will only be passed if the
            control is switched mid-run.  The default is None.
        last_output_cpsd : np.ndarray, optional
            A complex 3d numpy ndarray with dimensions frequency lines x drive
            channels x drive channels representing the last CPSD matrix of 
            drive outputs.  Will only be passed if the
            control is switched mid-run.  The default is None.
        """
        self.specification = specification
        
    def system_id_update(self,transfer_function : np.ndarray, buzz_cpsd : np.ndarray):
        """
        Updates the control law with the data from the system identification

        Parameters
        ----------
        transfer_function : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x excitation sources representing the FRF matrix measured
            by the system identification process between drive voltages and
            control response
        buzz_cpsd : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x control channels representing the CPSD matrix measured
            by the system identification process

        """
        pass

    def control(self,transfer_function : np.ndarray,
                last_response_cpsd : np.ndarray = None,
                last_output_cpsd : np.ndarray = None) -> np.ndarray:
        """
        Perform the control operations

        Parameters
        ----------
        transfer_function : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x excitation sources representing the FRF matrix measured
            by the system identification process between drive voltages and
            control response
        last_response_cpsd : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x control channels representing the last CPSD matrix of 
            control channel responses.  If no previous data exists (first time
            through control) it will be None.  The default is None.
        last_output_cpsd : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x drive
            channels x drive channels representing the last CPSD matrix of 
            drive outputs.  If no previous data exists (first time
            through control) it will be None.  The default is None.

        Returns
        -------
        next_output_cpsd : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x drive
            channels x drive channels representing the new CPSD matrix of 
            drive outputs that should be played to the shakers.
        """
        tf_pinv = np.linalg.pinv(transfer_function)
        return tf_pinv@self.specification@tf_pinv.conjugate().transpose(0,2,1)
        

def buzz_control(transfer_function,  # Transfer Functions
                 specification, # Specifications
                 buzz_cpsd, # Buzz test in case cross terms are to be computed
                 extra_parameters = '',
                 last_response_cpsd = None, # Last Response for Error Correction
                 last_output_cpsd = None, # Last output for Drive-based control
                 ):
    # Create a new specification using the autospectra from the original and
    # phase and coherence of the buzz_cpsd
    spec = match_coherence_phase(specification,buzz_cpsd)
    # Invert the transfer function using the pseudoinverse
    tf_pinv = np.linalg.pinv(transfer_function)
    # Return the least squares solution for the new output CPSD
    return tf_pinv@spec@tf_pinv.conjugate().transpose(0,2,1)

def shape_constrained_pseudoinverse(transfer_function,  # Transfer Functions
                                    specification, # Specifications
                                    buzz_cpsd, # Buzz test in case cross terms are to be computed
                                    extra_parameters = '',
                                    last_response_cpsd = None, # Last Response for Error Correction
                                    last_output_cpsd = None, # Last output for Drive-based control
                                    ):
    shape_constraint_threshold = float(extra_parameters)
    # Perform SVD on transfer function
    [U,S,Vh] = np.linalg.svd(transfer_function,full_matrices=False)
    V = Vh.conjugate().transpose(0,2,1)
    singular_value_ratios = S/S[:,0,np.newaxis]
    # Determine number of constraint vectors to use
    num_shape_vectors = np.sum(singular_value_ratios >= shape_constraint_threshold,axis=1)
    # We have to go into a For Loop here because V changes size on each iteration
    output = np.empty((transfer_function.shape[0],transfer_function.shape[2],transfer_function.shape[2]),dtype=complex)
    for i_f,(V_f,spec_f,H_f,num_shape_vectors_f) in enumerate(zip(V,specification,transfer_function,num_shape_vectors)):
        # Form constraint matrix
        constraint_matrix = V_f[:,:num_shape_vectors_f]
        # Constraint FRF matrix
        HC = H_f@constraint_matrix
        HC_pinv = np.linalg.pinv(HC)
        # Estimate inputs (constrained)
        SxxC = HC_pinv@spec_f@HC_pinv.conjugate().T
        # Convert to full inputs
        output[i_f] = constraint_matrix@SxxC@constraint_matrix.conjugate().T
    return output

def match_trace_pseudoinverse(transfer_function,  # Transfer Functions
                              specification, # Specifications
                              buzz_cpsd, # Buzz test in case cross terms are to be computed
                              extra_parameters = '',
                              last_response_cpsd = None, # Last Response for Error Correction
                              last_output_cpsd = None, # Last output for Drive-based control
                              ):
    # If it's the first time through, do the actual control
    if last_output_cpsd is None:
        # Invert the transfer function using the pseudoinverse
        tf_pinv = np.linalg.pinv(transfer_function)
        # Return the least squares solution for the new output CPSD
        output = tf_pinv@specification@tf_pinv.conjugate().transpose(0,2,1)
    else:
        # Scale the last output cpsd by the trace ratio between spec and last response
        trace_ratio = trace(specification)/trace(last_response_cpsd)
        trace_ratio[np.isnan(trace_ratio)] = 0
        output =  last_output_cpsd*trace_ratio[:,np.newaxis,np.newaxis]
    return output

def partitioned_control(transfer_function,  # Transfer Functions
                        specification, # Specifications
                        buzz_cpsd, # Buzz test in case cross terms are to be computed
                        extra_parameters = '',
                        last_response_cpsd = None, # Last Response for Error Correction
                        last_output_cpsd = None, # Last output for Drive-based control
                        ):
    try:
        control_dofs = np.array([int(i) for i in extra_parameters.split(',')])
    except ValueError:
        control_dofs = np.arange(specification.shape[-1])
    spec_partition = specification[...,control_dofs[:,np.newaxis],control_dofs]
    transfer_function_partition = transfer_function[...,control_dofs,:]
    tf_pinv = np.linalg.pinv(transfer_function_partition)
    # Return the least squares solution for the new output CPSD
    return tf_pinv@spec_partition@tf_pinv.conjugate().transpose(0,2,1)