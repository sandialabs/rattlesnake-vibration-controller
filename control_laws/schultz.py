# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 09:01:30 2021

@author: dprohe
"""
import numpy as np

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

#======= SCHULTZ'S ATTEMPT AT NEW CONTROL LAW =====================
def match_trace_buzz_pseudoinverse_control(transfer_function,  # Transfer Functions
                          specification, # Specifications
                          buzz_cpsd, # Buzz test in case cross terms are to be computed
                          extra_parameters = '', # example for shape constraints w/ sval threshold would be like 0.10 or similar (ratio of smallest/largest singular value)
                          last_response_cpsd = None, # Last Response for Error Correction
                          last_output_cpsd = None, # Last output for Drive-based control
                          ):
    #----------- SHAPE CONSTRAINT/SINGULAR VALUE REJECTION PIECE ----------------
    try:
        rcond = float(extra_parameters)
    except ValueError:
        rcond = 1e-14
    #-------------- BUZZ PIECE --------------------------------------------------
                # Create a new specification using the autospectra from the original and
    # phase and coherence of the buzz_cpsd
    specBuzz = match_coherence_phase(specification,buzz_cpsd)
    #--------------- PINV AND MATCH TRACE PIECE --------------------------------- 
    # If it's the first time through, do the actual control
    if last_output_cpsd is None:
        # Invert the transfer function using the pseudoinverse with rcond for shape constraint (singular value rejection)
        tf_pinv = np.linalg.pinv(transfer_function,rcond)
        # Return the least squares solution for the new output CPSD
        output = tf_pinv@specBuzz@tf_pinv.conjugate().transpose(0,2,1)                     
    else:
        # Scale the last output cpsd by the trace ratio between spec and last response
        trace_ratio = trace(specification)/trace(last_response_cpsd)
        trace_ratio[np.isnan(trace_ratio)] = 0
        output =  last_output_cpsd*trace_ratio[:,np.newaxis,np.newaxis]
    return output
