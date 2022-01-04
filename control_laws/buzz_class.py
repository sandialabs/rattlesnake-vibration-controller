# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 07:50:11 2021

@author: dprohe
"""

import numpy as np

class buzz_control_class:
    def __init__(self,specification : np.ndarray,extra_control_parameters : str,
                 transfer_function : np.ndarray = None,  # Transfer Functions
                 buzz_cpsd : np.ndarray = None, # Buzz test in case cross terms are to be computed
                 last_response_cpsd : np.ndarray = None, # Last Response for Error Correction
                 last_output_cpsd : np.ndarray = None, # Last output for Drive-based control
                 ):
        # Store the specification to the class
        if buzz_cpsd is None: # If it's the first time through we won't have a buzz test yet
            self.specification = specification
        else: # Otherwise we can compute the modified spec right away
            self.specification = self.match_coherence_phase(specification, buzz_cpsd)
            
    def system_id_update(self,transfer_function : np.ndarray, buzz_cpsd : np.ndarray):
        # Update the specification with the buzz_cpsd
        self.specification = self.match_coherence_phase(self.specification,buzz_cpsd)

    def control(self,transfer_function : np.ndarray,
                last_response_cpsd : np.ndarray = None,
                last_output_cpsd : np.ndarray = None) -> np.ndarray:
        # Perform the control
        tf_pinv = np.linalg.pinv(transfer_function)
        return tf_pinv@self.specification@tf_pinv.conjugate().transpose(0,2,1)
        
    def cpsd_coherence(self,cpsd):
        num = np.abs(cpsd)**2
        den = (cpsd[:,np.newaxis,np.arange(cpsd.shape[1]),np.arange(cpsd.shape[2])]*
               cpsd[:,np.arange(cpsd.shape[1]),np.arange(cpsd.shape[2]),np.newaxis])
        den[den==0.0] = 1 # Set to 1
        return np.real(num/
                       den)
    
    def cpsd_phase(self,cpsd):
        return np.angle(cpsd)
    
    def cpsd_from_coh_phs(self,asd,coh,phs):
        return np.exp(phs*1j)*np.sqrt(coh*asd[:,:,np.newaxis]*asd[:,np.newaxis,:])
    
    def cpsd_autospectra(self,cpsd):
        return np.einsum('ijj->ij',cpsd)
    
    def match_coherence_phase(self,cpsd_original,cpsd_to_match):
        coh = self.cpsd_coherence(cpsd_to_match)
        phs = self.cpsd_phase(cpsd_to_match)
        asd = self.cpsd_autospectra(cpsd_original)
        return self.cpsd_from_coh_phs(asd,coh,phs)
    
    