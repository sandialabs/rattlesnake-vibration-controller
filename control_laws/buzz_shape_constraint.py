# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 07:50:11 2021

@author: dprohe
"""

import numpy as np

class buzz_shape_constraint_control:
    def __init__(self,specification : np.ndarray,extra_control_parameters : str,
                 transfer_function : np.ndarray = None,  # Transfer Functions
                 buzz_cpsd : np.ndarray = None, # Buzz test in case cross terms are to be computed
                 last_response_cpsd : np.ndarray = None, # Last Response for Error Correction
                 last_output_cpsd : np.ndarray = None, # Last output for Drive-based control
                 ):
        # Store the specification to the class
        self.specification = specification
        self.shape_constraint_threshold = float(extra_control_parameters)
        
    def system_id_update(self,transfer_function : np.ndarray, buzz_cpsd : np.ndarray):
        # Update the specification with the buzz_cpsd
        self.specification = self.match_coherence_phase(self.specification,buzz_cpsd)

    def control(self,transfer_function : np.ndarray,
                last_response_cpsd : np.ndarray = None,
                last_output_cpsd : np.ndarray = None) -> np.ndarray:
        # Perform the control
        
        # Perform SVD on transfer function
        [U,S,Vh] = np.linalg.svd(transfer_function,full_matrices=False)
        V = Vh.conjugate().transpose(0,2,1)
        singular_value_ratios = S/S[:,0,np.newaxis]
        # Determine number of constraint vectors to use
        num_shape_vectors = np.sum(singular_value_ratios >= self.shape_constraint_threshold,axis=1)
        # We have to go into a For Loop here because V changes size on each iteration
        output = np.empty((transfer_function.shape[0],transfer_function.shape[2],transfer_function.shape[2]),dtype=complex)
        for i_f,(V_f,spec_f,H_f,num_shape_vectors_f) in enumerate(zip(V,self.specification,transfer_function,num_shape_vectors)):
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
    
    