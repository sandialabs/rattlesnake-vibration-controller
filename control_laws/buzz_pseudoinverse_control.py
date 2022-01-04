import numpy as np

# Helper functions
def cpsd_coherence(cpsd):
    num = np.abs(cpsd)**2
    den = (cpsd[:,np.newaxis,np.arange(cpsd.shape[1]),np.arange(cpsd.shape[2])]*
           cpsd[:,np.arange(cpsd.shape[1]),np.arange(cpsd.shape[2]),np.newaxis])
    den[den==0.0] = 1 # This prevents divide-by-zero errors from ruining the matrix for frequency lines where the specification is zero
    return np.real(num/
                   den)

def cpsd_phase(cpsd):
    return np.angle(cpsd)

def cpsd_autospectra(cpsd):
    return np.einsum('ijj->ij',cpsd)

def cpsd_from_coh_phs(asd,coh,phs):
    return np.exp(phs*1j)*np.sqrt(coh*asd[:,:,np.newaxis]*asd[:,np.newaxis,:])

def match_coherence_phase(cpsd_to_modify,cpsd_to_match):
    coh = cpsd_coherence(cpsd_to_match)
    phs = cpsd_phase(cpsd_to_match)
    asd = cpsd_autospectra(cpsd_to_modify)
    return cpsd_from_coh_phs(asd,coh,phs)

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