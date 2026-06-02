import numpy as np

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

def match_coherence_phase(cpsd_original,cpsd_to_match):
    coh = cpsd_coherence(cpsd_to_match)
    phs = cpsd_phase(cpsd_to_match)
    asd = cpsd_autospectra(cpsd_original)
    return cpsd_from_coh_phs(asd,coh,phs)

def trace(cpsd):
    return np.einsum('ijj->i',cpsd)

def pseudoinverse_control(specification, # Specifications
                          warning_levels, # Warning levels
                          abort_levels, # Abort Levels
                          transfer_function,  # Transfer Functions
                          noise_response_cpsd,  # Noise levels and correlation 
                          noise_reference_cpsd, # from the system identification
                          sysid_response_cpsd,  # Response levels and correlation
                          sysid_reference_cpsd, # from the system identification
                          multiple_coherence, # Coherence from the system identification
                          frames, # Number of frames in the CPSD and FRF matrices
                          total_frames, # Total frames that could be in the CPSD and FRF matrices
                          extra_parameters = '', # Extra parameters for the control law
                          last_response_cpsd = None, # Last Control Response for Error Correction
                          last_output_cpsd = None, # Last Control Excitation for Drive-based control
                          ):
    """
    A control law that simply performs a pseudoinverse on the transfer function
    matrix and pre- and post-multiplies the specification by that inverse via
    the formula Gvv = H^+ Gxx (H^*)^+.
    
    Parameters
    ----------
    specification : np.ndarray
        The response specification that the control law will attempt to achieve.
        Shape is (num_frequencies x num_control_channels x num_control_channels).
    warning_levels : np.ndarray
        The warning levels provided by the specification where the control will
        notify the user if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    abort_levels : np.ndarray
        The abort levels provided by the specification where the control will
        shut down if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    transfer_function : np.ndarray
        The system transfer function between the excitation voltage and the
        control channel responses.  Shape is (num_frequencies x 
        num_control_channels x num_excitation_channels)
    noise_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the response coordinates.  Shape
        is (num_frequencies x num_control_channels x num_control_channels).
    noise_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the reference coordinates.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    sysid_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        response coordinates for the transfer function calculation.  Can also
        be used to provide "preferred" relationships between the responses for
        uncorrelated inputs.  Shape is (num_frequencies x num_control_channels
        x num_control_channels).
    sysid_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        reference coordinates for the transfer function calculation.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    multiple_coherence : np.ndarray
        Multiple coherence function which shows how the measured responses are
        related to the measured excitation signals.  Multiple coherence will be
        1 if the measured responses are completely due to the input signals and
        0 if the measured responses are not related to the input signals at all.
        Can be used to determine which frequency lines are most accurately
        computed in the transfer function.
    frames : int
        Specifies the number of measurement frames used to compute the current
        system identification estimates.
    total_frames : int
        Specifies the number of frames specified to be used in the system
        identification estimate.
    extra_parameters : str, optional
        A string containing any optional parameters the control law may need to
        use. It is up to the control law to parse this string to extract the
        required information that it needs.  The default is ''.
    last_response_cpsd : np.ndarray, optional
        The CPSD measured from the control channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        response coordinates or to provide error-based control by comparing the
        achieved responses against the desired specification.  Shape is 
        (num_frequencies x num_control_channels x num_control_channels).
        If it is the first time through the control, and there is no previously
        measured response, this will be None.
    last_output_cpsd : np.ndarray, optional
        The CPSD measured from the excitation channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        reference coordinates or to provide drive-based control.  Shape is 
        (num_frequencies x num_excitation_channels x num_excitation_channels).
        If it is the first time through the control, and there is no previously
        measured excitation, this will be None.
    
    Returns
    -------
    np.ndarray
        The output CPSD matrix with shape
        (num_frequencies x num_excitation_channels x num_excitation_channels)
    
    """
    try:
        rcond = float(extra_parameters)
    except ValueError:
        rcond = 1e-15
    # Invert the transfer function using the pseudoinverse
    tf_pinv = np.linalg.pinv(transfer_function,rcond)
    # Return the least squares solution for the new output CPSD
    output = tf_pinv@specification@tf_pinv.conjugate().transpose(0,2,1)
    return output

def match_trace_pseudoinverse(specification, # Specifications
                              warning_levels, # Warning levels
                              abort_levels, # Abort Levels
                              transfer_function,  # Transfer Functions
                              noise_response_cpsd,  # Noise levels and correlation 
                              noise_reference_cpsd, # from the system identification
                              sysid_response_cpsd,  # Response levels and correlation
                              sysid_reference_cpsd, # from the system identification
                              multiple_coherence, # Coherence from the system identification
                              frames, # Number of frames in the CPSD and FRF matrices
                              total_frames, # Total frames that could be in the CPSD and FRF matrices
                              extra_parameters = '', # Extra parameters for the control law
                              last_response_cpsd = None, # Last Control Response for Error Correction
                              last_output_cpsd = None, # Last Control Excitation for Drive-based control
                              ):
    """
    A control law that initially performs a pseudoinverse on the transfer function
    matrix and pre- and post-multiplies the updated specification by that inverse
    via the formula Gvv = H^+ Gxx (H^*)^+.  On subsequent iterations, it will scale
    the output at each frequency line up or down depending on if the frequency line
    is on average higher or low.  This is equivalent to matching the "trace" (sum of
    the diagonal) of the CPSD specification in a closed-loop fashion.
    
    Parameters
    ----------
    specification : np.ndarray
        The response specification that the control law will attempt to achieve.
        Shape is (num_frequencies x num_control_channels x num_control_channels).
    warning_levels : np.ndarray
        The warning levels provided by the specification where the control will
        notify the user if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    abort_levels : np.ndarray
        The abort levels provided by the specification where the control will
        shut down if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    transfer_function : np.ndarray
        The system transfer function between the excitation voltage and the
        control channel responses.  Shape is (num_frequencies x 
        num_control_channels x num_excitation_channels)
    noise_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the response coordinates.  Shape
        is (num_frequencies x num_control_channels x num_control_channels).
    noise_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the reference coordinates.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    sysid_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        response coordinates for the transfer function calculation.  Can also
        be used to provide "preferred" relationships between the responses for
        uncorrelated inputs.  Shape is (num_frequencies x num_control_channels
        x num_control_channels).
    sysid_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        reference coordinates for the transfer function calculation.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    multiple_coherence : np.ndarray
        Multiple coherence function which shows how the measured responses are
        related to the measured excitation signals.  Multiple coherence will be
        1 if the measured responses are completely due to the input signals and
        0 if the measured responses are not related to the input signals at all.
        Can be used to determine which frequency lines are most accurately
        computed in the transfer function.
    frames : int
        Specifies the number of measurement frames used to compute the current
        system identification estimates.
    total_frames : int
        Specifies the number of frames specified to be used in the system
        identification estimate.
    extra_parameters : str, optional
        A string containing any optional parameters the control law may need to
        use. It is up to the control law to parse this string to extract the
        required information that it needs.  The default is ''.
    last_response_cpsd : np.ndarray, optional
        The CPSD measured from the control channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        response coordinates or to provide error-based control by comparing the
        achieved responses against the desired specification.  Shape is 
        (num_frequencies x num_control_channels x num_control_channels).
        If it is the first time through the control, and there is no previously
        measured response, this will be None.
    last_output_cpsd : np.ndarray, optional
        The CPSD measured from the excitation channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        reference coordinates or to provide drive-based control.  Shape is 
        (num_frequencies x num_excitation_channels x num_excitation_channels).
        If it is the first time through the control, and there is no previously
        measured excitation, this will be None.
    
    Returns
    -------
    np.ndarray
        The output CPSD matrix with shape
        (num_frequencies x num_excitation_channels x num_excitation_channels)
    
    """
    try:
        rcond = float(extra_parameters)
    except ValueError:
        rcond = 1e-15
    # If it's the first time through, do the actual control
    if last_output_cpsd is None:
        # Invert the transfer function using the pseudoinverse
        tf_pinv = np.linalg.pinv(transfer_function,rcond)
        # Return the least squares solution for the new output CPSD
        output = tf_pinv@specification@tf_pinv.conjugate().transpose(0,2,1)
    else:
        # Scale the last output cpsd by the trace ratio between spec and last response
        trace_ratio = trace(specification)/trace(last_response_cpsd)
        trace_ratio[np.isnan(trace_ratio)] = 0
        output =  last_output_cpsd*trace_ratio[:,np.newaxis,np.newaxis]
    return output

def buzz_control(specification, # Specifications
                 warning_levels, # Warning levels
                 abort_levels, # Abort Levels
                 transfer_function,  # Transfer Functions
                 noise_response_cpsd,  # Noise levels and correlation 
                 noise_reference_cpsd, # from the system identification
                 sysid_response_cpsd,  # Response levels and correlation
                 sysid_reference_cpsd, # from the system identification
                 multiple_coherence, # Coherence from the system identification
                 frames, # Number of frames in the CPSD and FRF matrices
                 total_frames, # Total frames that could be in the CPSD and FRF matrices
                 extra_parameters = '', # Extra parameters for the control law
                 last_response_cpsd = None, # Last Control Response for Error Correction
                 last_output_cpsd = None, # Last Control Excitation for Drive-based control
                 ):
    """
    A control law that updates the coherence and phase of the specification
    with the coherence and phase derived from the system identification phase.
    It then simply performs a pseudoinverse on the transfer function
    matrix and pre- and post-multiplies the updated specification by that inverse
    via the formula Gvv = H^+ Gxx (H^*)^+.
    
    Parameters
    ----------
    specification : np.ndarray
        The response specification that the control law will attempt to achieve.
        Shape is (num_frequencies x num_control_channels x num_control_channels).
    warning_levels : np.ndarray
        The warning levels provided by the specification where the control will
        notify the user if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    abort_levels : np.ndarray
        The abort levels provided by the specification where the control will
        shut down if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    transfer_function : np.ndarray
        The system transfer function between the excitation voltage and the
        control channel responses.  Shape is (num_frequencies x 
        num_control_channels x num_excitation_channels)
    noise_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the response coordinates.  Shape
        is (num_frequencies x num_control_channels x num_control_channels).
    noise_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the reference coordinates.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    sysid_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        response coordinates for the transfer function calculation.  Can also
        be used to provide "preferred" relationships between the responses for
        uncorrelated inputs.  Shape is (num_frequencies x num_control_channels
        x num_control_channels).
    sysid_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        reference coordinates for the transfer function calculation.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    multiple_coherence : np.ndarray
        Multiple coherence function which shows how the measured responses are
        related to the measured excitation signals.  Multiple coherence will be
        1 if the measured responses are completely due to the input signals and
        0 if the measured responses are not related to the input signals at all.
        Can be used to determine which frequency lines are most accurately
        computed in the transfer function.
    frames : int
        Specifies the number of measurement frames used to compute the current
        system identification estimates.
    total_frames : int
        Specifies the number of frames specified to be used in the system
        identification estimate.
    extra_parameters : str, optional
        A string containing any optional parameters the control law may need to
        use. It is up to the control law to parse this string to extract the
        required information that it needs.  The default is ''.
    last_response_cpsd : np.ndarray, optional
        The CPSD measured from the control channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        response coordinates or to provide error-based control by comparing the
        achieved responses against the desired specification.  Shape is 
        (num_frequencies x num_control_channels x num_control_channels).
        If it is the first time through the control, and there is no previously
        measured response, this will be None.
    last_output_cpsd : np.ndarray, optional
        The CPSD measured from the excitation channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        reference coordinates or to provide drive-based control.  Shape is 
        (num_frequencies x num_excitation_channels x num_excitation_channels).
        If it is the first time through the control, and there is no previously
        measured excitation, this will be None.
    
    Returns
    -------
    np.ndarray
        The output CPSD matrix with shape
        (num_frequencies x num_excitation_channels x num_excitation_channels)
    
    """
    try:
        rcond = float(extra_parameters)
    except ValueError:
        rcond = 1e-15
    # Create a new specification using the autospectra from the original and
    # phase and coherence of the buzz_cpsd
    modified_spec = match_coherence_phase(specification,sysid_response_cpsd)
    # Invert the transfer function using the pseudoinverse
    tf_pinv = np.linalg.pinv(transfer_function,rcond)
    # Return the least squares solution for the new output CPSD
    return tf_pinv@modified_spec@tf_pinv.conjugate().transpose(0,2,1)

def buzz_control_generator():
    output_cpsd = None
    modified_spec = None
    while True:
        (specification, # Specifications
         warning_levels, # Warning levels
         abort_levels, # Abort Levels
         transfer_function,  # Transfer Functions
         noise_response_cpsd,  # Noise levels and correlation 
         noise_reference_cpsd, # from the system identification
         sysid_response_cpsd,  # Response levels and correlation
         sysid_reference_cpsd, # from the system identification
         multiple_coherence, # Coherence from the system identification
         frames, # Number of frames in the CPSD and FRF matrices
         total_frames, # Total frames that could be in the CPSD and FRF matrices
         extra_parameters, # Extra parameters for the control law
         last_response_cpsd, # Last Control Response for Error Correction
         last_output_cpsd, # Last Control Excitation for Drive-based control
            ) = yield output_cpsd
        # Only comput the modified spec if it hasn't been yet.
        if modified_spec is None:
            modified_spec = match_coherence_phase(specification,sysid_response_cpsd)
         # Invert the transfer function using the pseudoinverse
        tf_pinv = np.linalg.pinv(transfer_function)
        # Assign the output_cpsd so it is yielded next time through the loop
        output_cpsd = tf_pinv@modified_spec@tf_pinv.conjugate().transpose(0,2,1)

class buzz_control_class:
    def __init__(self,
                 specification : np.ndarray, # Specifications
                 warning_levels  : np.ndarray, # Warning levels
                 abort_levels  : np.ndarray, # Abort Levels
                 extra_parameters : str, # Extra parameters for the control law
                 transfer_function : np.ndarray = None,  # Transfer Functions
                 noise_response_cpsd : np.ndarray = None,  # Noise levels and correlation 
                 noise_reference_cpsd : np.ndarray = None, # from the system identification
                 sysid_response_cpsd : np.ndarray = None,  # Response levels and correlation
                 sysid_reference_cpsd : np.ndarray = None, # from the system identification
                 multiple_coherence : np.ndarray = None, # Coherence from the system identification
                 frames = None, # Number of frames in the CPSD and FRF matrices
                 total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                 last_response_cpsd : np.ndarray = None, # Last Control Response for Error Correction
                 last_output_cpsd : np.ndarray = None, # Last Control Excitation for Drive-based control
                 ):
        # Store the specification to the class
        if sysid_response_cpsd is None: # If it's the first time through we won't have a buzz test yet
            self.specification = specification
        else: # Otherwise we can compute the modified spec right away
            self.specification = self.match_coherence_phase(specification, sysid_response_cpsd)
            
    def system_id_update(self,
                         transfer_function : np.ndarray = None,  # Transfer Functions
                         noise_response_cpsd : np.ndarray = None,  # Noise levels and correlation 
                         noise_reference_cpsd : np.ndarray = None, # from the system identification
                         sysid_response_cpsd : np.ndarray = None,  # Response levels and correlation
                         sysid_reference_cpsd : np.ndarray = None, # from the system identification
                         multiple_coherence : np.ndarray = None, # Coherence from the system identification
                         frames = None, # Number of frames in the CPSD and FRF matrices
                         total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                         ):
        # Update the specification with the buzz_cpsd
        self.specification = self.match_coherence_phase(self.specification,sysid_response_cpsd)

    def control(self,
                transfer_function : np.ndarray = None,  # Transfer Functions
                multiple_coherence : np.ndarray = None, # Coherence from the system identification
                frames = None, # Number of frames in the CPSD and FRF matrices
                total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                last_response_cpsd : np.ndarray = None, # Last Control Response for Error Correction
                last_output_cpsd : np.ndarray = None) -> np.ndarray:
        # Perform the control
        tf_pinv = np.linalg.pinv(transfer_function)
        return tf_pinv @ self.specification @ tf_pinv.conjugate().transpose(0,2,1)
        
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
