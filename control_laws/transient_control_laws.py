# -*- coding: utf-8 -*-
"""
A collection of several transient control laws written for the MIMO transient
module in Rattlesnake

Created on Tue May  4 16:37:10 2021

@author: dprohe
"""

import numpy as np

def pseudoinverse_control(
        sample_rate,
        specification_signals, # Signal to try to reproduce
        frequency_spacing,
        transfer_function, # Transfer Functions
        noise_response_cpsd,  # Noise levels and correlation 
        noise_reference_cpsd, # from the system identification
        sysid_response_cpsd,  # Response levels and correlation
        sysid_reference_cpsd, # from the system identification
        multiple_coherence, # Coherence from the system identification
        frames, # Number of frames in the CPSD and FRF matrices
        total_frames, # Total frames that could be in the CPSD and FRF matrices
        output_oversample_factor, # Oversample factor to output
        extra_parameters = '', # Extra parameters for the control law
        last_excitation_signals = None, # Last excitation signal for drive-based control
        last_response_signals = None, # Last response signal for error correction
        ):
    # Parse the input arguments in extra_parameters
    rcond = 1e-15
    zero_impulse_after = None
    # Split it up into lines
    for entry in extra_parameters.split('\n'):
        try:
            # For each entry, split the key from the value using the colon
            field,value = entry.split(':')
            # Strip any whitespace
            field = field.strip()
            # Check the field to figure out which value to assign
            if field == 'rcond':
                rcond = float(value)
            elif field == 'zero_impulse_after':
                zero_impulse_after = float(value)
            else:
                # Report if we cannot understand the parameter
                print('Unrecognized Parameter: {:}, skipping...'.format(field))
        except ValueError:
            # Report if we cannot parse the line
            print('Unable to Parse Line {:}, skipping...'.format(entry))

    # Compute impulse responses using the IFFT of the transfer function
    # We will zero pad the IFFT to do interpolation in the frequency domain
    # to match the length of the required signal
    impulse_response = np.fft.irfft(transfer_function,axis=0)

    # The impulse response should be going to zero at the end of the frame,
    # but practically there may be some gibbs phenomenon effects that make the
    # impulse response noncausal.  If we zero pad, this might be wrong.  We
    # therefore give the use the ability to zero out this non-causal poriton of
    # the impulse response.
    if zero_impulse_after is not None:
        # Remove noncausal portion
        impulse_response_abscissa = np.arange(impulse_response.shape[0])/sample_rate
        zero_indices = impulse_response_abscissa > zero_impulse_after
        impulse_response[zero_indices] = 0
        
    # Zero pad the impulse response to create a signal that is long enough for
    # the specification signal
    added_zeros = np.zeros((specification_signals.shape[-1]-impulse_response.shape[0],) 
                           + impulse_response.shape[1:])
    full_impulse_response = np.concatenate((impulse_response,added_zeros),axis=0)

    # Compute FRFs using the FFT from the impulse response.  This is now
    # interpolated such that it matches the frequency spacing of the specification
    # signal
    interpolated_transfer_function = np.fft.rfft(full_impulse_response,axis=0)

    # Perform convolution by frequency domain multiplication
    signal_fft = np.fft.rfft(specification_signals,axis=-1)
    # Invert the FRF matrix using the specified rcond parameter
    inverted_frf = np.linalg.pinv(interpolated_transfer_function,rcond=rcond)
    # Multiply the inverted FRFs by the response spectra to get the drive spectra
    drive_signals_fft = np.einsum('ijk,ki->ij',inverted_frf,signal_fft)

    # Zero pad the drive FFT to oversample to the output_oversample_factor
    drive_signals_fft_zero_padded = np.concatenate((drive_signals_fft[:-1],
        np.zeros((drive_signals_fft[:-1].shape[0]*(output_oversample_factor-1)+1,)
                 +drive_signals_fft.shape[1:])),axis=0)

    # Finally, take the IFFT to get the time domain signal.  We need to scale
    # by the output_oversample_factor due to how the IFFT is normalized.
    drive_signals_oversampled = np.fft.irfft(
        drive_signals_fft_zero_padded.T,axis=-1)*output_oversample_factor
    return drive_signals_oversampled

def pseudoinverse_control_generator():
    signal_fft = None
    inverted_frf = None
    drive_signals_oversampled = None
    while True:
        (sample_rate,
         specification_signals, # Signal to try to reproduce
         frequency_spacing,
         transfer_function, # Transfer Functions
         noise_response_cpsd,  # Noise levels and correlation 
         noise_reference_cpsd, # from the system identification
         sysid_response_cpsd,  # Response levels and correlation
         sysid_reference_cpsd, # from the system identification
         multiple_coherence, # Coherence from the system identification
         frames, # Number of frames in the CPSD and FRF matrices
         total_frames, # Total frames that could be in the CPSD and FRF matrices
         output_oversample_factor, # Oversample factor to output
         extra_parameters, # Extra parameters for the control law
         last_excitation_signals, # Last excitation signal for drive-based control
         last_response_signals, # Last response signal for error correction
         ) = yield drive_signals_oversampled
        if signal_fft is None:
            # Compute the FFT of the spec if it hasn't been done yet
            signal_fft = np.fft.rfft(specification_signals).T
        # Get a tolerance if specified
        rcond = 1e-15
        zero_impulse_after = None
        for entry in extra_parameters.split('\n'):
            field,value = entry.split(':')
            field = field.strip()
            if field == 'rcond':
                rcond = float(value)
            elif field == 'zero_impulse_after':
                zero_impulse_after = float(value)
            else:
                print('Unrecognized Parameter: {:}'.format(field))
        if inverted_frf is None:
            # Compute impulse responses
            impulse_response = np.fft.irfft(transfer_function,axis=0)

            if zero_impulse_after is not None:
                # Remove noncausal portion
                impulse_response_abscissa = np.arange(impulse_response.shape[0])/sample_rate
                zero_indices = impulse_response_abscissa > zero_impulse_after
                impulse_response[zero_indices] = 0
                
            # Zero pad the impulse response to create a signal that is long enough
            added_zeros = np.zeros((specification_signals.shape[-1]-impulse_response.shape[0],) + impulse_response.shape[1:])
            full_impulse_response = np.concatenate((impulse_response,added_zeros),axis=0)

            # Compute FRFs
            interpolated_transfer_function = np.fft.rfft(full_impulse_response,axis=0)

            # Perform convolution in frequency domain
            inverted_frf = np.linalg.pinv(interpolated_transfer_function,rcond=rcond)
            
        drive_signals_fft = np.einsum('ijk,ki->ij',inverted_frf,signal_fft)

        # Zero pad the FFT to oversample
        drive_signals_fft_zero_padded = np.concatenate((drive_signals_fft[:-1],
            np.zeros((drive_signals_fft[:-1].shape[0]*(output_oversample_factor-1)+1,)+drive_signals_fft.shape[1:])),axis=0)

        drive_signals_oversampled = np.fft.irfft(drive_signals_fft_zero_padded.T,axis=-1)*output_oversample_factor

class pseudoinverse_control_class:
    def __init__(self,
                 sample_rate,
                 specification_signals, # Signal to try to reproduce
                 output_oversample_factor, # Oversample factor to output
                 extra_parameters, # Extra parameters for the control law
                 frequency_spacing,
                 transfer_function, # Transfer Functions
                 noise_response_cpsd,  # Noise levels and correlation 
                 noise_reference_cpsd, # from the system identification
                 sysid_response_cpsd,  # Response levels and correlation
                 sysid_reference_cpsd, # from the system identification
                 multiple_coherence, # Coherence from the system identification
                 frames, # Number of frames in the CPSD and FRF matrices
                 total_frames, # Total frames that could be in the CPSD and FRF matrices
                 last_excitation_signals = None, # Last excitation signal for drive-based control
                 last_response_signals = None, # Last response signal for error correction
                 ):
        self.rcond = 1e-15
        self.zero_impulse_after = None
        for entry in extra_parameters.split('\n'):
            field,value = entry.split(':')
            field = field.strip()
            if field == 'rcond':
                self.rcond = float(value)
            elif field == 'zero_impulse_after':
                self.zero_impulse_after = float(value)
            else:
                print('Unrecognized Parameter: {:}'.format(field))
        self.sample_rate = sample_rate
        self.specification_signals = specification_signals
        self.signal_fft = np.fft.rfft(specification_signals).T
        if self.transfer_function is not None:
            self.system_id_update(
                frequency_spacing,
                transfer_function, # Transfer Functions
                noise_response_cpsd,  # Noise levels and correlation 
                noise_reference_cpsd, # from the system identification
                sysid_response_cpsd,  # Response levels and correlation
                sysid_reference_cpsd, # from the system identification
                multiple_coherence, # Coherence from the system identification
                frames, # Number of frames in the CPSD and FRF matrices
                total_frames, # Total frames that could be in the CPSD and FRF matrices
                )
        
    def system_id_update(self,
                         frequency_spacing,
                         transfer_function, # Transfer Functions
                         noise_response_cpsd,  # Noise levels and correlation 
                         noise_reference_cpsd, # from the system identification
                         sysid_response_cpsd,  # Response levels and correlation
                         sysid_reference_cpsd, # from the system identification
                         multiple_coherence, # Coherence from the system identification
                         frames, # Number of frames in the CPSD and FRF matrices
                         total_frames, # Total frames that could be in the CPSD and FRF matrices
                         ):
        # Compute impulse responses
        impulse_response = np.fft.irfft(transfer_function,axis=0)

        if self.zero_impulse_after is not None:
            # Remove noncausal portion
            impulse_response_abscissa = np.arange(impulse_response.shape[0])/self.sample_rate
            zero_indices = impulse_response_abscissa > self.zero_impulse_after
            impulse_response[zero_indices] = 0
            
        # Zero pad the impulse response to create a signal that is long enough
        added_zeros = np.zeros((self.specification_signals.shape[-1]-impulse_response.shape[0],) 
                               + impulse_response.shape[1:])
        full_impulse_response = np.concatenate((impulse_response,added_zeros),axis=0)

        # Compute FRFs
        interpolated_transfer_function = np.fft.rfft(full_impulse_response,axis=0)

        # Perform convolution in frequency domain
        self.inverted_frf = np.linalg.pinv(interpolated_transfer_function,rcond=self.rcond)
        
        drive_signals_fft = np.einsum('ijk,ki->ij',self.inverted_frf,self.signal_fft)

        # Zero pad the FFT to oversample
        drive_signals_fft_zero_padded = np.concatenate((drive_signals_fft[:-1],
            np.zeros((drive_signals_fft[:-1].shape[0]*(self.output_oversample_factor-1)+1,)
                     +drive_signals_fft.shape[1:])),axis=0)

        self.drive_signals_oversampled = np.fft.irfft(
            drive_signals_fft_zero_padded.T,axis=-1)*self.output_oversample_factor

    def control(self,
                last_excitation_signals = None, # Last excitation signal for drive-based control
                last_response_signals = None, # Last response signal for error correction
                ) -> np.ndarray:       
        # We could modify the output signal based on new data that we obtained
        # Otherwise just output the same
        
        return self.drive_signals_oversampled