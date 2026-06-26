# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 12:48:43 2023

@author: dprohe
"""

import numpy as np

import matlab.engine

import os

class matlab_control_class:
    def __init__(self,
                 sample_rate,
                 specification_signals, # Signal to try to reproduce
                 output_oversample_factor, # Oversample factor to output
                 extra_parameters, # Extra parameters for the control law
                 frequency_spacing, # Frequency spacing of the system identification
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
        """
        Initializes the control law

        Parameters
        ----------

        Returns
        -------
        None.

        """
        print('Initializing Control Law')
        # Connect to Matlab if possible
        shared_engines = matlab.engine.find_matlab()
        if len(shared_engines) > 0:
            self.engine = matlab.engine.connect_matlab()
        else:
            self.engine = matlab.engine.start_matlab()
        self.script = None
        self.script_path = None
        self.script_matlab_name = None
        self.matlab_function = None
        self.rcond = 1e-15
        self.zero_impulse_after = None
        self.control_indices = slice(None)
        self.extra_parameters = extra_parameters
        self.update_spec = False
        for entry in extra_parameters.split('\n'):
            try:
                field,value = entry.split('::')
                field = field.strip()
                if field == 'rcond':
                    self.rcond = float(value)
                elif field == 'zero_impulse_after':
                    self.zero_impulse_after = float(value)
                elif field == 'script':
                    self.script = value.strip()
                    self.script_path,name = os.path.split(value.strip())
                    self.engine.addpath(self.script_path,nargout=0)
                    self.script_matlab_name = os.path.splitext(name)[0]
                    self.matlab_function = getattr(self.engine,self.script_matlab_name)
                    print('Got matlab script: {:}.\nAdded {:} to Path\nMatlab Name: {:}'.format(self.script,self.script_path,self.script_matlab_name))
                elif field == 'control_indices':
                    self.control_indices = np.array([int(val) for val in value.split(',')])
                elif field == 'update_spec':
                    self.update_spec = value.strip().lower() == 'true'
                else:
                    print('Unrecognized Parameter: {:}'.format(field))
            except ValueError:
                print('Unable to Parse Line {:}, skipping...'.format(entry))
        self.sample_rate = sample_rate
        self.specification_signals = specification_signals
        self.output_oversample_factor = output_oversample_factor
        self.extra_parameters = extra_parameters
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
        """
        Updates the control law with the data from the system identification

        Parameters
        ----------
        transfer_function : np.ndarray
            A complex 3d numpy ndarray with dimensions frequency lines x control
            channels x excitation sources representing the FRF matrix measured
            by the system identification process between drive voltages and
            control response
        """
        print('Performing System ID Update')
        self.frequency_spacing = frequency_spacing
        self.transfer_function = transfer_function
        self.noise_response_cpsd = noise_response_cpsd
        self.noise_reference_cpsd = noise_reference_cpsd
        self.sysid_response_cpsd = sysid_response_cpsd
        self.sysid_reference_cpsd = sysid_reference_cpsd
        self.multiple_coherence = multiple_coherence
        self.frames = frames
        self.total_frames = total_frames

    def control(self,
                last_excitation_signals = None, # Last excitation signal for drive-based control
                last_response_signals = None, # Last response signal for error correction
                ) -> np.ndarray:
        """
        Perform the control operations

        Parameters
        ----------
        last_excitation_signals : np.ndarray, optional
            The most recent output signal, which can be used for error-based
            control. The default is None.
        last_response_signals : np.ndarray, optional
            The most recent responses to the last output signals, which can be
            used for error-based control. The default is None.

        Returns
        -------
        output_signal : np.ndarray
            A 2D numpy array consisting of number of outputs x signal samples *
            output_oversample_factor.  This signal will be played directly to the
            shakers.
        """
        print('Performing Control')
        # We could modify the output signal based on new data that we obtained
        # Otherwise just output the same
        # Prep signals for matlab
        if last_excitation_signals is None:
            matlab_last_excitation = matlab.double(np.array([]))
        else:
            matlab_last_excitation = matlab.double(last_excitation_signals, is_complex = False)
        if last_response_signals is None:
            matlab_last_response = matlab.double(np.array([]))
        else:
            matlab_last_response = matlab.double(last_response_signals, is_complex = False)
        specification_signals = np.asarray(self.matlab_function(
            matlab.double(float(self.sample_rate), is_complex = False),
            matlab.double(self.specification_signals, is_complex = False),
            matlab.int32(np.array(self.output_oversample_factor,dtype='int32'), is_complex = False),
            matlab.double(self.frequency_spacing, is_complex = False),
            matlab.double(self.transfer_function, size = self.transfer_function.shape,is_complex = True),
            matlab.double(self.noise_response_cpsd, size = self.noise_response_cpsd.shape,is_complex = True),
            matlab.double(self.noise_reference_cpsd, size = self.noise_reference_cpsd.shape,is_complex = True),
            matlab.double(self.sysid_response_cpsd, size = self.sysid_response_cpsd.shape,is_complex = True),
            matlab.double(self.sysid_reference_cpsd, size = self.sysid_reference_cpsd.shape,is_complex = True),
            matlab.double(self.multiple_coherence, size = self.multiple_coherence.shape,is_complex = False),
            matlab.int32(np.array(self.frames,dtype='int32'), is_complex = False),
            matlab.int32(np.array(self.total_frames,dtype='int32'), is_complex = False),
            matlab_last_excitation,
            matlab_last_response))
        print('Got Matlab Result: {:}'.format(specification_signals.shape))
        
        # If we're not updating the specification, then we just got the drives, so we return them.
        if not self.update_spec:
            return specification_signals
        
        # Compute impulse responses
        impulse_response = np.fft.irfft(self.transfer_function,axis=0)

        if self.zero_impulse_after is not None:
            # Remove noncausal portion
            impulse_response_abscissa = np.arange(impulse_response.shape[0])/self.sample_rate
            zero_indices = impulse_response_abscissa > self.zero_impulse_after
            impulse_response[zero_indices] = 0
            
        # Zero pad the impulse response to create a signal that is long enough
        added_zeros = np.zeros((specification_signals.shape[-1]-impulse_response.shape[0],) + impulse_response.shape[1:])
        full_impulse_response = np.concatenate((impulse_response,added_zeros),axis=0)

        # Compute FRFs
        interpolated_transfer_function = np.fft.rfft(full_impulse_response,axis=0)
        
        # Convert down to just the desired control degrees of freedom
        specification_signals = specification_signals[self.control_indices]
        interpolated_transfer_function = interpolated_transfer_function[...,self.control_indices,:]
        
        # Perform convolution in frequency domain
        signal_fft = np.fft.rfft(specification_signals,axis=-1)
        inverted_frf = np.linalg.pinv(interpolated_transfer_function,rcond=self.rcond)
        drive_signals_fft = np.einsum('ijk,ki->ij',inverted_frf,signal_fft)

        # Zero pad the FFT to oversample
        drive_signals_fft_zero_padded = np.concatenate((drive_signals_fft[:-1],
            np.zeros((drive_signals_fft[:-1].shape[0]*(self.output_oversample_factor-1)+1,)+drive_signals_fft.shape[1:])),axis=0)

        drive_signals_oversampled = np.fft.irfft(drive_signals_fft_zero_padded.T,axis=-1)*self.output_oversample_factor
        return drive_signals_oversampled