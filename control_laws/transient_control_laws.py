# -*- coding: utf-8 -*-
"""
A collection of several transient control laws written for the MIMO transient
module in Rattlesnake

Created on Tue May  4 16:37:10 2021

@author: dprohe
"""

import numpy as np

def pseudoinverse_control(transfer_function, # Transfer Functions
                          specification_signals, # Signal to try to reproduce
                          output_oversample_factor, # Factor by which the output should be oversampled
                          extra_parameters = '', # Any extra parametrs the controller might need
                          last_excitation_signals = None, # Last excitation signal for drive-based control
                          last_response_signals = None, # Last response signal for error correction
                          ):
    """
    Perform a simple pseudoinverse of the transfer function matrix.

    Parameters
    ----------
    transfer_function : np.ndarray
         A complex 3d numpy ndarray with dimensions frequency lines x control
        channels x excitation sources representing the FRF matrix measured
        by the system identification process between drive voltages and
        control response
    specification_signals : np.ndarray
        A real 2d numpy ndarray with dimensions control
        channels x signal samples representing the specification defined
        by one time history for each control channel
    output_oversample_factor : int
        The oversample factor on the output compared to the input.  I.e the
        output sample rate is output_oversample_factor * the acquisition sample
        rate.
    extra_parameters : str
        A string containing any extra parameters that might need to be
        passed to the control law.  This should be parsed by the function to
        produce meaningful data.
    last_excitation_signals : np.ndarray, optional
        The last excitation signal played to the shakers. The default is None.
    last_response_signals : np.ndarray, optional
        The last response signals measured by the controller. The default is None.

    Returns
    -------
    output_signal : np.ndarray
        A 2D numpy array consisting of number of outputs x signal samples *
        output_oversample_factor.  This signal will be played directly to the
        shakers.

    """
    # Get a tolerance if specified
    try:
        rcond = float(extra_parameters)
    except ValueError:
        rcond = 1e-15
    H_pinv = np.linalg.pinv(transfer_function,rcond)
    spec_fft = np.fft.rfft(specification_signals).T
    drive_signals_fft = H_pinv@spec_fft[...,np.newaxis]
    # Oversample
    drive_signals_fft_zero_padded = np.concatenate((drive_signals_fft[:-1],
        np.zeros((drive_signals_fft[:-1].shape[0]*(output_oversample_factor-1)+1,)+drive_signals_fft.shape[1:])),axis=0)
    # Scale by output oversample factor to correct for more frequency lines
    drive_signals_oversampled = np.fft.irfft(drive_signals_fft_zero_padded.squeeze().T,axis=-1)*output_oversample_factor 
    return drive_signals_oversampled

def pseudoinverse_control_generator():
    drive_signals_oversampled = None
    spec_fft = None
    rcond = None
    H_pinv = None
    while True:
        (transfer_function,specification_signals,output_oversample_factor,extra_parameters,
            last_excitation_signals,last_response_signals) = yield drive_signals_oversampled
        if spec_fft is None:
            # Compute the FFT of the spec if it hasn't been done yet
            spec_fft = np.fft.rfft(specification_signals).T
        if rcond is None:
            # Get the condition number from the extra parameters if specified
            try:
                rcond = float(extra_parameters)
            except ValueError:
                rcond = 1e-15
        if H_pinv is None:
            # Invert the transfer function using the pseudoinverse
            H_pinv = np.linalg.pinv(transfer_function,rcond)
        drive_signals_fft = H_pinv@spec_fft[...,np.newaxis]
        # Oversample
        drive_signals_fft_zero_padded = np.concatenate((drive_signals_fft[:-1],
            np.zeros((drive_signals_fft[:-1].shape[0]*(output_oversample_factor-1)+1,)+drive_signals_fft.shape[1:])),axis=0)
        # Scale by output oversample factor to correct for more frequency lines
        drive_signals_oversampled = np.fft.irfft(drive_signals_fft_zero_padded.squeeze().T,axis=-1)*output_oversample_factor 

class pseudoinverse_control_class:
    def __init__(self,specification_signals : np.ndarray,extra_control_parameters : str,
                 output_oversample : int,transfer_function : np.ndarray = None,  # Transfer Functions
                 last_excitation_signals : np.ndarray = None, # Last output for Drive-based control
                 last_response_signals : np.ndarray = None, # Last Response for Error Correction
                 ):
        """
        Initializes the control law

        Parameters
        ----------
        specification_signals : np.ndarray
            The target signal for the control channels
        extra_control_parameters : str
            Extra parameters for the control law to use
        output_oversample : int
            Oversampling of the output compared to the acquisition
        transfer_function : np.ndarray, optional
            The current estimate of the transfer function between the control
            responses and the output voltages. The default is None.
        last_excitation_signals : np.ndarray, optional
            The most recent output signal, which can be used for error-based
            control. The default is None.
        last_response_signals : np.ndarray, optional
            The most recent responses to the last output signals, which can be
            used for error-based control. The default is None.

        Returns
        -------
        None.

        """
        self.spec_fft = np.fft.rfft(specification_signals).T
        self.output_oversample_factor = output_oversample
        # Get the condition number from the extra parameters if specified
        try:
            self.rcond = float(extra_control_parameters)
        except ValueError:
            self.rcond = 1e-15
        
    def system_id_update(self,transfer_function : np.ndarray):
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
        self.H_pinv = np.linalg.pinv(transfer_function,self.rcond)

    def control(self,
                last_excitation_signals : np.ndarray = None, # Last output for Drive-based control
                last_response_signals : np.ndarray = None, # Last Response for Error Correction
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
        drive_signals_fft = self.H_pinv@self.spec_fft[...,np.newaxis]
        # Oversample
        drive_signals_fft_zero_padded = np.concatenate((drive_signals_fft[:-1],
            np.zeros((drive_signals_fft[:-1].shape[0]*(self.output_oversample_factor-1)+1,)+drive_signals_fft.shape[1:])),axis=0)
        # Scale by output oversample factor to correct for more frequency lines
        drive_signals_oversampled = np.fft.irfft(drive_signals_fft_zero_padded.squeeze().T,axis=-1)*self.output_oversample_factor 
        return drive_signals_oversampled