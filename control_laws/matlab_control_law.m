function output_signals = matlab_control_law(sample_rate, ...
    specification_signals, output_oversample, ...
    frequency_spacing, transfer_function, noise_response_cpsd, ...
    noise_reference_cpsd, sysid_response_cpsd, sysid_reference_cpsd, ...
    multiple_coherence, frames, total_frames, last_excitation_signal,...
    last_response_signal, n_control, n_drive, n_timestep, n_freq)
% Note, due to limitations of the matlab engine for Python, all arrays are
% flattened prior to sending
%
% sample rate : Sample rate of the control
% specification_signals : a n_control x n_timestep signal that specifies
%       the desired output
% output_oversample : an output oversampling factor that the control law
%       must provide on the output signal because some hardware devices
%       have minimum output rates.  Note that if you are simply modifying
%       the specification, you don't need to use this value.  However, if
%       you are actually constructing the output time history, you need to
%       upsample the output data by this value.
% frequency_spacing : the frequency spacing in the spectral matrices that
%       were provided by the system identification.
% transfer_function : the transfer function between shaker voltage and the
%       control degrees of freedom from the test.  The shape is
%       n_freq x n_control x n_drive.  The frequency values start at 0 and
%       increase by frequency_spacing
% noise_response_cpsd : the power spectral density matrix from the noise
%       characterization done during the system identification.  The shape
%       is n_freq x n_control x n_control.  The frequency values start at 0
%       and increase by frequency_spacing
% noise_reference_cpsd : the power spectral density matrix from the noise
%       characterization done during the system identification.  The shape
%       is n_freq x n_drive x n_drive.  The frequency values start at 0
%       and increase by frequency_spacing
% sysid_response_cpsd : the power spectral density matrix from the system
%       identification.  The shape is n_freq x n_control x n_control.  The 
%       frequency values start at 0 and increase by frequency_spacing
% sysid_reference_cpsd : the power spectral density matrix from the system
%       identification.  The shape is n_freq x n_drive x n_drive.  The 
%       frequency values start at 0 and increase by frequency_spacing
% multiple_coherence : the coherence from the system identification with
%       shape n_freq x n_control
% frames : the number of measurement frames used to compute the system
%       identification matrices
% total_frames : the largest number of frames that are used to compute the
%       system identification matrices
% last_excitation_signals : the last excitation signal that can be used for
%       drive-based control
% last_response_signals : the last response signal that can be used for
%       error-based control
% n_control : the number of control channels
% n_drive : the number of drive channels
% n_timestep : the number of timesteps in the specification signal
% n_freq : the number of frequency lines in the system identification
%       matrices
%
% output_signals : the output from this matlab function must be new signals
%       for the Python controller.  If the extra_parameter update_spec is
%       True, then simply return an updated version of the specification
%       that the controller will then attempt to control to.  Otherwise if
%       update_spec is False, the control law will expect that the
%       output_signals are the next voltage signals to be played, it which
%       case, it must be oversampled per the output_oversample argument.
%       This must be either a n_control x n_timestep array when update_spec
%       is True or a n_drive x (n_timestep*output_oversample) array when
%       update_spec is False.

save('matlab_control_law_data.mat')
whos_struct = whos();
for i = 1:length(whos_struct)
    disp([whos_struct(i).name,' ',num2str(whos_struct(i).size),' complex: ',num2str(whos_struct(i).complex)])
end
output_signals = specification_signals;