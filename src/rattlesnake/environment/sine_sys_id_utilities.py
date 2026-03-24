# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 10:19:37 2025

@author: dprohe
"""

import os

# import time

import numpy as np
from scipy.io import loadmat
from scipy.signal import windows
from scipy.sparse import linalg
from scipy import sparse
from scipy.signal import lfilter, lfiltic, butter

DEBUG = False

if DEBUG:
    import pickle


def load_specification(spec_path):
    """Loads a sine specification from a .mat or .npz file

    Assumes the phases are represented in degrees

    Parameters
    ----------
    spec_path : str
        The path to the specification to load.

    Returns
    -------
    frequency : ndarray
        The frequency breakpoint of the specificatyion.
    amplitude : ndarray
        The amplitudes at the breakpoints for each channel.
    phase : ndarray
        The phase in degrees of the specification
    sweep_type : ndarray
        The sweep types at each breakpoint.
    sweep_rate : ndarray
        The sweep rate at each breakpoint.
    warning : ndarray
        Upper, lower, left, and right warning levels at each
        breakpoint
    abort : ndarray
        Upper, lower, left, and right abort levels at each breakpoint.
    start_time : float
        The start time for the sine sweep.
    name : str
        The name of the sine sweep.

    """
    _, extension = os.path.splitext(spec_path)
    if extension.lower() == ".mat":
        data = loadmat(spec_path)
    else:
        data = np.load(spec_path)
    frequency = data["frequency"].flatten()
    amplitude = data["amplitude"]
    if "phase" in data:
        phase = data["phase"]  # Degrees
    else:
        phase = None
    if "sweep_rate" in data:
        sweep_rate = data["sweep_rate"].flatten()
    else:
        sweep_rate = None
    if "sweep_type" in data:
        sweep_type = data["sweep_type"].flatten()
    else:
        sweep_type = None
    if "warning" in data:
        warning = data["warning"]
    else:
        warning = None
    if "abort" in data:
        abort = data["abort"]
    else:
        abort = None
    if "start_time" in data:
        start_time = data["start_time"]
    else:
        start_time = None
    if "name" in data:
        name = data["name"][()]
    else:
        name = None
    return (
        frequency,
        amplitude,
        phase,
        sweep_type,
        sweep_rate,
        warning,
        abort,
        start_time,
        name,
    )


def sine_sweep(
    dt,
    frequencies,
    sweep_rates,
    sweep_types,
    amplitudes=1,
    phases=0,
    return_argument=False,
    return_frequency=False,
    return_amplitude=False,
    return_phase=False,
    return_abscissa=False,
    only_breakpoints=False,
):
    """
    Generates a sweeping sine wave with linear or logarithmic sweep rate

    Parameters
    ----------
    dt : float
        The time step of the output signal
    frequencies : iterable
        A list of frequency breakpoints for the sweep.  Can be ascending or
        decending or both.  Frequencies are specified in Hz, not rad/s.
    sweep_rates : iterable
        A list of sweep rates between the breakpoints.  This array should have
        one fewer element than the `frequencies` array.  The ith element of this
        array specifies the sweep rate between `frequencies[i]` and
        `frequencies[i+1]`. For a linear sweep,
        the rate is in Hz/s.  For a logarithmic sweep, the rate is in octave/s.
    sweep_types : iterable or str
        The type of sweep to perform between each frequency breakpoint.  Can be
        'lin' or 'log'.  If a string is specified, it will be used for all
        breakpoints.  Otherwise it should be an array containing strings with
        one fewer element than that of the `frequencies` array.
    amplitudes : iterable or float, optional
        Amplitude of the cosine wave at each of the frequency breakpoints.  Can
        be specified as a single floating point value, or as an array with a
        value specified for each breakpoint. The default is 1.
    phases : iterable or float, optional
        Phases in radians of the cosine wave at each of the frequency breakpoints.  Can
        be specified as a single floating point value, or as an array with a
        value specified for each breakpoint. Be aware that modifying the phase
        between breakpoints will effectively change the frequency of the signal,
        because the phase will change over time.  The default is 0.
    return_argument : bool
        If True, return cosine argument over time
    return_frequency : bool
        If True, return the instantaneous frequency over time
    return_amplitude : bool
        If True, return the instantaneous amplitude over time
    return_phase : bool
        If True, return the instantaneous phase over time
    return_abscissa : bool
        If True, return the instantaneous abscissa over time
    only_breakpoints : bool
        If True, only returns data at breakpoints.  Default is False

    Raises
    ------
    ValueError
        If the sweep rate and start and end frequency would result in a negative
        sweep time, for example if the start frequency is above the end frequency
        and a positive sweep rate is specified.

    Returns
    -------
    ordinate : np.ndarray
        A numpy array consisting of the generated sine sweep signal.  The length
        of the signal will be determined by the frequency breakpoints and sweep
        rates.
    arg_over_time : np.ndarray
        A numpy array consisting of the argument to the cosine wave over time.
    freq_over_time : np.ndarray
        A numpy array consisting of the frequency of the cosine wave over time.
    amp_over_time : np.ndarray
        A numpy array consistsing of the amplitude of the cosine wave over time.
    phs_over_time : np.ndarray
        A numpy array consisting of the added phase in radians of the cosine
        wave over time.
    abscissa : np.ndarray
        A numpy array consisting of the time value at each time step returned

    """
    last_phase = 0
    last_abscissa = 0
    abscissa = []
    ordinate = []
    arg_over_time = []
    freq_over_time = []
    amp_over_time = []
    phs_over_time = []

    # Go through each section
    for i in range(len(frequencies) - 1):
        # Extract the terms
        start_frequency = frequencies[i]
        end_frequency = frequencies[i + 1]
        omega_start = start_frequency * 2 * np.pi
        try:
            sweep_rate = sweep_rates[i]
        except TypeError:
            sweep_rate = sweep_rates
        if isinstance(sweep_types, str):
            sweep_type = sweep_types
        else:
            sweep_type = sweep_types[i]
        try:
            start_amplitude = amplitudes[i]
            end_amplitude = amplitudes[i + 1]
        except TypeError:
            start_amplitude = amplitudes
            end_amplitude = amplitudes
        try:
            start_phase = phases[i]  # Radians
            end_phase = phases[i + 1]  # Radians
        except TypeError:
            start_phase = phases  # Radians
            end_phase = phases  # Radians
        # Compute the length of this portion of the signal
        if sweep_type.lower() in ["lin", "linear"]:
            sweep_time = +(end_frequency - start_frequency) / sweep_rate
        elif sweep_type.lower() in ["log", "logarithmic"]:
            sweep_time = np.log(end_frequency / start_frequency) / (sweep_rate * np.log(2))
        else:
            raise ValueError("Sweep type should be one of lin, linear, log, or logarithmic")
        if sweep_time < 0:
            raise ValueError(f"Sweep time for segment index {i} is negative.  Check sweep rate.")
        sweep_samples = int(np.floor(sweep_time / dt))
        # Construct the abscissa
        if only_breakpoints:
            this_abscissa = np.array([0, sweep_samples * dt])
        else:
            this_abscissa = np.arange(sweep_samples + 1) * dt
        # Compute the phase over time
        if sweep_type.lower() in ["lin", "linear"]:
            this_argument = (1 / 2) * (
                sweep_rate * 2 * np.pi
            ) * this_abscissa**2 + omega_start * this_abscissa
            this_frequency = (sweep_rate) * this_abscissa + omega_start / (2 * np.pi)
        elif sweep_type.lower() in ["log", "logarithmic"]:
            this_argument = 2 ** (sweep_rate * this_abscissa) * omega_start / (
                sweep_rate * np.log(2)
            ) - omega_start / (sweep_rate * np.log(2))
            this_frequency = 2 ** (sweep_rate * this_abscissa) * omega_start / (2 * np.pi)
        else:
            raise ValueError("Invalid sweep type, should be linear, lin, logarithmic, or log")
        # Compute the phase at each time step
        if end_frequency > start_frequency:
            freq_interp = [start_frequency, end_frequency]
            phase_interp = [start_phase, end_phase]
            amp_interp = [start_amplitude, end_amplitude]
        else:
            freq_interp = [end_frequency, start_frequency]
            phase_interp = [end_phase, start_phase]
            amp_interp = [end_amplitude, start_amplitude]
        this_phases = np.interp(this_frequency, freq_interp, phase_interp)
        # Compute the amplitude at each time step
        this_amplitudes = np.interp(this_frequency, freq_interp, amp_interp)
        this_ordinate = this_amplitudes * np.cos(this_argument + this_phases + last_phase)
        if i == len(frequencies) - 2:
            last_index = None  # If it's the last segment, go up until the end
        else:
            last_index = -1  # Otherwise, we remove the last point because the first point of the
            # next segment will be this value
        arg_over_time.append(this_argument[:last_index] + last_phase)
        last_phase += this_argument[-1]
        abscissa.append(this_abscissa[:last_index] + last_abscissa)
        last_abscissa += this_abscissa[-1]
        ordinate.append(this_ordinate[:last_index])
        freq_over_time.append(this_frequency[:last_index])
        amp_over_time.append(this_amplitudes[:last_index])
        phs_over_time.append(this_phases[:last_index])
    ordinate = np.concatenate(ordinate)
    return_vals = [ordinate]
    if return_argument:
        return_vals.append(np.concatenate(arg_over_time))
    if return_frequency:
        return_vals.append(np.concatenate(freq_over_time))
    if return_amplitude:
        return_vals.append(np.concatenate(amp_over_time))
    if return_phase:
        return_vals.append(np.concatenate(phs_over_time))
    if return_abscissa:
        return_vals.append(np.concatenate(abscissa))
    if len(return_vals) == 1:
        return_vals = return_vals[0]
    else:
        return_vals = tuple(return_vals)
    return return_vals


def digital_tracking_filter_generator(
    dt,
    cutoff_frequency_ratio=0.15,
    filter_order=2,
    phase_estimate=None,
    amplitude_estimate=None,
):
    """
    Computes amplitudes and phases using a digital tracking filter

    Parameters
    ----------
    dt : float
        The time step of the signal
    cutoff_frequency_ratio : float
        The cutoff frequency of the low-pass filter compared to the lowest
        frequency sine tone in each block.  Default is 0.15.
    filter_order : float
        The filter order of the low-pass butterworth filter.  Default is 2.
    phase_estimate : float
        An estimate of the initial phase to seed the low-pass filter.
    amplitude_estimate : float
        An estimate of the initial amplitude to seed the low-pass filter.

    Sends
    ------
    xi : iterable
        The next block of the signal to be filtered
    fi : iterable
        The frequencies at the time steps in xi
    argsi : iterable
        The argument to a cosine function at the time steps in xi

    Yields
    -------
    amplitude : np.ndarray
        The amplitude at each time step
    phase : np.ndarray
        The phase at each time step
    """
    # if plot_results:
    #     fig,ax = plt.subplots(2,2,sharex=True)
    #     ax[0,0].set_ylabel('Signal and Amplitude')
    #     ax[0,1].set_ylabel('Phase')
    #     ax[1,0].set_ylabel('Filtered COLA Signal (cos)')
    #     ax[1,1].set_ylabel('Filtered COLA Signal (sin)')
    # sample_index = 0
    # fig.tight_layout()
    if phase_estimate is None:
        phase_estimate = 0
    if amplitude_estimate is None:
        amplitude_estimate = 0

    xi_0_filt = None
    xi_90_filt = None
    xi_0 = None
    xi_90 = None
    amplitude = None
    phase = None
    while True:
        xi, fi, argsi = yield amplitude, phase
        xi = np.array(xi)
        fi = np.array(fi)
        argsi = np.array(argsi)
        # print(f"{cutoff_frequency_ratio=}")
        # print(f"{np.min(fi)=}")
        # print(f"{cutoff_frequency_ratio*np.min(fi)=}")
        b, a = butter(filter_order, cutoff_frequency_ratio * np.min(fi), fs=1 / dt)
        if xi_0_filt is None:
            # Set up some fake data to initialize the filter to a good value
            past_ts = np.arange(-filter_order * 2 - 1, 0) * dt
            past_xs = amplitude_estimate * np.cos(2 * np.pi * fi[0] * past_ts + phase_estimate)
            xi_0 = np.cos(2 * np.pi * fi[0] * past_ts) * past_xs
            xi_90 = -np.sin(2 * np.pi * fi[0] * past_ts) * past_xs
            xi_0_filt = 0.5 * amplitude_estimate * np.cos(phase_estimate) * np.ones(xi_0.shape)
            xi_90_filt = 0.5 * amplitude_estimate * np.sin(phase_estimate) * np.ones(xi_90.shape)
            # if plot_results:
            #     ax[1,0].plot(past_ts,xi_0,'r')
            #     ax[1,0].plot(past_ts,xi_0_filt,'m')
            #     ax[1,1].plot(past_ts,xi_90,'r')
            #     ax[1,1].plot(past_ts,xi_90_filt,'m')
        # Set up the filter initial states
        z0i = lfiltic(b, a, xi_0_filt[::-1], xi_0[::-1])
        z90i = lfiltic(b, a, xi_90_filt[::-1], xi_90[::-1])
        # Now set up the tracking filter
        cola0 = np.cos(argsi)
        cola90 = -np.sin(argsi)
        xi_0 = cola0 * xi
        xi_90 = cola90 * xi
        xi_0_filt, z0i = lfilter(b, a, xi_0, zi=z0i)
        xi_90_filt, z90i = lfilter(b, a, xi_90, zi=z90i)
        phase = np.arctan2(xi_90_filt, xi_0_filt)
        amplitude = 2 * np.sqrt(xi_0_filt**2 + xi_90_filt**2)
        # if plot_results:
        #     ti = np.arange(sample_index,sample_index + xi.shape[-1])*dt
        #     ax[0,0].plot(ti,xi,'b')
        #     ax[0,0].plot(ti,amplitude,'g')
        #     ax[0,1].plot(ti,phase,'g')
        #     ax[1,0].plot(ti,xi_0,'b')
        #     ax[1,0].plot(ti,xi_0_filt,'g')
        #     ax[1,1].plot(ti,xi_90,'b')
        #     ax[1,1].plot(ti,xi_90_filt,'g')
        #     sample_index += xi.shape[-1]


class DefaultSineControlLaw:
    """A default control law for the sine environment"""

    def __unpickleable_fields__(self):
        """Defines fields that can't be pickled in the case of an error"""
        return ["tracking_filters"]

    def __getstate__(self):
        """Defines how the object is pickled"""
        state = self.__dict__.copy()
        for field in self.__unpickleable_fields__():
            if field in state:
                del state[field]
        return state

    def __setstate__(self, state):
        """Defines how the object is restored from a pickle"""
        self.__dict__.update(state)
        for field in self.__unpickleable_fields__():
            setattr(self, field, None)

    def __init__(
        self,
        sample_rate,  # Sample Rate of the data acquisition
        specifications,  # Specification structured array
        output_oversample,  # Oversampling required for output
        ramp_time,  # Length of the ramp on the start and end of the signal
        convergence_factor,  # Scale factor on the convergence of the closed-loop control
        block_size,  # Size of writing blocks
        buffer_blocks,  # Number of write blocks to keep in the buffer
        extra_control_parameters,  # Required parameters
        sysid_frequency_spacing,  # Frequency Spacing
        sysid_transfer_functions,  # Transfer Functions
        sysid_response_noise,  # Noise levels and correlation
        sysid_reference_noise,  # from the system identification
        sysid_response_cpsd,  # Response levels and correlation
        sysid_reference_cpsd,  # from the system identification
        sysid_coherence,  # Coherence from the system identification
        sysid_frames,  # Number of frames in the FRF matrices
    ):
        """
        Initialize the sine control law

        Parameters
        ----------
        sample_rate : float
            The sample rate of the data acquisition system.
        specifications : list of SineSpecification
            One SineSpecification object for each tone in the environment.
        output_oversample : int
            The oversample factor on the output.
        ramp_time : float
            The time to ramp up in level to start or ramp down to end.
        convergence_factor : float
            A value between 0 and 1 specifying how quickly the error correction
            should take place.
        block_size : int
            The number of samples generated for each analysis and error
            correction step
        buffer_blocks : int
            The number of blocks to keep in a buffer to ensure we don't run
            out of data to generate.
        extra_control_parameters : str
            A string containing any extr information the control law needs
            to know about.
        sysid_frequency_spacing : float
            The frequency spacing in the transfer function.
        sysid_transfer_functions : ndarray
            A 3D ndarray with shape num_freq, num_control, num_drive.
        sysid_response_noise : ndarray
            A 3D ndarray containing CPSDs with values populated from the
            noise floor check.  Shape is num_freq, num_control, num_control.
        sysid_reference_noise : ndarray
            A 3D ndarray containing CPSDs with the values populated from the
            noise floor check.  Shape is num_freq, num_drive, num_drive.
        sysid_response_cpsd : ndarray
            A 3D ndarray containing CPSDs with values populated from the
            system identification.  Shape is num_freq, num_control, num_control.
        sysid_reference_cpsd :  ndarray
            A 3D ndarray containing CPSDs with the values populated from the
            system identification.  Shape is num_freq, num_drive, num_drive.
        sysid_coherence : ndarray
            Multiple coherence of the control channels from the system
            identification
        sysid_frames : int
            NNumber of frames used in the system identification.

        """
        # start_time = time.time()
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.buffer_blocks = buffer_blocks
        self.specifications = [spec.copy() for spec in specifications]
        self.output_oversample = output_oversample
        self.extra_control_parameters = extra_control_parameters
        self.ramp_samples = int(ramp_time * sample_rate) * output_oversample
        self.convergence_factor = convergence_factor
        # Loop through the different specifications and perform the initial control
        (
            self.specified_response,
            self.specified_tone_response,
            self.specified_frequency,
            self.specified_argument,
            self.specified_amplitude,
            self.specified_phase,  # Radians
            _,
            _,
        ) = SineSpecification.create_combined_signals(
            self.specifications,
            self.sample_rate * self.output_oversample,
            self.ramp_samples,
        )
        self.specified_phase = self.specified_phase  # Radians
        self.tone_slices = []
        for amp in self.specified_amplitude:
            nonzero_indices = np.any(amp != 0, axis=0)
            self.tone_slices.append(
                slice(
                    np.argmax(nonzero_indices),
                    nonzero_indices.size - np.argmax(nonzero_indices[::-1]),
                )
            )

        # System ID Parameters
        self.frfs = None
        self.frf_frequency_spacing = None
        self.frf_frequencies = None
        self.sysid_response_noise = None
        self.sysid_reference_noise = None
        self.sysid_response_cpsd = None
        self.sysid_reference_cpsd = None
        self.sysid_coherence = None
        self.frames = None
        self.frf_pinv = None
        self.interpolated_frf_pinv = None
        self.largest_correction_factors = None

        # Preshaped drive parameters
        self.preshaped_drive_amplitudes = None
        self.preshaped_drive_phases = None  # Radians
        self.preshaped_drive_signals = None

        # Control parameters
        self.start_index = None
        self.end_index = None
        self.signal_slice = None
        self.control_tones = None
        self.control_ramp_up = None
        self.control_ramp_down = None
        self.target_ramp_up = None
        self.target_ramp_down = None
        self.control_response_signals = None
        self.control_response_amplitudes = None
        self.control_response_phases = None  # Radians
        self.control_drive_correction = None
        self.control_sent_complex_excitation = None
        self.control_analysis_index = None
        self.control_write_index = None
        self.max_singular_values = None

        self.maximum_drive_voltage = None
        self.harddisk_storage = None
        for string in extra_control_parameters.split("\n"):
            if string.strip() == "":
                continue
            try:
                command, value = string.split("=")
            except ValueError:
                print(f'Unable to Parse Extra Parameters Line"{string:}"')
                continue
            command = command.strip()
            value = value.strip()
            if command in [
                "maximum_drive_voltage",
                "max_drive_voltage",
                "maximum_excitation_voltage",
                "max_excitation_voltage",
            ]:
                self.maximum_drive_voltage = float(value)
                print(f"Set Maximum Drive Voltage to {self.maximum_drive_voltage}")
            elif command in ["harddisk_storage"]:
                self.harddisk_storage = value
            else:
                print(f"Unknown extra parameter {command}")

        if sysid_transfer_functions is not None:
            self.system_id_update(
                sysid_frequency_spacing,  # Frequency Spacing
                sysid_transfer_functions,  # Transfer Functions
                sysid_response_noise,  # Noise levels and correlation
                sysid_reference_noise,  # from the system identification
                sysid_response_cpsd,  # Response levels and correlation
                sysid_reference_cpsd,  # from the system identification
                sysid_coherence,  # Coherence from the system identification
                sysid_frames,  # Number of frames in the FRF matrices
            )
        # finish_time = time.time()
        # print(f'__init__ called in {finish_time - start_time:0.2f}s.')

    def system_id_update(
        self,
        sysid_frequency_spacing,  # Frequency Spacing
        sysid_transfer_functions,  # Transfer Functions
        sysid_response_noise,  # Noise levels and correlation
        sysid_reference_noise,  # from the system identification
        sysid_response_cpsd,  # Response levels and correlation
        sysid_reference_cpsd,  # from the system identification
        sysid_coherence,  # Coherence from the system identification
        sysid_frames,  # Number of frames in the FRF matrices
    ):
        """
        Updates the control law after system identification has finished

        Parameters
        ----------
        sysid_frequency_spacing : float
            The frequency spacing in the transfer function.
        sysid_transfer_functions : ndarray
            A 3D ndarray with shape num_freq, num_control, num_drive.
        sysid_response_noise : ndarray
            A 3D ndarray containing CPSDs with values populated from the
            noise floor check.  Shape is num_freq, num_control, num_control.
        sysid_reference_noise : ndarray
            A 3D ndarray containing CPSDs with the values populated from the
            noise floor check.  Shape is num_freq, num_drive, num_drive.
        sysid_response_cpsd : ndarray
            A 3D ndarray containing CPSDs with values populated from the
            system identification.  Shape is num_freq, num_control, num_control.
        sysid_reference_cpsd :  ndarray
            A 3D ndarray containing CPSDs with the values populated from the
            system identification.  Shape is num_freq, num_drive, num_drive.
        sysid_coherence : ndarray
            Multiple coherence of the control channels from the system
            identification
        sysid_frames : int
            Number of frames used in the system identification.

        Returns
        -------
        preshaped_drive_signals : ndarray
            Excitation signals with shape num_tones, num_drives, num_timesteps.
        specified_frequency : ndarray
            The instantaneous frequencies at each of the drive timesteps with
            shape num_tones, num_timesteps
        specified_argument : ndarray
            The instantaneous sine argument at each of the drive timesteps with
            shape num_tones, num_drives, num_timesteps
        preshaped_drive_amplitudes : ndarray
            The instantaneous amplitude at each of the drive timesteps with
            shape num_tones, num_drives, num_timesteps
        preshaped_drive_phases : ndarray
            The instantaneous phase in radians at each of the drive timesteps
            with shape num_tones, num_drives, num_timesteps.
        """
        # start_time = time.time()
        # print('Updating System ID Information')
        self.frf_frequency_spacing = sysid_frequency_spacing
        self.frfs = sysid_transfer_functions
        self.frf_frequencies = self.frf_frequency_spacing * np.arange(self.frfs.shape[0])
        # print('Inverting FRFs')
        self.frf_pinv = np.linalg.pinv(self.frfs)
        self.max_singular_values = np.max(
            np.linalg.svd(self.frfs, compute_uv=False, full_matrices=False), axis=-1
        )
        self.sysid_response_noise = sysid_response_noise
        self.sysid_reference_noise = sysid_reference_noise
        self.sysid_response_cpsd = sysid_response_cpsd
        self.sysid_reference_cpsd = sysid_reference_cpsd
        self.sysid_coherence = sysid_coherence
        self.frames = sysid_frames

        # Go through and compute the response amplitudes and phases from each of the sine tones
        # print('Preallocating Amplitude, Phase, FRFs, and Correction Factors')
        if self.harddisk_storage is not None:
            filename = os.path.join(self.harddisk_storage, "preshaped_drive_amplitudes.mmap")
            shape = (
                self.specified_frequency.shape[0],
                self.frfs.shape[-1],
                self.specified_frequency.shape[-1],
            )
            self.preshaped_drive_amplitudes = np.memmap(
                filename, dtype=float, shape=shape, mode="w+"
            )
            filename = os.path.join(self.harddisk_storage, "preshaped_drive_phases.mmap")
            shape = (
                self.specified_frequency.shape[0],
                self.frfs.shape[-1],
                self.specified_frequency.shape[-1],
            )
            self.preshaped_drive_phases = np.memmap(filename, dtype=float, shape=shape, mode="w+")
            filename = os.path.join(self.harddisk_storage, "largest_correction_factors.mmap")
            shape = self.specified_frequency.shape
            self.largest_correction_factors = np.memmap(
                filename, dtype=float, shape=shape, mode="w+"
            )
            filename = os.path.join(self.harddisk_storage, "interpolated_frf_pinv.mmap")
            shape = (
                self.specified_frequency.shape[0],
                self.specified_frequency.shape[-1],
            ) + self.frf_pinv.shape[-2:]
            self.interpolated_frf_pinv = np.memmap(filename, dtype="c16", shape=shape, mode="w+")
        else:
            self.preshaped_drive_amplitudes = np.zeros(
                (
                    self.specified_frequency.shape[0],
                    self.frfs.shape[-1],
                    self.specified_frequency.shape[-1],
                )
            )
            self.preshaped_drive_phases = np.zeros(  # Radians
                (
                    self.specified_frequency.shape[0],
                    self.frfs.shape[-1],
                    self.specified_frequency.shape[-1],
                )
            )
            self.largest_correction_factors = np.zeros(self.specified_frequency.shape)
            self.interpolated_frf_pinv = np.zeros(
                (
                    self.specified_frequency.shape[0],
                    self.specified_frequency.shape[-1],
                )
                + self.frf_pinv.shape[-2:],
                dtype="c16",
            )
        for tone_index, (freq, amp, phs, control_slice) in enumerate(  # phs is radians
            zip(
                self.specified_frequency,
                self.specified_amplitude,
                self.specified_phase,  # Radians
                self.tone_slices,
            )
        ):
            # print('Interpolating Tone {:}'.format(tone_index))
            control_amp = amp[..., control_slice]
            control_phs = phs[..., control_slice]  # Radians
            control_freq = freq[..., control_slice]
            # Interpolate the pseudoinverse of the FRF
            for index in np.ndindex(*self.frf_pinv.shape[1:]):
                # print('  Interpolating Response {:}'.format(index))
                interpolated_index = (tone_index, control_slice) + index
                frf_pinv_index = (Ellipsis,) + index
                self.interpolated_frf_pinv[interpolated_index] = np.interp(
                    control_freq, self.frf_frequencies, self.frf_pinv[frf_pinv_index]
                )
            # print('Computing Largest Correction Factors')
            self.largest_correction_factors[tone_index, control_slice] = (
                1 / np.interp(control_freq, self.frf_frequencies, self.max_singular_values) ** 2
            )
            # print('Computing Complex Response')
            complex_response = np.moveaxis(
                control_amp * np.exp(1j * control_phs), -1, 0
            )[  # Radians
                ..., np.newaxis
            ]
            # print('Computing Complex Excitation')
            complex_excitation = np.moveaxis(
                (self.interpolated_frf_pinv[tone_index, control_slice] @ complex_response)[..., 0],
                0,
                1,
            )
            # print('Extracting Excitation Amplitude and Phase')
            self.preshaped_drive_amplitudes[tone_index, :, control_slice] = np.abs(
                complex_excitation
            )
            self.preshaped_drive_phases[tone_index, :, control_slice] = np.unwrap(
                np.angle(complex_excitation)
            )  # Radians

        if self.maximum_drive_voltage is not None:
            # print('Truncating for Maximum Voltage')
            self.preshaped_drive_amplitudes[
                self.preshaped_drive_amplitudes > self.maximum_drive_voltage
            ] = self.maximum_drive_voltage
            self.preshaped_drive_amplitudes[
                self.preshaped_drive_amplitudes < -self.maximum_drive_voltage
            ] = -self.maximum_drive_voltage
        # print('Computing Excitation Signal')
        self.preshaped_drive_signals = self.preshaped_drive_amplitudes * np.cos(
            self.specified_argument[:, np.newaxis, :] + self.preshaped_drive_phases  # Radians
        )

        if self.harddisk_storage is not None:
            # print('Flushing memmaps')
            self.preshaped_drive_amplitudes.flush()
            self.preshaped_drive_phases.flush()  # Radians
            self.largest_correction_factors.flush()
            self.interpolated_frf_pinv.flush()

        if DEBUG:
            print("Writing Sine Debug Pickle")
            with open("debug_data/sine_control_law_debug.pkl", "wb") as f:
                pickle.dump(self, f)
            print("Done!")

        # finish_time = time.time()
        # print(f'system_id_update called in {finish_time - start_time:0.2f}s.')

        return (
            self.preshaped_drive_signals,
            self.specified_frequency,
            self.specified_argument,
            self.preshaped_drive_amplitudes,
            self.preshaped_drive_phases,  # Radians for return value
        )

    def get_control_targets(self, block_start, block_end):
        """Gets up the control targets for a specified block, adding ramps

        Parameters
        ----------
        block_start : int
            The starting index for the block of data that targets are computed from
        block_end : int
            The end index for the block of data that targets are computed from

        Returns
        -------
        amplitudes
            The amplitudes over time including ramp up and ramp down
        phases
            The phases over time including ramp up and ramp down portions, in radians
        arguments
            The argument of the cosine wave over time including the ramp
            up and ramp down portions
        """
        ramp_up_start = block_start - self.ramp_samples
        if ramp_up_start >= 0:
            ramp_up_start = self.ramp_samples
        ramp_up_end = block_end - self.ramp_samples
        if ramp_up_end >= 0:
            ramp_up_end = self.ramp_samples
        ramp_down_start = block_start - self.end_index + self.start_index + self.ramp_samples
        if ramp_down_start < 0:
            ramp_down_start = 0
        ramp_down_end = block_end - self.end_index + self.start_index + self.ramp_samples
        if ramp_down_end < 0:
            ramp_down_end = 0
        middle_start = block_start
        if middle_start < self.ramp_samples:
            middle_start = self.ramp_samples
        if middle_start > self.end_index - self.start_index - self.ramp_samples:
            middle_start = self.end_index - self.start_index - self.ramp_samples
        middle_end = block_end
        if middle_end < self.ramp_samples:
            middle_end = self.ramp_samples
        if middle_end > self.end_index - self.start_index - self.ramp_samples:
            middle_end = self.end_index - self.start_index - self.ramp_samples
        amplitudes = np.concatenate(
            (
                self.target_ramp_up[..., ramp_up_start:ramp_up_end],
                self.specified_amplitude[
                    self.control_tones,
                    ...,
                    self.start_index + middle_start : self.start_index + middle_end,
                ],
                self.target_ramp_down[..., ramp_down_start:ramp_down_end],
            ),
            axis=-1,
        )
        phases = self.specified_phase[  # Radians
            self.control_tones,
            ...,
            self.start_index + block_start : self.start_index + block_start + amplitudes.shape[-1],
        ]
        arguments = self.specified_argument[
            self.control_tones,
            self.start_index + block_start : self.start_index + block_start + amplitudes.shape[-1],
        ]
        return amplitudes, phases, arguments  # Radians

    def get_control_preshaped_excitations(self, block_start, block_end):
        """Gets the initial guess at an excitation signal over a portion of time

        Parameters
        ----------
        block_start : int
            The starting index for the block of data that excitations are computed from
        block_end : int
            The end index for the block of data that excitations are computed from

        Returns
        -------
        amplitudes
            The amplitudes over time including ramp up and ramp down
        phases
            The phases over time including ramp up and ramp down portions
        arguments
            The argument of the cosine wave over time including the ramp
            up and ramp down portions, in radians
        """
        ramp_up_start = block_start - self.ramp_samples
        if ramp_up_start >= 0:
            ramp_up_start = self.ramp_samples
        ramp_up_end = block_end - self.ramp_samples
        if ramp_up_end >= 0:
            ramp_up_end = self.ramp_samples
        ramp_down_start = block_start - self.end_index + self.start_index + self.ramp_samples
        if ramp_down_start < 0:
            ramp_down_start = 0
        ramp_down_end = block_end - self.end_index + self.start_index + self.ramp_samples
        if ramp_down_end < 0:
            ramp_down_end = 0
        middle_start = block_start
        if middle_start < self.ramp_samples:
            middle_start = self.ramp_samples
        if middle_start > self.end_index - self.start_index - self.ramp_samples:
            middle_start = self.end_index - self.start_index - self.ramp_samples
        middle_end = block_end
        if middle_end < self.ramp_samples:
            middle_end = self.ramp_samples
        if middle_end > self.end_index - self.start_index - self.ramp_samples:
            middle_end = self.end_index - self.start_index - self.ramp_samples
        amplitudes = np.concatenate(
            (
                self.control_ramp_up[..., ramp_up_start:ramp_up_end],
                self.preshaped_drive_amplitudes[
                    self.control_tones,
                    ...,
                    self.start_index + middle_start : self.start_index + middle_end,
                ],
                self.control_ramp_down[..., ramp_down_start:ramp_down_end],
            ),
            axis=-1,
        )
        phases = self.preshaped_drive_phases[  # Radians
            self.control_tones,
            ...,
            self.start_index + block_start : self.start_index + block_start + amplitudes.shape[-1],
        ]
        arguments = self.specified_argument[
            self.control_tones,
            self.start_index + block_start : self.start_index + block_start + amplitudes.shape[-1],
        ]
        return amplitudes, phases, arguments  # Phase in Radians

    def initialize_control(self, control_tones, start_index, end_index):
        """
        Initializes the control and creates a preshaped drive signal

        Aguments are provided to specify which tones and portion of time
        to generate the signal over

        Parameters
        ----------
        control_tones : ndarray or slice
            Indicies into the specifications to determine which control tones
            should be used.
        start_index : int
            The starting time step index.
        end_index : int
            The ending time step index.

        Returns
        -------
        excitation_signals : ndarray
            The drive signal at each shaker over time

        """
        # start_time = time.time()
        # Parse the frequency content to get the portion of the signal we care
        # about
        self.control_tones = control_tones
        self.start_index = start_index
        self.end_index = self.preshaped_drive_signals.shape[-1] if end_index is None else end_index

        if DEBUG:
            print("Writing Sine Debug Pickle")
            with open("debug_data/sine_control_law_initialize_control_debug.pkl", "wb") as f:
                pickle.dump(self, f)
            print("Done!")

        # Set up the analysis and write_indices
        self.control_analysis_index = 0
        self.control_write_index = self.ramp_samples + self.buffer_blocks * self.block_size

        # Set up the ramp-ups and ramp downs for the excitation signal
        self.control_ramp_up = (
            np.linspace(0, 1, self.ramp_samples)
            * self.preshaped_drive_amplitudes[
                self.control_tones,
                ...,
                self.start_index + self.ramp_samples,
                np.newaxis,
            ]
        )
        self.control_ramp_down = (
            np.linspace(1, 0, self.ramp_samples)
            * self.preshaped_drive_amplitudes[
                self.control_tones, ..., self.end_index - self.ramp_samples, np.newaxis
            ]
        )
        self.target_ramp_up = (
            np.linspace(0, 1, self.ramp_samples)
            * self.specified_amplitude[
                self.control_tones,
                ...,
                self.start_index + self.ramp_samples,
                np.newaxis,
            ]
        )
        self.target_ramp_down = (
            np.linspace(1, 0, self.ramp_samples)
            * self.specified_amplitude[
                self.control_tones, ..., self.end_index - self.ramp_samples, np.newaxis
            ]
        )

        (
            starting_drive_amplitudes,
            starting_drive_phases,
            starting_arguments,
        ) = self.get_control_preshaped_excitations(  # Radians
            0, self.control_write_index
        )  # Radians

        complex_excitation = starting_drive_amplitudes * (
            np.exp(1j * starting_drive_phases)
        )  # Radians
        excitation_signals = np.sum(
            starting_drive_amplitudes
            * np.cos(starting_drive_phases + starting_arguments[:, np.newaxis, :]),  # Radians
            axis=0,
        )

        # Set up control parameters
        self.control_drive_correction = np.zeros(starting_drive_amplitudes.shape[:2], dtype=complex)

        # Set up the amplitude and phase tracking
        self.control_response_amplitudes = []
        self.control_response_phases = []  # Radians
        self.control_response_signals = []
        self.control_sent_complex_excitation = []

        if DEBUG:
            print("Writing Sine Debug Pickle")
            with open("debug_data/sine_control_law_debug.pkl", "wb") as f:
                pickle.dump(self, f)
            print("Done!")

        # Sending ramp and first two blocks to start, so add them to the list of blocks sent.
        self.control_sent_complex_excitation.append(complex_excitation)

        # finish_time = time.time()
        # print(f'initialize_control called in {finish_time - start_time:0.2f}s.')
        return excitation_signals

    def update_control(
        self,
        control_signals,
        control_amplitudes,
        control_phases,  # Radians
        control_frequencies,  # pylint: disable=unused-argument
        time_delay,  # pylint: disable=unused-argument
    ):
        """
        Updates the control parameters based on previous responses

        Parameters
        ----------
        control_signals : ndarray
            Time histories acquired by the environment.
        control_amplitudes : ndarray
            Amplitudes extracted from the time signals with shape num_tones,
            num_channels, num_timesteps.
        control_phases : ndarray
            Phases extracted from the time signals in radians with shape
            num_tones, num_channels, num_timesteps
        control_frequencies : ndarray
            Instantaneous frequencies at the timesteps analyzed with shape
            num_tones, num_timesteps.
        time_delay : float
            Time delay computed between the acquisition and output signals,
            which can be used to adjust for phase drifts due to delays.

        Returns
        -------
        drive_correction : ndarray
            A correction factor on the drive signals with shape num_tones,
            num_drives.

        """

        # start_time = time.time()
        self.control_response_signals.append(control_signals)
        self.control_response_amplitudes.append(control_amplitudes)
        self.control_response_phases.append(control_phases)  # Radians

        if DEBUG:
            print("Writing Sine Debug Pickle")
            with open("debug_data/sine_control_law_update_control_debug.pkl", "wb") as f:
                pickle.dump(self, f)
            print("Done!")

        # Find the equivalent block in the signal
        block_start_index = self.control_analysis_index
        block_end_index = (
            control_signals.shape[-1] * self.output_oversample + self.control_analysis_index
        )
        if self.convergence_factor != 0:
            reduction_slice = slice(
                block_start_index + self.start_index,
                block_end_index + self.start_index,
                self.output_oversample,
            )
            # Compute the target of the current block
            target_response_amplitudes, target_response_phases, _ = (  # Radians
                self.get_control_targets(block_start_index, block_end_index)
            )
            complex_targets = target_response_amplitudes[..., :: self.output_oversample] * np.exp(
                1j * target_response_phases[..., :: self.output_oversample]
            )
            complex_achieved = control_amplitudes * np.exp(1j * control_phases)  # Radians
            complex_error = (
                complex_targets - complex_achieved
            )  # Number of Tones x Num Responses x Num Freqs
            block_correction_factor = self.convergence_factor * np.min(
                self.largest_correction_factors[self.control_tones, ..., reduction_slice],
                axis=-1,
                keepdims=True,
            )  # Num Tones x 1
            block_frf = self.interpolated_frf_pinv[
                self.control_tones, reduction_slice, ...
            ]  # Number of Tones x Num Freqs x Num Excitations x Num Responses
            block_drive_correction = (  # Number of Tones x Num Freqs x Num Excitations x 1
                block_frf  # Number of Tones x Num Freqs x Num Excitations x Num Responses
                @ complex_error.transpose(0, 2, 1)[..., np.newaxis]
            )  # Number of Tones x Number Freqs x Num Responses x 1
            self.control_drive_correction = (
                self.control_drive_correction  # Number of Tones x Number of Excitation Signals
                + block_correction_factor  # Number of Tones x 1
                * np.mean(block_drive_correction[..., 0], axis=1)
            )  # Mean across frequency lines, Number of Tones x Num Excitations
        self.control_analysis_index = block_end_index
        # finish_time = time.time()
        # print(f'update_control called in {finish_time - start_time:0.2f}s.')
        return self.control_drive_correction

    def generate_signal(self):
        """
        Generates the next portion of the signal during the control
        calculations

        Returns
        -------
        excitation_signal : ndarray
            The next portion of the signal to generate, with shape num_drives,
            num_timesteps.
        done_controlling : bool
            A flag specifying that the entire signal has been generated, so no
            more control decisions should be made.

        """
        if DEBUG:
            print("Writing Sine Debug Pickle")
            with open("debug_data/sine_control_law_generate_signal_debug.pkl", "wb") as f:
                pickle.dump(self, f)
            print("Done!")

        # start_time = time.time()
        start_index = self.control_write_index
        end_index = self.control_write_index + self.block_size
        excitation_amplitudes, excitation_phases, excitation_arguments = (  # Radians
            self.get_control_preshaped_excitations(start_index, end_index)
        )
        complex_excitation = (
            excitation_amplitudes * np.exp(1j * excitation_phases)
            + self.control_drive_correction[..., np.newaxis]
        )  # Num tones x num signals x num freqs
        amplitudes = np.abs(complex_excitation)
        if self.maximum_drive_voltage is not None:
            over_indices = amplitudes > self.maximum_drive_voltage
            complex_excitation[over_indices] = (
                self.maximum_drive_voltage
                * complex_excitation[over_indices]
                / amplitudes[over_indices]
            )
        excitation_signals = np.abs(complex_excitation) * np.cos(
            excitation_arguments[:, np.newaxis, :] + np.angle(complex_excitation)
        )
        # Combine all tones into one signal
        excitation_signal = np.sum(excitation_signals, axis=0)
        # Store this value so we know what was output
        self.control_sent_complex_excitation.append(complex_excitation)
        # Check if we've exhausted all of our data
        done_controlling = end_index >= (self.end_index - self.start_index)
        self.control_write_index = end_index

        # finish_time = time.time()
        # print(f'generate_signal called in {finish_time - start_time:0.2f}s.')
        return excitation_signal, done_controlling

    def finalize_control(self):
        """
        A method to update the control based on previous results, generating a
        new preshaped drive signal

        Returns
        -------
        preshaped_drive_signals : ndarray
            Excitation signals with shape num_tones, num_drives, num_timesteps.
        specified_frequency : ndarray
            The instantaneous frequencies at each of the drive timesteps with
            shape num_tones, num_timesteps
        specified_argument : ndarray
            The instantaneous sine argument at each of the drive timesteps with
            shape num_tones, num_drives, num_timesteps
        preshaped_drive_amplitudes : ndarray
            The instantaneous amplitude at each of the drive timesteps with
            shape num_tones, num_drives, num_timesteps
        preshaped_drive_phases : ndarray
            The instantaneous phase in radians at each of the drive timesteps
            with shape num_tones, num_drives, num_timesteps.
        ramp_samples : ndarray
            The number of ramp samples used in the signal

        """
        return (
            self.preshaped_drive_signals,
            self.specified_frequency,
            self.specified_argument,
            self.preshaped_drive_amplitudes,
            self.preshaped_drive_phases,  # Radians
            self.ramp_samples,
        )


def vold_kalman_filter(
    sample_rate,
    signal,
    arguments,
    filter_order=None,
    bandwidth=None,
    method=None,
    return_amp_phs=False,
    return_envelope=False,
    return_r=False,
):
    """
    Extract sinusoidal components from a signal using the second generation
    Vold-Kalman filter.

    Parameters
    ----------
    sample_rate : float
        The sample rate of the signal in Hz.
    signal : ndarray
        A 1D signal containing sinusoidal components that need to be extracted
    arguments : ndarray
        A 2D array consisting of the arguments to the sinusoidal components of
        the form exp(1j*argument).  This is the integral over time of the
        angular frequency, which can be approximated as
        2*np.pi*scipy.integrate.cumulative_trapezoid(frequencies,timesteps,initial=0)
        if frequencies is the frequency at each time step in Hz timesteps is
        the vector of time steps in seconds.  This is a 2D array where the
        number of rows is the
        number of different sinusoidal components that are desired to be
        extracted, and the number of columns are the number of time steps in
        the `signal` argument.
    filter_order : int, optional
        The order of the VK filter, which should be 1, 2, or 3. The default is
        2.  The low-pass filter roll-off is approximately -40 dB per times the
        filter order.
    bandwidth : ndarray, optional
        The prescribed bandwidth of the filter. This is related to the filter
        selectivity parameter `r` in the literature.  This will be broadcast to
        the same shape as the `arguments` argument.  The default is the sample
        rate divided by 1000.
    method : str, optional
        Can be set to either 'single' or 'multi'.  In a 'single' solution, each
        sinusoidal component will be solved independently without any coupling.
        This can be more efficient, but will result in errors if the
        frequencies of the sine waves cross.  The 'multi' solution will solve
        for all sinusoidal components simultaneously, resulting in a better
        estimate of crossing frequencies. The default is 'multi'.
    return_amp_phs : bool
        Returns the amplitude and phase of the reconstructed signals at each
        time step.  Default is False
    return_envelope : bool
        Returns the complex envelope and phasors at each time step.  Default is
        False
    return_r : bool
        Returns the computed selectivity parameters for the filter.  Default is
        False

    Raises
    ------
    ValueError
        If arguments are not the correct size or values.

    Returns
    -------
    reconstructed_signals : ndarray
        Returns a time history the same size as `signal` for each of the
        sinusoidal components solved for.
    reconstructed_amplitudes : ndarray
        Returns the amplitude over time for each of the sinusoidal components
        solved for.  Only returned if return_amp_phs is True.
    reconstructed_phases : ndarray
        Returns the phase over time for each of the sinusoidal components
        solved for.  Only returned if return_amp_phs is True.
    reconstructed_envelope : ndarray
        Returns the complex envelope `x` over time for each of the sinusoidal
        components solved for.  Only returned if return_envelope is True.
    reconstructed_phasor : ndarray
        Returns the phasor `c` over time for each of the sinusoidal components
        solved for.  Only returned if return_envelope is True.
    r : ndarray
        Returns the selectivity `r` over time for each of the sinusoidal
        components solved for.  Only returned if return_r is True.

    """
    # pylint: disable=invalid-name
    if filter_order is None:
        filter_order = 2

    if bandwidth is None:
        bandwidth = sample_rate / 1000

    # Make sure input data are numpy arrays
    signal = np.array(signal)
    arguments = np.atleast_2d(arguments)
    bandwidth = np.atleast_2d(bandwidth)
    bandwidth = np.broadcast_to(bandwidth, arguments.shape)
    relative_bandwidth = bandwidth / sample_rate

    # Extract some sizes to make sure everything is correctly sized
    n_samples = signal.shape[-1]

    n_orders_arg, n_arg = arguments.shape
    if n_arg != n_samples:
        raise ValueError(
            "Argument array must have identical number of columns as samples in signal"
        )

    if method is None:
        if n_orders_arg > 1:
            method = "multi"
        else:
            method = "single"
    if method.lower() not in ["multi", "single"]:
        raise ValueError('`method` must be either "multi" or "single"')

    # Construct phasors to multiply the signals by
    phasor = np.exp(1j * arguments)

    # Construct the matrices for the least squares solution
    if filter_order == 1:
        coefs = np.array([1, -1])
        r = np.sqrt((np.sqrt(2) - 1) / (2 * (1 - np.cos(np.pi * relative_bandwidth))))
    elif filter_order == 2:
        coefs = np.array([1, -2, 1])
        r = np.sqrt(
            (np.sqrt(2) - 1)
            / (
                6
                - 8 * np.cos(np.pi * relative_bandwidth)
                + 2 * np.cos(2 * np.pi * relative_bandwidth)
            )
        )
    elif filter_order == 3:
        coefs = np.array([1, -3, 3, -1])
        r = np.sqrt(
            (np.sqrt(2) - 1)
            / (
                20
                - 30 * np.cos(np.pi * relative_bandwidth)
                + 12 * np.cos(2 * np.pi * relative_bandwidth)
                - 2 * np.cos(3 * np.pi * relative_bandwidth)
            )
        )
    else:
        raise ValueError("filter order must be 1, 2, or 3")

    # Construct the solution matrices
    A = sparse.spdiags(
        np.tile(coefs, (n_samples, 1)).T,
        np.arange(filter_order + 1),
        n_samples - filter_order,
        n_samples,
    )
    B = []
    for rvec in r:
        R = sparse.spdiags(rvec, 0, n_samples, n_samples)
        AR = A @ R
        B.append((AR).T @ (AR) + sparse.eye(n_samples))

    if method.lower() == "multi":
        # This solves the multiple order approach, constructing a big matrix of
        # Bs on the diagonal and CHCs on the off-diagonals.  We can set up the
        # matrix as diagonals and upper diagonals then add the transpose to get the
        # lower diagonals
        B_multi_diagonal = sparse.block_diag(B)
        # There will be number of orders**2 B matrices, and number of orders
        # diagonals, so there will be n_orders**2-n_orders off diagonals, half on
        # on the upper triangle.  We need to fill in all of these values for all
        # time steps.
        num_off_diags = (n_orders_arg**2 - n_orders_arg) // 2
        row_indices = np.zeros((n_samples, num_off_diags), dtype=int)
        col_indices = np.zeros((n_samples, num_off_diags), dtype=int)
        CHC = np.zeros((n_samples, num_off_diags), dtype="c16")
        # Keep track of the off-diagonal index so we know which column to put the
        # data in
        off_diagonal_index = 0
        # Now we need to step through the off-diagonal blocks and create the arrays
        for row_index in range(n_orders_arg):
            # Since we need to stay on the upper triangle, column indices will start
            # after the diagonal entry
            for col_index in range(row_index + 1, n_orders_arg):
                row_indices[:, off_diagonal_index] = np.arange(
                    row_index * n_samples, (row_index + 1) * n_samples
                )
                col_indices[:, off_diagonal_index] = np.arange(
                    col_index * n_samples, (col_index + 1) * n_samples
                )
                CHC[:, off_diagonal_index] = phasor[row_index].conj() * phasor[col_index]
                off_diagonal_index += 1
        # We set up the variables as multidimensional so we could store them easier,
        # but now we need to flatten them to put them into the sparse matrix.
        # We choose CSR because we can do math with it easier
        B_multi_utri = sparse.csr_matrix(
            (CHC.flatten(), (row_indices.flatten(), col_indices.flatten())),
            shape=B_multi_diagonal.shape,
        )

        # Now we can assemble the entire matrix by adding with the complex conjugate
        # of the upper triangle to get the lower triangle
        B_multi = B_multi_diagonal + B_multi_utri + B_multi_utri.getH()

        # We also need to construct the right hand side of the equation.  This
        # should be a multiplication of the phasor^H with the signal
        RHS = phasor.flatten().conj() * np.tile(signal, n_orders_arg)

        x_multi = linalg.spsolve(B_multi, RHS[:, np.newaxis])
        x = 2 * x_multi.reshape(
            n_orders_arg, -1
        )  # Multiply by 2 to account for missing negative frequency components
    else:
        # This solves the single order approach.  If the user has put in multiple
        # orders, it will solve them all independently instead of combining them
        # into a single larger solve.
        x = np.zeros((n_orders_arg, n_samples), dtype=np.complex128)
        for i, (phasor_i, B_i) in enumerate(zip(phasor, B)):
            # We already have the left side of the equation B, now we just need the
            # right side of the equation, which is the phasor hermetian
            # times the signal elementwise-multiplied
            RHS = phasor_i.conj() * signal
            x[i] = 2 * linalg.spsolve(B_i, RHS)

    return_value = [np.real(x * phasor)]
    if return_amp_phs:
        return_value.extend([np.abs(x), np.angle(x)])
    if return_envelope:
        return_value.extend([x, phasor])
    if return_r:
        return_value.extend([r])
    if len(return_value) == 1:
        return return_value[0]
    else:
        return return_value


def vold_kalman_filter_generator(
    sample_rate,
    num_orders,
    block_size,
    overlap,
    filter_order=None,
    bandwidth=None,
    method=None,
    buffer_size_factor=3,
):
    """
    Extracts sinusoidal information using a Vold-Kalman Filter

    This uses an windowed-overlap-and-add process to solve for the signal while
    removing start and end effects of the filter.  Each time the generator is
    called, it will yield a further section of the results up until the overlap
    section.

    Parameters
    ----------
    sample_rate : float
        The sample rate of the signal in Hz.
    num_orders : int
        The number of orders that will be found in the signal
    block_size : int
        The size of the blocks used in the analysis.
    overlap : float, optional
        Fraction of the block size to overlap when computing the results.  If
        not specified, it will default to 0.15.
    filter_order : int, optional
        The order of the VK filter, which should be 1, 2, or 3. The default is
        2.  The low-pass filter roll-off is approximately -40 dB per times the
        filter order.
    bandwidth : ndarray, optional
        The prescribed bandwidth of the filter. This is related to the filter
        selectivity parameter `r` in the literature.  This will be broadcast to
        the same shape as the `arguments` argument.  The default is the sample
        rate divided by 1000.
    method : str, optional
        Can be set to either 'single' or 'multi'.  In a 'single' solution, each
        sinusoidal component will be solved independently without any coupling.
        This can be more efficient, but will result in errors if the
        frequencies of the sine waves cross.  The 'multi' solution will solve
        for all sinusoidal components simultaneously, resulting in a better
        estimate of crossing frequencies. The default is 'multi'.
    buffer_size_factor : int, optional
        Specifies the size of the buffer.  buffer_size_factor * block_size is
        the size of the buffer.

    Raises
    ------
    ValueError
        If arguments are not the correct size or values.
    ValueError
        If data is provided subsequently to specifying last_signal = True

    Sends
    -----
    xi : iterable
        The next block of the signal to be filtered.  This should be a 1D
        signal containing sinusoidal components that need to be extracted.
    argsi : iterable
        A 2D array consisting of the arguments to the sinusoidal components of
        the form exp(1j*argsi).  This is the integral over time of the
        angular frequency, which can be approximated as
        2*np.pi*scipy.integrate.cumulative_trapezoid(frequencies,timesteps,initial=0)
        if frequencies is the frequency at each time step in Hz timesteps is
        the vector of time steps in seconds.  This is a 2D array where the
        number of rows is the
        number of different sinusoidal components that are desired to be
        extracted, and the number of columns are the number of time steps in
        the `signal` argument.
    last_signal : bool
        If True, the remainder of the data will be returned and the
        overlap-and-add process will be finished.

    Yields
    -------
    reconstructed_signals : ndarray
        Returns a time history the same size as `signal` for each of the
        sinusoidal components solved for.
    reconstructed_amplitudes : ndarray
        Returns the amplitude over time for each of the sinusoidal components
        solved for.  Only returned if return_amp_phs is True.
    reconstructed_phases : ndarray
        Returns the phase over time for each of the sinusoidal components
        solved for.  Only returned if return_amp_phs is True.

    """
    previous_envelope = None
    reconstructed_signals = None
    reconstructed_amplitudes = None
    reconstructed_phases = None
    overlap_samples = int(overlap * block_size)
    window = windows.hann(overlap_samples * 2, False)
    start_window = window[:overlap_samples]
    end_window = window[overlap_samples:]
    buffer = CircularBufferWithOverlap(
        buffer_size_factor * block_size,
        block_size,
        overlap_samples,
        data_shape=(num_orders + 1,),
    )
    first_output = True
    last_signal = False
    while True:
        xi, argsi, check_last_signal = (
            yield reconstructed_signals,
            reconstructed_amplitudes,
            reconstructed_phases,
        )
        if last_signal and check_last_signal:
            raise ValueError("Generator has been exhausted.")
        last_signal = check_last_signal
        argsi = np.atleast_2d(argsi)
        buffer_data = np.concatenate([xi[np.newaxis], argsi])
        # print(f"{buffer_data.shape=}")
        buffer_output = buffer.write_get_data(buffer_data, last_signal)
        if buffer_output is not None:
            # print(f"{buffer_output.shape=}")
            if first_output:
                buffer_output = buffer_output[..., overlap_samples:]
                first_output = False
                signal = buffer_output[0]
                arguments = buffer_output[1:]
            else:
                signal = buffer_output[0]
                arguments = buffer_output[1:]
                signal[:overlap_samples] = signal[:overlap_samples] * start_window
            if not last_signal:
                signal[-overlap_samples:] = signal[-overlap_samples:] * end_window
            # print(f"{signal.shape=}")
            # Do the VK Filtering
            _, vk_envelope, vk_phasor = vold_kalman_filter(
                sample_rate,
                signal,
                arguments,
                filter_order,
                bandwidth,
                method,
                return_envelope=True,
            )
            # print(f"{vk_signal.shape=}")
            # If necessary, do the overlap
            if previous_envelope is not None:
                vk_envelope[..., :overlap_samples] = (
                    vk_envelope[..., :overlap_samples] + previous_envelope[..., -overlap_samples:]
                )
            if not last_signal:
                reconstructed_signals = np.real(
                    vk_envelope[..., :-overlap_samples] * vk_phasor[..., :-overlap_samples]
                )
                reconstructed_amplitudes = np.abs(vk_envelope[..., :-overlap_samples])
                reconstructed_phases = np.angle(vk_envelope[..., :-overlap_samples])
            else:
                reconstructed_signals = np.real(vk_envelope * vk_phasor)
                reconstructed_amplitudes = np.abs(vk_envelope)
                reconstructed_phases = np.angle(vk_envelope)
            previous_envelope = vk_envelope
        else:
            # print(f"{buffer_output=}")
            reconstructed_signals = None
            reconstructed_amplitudes = None
            reconstructed_phases = None


class CircularBufferWithOverlap:
    """
    A Circular buffer that allows data to be added and removed
    from the buffer with overlap
    """

    def __init__(self, buffer_size, block_size, overlap_size, dtype="float", data_shape=()):
        """Initialize the circular buffer

        Parameters
        ----------
        buffer_size : int
            The total size of the circular buffer
        block_size : int
            The number of samples written or read in each block
        overlap_size : int
            The number of samples from the previous block to include in the read.
        dtype : dtype, optional
            The type of data in the buffer. The default is "float".
        data_shape : tuple, optional
            The shape of data in the buffer. The default is ().
        """
        self.buffer_size = buffer_size
        self.block_size = block_size
        self.overlap_size = overlap_size
        self.buffer = np.zeros(
            tuple(data_shape) + (buffer_size,), dtype=dtype
        )  # Initialize buffer with zeros
        self.buffer_read = np.ones((buffer_size,), dtype=bool)
        self.write_index = 0  # Index where the next block will be written
        self.read_index = 0  # Index where the next block will be read from
        self.debug = False
        if self.debug:
            self.report_buffer_state()

    def report_buffer_state(self):
        """Prints the current buffer state"""
        read_samples = np.sum(self.buffer_read)
        write_samples = self.buffer_size - read_samples
        print(f"{read_samples} of {self.buffer_size} have been read")
        print(f"{write_samples} of {self.buffer_size} have been written but not read")

    def write_get_data(self, data, read_remaining=False):
        """
        Writes a block of data and then returns a block if available

        Parameters:
        - data: Array to write to the buffer.
        """
        self.write(data)
        try:
            return self.read(read_remaining)
        except ValueError:
            return None

    def write(self, data):
        """
        Write a block of data to the circular buffer.

        Parameters:
        - data: Array to write to the buffer.
        """
        # Compute the end index for the write operation
        indices = (
            np.arange(self.write_index, self.write_index + data.shape[-1] + self.overlap_size)
            % self.buffer_size
        )

        if np.any(~self.buffer_read[indices]):
            raise ValueError(
                "Overwriting data on buffer that has not been read.  "
                "Read data before writing again."
            )

        self.buffer[..., indices[: None if self.overlap_size == 0 else -self.overlap_size]] = data
        self.buffer_read[indices[: None if self.overlap_size == 0 else -self.overlap_size]] = False

        # Update the write index
        self.write_index = (self.write_index + data.shape[-1]) % self.buffer_size

        if self.debug:
            print("Wrote Data to Buffer")
            self.report_buffer_state()
        # print(self.buffer)
        # print(self.buffer_read)

    def read(self, read_remaining=False):
        """
        Reads data from the buffer

        Parameters
        ----------
        read_remaining : bool, optional
            If true, read everything left on the buffer that hasn't yet been
            read. The default is False.

        Raises
        ------
        ValueError
            If there is not a block of data on the buffer and data would be
            read a second time.

        Returns
        -------
        return_data : ndarray
            A block of data read from the buffer.

        """
        indices = (
            np.arange(self.read_index - self.overlap_size, self.read_index + self.block_size)
            % self.buffer_size
        )
        if read_remaining:
            # Pick out just the indices that are ok to read
            # print('Reading Remaining:')
            # print(f"{indices.copy()=}")
            indices = np.concatenate(
                (
                    indices[: self.overlap_size],
                    indices[self.overlap_size :][~self.buffer_read[indices[self.overlap_size :]]],
                )
            )
            # print(f"{indices.copy()=}")
        if np.any(self.buffer_read[indices[self.overlap_size :]]):
            raise ValueError("Data would be read multiple times.  Write data before reading again.")
        return_data = self.buffer[..., indices]
        self.buffer_read[indices[self.overlap_size :]] = True
        self.read_index = (
            self.read_index + (return_data.shape[-1] - self.overlap_size)
        ) % self.buffer_size
        if self.debug:
            print("Read Data from Buffer")
            self.report_buffer_state()
        # print(self.buffer)
        # print(self.buffer_read)
        return return_data


class SineSpecification:
    """A class representing a sine specification"""

    def __init__(
        self,
        name,
        start_time,
        num_control,
        num_breakpoints=None,
        frequency_breakpoints=None,
        amplitude_breakpoints=None,
        phase_breakpoints=None,
        sweep_type_breakpoints=None,
        sweep_rate_breakpoints=None,
        warning_breakpoints=None,
        abort_breakpoints=None,
    ):
        """
        Initializes the sine specification

        Parameters
        ----------
        name : str
            Name of the sine tone.
        start_time : float
            The starting time of the sine tone.
        num_control : int
            The number of control channels in the specification.
        num_breakpoints : int, optional
            The number of frequency breakpoints in the specification.  Either
            this or frequency_breakpoints must be specified.
        frequency_breakpoints : ndarray, optional
            The frequency breakpoints in the specification.  Either this or
            num_breakpoints must be specified.
        amplitude_breakpoints : ndarray, optional
            The amplitude breakpoints of the specification, with shape num_freq,
            num_channels. If not specified, amplitudes will be 0.
        phase_breakpoints : ndarray, optional
            The phase breakpoints of the specification, with shape num_freq,
            num_channels. If not specified, phase will be 0.  Phases should be
            in radians.
        sweep_type_breakpoints : ndarray, optional
            Should be a 0 if linear sweep or 1 if logarithmic sweep.  Linear if
            not specified.
        sweep_rate_breakpoints : ndarray, optional
            Sweep rate at each breakpoint.  Hz/s if linear and oct/min if
            logarithmic sweep
        warning_breakpoints : ndarray, optional
            A 4D array of warning amplitudes with shape num_freq, 2, 2,
            num_channels.  The second dimension uses the 0 index for the lower
            warning limit and the 1 index for the upper warning limit.  The
            third dimension uses the 0 index for the "left" or "lower frequency"
            side of the breakpoint and 1 for the "right" or "higher frequency"
            side of the breakpoing.  If not specified, no warnings will be
            specified.
        abort_breakpoints : ndarray, optional
            A 4D array of abort amplitudes with shape num_freq, 2, 2,
            num_channels.  The second dimension uses the 0 index for the lower
            warning limit and the 1 index for the upper warning limit.  The
            third dimension uses the 0 index for the "left" or "lower frequency"
            side of the breakpoint and 1 for the "right" or "higher frequency"
            side of the breakpoing.  If not specified, no aborts will be
            specified.

        Raises
        ------
        ValueError
            if not one of frequency_breakpoints or num_breakpoints is specified
        """
        spec_dtype = [
            ("frequency", "f8"),
            ("amplitude", "f8", (num_control,)),
            ("phase", "f8", (num_control,)),
            ("sweep_type", "u1"),
            ("sweep_rate", "f8"),
            ("warning", "f8", (2, 2, num_control)),
            ("abort", "f8", (2, 2, num_control)),
        ]
        if frequency_breakpoints is None and num_breakpoints is None:
            raise ValueError("Must specify either number of breakpoints or breakpoint frequencies.")
        if frequency_breakpoints is None:
            self.breakpoint_table = np.zeros(num_breakpoints, dtype=spec_dtype)
        else:
            self.breakpoint_table = np.zeros(frequency_breakpoints.shape[0], dtype=spec_dtype)
            self.breakpoint_table["frequency"] = frequency_breakpoints
        if amplitude_breakpoints is not None:
            self.breakpoint_table["amplitude"] = amplitude_breakpoints
        if phase_breakpoints is not None:
            self.breakpoint_table["phase"] = phase_breakpoints  # Radians
        if sweep_type_breakpoints is not None:
            self.breakpoint_table["sweep_type"][:-1] = sweep_type_breakpoints
        if sweep_rate_breakpoints is not None:
            self.breakpoint_table["sweep_rate"][:-1] = sweep_rate_breakpoints
        if warning_breakpoints is not None:
            self.breakpoint_table["warning"] = warning_breakpoints
        else:
            self.breakpoint_table["warning"] = np.nan
        if abort_breakpoints is not None:
            self.breakpoint_table["abort"] = abort_breakpoints
        else:
            self.breakpoint_table["abort"] = np.nan
        self.start_time = start_time
        self.name = name

    def copy(self):
        """Creates a copy of the sine specification"""
        return SineSpecification(
            self.name,
            self.start_time,
            self.breakpoint_table["amplitude"].shape[-1],
            frequency_breakpoints=self.breakpoint_table["frequency"].copy(),
            amplitude_breakpoints=self.breakpoint_table["amplitude"].copy(),
            phase_breakpoints=self.breakpoint_table["phase"].copy(),
            sweep_type_breakpoints=self.breakpoint_table["sweep_type"][:-1].copy(),
            sweep_rate_breakpoints=self.breakpoint_table["sweep_rate"][:-1].copy(),
            warning_breakpoints=self.breakpoint_table["warning"].copy(),
            abort_breakpoints=self.breakpoint_table["abort"].copy(),
        )

    def create_signal(
        self,
        sample_rate,
        ramp_samples=0,
        control_index=None,
        ignore_start_time=False,
        only_breakpoints=False,
    ):
        """
        Creates a signal from the sine specification

        Parameters
        ----------
        sample_rate : float
            The sample rate of the signal generated
        ramp_samples : int, optional
            The number of samples to add to the start and end due to ramp up
            or ramp down. The default is 0.
        control_index : int, optional
            The channel index to generate a signal for.  If not specified,
            all channels will be generated.
        ignore_start_time : bool, optional
            If True, ignore the start time and have the sine sweep start
            immediately in the signal. The default is False.
        only_breakpoints : bool, optional
            If True, only generate values at the breakpoints. The default is
            False.

        Returns
        -------
        ordinate : ndarray
            The generated signals
        frequency : ndarray
            The instantaneous frequency at each time step
        argument : ndarray
            The instantaneous argument at each time step.
        amplitude : ndarray
            The instantaneous amplitude at each timestep.
        phase : ndarray
            The instantaneous phase at each timestep in radians.
        abscissa : ndarray
            The abscissa at each time step.
        start_index : int
            The sample at which the specification starts taking into account
            the start time and the ramp samples.
        end_index : int
            The sample at which the specification ends taking into accound the
            ramp samples.

        """
        # Convert octave per min to octave per second
        sweep_rates = self.breakpoint_table["sweep_rate"].copy()
        sweep_rates[self.breakpoint_table["sweep_type"] == 1] = (
            sweep_rates[self.breakpoint_table["sweep_type"] == 1] / 60
        )
        # Create the sweep types array
        sweep_types = [
            "lin" if sweep_type == 0 else "log"
            for sweep_type in self.breakpoint_table["sweep_type"][:-1]
        ]
        if control_index is None:
            ordinate = []
            amplitude = []
            phase = []
            for control_index in range(self.breakpoint_table["amplitude"].shape[-1]):
                (
                    this_ordinate,
                    argument,
                    frequency,
                    this_amplitude,
                    this_phase,
                    abscissa,
                ) = sine_sweep(
                    1 / sample_rate,
                    self.breakpoint_table["frequency"],
                    sweep_rates,
                    sweep_types,
                    self.breakpoint_table["amplitude"][:, control_index],
                    self.breakpoint_table["phase"][:, control_index],
                    return_frequency=True,
                    return_argument=True,
                    return_amplitude=True,
                    return_phase=True,
                    return_abscissa=True,
                    only_breakpoints=only_breakpoints,
                )
                ordinate.append(this_ordinate)
                amplitude.append(this_amplitude)
                phase.append(this_phase)
            ordinate = np.array(ordinate)
            amplitude = np.array(amplitude)
            phase = np.array(phase)
        else:
            ordinate, argument, frequency, amplitude, phase, abscissa = sine_sweep(
                1 / sample_rate,
                self.breakpoint_table["frequency"],
                sweep_rates,
                sweep_types,
                self.breakpoint_table["amplitude"][:, control_index],
                self.breakpoint_table["phase"][:, control_index],
                return_frequency=True,
                return_argument=True,
                return_amplitude=True,
                return_phase=True,
                return_abscissa=True,
                only_breakpoints=only_breakpoints,
            )

        if ignore_start_time:
            delay_samples = 0
        else:
            delay_samples = int(sample_rate * self.start_time)
        start_index = ramp_samples + delay_samples
        if ramp_samples > 0 or delay_samples > 0:
            # Create the pre-signal ramp
            begin_abscissa = np.arange(-ramp_samples - delay_samples, 0) / sample_rate
            begin_arguments = 2 * np.pi * frequency[0] * begin_abscissa + argument[0]
            begin_frequencies = np.ones(ramp_samples + delay_samples) * frequency[0]
            begin_amplitudes = (
                np.concatenate((np.zeros(delay_samples), np.linspace(0, 1, ramp_samples)))
                * amplitude[..., [0]]
            )
            begin_phases = np.ones(ramp_samples + delay_samples) * phase[..., [0]]
            begin_signal = begin_amplitudes * np.cos(begin_arguments + begin_phases)

            abscissa = np.concatenate((begin_abscissa, abscissa), axis=-1)
            ordinate = np.concatenate((begin_signal, ordinate), axis=-1)
            frequency = np.concatenate((begin_frequencies, frequency), axis=-1)
            amplitude = np.concatenate((begin_amplitudes, amplitude), axis=-1)
            phase = np.concatenate((begin_phases, phase), axis=-1)
            argument = np.concatenate((begin_arguments, argument), axis=-1)
        if ramp_samples > 0:
            end_abscissa = np.arange(1, ramp_samples + 1) / sample_rate
            end_arguments = 2 * np.pi * frequency[-1] * end_abscissa + argument[-1]
            end_frequencies = np.ones(ramp_samples) * frequency[-1]
            end_amplitudes = np.linspace(1, 0, ramp_samples) * amplitude[..., [-1]]
            end_phases = np.ones(ramp_samples) * phase[..., [-1]]
            end_signal = end_amplitudes * np.cos(end_arguments + end_phases)

            abscissa = np.concatenate((abscissa, end_abscissa), axis=-1)
            ordinate = np.concatenate((ordinate, end_signal), axis=-1)
            frequency = np.concatenate((frequency, end_frequencies), axis=-1)
            argument = np.concatenate((argument, end_arguments), axis=-1)
            amplitude = np.concatenate((amplitude, end_amplitudes), axis=-1)
            phase = np.concatenate((phase, end_phases), axis=-1)
        end_index = abscissa.shape[-1] - ramp_samples

        return (
            ordinate,
            frequency,
            argument,
            amplitude,
            phase,
            abscissa,
            start_index,
            end_index,
        )

    def interpolate_warning(self, channel_index, frequencies):
        """
        Interpolates the warning array at the specified frequencies

        Parameters
        ----------
        channel_index : int
            The channel to compute the warning levels with.
        frequencies : ndarray
            The frequencies at which to compute the warning levels.

        Returns
        -------
        warning_levels
            A 2 x num_frequencies array containing the warning levels.  The
            first index is the lower warning level and the second index is the
            upper warning level.  If warnings are not specified for certain
            values, they will be set to NaN.

        """
        abscissa = np.repeat(self.breakpoint_table["frequency"], 2)
        lower_ordinate = self.breakpoint_table["warning"][:, 0, :, channel_index].flatten()
        upper_ordinate = self.breakpoint_table["warning"][:, 1, :, channel_index].flatten()
        return np.array(
            [
                np.interp(frequencies, abscissa, lower_ordinate),
                np.interp(frequencies, abscissa, upper_ordinate),
            ]
        )

    def interpolate_abort(self, channel_index, frequencies):
        """
        Interpolates the abort array at the specified frequencies

        Parameters
        ----------
        channel_index : int
            The channel to compute the abort levels with.
        frequencies : ndarray
            The frequencies at which to compute the abort levels.

        Returns
        -------
        abort_levels
            A 2 x num_frequencies array containing the abort levels.  The
            first index is the lower warning level and the second index is the
            upper warning level.  If warnings are not specified for certain
            values, they will be set to NaN.

        """
        abscissa = np.repeat(self.breakpoint_table["frequency"], 2)
        lower_ordinate = self.breakpoint_table["abort"][:, 0, :, channel_index].flatten()
        upper_ordinate = self.breakpoint_table["abort"][:, 1, :, channel_index].flatten()
        return np.array(
            [
                np.interp(frequencies, abscissa, lower_ordinate),
                np.interp(frequencies, abscissa, upper_ordinate),
            ]
        )

    @staticmethod
    def structured_array_equal(arr1, arr2):
        """
        A method to check if two sine specification breakpoint tables are equal

        Parameters
        ----------
        arr1 : ndarray
            A structured array representing a breakpoint table.
        arr2 : ndarray
            A structured array representing a breakpoint table.

        Returns
        -------
        bool
            True if the two arrays are equal.  False otherwise.

        """
        if arr1.dtype != arr2.dtype:
            # print('DTypes Not Equal')
            return False
        for field in arr1.dtype.names:
            field1 = arr1[field]
            field2 = arr2[field]
            if not np.array_equal(field1, field2, equal_nan=True):
                # print(f'Field {field} Not Equal')
                # print(field1)
                # print(field2)
                return False
        return True

    def __eq__(self, other):
        """
        A method to check if two sine specifications are equal.

        Parameters
        ----------
        other : SineSpecification
            The SineSpecification object to compare against.

        Returns
        -------
        bool
            True if the two SineSpecification objects are equal.

        """
        if not SineSpecification.structured_array_equal(
            self.breakpoint_table, other.breakpoint_table
        ):
            return False
        if self.start_time != other.start_time:
            return False
        return True

    @staticmethod
    def create_combined_signals(specifications, sample_rate, ramp_samples, control_index=None):
        """
        Creates a combined signal from many specifications

        Parameters
        ----------
        specifications : list of SineSpecification objects
            The various specification objects to combine together to creat the
            combined signals.
        sample_rate : float
            The sample rate at which the signals should be generated.
        ramp_samples : int
            The number of samples to use in the ramp portion of the signals.
        control_index : int, optional
            The control channel index at which the signals should be generated.
            The default is to generate all channels.

        Returns
        -------
        signal : ndarray
            A number of channels by number of timesteps array of time history
            values.
        order_signals : ndarray
            Separate signals for each of the specification tones, in a
            num_tones, num_channels, num_timesteps array.
        order_frequencies : ndarray
            The frequencies associated with each sine tone at each time step
            in a num_tones, num_timesteps array.
        order_arguments : ndarray
            The arguments associated with each sine tone at each time step in a
            num_tones, num_timestep array
        order_amplitudes : ndarray
            The instantaneous amplitudes associated with each sine tone and
            channel at each time step in a num_tones, num_channels,
            num_timesteps shaped array
        order_phases : ndarray
            The instantaneous phase associated with each sine tone and
            channel at each time step in a num_tones, num_channels,
            num_timesteps shaped array.  Phases in Radians.
        order_start_samples : ndarray
            The starting sample for each tone taking into account the start time
            and the ramp samples
        order_end_samples : ndarray
            The end sample for each tone taking into account the ramp samples.

        """
        order_signals = []
        order_arguments = []
        order_frequencies = []
        order_amplitudes = []
        order_phases = []
        order_start_samples = []
        order_end_samples = []
        longest_signal = 0
        for spec in specifications:
            (
                ordinate,
                frequency,
                argument,
                amplitude,
                phase,
                _,
                start_index,
                end_index,
            ) = spec.create_signal(sample_rate, ramp_samples, control_index)
            order_signals.append(ordinate)
            order_frequencies.append(frequency)
            order_amplitudes.append(amplitude)
            order_phases.append(phase)
            order_arguments.append(argument)
            order_start_samples.append(start_index)
            order_end_samples.append(end_index)

            if order_signals[-1].shape[-1] > longest_signal:
                longest_signal = order_signals[-1].shape[-1]

        # Now that we know the longest signals, we know how much we need to pad
        # to make all signals the same length
        for i, signal in enumerate(order_signals):
            extra_samples = longest_signal - signal.shape[-1]
            end_abscissa = np.arange(1, extra_samples + 1) / sample_rate
            end_arguments = (
                2 * np.pi * order_frequencies[i][-1] * end_abscissa + order_arguments[i][-1]
            )
            end_frequencies = np.ones(extra_samples) * order_frequencies[i][-1]
            end_amplitudes = np.zeros((extra_samples)) * order_amplitudes[i][..., [-1]]
            end_phases = np.ones(extra_samples) * order_phases[i][..., [-1]]
            end_signal = np.zeros(extra_samples) * signal[..., [-1]]
            order_signals[i] = np.concatenate((order_signals[i], end_signal), axis=-1)
            order_frequencies[i] = np.concatenate((order_frequencies[i], end_frequencies), axis=-1)
            order_arguments[i] = np.concatenate((order_arguments[i], end_arguments), axis=-1)
            order_amplitudes[i] = np.concatenate((order_amplitudes[i], end_amplitudes), axis=-1)
            order_phases[i] = np.concatenate((order_phases[i], end_phases), axis=-1)

        order_signals = np.array(order_signals)
        order_frequencies = np.array(order_frequencies)
        order_arguments = np.array(order_arguments)
        order_amplitudes = np.array(order_amplitudes)
        order_phases = np.array(order_phases)
        order_start_samples = np.array(order_start_samples)
        order_end_samples = np.array(order_end_samples)
        signal = np.sum(order_signals, axis=0)

        return (
            signal,
            order_signals,
            order_frequencies,
            order_arguments,
            order_amplitudes,
            order_phases,
            order_start_samples,
            order_end_samples,
        )
