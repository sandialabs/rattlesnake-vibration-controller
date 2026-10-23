# -*- coding: utf-8 -*-
"""
Default control law for the sine environment.

This module is loaded by name (see DEFAULT_CONTROL_SCRIPT in
sine_sys_id_environment.py) when no custom control script is specified.

@author: dprohe
"""

import os

import numpy as np

from rattlesnake.environment.sine_sys_id_utilities import SineSpecification

DEBUG = False

if DEBUG:
    import pickle


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
        self.frf_frequencies = self.frf_frequency_spacing * np.arange(
            self.frfs.shape[0]
        )
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
            filename = os.path.join(
                self.harddisk_storage, "preshaped_drive_amplitudes.mmap"
            )
            shape = (
                self.specified_frequency.shape[0],
                self.frfs.shape[-1],
                self.specified_frequency.shape[-1],
            )
            self.preshaped_drive_amplitudes = np.memmap(
                filename, dtype=float, shape=shape, mode="w+"
            )
            filename = os.path.join(
                self.harddisk_storage, "preshaped_drive_phases.mmap"
            )
            shape = (
                self.specified_frequency.shape[0],
                self.frfs.shape[-1],
                self.specified_frequency.shape[-1],
            )
            self.preshaped_drive_phases = np.memmap(
                filename, dtype=float, shape=shape, mode="w+"
            )
            filename = os.path.join(
                self.harddisk_storage, "largest_correction_factors.mmap"
            )
            shape = self.specified_frequency.shape
            self.largest_correction_factors = np.memmap(
                filename, dtype=float, shape=shape, mode="w+"
            )
            filename = os.path.join(self.harddisk_storage, "interpolated_frf_pinv.mmap")
            shape = (
                self.specified_frequency.shape[0],
                self.specified_frequency.shape[-1],
            ) + self.frf_pinv.shape[-2:]
            self.interpolated_frf_pinv = np.memmap(
                filename, dtype="c16", shape=shape, mode="w+"
            )
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
                1
                / np.interp(
                    control_freq, self.frf_frequencies, self.max_singular_values
                )
                ** 2
            )
            # print('Computing Complex Response')
            complex_response = np.moveaxis(
                control_amp * np.exp(1j * control_phs), -1, 0
            )[  # Radians
                ..., np.newaxis
            ]
            # print('Computing Complex Excitation')
            complex_excitation = np.moveaxis(
                (
                    self.interpolated_frf_pinv[tone_index, control_slice]
                    @ complex_response
                )[..., 0],
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
            self.specified_argument[:, np.newaxis, :]
            + self.preshaped_drive_phases  # Radians
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
        ramp_down_start = (
            block_start - self.end_index + self.start_index + self.ramp_samples
        )
        if ramp_down_start < 0:
            ramp_down_start = 0
        ramp_down_end = (
            block_end - self.end_index + self.start_index + self.ramp_samples
        )
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
            self.start_index
            + block_start : self.start_index
            + block_start
            + amplitudes.shape[-1],
        ]
        arguments = self.specified_argument[
            self.control_tones,
            self.start_index
            + block_start : self.start_index
            + block_start
            + amplitudes.shape[-1],
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
        ramp_down_start = (
            block_start - self.end_index + self.start_index + self.ramp_samples
        )
        if ramp_down_start < 0:
            ramp_down_start = 0
        ramp_down_end = (
            block_end - self.end_index + self.start_index + self.ramp_samples
        )
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
            self.start_index
            + block_start : self.start_index
            + block_start
            + amplitudes.shape[-1],
        ]
        arguments = self.specified_argument[
            self.control_tones,
            self.start_index
            + block_start : self.start_index
            + block_start
            + amplitudes.shape[-1],
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
        self.end_index = (
            self.preshaped_drive_signals.shape[-1] if end_index is None else end_index
        )

        if DEBUG:
            print("Writing Sine Debug Pickle")
            with open(
                "debug_data/sine_control_law_initialize_control_debug.pkl", "wb"
            ) as f:
                pickle.dump(self, f)
            print("Done!")

        # Set up the analysis and write_indices
        self.control_analysis_index = 0
        self.control_write_index = (
            self.ramp_samples + self.buffer_blocks * self.block_size
        )

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
            * np.cos(
                starting_drive_phases + starting_arguments[:, np.newaxis, :]
            ),  # Radians
            axis=0,
        )

        # Set up control parameters
        self.control_drive_correction = np.zeros(
            starting_drive_amplitudes.shape[:2], dtype=complex
        )

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
            with open(
                "debug_data/sine_control_law_update_control_debug.pkl", "wb"
            ) as f:
                pickle.dump(self, f)
            print("Done!")

        # Find the equivalent block in the signal
        block_start_index = self.control_analysis_index
        block_end_index = (
            control_signals.shape[-1] * self.output_oversample
            + self.control_analysis_index
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
            complex_targets = target_response_amplitudes[
                ..., :: self.output_oversample
            ] * np.exp(1j * target_response_phases[..., :: self.output_oversample])
            complex_achieved = control_amplitudes * np.exp(
                1j * control_phases
            )  # Radians
            complex_error = (
                complex_targets - complex_achieved
            )  # Number of Tones x Num Responses x Num Freqs
            block_correction_factor = self.convergence_factor * np.min(
                self.largest_correction_factors[
                    self.control_tones, ..., reduction_slice
                ],
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
            with open(
                "debug_data/sine_control_law_generate_signal_debug.pkl", "wb"
            ) as f:
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
