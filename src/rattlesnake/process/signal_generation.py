# -*- coding: utf-8 -*-
"""
Rattlesnake Vibration Control Software
Copyright (C) 2021  National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from abc import abstractmethod, ABC
from enum import Enum

import numpy as np
import scipy.signal as sig
from scipy.stats import norm


class SignalTypes(Enum):
    """Enumeration of valid types of signals we can generate"""

    RANDOM = 0
    PSEUDORANDOM = 1
    BURST_RANDOM = 2
    CHIRP = 3
    SINE = 4
    SQUARE = 5
    CPSD = 6
    TRANSIENT = 7
    CONTINUOUSTRANSIENT = 8


DEBUG = False

if DEBUG:
    from glob import glob

    FILE_OUTPUT = "debug_data/signal_generator_{:}.npz"


def cola(
    signal_samples: int,
    end_samples: int,
    signals: np.ndarray,
    window_name: str,
    window_exponent: float = 0.5,
):
    """
    Constant Overlap and Addition of signals to blend them together

    This function creates long signals of individual realizations by windowing,
    overlapping, and adding the signals together.

    Parameters
    ----------
    signal_samples : int
        Number of samples in the overlapped region of the signal.
    end_samples : int
        Number of samples in the rest of the signal.
    signals : np.ndarray
        3D numpy array where each of the two rows is a signal that will be
        windowed, overlapped, and added.
    window_name : str
        Name of the window function that will be used to window the signals.
    window_exponent : float
        Exponent on the window function.  Set to 0.5 for constant variance in
        the signals (Default Value = 0.5)

    Returns
    -------
    output : np.ndarray
        Combined signal that has been windowed, overlapped, and added.

    Notes
    -----
    Uses the constant overlap and add process described in [1]_

    .. [1] R. Schultz and G. Nelson, "Input signal synthesis for open-loop
       multiple-input/multiple-output testing," Proceedings of the International
       Modal Analysis Conference, 2019.

    """
    total_samples = signal_samples + end_samples
    if window_name == "tukey":
        window_name = ("tukey", 2 * (end_samples / total_samples))
    window = sig.get_window(window_name, total_samples, fftbins=True) ** window_exponent
    # Create the new signal
    last_signal, current_signal = signals
    output = current_signal[:, :signal_samples] * window[:signal_samples]
    if end_samples > 0:
        output[:, :end_samples] += np.array(last_signal)[:, -end_samples:] * window[-end_samples:]
    return output


def cpsd_to_time_history(cpsd_matrix, sample_rate, df, output_oversample=1):
    """Generates a time history realization from a CPSD matrix

    Parameters
    ----------
    cpsd_matrix : np.ndarray :
        A 3D complex np.ndarray representing a CPSD matrix where the first
        dimension is the frequency line and the second two dimensions are the
        rows and columns of the matrix at each frequency line.
    sample_rate: float :
        The sample rate of the controller in samples per second
    df : float :
        The frequency spacing of the cpsd matrix


    Returns
    -------
    output : np.ndarray :
        A numpy array containing the generated signals

    Notes
    -----
    Uses the process described in [1]_

    .. [1] R. Schultz and G. Nelson, "Input signal synthesis for open-loop
       multiple-input/multiple-output testing," Proceedings of the International
       Modal Analysis Conference, 2019.

    """
    # pylint: disable=invalid-name
    # svd_start = time()
    # Compute SVD broadcasting over all frequency lines
    [U, S, Vh] = np.linalg.svd(cpsd_matrix, full_matrices=False)
    # svd_end = time()
    # print('SVD Time: {:0.4f}'.format(svd_end-svd_start))
    # Reform using the sqrt of the S matrix
    Lsvd = U * np.sqrt(S[:, np.newaxis, :]) @ Vh
    # Compute Random Process
    W = np.sqrt(0.5) * (
        np.random.randn(*cpsd_matrix.shape[:-1], 1)
        + 1j * np.random.randn(*cpsd_matrix.shape[:-1], 1)
    )
    Xv = 1 / np.sqrt(df) * Lsvd @ W
    # Ensure that the signal is real by setting the nyquist and DC component to 0
    Xv[[0, -1], :, :] = 0
    # Compute the IFFT, using the real version makes it so you don't need negative frequencies
    zero_padding = np.zeros(
        [(output_oversample - 1) * (Xv.shape[0] - 1)] + list(Xv.shape[1:]),
        dtype=Xv.dtype,
    )
    # ifft_start = time()
    xv = (
        np.fft.irfft(np.concatenate((Xv, zero_padding), axis=0) / np.sqrt(2), axis=0)
        * output_oversample
        * sample_rate
    )
    # ifft_end = time()
    # print('FFT Time: {:0.4f}'.format(ifft_end-ifft_start))
    output = xv[:, :, 0].T
    return output


class SignalGenerator(ABC):
    """Abstract base class showing required methods of each signal generator class"""

    @abstractmethod
    def generate_frame(self):
        """This method generates and returns a frame of data from the signal generator"""

    # TODO: update the parameters passing approach to take one argument (like a dictionary)
    # so calling signature is maintained in child classes.
    def update_parameters(self):
        """This method accepts various arguments to update the parameters of the signal generator"""

    @property
    @abstractmethod
    def ready_for_next_output(self):
        """This method returns True if the signal generator can currently produce a frame of data"""


class RandomSignalGenerator(SignalGenerator):
    """Signal generator that produces true random signals using a COLA procedure"""

    def __init__(
        self,
        rms,
        sample_rate,
        num_samples_per_frame,
        num_signals,
        low_frequency_cutoff,
        high_frequency_cutoff,
        cola_overlap,
        cola_window,
        cola_exponent,
        output_oversample,
    ):
        self.rms = rms
        self.sample_rate = sample_rate
        self.num_samples = num_samples_per_frame
        self.num_signals = num_signals
        self.low_frequency_cutoff = 0 if low_frequency_cutoff is None else low_frequency_cutoff
        self.high_frequency_cutoff = (
            sample_rate / 2 if high_frequency_cutoff is None else high_frequency_cutoff
        )
        self.cola_overlap = cola_overlap
        self.cola_window = cola_window.lower()
        self.cola_exponent = cola_exponent
        self.output_oversample = output_oversample
        self.cola_queue = np.zeros((2, self.num_signals, self.num_samples * self.output_oversample))

        # Set up the queue the first time
        self.generate_frame()

    @property
    def samples_per_output(self):
        """Property returning the samples per output given the COLA overlap"""
        return int(self.num_samples * (1 - self.cola_overlap))

    @property
    def overlapped_output_samples(self):
        """Property returning the number of output samples that are overlapped."""
        return self.num_samples - self.samples_per_output

    @property
    def ready_for_next_output(self):
        """Random signals are always ready to produce next outputs"""
        return True

    def generate_frame(self):
        """Generates a random frame of data and overlaps and adds it to the previous data"""
        # Create a signal
        signal = self.rms * np.random.randn(
            self.num_signals, self.num_samples * self.output_oversample
        )
        # Band limit it
        fft = np.fft.rfft(signal, axis=-1)
        freq = np.fft.rfftfreq(signal.shape[-1], 1 / (self.sample_rate * self.output_oversample))
        invalid_frequencies = (freq < self.low_frequency_cutoff) | (
            freq > self.high_frequency_cutoff
        )
        scale_factor = (~invalid_frequencies).sum() / len(invalid_frequencies)
        fft[..., invalid_frequencies] = 0
        bandlimited_signal = np.fft.irfft(fft) / np.sqrt(scale_factor)
        # Roll the queue
        self.cola_queue = np.roll(self.cola_queue, -1, axis=0)
        self.cola_queue[-1, ...] = bandlimited_signal
        output_signal = cola(
            self.samples_per_output * self.output_oversample,
            self.overlapped_output_samples * self.output_oversample,
            self.cola_queue,
            self.cola_window,
            self.cola_exponent,
        )
        return output_signal, False


class PseudorandomSignalGenerator(SignalGenerator):
    """Signal generator that produces a periodic signal that looks like a random signal"""

    def __init__(
        self,
        rms,
        sample_rate,
        num_samples_per_frame,
        num_signals,
        low_frequency_cutoff,
        high_frequency_cutoff,
        output_oversample,
    ):
        freq = np.fft.rfftfreq(
            num_samples_per_frame * output_oversample,
            1 / (sample_rate * output_oversample),
        )
        fft = np.zeros(
            (num_signals, num_samples_per_frame * output_oversample // 2 + 1),
            dtype=complex,
        )
        low_frequency_cutoff = 0 if low_frequency_cutoff is None else low_frequency_cutoff
        high_frequency_cutoff = (
            sample_rate / 2 if high_frequency_cutoff is None else high_frequency_cutoff
        )
        valid_frequencies = (freq >= low_frequency_cutoff) & (freq <= high_frequency_cutoff)
        fft[..., valid_frequencies] = np.exp(
            1j * 2 * np.pi * np.random.rand(num_signals, valid_frequencies.sum())
        )
        self.signal = np.fft.irfft(fft)
        signal_rms = np.sqrt(np.mean(self.signal**2, axis=-1, keepdims=True))
        self.signal *= rms / signal_rms

    def generate_frame(self):
        """Generates one realization of the periodic pseudorandom signal"""
        return self.signal.copy(), False

    @property
    def ready_for_next_output(self):
        """Pseudorandom signals are always ready, as they just repeat the frame"""
        return True


class BurstRandomSignalGenerator(SignalGenerator):
    """Signal generator that produces a burst random excitation signal"""

    def __init__(
        self,
        rms,
        sample_rate,
        num_samples_per_frame,
        num_signals,
        low_frequency_cutoff,
        high_frequency_cutoff,
        on_fraction,
        ramp_fraction,
        output_oversample,
    ):
        self.rms = rms
        self.sample_rate = sample_rate
        self.num_samples = num_samples_per_frame
        self.num_signals = num_signals
        self.low_frequency_cutoff = 0 if low_frequency_cutoff is None else low_frequency_cutoff
        self.high_frequency_cutoff = (
            sample_rate / 2 if high_frequency_cutoff is None else high_frequency_cutoff
        )
        self.on_fraction = on_fraction
        if ramp_fraction > 0.5:
            raise ValueError("ramp_fraction cannot be more that 0.5")
        self.ramp_fraction = ramp_fraction
        self.output_oversample = output_oversample

        self.envelope = np.zeros(self.num_samples * self.output_oversample)
        self.envelope[: self.ramp_samples] = np.linspace(0, 1, self.ramp_samples)
        self.envelope[self.ramp_samples : self.ramp_samples + self.on_samples] = 1
        self.envelope[
            self.ramp_samples + self.on_samples : self.ramp_samples * 2 + self.on_samples
        ] = np.linspace(1, 0, self.ramp_samples)

    @property
    def ramp_samples(self):
        """Property computing how many samples are in the ramp-up and ramp-down of the burst"""
        return int(
            self.num_samples * self.output_oversample * self.on_fraction * self.ramp_fraction
        )

    @property
    def on_samples(self):
        """Property computing how many samples the burst is active for"""
        return int(
            self.num_samples * self.output_oversample * self.on_fraction - 2 * self.ramp_samples
        )

    @property
    def ready_for_next_output(self):
        """Burst random is always ready for the next output"""
        return True

    def generate_frame(self):
        """Generates one burst cycle of data"""
        # Create a signal
        signal = self.rms * np.random.randn(
            self.num_signals, self.num_samples * self.output_oversample
        )
        # Band limit it
        fft = np.fft.rfft(signal, axis=-1)
        freq = np.fft.rfftfreq(signal.shape[-1], 1 / (self.sample_rate * self.output_oversample))
        invalid_frequencies = (freq < self.low_frequency_cutoff) | (
            freq > self.high_frequency_cutoff
        )
        scale_factor = (~invalid_frequencies).sum() / len(invalid_frequencies)
        fft[..., invalid_frequencies] = 0
        bandlimited_signal = np.fft.irfft(fft) / np.sqrt(scale_factor)
        return bandlimited_signal * self.envelope, False


class ChirpSignalGenerator(SignalGenerator):
    """Signal generator that generates a periodic fast sine sweep from low to high frequency"""

    def __init__(
        self,
        level,
        sample_rate,
        num_samples_per_frame,
        num_signals,
        low_frequency_cutoff,
        high_frequency_cutoff,
        output_oversample,
    ):
        times = np.arange(num_samples_per_frame * output_oversample) / (
            sample_rate * output_oversample
        )
        signal_length = num_samples_per_frame / sample_rate
        n_cycles = np.ceil(high_frequency_cutoff * signal_length)
        high_frequency_cutoff = n_cycles / signal_length
        frequency_slope = (high_frequency_cutoff - low_frequency_cutoff) / signal_length
        argument = frequency_slope / 2 * times**2 + low_frequency_cutoff * times
        self.signal = np.tile(level * np.sin(2 * np.pi * argument), (num_signals, 1))

    def generate_frame(self):
        """Generates a single realization of the sweep"""
        return self.signal.copy(), False

    @property
    def ready_for_next_output(self):
        """Chirp signals are always ready for next output, as they just repeat the same signal"""
        return True


class SineSignalGenerator(SignalGenerator):
    """Signal generator that produces stationary sine signals and tracks the instantaneous phase"""

    def __init__(
        self,
        level,
        sample_rate,
        num_samples_per_frame,
        num_signals,
        frequency,
        phase,
        output_oversample,
    ):
        self.level = np.broadcast_to(level, (num_signals, 1)).copy()
        self.sample_rate = sample_rate
        self.num_samples = num_samples_per_frame
        self.num_signals = num_signals
        self.frequency = None if frequency is None else np.array(frequency, dtype=float)
        self.phase = None if phase is None else np.array(phase, dtype=float)
        self.output_oversample = output_oversample
        self.times = np.arange(self.num_samples * self.output_oversample) / (
            self.sample_rate * self.output_oversample
        )

    @property
    def phase_per_sample(self):
        """Property computing the phase change per sample"""
        return 2 * np.pi * self.frequency / self.sample_rate

    @property
    def phase_per_frame(self):
        """Property computing the phase change per frame"""
        return self.phase_per_sample * self.num_samples

    @property
    def ready_for_next_output(self):
        """Sine signals are ready for output if all parameters are defined"""
        return self.frequency is not None and self.phase is not None

    def update_parameters(self, frequency, level, phase=None):
        """Updates the parameters of the sinusoidal signal

        Parameters
        ----------
        frequency : np.ndarray
            The new frequencies to use for the sine waves
        level : np.ndarray
            The new amplitudes to use for the sine waves
        phase : np.ndarray, optional
            The new phases to use for the sine waves.  If not specified, it will not be updated

        Notes
        -----
        All parameters broadcast when creating the sine signals.  The time dimension will be
        appended to each of these parameters as a new axis.  However, they must consistently
        broadcast with one another.
        """
        self.frequency[...] = frequency
        self.level[...] = level
        if phase is not None:
            self.phase[...] = phase

    def generate_frame(self):
        """Generates a frame of sine data while tracking the phase change"""
        signal = self.level * np.sin(
            2 * np.pi * self.frequency[..., np.newaxis] * self.times + self.phase[..., np.newaxis]
        )
        self.phase += self.phase_per_frame
        return signal, False


class SquareSignalGenerator(SignalGenerator):
    """Signal generator that produces a square wave and tracks the instantaneous phase"""

    def __init__(
        self,
        level,
        sample_rate,
        num_samples_per_frame,
        num_signals,
        frequency,
        phase,
        on_fraction,
        output_oversample,
    ):
        self.level = np.broadcast_to(level, (num_signals, 1)).copy()
        self.sample_rate = sample_rate
        self.num_samples = num_samples_per_frame
        self.num_signals = num_signals
        self.frequency = None if frequency is None else np.array(frequency, dtype=float)
        self.phase = None if phase is None else np.array(phase, dtype=float)
        self.on_fraction = on_fraction
        self.output_oversample = output_oversample
        self.times = np.arange(self.num_samples * self.output_oversample) / (
            self.sample_rate * self.output_oversample
        )

    @property
    def phase_per_sample(self):
        """Computes the phase change per sample based on the frequency"""
        return 2 * np.pi * self.frequency / self.sample_rate

    @property
    def phase_per_frame(self):
        """Computes the phase change per frame based on frequency and number of samples"""
        return self.phase_per_sample * self.num_samples

    @property
    def ready_for_next_output(self):
        """Square waves are ready for output as long as frequency and phase are defined"""
        return self.frequency is not None and self.phase is not None

    def update_parameters(self, frequency, phase=None):
        """Updates the parameters of the square wave signal

        Parameters
        ----------
        frequency : np.ndarray
            The new frequencies to use for the square waves
        phase : np.ndarray, optional
            The new phases to use for the square waves.  If not specified, it will not be updated

        Notes
        -----
        All parameters broadcast when creating the sine signals.  The time dimension will be
        appended to each of these parameters as a new axis.  However, they must consistently
        broadcast with one another.
        """
        self.frequency[...] = frequency
        if phase is not None:
            self.phase[...] = phase

    def generate_frame(self):
        """Generates a frame of data while tracking the instantaneous phase change"""
        signal = self.level * (
            2
            * (
                (
                    (
                        2 * np.pi * self.frequency[..., np.newaxis] * self.times
                        + self.phase[..., np.newaxis]
                    )
                    % (2 * np.pi)
                )
                < 2 * np.pi * self.on_fraction
            ).astype(int)
            - 1
        )
        self.phase += self.phase_per_frame
        return signal, False


class CPSDSignalGenerator(SignalGenerator):
    """Signal generator that generates time histories satisfying a prescribed CPSD matrix"""

    def __init__(
        self,
        sample_rate,
        num_samples_per_frame,
        num_signals,
        cpsd_matrix,
        cola_overlap,
        cola_window,
        cola_exponent,
        sigma_clip,
        output_oversample,
    ):
        self.sample_rate = sample_rate
        self.num_samples = num_samples_per_frame
        self.num_signals = num_signals
        if sigma_clip is None:
            self.sigma_clip = None
        elif isinstance(sigma_clip, np.ndarray):
            self.sigma_clip = sigma_clip.squeeze()[:, np.newaxis]  # force to n x 1 array
            if np.all(self.sigma_clip >= 5.0):
                self.sigma_clip = None
        elif isinstance(sigma_clip, (int, float)):
            self.sigma_clip = sigma_clip
            if self.sigma_clip >= 5.0:
                self.sigma_clip = None
        self.update_parameters(cpsd_matrix)
        self.cola_overlap = cola_overlap
        self.cola_window = cola_window.lower()
        self.cola_exponent = cola_exponent
        self.output_oversample = output_oversample
        self.cola_queue = np.zeros((2, self.num_signals, self.num_samples * self.output_oversample))
        self.cola_initialized = False

    @property
    def samples_per_output(self):
        """Property returning the samples per output given the COLA overlap"""
        return int(self.num_samples * (1 - self.cola_overlap))

    @property
    def overlapped_output_samples(self):
        """Property returning the number of output samples that are overlapped."""
        return self.num_samples - self.samples_per_output

    @property
    def frequency_spacing(self):
        """Property returning frequency line spacing given the sampling parameters"""
        return self.sample_rate / self.num_samples

    @property
    def ready_for_next_output(self):
        """Ready for output as long as the CPSD matrix we are targetting is defined"""
        return self.cpsd_matrix is not None

    def update_parameters(self, cpsd_matrix):
        """Updates the CPSD target

        Parameters
        ----------
        cpsd_matrix : np.ndarray
            A 3D CPSD matrix that will be matched by the time histories being generated
        """
        # pylint: disable=invalid-name
        self.cpsd_matrix = cpsd_matrix
        if self.cpsd_matrix is None:
            self.Lsvd = None
            return
        # Determine rms and rescaling factors for sigma clipping (rescale factors used to
        # maintain rms levels when using low clipping thresholds)
        self._rms = (
            np.trapz(  # TODO: This is deprecated, fix to use Trapezoid
                self.cpsd_matrix.diagonal(axis1=1, axis2=2),
                np.arange(self.cpsd_matrix.shape[0]) * self.frequency_spacing,
                axis=0,
            )
            ** 0.5
        )[:, np.newaxis]
        if self.sigma_clip is None:
            self._scale_factor = None
        else:
            # this is based on a curve fit between clipping threshold and
            # rms error: [-1/(x + 0.5)^3 + 1] (less effective at lower clipping threshold)
            self._scale_factor = -1 / (self.sigma_clip + 0.5) ** 3 + 1
        self._size = (*self.cpsd_matrix.shape[:-1], 1)
        # svd_start = time()
        # Compute SVD broadcasting over all frequency lines
        [U, S, Vh] = np.linalg.svd(cpsd_matrix, full_matrices=False)
        # svd_end = time()
        # print('SVD Time: {:0.4f}'.format(svd_end-svd_start))
        # Reform using the sqrt of the S matrix
        self.Lsvd = U * np.sqrt(S[:, np.newaxis, :]) @ Vh

    def rejection_sample(self, size, threshold=None) -> np.ndarray:
        """Handles sigma clipping for the randomly generated signals"""
        # `size` should be (n_samples x n_channels x 1)
        # (this is the size needed to add to the cola queue)
        if threshold is None:
            return
        oversample = np.max(size[0] + np.ceil((1 - norm.cdf(threshold)) * 100 * size[0])).astype(
            int
        )
        # arr needs to be (n_channels x n_samples) (so that when we mask it,
        # it gets flattened in the right order)
        arr = np.random.randn(size[1], oversample)
        mask = np.abs(arr) <= threshold
        # total number of samples rejected for each channel
        num_rejected = np.cumsum(np.sum(~mask, axis=1))
        # roll forward by 1 and set first value to zero
        num_rejected = np.roll(num_rejected, 1, axis=0)
        num_rejected[0] = 0
        # starting indices from original arr after masked values are removed
        shifted_indices = (
            np.array([oversample * j for j in range(size[1])], dtype=int) - num_rejected
        )
        indices = np.concatenate([np.arange(ind, ind + size[0]) for ind in shifted_indices])
        # pull out masked values, reshape in correct order, and swapaxes to match dims of `size`
        return arr[mask][indices].reshape((size[1], size[0], size[2])).swapaxes(0, 1)

    def generate_frame(self):
        """Generates a single frame of data and overlaps it with the previous frame of data"""
        if not self.cola_initialized:
            self.cola_initialized = True
            self.generate_frame()
        # Create a signal
        if self.sigma_clip is None:
            real = np.random.randn(*self._size)
            imag = np.random.randn(*self._size)
        else:
            # Apply sigma clipping via rejection sampling
            # (apply correction factor to attempt to preserve rms levels)
            real = self.rejection_sample(self._size, self.sigma_clip) / self._scale_factor
            imag = self.rejection_sample(self._size, self.sigma_clip) / self._scale_factor
        # print('after ', len(real), len(imag))
        # Compute Random Process
        W = np.sqrt(0.5) * (real + 1j * imag)  # pylint: disable=invalid-name
        Xv = 1 / np.sqrt(self.frequency_spacing) * self.Lsvd @ W  # pylint: disable=invalid-name
        # Ensure that the signal is real by setting the nyquist and DC component to 0
        Xv[[0, -1], :, :] = 0
        # Compute the IFFT, using the real version makes it so you don't need negative frequencies
        zero_padding = np.zeros(
            [(self.output_oversample - 1) * (Xv.shape[0] - 1)] + list(Xv.shape[1:]),
            dtype=Xv.dtype,
        )
        # ifft_start = time()
        xv = (
            np.fft.irfft(np.concatenate((Xv, zero_padding), axis=0) / np.sqrt(2), axis=0)
            * self.output_oversample
            * self.sample_rate
        )
        # ifft_end = time()
        # print('FFT Time: {:0.4f}'.format(ifft_end-ifft_start))
        signal = xv[:, :, 0].T
        # Band limit it
        # Roll the queue
        self.cola_queue = np.roll(self.cola_queue, -1, axis=0)
        self.cola_queue[-1, ...] = signal
        output_signal = cola(
            self.samples_per_output * self.output_oversample,
            self.overlapped_output_samples * self.output_oversample,
            self.cola_queue,
            self.cola_window,
            self.cola_exponent,
        )

        # mirror tail ends of the distribution about the sigma clipping threshold
        if self.sigma_clip is not None:
            mask = np.abs(output_signal) >= (self._rms * self.sigma_clip)
            if len(mask) > 0:
                twosigma = np.sign(output_signal).real * self._rms.real * self.sigma_clip * 2
                output_signal[mask] *= -1
                output_signal[mask] += twosigma[mask]
        return output_signal, False


class ContinuousTransientSignalGenerator(SignalGenerator):
    """Signal generator that constantly receives data to later generate"""

    def __init__(self, num_samples_per_frame, num_signals, signal, last_signal):
        self.num_samples = num_samples_per_frame
        self.num_signals = num_signals
        self.signal = np.zeros((self.num_signals, 0)) if signal is None else signal
        self.no_more_signal_incoming = last_signal

    @property
    def ready_for_next_output(self):
        """Ready to output if there is enough data on the signal buffer to create a frame"""
        return self.signal.shape[-1] >= self.num_samples or self.no_more_signal_incoming

    def update_parameters(self, signal, last_signal):
        """Updates the parameters of the transient signal generator

        Parameters
        ----------
        signal : np.ndarray
            New portions of signals to add to the output signal buffer
        last_signal : bool
            True if this is the last signal that will be given to the signal generator
        """
        self.signal = np.concatenate((self.signal, signal), axis=-1)
        self.no_more_signal_incoming = last_signal

    def generate_frame(self):
        """Generates a frame of data and a flag letting the caller know the signal is done"""
        output_signal = self.signal[..., : self.num_samples]
        self.signal = self.signal[..., self.num_samples :]
        if DEBUG:
            num_files = len(glob(FILE_OUTPUT.format("*")))
            np.savez(
                FILE_OUTPUT.format(num_files),
                output_signal=output_signal,
                last_signal=(self.no_more_signal_incoming and self.signal.shape[-1] == 0),
            )
        return output_signal, (self.no_more_signal_incoming and self.signal.shape[-1] == 0)


class TransientSignalGenerator(SignalGenerator):
    """Signal generator to generate a specified signal"""

    def __init__(self, signal, repeat):
        self.signal = signal
        self.repeat = repeat

    @property
    def ready_for_next_output(self):
        """Ready for output if the signal is defined"""
        return self.signal is not None

    def update_parameters(self, signal, repeat):
        """Updates with a new signal and a flag to repeat the signal or not"""
        self.signal = signal
        self.repeat = repeat

    def generate_frame(self):
        """Generates the signal.  The done flag will be set if the signal is not repeating"""
        return self.signal, not self.repeat
