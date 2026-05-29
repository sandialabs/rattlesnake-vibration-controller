"""
Signal Generator Class Testing

This code tests the signal generator classes and the accuracies
of the generate_frame functions.

The following code is tested:
signal_generation.py
- SquareSignalGenerator.__init__
- SquareSignalGenerator.generate_frame
- SineSignalGenerator.__init__
- SineSignalGenerator.generate_frame
- ChirpSignalGenerator.__init__
- ChirpSignalGenerator.generate_frame
- RandomSignalGenerator.__init__
- RandomSignalGenerator.generate_frame
- PseudorandomSignalGenerator.__init__
- PseudorandomSignalGenerator.generate_frame
- BurstRandomSignalGenerator.__init__
- BurstRandomSignalGenerator.generate_frame
"""

from rattlesnake.process.signal_generation import (
    SignalTypes,
    SignalGenerator,
    cola,
    cpsd_to_time_history,
    RandomSignalGenerator,
    PseudorandomSignalGenerator,
    BurstRandomSignalGenerator,
    ChirpSignalGenerator,
    SineSignalGenerator,
    SquareSignalGenerator,
    CPSDSignalGenerator,
    ContinuousTransientSignalGenerator,
    TransientSignalGenerator,
)
import functions.signal_generation_functions as sig_fun
import pytest
import sys
from pathlib import Path
import numpy as np
import scipy.signal as sig
from unittest import mock


# Create a dictionary of arguments for signals, easier than redefining these for every test
def get_test_arguments():
    return [
        {
            "rms": 2,
            "sample_rate": 1000,
            "num_samples_per_frame": 100,
            "num_signals": 9,
            "low_frequency_cutoff": 0,
            "high_frequency_cutoff": 500,
            "cola_overlap": 0.5,
            "cola_window": "boxcar",
            "cola_exponent": 0.1,
            "output_oversample": 10,
            "cola_queue": 10,
            "on_fraction": 0.5,
            "ramp_fraction": 0.25,
            "level": 1,
            "frequency": 5,
            "phase": 0,
            "cpsd_matrix": "tk",
            "signal": "tk",
            "last_signal": "tk",
            "repeat": "tk",
            "num_frames": 5,
            "random_seed": 45,
        },
        {
            "rms": 2,
            "sample_rate": 100,
            "num_samples_per_frame": 300,
            "num_signals": 9,
            "low_frequency_cutoff": 0,
            "high_frequency_cutoff": 500,
            "cola_overlap": 0.2,
            "cola_window": "boxcar",
            "cola_exponent": 0.1,
            "output_oversample": 10,
            "cola_queue": 10,
            "on_fraction": 0.5,
            "ramp_fraction": 0.25,
            "level": 1,
            "frequency": 5,
            "phase": 0,
            "cpsd_matrix": "tk",
            "signal": "tk",
            "last_signal": "tk",
            "repeat": "tk",
            "num_frames": 5,
            "random_seed": 28,
        },
        {
            "rms": 2,
            "sample_rate": 500,
            "num_samples_per_frame": 250,
            "num_signals": 9,
            "low_frequency_cutoff": 0,
            "high_frequency_cutoff": 500,
            "cola_overlap": 0.1,
            "cola_window": "hamming",
            "cola_exponent": 0.2,
            "output_oversample": 10,
            "cola_queue": 10,
            "on_fraction": 0.5,
            "ramp_fraction": 0.25,
            "level": 1,
            "frequency": 5,
            "phase": 0,
            "cpsd_matrix": "tk",
            "signal": "tk",
            "last_signal": "tk",
            "repeat": "tk",
            "num_frames": 5,
            "random_seed": 36,
        },
    ]


class DummySignalGenerator(SignalGenerator):
    def __init__(self):
        super().__init__()

    def update_parameters(self, *args, **kwargs):
        return super().update_parameters(*args, **kwargs)

    @property
    def ready_for_next_output(self):
        pass

    def generate_frame(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("signal_idx", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_signal_types(signal_idx):
    signal_type = SignalTypes(signal_idx)

    assert isinstance(signal_type, SignalTypes)


def test_signal_generator_init():
    signal_generator = DummySignalGenerator()

    assert isinstance(signal_generator, DummySignalGenerator)


# Test SquareSignalGenerator
# Loop through input arguments
@pytest.mark.parametrize(
    "argument_dict",
    [(get_test_arguments()[0]), (get_test_arguments()[1]), (get_test_arguments()[2])],
)
def test_square_wave(argument_dict):
    # Generate square wave with generator class
    signal_generator = sig_fun.generate_square_signal(argument_dict)
    data = signal_generator.generate_frame()[0]

    # Generate square wave
    assert_data = sig_fun.generate_assert_square_signal(argument_dict)

    # Test if the generator made the correct square wave
    np.testing.assert_array_almost_equal(data, assert_data)


def test_square_wave_ready_output():
    square_wave = sig_fun.generate_square_signal(get_test_arguments()[0])
    data = square_wave.ready_for_next_output

    assert data == True


def test_square_wave_update_parameters():
    square_wave = sig_fun.generate_square_signal(get_test_arguments()[0])
    frequency = np.array(10, dtype=float)
    phase = np.array(0.5, dtype=float)

    square_wave.update_parameters(frequency, phase)

    np.testing.assert_array_equal(square_wave.frequency, frequency)
    np.testing.assert_array_equal(square_wave.phase, phase)


# Test SineWaveGenerator
# Loop through input arguments
@pytest.mark.parametrize(
    "argument_dict",
    [(get_test_arguments()[0]), (get_test_arguments()[1]), (get_test_arguments()[2])],
)
def test_sine_wave(argument_dict):
    # Generate sine wave with generator class
    signal_generator = sig_fun.generate_sine_signal(argument_dict)
    data = signal_generator.generate_frame()[0]

    # Generate sine wave
    assert_data = sig_fun.generate_assert_sine_signal(argument_dict)

    # Test if generator made the correct sine wave
    np.testing.assert_array_almost_equal(data, assert_data)


def test_sine_wave_ready_output():
    sine_wave = sig_fun.generate_sine_signal(get_test_arguments()[0])
    data = sine_wave.ready_for_next_output

    assert data == True


def test_sine_wae_update_parameters():
    sine_wave = sig_fun.generate_sine_signal(get_test_arguments()[0])
    frequency = np.array(10, dtype=float)
    level = np.broadcast_to(2, (sine_wave.num_signals, 1)).copy()
    phase = np.array(0.5, dtype=float)

    sine_wave.update_parameters(frequency, level, phase)

    np.testing.assert_array_equal(sine_wave.frequency, frequency)
    np.testing.assert_array_equal(sine_wave.level, level)
    np.testing.assert_array_equal(sine_wave.phase, phase)


# Test ChirpSignalGenerator
# Loop through input arguments
@pytest.mark.parametrize(
    "argument_dict",
    [(get_test_arguments()[0]), (get_test_arguments()[1]), (get_test_arguments()[2])],
)
def test_chirp_wave(argument_dict):
    # Generate chirp signal with generator class
    signal_generator = sig_fun.generate_chirp_signal(argument_dict)
    data = signal_generator.generate_frame()[0]

    # Generate chirp signal without class
    assert_data = sig_fun.generate_assert_chirp_signal(argument_dict)

    # Test if generator made correct chirp signal
    np.testing.assert_array_almost_equal(data, assert_data)


def test_chirp_wave_ready_output():
    chirp_wave = sig_fun.generate_chirp_signal(get_test_arguments()[0])
    data = chirp_wave.ready_for_next_output

    assert data == True


# Burst Random Signals
# Loop through input arguments
@pytest.mark.parametrize(
    "argument_dict",
    [(get_test_arguments()[0]), (get_test_arguments()[1]), (get_test_arguments()[2])],
)
def test_burst_random_wave(argument_dict):
    # Generate burst random signal at a random seed with generator class
    np.random.seed(argument_dict["random_seed"])
    signal_generator = sig_fun.generate_burst_random_signal(argument_dict)
    data = signal_generator.generate_frame()[0]

    # Generate burst random signal at same seed without class
    np.random.seed(argument_dict["random_seed"])
    assert_data = sig_fun.generate_assert_burst_random_signal(argument_dict)

    # Test if generator made correct burst random signal
    np.testing.assert_array_almost_equal(data, assert_data)


def test_burst_wave_ready_output():
    burst_wave = sig_fun.generate_burst_random_signal(get_test_arguments()[0])
    data = burst_wave.ready_for_next_output

    assert data == True


# Random Signal
# Loop through input arguments
@pytest.mark.parametrize(
    "argument_dict",
    [(get_test_arguments()[0]), (get_test_arguments()[1]), (get_test_arguments()[2])],
)
def test_random_wave(argument_dict):
    # Generate random signal at seed with generator class
    np.random.seed(argument_dict["random_seed"])
    signal_generator = sig_fun.generate_random_signal(argument_dict)
    data = signal_generator.generate_frame()[0]

    # Create a signal at the same seed
    np.random.seed(argument_dict["random_seed"])
    cola_queue = np.zeros(
        (
            2,
            argument_dict["num_signals"],
            argument_dict["num_samples_per_frame"] * argument_dict["output_oversample"],
        )
    )
    assert_data, cola_queue = sig_fun.generate_random_queue(argument_dict, cola_queue)
    assert_data, cola_queue = sig_fun.generate_random_queue(argument_dict, cola_queue)

    # Test if generator made correct random signal
    np.testing.assert_array_almost_equal(data, assert_data)


def test_random_wave_ready_output():
    random_wave = sig_fun.generate_random_signal(get_test_arguments()[0])
    data = random_wave.ready_for_next_output

    assert data == True


# Pseudorandom signal
# Loop through input arguments
@pytest.mark.parametrize(
    "argument_dict",
    [(get_test_arguments()[0]), (get_test_arguments()[1]), (get_test_arguments()[2])],
)
def test_pseudorandom_wave(argument_dict):
    # Generate a pseudorandom signal with generator class
    np.random.seed(argument_dict["random_seed"])
    signal_generator = sig_fun.generate_pseudorandom_signal(argument_dict)
    data = signal_generator.generate_frame()[0]

    # Generate same signal without class
    np.random.seed(argument_dict["random_seed"])
    assert_data = sig_fun.generate_assert_pseudorandom_signal(argument_dict)

    # Test if generator class made the correct signal
    np.testing.assert_array_almost_equal(data, assert_data)


def test_pseudorandom_wave_ready_output():
    pseudorandom_wave = sig_fun.generate_pseudorandom_signal(get_test_arguments()[0])
    data = pseudorandom_wave.ready_for_next_output

    assert data == True


if __name__ == "__main__":
    # test_square_wave(get_test_arguments()[1])
    # test_sine_wave(get_test_arguments()[1])
    # test_chirp_wave(get_test_arguments()[1])
    # test_burst_random_wave(get_test_arguments()[1])
    # test_random_wave(get_test_arguments()[1])
    test_pseudorandom_wave(get_test_arguments()[1])
    pass
