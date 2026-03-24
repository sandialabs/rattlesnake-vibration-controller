from rattlesnake.process.signal_generation import (
    RandomSignalGenerator,
    BurstRandomSignalGenerator,
    PseudorandomSignalGenerator,
    ChirpSignalGenerator,
    SquareSignalGenerator,
    SineSignalGenerator,
)
import numpy as np
import scipy.signal as sig


def generate_square_signal(argument_dict):
    signal_generator = SquareSignalGenerator(
        level=argument_dict["level"],
        sample_rate=argument_dict["sample_rate"],
        num_samples_per_frame=argument_dict["num_samples_per_frame"],
        num_signals=argument_dict["num_signals"],
        frequency=argument_dict["frequency"],
        phase=argument_dict["phase"],
        on_fraction=argument_dict["on_fraction"],
        output_oversample=argument_dict["output_oversample"],
    )

    return signal_generator


def generate_assert_square_signal(argument_dict):
    amplitude = np.broadcast_to(argument_dict["level"], (argument_dict["num_signals"], 1)).copy()
    frequency = np.array(argument_dict["frequency"], dtype=float)
    time = np.arange(
        argument_dict["num_samples_per_frame"] * argument_dict["output_oversample"]
    ) / (argument_dict["sample_rate"] * argument_dict["output_oversample"])
    phase = np.array(argument_dict["phase"], dtype=float)
    assert_data = amplitude * (
        2
        * (
            ((2 * np.pi * frequency[..., np.newaxis] * time + phase[..., np.newaxis]) % (2 * np.pi))
            < 2 * np.pi * argument_dict["on_fraction"]
        ).astype(int)
        - 1
    )

    return assert_data


def generate_sine_signal(argument_dict):
    signal_generator = SineSignalGenerator(
        level=argument_dict["level"],
        sample_rate=argument_dict["sample_rate"],
        num_samples_per_frame=argument_dict["num_samples_per_frame"],
        num_signals=argument_dict["num_signals"],
        frequency=argument_dict["frequency"],
        phase=argument_dict["phase"],
        output_oversample=argument_dict["output_oversample"],
    )

    return signal_generator


def generate_assert_sine_signal(argument_dict):
    amplitude = np.broadcast_to(argument_dict["level"], (argument_dict["num_signals"], 1)).copy()
    frequency = np.array(argument_dict["frequency"], dtype=float)
    time = np.arange(
        argument_dict["num_samples_per_frame"] * argument_dict["output_oversample"]
    ) / (argument_dict["sample_rate"] * argument_dict["output_oversample"])
    phase = np.array(argument_dict["phase"], dtype=float)
    assert_data = amplitude * np.sin(
        2 * np.pi * frequency[..., np.newaxis] * time + phase[..., np.newaxis]
    )

    return assert_data


def generate_chirp_signal(argument_dict):
    signal_generator = ChirpSignalGenerator(
        level=argument_dict["level"],
        sample_rate=argument_dict["sample_rate"],
        num_samples_per_frame=argument_dict["num_samples_per_frame"],
        num_signals=argument_dict["num_signals"],
        low_frequency_cutoff=argument_dict["low_frequency_cutoff"],
        high_frequency_cutoff=argument_dict["high_frequency_cutoff"],
        output_oversample=argument_dict["output_oversample"],
    )

    return signal_generator


def generate_assert_chirp_signal(argument_dict):
    amplitude = argument_dict["level"]
    time = np.arange(
        argument_dict["num_samples_per_frame"] * argument_dict["output_oversample"]
    ) / (argument_dict["sample_rate"] * argument_dict["output_oversample"])
    signal_length = argument_dict["num_samples_per_frame"] / argument_dict["sample_rate"]
    n_cycles = np.ceil(argument_dict["high_frequency_cutoff"] * signal_length)
    high_frequency_cutoff = n_cycles / signal_length
    frequency_slope = (
        high_frequency_cutoff - argument_dict["low_frequency_cutoff"]
    ) / signal_length
    argument = frequency_slope / 2 * time**2 + argument_dict["low_frequency_cutoff"] * time
    assert_data = np.tile(
        amplitude * np.sin(2 * np.pi * argument), (argument_dict["num_signals"], 1)
    )

    return assert_data


def generate_random_signal(argument_dict):
    signal_generator = RandomSignalGenerator(
        rms=argument_dict["rms"],
        sample_rate=argument_dict["sample_rate"],
        num_samples_per_frame=argument_dict["num_samples_per_frame"],
        num_signals=argument_dict["num_signals"],
        low_frequency_cutoff=argument_dict["low_frequency_cutoff"],
        high_frequency_cutoff=argument_dict["high_frequency_cutoff"],
        cola_overlap=argument_dict["cola_overlap"],
        cola_window=argument_dict["cola_window"],
        cola_exponent=argument_dict["cola_exponent"],
        output_oversample=argument_dict["output_oversample"],
    )
    # frame_output_samples = int(argument_dict["num_samples_per_frame"]*argument_dict["output_oversample"])
    # data = signal_generator.generate_frame()[0]
    # while data.shape[-1] < frame_output_samples:
    #     data = np.concatenate((data,signal_generator.generate_frame()[0]),axis=-1)
    # data = data[...,:frame_output_samples]
    # times = np.arange(frame_output_samples)/(argument_dict["sample_rate"]*argument_dict["output_oversample"])

    return signal_generator


def generate_random_queue(argument_dict, cola_queue):
    signal = argument_dict["rms"] * np.random.randn(
        argument_dict["num_signals"],
        argument_dict["num_samples_per_frame"] * argument_dict["output_oversample"],
    )
    # Band limit
    fft = np.fft.rfft(signal, axis=-1)
    freq = np.fft.rfftfreq(
        signal.shape[-1], 1 / (argument_dict["sample_rate"] * argument_dict["output_oversample"])
    )
    invalid_frequencies = (freq < argument_dict["low_frequency_cutoff"]) | (
        freq > argument_dict["high_frequency_cutoff"]
    )
    scale_factor = (~invalid_frequencies).sum() / len(invalid_frequencies)
    fft[..., invalid_frequencies] = 0
    bandlimited_signal = np.fft.irfft(fft) / np.sqrt(scale_factor)
    # Roll the queue
    cola_queue = np.roll(cola_queue, -1, axis=0)
    cola_queue[-1, ...] = bandlimited_signal

    signal_samples = (
        int(argument_dict["num_samples_per_frame"] * (1 - argument_dict["cola_overlap"]))
        * argument_dict["output_oversample"]
    )
    end_samples = (
        argument_dict["num_samples_per_frame"]
        - int(argument_dict["num_samples_per_frame"] * (1 - argument_dict["cola_overlap"]))
    ) * argument_dict["output_oversample"]
    window_name = argument_dict["cola_window"]

    total_samples = signal_samples + end_samples
    if window_name == "tukey":
        window_name = ("tukey", 2 * (end_samples / total_samples))
    window = (
        sig.get_window(window_name, total_samples, fftbins=True) ** argument_dict["cola_exponent"]
    )
    # Create the new signal
    last_signal, current_signal = cola_queue
    assert_data = current_signal[:, :signal_samples] * window[:signal_samples]
    if end_samples > 0:
        assert_data[:, :end_samples] += (
            np.array(last_signal)[:, -end_samples:] * window[-end_samples:]
        )

    return assert_data, cola_queue


def generate_burst_random_signal(argument_dict):
    signal_generator = BurstRandomSignalGenerator(
        rms=argument_dict["rms"],
        sample_rate=argument_dict["sample_rate"],
        num_samples_per_frame=argument_dict["num_samples_per_frame"],
        num_signals=argument_dict["num_signals"],
        low_frequency_cutoff=argument_dict["low_frequency_cutoff"],
        high_frequency_cutoff=argument_dict["high_frequency_cutoff"],
        on_fraction=argument_dict["on_fraction"],
        ramp_fraction=argument_dict["ramp_fraction"],
        output_oversample=argument_dict["output_oversample"],
    )

    # frame_output_samples = int(
    #     argument_dict["num_samples_per_frame"]*argument_dict["output_oversample"])
    # times = np.arange(frame_output_samples) / \
    #     (argument_dict["sample_rate"]*argument_dict["output_oversample"])

    return signal_generator


def generate_assert_burst_random_signal(argument_dict):
    signal = argument_dict["rms"] * np.random.randn(
        argument_dict["num_signals"],
        argument_dict["num_samples_per_frame"] * argument_dict["output_oversample"],
    )
    # Band limit
    fft = np.fft.rfft(signal, axis=-1)
    freq = np.fft.rfftfreq(
        signal.shape[-1], 1 / (argument_dict["sample_rate"] * argument_dict["output_oversample"])
    )
    invalid_frequencies = (freq < argument_dict["low_frequency_cutoff"]) | (
        freq > argument_dict["high_frequency_cutoff"]
    )
    scale_factor = (~invalid_frequencies).sum() / len(invalid_frequencies)
    fft[..., invalid_frequencies] = 0
    bandlimited_signal = np.fft.irfft(fft) / np.sqrt(scale_factor)
    # Ramping function
    ramp_samples = int(
        argument_dict["num_samples_per_frame"]
        * argument_dict["output_oversample"]
        * argument_dict["on_fraction"]
        * argument_dict["ramp_fraction"]
    )
    on_samples = int(
        argument_dict["num_samples_per_frame"]
        * argument_dict["output_oversample"]
        * argument_dict["on_fraction"]
        - 2 * ramp_samples
    )
    envelope = np.zeros(argument_dict["num_samples_per_frame"] * argument_dict["output_oversample"])
    envelope[:ramp_samples] = np.linspace(0, 1, ramp_samples)
    envelope[ramp_samples : ramp_samples + on_samples] = 1
    envelope[ramp_samples + on_samples : ramp_samples * 2 + on_samples] = np.linspace(
        1, 0, ramp_samples
    )
    assert_data = bandlimited_signal * envelope

    return assert_data


def generate_pseudorandom_signal(argument_dict):
    signal_generator = PseudorandomSignalGenerator(
        rms=argument_dict["rms"],
        sample_rate=argument_dict["sample_rate"],
        num_samples_per_frame=argument_dict["num_samples_per_frame"],
        num_signals=argument_dict["num_signals"],
        low_frequency_cutoff=argument_dict["low_frequency_cutoff"],
        high_frequency_cutoff=argument_dict["high_frequency_cutoff"],
        output_oversample=argument_dict["output_oversample"],
    )

    return signal_generator


def generate_assert_pseudorandom_signal(argument_dict):
    freq = np.fft.rfftfreq(
        argument_dict["num_samples_per_frame"] * argument_dict["output_oversample"],
        1 / (argument_dict["sample_rate"] * argument_dict["output_oversample"]),
    )
    fft = np.zeros(
        (
            argument_dict["num_signals"],
            argument_dict["num_samples_per_frame"] * argument_dict["output_oversample"] // 2 + 1,
        ),
        dtype=complex,
    )
    low_frequency_cutoff = (
        0
        if argument_dict["low_frequency_cutoff"] is None
        else argument_dict["low_frequency_cutoff"]
    )
    high_frequency_cutoff = (
        argument_dict["sample_rate"] / 2
        if argument_dict["high_frequency_cutoff"] is None
        else argument_dict["high_frequency_cutoff"]
    )
    valid_frequencies = (freq >= low_frequency_cutoff) & (freq <= high_frequency_cutoff)
    fft[..., valid_frequencies] = np.exp(
        1j * 2 * np.pi * np.random.rand(argument_dict["num_signals"], valid_frequencies.sum())
    )
    assert_data = np.fft.irfft(fft)
    signal_rms = np.sqrt(np.mean(assert_data**2, axis=-1, keepdims=True))
    assert_data *= argument_dict["rms"] / signal_rms

    return assert_data


if __name__ == "__main__":
    test_dict = {
        "rms": 2,
        "sample_rate": 1000,
        "num_samples_per_frame": 100,
        "num_signals": 9,
        "low_frequency_cutoff": 1,
        "high_frequency_cutoff": 1000,
        "cola_overlap": 0.5,
        "cola_window": "hann",
        "cola_exponent": 0.5,
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
    }
    # generate_burst_random_signal(test_dict)
