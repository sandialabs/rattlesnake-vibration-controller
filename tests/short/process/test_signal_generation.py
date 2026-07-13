import numpy as np
import pytest

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


# region Helpers
class DummySignalGenerator(SignalGenerator):
    def __init__(self):
        self.ready = True
        self.updated = False

    def generate_frame(self):
        return np.zeros((1, 1)), False

    def update_parameters(self, *args, **kwargs):
        self.updated = True
        return super().update_parameters()

    @property
    def ready_for_next_output(self):
        return self.ready


# endregion


# region SignalTypes
def test_signal_types_unique_integer_values():
    """
    Verifies that signal type enum values are unique integers.
    """
    values = [signal_type.value for signal_type in SignalTypes]

    assert all(isinstance(value, int) for value in values)
    assert len(values) == len(set(values))


@pytest.mark.parametrize("signal_idx", range(9))
def test_signal_types(signal_idx):
    """
    Verifies that signal type enum values construct valid ``SignalTypes``
    members.
    """
    signal_type = SignalTypes(signal_idx)

    assert isinstance(signal_type, SignalTypes)


def test_signal_types_expected_values():
    """
    Verifies that signal type enum members have expected values.
    """
    assert SignalTypes.RANDOM.value == 0
    assert SignalTypes.PSEUDORANDOM.value == 1
    assert SignalTypes.BURST_RANDOM.value == 2
    assert SignalTypes.CHIRP.value == 3
    assert SignalTypes.SINE.value == 4
    assert SignalTypes.SQUARE.value == 5
    assert SignalTypes.CPSD.value == 6
    assert SignalTypes.TRANSIENT.value == 7
    assert SignalTypes.CONTINUOUSTRANSIENT.value == 8


# endregion


# region Utility Functions
def test_cola_no_overlap():
    """
    Verifies COLA output when there are no overlapped samples.
    """
    signals = np.array(
        [
            [[10.0, 10.0, 10.0, 10.0]],
            [[1.0, 2.0, 3.0, 4.0]],
        ]
    )

    output = cola(
        signal_samples=4,
        end_samples=0,
        signals=signals,
        window_name="boxcar",
        window_exponent=1.0,
    )

    np.testing.assert_array_equal(output, np.array([[1.0, 2.0, 3.0, 4.0]]))


def test_cola_with_overlap_boxcar():
    """
    Verifies COLA output with overlapped samples using a boxcar window.
    """
    signals = np.array(
        [
            [[10.0, 20.0, 30.0, 40.0]],
            [[1.0, 2.0, 3.0, 4.0]],
        ]
    )

    output = cola(
        signal_samples=2,
        end_samples=2,
        signals=signals,
        window_name="boxcar",
        window_exponent=1.0,
    )

    np.testing.assert_array_equal(output, np.array([[31.0, 42.0]]))


def test_cola_tukey_window():
    """
    Verifies that Tukey window handling returns an output with expected shape.
    """
    signals = np.ones((2, 2, 10))

    output = cola(
        signal_samples=6,
        end_samples=4,
        signals=signals,
        window_name="tukey",
        window_exponent=0.5,
    )

    assert output.shape == (2, 6)


def test_cpsd_to_time_history_shape():
    """
    Verifies that CPSD synthesis returns a channel-by-samples time history.
    """
    np.random.seed(1)

    cpsd_matrix = np.zeros((5, 2, 2), dtype=complex)
    cpsd_matrix[:, 0, 0] = 1.0
    cpsd_matrix[:, 1, 1] = 1.0

    output = cpsd_to_time_history(
        cpsd_matrix,
        sample_rate=100.0,
        df=1.0,
        output_oversample=1,
    )

    assert output.shape == (2, 8)
    assert np.isrealobj(output)


def test_cpsd_to_time_history_output_oversample_shape():
    """
    Verifies that CPSD synthesis honors output oversampling.
    """
    np.random.seed(1)

    cpsd_matrix = np.zeros((5, 1, 1), dtype=complex)
    cpsd_matrix[:, 0, 0] = 1.0

    output = cpsd_to_time_history(
        cpsd_matrix,
        sample_rate=100.0,
        df=1.0,
        output_oversample=2,
    )

    assert output.shape == (1, 16)


# endregion


# region SignalGenerator Base
def test_signal_generator_init():
    """
    Verifies that a concrete dummy signal generator subclass can be constructed.
    """
    signal_generator = DummySignalGenerator()

    assert isinstance(signal_generator, DummySignalGenerator)
    assert isinstance(signal_generator, SignalGenerator)
    assert signal_generator.ready_for_next_output is True


def test_signal_generator_generate_frame():
    """
    Verifies that the dummy signal generator returns a frame and done flag.
    """
    signal_generator = DummySignalGenerator()

    frame, done = signal_generator.generate_frame()

    np.testing.assert_array_equal(frame, np.zeros((1, 1)))
    assert done is False


# endregion


# region RandomSignalGenerator
def test_random_signal_generator_init():
    """
    Verifies random signal generator initialization and derived properties.
    """
    np.random.seed(1)

    generator = RandomSignalGenerator(
        rms=2.0,
        sample_rate=1000.0,
        num_samples_per_frame=100,
        num_signals=3,
        low_frequency_cutoff=10.0,
        high_frequency_cutoff=200.0,
        cola_overlap=0.5,
        cola_window="hann",
        cola_exponent=0.5,
        output_oversample=2,
    )

    assert isinstance(generator, RandomSignalGenerator)
    assert generator.samples_per_output == 50
    assert generator.overlapped_output_samples == 50
    assert generator.ready_for_next_output is True
    assert generator.cola_queue.shape == (2, 3, 200)


def test_random_signal_generator_generate_frame_shape():
    """
    Verifies random signal generator output shape and done flag.
    """
    np.random.seed(1)

    generator = RandomSignalGenerator(
        rms=1.0,
        sample_rate=1000.0,
        num_samples_per_frame=100,
        num_signals=2,
        low_frequency_cutoff=None,
        high_frequency_cutoff=None,
        cola_overlap=0.5,
        cola_window="boxcar",
        cola_exponent=1.0,
        output_oversample=1,
    )

    frame, done = generator.generate_frame()

    assert frame.shape == (2, 50)
    assert done is False
    assert np.all(np.isfinite(frame))


def test_random_signal_generator_band_limits_signal():
    """
    Verifies that generated random data are finite when cutoffs are supplied.
    """
    np.random.seed(2)

    generator = RandomSignalGenerator(
        rms=1.0,
        sample_rate=1000.0,
        num_samples_per_frame=128,
        num_signals=1,
        low_frequency_cutoff=50.0,
        high_frequency_cutoff=200.0,
        cola_overlap=0.0,
        cola_window="boxcar",
        cola_exponent=1.0,
        output_oversample=1,
    )

    frame, _ = generator.generate_frame()

    assert frame.shape == (1, 128)
    assert np.all(np.isfinite(frame))


# endregion


# region PseudorandomSignalGenerator
def test_pseudorandom_signal_generator_init_and_generate():
    """
    Verifies pseudorandom signal generation shape, repetition, and readiness.
    """
    np.random.seed(1)

    generator = PseudorandomSignalGenerator(
        rms=2.0,
        sample_rate=1000.0,
        num_samples_per_frame=128,
        num_signals=2,
        low_frequency_cutoff=10.0,
        high_frequency_cutoff=200.0,
        output_oversample=1,
    )

    frame_1, done_1 = generator.generate_frame()
    frame_2, done_2 = generator.generate_frame()

    assert generator.ready_for_next_output is True
    assert frame_1.shape == (2, 128)
    assert done_1 is False
    assert done_2 is False
    np.testing.assert_array_equal(frame_1, frame_2)


def test_pseudorandom_signal_generator_rms_scaling():
    """
    Verifies pseudorandom signal RMS is scaled approximately to the requested
    RMS.
    """
    np.random.seed(2)

    generator = PseudorandomSignalGenerator(
        rms=3.0,
        sample_rate=1000.0,
        num_samples_per_frame=256,
        num_signals=2,
        low_frequency_cutoff=None,
        high_frequency_cutoff=None,
        output_oversample=1,
    )

    frame, _ = generator.generate_frame()
    rms = np.sqrt(np.mean(frame**2, axis=-1))

    np.testing.assert_allclose(rms, np.array([3.0, 3.0]), rtol=1e-12, atol=1e-12)


# endregion


# region BurstRandomSignalGenerator
def test_burst_random_signal_generator_init():
    """
    Verifies burst random signal generator initialization and derived sample
    counts.
    """
    generator = BurstRandomSignalGenerator(
        rms=1.0,
        sample_rate=1000.0,
        num_samples_per_frame=100,
        num_signals=2,
        low_frequency_cutoff=10.0,
        high_frequency_cutoff=200.0,
        on_fraction=0.5,
        ramp_fraction=0.2,
        output_oversample=2,
    )

    assert isinstance(generator, BurstRandomSignalGenerator)
    assert generator.ramp_samples == 20
    assert generator.on_samples == 60
    assert generator.ready_for_next_output is True
    assert generator.envelope.shape == (200,)


def test_burst_random_signal_generator_invalid_ramp_fraction():
    """
    Verifies that ramp fractions greater than 0.5 raise ``ValueError``.
    """
    with pytest.raises(ValueError):
        BurstRandomSignalGenerator(
            rms=1.0,
            sample_rate=1000.0,
            num_samples_per_frame=100,
            num_signals=1,
            low_frequency_cutoff=10.0,
            high_frequency_cutoff=200.0,
            on_fraction=0.5,
            ramp_fraction=0.6,
            output_oversample=1,
        )


def test_burst_random_signal_generator_generate_frame_shape():
    """
    Verifies burst random output shape and envelope zeros outside the burst.
    """
    np.random.seed(1)

    generator = BurstRandomSignalGenerator(
        rms=1.0,
        sample_rate=1000.0,
        num_samples_per_frame=100,
        num_signals=1,
        low_frequency_cutoff=None,
        high_frequency_cutoff=None,
        on_fraction=0.5,
        ramp_fraction=0.2,
        output_oversample=1,
    )

    frame, done = generator.generate_frame()

    assert frame.shape == (1, 100)
    assert done is False
    np.testing.assert_array_equal(frame[:, 50:], np.zeros((1, 50)))


# endregion


# region ChirpSignalGenerator
def test_chirp_signal_generator_init_and_generate():
    """
    Verifies chirp signal generation shape, repetition, and readiness.
    """
    generator = ChirpSignalGenerator(
        level=2.0,
        sample_rate=1000.0,
        num_samples_per_frame=100,
        num_signals=3,
        low_frequency_cutoff=10.0,
        high_frequency_cutoff=100.0,
        output_oversample=2,
    )

    frame_1, done_1 = generator.generate_frame()
    frame_2, done_2 = generator.generate_frame()

    assert generator.ready_for_next_output is True
    assert frame_1.shape == (3, 200)
    assert done_1 is False
    assert done_2 is False
    np.testing.assert_array_equal(frame_1, frame_2)
    np.testing.assert_array_less(np.abs(frame_1), 2.0 + 1e-12)


# endregion


# region SineSignalGenerator
def test_sine_signal_generator_init_properties_ready():
    """
    Verifies sine generator initialization, phase properties, and readiness.
    """
    generator = SineSignalGenerator(
        level=2.0,
        sample_rate=1000.0,
        num_samples_per_frame=100,
        num_signals=2,
        frequency=5.0,
        phase=0.0,
        output_oversample=2,
    )

    assert isinstance(generator, SineSignalGenerator)
    assert generator.ready_for_next_output is True
    assert generator.phase_per_sample == pytest.approx(2 * np.pi * 5.0 / 1000.0)
    assert generator.phase_per_frame == pytest.approx(2 * np.pi * 5.0 / 1000.0 * 100)
    assert generator.times.shape == (200,)


@pytest.mark.parametrize(
    "frequency, phase, expected_ready",
    [
        (None, 0.0, False),
        (5.0, None, False),
        (5.0, 0.0, True),
    ],
)
def test_sine_signal_generator_ready_for_next_output(
    frequency,
    phase,
    expected_ready,
):
    """
    Verifies sine readiness depends on frequency and phase being defined.
    """
    generator = SineSignalGenerator(
        level=1.0,
        sample_rate=1000.0,
        num_samples_per_frame=10,
        num_signals=1,
        frequency=frequency,
        phase=phase,
        output_oversample=1,
    )

    assert generator.ready_for_next_output is expected_ready


def test_sine_signal_generator_update_parameters():
    """
    Verifies sine frequency, level, and phase updates.
    """
    generator = SineSignalGenerator(
        level=1.0,
        sample_rate=1000.0,
        num_samples_per_frame=10,
        num_signals=2,
        frequency=5.0,
        phase=0.0,
        output_oversample=1,
    )

    level = np.array([[2.0], [3.0]])
    generator.update_parameters(frequency=10.0, level=level, phase=0.5)

    np.testing.assert_array_equal(generator.frequency, np.array(10.0))
    np.testing.assert_array_equal(generator.level, level)
    np.testing.assert_array_equal(generator.phase, np.array(0.5))


def test_sine_signal_generator_generate_frame_and_phase_advance():
    """
    Verifies sine generation and phase advancement.
    """
    generator = SineSignalGenerator(
        level=1.0,
        sample_rate=100.0,
        num_samples_per_frame=10,
        num_signals=1,
        frequency=5.0,
        phase=0.0,
        output_oversample=1,
    )

    initial_phase = generator.phase.copy()
    frame, done = generator.generate_frame()

    expected = np.sin(2 * np.pi * 5.0 * np.arange(10) / 100.0).reshape(1, -1)

    np.testing.assert_allclose(frame, expected)
    assert done is False
    assert generator.phase == pytest.approx(initial_phase + generator.phase_per_frame)


# endregion


# region SquareSignalGenerator
def test_square_signal_generator_init_properties_ready():
    """
    Verifies square generator initialization, phase properties, and readiness.
    """
    generator = SquareSignalGenerator(
        level=2.0,
        sample_rate=1000.0,
        num_samples_per_frame=100,
        num_signals=2,
        frequency=5.0,
        phase=0.0,
        on_fraction=0.5,
        output_oversample=2,
    )

    assert isinstance(generator, SquareSignalGenerator)
    assert generator.ready_for_next_output is True
    assert generator.phase_per_sample == pytest.approx(2 * np.pi * 5.0 / 1000.0)
    assert generator.phase_per_frame == pytest.approx(2 * np.pi * 5.0 / 1000.0 * 100)


@pytest.mark.parametrize(
    "frequency, phase, expected_ready",
    [
        (None, 0.0, False),
        (5.0, None, False),
        (5.0, 0.0, True),
    ],
)
def test_square_signal_generator_ready_for_next_output(
    frequency,
    phase,
    expected_ready,
):
    """
    Verifies square readiness depends on frequency and phase being defined.
    """
    generator = SquareSignalGenerator(
        level=1.0,
        sample_rate=1000.0,
        num_samples_per_frame=10,
        num_signals=1,
        frequency=frequency,
        phase=phase,
        on_fraction=0.5,
        output_oversample=1,
    )

    assert generator.ready_for_next_output is expected_ready


def test_square_signal_generator_update_parameters():
    """
    Verifies square frequency and phase updates.
    """
    generator = SquareSignalGenerator(
        level=1.0,
        sample_rate=1000.0,
        num_samples_per_frame=10,
        num_signals=1,
        frequency=5.0,
        phase=0.0,
        on_fraction=0.5,
        output_oversample=1,
    )

    generator.update_parameters(frequency=10.0, phase=0.5)

    np.testing.assert_array_equal(generator.frequency, np.array(10.0))
    np.testing.assert_array_equal(generator.phase, np.array(0.5))


def test_square_signal_generator_generate_frame_and_phase_advance():
    """
    Verifies square wave generation and phase advancement.
    """
    generator = SquareSignalGenerator(
        level=1.0,
        sample_rate=8.0,
        num_samples_per_frame=8,
        num_signals=1,
        frequency=1.0,
        phase=0.0,
        on_fraction=0.5,
        output_oversample=1,
    )

    initial_phase = generator.phase.copy()
    frame, done = generator.generate_frame()

    expected = np.array([[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]])

    np.testing.assert_array_equal(frame, expected)
    assert done is False
    assert generator.phase == pytest.approx(initial_phase + generator.phase_per_frame)


# endregion


# region CPSDSignalGenerator
def make_cpsd_matrix(num_frequency_lines=5, num_signals=2):
    cpsd_matrix = np.zeros(
        (num_frequency_lines, num_signals, num_signals), dtype=complex
    )
    for i in range(num_signals):
        cpsd_matrix[:, i, i]
