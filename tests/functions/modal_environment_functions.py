from rattlesnake.environment.modal_environment import ModalCommands
from rattlesnake.process.data_collector import (
    AcquisitionType,
    Acceptance,
    TriggerSlope,
    Window,
    DataCollectorCommands,
)
from rattlesnake.process.spectral_processing import (
    AveragingTypes,
    Estimator,
    SpectralProcessingCommands,
)
from rattlesnake.process.signal_generation_process import SignalGenerationCommands
from unittest import mock
import numpy as np


def create_signal_generator_dict():
    signal_generator_dict = {
        0: ("none", "rattlesnake.environment.modal_environment.PseudorandomSignalGenerator"),
        1: ("random", "rattlesnake.environment.modal_environment.RandomSignalGenerator"),
        2: (
            "pseudorandom",
            "rattlesnake.environment.modal_environment.PseudorandomSignalGenerator",
        ),
        3: ("burst", "rattlesnake.environment.modal_environment.BurstRandomSignalGenerator"),
        4: ("chirp", "rattlesnake.environment.modal_environment.ChirpSignalGenerator"),
        5: ("square", "rattlesnake.environment.modal_environment.SquareSignalGenerator"),
        6: ("sine", "rattlesnake.environment.modal_environment.SineSignalGenerator"),
    }
    return signal_generator_dict


def create_modal_commands_dict():
    modal_commands_dict = {
        0: ModalCommands.START_CONTROL,
        1: ModalCommands.STOP_CONTROL,
        2: ModalCommands.ACCEPT_FRAME,
        3: ModalCommands.RUN_CONTROL,
        4: ModalCommands.CHECK_FOR_COMPLETE_SHUTDOWN,
    }
    return modal_commands_dict


def create_environment_template_calls():
    environment_template_calls = [
        mock.call(1, 1, "Control Type"),
        mock.call(1, 2, "Modal"),
        mock.call(2, 1, "Samples Per Frame:"),
        mock.call(2, 2, "# Number of Samples per Measurement Frame"),
        mock.call(3, 1, "Averaging Type:"),
        mock.call(3, 2, "# Averaging Type"),
        mock.call(4, 1, "Number of Averages:"),
        mock.call(4, 2, "# Number of Averages used when computing the FRF"),
        mock.call(5, 1, "Averaging Coefficient:"),
        mock.call(5, 2, "# Averaging Coefficient for Exponential Averaging"),
        mock.call(6, 1, "FRF Technique:"),
        mock.call(6, 2, "# FRF Technique"),
        mock.call(7, 1, "FRF Window:"),
        mock.call(7, 2, "# Window used to compute FRF"),
        mock.call(8, 1, "Exponential Window End Value:"),
        mock.call(
            8,
            2,
            "# Exponential Window Value at the end of the measurement frame (0.5 or 50%, not 50)",
        ),
        mock.call(9, 1, "FRF Overlap:"),
        mock.call(9, 2, "# Overlap for FRF calculations (0.5 or 50%, not 50)"),
        mock.call(10, 1, "Triggering Type:"),
        mock.call(10, 2, '# One of "Free Run", "First Frame", or "Every Frame"'),
        mock.call(11, 1, "Average Acceptance:"),
        mock.call(11, 2, '# One of "Accept All", "Manual", or "Autoreject"'),
        mock.call(12, 1, "Trigger Channel"),
        mock.call(12, 2, "# Channel number (1-based) to use for triggering"),
        mock.call(13, 1, "Pretrigger"),
        mock.call(13, 2, "# Amount of frame to use as pretrigger (0.5 or 50%, not 50)"),
        mock.call(14, 1, "Trigger Slope"),
        mock.call(14, 2, '# One of "Positive" or "Negative"'),
        mock.call(15, 1, "Trigger Level"),
        mock.call(
            15,
            2,
            "# Level to use to trigger the test as a fraction of the total range of the channel (0.5 or 50%, not 50)",
        ),
        mock.call(16, 1, "Hysteresis Level"),
        mock.call(
            16,
            2,
            "# Level that a channel must fall below before another trigger can be considered (0.5 or 50%, not 50)",
        ),
        mock.call(17, 1, "Hysteresis Frame Fraction"),
        mock.call(
            17,
            2,
            "# Fraction of the frame that a channel maintain hysteresis condition before another trigger can be considered (0.5 or 50%, not 50)",
        ),
        mock.call(18, 1, "Signal Generator Type"),
        mock.call(
            18,
            2,
            '# One of "None", "Random", "Burst Random", "Pseudorandom", "Chirp", "Square", or "Sine"',
        ),
        mock.call(19, 1, "Signal Generator Level"),
        mock.call(
            19,
            2,
            "# RMS voltage level for random signals, Peak voltage level for chirp, sine, and square pulse",
        ),
        mock.call(20, 1, "Signal Generator Frequency 1"),
        mock.call(
            20,
            2,
            "# Minimum frequency for broadband signals or frequency for sine and square  pulse",
        ),
        mock.call(21, 1, "Signal Generator Frequency 2"),
        mock.call(
            21, 2, "# Maximum frequency for broadband signals.  Ignored for sine and square pulse"
        ),
        mock.call(22, 1, "Signal Generator On Fraction"),
        mock.call(
            22, 2, "# Fraction of time that the burst or square wave is on (0.5 or 50%, not 50)"
        ),
        mock.call(23, 1, "Wait Time for Steady State"),
        mock.call(
            23, 2, "# Time to wait after output starts to allow the system to reach steady state"
        ),
        mock.call(24, 1, "Autoaccept Script"),
        mock.call(24, 2, "# File in which an autoacceptance function is defined"),
        mock.call(25, 1, "Autoaccept Function"),
        mock.call(25, 2, "# Function name in which the autoacceptance function is defined"),
        mock.call(26, 1, "Reference Channels"),
        mock.call(26, 2, "# List of channels, one per cell on this row"),
        mock.call(27, 1, "Disabled Channels"),
        mock.call(27, 2, "# List of channels, one per cell on this row"),
    ]
    return environment_template_calls


def create_environment_collector_metadata_calls():
    metadata_call = [
        mock.call(
            1,
            [1, 2],
            [],
            AcquisitionType.FREE_RUN,
            Acceptance.AUTOMATIC,
            None,
            0.0,
            0,
            TriggerSlope.POSITIVE,
            0.01,
            0.02,
            0,
            0.0,
            10,
            Window.RECTANGLE,
            response_transformation_matrix=None,
            reference_transformation_matrix=None,
            window_parameter_2=-10 / np.log(0.25),
        )
    ]
    return metadata_call


def create_environment_spectral_metadata_calls():
    metadata_call = [
        mock.call(
            AveragingTypes.LINEAR,
            30,
            0.1,
            Estimator.H1,
            2,
            0,
            204.8,
            2048,
            6,
            compute_cpsd=False,
            compute_apsd=True,
        )
    ]
    return metadata_call


def create_environment_signal_metadata_calls():
    metadata_call = [
        mock.call(
            samples_per_write=102400,
            level_ramp_samples=1,
            output_transformation_matrix=None,
            disabled_signals=[0],
        )
    ]
    return metadata_call


def create_environment_start_calls(mock_collect, mock_signal, mock_siggen, mock_spectral):
    environment_name = "Environment_name"
    start_calls = [
        mock.call(
            environment_name, (DataCollectorCommands.FORCE_INITIALIZE_COLLECTOR, mock_collect())
        ),
        mock.call(environment_name, (DataCollectorCommands.SET_TEST_LEVEL, (0, 1))),
        mock.call(
            environment_name, (SignalGenerationCommands.INITIALIZE_PARAMETERS, mock_signal())
        ),
        mock.call(
            environment_name, (SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR, mock_siggen())
        ),
        mock.call(environment_name, (SignalGenerationCommands.MUTE, None)),
        mock.call(environment_name, (SignalGenerationCommands.ADJUST_TEST_LEVEL, 1.0)),
        mock.call(environment_name, (DataCollectorCommands.ACQUIRE, None)),
        mock.call(environment_name, (SignalGenerationCommands.GENERATE_SIGNALS, None)),
        mock.call(
            environment_name, (SpectralProcessingCommands.INITIALIZE_PARAMETERS, mock_spectral())
        ),
        mock.call(environment_name, (SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING, None)),
        mock.call(environment_name, (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING, None)),
        mock.call(environment_name, (ModalCommands.RUN_CONTROL, None)),
    ]
    return start_calls
