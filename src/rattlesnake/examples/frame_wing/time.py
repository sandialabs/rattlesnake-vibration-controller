import openpyxl as opxl
import numpy as np

from rattlesnake.engine import RattlesnakeController
from rattlesnake.main import launch_rattlesnake_ui
from rattlesnake.user_interface.user_interface import build_rattlesnake_app
from rattlesnake.hardware.state_space_virtual_hardware import StateSpaceMetadata
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.time_environment import TimeMetadata
from rattlesnake.process.streaming import StreamType, StreamMetadata
from rattlesnake.examples.defaults import DIRECTORY as rattlesnake_directory


def build_time_signal(sample_rate):
    """
    Build a 3-channel, 1-second signal with a short haversine pulse near the start.

    Parameters
    ----------
    sample_rate : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        Array of shape (3, sample_rate)
    """
    n_samples = sample_rate
    t = np.arange(n_samples) / sample_rate

    signal = np.zeros((3, n_samples))

    # Short haversine pulse near the start
    pulse_start = int(0.05 * sample_rate)
    pulse_width = int(0.025 * sample_rate)  # 25 ms wide
    pulse_t = np.arange(pulse_width) / pulse_width
    pulse = 0.5 * (1 - np.cos(2 * np.pi * pulse_t))

    signal[0, pulse_start : pulse_start + pulse_width] = pulse
    signal[1, pulse_start : pulse_start + pulse_width] = 0.7 * pulse
    signal[2, pulse_start : pulse_start + pulse_width] = 0.4 * pulse

    return signal


def build_time_controller(threaded=False):
    """
    Build a Rattlesnake controller initialized to a populated Time example state.

    Uses the shared frame wing channel table and state-space virtual hardware.
    A 3-channel, 1-second transient drive signal is loaded into the environment.

    Returns
    -------
    RattlesnakeController
        Controller with hardware, time environment, and acquisition initialized.
    """
    rattlesnake = RattlesnakeController(threaded=threaded)

    channel_list = HardwareMetadata.load_channel_table_from_workbook(
        opxl.load_workbook(rattlesnake_directory + "/frame_wing/data/channel_table.xlsx")
    )

    hardware_metadata = StateSpaceMetadata(
        channel_list,
        sample_rate=8192,
        time_per_read=0.25,
        time_per_write=0.25,
        output_oversample=10,
        hardware_file=rattlesnake_directory + "/frame_wing/data/rattlesnake_state_space_system.npz",
    )

    rattlesnake.initialize_hardware(hardware_metadata)

    output_signal = build_time_signal(hardware_metadata.output_sample_rate)

    environment_metadata = TimeMetadata(
        environment_name="Time",
        channel_list_bools=[True] * len(hardware_metadata.channel_list),
        sample_rate=hardware_metadata.sample_rate,
        output_oversample=hardware_metadata.output_oversample,
        output_signal=output_signal,
        cancel_rampdown_time=0.5,
    )

    rattlesnake.initialize_environments([environment_metadata])
    rattlesnake.initialize_profile_event_list([])

    stream_metadata = StreamMetadata(StreamType.NO_STREAM, stream_file=None)
    rattlesnake.start_acquisition(stream_metadata)

    return rattlesnake


def build_time_app(threaded=False, display_errors=False):
    """
    Build a populated Time example UI without entering the Qt event loop.

    Returns
    -------
    RattlesnakeAppHandle
        Handle containing controller, UI, and QApplication.
    """
    rattlesnake = build_time_controller(threaded=threaded)
    return build_rattlesnake_app(rattlesnake, display_errors=display_errors)


def get_time_ui(threaded=False, display_errors=False):
    """
    Convenience helper returning the Time environment UI.

    Returns
    -------
    tuple
        (handle, time_ui)
    """
    handle = build_time_app(threaded=threaded, display_errors=display_errors)
    time_ui = handle.rattlesnake_ui.environment_uis["Time"]
    return handle, time_ui


if __name__ == "__main__":
    rattlesnake = build_time_controller(threaded=False)
    launch_rattlesnake_ui(rattlesnake)
