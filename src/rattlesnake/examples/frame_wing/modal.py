import openpyxl as opxl

from rattlesnake.engine import RattlesnakeController
from rattlesnake.main import launch_rattlesnake_ui
from rattlesnake.user_interface.user_interface import build_rattlesnake_app
from rattlesnake.hardware.state_space_virtual_hardware import StateSpaceMetadata
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.modal_environment import ModalMetadata
from rattlesnake.process.streaming import StreamType, StreamMetadata
from rattlesnake.examples.defaults import DIRECTORY as rattlesnake_directory


def build_modal_controller(threaded=False):
    """
    Build a Rattlesnake controller initialized to a populated Modal example state.

    Uses the shared frame wing channel table and state-space virtual hardware.
    Force channels are assigned as references, and a burst-random excitation is configured.

    Returns
    -------
    RattlesnakeController
        Controller with hardware, modal environment, and acquisition initialized.
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

    # In the modal environment, reference channels are chosen from the active channels.
    # Use physical force channels as references.
    reference_channel_indices = [
        i for i, ch in enumerate(channel_list) if ch.channel_type == "Force"
    ]

    # Response channels are enabled non-reference channels.
    response_channel_indices = [
        i for i, ch in enumerate(channel_list) if ch.channel_type != "Force"
    ]

    output_channel_indices = [
        i for i, ch in enumerate(channel_list) if ch.feedback_device is not None
    ]

    environment_metadata = ModalMetadata(
        environment_name="Modal",
        channel_list_bools=[True] * len(hardware_metadata.channel_list),
        sample_rate=hardware_metadata.sample_rate,
        samples_per_frame=2 * hardware_metadata.sample_rate,  # 2 second frame
        averaging_type="Linear",
        num_averages=15,
        averaging_coefficient=0.1,
        frf_technique="H1",
        frf_window="rectangle",
        overlap_percent=0.0,
        trigger_type="First Frame",
        accept_type="Accept All",
        wait_for_steady_state=0.0,
        trigger_channel=23,
        pretrigger_percent=2.0,
        trigger_slope_positive=True,
        trigger_level_percent=0.5,
        hysteresis_level_percent=0.25,
        hysteresis_frame_percent=10.0,
        signal_generator_type="burst",
        signal_generator_level=0.1,
        signal_generator_min_frequency=20.0,
        signal_generator_max_frequency=2000.0,
        signal_generator_on_percent=50.0,
        acceptance_function=None,
        reference_channel_indices=reference_channel_indices,
        response_channel_indices=response_channel_indices,
        output_channel_indices=output_channel_indices,
        output_oversample=hardware_metadata.output_oversample,
        exponential_window_value_at_frame_end=0.5,
    )

    rattlesnake.initialize_environments([environment_metadata])
    rattlesnake.initialize_profile_event_list([])

    stream_metadata = StreamMetadata(StreamType.NO_STREAM, stream_file=None)
    rattlesnake.start_acquisition(stream_metadata)

    return rattlesnake


def build_modal_app(threaded=False, display_errors=False):
    """
    Build a populated Modal example UI without entering the Qt event loop.

    Returns
    -------
    RattlesnakeAppHandle
        Handle containing controller, UI, and QApplication.
    """
    rattlesnake = build_modal_controller(threaded=threaded)
    return build_rattlesnake_app(rattlesnake, display_errors=display_errors)


def get_modal_ui(threaded=False, display_errors=False):
    """
    Convenience helper returning the Modal environment UI.

    Returns
    -------
    tuple
        (handle, modal_ui)
    """
    handle = build_modal_app(threaded=threaded, display_errors=display_errors)
    modal_ui = handle.rattlesnake_ui.environment_uis["Modal"]
    return handle, modal_ui


if __name__ == "__main__":
    rattlesnake = build_modal_controller(threaded=False)
    launch_rattlesnake_ui(rattlesnake)
