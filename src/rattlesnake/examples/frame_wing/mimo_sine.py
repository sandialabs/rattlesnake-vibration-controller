import openpyxl as opxl

from rattlesnake.engine import RattlesnakeController
from rattlesnake.main import launch_rattlesnake_ui
from rattlesnake.user_interface.user_interface import build_rattlesnake_app
from rattlesnake.hardware.state_space_virtual_hardware import StateSpaceMetadata
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.sine_sys_id_environment import SineMetadata
from rattlesnake.environment.sine_sys_id_utilities import load_specification, SineSpecification
from rattlesnake.process.abstract_sysid_data_analysis import SysIdMetadata
from rattlesnake.process.streaming import StreamType, StreamMetadata
from rattlesnake.examples.defaults import DIRECTORY as rattlesnake_directory


def build_mimo_sine_controller(threaded=False):
    """
    Build a Rattlesnake controller initialized to a populated MIMO Sine example state.

    Returns
    -------
    RattlesnakeController
        Controller with hardware, environment, system identification, and
        acquisition initialized.
    """
    rattlesnake = RattlesnakeController(threaded=threaded, timeout=1000)

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

    sysid_parameters = SysIdMetadata(
        sample_rate=hardware_metadata.sample_rate,
        sysid_frame_size=int(hardware_metadata.sample_rate // 2),
        sysid_averaging_type="Linear",
        sysid_noise_averages=5,
        sysid_averages=20,
        sysid_exponential_averaging_coefficient=0.01,
        sysid_estimator="H1",
        sysid_level=0.01,
        sysid_level_ramp_time=0.5,
        sysid_signal_type="Random",
        sysid_window="Hann",
        sysid_overlap=0.5,
        sysid_burst_on=0.5,
        sysid_pretrigger=0.05,
        sysid_burst_ramp_fraction=0.05,
        sysid_low_frequency_cutoff=0,
        sysid_high_frequency_cutoff=hardware_metadata.sample_rate // 2,
        stream_file=None,
    )

    control_channel_indices = [
        index
        for index, channel in enumerate(channel_list)
        if channel.channel_type == "Acceleration" and channel.node_direction == "Y+"
    ]

    (
        frequencies,
        amplitudes,
        phases,
        sweep_types,
        sweep_rates,
        warnings,
        aborts,
        start_time,
        name,
    ) = load_specification(rattlesnake_directory + "/frame_wing/data/sine_specification.npz")

    spec = SineSpecification(
        name=name,
        start_time=start_time,
        num_control=len(control_channel_indices),
        frequency_breakpoints=frequencies,
        amplitude_breakpoints=amplitudes,
        phase_breakpoints=phases,
        sweep_type_breakpoints=sweep_types,
        sweep_rate_breakpoints=sweep_rates,
        warning_breakpoints=warnings,
        abort_breakpoints=aborts,
    )

    environment_metadata = SineMetadata(
        environment_name="Sine",
        channel_list_bools=[True] * len(hardware_metadata.channel_list),
        sample_rate=hardware_metadata.sample_rate,
        samples_per_frame=hardware_metadata.samples_per_read,
        number_of_channels=len(hardware_metadata.channel_list),
        specifications=[spec],
        ramp_time=0.5,
        buffer_blocks=2,
        control_convergence=0.0,
        update_drives_after_environment=False,
        phase_fit=False,
        allow_automatic_aborts=False,
        tracking_filter_type=0,
        tracking_filter_cutoff=0.15,
        tracking_filter_order=2,
        vk_filter_order=2,
        vk_filter_bandwidth=20,
        vk_filter_blocksize=5000,
        vk_filter_overlap=0.5,
        control_python_script=None,
        control_python_class=None,
        control_python_parameters="",
        control_channel_indices=control_channel_indices,
        output_channel_indices=[
            index for index, channel in enumerate(channel_list) if channel.feedback_device is not None
        ],
        response_transformation_matrix=None,
        output_transformation_matrix=None,
        sysid_metadata=sysid_parameters,
    )

    rattlesnake.initialize_environments([environment_metadata])

    rattlesnake.initialize_system_id(sysid_parameters, environment_metadata.environment_name)
    rattlesnake.run_system_id(sysid_parameters, environment_metadata.environment_name)

    rattlesnake.initialize_profile_event_list([])

    stream_metadata = StreamMetadata(StreamType.NO_STREAM, stream_file=None)
    rattlesnake.start_acquisition(stream_metadata)

    return rattlesnake


def build_mimo_sine_app(threaded=False, display_errors=False):
    """
    Build a populated MIMO Sine example UI without entering the Qt event loop.

    Returns
    -------
    RattlesnakeAppHandle
        Handle containing controller, UI, and QApplication.
    """
    rattlesnake = build_mimo_sine_controller(threaded=threaded)
    return build_rattlesnake_app(rattlesnake, display_errors=display_errors)


def get_mimo_sine_ui(threaded=False, display_errors=False):
    """
    Convenience helper returning the Sine environment UI.

    Returns
    -------
    tuple
        (handle, sine_ui)
    """
    handle = build_mimo_sine_app(threaded=threaded, display_errors=display_errors)
    sine_ui = handle.rattlesnake_ui.environment_uis["Sine"]
    return handle, sine_ui


if __name__ == "__main__":
    rattlesnake = build_mimo_sine_controller(threaded=False)
    launch_rattlesnake_ui(rattlesnake)