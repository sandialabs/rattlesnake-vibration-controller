import openpyxl as opxl

from rattlesnake.engine import RattlesnakeController
from rattlesnake.main import launch_rattlesnake_ui
from rattlesnake.user_interface.user_interface import build_rattlesnake_app
from rattlesnake.hardware.state_space_virtual_hardware import StateSpaceMetadata
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.random_vibration_sys_id_environment import (
    RandomVibrationMetadata,
)
from rattlesnake.environment.random_vibration_sys_id_utilities import load_specification
from rattlesnake.process.abstract_sysid_data_analysis import SysIdMetadata
from rattlesnake.process.streaming import StreamType, StreamMetadata
from rattlesnake.examples.defaults import DIRECTORY as rattlesnake_directory


def build_mimo_random_controller(threaded=False):
    """
    Build a Rattlesnake controller initialized to a populated MIMO Random example state.

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
        sysid_frame_size=4096,
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

    spec_freq_lines, spec_cpsd_matrix, spec_warning_matrix, spec_abort_matrix = load_specification(
        spec_path=rattlesnake_directory + "/frame_wing/data/random_vibration_specification.npz",
        n_freq_lines=sysid_parameters.sysid_frame_size // 2 + 1,
        df=sysid_parameters.sample_rate / sysid_parameters.sysid_frame_size,
    )

    environment_metadata = RandomVibrationMetadata(
        environment_name="Random",
        channel_list_bools=[True] * len(hardware_metadata.channel_list),
        sample_rate=hardware_metadata.sample_rate,
        number_of_channels=len(hardware_metadata.channel_list),
        samples_per_frame=sysid_parameters.sysid_frame_size,
        test_level_ramp_time=0.5,
        cola_window="Tukey",
        cola_overlap=0.5,
        cola_window_exponent=0.5,
        sigma_clip=5,
        update_tf_during_control=False,
        frames_in_cpsd=20,
        cpsd_window="Hann",
        cpsd_overlap=0.5,
        percent_lines_out=10.0,
        allow_automatic_aborts=False,
        control_python_script=rattlesnake_directory + "/control_laws/control_laws.py",
        control_python_function="buzz_control",
        control_python_function_type=0,
        control_python_function_parameters="",
        control_channel_indices=[
            index
            for index, channel in enumerate(channel_list)
            if channel.channel_type == "Acceleration" and channel.node_direction == "Y+"
        ],
        output_channel_indices=[
            index for index, channel in enumerate(channel_list) if channel.feedback_device is not None
        ],
        specification_frequency_lines=spec_freq_lines,
        specification_cpsd_matrix=spec_cpsd_matrix,
        specification_warning_matrix=spec_warning_matrix,
        specification_abort_matrix=spec_abort_matrix,
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


def build_mimo_random_app(threaded=False, display_errors=False):
    """
    Build a populated MIMO Random example UI without entering the Qt event loop.

    Returns
    -------
    RattlesnakeAppHandle
        Handle containing controller, UI, and QApplication.
    """
    rattlesnake = build_mimo_random_controller(threaded=threaded)
    return build_rattlesnake_app(rattlesnake, display_errors=display_errors)


def get_mimo_random_ui(threaded=False, display_errors=False):
    """
    Convenience helper returning the Random environment UI.

    Returns
    -------
    tuple
        (handle, random_ui)
    """
    handle = build_mimo_random_app(threaded=threaded, display_errors=display_errors)
    random_ui = handle.rattlesnake_ui.environment_uis["Random"]
    return handle, random_ui


if __name__ == "__main__":
    rattlesnake = build_mimo_random_controller(threaded=False)
    launch_rattlesnake_ui(rattlesnake)