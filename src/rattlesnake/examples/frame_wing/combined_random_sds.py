import openpyxl as opxl
import numpy as np

from rattlesnake.engine import RattlesnakeController
from rattlesnake.main import launch_rattlesnake_ui
from rattlesnake.user_interface.user_interface import build_rattlesnake_app
from rattlesnake.hardware.state_space_virtual_hardware import StateSpaceMetadata
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.random_vibration_sys_id_environment import (
    RandomVibrationMetadata,
    RandomVibrationInstructions,
    RandomVibrationCommands,
)
from rattlesnake.environment.sds_sys_id_metadata import (
    SDSMetadata,
    ToneParameters,
    ToneStrategy,
    CompPulseParameters,
    DecayParameters,
    DecayStrategy,
    SRSParameters,
    SRSType,
    SRSDisplacementType,
    SDSParameters,
    ControlParameters,
    ControlLawType,
    SpecParameters,
)
from rattlesnake.environment.sds_sys_id_utilities import SDSInstructions, SDSCommands
from rattlesnake.environment.random_vibration_sys_id_utilities import load_specification
from rattlesnake.process.abstract_sysid_data_analysis import SysIdMetadata
from rattlesnake.process.streaming import StreamType, StreamMetadata
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.utilities import GlobalCommands
from rattlesnake.examples.defaults import DIRECTORY as rattlesnake_directory



def build_combined_controller(threaded=False):
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

    random_environment_metadata = RandomVibrationMetadata(
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
            index
            for index, channel in enumerate(channel_list)
            if channel.feedback_device is not None
        ],
        specification_frequency_lines=spec_freq_lines,
        specification_cpsd_matrix=spec_cpsd_matrix,
        specification_warning_matrix=spec_warning_matrix,
        specification_abort_matrix=spec_abort_matrix,
        response_transformation_matrix=None,
        output_transformation_matrix=None,
        sysid_metadata=sysid_parameters,
    )

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
    output_channel_indices = [
        index for index, channel in enumerate(channel_list) if channel.feedback_device is not None
    ]

    tone_parameters = ToneParameters(
        tone_strategy=ToneStrategy.FROM_SPEC,
        tone_data=None,
    )

    compensation_pulse_parameters = CompPulseParameters(
        use_compensation_pulse=True,
        compensation_frequency=None,
        compensation_decay=0.95,
    )

    decay_parameters = DecayParameters(
        decay_strategy=DecayStrategy.NUM_TIME_CONSTANTS,
        common_decay=True,
        decay_data=5.0,
    )

    srs_parameters = SRSParameters(
        srs_type=SRSType.MAXIMUM_ABSMAX,
        srs_displacement=SRSDisplacementType.ABSOLUTE,
        srs_damping=0.03,
    )

    sds_parameters = SDSParameters(
        iterations=3,
        convergence=0.8,
        scale_factor=1.02,
        error_tolerance=0.05,
    )

    spec_data = np.load(rattlesnake_directory + "/frame_wing/data/srs_specification.npz")
    specification = SpecParameters(
        frequencies=spec_data["f"],
        srs_spec=spec_data["srs"],
        srs_lower_limit=spec_data["lower_limit"],
        srs_upper_limit=spec_data["upper_limit"],
        num_hits=spec_data["num_hits"],
    )

    control_parameters = ControlParameters(
        control_script="rattlesnake.environment.sds_sys_id_control_law",
        control_object="default_control_law",
        control_type=ControlLawType.FUNCTION,
        control_parameters={},
    )

    sds_environment_metadata = SDSMetadata(
        environment_name="Shock",
        channel_list_bools=[True] * len(hardware_metadata.channel_list),
        sample_rate=hardware_metadata.sample_rate,
        num_channels=len(hardware_metadata.channel_list),
        block_size=int(hardware_metadata.sample_rate),
        tone_data=tone_parameters,
        compensation_pulse_data=compensation_pulse_parameters,
        decay_data=decay_parameters,
        srs_data=srs_parameters,
        sds_data=sds_parameters,
        control_script_data=control_parameters,
        control_channel_indices=control_channel_indices,
        output_channel_indices=output_channel_indices,
        response_transformation_matrix=None,
        excitation_transformation_matrix=None,
        specification_data=specification,
        sysid_metadata=sysid_parameters,
    )

    rattlesnake.initialize_environments([random_environment_metadata, sds_environment_metadata])

    rattlesnake.initialize_system_id(sysid_parameters, random_environment_metadata.environment_name)
    rattlesnake.run_system_id(sysid_parameters, random_environment_metadata.environment_name)

    rattlesnake.initialize_system_id(sysid_parameters, sds_environment_metadata.environment_name)
    rattlesnake.run_system_id(sysid_parameters, sds_environment_metadata.environment_name)

    profile_event_list = []
    timestamp = 0
    command = RandomVibrationCommands.ADJUST_TEST_LEVEL
    data = -6
    profile_event_list.append(
        ProfileEvent(timestamp, random_environment_metadata.environment_name, command, data)
    )

    timestamp = 0
    command = GlobalCommands.START_ENVIRONMENT
    instructions = RandomVibrationInstructions(
        random_environment_metadata.environment_name, control_test_level=-6
    )
    profile_event_list.append(
        ProfileEvent(timestamp, random_environment_metadata.environment_name, command, instructions)
    )

    timestamp = 5
    command = RandomVibrationCommands.ADJUST_TEST_LEVEL
    data = -3
    profile_event_list.append(
        ProfileEvent(timestamp, random_environment_metadata.environment_name, command, data)
    )

    timestamp = 10
    command = RandomVibrationCommands.ADJUST_TEST_LEVEL
    data = 0
    profile_event_list.append(
        ProfileEvent(timestamp, random_environment_metadata.environment_name, command, data)
    )

    timestamp = 20
    command = GlobalCommands.START_ENVIRONMENT
    instructions = SDSInstructions(
        sds_environment_metadata.environment_name,
        control_test_level=0,
        target_hits_at_level=5,
        automatic_hits=True,
        automatic_interval=1,
        sds_table=None,
        allow_automatic_updates=False,
    )
    profile_event_list.append(
        ProfileEvent(timestamp, sds_environment_metadata.environment_name, command, instructions)
    )

    timestamp = 40
    command = GlobalCommands.STOP_ENVIRONMENT
    profile_event_list.append(
        ProfileEvent(timestamp, random_environment_metadata.environment_name, command)
    )

    rattlesnake.initialize_profile_event_list(profile_event_list)

    stream_metadata = StreamMetadata(StreamType.NO_STREAM, stream_file=None)
    rattlesnake.start_acquisition(stream_metadata)

    return rattlesnake


def build_combined_app(threaded=False, display_errors=False):
    """
    Build a populated MIMO Random example UI without entering the Qt event loop.

    Returns
    -------
    RattlesnakeAppHandle
        Handle containing controller, UI, and QApplication.
    """
    rattlesnake = build_combined_controller(threaded=threaded)
    return build_rattlesnake_app(rattlesnake, display_errors=display_errors)


def get_combined_ui(threaded=False, display_errors=False):
    """
    Convenience helper returning the Random environment UI.

    Returns
    -------
    tuple
        (handle, random_ui)
    """
    handle = build_combined_app(threaded=threaded, display_errors=display_errors)
    random_ui = handle.rattlesnake_ui.environment_uis["Random"]
    return handle, random_ui


if __name__ == "__main__":
    rattlesnake = build_combined_controller(threaded=False)
    launch_rattlesnake_ui(rattlesnake)
