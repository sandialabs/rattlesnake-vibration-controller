import openpyxl as opxl
import numpy as np

from rattlesnake.engine import RattlesnakeController
from rattlesnake.main import launch_rattlesnake_ui
from rattlesnake.user_interface.user_interface import build_rattlesnake_app
from rattlesnake.hardware.state_space_virtual_hardware import StateSpaceMetadata
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
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
from rattlesnake.process.abstract_sysid_data_analysis import SysIdMetadata
from rattlesnake.process.streaming import StreamType, StreamMetadata
from rattlesnake.examples.defaults import DIRECTORY as rattlesnake_directory


def build_mimo_sds_controller(threaded=False):
    """
    Build a Rattlesnake controller initialized to a populated MIMO SDS example state.

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

    environment_metadata = SDSMetadata(
        environment_name="SDS",
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

    rattlesnake.initialize_environments([environment_metadata])

    # rattlesnake.initialize_system_id(sysid_parameters, environment_metadata.environment_name)
    # rattlesnake.run_system_id(sysid_parameters, environment_metadata.environment_name)

    # rattlesnake.initialize_profile_event_list([])

    # stream_metadata = StreamMetadata(StreamType.NO_STREAM, stream_file=None)
    # rattlesnake.start_acquisition(stream_metadata)

    return rattlesnake


def build_mimo_sds_app(threaded=False, display_errors=False):
    """
    Build a populated MIMO SDS example UI without entering the Qt event loop.

    Returns
    -------
    RattlesnakeAppHandle
        Handle containing controller, UI, and QApplication.
    """
    rattlesnake = build_mimo_sds_controller(threaded=threaded)
    return build_rattlesnake_app(rattlesnake, display_errors=display_errors)


def get_mimo_sds_ui(threaded=False, display_errors=False):
    """
    Convenience helper returning the SDS environment UI.

    Returns
    -------
    tuple
        (handle, sds_ui)
    """
    handle = build_mimo_sds_app(threaded=threaded, display_errors=display_errors)
    sds_ui = handle.rattlesnake_ui.environment_uis["SDS"]
    return handle, sds_ui


if __name__ == "__main__":
    rattlesnake = build_mimo_sds_controller(threaded=False)
    launch_rattlesnake_ui(rattlesnake)