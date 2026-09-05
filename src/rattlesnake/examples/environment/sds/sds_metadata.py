import numpy as np
import netCDF4 as nc4
import openpyxl

import rattlesnake.examples.defaults as defaults
from rattlesnake.utilities import GlobalCommands
from rattlesnake.load_utilities import load_profile_from_workbook
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
from rattlesnake.environment.sds_sys_id_environment import SDSEnvironment
from rattlesnake.environment.sds_sys_id_utilities import (
    SDSInstructions,
    decayed_sine_table,
    octspace,
)
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.utilities import DIRECTORY

ENVIRONMENT_NAME = "SDS 0"


def manual_sds_metadata(hardware_metadata, **overrides):
    channel_list_bools = [True] * len(hardware_metadata.channel_list)
    sample_rate = hardware_metadata.sample_rate
    num_channels = 21
    block_size = 2048
    tone_data = ToneParameters(
        tone_strategy=ToneStrategy.FROM_SPEC,
        tone_data=None,
    )
    compensation_pulse_data = CompPulseParameters(
        use_compensation_pulse=True,
        compensation_frequency=None,
        compensation_decay=0.95,
    )
    decay_data = DecayParameters(
        decay_strategy=DecayStrategy.NUM_TIME_CONSTANTS,
        common_decay=True,
        decay_data=5,
    )
    srs_data = SRSParameters(
        srs_type=SRSType.MAXIMUM_ABSMAX,
        srs_displacement=SRSDisplacementType.ABSOLUTE,
        srs_damping=0.03,
    )
    sds_data = SDSParameters(
        iterations=3,
        convergence=0.8,
        scale_factor=1.02,
        error_tolerance=5,
    )
    control_script_data = ControlParameters(
        control_script=DIRECTORY + r"\environment\sds_sys_id_control_law.py",
        control_object="default_control_law",
        control_type=ControlLawType.FUNCTION,
        control_parameters={"rcond": 1e-10, "accuracy_weight": 100, "input_weight": 1},
    )
    control_channel_indices = [0, 1, 2]
    output_channel_indices = [12, 13, 14, 15, 16, 17, 18, 19, 20]
    response_transformation_matrix = None
    excitation_transformation_matrix = None
    frequencies = octspace(20, 0.9 * sample_rate / 2, 3)
    num_freq = len(frequencies)

    num_control_channels = len(control_channel_indices)
    specification_data = SpecParameters(
        frequencies=frequencies,
        srs_spec=np.ones((num_freq, num_control_channels)),
        srs_lower_limit=np.ones((num_freq, num_control_channels)) * 0.5,
        srs_upper_limit=np.ones((num_freq, num_control_channels)) * 1.5,
        num_hits=10,
    )

    kwargs = dict(
        environment_name=ENVIRONMENT_NAME,
        channel_list_bools=channel_list_bools,
        sample_rate=sample_rate,
        num_channels=num_channels,
        block_size=block_size,
        tone_data=tone_data,
        compensation_pulse_data=compensation_pulse_data,
        decay_data=decay_data,
        srs_data=srs_data,
        sds_data=sds_data,
        control_script_data=control_script_data,
        control_channel_indices=control_channel_indices,
        output_channel_indices=output_channel_indices,
        response_transformation_matrix=response_transformation_matrix,
        excitation_transformation_matrix=excitation_transformation_matrix,
        specification_data=specification_data,
        sysid_metadata=None,
    )
    kwargs.update(overrides)
    return SDSMetadata(**kwargs)


def netcdf_sds_metadata():
    pass


def worksheet_sds_metadata():
    pass


def sds_instructions():
    control_test_level = 0.0
    target_hits_at_level = 1
    automatic_hits = False
    automatic_interval = None
    allow_automatic_updates = False
    tone_frequencies = octspace(20, 0.9 * defaults.SAMPLE_RATE / 2, 3)
    compensation_frequency = tone_frequencies.min() / 3
    frequencies = np.concatenate((tone_frequencies, [compensation_frequency]))
    num_drive_channels = 9
    sds_table = decayed_sine_table(
        frequency=frequencies,
        amplitude=np.zeros((len(frequencies), num_drive_channels)),
        decay=np.zeros((len(frequencies), num_drive_channels)),
        delay=np.zeros((len(frequencies), num_drive_channels)),
    )

    return SDSInstructions(
        environment_name=ENVIRONMENT_NAME,
        control_test_level=control_test_level,
        target_hits_at_level=target_hits_at_level,
        automatic_hits=automatic_hits,
        automatic_interval=automatic_interval,
        sds_table=sds_table,
        allow_automatic_updates=allow_automatic_updates,
    )


def sds_event_list():
    return []


def worksheet_sds_event_list():
    pass
