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
    ControlLawType
)
from rattlesnake.environment.sds_sys_id_environment import SDSEnvironment
from rattlesnake.environment.sds_sys_id_utilities import SDSInstructions
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.environment.environment_utilities import EnvironmentType

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
        compensation_frequency=0,
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
        srs_damping=3,
    )
    sds_data = SDSParameters(
        iterations=3,
        convergence=0.8,
        scale_factor=1.02,
        error_tolerance=5,
    )
    control_script_data = ControlParameters(
        control_script=defaults.DIRECTORY + "environment/sds_sys_id_control_law.py",
        control_object="default_control_law",
        control_type=ControlLawType.FUNCTION
        control_parameters="round=1e-10\naccuracy_weight=100\ninput_weight=1"
    )
    control_channel_indices = [0, 1, 2]
    output_channel_indices = [12, 13, 14, 15, 16, 17, 18, 19, 20]
    response_transformation_matrix = None
    excitation_transformation_matrix = None

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
    pass


def sds_event_list():
    pass


def worksheet_sds_event_list():
    pass
