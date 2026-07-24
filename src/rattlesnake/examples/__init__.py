from rattlesnake.examples.headless_example import build_rattlesnake_object
from rattlesnake.examples.hardware.exodus.exodus_metadata import (
    worksheet_exodus_metadata,
    netcdf_exodus_metadata,
    manual_exodus_metadata,
)
from rattlesnake.examples.hardware.sdynpy_frf.sdynpy_frf_metadata import (
    worksheet_sdynpy_frf_metadata,
    netcdf_sdynpy_frf_metadata,
    manual_sdynpy_frf_metadata,
)
from rattlesnake.examples.hardware.sdynpy_system.sdynpy_system_metadata import (
    worksheet_sdynpy_system_metadata,
    netcdf_sdynpy_system_metadata,
    manual_sdynpy_system_metadata,
)
from rattlesnake.examples.hardware.state_space.state_space_metadata import (
    worksheet_state_space_metadata,
    netcdf_state_space_metadata,
    manual_state_space_metadata,
)
from rattlesnake.examples.hardware.stream_metadata import (
    stream_metadata_no,
    stream_metadata_immediate,
    stream_metadata_manual,
    stream_metadata_test_level,
    stream_metadata_profile,
)

from rattlesnake.examples.environment.modal.modal_metadata import (
    worksheet_modal_metadata,
    netcdf_modal_metadata,
    manual_modal_metadata,
    modal_instructions,
    modal_event_list,
    worksheet_modal_event_list,
)
from rattlesnake.examples.environment.random.random_metadata import (
    worksheet_random_metadata,
    netcdf_random_metadata,
    manual_random_metadata,
    create_sine_specification as create_random_cpsd_specification,
    random_instructions,
    random_event_list,
    worksheet_random_event_list,
)
from rattlesnake.examples.environment.sine.sine_metadata import (
    worksheet_sine_metadata,
    netcdf_sine_metadata,
    manual_sine_metadata,
    create_sine_specification,
    sine_instructions,
    sine_event_list,
    worksheet_sine_event_list,
)
from rattlesnake.examples.environment.skeleton.skeleton_metadata import (
    worksheet_skeleton_metadata,
    netcdf_skeleton_metadata,
    manual_skeleton_metadata,
    skeleton_instructions,
    skeleton_event_list,
    worksheet_skeleton_event_list,
)
from rattlesnake.examples.environment.skeleton_sysid.skeleton_sysid_metadata import (
    worksheet_skeleton_sysid_metadata,
    netcdf_skeleton_sysid_metadata,
    manual_skeleton_sysid_metadata,
    skeleton_sysid_instructions,
)
from rattlesnake.examples.environment.sysid.sysid_metadata import (
    netcdf_sysid_data_package,
    netcdf_sysid_metadata,
    worksheet_sysid_metadata,
    manual_sysid_metadata,
)
from rattlesnake.examples.environment.time.time_metadata import (
    create_time_signal,
    worksheet_time_metadata,
    netcdf_time_metadata,
    manual_time_metadata,
    time_instructions,
    time_event_list,
    worksheet_time_event_list,
)
from rattlesnake.examples.environment.transient.transient_metadata import (
    worksheet_transient_metadata,
    netcdf_transient_metadata,
    manual_transient_metadata,
    create_control_signal,
    transient_instructions,
    transient_event_list,
    worksheet_transient_event_list,
)

__all__ = [
    # Main
    "build_rattlesnake_object",
    # Exodus
    "worksheet_exodus_metadata",
    "netcdf_exodus_metadata",
    "manual_exodus_metadata",
    # Sdynpy_frf
    "worksheet_sdynpy_frf_metadata",
    "netcdf_sdynpy_frf_metadata",
    "manual_sdynpy_frf_metadata",
    # Sdynpy_system
    "worksheet_sdynpy_system_metadata",
    "netcdf_sdynpy_system_metadata",
    "manual_sdynpy_system_metadata",
    # State_space
    "worksheet_state_space_metadata",
    "netcdf_state_space_metadata",
    "manual_state_space_metadata",
    # Streaming
    "stream_metadata_no",
    "stream_metadata_immediate",
    "stream_metadata_manual",
    "stream_metadata_test_level",
    "stream_metadata_profile",
    # Modal
    "worksheet_modal_metadata",
    "netcdf_modal_metadata",
    "manual_modal_metadata",
    "modal_instructions",
    "modal_event_list",
    "worksheet_modal_event_list",
    # Random
    "worksheet_random_metadata",
    "netcdf_random_metadata",
    "manual_random_metadata",
    "create_random_cpsd_specification",
    "random_instructions",
    "random_event_list",
    "worksheet_random_event_list",
    # Sine
    "worksheet_sine_metadata",
    "netcdf_sine_metadata",
    "manual_sine_metadata",
    "create_sine_specification",
    "sine_instructions",
    "sine_event_list",
    "worksheet_sine_event_list",
    # Skeleton
    "worksheet_skeleton_metadata",
    "netcdf_skeleton_metadata",
    "manual_skeleton_metadata",
    "skeleton_instructions",
    "skeleton_event_list",
    "worksheet_skeleton_event_list",
    # Skeleton_sysid
    "worksheet_skeleton_sysid_metadata",
    "netcdf_skeleton_sysid_metadata",
    "manual_skeleton_sysid_metadata",
    "skeleton_sysid_instructions",
    # Sysid
    "netcdf_sysid_data_package",
    "netcdf_sysid_metadata",
    "worksheet_sysid_metadata",
    "manual_sysid_metadata",
    # Time
    "create_time_signal",
    "worksheet_time_metadata",
    "netcdf_time_metadata",
    "manual_time_metadata",
    "time_instructions",
    "time_event_list",
    "worksheet_time_event_list",
    # Transient
    "worksheet_transient_metadata",
    "netcdf_transient_metadata",
    "manual_transient_metadata",
    "create_control_signal",
    "transient_instructions",
    "transient_event_list",
    "worksheet_transient_event_list",
]
