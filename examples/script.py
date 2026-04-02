from rattlesnake.rattlesnake import RattlesnakeController
from rattlesnake.main import launch_rattlesnake_ui
from .metadata import (
    make_sdynpy_system_metadata,
    make_time_environment_metadata,
    make_time_environment_event_list,
    make_time_environment_stream_metadata,
    make_time_environment_instructions,
    make_modal_environment_metadata,
    make_sine_environment_metadata,
)


def build_time_environment():
    rattlesnake = RattlesnakeController(threaded=True, timeout=30)
    hardware_metadata = make_sdynpy_system_metadata()
    time_environment_metadata = make_time_environment_metadata(hardware_metadata)
    time_profile_event_list = make_time_environment_event_list()
    time_stream_metadata = make_time_environment_stream_metadata()
    time_environment_instructions = make_time_environment_instructions()

    rattlesnake.initialize_hardware_metadata(hardware_metadata)
    # rattlesnake.set_environments([time_environment_metadata])
    # rattlesnake.set_profile_event_list(time_profile_event_list)
    # rattlesnake.set_stream_metadata(time_stream_metadata)
    # rattlesnake.start_acquisition(time_stream_metadata)
    # rattlesnake.start_environment(time_environment_instructions)

    return rattlesnake


if __name__ == "__main__":
    rattlesnake = build_time_environment()

    launch_rattlesnake_ui(rattlesnake)
