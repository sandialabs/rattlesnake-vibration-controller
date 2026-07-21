import time

import netCDF4 as nc4

import rattlesnake.headless as headless
from rattlesnake.utilities import GlobalCommands
from rattlesnake.profile_manager import ProfileEvent
from rattlesnake.main import launch_rattlesnake_ui
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.modal_environment import ModalInstructions
from rattlesnake.process.streaming import StreamType
from rattlesnake.examples.headless_example import build_rattlesnake_object


def transient_zero_trac():
    """Transient example problem results in zero division at the end which sends TRAC values to 0. When starting
    transient environment from headless, Unable to Parse Line is written"""
    rattlesnake = build_rattlesnake_object(
        threaded=False,
        timeout=20,
        import_method="manual",
        hardware_type=HardwareType.SDYNPY_SYSTEM,
        environment_type=EnvironmentType.TRANSIENT,
        stream_type=StreamType.NO_STREAM,
        load_sysid=True,
        run_sysid=False,
        start_hardware=True,
        start_environment=True,
        run_profile=False,
    )
    time.sleep(10)
    rattlesnake.stop_environment("Transient 0")
    rattlesnake.shutdown()


def profile_event_crash():
    """Profile events can error out when going to fast. (ex. Stop Environment then immediate Start Environment). It should
    either skip validation and send the command anyways or just stop the profile from firing future events and continuing
    the crash"""
    rattlesnake = build_rattlesnake_object(
        threaded=False,
        timeout=20,
        import_method="manual",
        hardware_type=HardwareType.SDYNPY_SYSTEM,
        environment_type=EnvironmentType.MODAL,
        stream_type=StreamType.NO_STREAM,
        load_sysid=True,
        run_sysid=False,
        start_hardware=True,
        start_environment=False,
        run_profile=False,
    )

    timestamp = 1
    command = GlobalCommands.START_ENVIRONMENT
    instructions = ModalInstructions("Modal 0")
    start_environment_event = ProfileEvent(timestamp, "Modal 0", command, instructions)

    timestamp = 1.01
    command = GlobalCommands.STOP_ENVIRONMENT
    stop_environment_event = ProfileEvent(timestamp, "Modal 0", command)

    timestamp = 5
    command = GlobalCommands.START_ENVIRONMENT
    instructions = ModalInstructions("Modal 0")
    start_environment_event_2 = ProfileEvent(
        timestamp, "Modal 0", command, instructions
    )

    profile_event_list = [
        start_environment_event,
        stop_environment_event,
        start_environment_event_2,
    ]

    rattlesnake.start_profile(profile_event_list)

    time.sleep(10)
    rattlesnake.shutdown()


def sysid_environment_zero_divide():
    """Figure out the zero devide in this to see if it is even a real issue"""
    rattlesnake = build_rattlesnake_object(
        threaded=True,
        timeout=20,
        import_method="manual",
        hardware_type=HardwareType.SDYNPY_SYSTEM,
        environment_type=EnvironmentType.SINE,
        stream_type=StreamType.NO_STREAM,
        load_sysid=True,
        run_sysid=False,
        start_hardware=False,
        start_environment=False,
        run_profile=False,
    )
    time.sleep(2)
    rattlesnake.shutdown()


def profile_event_button_disable():
    """Start profile directly into stop acqusition causes the acqusition to time out and the start/stop profile button to both be
    disabled. I think this happens because the profile timers are not fully started yet so the stop button runs stop_profile but
    the timers are not started so it errs out. This really matters for the user interface button stuff more than the crash since
    it is not common to want to start a profile and immediately exit out."""
    rattlesnake = build_rattlesnake_object(
        threaded=False,
        timeout=20,
        import_method="manual",
        hardware_type=HardwareType.SDYNPY_SYSTEM,
        environment_type=EnvironmentType.MODAL,
        stream_type=StreamType.NO_STREAM,
        load_sysid=False,
        run_sysid=False,
        start_hardware=True,
        start_environment=False,
        run_profile=False,
    )

    timestamp = 1
    command = GlobalCommands.START_ENVIRONMENT
    instructions = ModalInstructions("Modal 0")
    start_environment_event = ProfileEvent(timestamp, "Modal 0", command, instructions)

    timestamp = 10
    command = GlobalCommands.STOP_ENVIRONMENT
    stop_environment_event = ProfileEvent(timestamp, "Modal 0", command)

    profile_event_list = [
        start_environment_event,
        stop_environment_event,
    ]
    print("Starting Profile")

    rattlesnake.start_profile(profile_event_list)
    print("Stopping Acqusition")
    rattlesnake.stop_acquisition()

    rattlesnake.shutdown()


from rattlesnake.examples.hardware.sdynpy_system.sdynpy_system_metadata import (
    manual_sdynpy_system_metadata,
)
from rattlesnake.examples.environment.modal.modal_metadata import manual_modal_metadata


def modal_environment_slow_writing():
    rattlesnake = headless.RattlesnakeController()

    hardware_metadata = manual_sdynpy_system_metadata()
    hardware_metadata.sample_rate = 15000
    rattlesnake.initialize_hardware(hardware_metadata)

    environment_metadata = manual_modal_metadata(hardware_metadata)
    environment_metadata.samples_per_frame = 30000
    rattlesnake.initialize_environments([environment_metadata])

    rattlesnake.start_acquisition(headless.StreamMetadata())
    time.sleep(1)

    modal_instructions = ModalInstructions("Modal 0", "test_file.nc4")
    rattlesnake.start_environment(modal_instructions)
    time.sleep(80)

    rattlesnake.stop_acquisition()

    time.sleep(5)

    rattlesnake.shutdown()

    dataset = nc4.Dataset("test_file.nc4", mode="r+")
    print(f"{dataset.variables["time_data"].shape}")


if __name__ == "__main__":

    dataset = nc4.Dataset("test_file.nc4", mode="r+")
    dataset.close()
    modal_environment_slow_writing()
