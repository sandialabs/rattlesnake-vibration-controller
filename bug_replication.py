import time

import netCDF4 as nc4

from rattlesnake.headless import *
from rattlesnake.examples import *



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

if __name__ == "__main__":
    pass