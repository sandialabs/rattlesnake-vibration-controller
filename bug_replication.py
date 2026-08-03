import time
import threading

import netCDF4 as nc4
from qtpy import QtCore

from rattlesnake.headless import *
from rattlesnake.examples import *
from rattlesnake.testing import *


def modal_environment_plot_issues():
    with test_example_rattlesnake_object(
        hardware_type=HardwareType.SDYNPY_SYSTEM,
        environment_type=EnvironmentType.MODAL,
        start_hardware=True,
    ) as rattlesnake:
        ui_event_list = [
            UIEvent(5, lambda ui: ui.environment_uis["Modal 0"].preview_acquisition()),
            UIEvent(7.5, lambda ui: ui.environment_uis["Modal 0"].new_window()),
            UIEvent(10, lambda ui: ui.environment_uis["Modal 0"].stop_environment()),
            UIEvent(
                12.5,
                lambda ui: ui.environment_uis[
                    "Modal 0"
                ].definition_widget.samples_per_frame_selector.setValue(2048),
            ),
            UIEvent(15, lambda ui: ui.initialize_environments()),
            UIEvent(17.5, lambda ui: ui.stop_acquisition()),
            UIEvent(20, lambda ui: ui.initialize_environments()),
        ]
        launch_temporary_rattlesnake_ui(
            rattlesnake, ui_event_list, closeout_time=30, display_errors=False
        )


def profile_event_crash():
    """Profile events can error out when going to fast. (ex. Stop Environment then immediate Start Environment). It should
    either skip validation and send the command anyways or just stop the profile from firing future events and continuing
    the crash"""
    rattlesnake = build_example_rattlesnake_object(
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


def dual_sysid_environment():
    with test_example_rattlesnake_object(
        threaded=True, hardware_type=HardwareType.SDYNPY_SYSTEM
    ) as rattlesnake:
        metadata_1 = manual_random_metadata(
            rattlesnake.hardware_metadata, environment_name="Random 0"
        )
        metadata_1.sysid_metadata = manual_sysid_metadata(rattlesnake.hardware_metadata)
        metadata_1.sysid_metadata.sysid_level_ramp_time = 2.5
        metadata_2 = manual_random_metadata(
            rattlesnake.hardware_metadata, environment_name="Random 1"
        )
        metadata_2.sysid_metadata = manual_sysid_metadata(rattlesnake.hardware_metadata)

        rattlesnake.initialize_environments([metadata_1, metadata_2])

        # rattlesnake.initialize_system_id(
        #     manual_sysid_metadata(rattlesnake.hardware_metadata), "Random 0"
        # )
        rattlesnake.load_system_id_from_package("Random 0", netcdf_sysid_data_package())

        launch_rattlesnake_ui(rattlesnake)


if __name__ == "__main__":
    modal_environment_plot_issues()
