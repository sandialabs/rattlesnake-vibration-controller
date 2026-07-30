from qtpy import QtCore

from rattlesnake.engine import RattlesnakeController
from rattlesnake.main import build_rattlesnake_app
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.environment_registry import SYSID_ENVIRONMENTS
from rattlesnake.process.streaming import StreamType


def mock_rattlesnake_object(**overrides):
    """Builds a RattlesnakeController, defaulting to the smallest object possible.

    Mirrors ``build_rattlesnake_object`` in
    ``rattlesnake.examples.headless_example``, but defaults to no hardware,
    no environment, and nothing started/run. Override any keyword the same
    way the ``mock_utilities`` builders are overridden, e.g.
    ``mock_rattlesnake_object(hardware_type=HardwareType.SDYNPY_SYSTEM)``.
    """
    # Imported here (rather than at module level) because
    # rattlesnake.examples.__init__ imports build_rattlesnake_object from
    # headless_example, which imports from this module -- importing
    # rattlesnake.examples anything at module level here would be circular.
    from rattlesnake.examples.example_registry import (
        HARDWARE_DICT,
        ENVIRONMENT_DICT,
        SYSID_DICT,
        SYSID_LOAD_DICT,
        STREAM_DICT,
        INSTRUCTIONS_DICT,
        EVENT_DICT,
    )

    kwargs = dict(
        threaded=False,
        timeout=20,
        import_method="manual",
        hardware_type=HardwareType.NONE,
        environment_type=EnvironmentType.NONE,
        stream_type=StreamType.NO_STREAM,
        load_sysid=False,
        run_sysid=False,
        start_hardware=False,
        start_environment=False,
        run_profile=False,
    )
    kwargs.update(overrides)

    rattlesnake = RattlesnakeController(
        threaded=kwargs["threaded"], timeout=kwargs["timeout"]
    )

    # Initialize hardware
    if kwargs["hardware_type"] is HardwareType.NONE:
        return rattlesnake
    hardware_metadata = HARDWARE_DICT[kwargs["hardware_type"]][kwargs["import_method"]]()
    rattlesnake.initialize_hardware(hardware_metadata)

    # Initialize environment
    if kwargs["environment_type"] is EnvironmentType.NONE:
        return rattlesnake
    environment_metadata = ENVIRONMENT_DICT[kwargs["environment_type"]][
        kwargs["import_method"]
    ](hardware_metadata)
    environment_name = getattr(environment_metadata, "environment_name", None)
    rattlesnake.initialize_environments([environment_metadata])

    # Run System Identification
    if kwargs["environment_type"] in SYSID_ENVIRONMENTS:
        sysid_metadata = SYSID_DICT[kwargs["import_method"]](hardware_metadata)
        rattlesnake.initialize_system_id(sysid_metadata, environment_name)
        if kwargs["run_sysid"]:
            rattlesnake.run_system_id(sysid_metadata, environment_name)
        if kwargs["load_sysid"]:
            sysid_package = SYSID_LOAD_DICT["netcdf"]()
            rattlesnake.load_system_id_from_package(environment_name, sysid_package)

    # Initialize profile event list and stream metadata in the UI (Don't normally need this for headless)
    event_list = EVENT_DICT[kwargs["environment_type"]][kwargs["import_method"]]()
    stream_metadata = STREAM_DICT[kwargs["stream_type"]](environment_name)
    rattlesnake.initialize_profile_event_list(event_list)
    rattlesnake.set_stream_metadata(stream_metadata)

    # Start Hardware
    if not kwargs["start_hardware"]:
        return rattlesnake
    rattlesnake.start_acquisition(stream_metadata)

    # Start Environment
    if kwargs["run_profile"]:
        rattlesnake.start_profile(event_list)
    elif kwargs["start_environment"]:
        instructions = INSTRUCTIONS_DICT[kwargs["environment_type"]]()
        rattlesnake.start_environment(instructions=instructions)
    return rattlesnake


def launch_temporary_rattlesnake_ui_profile(
    rattlesnake: RattlesnakeController, closeout_time: float
):
    """
    Launches the rattlesnake ui, auto-starting a profile shortly after open
    and auto-closing after closeout_time, for unattended testing.

    Parameters
    ----------
    rattlesnake : RattlesnakeController
        The rattlesnake controller object that the UI is going to represent.
    closeout_time : float
        Time, in seconds, after which the UI will automatically close.
    """
    app, ui = build_rattlesnake_app(rattlesnake)
    ui.show()
    QtCore.QTimer.singleShot(int(closeout_time * 1000), ui.close)
    profile_start = 20
    QtCore.QTimer.singleShot(int(profile_start * 1000), ui.start_profile)
    app.exec_()

    # Shutdown processes
    ui.shutdown()
    rattlesnake.shutdown()


def launch_temporary_rattlesnake_ui_environment(
    rattlesnake: RattlesnakeController, closeout_time: float
):
    """
    Launches the rattlesnake ui, auto-stopping acquisition partway through
    and auto-closing after closeout_time, for unattended testing.

    Parameters
    ----------
    rattlesnake : RattlesnakeController
        The rattlesnake controller object that the UI is going to represent.
    closeout_time : float
        Time, in seconds, after which the UI will automatically close.
    """
    app, ui = build_rattlesnake_app(rattlesnake)
    ui.show()
    QtCore.QTimer.singleShot(int(closeout_time * 1000), ui.close)
    environment_end = 40
    QtCore.QTimer.singleShot(int(environment_end * 1000), ui.stop_acquisition)
    app.exec_()

    # Shutdown processes
    ui.shutdown()
    rattlesnake.shutdown()
