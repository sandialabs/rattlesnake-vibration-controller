import functools

from qtpy import QtCore, QtWidgets

from rattlesnake.engine import RattlesnakeController
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.process.streaming import StreamType
from rattlesnake.user_interface.user_interface import (
    build_rattlesnake_app,
    RattlesnakeUI,
)
from rattlesnake.examples.headless_example import build_example_rattlesnake_object


# region Rattlesnake
def test_example_rattlesnake_object(**overrides):
    """
    Builds a rattlesnake object with overrides
    """
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

    return build_example_rattlesnake_object(**kwargs)


def initialize_rattlesnake_object(
    hardware_metadata=None,
    environment_metadata_list=None,
    profile_event_list=None,
    stream_metadata=None,
):
    rattlesnake = RattlesnakeController()

    if not hardware_metadata:
        return rattlesnake
    rattlesnake.initialize_hardware(hardware_metadata)

    if not environment_metadata_list:
        return rattlesnake
    rattlesnake.initialize_environments(environment_metadata_list)

    if not profile_event_list:
        return rattlesnake
    rattlesnake.initialize_profile_event_list(profile_event_list)

    if not stream_metadata:
        return rattlesnake
    rattlesnake.set_stream_metadata(stream_metadata)

    return rattlesnake


# endregion


# region User Interface
class UIEvent:
    """A single scripted action to fire against a live RattlesnakeUI.

    Mirrors ``rattlesnake.profile_manager.ProfileEvent``'s
    ``(timestamp, target, command, data)`` shape, but targets a live UI
    instead of an environment command queue.

    Parameters
    ----------
    timestamp : float
        Time, in seconds after the UI is shown, at which to fire the event.
    action : str or Callable
        If a string, the name of a ``RattlesnakeUI`` method to call, e.g.
        ``UIEvent(20, "start_profile")``. If a callable, called as
        ``action(ui, *args, **kwargs)`` -- an escape hatch for anything finer
        than a single method call, e.g.
        ``UIEvent(5, lambda ui: QTest.mouseClick(ui.some_button, QtCore.Qt.LeftButton))``.
    *args, **kwargs
        Passed through to ``action`` when it fires.
    """

    def __init__(self, timestamp: float, action, *args, **kwargs):
        self.timestamp = timestamp
        self.action = action
        self.args = args
        self.kwargs = kwargs

    def fire(self, ui):
        if isinstance(self.action, str):
            return getattr(ui, self.action)(*self.args, **self.kwargs)
        return self.action(ui, *self.args, **self.kwargs)


def launch_temporary_rattlesnake_ui(
    rattlesnake: RattlesnakeController,
    ui_event_list: list = [],
    closeout_time: float = None,
    *,
    rattlesnake_ui: RattlesnakeUI = None,
    app: QtWidgets.QApplication = None,
    set_font_size: bool = True,
    display_errors: bool = False,
) -> None:
    """
    Launches the rattlesnake ui, firing each UIEvent in ``ui_event_list`` at
    its scheduled timestamp and auto-closing after closeout_time, for
    unattended testing.

    Parameters
    ----------
    rattlesnake : RattlesnakeController
        The rattlesnake controller object that the UI is going to represent.
    ui_event_list : list[UIEvent]
        The scripted actions to fire against the UI, e.g.
        ``[UIEvent(20, "start_profile")]``.
    closeout_time : float
        Time, in seconds, after which the UI will automatically close.
    """
    if rattlesnake_ui is None and app is None:
        with build_rattlesnake_app(
            rattlesnake,
            set_font_size=set_font_size,
            display_errors=display_errors,
        ) as (
            rattlesnake,
            rattlesnake_ui,
            app,
        ):
            rattlesnake_ui.show()
            for event in ui_event_list:
                QtCore.QTimer.singleShot(
                    int(event.timestamp * 1000),
                    functools.partial(event.fire, rattlesnake_ui),
                )
            if closeout_time:
                QtCore.QTimer.singleShot(
                    int(closeout_time * 1000), rattlesnake_ui.close
                )
            app.exec_()
    else:
        rattlesnake_ui.show()
        for event in ui_event_list:
            QtCore.QTimer.singleShot(
                int(event.timestamp * 1000),
                functools.partial(event.fire, rattlesnake_ui),
            )
        if closeout_time:
            QtCore.QTimer.singleShot(int(closeout_time * 1000), rattlesnake_ui.close)
        app.exec_()


# endregion
