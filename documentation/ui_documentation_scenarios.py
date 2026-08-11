from dataclasses import dataclass
from typing import Dict, Any

from qtpy import QtWidgets, QtTest
import time

from rattlesnake.examples.frame_wing.mimo_random import get_mimo_random_ui
from rattlesnake.examples.frame_wing.mimo_transient import get_mimo_transient_ui
from rattlesnake.examples.frame_wing.mimo_sine import get_mimo_sine_ui
from rattlesnake.examples.frame_wing.mimo_sds import get_mimo_sds_ui


@dataclass
class UIDocScenarioResult:
    name: str
    handle: Any
    main_ui: Any
    environment_ui: Any
    widgets: Dict[str, QtWidgets.QWidget]

    def cleanup(self):
        try:
            self.main_ui.shutdown()
        except Exception:
            pass
        try:
            self.handle.rattlesnake.shutdown()
        except Exception:
            pass


def _build_result(name, handle, env_ui):
    return UIDocScenarioResult(
        name=name,
        handle=handle,
        main_ui=handle.rattlesnake_ui,
        environment_ui=env_ui,
        widgets={
            "main": handle.rattlesnake_ui,
            "definition": env_ui.definition_widget,
            "system_id": env_ui.system_id_widget,
            "prediction": env_ui.prediction_widget,
            "run": env_ui.run_widget,
        },
    )


def get_environment_queue_name(handle, environment_name):
    return handle.rattlesnake.environment_manager.queue_names_dict[environment_name]


def wait_for_environment_state(handle, environment_name, active, timeout=60.0):
    queue_name = get_environment_queue_name(handle, environment_name)
    event = handle.rattlesnake.event_container.environment_active_events[queue_name]

    start = time.time()
    while True:
        QtWidgets.QApplication.processEvents()
        if event.is_set() == active:
            return True
        if time.time() - start > timeout:
            return False
        QtTest.QTest.qWait(100)


def run_environment_for_documentation(
    handle,
    env_ui,
    *,
    manual_stop_after_seconds=None,
    completion_timeout=120.0,
):
    env_ui.start_environment()

    started = wait_for_environment_state(
        handle,
        env_ui.environment_name,
        active=True,
        timeout=30.0,
    )
    if not started:
        raise RuntimeError(f"Environment {env_ui.environment_name} did not become active in time.")

    QtWidgets.QApplication.processEvents()

    if manual_stop_after_seconds is not None:
        QtTest.QTest.qWait(int(manual_stop_after_seconds * 1000))
        QtWidgets.QApplication.processEvents()
        env_ui.stop_environment()
        finished = wait_for_environment_state(
            handle,
            env_ui.environment_name,
            active=False,
            timeout=completion_timeout,
        )
    else:
        finished = wait_for_environment_state(
            handle,
            env_ui.environment_name,
            active=False,
            timeout=completion_timeout,
        )

    if not finished:
        raise RuntimeError(f"Environment {env_ui.environment_name} did not finish in time.")

    QtWidgets.QApplication.processEvents()


def random_environment_scenario(display_errors=False):
    handle, env_ui = get_mimo_random_ui(display_errors=display_errors)
    QtWidgets.QApplication.processEvents()

    run_environment_for_documentation(
        handle,
        env_ui,
        manual_stop_after_seconds=10.0,
        completion_timeout=60.0,
    )

    return _build_result("random", handle, env_ui)


def transient_environment_scenario(display_errors=False):
    handle, env_ui = get_mimo_transient_ui(display_errors=display_errors)
    QtWidgets.QApplication.processEvents()

    run_environment_for_documentation(
        handle,
        env_ui,
        manual_stop_after_seconds=None,
        completion_timeout=120.0,
    )

    return _build_result("transient", handle, env_ui)


def sine_environment_scenario(display_errors=False):
    handle, env_ui = get_mimo_sine_ui(display_errors=display_errors)
    QtWidgets.QApplication.processEvents()

    run_environment_for_documentation(
        handle,
        env_ui,
        manual_stop_after_seconds=None,
        completion_timeout=120.0,
    )

    return _build_result("sine", handle, env_ui)


def sds_environment_scenario(display_errors=False):
    handle, env_ui = get_mimo_sds_ui(display_errors=display_errors)
    QtWidgets.QApplication.processEvents()

    run_environment_for_documentation(
        handle,
        env_ui,
        manual_stop_after_seconds=None,
        completion_timeout=180.0,
    )

    return _build_result("sds", handle, env_ui)


ENVIRONMENT_SCENARIOS = {
    "random": random_environment_scenario,
    "transient": transient_environment_scenario,
    "sine": sine_environment_scenario,
    "sds": sds_environment_scenario,
}