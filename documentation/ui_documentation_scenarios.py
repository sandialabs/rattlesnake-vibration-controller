from dataclasses import dataclass
from typing import Dict, Any

from qtpy import QtWidgets

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


def random_environment_scenario(display_errors=False):
    handle, env_ui = get_mimo_random_ui(display_errors=display_errors)
    QtWidgets.QApplication.processEvents()
    return _build_result("random", handle, env_ui)


def transient_environment_scenario(display_errors=False):
    handle, env_ui = get_mimo_transient_ui(display_errors=display_errors)
    QtWidgets.QApplication.processEvents()
    return _build_result("transient", handle, env_ui)


def sine_environment_scenario(display_errors=False):
    handle, env_ui = get_mimo_sine_ui(display_errors=display_errors)
    QtWidgets.QApplication.processEvents()
    return _build_result("sine", handle, env_ui)


def sds_environment_scenario(display_errors=False):
    handle, env_ui = get_mimo_sds_ui(display_errors=display_errors)
    QtWidgets.QApplication.processEvents()
    return _build_result("sds", handle, env_ui)


ENVIRONMENT_SCENARIOS = {
    "random": random_environment_scenario,
    "transient": transient_environment_scenario,
    "sine": sine_environment_scenario,
    "sds": sds_environment_scenario,
}