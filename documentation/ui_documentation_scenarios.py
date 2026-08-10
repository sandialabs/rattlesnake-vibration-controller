from dataclasses import dataclass
from typing import Callable, Dict, Any

from qtpy import QtWidgets

from rattlesnake.examples.frame_wing.mimo_random import get_mimo_random_ui
from rattlesnake.examples.frame_wing.mimo_transient import get_mimo_transient_ui
from rattlesnake.examples.frame_wing.mimo_sine import get_mimo_sine_ui
from rattlesnake.examples.frame_wing.mimo_sds import get_mimo_sds_ui


@dataclass
class UIDocScenarioResult:
    """
    Container returned by a UI documentation scenario builder.
    """

    name: str
    handle: Any
    main_ui: Any
    environment_ui: Any
    widgets: Dict[str, QtWidgets.QWidget]

    def cleanup(self):
        """
        Shut down the example cleanly.
        """
        try:
            self.main_ui.shutdown()
        except Exception:
            pass
        try:
            self.handle.rattlesnake.shutdown()
        except Exception:
            pass


def _build_result(name, handle, environment_name, env_ui):
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


# -----------------------------------------------------------------------------
# Random scenarios
# -----------------------------------------------------------------------------
def random_definition_scenario(display_errors=False):
    handle, env_ui = get_mimo_random_ui(display_errors=display_errors)
    handle.rattlesnake_ui.rattlesnake_tabs.setCurrentIndex(1)  # Environment Definition
    for i in range(handle.rattlesnake_ui.environment_definition_environment_tabs.count()):
        if (
            handle.rattlesnake_ui.environment_definition_environment_tabs.tabText(i)
            == env_ui.environment_name
        ):
            handle.rattlesnake_ui.environment_definition_environment_tabs.setCurrentIndex(i)
            break
    QtWidgets.QApplication.processEvents()
    return _build_result("random_definition", handle, env_ui.environment_name, env_ui)


def random_prediction_scenario(display_errors=False):
    handle, env_ui = get_mimo_random_ui(display_errors=display_errors)
    handle.rattlesnake_ui.rattlesnake_tabs.setCurrentIndex(3)  # Test Predictions
    for i in range(handle.rattlesnake_ui.test_prediction_environment_tabs.count()):
        if (
            handle.rattlesnake_ui.test_prediction_environment_tabs.tabText(i)
            == env_ui.environment_name
        ):
            handle.rattlesnake_ui.test_prediction_environment_tabs.setCurrentIndex(i)
            break
    QtWidgets.QApplication.processEvents()
    return _build_result("random_prediction", handle, env_ui.environment_name, env_ui)


def random_run_scenario(display_errors=False):
    handle, env_ui = get_mimo_random_ui(display_errors=display_errors)
    handle.rattlesnake_ui.rattlesnake_tabs.setCurrentIndex(5)  # Run Test
    for i in range(handle.rattlesnake_ui.run_environment_tabs.count()):
        if handle.rattlesnake_ui.run_environment_tabs.tabText(i) == env_ui.environment_name:
            handle.rattlesnake_ui.run_environment_tabs.setCurrentIndex(i)
            break
    QtWidgets.QApplication.processEvents()
    return _build_result("random_run", handle, env_ui.environment_name, env_ui)


# -----------------------------------------------------------------------------
# Transient scenarios
# -----------------------------------------------------------------------------
def transient_definition_scenario(display_errors=False):
    handle, env_ui = get_mimo_transient_ui(display_errors=display_errors)
    handle.rattlesnake_ui.rattlesnake_tabs.setCurrentIndex(1)
    for i in range(handle.rattlesnake_ui.environment_definition_environment_tabs.count()):
        if (
            handle.rattlesnake_ui.environment_definition_environment_tabs.tabText(i)
            == env_ui.environment_name
        ):
            handle.rattlesnake_ui.environment_definition_environment_tabs.setCurrentIndex(i)
            break
    QtWidgets.QApplication.processEvents()
    return _build_result("transient_definition", handle, env_ui.environment_name, env_ui)


def transient_prediction_scenario(display_errors=False):
    handle, env_ui = get_mimo_transient_ui(display_errors=display_errors)
    handle.rattlesnake_ui.rattlesnake_tabs.setCurrentIndex(3)
    for i in range(handle.rattlesnake_ui.test_prediction_environment_tabs.count()):
        if (
            handle.rattlesnake_ui.test_prediction_environment_tabs.tabText(i)
            == env_ui.environment_name
        ):
            handle.rattlesnake_ui.test_prediction_environment_tabs.setCurrentIndex(i)
            break
    QtWidgets.QApplication.processEvents()
    return _build_result("transient_prediction", handle, env_ui.environment_name, env_ui)


def transient_run_scenario(display_errors=False):
    handle, env_ui = get_mimo_transient_ui(display_errors=display_errors)
    handle.rattlesnake_ui.rattlesnake_tabs.setCurrentIndex(5)
    for i in range(handle.rattlesnake_ui.run_environment_tabs.count()):
        if handle.rattlesnake_ui.run_environment_tabs.tabText(i) == env_ui.environment_name:
            handle.rattlesnake_ui.run_environment_tabs.setCurrentIndex(i)
            break
    QtWidgets.QApplication.processEvents()
    return _build_result("transient_run", handle, env_ui.environment_name, env_ui)


# -----------------------------------------------------------------------------
# Sine scenarios
# -----------------------------------------------------------------------------
def sine_definition_scenario(display_errors=False):
    handle, env_ui = get_mimo_sine_ui(display_errors=display_errors)
    handle.rattlesnake_ui.rattlesnake_tabs.setCurrentIndex(1)
    for i in range(handle.rattlesnake_ui.environment_definition_environment_tabs.count()):
        if (
            handle.rattlesnake_ui.environment_definition_environment_tabs.tabText(i)
            == env_ui.environment_name
        ):
            handle.rattlesnake_ui.environment_definition_environment_tabs.setCurrentIndex(i)
            break
    QtWidgets.QApplication.processEvents()
    return _build_result("sine_definition", handle, env_ui.environment_name, env_ui)


def sine_prediction_scenario(display_errors=False):
    handle, env_ui = get_mimo_sine_ui(display_errors=display_errors)
    handle.rattlesnake_ui.rattlesnake_tabs.setCurrentIndex(3)
    for i in range(handle.rattlesnake_ui.test_prediction_environment_tabs.count()):
        if (
            handle.rattlesnake_ui.test_prediction_environment_tabs.tabText(i)
            == env_ui.environment_name
        ):
            handle.rattlesnake_ui.test_prediction_environment_tabs.setCurrentIndex(i)
            break
    QtWidgets.QApplication.processEvents()
    return _build_result("sine_prediction", handle, env_ui.environment_name, env_ui)


def sine_run_scenario(display_errors=False):
    handle, env_ui = get_mimo_sine_ui(display_errors=display_errors)
    handle.rattlesnake_ui.rattlesnake_tabs.setCurrentIndex(5)
    for i in range(handle.rattlesnake_ui.run_environment_tabs.count()):
        if handle.rattlesnake_ui.run_environment_tabs.tabText(i) == env_ui.environment_name:
            handle.rattlesnake_ui.run_environment_tabs.setCurrentIndex(i)
            break
    QtWidgets.QApplication.processEvents()
    return _build_result("sine_run", handle, env_ui.environment_name, env_ui)


# -----------------------------------------------------------------------------
# SDS scenarios
# -----------------------------------------------------------------------------
def sds_definition_scenario(display_errors=False):
    handle, env_ui = get_mimo_sds_ui(display_errors=display_errors)
    handle.rattlesnake_ui.rattlesnake_tabs.setCurrentIndex(1)
    for i in range(handle.rattlesnake_ui.environment_definition_environment_tabs.count()):
        if (
            handle.rattlesnake_ui.environment_definition_environment_tabs.tabText(i)
            == env_ui.environment_name
        ):
            handle.rattlesnake_ui.environment_definition_environment_tabs.setCurrentIndex(i)
            break
    QtWidgets.QApplication.processEvents()
    return _build_result("sds_definition", handle, env_ui.environment_name, env_ui)


def sds_prediction_scenario(display_errors=False):
    handle, env_ui = get_mimo_sds_ui(display_errors=display_errors)
    handle.rattlesnake_ui.rattlesnake_tabs.setCurrentIndex(3)
    for i in range(handle.rattlesnake_ui.test_prediction_environment_tabs.count()):
        if (
            handle.rattlesnake_ui.test_prediction_environment_tabs.tabText(i)
            == env_ui.environment_name
        ):
            handle.rattlesnake_ui.test_prediction_environment_tabs.setCurrentIndex(i)
            break
    QtWidgets.QApplication.processEvents()
    return _build_result("sds_prediction", handle, env_ui.environment_name, env_ui)


def sds_run_scenario(display_errors=False):
    handle, env_ui = get_mimo_sds_ui(display_errors=display_errors)
    handle.rattlesnake_ui.rattlesnake_tabs.setCurrentIndex(5)
    for i in range(handle.rattlesnake_ui.run_environment_tabs.count()):
        if handle.rattlesnake_ui.run_environment_tabs.tabText(i) == env_ui.environment_name:
            handle.rattlesnake_ui.run_environment_tabs.setCurrentIndex(i)
            break
    QtWidgets.QApplication.processEvents()
    return _build_result("sds_run", handle, env_ui.environment_name, env_ui)


UI_DOC_SCENARIOS = {
    "random_definition": random_definition_scenario,
    "random_prediction": random_prediction_scenario,
    "random_run": random_run_scenario,
    "transient_definition": transient_definition_scenario,
    "transient_prediction": transient_prediction_scenario,
    "transient_run": transient_run_scenario,
    "sine_definition": sine_definition_scenario,
    "sine_prediction": sine_prediction_scenario,
    "sine_run": sine_run_scenario,
    "sds_definition": sds_definition_scenario,
    "sds_prediction": sds_prediction_scenario,
    "sds_run": sds_run_scenario,
}
