# -*- coding: utf-8 -*-
"""
This script is a first attempt to automate documentation of the Rattlesnake user interface,
pulling the layout of the widgets and their tooltips and creating a markdown file that can
be included in the main Rattlesnake documentation.

Rattlesnake Vibration Control Software
Copyright (C) 2021  National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import re
import sys
from qtpy import QtWidgets, uic, QtTest, QtCore
from qtpy.QtCore import QRect, QPoint
from qtpy.QtGui import QPixmap, QPainter, QPen, QColor
from ui_documentation_scenarios import ENVIRONMENT_SCENARIOS

try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    dir_path = "."

generated_dir = os.path.join(dir_path, "book", "src", "_generated")
figures_dir = os.path.join(generated_dir, "figures")

os.makedirs(generated_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

files = [
    dir_path + "/" + v
    for v in [
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/random_vibration_definition.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/random_vibration_prediction.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/random_vibration_run.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/modal_definition.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/modal_run.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/transient_definition.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/transient_prediction.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/transient_run.ui",
        "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/time_run.ui",
        "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/time_definition.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/system_identification.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/sine_definition.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/sine_prediction.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/sine_run.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/srs_sds_definition.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/srs_sds_prediction.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/srs_sds_run.ui",
    ]
]

# "../src/rattlesnake/components/ip_manager.ui",
# "../src/rattlesnake/components/environment_selector.ui",
# "../src/rattlesnake/components/combined_environments_controller.ui",
# "../src/rattlesnake/components/control_select.ui",

UI_FILE_TO_SCENARIO = {
    "random_vibration_definition.ui": ("random", "definition"),
    "random_vibration_prediction.ui": ("random", "prediction"),
    "random_vibration_run.ui": ("random", "run"),
    "system_identification.ui": ("random", "system_id"),
    "transient_definition.ui": ("transient", "definition"),
    "transient_prediction.ui": ("transient", "prediction"),
    "transient_run.ui": ("transient", "run"),
    "sine_definition.ui": ("sine", "definition"),
    "sine_prediction.ui": ("sine", "prediction"),
    "sine_run.ui": ("sine", "run"),
    "srs_sds_definition.ui": ("sds", "definition"),
    "srs_sds_prediction.ui": ("sds", "prediction"),
    "srs_sds_run.ui": ("sds", "run"),
    "modal_definition.ui": ("modal", "definition"),
    "modal_run.ui": ("modal", "run"),
    "time_definition.ui": ("time", "definition"),
    "time_run.ui": ("time", "run"),
}

PAGE_STATES = {
    "sine_definition.ui": {
        "states": [
            {
                "name": "tracking_filter",
                "actions": [
                    {"action": "set_current_index", "widget": "filter_type_selector", "value": 0},
                ],
                "widgets": [
                    "tracking_filter_cutoff_label",
                    "tracking_filter_cutoff_selector",
                    "tracking_filter_order_label",
                    "tracking_filter_order_selector",
                ],
            },
            {
                "name": "vold_kalman_filter",
                "actions": [
                    {"action": "set_current_index", "widget": "filter_type_selector", "value": 1},
                ],
                "widgets": [
                    "vk_filter_order_label",
                    "vk_filter_order_selector",
                    "vk_filter_bandwidth_label",
                    "vk_filter_bandwidth_selector",
                    "vk_filter_block_size_label",
                    "vk_filter_block_size_selector",
                    "vk_filter_block_overlap_label",
                    "vk_filter_block_overlap_selector",
                ],
            },
            {
                "name": "filter_explorer_dialog",
                "actions": [
                    {
                        "action": "call",
                        "callable": lambda scenario: (
                            scenario.environment_ui.explore_filter_settings(blocking=False)
                        ),
                    },
                ],
                "widgets": [
                    "channel_selector",
                    "filter_type_selector",
                    "compute_button",
                    "full_time_history_plot",
                    "order_time_history_plot",
                    "order_amplitude_plot",
                    "order_phase_plot",
                    "accept_button",
                    "reject_button",
                ],
                "root_getter": lambda scenario: scenario.environment_ui.filter_explorer_dialog,
                "teardown": [
                    {"action": "close_root"},
                ],
            },
            {
                "name": "sine_table_breakpoint_tab",
                "actions": [
                    {
                        "action": "call",
                        "callable": lambda scenario: (
                            scenario.environment_ui.definition_widget.sine_table_tab_widget.setCurrentIndex(
                                0
                            )
                        ),
                    },
                    {
                        "action": "call",
                        "callable": lambda scenario: scenario.environment_ui.sine_tables[
                            0
                        ].widget.tabWidget.setCurrentIndex(0),
                    },
                ],
                "widgets": [
                    "breakpoint_table",
                    "name_editor",
                    "start_time_selector",
                    "add_breakpoint_button",
                    "remove_breakpoint_button",
                    "load_breakpoints_button",
                    "remove_tone_button",
                ],
                "root_getter": lambda scenario: scenario.environment_ui.sine_tables[0].widget,
            },
            {
                "name": "sine_table_warning_tab",
                "actions": [
                    {
                        "action": "call",
                        "callable": lambda scenario: (
                            scenario.environment_ui.definition_widget.sine_table_tab_widget.setCurrentIndex(
                                0
                            )
                        ),
                    },
                    {
                        "action": "call",
                        "callable": lambda scenario: scenario.environment_ui.sine_tables[
                            0
                        ].widget.tabWidget.setCurrentIndex(1),
                    },
                ],
                "widgets": [
                    "warning_table",
                ],
                "root_getter": lambda scenario: scenario.environment_ui.sine_tables[0].widget,
            },
            {
                "name": "sine_table_abort_tab",
                "actions": [
                    {
                        "action": "call",
                        "callable": lambda scenario: (
                            scenario.environment_ui.definition_widget.sine_table_tab_widget.setCurrentIndex(
                                0
                            )
                        ),
                    },
                    {
                        "action": "call",
                        "callable": lambda scenario: scenario.environment_ui.sine_tables[
                            0
                        ].widget.tabWidget.setCurrentIndex(2),
                    },
                ],
                "widgets": [
                    "abort_table",
                ],
                "root_getter": lambda scenario: scenario.environment_ui.sine_tables[0].widget,
            },
        ]
    },
    "srs_sds_definition.ui": {
        "states": [
            {
                "name": "from_spec_tones",
                "actions": [
                    {"action": "set_checked", "widget": "from_spec_button", "value": True},
                ],
                "widgets": [
                    "tone_table",
                ],
            },
            {
                "name": "octave_tones",
                "actions": [
                    {"action": "set_checked", "widget": "octave_button", "value": True},
                ],
                "widgets": [
                    "min_frequency_label",
                    "min_frequency_selector",
                    "max_frequency_label",
                    "max_frequency_selector",
                    "tones_per_octave_label",
                    "tones_per_octave_selector",
                ],
            },
            {
                "name": "manual_tones",
                "actions": [
                    {"action": "set_checked", "widget": "manual_button", "value": True},
                ],
                "widgets": [
                    "tone_table",
                    "add_tone_button",
                    "remove_tone_button",
                ],
            },
            {
                "name": "common_decay",
                "actions": [
                    {"action": "set_checked", "widget": "common_decay_checkbox", "value": True},
                ],
                "widgets": [
                    "decay_value_selector",
                ],
            },
            {
                "name": "per_tone_decay",
                "actions": [
                    {"action": "set_checked", "widget": "common_decay_checkbox", "value": False},
                ],
                "widgets": [
                    "tone_table",
                ],
            },
            {
                "name": "spec_breakpoint_tab",
                "actions": [
                    {
                        "action": "set_current_index",
                        "widget": "specification_tabwidget",
                        "value": 0,
                    },
                ],
                "widgets": [
                    "breakpoint_table",
                    "num_hits_spinbox",
                    "add_breakpoint_button",
                    "remove_breakpoint_button",
                    "load_breakpoints_button",
                ],
            },
            {
                "name": "spec_lower_limit_tab",
                "actions": [
                    {
                        "action": "set_current_index",
                        "widget": "specification_tabwidget",
                        "value": 1,
                    },
                ],
                "widgets": [
                    "lower_limit_table",
                ],
            },
            {
                "name": "spec_upper_limit_tab",
                "actions": [
                    {
                        "action": "set_current_index",
                        "widget": "specification_tabwidget",
                        "value": 2,
                    },
                ],
                "widgets": [
                    "upper_limit_table",
                ],
            },
        ]
    },
    "srs_sds_run.ui": {
        "states": [
            {
                "name": "run_table_dialog",
                "actions": [
                    {
                        "action": "call",
                        "callable": lambda scenario: scenario.environment_ui.show_run_table(),
                    },
                ],
                "widgets": [
                    "save_table_button",
                    "load_table_button",
                    "allow_manual_updates_checkbox",
                    "allow_automatic_updates_checkbox",
                    "sds_table",
                    "excitation_voltage_list",
                    "response_error_list",
                    "excitation_display_plot",
                    "response_display_plot",
                    "response_srs_plot",
                    "excitation_selector",
                    "response_selector",
                    "maximum_error_button",
                    "minimum_error_button",
                    "maximum_voltage_button",
                    "minimum_voltage_button",
                ],
                "root_getter": lambda scenario: scenario.environment_ui.run_table_dialog,
                "teardown": [
                    {"action": "close_root"},
                ],
            },
            {
                "name": "shock_history_dialog",
                "actions": [
                    {
                        "action": "call",
                        "callable": lambda scenario: scenario.environment_ui.show_shock_history(),
                    },
                ],
                "widgets": [
                    "summary_groupbox",
                    "hits_by_level_plot",
                    "shock_timeline_plot",
                    "show_history_table_checkbox",
                    "close_button",
                    "total_hits_display",
                    "target_hits_display",
                    "distinct_levels_display",
                    "hits_at_target_display",
                ],
                "root_getter": lambda scenario: scenario.environment_ui.shock_history_dialog,
                "teardown": [
                    {"action": "close_root"},
                ],
            },
            {
                "name": "shock_history_dialog_with_table",
                "actions": [
                    {
                        "action": "call",
                        "callable": lambda scenario: scenario.environment_ui.show_shock_history(),
                    },
                    {
                        "action": "set_checked",
                        "widget": "show_history_table_checkbox",
                        "value": True,
                    },
                ],
                "widgets": [
                    "history_table_groupbox",
                    "history_table",
                ],
                "root_getter": lambda scenario: scenario.environment_ui.shock_history_dialog,
                "teardown": [
                    {"action": "close_root"},
                ],
            },
        ]
    },
    "system_identification.ui": {
        "states": [
            {
                "name": "impulse_response",
                "actions": [
                    {"action": "set_checked", "widget": "impulse_checkbox", "value": True},
                    {"action": "set_checked", "widget": "coherence_checkbox", "value": False},
                    {"action": "set_checked", "widget": "levels_checkbox", "value": False},
                    {"action": "set_checked", "widget": "kurtosis_checkbox", "value": False},
                ],
                "widgets": [
                    "impulse_groupbox",
                    "impulse_graphicslayout",
                ],
            },
            {
                "name": "coherence_conditioning",
                "actions": [
                    {"action": "set_checked", "widget": "impulse_checkbox", "value": False},
                    {"action": "set_checked", "widget": "coherence_checkbox", "value": True},
                    {"action": "set_checked", "widget": "levels_checkbox", "value": False},
                    {"action": "set_checked", "widget": "kurtosis_checkbox", "value": False},
                ],
                "widgets": [
                    "coherence_groupbox",
                    "coherence_graphicslayout",
                ],
            },
            {
                "name": "levels",
                "actions": [
                    {"action": "set_checked", "widget": "impulse_checkbox", "value": False},
                    {"action": "set_checked", "widget": "coherence_checkbox", "value": False},
                    {"action": "set_checked", "widget": "levels_checkbox", "value": True},
                    {"action": "set_checked", "widget": "kurtosis_checkbox", "value": False},
                ],
                "widgets": [
                    "levels_groupbox",
                    "levels_graphicslayout",
                ],
            },
            {
                "name": "kurtosis",
                "actions": [
                    {"action": "set_checked", "widget": "impulse_checkbox", "value": False},
                    {"action": "set_checked", "widget": "coherence_checkbox", "value": False},
                    {"action": "set_checked", "widget": "levels_checkbox", "value": False},
                    {"action": "set_checked", "widget": "kurtosis_checkbox", "value": True},
                ],
                "widgets": [
                    "kurtosis_groupbox",
                    "kurtosis_graphicslayout",
                ],
            },
        ]
    },
    "modal_definition.ui": {
        "states": [
            {
                "name": "signal_generator_none",
                "actions": [
                    {
                        "action": "set_current_index",
                        "widget": "signal_generator_selector",
                        "value": 0,
                    },
                ],
                "widgets": [
                    "no_excitation_label",
                ],
            },
            {
                "name": "signal_generator_random",
                "actions": [
                    {
                        "action": "set_current_index",
                        "widget": "signal_generator_selector",
                        "value": 1,
                    },
                ],
                "widgets": [
                    "random_rms_label",
                    "random_rms_selector",
                    "random_rms_selector_label",
                    "random_frequency_range_label",
                    "random_min_frequency_selector",
                    "random_frequency_range_selector_label",
                    "random_max_frequency_selector",
                ],
            },
            {
                "name": "signal_generator_burst",
                "actions": [
                    {
                        "action": "set_current_index",
                        "widget": "signal_generator_selector",
                        "value": 2,
                    },
                ],
                "widgets": [
                    "burst_rms_label",
                    "burst_rms_selector",
                    "burst_rms_selector_label",
                    "burst_frequency_range_label",
                    "burst_min_frequency_selector",
                    "burst_frequency_range_selector_label",
                    "burst_max_frequency_selector",
                    "burst_on_percentage_label",
                    "burst_on_percentage_selector",
                ],
            },
            {
                "name": "signal_generator_pseudorandom",
                "actions": [
                    {
                        "action": "set_current_index",
                        "widget": "signal_generator_selector",
                        "value": 3,
                    },
                ],
                "widgets": [
                    "pseudorandom_rms_label",
                    "pseudorandom_rms_selector",
                    "pseudorandom_rms_selector_label",
                    "pseudorandom_frequency_range_label",
                    "pseudorandom_min_frequency_selector",
                    "pseudorandom_frequency_range_selector_label",
                    "pseudorandom_max_frequency_selector",
                ],
            },
            {
                "name": "signal_generator_chirp",
                "actions": [
                    {
                        "action": "set_current_index",
                        "widget": "signal_generator_selector",
                        "value": 4,
                    },
                ],
                "widgets": [
                    "chirp_level_label",
                    "chirp_level_selector",
                    "chirp_level_selector_label",
                    "chirp_frequency_label",
                    "chirp_min_frequency_selector",
                    "chirp_frequency_range_selector_label",
                    "chirp_max_frequency_selector",
                ],
            },
            {
                "name": "signal_generator_square",
                "actions": [
                    {
                        "action": "set_current_index",
                        "widget": "signal_generator_selector",
                        "value": 5,
                    },
                ],
                "widgets": [
                    "square_level_label",
                    "square_level_selector",
                    "square_level_selector_label",
                    "square_frequency_label",
                    "square_frequency_selector",
                    "square_frequency_selector_label",
                    "square_percent_on_label",
                    "square_percent_on_selector",
                ],
            },
            {
                "name": "signal_generator_sine",
                "actions": [
                    {
                        "action": "set_current_index",
                        "widget": "signal_generator_selector",
                        "value": 6,
                    },
                ],
                "widgets": [
                    "sine_level_label",
                    "sine_level_selector",
                    "sine_level_selector_label",
                    "sine_frequency_label",
                    "sine_frequency_selector",
                    "sine_frequency_selector_label",
                ],
            },
        ]
    },
}


def get_named_widget(root_widget, widget_name):
    """
    Find a widget or QObject by objectName starting from the provided root.
    """
    if root_widget is None:
        return None

    if hasattr(root_widget, widget_name):
        return getattr(root_widget, widget_name)

    found = root_widget.findChild(QtWidgets.QWidget, widget_name)
    if found is not None:
        return found

    found = root_widget.findChild(QtCore.QObject, widget_name)
    return found


def force_render(widget, delay_ms=500):
    """
    Force a Qt widget to become visible and painted before grabbing screenshots.
    """
    widget.show()
    try:
        widget.raise_()
        widget.activateWindow()
    except Exception:
        pass

    QtWidgets.QApplication.processEvents()

    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(delay_ms, loop.quit)
    loop.exec_()

    QtWidgets.QApplication.processEvents()
    widget.repaint()
    QtWidgets.QApplication.processEvents()

def apply_state_actions(root_widget, scenario_result, actions):
    """
    Apply a sequence of UI actions to enter a valid documentation state.

    Returns
    -------
    action_root : QWidget | None
        If one of the actions creates/returns a dialog or alternate root widget,
        that object is returned and becomes the active root for subsequent actions.
    """
    action_root = root_widget

    for action in actions:
        action_type = action["action"]

        if action_type == "call":
            result = action["callable"](scenario_result)
            if isinstance(result, QtWidgets.QWidget):
                action_root = result
            QtWidgets.QApplication.processEvents()
            continue

        if action_type == "close_root":
            if action_root is not None:
                action_root.close()
                QtWidgets.QApplication.processEvents()
            continue

        widget_name = action.get("widget")
        widget = get_named_widget(action_root, widget_name)

        if widget is None:
            raise RuntimeError(
                f"Could not find widget '{widget_name}' while applying state action."
            )

        if action_type == "set_checked":
            widget.setChecked(bool(action["value"]))
        elif action_type == "set_current_index":
            widget.setCurrentIndex(int(action["value"]))
        elif action_type == "set_value":
            widget.setValue(action["value"])
        elif action_type == "set_text":
            widget.setText(str(action["value"]))
        elif action_type == "click":
            widget.click()
        else:
            raise RuntimeError(f"Unknown action type '{action_type}'")

        QtWidgets.QApplication.processEvents()

    QtWidgets.QApplication.processEvents()
    return action_root


def collect_reduced_widget_names(reduced_structure):
    names = set()
    for _, struct in reduced_structure.items():
        names.add(struct["name"])
        children = struct.get("children", {})
        if children:
            names.update(collect_reduced_widget_names(children))
    return names


def collect_state_widget_names(page_state_spec):
    names = set()
    for state in page_state_spec.get("states", []):
        for widget_name in state.get("widgets", []):
            names.add(widget_name)
    return names


def filter_reduced_structure_by_widget_names(reduced_structure, allowed_names):
    filtered = {}

    for key, struct in reduced_structure.items():
        include_self = struct["name"] in allowed_names
        children = struct.get("children", {})
        filtered_children = (
            filter_reduced_structure_by_widget_names(children, allowed_names) if children else {}
        )

        if include_self or filtered_children:
            new_struct = struct.copy()
            if "children" in new_struct:
                new_struct["children"] = filtered_children
            filtered[key] = new_struct

    return filtered


def compute_default_reduced_structure(ui_file, reduced_structure):
    basename = os.path.basename(ui_file)
    page_state_spec = PAGE_STATES.get(basename)

    if page_state_spec is None:
        return reduced_structure

    all_widget_names = collect_reduced_widget_names(reduced_structure)
    state_widget_names = collect_state_widget_names(page_state_spec)
    default_widget_names = all_widget_names - state_widget_names

    return filter_reduced_structure_by_widget_names(
        reduced_structure,
        default_widget_names,
    )


def generate_state_markdown(ui_analyzer, scenario_result, ui_file, reduced_structure):
    basename = os.path.basename(ui_file)
    page_state_spec = PAGE_STATES.get(basename)
    if page_state_spec is None:
        return ""

    state_markdown_blocks = []

    for state in page_state_spec.get("states", []):
        print(f"  Rendering state: {state['name']}")

        action_root = getattr(ui_analyzer, "central_widget", None)

        action_root = apply_state_actions(
            action_root,
            scenario_result,
            state.get("actions", []),
        )
        QtWidgets.QApplication.processEvents()

        capture_root = action_root
        if "root_getter" in state:
            capture_root = state["root_getter"](scenario_result)

        if capture_root is None:
            raise RuntimeError(f"State '{state['name']}' did not produce a valid capture root.")

        force_render(capture_root, delay_ms=400)

        # IMPORTANT:
        # Build reduced structure from the correct root widget for this state.
        if capture_root is getattr(ui_analyzer, "central_widget", None):
            working_reduced_structure = reduced_structure
        else:
            dialog_analyzer = UIAnalyzer(ui_file, live_widget=capture_root)
            working_reduced_structure = dialog_analyzer.reduced_structure()

        allowed_names = set(state.get("widgets", []))
        state_reduced = filter_reduced_structure_by_widget_names(
            working_reduced_structure,
            allowed_names,
        )

        state_name_prefix = f"{ui_analyzer.name}__{state['name']}"

        state_text, state_figures = ui_analyzer._generate_item_markdown(
            state_reduced,
            capture_root=capture_root,
            name_prefix=state_name_prefix,
        )

        block = ""
        if state_text.strip():
            block += "\n\n" + state_text
        if state_figures.strip():
            block += "\n\n" + state_figures

        state_markdown_blocks.append(block)

        if "teardown" in state:
            apply_state_actions(capture_root, scenario_result, state["teardown"])
            QtWidgets.QApplication.processEvents()

    return "".join(state_markdown_blocks)


class UIAnalyzer(QtWidgets.QMainWindow):
    """A Class to analyze the contents of a .ui file and create a markdown file documenting it"""

    def __init__(self, ui_file, live_widget=None):
        super().__init__()
        self.name = os.path.splitext(os.path.split(ui_file)[-1])[0]
        self.live_widget = live_widget
        self.print_depth = 0
        self.all_widgets = None
        self.all_layouts = None

        if self.live_widget is None:
            self.load_ui(ui_file)
            self.resize(1800, 1000)
        else:
            # Use the provided runtime widget for screenshots, but still keep the ui file path
            self.central_widget = live_widget
            self.base_class = type(live_widget)

    def load_ui(self, ui_file):
        """Loads in a ui file and shows it in a main window

        Parameters
        ----------
        ui_file : str
            The path to the ui file to load
        """
        # Load the UI file
        self.form_class, self.base_class = uic.loadUiType(ui_file)

        # Check if the loaded UI is a QMainWindow
        if issubclass(self.base_class, QtWidgets.QMainWindow):
            # If it's a QMainWindow, load it directly into self
            self.ui = self.form_class()
            self.ui.setupUi(self)
        else:
            # If it's a QWidget, create a central widget and load the UI into it
            self.central_widget = QtWidgets.QWidget(self)
            self.setCentralWidget(self.central_widget)
            self.ui = self.form_class()
            self.ui.setupUi(self.central_widget)

        self.show()

    def export_structure(self):
        root = self.central_widget if hasattr(self, "central_widget") else self
        self.all_widgets = root.findChildren(QtWidgets.QWidget)
        self.all_layouts = root.findChildren(QtWidgets.QLayout)
        self.all_widgets.append(root)
        return self._get_widget_structure(root)

    def _get_widget_structure(self, item):
        try:
            name = item.objectName()
        except AttributeError:
            name = type(item).__name__
        print(f"Analyzing {name}")
        try:
            item_rect = item.rect()
            position = item.mapToGlobal(item_rect.topLeft())
            height = item_rect.height()
            width = item_rect.width()
            box = [position.x(), position.y(), width, height]
            position = [position.x(), position.y()]
        except AttributeError:
            position = None
            box = None
        try:
            tool_tip = item.toolTip()
        except AttributeError:
            tool_tip = None
        structure = {
            "name": name,
            "type": type(item),
            "tooltip": tool_tip,
            "pos": position,
            "box": box,
            "children": [],
            "widget": item,
        }
        # Remove ourselves from the list since we've been accounted for
        if isinstance(item, QtWidgets.QWidget):
            self.all_widgets.remove(item)
        if isinstance(item, QtWidgets.QLayout):
            self.all_layouts.remove(item)
        print(f"Removed {name} from global lists")

        # If we are a layout, go through all of the items in the layout
        if isinstance(item, QtWidgets.QLayout):
            print("Stepping through Layout")
            for index in range(item.count()):
                print(f"Item {index}")
                # Get the item
                childitem = item.itemAt(index)
                widget = childitem.widget()
                if widget is None:
                    # If the item is not a widget, it will be a layout
                    child = childitem
                else:
                    # Otherwise, it will be an item that we need to get the
                    # widget from
                    child = widget
                structure["children"].append(self._get_widget_structure(child))

        try:
            # Get the remaining children layouts and widgets
            child_layouts = [
                it
                for it in item.children()
                if isinstance(it, QtWidgets.QLayout)
                if it in self.all_layouts
            ]
            print(f"Remaining Child Layouts {child_layouts}")
            for child in child_layouts:
                structure["children"].append(self._get_widget_structure(child))
            child_widgets = [
                it
                for it in item.children()
                if not isinstance(it, QtWidgets.QLayout)
                if it in self.all_widgets
            ]
            print(f"Remaining Child Widgets {child_widgets}")
            for child in child_widgets:
                structure["children"].append(self._get_widget_structure(child))
        except AttributeError:
            child_layouts = []
            child_widgets = []

        # Sort the children based on left-to-right, top-to-bottom
        structure["children"].sort(key=lambda child: (child["pos"][1], child["pos"][0]))

        # Go through the children and set the position of this widget based on
        # the values of the children
        if position is None and len(structure["children"]) > 0:
            xs = []
            ys = []
            for child in structure["children"]:
                pos = child["pos"]
                if pos is not None:
                    x, y = child["pos"]
                    xs.append(x)
                    ys.append(y)
            structure["pos"] = [min(xs), min(ys)]
        elif position is None and len(structure["children"]) == 0:
            structure["pos"] = [100000000, 1000000000]

        return structure

    def reduced_structure(self, full_structure=None, structure_dictionary=None):
        if full_structure is None:
            full_structure = self.export_structure()
        if structure_dictionary is None:
            structure_dictionary = {}

        if isinstance(full_structure["widget"], QtWidgets.QGroupBox):
            structure_dictionary[full_structure["widget"].title()] = full_structure.copy()
            del structure_dictionary[full_structure["widget"].title()]["children"]
            structure_dictionary[full_structure["widget"].title()]["children"] = {}
            children_dictionary = structure_dictionary[full_structure["widget"].title()]["children"]
        else:
            children_dictionary = structure_dictionary

        if full_structure["tooltip"] is not None and full_structure["tooltip"].strip() != "":
            label, message = self.parse_tooltip(full_structure["tooltip"])
            structure_dictionary[label] = full_structure.copy()
            del structure_dictionary[label]["children"]

        # Now go analyze the children
        for child in full_structure["children"]:
            self.reduced_structure(child, children_dictionary)

        return structure_dictionary

    def print_structure(self):
        """Prints a representation of the ui hierarchy"""
        struct = self.export_structure()
        self._print_struct_item(struct)

    def _print_struct_item(self, struct):
        name = struct["name"]
        position = struct["pos"]
        # tooltip = struct['tooltip']
        typ = struct["type"]
        children = struct["children"]
        print(" " * self.print_depth * 4 + f" {name} ({typ}) at {position}")
        self.print_depth += 1
        for child in children:
            self._print_struct_item(child)
        self.print_depth -= 1

    def print_reduced_structure(self):
        reduced_structure = self.reduced_structure()
        for name, data in reduced_structure.items():
            self._print_reduced_struct_item(name, data)

    def _print_reduced_struct_item(self, key, data):
        name = data["name"]
        position = data["pos"]
        # tooltip = struct['tooltip']
        typ = data["type"]
        try:
            children = data["children"]
        except KeyError:
            children = {}
        print(" " * self.print_depth * 4 + f"{key} {name} ({typ}) at {position}")
        self.print_depth += 1
        for child_key, child_data in children.items():
            self._print_reduced_struct_item(child_key, child_data)
        self.print_depth -= 1

    def parse_tooltip(self, tooltip):
        """Parses a tooltip to extract the widget's label and documentation

        Parameters
        ----------
        tooltip : str
            HTML-based tooltip from the UI.  It will be separated by paragraph tags (</p>).  The
            first line is used as the label.  The remaining lines are used as documentation.  All
            other HTML tags are discarded.

        Returns
        -------
        label : str
            The name of the widget in the documentation
        documentation : str
            The documentation to go along with the widget
        """
        # Use regex to remove HTML tags
        tooltip_data = []
        lines = tooltip.split("</p>")
        for line in lines:
            text = re.sub(r"<[^>]+>", "", line)  # Remove all HTML tags
            text = text.strip()  # Remove leading and trailing whitespace
            if len(text) > 0:
                tooltip_data.append(text)
        tooltip_data = [
            tooltip_data[0].replace(r"&lt;", r"<").replace(r"&gt;", r">"),
            "  ".join(tooltip_data[1:])
            .replace(r"&lt;", r"<")
            .replace(r"&gt;", r">")
            .replace("&quot;", '"'),
        ]
        return tooltip_data

    def generate_markdown(self, scenario_result=None, ui_file=None):
        reduced = self.reduced_structure()

        if ui_file is None:
            ui_file = self.name + ".ui"

        default_reduced = compute_default_reduced_structure(ui_file, reduced)
        default_capture_root = getattr(self, "central_widget", None)
        markdown_text, markdown_figures = self._generate_item_markdown(
            default_reduced,
            capture_root=default_capture_root,
        )

        state_markdown = ""
        if scenario_result is not None and ui_file is not None:
            state_markdown = generate_state_markdown(
                self,
                scenario_result,
                ui_file,
                reduced,
            )

        return (
            f"---\nnumbering:\n  figure: false\n---\n# {self.name}\n\n"
            + markdown_text
            + ("\n\n" + markdown_figures if markdown_figures.strip() else "")
            + ("\n\n" + state_markdown if state_markdown.strip() else "")
        )

    def _generate_item_markdown(self, reduced_structure, capture_root=None, name_prefix=None):
        if name_prefix is None:
            name_prefix = self.name

        this_text_markdown = ""
        this_figure_markdown = ""

        for name, struct in reduced_structure.items():
            if isinstance(struct["widget"], QtWidgets.QGroupBox):
                figure_file_name = name_prefix + "__" + struct["name"] + ".png"
                figure_full_path = os.path.join(figures_dir, figure_file_name)
                figure_rel_path = os.path.join("figures", figure_file_name).replace("\\", "/")
                figure_ref_name = "fig:" + name_prefix + ":" + struct["name"]

                print(
                    "Generating figure:",
                    name,
                    "objectName=",
                    struct["widget"].objectName()
                    if hasattr(struct["widget"], "objectName")
                    else None,
                    "type=",
                    type(struct["widget"]),
                )

                px = self.generate_documentation_figure(
                    struct["widget"],
                    capture_root=capture_root,
                )
                saved = px.save(figure_full_path)
                print(f"Save returned {saved} for {figure_full_path}")

                block_label = "sec:" + name_prefix + ":" + struct["name"]
                this_text_markdown = this_text_markdown + f"\n\n({block_label})="
                this_figure_markdown += (
                    f"\n\n:::{{figure}} {figure_rel_path}\n"
                    f":label: {figure_ref_name}\n"
                    f" {name} Settings\n:::"
                )

            if struct["tooltip"] is not None and struct["tooltip"].strip() != "":
                figure_file_name = name_prefix + "__" + struct["name"] + ".png"
                figure_full_path = os.path.join(figures_dir, figure_file_name)
                figure_rel_path = os.path.join("figures", figure_file_name).replace("\\", "/")
                figure_ref_name = "fig:" + name_prefix + ":" + struct["name"]

                label, message = self.parse_tooltip(struct["tooltip"])

                print(
                    "Generating figure:",
                    name,
                    "objectName=",
                    struct["widget"].objectName()
                    if hasattr(struct["widget"], "objectName")
                    else None,
                    "type=",
                    type(struct["widget"]),
                )

                px = self.generate_documentation_figure(
                    struct["widget"],
                    capture_root=capture_root,
                )
                print(
                    f"Pixmap info for {struct['name']}: "
                    f"isNull={px.isNull()}, size=({px.width()} x {px.height()})"
                )
                saved = px.save(figure_full_path)
                print(f"Save returned {saved} for {figure_full_path}")

                this_figure_markdown += (
                    f"\n\n:::{{figure}} {figure_rel_path}\n"
                    f":label: {figure_ref_name}\n"
                    f" **{label}** {message}\n:::"
                )
                this_text_markdown = (
                    this_text_markdown + f"\n* [**{label}**](#{figure_ref_name}) {message}"
                )

            if "children" in struct:
                child_text, child_figure = self._generate_item_markdown(
                    struct["children"],
                    capture_root=capture_root,
                    name_prefix=name_prefix,
                )
                this_text_markdown = this_text_markdown + child_text
                this_figure_markdown = this_figure_markdown + child_figure

            if isinstance(struct["widget"], QtWidgets.QGroupBox):
                this_text_markdown = this_text_markdown + "\n"

        return this_text_markdown, this_figure_markdown

    def generate_documentation_figure(
        self,
        widget,
        capture_root=None,
        padding=120,
        box_thickness=2,
        box_padding=5,
    ):
        QtWidgets.QApplication.processEvents()

        if capture_root is None:
            capture_root = getattr(self, "central_widget", None)
        if capture_root is None:
            capture_root = widget

        # If widget is not a descendant of capture_root, fall back to widget grab.
        ancestor = widget
        is_descendant = False
        while ancestor is not None:
            if ancestor is capture_root:
                is_descendant = True
                break
            ancestor = ancestor.parentWidget()

        if not is_descendant:
            capture_root = widget

        force_render(capture_root, delay_ms=500)
        if widget is not capture_root:
            force_render(widget, delay_ms=200)
        root_pixmap = capture_root.grab()

        # Map widget-local rect into capture-root coordinates
        top_left_in_root = widget.mapTo(capture_root, QPoint(0, 0))
        widget_rect = QRect(
            top_left_in_root.x(),
            top_left_in_root.y(),
            widget.width(),
            widget.height(),
        )

        # Expand box slightly so border doesn't intersect the widget edge
        box_rect = QRect(
            widget_rect.left() - box_padding,
            widget_rect.top() - box_padding,
            widget_rect.width() + 2 * box_padding,
            widget_rect.height() + 2 * box_padding,
        )

        # Draw the red rectangle on the captured root pixmap
        painter = QPainter(root_pixmap)
        pen = QPen(QColor("red"))
        pen.setWidth(box_thickness)
        painter.setPen(pen)
        painter.drawRect(box_rect)
        painter.end()

        # Expand crop region for surrounding context
        expanded_rect = QRect(
            widget_rect.left() - padding,
            widget_rect.top() - padding,
            widget_rect.width() + 2 * padding,
            widget_rect.height() + 2 * padding,
        )

        expanded_rect = expanded_rect.intersected(root_pixmap.rect())

        cropped_pixmap = root_pixmap.copy(expanded_rect)
        return cropped_pixmap

if __name__ == "__main__":
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    # Group files by environment scenario key
    grouped_files = {}
    static_files = []

    for file in files:
        basename = os.path.basename(file)
        if basename in UI_FILE_TO_SCENARIO:
            environment_key, widget_key = UI_FILE_TO_SCENARIO[basename]
            grouped_files.setdefault(environment_key, []).append((file, widget_key))
        else:
            static_files.append(file)

    try:
        # First process scenario-backed UI files one environment at a time
        for environment_key, file_entries in grouped_files.items():
            print(f"\n=== Building scenario for environment: {environment_key} ===")
            scenario_builder = ENVIRONMENT_SCENARIOS[environment_key]
            scenario_result = scenario_builder(display_errors=False)

            try:
                for file, widget_key in file_entries:
                    print(f"Analyzing {file} using scenario {environment_key}:{widget_key}")
                    basename = os.path.basename(file)
                    live_widget = scenario_result.widgets[widget_key]

                    # Switch to the correct top-level tab for the screenshot
                    if widget_key == "definition":
                        scenario_result.main_ui.rattlesnake_tabs.setCurrentIndex(1)
                        env_tabs = scenario_result.main_ui.environment_definition_environment_tabs
                    elif widget_key == "system_id":
                        scenario_result.main_ui.rattlesnake_tabs.setCurrentIndex(2)
                        env_tabs = scenario_result.main_ui.system_id_environment_tabs
                    elif widget_key == "prediction":
                        scenario_result.main_ui.rattlesnake_tabs.setCurrentIndex(3)
                        env_tabs = scenario_result.main_ui.test_prediction_environment_tabs
                    elif widget_key == "run":
                        scenario_result.main_ui.rattlesnake_tabs.setCurrentIndex(5)
                        env_tabs = scenario_result.main_ui.run_environment_tabs
                    else:
                        env_tabs = None

                    # Select correct environment tab inside that page
                    if env_tabs is not None:
                        for i in range(env_tabs.count()):
                            if (
                                env_tabs.tabText(i)
                                == scenario_result.environment_ui.environment_name
                            ):
                                env_tabs.setCurrentIndex(i)
                                break

                    QtWidgets.QApplication.processEvents()

                    if live_widget is not None:
                        force_render(scenario_result.main_ui, delay_ms=400)
                        force_render(live_widget, delay_ms=400)

                    ui = UIAnalyzer(file, live_widget=live_widget)
                    markdown_text = ui.generate_markdown(
                        scenario_result=scenario_result,
                        ui_file=file,
                    )

                    filename = os.path.splitext(os.path.split(file)[1])[0]
                    output_md = os.path.join(
                        dir_path, "book", "src", "_generated", f"{filename}_doc.md"
                    )

                    with open(output_md, "w", encoding="utf-8") as f:
                        f.write(markdown_text)

                    print(f"  Wrote markdown to {output_md}")

            finally:
                print(f"Cleaning up scenario {environment_key}")
                scenario_result.cleanup()
                QtWidgets.QApplication.processEvents()

        # Then process any plain static UI files that do not use scenarios
        for file in static_files:
            print(f"\n=== Analyzing static UI file: {file} ===")
            ui = UIAnalyzer(file)
            markdown_text = ui.generate_markdown(
                scenario_result=None,
                ui_file=file,
            )

            filename = os.path.splitext(os.path.split(file)[1])[0]
            output_md = os.path.join(dir_path, "book", "src", "_generated", f"{filename}_doc.md")

            with open(output_md, "w", encoding="utf-8") as f:
                f.write(markdown_text)

            print(f"  Wrote markdown to {output_md}")

    finally:
        QtWidgets.QApplication.processEvents()