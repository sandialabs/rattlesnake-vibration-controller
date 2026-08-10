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
from qtpy import QtWidgets, uic
from qtpy.QtCore import QRect, QPoint
from qtpy.QtGui import QPixmap, QPainter, QPen, QColor
from ui_documentation_scenarios import UI_DOC_SCENARIOS

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
        "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/random_vibration_definition.ui",
        "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/random_vibration_prediction.ui",
        "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/random_vibration_run.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/modal_definition.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/modal_run.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/modal_acquisition_window.ui",
        "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/transient_definition.ui",
        "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/transient_prediction.ui",
        "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/transient_run.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/transformation_matrices.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/time_run.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/time_definition.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/system_identification.ui",
        "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/sine_definition.ui",
        "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/sine_prediction.ui",
        "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/sine_run.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/sine_sweep_table.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/sine_filter_explorer.ui",
        "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/srs_sds_definition.ui",
        "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/srs_sds_prediction.ui",
        "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/srs_sds_run.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/srs_sds_prediction_table.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/srs_sds_run_table.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/srs_sds_shock_history.ui",
        # "../src/rattlesnake/src/rattlesnake/user_interface/ui_files/srs_sds_synthesize_dialog.ui",
    ]
]

# "../src/rattlesnake/components/ip_manager.ui",
# "../src/rattlesnake/components/environment_selector.ui",
# "../src/rattlesnake/components/combined_environments_controller.ui",
# "../src/rattlesnake/components/control_select.ui",

UI_FILE_TO_SCENARIO = {
    "random_vibration_definition.ui": ("random_definition", "definition"),
    "random_vibration_prediction.ui": ("random_prediction", "prediction"),
    "random_vibration_run.ui": ("random_run", "run"),
    "transient_definition.ui": ("transient_definition", "definition"),
    "transient_prediction.ui": ("transient_prediction", "prediction"),
    "transient_run.ui": ("transient_run", "run"),
    "sine_definition.ui": ("sine_definition", "definition"),
    "sine_prediction.ui": ("sine_prediction", "prediction"),
    "sine_run.ui": ("sine_run", "run"),
    "srs_sds_definition.ui": ("sds_definition", "definition"),
    "srs_sds_prediction.ui": ("sds_prediction", "prediction"),
    "srs_sds_run.ui": ("sds_run", "run"),
}


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

    def generate_markdown(self):
        """Generates a string of markdown text describing the user interface

        Returns
        -------
        str
            A string containing markdown text that can be included into the main documentation.
        """
        struct = self.reduced_structure()
        markdown_text, markdown_figures = self._generate_item_markdown(struct)
        return (
            f"---\nnumbering:\n  figure: false\n---\n# {self.name}\n\n"
            + markdown_text
            + "\n\n"
            + markdown_figures
        )

    def _generate_item_markdown(self, reduced_structure):
        this_text_markdown = ""
        this_figure_markdown = ""

        for name, struct in reduced_structure.items():
            if isinstance(struct["widget"], QtWidgets.QGroupBox):
                figure_file_name = self.name + "__" + struct["name"] + ".png"
                figure_full_path = os.path.join(
                    "book", "src", "_generated", "figures", figure_file_name
                ).replace("\\", "/")
                figure_rel_path = os.path.join("figures", figure_file_name).replace("\\", "/")
                figure_ref_name = "fig:" + self.name + ":" + struct["name"]
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
                px = self.generate_documentation_figure(struct["widget"])
                px.save(figure_full_path)

                block_label = "sec:" + self.name + ":" + struct["name"]
                this_text_markdown = this_text_markdown + f"\n\n({block_label})="
                this_figure_markdown += f"\n\n:::{{figure}} {figure_rel_path}\n:label: {figure_ref_name}\n {name} Settings\n:::"

            if struct["tooltip"] is not None and struct["tooltip"].strip() != "":
                # This means we would like to build documentation with this widget
                figure_file_name = self.name + "__" + struct["name"] + ".png"
                figure_full_path = os.path.join(
                    "book", "src", "_generated", "figures", figure_file_name
                ).replace("\\", "/")
                figure_rel_path = os.path.join("figures", figure_file_name).replace("\\", "/")
                figure_ref_name = "fig:" + self.name + ":" + struct["name"]
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
                px = self.generate_documentation_figure(struct["widget"])
                px.save(figure_full_path)

                this_figure_markdown += f"\n\n:::{{figure}} {figure_rel_path}\n:label: {figure_ref_name}\n **{label}** {message}\n:::"
                this_text_markdown = (
                    this_text_markdown + f"\n* [**{label}**](#{figure_ref_name}) {message}"
                )

            # Go through its children
            if "children" in struct:
                child_text, child_figure = self._generate_item_markdown(struct["children"])
                this_text_markdown = this_text_markdown + child_text
                this_figure_markdown = this_figure_markdown + child_figure

            if isinstance(struct["widget"], QtWidgets.QGroupBox):
                this_text_markdown = this_text_markdown + "\n"

        return this_text_markdown, this_figure_markdown

    def generate_documentation_figure(self, widget, padding=40, box_thickness=2, box_padding=5):
        QtWidgets.QApplication.processEvents()

        # Use the analyzer's central widget as the capture root if possible,
        # otherwise fall back to the widget itself.
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

        # Grab the capture root
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
    for file in files:
        print(f"Analyzing {file}")
        basename = os.path.basename(file)

        scenario_result = None
        live_widget = None

        try:
            if basename in UI_FILE_TO_SCENARIO:
                scenario_name, widget_key = UI_FILE_TO_SCENARIO[basename]
                scenario_builder = UI_DOC_SCENARIOS[scenario_name]
                print(f"  Using scenario: {scenario_name}")
                scenario_result = scenario_builder(display_errors=False)
                live_widget = scenario_result.widgets[widget_key]
                QtWidgets.QApplication.processEvents()

            if live_widget is not None:
                live_widget.show()
                live_widget.raise_()
                QtWidgets.QApplication.processEvents()

            ui = UIAnalyzer(file, live_widget=live_widget)
            markdown_text = ui.generate_markdown()

            filename = os.path.splitext(os.path.split(file)[1])[0]
            with open(
                dir_path + "/" + f"book/src/_generated/{filename}_doc.md",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(markdown_text)

        finally:
            if scenario_result is not None:
                scenario_result.cleanup()
                QtWidgets.QApplication.processEvents()