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
from qtpy.QtCore import QRect
from qtpy.QtGui import QPixmap, QPainter, QPen, QColor
import pyqtgraph

pyqtgraph.setConfigOption("background", "w")
pyqtgraph.setConfigOption("foreground", "k")

try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    dir_path = "."

files = [
    dir_path + "/" + v
    for v in [
        "../src/rattlesnake/components/random_vibration_prediction.ui",
        "../src/rattlesnake/components/modal_run.ui",
        "../src/rattlesnake/components/modal_definition.ui",
        "../src/rattlesnake/components/modal_acquisition_window.ui",
        "../src/rattlesnake/components/transient_run.ui",
        "../src/rattlesnake/components/transient_prediction.ui",
        "../src/rattlesnake/components/transient_definition.ui",
        "../src/rattlesnake/components/transformation_matrices.ui",
        "../src/rattlesnake/components/time_run.ui",
        "../src/rattlesnake/components/time_definition.ui",
        "../src/rattlesnake/components/system_identification.ui",
        "../src/rattlesnake/components/sine_run.ui",
        "../src/rattlesnake/components/sine_prediction.ui",
        "../src/rattlesnake/components/random_vibration_definition.ui",
        "../src/rattlesnake/components/sine_sweep_table.ui",
        "../src/rattlesnake/components/sine_filter_explorer.ui",
        "../src/rattlesnake/components/sine_definition.ui",
        "../src/rattlesnake/components/random_vibration_run.ui",
    ]
]


# "../src/rattlesnake/components/ip_manager.ui",
# "../src/rattlesnake/components/environment_selector.ui",
# "../src/rattlesnake/components/combined_environments_controller.ui",
# "../src/rattlesnake/components/control_select.ui",

class UIAnalyzer(QtWidgets.QMainWindow):
    """A Class to analyze the contents of a .ui file and create a markdown file documenting it
    """

    def __init__(self, ui_file):
        """Initializes the UI Analyzer object

        Parameters
        ----------
        file : str
            Path to a .ui file to analyze and document
        """
        super().__init__()
        self.name = os.path.splitext(os.path.split(ui_file)[-1])[0]
        self.load_ui(ui_file)
        self.print_depth = 0
        self.all_widgets = None
        self.all_layouts = None
        self.resize(1800, 1000)

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
        """Creates a nested dictionary structure of the widgets and layouts in the user interface"""
        # Start the recursive structure export from the central widget or self
        if self.base_class == QtWidgets.QMainWindow:
            self.all_widgets = self.findChildren(QtWidgets.QWidget)
            self.all_layouts = self.findChildren(QtWidgets.QLayout)
            self.all_widgets.append(self)
            return self._get_widget_structure(self)
        else:
            self.all_widgets = self.central_widget.findChildren(QtWidgets.QWidget)
            self.all_layouts = self.central_widget.findChildren(QtWidgets.QLayout)
            self.all_widgets.append(self.central_widget)
            return self._get_widget_structure(self.central_widget)

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
        return f"---\nnumbering:\n  figure: false\n---\n# {self.name}\n\n"+markdown_text + "\n\n" + markdown_figures

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
                ).replace('\\','/')
                figure_rel_path = os.path.join("figures", figure_file_name).replace('\\','/')
                figure_ref_name = "fig:" + self.name + ":" + struct["name"]
                label, message = self.parse_tooltip(struct["tooltip"])
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

    def generate_documentation_figure(self, widget, padding=100, box_thickness=2, box_padding=5):
        item_rect = widget.rect()
        parent_window = widget.window()
        position = widget.mapTo(parent_window, item_rect.topLeft())
        height = item_rect.height()
        width = item_rect.width()
        widget_rect = QRect(position.x(), position.y(), width, height)
        window_pixmap = parent_window.grab()

        box_rect = QRect(
            widget_rect.left() - box_padding,
            widget_rect.top() - box_padding,
            widget_rect.width() + 2 * box_padding,
            widget_rect.height() + 2 * box_padding,
        )

        # Draw a red box around the widget's geometry within the cropped image
        painter = QPainter(window_pixmap)
        pen = QPen(QColor("red"))
        pen.setWidth(box_thickness)
        painter.setPen(pen)

        # Draw the rectangle
        painter.drawRect(box_rect)
        painter.end()

        # return window_pixmap
        # Calculate the expanded rectangle around the widget
        expanded_rect = QRect(
            widget_rect.left() - padding,
            widget_rect.top() - padding,
            widget_rect.width() + 2 * padding,
            widget_rect.height() + 2 * padding,
        )

        # Ensure the expanded rectangle stays within the bounds of the window
        expanded_rect = expanded_rect.intersected(parent_window.rect())

        # Crop the pixmap to the expanded rectangle
        cropped_pixmap = window_pixmap.copy(expanded_rect)

        return cropped_pixmap


app = QtWidgets.QApplication(sys.argv)

for file in files:
    print(f"Analyzing {file}")
    ui = UIAnalyzer(file)
    markdown_text = ui.generate_markdown()
    filename = os.path.splitext(os.path.split(file)[1])[0]
    with open(
        dir_path + "/" + f"book/src/_generated/{filename}_doc.md", "w", encoding="utf-8"
    ) as f:
        f.write(markdown_text)
