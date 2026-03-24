from rattlesnake.user_interface.abstract_sys_id_user_interface import AbstractSysIdUI
from rattlesnake.environment.sine_sys_id_environment import (
    SineCommands,
    SineUICommands,
    SineMetadata,
)
from rattlesnake.utilities import GlobalCommands, VerboseMessageQueue, load_python_module, db2scale
from rattlesnake.environment.environment_utilities import ControlTypes
from rattlesnake.user_interface.ui_utilities import (
    environment_definition_ui_paths,
    environment_prediction_ui_paths,
    environment_run_ui_paths,
)
from rattlesnake.user_interface.sine_sys_id_ui_utilities import (
    FilterExplorer,
    PlotSineWindow,
    SineSweepTable,
)
from rattlesnake.user_interface.ui_utilities import (
    TransformationMatrixWindow,
    VaryingNumberOfLinePlot,
    blended_scatter_plot,
    multiline_plotter,
)
from qtpy import QtWidgets, uic
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor  # pylint: disable=no-name-in-module
from multiprocessing.queues import Queue
import numpy as np
import inspect
import netCDF4 as nc4


CONTROL_TYPE = ControlTypes.SINE
MAXIMUM_NAME_LENGTH = 50


# region: User Interface
class SineUI(AbstractSysIdUI):
    """Class to represent the user interface of the MIMO sine module"""

    def __init__(
        self,
        environment_name: str,
        definition_tabwidget: QtWidgets.QTabWidget,
        system_id_tabwidget: QtWidgets.QTabWidget,
        test_predictions_tabwidget: QtWidgets.QTabWidget,
        run_tabwidget: QtWidgets.QTabWidget,
        environment_command_queue: VerboseMessageQueue,
        controller_communication_queue: VerboseMessageQueue,
        log_file_queue: Queue,
    ):
        """Initializes a Sine Environment User Interface

        Parameters
        ----------
        environment_name : str
            The name of the environment
        definition_tabwidget : QtWidgets.QTabWidget
            The tab widget containing the environment definitions, into which the
            definition widget will be placed.
        system_id_tabwidget : QtWidgets.QTabWidget
            The tab widget containing the system identification operations, into
            which the system id widget is placed by the abstract parent class
        test_predictions_tabwidget : QtWidgets.QTabWidget
            The tab widget containing the test predictions, into which the
            prediction widget will be placed
        run_tabwidget : QtWidgets.QTabWidget
            The tab widget containing the operations to run the environment,
            into which the run widget will be placed.
        environment_command_queue : VerboseMessageQueue
            The queue to put commands for the environment process
        controller_communication_queue : VerboseMessageQueue
            The queue to put commands for the environment
        log_file_queue : Queue
            The queue to put logging information
        """
        super().__init__(
            environment_name,
            environment_command_queue,
            controller_communication_queue,
            log_file_queue,
            system_id_tabwidget,
        )
        # Add the page to the control definition tabwidget
        self.definition_widget = QtWidgets.QWidget()
        uic.loadUi(environment_definition_ui_paths[CONTROL_TYPE], self.definition_widget)
        definition_tabwidget.addTab(self.definition_widget, self.environment_name)
        # Add the page to the control prediction tabwidget
        self.prediction_widget = QtWidgets.QWidget()
        uic.loadUi(environment_prediction_ui_paths[CONTROL_TYPE], self.prediction_widget)
        test_predictions_tabwidget.addTab(self.prediction_widget, self.environment_name)
        # Add the page to the run tabwidget
        self.run_widget = QtWidgets.QWidget()
        uic.loadUi(environment_run_ui_paths[CONTROL_TYPE], self.run_widget)
        run_tabwidget.addTab(self.run_widget, self.environment_name)

        self.run_widget.splitter.setSizes([200000, 100000])

        self.physical_channel_names = None
        self.physical_output_indices = None
        self.physical_unit_names = None
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.python_control_module = None
        self.plot_data_items = {}
        self.achieved_response_signals_combined = None
        self.achieved_response_signals = None
        self.achieved_response_amplitudes = None
        self.achieved_response_phases = None
        self.complex_drive_modifications = None
        self.achieved_excitation_signals_combined = None
        self.achieved_excitation_signals = None
        self.achieved_excitation_frequencies = None
        self.achieved_excitation_arguments = None
        self.achieved_excitation_amplitudes = None
        self.achieved_excitation_phases = None
        self.plot_downsample = None
        self.specification_signals_combined = None
        self.specification_signals = None
        self.specification_frequencies = None
        self.specification_arguments = None
        self.specification_amplitudes = None
        self.specification_phases = None
        self.sine_tables = []
        self.plot_windows = []
        self.spec_time = None
        self.shutdown_sent = False

        self.control_selector_widgets = [
            self.definition_widget.specification_row_selector,
            self.prediction_widget.response_selector,
            self.run_widget.control_channel_selector,
        ]
        self.output_selector_widgets = [
            self.prediction_widget.excitation_selector,
        ]

        self.spec_display_plotwidgets = [
            self.definition_widget.specification_all_frequencies_plot,
            self.definition_widget.specification_all_amplitudes_plot,
            self.definition_widget.specification_channel_amplitude_plot,
            self.definition_widget.specification_channel_phase_plot,
        ]

        self.spec_display_viewboxes = []
        self.spec_display_imgviews = []
        for plotwidget in self.spec_display_plotwidgets:
            plot_item = plotwidget.getPlotItem()
            plot_item.showGrid(True, True, 0.25)
            plot_item.enableAutoRange()
            plot_item.getViewBox().enableAutoRange(enable=True)

        for widget in [
            self.run_widget.control_updates_signal_selector,
            self.run_widget.signal_selector,
        ]:
            widget.setSelectionMode(QtWidgets.QTableWidget.SingleSelection)
        for widget in [self.run_widget.partial_environment_tone_selector]:
            widget.setSelectionMode(QtWidgets.QListWidget.MultiSelection)

        plotitem = self.definition_widget.specification_all_frequencies_plot.getPlotItem()
        plotitem.setLabel("bottom", "Time")
        plotitem.setLabel("left", "Frequency")
        plotitem = self.definition_widget.specification_all_amplitudes_plot.getPlotItem()
        plotitem.setLabel("bottom", "Frequency")
        plotitem.setLabel("left", "Amplitude")
        plotitem = self.definition_widget.specification_channel_amplitude_plot.getPlotItem()
        plotitem.setLabel("bottom", "Frequency")
        plotitem.setLabel("left", "Amplitude")
        plotitem = self.definition_widget.specification_channel_phase_plot.getPlotItem()
        plotitem.setLabel("bottom", "Frequency")
        plotitem.setLabel("left", "Phase (deg)")

        self.change_filter_setting_visibility()

        self.connect_callbacks()

        # Complete the profile commands
        self.command_map["Set Test Level"] = self.change_test_level_from_profile
        self.command_map["Save Control Data"] = self.save_control_data_from_profile

    def connect_callbacks(self):
        """Connects UI callbacks to object methods"""
        # Definition
        self.definition_widget.transformation_matrices_button.clicked.connect(
            self.define_transformation_matrices
        )
        self.definition_widget.control_channels_selector.itemChanged.connect(
            self.update_control_channels
        )
        self.definition_widget.check_selected_button.clicked.connect(
            self.check_selected_control_channels
        )
        self.definition_widget.uncheck_selected_button.clicked.connect(
            self.uncheck_selected_control_channels
        )
        self.definition_widget.specification_row_selector.currentIndexChanged.connect(
            self.update_specification
        )
        self.definition_widget.script_load_file_button.clicked.connect(self.select_python_module)
        self.definition_widget.sine_table_tab_widget.currentChanged.connect(
            self.sine_table_tab_changed
        )
        self.definition_widget.explore_filter_button.clicked.connect(self.explore_filter_settings)
        self.definition_widget.filter_type_selector.currentIndexChanged.connect(
            self.change_filter_setting_visibility
        )
        # Prediction
        self.prediction_widget.excitation_selector.currentIndexChanged.connect(
            self.send_excitation_prediction_plot_choices
        )
        self.prediction_widget.excitation_display_type.currentIndexChanged.connect(
            self.update_excitation_prediction_type
        )
        self.prediction_widget.excitation_display_tone.currentIndexChanged.connect(
            self.update_excitation_prediction_tone
        )
        self.prediction_widget.response_selector.currentIndexChanged.connect(
            self.send_response_prediction_plot_choices
        )
        self.prediction_widget.response_display_type.currentIndexChanged.connect(
            self.update_response_prediction_type
        )
        self.prediction_widget.response_display_tone.currentIndexChanged.connect(
            self.update_response_prediction_tone
        )
        self.prediction_widget.excitation_voltage_list.itemDoubleClicked.connect(
            self.update_excitation_prediction_from_table
        )
        self.prediction_widget.response_error_table.cellDoubleClicked.connect(
            self.update_response_prediction_from_table
        )
        # Run Test
        self.run_widget.start_test_button.clicked.connect(self.start_control)
        self.run_widget.stop_test_button.clicked.connect(self.stop_control)
        self.run_widget.create_window_button.clicked.connect(self.create_window)
        self.run_widget.show_all_channels_button.clicked.connect(self.show_all_channels)
        self.run_widget.tile_windows_button.clicked.connect(self.tile_windows)
        self.run_widget.close_windows_button.clicked.connect(self.close_windows)
        self.run_widget.control_updates_signal_selector.itemSelectionChanged.connect(
            self.update_control_run_plot
        )
        self.run_widget.signal_selector.currentCellChanged.connect(self.update_run_plot)
        self.run_widget.save_control_data_button.clicked.connect(self.save_control_data)
        self.run_widget.partial_environment_selector.stateChanged.connect(
            self.enable_disable_partial_environment
        )

    # %% Data Acquisition

    def initialize_data_acquisition(self, data_acquisition_parameters):
        super().initialize_data_acquisition(data_acquisition_parameters)
        # Initialize Plots
        for plotwidget in self.spec_display_plotwidgets:
            plotwidget.clear()
        self.plot_data_items["specification_all_frequencies"] = VaryingNumberOfLinePlot(
            self.definition_widget.specification_all_frequencies_plot.getPlotItem()
        )
        self.plot_data_items["specification_all_amplitudes"] = VaryingNumberOfLinePlot(
            self.definition_widget.specification_all_amplitudes_plot.getPlotItem()
        )
        self.plot_data_items[
            "specification_channel_phase"
        ] = self.definition_widget.specification_channel_phase_plot.getPlotItem().plot(
            np.array([0, 1]), np.zeros(2), pen={"color": "b", "width": 1}, name="Phase"
        )
        self.plot_data_items[
            "specification_channel_amplitude"
        ] = self.definition_widget.specification_channel_amplitude_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.zeros(2),
            pen={"color": "b", "width": 1},
            name="Amplitude",
        )
        self.plot_data_items[
            "specification_channel_warning_upper"
        ] = self.definition_widget.specification_channel_amplitude_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.zeros(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
            name="Warning",
        )
        self.plot_data_items[
            "specification_channel_warning_lower"
        ] = self.definition_widget.specification_channel_amplitude_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.zeros(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
        )
        self.plot_data_items[
            "specification_channel_abort_upper"
        ] = self.definition_widget.specification_channel_amplitude_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.zeros(2),
            pen={"color": (153, 0, 0), "width": 1, "style": Qt.DashLine},
            name="Abort",
        )
        self.plot_data_items[
            "specification_channel_abort_lower"
        ] = self.definition_widget.specification_channel_amplitude_plot.getPlotItem().plot(
            np.array([0, 1]),
            np.zeros(2),
            pen={"color": (153, 0, 0), "width": 1, "style": Qt.DashLine},
        )
        self.definition_widget.specification_channel_amplitude_plot.getPlotItem().addLegend()

        # Set up channel names
        self.physical_channel_names = [
            (
                f"{'' if channel.channel_type is None else channel.channel_type} "
                f"{channel.node_number} "
                f"{'' if channel.node_direction is None else channel.node_direction}"
            )[:MAXIMUM_NAME_LENGTH]
            for channel in data_acquisition_parameters.channel_list
        ]
        self.physical_unit_names = [
            f"{'-' if channel.unit is None else channel.unit}"
            for channel in data_acquisition_parameters.channel_list
        ]
        self.physical_output_indices = [
            i
            for i, channel in enumerate(data_acquisition_parameters.channel_list)
            if channel.feedback_device
        ]
        # Set up widgets
        self.definition_widget.sample_rate_display.setValue(data_acquisition_parameters.sample_rate)
        self.system_id_widget.samplesPerFrameSpinBox.setValue(
            data_acquisition_parameters.sample_rate
        )
        self.definition_widget.samples_per_acquire_display.setValue(
            data_acquisition_parameters.samples_per_read
        )
        self.definition_widget.samples_per_write_display.setValue(
            data_acquisition_parameters.samples_per_write
        )
        self.definition_widget.frame_time_display.setValue(
            data_acquisition_parameters.samples_per_read / data_acquisition_parameters.sample_rate
        )
        self.definition_widget.nyquist_frequency_display.setValue(
            data_acquisition_parameters.sample_rate / 2
        )
        self.definition_widget.control_channels_selector.clear()
        for channel_name in self.physical_channel_names:
            item = QtWidgets.QListWidgetItem()
            item.setText(channel_name)
            item.setFlags(
                item.flags() | Qt.ItemIsUserCheckable
            )  # | Qt.ItemIsUserTristate) # We will add this when we implement limits
            item.setCheckState(Qt.Unchecked)
            self.definition_widget.control_channels_selector.addItem(item)
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.define_transformation_matrices(None, False)
        self.definition_widget.input_channels_display.setValue(len(self.physical_channel_names))
        self.definition_widget.output_channels_display.setValue(len(self.physical_output_indices))
        self.definition_widget.control_channels_display.setValue(0)
        if self.definition_widget.sine_table_tab_widget.count() == 1:
            self.sine_tables.append(
                SineSweepTable(
                    self.definition_widget.sine_table_tab_widget,
                    self.update_specification,
                    self.remove_sine_table_entry,
                    (
                        self.physical_control_names
                        if self.response_transformation_matrix is None
                        else [
                            f"Transformed Response {i}"
                            for i in range(self.response_transformation_matrix.shape[0])
                        ]
                    ),
                    self.data_acquisition_parameters,
                )
            )
        self.clear_and_update_specification_table()

    @property
    def physical_output_names(self):
        """Defines names of the physical drive channels"""
        return [self.physical_channel_names[i] for i in self.physical_output_indices]

    # %% Environment

    @property
    def physical_control_indices(self):
        """Gets the physical control indices currently checked"""
        return [
            i
            for i in range(self.definition_widget.control_channels_selector.count())
            if self.definition_widget.control_channels_selector.item(i).checkState() == Qt.Checked
        ]

    @property
    def physical_control_names(self):
        """Gets the names for the physical control channels currently checked"""
        return [self.physical_channel_names[i] for i in self.physical_control_indices]

    @property
    def physical_control_units(self):
        """Gets the unit for the control channels currently checked"""
        return [self.physical_unit_names[i] for i in self.physical_control_indices]

    @property
    def initialized_control_names(self):
        """Gets the names of the control channels that have been initialized"""
        if self.environment_parameters.response_transformation_matrix is None:
            return [
                self.physical_channel_names[i]
                for i in self.environment_parameters.control_channel_indices
            ]
        return [
            f"Transformed Response {i + 1}"
            for i in range(self.environment_parameters.response_transformation_matrix.shape[0])
        ]

    @property
    def initialized_output_names(self):
        """Gets the names of the drive channels that have been initialized"""
        if self.environment_parameters.reference_transformation_matrix is None:
            return self.physical_output_names
        else:
            return [
                f"Transformed Drive {i + 1}"
                for i in range(self.environment_parameters.reference_transformation_matrix.shape[0])
            ]

    def update_control_channels(self):
        """Updates the control channels due to selection changes"""
        self.response_transformation_matrix = None
        self.output_transformation_matrix = None
        self.definition_widget.control_channels_display.setValue(len(self.physical_control_indices))
        self.define_transformation_matrices(None, False)
        self.clear_and_update_specification_table()

    def check_selected_control_channels(self):
        """Checks the selected control channels on the UI"""
        for item in self.definition_widget.control_channels_selector.selectedItems():
            item.setCheckState(Qt.Checked)

    def uncheck_selected_control_channels(self):
        """Unchecks the selected control channels on the UI"""
        for item in self.definition_widget.control_channels_selector.selectedItems():
            item.setCheckState(Qt.Unchecked)

    def clear_and_update_specification_table(self):
        """Clears the specification table of all information"""
        control_names = (
            self.physical_control_names
            if self.response_transformation_matrix is None
            else [
                f"Transformed Response {i+1}"
                for i in range(self.response_transformation_matrix.shape[0])
            ]
        )
        for sine_table in self.sine_tables:
            sine_table.clear_and_update_specification_table(control_names=control_names)

    def add_sine_table_tab(self):
        """Adds a new sine tone to the sine specification table"""
        self.definition_widget.sine_table_tab_widget.blockSignals(True)
        self.sine_tables.append(
            SineSweepTable(
                self.definition_widget.sine_table_tab_widget,
                self.update_specification,
                self.remove_sine_table_entry,
                (
                    self.physical_control_names
                    if self.response_transformation_matrix is None
                    else [
                        f"Transformed Response {i}"
                        for i in range(self.response_transformation_matrix.shape[0])
                    ]
                ),
                self.data_acquisition_parameters,
            )
        )
        self.definition_widget.sine_table_tab_widget.blockSignals(False)

    def sine_table_tab_changed(self, index):
        """Updates the displayed sine table and adds a new index if necessary."""
        if index == self.definition_widget.sine_table_tab_widget.count() - 1:
            self.add_sine_table_tab()
        else:
            self.update_specification()

    def remove_sine_table_entry(self, index):
        """Removes a tone from the sine table"""
        self.definition_widget.sine_table_tab_widget.setCurrentIndex(0)
        self.definition_widget.sine_table_tab_widget.removeTab(index)
        self.sine_tables.pop(index)
        for i, table in enumerate(self.sine_tables):
            table.index = i
        self.update_specification()

    def change_filter_setting_visibility(self):
        """Updates which settings are available depending on the selected filter type"""
        isdtf = self.definition_widget.filter_type_selector.currentIndex() == 0
        for widget in [
            self.definition_widget.vk_filter_order_label,
            self.definition_widget.vk_filter_order_selector,
            self.definition_widget.vk_filter_block_overlap_label,
            self.definition_widget.vk_filter_block_overlap_selector,
            self.definition_widget.vk_filter_bandwidth_label,
            self.definition_widget.vk_filter_bandwidth_selector,
            self.definition_widget.vk_filter_block_size_label,
            self.definition_widget.vk_filter_block_size_selector,
        ]:
            widget.setVisible(not isdtf)
        for widget in [
            self.definition_widget.tracking_filter_cutoff_label,
            self.definition_widget.tracking_filter_cutoff_selector,
            self.definition_widget.tracking_filter_order_label,
            self.definition_widget.tracking_filter_order_selector,
        ]:
            widget.setVisible(isdtf)

    def collect_specification(self):
        """Collects the specifications defined in the sine table"""
        specs = []
        for sine_table in self.sine_tables:
            spec = sine_table.get_specification()
            specs.append(spec)
        return specs

    def update_specification(self):
        """Updates the specification in the table and plots based on the selection"""
        # print('Updating Specification Plot')
        # Go through each of the sine signals
        for sine_table in self.sine_tables:
            # Go through and update the prefixes
            for row in range(sine_table.widget.breakpoint_table.rowCount() - 1):
                combobox = sine_table.widget.breakpoint_table.cellWidget(row, 1)
                spinbox = sine_table.widget.breakpoint_table.cellWidget(row, 2)
                if combobox.currentIndex() == 0:
                    spinbox.setSuffix(" Hz/s")
                else:
                    spinbox.setSuffix(" oct/min")
        # Generate representative time signals
        specs = self.collect_specification()
        if len(specs) == 0:
            return
        table_index = self.definition_widget.sine_table_tab_widget.currentIndex()
        control_index = self.definition_widget.specification_row_selector.currentIndex()
        all_ordinate = []
        all_abscissa = []
        all_frequency = []
        all_amplitude = []
        all_phase = []
        for spec in specs:
            (
                ordinate,
                frequency,
                _,
                amplitude,
                phase,
                abscissa,
                _,
                _,
            ) = spec.create_signal(
                self.data_acquisition_parameters.sample_rate,
                control_index=control_index,
                ignore_start_time=True,
                only_breakpoints=True,
            )
            # print(f'Shapes: {ordinate.shape=}, {frequency.shape=}, '
            # '{amplitude.shape=}, {phase.shape=}')
            all_ordinate.append(ordinate)
            all_abscissa.append(abscissa + spec.start_time)
            all_frequency.append(frequency)
            all_amplitude.append(amplitude)
            all_phase.append(phase)
        self.plot_data_items["specification_all_frequencies"].set_data(all_abscissa, all_frequency)
        self.plot_data_items["specification_all_amplitudes"].set_data(all_frequency, all_amplitude)

        self.plot_data_items["specification_channel_amplitude"].setData(
            all_frequency[table_index], all_amplitude[table_index]
        )
        self.plot_data_items["specification_channel_phase"].setData(
            all_frequency[table_index], all_phase[table_index] * 180 / np.pi
        )
        self.plot_data_items["specification_channel_warning_lower"].setData(
            np.repeat(specs[table_index].breakpoint_table["frequency"], 2),
            specs[table_index].breakpoint_table["warning"][:, 0, :, control_index].flatten(),
        )
        self.plot_data_items["specification_channel_warning_upper"].setData(
            np.repeat(specs[table_index].breakpoint_table["frequency"], 2),
            specs[table_index].breakpoint_table["warning"][:, 1, :, control_index].flatten(),
        )
        self.plot_data_items["specification_channel_abort_lower"].setData(
            np.repeat(specs[table_index].breakpoint_table["frequency"], 2),
            specs[table_index].breakpoint_table["abort"][:, 0, :, control_index].flatten(),
        )
        self.plot_data_items["specification_channel_abort_upper"].setData(
            np.repeat(specs[table_index].breakpoint_table["frequency"], 2),
            specs[table_index].breakpoint_table["abort"][:, 1, :, control_index].flatten(),
        )
        # Return the length of the specification
        return max(max(abscissa) for abscissa in all_abscissa)

    def explore_filter_settings(self):
        """Brings up a dialog box to explore filter settings"""
        control_names = (
            self.physical_control_names
            if self.response_transformation_matrix is None
            else [
                f"Transformed Response {i+1}"
                for i in range(self.response_transformation_matrix.shape[0])
            ]
        )
        order_names = [
            self.definition_widget.sine_table_tab_widget.tabText(i)
            for i in range(self.definition_widget.sine_table_tab_widget.count() - 1)
        ]
        specs = self.collect_specification()
        (
            result,
            filter_type,
            dtf_cutoff,
            dtf_order,
            vk_order,
            vk_bandwidth,
            vk_blocksize,
            vk_overlap,
        ) = FilterExplorer.explore_filter_settings(
            control_names,
            order_names,
            specs,
            self.definition_widget.filter_type_selector.currentIndex(),
            self.definition_widget.tracking_filter_cutoff_selector.value(),
            self.definition_widget.tracking_filter_order_selector.value(),
            self.definition_widget.vk_filter_order_selector.currentIndex() + 1,
            self.definition_widget.vk_filter_bandwidth_selector.value(),
            self.definition_widget.vk_filter_block_size_selector.value(),
            self.definition_widget.vk_filter_block_overlap_selector.value(),
            self.data_acquisition_parameters.sample_rate,
            self.definition_widget.ramp_time_spinbox.value(),
            self.data_acquisition_parameters.samples_per_read,
            self.definition_widget,
        )
        if result:
            self.definition_widget.filter_type_selector.setCurrentIndex(filter_type)
            self.definition_widget.tracking_filter_cutoff_selector.setValue(dtf_cutoff)
            self.definition_widget.tracking_filter_order_selector.setValue(dtf_order)
            self.definition_widget.vk_filter_order_selector.setCurrentIndex(vk_order - 1)
            self.definition_widget.vk_filter_bandwidth_selector.setValue(vk_bandwidth)
            self.definition_widget.vk_filter_block_size_selector.setValue(vk_blocksize)
            self.definition_widget.vk_filter_block_overlap_selector.setValue(vk_overlap)

    def select_python_module(self, clicked, filename=None):  # pylint: disable=unused-argument
        """Loads a Python module using a dialog or the specified filename

        Parameters
        ----------
        clicked :
            The clicked event that triggered the callback.
        filename :
            File name defining the Python module for bypassing the callback when
            loading from a file (Default value = None).

        """
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.definition_widget,
                "Select Python Module",
                filter="Python Modules (*.py)",
            )
            if filename == "":
                return
        self.python_control_module = load_python_module(filename)
        classes = [
            function
            for function in inspect.getmembers(self.python_control_module)
            if (
                inspect.isclass(function[1])
                and all(
                    [method in function[1].__dict__ for method in ["system_id_update", "control"]]
                )
            )
        ]
        self.log(
            f"Loaded module {self.python_control_module.__name__} with classes "
            f"{[control_class[0] for control_class in classes]}"
        )
        self.definition_widget.python_class_input.clear()
        self.definition_widget.script_file_path_input.setText(filename)
        for control_class in classes:
            self.definition_widget.python_class_input.addItem(control_class[0])

    def collect_environment_definition_parameters(self):
        if self.python_control_module is None:
            control_module = None
            control_class = None
            control_class_parameters = (
                self.definition_widget.control_parameters_text_input.toPlainText()
            )
        else:
            control_module = self.definition_widget.script_file_path_input.text()
            control_class = self.definition_widget.python_class_input.itemText(
                self.definition_widget.python_class_input.currentIndex()
            )
            control_class_parameters = (
                self.definition_widget.control_parameters_text_input.toPlainText()
            )
        return SineMetadata(
            sample_rate=self.definition_widget.sample_rate_display.value(),
            samples_per_frame=self.definition_widget.samples_per_acquire_display.value(),
            number_of_channels=len(self.data_acquisition_parameters.channel_list),
            specifications=self.collect_specification(),
            ramp_time=self.definition_widget.ramp_time_spinbox.value(),
            buffer_blocks=self.definition_widget.buffer_blocks_selector.value(),
            control_convergence=self.definition_widget.control_convergence_selector.value(),
            update_drives_after_environment=self.definition_widget.update_drives_after_environment_selector.isChecked(),
            phase_fit=self.definition_widget.best_fit_phase_checkbox.isChecked(),
            allow_automatic_aborts=self.definition_widget.auto_abort_checkbox.isChecked(),
            tracking_filter_type=self.definition_widget.filter_type_selector.currentIndex(),
            tracking_filter_cutoff=self.definition_widget.tracking_filter_cutoff_selector.value()
            / 100,
            tracking_filter_order=self.definition_widget.tracking_filter_order_selector.value(),
            vk_filter_order=self.definition_widget.vk_filter_order_selector.currentIndex() + 1,
            vk_filter_bandwidth=self.definition_widget.vk_filter_bandwidth_selector.value(),
            vk_filter_blocksize=self.definition_widget.vk_filter_block_size_selector.value(),
            vk_filter_overlap=self.definition_widget.vk_filter_block_overlap_selector.value(),
            control_python_script=control_module,
            control_python_class=control_class,
            control_python_parameters=control_class_parameters,
            control_channel_indices=self.physical_control_indices,
            output_channel_indices=self.physical_output_indices,
            response_transformation_matrix=self.response_transformation_matrix,
            output_transformation_matrix=self.output_transformation_matrix,
        )

    def initialize_environment(self):
        super().initialize_environment()
        # Set up channel names in selectors
        for widget in [
            self.prediction_widget.response_selector,
            self.run_widget.control_channel_selector,
        ]:
            widget.blockSignals(True)
            widget.clear()
            for i, control_name in enumerate(self.initialized_control_names):
                widget.addItem(f"{i + 1}: {control_name}")
            if isinstance(widget, QtWidgets.QListWidget):
                widget.setCurrentRow(0)
            widget.blockSignals(False)
        for widget in [self.prediction_widget.excitation_selector]:
            widget.blockSignals(True)
            widget.clear()
            for i, drive_name in enumerate(self.initialized_output_names):
                widget.addItem(f"{i + 1}: {drive_name}")
            if isinstance(widget, QtWidgets.QListWidget):
                widget.setCurrentRow(0)
            widget.blockSignals(False)
        # Set up tone names in selectors
        for widget in [
            self.prediction_widget.response_display_tone,
            self.prediction_widget.excitation_display_tone,
            self.run_widget.partial_environment_tone_selector,
            self.run_widget.control_tone_selector,
        ]:
            widget.blockSignals(True)
            widget.clear()
            if widget not in [
                self.run_widget.partial_environment_tone_selector,
                self.run_widget.control_tone_selector,
            ]:
                widget.addItem("All Tones")
            for table in self.sine_tables:
                widget.addItem(table.widget.name_editor.text())
            if isinstance(widget, QtWidgets.QListWidget):
                widget.setCurrentRow(0)
            widget.blockSignals(False)
        # Set up the run widget tables
        for widget, channel_names in zip(
            [
                self.run_widget.signal_selector,
                self.run_widget.control_updates_signal_selector,
            ],
            [self.initialized_control_names, self.initialized_output_names],
        ):
            widget.blockSignals(True)
            widget.clear()
            widget.setRowCount(len(self.sine_tables))
            widget.setColumnCount(len(channel_names))
            for i in range(widget.rowCount()):
                for j in range(widget.columnCount()):
                    item = QtWidgets.QTableWidgetItem("0.000")
                    widget.setItem(i, j, item)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            widget.blockSignals(False)

        # Set up the prediction and run plots
        self.prediction_widget.excitation_display_plot.getPlotItem().clear()
        self.prediction_widget.response_display_plot.getPlotItem().clear()
        self.run_widget.control_updates_plot.getPlotItem().clear()
        self.run_widget.amplitude_plot.getPlotItem().clear()
        self.run_widget.phase_plot.getPlotItem().clear()
        self.run_widget.amplitude_plot.getPlotItem().addLegend()
        self.run_widget.phase_plot.getPlotItem().addLegend()
        self.prediction_widget.excitation_display_plot.getPlotItem().addLegend()
        self.prediction_widget.response_display_plot.getPlotItem().addLegend()
        self.plot_data_items["response_prediction"] = multiline_plotter(
            np.arange(2),
            np.zeros((2, 2)),
            widget=self.prediction_widget.response_display_plot,
            other_pen_options={"width": 1},
            names=["Prediction", "Spec"],
        )
        self.plot_data_items["excitation_prediction"] = multiline_plotter(
            np.arange(2),
            np.zeros((1, 2)),
            widget=self.prediction_widget.excitation_display_plot,
            other_pen_options={"width": 1},
            names=["Prediction"],
        )
        self.plot_data_items["control_amplitude"] = multiline_plotter(
            np.arange(2),
            np.zeros((2, 2)),
            widget=self.run_widget.amplitude_plot,
            other_pen_options={"width": 1},
            names=["Achieved", "Spec"],
        )
        self.plot_data_items["control_phase"] = multiline_plotter(
            np.arange(2),
            np.zeros((2, 2)),
            widget=self.run_widget.phase_plot,
            other_pen_options={"width": 1},
            names=["Achieved", "Spec"],
        )
        self.plot_data_items[
            "control_warning_upper"
        ] = self.run_widget.amplitude_plot.getPlotItem().plot(
            np.array([0, 0]),
            np.zeros(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
            name="Warning",
        )
        self.plot_data_items[
            "control_warning_lower"
        ] = self.run_widget.amplitude_plot.getPlotItem().plot(
            np.array([0, 0]),
            np.zeros(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
        )
        self.plot_data_items[
            "control_abort_upper"
        ] = self.run_widget.amplitude_plot.getPlotItem().plot(
            np.array([0, 0]),
            np.zeros(2),
            pen={"color": (153, 0, 0), "width": 1, "style": Qt.DashLine},
            name="Abort",
        )
        self.plot_data_items[
            "control_abort_lower"
        ] = self.run_widget.amplitude_plot.getPlotItem().plot(
            np.array([0, 0]),
            np.zeros(2),
            pen={"color": (153, 0, 0), "width": 1, "style": Qt.DashLine},
        )
        self.plot_data_items[
            "prediction_warning_upper"
        ] = self.prediction_widget.response_display_plot.getPlotItem().plot(
            np.array([0, 0]),
            np.zeros(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
            name="Warning",
        )
        self.plot_data_items[
            "prediction_warning_lower"
        ] = self.prediction_widget.response_display_plot.getPlotItem().plot(
            np.array([0, 0]),
            np.zeros(2),
            pen={"color": (255, 204, 0), "width": 1, "style": Qt.DashLine},
        )
        self.plot_data_items[
            "prediction_abort_upper"
        ] = self.prediction_widget.response_display_plot.getPlotItem().plot(
            np.array([0, 0]),
            np.zeros(2),
            pen={"color": (153, 0, 0), "width": 1, "style": Qt.DashLine},
            name="Abort",
        )
        self.plot_data_items[
            "prediction_abort_lower"
        ] = self.prediction_widget.response_display_plot.getPlotItem().plot(
            np.array([0, 0]),
            np.zeros(2),
            pen={"color": (153, 0, 0), "width": 1, "style": Qt.DashLine},
        )
        self.plot_data_items["control_updates"] = blended_scatter_plot(
            np.zeros((10, 2)), widget=self.run_widget.control_updates_plot
        )

        # Make sure the specification starts at 0
        min_time = min(spec.start_time for spec in self.collect_specification())
        for sine_table in self.sine_tables:
            sine_table.widget.start_time_selector.setValue(
                sine_table.widget.start_time_selector.value() - min_time
            )

        self.spec_time = self.update_specification()

        for widget in [
            self.run_widget.start_time_selector,
            self.run_widget.stop_time_selector,
        ]:
            widget.setMinimum(0)
            widget.setMaximum(self.spec_time)
        self.run_widget.stop_time_selector.setValue(self.spec_time)

        return self.environment_parameters

    def define_transformation_matrices(
        self, clicked, dialog=True
    ):  # pylint: disable=unused-argument
        """Defines the transformation matrices using the dialog box"""
        if dialog:
            (response_transformation, output_transformation, result) = (
                TransformationMatrixWindow.define_transformation_matrices(
                    self.response_transformation_matrix,
                    self.definition_widget.control_channels_display.value(),
                    self.output_transformation_matrix,
                    self.definition_widget.output_channels_display.value(),
                    self.definition_widget,
                )
            )
        else:
            response_transformation = self.response_transformation_matrix
            output_transformation = self.output_transformation_matrix
            result = True
        if result:
            # Update the control names
            for widget in self.control_selector_widgets:
                widget.blockSignals(True)
                widget.clear()
            if response_transformation is None:
                for i, control_name in enumerate(self.physical_control_names):
                    for widget in self.control_selector_widgets:
                        widget.addItem(f"{i + 1}: {control_name}")
                self.definition_widget.transform_channels_display.setValue(
                    len(self.physical_control_names)
                )
            else:
                for i in range(response_transformation.shape[0]):
                    for widget in self.control_selector_widgets:
                        widget.addItem(f"{i + 1}: Virtual Response")
                self.definition_widget.transform_channels_display.setValue(
                    response_transformation.shape[0]
                )
            for widget in self.control_selector_widgets:
                widget.blockSignals(False)
            # Update the output names
            for widget in self.output_selector_widgets:
                widget.blockSignals(True)
                widget.clear()
            if output_transformation is None:
                for i, drive_name in enumerate(self.physical_output_names):
                    for widget in self.output_selector_widgets:
                        widget.addItem(f"{i + 1}: {drive_name}")
                self.definition_widget.transform_outputs_display.setValue(
                    len(self.physical_output_names)
                )
            else:
                for i in range(output_transformation.shape[0]):
                    for widget in self.output_selector_widgets:
                        widget.addItem(f"{i + 1}: Virtual Drive")
                self.definition_widget.transform_outputs_display.setValue(
                    output_transformation.shape[0]
                )
            for widget in self.output_selector_widgets:
                widget.blockSignals(False)
            self.response_transformation_matrix = response_transformation
            self.output_transformation_matrix = output_transformation
            self.clear_and_update_specification_table()

    # %% Predictions

    def update_response_prediction_tone(self):
        """Called when the tone is changed, sends selection to environment"""
        type_index = self.prediction_widget.response_display_type.currentIndex()
        tone_index = (
            self.prediction_widget.response_display_tone.currentIndex() - 1
        )  # All tones is first
        if tone_index < 0 and type_index != 0:  # For all tones we can only show time histories
            self.prediction_widget.response_display_type.blockSignals(True)
            self.prediction_widget.response_display_type.setCurrentIndex(0)
            self.prediction_widget.response_display_type.blockSignals(False)
        self.send_response_prediction_plot_choices()

    def update_excitation_prediction_tone(self):
        """Called when the tone is changed, sends selection to the environment"""
        # Excitation
        type_index = self.prediction_widget.excitation_display_type.currentIndex()
        tone_index = (
            self.prediction_widget.excitation_display_tone.currentIndex() - 1
        )  # All tones is first
        if tone_index < 0 and type_index != 0:  # For all tones we can only show time histories
            self.prediction_widget.excitation_display_type.blockSignals(True)
            self.prediction_widget.excitation_display_type.setCurrentIndex(0)
            self.prediction_widget.excitation_display_type.blockSignals(False)
        self.send_excitation_prediction_plot_choices()

    def update_response_prediction_type(self):
        """Called when the response type is changed, sends selection to environment"""
        type_index = self.prediction_widget.response_display_type.currentIndex()
        tone_index = (
            self.prediction_widget.response_display_tone.currentIndex() - 1
        )  # All tones is first
        if tone_index < 0 and type_index != 0:  # For all tones we can only show time histories
            self.prediction_widget.response_display_tone.blockSignals(True)
            self.prediction_widget.response_display_tone.setCurrentIndex(1)
            self.prediction_widget.response_display_tone.blockSignals(False)
        self.send_response_prediction_plot_choices()

    def update_excitation_prediction_type(self):
        """Called when the drive type is changed, sends selection to environment"""
        # Excitation
        type_index = self.prediction_widget.excitation_display_type.currentIndex()
        tone_index = (
            self.prediction_widget.excitation_display_tone.currentIndex() - 1
        )  # All tones is first
        if tone_index < 0 and type_index != 0:  # For all tones we can only show time histories
            self.prediction_widget.excitation_display_tone.blockSignals(True)
            self.prediction_widget.excitation_display_tone.setCurrentIndex(1)
            self.prediction_widget.excitation_display_tone.blockSignals(False)
        self.send_excitation_prediction_plot_choices()

    def send_response_prediction_plot_choices(self):
        """Sends the response prediction plot choices to the environment"""
        channel_index = self.prediction_widget.response_selector.currentIndex()
        type_index = self.prediction_widget.response_display_type.currentIndex()
        tone_index = (
            self.prediction_widget.response_display_tone.currentIndex() - 1
        )  # All tones is first
        self.environment_command_queue.put(
            self.log_name,
            (
                SineCommands.SEND_RESPONSE_PREDICTION,
                (channel_index, type_index, tone_index),
            ),
        )
        self.plot_prediction_warnings_and_aborts()  # Update the plots for the warning/abort limits

    def send_excitation_prediction_plot_choices(self):
        """Sends the drive prediction plot choices to the environment"""
        channel_index = self.prediction_widget.excitation_selector.currentIndex()
        type_index = self.prediction_widget.excitation_display_type.currentIndex()
        tone_index = (
            self.prediction_widget.excitation_display_tone.currentIndex() - 1
        )  # All tones is first
        self.environment_command_queue.put(
            self.log_name,
            (
                SineCommands.SEND_EXCITATION_PREDICTION,
                (channel_index, type_index, tone_index),
            ),
        )

    def plot_prediction_warnings_and_aborts(self):
        """Adds warning and aborts to the prediction tab"""
        if self.prediction_widget.response_display_type.currentIndex() == 3:
            # Plot the response
            specs = self.environment_parameters.specifications
            table_index = self.prediction_widget.response_display_tone.currentIndex() - 1
            control_index = self.prediction_widget.response_selector.currentIndex()
            self.plot_data_items["prediction_warning_lower"].setData(
                np.repeat(specs[table_index].breakpoint_table["frequency"], 2),
                specs[table_index].breakpoint_table["warning"][:, 0, :, control_index].flatten(),
            )
            self.plot_data_items["prediction_warning_upper"].setData(
                np.repeat(specs[table_index].breakpoint_table["frequency"], 2),
                specs[table_index].breakpoint_table["warning"][:, 1, :, control_index].flatten(),
            )
            self.plot_data_items["prediction_abort_lower"].setData(
                np.repeat(specs[table_index].breakpoint_table["frequency"], 2),
                specs[table_index].breakpoint_table["abort"][:, 0, :, control_index].flatten(),
            )
            self.plot_data_items["prediction_abort_upper"].setData(
                np.repeat(specs[table_index].breakpoint_table["frequency"], 2),
                specs[table_index].breakpoint_table["abort"][:, 1, :, control_index].flatten(),
            )
        else:
            for item in [
                "prediction_warning_lower",
                "prediction_warning_upper",
                "prediction_abort_lower",
                "prediction_abort_upper",
            ]:
                self.plot_data_items[item].setData(np.zeros(2), np.nan * np.ones(2))

    def plot_excitation_prediction(self, abscissa, ordinate):
        """Plots the recieved drive prediction"""
        self.plot_data_items["excitation_prediction"][0].setData(abscissa, ordinate)

    def plot_response_prediction(self, abscissa, ordinate):
        """Plots the recieved control prediction"""
        for index, this_ordinate in enumerate(ordinate[::-1]):
            plot_length = min(abscissa.shape[-1], this_ordinate.shape[-1])
            self.plot_data_items["response_prediction"][index].setData(
                abscissa[:plot_length], this_ordinate[:plot_length]
            )

    def update_voltage_list(self, voltages):
        """Updates the voltage list with predicted values"""
        self.prediction_widget.excitation_voltage_list.clear()
        for value in voltages:
            self.prediction_widget.excitation_voltage_list.addItem(f"{value:.3f}")

    def update_response_matrix(self, amplitude_error, warning_matrix, abort_matrix):
        """Updates the response error predictions in the table"""
        self.prediction_widget.response_error_table.clear()
        self.prediction_widget.response_error_table.setRowCount(amplitude_error.shape[0])
        self.prediction_widget.response_error_table.setColumnCount(amplitude_error.shape[1])
        for i in range(amplitude_error.shape[0]):
            for j in range(amplitude_error.shape[1]):
                error_value = amplitude_error[i, j]
                item = QtWidgets.QTableWidgetItem(f"{error_value:0.3f}")
                self.prediction_widget.response_error_table.setItem(i, j, item)
                if abort_matrix[i, j]:
                    item.setBackground(QColor(255, 125, 125))
                elif warning_matrix[i, j]:
                    item.setBackground(QColor(255, 255, 125))
                else:
                    item.setBackground(QColor(255, 255, 255))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

    def update_response_prediction_from_table(self, row, column):
        """Selects the specified tone and channel from a double-click on the prediction table"""
        widgets = [
            self.prediction_widget.response_display_type,
            self.prediction_widget.response_display_tone,
            self.prediction_widget.response_selector,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        self.prediction_widget.response_display_type.setCurrentIndex(3)
        self.prediction_widget.response_display_tone.setCurrentIndex(row + 1)  # All tones is first
        self.prediction_widget.response_selector.setCurrentIndex(column)
        for widget in widgets:
            widget.blockSignals(False)
        self.send_response_prediction_plot_choices()

    def update_excitation_prediction_from_table(self, item):
        """Updates the specified tone and channel from a double-click on the predicted voltage"""
        index = self.prediction_widget.excitation_voltage_list.row(item)
        widgets = [
            self.prediction_widget.excitation_display_type,
            self.prediction_widget.excitation_display_tone,
            self.prediction_widget.excitation_selector,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        self.prediction_widget.excitation_display_type.setCurrentIndex(0)
        self.prediction_widget.excitation_display_tone.setCurrentIndex(0)  # All tones is first
        self.prediction_widget.excitation_selector.setCurrentIndex(index)
        for widget in widgets:
            widget.blockSignals(False)
        self.send_excitation_prediction_plot_choices()

    # %% Control

    def start_control(self):
        """Sets itself up to start controlling and sends a signal to the environment to start"""
        self.achieved_response_signals_combined = []
        self.achieved_response_signals = []
        self.achieved_response_amplitudes = []
        self.achieved_response_phases = []
        self.complex_drive_modifications = []
        self.achieved_excitation_signals_combined = []
        self.achieved_excitation_signals = []
        self.achieved_excitation_frequencies = []
        self.achieved_excitation_arguments = []
        self.achieved_excitation_amplitudes = []
        self.achieved_excitation_phases = []
        self.enable_control(False)
        self.shutdown_sent = False
        self.controller_communication_queue.put(
            self.log_name, (GlobalCommands.START_ENVIRONMENT, self.environment_name)
        )
        self.environment_command_queue.put(
            self.log_name,
            (
                SineCommands.START_CONTROL,
                (
                    db2scale(self.run_widget.test_level_selector.value()),
                    (
                        [
                            self.run_widget.partial_environment_tone_selector.row(item)
                            for item in self.run_widget.partial_environment_tone_selector.selectedItems()
                        ]
                        if self.run_widget.partial_environment_selector.isChecked()
                        else None
                    ),
                    (
                        self.run_widget.start_time_selector.value()
                        if self.run_widget.partial_environment_selector.isChecked()
                        else None
                    ),
                    (
                        self.run_widget.stop_time_selector.value()
                        if self.run_widget.partial_environment_selector.isChecked()
                        else None
                    ),
                ),
            ),
        )
        if self.run_widget.test_level_selector.value() >= 0:
            self.controller_communication_queue.put(
                self.log_name, (GlobalCommands.AT_TARGET_LEVEL, self.environment_name)
            )

    def enable_control(self, enabled):
        """Enables or disables the widgets to start or modify the control

        Parameters
        ----------
        enabled : bool
            If True, enables the widgets.  Otherwise, it disables the widgets
        """
        for widget in [
            self.run_widget.test_level_selector,
            self.run_widget.partial_environment_selector,
            self.run_widget.partial_environment_tone_selector,
            self.run_widget.start_time_selector,
            self.run_widget.stop_time_selector,
            self.run_widget.start_test_button,
        ]:
            widget.setEnabled(enabled)
        for widget in [self.run_widget.stop_test_button]:
            widget.setEnabled(not enabled)
        if enabled:
            self.enable_disable_partial_environment()

    def stop_control(self):
        """Sends a signal to shut down the control"""
        self.shutdown_sent = True
        self.environment_command_queue.put(self.log_name, (SineCommands.STOP_CONTROL, None))

    def change_test_level_from_profile(self, test_level):
        """Changes the value of the test level from a profile.

        Parameters
        ----------
        test_level : int
            The value in decibels to set the test level to
        """
        self.run_widget.test_level_selector.setValue(int(test_level))

    def create_window(
        self, event, tone_index=None, channel_index=None
    ):  # pylint: disable=unused-argument
        """Creates a window with the specified tone and channel index.

        Parameters
        ----------
        event : event
            The button clicked event that triggered this callback.  Not used.
        tone_index : int, optional
            The tone index to visualize.  By default, it will be the one
            currently selected in the UI
        channel_index : int, optional
            The channel index to visualize.  By default, it will be the one
            currently selected in the UI
        """
        if tone_index is None:
            tone_index = self.run_widget.control_tone_selector.currentIndex()
        if channel_index is None:
            channel_index = self.run_widget.control_channel_selector.currentIndex()
        self.plot_windows.append(PlotSineWindow(None, self, tone_index, channel_index))

    def show_all_channels(self):
        """Creates a window for all pairs of tone and channel"""
        for i in range(self.run_widget.control_tone_selector.count()):
            for j in range(self.run_widget.control_channel_selector.count()):
                self.create_window(None, i, j)
        self.tile_windows()

    def tile_windows(self):
        """Tiles the plot windows across the monitor"""
        screen_rect = QtWidgets.QApplication.desktop().screenGeometry()
        # Go through and remove any closed windows
        self.plot_windows = [window for window in self.plot_windows if window.isVisible()]
        num_windows = len(self.plot_windows)
        if num_windows == 0:
            return
        ncols = int(np.ceil(np.sqrt(num_windows)))
        nrows = int(np.ceil(num_windows / ncols))
        window_width = int(screen_rect.width() / ncols)
        window_height = int(screen_rect.height() / nrows)
        for index, window in enumerate(self.plot_windows):
            window.resize(window_width, window_height)
            row_ind = index // ncols
            col_ind = index % ncols
            window.move(col_ind * window_width, row_ind * window_height)

    def close_windows(self):
        """Closes all the plot windows"""
        for window in self.plot_windows:
            window.close()

    def save_control_data_from_profile(self, filename):
        """Saves the current control data to the specified file name"""
        self.save_control_data(None, filename)

    def save_control_data(self, clicked, filename=None):  # pylint: disable=unused-argument
        """Saves the control data to the specified filename, or via a dialog box"""
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self.definition_widget,
                "Select File to Save Spectral Data",
                filter="NumPy File (*.npz)",
            )
        if filename == "":
            return
        self.environment_command_queue.put(
            self.log_name, (SineCommands.SAVE_CONTROL_DATA, filename)
        )

    def update_control_run_plot(self, tone_index=None, channel_index=None):
        """Updates the drive plot showing the modifications to the control

        Parameters
        ----------
        tone_index : int, optional
            The tone index to display.  By default, the currently selected tone is displayed.
        channel_index : int, optional
            The channel index to display.  By default, the currently selected channel is displayed.
        """
        if self.complex_drive_modifications is None:
            return
        if channel_index is None:
            channel_index = self.run_widget.control_updates_signal_selector.currentColumn()
        if tone_index is None:
            tone_index = self.run_widget.control_updates_signal_selector.currentRow()
        response_over_time = [
            cuh[tone_index, channel_index] for cuh in self.complex_drive_modifications[::-1]
        ]
        # print(response_over_time)
        for rot, marker in zip(response_over_time, self.plot_data_items["control_updates"][::-1]):
            marker.setData([np.real(rot)], [np.imag(rot)])
        for tone_index in range(self.run_widget.control_updates_signal_selector.rowCount()):
            for channel_index in range(
                self.run_widget.control_updates_signal_selector.columnCount()
            ):
                item = self.run_widget.control_updates_signal_selector.item(
                    tone_index, channel_index
                )
                item.setText(
                    f"{np.abs(self.complex_drive_modifications[-1][tone_index, channel_index]):0.3g}"
                )

    def update_run_plot(
        self,
        tone_index=None,
        channel_index=None,
        previous_tone=None,  # pylint: disable=unused-argument
        previous_channel=None,  # pylint: disable=unused-argument
        update_spec=True,
    ):
        """Updates the control plots showing amplitude and phase

        Parameters
        ----------
        tone_index : int, optional
            The tone index to display.  By default, the currently selected tone is displayed.
        channel_index : int, optional
            The channel index to display.  By default, the currently selected channel is displayed.
        previous_tone : int, optional
            Not used but required by callback signature
        previous_channel : int, optional
            Not used but required by callback signature
        update_spec : bool, optional
            If True, the specification will also be updated, by default True.
        """
        if channel_index is None:
            channel_index = self.run_widget.signal_selector.currentColumn()
        if tone_index is None:
            tone_index = self.run_widget.signal_selector.currentRow()
        if update_spec:
            spec_frequency = self.specification_frequencies[tone_index]
            spec_amplitude = self.specification_amplitudes[tone_index, channel_index]
            self.plot_data_items["control_amplitude"][1].setData(spec_frequency, spec_amplitude)
            spec_phase = self.specification_phases[tone_index, channel_index]
            self.plot_data_items["control_phase"][1].setData(spec_frequency, spec_phase)
            # Get the warning and abort limits
            spec = self.environment_parameters.specifications[tone_index]
            self.plot_data_items["control_warning_lower"].setData(
                np.repeat(spec.breakpoint_table["frequency"], 2),
                spec.breakpoint_table["warning"][:, 0, :, channel_index].flatten(),
            )
            self.plot_data_items["control_warning_upper"].setData(
                np.repeat(spec.breakpoint_table["frequency"], 2),
                spec.breakpoint_table["warning"][:, 1, :, channel_index].flatten(),
            )
            self.plot_data_items["control_abort_lower"].setData(
                np.repeat(spec.breakpoint_table["frequency"], 2),
                spec.breakpoint_table["abort"][:, 0, :, channel_index].flatten(),
            )
            self.plot_data_items["control_abort_upper"].setData(
                np.repeat(spec.breakpoint_table["frequency"], 2),
                spec.breakpoint_table["abort"][:, 1, :, channel_index].flatten(),
            )
        if self.achieved_excitation_frequencies is not None:
            achieved_frequency = np.concatenate(
                [fh[tone_index] for fh in self.achieved_excitation_frequencies]
            )
            if self.achieved_response_amplitudes is not None:
                achieved_amplitude = np.concatenate(
                    [ah[tone_index, channel_index] for ah in self.achieved_response_amplitudes]
                )
                self.plot_data_items["control_amplitude"][0].setData(
                    achieved_frequency, achieved_amplitude
                )
            if self.achieved_response_phases is not None:
                achieved_phase = np.concatenate(
                    [ph[tone_index, channel_index] for ph in self.achieved_response_phases]
                )
                self.plot_data_items["control_phase"][0].setData(achieved_frequency, achieved_phase)
        # Go through and remove any closed windows
        self.plot_windows = [window for window in self.plot_windows if window.isVisible()]
        for window in self.plot_windows:
            window.update_plot()

    def update_control_error_table(self, last_errors, last_warning_flags, last_abort_flags):
        """Updates the values in the control table, including color changes

        Parameters
        ----------
        last_errors : ndarray
            An array of shape (tone,channel) containing the amplitude errors
        last_warning_flags : ndarray
            An array of shape (tone,channel) booleans denoting if a warning has been hit
        last_abort_flags : ndarray
            An array of shape (tone,channel) booleans denoting if an abort has been hit
        """
        for i in range(last_errors.shape[0]):
            for j in range(last_errors.shape[1]):
                item = self.run_widget.signal_selector.item(i, j)
                if last_abort_flags[i, j]:
                    item.setBackground(QColor(255, 125, 125))
                elif last_warning_flags[i, j]:
                    item.setBackground(QColor(255, 255, 125))
                else:
                    item.setBackground(QColor(255, 255, 255))
                item.setText(f"{last_errors[i, j]:0.3f}")
        if (
            np.any(last_abort_flags)
            and self.environment_parameters.allow_automatic_aborts
            and not self.shutdown_sent
        ):
            self.log("Sending Abort Signal!")
            self.stop_control()

    def enable_disable_partial_environment(self):
        """Enables or disables the partial environment widgets"""
        for widget in [
            self.run_widget.start_time_selector,
            self.run_widget.stop_time_selector,
            self.run_widget.partial_environment_tone_selector,
        ]:
            widget.setEnabled(self.run_widget.partial_environment_selector.isChecked())

    # %% Misc

    def retrieve_metadata(
        self,
        netcdf_handle: nc4._netCDF4.Dataset,  # pylint: disable=c-extension-no-member
        environment_name: str = None,
    ) -> nc4._netCDF4.Group:  # pylint: disable=c-extension-no-member
        """Retrieves metadata from a netcdf file and sets the UI appropriately."""
        # Get all the system identification information
        super().retrieve_metadata(netcdf_handle, environment_name)
        # Get the group
        group = netcdf_handle.groups[self.environment_name]
        self.definition_widget.ramp_time_spinbox.setValue(group.ramp_time)
        self.definition_widget.buffer_blocks_selector.setValue(group.buffer_blocks)
        self.definition_widget.control_convergence_selector.setValue(group.control_convergence)
        self.definition_widget.update_drives_after_environment_selector.setChecked(
            bool(group.update_drives_after_environment)
        )
        self.definition_widget.best_fit_phase_checkbox.setChecked(bool(group.phase_fit))
        self.definition_widget.auto_abort_checkbox.setChecked(bool(group.allow_automatic_aborts))
        self.definition_widget.filter_type_selector.setCurrentIndex(group.tracking_filter_type)
        self.definition_widget.tracking_filter_cutoff_selector.setValue(
            group.tracking_filter_cutoff * 100
        )
        self.definition_widget.tracking_filter_order_selector.setValue(group.tracking_filter_order)
        self.definition_widget.vk_filter_order_selector.setCurrentIndex(group.vk_filter_order - 1)
        self.definition_widget.vk_filter_bandwidth_selector.setValue(group.vk_filter_bandwidth)
        self.definition_widget.vk_filter_block_size_selector.setValue(group.vk_filter_blocksize)
        self.definition_widget.vk_filter_block_overlap_selector.setValue(group.vk_filter_overlap)
        if group.control_python_script != "":
            self.select_python_module(None, group.control_python_script)
            self.definition_widget.control_function_input.setCurrentIndex(
                self.definition_widget.control_function_input.findText(group.control_python_class)
            )
            self.definition_widget.control_parameters_text_input.setText(
                group.control_python_function_parameters
            )
        # Control channels
        for i in group.variables["control_channel_indices"][...]:
            item = self.definition_widget.control_channels_selector.item(i)
            item.setCheckState(Qt.Checked)
        # Transformation matrices
        try:
            self.response_transformation_matrix = group.variables["response_transformation_matrix"][
                ...
            ].data
        except KeyError:
            self.response_transformation_matrix = None
        try:
            self.output_transformation_matrix = group.variables["output_transformation_matrix"][
                ...
            ].data
        except KeyError:
            self.output_transformation_matrix = None
        self.define_transformation_matrices(None, dialog=False)
        # Specifications
        self.clear_and_update_specification_table()
        for index, (spec_name, spec_group) in enumerate(group["specifications"].groups.items()):
            if index > 0:
                self.add_sine_table_tab()
            frequency = spec_group["spec_frequency"][...]
            amplitude = spec_group["spec_amplitude"][...].transpose(1, 0)
            phase = spec_group["spec_phase"][...].transpose(1, 0)
            sweep_type = spec_group["spec_sweep_type"][...]
            sweep_rate = spec_group["spec_sweep_rate"][...].copy()
            sweep_rate[sweep_type == 1] = sweep_rate[sweep_type == 1] / 60
            sweep_type = ["lin" if val == 0 else "log" for val in sweep_type]
            warning = spec_group["spec_warning"][...].transpose(1, 2, 3, 0)
            abort = spec_group["spec_abort"][...].transpose(1, 2, 3, 0)
            start_time = spec_group.start_time
            self.sine_tables[-1].clear_and_update_specification_table(
                frequency,
                amplitude,
                phase,
                sweep_type,
                sweep_rate,
                warning,
                abort,
                start_time,
                spec_name,
            )

    def update_gui(self, queue_data):
        if super().update_gui(queue_data):
            return
        message, data = queue_data
        if message == SineUICommands.REQUEST_PREDICTION_PLOT_CHOICES:
            self.log("Sending Prediction Plot Choices...")
            self.send_response_prediction_plot_choices()
            self.send_excitation_prediction_plot_choices()
        elif message == SineUICommands.EXCITATION_PREDICTION:
            self.plot_excitation_prediction(*data)
        elif message == SineUICommands.RESPONSE_PREDICTION:
            self.plot_response_prediction(*data)
        elif message == SineUICommands.RESPONSE_ERROR_MATRIX:
            self.update_response_matrix(*data)
        elif message == SineUICommands.EXCITATION_VOLTAGE_LIST:
            self.update_voltage_list(data)
        elif message == SineUICommands.SPECIFICATION_FOR_PLOTTING:
            (
                self.specification_signals_combined,
                self.specification_signals,
                self.specification_frequencies,
                self.specification_arguments,
                self.specification_amplitudes,
                self.specification_phases,
                self.plot_downsample,
            ) = data
            self.log(f"Plot Downsample: {self.plot_downsample}")
            self.update_run_plot(update_spec=True)
        elif message == SineUICommands.TIME_DATA:
            (last_excitation, last_control) = data
            self.achieved_excitation_signals_combined.append(last_excitation)
            self.achieved_response_signals_combined.append(last_control)
        elif message == SineUICommands.CONTROL_DATA:
            (
                last_signals,
                last_amplitudes,
                last_phases,
                last_frequencies,
                last_correction,
                last_errors,
                last_warning_flags,
                last_abort_flags,
            ) = data
            self.achieved_response_amplitudes.append(last_amplitudes)
            self.achieved_response_phases.append(last_phases)
            self.complex_drive_modifications.append(last_correction)
            self.achieved_excitation_frequencies.append(last_frequencies)
            self.achieved_excitation_signals.append(last_signals)
            self.update_control_run_plot()
            self.update_run_plot(update_spec=False)
            self.update_control_error_table(last_errors, last_warning_flags, last_abort_flags)
        elif message == SineUICommands.ENABLE_CONTROL:
            self.enable_control(True)
        else:
            widget = None
            for parent in [
                self.definition_widget,
                self.run_widget,
                self.system_id_widget,
                self.prediction_widget,
            ]:
                try:
                    widget = getattr(parent, message)
                    break
                except AttributeError:
                    continue
            if widget is None:
                raise ValueError(f"Cannot Update Widget {message}: not found in UI")
            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.setValue(data)
            elif isinstance(widget, QtWidgets.QSpinBox):
                widget.setValue(data)
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(data)
            elif isinstance(widget, QtWidgets.QListWidget):
                widget.clear()
                widget.addItems([f"{d:.3f}" for d in data])

    def set_parameters_from_template(self, worksheet):
        self.definition_widget.ramp_time_spinbox.setValue(float(worksheet.cell(2, 2).value))
        self.definition_widget.control_convergence_selector.setValue(
            float(worksheet.cell(3, 2).value)
        )
        self.definition_widget.update_drives_after_environment_selector.setChecked(
            worksheet.cell(4, 2).value.upper() == "Y"
        )
        self.definition_widget.best_fit_phase_checkbox.setChecked(
            worksheet.cell(5, 2).value.upper() == "Y"
        )
        self.definition_widget.auto_abort_checkbox.setChecked(
            worksheet.cell(6, 2).value.upper() == "Y"
        )
        self.definition_widget.buffer_blocks_selector.setValue(int(worksheet.cell(7, 2).value))
        self.definition_widget.filter_type_selector.setCurrentIndex(
            1 if worksheet.cell(8, 2).value.upper() == "VK" else 0
        )
        self.definition_widget.tracking_filter_cutoff_selector.setValue(
            float(worksheet.cell(9, 2).value)
        )
        self.definition_widget.tracking_filter_order_selector.setValue(
            int(worksheet.cell(10, 2).value)
        )
        self.definition_widget.vk_filter_order_selector.setCurrentIndex(
            int(worksheet.cell(11, 2).value) - 1
        )
        self.definition_widget.vk_filter_bandwidth_selector.setValue(
            float(worksheet.cell(12, 2).value)
        )
        self.definition_widget.vk_filter_block_size_selector.setValue(
            int(worksheet.cell(13, 2).value)
        )
        self.definition_widget.vk_filter_block_overlap_selector.setValue(
            float(worksheet.cell(14, 2).value)
        )
        if worksheet.cell(15, 2).value is not None and worksheet.cell(15, 2).value != "":
            self.select_python_module(None, worksheet.cell(15, 2).value)
            self.definition_widget.python_class_input.setCurrentIndex(
                self.definition_widget.python_class_input.findText(worksheet.cell(16, 2).value)
            )
        self.definition_widget.control_parameters_text_input.setText(
            "" if worksheet.cell(17, 2).value is None else str(worksheet.cell(17, 2).value)
        )
        column_index = 2
        while True:
            value = worksheet.cell(18, column_index).value
            if value is None or (isinstance(value, str) and value.strip() == ""):
                break
            item = self.definition_widget.control_channels_selector.item(int(value) - 1)
            item.setCheckState(Qt.Checked)
            column_index += 1
        self.system_id_widget.samplesPerFrameSpinBox.setValue(int(worksheet.cell(19, 2).value))
        self.system_id_widget.averagingTypeComboBox.setCurrentIndex(
            self.system_id_widget.averagingTypeComboBox.findText(worksheet.cell(20, 2).value)
        )
        self.system_id_widget.noiseAveragesSpinBox.setValue(int(worksheet.cell(21, 2).value))
        self.system_id_widget.systemIDAveragesSpinBox.setValue(int(worksheet.cell(22, 2).value))
        self.system_id_widget.averagingCoefficientDoubleSpinBox.setValue(
            float(worksheet.cell(23, 2).value)
        )
        self.system_id_widget.estimatorComboBox.setCurrentIndex(
            self.system_id_widget.estimatorComboBox.findText(worksheet.cell(24, 2).value)
        )
        self.system_id_widget.levelDoubleSpinBox.setValue(float(worksheet.cell(25, 2).value))
        self.system_id_widget.levelRampTimeDoubleSpinBox.setValue(
            float(worksheet.cell(26, 2).value)
        )
        self.system_id_widget.signalTypeComboBox.setCurrentIndex(
            self.system_id_widget.signalTypeComboBox.findText(worksheet.cell(27, 2).value)
        )
        self.system_id_widget.windowComboBox.setCurrentIndex(
            self.system_id_widget.windowComboBox.findText(worksheet.cell(28, 2).value)
        )
        self.system_id_widget.overlapDoubleSpinBox.setValue(float(worksheet.cell(29, 2).value))
        self.system_id_widget.onFractionDoubleSpinBox.setValue(float(worksheet.cell(30, 2).value))
        self.system_id_widget.pretriggerDoubleSpinBox.setValue(float(worksheet.cell(31, 2).value))
        self.system_id_widget.rampFractionDoubleSpinBox.setValue(float(worksheet.cell(32, 2).value))

        # Now we need to find the transformation matrices' sizes
        response_channels = self.definition_widget.control_channels_display.value()
        output_channels = self.definition_widget.output_channels_display.value()
        output_transform_row = 35
        if (
            isinstance(worksheet.cell(34, 2).value, str)
            and worksheet.cell(34, 2).value.lower() == "none"
        ):
            self.response_transformation_matrix = None
        else:
            while True:
                if worksheet.cell(output_transform_row, 1).value == "Output Transformation Matrix:":
                    break
                output_transform_row += 1
            response_size = output_transform_row - 34
            response_transformation = []
            for i in range(response_size):
                response_transformation.append([])
                for j in range(response_channels):
                    response_transformation[-1].append(float(worksheet.cell(34 + i, 2 + j).value))
            self.response_transformation_matrix = np.array(response_transformation)
        if (
            isinstance(worksheet.cell(output_transform_row, 2).value, str)
            and worksheet.cell(output_transform_row, 2).value.lower() == "none"
        ):
            self.output_transformation_matrix = None
        else:
            output_transformation = []
            i = 0
            while True:
                if worksheet.cell(output_transform_row + i, 2).value is None or (
                    isinstance(worksheet.cell(output_transform_row + i, 2).value, str)
                    and worksheet.cell(output_transform_row + i, 2).value.strip() == ""
                ):
                    break
                output_transformation.append([])
                for j in range(output_channels):
                    output_transformation[-1].append(
                        float(worksheet.cell(output_transform_row + i, 2 + j).value)
                    )
                i += 1
            self.output_transformation_matrix = np.array(output_transformation)
        self.define_transformation_matrices(None, dialog=False)

        # Load in the specification
        if worksheet.cell(33, 2).value:
            self.sine_tables[0].load_specification(None, worksheet.cell(33, 2).value)
        column_index = 3
        while True:
            if worksheet.cell(33, column_index).value:
                self.add_sine_table_tab()
                self.sine_tables[-1].load_specification(
                    None, worksheet.cell(33, column_index).value
                )
                column_index += 1
            else:
                break

    @staticmethod
    def create_environment_template(environment_name, workbook):
        worksheet = workbook.create_sheet(environment_name)
        worksheet.cell(1, 1, "Control Type")
        worksheet.cell(1, 2, "Sine")
        worksheet.cell(
            1,
            4,
            "Note: Replace cells with hash marks (#) to provide the requested parameters.",
        )
        worksheet.cell(2, 1, "Test Ramp Time")
        worksheet.cell(2, 2, "# Time for the test to ramp up or down when starting or stopping")
        worksheet.cell(3, 1, "Control Convergence")
        worksheet.cell(
            3,
            2,
            "# A scale factor on the closed-loop update to "
            "balance stability with speed of convergence",
        )
        worksheet.cell(4, 1, "Update Drives after Environment:")
        worksheet.cell(
            4,
            2,
            "# If Y, then a control calculation will be performed after the "
            "environment finishes to update the next drive signal (Y/N)",
        )
        worksheet.cell(5, 1, "Fit Phases")
        worksheet.cell(
            5,
            2,
            "# If Y, perform a best fit to phase quantities to accommodate time delays (Y/N)",
        )
        worksheet.cell(6, 1, "Allow Automatic Aborts")
        worksheet.cell(
            6,
            2,
            "# Shut down the test automatically if an abort level is reached (Y/N)",
        )
        worksheet.cell(7, 1, "Buffer Blocks")
        worksheet.cell(
            7,
            2,
            "# Number of write blocks to keep in the buffer to "
            "guard against running out of samples to generate",
        )
        worksheet.cell(8, 1, "Tracking Filter Type")
        worksheet.cell(
            8,
            2,
            "# Select the tracking filter type to use "
            "(VK - Vold-Kalman / DFT - Digital Tracking Filter)",
        )
        worksheet.cell(9, 1, "Digital Tracking Filter Cutoff Percent:")
        worksheet.cell(
            9,
            2,
            "# Tracking filter cutoff frequency compared to the instantaneous frequency",
        )
        worksheet.cell(10, 1, "Digital Tracking Filter Order")
        worksheet.cell(10, 2, "# Order of the Butterworth filter used in the tracking filter")
        worksheet.cell(11, 1, "VK Filter Order")
        worksheet.cell(11, 2, "# Order of the Vold-Kalman Filter (1, 2, or 3)")
        worksheet.cell(12, 1, "VK Filter Bandwidth")
        worksheet.cell(12, 2, "# Bandwidth of the Vold-Kalman Filter")
        worksheet.cell(13, 1, "VK Filter Block Size")
        worksheet.cell(13, 2, "# Number of samples in the filter blocks for the Vold-Kalman Filter")
        worksheet.cell(14, 1, "VK Filter Overlap")
        worksheet.cell(14, 2, "Overlap between frames in the VK filter as a fraction (0.5, not 50)")
        worksheet.cell(15, 1, "Custom Control Python Script:")
        worksheet.cell(15, 2, "# Path to the Python script containing the control law")
        worksheet.cell(16, 1, "Custom Control Python Class:")
        worksheet.cell(
            16,
            2,
            "# Class name within the Python Script that will serve as the control law",
        )
        worksheet.cell(17, 1, "Control Parameters:")
        worksheet.cell(17, 2, "# Extra parameters used in the control law")
        worksheet.cell(18, 1, "Control Channels (1-based):")
        worksheet.cell(18, 2, "# List of channels, one per cell on this row")
        worksheet.cell(19, 1, "System ID Samples per Frame")
        worksheet.cell(
            19,
            2,
            "# Number of Samples per Measurement Frame in the System Identification",
        )
        worksheet.cell(20, 1, "System ID Averaging:")
        worksheet.cell(20, 2, "# Averaging Type, should be Linear or Exponential")
        worksheet.cell(21, 1, "Noise Averages:")
        worksheet.cell(21, 2, "# Number of Averages used when characterizing noise")
        worksheet.cell(22, 1, "System ID Averages:")
        worksheet.cell(22, 2, "# Number of Averages used when computing the FRF")
        worksheet.cell(23, 1, "Exponential Averaging Coefficient:")
        worksheet.cell(23, 2, "# Averaging Coefficient for Exponential Averaging (if used)")
        worksheet.cell(24, 1, "System ID Estimator:")
        worksheet.cell(
            24,
            2,
            "# Technique used to compute system ID.  Should be one of H1, H2, H3, or Hv.",
        )
        worksheet.cell(25, 1, "System ID Level (V RMS):")
        worksheet.cell(
            25,
            2,
            "# RMS Value of Flat Voltage Spectrum used for System Identification.",
        )
        worksheet.cell(26, 1, "System ID Ramp Time")
        worksheet.cell(
            26,
            2,
            "# Time for the system identification to ramp between levels or from start or to stop.",
        )
        worksheet.cell(27, 1, "System ID Signal Type:")
        worksheet.cell(27, 2, "# Signal to use for the system identification")
        worksheet.cell(28, 1, "System ID Window:")
        worksheet.cell(
            28,
            2,
            "# Window used to compute FRFs during system ID.  Should be one of Hann or None",
        )
        worksheet.cell(29, 1, "System ID Overlap %:")
        worksheet.cell(29, 2, "# Overlap to use in the system identification")
        worksheet.cell(30, 1, "System ID Burst On %:")
        worksheet.cell(30, 2, "# Percentage of a frame that the burst random is on for")
        worksheet.cell(31, 1, "System ID Burst Pretrigger %:")
        worksheet.cell(
            31,
            2,
            "# Percentage of a frame that occurs before the burst starts in a burst random signal",
        )
        worksheet.cell(32, 1, "System ID Ramp Fraction %:")
        worksheet.cell(
            32,
            2,
            '# Percentage of the "System ID Burst On %" that will be used to ramp up to full level',
        )
        worksheet.cell(33, 1, "Specification File:")
        worksheet.cell(33, 2, "# Path to the file containing the Specification")
        worksheet.cell(34, 1, "Response Transformation Matrix:")
        worksheet.cell(
            34,
            2,
            (
                "# Transformation matrix to apply to the response channels.  Type None if there is "
                "none.  Otherwise, make this a 2D array in the spreadsheet and move the Output "
                "Transformation Matrix line down so it will fit.  The number of columns should be "
                "the number of physical control channels."
            ),
        )
        worksheet.cell(35, 1, "Output Transformation Matrix:")
        worksheet.cell(
            35,
            2,
            "# Transformation matrix to apply to the outputs.  Type None if there is none.  "
            "Otherwise, make this a 2D array in the spreadsheet.  The number of columns should be "
            "the number of physical output channels in the environment.",
        )
