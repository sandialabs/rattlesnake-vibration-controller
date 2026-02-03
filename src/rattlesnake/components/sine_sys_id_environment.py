# -*- coding: utf-8 -*-
"""
This file defines a sine environment that utilizes system
identification.

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

import importlib
import inspect
import multiprocessing as mp
import os
import time
import traceback
from enum import Enum
from multiprocessing.queues import Queue

import netCDF4 as nc4
import numpy as np
import scipy.signal as sig
from qtpy import QtWidgets, uic
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor  # pylint: disable=no-name-in-module

from .abstract_sysid_environment import (
    AbstractSysIdEnvironment,
    AbstractSysIdMetadata,
    AbstractSysIdUI,
)
from .environments import (
    ControlTypes,
    environment_definition_ui_paths,
    environment_prediction_ui_paths,
    environment_run_ui_paths,
)
from .sine_sys_id_utilities import (
    DefaultSineControlLaw,
    FilterExplorer,
    PlotSineWindow,
    SineSpecification,
    SineSweepTable,
    digital_tracking_filter_generator,
    sine_sweep,
    vold_kalman_filter_generator,
)
from .ui_utilities import (
    TransformationMatrixWindow,
    VaryingNumberOfLinePlot,
    blended_scatter_plot,
    multiline_plotter,
)
from .utilities import (
    GlobalCommands,
    VerboseMessageQueue,
    db2scale,
    flush_queue,
    load_python_module,
    scale2db,
    wrap,
)

# %% Global Variables
CONTROL_TYPE = ControlTypes.SINE
MAXIMUM_NAME_LENGTH = 50
MAXIMUM_SAMPLES_TO_PLOT = 1000000

DEBUG = False

if DEBUG:
    from glob import glob

    FILE_OUTPUT = "debug_data/sine_control_{:}.npz"


# %% Commands
class SineCommands(Enum):
    """Enumeration containing sine commands"""

    START_CONTROL = 0
    STOP_CONTROL = 1
    SAVE_CONTROL_DATA = 2
    PERFORM_CONTROL_PREDICTION = 3
    SEND_EXCITATION_PREDICTION = 4
    SEND_RESPONSE_PREDICTION = 5


# %% Queues
class SineQueues:
    """A container class for the queues that this environment will manage."""

    def __init__(
        self,
        environment_name: str,
        environment_command_queue: VerboseMessageQueue,
        gui_update_queue: Queue,
        controller_communication_queue: VerboseMessageQueue,
        data_in_queue: Queue,
        data_out_queue: Queue,
        log_file_queue: VerboseMessageQueue,
    ):
        """A container class for the queues that sine vibration will manage.

        The environment uses many queues to pass data between the various
        pieces.  This class organizes those queues into one common namespace.

        Parameters
        ----------
        environment_name : str
            Name of the environment
        environment_command_queue : VerboseMessageQueue
            Queue that is read by the environment for environment commands
        gui_update_queue : mp.queues.Queue
            Queue where various subtasks put instructions for updating the
            widgets in the user interface
        controller_communication_queue : VerboseMessageQueue
            Queue that is read by the controller for global controller commands
        data_in_queue : mp.queues.Queue
            Multiprocessing queue that connects the acquisition subtask to the
            environment subtask.  Each environment will retrieve acquired data
            from this queue.
        data_out_queue : mp.queues.Queue
            Multiprocessing queue that connects the output subtask to the
            environment subtask.  Each environment will put data that it wants
            the controller to generate in this queue.
        log_file_queue : VerboseMessageQueue
            Queue for putting logging messages that will be read by the logging
            subtask and written to a file.
        """
        self.environment_command_queue = environment_command_queue
        self.gui_update_queue = gui_update_queue
        self.data_analysis_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Data Analysis Command Queue"
        )
        self.signal_generation_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Signal Generation Command Queue"
        )
        self.spectral_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Spectral Computation Command Queue"
        )
        self.collector_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Data Collector Command Queue"
        )
        self.controller_communication_queue = controller_communication_queue
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.data_for_spectral_computation_queue = mp.Queue()
        self.updated_spectral_quantities_queue = mp.Queue()
        self.time_history_to_generate_queue = mp.Queue()
        self.log_file_queue = log_file_queue


# %% Metadata
class SineMetadata(AbstractSysIdMetadata):
    """Metadata describing the Sine environment"""

    def __init__(
        self,
        *,
        sample_rate,
        samples_per_frame,
        number_of_channels,
        specifications,
        ramp_time,
        buffer_blocks,
        control_convergence,
        update_drives_after_environment,
        phase_fit,
        allow_automatic_aborts,
        tracking_filter_type,
        tracking_filter_cutoff,
        tracking_filter_order,
        vk_filter_order,
        vk_filter_bandwidth,
        vk_filter_blocksize,
        vk_filter_overlap,
        control_python_script,
        control_python_class,
        control_python_parameters,
        control_channel_indices,
        output_channel_indices,
        response_transformation_matrix,
        output_transformation_matrix,
    ):
        """Creates a Metadata object defining a Sine environment

        Parameters
        ----------
        sample_rate : float
            The sample rate in Hz
        samples_per_frame : int
            The number of samples per acquisition block
        number_of_channels : int
            The number of channels in the environment
        specifications : array of SineSpecification
            One SineSpecification object for each tone in the environment
        ramp_time : float
            The time to ramp to the initial level and ramp down from the final level
        buffer_blocks : int
            The number of blocks of data to use in various buffered activities,
            such as filtering and overlapping and adding data
        control_convergence : float
            A value between 0 and 1 that controls the proportional correction
            factor in the on-line control.
        update_drives_after_environment : bool
            If True, the control class will call the finalize_control method to
            update the preshaped drives based on the results of the previous test
        phase_fit : bool
            If True, the achieved phases will be best-fit to the specification
            phase.  This should handle time delays between acquisition and output
        allow_automatic_aborts : bool
            If True, the test will shut down if an abort level is reached
        tracking_filter_type : int
            0 for digital tracking filter, 1 for vold kalman filter
        tracking_filter_cutoff : float
            Filter cutoff for the digital tracking filter.
        tracking_filter_order : int
            Filter order for the digital tracking filter.
        vk_filter_order : int
            Order of the Vold-Kalman filter.
        vk_filter_bandwidth : float
            Bandwidth of the Vold-Kalman filter
        vk_filter_blocksize : int
            Number of samples in each Vold-Kalman filter segment
        vk_filter_overlap : float
            Overlap fraction between 0 and 0.5 for the Vold-Kalman filter
        control_python_script : str
            Path to a Python script containing an alternative Sine control law class
        control_python_class : str
            Name of the sine control law class
        control_python_parameters : str
            Extra parameters passed to the sine control law
        control_channel_indices : array of int
            Indices into the channels specifying the channels used as control channels
        output_channel_indices : array of int
            Indices into the channels specifying the channels used as drive channels
        response_transformation_matrix : ndarray
            A 2D np.ndarray consisting of a transformation matrix applied to the
            control channels
        output_transformation_matrix : _type_
            A 2D np.ndarray consisting of a transformation matrix applied to the
            drive channels
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.samples_per_frame = samples_per_frame
        self.number_of_channels = number_of_channels
        self.specifications = specifications
        self.ramp_time = ramp_time
        self.buffer_blocks = buffer_blocks
        self.control_convergence = control_convergence
        self.update_drives_after_environment = update_drives_after_environment
        self.phase_fit = phase_fit
        self.allow_automatic_aborts = allow_automatic_aborts
        self.tracking_filter_type = tracking_filter_type
        self.tracking_filter_cutoff = tracking_filter_cutoff
        self.tracking_filter_order = tracking_filter_order
        self.vk_filter_order = vk_filter_order
        self.vk_filter_bandwidth = vk_filter_bandwidth
        self.vk_filter_blocksize = vk_filter_blocksize
        self.vk_filter_overlap = vk_filter_overlap
        self.control_python_script = control_python_script
        self.control_python_class = control_python_class
        self.control_python_parameters = control_python_parameters
        self.control_channel_indices = control_channel_indices
        self.output_channel_indices = output_channel_indices
        self.response_transformation_matrix = response_transformation_matrix
        self.reference_transformation_matrix = output_transformation_matrix

    @property
    def sample_rate(self):
        """Sample rate of the data acquisition system"""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        """Sets the stored sample rate parameter"""
        self._sample_rate = value

    @property
    def ramp_samples(self):
        """Number of samples in the ramp time"""
        return int(self.ramp_time * self.sample_rate)

    @property
    def number_of_channels(self):
        """Total number of channels in the environment"""
        return self._number_of_channels

    @number_of_channels.setter
    def number_of_channels(self, value):
        """Sets the stored value of the number of channels in the environment"""
        self._number_of_channels = value

    @property
    def reference_channel_indices(self):
        """Indices corresponding to the drive channels"""
        return self.output_channel_indices

    @property
    def response_channel_indices(self):
        """Indices corresponding to the control channels"""
        return self.control_channel_indices

    @property
    def response_transformation_matrix(self):
        """Transformation matrix applied to the control channels"""
        return self._response_transformation_matrix

    @response_transformation_matrix.setter
    def response_transformation_matrix(self, value):
        """Sets the transformation matrix applied to the control channels"""
        self._response_transformation_matrix = value

    @property
    def reference_transformation_matrix(self):
        """Transformation matrix applied to the drive channels"""
        return self._reference_transformation_matrix

    @reference_transformation_matrix.setter
    def reference_transformation_matrix(self, value):
        """Sets the transformation matrix applied to the drive channels"""
        self._reference_transformation_matrix = value

    def store_to_netcdf(
        self,
        netcdf_group_handle: nc4._netCDF4.Group,  # pylint: disable=c-extension-no-member
    ):
        """Store parameters to a group in a netCDF streaming file.

        This function stores parameters from the environment into the netCDF
        file in a group with the environment's name as its name.  The function
        will receive a reference to the group within the dataset and should
        store the environment's parameters into that group in the form of
        attributes, dimensions, or variables.

        This function is the "write" counterpart to the retrieve_metadata
        function in the AbstractUI class, which will read parameters from
        the netCDF file to populate the parameters in the user interface.

        Parameters
        ----------
        netcdf_group_handle : nc4._netCDF4.Group
            A reference to the Group within the netCDF dataset where the
            environment's metadata is stored.

        """
        super().store_to_netcdf(netcdf_group_handle)

        netcdf_group_handle.sample_rate = self.sample_rate
        netcdf_group_handle.samples_per_frame = self.samples_per_frame
        netcdf_group_handle.ramp_time = self.ramp_time

        netcdf_group_handle.number_of_channels = self.number_of_channels
        netcdf_group_handle.update_drives_after_environment = (
            1 if self.update_drives_after_environment else 0
        )
        netcdf_group_handle.phase_fit = 1 if self.phase_fit else 0
        netcdf_group_handle.control_convergence = self.control_convergence
        netcdf_group_handle.allow_automatic_aborts = 1 if self.allow_automatic_aborts else 0
        netcdf_group_handle.control_python_script = (
            "" if self.control_python_script is None else self.control_python_script
        )
        netcdf_group_handle.control_python_class = (
            "" if self.control_python_script is None else self.control_python_class
        )
        netcdf_group_handle.control_python_parameters = (
            "" if self.control_python_script is None else self.control_python_parameters
        )
        netcdf_group_handle.tracking_filter_type = self.tracking_filter_type
        netcdf_group_handle.tracking_filter_cutoff = self.tracking_filter_cutoff
        netcdf_group_handle.tracking_filter_order = self.tracking_filter_order
        netcdf_group_handle.vk_filter_order = self.vk_filter_order
        netcdf_group_handle.vk_filter_bandwidth = self.vk_filter_bandwidth
        netcdf_group_handle.vk_filter_blocksize = self.vk_filter_blocksize
        netcdf_group_handle.vk_filter_overlap = self.vk_filter_overlap
        netcdf_group_handle.buffer_blocks = self.buffer_blocks

        # Control channels
        netcdf_group_handle.createDimension("control_channels", len(self.control_channel_indices))
        var = netcdf_group_handle.createVariable(
            "control_channel_indices", "i4", ("control_channels")
        )
        var[...] = self.control_channel_indices
        # Transformation matrices
        if self.response_transformation_matrix is not None:
            netcdf_group_handle.createDimension(
                "response_transformation_rows",
                self.response_transformation_matrix.shape[0],
            )
            netcdf_group_handle.createDimension(
                "response_transformation_cols",
                self.response_transformation_matrix.shape[1],
            )
            var = netcdf_group_handle.createVariable(
                "response_transformation_matrix",
                "f8",
                ("response_transformation_rows", "response_transformation_cols"),
            )
            var[...] = self.response_transformation_matrix
        if self.reference_transformation_matrix is not None:
            netcdf_group_handle.createDimension(
                "reference_transformation_rows",
                self.reference_transformation_matrix.shape[0],
            )
            netcdf_group_handle.createDimension(
                "reference_transformation_cols",
                self.reference_transformation_matrix.shape[1],
            )
            var = netcdf_group_handle.createVariable(
                "reference_transformation_matrix",
                "f8",
                ("reference_transformation_rows", "reference_transformation_cols"),
            )
            var[...] = self.reference_transformation_matrix
        # Specification
        spec_group = netcdf_group_handle.createGroup("specifications")
        for specification in self.specifications:
            specification: SineSpecification
            grp = spec_group.createGroup(specification.name)
            grp.start_time = specification.start_time
            grp.createDimension("num_breakpoints", len(specification.breakpoint_table))
            grp.createDimension(
                "specification_channels",
                specification.breakpoint_table["amplitude"].shape[-1],
            )
            grp.createDimension("two", 2)
            var = grp.createVariable("spec_frequency", "f8", ("num_breakpoints"))
            var[...] = specification.breakpoint_table["frequency"]
            var = grp.createVariable(
                "spec_amplitude", "f8", ("num_breakpoints", "specification_channels")
            )
            var[...] = specification.breakpoint_table["amplitude"]
            var = grp.createVariable(
                "spec_phase", "f8", ("num_breakpoints", "specification_channels")
            )
            var[...] = specification.breakpoint_table["phase"]
            var = grp.createVariable("spec_sweep_type", "i1", ("num_breakpoints"))
            var[...] = specification.breakpoint_table["sweep_type"]
            var = grp.createVariable("spec_sweep_rate", "f8", ("num_breakpoints"))
            var[...] = specification.breakpoint_table["sweep_rate"]
            var = grp.createVariable(
                "spec_warning",
                "f8",
                ("num_breakpoints", "two", "two", "specification_channels"),
            )
            var[...] = specification.breakpoint_table["warning"]
            var = grp.createVariable(
                "spec_abort",
                "f8",
                ("num_breakpoints", "two", "two", "specification_channels"),
            )
            var[...] = specification.breakpoint_table["abort"]


# %% Additional Imports
# These need to be here to avoid circular imports
from .abstract_sysid_data_analysis import (  # pylint: disable=wrong-import-position # noqa: E402
    sysid_data_analysis_process,
)
from .data_collector import (  # pylint: disable=wrong-import-position # noqa: E402
    data_collector_process,
)
from .signal_generation import (  # pylint: disable=wrong-import-position # noqa: E402
    ContinuousTransientSignalGenerator,
)
from .signal_generation_process import (  # pylint: disable=wrong-import-position # noqa: E402
    SignalGenerationCommands,
    SignalGenerationMetadata,
    signal_generation_process,
)
from .spectral_processing import (  # pylint: disable=wrong-import-position # noqa: E402
    spectral_processing_process,
)


# %% UI
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
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
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
        if message == "request_prediction_plot_choices":
            self.log("Sending Prediction Plot Choices...")
            self.send_response_prediction_plot_choices()
            self.send_excitation_prediction_plot_choices()
        elif message == "excitation_prediction":
            self.plot_excitation_prediction(*data)
        elif message == "response_prediction":
            self.plot_response_prediction(*data)
        elif message == "response_error_matrix":
            self.update_response_matrix(*data)
        elif message == "excitation_voltage_list":
            self.update_voltage_list(data)
        elif message == "specification_for_plotting":
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
        elif message == "time_data":
            (last_excitation, last_control) = data
            self.achieved_excitation_signals_combined.append(last_excitation)
            self.achieved_response_signals_combined.append(last_control)
        elif message == "control_data":
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
        elif message == "enable_control":
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


# %% Environment


class SineEnvironment(AbstractSysIdEnvironment):
    """Class representing the environment computations on a separate process from the main UI"""

    def __init__(
        self,
        environment_name: str,
        queue_container: SineQueues,
        acquisition_active,
        output_active,
    ):
        """Initializes the sine environment computation class

        Parameters
        ----------
        environment_name : str
            Name of the environment
        queue_container : SineQueues
            A container containing all the queues that the Sine environment needs to
            pass information between its parts
        acquisition_active : int
            A multiprocessing shared value that is used to tell all processes whether or not
            the acquisition is currently running
        output_active : int
            A multiprocessing shared value that is used to tell all processes whether or not
            the output is currently running
        """
        super().__init__(
            environment_name,
            queue_container.environment_command_queue,
            queue_container.gui_update_queue,
            queue_container.controller_communication_queue,
            queue_container.log_file_queue,
            queue_container.collector_command_queue,
            queue_container.signal_generation_command_queue,
            queue_container.spectral_command_queue,
            queue_container.data_analysis_command_queue,
            queue_container.data_in_queue,
            queue_container.data_out_queue,
            acquisition_active,
            output_active,
        )
        self.map_command(SineCommands.PERFORM_CONTROL_PREDICTION, self.perform_control_prediction)
        self.map_command(SineCommands.START_CONTROL, self.start_control)
        self.map_command(SineCommands.STOP_CONTROL, self.stop_environment)
        self.map_command(SineCommands.SAVE_CONTROL_DATA, self.save_control_data)
        self.map_command(SineCommands.SEND_RESPONSE_PREDICTION, self.send_response_prediction)
        self.map_command(SineCommands.SEND_EXCITATION_PREDICTION, self.send_excitation_prediction)
        # Persistent data
        self.data_acquisition_parameters = None
        self.environment_parameters = None
        self.queue_container = queue_container
        self.plot_downsample = None
        # Control data
        self.sysid_frequencies = None
        self.sysid_frf = None
        self.sysid_coherence = None
        self.sysid_response_cpsd = None
        self.sysid_reference_cpsd = None
        self.sysid_condition = None
        self.sysid_response_noise = None
        self.sysid_reference_noise = None
        self.sysid_frames = None
        self.control_class = None
        self.extra_control_parameters = None
        # Specification data
        self.specification_signals_combined = None
        self.specification_signals = None
        self.specification_frequencies = None
        self.specification_arguments = None
        self.specification_amplitudes = None
        self.specification_phases = None
        self.specification_start_indices = None
        self.specification_end_indices = None
        self.ramp_samples = None
        # Excitation Signal Data
        self.excitation_signals = None
        self.excitation_signals_combined = None
        self.excitation_signal_frequencies = None
        self.excitation_signal_arguments = None
        self.excitation_signal_amplitudes = None
        self.excitation_signal_phases = None
        self.peak_voltages = None
        # Predicted Response Data
        self.predicted_response_signals_combined = None
        self.predicted_response_signals = None
        self.predicted_response_amplitudes = None
        self.predicted_response_phases = None
        self.predicted_warning_matrix = None
        self.predicted_abort_matrix = None
        self.predicted_amplitude_error = None
        # Running data
        self.control_test_level = 0
        self.control_tones = None
        self.control_tone_indices = None
        self.control_start_time = None
        self.control_end_time = None
        self.control_time_delay = None
        self.control_write_index = 0
        self.control_read_index = 0
        self.control_analysis_index = 0
        self.control_finished = False
        self.control_analysis_finished = False
        self.control_startup = True
        self.control_first_signal = None
        self.control_response_signals_combined = None
        self.control_response_amplitudes = None
        self.control_response_phases = None
        self.control_response_frequencies = None
        self.control_response_arguments = None
        self.control_target_phases = None
        self.control_target_amplitudes = None
        self.control_specification_arguments = None
        self.control_drive_modifications = None
        self.control_block_size = None
        self.control_filters = None
        self.control_warning_flags = None
        self.control_abort_flags = None
        self.control_amplitude_errors = None
        self.control_start_index = None
        self.control_end_index = None
        self.good_line_threshold = 0.25

    def initialize_environment_test_parameters(self, environment_parameters: SineMetadata):
        # Check if all specifications are equal
        if (
            self.environment_parameters is None
            or not np.array_equal(
                self.environment_parameters.control_channel_indices,
                environment_parameters.control_channel_indices,
            )
            or not (
                all(
                    [
                        spec1 == spec2
                        for spec1, spec2 in zip(
                            self.environment_parameters.specifications,
                            environment_parameters.specifications,
                        )
                    ]
                )
                and (
                    len(self.environment_parameters.specifications)
                    == len(environment_parameters.specifications)
                )
            )
        ):
            self.sysid_frequencies = None
            self.sysid_frf = None
            self.sysid_coherence = None
            self.sysid_response_cpsd = None
            self.sysid_reference_cpsd = None
            self.sysid_condition = None
            self.sysid_response_noise = None
            self.sysid_reference_noise = None
            self.sysid_frames = None
            self.control_class = None
            self.extra_control_parameters = None
            self.excitation_signals_combined = None
            self.excitation_signals = None
            self.excitation_signal_frequencies = None
            self.excitation_signal_arguments = None
            self.excitation_signal_amplitudes = None
            self.excitation_signal_phases = None
            self.predicted_response_signals_combined = None
            self.predicted_response_signals = None
            self.predicted_response_amplitudes = None
            self.predicted_response_phases = None
            self.ramp_samples = None
        super().initialize_environment_test_parameters(environment_parameters)
        self.environment_parameters: SineMetadata
        if environment_parameters.control_python_script is None:
            control_class = DefaultSineControlLaw
            self.extra_control_parameters = environment_parameters.control_python_parameters
        else:
            _, file = os.path.split(environment_parameters.control_python_script)
            file, _ = os.path.splitext(file)
            spec = importlib.util.spec_from_file_location(
                file, environment_parameters.control_python_script
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.extra_control_parameters = environment_parameters.control_python_parameters
            control_class = getattr(module, environment_parameters.control_python_class)
        self.control_class = control_class(
            self.data_acquisition_parameters.sample_rate,
            self.environment_parameters.specifications,
            self.data_acquisition_parameters.output_oversample,
            self.environment_parameters.ramp_time,
            self.environment_parameters.control_convergence,
            self.data_acquisition_parameters.samples_per_write,
            self.environment_parameters.buffer_blocks,
            self.extra_control_parameters,  # Required parameters
            self.environment_parameters.sysid_frequency_spacing,  # Frequency Spacing
            self.sysid_frf,  # Transfer Functions
            self.sysid_response_noise,  # Noise levels and correlation
            self.sysid_reference_noise,  # from the system identification
            self.sysid_response_cpsd,  # Response levels and correlation
            self.sysid_reference_cpsd,  # from the system identification
            self.sysid_coherence,  # Coherence from the system identification
            self.sysid_frames,  # Number of frames in the FRF matrices
        )
        self.log("Creating Specification Signals...")
        (
            self.specification_signals_combined,
            self.specification_signals,
            self.specification_frequencies,
            self.specification_arguments,
            self.specification_amplitudes,
            self.specification_phases,
            self.specification_start_indices,
            self.specification_end_indices,
        ) = SineSpecification.create_combined_signals(
            self.environment_parameters.specifications,
            self.data_acquisition_parameters.sample_rate
            * self.data_acquisition_parameters.output_oversample,
            self.environment_parameters.ramp_samples
            * self.data_acquisition_parameters.output_oversample,
        )
        self.ramp_samples = (
            self.environment_parameters.ramp_samples
            * self.data_acquisition_parameters.output_oversample
        )
        self.plot_downsample = (
            self.specification_signals_combined.shape[-1]
            // self.data_acquisition_parameters.output_oversample
            // MAXIMUM_SAMPLES_TO_PLOT
            + 1
        )
        self.gui_update_queue.put(
            (
                self.environment_name,
                (
                    "specification_for_plotting",
                    (
                        self.specification_signals_combined[
                            ...,
                            :: self.data_acquisition_parameters.output_oversample
                            * self.plot_downsample,
                        ],
                        self.specification_signals[
                            ...,
                            :: self.data_acquisition_parameters.output_oversample
                            * self.plot_downsample,
                        ],
                        self.specification_frequencies[
                            ...,
                            :: self.data_acquisition_parameters.output_oversample
                            * self.plot_downsample,
                        ],
                        self.specification_arguments[
                            ...,
                            :: self.data_acquisition_parameters.output_oversample
                            * self.plot_downsample,
                        ],
                        self.specification_amplitudes[
                            ...,
                            :: self.data_acquisition_parameters.output_oversample
                            * self.plot_downsample,
                        ],
                        wrap(
                            self.specification_phases[
                                ...,
                                :: self.data_acquisition_parameters.output_oversample
                                * self.plot_downsample,
                            ]
                        )
                        * 180
                        / np.pi,
                        self.plot_downsample,
                    ),
                ),
            )
        )
        self.log("Done!")

    def system_id_complete(self, data):
        # print('Finished System Identification')
        self.log("Finished System Identification")
        super().system_id_complete(data)
        (
            self.sysid_frames,
            _,
            self.sysid_frequencies,
            self.sysid_frf,
            self.sysid_coherence,
            self.sysid_response_cpsd,
            self.sysid_reference_cpsd,
            self.sysid_condition,
            self.sysid_response_noise,
            self.sysid_reference_noise,
        ) = data
        # Perform the control prediction
        self.perform_control_prediction(True)

    def filter_predicted_signal(self):
        """Extract amplitude and phase information from predicted signals"""
        # print('Filtering Predicted Signal')
        predicted_signals = []
        predicted_amplitudes = []
        predicted_phases = []
        arguments = self.excitation_signal_arguments[
            ..., :: self.data_acquisition_parameters.output_oversample
        ]
        frequencies = self.excitation_signal_frequencies[
            ..., :: self.data_acquisition_parameters.output_oversample
        ]
        for signal in self.predicted_response_signals_combined:
            if self.environment_parameters.tracking_filter_type == 0:
                block_size = self.data_acquisition_parameters.samples_per_read
                generator = [
                    digital_tracking_filter_generator(
                        dt=1 / self.environment_parameters.sample_rate,
                        cutoff_frequency_ratio=self.environment_parameters.tracking_filter_cutoff,
                        filter_order=self.environment_parameters.tracking_filter_order,
                    )
                    for tone in self.excitation_signals
                ]
                for gen in generator:
                    gen.send(None)
            else:
                block_size = self.environment_parameters.vk_filter_blocksize
                generator = vold_kalman_filter_generator(
                    sample_rate=self.environment_parameters.sample_rate,
                    num_orders=self.excitation_signals.shape[0],
                    block_size=block_size,
                    overlap=self.environment_parameters.vk_filter_overlap,
                    bandwidth=self.environment_parameters.vk_filter_bandwidth,
                    filter_order=self.environment_parameters.vk_filter_order,
                    buffer_size_factor=self.environment_parameters.buffer_blocks + 1,
                )
                generator.send(None)

            # print(f"{self.signal.shape=}")
            start_index = 0
            reconstructed_signals = []
            reconstructed_amplitudes = []
            reconstructed_phases = []

            last_data = False
            while not last_data:
                end_index = start_index + block_size
                block = signal[start_index:end_index]
                block_arguments = arguments[:, start_index:end_index]
                block_frequencies = frequencies[:, start_index:end_index]
                last_data = end_index >= signal.size
                # print(f"{block.shape=}, {block_arguments.shape=}, "
                # f"{block_frequencies.shape=}, {last_data=}, {block_size=}")
                if self.environment_parameters.tracking_filter_type == 0:
                    amps = []
                    phss = []
                    for arg, freq, gen in zip(block_arguments, block_frequencies, generator):
                        amp, phs = gen.send((block, freq, arg))
                        amps.append(amp)
                        phss.append(phs)
                    reconstructed_amplitudes.append(np.array(amps))
                    reconstructed_phases.append(np.array(phss))
                    reconstructed_signals.append(
                        np.array(amps) * np.cos(block_arguments + np.array(phss))
                    )
                else:
                    vk_signals, vk_amplitudes, vk_phases = generator.send(
                        (block, block_arguments, last_data)
                    )
                    if vk_signals is not None:
                        reconstructed_signals.append(vk_signals)
                        reconstructed_amplitudes.append(vk_amplitudes)
                        reconstructed_phases.append(vk_phases)
                start_index += block_size

            predicted_signals.append(np.concatenate(reconstructed_signals, axis=-1))
            predicted_amplitudes.append(np.concatenate(reconstructed_amplitudes, axis=-1))
            predicted_phases.append(np.concatenate(reconstructed_phases, axis=-1))

        self.predicted_response_signals = np.array(predicted_signals).transpose(1, 0, 2)
        self.predicted_response_amplitudes = np.array(predicted_amplitudes).transpose(1, 0, 2)
        self.predicted_response_phases = np.array(predicted_phases).transpose(1, 0, 2)
        # Pull out the data to compute maximum amplitude error
        self.predicted_amplitude_error = np.zeros(self.specification_amplitudes.shape[:2])
        for tone_index, (specs, preds, start_index, end_index) in enumerate(
            zip(
                self.specification_amplitudes,
                self.predicted_response_amplitudes,
                self.specification_start_indices,
                self.specification_end_indices,
            )
        ):
            for channel_index, (spec, pred) in enumerate(zip(specs, preds)):
                spec = spec[
                    start_index : end_index : self.data_acquisition_parameters.output_oversample
                ]
                pred = pred[
                    start_index
                    // self.data_acquisition_parameters.output_oversample : start_index
                    // self.data_acquisition_parameters.output_oversample
                    + spec.size
                ]
                max_error = np.max(np.abs(scale2db(pred / spec)))
                self.predicted_amplitude_error[tone_index, channel_index] = max_error

    def compare_predictions_to_warning_and_abort(self):
        """Compares the extracted prediction information to abort and warning levels"""
        specs = self.environment_parameters.specifications
        amps = self.predicted_response_amplitudes
        warning_matrix = np.zeros(amps.shape[:2], dtype=bool)
        abort_matrix = np.zeros(amps.shape[:2], dtype=bool)
        for tone_index in range(amps.shape[0]):
            freqs = self.excitation_signal_frequencies[
                tone_index,
                self.specification_start_indices[tone_index] : self.specification_end_indices[
                    tone_index
                ] : self.data_acquisition_parameters.output_oversample,
            ]
            for channel_index in range(amps.shape[1]):
                warning_levels = specs[tone_index].interpolate_warning(channel_index, freqs)
                abort_levels = specs[tone_index].interpolate_abort(channel_index, freqs)
                predicted = amps[
                    tone_index,
                    channel_index,
                    self.specification_start_indices[tone_index]
                    // self.data_acquisition_parameters.output_oversample : self.specification_start_indices[
                        tone_index
                    ]
                    // self.data_acquisition_parameters.output_oversample
                    + freqs.size,
                ]
                warning_ratio = predicted / warning_levels
                if np.any(warning_ratio[0] < 1.0):
                    warning_matrix[tone_index, channel_index] = True
                if np.any(warning_ratio[1] > 1.0):
                    warning_matrix[tone_index, channel_index] = True
                abort_ratio = predicted / abort_levels
                if np.any(abort_ratio[0] < 1.0):
                    abort_matrix[tone_index, channel_index] = True
                if np.any(abort_ratio[1] > 1.0):
                    abort_matrix[tone_index, channel_index] = True
        self.predicted_warning_matrix = warning_matrix
        self.predicted_abort_matrix = abort_matrix

    def perform_control_prediction(self, sysid_update):
        """Compute the prediction from the test by convolving with the transfer functions"""
        # print('Performing Control Prediction')
        if self.sysid_frf is None:
            self.gui_update_queue.put(
                (
                    "error",
                    (
                        "Perform System Identification",
                        "Perform System ID before performing test predictions",
                    ),
                )
            )
            return
        # print('Computing Drive Signal')
        if sysid_update:
            self.log("Updating System Identification...")
            (
                self.excitation_signals,
                self.excitation_signal_frequencies,
                self.excitation_signal_arguments,
                self.excitation_signal_amplitudes,
                self.excitation_signal_phases,
            ) = self.control_class.system_id_update(
                self.environment_parameters.sysid_frequency_spacing,
                self.sysid_frf,  # Transfer Functions
                self.sysid_response_noise,  # Noise levels and correlation
                self.sysid_reference_noise,  # from the system identification
                self.sysid_response_cpsd,  # Response levels and correlation
                self.sysid_reference_cpsd,  # from the system identification
                self.sysid_coherence,  # Coherence from the system identification
                self.sysid_frames,  # Number of frames in the CPSD and FRF matrices
            )
            self.excitation_signals_combined = np.sum(self.excitation_signals, axis=0)
            self.peak_voltages = np.max(np.abs(self.excitation_signals_combined), axis=-1)
            self.log("Done!")
        # print('Performing Response Prediction')
        # print('Drive Signals {:}'.format(self.next_drive.shape))
        drive_signals = self.excitation_signals_combined[
            :, :: self.data_acquisition_parameters.output_oversample
        ]
        impulse_responses = np.moveaxis(np.fft.irfft(self.sysid_frf, axis=0), 0, -1)

        self.log("Predicting Test Response...")
        self.predicted_response_signals_combined = np.zeros(
            (impulse_responses.shape[0], drive_signals.shape[-1])
        )

        for i, impulse_response_row in enumerate(impulse_responses):
            for impulse, drive in zip(impulse_response_row, drive_signals):
                # print('Convolving {:},{:}'.format(i,j))
                self.predicted_response_signals_combined[i, :] += sig.convolve(
                    drive, impulse, "full"
                )[: drive_signals.shape[-1]]
        self.log("Done!")

        self.log("Filtering Predicted Signals...")
        self.filter_predicted_signal()
        self.log("Done!")

        # print('From Performing Control Predictions')
        # print(f'{self.excitation_signals_combined.shape=}')
        # print(f'{self.excitation_signals.shape=}')
        # print(f'{self.excitation_signal_frequencies.shape=}')
        # print(f'{self.excitation_signal_arguments.shape=}')
        # print(f'{self.excitation_signal_amplitudes.shape=}')
        # print(f'{self.excitation_signal_phases.shape=}')
        # print(f'{self.ramp_samples=}')
        # print(f'{self.predicted_response_signals_combined.shape=}')
        # print(f'{self.predicted_response_signals.shape=}')
        # print(f'{self.predicted_response_amplitudes.shape=}')
        # print(f'{self.predicted_response_phases.shape=}')

        self.log("Comparing to Warning and Abort Curves")
        self.compare_predictions_to_warning_and_abort()
        self.log("Done!")

        self.log("Showing Test Predictions...")
        self.show_test_prediction()
        self.log("Done!")

    def show_test_prediction(self):
        """Starts the process to show the predictions by requesting the current plot choices"""
        self.gui_update_queue.put(
            (self.environment_name, ("request_prediction_plot_choices", None))
        )
        self.gui_update_queue.put(
            (self.environment_name, ("excitation_voltage_list", self.peak_voltages))
        )
        self.gui_update_queue.put(
            (
                self.environment_name,
                (
                    "response_error_matrix",
                    (
                        self.predicted_amplitude_error,
                        self.predicted_warning_matrix,
                        self.predicted_abort_matrix,
                    ),
                ),
            )
        )

    def send_excitation_prediction(self, excitation_plot_choices):
        """Sends the predicted excitation for the channel, tone, and data type requested"""
        channel_index, type_index, tone_index = excitation_plot_choices
        # print(f'Excitation Predictions: {channel_index=}, {type_index=}, {tone_index}')
        if type_index == 0:  # Time histories
            if tone_index == -1:
                ordinate = self.excitation_signals_combined[
                    channel_index,
                    :: self.plot_downsample * self.data_acquisition_parameters.output_oversample,
                ]
                abscissa = (
                    np.arange(ordinate.shape[-1])
                    / self.data_acquisition_parameters.sample_rate
                    * self.plot_downsample
                )
            else:
                ordinate = self.excitation_signals[
                    tone_index,
                    channel_index,
                    :: self.plot_downsample * self.data_acquisition_parameters.output_oversample,
                ]
                abscissa = (
                    np.arange(ordinate.shape[-1])
                    / self.data_acquisition_parameters.sample_rate
                    * self.plot_downsample
                )
        elif type_index == 1:  # Amplitude Vs Time
            ordinate = self.excitation_signal_amplitudes[
                tone_index,
                channel_index,
                :: self.plot_downsample * self.data_acquisition_parameters.output_oversample,
            ]
            abscissa = (
                np.arange(ordinate.shape[-1])
                / self.data_acquisition_parameters.sample_rate
                * self.plot_downsample
            )
        elif type_index == 2:  # Phase Vs Time
            ordinate = (
                wrap(
                    self.excitation_signal_phases[
                        tone_index,
                        channel_index,
                        :: self.plot_downsample
                        * self.data_acquisition_parameters.output_oversample,
                    ]
                )
                * 180
                / np.pi
            )
            abscissa = (
                np.arange(ordinate.shape[-1])
                / self.data_acquisition_parameters.sample_rate
                * self.plot_downsample
            )
        elif type_index == 3:  # Amplitude Vs Frequency
            ordinate = self.excitation_signal_amplitudes[
                tone_index,
                channel_index,
                :: self.plot_downsample * self.data_acquisition_parameters.output_oversample,
            ]
            abscissa = self.specification_frequencies[
                tone_index,
                :: self.plot_downsample * self.data_acquisition_parameters.output_oversample,
            ]
        elif type_index == 4:  # Phase Vs Frequency
            ordinate = (
                wrap(
                    self.excitation_signal_phases[
                        tone_index,
                        channel_index,
                        :: self.plot_downsample
                        * self.data_acquisition_parameters.output_oversample,
                    ]
                )
                * 180
                / np.pi
            )
            abscissa = self.specification_frequencies[
                tone_index,
                :: self.plot_downsample * self.data_acquisition_parameters.output_oversample,
            ]
        else:
            raise ValueError(f"Undefined type_index {type_index}")
        # print(f'{ordinate.shape=}, {abscissa.shape=}')
        # print(f'{abscissa.min()=}, {abscissa.max()=}')
        self.gui_update_queue.put(
            (self.environment_name, ("excitation_prediction", (abscissa, ordinate)))
        )

    def send_response_prediction(self, response_plot_choices):
        """Sends the response predictions at the requested channel, tone, and data type"""
        channel_index, type_index, tone_index = response_plot_choices
        # print(f'Response Predictions: {channel_index=}, {type_index=}, {tone_index}')
        if type_index == 0:  # Time histories
            if tone_index == -1:
                ordinate = [
                    self.specification_signals_combined[
                        channel_index,
                        :: self.plot_downsample
                        * self.data_acquisition_parameters.output_oversample,
                    ],
                    self.predicted_response_signals_combined[
                        channel_index, :: self.plot_downsample
                    ],
                ]
                abscissa = (
                    np.arange(max(v.shape[-1] for v in ordinate))
                    / self.data_acquisition_parameters.sample_rate
                    * self.plot_downsample
                )
            else:
                ordinate = [
                    self.specification_signals[
                        tone_index,
                        channel_index,
                        :: self.plot_downsample
                        * self.data_acquisition_parameters.output_oversample,
                    ],
                    self.predicted_response_signals[
                        tone_index, channel_index, :: self.plot_downsample
                    ],
                ]
                abscissa = (
                    np.arange(max(v.shape[-1] for v in ordinate))
                    / self.data_acquisition_parameters.sample_rate
                    * self.plot_downsample
                )
        elif type_index == 1:  # Amplitude Vs Time
            ordinate = [
                self.specification_amplitudes[
                    tone_index,
                    channel_index,
                    :: self.plot_downsample * self.data_acquisition_parameters.output_oversample,
                ],
                self.predicted_response_amplitudes[
                    tone_index, channel_index, :: self.plot_downsample
                ],
            ]
            abscissa = (
                np.arange(max(v.shape[-1] for v in ordinate))
                / self.data_acquisition_parameters.sample_rate
                * self.plot_downsample
            )
        elif type_index == 2:  # Phase Vs Time
            ordinate = [
                self.specification_phases[
                    tone_index,
                    channel_index,
                    :: self.plot_downsample * self.data_acquisition_parameters.output_oversample,
                ]
                * 180
                / np.pi,
                self.predicted_response_phases[tone_index, channel_index, :: self.plot_downsample]
                * 180
                / np.pi,
            ]
            abscissa = (
                np.arange(max(v.shape[-1] for v in ordinate))
                / self.data_acquisition_parameters.sample_rate
                * self.plot_downsample
            )
        elif type_index == 3:  # Amplitude Vs Frequency
            ordinate = [
                self.specification_amplitudes[
                    tone_index,
                    channel_index,
                    :: self.plot_downsample * self.data_acquisition_parameters.output_oversample,
                ],
                self.predicted_response_amplitudes[
                    tone_index, channel_index, :: self.plot_downsample
                ],
            ]
            abscissa = self.specification_frequencies[
                tone_index,
                :: self.plot_downsample * self.data_acquisition_parameters.output_oversample,
            ]
        elif type_index == 4:  # Phase Vs Frequency
            ordinate = [
                self.specification_phases[
                    tone_index,
                    channel_index,
                    :: self.plot_downsample * self.data_acquisition_parameters.output_oversample,
                ]
                * 180
                / np.pi,
                self.predicted_response_phases[tone_index, channel_index, :: self.plot_downsample]
                * 180
                / np.pi,
            ]
            abscissa = self.specification_frequencies[
                tone_index,
                :: self.plot_downsample * self.data_acquisition_parameters.output_oversample,
            ]
        else:
            raise ValueError(f"Undefined type_index {type_index}")
        # print(f'{ordinate[0].shape=}, {ordinate[1].shape=}, {abscissa.shape=}')
        # print(f'{abscissa.min()=}, {abscissa.max()=}')
        self.gui_update_queue.put(
            (self.environment_name, ("response_prediction", (abscissa, ordinate)))
        )

    def compute_spec_amplitudes_and_phases(self):
        """Computes amplitude and phase information from the specification"""
        spec_ordinates = []
        spec_amplitudes = []
        spec_phases = []
        spec_frequencies = []
        spec_arguments = []

        for channel_index in range(len(self.environment_parameters.control_channel_indices)):
            spec = self.environment_parameters.specification
            # Convert octave per min to octave per second
            sweep_rates = spec["sweep_rate"].copy()
            sweep_rates[spec["sweep_type"] == 1] = sweep_rates[spec["sweep_type"] == 1] / 60
            # Create the sweep types array
            sweep_types = [
                "lin" if sweep_type == 0 else "log" for sweep_type in spec["sweep_type"][:-1]
            ]
            spec_ordinate, spec_argument, spec_frequency, spec_amplitude, spec_phase = sine_sweep(
                1 / self.data_acquisition_parameters.sample_rate,
                spec["frequency"],
                sweep_rates,
                sweep_types,
                spec["amplitude"][:, channel_index],
                spec["phase"][:, channel_index],
                return_frequency=True,
                return_argument=True,
                return_amplitude=True,
                return_phase=True,
            )
            spec_ordinates.append(spec_ordinate)
            spec_amplitudes.append(spec_amplitude)
            spec_phases.append(spec_phase)
            spec_arguments.append(spec_argument)
            spec_frequencies.append(spec_frequency)

        return (
            np.array(spec_ordinates),
            np.array(spec_arguments),
            np.array(spec_frequencies),
            np.array(spec_amplitudes),
            np.array(spec_phases),
        )

    def get_signal_generation_metadata(self):
        """Gets a SignalGenerationMetadata object for the current environment"""
        return SignalGenerationMetadata(
            samples_per_write=self.data_acquisition_parameters.samples_per_write,
            level_ramp_samples=self.environment_parameters.ramp_time
            * self.environment_parameters.sample_rate
            * self.data_acquisition_parameters.output_oversample,
            output_transformation_matrix=self.environment_parameters.reference_transformation_matrix,
        )

    def start_control(self, data):
        """
        Starts up and runs the control with the specified test level,
        tones, start and end times
        """
        if self.control_startup:
            self.log("Starting Environment")
            # Read in the starting parameters
            (
                self.control_test_level,
                self.control_tones,
                self.control_start_time,
                self.control_end_time,
            ) = data
            if self.control_tones is not None and len(self.control_tones) == 0:
                self.control_tones = None
            if self.control_tones is None:
                self.control_tones = slice(None)
                self.control_tone_indices = np.arange(self.excitation_signal_arguments.shape[0])
            else:
                self.control_tone_indices = self.control_tones
            # Precompute the number of channels for convenience
            n_control_channels = (
                len(self.environment_parameters.control_channel_indices)
                if self.environment_parameters.response_transformation_matrix is None
                else self.environment_parameters.response_transformation_matrix.shape[0]
            )
            n_output_channels = (
                len(self.environment_parameters.output_channel_indices)
                if self.environment_parameters.reference_transformation_matrix is None
                else self.environment_parameters.reference_transformation_matrix.shape[0]
            )

            self.control_time_delay = (
                None  # We will need to compute this when we get our first data point
            )
            if DEBUG:
                num_files = len(glob(FILE_OUTPUT.format("*")))
                np.savez(
                    FILE_OUTPUT.format(num_files),
                    excitation_signal=self.control_first_signal,
                    done_controlling=False,
                )
            # print('Parsing Indices')
            # Parse out frequency information to get the indices into the full
            # arrays that we are controlling to
            if self.control_start_time is None:
                self.control_start_index = 0
            else:
                self.control_start_index = int(
                    self.control_start_time
                    * self.data_acquisition_parameters.sample_rate
                    * self.data_acquisition_parameters.output_oversample
                )
                if self.control_start_index < 0:
                    self.control_start_index = 0
            if self.control_end_time is None:
                self.control_end_index = None
            else:
                self.control_end_index = (
                    int(
                        self.control_end_time
                        * self.data_acquisition_parameters.sample_rate
                        * self.data_acquisition_parameters.output_oversample
                    )
                    + 2 * self.ramp_samples
                )
            if (
                self.control_end_index is None
                or self.control_end_index > self.specification_arguments.shape[-1]
            ):
                self.control_end_index = self.specification_arguments.shape[-1]
            # Check to make sure if we are controlling to a subset of the tones that we
            # aren't starting with a bunch of zeros
            if self.control_tones is not None:
                first_nonzero_index = np.min(
                    np.argmax(
                        self.excitation_signal_amplitudes[self.control_tones] != 0,
                        axis=-1,
                    )
                )
                if first_nonzero_index > self.control_start_index:
                    self.control_start_index = first_nonzero_index
            control_slice = slice(self.control_start_index, self.control_end_index)

            # print('Initializing Control')
            # Call the control law to get the first of its signals
            self.control_first_signal = self.control_class.initialize_control(
                self.control_tones, self.control_start_index, self.control_end_index
            )

            # print('Constructing Control Arrays')
            # Construct the full arrays that we are controlling to
            self.control_specification_arguments = self.excitation_signal_arguments[
                self.control_tones, control_slice
            ]

            # print('Setting Up Tracking Filters')
            # Set up the tracking filters to track amplitude and phase information
            self.control_filters = []
            if self.environment_parameters.tracking_filter_type == 0:
                self.control_block_size = self.data_acquisition_parameters.samples_per_read
            else:
                self.control_block_size = self.environment_parameters.vk_filter_blocksize
            for signal in self.predicted_response_signals_combined:
                if self.environment_parameters.tracking_filter_type == 0:
                    generator = [
                        digital_tracking_filter_generator(
                            dt=1 / self.environment_parameters.sample_rate,
                            cutoff_frequency_ratio=self.environment_parameters.tracking_filter_cutoff,
                            filter_order=self.environment_parameters.tracking_filter_order,
                        )
                        for tone in self.control_specification_arguments
                    ]
                    for gen in generator:
                        gen.send(None)
                    self.control_filters.append(generator)
                else:
                    generator = vold_kalman_filter_generator(
                        sample_rate=self.environment_parameters.sample_rate,
                        num_orders=self.control_specification_arguments.shape[0],
                        block_size=self.control_block_size,
                        overlap=self.environment_parameters.vk_filter_overlap,
                        bandwidth=self.environment_parameters.vk_filter_bandwidth,
                        filter_order=self.environment_parameters.vk_filter_order,
                        buffer_size_factor=self.environment_parameters.buffer_blocks + 1,
                    )
                    generator.send(None)
                    self.control_filters.append(generator)

            # Set up empty warning and abort flags
            self.control_warning_flags = np.zeros(
                (self.control_specification_arguments.shape[0], n_control_channels),
                dtype=bool,
            )
            self.control_abort_flags = np.zeros(
                (self.control_specification_arguments.shape[0], n_control_channels),
                dtype=bool,
            )
            self.control_amplitude_errors = np.zeros(
                (self.control_specification_arguments.shape[0], n_control_channels)
            )

            # print('Setting up Signal Generation')
            # Set up the signal generation
            self.siggen_shutdown_achieved = False
            self.queue_container.signal_generation_command_queue.put(
                self.environment_name,
                (
                    SignalGenerationCommands.INITIALIZE_PARAMETERS,
                    self.get_signal_generation_metadata(),
                ),
            )
            self.queue_container.signal_generation_command_queue.put(
                self.environment_name,
                (
                    SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,
                    ContinuousTransientSignalGenerator(
                        num_samples_per_frame=self.data_acquisition_parameters.samples_per_write,
                        num_signals=n_output_channels,
                        signal=self.control_first_signal,
                        last_signal=False,
                    ),
                ),
            )
            self.queue_container.signal_generation_command_queue.put(
                self.environment_name,
                (SignalGenerationCommands.SET_TEST_LEVEL, self.control_test_level),
            )
            # Tell the signal generation to start generating signals
            self.queue_container.signal_generation_command_queue.put(
                self.environment_name, (SignalGenerationCommands.GENERATE_SIGNALS, None)
            )

            # print('Setting up last arguments')
            self.control_write_index = self.control_first_signal.shape[-1]
            self.control_read_index = 0
            self.control_analysis_index = 0
            self.control_finished = False
            self.control_analysis_finished = False
            self.control_response_signals_combined = []
            self.control_response_amplitudes = []
            self.control_response_phases = []
            self.control_drive_modifications = []
            self.control_response_frequencies = self.excitation_signal_frequencies[
                self.control_tones,
                self.control_start_index : self.control_end_index : self.data_acquisition_parameters.output_oversample,
            ]
            self.control_response_arguments = self.excitation_signal_arguments[
                self.control_tones,
                self.control_start_index : self.control_end_index : self.data_acquisition_parameters.output_oversample,
            ]
            self.control_target_phases = self.specification_phases[
                self.control_tones,
                :,
                self.control_start_index : self.control_end_index : self.data_acquisition_parameters.output_oversample,
            ]
            self.control_target_amplitudes = self.specification_amplitudes[
                self.control_tones,
                :,
                self.control_start_index : self.control_end_index : self.data_acquisition_parameters.output_oversample,
            ]
            self.control_startup = False
        # See if any data has come in
        try:
            # print('Listening for Data')
            acquisition_data, last_acquisition = self.queue_container.data_in_queue.get_nowait()
            # print('Got Data')
            if last_acquisition:
                self.log(
                    "Acquired Last Data, Signal Generation Shutdown "
                    f"Achieved: {self.siggen_shutdown_achieved}"
                )
            else:
                self.log("Acquired Data")
            scale_factor = 0.0 if self.control_test_level < 1e-10 else 1 / self.control_test_level
            # print('Parsing Control and Excitation Data')
            control_data = (
                acquisition_data[self.environment_parameters.control_channel_indices] * scale_factor
            )
            if self.environment_parameters.response_transformation_matrix is not None:
                control_data = (
                    self.environment_parameters.response_transformation_matrix @ control_data
                )
            excitation_data = (
                acquisition_data[self.environment_parameters.output_channel_indices] * scale_factor
            )
            if self.environment_parameters.reference_transformation_matrix is not None:
                excitation_data = (
                    self.environment_parameters.reference_transformation_matrix @ excitation_data
                )
            block_size = acquisition_data.shape[-1]
            block_slice = slice(self.control_read_index, self.control_read_index + block_size)
            # print(f'Read Block Range {block_slice}')
            # Send the time data to the GUI
            # print('Sending Time Data')
            self.gui_update_queue.put(
                (
                    self.environment_name,
                    (
                        "time_data",
                        (
                            excitation_data[..., :: self.plot_downsample],
                            control_data[..., :: self.plot_downsample],
                        ),
                    ),
                )
            )
            self.control_response_signals_combined.append(control_data)
            # Find the time delay between the first signal and what we've
            # measured
            # print('Computing Time Delay')
            if self.control_first_signal is not None:
                first_signal = self.control_first_signal[
                    ..., :: self.data_acquisition_parameters.output_oversample
                ][..., : excitation_data.shape[-1]]
                reference_fft = np.fft.rfft(first_signal, axis=-1)
                this_fft = np.fft.rfft(excitation_data, axis=-1)
                freq = np.fft.rfftfreq(
                    first_signal.shape[-1],
                    1 / self.data_acquisition_parameters.sample_rate,
                )
                good_lines = (
                    np.abs(reference_fft) / np.max(np.abs(reference_fft), axis=-1, keepdims=True)
                    > self.good_line_threshold
                )
                good_lines[..., 0] = False
                phase_difference = np.angle(this_fft / reference_fft)
                phase_slope = np.median(
                    phase_difference[good_lines]
                    / np.broadcast_to(freq, phase_difference.shape)[good_lines]
                )
                self.control_time_delay = phase_slope / (2 * np.pi)
                # print(f"Time Delay: {self.control_time_delay=}")
                if DEBUG:
                    np.savez(
                        "debug_data/first_signal_sine_control.npz",
                        first_signal=self.control_first_signal,
                        excitation_data=excitation_data,
                    )
                self.control_first_signal = None
            # Analyze the recovered data
            if not self.control_analysis_finished:
                # print('Analyzing Data')
                achieved_signals = []
                achieved_amplitudes = []
                achieved_phases = []
                block_frequencies = self.control_response_frequencies[..., block_slice]
                block_arguments = self.control_response_arguments[..., block_slice]
                # print('Block Frequencies')
                # print(block_frequencies)
                # Check if this is the last control data we will be getting
                self.control_analysis_finished = (
                    block_slice.stop >= self.control_response_frequencies.shape[-1]
                )
                # print(f'Is Last Data? {self.control_analysis_finished=}')
                # Truncate just in case we've gotten some extra data in the acquisition
                block_signal = control_data[..., : block_arguments.shape[-1]]
                # print('Filtering Data to extract amplitude and phase')
                start_time = time.time()
                if self.environment_parameters.tracking_filter_type == 0:
                    for signal, tone_filters in zip(block_signal, self.control_filters):
                        amps = []
                        phss = []
                        for tone_argument, tone_frequency, tone_filter in zip(
                            block_arguments, block_frequencies, tone_filters
                        ):
                            amp, phs = tone_filter.send((signal, tone_frequency, tone_argument))
                            amps.append(amp)
                            phss.append(phs)  # Radians
                        achieved_amplitudes.append(np.array(amps))
                        achieved_phases.append(np.array(phss))
                        achieved_signals.append(
                            np.array(amps) * np.cos(block_arguments + np.array(phss))  # Radians
                        )
                else:
                    for signal, vk_filter in zip(block_signal, self.control_filters):
                        vk_signals, vk_amplitudes, vk_phases = vk_filter.send(
                            (signal, block_arguments, self.control_analysis_finished)
                        )
                        achieved_amplitudes.append(vk_amplitudes)
                        achieved_phases.append(vk_phases)  # Radians
                        achieved_signals.append(vk_signals)

                finish_time = time.time()
                self.log(f"Signal filtering achieved in {finish_time - start_time:0.2f}s.")
                achieved_signals = np.array(achieved_signals)
                achieved_amplitudes = np.array(achieved_amplitudes)
                achieved_phases = np.array(achieved_phases)
                if not np.all(
                    achieved_signals == None  # noqa: E711 # pylint: disable=singleton-comparison
                ):
                    # print('Got Amplitude and Phase Data')
                    block_start = self.control_analysis_index
                    block_end = self.control_analysis_index + achieved_signals.shape[-1]
                    block_slice = slice(block_start, block_end)
                    # print(f'Analysis Block Range {block_slice}')
                    achieved_frequencies = self.control_response_frequencies[..., block_slice]
                    # print('Analysis Frequencies')
                    # print(achieved_frequencies)
                    # print(f'Achieved Frequency Size {achieved_frequencies.shape=}')
                    achieved_signals = achieved_signals.transpose(1, 0, 2)
                    self.log(f"Analyzing Dataset With Size {achieved_signals.shape}")
                    # print(f'Achieved Signals Size {achieved_signals.shape=}')
                    achieved_amplitudes = achieved_amplitudes.transpose(1, 0, 2)
                    # print(f'Achieved Amplitudes Size {achieved_signals.shape=}')
                    # Correct for time delay on the phases, newaxis to broadcast across signals
                    achieved_phases = (
                        achieved_phases.transpose(1, 0, 2)
                        - self.control_time_delay
                        * 2
                        * np.pi
                        * achieved_frequencies[:, np.newaxis, :]
                    )
                    # print(f'Achieved Phases Size {achieved_phases.shape=}')
                    # Here I want to do the best fit to the phases, need to compare phase
                    # achieved vs phase desired
                    if self.environment_parameters.phase_fit:
                        # print('Fitting Phases')
                        target = self.control_target_amplitudes[..., block_slice] * np.exp(
                            1j * self.control_target_phases[..., block_slice]
                        )
                        achieved = achieved_amplitudes * np.exp(1j * achieved_phases)
                        phase_change_fit = np.angle(np.sum(target * achieved.conj()))
                        achieved_phases = achieved_phases + phase_change_fit
                    # print('Computing Drive Updates')
                    drive_modification = self.control_class.update_control(
                        achieved_signals,
                        achieved_amplitudes,
                        achieved_phases,  # Radians
                        achieved_frequencies,
                        self.control_time_delay,
                    )
                    self.control_drive_modifications.append(drive_modification)
                    # Need to develop data for the table.  First we need to pick out "valid"
                    # data, which will exclude the ramp-ups and ramp-downs
                    for tone_index in range(achieved_signals.shape[0]):
                        full_tone_index = self.control_tone_indices[tone_index]
                        compare_start = max(
                            (
                                self.specification_start_indices[full_tone_index]
                                - self.control_start_index
                            )
                            // self.data_acquisition_parameters.output_oversample,
                            block_start,
                            self.ramp_samples // self.data_acquisition_parameters.output_oversample,
                        )
                        compare_end = min(
                            (
                                self.specification_end_indices[full_tone_index]
                                - self.control_start_index
                            )
                            // self.data_acquisition_parameters.output_oversample,
                            block_end,
                            (self.control_end_index - self.control_start_index - self.ramp_samples)
                            // self.data_acquisition_parameters.output_oversample,
                        )
                        if compare_start >= compare_end:
                            continue
                        block_start_offset = compare_start - block_start
                        block_end_offset = compare_end - block_start
                        amplitudes = achieved_amplitudes[
                            tone_index, :, block_start_offset:block_end_offset
                        ]
                        compare_amplitudes = self.control_target_amplitudes[
                            tone_index, :, compare_start:compare_end
                        ]
                        compare_frequencies = self.control_response_frequencies[
                            tone_index, compare_start:compare_end
                        ]
                        self.control_amplitude_errors[tone_index] = np.max(
                            np.abs(scale2db(amplitudes / compare_amplitudes)), axis=-1
                        )
                        if np.any(np.isinf(self.control_amplitude_errors[tone_index])):
                            self.log(
                                f"Found Infinities:\n{amplitudes.shape=} "
                                f"{compare_amplitudes.shape=}"
                            )
                            self.log(f"Infinity Frequencies: {compare_frequencies}")
                            self.log(
                                f"Comparison Amplitudes: "
                                f"{compare_amplitudes[np.isinf(self.control_amplitude_errors[tone_index])]}"
                            )
                        for channel_index in range(amplitudes.shape[0]):
                            compare_warnings = self.environment_parameters.specifications[
                                full_tone_index
                            ].interpolate_warning(channel_index, compare_frequencies)
                            compare_aborts = self.environment_parameters.specifications[
                                full_tone_index
                            ].interpolate_abort(channel_index, compare_frequencies)
                            warning_ratio = amplitudes[channel_index] / compare_warnings
                            abort_ratio = amplitudes[channel_index] / compare_aborts
                            if np.any(warning_ratio[0] < 1.0):
                                self.control_warning_flags[tone_index, channel_index] = True
                                self.log(
                                    f"Lower Warning at Tone {full_tone_index} Channel "
                                    f"{channel_index} Frequency "
                                    f"{compare_frequencies[warning_ratio[0] < 1.0]}"
                                )
                                self.log(
                                    f"Amplitudes: {amplitudes[channel_index, warning_ratio[0] < 1.0]}"
                                )
                                self.log(
                                    f"Warning Level: {compare_warnings[0, warning_ratio[0] < 1.0]}"
                                )
                            if np.any(warning_ratio[1] > 1.0):
                                self.control_warning_flags[tone_index, channel_index] = True
                                self.log(
                                    f"Upper Warning at Tone {full_tone_index} Channel "
                                    f"{channel_index} Frequency "
                                    f"{compare_frequencies[warning_ratio[1] > 1.0]}"
                                )
                                self.log(
                                    f"Amplitudes: {amplitudes[channel_index, warning_ratio[1] > 1.0]}"
                                )
                                self.log(
                                    f"Warning Level: {compare_warnings[1, warning_ratio[1] > 1.0]}"
                                )
                            if np.any(abort_ratio[0] < 1.0):
                                self.control_abort_flags[tone_index, channel_index] = True
                                self.log(
                                    f"Lower Abort at Tone {full_tone_index} Channel "
                                    f"{channel_index} Frequency "
                                    f"{compare_frequencies[abort_ratio[0] < 1.0]}"
                                )
                                self.log(
                                    f"Amplitudes: {amplitudes[channel_index, abort_ratio[0] < 1.0]}"
                                )
                                self.log(f"Abort Level: {compare_aborts[0, abort_ratio[0] < 1.0]}")
                            if np.any(abort_ratio[1] > 1.0):
                                self.control_abort_flags[tone_index, channel_index] = True
                                self.log(
                                    f"Upper Abort at Tone {full_tone_index} Channel {channel_index} "
                                    f"Frequency {compare_frequencies[abort_ratio[1] > 1.0]}"
                                )
                                self.log(
                                    f"Amplitudes: {amplitudes[channel_index, abort_ratio[1] > 1.0]}"
                                )
                                self.log(f"Abort Level: {compare_aborts[1, abort_ratio[1] > 1.0]}")
                    # print('Populating Full Block Data')
                    full_achieved_signals = np.zeros(
                        (self.specification_signals.shape[0],) + achieved_signals.shape[1:]
                    )
                    full_achieved_signals[self.control_tones] = achieved_signals
                    full_achieved_amplitudes = np.zeros(
                        (self.specification_amplitudes.shape[0],) + achieved_amplitudes.shape[1:]
                    )
                    full_achieved_amplitudes[self.control_tones] = achieved_amplitudes
                    full_achieved_phases = np.zeros(
                        (self.specification_phases.shape[0],) + achieved_phases.shape[1:]
                    )
                    full_achieved_phases[self.control_tones] = achieved_phases
                    full_achieved_frequencies = np.zeros(
                        (self.specification_frequencies.shape[0],) + achieved_frequencies.shape[1:]
                    )
                    full_achieved_frequencies[self.control_tones] = achieved_frequencies
                    full_drive_modification = np.zeros(
                        (self.specification_frequencies.shape[0],) + drive_modification.shape[1:],
                        dtype=complex,
                    )
                    full_drive_modification[self.control_tones] = drive_modification
                    full_achieved_amplitude_errors = np.zeros(
                        (self.specification_signals.shape[0],)
                        + self.control_amplitude_errors.shape[1:]
                    )
                    full_achieved_amplitude_errors[self.control_tones] = (
                        self.control_amplitude_errors
                    )
                    full_achieved_warning_flags = np.zeros(
                        (self.specification_signals.shape[0],)
                        + self.control_warning_flags.shape[1:],
                        dtype=bool,
                    )
                    full_achieved_warning_flags[self.control_tones] = self.control_warning_flags
                    full_achieved_abort_flags = np.zeros(
                        (self.specification_signals.shape[0],) + self.control_abort_flags.shape[1:],
                        dtype=bool,
                    )
                    full_achieved_abort_flags[self.control_tones] = self.control_abort_flags
                    self.control_response_amplitudes.append(achieved_amplitudes)
                    self.control_response_phases.append(achieved_phases)

                    # print('Sending Block Data to GUI')
                    self.gui_update_queue.put(
                        (
                            self.environment_name,
                            (
                                "control_data",
                                (
                                    full_achieved_signals[..., :: self.plot_downsample],
                                    full_achieved_amplitudes[..., :: self.plot_downsample],
                                    full_achieved_phases[..., :: self.plot_downsample]
                                    * 180
                                    / np.pi,
                                    full_achieved_frequencies[..., :: self.plot_downsample],
                                    full_drive_modification,
                                    full_achieved_amplitude_errors,
                                    full_achieved_warning_flags,
                                    full_achieved_abort_flags,
                                ),
                            ),
                        )
                    )

                    self.control_analysis_index += achieved_signals.shape[-1]
                # Check if we're done analyzing control outputs and generate the next outputs
                if self.control_analysis_finished:
                    # print('Analysis is Finished')
                    self.log("Analysis Finished")
                    (
                        self.excitation_signals,
                        self.excitation_signal_frequencies,
                        self.excitation_signal_arguments,
                        self.excitation_signal_amplitudes,
                        self.excitation_signal_phases,  # Degrees
                        self.ramp_samples,
                    ) = self.control_class.finalize_control()
            self.control_read_index += acquisition_data.shape[-1]
            # Now we need to see if we need to write more data to the controller
            # This will be related to our current read index and write index for the control;
            # we don't want to run out of samples being generated on the output hardware
            # print('Checking if New Data is Needed')
            if (
                self.control_write_index // self.data_acquisition_parameters.output_oversample
                < self.control_read_index
                + self.environment_parameters.buffer_blocks
                * self.data_acquisition_parameters.samples_per_write
            ) and not self.control_finished:
                # print('Generating New Data')
                self.log("Generating New Data")
                excitation_signal, self.control_finished = self.control_class.generate_signal()
                # print('Data Generation Complete')
                if self.control_finished:
                    # print('Generated Last Data')
                    self.log("Control Finished")
                if DEBUG:
                    print("Writing Debug File")
                    num_files = len(glob(FILE_OUTPUT.format("*")))
                    np.savez(
                        FILE_OUTPUT.format(num_files),
                        excitation_signal=excitation_signal,
                        done_controlling=self.control_finished,
                    )
                self.log(f"Excitation Size: {np.sqrt(np.mean(excitation_signal**2))=}")
                self.queue_container.time_history_to_generate_queue.put(
                    (excitation_signal, self.control_finished)
                )
                self.control_write_index += excitation_signal.shape[-1]
        except mp.queues.Empty:
            # print("Didn't Find Data")
            last_acquisition = False
        # See if we need to keep going
        if self.siggen_shutdown_achieved and last_acquisition:
            self.shutdown()
        else:
            self.queue_container.environment_command_queue.put(
                self.environment_name, (SineCommands.START_CONTROL, None)
            )

    def shutdown(self):
        """Handles the environment after it has shut down"""
        self.log("Environment Shut Down")
        self.log(f"Before Flush: {self.queue_container.time_history_to_generate_queue.qsize()=}")
        flush_queue(self.queue_container.time_history_to_generate_queue, timeout=0.01)
        self.log(f"After Flush: {self.queue_container.time_history_to_generate_queue.qsize()=}")
        self.gui_update_queue.put((self.environment_name, ("enable_control", None)))
        self.control_startup = True

    def stop_environment(self, data):
        """Sends a signal to start the shutdown process"""
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.START_SHUTDOWN, None)
        )

    def save_control_data(self, filename):
        """Saves the control data to a numpy file"""
        output_dict = {}
        for label in [
            "control_response_signals_combined",
            "control_response_amplitudes",
            "control_response_phases",
            "control_drive_modifications",
        ]:
            for index, array in enumerate(getattr(self, label)):
                output_dict[f"{label}_{index}"] = array
        for label in [
            "control_response_frequencies",
            "control_response_arguments",
            "control_target_phases",
            "control_target_amplitudes",
        ]:
            output_dict[label] = getattr(self, label)
        output_dict["sample_rate"] = self.data_acquisition_parameters.sample_rate
        output_dict["output_oversample"] = self.data_acquisition_parameters.output_oversample
        output_dict["names"] = [spec.name for spec in self.environment_parameters.specifications]
        np.savez(filename, **output_dict)


# %% Process


def sine_process(
    environment_name: str,
    input_queue: VerboseMessageQueue,
    gui_update_queue: Queue,
    controller_communication_queue: VerboseMessageQueue,
    log_file_queue: Queue,
    data_in_queue: Queue,
    data_out_queue: Queue,
    acquisition_active,
    output_active,
):
    """A function to be used by multiprocessing to run the Sine environment.  It sets up
    the class and kicks off the run loop.

    Parameters
    ----------
    environment_name : str
        The name of the environment
    input_queue : VerboseMessageQueue
        A queue used to provide commands to the environment
    gui_update_queue : Queue
        A queue used to provide updates to the user interface from the environment
    controller_communication_queue : VerboseMessageQueue
        A queue used to communicate with the larger controller
    log_file_queue : Queue
        A queue used to handle logging
    data_in_queue : Queue
        A queue used to send data to the environment from the acqusition process
    data_out_queue : Queue
        A queue used to send data to the output process from the environment
    acquisition_active : int
        A multiprocessing value used as a flag to show when the acquisition is running
    output_active : int
        A multiprocessing value used as a flag to show when the output is running
    """
    try:
        # Create vibration queues
        queue_container = SineQueues(
            environment_name,
            input_queue,
            gui_update_queue,
            controller_communication_queue,
            data_in_queue,
            data_out_queue,
            log_file_queue,
        )

        spectral_proc = mp.Process(
            target=spectral_processing_process,
            args=(
                environment_name,
                queue_container.spectral_command_queue,
                queue_container.data_for_spectral_computation_queue,
                queue_container.updated_spectral_quantities_queue,
                queue_container.environment_command_queue,
                queue_container.gui_update_queue,
                queue_container.log_file_queue,
            ),
        )
        spectral_proc.start()
        analysis_proc = mp.Process(
            target=sysid_data_analysis_process,
            args=(
                environment_name,
                queue_container.data_analysis_command_queue,
                queue_container.updated_spectral_quantities_queue,
                queue_container.time_history_to_generate_queue,
                queue_container.environment_command_queue,
                queue_container.gui_update_queue,
                queue_container.log_file_queue,
            ),
        )
        analysis_proc.start()
        siggen_proc = mp.Process(
            target=signal_generation_process,
            args=(
                environment_name,
                queue_container.signal_generation_command_queue,
                queue_container.time_history_to_generate_queue,
                queue_container.data_out_queue,
                queue_container.environment_command_queue,
                queue_container.log_file_queue,
                queue_container.gui_update_queue,
            ),
        )
        siggen_proc.start()
        collection_proc = mp.Process(
            target=data_collector_process,
            args=(
                environment_name,
                queue_container.collector_command_queue,
                queue_container.data_in_queue,
                [queue_container.data_for_spectral_computation_queue],
                queue_container.environment_command_queue,
                queue_container.log_file_queue,
                queue_container.gui_update_queue,
            ),
        )
        collection_proc.start()

        process_class = SineEnvironment(
            environment_name, queue_container, acquisition_active, output_active
        )
        process_class.run()

        # Rejoin all the processes
        process_class.log("Joining Subprocesses")
        process_class.log("Joining Spectral Computation")
        spectral_proc.join()
        process_class.log("Joining Data Analysis")
        analysis_proc.join()
        process_class.log("Joining Signal Generation")
        siggen_proc.join()
        process_class.log("Joining Data Collection")
        collection_proc.join()
    except Exception:  # pylint: disable=broad-exception-caught
        print(traceback.format_exc())
