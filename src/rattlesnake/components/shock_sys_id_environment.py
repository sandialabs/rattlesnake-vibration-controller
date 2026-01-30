# -*- coding: utf-8 -*-
"""
This file defines a shock environment that utilizes system
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
import multiprocessing.sharedctypes  # pylint: disable=unused-import
import os
import traceback
from enum import Enum
from multiprocessing.queues import Queue

import netCDF4 as nc4
import numpy as np
import scipy.signal as sig
from qtpy import QtCore, QtWidgets, uic
from qtpy.QtCore import Qt

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
from .shock_sys_id_utilities import (
    ToneParameters,
    CompPulseParameters,
    DecayParameters,
    SRSParameters,
    SpecParameters,
    ControlParameters,
)
from .ui_utilities import (
    PlotTimeWindow,
    TransformationMatrixWindow,
    colororder,
    load_time_history,
    multiline_plotter,
)
from .utilities import (
    DataAcquisitionParameters,
    GlobalCommands,
    VerboseMessageQueue,
    align_signals,
    db2scale,
    load_python_module,
    rms_time,
    shift_signal,
    trac,
)

# %% Global Variables
CONTROL_TYPE = ControlTypes.SHOCK
MAXIMUM_NAME_LENGTH = 50
BUFFER_SIZE_SAMPLES_PER_READ_MULTIPLIER = 2


# %% Commands
class ShockCommands(Enum):
    """Valid commands for the Shock environment"""

    START_CONTROL = 0
    STOP_CONTROL = 1
    PERFORM_CONTROL_PREDICTION = 3
    # UPDATE_INTERACTIVE_CONTROL_PARAMETERS = 4


# %% Queues


class ShockQueues:
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
        """A container class for the queues that Shock will manage.

        The environment uses many queues to pass data between the various pieces.
        This class organizes those queues into one common namespace.

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


class ShockMetadata(AbstractSysIdMetadata):
    """Metadata required to define a Shock control law in rattlesnake."""

    def __init__(
        self,
        *,
        sample_rate: int,
        block_size: int,
        tone_data: ToneParameters,
        compensation_pulse_data: CompPulseParameters,
        decay_data: DecayParameters,
        srs_data: SRSParameters,
        control_script_data: ControlParameters,
        control_channel_indices: np.ndarray,
        output_channel_indices: np.ndarray,
        response_transformation_matrix: None | np.ndarray,
        excitation_transformation_matrix: None | np.ndarray,
        specification_data: SpecParameters,
    ):
        super().__init__()
        self.number_of_channels = number_of_channels
        self.block_size = block_size
        self.compensation_pulser_data = compensation_pulse_data
        self.decay_data = decay_data
        self.srs_data = srs_data
        self.control_script_data = control_script_data
        self.control_channel_indices = control_channel_indices
        self.output_channel_indices = output_channel_indices
        self.response_transformation_matrix = response_transformation_matrix
        self.reference_transformation_matrix = excitation_transformation_matrix
        self.specification_data = specification_data

    @property
    def ramp_samples(self):
        """Number of samples to ramp down to zero when aborting a test"""
        return int(self.test_level_ramp_time * self.sample_rate)

    @property
    def number_of_channels(self):
        """Total number of channels in the environment"""
        return self._number_of_channels

    @number_of_channels.setter
    def number_of_channels(self, value):
        """Sets the total number of channels in the environment"""
        self._number_of_channels = value

    @property
    def response_channel_indices(self):
        """Indices identifying which channels are control channels"""
        return self.control_channel_indices

    @property
    def reference_channel_indices(self):
        """Indices identifying which channels are reference or excitation channels"""
        return self.output_channel_indices

    @property
    def response_transformation_matrix(self):
        """Transformation matrix applied to the control channels"""
        return self._response_transformation_matrix

    @response_transformation_matrix.setter
    def response_transformation_matrix(self, value):
        """Sets the transformation matrix for the control channels"""
        self._response_transformation_matrix = value

    @property
    def reference_transformation_matrix(self):
        """Transformation matrix applied to the excitation channels"""
        return self._reference_transformation_matrix

    @reference_transformation_matrix.setter
    def reference_transformation_matrix(self, value):
        """Sets the transformation matrix applied to the excitation channels"""
        self._reference_transformation_matrix = value

    @property
    def sample_rate(self):
        """Gets the sample rate of the data acquisition system"""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        """Sets the sample rate of the data acquisition system"""
        self._sample_rate = value

    def store_to_netcdf(
        self, netcdf_group_handle: nc4._netCDF4.Group  # pylint: disable=c-extension-no-member
    ):
        """Stores the metadata in a netcdf group

        Parameters
        ----------
        netcdf_group_handle : nc4._netCDF4.Group
            A group in a NetCDF4 group defining the environment's medatadata
        """
        super().store_to_netcdf(netcdf_group_handle)
        # netcdf_group_handle.control_python_script = self.control_python_script
        # netcdf_group_handle.control_python_function = self.control_python_function
        # netcdf_group_handle.control_python_function_type = self.control_python_function_type
        # netcdf_group_handle.control_python_function_parameters = (
        #     self.control_python_function_parameters
        # )
        # Save the output signal
        netcdf_group_handle.createDimension("control_channels", len(self.control_channel_indices))
        if self.response_transformation_matrix is None:
            netcdf_group_handle.createDimension(
                "specification_channels", len(self.control_channel_indices)
            )
        else:
            netcdf_group_handle.createDimension(
                "specification_channels", self.response_transformation_matrix.shape[0]
            )
        # Control Channels
        var = netcdf_group_handle.createVariable(
            "control_channel_indices", "i4", ("control_channels")
        )
        var[...] = self.control_channel_indices
        # Transformation Matrix
        if self.response_transformation_matrix is not None:
            var = netcdf_group_handle.createVariable(
                "response_transformation_matrix",
                "f8",
                ("specification_channels", "control_channels"),
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


# %% UI

from .abstract_interactive_control_law import (  # noqa: E402 pylint: disable=wrong-import-position
    AbstractControlLawComputation,
)
from .abstract_sysid_data_analysis import (  # noqa: E402 pylint: disable=wrong-import-position
    sysid_data_analysis_process,
)
from .data_collector import (  # noqa: E402 pylint: disable=wrong-import-position
    FrameBuffer,
    data_collector_process,
)
from .signal_generation import (  # noqa: E402 pylint: disable=wrong-import-position
    TransientSignalGenerator,
)
from .signal_generation_process import (  # noqa: E402 pylint: disable=wrong-import-position
    SignalGenerationCommands,
    SignalGenerationMetadata,
    signal_generation_process,
)
from .spectral_processing import (  # noqa: E402 pylint: disable=wrong-import-position
    spectral_processing_process,
)


class ShockUI(AbstractSysIdUI):
    """Class defining the user interface for the Shock environment"""

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

    def connect_callbacks(self):
        """Connects the callbacks to the Shock UI widgets"""
        # Definition
        # Prediction
        # Run Test

    # %% Data Acquisition

    def initialize_data_acquisition(self, data_acquisition_parameters):
        super().initialize_data_acquisition(data_acquisition_parameters)

    @property
    def physical_output_names(self):
        """Names of the physical drive channels"""
        return [self.physical_channel_names[i] for i in self.physical_output_indices]

    # %% Environment

    @property
    def physical_control_indices(self):
        """Indices of the control channels"""
        return [
            i
            for i in range(self.definition_widget.control_channels_selector.count())
            if self.definition_widget.control_channels_selector.item(i).checkState() == Qt.Checked
        ]

    @property
    def physical_control_names(self):
        """Names of the selected control channels"""
        return [self.physical_channel_names[i] for i in self.physical_control_indices]

    @property
    def initialized_control_names(self):
        """Names of the control channels that have been initialized"""
        if self.environment_parameters.response_transformation_matrix is None:
            return [
                self.physical_channel_names[i]
                for i in self.environment_parameters.control_channel_indices
            ]
        else:
            return [
                f"Transformed Response {i + 1}"
                for i in range(self.environment_parameters.response_transformation_matrix.shape[0])
            ]

    @property
    def initialized_output_names(self):
        """Names of the drive channels that have been initialized"""
        if self.environment_parameters.reference_transformation_matrix is None:
            return self.physical_output_names
        else:
            return [
                f"Transformed Drive {i + 1}"
                for i in range(self.environment_parameters.reference_transformation_matrix.shape[0])
            ]

    def collect_environment_definition_parameters(self):
        """Collects the metadata defining the environment from the UI widgets"""
        if self.python_control_module is None:
            control_module = None
            control_function = None
            control_function_type = None
            control_function_parameters = None
        else:
            control_module = self.definition_widget.control_script_file_path_input.text()
            control_function = self.definition_widget.control_function_input.itemText(
                self.definition_widget.control_function_input.currentIndex()
            )
            control_function_type = (
                self.definition_widget.control_function_generator_selector.currentIndex()
            )
            control_function_parameters = (
                self.definition_widget.control_parameters_text_input.toPlainText()
            )
        return ShockMetadata()

    def initialize_environment(self):
        super().initialize_environment()
        return self.environment_parameters

    # %% Predictions

    def show_max_voltage_prediction(self):
        """Callback to find and plot the time history showing the maximum drive voltage required"""
        widget = self.prediction_widget.excitation_voltage_list
        index = np.argmax([float(widget.item(v).text()) for v in range(widget.count())])
        self.prediction_widget.excitation_selector.setCurrentIndex(index)

    def show_min_voltage_prediction(self):
        """Callback to find and plot the time history showing the minimum drive voltage required"""
        widget = self.prediction_widget.excitation_voltage_list
        index = np.argmin([float(widget.item(v).text()) for v in range(widget.count())])
        self.prediction_widget.excitation_selector.setCurrentIndex(index)

    def show_max_error_prediction(self):
        """Callback to find and plot the time history with the largest error compared to spec"""
        widget = self.prediction_widget.response_error_list
        index = np.argmax([float(widget.item(v).text()) for v in range(widget.count())])
        self.prediction_widget.response_selector.setCurrentIndex(index)

    def show_min_error_prediction(self):
        """Callback to find and plot the time history with the smallest error compared to spec"""
        widget = self.prediction_widget.response_error_list
        index = np.argmin([float(widget.item(v).text()) for v in range(widget.count())])
        self.prediction_widget.response_selector.setCurrentIndex(index)

    def update_response_error_prediction_selector(self, item):
        """Callback to update the response prediction selector when an item is doubleclicked"""
        index = self.prediction_widget.response_error_list.row(item)
        self.prediction_widget.response_selector.setCurrentIndex(index)

    def update_excitation_prediction_selector(self, item):
        """Callback to update the drive predition selector when an item is doubleclicked"""
        index = self.prediction_widget.excitation_voltage_list.row(item)
        self.prediction_widget.excitation_selector.setCurrentIndex(index)

    def recompute_predictions(self):
        """Recomputes the control predictions"""
        self.environment_command_queue.put(
            self.log_name, (ShockCommands.PERFORM_CONTROL_PREDICTION, False)
        )

    # %% Control

    def start_control(self):
        """Starts the chain of events to start the environment"""

    def stop_control(self):
        """Starts the sequence of events to stop the controller prematurely"""
        self.environment_command_queue.put(self.log_name, (ShockCommands.STOP_CONTROL, None))

    def enable_control(self, enabled):
        """Enables or disables the buttons to start control if it's already running"""

    def change_test_level_from_profile(self, test_level):
        """Updates the test level based on a profile event"""
        self.run_widget.test_level_selector.setValue(int(test_level))

    # %% Misc

    def retrieve_metadata(self, netcdf_handle=None, environment_name=None):
        group = super().retrieve_metadata(netcdf_handle, environment_name)

    def update_gui(self, queue_data):
        if super().update_gui(queue_data):
            return

    def set_parameters_from_template(self, worksheet):
        pass

    @staticmethod
    def create_environment_template(environment_name, workbook):

        worksheet.cell(22, 1, "Response Transformation Matrix:")
        worksheet.cell(
            22,
            2,
            "# Transformation matrix to apply to the response channels.  Type None if there "
            "is none.  Otherwise, make this a 2D array in the spreadsheet and move the Output "
            "Transformation Matrix line down so it will fit.  The number of columns should be "
            "the number of physical control channels.",
        )
        worksheet.cell(23, 1, "Output Transformation Matrix:")
        worksheet.cell(
            23,
            2,
            "# Transformation matrix to apply to the outputs.  Type None if there is none.  "
            "Otherwise, make this a 2D array in the spreadsheet.  The number of columns should "
            "be the number of physical output channels in the environment.",
        )


# %% Environment


class ShockEnvironment(AbstractSysIdEnvironment):
    """Class defining calculations for the Shock environment"""

    def __init__(
        self,
        environment_name: str,
        queue_container: ShockQueues,
        acquisition_active: mp.sharedctypes.Synchronized,
        output_active: mp.sharedctypes.Synchronized,
    ):
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
        self.map_command(
            ShockCommands.PERFORM_CONTROL_PREDICTION,
            self.perform_control_prediction,
        )
        self.map_command(ShockCommands.START_CONTROL, self.start_control)
        self.map_command(ShockCommands.STOP_CONTROL, self.stop_environment)
        # self.map_command(
        #     GlobalCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS,
        #     self.update_interactive_control_parameters,
        # )
        # self.map_command(GlobalCommands.SEND_INTERACTIVE_COMMAND, self.send_interactive_command)

    def initialize_environment_test_parameters(self, environment_parameters: ShockMetadata):
        super().initialize_environment_test_parameters(environment_parameters)
        self.environment_parameters: ShockMetadata

    def system_id_complete(self, data):
        """Sends the message that system identification is complete and control calculations
        should be performed"""
        super().system_id_complete(data)
        (
            self.frames,
            _,  # avg,
            self.frequencies,
            self.frf,
            self.sysid_coherence,
            self.sysid_response_cpsd,
            self.sysid_reference_cpsd,
            self.sysid_condition,
            self.sysid_response_noise,
            self.sysid_reference_noise,
        ) = data
        # Perform the control prediction
        self.perform_control_prediction(True)

    def perform_control_prediction(self, sysid_update):
        """Performs the control prediction based on system identification information"""
        self.show_test_prediction()

    def show_test_prediction(self):
        """Sends the test predictions to the UI"""

    def get_signal_generation_metadata(self):
        """Collects the metadata required to define the signal generation process"""
        return SignalGenerationMetadata(
            samples_per_write=self.data_acquisition_parameters.samples_per_write,
            level_ramp_samples=self.environment_parameters.test_level_ramp_time
            * self.environment_parameters.sample_rate
            * self.data_acquisition_parameters.output_oversample,
            output_transformation_matrix=self.environment_parameters.reference_transformation_matrix,
        )

    def start_control(self, data):
        """Starts up the control to generate the signal"""
        if self.startup:
            pass

    def shutdown(self):
        """Let the UI know that this environment has completely shut down"""
        self.log("Environment Shut Down")
        self.gui_update_queue.put((self.environment_name, ("enable_control", None)))
        self.startup = True

    def stop_environment(self, data):
        """Starts the shutdown sequence based on commands from the UI"""
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.START_SHUTDOWN, None)
        )


# %% Process


def shock_process(
    environment_name: str,
    input_queue: VerboseMessageQueue,
    gui_update_queue: Queue,
    controller_communication_queue: VerboseMessageQueue,
    log_file_queue: Queue,
    data_in_queue: Queue,
    data_out_queue: Queue,
    acquisition_active: mp.sharedctypes.Synchronized,
    output_active: mp.sharedctypes.Synchronized,
):
    """
    Shock environment process function called by multiprocessing

    This function defines the Shock Environment process that
    gets run by the multiprocessing module when it creates a new process.  It
    creates a ShockEnvironment object and runs it.

    Parameters
    ----------
    environment_name : str :
        Name of the environment
    input_queue : VerboseMessageQueue :
        Queue containing instructions for the environment
    gui_update_queue : Queue :
        Queue where GUI updates are put
    controller_communication_queue : Queue :
        Queue for global communications with the controller
    log_file_queue : Queue :
        Queue for writing log file messages
    data_in_queue : Queue :
        Queue from which data will be read by the environment
    data_out_queue : Queue :
        Queue to which data will be written that will be output by the hardware.
    acquisition_active : mp.sharedctypes.Synchronized
        A synchronized value that indicates when the acquisition is active
    output_active : mp.sharedctypes.Synchronized
        A synchronized value that indicates when the output is active
    """
    try:
        # Create vibration queues
        queue_container = ShockQueues(
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

        process_class = ShockEnvironment(
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
    except Exception:  # pylint: disable = broad-exception-caught
        print(traceback.format_exc())
