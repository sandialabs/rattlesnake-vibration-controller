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
import multiprocessing as mp
import os
import time
import traceback
from enum import Enum
from multiprocessing.queues import Queue

import netCDF4 as nc4
import numpy as np
import scipy.signal as sig

from rattlesnake.environment.abstract_sysid_environment import (
    AbstractSysIdEnvironment,
    AbstractSysIdMetadata,
)
from rattlesnake.user_interface.ui_utilities import UICommands
from rattlesnake.environment.environment_utilities import ControlTypes
from rattlesnake.environment.sine_sys_id_utilities import (
    DefaultSineControlLaw,
    SineSpecification,
    digital_tracking_filter_generator,
    sine_sweep,
    vold_kalman_filter_generator,
)

from rattlesnake.utilities import (
    VerboseMessageQueue,
    flush_queue,
    scale2db,
    wrap,
)
from rattlesnake.process.abstract_sysid_data_analysis import (  # pylint: disable=wrong-import-position # noqa: E402
    sysid_data_analysis_process,
)
from rattlesnake.process.data_collector import (  # pylint: disable=wrong-import-position # noqa: E402
    data_collector_process,
)
from rattlesnake.process.signal_generation import (  # pylint: disable=wrong-import-position # noqa: E402
    ContinuousTransientSignalGenerator,
)
from rattlesnake.process.signal_generation_process import (  # pylint: disable=wrong-import-position # noqa: E402
    SignalGenerationCommands,
    SignalGenerationMetadata,
    signal_generation_process,
)
from rattlesnake.process.spectral_processing import (  # pylint: disable=wrong-import-position # noqa: E402
    spectral_processing_process,
)

# %% Global Variables
CONTROL_TYPE = ControlTypes.SINE
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


class SineUICommands(Enum):
    SPECIFICATION_FOR_PLOTTING = 0
    REQUEST_PREDICTION_PLOT_CHOICES = 1
    EXCITATION_VOLTAGE_LIST = 2
    RESPONSE_ERROR_MATRIX = 3
    EXCITATION_PREDICTION = 4
    RESPONSE_PREDICTION = 5
    TIME_DATA = 6
    CONTROL_DATA = 7
    ENABLE_CONTROL = 8


# region: Queues
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


# region: Enviornment
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
                    SineUICommands.SPECIFICATION_FOR_PLOTTING,
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
                    UICommands.ERROR,
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
            (self.environment_name, (SineUICommands.REQUEST_PREDICTION_PLOT_CHOICES, None))
        )
        self.gui_update_queue.put(
            (self.environment_name, (SineUICommands.EXCITATION_VOLTAGE_LIST, self.peak_voltages))
        )
        self.gui_update_queue.put(
            (
                self.environment_name,
                (
                    SineUICommands.RESPONSE_ERROR_MATRIX,
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
            (self.environment_name, (SineUICommands.EXCITATION_PREDICTION, (abscissa, ordinate)))
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
            (self.environment_name, (SineUICommands.RESPONSE_PREDICTION, (abscissa, ordinate)))
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
                        SineUICommands.TIME_DATA,
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
                                SineUICommands.CONTROL_DATA,
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
        self.gui_update_queue.put((self.environment_name, (SineUICommands.ENABLE_CONTROL, None)))
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


# region: Process
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
