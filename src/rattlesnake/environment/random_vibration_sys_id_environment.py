# -*- coding: utf-8 -*-
"""
This file defines a Random Vibration Environment where a specification is
defined and the controller solves for excitations that will cause the test
article to match the specified response.

This environment has a number of subprocesses, including CPSD and FRF
computation, data analysis, and signal generation.

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

import multiprocessing as mp
import multiprocessing.sharedctypes  # pylint: disable=unused-import
import time
from enum import Enum
from multiprocessing.queues import Queue

import netCDF4 as nc4
import numpy as np

# %% Imports
from rattlesnake.environment.abstract_sysid_environment import (
    AbstractSysIdEnvironment,
    AbstractSysIdMetadata,
)
from rattlesnake.utilities import (
    GlobalCommands,
    VerboseMessageQueue,
)
from rattlesnake.process.data_collector import (
    Acceptance,
    AcquisitionType,
    CollectorMetadata,
    DataCollectorCommands,
    TriggerSlope,
    Window,
    data_collector_process,
)
from rattlesnake.process.signal_generation import (
    CPSDSignalGenerator,
)
from rattlesnake.process.signal_generation_process import (
    SignalGenerationCommands,
    SignalGenerationMetadata,
    signal_generation_process,
)
from rattlesnake.process.spectral_processing import (
    AveragingTypes,
    Estimator,
    SpectralProcessingCommands,
    SpectralProcessingMetadata,
    spectral_processing_process,
)


# %% Commands
class RandomVibrationCommands(Enum):
    """Valid random vibration commands"""

    ADJUST_TEST_LEVEL = 0
    START_CONTROL = 1
    STOP_CONTROL = 2
    CHECK_FOR_COMPLETE_SHUTDOWN = 3
    RECOMPUTE_PREDICTION = 4
    # UPDATE_INTERACTIVE_CONTROL_PARAMETERS = 5


class RandomVibrationUICommands(Enum):
    ENABLE_CONTROL = 0


# region: Queues
class RandomVibrationQueues:
    """A container class for the queues that random vibration will manage."""

    def __init__(
        self,
        environment_name: str,
        environment_command_queue: VerboseMessageQueue,
        gui_update_queue: mp.queues.Queue,
        controller_communication_queue: VerboseMessageQueue,
        data_in_queue: mp.queues.Queue,
        data_out_queue: mp.queues.Queue,
        log_file_queue: VerboseMessageQueue,
    ):
        """A container class for the queues that random vibration will manage.

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
        self.cpsd_to_generate_queue = mp.Queue()
        self.log_file_queue = log_file_queue


# region: Metadata
class RandomVibrationMetadata(AbstractSysIdMetadata):
    """Container to hold the signal processing parameters of the environment"""

    def __init__(
        self,
        number_of_channels,
        sample_rate,
        samples_per_frame,
        test_level_ramp_time,
        cola_window,
        cola_overlap,
        cola_window_exponent,
        sigma_clip,
        update_tf_during_control,
        frames_in_cpsd,
        cpsd_window,
        cpsd_overlap,
        percent_lines_out,
        allow_automatic_aborts,
        control_python_script,
        control_python_function,
        control_python_function_type,
        control_python_function_parameters,
        control_channel_indices,
        output_channel_indices,
        specification_frequency_lines,
        specification_cpsd_matrix,
        specification_warning_matrix,
        specification_abort_matrix,
        response_transformation_matrix,
        output_transformation_matrix,
    ):
        super().__init__()
        self.number_of_channels = number_of_channels
        self.sample_rate = sample_rate
        self.samples_per_frame = samples_per_frame
        self.test_level_ramp_time = test_level_ramp_time
        self.cpsd_overlap = cpsd_overlap
        self.update_tf_during_control = update_tf_during_control
        self.cola_window = cola_window
        self.cola_overlap = cola_overlap
        self.cola_window_exponent = cola_window_exponent
        self.sigma_clip = sigma_clip
        self.frames_in_cpsd = frames_in_cpsd
        self.cpsd_window = cpsd_window
        self.response_transformation_matrix = response_transformation_matrix
        self.reference_transformation_matrix = output_transformation_matrix
        self.control_python_script = control_python_script
        self.control_python_function = control_python_function
        self.control_python_function_type = control_python_function_type
        self.control_python_function_parameters = control_python_function_parameters
        self.control_channel_indices = control_channel_indices
        self.output_channel_indices = output_channel_indices
        self.specification_frequency_lines = specification_frequency_lines
        self.specification_cpsd_matrix = specification_cpsd_matrix
        self.specification_warning_matrix = specification_warning_matrix
        self.specification_abort_matrix = specification_abort_matrix
        self.percent_lines_out = percent_lines_out
        self.allow_automatic_aborts = allow_automatic_aborts

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value

    @property
    def number_of_channels(self):
        return self._number_of_channels

    @number_of_channels.setter
    def number_of_channels(self, value):
        self._number_of_channels = value

    @property
    def reference_channel_indices(self):
        return self.output_channel_indices

    @property
    def response_channel_indices(self):
        return self.control_channel_indices

    @property
    def response_transformation_matrix(self):
        return self._response_transformation_matrix

    @response_transformation_matrix.setter
    def response_transformation_matrix(self, value):
        self._response_transformation_matrix = value

    @property
    def reference_transformation_matrix(self):
        return self._reference_transformation_matrix

    @reference_transformation_matrix.setter
    def reference_transformation_matrix(self, value):
        self._reference_transformation_matrix = value

    @property
    def samples_per_acquire(self):
        """Property returning the samples per acquisition step given the overlap"""
        return int(self.samples_per_frame * (1 - self.cpsd_overlap))

    @property
    def frame_time(self):
        """Property returning the time per measurement frame"""
        return self.samples_per_frame / self.sample_rate

    @property
    def nyquist_frequency(self):
        """Property returning half the sample rate"""
        return self.sample_rate / 2

    @property
    def fft_lines(self):
        """Property returning the frequency lines given the sampling parameters"""
        return self.samples_per_frame // 2 + 1

    @property
    def frequency_spacing(self):
        """Property returning frequency line spacing given the sampling parameters"""
        return self.sample_rate / self.samples_per_frame

    @property
    def samples_per_output(self):
        """Property returning the samples per output given the COLA overlap"""
        return int(self.samples_per_frame * (1 - self.cola_overlap))

    @property
    def overlapped_output_samples(self):
        """Property returning the number of output samples that are overlapped."""
        return self.samples_per_frame - self.samples_per_output

    @property
    def skip_frames(self):
        """Property returning the number of frames to skip when changing levels"""
        return int(
            np.ceil(
                self.test_level_ramp_time
                * self.sample_rate
                / (self.samples_per_frame * (1 - self.cpsd_overlap))
            )
        )

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
        function in the RandomVibrationUI class, which will read parameters from
        the netCDF file to populate the parameters in the user interface.

        Parameters
        ----------
        netcdf_group_handle : nc4._netCDF4.Group
            A reference to the Group within the netCDF dataset where the
            environment's metadata is stored.

        """
        super().store_to_netcdf(netcdf_group_handle)
        netcdf_group_handle.samples_per_frame = self.samples_per_frame
        netcdf_group_handle.test_level_ramp_time = self.test_level_ramp_time
        netcdf_group_handle.cpsd_overlap = self.cpsd_overlap
        netcdf_group_handle.update_tf_during_control = 1 if self.update_tf_during_control else 0
        netcdf_group_handle.cola_window = self.cola_window
        netcdf_group_handle.cola_overlap = self.cola_overlap
        netcdf_group_handle.cola_window_exponent = self.cola_window_exponent
        netcdf_group_handle.frames_in_cpsd = self.frames_in_cpsd
        netcdf_group_handle.cpsd_window = self.cpsd_window
        netcdf_group_handle.control_python_script = self.control_python_script
        netcdf_group_handle.control_python_function = self.control_python_function
        netcdf_group_handle.control_python_function_type = self.control_python_function_type
        netcdf_group_handle.control_python_function_parameters = (
            self.control_python_function_parameters
        )
        netcdf_group_handle.allow_automatic_aborts = 1 if self.allow_automatic_aborts else 0
        # Specifications
        netcdf_group_handle.createDimension("fft_lines", self.fft_lines)
        netcdf_group_handle.createDimension("two", 2)
        netcdf_group_handle.createDimension(
            "specification_channels", self.specification_cpsd_matrix.shape[-1]
        )
        var = netcdf_group_handle.createVariable(
            "specification_frequency_lines", "f8", ("fft_lines",)
        )
        var[...] = self.specification_frequency_lines
        var = netcdf_group_handle.createVariable(
            "specification_cpsd_matrix_real",
            "f8",
            ("fft_lines", "specification_channels", "specification_channels"),
        )
        var[...] = self.specification_cpsd_matrix.real
        var = netcdf_group_handle.createVariable(
            "specification_cpsd_matrix_imag",
            "f8",
            ("fft_lines", "specification_channels", "specification_channels"),
        )
        var[...] = self.specification_cpsd_matrix.imag
        var = netcdf_group_handle.createVariable(
            "specification_warning_matrix",
            "f8",
            ("two", "fft_lines", "specification_channels"),
        )
        var[...] = self.specification_warning_matrix.real
        var = netcdf_group_handle.createVariable(
            "specification_abort_matrix",
            "f8",
            ("two", "fft_lines", "specification_channels"),
        )
        var[...] = self.specification_abort_matrix.real
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
        # Control channels
        netcdf_group_handle.createDimension("control_channels", len(self.control_channel_indices))
        var = netcdf_group_handle.createVariable(
            "control_channel_indices", "i4", ("control_channels")
        )
        var[...] = self.control_channel_indices


from rattlesnake.process.random_vibration_sys_id_data_analysis import (  # noqa: E402 pylint: disable=wrong-import-position
    RandomVibrationDataAnalysisCommands,
    random_data_analysis_process,
)


# region: Environment
class RandomVibrationEnvironment(AbstractSysIdEnvironment):
    """Random Environment class defining the interface with the controller"""

    def __init__(
        self,
        environment_name: str,
        queue_container: RandomVibrationQueues,
        acquisition_active: mp.sharedctypes.Synchronized,
        output_active: mp.sharedctypes.Synchronized,
    ):
        """
        Random Vibration Environment Constructor that fills out the ``command_map``

        Parameters
        ----------
        environment_name : str
            Name of the environment.
        queue_container : RandomVibrationQueues
            Container of queues used by the Random Vibration Environment.

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
        self.map_command(RandomVibrationCommands.START_CONTROL, self.start_control)
        self.map_command(RandomVibrationCommands.STOP_CONTROL, self.stop_environment)
        self.map_command(RandomVibrationCommands.ADJUST_TEST_LEVEL, self.adjust_test_level)
        self.map_command(
            RandomVibrationCommands.CHECK_FOR_COMPLETE_SHUTDOWN,
            self.check_for_control_shutdown,
        )
        self.map_command(RandomVibrationCommands.RECOMPUTE_PREDICTION, self.recompute_prediction)
        self.map_command(
            GlobalCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS,
            self.update_interactive_control_parameters,
        )
        self.map_command(GlobalCommands.SEND_INTERACTIVE_COMMAND, self.send_interactive_command)
        self.queue_container = queue_container

    def initialize_environment_test_parameters(
        self, environment_parameters: RandomVibrationMetadata
    ):
        """
        Initialize the environment parameters specific to this environment

        The environment will recieve parameters defining itself from the
        user interface and must set itself up accordingly.

        Parameters
        ----------
        environment_parameters : RandomVibrationMetadata
            A container containing the parameters defining the environment

        """
        super().initialize_environment_test_parameters(environment_parameters)

        # Set up the collector
        self.queue_container.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.INITIALIZE_COLLECTOR,
                self.get_data_collector_metadata(),
            ),
        )
        # Set up the signal generation
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name,
            (
                SignalGenerationCommands.INITIALIZE_PARAMETERS,
                self.get_signal_generation_metadata(),
            ),
        )
        # Set up the spectral processing
        self.queue_container.spectral_command_queue.put(
            self.environment_name,
            (
                SpectralProcessingCommands.INITIALIZE_PARAMETERS,
                self.get_spectral_processing_metadata(),
            ),
        )
        # Set up the data analysis
        self.queue_container.data_analysis_command_queue.put(
            self.environment_name,
            (
                RandomVibrationDataAnalysisCommands.INITIALIZE_PARAMETERS,
                self.environment_parameters,
            ),
        )

    def update_interactive_control_parameters(self, parameters):
        """Sends updated parameters to the interactive control law on the data analysis process"""
        self.queue_container.data_analysis_command_queue.put(
            self.environment_name,
            (GlobalCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS, parameters),
        )

    def send_interactive_command(self, command):
        """General method that can be used by an interactive UI object to pass commands and data to
        its corresponding computation object"""
        if self.environment_parameters.control_python_function_type == 3:  # Interactive
            self.queue_container.data_analysis_command_queue.put(
                self.environment_name,
                (GlobalCommands.SEND_INTERACTIVE_COMMAND, command),
            )
        else:
            raise ValueError(
                "Received an SEND_INTERACTIVE_COMMAND signal without an interactive control law.  "
                "How did this happen?"
            )

    def system_id_complete(self, data):
        """Triggered when system identification has been completed, starting control predictions"""
        super().system_id_complete(data)
        self.queue_container.data_analysis_command_queue.put(
            self.environment_name,
            (RandomVibrationDataAnalysisCommands.PERFORM_CONTROL_PREDICTION, None),
        )

    def get_data_collector_metadata(self):
        """Gets relevant metadata for the data collector process"""
        num_channels = self.environment_parameters.number_of_channels
        response_channel_indices = self.environment_parameters.response_channel_indices
        reference_channel_indices = self.environment_parameters.reference_channel_indices
        acquisition_type = AcquisitionType.FREE_RUN
        acceptance = Acceptance.AUTOMATIC
        acceptance_function = None
        overlap_fraction = self.environment_parameters.cpsd_overlap
        trigger_channel_index = 0
        trigger_slope = TriggerSlope.POSITIVE
        trigger_level = 0
        trigger_hysteresis = 0
        trigger_hysteresis_samples = 0
        pretrigger_fraction = 0
        frame_size = self.environment_parameters.samples_per_frame
        window = Window.HANN if self.environment_parameters.cpsd_window == "Hann" else None
        # use number of sysid averages as kurtosis buffer size
        # (could maybe make this match the test duration if user is using the "Time at Level"
        # function, would need to pass info from the RandomVibrationUI object)
        kurtosis_buffer_length = self.environment_parameters.sysid_averages

        return CollectorMetadata(
            num_channels,
            response_channel_indices,
            reference_channel_indices,
            acquisition_type,
            acceptance,
            acceptance_function,
            overlap_fraction,
            trigger_channel_index,
            trigger_slope,
            trigger_level,
            trigger_hysteresis,
            trigger_hysteresis_samples,
            pretrigger_fraction,
            frame_size,
            window,
            kurtosis_buffer_length=kurtosis_buffer_length,
            response_transformation_matrix=self.environment_parameters.response_transformation_matrix,
            reference_transformation_matrix=self.environment_parameters.reference_transformation_matrix,
        )

    def get_signal_generation_metadata(self):
        """Gets relevant metadata for the signal generation process"""
        return SignalGenerationMetadata(
            samples_per_write=self.data_acquisition_parameters.samples_per_write,
            level_ramp_samples=self.environment_parameters.test_level_ramp_time
            * self.environment_parameters.sample_rate
            * self.data_acquisition_parameters.output_oversample,
            output_transformation_matrix=self.environment_parameters.reference_transformation_matrix,
        )

    def get_signal_generator(self):
        """Gets the signal generator object that will generate signals for the environment"""
        return CPSDSignalGenerator(
            self.environment_parameters.sample_rate,
            self.environment_parameters.samples_per_frame,
            self.environment_parameters.num_reference_channels,
            None,
            self.environment_parameters.cola_overlap,
            self.environment_parameters.cola_window,
            self.environment_parameters.cola_window_exponent,
            self.environment_parameters.sigma_clip,
            self.data_acquisition_parameters.output_oversample,
        )

    def get_spectral_processing_metadata(self):
        """Gets the required metadata for the spectral processing process"""
        averaging_type = AveragingTypes.LINEAR
        averages = self.environment_parameters.frames_in_cpsd
        exponential_averaging_coefficient = 0
        if self.environment_parameters.sysid_estimator == "H1":
            frf_estimator = Estimator.H1
        elif self.environment_parameters.sysid_estimator == "H2":
            frf_estimator = Estimator.H2
        elif self.environment_parameters.sysid_estimator == "H3":
            frf_estimator = Estimator.H3
        elif self.environment_parameters.sysid_estimator == "Hv":
            frf_estimator = Estimator.HV
        else:
            raise ValueError(f"Invalid FRF Estimator {self.environment_parameters.sysid_estimator}")
        num_response_channels = self.environment_parameters.num_response_channels
        num_reference_channels = self.environment_parameters.num_reference_channels
        frequency_spacing = self.environment_parameters.frequency_spacing
        sample_rate = self.environment_parameters.sample_rate
        num_frequency_lines = self.environment_parameters.fft_lines
        return SpectralProcessingMetadata(
            averaging_type,
            averages,
            exponential_averaging_coefficient,
            frf_estimator,
            num_response_channels,
            num_reference_channels,
            frequency_spacing,
            sample_rate,
            num_frequency_lines,
        )

    def recompute_prediction(self, data):  # pylint: disable=unused-argument
        """Sends a signal to the data analysis process to recompute test predictions"""
        self.queue_container.data_analysis_command_queue.put(
            self.environment_name,
            (RandomVibrationDataAnalysisCommands.PERFORM_CONTROL_PREDICTION, None),
        )

    def start_control(self, data):
        """Starts the environment at the specified test level"""
        self.log("Starting Control")
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False
        self.analysis_shutdown_achieved = False
        self.queue_container.controller_communication_queue.put(
            self.environment_name,
            (GlobalCommands.START_ENVIRONMENT, self.environment_name),
        )
        # Set up the collector
        self.queue_container.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.INITIALIZE_COLLECTOR,
                self.get_data_collector_metadata(),
            ),
        )

        self.queue_container.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.SET_TEST_LEVEL,
                (self.environment_parameters.skip_frames, data),
            ),
        )
        time.sleep(0.01)

        # Set up the signal generation
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
                self.get_signal_generator(),
            ),
        )

        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.MUTE, None)
        )

        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.ADJUST_TEST_LEVEL, data)
        )

        # Tell the collector to start acquiring data
        self.queue_container.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.ACQUIRE, None)
        )

        # Tell the signal generation to start generating signals
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.GENERATE_SIGNALS, None)
        )

        # # Set up the data analysis
        # self.queue_container.data_analysis_command_queue.put(
        #     self.environment_name,
        #     (RandomVibrationDataAnalysisCommands.INITIALIZE_PARAMETERS,
        #      self.environment_parameters))

        # Start the data analysis running
        self.queue_container.data_analysis_command_queue.put(
            self.environment_name,
            (RandomVibrationDataAnalysisCommands.RUN_CONTROL, None),
        )

        # Set up the spectral processing
        self.queue_container.spectral_command_queue.put(
            self.environment_name,
            (
                SpectralProcessingCommands.INITIALIZE_PARAMETERS,
                self.get_spectral_processing_metadata(),
            ),
        )

        # Tell the spectral analysis to clear and start acquiring
        self.queue_container.spectral_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING, None),
        )

        self.queue_container.spectral_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING, None),
        )

    def stop_environment(self, data):
        """Stop the environment gracefully

        This function defines the operations to shut down the environment
        gracefully so there is no hard stop that might damage test equipment
        or parts.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.log("Stopping Control")
        self.queue_container.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.SET_TEST_LEVEL,
                (self.environment_parameters.skip_frames * 10, 1),
            ),
        )
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.START_SHUTDOWN, None)
        )
        self.queue_container.spectral_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.STOP_SPECTRAL_PROCESSING, None),
        )
        self.queue_container.data_analysis_command_queue.put(
            self.environment_name,
            (RandomVibrationDataAnalysisCommands.STOP_CONTROL, None),
        )
        self.queue_container.environment_command_queue.put(
            self.environment_name,
            (RandomVibrationCommands.CHECK_FOR_COMPLETE_SHUTDOWN, None),
        )

    def check_for_control_shutdown(self, data):  # pylint: disable=unused-argument
        """Checks the different processes to see if the controller has shut down gracefully"""
        if (
            self.siggen_shutdown_achieved
            and self.collector_shutdown_achieved
            and self.spectral_shutdown_achieved
            and self.analysis_shutdown_achieved
        ):
            self.log("Shutdown Achieved")
            self.gui_update_queue.put(
                (self.environment_name, (RandomVibrationUICommands.ENABLE_CONTROL, None))
            )
        else:
            # Recheck some time later
            time.sleep(1)
            self.environment_command_queue.put(
                self.environment_name,
                (RandomVibrationCommands.CHECK_FOR_COMPLETE_SHUTDOWN, None),
            )

    def adjust_test_level(self, data):
        """Adjusts the test level of the environment to the specified level"""
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.ADJUST_TEST_LEVEL, data)
        )
        self.queue_container.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.SET_TEST_LEVEL,
                (self.environment_parameters.skip_frames, data),
            ),
        )

    def quit(self, data):
        """Closes down the environment permanently as the software is exiting"""
        for queue in [
            self.queue_container.spectral_command_queue,
            self.queue_container.data_analysis_command_queue,
            self.queue_container.signal_generation_command_queue,
            self.queue_container.collector_command_queue,
        ]:
            queue.put(self.environment_name, (GlobalCommands.QUIT, None))
        # Return true to stop the task
        return True


# %% Process


def random_vibration_process(
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
    """Random vibration environment process function called by multiprocessing

    This function defines the Random Vibration Environment process that
    gets run by the multiprocessing module when it creates a new process.  It
    creates a RandomVibrationEnvironment object and runs it.

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
    # Create vibration queues
    queue_container = RandomVibrationQueues(
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
        target=random_data_analysis_process,
        args=(
            environment_name,
            queue_container.data_analysis_command_queue,
            queue_container.updated_spectral_quantities_queue,
            queue_container.cpsd_to_generate_queue,
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
            queue_container.cpsd_to_generate_queue,
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
    process_class = RandomVibrationEnvironment(
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
