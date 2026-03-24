# -*- coding: utf-8 -*-
"""
This file defines a transient environment that utilizes system
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
import multiprocessing.sharedctypes  # pylint: disable=unused-import
import os
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
from rattlesnake.environment.environment_utilities import ControlTypes
from rattlesnake.utilities import (
    GlobalCommands,
    VerboseMessageQueue,
    align_signals,
    shift_signal,
    trac,
)
from rattlesnake.user_interface.ui_utilities import UICommands
from rattlesnake.process.abstract_sysid_data_analysis import (
    sysid_data_analysis_process,
)
from rattlesnake.process.data_collector import (
    FrameBuffer,
    data_collector_process,
)
from rattlesnake.process.signal_generation import (
    TransientSignalGenerator,
)
from rattlesnake.process.signal_generation_process import (
    SignalGenerationCommands,
    SignalGenerationMetadata,
    signal_generation_process,
)
from rattlesnake.process.spectral_processing import (
    spectral_processing_process,
)

# %% Global Variables
CONTROL_TYPE = ControlTypes.TRANSIENT
BUFFER_SIZE_SAMPLES_PER_READ_MULTIPLIER = 2


# %% Commands
class TransientCommands(Enum):
    """Valid commands for the transient environment"""

    START_CONTROL = 0
    STOP_CONTROL = 1
    PERFORM_CONTROL_PREDICTION = 3
    # UPDATE_INTERACTIVE_CONTROL_PARAMETERS = 4


class TransientUICommands(Enum):
    INTERACTIVE_CONTROL_SYSID_UPDATE = 0
    CONTROL_PREDICTIONS = 1
    TIME_DATA = 2
    CONTROL_DATA = 3
    ENABLE_CONTROL = 4


# region: Queues
class TransientQueues:
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
        """A container class for the queues that transient will manage.

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


# region: Metadata
class TransientMetadata(AbstractSysIdMetadata):
    """Metadata required to define a transient control law in rattlesnake."""

    def __init__(
        self,
        number_of_channels,
        sample_rate,
        control_signal,
        ramp_time,
        control_python_script,
        control_python_function,
        control_python_function_type,
        control_python_function_parameters,
        control_channel_indices,
        output_channel_indices,
        response_transformation_matrix,
        output_transformation_matrix,
    ):
        super().__init__()
        self.number_of_channels = number_of_channels
        self.sample_rate = sample_rate
        self.control_signal = control_signal
        self.test_level_ramp_time = ramp_time
        self.control_python_script = control_python_script
        self.control_python_function = control_python_function
        self.control_python_function_type = control_python_function_type
        self.control_python_function_parameters = control_python_function_parameters
        self.control_channel_indices = control_channel_indices
        self.output_channel_indices = output_channel_indices
        self.response_transformation_matrix = response_transformation_matrix
        self.reference_transformation_matrix = output_transformation_matrix

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

    @property
    def signal_samples(self):
        """Gets the number of samples in the signal that is being controlled to"""
        return self.control_signal.shape[-1]

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
        netcdf_group_handle.test_level_ramp_time = self.test_level_ramp_time
        netcdf_group_handle.control_python_script = self.control_python_script
        netcdf_group_handle.control_python_function = self.control_python_function
        netcdf_group_handle.control_python_function_type = self.control_python_function_type
        netcdf_group_handle.control_python_function_parameters = (
            self.control_python_function_parameters
        )
        # Save the output signal
        netcdf_group_handle.createDimension("control_channels", len(self.control_channel_indices))
        netcdf_group_handle.createDimension("specification_channels", self.control_signal.shape[0])
        netcdf_group_handle.createDimension("signal_samples", self.signal_samples)
        var = netcdf_group_handle.createVariable(
            "control_signal", "f8", ("specification_channels", "signal_samples")
        )
        var[...] = self.control_signal
        # Control Channels
        var = netcdf_group_handle.createVariable(
            "control_channel_indices", "i4", ("control_channels")
        )
        var[...] = self.control_channel_indices
        # Transformation Matrix
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


# region: Environment
class TransientEnvironment(AbstractSysIdEnvironment):
    """Class defining calculations for the transient environment"""

    def __init__(
        self,
        environment_name: str,
        queue_container: TransientQueues,
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
            TransientCommands.PERFORM_CONTROL_PREDICTION,
            self.perform_control_prediction,
        )
        self.map_command(TransientCommands.START_CONTROL, self.start_control)
        self.map_command(TransientCommands.STOP_CONTROL, self.stop_environment)
        self.map_command(
            GlobalCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS,
            self.update_interactive_control_parameters,
        )
        self.map_command(GlobalCommands.SEND_INTERACTIVE_COMMAND, self.send_interactive_command)
        # Persistent data
        self.data_acquisition_parameters = None
        self.environment_parameters = None
        self.queue_container = queue_container
        self.frames = None
        self.frequencies = None
        self.frf = None
        self.sysid_coherence = None
        self.sysid_response_cpsd = None
        self.sysid_reference_cpsd = None
        self.sysid_condition = None
        self.sysid_response_noise = None
        self.sysid_reference_noise = None
        self.control_function_type = None
        self.extra_control_parameters = None
        self.control_function = None
        self.aligned_output = None
        self.aligned_response = None
        self.next_drive = None
        self.predicted_response = None
        self.startup = True
        self.shutdown_flag = False
        self.repeat = False
        self.test_level = 0
        self.control_buffer = None
        self.output_buffer = None
        self.last_signal_found = None
        self.has_sent_interactive_control_transfer_function_results = False
        self.last_interactive_parameters = None

    def initialize_environment_test_parameters(self, environment_parameters: TransientMetadata):
        if (
            self.environment_parameters is None
            or self.environment_parameters.control_signal.shape
            != environment_parameters.control_signal.shape
        ):
            self.frames = None
            self.frequencies = None
            self.frf = None
            self.sysid_coherence = None
            self.sysid_response_cpsd = None
            self.sysid_reference_cpsd = None
            self.sysid_condition = None
            self.sysid_response_noise = None
            self.sysid_reference_noise = None
            self.control_function_type = None
            self.extra_control_parameters = None
            self.control_function = None
            self.aligned_output = None
            self.aligned_response = None
            self.next_drive = None
            self.predicted_response = None
        super().initialize_environment_test_parameters(environment_parameters)
        self.environment_parameters: TransientMetadata
        # Load in the control law
        _, file = os.path.split(environment_parameters.control_python_script)
        file, _ = os.path.splitext(file)
        spec = importlib.util.spec_from_file_location(
            file, environment_parameters.control_python_script
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.control_function_type = environment_parameters.control_python_function_type
        self.extra_control_parameters = environment_parameters.control_python_function_parameters
        if self.control_function_type == 1:  # Generator
            # Get the generator function
            generator_function = getattr(module, environment_parameters.control_python_function)()
            # Get us to the first yield statement
            next(generator_function)
            # Define the control function as the generator's send function
            self.control_function = generator_function.send
        elif self.control_function_type == 2:  # Class
            self.control_function = getattr(module, environment_parameters.control_python_function)(
                self.data_acquisition_parameters.sample_rate,
                self.environment_parameters.control_signal,
                self.data_acquisition_parameters.output_oversample,
                self.extra_control_parameters,  # Required parameters
                self.environment_parameters.sysid_frequency_spacing,  # Frequency Spacing
                self.frf,  # Transfer Functions
                self.sysid_response_noise,  # Noise levels and correlation
                self.sysid_reference_noise,  # from the system identification
                self.sysid_response_cpsd,  # Response levels and correlation
                self.sysid_reference_cpsd,  # from the system identification
                self.sysid_coherence,  # Coherence from the system identification
                self.frames,  # Number of frames in the CPSD and FRF matrices
                self.environment_parameters.sysid_averages,  # Total frames that
                # could be in the CPSD and FRF matrices
                self.aligned_output,  # Last excitation signal for drive-based control
                self.aligned_response,
            )  # Last response signal for error-based correction
        elif self.control_function_type == 3:  # Interactive Class
            control_class = getattr(module, environment_parameters.control_python_function)
            self.control_function = control_class(
                self.environment_name,
                self.gui_update_queue,
                self.data_acquisition_parameters.sample_rate,
                self.environment_parameters.control_signal,
                self.data_acquisition_parameters.output_oversample,
                self.extra_control_parameters,  # Required parameters
                self.environment_parameters.sysid_frequency_spacing,  # Frequency Spacing
                self.frf,  # Transfer Functions
                self.sysid_response_noise,  # Noise levels and correlation
                self.sysid_reference_noise,  # from the system identification
                self.sysid_response_cpsd,  # Response levels and correlation
                self.sysid_reference_cpsd,  # from the system identification
                self.sysid_coherence,  # Coherence from the system identification
                self.frames,  # Number of frames in the CPSD and FRF matrices
                self.environment_parameters.sysid_averages,  # Total frames tha
                # could be in the CPSD and FRF matrices
                self.aligned_output,  # Last excitation signal for drive-based control
                self.aligned_response,
            )  # Last response signal for error-based correction
            self.last_interactive_parameters = None
            self.has_sent_interactive_control_transfer_function_results = False
        else:  # Function
            self.control_function = getattr(module, environment_parameters.control_python_function)

    def update_interactive_control_parameters(self, interactive_control_parameters):
        """Updates the interactive control law based on received parameters"""
        if self.environment_parameters.control_python_function_type == 3:  # Interactive
            self.control_function.update_parameters(interactive_control_parameters)
            self.last_interactive_parameters = interactive_control_parameters
        else:
            raise ValueError(
                "Received an UPDATE_INTERACTIVE_CONTROL_PARAMETERS signal without an "
                "interactive control law.  How did this happen?"
            )

    def send_interactive_command(self, command):
        """General method that can be used by an interactive UI object to pass commands
        and data to its corresponding computation object"""
        if self.environment_parameters.control_python_function_type == 3:  # Interactive
            self.control_function.send_command(command)
        else:
            raise ValueError(
                "Received an SEND_INTERACTIVE_COMMAND signal without an interactive "
                "control law.  How did this happen?"
            )

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
        if self.frf is None:
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
        if self.control_function_type == 1:  # Generator
            output_time_history = self.control_function(
                (
                    self.data_acquisition_parameters.sample_rate,
                    self.environment_parameters.control_signal,
                    self.environment_parameters.sysid_frequency_spacing,
                    self.frf,  # Transfer Functions
                    self.sysid_response_noise,  # Noise levels and correlation
                    self.sysid_reference_noise,  # from the system identification
                    self.sysid_response_cpsd,  # Response levels and correlation
                    self.sysid_reference_cpsd,  # from the system identification
                    self.sysid_coherence,  # Coherence from the system identification
                    self.frames,  # Number of frames in the CPSD and FRF matrices
                    self.environment_parameters.sysid_averages,  # Total frames that could be in
                    #  the CPSD and FRF matrices
                    self.data_acquisition_parameters.output_oversample,
                    self.extra_control_parameters,  # Required parameters
                    self.next_drive,  # Last excitation signal for drive-based control
                    self.predicted_response,  # Last response signal for error correction
                )
            )
        elif self.control_function_type in [2, 3]:  # Class or Interactive Class
            if (
                self.environment_parameters.control_python_function == 2
                or not self.has_sent_interactive_control_transfer_function_results
            ):
                if sysid_update:
                    self.control_function.system_id_update(
                        self.environment_parameters.sysid_frequency_spacing,
                        self.frf,  # Transfer Functions
                        self.sysid_response_noise,  # Noise levels and correlation
                        self.sysid_reference_noise,  # from the system identification
                        self.sysid_response_cpsd,  # Response levels and correlation
                        self.sysid_reference_cpsd,  # from the system identification
                        self.sysid_coherence,  # Coherence from the system identification
                        self.frames,  # Number of frames in the CPSD and FRF matrices
                        self.environment_parameters.sysid_averages,  # Total frames that
                        # could be in the CPSD and FRF matrices
                    )

                if self.environment_parameters.control_python_function_type == 3:
                    self.gui_update_queue.put(
                        (
                            self.environment_name,
                            (
                                TransientUICommands.INTERACTIVE_CONTROL_SYSID_UPDATE,
                                (
                                    self.frf,
                                    self.sysid_response_noise,
                                    self.sysid_reference_noise,
                                    self.sysid_response_cpsd,
                                    self.sysid_reference_cpsd,
                                    self.sysid_coherence,
                                ),
                            ),
                        )
                    )
                    self.has_sent_interactive_control_transfer_function_results = True
            if (
                self.environment_parameters.control_python_function_type == 2
                or self.last_interactive_parameters is not None
            ):
                output_time_history = self.control_function.control(
                    self.next_drive, self.predicted_response
                )
            else:
                self.log("Have not yet received control parameters from interactive control law!")
                output_time_history = None
                return
        else:  # Function
            output_time_history = self.control_function(
                self.data_acquisition_parameters.sample_rate,
                self.environment_parameters.control_signal,
                self.environment_parameters.sysid_frequency_spacing,
                self.frf,  # Transfer Functions
                self.sysid_response_noise,  # Noise levels and correlation
                self.sysid_reference_noise,  # from the system identification
                self.sysid_response_cpsd,  # Response levels and correlation
                self.sysid_reference_cpsd,  # from the system identification
                self.sysid_coherence,  # Coherence from the system identification
                self.frames,  # Number of frames in the CPSD and FRF matrices
                self.environment_parameters.sysid_averages,  # Total frames that could
                # be in the CPSD and FRF matrices
                self.data_acquisition_parameters.output_oversample,
                self.extra_control_parameters,  # Required parameters
                self.next_drive,  # Last excitation signal for drive-based control
                self.predicted_response,  # Last response signal for error correction
            )
        self.next_drive = output_time_history
        self.show_test_prediction()

    def show_test_prediction(self):
        """Sends the test predictions to the UI"""
        # print('Drive Signals {:}'.format(self.next_drive.shape))
        drive_signals = self.next_drive[:, :: self.data_acquisition_parameters.output_oversample]
        impulse_responses = np.moveaxis(np.fft.irfft(self.frf, axis=0), 0, -1)

        self.predicted_response = np.zeros((impulse_responses.shape[0], drive_signals.shape[-1]))

        for i, impulse_response_row in enumerate(impulse_responses):
            for _, (impulse, drive) in enumerate(zip(impulse_response_row, drive_signals)):
                # print('Convolving {:},{:}'.format(i,j))
                self.predicted_response[i, :] += sig.convolve(drive, impulse, "full")[
                    : drive_signals.shape[-1]
                ]

        # print('Response Prediction {:}'.format(self.predicted_response.shape))
        # print('Control Signal {:}'.format(self.environment_parameters.control_signal.shape))
        time_trac = trac(self.predicted_response, self.environment_parameters.control_signal)
        peak_voltages = np.max(np.abs(self.next_drive), axis=-1)
        self.gui_update_queue.put(
            (self.environment_name, ("excitation_voltage_list", peak_voltages))
        )
        self.gui_update_queue.put((self.environment_name, ("response_error_list", time_trac)))
        self.gui_update_queue.put(
            (
                self.environment_name,
                (
                    TransientUICommands.CONTROL_PREDICTIONS,
                    (
                        np.arange(self.environment_parameters.control_signal.shape[-1])
                        / self.data_acquisition_parameters.sample_rate,
                        drive_signals,
                        self.predicted_response,
                        self.environment_parameters.control_signal,
                    ),
                ),
            )
        )

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
            self.test_level, self.repeat = data
            self.log("Starting Environment")
            self.siggen_shutdown_achieved = False
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
                    TransientSignalGenerator(self.next_drive, self.repeat),
                ),
            )
            self.queue_container.signal_generation_command_queue.put(
                self.environment_name,
                (SignalGenerationCommands.SET_TEST_LEVEL, self.test_level),
            )
            # Tell the signal generation to start generating signals
            self.queue_container.signal_generation_command_queue.put(
                self.environment_name, (SignalGenerationCommands.GENERATE_SIGNALS, None)
            )
            # Set up the measurement buffers
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
            self.control_buffer = FrameBuffer(
                n_control_channels,
                0,
                0,
                False,
                0,
                0,
                0,
                self.environment_parameters.control_signal.shape[-1],
                0,
                False,
                False,
                False,
                0,
                buffer_size_frame_multiplier=1
                + (
                    self.data_acquisition_parameters.samples_per_read
                    * BUFFER_SIZE_SAMPLES_PER_READ_MULTIPLIER
                    / self.environment_parameters.control_signal.shape[-1]
                ),
                starting_value=0.0,
            )
            self.output_buffer = FrameBuffer(
                n_output_channels,
                0,
                0,
                False,
                0,
                0,
                0,
                self.environment_parameters.control_signal.shape[-1],
                0,
                False,
                False,
                False,
                0,
                buffer_size_frame_multiplier=1
                + (
                    self.data_acquisition_parameters.samples_per_read
                    * BUFFER_SIZE_SAMPLES_PER_READ_MULTIPLIER
                    / self.environment_parameters.control_signal.shape[-1]
                ),
                starting_value=0.0,
            )
            self.startup = False
        # See if any data has come in
        try:
            acquisition_data, last_acquisition = self.queue_container.data_in_queue.get_nowait()
            if self.last_signal_found is not None:
                self.last_signal_found -= self.data_acquisition_parameters.samples_per_read
            if last_acquisition:
                self.log(
                    f"Acquired Last Data, Signal Generation "
                    f"Shutdown Achieved: {self.siggen_shutdown_achieved}"
                )
            else:
                self.log("Acquired Data")
            scale_factor = 0.0 if self.test_level < 1e-10 else 1 / self.test_level
            control_data = (
                acquisition_data[self.environment_parameters.control_channel_indices] * scale_factor
            )
            if self.environment_parameters.response_transformation_matrix is not None:
                control_data = (
                    self.environment_parameters.response_transformation_matrix @ control_data
                )
            output_data = (
                acquisition_data[self.environment_parameters.output_channel_indices] * scale_factor
            )
            if self.environment_parameters.reference_transformation_matrix is not None:
                output_data = (
                    self.environment_parameters.reference_transformation_matrix @ output_data
                )
            # Add the data to the buffers
            self.control_buffer.add_data(control_data)
            self.output_buffer.add_data(output_data)
            if last_acquisition:
                # Find alignment with the specification via output
                self.log("Aligning signal with specification")
                (
                    self.aligned_output,
                    sample_delay,
                    phase_change,
                    _,
                ) = align_signals(
                    self.output_buffer[:],
                    self.next_drive[:, :: self.data_acquisition_parameters.output_oversample],
                    correlation_threshold=0.5,
                )
            else:
                (
                    self.aligned_output,
                    sample_delay,
                    phase_change,
                    _,
                ) = (None, None, None, None)
            self.queue_container.gui_update_queue.put(
                (
                    self.environment_name,
                    (TransientUICommands.TIME_DATA, (control_data, output_data, sample_delay)),
                )
            )  # Sample_delay will be None if the alignment is not found
            if self.aligned_output is not None:
                self.log(f"Alignment Found at {sample_delay} samples")
                self.aligned_response = shift_signal(
                    self.control_buffer[:],
                    self.environment_parameters.control_signal.shape[-1],
                    sample_delay,
                    phase_change,
                )
                time_trac = trac(self.aligned_response, self.environment_parameters.control_signal)
                self.gui_update_queue.put(
                    (
                        self.environment_name,
                        ("control_response_error_list", time_trac),
                    )
                )
                self.queue_container.gui_update_queue.put(
                    (
                        self.environment_name,
                        (
                            TransientUICommands.CONTROL_DATA,
                            (self.aligned_response, self.aligned_output),
                        ),
                    )
                )
                # Do the next control
                self.log(
                    f"Last Signal Found: {self.last_signal_found}, "
                    f"Current Signal Found: {sample_delay}"
                )
                # We don't want to keep a signal if it starts during the last signal.
                # Multiply by 0.8 to give a little wiggle room in case the
                # last signal wasn't found exactly at the right place.
                if (
                    self.last_signal_found is None
                    or (
                        self.last_signal_found
                        + self.environment_parameters.control_signal.shape[-1] * 0.8
                    )
                    < sample_delay
                ):
                    self.next_drive = self.aligned_output
                    self.predicted_response = self.aligned_response
                    self.log("Computing next signal via control law")
                    self.perform_control_prediction(False)
                    self.last_signal_found = sample_delay
                else:
                    self.log("Signal was found previously, not controlling")
        except mp.queues.Empty:
            last_acquisition = False
        # See if we need to keep going
        if self.siggen_shutdown_achieved and last_acquisition:
            self.shutdown()
        else:
            self.queue_container.environment_command_queue.put(
                self.environment_name, (TransientCommands.START_CONTROL, None)
            )

    def shutdown(self):
        """Let the UI know that this environment has completely shut down"""
        self.log("Environment Shut Down")
        self.gui_update_queue.put(
            (self.environment_name, (TransientUICommands.ENABLE_CONTROL, None))
        )
        self.startup = True

    def stop_environment(self, data):
        """Starts the shutdown sequence based on commands from the UI"""
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.START_SHUTDOWN, None)
        )


# %% Process


def transient_process(
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
    Transient vibration environment process function called by multiprocessing

    This function defines the Transient Vibration Environment process that
    gets run by the multiprocessing module when it creates a new process.  It
    creates a TransientEnvironment object and runs it.

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
        queue_container = TransientQueues(
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

        process_class = TransientEnvironment(
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
