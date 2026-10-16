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

import inspect
import threading
import multiprocessing as mp
import multiprocessing.sharedctypes  # pylint: disable=unused-import
import time
from enum import Enum
from multiprocessing.queues import Queue
from typing import List

import netCDF4 as nc4
import numpy as np
import openpyxl
import scipy.signal as sig

from rattlesnake.environment.random_vibration_sys_id_utilities import (
    load_specification,
)
from rattlesnake.environment.abstract_interactive_control_law import (
    AbstractControlLawComputation,
)
from rattlesnake.environment.environment_utilities import (
    EnvironmentType,
)
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.abstract_environment import (
    EnvironmentInstructions,
    EnvironmentCommands,
)
from rattlesnake.environment.abstract_sysid_environment import (
    SysIdEnvironment,
    SysIdEnvironmentMetadata,
)
from rattlesnake.process.abstract_sysid_data_analysis import SysIdMetadata
from rattlesnake.utilities import (
    GlobalCommands,
    RattlesnakeError,
    VerboseMessageQueue,
    db2scale,
    load_python_module,
    worksheet_cell_str,
    _direction_map,
)
from rattlesnake.environment.abstract_interactive_control_law import ControlLawCommands
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
from rattlesnake.user_interface.ui_utilities import UICommands

CONTROL_TYPE = EnvironmentType.RANDOM


# region Commands
class RandomVibrationCommands(EnvironmentCommands):
    """Valid random vibration commands"""

    ADJUST_TEST_LEVEL = 0
    CHECK_FOR_COMPLETE_SHUTDOWN = 1
    RECOMPUTE_PREDICTION = 2
    CHANGE_SPECIFICATION = 3
    SAVE_CONTROL_DATA = 4

    VALID_PROFILE_COMMANDS = (
        ADJUST_TEST_LEVEL,
        CHANGE_SPECIFICATION,
        SAVE_CONTROL_DATA,
    )
    VALID_DATA = {
        ADJUST_TEST_LEVEL: float,
        CHANGE_SPECIFICATION: str,
        SAVE_CONTROL_DATA: str,
    }


class RandomVibrationUICommands(Enum):
    ENABLE_CONTROL = 0
    CHANGE_SPECIFICATION = 1
    ADJUST_TEST_LEVEL = 2


# endregion

# region Instructions
class RandomVibrationInstructions(EnvironmentInstructions):
    def __init__(self, environment_name, control_test_level):
        super().__init__(CONTROL_TYPE, environment_name)
        self.control_test_level = control_test_level

    def validate(self):
        return super().validate()


# endregion


# region Queues
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
            log_file_queue,
            mp.Queue(),
            environment_name + " Data Analysis Command Queue",
        )
        self.signal_generation_command_queue = VerboseMessageQueue(
            log_file_queue,
            mp.Queue(),
            environment_name + " Signal Generation Command Queue",
        )
        self.spectral_command_queue = VerboseMessageQueue(
            log_file_queue,
            mp.Queue(),
            environment_name + " Spectral Computation Command Queue",
        )
        self.collector_command_queue = VerboseMessageQueue(
            log_file_queue,
            mp.Queue(),
            environment_name + " Data Collector Command Queue",
        )
        self.controller_communication_queue = controller_communication_queue
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.data_for_spectral_computation_queue = mp.Queue()
        self.updated_spectral_quantities_queue = mp.Queue()
        self.cpsd_to_generate_queue = mp.Queue()
        self.log_file_queue = log_file_queue


# endregion

from rattlesnake.process.random_vibration_sys_id_data_analysis import (  # noqa: E402 pylint: disable=wrong-import-position
    RandomVibrationDataAnalysisCommands,
    random_data_analysis_process,
)


# region Environment
class RandomVibrationEnvironment(SysIdEnvironment):
    """Random Environment class defining the interface with the controller"""

    def __init__(
        self,
        environment_name: str,
        queue_name: str,
        queue_container: RandomVibrationQueues,
        acquisition_active_event: mp.synchronize.Event,
        output_active_event: mp.synchronize.Event,
        active_event: mp.synchronize.Event,
        ready_event: mp.synchronize.Event,
        sysid_active_event: mp.synchronize.Event,
        sysid_stored_event: mp.synchronize.Event,
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
            queue_name,
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
            acquisition_active_event,
            output_active_event,
            active_event,
            ready_event,
            sysid_active_event,
            sysid_stored_event,
        )
        self.map_command(GlobalCommands.START_ENVIRONMENT, self.start_control)
        self.map_command(
            RandomVibrationDataAnalysisCommands.STOP_CONTROL, self.stop_environment
        )
        self.map_command(
            RandomVibrationCommands.ADJUST_TEST_LEVEL, self.adjust_test_level
        )
        self.map_command(
            RandomVibrationCommands.SAVE_CONTROL_DATA, self.save_spectral_data
        )
        self.map_command(
            RandomVibrationCommands.CHANGE_SPECIFICATION, self.change_specification
        )
        self.map_command(
            RandomVibrationCommands.CHECK_FOR_COMPLETE_SHUTDOWN,
            self.check_for_control_shutdown,
        )
        self.map_command(
            RandomVibrationCommands.RECOMPUTE_PREDICTION, self.recompute_prediction
        )
        self.map_command(
            ControlLawCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS,
            self.update_interactive_control_parameters,
        )
        self.map_command(
            ControlLawCommands.SEND_INTERACTIVE_COMMAND, self.send_interactive_command
        )
        self.queue_container = queue_container

        self.set_ready()

    # region StateSync
    def initialize_hardware(self, hardware_metadata):
        super().initialize_hardware(hardware_metadata)

        self.set_ready()

    def initialize_environment(self, environment_metadata: RandomVibrationMetadata):
        super().initialize_environment(environment_metadata)

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

        self.set_ready()

    def initialize_sysid(self, sysid_metadata):
        super().initialize_sysid(sysid_metadata)

        self.set_ready()

    def update_interactive_control_parameters(self, parameters):
        """Sends updated parameters to the interactive control law on the data analysis process"""
        self.queue_container.data_analysis_command_queue.put(
            self.environment_name,
            (ControlLawCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS, parameters),
        )

    def get_data_collector_metadata(self):
        """Gets relevant metadata for the data collector process"""
        num_channels = self.environment_metadata.number_of_channels
        response_channel_indices = self.environment_metadata.response_channel_indices
        reference_channel_indices = self.environment_metadata.reference_channel_indices
        acquisition_type = AcquisitionType.FREE_RUN
        acceptance = Acceptance.AUTOMATIC
        acceptance_function = None
        overlap_fraction = self.environment_metadata.cpsd_overlap
        trigger_channel_index = 0
        trigger_slope = TriggerSlope.POSITIVE
        trigger_level = 0
        trigger_hysteresis = 0
        trigger_hysteresis_samples = 0
        pretrigger_fraction = 0
        frame_size = self.environment_metadata.samples_per_frame
        window = (
            Window.HANN if self.environment_metadata.cpsd_window == "Hann" else None
        )
        # use number of sysid averages as kurtosis buffer size
        # (could maybe make this match the test duration if user is using the "Time at Level"
        # function, would need to pass info from the RandomVibrationUI object)
        kurtosis_buffer_length = self.environment_metadata.sysid_metadata.sysid_averages

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
            response_transformation_matrix=self.environment_metadata.response_transformation_matrix,
            reference_transformation_matrix=self.environment_metadata.reference_transformation_matrix,
        )

    def get_signal_generation_metadata(self):
        """Gets relevant metadata for the signal generation process"""
        return SignalGenerationMetadata(
            samples_per_write=self.hardware_metadata.samples_per_write,
            level_ramp_samples=self.environment_metadata.test_level_ramp_time
            * self.environment_metadata.sample_rate
            * self.hardware_metadata.output_oversample,
            output_transformation_matrix=self.environment_metadata.reference_transformation_matrix,
        )

    def get_signal_generator(self):
        """Gets the signal generator object that will generate signals for the environment"""
        return CPSDSignalGenerator(
            self.environment_metadata.sample_rate,
            self.environment_metadata.samples_per_frame,
            self.environment_metadata.num_reference_channels,
            None,
            self.environment_metadata.cola_overlap,
            self.environment_metadata.cola_window,
            self.environment_metadata.cola_window_exponent,
            self.environment_metadata.sigma_clip,
            self.hardware_metadata.output_oversample,
        )

    def get_spectral_processing_metadata(self):
        """Gets the required metadata for the spectral processing process"""
        averaging_type = AveragingTypes.LINEAR
        averages = self.environment_metadata.frames_in_cpsd
        exponential_averaging_coefficient = 0
        if self.environment_metadata.sysid_metadata.sysid_estimator == "H1":
            frf_estimator = Estimator.H1
        elif self.environment_metadata.sysid_metadata.sysid_estimator == "H2":
            frf_estimator = Estimator.H2
        elif self.environment_metadata.sysid_metadata.sysid_estimator == "H3":
            frf_estimator = Estimator.H3
        elif self.environment_metadata.sysid_metadata.sysid_estimator == "Hv":
            frf_estimator = Estimator.HV
        else:
            raise ValueError(
                f"Invalid FRF Estimator {self.environment_metadata.sysid_metadata.sysid_estimator}"
            )
        num_response_channels = self.environment_metadata.num_response_channels
        num_reference_channels = self.environment_metadata.num_reference_channels
        frequency_spacing = self.environment_metadata.frequency_spacing
        sample_rate = self.environment_metadata.sample_rate
        num_frequency_lines = self.environment_metadata.fft_lines
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

    # endregion

    # region Commands
    def adjust_test_level(self, data):
        """Adjusts the test level of the environment to the specified level"""
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.ADJUST_TEST_LEVEL, db2scale(data)),
        )
        self.queue_container.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.SET_TEST_LEVEL,
                (self.environment_metadata.skip_frames, db2scale(data)),
            ),
        )
        self.gui_update_queue.put(
            (
                self.environment_name,
                (RandomVibrationUICommands.ADJUST_TEST_LEVEL, data),
            )
        )

    def send_interactive_command(self, command):
        """General method that can be used by an interactive UI object to pass commands and data to
        its corresponding computation object"""
        if self.environment_metadata.control_python_function_type == 3:  # Interactive
            self.queue_container.data_analysis_command_queue.put(
                self.environment_name,
                (ControlLawCommands.SEND_INTERACTIVE_COMMAND, command),
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
        self.set_sysid_stored()

    def recompute_prediction(self, data):  # pylint: disable=unused-argument
        """Sends a signal to the data analysis process to recompute test predictions"""
        self.queue_container.data_analysis_command_queue.put(
            self.environment_name,
            (RandomVibrationDataAnalysisCommands.PERFORM_CONTROL_PREDICTION, None),
        )

    def start_control(self, data: RandomVibrationInstructions):
        """Starts the environment at the specified test level"""
        self.log("Starting Control")
        test_level = db2scale(data.control_test_level)
        self.gui_update_queue.put(
            (
                self.environment_name,
                (UICommands.SET_ENVIRONMENT_INSTRUCTIONS, data),
            )
        )
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False
        self.analysis_shutdown_achieved = False
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
                (self.environment_metadata.skip_frames, test_level),
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
            self.environment_name,
            (SignalGenerationCommands.ADJUST_TEST_LEVEL, test_level),
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
        #      self.environment_metadata))

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

        self.set_active()

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
                (self.environment_metadata.skip_frames * 10, 1),
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

    def save_spectral_data(self, data):
        filename = data
        netcdf_dataset = nc4.Dataset(  # pylint: disable=no-member
            filename, "w", format="NETCDF4", clobber=True
        )
        if self.environment_name not in netcdf_dataset.groups:
            netcdf_handle = netcdf_dataset.createGroup(self.environment_name)
        else:
            netcdf_handle = netcdf_dataset.groups[self.environment_name]
        self.environment_metadata.save_metadata_to_netcdf(netcdf_handle)
        netcdf_dataset.close()

        self.data_analysis_command_queue.put(
            self.environment_name,
            (RandomVibrationDataAnalysisCommands.SAVE_CONTROL_DATA, filename),
        )

    def change_specification(self, data):
        """
        Loads in a new specification and starts controlling to it

        Parameters
        ----------
        new_specification_file : str
            File path to a new specification file

        """
        filename = data
        new_metadata = self.environment_metadata
        new_metadata.load_specification(self.hardware_metadata.channel_list, filename)
        self.initialize_environment(new_metadata)

        self.gui_update_queue.put(
            (
                self.environment_name,
                (
                    RandomVibrationUICommands.CHANGE_SPECIFICATION,
                    (filename, new_metadata),
                ),
            )
        )

    # endregion

    # region Shutdown
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
                (
                    self.environment_name,
                    (RandomVibrationUICommands.ENABLE_CONTROL, None),
                )
            )
            self.clear_active()
        else:
            # Recheck some time later
            time.sleep(1)
            self.environment_command_queue.put(
                self.environment_name,
                (RandomVibrationCommands.CHECK_FOR_COMPLETE_SHUTDOWN, None),
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

    # endregion


# endregion


# region Process
def random_vibration_process(
    environment_name: str,
    queue_name: str,
    input_queue: VerboseMessageQueue,
    gui_update_queue: mp.Queue,
    controller_command_queue: VerboseMessageQueue,
    log_file_queue: mp.Queue,
    data_in_queue: mp.Queue,
    data_out_queue: mp.Queue,
    acquisition_active_event: mp.synchronize.Event,
    output_active_event: mp.synchronize.Event,
    active_event: mp.synchronize.Event,
    ready_event: mp.synchronize.Event,
    shutdown_event: mp.synchronize.Event,
    sysid_active_event: mp.synchronize.Event,
    sysid_stored_event: mp.synchronize.Event,
    ping_alive_event: mp.synchronize.Event,
    threaded: bool,
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
    if threaded:
        new_process = threading.Thread  # worker threads
    else:
        new_process = mp.Process  # worker processes
    queue_container = RandomVibrationQueues(
        environment_name,
        input_queue,
        gui_update_queue,
        controller_command_queue,
        data_in_queue,
        data_out_queue,
        log_file_queue,
    )

    spectral_proc = new_process(
        target=spectral_processing_process,
        args=(
            environment_name,
            queue_container.spectral_command_queue,
            queue_container.data_for_spectral_computation_queue,
            queue_container.updated_spectral_quantities_queue,
            queue_container.environment_command_queue,
            queue_container.gui_update_queue,
            queue_container.log_file_queue,
            shutdown_event,
        ),
    )
    spectral_proc.start()
    analysis_proc = new_process(
        target=random_data_analysis_process,
        args=(
            environment_name,
            queue_container.data_analysis_command_queue,
            queue_container.updated_spectral_quantities_queue,
            queue_container.cpsd_to_generate_queue,
            queue_container.environment_command_queue,
            queue_container.gui_update_queue,
            queue_container.log_file_queue,
            ping_alive_event,
            shutdown_event,
        ),
    )
    analysis_proc.start()
    siggen_proc = new_process(
        target=signal_generation_process,
        args=(
            environment_name,
            queue_container.signal_generation_command_queue,
            queue_container.cpsd_to_generate_queue,
            queue_container.data_out_queue,
            queue_container.environment_command_queue,
            queue_container.log_file_queue,
            queue_container.gui_update_queue,
            shutdown_event,
        ),
    )
    siggen_proc.start()
    collection_proc = new_process(
        target=data_collector_process,
        args=(
            environment_name,
            queue_container.collector_command_queue,
            queue_container.data_in_queue,
            [queue_container.data_for_spectral_computation_queue],
            queue_container.environment_command_queue,
            queue_container.log_file_queue,
            queue_container.gui_update_queue,
            shutdown_event,
        ),
    )

    collection_proc.start()
    process_class = RandomVibrationEnvironment(
        environment_name,
        queue_name,
        queue_container,
        acquisition_active_event,
        output_active_event,
        active_event,
        ready_event,
        sysid_active_event,
        sysid_stored_event,
    )
    process_class.run(shutdown_event)

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


# endregion
