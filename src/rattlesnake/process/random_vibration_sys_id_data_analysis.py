# -*- coding: utf-8 -*-
"""
This file defines the data analysis and control law for the Random Vibration
Environment

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
from enum import Enum

import numpy as np

from rattlesnake.process.abstract_sysid_data_analysis import (
    AbstractSysIDAnalysisProcess,
    SysIDDataAnalysisCommands,
)
from rattlesnake.environment.random_vibration_sys_id_environment import (
    RandomVibrationCommands,
    RandomVibrationMetadata,
)
from rattlesnake.utilities import (
    GlobalCommands,
    VerboseMessageQueue,
    flush_queue,
    power2db,
    rms_csd,
    rms_time,
)


class RandomVibrationDataAnalysisCommands(Enum):
    """Enumeration containing valid commands for the random data analysis process"""

    INITIALIZE_PARAMETERS = 0
    PERFORM_CONTROL_PREDICTION = 1
    RUN_CONTROL = 2
    STOP_CONTROL = 3
    # SHUTDOWN_ACHIEVED = 4
    # UPDATE_INTERACTIVE_CONTROL_PARAMETERS = 5


class RandomVibrationDataAnalysisUICommands(Enum):
    INTERACTIVE_CONTROL_SYSID_UPDATE = 0
    CONTROL_PREDICTIONS = 1
    CONTROL_UPDATE = 2
    UPDATE_TEST_RESPONSE_ERROR_LIST = 3


class RandomVibrationDataAnalysisProcess(AbstractSysIDAnalysisProcess):
    """Control calculations for the Random Vibration environment"""

    def __init__(
        self,
        process_name: str,
        command_queue: VerboseMessageQueue,
        data_in_queue: mp.queues.Queue,
        data_out_queue: mp.queues.Queue,
        environment_command_queue: VerboseMessageQueue,
        log_file_queue: mp.queues.Queue,
        gui_update_queue: mp.queues.Queue,
        environment_name: str,
    ):
        super().__init__(
            process_name,
            command_queue,
            data_in_queue,
            data_out_queue,
            environment_command_queue,
            log_file_queue,
            gui_update_queue,
            environment_name,
        )
        self.map_command(
            RandomVibrationDataAnalysisCommands.INITIALIZE_PARAMETERS,
            self.initialize_sysid_parameters,
        )
        self.map_command(
            RandomVibrationDataAnalysisCommands.PERFORM_CONTROL_PREDICTION,
            self.perform_control_prediction,
        )
        self.map_command(RandomVibrationDataAnalysisCommands.RUN_CONTROL, self.run_control)
        self.map_command(RandomVibrationDataAnalysisCommands.STOP_CONTROL, self.stop_control)
        self.map_command(
            GlobalCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS,
            self.update_interactive_control_parameters,
        )
        self.map_command(GlobalCommands.SEND_INTERACTIVE_COMMAND, self.send_interactive_command)
        self.error_indices = None
        self.control_function = None
        self.response_cpsd_prediction = None
        self.drive_cpsd_prediction = None
        self.control_frf = None
        self.control_coherence = None
        self.control_frf_condition = None
        self.last_response_cpsd = None
        self.last_drive_cpsd = None
        self.startup = True
        self.has_sent_interactive_control_transfer_function_results = False
        self.last_interactive_parameters = None

    def initialize_sysid_parameters(self, data: RandomVibrationMetadata):
        self.parameters: RandomVibrationMetadata
        super().initialize_sysid_parameters(data)  # This defines self.parameters

        # Find the frequency lines to perform control and compute error over
        # print(type(data.specification_cpsd_matrix))
        # print(data.specification_cpsd_matrix)
        self.error_indices = ~np.all(
            (data.specification_cpsd_matrix == 0) | np.isnan(data.specification_cpsd_matrix),
            axis=(-1, -2),
        )
        self.frequencies = self.parameters.frequency_spacing * np.arange(self.parameters.fft_lines)
        # Load in the control script
        _, file = os.path.split(data.control_python_script)
        file, _ = os.path.splitext(file)
        spec = importlib.util.spec_from_file_location(file, data.control_python_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Pull out the function from the loaded module
        if self.parameters.control_python_function_type == 1:  # Generator
            # Get the generator function
            generator_function = getattr(module, data.control_python_function)()
            # Get us to the first yield statement
            next(generator_function)
            # Define the control function as the generator's send function
            self.control_function = generator_function.send
        elif self.parameters.control_python_function_type == 2:  # Class
            control_frf = (
                self.control_frf if self.parameters.update_tf_during_control else self.sysid_frf
            )
            control_coherence = (
                self.control_coherence
                if self.parameters.update_tf_during_control
                else self.sysid_coherence
            )
            self.control_function = getattr(module, data.control_python_function)(
                data.specification_cpsd_matrix,  # Specifications
                data.specification_warning_matrix,  # Warning levels
                data.specification_abort_matrix,  # Abort Levels
                data.control_python_function_parameters,  # Extra parameters for the control law
                control_frf,  # Transfer Functions
                self.sysid_response_noise,  # Noise levels and correlation
                self.sysid_reference_noise,  # from the system identification
                self.sysid_response_cpsd,  # Response levels and correlation
                self.sysid_reference_cpsd,  # from the system identification
                control_coherence,  # Coherence from the system identification
                self.frames,  # Number of frames in the CPSD
                self.parameters.frames_in_cpsd,  # Total number of frames
                self.last_response_cpsd,  # Last Control Response for Error Correction
                self.last_drive_cpsd,  # Last Control Excitation for Drive-based control
            )
        elif self.parameters.control_python_function_type == 3:  # Interactive
            control_frf = (
                self.control_frf if self.parameters.update_tf_during_control else self.sysid_frf
            )
            control_coherence = (
                self.control_coherence
                if self.parameters.update_tf_during_control
                else self.sysid_coherence
            )
            control_class = getattr(module, data.control_python_function)
            self.control_function = control_class(
                self.environment_name,
                self.gui_update_queue,
                data.specification_cpsd_matrix,  # Specifications
                data.specification_warning_matrix,  # Warning levels
                data.specification_abort_matrix,  # Abort Levels
                data.control_python_function_parameters,  # Extra parameters for the control law
                control_frf,  # Transfer Functions
                self.sysid_response_noise,  # Noise levels and correlation
                self.sysid_reference_noise,  # from the system identification
                self.sysid_response_cpsd,  # Response levels and correlation
                self.sysid_reference_cpsd,  # from the system identification
                control_coherence,  # Coherence from the system identification
                self.frames,  # Number of frames in the CPSD
                self.parameters.frames_in_cpsd,  # Total number of frames
                self.last_response_cpsd,  # Last Control Response for Error Correction
                self.last_drive_cpsd,  # Last Control Excitation for Drive-based control
            )
            self.last_interactive_parameters = None
            self.has_sent_interactive_control_transfer_function_results = False
        else:  # Function
            self.control_function = getattr(module, data.control_python_function)

    def perform_control_prediction(self, data):  # pylint: disable=unused-argument
        """Runs the control law with system identification information to predict control"""
        if self.sysid_frf is None:
            return
        if self.parameters.control_python_function_type == 1:  # Generator
            output_cpsd = self.control_function(
                (
                    self.parameters.specification_cpsd_matrix,  # Specifications
                    self.parameters.specification_warning_matrix,  # Warning levels
                    self.parameters.specification_abort_matrix,  # Abort Levels
                    self.sysid_frf,  # Transfer Functions
                    self.sysid_response_noise,  # Noise levels and correlation
                    self.sysid_reference_noise,  # from the system identification
                    self.sysid_response_cpsd,  # Response levels and correlation
                    self.sysid_reference_cpsd,  # from the system identification
                    self.sysid_coherence,  # Coherence from the system identification
                    self.parameters.sysid_averages,  # Number of frames in the CPSD
                    self.parameters.sysid_averages,  # Total number of frames
                    self.parameters.control_python_function_parameters,  # Extra parameters
                    None,  # Last Control Response for Error Correction
                    None,  # Last Control Excitation for Drive-based control
                )
            )
        elif self.parameters.control_python_function_type in [
            2,
            3,
        ]:  # Class or Interactive
            if (
                self.parameters.control_python_function == 2
                or not self.has_sent_interactive_control_transfer_function_results
            ):
                self.control_function.system_id_update(
                    self.sysid_frf,  # Transfer Functions
                    self.sysid_response_noise,  # Noise levels and correlation
                    self.sysid_reference_noise,  # from the system identification
                    self.sysid_response_cpsd,  # Response levels and correlation
                    self.sysid_reference_cpsd,  # from the system identification
                    self.sysid_coherence,  # Coherence from the system identification
                    self.parameters.sysid_averages,  # Number of frames in the CPSD
                    self.parameters.sysid_averages,  # Total number of frames
                )
                if self.parameters.control_python_function_type == 3:
                    self.gui_update_queue.put(
                        (
                            self.environment_name,
                            (
                                RandomVibrationDataAnalysisUICommands.INTERACTIVE_CONTROL_SYSID_UPDATE,
                                (
                                    self.sysid_frf,  # Transfer Functions
                                    self.sysid_response_noise,  # Noise levels and correlation
                                    self.sysid_reference_noise,  # from the system identification
                                    self.sysid_response_cpsd,  # Response levels and correlation
                                    self.sysid_reference_cpsd,  # from the system identification
                                    self.sysid_coherence,  # Coherence from the system id
                                ),
                            ),
                        )
                    )
                    self.has_sent_interactive_control_transfer_function_results = True
            if (
                self.parameters.control_python_function_type == 2
                or self.last_interactive_parameters is not None
            ):
                output_cpsd = self.control_function.control(
                    self.sysid_frf,  # Transfer Functions
                    self.sysid_coherence,  # Coherence from the system identification
                    self.parameters.sysid_averages,  # Number of frames in the CPSD
                    self.parameters.sysid_averages,  # Total number of frames
                    None,
                    None,
                )
            else:
                self.log("Have not yet received control parameters from interactive control law!")
                self.command_queue.put(
                    self.process_name,
                    (
                        RandomVibrationDataAnalysisCommands.PERFORM_CONTROL_PREDICTION,
                        None,
                    ),
                )
                time.sleep(0.25)
                return
        else:  # Function
            output_cpsd = self.control_function(
                self.parameters.specification_cpsd_matrix,  # Specifications
                self.parameters.specification_warning_matrix,  # Warning levels
                self.parameters.specification_abort_matrix,  # Abort Levels
                self.sysid_frf,  # Transfer Functions
                self.sysid_response_noise,  # Noise levels and correlation
                self.sysid_reference_noise,  # from the system identification
                self.sysid_response_cpsd,  # Response levels and correlation
                self.sysid_reference_cpsd,  # from the system identification
                self.sysid_coherence,  # Coherence from the system identification
                self.parameters.sysid_averages,  # Number of frames in the CPSD
                self.parameters.sysid_averages,  # Total number of frames
                self.parameters.control_python_function_parameters,  # Extra parameters
                None,  # Last Control Response for Error Correction
                None,  # Last Control Excitation for Drive-based control
            )
        response_cpsd = self.sysid_frf @ output_cpsd @ self.sysid_frf.conjugate().transpose(0, 2, 1)
        self.drive_cpsd_prediction = output_cpsd
        self.response_cpsd_prediction = response_cpsd
        rms_drives = rms_csd(self.drive_cpsd_prediction, self.parameters.frequency_spacing)
        response_db_error = power2db(
            np.einsum("ijj->ij", self.response_cpsd_prediction[self.error_indices]).real
        ) - power2db(
            np.einsum("ijj->ij", self.parameters.specification_cpsd_matrix[self.error_indices]).real
        )
        rms_db_error = rms_time(response_db_error, axis=0)
        self.gui_update_queue.put(
            (
                self.environment_name,
                (
                    RandomVibrationDataAnalysisUICommands.CONTROL_PREDICTIONS,
                    (
                        self.frequencies,
                        self.drive_cpsd_prediction,
                        self.response_cpsd_prediction,
                        self.parameters.specification_cpsd_matrix,
                        rms_drives,
                        rms_db_error,
                    ),
                ),
            )
        )
        if self.parameters.control_python_function_type == 3:
            self.has_sent_interactive_control_transfer_function_results = False

    def run_control(self, data):  # pylint: disable=unused-argument
        """Runs the control law to generate new output CPSDs"""
        if self.startup:
            self.log("Starting Control")
            self.frames = 0
            self.data_out_queue.put([self.drive_cpsd_prediction])
            self.startup = False
        spectral_data = flush_queue(self.data_in_queue)
        if len(spectral_data) > 0:
            self.log("Obtained Spectral Data")
            (
                self.frames,
                self.frequencies,
                self.control_frf,
                self.control_coherence,
                self.last_response_cpsd,
                self.last_drive_cpsd,
                self.control_frf_condition,
            ) = spectral_data[-1]
            self.gui_update_queue.put(
                (
                    self.environment_name,
                    (
                        RandomVibrationDataAnalysisUICommands.CONTROL_UPDATE,
                        (
                            self.frames,
                            self.parameters.frames_in_cpsd,
                            self.frequencies,
                            self.control_frf,
                            self.control_coherence,
                            self.last_response_cpsd,
                            self.last_drive_cpsd,
                            self.control_frf_condition,
                        ),
                    ),
                )
            )
            # Check to see if there are any aborts or warnings
            warning_channels = []
            abort_channels = []
            with np.errstate(invalid="ignore"):
                lines_out = (self.parameters.percent_lines_out / 100) * self.parameters.fft_lines
                for i in range(self.last_response_cpsd.shape[-1]):
                    if (
                        sum(
                            self.last_response_cpsd[:, i, i]
                            > self.parameters.specification_abort_matrix[1, :, i]
                        )
                        > lines_out
                    ):
                        abort_channels.append(i)
                    elif (
                        sum(
                            self.last_response_cpsd[:, i, i]
                            < self.parameters.specification_abort_matrix[0, :, i]
                        )
                        > lines_out
                    ):
                        abort_channels.append(i)
                    elif (
                        sum(
                            self.last_response_cpsd[:, i, i]
                            > self.parameters.specification_warning_matrix[1, :, i]
                        )
                        > lines_out
                    ):
                        warning_channels.append(i)
                    elif (
                        sum(
                            self.last_response_cpsd[:, i, i]
                            < self.parameters.specification_warning_matrix[0, :, i]
                        )
                        > lines_out
                    ):
                        warning_channels.append(i)
            if (
                len(abort_channels) > 0
                and self.frames == self.parameters.frames_in_cpsd
                and self.parameters.allow_automatic_aborts
            ):
                print(f"Aborting due to channel indices {abort_channels}")
                self.log(f"Aborting due to channel indices {abort_channels}")
                self.environment_command_queue.put(
                    self.process_name, (RandomVibrationCommands.STOP_CONTROL, None)
                )
            response_db_error = power2db(
                np.einsum("ijj->ij", self.last_response_cpsd[self.error_indices]).real
            ) - power2db(
                np.einsum(
                    "ijj->ij",
                    self.parameters.specification_cpsd_matrix[self.error_indices],
                ).real
            )
            rms_db_error = rms_time(response_db_error, axis=0)
            self.gui_update_queue.put(
                (
                    self.environment_name,
                    (
                        RandomVibrationDataAnalysisUICommands.UPDATE_TEST_RESPONSE_ERROR_LIST,
                        (rms_db_error, warning_channels, abort_channels),
                    ),
                )
            )
            self.log("Controlling")
            # Create the new control output
            control_frf = (
                self.control_frf if self.parameters.update_tf_during_control else self.sysid_frf
            )
            control_coherence = (
                self.control_coherence
                if self.parameters.update_tf_during_control
                else self.sysid_coherence
            )
            if self.parameters.control_python_function_type == 1:  # Generator
                output_cpsd = self.control_function(
                    (
                        self.parameters.specification_cpsd_matrix,  # Specifications
                        self.parameters.specification_warning_matrix,  # Warning levels
                        self.parameters.specification_abort_matrix,  # Abort Levels
                        control_frf,  # Transfer Functions
                        self.sysid_response_noise,  # Noise levels and correlation
                        self.sysid_reference_noise,  # from the system identification
                        self.sysid_response_cpsd,  # Response levels and correlation
                        self.sysid_reference_cpsd,  # from the system identification
                        control_coherence,  # Coherence
                        self.frames,  # Number of frames in the CPSD
                        self.parameters.frames_in_cpsd,  # Total number of frames
                        self.parameters.control_python_function_parameters,  # Extra parameters
                        self.last_response_cpsd,  # Last Control Response for Error Correction
                        self.last_drive_cpsd,  # Last Control Excitation for Drive-based control
                    )
                )
            elif self.parameters.control_python_function_type in [
                2,
                3,
            ]:  # Class or interactive class
                output_cpsd = self.control_function.control(
                    control_frf,  # Transfer Functions
                    control_coherence,  # Coherence
                    self.frames,  # Number of frames in the CPSD
                    self.parameters.frames_in_cpsd,  # Total number of frames
                    self.last_response_cpsd,  # Last Control Response for Error Correction
                    self.last_drive_cpsd,  # Last Control Excitation for Drive-based control
                )
            else:  # Function
                output_cpsd = self.control_function(
                    self.parameters.specification_cpsd_matrix,  # Specifications
                    self.parameters.specification_warning_matrix,  # Warning levels
                    self.parameters.specification_abort_matrix,  # Abort Levels
                    control_frf,  # Transfer Functions
                    self.sysid_response_noise,  # Noise levels and correlation
                    self.sysid_reference_noise,  # from the system identification
                    self.sysid_response_cpsd,  # Response levels and correlation
                    self.sysid_reference_cpsd,  # from the system identification
                    control_coherence,  # Coherence
                    self.frames,  # Number of frames in the CPSD
                    self.parameters.frames_in_cpsd,  # Total number of frames
                    self.parameters.control_python_function_parameters,  # Extra parameters
                    self.last_response_cpsd,  # Last Control Response for Error Correction
                    self.last_drive_cpsd,  # Last Control Excitation for Drive-based control
                )
            self.log(
                f"RMS Outputs from Control \n  "
                f"{rms_csd(output_cpsd, self.parameters.frequency_spacing)}"
            )
            self.data_out_queue.put([output_cpsd])
            self.log("Finished Controlling")
            rms_voltages = rms_csd(output_cpsd, self.parameters.frequency_spacing)
            self.gui_update_queue.put(
                (
                    self.environment_name,
                    ("test_output_voltage_list", rms_voltages),
                )
            )
        self.command_queue.put(
            self.process_name, (RandomVibrationDataAnalysisCommands.RUN_CONTROL, None)
        )

    def update_interactive_control_parameters(self, interactive_control_parameters):
        """Updates parameters for the interactive control law"""
        if self.parameters.control_python_function_type == 3:  # Interactive
            self.control_function.update_parameters(interactive_control_parameters)
            self.last_interactive_parameters = interactive_control_parameters
        else:
            raise ValueError(
                "Received an UPDATE_INTERACTIVE_CONTROL_PARAMETERS signal without an "
                "interactive control law.  How did this happen?"
            )

    def send_interactive_command(self, command):
        """Sends a command to the interactive control law"""
        if self.parameters.control_python_function_type == 3:  # Interactive
            self.control_function.send_command(command)
        else:
            raise ValueError(
                "Received an UPDATE_INTERACTIVE_CONTROL_PARAMETERS signal without an "
                "interactive control law.  How did this happen?"
            )

    def stop_control(self, data):  # pylint: disable=unused-argument
        """Stops the data acquisition process"""
        # Remove any run_transfer_function or run_control from the queue
        instructions = self.command_queue.flush(self.process_name)
        for instruction in instructions:
            if not instruction[0] in [RandomVibrationDataAnalysisCommands.RUN_CONTROL]:
                self.command_queue.put(self.process_name, instruction)
        flush_queue(self.data_out_queue)
        self.startup = True
        self.control_frf = None
        self.control_coherence = None
        self.control_frf_condition = None
        self.last_response_cpsd = None
        self.last_drive_cpsd = None
        self.environment_command_queue.put(
            self.process_name, (SysIDDataAnalysisCommands.SHUTDOWN_ACHIEVED, None)
        )


def random_data_analysis_process(
    environment_name: str,
    command_queue: VerboseMessageQueue,
    data_in_queue: mp.queues.Queue,
    data_out_queue: mp.queues.Queue,
    environment_command_queue: VerboseMessageQueue,
    gui_update_queue: mp.queues.Queue,
    log_file_queue: mp.queues.Queue,
    process_name=None,
):
    """Process defining the random vibration control calculations and data analysis

    Parameters
    ----------
    environment_name : str
        The name of the random vibration environment
    command_queue : VerboseMessageQueue
        A message queue that will provide commands to this process
    data_in_queue : mp.queues.Queue
        A queue from which data will be received to use for control updates
    data_out_queue : mp.queues.Queue
        A queue into which the results of control calculations will be placed
    environment_command_queue : VerboseMessageQueue
        A message queue to send commands back to the main controller
    gui_update_queue : mp.queues.Queue
        A queue to send updates to the graphical user interface for updating interactive control
        laws
    log_file_queue : mp.queues.Queue
        A queue to send messages that will be logged in the log file
    process_name : str, optional
        A name for the process.  If not specified, it will be
    """
    data_analysis_instance = RandomVibrationDataAnalysisProcess(
        environment_name + " Data Analysis" if process_name is None else process_name,
        command_queue,
        data_in_queue,
        data_out_queue,
        environment_command_queue,
        log_file_queue,
        gui_update_queue,
        environment_name,
    )

    data_analysis_instance.run()
