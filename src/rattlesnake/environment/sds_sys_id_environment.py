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

# region Imports
import importlib
import threading
import multiprocessing as mp
import multiprocessing.sharedctypes  # pylint: disable=unused-import
import os
import traceback
from multiprocessing.queues import Queue

import netCDF4 as nc4
import numpy as np
import scipy.signal as sig

from rattlesnake.environment.abstract_sysid_environment import (
    SysIdEnvironment,
)
from rattlesnake.environment.environment_utilities import (
    EnvironmentType,
)
from rattlesnake.environment.sds_sys_id_metadata import (
    SDSMetadata,
    ControlLawType,
    ControlParameters,
)
from rattlesnake.environment.sds_sys_id_utilities import (
    SDSQueues,
    SDSCommands,
    sum_decayed_sines_reconstruction,
    srs as srs_function,
)


from rattlesnake.hardware.abstract_hardware import (
    HardwareMetadata,
)
from rattlesnake.utilities import (
    GlobalCommands,
    VerboseMessageQueue,
    db2scale,
    load_python_module,
)


from rattlesnake.environment.abstract_interactive_control_law import (
    AbstractControlLawComputation,
    ControlLawCommands,
)
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

# region Globals
CONTROL_TYPE = EnvironmentType.SDS
BUFFER_SIZE_SAMPLES_PER_READ_MULTIPLIER = 2

# region Environment Process


class SDSEnvironment(SysIdEnvironment):
    """Class defining calculations for the SDS environment"""

    def __init__(
        self,
        environment_name: str,
        queue_name: str,
        queue_container: SDSQueues,
        acquisition_active_event: mp.synchronize.Event,
        output_active_event: mp.synchronize.Event,
        active_event: mp.synchronize.Event,
        ready_event: mp.synchronize.Event,
        sysid_active_event: mp.synchronize.Event,
        sysid_stored_event: mp.synchronize.Event,
    ):
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
        self.map_command(
            SDSCommands.PERFORM_CONTROL_PREDICTION,
            self.perform_control_prediction,
        )
        self.map_command(
            SDSCommands.SDS_TABLE_PREDICTION,
            self.perform_prediction_table_prediction,
        )
        self.map_command(
            SDSCommands.SDS_RUN_TABLE_PREDICTION,
            self.perform_run_table_prediction,
        )
        self.map_command(SDSCommands.START_CONTROL, self.start_control)
        self.map_command(SDSCommands.STOP_CONTROL, self.stop_environment)
        self.map_command(
            ControlLawCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS,
            self.update_interactive_control_parameters,
        )
        self.map_command(ControlLawCommands.SEND_INTERACTIVE_COMMAND, self.send_interactive_command)

        # Persistent Data
        self.data_acquisition_parameters = None
        self.environment_parameters = None
        self.queue_container = queue_container
        # System ID information
        self.sysid_frames = None
        self.sysid_frequencies = None
        self.sysid_frf = None
        self.sysid_coherence = None
        self.sysid_response_cpsd = None
        self.sysid_reference_cpsd = None
        self.sysid_condition = None
        self.sysid_response_noise = None
        self.sysid_reference_noise = None
        # Control information
        self.control_module = None
        self.control_law = None
        self.control_last_interactive_parameters = None
        self.control_has_sent_interactive_control_transfer_function_results = False
        self.last_response_srs = None
        self.last_response_time_history = None
        self.last_drive_amplitudes = None
        self.last_drive_decays = None
        self.last_drive_delays = None
        self.hit_level_history = None
        self.run_metadata = None
        # Prediction information
        self.predicted_response_srs = None
        self.predicted_response_time_history = None
        self.predicted_amplitudes = None
        self.predicted_decays = None
        self.predicted_delays = None
        self.predicted_drive_time_history = None

    # region Environment
    def initialize_environment_test_parameters(self, environment_parameters: SDSMetadata):
        print("Environment Initialized Parameters")
        # Check if things need to be reset
        if self.environment_parameters is None or not np.array_equal(
            self.environment_parameters.control_channel_indices,
            environment_parameters.control_channel_indices,
        ):
            # System ID information
            self.sysid_frames = None
            self.sysid_frequencies = None
            self.sysid_frf = None
            self.sysid_coherence = None
            self.sysid_response_cpsd = None
            self.sysid_reference_cpsd = None
            self.sysid_condition = None
            self.sysid_response_noise = None
            self.sysid_reference_noise = None
            self.control_last_interactive_parameters = None
            self.control_has_sent_interactive_control_transfer_function_results = False
            self.last_response_srs = None
            self.last_drive_amplitudes = None
            self.last_drive_decays = None
            self.last_drive_delays = None
        super().initialize_environment_test_parameters(environment_parameters)
        self.environment_parameters: SDSMetadata
        # Load in the control law
        if (
            self.environment_parameters.control_script_data.control_script
            == "rattlesnake.environment.sds_sys_id_control_law"
        ):
            self.control_module = importlib.import_module(
                "rattlesnake.environment.sds_sys_id_control_law"
            )
        else:
            self.control_module = load_python_module(
                self.environment_parameters.control_script_data.control_script
            )
        # Depending on the type, initialize the control law
        if self.environment_parameters.control_script_data.control_type == ControlLawType.FUNCTION:
            self.control_law = getattr(
                self.control_module, self.environment_parameters.control_script_data.control_object
            )
        elif self.environment_parameters.control_script_data.control_type == ControlLawType.CLASS:
            self.control_law = getattr(
                self.control_module, self.environment_parameters.control_script_data.control_object
            )(
                environment_parameters=self.environment_parameters,
                transfer_function_frequencies=self.sysid_frequencies,
                transfer_function=self.sysid_frf,
                noise_response_cpsd=self.sysid_response_noise,
                noise_reference_cpsd=self.sysid_reference_noise,
                sysid_response_cpsd=self.sysid_response_cpsd,
                sysid_reference_cpsd=self.sysid_reference_cpsd,
                multiple_coherence=self.sysid_coherence,
                frames=self.sysid_frames,
                last_response_srs=self.last_response_srs,
                last_drive_amplitudes=self.last_drive_amplitudes,
                last_drive_decays=self.last_drive_decays,
                last_drive_delays=self.last_drive_delays,
                **self.environment_parameters.control_script_data.control_parameters,
            )
        elif (
            self.environment_parameters.control_script_data.control_type
            == ControlLawType.INTERACTIVE_CLASS
        ):
            self.control_law = getattr(
                self.control_module, self.environment_parameters.control_script_data.control_object
            )(
                environment_parameters=self.environment_parameters,
                transfer_function_frequencies=self.sysid_frequencies,
                transfer_function=self.sysid_frf,
                noise_response_cpsd=self.sysid_response_noise,
                noise_reference_cpsd=self.sysid_reference_noise,
                sysid_response_cpsd=self.sysid_response_cpsd,
                sysid_reference_cpsd=self.sysid_reference_cpsd,
                multiple_coherence=self.sysid_coherence,
                frames=self.sysid_frames,
                last_response_srs=self.last_response_srs,
                last_drive_amplitudes=self.last_drive_amplitudes,
                last_drive_decays=self.last_drive_decays,
                last_drive_delays=self.last_drive_delays,
                **self.environment_parameters.control_script_data.control_parameters,
            )
            self.control_last_interactive_parameters = None
            self.control_has_sent_interactive_control_transfer_function_results = False
        else:
            raise ValueError(
                f"Invalid type {self.environment_parameters.control_script_data.control_type}. "
                "How did you get here?!"
            )

    # region Interactive Control Law
    def update_interactive_control_parameters(self, interactive_control_parameters):
        """Updates the interactive control law based on received parameters"""
        if (
            self.environment_parameters.control_script_data.control_type
            == ControlLawType.INTERACTIVE_CLASS
        ):
            self.control_law.update_parameters(interactive_control_parameters)
            self.control_last_interactive_parameters = interactive_control_parameters
        else:
            raise ValueError(
                "Received an UPDATE_INTERACTIVE_CONTROL_PARAMETERS signal without an "
                "interactive control law.  How did this happen?"
            )

    def send_interactive_command(self, command):
        """General method that can be used by an interactive UI object to pass commands
        and data to its corresponding computation object"""
        if (
            self.environment_parameters.control_script_data.control_type
            == ControlLawType.INTERACTIVE_CLASS
        ):
            self.control_law.send_command(command)
        else:
            raise ValueError(
                "Received an SEND_INTERACTIVE_COMMAND signal without an interactive "
                "control law.  How did this happen?"
            )

    # region System ID
    def system_id_complete(self, data):
        """Sends the message that system identification is complete and control calculations
        should be performed"""
        print("Environment System ID Complete!")
        super().system_id_complete(data)
        (
            self.sysid_frames,
            _,  # avg,
            self.sysid_frequencies,
            self.sysid_frf,
            self.sysid_coherence,
            self.sysid_response_cpsd,
            self.sysid_reference_cpsd,
            self.sysid_condition,
            self.sysid_response_noise,
            self.sysid_reference_noise,
        ) = data
        self.perform_control_prediction(True)

    # region Prediction
    def perform_control_prediction(self, sysid_update):
        """Performs the control prediction based on system identification information"""
        print("Performing Control Prediction")
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
        # Perform the control prediction
        # Depending on the type, initialize the control law
        if self.environment_parameters.control_script_data.control_type == ControlLawType.FUNCTION:
            output_amplitudes, output_decays, output_delays = self.control_law(
                environment_parameters=self.environment_parameters,
                transfer_function_frequencies=self.sysid_frequencies,
                transfer_function=self.sysid_frf,
                noise_response_cpsd=self.sysid_response_noise,
                noise_reference_cpsd=self.sysid_reference_noise,
                sysid_response_cpsd=self.sysid_response_cpsd,
                sysid_reference_cpsd=self.sysid_reference_cpsd,
                multiple_coherence=self.sysid_coherence,
                frames=self.sysid_frames,
                last_response_srs=self.last_response_srs,
                last_drive_amplitudes=self.last_drive_amplitudes,
                last_drive_decays=self.last_drive_decays,
                last_drive_delays=self.last_drive_delays,
                **self.environment_parameters.control_script_data.control_parameters,
            )
        elif (
            self.environment_parameters.control_script_data.control_type == ControlLawType.CLASS
            or self.environment_parameters.control_script_data.control_type
            == ControlLawType.INTERACTIVE_CLASS
        ):
            if sysid_update:
                self.control_law.system_id_update(
                    transfer_function_frequencies=self.sysid_frequencies,
                    transfer_function=self.sysid_frf,
                    noise_response_cpsd=self.sysid_response_noise,
                    noise_reference_cpsd=self.sysid_reference_noise,
                    sysid_response_cpsd=self.sysid_response_cpsd,
                    sysid_reference_cpsd=self.sysid_reference_cpsd,
                    multiple_coherence=self.sysid_coherence,
                    frames=self.sysid_frames,
                )
                if (
                    self.environment_parameters.control_script_data.control_type
                    == ControlLawType.INTERACTIVE_CLASS
                ):
                    self.gui_update_queue.put(
                        (
                            self.environment_name,
                            (
                                "interactive_control_sysid_update",
                                (
                                    self.sysid_frequencies,
                                    self.sysid_frf,
                                    self.sysid_response_noise,
                                    self.sysid_reference_noise,
                                    self.sysid_response_cpsd,
                                    self.sysid_reference_cpsd,
                                    self.sysid_coherence,
                                    self.sysid_frames,
                                ),
                            ),
                        )
                    )
                    self.control_has_sent_interactive_control_transfer_function_results = True
            if (
                self.environment_parameters.control_script_data.control_type == ControlLawType.CLASS
                or self.control_last_interactive_parameters is not None
            ):
                output_amplitudes, output_decays, output_delays = self.control_law.control(
                    last_response_srs=self.last_response_srs,
                    last_drive_amplitudes=self.last_drive_amplitudes,
                    last_drive_decays=self.last_drive_decays,
                    last_drive_delays=self.last_drive_delays,
                )
            else:
                self.log("Have not yet received control parameters from interactive control law!")
                return
        else:
            raise ValueError(
                f"Invalid type {self.environment_parameters.control_script_data.control_type}. "
                "How did you get here?!"
            )

        (
            self.predicted_drive_time_history,
            self.predicted_response_time_history,
            self.predicted_response_srs,
        ) = self.simulate_response((output_amplitudes, output_decays, output_delays))
        self.predicted_amplitudes = output_amplitudes
        self.predicted_decays = output_decays
        self.predicted_delays = output_delays
        self.show_test_prediction()

    def perform_prediction_table_prediction(self, sds_table):
        self.perform_table_prediction(sds_table, run_table=False)

    def perform_run_table_prediction(self, sds_table):
        self.perform_table_prediction(sds_table, run_table=True)

    def perform_table_prediction(self, sds_table, run_table: bool):
        output_amplitudes = sds_table["amplitude"]
        output_decays = sds_table["decay"]
        output_delays = sds_table["delay"]
        (
            predicted_drive_time_history,
            predicted_response_time_history,
            predicted_response_srs,
        ) = self.simulate_response((output_amplitudes, output_decays, output_delays))
        self.gui_update_queue.put(
            (
                self.environment_name,
                (
                    "control_run_predictions" if run_table else "control_predictions",
                    (
                        output_amplitudes,
                        output_delays,
                        output_decays,
                        predicted_drive_time_history,
                        predicted_response_time_history,
                        predicted_response_srs,
                    ),
                ),
            )
        )

    def simulate_response(self, data):
        print("Reconstructing Drives")
        # Reconstruct drive signals
        amplitudes, decays, delays = data
        frequencies = self.environment_parameters.get_sds_frequencies_w_compensation_pulse()
        drive_signals = sum_decayed_sines_reconstruction(
            frequencies,
            amplitudes[:, np.newaxis, :].T,
            decays[:, np.newaxis, :].T,
            delays[:, np.newaxis, :].T,
            self.environment_parameters.sample_rate,
            self.environment_parameters.block_size,
        )
        # Simulate responses to those drive signals
        print("Computing Impulse Response")
        impulse_responses = np.moveaxis(np.fft.irfft(self.sysid_frf, axis=0), 0, -1)

        predicted_response_time_history = np.zeros(
            (impulse_responses.shape[0], drive_signals.shape[-1])
        )
        print("Simulating Response")
        for i, impulse_response_row in enumerate(impulse_responses):
            for impulse, drive in zip(impulse_response_row, drive_signals):
                # print('Convolving {:},{:}'.format(i,j))
                predicted_response_time_history[i, :] += sig.convolve(drive, impulse, "full")[
                    : drive_signals.shape[-1]
                ]

        srss = []
        print("Computing SRS")
        for signal in predicted_response_time_history:
            srss.append(
                srs_function(
                    signal,
                    1 / self.environment_parameters.sample_rate,
                    self.environment_parameters.get_sds_frequencies(),
                    self.environment_parameters.srs_data.srs_damping,
                    self.environment_parameters.srs_data.srs_type.value
                    * self.environment_parameters.srs_data.srs_displacement.value,
                )[0]
            )
        srss = np.array(srss).T
        return drive_signals, predicted_response_time_history, srss

    def show_test_prediction(self):
        """Sends the test predictions to the UI"""
        for message in ("control_predictions", "control_run_predictions"):
            self.gui_update_queue.put(
                (
                    self.environment_name,
                    (
                        message,
                        (
                            self.predicted_amplitudes,
                            self.predicted_delays,
                            self.predicted_decays,
                            self.predicted_drive_time_history,
                            self.predicted_response_time_history,
                            self.predicted_response_srs,
                        ),
                    ),
                )
            )

    # region Control
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


# region Process


def sds_process(
    environment_name: str,
    queue_name: str,
    input_queue: VerboseMessageQueue,
    gui_update_queue: Queue,
    controller_communication_queue: VerboseMessageQueue,
    log_file_queue: Queue,
    data_in_queue: Queue,
    data_out_queue: Queue,
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
    """
    SDS environment process function called by multiprocessing

    This function defines the SDS Environment process that
    gets run by the multiprocessing module when it creates a new process.  It
    creates a SDSEnvironment object and runs it.

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
    try:
        # Create vibration queues
        queue_container = SDSQueues(
            environment_name,
            input_queue,
            gui_update_queue,
            controller_communication_queue,
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
            ),
        )
        spectral_proc.start()
        analysis_proc = new_process(
            target=sysid_data_analysis_process,
            args=(
                environment_name,
                queue_container.data_analysis_command_queue,
                queue_container.updated_spectral_quantities_queue,
                queue_container.time_history_to_generate_queue,
                queue_container.environment_command_queue,
                queue_container.gui_update_queue,
                queue_container.log_file_queue,
                ping_alive_event,
            ),
        )
        analysis_proc.start()
        siggen_proc = new_process(
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
            ),
        )
        collection_proc.start()

        process_class = SDSEnvironment(
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
    except Exception:  # pylint: disable = broad-exception-caught
        print(traceback.format_exc())
