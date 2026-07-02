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
import time
from datetime import datetime

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
    SDSUICommands,
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
    align_signals,
    shift_signal,
)


from rattlesnake.environment.abstract_interactive_control_law import (
    AbstractControlLawComputation,
    ControlLawCommands,
)
from rattlesnake.process.abstract_sysid_data_analysis import (
    sysid_data_analysis_process,
    SysIdMetadata,
    SysIdDataPackage,
)
from rattlesnake.process.data_collector import (
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
from rattlesnake.user_interface.ui_utilities import UICommands

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
        # print(f"Building environment for {environment_name}")
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
        self.map_command(SDSCommands.MONITOR_HIT, self.monitor_hit)
        self.map_command(GlobalCommands.START_ENVIRONMENT, self.start_control)
        self.map_command(SDSCommands.STOP_CONTROL, self.stop_environment)
        self.map_command(
            ControlLawCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS,
            self.update_interactive_control_parameters,
        )
        self.map_command(ControlLawCommands.SEND_INTERACTIVE_COMMAND, self.send_interactive_command)

        # Persistent Data
        self.queue_container = queue_container
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
        # Prediction information
        self.predicted_response_srs = None
        self.predicted_response_time_history = None
        self.predicted_amplitudes = None
        self.predicted_decays = None
        self.predicted_delays = None
        self.predicted_drive_time_history = None
        # Run Information
        self.run_instructions = None
        self.hit_history = []
        self.run_sds_table = None

        self.total_hits = 0
        self.hits_at_target = 0
        self.current_test_level_db = 0.0
        self.current_test_level_scale = 1.0

        self.sequence_active = False
        self.automatic_hits = False
        self.automatic_interval = None
        self.stop_requested = False

        self.hit_in_progress = False
        self.pending_next_hit_time = None

        self.last_drive_signal = None
        self.last_measured_drive_signal = None
        self.last_response_signal = None
        self.last_response_srs = None

        self.current_hit_control_data = []
        self.current_hit_output_data = []

        self.allow_automatic_updates = False
        self.last_hit_completion_time = None

        self.set_ready()

    def initialize_hardware(self, hardware_metadata):
        super().initialize_hardware(hardware_metadata)

        self.set_ready()

    # region Environment
    def initialize_environment(self, environment_metadata: SDSMetadata):
        print("Environment Initialized Parameters")
        # Check if things need to be reset
        if self.environment_metadata is None or not np.array_equal(
            self.environment_metadata.control_channel_indices,
            environment_metadata.control_channel_indices,
        ):
            # System ID information
            self.sysid_data = SysIdDataPackage()
            self.control_last_interactive_parameters = None
            self.control_has_sent_interactive_control_transfer_function_results = False
            self.last_response_srs = None
            self.last_drive_amplitudes = None
            self.last_drive_decays = None
            self.last_drive_delays = None
        super().initialize_environment(environment_metadata)
        self.environment_metadata: SDSMetadata
        # Load in the control law
        if (
            self.environment_metadata.control_script_data.control_script
            == "rattlesnake.environment.sds_sys_id_control_law"
        ):
            self.control_module = importlib.import_module(
                "rattlesnake.environment.sds_sys_id_control_law"
            )
        else:
            self.control_module = load_python_module(
                self.environment_metadata.control_script_data.control_script
            )
        # Depending on the type, initialize the control law
        if self.environment_metadata.control_script_data.control_type == ControlLawType.FUNCTION:
            self.control_law = getattr(
                self.control_module, self.environment_metadata.control_script_data.control_object
            )
        elif self.environment_metadata.control_script_data.control_type == ControlLawType.CLASS:
            self.control_law = getattr(
                self.control_module, self.environment_metadata.control_script_data.control_object
            )(
                environment_metadata=self.environment_metadata,
                sysid_data=self.sysid_data,
                last_response_srs=self.last_response_srs,
                last_drive_amplitudes=self.last_drive_amplitudes,
                last_drive_decays=self.last_drive_decays,
                last_drive_delays=self.last_drive_delays,
                **self.environment_metadata.control_script_data.control_parameters,
            )
        elif (
            self.environment_metadata.control_script_data.control_type
            == ControlLawType.INTERACTIVE_CLASS
        ):
            self.control_law = getattr(
                self.control_module, self.environment_metadata.control_script_data.control_object
            )(
                environment_metadata=self.environment_metadata,
                sysid_data=self.sysid_data,
                last_response_srs=self.last_response_srs,
                last_drive_amplitudes=self.last_drive_amplitudes,
                last_drive_decays=self.last_drive_decays,
                last_drive_delays=self.last_drive_delays,
                **self.environment_metadata.control_script_data.control_parameters,
            )
            self.control_last_interactive_parameters = None
            self.control_has_sent_interactive_control_transfer_function_results = False
        else:
            raise ValueError(
                f"Invalid type {self.environment_metadata.control_script_data.control_type}. "
                "How did you get here?!"
            )

        self.set_ready()

    def initialize_sysid(self, sysid_metadata: SysIdMetadata):
        super().initialize_sysid(sysid_metadata)

        self.set_ready()

    # region Interactive Control Law
    def update_interactive_control_parameters(self, interactive_control_parameters):
        """Updates the interactive control law based on received parameters"""
        if (
            self.environment_metadata.control_script_data.control_type
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
            self.environment_metadata.control_script_data.control_type
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
        self.perform_control_prediction(True)
        self.set_sysid_stored()

    # region Prediction
    def perform_control_prediction(self, sysid_update):
        """Performs the control prediction based on system identification information"""
        print("Performing Control Prediction")
        if self.sysid_data.sysid_frf is None:
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
        if self.environment_metadata.control_script_data.control_type == ControlLawType.FUNCTION:
            output_amplitudes, output_decays, output_delays = self.control_law(
                environment_metadata=self.environment_metadata,
                sysid_data=self.sysid_data,
                last_response_srs=self.last_response_srs,
                last_drive_amplitudes=self.last_drive_amplitudes,
                last_drive_decays=self.last_drive_decays,
                last_drive_delays=self.last_drive_delays,
                **self.environment_metadata.control_script_data.control_parameters,
            )
        elif (
            self.environment_metadata.control_script_data.control_type == ControlLawType.CLASS
            or self.environment_metadata.control_script_data.control_type
            == ControlLawType.INTERACTIVE_CLASS
        ):
            if sysid_update:
                self.control_law.system_id_update(sysid_data=self.sysid_data)
                if (
                    self.environment_metadata.control_script_data.control_type
                    == ControlLawType.INTERACTIVE_CLASS
                ):
                    self.gui_update_queue.put(
                        (
                            self.environment_name,
                            (
                                "interactive_control_sysid_update",
                                self.sysid_data,
                            ),
                        )
                    )
                    self.control_has_sent_interactive_control_transfer_function_results = True
            if (
                self.environment_metadata.control_script_data.control_type == ControlLawType.CLASS
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
                f"Invalid type {self.environment_metadata.control_script_data.control_type}. "
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
                    (
                        SDSUICommands.RUN_CONTROL_PREDICTIONS
                        if run_table
                        else SDSUICommands.CONTROL_PREDICTIONS
                    ),
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
        frequencies = self.environment_metadata.get_sds_frequencies_w_compensation_pulse()
        drive_signals = sum_decayed_sines_reconstruction(
            frequencies,
            amplitudes[:, np.newaxis, :].T,
            decays[:, np.newaxis, :].T,
            delays[:, np.newaxis, :].T,
            self.environment_metadata.sample_rate,
            self.environment_metadata.block_size,
        )
        # Simulate responses to those drive signals
        print("Computing Impulse Response")
        impulse_responses = np.moveaxis(np.fft.irfft(self.sysid_data.sysid_frf, axis=0), 0, -1)

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
                    1 / self.environment_metadata.sample_rate,
                    self.environment_metadata.get_sds_frequencies(),
                    self.environment_metadata.srs_data.srs_damping,
                    self.environment_metadata.srs_data.srs_type.value
                    * self.environment_metadata.srs_data.srs_displacement.value,
                )[0]
            )
        srss = np.array(srss).T
        return drive_signals, predicted_response_time_history, srss

    def show_test_prediction(self):
        """Sends the test predictions to the UI"""
        for message in (SDSUICommands.CONTROL_PREDICTIONS, SDSUICommands.RUN_CONTROL_PREDICTIONS):
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
            samples_per_write=self.hardware_metadata.samples_per_write,
            level_ramp_samples=0.5  # This isn't really necessary since we won't cancel during a hit
            * self.environment_metadata.sample_rate
            * self.hardware_metadata.output_oversample,
            output_transformation_matrix=self.environment_metadata.reference_transformation_matrix,
        )

    def start_control(self, data):
        """
        Start the SDS environment.

        Manual mode:
            One START_CONTROL performs one hit and then the environment ends.

        Automatic mode:
            One START_CONTROL performs repeated hits until:
              - target_hits_at_level is reached, or
              - stop_environment is requested.

        Hit counters and history are cumulative across runs and are NOT reset here.
        """
        instructions = data

        if self.active:
            self.log("SDS environment already active; ignoring duplicate start.")
            return

        instructions.validate()

        self.run_instructions = instructions
        self.run_sds_table = instructions.sds_table.copy()

        self.current_test_level_db = instructions.control_test_level
        self.current_test_level_scale = db2scale(instructions.control_test_level)

        self.automatic_hits = instructions.automatic_hits
        self.automatic_interval = instructions.automatic_interval
        self.allow_automatic_updates = instructions.allow_automatic_updates

        self.sequence_active = True
        self.stop_requested = False
        self.hit_in_progress = False
        self.pending_next_hit_time = None
        self.last_hit_completion_time = None

        self.last_drive_signal = None
        self.last_measured_drive_signal = None
        self.last_response_signal = None
        self.last_response_srs = None

        self.current_hit_control_data = []
        self.current_hit_output_data = []

        self.gui_update_queue.put(
            (
                self.environment_name,
                (UICommands.SET_ENVIRONMENT_INSTRUCTIONS, instructions),
            )
        )

        self.set_active()
        self.gui_update_queue.put((self.environment_name, (UICommands.ENVIRONMENT_STARTED, None)))

        self.log(
            f"Starting SDS environment: automatic_hits={self.automatic_hits}, "
            f"target_hits_at_level={instructions.target_hits_at_level}, "
            f"test_level_db={self.current_test_level_db}, "
            f"allow_automatic_updates={self.allow_automatic_updates}"
        )

        # Always perform at least one hit when start is pressed
        self.launch_hit()

    def launch_hit(self):
        """
        Launch a single SDS hit by synthesizing one transient drive waveform from
        the current run SDS table and sending it to the signal generation process.
        """
        if self.hit_in_progress:
            self.log("Attempted to launch a hit while one was already in progress.")
            return

        self.log("Launching SDS hit")

        frequencies = self.run_sds_table["frequency"]
        amplitudes = self.run_sds_table["amplitude"]
        decays = self.run_sds_table["decay"]
        delays = self.run_sds_table["delay"]

        drive_signal = sum_decayed_sines_reconstruction(
            frequencies,
            amplitudes[:, np.newaxis, :].T,
            decays[:, np.newaxis, :].T,
            delays[:, np.newaxis, :].T,
            self.environment_metadata.sample_rate,
            self.environment_metadata.block_size,
        )

        self.last_drive_signal = drive_signal
        self.hit_in_progress = True

        # Reset per-hit acquisition accumulation
        self.current_hit_control_data = []
        self.current_hit_output_data = []

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
                TransientSignalGenerator(drive_signal, repeat=False),
            ),
        )
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.SET_TEST_LEVEL, self.current_test_level_scale),
        )
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.GENERATE_SIGNALS, None),
        )

        # Begin monitoring this hit
        self.queue_container.environment_command_queue.put(
            self.environment_name, (SDSCommands.MONITOR_HIT, None)
        )

    def complete_hit(self, full_control, full_output):
        """
        Complete one SDS hit by postprocessing the measured time histories.

        Parameters
        ----------
        full_control : np.ndarray
            Measured response/control-channel time history for the hit,
            shape (num_control_channels, num_samples)
        full_output : np.ndarray
            Measured output/drive-channel time history for the hit,
            shape (num_drive_channels, num_samples)
        """
        self.log("Completing SDS hit")

        expected_output = self.last_drive_signal[:, :: self.hardware_metadata.output_oversample]

        aligned_output, sample_delay, phase_change, found_correlation = align_signals(
            full_output,
            expected_output,
            correlation_threshold=0.5,
            perform_subsample=True,
        )

        if aligned_output is None:
            self.log("Could not align measured output to expected drive signal.")
            samples_to_keep = min(full_control.shape[-1], self.environment_metadata.block_size)
            self.last_response_signal = full_control[..., :samples_to_keep]
            measured_drive_signal = full_output[..., :samples_to_keep]
        else:
            self.log(
                f"Alignment found: sample_delay={sample_delay}, "
                f"phase_change={phase_change}, correlation={found_correlation}"
            )
            samples_to_keep = min(
                self.environment_metadata.block_size,
                full_control.shape[-1] - sample_delay,
                full_output.shape[-1] - sample_delay,
                expected_output.shape[-1],
            )

            measured_drive_signal = shift_signal(
                full_output,
                samples_to_keep,
                sample_delay,
                phase_change,
            )
            self.last_response_signal = shift_signal(
                full_control,
                samples_to_keep,
                sample_delay,
                phase_change,
            )

        self.last_measured_drive_signal = measured_drive_signal

        response_srs = []
        for signal in self.last_response_signal:
            response_srs.append(
                srs_function(
                    signal,
                    1 / self.environment_metadata.sample_rate,
                    self.environment_metadata.get_sds_frequencies(),
                    self.environment_metadata.srs_data.srs_damping,
                    self.environment_metadata.srs_data.srs_type.value
                    * self.environment_metadata.srs_data.srs_displacement.value,
                )[0]
            )
        self.last_response_srs = np.array(response_srs).T

        # Preserve latest drive table used for this hit
        self.last_drive_amplitudes = self.run_sds_table["amplitude"].copy()
        self.last_drive_decays = self.run_sds_table["decay"].copy()
        self.last_drive_delays = self.run_sds_table["delay"].copy()

        # Always call control law after each hit
        output_amplitudes, output_decays, output_delays = self.control_law(
            environment_metadata=self.environment_metadata,
            sysid_data=self.sysid_data,
            last_response_srs=self.last_response_srs,
            last_response_signals=self.last_response_signal,
            last_drive_amplitudes=self.last_drive_amplitudes,
            last_drive_decays=self.last_drive_decays,
            last_drive_delays=self.last_drive_delays,
            last_drive_signals=measured_drive_signal,
            **self.environment_metadata.control_script_data.control_parameters,
        )

        # Only persist updated SDS table if automatic updates are enabled
        if self.allow_automatic_updates:
            self.run_sds_table["amplitude"] = output_amplitudes
            self.run_sds_table["decay"] = output_decays
            self.run_sds_table["delay"] = output_delays

        # Update cumulative counters
        self.total_hits += 1
        counted_at_target = abs(self.current_test_level_db) < 1e-12
        if counted_at_target:
            self.hits_at_target += 1

        self.last_hit_completion_time = time.time()

        self.hit_history.append(
            {
                "hit_index": self.total_hits,
                "timestamp": datetime.now().isoformat(),
                "test_level_db": self.current_test_level_db,
                "counted_at_target": counted_at_target,
                "total_hits": self.total_hits,
                "hits_at_target": self.hits_at_target,
                "target_hits_at_level": (
                    None
                    if self.run_instructions is None
                    else self.run_instructions.target_hits_at_level
                ),
            }
        )

        self.log(
            f"Hit complete: total_hits={self.total_hits}, "
            f"hits_at_target={self.hits_at_target}, "
            f"counted_at_target={counted_at_target}"
        )

        self.hit_in_progress = False

        self.emit_control_update()

    def monitor_hit(self, data):
        """
        Monitor an in-progress SDS hit, or wait between hits during an automatic sequence.

        Responsibilities
        ----------------
        - While a hit is active:
            * gather acquisition blocks
            * detect hit completion via last_acquisition flag
            * call complete_hit(...)
        - After a hit completes:
            * if manual mode -> finish
            * if automatic mode -> wait until the next launch time, then launch next hit
        """
        if not self.active or not self.sequence_active:
            return

        # Case 1: a hit is currently active
        if self.hit_in_progress:
            try:
                acquisition_data, last_acquisition = self.queue_container.data_in_queue.get_nowait()

                control_data = acquisition_data[self.environment_metadata.control_channel_indices]
                if self.environment_metadata.response_transformation_matrix is not None:
                    control_data = (
                        self.environment_metadata.response_transformation_matrix @ control_data
                    )

                output_data = acquisition_data[self.environment_metadata.output_channel_indices]
                if self.environment_metadata.reference_transformation_matrix is not None:
                    output_data = (
                        self.environment_metadata.reference_transformation_matrix @ output_data
                    )

                self.current_hit_control_data.append(control_data)
                self.current_hit_output_data.append(output_data)

                if last_acquisition:
                    self.log("Received last acquisition block for SDS hit")

                    full_control = np.concatenate(self.current_hit_control_data, axis=-1)
                    full_output = np.concatenate(self.current_hit_output_data, axis=-1)

                    self.complete_hit(full_control, full_output)

                    # Manual mode: one hit and done
                    if not self.automatic_hits:
                        self.finish_sequence()
                        return

                    # Automatic mode: stop if requested or threshold reached
                    if self.stop_requested:
                        self.finish_sequence()
                        return

                    if self.hits_at_target >= self.run_instructions.target_hits_at_level:
                        self.finish_sequence()
                        return

                    # Otherwise schedule next hit.
                    # If computation took longer than the interval, the next hit will occur immediately.
                    self.pending_next_hit_time = (
                        self.last_hit_completion_time + self.automatic_interval
                    )
                    self.log(f"Next automatic hit time set to {self.pending_next_hit_time:.3f}")

                if self.sequence_active:
                    time.sleep(0.05)
                    self.queue_container.environment_command_queue.put(
                        self.environment_name, (SDSCommands.MONITOR_HIT, None)
                    )
                return

            except Exception:
                # No data available yet
                time.sleep(0.05)
                self.queue_container.environment_command_queue.put(
                    self.environment_name, (SDSCommands.MONITOR_HIT, None)
                )
                return

        # Case 2: waiting between automatic hits
        if self.automatic_hits and self.pending_next_hit_time is not None:
            if self.stop_requested:
                self.finish_sequence()
                return

            if self.hits_at_target >= self.run_instructions.target_hits_at_level:
                self.finish_sequence()
                return

            if time.time() >= self.pending_next_hit_time:
                self.pending_next_hit_time = None
                self.launch_hit()
                return

            time.sleep(0.05)
            self.queue_container.environment_command_queue.put(
                self.environment_name, (SDSCommands.MONITOR_HIT, None)
            )
            return

    def shutdown(self):
        """
        Final SDS environment shutdown.
        """
        self.log("Environment Shut Down")

        self.sequence_active = False
        self.hit_in_progress = False
        self.pending_next_hit_time = None
        self.stop_requested = False

        self.clear_active()
        self.gui_update_queue.put((self.environment_name, (UICommands.ENVIRONMENT_ENDED, None)))

    def stop_environment(self, data):
        """
        Stop the SDS environment gracefully.

        A currently active hit is allowed to finish. This only prevents future
        hits from starting in an automatic sequence.
        """
        self.log("Stop requested for SDS environment")
        self.stop_requested = True

        if not self.hit_in_progress:
            self.finish_sequence()
            return

        self.log("A hit is currently in progress; stopping after hit completion.")

    def emit_control_update(self):
        self.gui_update_queue.put(
            (
                self.environment_name,
                (
                    SDSUICommands.CONTROL_UPDATE,
                    {
                        "measured_drive_time_history": (
                            None
                            if self.last_measured_drive_signal is None
                            else self.last_measured_drive_signal.copy()
                        ),
                        "measured_response_time_history": (
                            None
                            if self.last_response_signal is None
                            else self.last_response_signal.copy()
                        ),
                        "measured_response_srs": (
                            None
                            if self.last_response_srs is None
                            else self.last_response_srs.copy()
                        ),
                        "run_sds_table": (
                            self.run_sds_table.copy()
                            if self.allow_automatic_updates and self.run_sds_table is not None
                            else None
                        ),
                        "total_hits": self.total_hits,
                        "hits_at_target": self.hits_at_target,
                        "target_hits_at_level": (
                            None
                            if self.run_instructions is None
                            else self.run_instructions.target_hits_at_level
                        ),
                        "hit_history": list(self.hit_history),
                    },
                ),
            )
        )

    def finish_sequence(self):
        self.log("Finishing SDS hit sequence")
        self.sequence_active = False
        self.hit_in_progress = False
        self.pending_next_hit_time = None
        self.shutdown()


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
            threaded,
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
