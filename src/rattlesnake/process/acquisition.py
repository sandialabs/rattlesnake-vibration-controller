# -*- coding: utf-8 -*-
"""
Controller Subsystem that handles the reading of data from the hardware.

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
import multiprocessing.queues as mpqueue
import multiprocessing.synchronize  # pylint: disable=unused-import
import queue as thqueue
from time import sleep, time
from typing import Dict
import os

import numpy as np

from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.hardware.hardware_registry import HARDWARE_ACQUISITION
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.abstract_environment import EnvironmentMetadata
from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.utilities import (
    GlobalCommands,
    QueueContainer,
    align_signals,
    correlation_norm_signal_spec_ratio,
    flush_queue,
)
from rattlesnake.user_interface.ui_utilities import UICommands

TASK_NAME = "Acquisition"

DEBUG = False
DEBUG_DIRECTORY = "debug_data"


# region Acquisition
class AcquisitionProcess(AbstractMessageProcess):
    """Class defining the acquisition behavior of the controller

    This class will handle reading data from the hardware and then sending it
    to the individual environment processes.

    See AbstractMesssageProcess for inherited class members.
    """

    def __init__(
        self,
        process_name: str,
        queue_container: QueueContainer,
        acquisition_active_event: mp.synchronize.Event,
        streaming_active_event: mp.synchronize.Event,
        ready_event: mp.synchronize.Event,
        ping_alive_event: mp.synchronize.Event,
    ):
        """
        Constructor for the AcquisitionProcess class

        Sets up the ``command_map`` and initializes all data members.

        Parameters
        ----------
        process_name : str
            The name of the process.
        queue_container : QueueContainer
            A container containing the queues used to communicate between
            controller processes
        environments : list
            A list of ``(ControlType,environment_name)`` pairs that define the
            environments in the controller.


        """
        super().__init__(
            process_name,
            queue_container.log_file_queue,
            queue_container.acquisition_command_queue,
            queue_container.gui_update_queue,
            ready_event,
        )
        self.map_command(GlobalCommands.INITIALIZE_HARDWARE, self.initialize_hardware)
        self.map_command(GlobalCommands.RUN_HARDWARE, self.acquire_signal)
        self.map_command(GlobalCommands.STOP_HARDWARE, self.stop_acquisition)
        self.map_command(
            GlobalCommands.INITIALIZE_ENVIRONMENT, self.initialize_environment
        )
        self.map_command(GlobalCommands.STOP_ENVIRONMENT, self.stop_environment)
        self.map_command(GlobalCommands.START_STREAMING, self.start_streaming)
        self.map_command(GlobalCommands.STOP_STREAMING, self.stop_streaming)

        # Communication
        self.queue_container = queue_container
        self.startup = True
        self.shutdown_flag = False
        self.any_environments_started = False
        self.ping_alive_event = ping_alive_event
        # Sampling data
        self.sample_rate = None
        self.read_size = None
        # Environment Data
        self.environment_list = []
        self.environment_acquisition_channels = {}
        self.environment_active_flags = {}
        self.environment_last_data = {}
        self.environment_samples_remaining_to_read = {}
        self.environment_first_data = {}
        # Hardware data
        self.hardware = None
        self.hardware_metadata = None
        # Streaming Information
        self.has_streamed = False
        # Persistent data
        self.read_data = None
        self.output_indices = None
        # Abort and Warning Limits
        self.abort_limits = None
        self.warning_limits = None
        self._acquisition_active_event = acquisition_active_event
        self._streaming_active_event = streaming_active_event
        # Debug state
        self.debug_acquisition_counter = 0
        self.debug_environment_in_counters = {}
        # print('acquisition setup')

    # region Debug
    def _reset_debug_counters(self):
        self.debug_acquisition_counter = 0
        self.debug_environment_in_counters = {}

    def _ensure_debug_directory(self):
        if DEBUG:
            os.makedirs(DEBUG_DIRECTORY, exist_ok=True)

    @staticmethod
    def _safe_environment_name(environment):
        return str(environment).replace(" ", "_")

    def _debug_should_save(self):
        return (
            any(self.environment_active_flags.values())
            or any(self.environment_last_data.values())
            or any(v is not None for v in self.environment_first_data.values())
            or self.shutdown_flag
        )

    def _save_acquisition_debug(self, read_data):
        if not DEBUG:
            return
        if read_data is None:
            return
        if not self._debug_should_save():
            return

        self._ensure_debug_directory()
        filename = os.path.join(
            DEBUG_DIRECTORY,
            f"acquisition_debug_{self.debug_acquisition_counter:04d}.npz",
        )
        np.savez(
            filename,
            timestamp=np.array(time()),
            read_data=read_data,
            read_data_buffer=self.read_data.copy(),
            output_indices=np.array(self.output_indices),
            shutdown_flag=self.shutdown_flag,
            active_environments=np.array(
                [k for k, v in self.environment_active_flags.items() if v], dtype=object
            ),
            last_data_flags=np.array(
                [f"{k}:{int(v)}" for k, v in self.environment_last_data.items()],
                dtype=object,
            ),
            first_data_pending=np.array(
                [k for k, v in self.environment_first_data.items() if v is not None],
                dtype=object,
            ),
            environment_samples_remaining_to_read=np.array(
                [
                    (f"{k}:{self.environment_samples_remaining_to_read.get(k, np.nan)}")
                    for k in self.environment_samples_remaining_to_read
                ],
                dtype=object,
            ),
        )
        self.debug_acquisition_counter += 1

    def _save_first_data_alignment_debug(
        self,
        environment,
        expected_first_data,
        read_data_buffer,
        aligned_output_buffer,
        delay,
        environment_data,
    ):
        if not DEBUG:
            return

        self._ensure_debug_directory()
        safe_environment = self._safe_environment_name(environment)
        index = self.debug_environment_in_counters.get(f"{safe_environment}_first", 0)
        filename = os.path.join(
            DEBUG_DIRECTORY,
            f"{safe_environment}_first_data_alignment_{index:04d}.npz",
        )
        np.savez(
            filename,
            timestamp=np.array(time()),
            expected_first_data=expected_first_data,
            read_data_buffer=read_data_buffer,
            aligned_output_buffer=aligned_output_buffer,
            delay=(np.nan if delay is None else delay),
            environment_data=environment_data,
            output_indices=np.array(self.output_indices),
            acquisition_channels=np.array(self.environment_acquisition_channels[environment]),
        )

    def _save_environment_data_in_debug(
        self,
        environment,
        environment_data,
        environment_finished,
        first_acquisition_for_environment,
        samples_remaining_to_read=None,
    ):
        if not DEBUG:
            return
        if environment_data is None:
            return

        self._ensure_debug_directory()
        safe_environment = self._safe_environment_name(environment)
        index = self.debug_environment_in_counters.get(safe_environment, 0)
        filename = os.path.join(
            DEBUG_DIRECTORY,
            f"{safe_environment}_data_in_{index:04d}.npz",
        )
        np.savez(
            filename,
            timestamp=np.array(time()),
            environment_data=environment_data,
            environment_finished=environment_finished,
            first_acquisition_for_environment=first_acquisition_for_environment,
            samples_remaining_to_read=(
                np.nan if samples_remaining_to_read is None else samples_remaining_to_read
            ),
            environment_active_flag=self.environment_active_flags.get(environment, False),
            environment_last_data_flag=self.environment_last_data.get(environment, False),
        )
        self.debug_environment_in_counters[safe_environment] = index + 1

    # region State Sync
    @property
    def acquisition_active(self):
        return self._acquisition_active_event.is_set()

    @property
    def streaming(self):
        return self._streaming_active_event.is_set()

    def set_active(self):
        self._acquisition_active_event.set()

    def clear_active(self):
        self._acquisition_active_event.clear()

    def set_streaming(self):
        self._streaming_active_event.set()

    def clear_streaming(self):
        self._streaming_active_event.clear()

    def initialize_hardware(self, metadata: HardwareMetadata):
        """Sets up the acquisition according to the specified parameters

        Parameters
        ----------
        data : tuple
            A tuple consisting of data acquisition parameters and the channels
            used by each environment.

        """
        self.log("Initializing Hardware")
        # Store pertinent data
        self.sample_rate = metadata.sample_rate
        self.read_size = metadata.samples_per_read
        # Check which type of hardware we have
        if self.hardware is not None:
            self.hardware.close()

        hardware_acquisition_class = HARDWARE_ACQUISITION[metadata.hardware_type]
        self.hardware = hardware_acquisition_class(
            self.ping_alive_event,
            self.queue_container.single_process_hardware_queue,
        )
        # Initialize hardware and create channels
        self.hardware.initialize_hardware(metadata)
        # Set up warning and abort limits
        self.abort_limits = []
        self.warning_limits = []
        for channel in metadata.channel_list:
            try:
                warning_limit = float(channel.warning_level)
            except (ValueError, TypeError):
                warning_limit = float("inf")  # Never warn on this channel
            try:
                abort_limit = float(channel.abort_level)
            except (ValueError, TypeError):
                abort_limit = float(
                    "inf"
                )  # Never abort on this channel if not specified
            self.warning_limits.append(warning_limit)
            self.abort_limits.append(abort_limit)
        self.abort_limits = np.array(self.abort_limits)
        self.warning_limits = np.array(self.warning_limits)
        self.output_indices = [
            index
            for index, channel in enumerate(metadata.channel_list)
            if (channel.feedback_device is not None)
            and not (
                channel.feedback_device.startswith("#")
                or channel.feedback_device.strip() == ""
            )
        ]
        self.read_data = np.zeros(
            (
                len(metadata.channel_list),
                4
                * np.max(
                    [
                        metadata.samples_per_read,
                        metadata.samples_per_write // metadata.output_oversample,
                    ]
                ),
            )
        )

        self.hardware_metadata = metadata
        self.set_ready()

    def initialize_environment(self, metadata_dict: Dict[str, EnvironmentMetadata]):
        self.log("Initializing Environment")
        self.environment_list = []
        self.environment_acquisition_channels = {}
        self.environment_active_flags = {}
        self.environment_last_data = {}
        self.environment_samples_remaining_to_read = {}
        self.environment_first_data = {}
        for queue_name, metadata in metadata_dict.items():
            self.environment_list.append(queue_name)
            self.environment_acquisition_channels[queue_name] = metadata.channel_indices
            self.environment_active_flags[queue_name] = False
            self.environment_last_data[queue_name] = False
            self.environment_samples_remaining_to_read[queue_name] = 0
            self.environment_first_data[queue_name] = None
        self.set_ready()

    # endregion

    # region Commands
    def stop_environment(self, data):
        """Sets flags stating that the specified environment will be ending.

        Parameters
        ----------
        data : str
            The environment name that should be deactivated

        """
        self.log(f"Deactivating Environment {data}")
        self.environment_active_flags[data] = False
        self.environment_last_data[data] = True
        self.environment_samples_remaining_to_read[data] = self.hardware.get_acquisition_delay()

    def start_streaming(self, data):  # pylint: disable=unused-argument
        """Sets the flag to tell the acquisition to write data to disk

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.set_streaming()
        if self.has_streamed:
            self.queue_container.streaming_command_queue.put(
                self.process_name, (GlobalCommands.CREATE_NEW_STREAM, None)
            )
        else:
            self.has_streamed = True

    def stop_streaming(self, data):  # pylint: disable=unused-argument
        """Sets the flag to tell the acquisition to not write data to disk

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.clear_streaming()

    def acquire_signal(self, data):
        """The main acquisition loop of the controller.

        If it is the first time through this loop, startup will be set to True
        and the hardware will be started.

        If it is the last time through this loop, the hardware will be shut
        down.

        The function will simply read the data from the hardware and pass it
        to any active environment and to the streaming process if the process
        is active.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        if self.startup:
            if DEBUG:
                self._reset_debug_counters()
            self.any_environments_started = False
            self.log("Waiting for Output to Start")
            start_wait_time = time()
            while True:
                # Try to get data from the measurement if we can
                try:
                    environment, data = (
                        self.queue_container.input_output_sync_queue.get_nowait()
                    )
                except (thqueue.Empty, mpqueue.Empty):
                    if time() - start_wait_time > 30:
                        self.queue_container.gui_update_queue.put(
                            (
                                UICommands.ERROR,
                                (
                                    "Acquisition Error",
                                    "Acquisition timed out waiting for output to start.  "
                                    "Check output task for errors!",
                                ),
                            )
                        )
                        break
                    sleep(0.1)
                    continue
                if environment is None:
                    self.log("Detected Output Started")
                    break
                else:
                    self.log(f"Listening for first data for environment {environment}")
                    self.environment_first_data[environment] = data
                    self.any_environments_started = True
            self.log("Starting Hardware Acquisition")
            self.hardware.start()
            self.startup = False
            self.set_active()
            self.gui_update_queue.put((UICommands.HARDWARE_STARTED, None))
            # print('started acquisition')
        self.get_first_output_data()
        if (
            self.shutdown_flag  # We're shutting down
            and all(
                [
                    not flag
                    for environment, flag in self.environment_active_flags.items()
                ]
            )  # All the environments are inactive
            and all(
                [
                    flag is None
                    for environment, flag in self.environment_first_data.items()
                ]
            )  # All the environments are not starting
            and all(
                [not flag for environment, flag in self.environment_last_data.items()]
            )  # None of the environments are expecting their last data
        ):
            self.log("Acquiring Remaining Data")
            read_data = self.hardware.read_remaining()
            self.add_data_to_buffer(read_data)
            self._save_acquisition_debug(read_data)
            if read_data.shape[-1] != 0:
                max_vals = np.max(np.abs(read_data), axis=-1)
                self.gui_update_queue.put((UICommands.MONITOR, max_vals))
                warn_channels = max_vals > self.warning_limits
                if np.any(warn_channels):
                    warning_numbers = [
                        i + 1 for i in range(len(warn_channels)) if warn_channels[i]
                    ]
                    print(f"Channels {warning_numbers} Reached Warning Limit")
                    self.log(f"Channels {warning_numbers} Reached Warning Limit")
                abort_channels = max_vals > self.abort_limits
                if np.any(abort_channels):
                    abort_numbers = [
                        i + 1 for i in range(len(abort_channels)) if abort_channels[i]
                    ]
                    print(f"Channels {abort_numbers} Reached Abort Limit")
                    self.log(f"Channels {abort_numbers} Reached Abort Limit")
                    # Don't stop because we're already shutting down.
            self.hardware.stop()
            self.shutdown_flag = False
            self.startup = True
            # print('{:} {:}'.format(self.streaming,self.any_environments_started))
            if self.streaming and self.any_environments_started:
                self.queue_container.streaming_command_queue.put(
                    self.process_name, (GlobalCommands.STREAMING_DATA, read_data.copy())
                )
                self.clear_streaming()
            if self.has_streamed and self.any_environments_started:
                # self.queue_container.streaming_command_queue.put(self.process_name, (GlobalCommands.FINALIZE_STREAMING, None))
                self.has_streamed = False
            self.clear_active()
            self.gui_update_queue.put((UICommands.HARDWARE_ENDED, None))
            self.log("Acquisition Shut Down")
        else:
            aquiring_environments = [
                name for name, flag in self.environment_active_flags.items() if flag
            ]
            self.log(f"Acquiring Data for {aquiring_environments} environments")
            read_data = self.hardware.read()
            self.add_data_to_buffer(read_data)
            self._save_acquisition_debug(read_data)
            if read_data.shape[-1] != 0:
                max_vals = np.max(np.abs(read_data), axis=-1)
                self.gui_update_queue.put((UICommands.MONITOR, max_vals))
                warn_channels = max_vals > self.warning_limits
                if np.any(warn_channels):
                    warning_numbers = [
                        i + 1 for i in range(len(warn_channels)) if warn_channels[i]
                    ]
                    print(f"Channels {warning_numbers} Reached Warning Limit")
                    self.log(f"Channels {warning_numbers} Reached Warning Limit")
                abort_channels = max_vals > self.abort_limits
                if np.any(abort_channels):
                    abort_numbers = [
                        i + 1 for i in range(len(abort_channels)) if abort_channels[i]
                    ]
                    print(f"Channels {abort_numbers} Reached Abort Limit")
                    self.log(f"Channels {abort_numbers} Reached Abort Limit")
                    self.gui_update_queue.put((UICommands.STOP, None))

            # Send the data to the different channels
            for environment in self.environment_list:
                # Check to see if we're waiting for the first data for this environment
                if self.environment_first_data[environment] is not None:
                    # This environment never found a valid input/output sync.
                    if (
                        self.environment_last_data[environment]
                        and self.environment_samples_remaining_to_read[environment] <= 0
                    ):
                        # Shut down the environment in the case that the input/output never synced
                        self.log(
                            f"Abandoning input/output sync search for {environment} "
                            "because environment is shutting down"
                        )
                        self.environment_first_data[environment] = None
                        continue

                    expected_first_data = self.environment_first_data[environment]

                    if np.all(np.abs(expected_first_data) < 1e-10):
                        delay = -self.read_size
                    else:
                        correlation_start_time = time()
                        _, delay, _, found_correlation = align_signals(
                            self.read_data[self.output_indices],
                            expected_first_data,
                            perform_subsample=False,
                            correlation_threshold=0.5,
                            correlation_metric=correlation_norm_signal_spec_ratio,
                        )
                        correlation_end_time = time()
                        corr_time = correlation_end_time - correlation_start_time
                        self.log(
                            f"Correlation check for environment {environment} took "
                            f"{corr_time:0.2f} seconds and achieved {found_correlation:0.2f}"
                        )

                        self.log(
                            f"{environment}: first-data alignment returned delay={delay}, "
                            f"read_buffer_shape={self.read_data.shape}, "
                            f"expected_first_shape={expected_first_data.shape}"
                        )

                        # Adding a criteria that the delay must be in the first half
                        # of the buffer, otherwise we could still be increasing
                        # in correlation as more data is acquired.
                        if delay is None or delay > self.read_data.shape[-1] // 2:
                            continue

                    environment_data = self.read_data[
                        self.environment_acquisition_channels[environment], delay:
                    ]
                    first_acquisition_for_environment = True

                    self._save_first_data_alignment_debug(
                        environment,
                        expected_first_data,
                        self.read_data.copy(),
                        self.read_data[self.output_indices].copy(),
                        delay,
                        environment_data.copy(),
                    )

                    self.log(f"Found First Data for Environment {environment}")
                    self.environment_first_data[environment] = None
                    if not self.environment_last_data[environment]:
                        self.environment_active_flags[environment] = True
                    else:
                        self.log(
                            f"Already received environment {environment} "
                            "shutdown signal, not starting"
                        )
                # Check to see if the environment is active
                elif (
                    self.environment_active_flags[environment]
                    or self.environment_last_data[environment]
                ):
                    environment_data = read_data[
                        self.environment_acquisition_channels[environment]
                    ].copy()
                    first_acquisition_for_environment = False
                # Otherwise the environment isn't active
                else:
                    continue
                if self.environment_last_data[environment]:
                    self.environment_samples_remaining_to_read[
                        environment
                    ] -= self.read_size
                    self.log(
                        f"Reading last data for {environment}, "
                        f"{self.environment_samples_remaining_to_read[environment]} samples "
                        f"remaining"
                    )
                environment_finished = (
                    self.environment_last_data[environment]
                    and self.environment_samples_remaining_to_read[environment] <= 0
                )
                self.log(
                    f"Sending {environment_data.shape} data to {environment} environment"
                )
                self._save_environment_data_in_debug(
                    environment,
                    environment_data,
                    environment_finished,
                    first_acquisition_for_environment,
                    self.environment_samples_remaining_to_read.get(environment),
                )
                self.queue_container.environment_data_in_queues[environment].put(
                    (environment_data, environment_finished)
                )
                if environment_finished:
                    self.environment_last_data[environment] = False
                    self.log(f"Delivered last data to {environment}")
            #  np.savez('test_data/acquisition_data_check.npz',
            #           read_data = self.read_data,
            #           environment_data = environment_data,
            #           environment_channels = self.environment_acquisition_channels[environment])
            self.queue_container.acquisition_command_queue.put(
                self.process_name, (GlobalCommands.RUN_HARDWARE, None)
            )
            # print('{:} {:}'.format(self.streaming,self.any_environments_started))
            if self.streaming and self.any_environments_started:
                self.queue_container.streaming_command_queue.put(
                    self.process_name, (GlobalCommands.STREAMING_DATA, read_data.copy())
                )

    def add_data_to_buffer(self, data):
        """Adds data to the end of the buffer and shifts existing in the buffer forward

        Parameters
        ----------
        data : np.ndarray
            A 2D array with shape num_channels x num_samples
        """
        # Roll the buffer with new data
        read_size = data.shape[-1]
        if read_size != 0:
            self.read_data[..., :-read_size] = self.read_data[..., read_size:]
            self.read_data[..., -read_size:] = data

    def get_first_output_data(self):
        """Searches through the sync queue for first data packages from the output process"""
        first_output_data = flush_queue(self.queue_container.input_output_sync_queue)
        for environment, data in first_output_data:
            self.log(f"Listening for first data for environment {environment}")
            self.environment_first_data[environment] = data
            self.any_environments_started = True

    def stop_acquisition(self, data):  # pylint: disable=unused-argument
        """Sets a flag telling the acquisition that it should start shutting down

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.shutdown_flag = True

    # endregion

    # region Shutdown
    def quit(self, data):
        """Stops the process and shuts down the hardware if necessary.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``
        """
        # Pull any data off the queues that have been put to
        queue_flush_sum = 0
        for queue in [
            q for name, q in self.queue_container.environment_data_in_queues.items()
        ] + [self.queue_container.acquisition_command_queue]:
            queue_flush_sum += len(flush_queue(queue))
        self.log(f"Flushed {queue_flush_sum} items out of queues")
        if self.hardware is not None:
            self.hardware.close()
        return True

    # endregion


# endregion


# region Process
def acquisition_process(
    queue_container: QueueContainer,
    acquisition_active_event: mp.synchronize.Event,
    streaming_active_event: mp.synchronize.Event,
    ready_event: mp.synchronize.Event,
    shutdown_event: mp.synchronize.Event,
    ping_alive_event: mp.synchronize.Event,
):
    """Function passed to multiprocessing as the acquisition process

    This process creates the ``AcquisitionProcess`` object and calls the ``run``
    command.

    Parameters
    ----------
    queue_container : QueueContainer
        A container containing the queues used to communicate between
        controller processes

    """

    acquisition_instance = AcquisitionProcess(
        TASK_NAME,
        queue_container,
        acquisition_active_event,
        streaming_active_event,
        ready_event,
        ping_alive_event,
    )

    acquisition_instance.run(shutdown_event)


# endregion
