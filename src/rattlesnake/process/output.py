# -*- coding: utf-8 -*-
"""
Controller subsystem that handles the output from the hardware to the
shaker amplifiers or other excitation device.

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
from typing import Dict

import numpy as np

from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.hardware.hardware_registry import HARDWARE_OUTPUT
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.abstract_environment import EnvironmentMetadata
from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.utilities import GlobalCommands, QueueContainer, flush_queue, rms_time

TASK_NAME = "Output"

DEBUG = False
if DEBUG:
    from glob import glob

    FILE_OUTPUT = "debug_data/output_{:}.npz"
    ENV_OUTPUT = "debug_data/output_first_data_{:}.npz"


# region Output
class OutputProcess(AbstractMessageProcess):
    """Class defining the output behavior of the controller

    This class will handle collecting data from the environments and writing
    data to be output to the hardware

    See AbstractMessageProcess for inherited members.
    """

    def __init__(
        self,
        process_name: str,
        queue_container: QueueContainer,
        output_active_event: mp.synchronize.Event,
        ready_event: mp.synchronize.Event,
    ):
        """
        Constructor for the OutputProcess Class

        Sets up the ``command_map`` and initializes all data members.

        Parameters
        ----------
        process_name : str
            The name of the process.
        queue_container : QueueContainer
            A container containing the queues used to communicate between
            controller processes
        """
        super().__init__(
            process_name,
            queue_container.log_file_queue,
            queue_container.output_command_queue,
            queue_container.gui_update_queue,
            ready_event,
        )
        self.map_command(GlobalCommands.INITIALIZE_HARDWARE, self.initialize_hardware)
        self.map_command(GlobalCommands.RUN_HARDWARE, self.output_signal)
        self.map_command(GlobalCommands.STOP_HARDWARE, self.stop_output)
        self.map_command(
            GlobalCommands.INITIALIZE_ENVIRONMENT, self.initialize_environment
        )
        self.map_command(GlobalCommands.START_ENVIRONMENT, self.start_environment)
        # Communication
        self.queue_container = queue_container
        self.startup = True
        self.shutdown_flag = False
        # Sampling data
        self.sample_rate = None
        self.write_size = None
        self.num_outputs = None
        self.output_oversample = None
        # Environment Data
        self.environment_list = []
        self.environment_output_channels = None
        self.environment_active_flags = {}
        self.environment_starting_up_flags = {}
        self.environment_shutting_down_flags = {}
        self.environment_data_out_remainders = None
        self.environment_first_data = {}
        # Hardware data
        self.hardware = None
        self.hardware_metadata = None
        # Shared memory to record activity
        self._output_active_event = output_active_event
        # print('output setup')

    # endregion

    # region State Sync
    @property
    def output_active(self):
        """Returns True if the output is currently active"""
        return self._output_active_event.is_set()

    def set_active(self):
        self._output_active_event.set()

    def clear_active(self):
        self._output_active_event.clear()

    def initialize_hardware(self, metadata: HardwareMetadata):
        """
        Sets up the output according to the specified parameters

        Parameters
        ----------
        data : tuple
            A tuple consisting of data acquisition parameters and the channels
            used by each environment.

        """
        self.log("Initializing Hardware")
        # Pull out invormation from the queue
        # Store pertinent data
        self.sample_rate = metadata.sample_rate
        self.write_size = metadata.samples_per_write
        self.output_oversample = metadata.output_oversample
        # Check which type of hardware we have
        if self.hardware is not None:
            self.hardware.close()

        hardware_output_class = HARDWARE_OUTPUT[metadata.hardware_type]
        self.hardware = hardware_output_class(
            self.queue_container.single_process_hardware_queue
        )
        # Initialize hardware and create channels
        self.hardware.initialize_hardware(metadata)
        # Get the environment output channels in reference to all the output channels
        output_indices = [
            index
            for index, channel in enumerate(metadata.channel_list)
            if (channel.feedback_device is not None)
            and not (channel.feedback_device.strip() == "")
        ]
        self.num_outputs = len(output_indices)

        self.hardware_metadata = metadata
        self.set_ready()

    def initialize_environment(self, metadata_dict: Dict[str, EnvironmentMetadata]):
        self.log("Initializing Environment")

        hardware_output_indices = [
            index
            for index, channel in enumerate(self.hardware_metadata.channel_list)
            if (channel.feedback_device is not None)
            and not (channel.feedback_device.strip() == "")
        ]
        self.environment_list = []
        self.environment_output_channels = {}
        self.environment_active_flags = {}
        self.environment_starting_up_flags = {}
        self.environment_shutting_down_flags = {}
        self.environment_data_out_remainders = {}
        self.environment_first_data = {}
        for queue_name, metadata in metadata_dict.items():
            # Initialize environment dicts
            self.environment_list.append(queue_name)
            self.environment_active_flags[queue_name] = False
            self.environment_starting_up_flags[queue_name] = False
            self.environment_shutting_down_flags[queue_name] = False
            self.environment_first_data[queue_name] = False

            # Build output mapping dicts
            common_indices, out_inds, _ = np.intersect1d(
                hardware_output_indices,
                metadata.channel_indices,
                return_indices=True,
            )
            self.environment_output_channels[queue_name] = out_inds
            self.environment_data_out_remainders[queue_name] = np.zeros(
                (len(common_indices), 0)
            )
        self.set_ready()

    # endregion

    # region Commands
    def output_signal(self, data):  # pylint: disable=unused-argument
        """The main output loop of the controller.

        If it is the first time through the loop, ``self.startup`` will be set
        to ``True`` and the hardware will be started.

        The output task must be notified that a given channel has started before
        it will check the channel for data.

        The function receives data from each of the environment data_out queues
        as well as a flag specifying whether the signal is the last signal.  If
        so, the output will deactivate the channel and stop looking for data
        from it until the next time it is activated.

        If the output is shut down, it will activate a ``shutdown_flag``, after
        which the output will wait for all environments to pass their last data
        and be deactivated.  Once all environments are deactivated, the output
        will stop and shutdown the hardware.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        # Skip hardware operations if there are no channels
        skip_hardware = self.num_outputs == 0
        # Go through each environment and collect data no matter what.
        # Start with ready to write and set to false if any environments are not
        ready_to_write = True
        for environment in self.environment_list:
            # If the task isn't active or currently shutting down, we don't need more data,
            # so just skip it.
            if (
                not self.environment_active_flags[environment]
                and not self.environment_starting_up_flags[environment]
            ) or self.environment_shutting_down_flags[environment]:
                continue
            # Check if we need more data from that environment
            if (
                self.environment_data_out_remainders[environment].shape[-1]
                < self.write_size
            ):
                if not self.environment_starting_up_flags[environment]:
                    ready_to_write = False
                try:
                    # Try to grab data from the queue and add it to the remainders.
                    environment_data, last_run = (
                        self.queue_container.environment_data_out_queues[
                            environment
                        ].get_nowait()
                    )
                except (thqueue.Empty, mpqueue.Empty):
                    # If data is not ready yet, simply continue to the next environment and we'll
                    # try on the next time around.
                    continue
                self.log(
                    f"Got {' x '.join([f'{shape}' for shape in environment_data.shape])} "
                    f"data from {environment} Environment"
                )
                if last_run:
                    self.log(f"Deactivating {environment} Environment")
                    self.environment_shutting_down_flags[environment] = True
                    if self.environment_starting_up_flags[environment]:
                        self.environment_starting_up_flags[environment] = False
                        self.environment_active_flags[environment] = True
                        self.environment_first_data[environment] = True
                self.environment_data_out_remainders[environment] = np.concatenate(
                    (
                        self.environment_data_out_remainders[environment],
                        environment_data,
                    ),
                    axis=-1,
                )
            else:
                if self.environment_starting_up_flags[environment]:
                    self.environment_starting_up_flags[environment] = False
                    self.environment_active_flags[environment] = True
                    self.log(
                        f"Received First Complete Data Write for {environment} Environment"
                    )
                    self.environment_first_data[environment] = True

        # If we got through that previous loop still ready to write, we can
        # output the next signal if the hardware is ready
        if ready_to_write and (
            self.startup or skip_hardware or self.hardware.ready_for_new_output()
        ):
            remainder_log = [
                (environment, remainder.shape[-1])
                for environment, remainder in self.environment_data_out_remainders.items()
                if self.environment_active_flags[environment]
            ]
            self.log(f"Ready to Write: Environment Remainders {remainder_log}")
            write_data = np.zeros((self.num_outputs, self.write_size))
            for environment in self.environment_list:
                # If the task is shutting down and all the data has been drained from it,
                # make it inactive and just skip it.
                if (
                    self.environment_shutting_down_flags[environment]
                    and self.environment_data_out_remainders[environment].shape[-1] == 0
                ):
                    self.environment_active_flags[environment] = False
                    self.environment_starting_up_flags[environment] = False
                    self.queue_container.acquisition_command_queue.put(
                        self.process_name,
                        (GlobalCommands.STOP_ENVIRONMENT, environment),
                    )
                    self.environment_shutting_down_flags[environment] = False
                    continue
                # If the task is inactive, also just skip it
                elif not self.environment_active_flags[environment]:
                    continue
                # Get the indices corresponding to the output channels
                output_indices = self.environment_output_channels[environment]
                # Determine how many time steps are available to write
                output_timesteps = min(
                    self.environment_data_out_remainders[environment].shape[-1],
                    self.write_size,
                )
                # Write one portion of the environment output to write_data
                write_data[
                    output_indices, :output_timesteps
                ] += self.environment_data_out_remainders[environment][
                    :, :output_timesteps
                ]
                self.environment_data_out_remainders[environment] = (
                    self.environment_data_out_remainders[environment][
                        :, output_timesteps:
                    ]
                )
            # Now that we have each environment accounted for in the output we
            # can write to the hardware
            self.log(
                f"Writing {' x '.join(f'{shape}' for shape in write_data.shape)} data to hardware"
            )
            if not skip_hardware:
                self.log(
                    f"Output Writing Data to Hardware RMS: \n  {rms_time(write_data, axis=-1)}"
                )
                for environment in self.environment_first_data:
                    if self.environment_first_data[environment]:
                        self.log(
                            f"Sending first data for environment {environment} to Acquisition "
                            "for syncing"
                        )
                        self.queue_container.input_output_sync_queue.put(
                            (
                                environment,
                                write_data[..., :: self.output_oversample].copy(),
                            )
                        )
                        self.environment_first_data[environment] = False
                        if DEBUG:
                            np.savez(
                                ENV_OUTPUT.format(environment), write_data=write_data
                            )
                if DEBUG:
                    num_files = len(glob(FILE_OUTPUT.format("*")))
                    np.savez(FILE_OUTPUT.format(num_files), write_data=write_data)
                self.hardware.write(write_data.copy())
            else:
                if self.environment_first_data[environment]:
                    self.queue_container.input_output_sync_queue.put((environment, 0))
                    self.environment_first_data[environment] = False
            #            np.savez('test_data/output_data_check.npz',output_data = write_data)
            # Now check and see if we are starting up and start the hardare if so
            if self.startup:
                self.log("Starting Hardware Output")
                if not skip_hardware:
                    self.hardware.start()
                # Send something to the sync queue to tell acquisition to start now
                # We will send None because it is unique and we won't have an
                # environment with that name
                self.queue_container.input_output_sync_queue.put((None, True))
                self.startup = False
                self.set_active()
                # print('started output')
        # Now check if we need to shut down.
        if (
            self.shutdown_flag  # Time to shut down
            and all(
                [
                    not flag
                    for environment, flag in self.environment_active_flags.items()
                ]
            )  # Check that all environments are not active
            and all(
                [
                    not flag
                    for environment, flag in self.environment_starting_up_flags.items()
                ]
            )  # Check that all environments are not starting up
            and all(
                [
                    remainder.shape[-1] == 0
                    for environment, remainder in self.environment_data_out_remainders.items()
                ]
            )  # Check that all data is written
        ):
            self.log("Stopping Hardware")
            if not skip_hardware:
                self.hardware.stop()
            self.startup = True
            self.shutdown_flag = False
            flush_queue(self.queue_container.input_output_sync_queue)
            self.clear_active()
        else:
            # Otherwise keep going
            self.queue_container.output_command_queue.put(
                self.process_name, (GlobalCommands.RUN_HARDWARE, None)
            )

    def stop_output(self, data):  # pylint: disable=unused-argument
        """Sets a flag telling the output that it should start shutting down

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.log("Starting Shutdown Procedure")
        self.shutdown_flag = True

    def start_environment(self, data):
        """Sets the flag stating the specified environment is active

        Parameters
        ----------
        data : str
            The environment name that should be activated.

        """
        self.log(f"Started Environment {data}")
        self.environment_starting_up_flags[data] = True
        self.environment_shutting_down_flags[data] = False
        self.environment_active_flags[data] = False

    # endregion

    # region Shutdown
    def quit(self, data):  # pylint: disable=unused-argument
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
            q for _, q in self.queue_container.environment_data_out_queues.items()
        ] + [
            self.queue_container.output_command_queue,
            self.queue_container.input_output_sync_queue,
            self.queue_container.single_process_hardware_queue,
        ]:
            queue_flush_sum += len(flush_queue(queue))
        self.log(f"Flushed {queue_flush_sum} items out of queues")
        if self.hardware is not None:
            self.hardware.close()
        return True

    # endregion


# region Process
def output_process(
    queue_container: QueueContainer,
    output_active_event: mp.synchronize.Event,
    ready_event: mp.synchronize.Event,
    shutdown_event: mp.synchronize.Event,
    ping_alive_event: mp.synchronize.Event,
):
    """Function passed to multiprocessing as the output process

    This process creates the ``OutputProcess`` object and calls the ``run``
    command.

    Parameters
    ----------
    queue_container : QueueContainer
        A container containing the queues used to communicate between
        controller processes
    environments : list
        A list of ``(ControlType,environment_name)`` pairs that define the
        environments in the controller.

    """

    output_instance = OutputProcess(
        TASK_NAME, queue_container, output_active_event, ready_event
    )

    output_instance.run(shutdown_event)


# endregion
