"""Classes for validating and running a test profile of timed events.

A profile event list schedules commands (e.g., start/stop an environment,
start/stop streaming) to be fired at specific timestamps during a test.

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

import threading
import multiprocessing.queues as mp_queues
from datetime import datetime
from enum import Enum
from typing import Any, List

from rattlesnake.utilities import (
    GlobalCommands,
    QueueContainer,
    RattlesnakeError,
    VerboseMessageQueue,
)
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.user_interface.ui_utilities import UICommands
from rattlesnake.environment.abstract_environment import EnvironmentInstructions
from rattlesnake.environment.environment_registry import ENVIRONMENT_COMMANDS

EXTRA_CLOSEOUT_TIME = 0.1  # Adds seconds to let the last profile event happen
TASK_NAME = "Profile Manager"
VALID_COMMANDS = {
    "Global": [
        GlobalCommands.STOP_HARDWARE,
        GlobalCommands.START_STREAMING,
        GlobalCommands.STOP_STREAMING,
    ]
}
for control_type, command_type in ENVIRONMENT_COMMANDS.items():
    VALID_COMMANDS.update(
        {
            control_type: [
                GlobalCommands.START_ENVIRONMENT,
                GlobalCommands.STOP_ENVIRONMENT,
                UICommands.SET_ENVIRONMENT_INSTRUCTIONS,
                *command_type.valid_profile_commands(),
            ]
        }
    )


VALID_DATA = {
    GlobalCommands.STOP_HARDWARE: type(None),
    GlobalCommands.START_STREAMING: type(None),
    GlobalCommands.STOP_STREAMING: type(None),
    GlobalCommands.START_ENVIRONMENT: EnvironmentInstructions,
    GlobalCommands.STOP_ENVIRONMENT: type(None),
    UICommands.SET_ENVIRONMENT_INSTRUCTIONS: EnvironmentInstructions,
}
for control_type, command_type in ENVIRONMENT_COMMANDS.items():
    for command in command_type.valid_profile_commands():
        VALID_DATA[command] = command_type.valid_data()[command]


# region ProfileEvent
class ProfileEvent:
    """
    A container for a schedule command within a profile.

    Stores the timestamp, the environment it applies to, the command, and any data the
    command requires. The queue name and environment type are assigned by the
    ``EnvironmentManager`` during validation.

    Parameters
    ----------
    timestamp : float
        Time, in seconds from the start of the profile, at which the event fires.
    environment_name : str
        Name of the environment the event applies to, or ``"Global"`` for
        controller-wide commands.
    command : Enum
        ``GlobalCommands`` or environment-specific command to execute.
    data : Any, optional
        Data to pass along with the command. Defaults to ``None``.

    Attributes
    ----------
    timestamp : float
        Time, in seconds from the start of the profile, at which the event fires.
    environment_name : str
        Name of the environment the event applies to.
    command : Enum
        Command to execute when the event fires.
    data : Any
        Data to pass along with the command.
    environment_type : EnvironmentType or str or None
        Environment type assigned to the event.
    queue_name : str or None
        Queue name assigned to the event.

    Unit Tests
    ----------
    """

    def __init__(
        self,
        timestamp: float,
        environment_name: str,
        command: Enum,
        data: Any = None,
    ) -> None:
        """
        Initializes a profile event without a queue name or environment type.

        Parameters
        ----------
        timestamp : float
            Time, in seconds from the start of the profile, at which the event fires.
        environment_name : str
            Name of the environment the event applies to, or ``"Global"`` for
            controller-wide commands.
        command : Enum
            ``GlobalCommands`` or environment-specific command to execute.
        data : Any, optional
            Data to pass along with the command. Defaults to ``None``.

        Unit Tests
        ----------
        """
        self.timestamp = timestamp
        self.environment_name = environment_name
        self.command = command
        self.data = data
        self._environment_type = None
        self._queue_name = None

    @property
    def environment_type(self) -> EnvironmentType | str | None:
        """
        Return the environment type assigned to the event.

        Returns
        -------
        EnvironmentType or str or None
            Environment type of ``environment_name``, ``"Global"`` for controller-wide
            commands, or ``None`` if it has not been assigned.

        Unit Tests
        ----------
        """
        return self._environment_type

    @property
    def queue_name(self) -> str | None:
        """
        Return the queue name assigned to the event.

        Returns
        -------
        str or None
            Queue name of ``environment_name``, ``"Global"`` for controller-wide
            commands, or ``None`` if it has not been assigned.

        Unit Tests
        ----------
        """
        return self._queue_name

    def validate(self) -> None:
        """
        Validate that the event is ready to be fired.

        Raises
        ------
        RattlesnakeError
            If ``environment_name`` is not a string.
        RattlesnakeError
            If ``timestamp`` is not a non-negative number.
        RattlesnakeError
            If the environment type is not valid or has not been assigned.
        RattlesnakeError
            If ``command`` is not valid for the environment type.
        RattlesnakeError
            If the queue name has not been assigned.
        RattlesnakeError
            If ``data`` is not the type required by ``command``.
        RattlesnakeError
            If the instructions in ``data`` are for a different environment.

        Unit Tests
        ----------
        """
        # Check if environment_name is a string
        if not isinstance(self.environment_name, str):
            raise RattlesnakeError(
                f"{self.environment_name} is not a valid environment_name "
                "for a profile event"
            )
        # Check if timestamp is a number
        if not isinstance(self.timestamp, (int, float)) or self.timestamp < 0:
            raise RattlesnakeError(
                f"{self.environment_name} profile event was not given a valid timestamp"
            )
        # Check if a valid environment_type was given
        if self.environment_type not in VALID_COMMANDS.keys():
            raise RattlesnakeError(
                f"{self.environment_name} not given a valid environment "
                f"type: {self.environment_type}"
            )
        # Check if the environment_type has logic for that given command
        if self.command not in VALID_COMMANDS[self.environment_type]:
            raise RattlesnakeError(
                f"{self.command} is not a valid command for {self.environment_name}"
            )
        # Check if the environment_manager assigned a queue_name to the event yet
        if not self.queue_name:
            raise RattlesnakeError(
                f"{self.environment_name} was not given a valid queue_name "
                "before assignment"
            )
        # Validate data type going into command
        if self.command in VALID_DATA.keys():
            valid_data_type = VALID_DATA[self.command]
            if not isinstance(self.data, valid_data_type):
                raise RattlesnakeError(
                    f"{self.command} profile event was provided "
                    f"{type(self.data)}, but requires {valid_data_type}."
                )

            if valid_data_type is EnvironmentInstructions:
                if not self.data.environment_name == self.environment_name:
                    raise RattlesnakeError(
                        "Invalid environment instruction assigned to "
                        f"{self.environment_name} profile event"
                    )
                if not self.data.environment_type == self.environment_type:
                    raise RattlesnakeError(
                        "Invalid environment instruction assigned to "
                        f"{self.environment_name} profile event"
                    )


# endregion


# region Manager
class ProfileManager:
    """
    Manager that builds out timers and fires commands to perform profile events.

    This class schedules a timer for each ``ProfileEvent`` from a profile. When an
    event's timer expires, the event is dispatched through ``command_map`` to a handler
    that puts the command on the controller command queue. After the last event, a
    closeout event tells the controller that the profile has finished.

    Parameters
    ----------
    queue_container : QueueContainer
        Container holding the log file and controller command queues.

    Attributes
    ----------
    log_file_queue : multiprocessing.queues.Queue
        Queue used to write messages to the log file.
    controller_command_queue : VerboseMessageQueue
        Queue used to send commands to the controller process.
    profile_timers : list of threading.Timer
        Timers started for the current profile, including closeout timers.
    command_map : dict of Enum to callable
        Mapping from commands to the handlers that send them to the controller.

    Unit Tests
    ----------
    """

    def __init__(self, queue_container: QueueContainer) -> None:
        """
        Initializes a profile manager without a running profile.

        Stores the queues and maps the commands to their handlers. Environment commands
        are mapped based off the dictionaries set up in the registry.

        Parameters
        ----------
        queue_container : QueueContainer
            Container holding the log file and controller command queues.

        Unit Tests
        ----------
        """
        self._log_file_queue = queue_container.log_file_queue
        self._controller_command_queue = queue_container.controller_command_queue

        self.profile_timers = []

        self.command_map = {}
        self.command_map[GlobalCommands.STOP_HARDWARE] = self.stop_hardware
        self.command_map[GlobalCommands.START_STREAMING] = self.start_streaming
        self.command_map[GlobalCommands.STOP_STREAMING] = self.stop_streaming
        self.command_map[GlobalCommands.START_ENVIRONMENT] = self.start_environment
        self.command_map[GlobalCommands.STOP_ENVIRONMENT] = self.stop_environment
        for command_type in ENVIRONMENT_COMMANDS.values():
            self.command_map.update(
                {command: self.send_environment_command for command in command_type}
            )

    @property
    def log_file_queue(self) -> mp_queues.Queue:
        """
        Return the queue used to write messages to the log file.

        Returns
        -------
        multiprocessing.queues.Queue
            Queue read by the log file process.

        Unit Tests
        ----------
        """
        return self._log_file_queue

    @property
    def controller_command_queue(self) -> VerboseMessageQueue:
        """
        Return the queue used to send commands to the controller.

        Returns
        -------
        VerboseMessageQueue
            Queue read by the controller process.

        Unit Tests
        ----------
        """
        return self._controller_command_queue

    def validate_profile_list(self, profile_event_list: List[ProfileEvent]) -> None:
        """
        Validate a list of profile events and sort it by timestamp.

        Each event is checked with ``ProfileEvent.validate``. The list is only sorted if
        every event is valid.

        Parameters
        ----------
        profile_event_list : list of ProfileEvent
            Profile events to validate. Sorted in place by timestamp.

        Raises
        ------
        RattlesnakeError
            If an item in ``profile_event_list`` is not a ``ProfileEvent``.
        RattlesnakeError
            If a profile event fails ``ProfileEvent.validate``.
        RattlesnakeError
            If a profile event's command does not have a handler in ``command_map``.

        Unit Tests
        ----------
        """
        for profile_event in profile_event_list:
            if not isinstance(profile_event, ProfileEvent):
                raise RattlesnakeError("Profile event list contains invalid type")
            # Validate profile event
            profile_event.validate()

            # Validate command has been implemented in profile_manager
            command = profile_event.command
            if command not in self.command_map.keys():
                raise RattlesnakeError(
                    f"No profile event has been implemented for {profile_event.command}"
                )

        # Sort profile_event_list by timestamp
        profile_event_list.sort(key=lambda event: event.timestamp)

    def start_profile(self, profile_event_list: List[ProfileEvent]) -> None:
        """
        Start the timers for each profile event.

        Timestamps are measured from when this function is called. A closeout event is
        also scheduled ``EXTRA_CLOSEOUT_TIME`` seconds after the latest profile event to
        tell the controller that the profile has finished. This function returns once
        the timers are started and does not wait for the profile to finish.

        Parameters
        ----------
        profile_event_list : list of ProfileEvent
            Validated profile events to run.

        Unit Tests
        ----------
        """
        self.log("Starting Profile")
        self.profile_timers = []
        max_timestamp = 0
        for profile_event in profile_event_list:
            # Expand data
            timestamp = profile_event.timestamp
            queue_name = profile_event.queue_name
            command = profile_event.command
            data = profile_event.data

            # Fire event
            timer = threading.Timer(
                timestamp, self.fire_profile_event, args=(queue_name, command, data)
            )
            timer.start()
            self.profile_timers.append(timer)

            if timestamp > max_timestamp:
                max_timestamp = timestamp

        # Fire a last profile event to tell Rattlesnake it is good to stop_acquisition
        timer = threading.Timer(
            max_timestamp + EXTRA_CLOSEOUT_TIME, self.fire_closeout_event
        )
        timer.start()
        self.profile_timers.append(timer)

    def fire_profile_event(self, queue_name: str, command: Enum, data: Any) -> None:
        """
        The function that runs when a profile timer expires.

        Checks command map for which function to run and logs that the command has been
        fired.

        Parameters
        ----------
        queue_name : str
            Queue name assigned to the event.
        command : Enum
            ``GlobalCommands`` or environment-specific command to execute.
        data : Any
            Data to pass along with the command.

        Unit Tests
        ----------
        """
        self.log(f"Profile Firing Event {queue_name} {command} {data}")
        self.command_map[command](queue_name, command, data)

    def stop_profile(self) -> None:
        """
        Cancel the pending profile timers and schedules a new closeout event.

        Events that have already fired are not undone. The closeout event fires
        ``EXTRA_CLOSEOUT_TIME`` seconds after this function is called.

        Unit Tests
        ----------
        """
        self.log("Stopping Profile")
        for timer in self.profile_timers:
            timer.cancel()

        # Add closeout event
        timer = threading.Timer(EXTRA_CLOSEOUT_TIME, self.fire_closeout_event)
        timer.start()
        self.profile_timers.append(timer)

    def stop_hardware(
        self, queue_name: str, command: GlobalCommands, data: None
    ) -> None:
        """
        Send a ``STOP_HARDWARE`` command to the controller.

        Parameters
        ----------
        queue_name : str
            Unused. Required by the ``command_map`` handler signature.
        command : GlobalCommands
            Unused. Required by the ``command_map`` handler signature.
        data : None
            Unused. Required by the ``command_map`` handler signature.

        Unit Tests
        ----------
        """
        self.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.STOP_HARDWARE, None)
        )

    def start_streaming(
        self, queue_name: str, command: GlobalCommands, data: None
    ) -> None:
        """
        Send a ``START_STREAMING`` command to the controller.

        The command is sent without the override, so streaming only starts if the stream
        type is ``StreamType.PROFILE_INSTRUCTION``.

        Parameters
        ----------
        queue_name : str
            Unused. Required by the ``command_map`` handler signature.
        command : GlobalCommands
            Unused. Required by the ``command_map`` handler signature.
        data : None
            Unused. Required by the ``command_map`` handler signature.

        Unit Tests
        ----------
        """
        self.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.START_STREAMING, False)
        )

    def stop_streaming(
        self, queue_name: str, command: GlobalCommands, data: None
    ) -> None:
        """
        Send a ``STOP_STREAMING`` command to the controller.

        Parameters
        ----------
        queue_name : str
            Unused. Required by the ``command_map`` handler signature.
        command : GlobalCommands
            Unused. Required by the ``command_map`` handler signature.
        data : None
            Unused. Required by the ``command_map`` handler signature.

        Unit Tests
        ----------
        """
        self.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.STOP_STREAMING, None)
        )

    def start_environment(
        self, queue_name: str, command: GlobalCommands, data: EnvironmentInstructions
    ) -> None:
        """
        Send a ``START_ENVIRONMENT`` command to the controller.

        The controller forwards the command to the environment.

        Parameters
        ----------
        queue_name : str
            Queue name of the environment to start.
        command : GlobalCommands
            Unused. Required by the ``command_map`` handler signature.
        data : EnvironmentInstructions
            Instructions used to start the environment.

        Unit Tests
        ----------
        """
        instructions = data
        self.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.START_ENVIRONMENT, (queue_name, instructions))
        )

    def stop_environment(
        self, queue_name: str, command: GlobalCommands, data: None
    ) -> None:
        """
        Send a ``STOP_ENVIRONMENT`` command to the controller.

        The controller forwards the command to the environment.

        Parameters
        ----------
        queue_name : str
            Queue name of the environment to stop.
        command : GlobalCommands
            Unused. Required by the ``command_map`` handler signature.
        data : None
            Unused. Required by the ``command_map`` handler signature.

        Unit Tests
        ----------
        """
        self.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.STOP_ENVIRONMENT, queue_name)
        )

    def send_environment_command(
        self, queue_name: str, command: Enum, data: Any
    ) -> None:
        """
        Send an environment-specific command to the controller.

        The controller forwards the command to the environment.

        Parameters
        ----------
        queue_name : str
            Queue name of the environment to send the command to.
        command : Enum
            Environment-specific command to send.
        data : Any
            Data to pass along with the command.

        Unit Tests
        ----------
        """
        self.controller_command_queue.put(
            TASK_NAME,
            (GlobalCommands.SEND_ENVIRONMENT_COMMAND, (queue_name, command, data)),
        )

    def fire_closeout_event(self) -> None:
        """
        Send a ``PROFILE_CLOSEOUT`` command to the controller.

        This tells the controller that the profile has finished so it can set its ready
        event. This event signals to the ``RattlesnakeController`` that a profile is no
        longer being run.

        Unit Tests
        ----------
        """
        self.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.PROFILE_CLOSEOUT, None)
        )

    def log(self, message: str) -> None:
        """
        Send a message to the log file task.

        Parameters
        ----------
        message : str
            A message that will be written to the log file.

        Unit Tests
        ----------
        """
        self.log_file_queue.put(f"{datetime.now()}: {TASK_NAME} -- {message}\n")


# endregion
