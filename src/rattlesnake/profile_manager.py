"""Classes for validating and running a test profile of timed events.

A profile event list schedules commands (e.g., start/stop an environment,
start/stop streaming) to be fired at specific timestamps during a test."""

import threading
from datetime import datetime
from typing import List

from rattlesnake.environment.abstract_environment import EnvironmentInstructions
from rattlesnake.environment.environment_registry import ENVIRONMENT_COMMANDS

# unused import
# from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.user_interface.ui_utilities import UICommands
from rattlesnake.utilities import GlobalCommands, QueueContainer, RattlesnakeError

EXTRA_CLOSEOUT_TIME = 0.1  # Adds seconds to let the last profile event happen
TASK_NAME = "Profile Manager"
VALID_COMMANDS = {
    "Global": (
        GlobalCommands.STOP_HARDWARE,
        GlobalCommands.START_STREAMING,
        GlobalCommands.STOP_STREAMING,
    )
}
for control_type, command_type in ENVIRONMENT_COMMANDS.items():
    VALID_COMMANDS.update(
        {
            control_type: (
                GlobalCommands.START_ENVIRONMENT,
                GlobalCommands.STOP_ENVIRONMENT,
                UICommands.SET_ENVIRONMENT_INSTRUCTIONS,
                *command_type.valid_profile_commands(),
            )
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
    """A single scheduled command in a test profile

    Stores the timestamp at which the command should fire, the environment
    it applies to, the command itself, and any data the command requires.
    The queue_name and environment_type are assigned later by the
    environment manager once the event has been validated."""

    def __init__(self, timestamp: float, environment_name: str, command, data=None):
        """
        Parameters
        ----------
        timestamp : float
            The time, in seconds from the start of the profile, at which
            this event should fire
        environment_name : str
            The name of the environment this event applies to, or "Global"
            for controller-wide commands
        command : GlobalCommands or an environment-specific command Enum
            The command to execute when this event fires
        data : optional
            Data to pass along with the command (Default value = None)
        """
        self.timestamp = timestamp
        self.environment_name = environment_name
        self.command = command
        self.data = data
        self._environment_type = None
        self._queue_name = None

    @property
    def environment_type(self):
        """The EnvironmentType assigned to this event's environment_name"""
        return self._environment_type

    @property
    def queue_name(self):
        """The queue_name assigned to this event's environment_name"""
        return self._queue_name

    def validate(self):
        """Validates that this event is well-formed and ready to be fired

        Checks that environment_name, timestamp, environment_type, command,
        queue_name, and data are all valid and consistent with each other.

        Raises
        ------
        RattlesnakeError
            If any of the event's fields are invalid
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
    """Validates and runs a test profile, firing timed commands to the
    controller as each profile event's timestamp is reached."""

    def __init__(self, queue_container: QueueContainer):
        """
        Parameters
        ----------
        queue_container : QueueContainer
            The container holding the queues used to log messages and send
            commands to the controller
        """
        self._log_file_queue = queue_container.log_file_queue
        self._controller_command_queue = queue_container.controller_command_queue

        self.profile_timers = []
        self.gui_timer = None

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
    def log_file_queue(self):
        """The queue used to write messages to the log file"""
        return self._log_file_queue

    @property
    def controller_command_queue(self):
        """The queue used to send commands to the controller"""
        return self._controller_command_queue

    def validate_profile_list(self, profile_event_list: List[ProfileEvent]):
        """Validates a list of profile events and sorts it by timestamp

        Each event is checked for a valid type, internal consistency (via
        ProfileEvent.validate), and that its command has a handler
        registered in command_map.

        Parameters
        ----------
        profile_event_list : List[ProfileEvent]
            The list of profile events to validate. Sorted in place by
            timestamp once validation succeeds.

        Raises
        ------
        RattlesnakeError
            If the list contains an invalid type or an unimplemented
            command
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

    def start_profile(self, profile_event_list: List[ProfileEvent]):
        """Schedules a timer to fire each profile event at its timestamp

        Also schedules a final closeout event, EXTRA_CLOSEOUT_TIME seconds
        after the last profile event, to tell Rattlesnake the profile is
        finished.

        Parameters
        ----------
        profile_event_list : List[ProfileEvent]
            The validated, timestamp-sorted list of profile events to run
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

    def fire_profile_event(self, queue_name, command, data):
        """Logs and dispatches a single profile event to its command handler

        Parameters
        ----------
        queue_name : str
            The queue name of the environment this event applies to
        command : GlobalCommands or an environment-specific command Enum
            The command to execute
        data : optional
            Data to pass along with the command
        """
        self.log(f"Profile Firing Event {queue_name} {command} {data}")
        self.command_map[command](queue_name, command, data)

    def stop_profile(self):
        """Cancels all pending profile timers and schedules a closeout event"""
        self.log("Stopping Profile")
        for timer in self.profile_timers:
            timer.cancel()

        # Add closeout event
        timer = threading.Timer(EXTRA_CLOSEOUT_TIME, self.fire_closeout_event)
        timer.start()
        self.profile_timers.append(timer)

    def stop_hardware(self, queue_name: str, command: GlobalCommands, data: None):
        """Sends a STOP_HARDWARE command to the controller

        Parameters
        ----------
        queue_name : str
            Unused; present for a uniform command_map handler signature
        command : GlobalCommands
            Unused; present for a uniform command_map handler signature
        data : None
            Unused; present for a uniform command_map handler signature
        """
        self.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.STOP_HARDWARE, None)
        )

    def start_streaming(self, queue_name: str, command: GlobalCommands, data: None):
        """Sends a START_STREAMING command to the controller

        Parameters
        ----------
        queue_name : str
            Unused; present for a uniform command_map handler signature
        command : GlobalCommands
            Unused; present for a uniform command_map handler signature
        data : None
            Unused; present for a uniform command_map handler signature
        """
        self.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.START_STREAMING, False)
        )

    def stop_streaming(self, queue_name: str, command: GlobalCommands, data: None):
        """Sends a STOP_STREAMING command to the controller

        Parameters
        ----------
        queue_name : str
            Unused; present for a uniform command_map handler signature
        command : GlobalCommands
            Unused; present for a uniform command_map handler signature
        data : None
            Unused; present for a uniform command_map handler signature
        """
        self.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.STOP_STREAMING, None)
        )

    def start_environment(
        self, queue_name: str, command: GlobalCommands, data: EnvironmentInstructions
    ):
        """Sends a START_ENVIRONMENT command for the given environment

        Parameters
        ----------
        queue_name : str
            The queue name of the environment to start
        command : GlobalCommands
            Unused; present for a uniform command_map handler signature
        data : EnvironmentInstructions
            The instructions to start the environment with
        """
        instructions = data
        self.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.START_ENVIRONMENT, (queue_name, instructions))
        )

    def stop_environment(self, queue_name: str, command: GlobalCommands, data: None):
        """Sends a STOP_ENVIRONMENT command for the given environment

        Parameters
        ----------
        queue_name : str
            The queue name of the environment to stop
        command : GlobalCommands
            Unused; present for a uniform command_map handler signature
        data : None
            Unused; present for a uniform command_map handler signature
        """
        self.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.STOP_ENVIRONMENT, queue_name)
        )

    def send_environment_command(self, queue_name: str, command: GlobalCommands, data):
        """Forwards an environment-specific command to the given environment

        Parameters
        ----------
        queue_name : str
            The queue name of the environment to send the command to
        command : GlobalCommands or an environment-specific command Enum
            The environment-specific command to send
        data : optional
            Data to pass along with the command
        """
        self.controller_command_queue.put(
            TASK_NAME,
            (GlobalCommands.SEND_ENVIRONMENT_COMMAND, (queue_name, command, data)),
        )

    def fire_closeout_event(self):
        """Sends a PROFILE_CLOSEOUT command telling Rattlesnake the profile
        has finished firing all of its events"""
        self.controller_command_queue.put(
            TASK_NAME, (GlobalCommands.PROFILE_CLOSEOUT, None)
        )

    def log(self, message):
        """Write a message to the log file

        This function puts a message onto the ``log_file_queue`` so it will
        eventually be written to the log file.

        When written to the log file, the message will include the date and
        time that the message was queued, the name of the environment, and
        then the message itself.

        Parameters
        ----------
        message : str :
            A message that will be written to the log file.

        """
        self.log_file_queue.put(f"{datetime.now()}: {TASK_NAME} -- {message}\n")


# endregion
