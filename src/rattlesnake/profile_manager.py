from rattlesnake.utilities import RattlesnakeError, QueueContainer, GlobalCommands
from rattlesnake.environment.abstract_environment import EnvironmentInstructions
from rattlesnake.environment.environment_utilities import ControlTypes
from rattlesnake.environment.environment_registry import ENVIRONMENT_COMMANDS
from rattlesnake.environment.time_environment import TimeCommands
from rattlesnake.environment.sine_environment import SineCommands
from rattlesnake.user_interface.ui_utilities import UICommands
import threading
from typing import List
from datetime import datetime

EXTRA_CLOSEOUT_TIME = 0.1  # Adds seconds to let the last profile event happen
TASK_NAME = "Profile Manager"
VALID_COMMANDS = {"Global": (GlobalCommands.STOP_HARDWARE, GlobalCommands.START_STREAMING, GlobalCommands.STOP_STREAMING)}
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


# region: ProfileEvent
class ProfileEvent:
    def __init__(self, timestamp: float, environment_name: str, command, data=None):
        self.timestamp = timestamp
        self.environment_name = environment_name
        self.command = command
        self.data = data
        self._environment_type = None
        self._queue_name = None

    @property
    def environment_type(self):
        return self._environment_type

    @property
    def queue_name(self):
        return self._queue_name

    def validate(self):
        # Check if environment_name is a string
        if not isinstance(self.environment_name, str):
            raise RattlesnakeError(f"{self.environment_name} is not a valid environment_name for a profile event")
        # Check if timestamp is a number
        if not isinstance(self.timestamp, (int, float)) or self.timestamp < 0:
            raise RattlesnakeError(f"{self.environment_name} profile event was not given a valid timestamp")
        # Check if a valid environment_type was given
        if self.environment_type not in VALID_COMMANDS.keys():
            raise RattlesnakeError(f"{self.environment_name} not given a valid environment type: {self.environment_type}")
        # Check if the environment_type has logic for that given command
        if self.command not in VALID_COMMANDS[self.environment_type]:
            raise RattlesnakeError(f"{self.command} is not a valid command for {self.environment_name}")
        # Check if the environment_manager assigned a queue_name to the event yet
        if not self.queue_name:
            raise RattlesnakeError(f"{self.environment_name} was not given a valid queue_name before assignment")
        # Validate data type going into command
        if self.command in VALID_DATA.keys():
            valid_data_type = VALID_DATA[self.command]
            if not isinstance(self.data, valid_data_type):
                raise RattlesnakeError(f"{self.command} profile event was provided {type(self.data)}, but requires {valid_data_type}.")

            if valid_data_type is EnvironmentInstructions:
                if not self.data.environment_name == self.environment_name:
                    raise RattlesnakeError(f"Invalid environment instruction assigned to {self.environment_name} profile event")
                if not self.data.environment_type == self.environment_type:
                    raise RattlesnakeError(f"Invalid environment instruction assigned to {self.environment_name} profile event")

        return True


# region: ProfileManager
class ProfileManager:
    def __init__(self, queue_container: QueueContainer):
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
            self.command_map.update({command: self.send_environment_command for command in command_type})

    @property
    def log_file_queue(self):
        return self._log_file_queue

    @property
    def controller_command_queue(self):
        return self._controller_command_queue

    def validate_profile_list(self, profile_event_list: List[ProfileEvent]):
        """Validate list of profile events. Since each event needs"""
        for profile_event in profile_event_list:
            if not isinstance(profile_event, ProfileEvent):
                raise RattlesnakeError("Profile event list contains invalid type")
            # Validate profile event
            valid_profile = profile_event.validate()
            if not valid_profile:
                raise RattlesnakeError("Rattlesnake.set_profile requires a valid list of ProfileEvents")

            # Validate command has been implemented in profile_manager
            command = profile_event.command
            if command not in self.command_map.keys():
                raise RattlesnakeError(f"No profile event has been implemented for {profile_event.command}")

        # Sort profile_event_list by timestamp
        profile_event_list.sort(key=lambda event: event.timestamp)

        return True

    def start_profile(self, profile_event_list: List[ProfileEvent]):
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
            timer = threading.Timer(timestamp, self.fire_profile_event, args=(queue_name, command, data))
            timer.start()
            self.profile_timers.append(timer)

            if timestamp > max_timestamp:
                max_timestamp = timestamp

        # Fire a last profile event to tell Rattlesnake it is good to stop_acquisition
        timer = threading.Timer(max_timestamp + EXTRA_CLOSEOUT_TIME, self.fire_closeout_event)
        timer.start()
        self.profile_timers.append(timer)

    def fire_profile_event(self, queue_name, command, data):
        self.log(f"Profile Firing Event {queue_name} {command} {data}")
        self.command_map[command](queue_name, command, data)

    def stop_profile(self):
        self.log("Stopping Profile")
        for timer in self.profile_timers:
            timer.cancel()

        # Add closeout event
        timer = threading.Timer(EXTRA_CLOSEOUT_TIME, self.fire_closeout_event)
        timer.start()
        self.profile_timers.append(timer)

    def stop_hardware(self, queue_name: str, command: GlobalCommands, data: None):
        self.controller_command_queue.put(TASK_NAME, (GlobalCommands.STOP_HARDWARE, None))

    def start_streaming(self, queue_name: str, command: GlobalCommands, data: None):
        self.controller_command_queue.put(TASK_NAME, (GlobalCommands.START_STREAMING, False))

    def stop_streaming(self, queue_name: str, command: GlobalCommands, data: None):
        self.controller_command_queue.put(TASK_NAME, (GlobalCommands.STOP_STREAMING, None))

    def start_environment(self, queue_name: str, command: GlobalCommands, data: EnvironmentInstructions):
        instructions = data
        self.controller_command_queue.put(TASK_NAME, (GlobalCommands.START_ENVIRONMENT, (queue_name, instructions)))

    def stop_environment(self, queue_name: str, command: GlobalCommands, data: None):
        self.controller_command_queue.put(TASK_NAME, (GlobalCommands.STOP_ENVIRONMENT, queue_name))

    def send_environment_command(self, queue_name: str, command: GlobalCommands, data):
        self.controller_command_queue.put(TASK_NAME, (GlobalCommands.SEND_ENVIRONMENT_COMMAND, (queue_name, command, data)))

    def fire_closeout_event(self):
        self.controller_command_queue.put(TASK_NAME, (GlobalCommands.PROFILE_CLOSEOUT, None))

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
