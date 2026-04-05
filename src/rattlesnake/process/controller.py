import multiprocessing as mp

from rattlesnake.environment.abstract_environment import EnvironmentInstructions
from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.process.streaming import StreamMetadata, StreamType
from rattlesnake.utilities import QueueContainer, GlobalCommands

TASK_NAME = "Controller"


# region: ControllerProcess
class ControllerProcess(AbstractMessageProcess):
    """Class defining behavior during the OUTPUT_START states of Rattlesnake.

    This class recieves commands from controller_command_queue and sends those
    commands to the correct processes that need them. These commands should be
    sent from the Rattlesnake() class or ProfileManager() class to avoid
    confusion. The main goal for this class is to automatically perform tasks
    that the user would usually do in the UI.

    This class is also in charge of the stream_process

    See AbstractMesssageProcess for inherited class members.
    """

    def __init__(
        self,
        process_name: str,
        queue_container: QueueContainer,
        acquisition_active_event: mp.synchronize.Event,
        output_active_event: mp.synchronize.Event,
        streaming_active_event: mp.synchronize.Event,
        environment_active_event: mp.synchronize.Event,
        environment_sysid_event: mp.synchronize.Event,
        ready_event: mp.synchronize.Event,
    ):
        """Constructor for the Controller class

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
            queue_container.controller_command_queue,
            queue_container.gui_update_queue,
            ready_event,
        )
        self.queue_container = queue_container
        self._acquisition_active_event = acquisition_active_event
        self._output_active_event = output_active_event
        self._streaming_active_event = streaming_active_event
        self._environment_active_event = environment_active_event
        self._environment_sysid_event = environment_sysid_event
        self.stream_metadata = StreamMetadata()
        self.map_command(GlobalCommands.RUN_HARDWARE, self.run_hardware)
        self.map_command(GlobalCommands.STOP_HARDWARE, self.stop_hardware)
        self.map_command(
            GlobalCommands.START_SYSTEM_ID_NOISE, self.start_system_id_noise
        )
        self.map_command(
            GlobalCommands.START_SYSTEM_ID_TRANSFER, self.start_system_id_transfer
        )
        self.map_command(GlobalCommands.STOP_SYSTEM_ID, self.stop_system_id)
        # self.map_command(GlobalCommands.START_ENVIRONMENT, self.start_environment)
        self.map_command(GlobalCommands.STOP_ENVIRONMENT, self.stop_environment)
        self.map_command(GlobalCommands.START_STREAMING, self.start_streaming)
        self.map_command(GlobalCommands.STOP_STREAMING, self.stop_streaming)
        self.map_command(GlobalCommands.STREAM_AT_TARGET_LEVEL, self.at_target_level)
        self.map_command(GlobalCommands.STREAM_MANUAL, self.manual_stream)
        self.map_command(GlobalCommands.PROFILE_CLOSEOUT, self.profile_closeout)
        self.map_command(
            GlobalCommands.SEND_ENVIRONMENT_COMMAND, self.send_environment_command
        )

    @property
    def acquisition_active(self):
        return self._acquisition_active_event.is_set()

    @property
    def output_active(self):
        return self._output_active_event.is_set()

    @property
    def streaming_active(self):
        return self._streaming_active_event.is_set()

    @property
    def environments_active(self):
        return [
            queue_name
            for queue_name, active_event in self._environment_active_event.items()
            if active_event.is_set()
        ]

    @property
    def environments_sysid_active(self):
        return [
            queue_name
            for queue_name, sysid_event in self._environment_sysid_event.items()
            if sysid_event.is_set()
        ]

    def run_hardware(self, data: StreamMetadata):
        self.stream_metadata = data
        if self.acquisition_active:
            raise RuntimeError(
                "Tried to start hardware while acquisition is still running"
            )
        self.queue_container.acquisition_command_queue.put(
            TASK_NAME, (GlobalCommands.RUN_HARDWARE, None)
        )
        if self.output_active:
            raise RuntimeError("Tried to start hardware while output is still running")
        self.queue_container.output_command_queue.put(
            TASK_NAME, (GlobalCommands.RUN_HARDWARE, None)
        )
        if self.stream_metadata.stream_type == StreamType.IMMEDIATELY:
            self.start_streaming(True)

    def stop_hardware(self, data: None = None):
        # Stop environments
        for queue_name in self.environments_sysid_active:
            self.stop_system_id(queue_name)
        for queue_name in self.environments_active:
            self.stop_environment(queue_name)
        if self.streaming_active:
            self.stop_streaming()
        if self.stream_metadata.stream_type is not StreamType.NO_STREAM:
            self.queue_container.streaming_command_queue.put(
                self.process_name, (GlobalCommands.FINALIZE_STREAMING, None)
            )

        # Stop acquisition. I flip these for error handling
        self.queue_container.acquisition_command_queue.put(
            TASK_NAME, (GlobalCommands.STOP_HARDWARE, None)
        )
        self.queue_container.output_command_queue.put(
            TASK_NAME, (GlobalCommands.STOP_HARDWARE, None)
        )
        if not self.acquisition_active:
            raise RuntimeError(
                "Tried to stop hardware when acquisition was not running"
            )
        if not self.output_active:
            raise RuntimeError("Tried to start hardware when output was not running")

    def start_system_id_noise(self, data):
        queue_name = data

        # Have output send data to environment
        self.queue_container.output_command_queue.put(
            TASK_NAME, (GlobalCommands.START_ENVIRONMENT, queue_name)
        )
        self.queue_container.environment_command_queues[queue_name].put(
            TASK_NAME, (GlobalCommands.START_SYSTEM_ID_NOISE, None)
        )

    def start_system_id_transfer(self, data):
        queue_name = data
        self.queue_container.output_command_queue.put(
            TASK_NAME, (GlobalCommands.START_ENVIRONMENT, queue_name)
        )
        self.queue_container.environment_command_queues[queue_name].put(
            TASK_NAME, (GlobalCommands.START_SYSTEM_ID_TRANSFER, None)
        )

    def stop_system_id(self, data):
        queue_name = data
        self.queue_container.environment_command_queues[queue_name].put(
            TASK_NAME, (GlobalCommands.STOP_SYSTEM_ID, True)
        )

    def start_environment(self, data: tuple[str, EnvironmentInstructions]):
        queue_name, instruction = data
        if queue_name in self.environments_active:
            raise RuntimeError(
                f"Tried to start {queue_name} environment while it was still running"
            )
        self.queue_container.output_command_queue.put(
            TASK_NAME, (GlobalCommands.START_ENVIRONMENT, queue_name)
        )
        self.queue_container.environment_command_queues[queue_name].put(
            TASK_NAME, (GlobalCommands.START_ENVIRONMENT, instruction)
        )

    def stop_environment(self, data: str):
        queue_name = data
        if queue_name not in self.environments_active:
            raise RuntimeError(
                f"Tried to stop {queue_name} environment when it was not running"
            )
        self.queue_container.environment_command_queues[queue_name].put(
            TASK_NAME, (GlobalCommands.STOP_ENVIRONMENT, None)
        )

    def start_streaming(self, data: bool = False):
        # This function has an override so that the controller can still start streaming through this even if the stream_type
        # is not STREAM_TYPE.PROFILE_INSTRUCTION
        override = data
        # I split these up for debugging purposes
        if override:
            self.queue_container.acquisition_command_queue.put(
                TASK_NAME, (GlobalCommands.START_STREAMING, None)
            )
        elif self.stream_metadata.stream_type == StreamType.PROFILE_INSTRUCTION:
            self.queue_container.acquisition_command_queue.put(
                TASK_NAME, (GlobalCommands.START_STREAMING, None)
            )

    def stop_streaming(self, data: None = None):
        self.queue_container.acquisition_command_queue.put(
            TASK_NAME, (GlobalCommands.STOP_STREAMING, None)
        )

    def at_target_level(self, data: str):
        environment_name = data
        if (
            self.stream_metadata.stream_type == StreamType.TEST_LEVEL
            and self.stream_metadata.test_level_environment_name == environment_name
        ):
            self.start_streaming(True)

    def manual_stream(self, data: None = None):
        if self.stream_metadata.stream_type == StreamType.MANUAL:
            self.start_streaming(True)

    def send_environment_command(self, data):
        queue_name, command, command_data = data
        self.queue_container.environment_command_queues[queue_name].put(
            TASK_NAME, (command, command_data)
        )

    def profile_closeout(self, data: None):
        self.set_ready()


# region: controller_process
def controller_process(
    queue_container: QueueContainer,
    acquisition_active_event: mp.synchronize.Event,
    output_active_event: mp.synchronize.Event,
    streaming_active_event: mp.synchronize.Event,
    environment_active_event: mp.synchronize.Event,
    environment_sysid_event: mp.synchronize.Event,
    ready_event: mp.synchronize.Event,
    shutdown_event: mp.synchronize.Event,
):
    """Function passed to multiprocessing as the controller process

    This process creates the ``Controller`` object and calls the ``run``
    command.

    Parameters
    ----------
    queue_container : QueueContainer
        A container containing the queues used to communicate between
        controller processes

    """

    acquisition_instance = ControllerProcess(
        TASK_NAME,
        queue_container,
        acquisition_active_event,
        output_active_event,
        streaming_active_event,
        environment_active_event,
        environment_sysid_event,
        ready_event,
    )

    acquisition_instance.run(shutdown_event)
