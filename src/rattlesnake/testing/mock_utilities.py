from rattlesnake.hardware.hardware_utilities import Channel
from rattlesnake.utilities import (
    QueueContainer,
    EventContainer,
    VerboseMessageQueue,
    flush_queue,
)
import atexit
import multiprocessing as mp
import threading
import queue as thqueue
from unittest import mock

MAX_ENVIRONMENTS = 4

# A single shared manager for queue names.  Creating one mp.Manager per
# mock_queue_container call starts a new server process per test, and the
# accumulated leaked processes can hang the interpreter at exit on spawn
# platforms (macOS).
_QUEUE_NAME_MANAGER = None


def queue_name_manager():
    global _QUEUE_NAME_MANAGER
    if _QUEUE_NAME_MANAGER is None:
        _QUEUE_NAME_MANAGER = mp.Manager()
        atexit.register(_QUEUE_NAME_MANAGER.shutdown)
    return _QUEUE_NAME_MANAGER


def flush_queue_container(queue_container):
    """Drains every real queue in a QueueContainer and detaches its feeder.

    Use in fixture teardown: an mp.Queue with undelivered data leaves its
    feeder thread blocked on the pipe, which hangs the interpreter at exit.
    Draining is best-effort (non-blocking); cancel_join_thread guarantees
    the exit does not wait on the feeder regardless.  Queues a test replaced
    with mocks are skipped — a mock's get() never raises Empty, so flushing
    one would loop forever.
    """
    queues = [
        queue_container.controller_command_queue,
        queue_container.acquisition_command_queue,
        queue_container.output_command_queue,
        queue_container.streaming_command_queue,
        queue_container.input_output_sync_queue,
        queue_container.single_process_hardware_queue,
        queue_container.gui_update_queue,
        *queue_container.environment_command_queues.values(),
        *queue_container.environment_data_in_queues.values(),
        *queue_container.environment_data_out_queues.values(),
        queue_container.log_file_queue,
    ]
    for queue in queues:
        if isinstance(queue, VerboseMessageQueue):
            base_queue = queue.base_queue
        elif isinstance(queue, (mp.queues.Queue, thqueue.Queue)):
            base_queue = queue
        else:
            continue
        flush_queue(queue)
        if hasattr(base_queue, "cancel_join_thread"):
            base_queue.cancel_join_thread()


def mock_channel_list():
    response_channel = Channel()
    for attr in response_channel.channel_attr_list:
        setattr(response_channel, attr, attr)
    response_channel.feedback_channel = None
    response_channel.feedback_device = None

    excitation_channel = Channel()
    for attr in excitation_channel.channel_attr_list:
        setattr(excitation_channel, attr, attr)

    return [response_channel, excitation_channel]


def mock_channel_list_bools():
    return [True, True]


def mock_queue_container(use_thread):
    if use_thread:
        new_queue = thqueue.Queue
    else:
        new_queue = mp.Queue

    controller_queue_name_manager = queue_name_manager()
    log_file_queue = mp.Queue()
    controller_command_queue = VerboseMessageQueue(
        log_file_queue,
        new_queue(),
        "Controller Command Queue",
        controller_queue_name_manager,
    )
    acquisition_command_queue = VerboseMessageQueue(
        log_file_queue,
        new_queue(),
        "Acquisition Command Queue",
        controller_queue_name_manager,
    )
    output_command_queue = VerboseMessageQueue(
        log_file_queue,
        mp.Queue(),
        "Output Command Queue",
        controller_queue_name_manager,
    )
    streaming_command_queue = VerboseMessageQueue(
        log_file_queue,
        new_queue(),
        "Streaming Command Queue",
        controller_queue_name_manager,
    )
    environment_command_queues = {}
    environment_data_in_queues = {}
    environment_data_out_queues = {}
    for env_idx in range(MAX_ENVIRONMENTS):
        environment_name = "Environment {:}".format(env_idx)
        environment_command_queues[environment_name] = VerboseMessageQueue(
            log_file_queue,
            mp.Queue(),
            environment_name + " Command Queue",
            controller_queue_name_manager,
        )
        environment_data_in_queues[environment_name] = new_queue()
        environment_data_out_queues[environment_name] = new_queue()

    input_output_sync_queue = new_queue()
    single_process_hardware_queue = new_queue()
    gui_update_queue = new_queue()
    queue_container = QueueContainer(
        controller_command_queue,
        acquisition_command_queue,
        output_command_queue,
        streaming_command_queue,
        log_file_queue,
        input_output_sync_queue,
        single_process_hardware_queue,
        gui_update_queue,
        environment_command_queues,
        environment_data_in_queues,
        environment_data_out_queues,
    )

    return queue_container


def mock_event_container(use_thread):
    if use_thread:
        new_event = threading.Event
    else:
        new_event = mp.Event

    log_close_event = mp.Event()
    controller_close_event = new_event()
    controller_ready_event = new_event()
    acquisition_close_event = new_event()
    acquisition_ready_event = new_event()
    acquisition_active_event = new_event()
    acquisition_active_event.clear()
    output_close_event = new_event()
    output_ready_event = new_event()
    output_active_event = new_event()
    output_active_event.clear()
    streaming_close_event = new_event()
    streaming_ready_event = new_event()
    streaming_active_event = new_event()
    streaming_active_event.clear()

    environment_close_events = {}
    environment_ready_events = {}
    environment_active_events = {}
    environment_sysid_active_events = {}
    environment_sysid_stored_events = {}
    for env_idx in range(MAX_ENVIRONMENTS):
        environment_name = "Environment {:}".format(env_idx)
        environment_close_events[environment_name] = new_event()
        environment_ready_events[environment_name] = new_event()
        environment_active_events[environment_name] = new_event()
        environment_sysid_active_events[environment_name] = new_event()
        environment_sysid_stored_events[environment_name] = new_event()
        environment_active_events[environment_name].clear()
        environment_sysid_active_events[environment_name].clear()
    ping_alive_event = new_event()

    event_container = EventContainer(
        controller_ready_event,
        acquisition_ready_event,
        output_ready_event,
        streaming_ready_event,
        environment_ready_events,
        log_close_event,
        controller_close_event,
        acquisition_close_event,
        output_close_event,
        streaming_close_event,
        environment_close_events,
        acquisition_active_event,
        output_active_event,
        streaming_active_event,
        environment_active_events,
        environment_sysid_active_events,
        environment_sysid_stored_events,
        ping_alive_event,
    )

    return event_container


def fake_time():
    return "Datetime"


def clear_verbose_queue(q, task_name, verbose_array):
    # Mock the datetime and message_id objects used during the log message in the VerboseQueue.get function
    with (
        mock.patch("rattlesnake.utilities.datetime") as mock_time,
        mock.patch(
            "rattlesnake.utilities.VerboseMessageQueue.generate_message_id"
        ) as mock_id,
    ):
        mock_time.now = fake_time
        mock_id.return_value = "1"

        # Clear the queue and store data to verbose_array
        idx = 0
        while not q.empty():
            output_value = q.get(task_name)
            verbose_array[idx] = output_value[1]
            idx += 1


# Clear the log_file_queue
def clear_log_queue(q, log_string):
    # Get string from queue and store it to the log_string bstring
    while not q.empty():
        output_string = q.get()
        output_string = output_string.encode("utf-8")
        log_string.value = log_string.value + output_string
