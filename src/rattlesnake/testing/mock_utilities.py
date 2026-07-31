import inspect
from unittest import mock
import threading
import queue as thqueue
import signal

import numpy as np

from rattlesnake.hardware.hardware_utilities import Channel, HardwareType
from rattlesnake.hardware.skeleton_hardware import SkeletonHardwareMetadata
from rattlesnake.hardware.hardware_registry import UNIMPLEMENTED_HARDWARE
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.skeleton_environment import (
    SkeletonEnvironment,
    SkeletonInstructions,
    SkeletonMetadata,
    SkeletonQueues,
)
from rattlesnake.environment.skeleton_sys_id_environment import (
    SkeletonSysIdEnvironment,
    SkeletonSysIdMetadata,
    SkeletonSysIdQueues,
)
from rattlesnake.environment.environment_registry import UNIMPLEMENTED_ENVIRONMENT
from rattlesnake.process.abstract_sysid_data_analysis import (
    SysIdDataPackage,
    SysIdMetadata,
)
from rattlesnake.utilities import QueueContainer, EventContainer, VerboseMessageQueue
import multiprocessing as mp

MAX_ENVIRONMENTS = 4


# region Main
def instantiate_with_mocks(cls, **overrides):
    """
    Instantiate `cls` by passing a MagicMock for every constructor argument,
    except arguments explicitly provided in `overrides`.
    """
    signature = inspect.signature(cls)

    positional_args = []
    keyword_args = {}

    for name, parameter in signature.parameters.items():
        if name == "self":
            continue

        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            continue

        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        value = overrides.get(name, mock.MagicMock(name=name))

        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            positional_args.append(value)
        else:
            keyword_args[name] = value

    return cls(*positional_args, **keyword_args)


def mock_queue_container(use_thread=True):
    """Build out queue container with multiprocessing or threaded queues."""
    if use_thread:
        new_queue = thqueue.Queue
    else:
        new_queue = mp.Queue

    controller_queue_name_manager = mp.Manager()
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


def mock_event_container(use_thread=True):
    """Build out an event container with multiprocessing or threaded events."""
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
    """
    datetime.datetime is a weird function and requires another function to be able to
    mock it consistently.
    """
    return "Datetime"


def clear_verbose_queue(queue, task_name, verbose_array):
    """Function to clear verbose queue in seperate process for testing purposes."""
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
        while not queue.empty():
            output_value = queue.get(task_name)
            verbose_array[idx] = output_value[1]
            idx += 1


def clear_log_queue(queue, log_string):
    """Get string from queue and store it to the log_string bstring."""
    while not queue.empty():
        output_string = queue.get()
        output_string = output_string.encode("utf-8")
        log_string.value = log_string.value + output_string


def keyboard_interrupt():
    signal.raise_signal(signal.SIGINT)


# region Hardware
IMPLEMENTED_HARDWARE = [
    hardware for hardware in HardwareType if hardware not in UNIMPLEMENTED_HARDWARE
]


def mock_channel_list():
    """Builds a default channel list with one physical and one feedback device"""
    response_channel = Channel()
    for attr in response_channel.channel_attr_list:
        setattr(response_channel, attr, attr)
    response_channel.feedback_channel = None
    response_channel.feedback_device = None

    excitation_channel = Channel()
    for attr in excitation_channel.channel_attr_list:
        setattr(excitation_channel, attr, attr)

    return [response_channel, excitation_channel]


def skeleton_hardware_metadata(**overrides):
    """Builds a SkeletonHardwareMetadata with sensible test defaults."""
    kwargs = dict(
        channel_list=mock_channel_list(),
        sample_rate=1024,
        time_per_read=0.25,
        time_per_write=0.125,
        output_oversample=1,
    )
    kwargs.update(overrides)
    return SkeletonHardwareMetadata(**kwargs)


# endregion


# region Environment
IMPLEMENTED_ENVIRONMENT = [
    environment
    for environment in EnvironmentType
    if environment not in UNIMPLEMENTED_ENVIRONMENT
]


def mock_channel_list_bools():
    return [True, True]


def skeleton_environment_metadata(**overrides):
    """Builds a SkeletonMetadata with test defaults."""
    kwargs = dict(
        environment_name="Skeleton Environment",
        channel_list_bools=mock_channel_list_bools(),
        sample_rate=1024,
        example_window_size="Example Window Size",
    )
    kwargs.update(overrides)
    return SkeletonMetadata(**kwargs)


def skeleton_environment_instructions(**overrides):
    """Builds a SkeletonInstructions with test defaults."""
    kwargs = dict(environment_name="Skeleton Environment", example_test_level=1.0)
    kwargs.update(overrides)
    return SkeletonInstructions(**kwargs)


def skeleton_queues(**overrides):
    """Builds a SkeletonQueues with test defaults."""
    log_file_queue = overrides.get("log_file_queue", mp.Queue())
    kwargs = dict(
        environment_command_queue=VerboseMessageQueue(
            log_file_queue, mp.Queue(), "Command Queue"
        ),
        gui_update_queue=mp.Queue(),
        controller_communication_queue=VerboseMessageQueue(
            log_file_queue, mp.Queue(), "Controller Queue"
        ),
        data_in_queue=mp.Queue(),
        data_out_queue=mp.Queue(),
        log_file_queue=log_file_queue,
    )
    kwargs.update(overrides)
    return SkeletonQueues(**kwargs)


def skeleton_environment(**overrides):
    """Builds a SkeletonEnvironment with test defaults."""
    kwargs = dict(
        environment_name="Environment Name",
        queue_name="Queue Name",
        queue_container=skeleton_queues(),
        acquisition_active_event=mp.Event(),
        output_active_event=mp.Event(),
        active_event=mp.Event(),
        ready_event=mp.Event(),
    )
    kwargs.update(overrides)
    return SkeletonEnvironment(**kwargs)


# endregion


# region SysId Environment
def skeleton_sysid_environment_metadata(**overrides):
    """Builds a SkeletonMetadata with test defaults."""
    kwargs = dict(
        environment_name="Skeleton SysId Environment",
        channel_list_bools=mock_channel_list_bools(),
        sample_rate=1024,
    )
    kwargs.update(overrides)
    return SkeletonSysIdMetadata(**kwargs)


def skeleton_sysid_queues(**overrides):
    """Builds a SkeletonQueues (sysid environment) with fresh default queues."""
    log_file_queue = overrides.get("log_file_queue", mp.Queue())
    kwargs = dict(
        environment_name="Skeleton SysId Environment",
        environment_command_queue=VerboseMessageQueue(
            log_file_queue, mp.Queue(), "Environment Command Queue"
        ),
        gui_update_queue=mp.Queue(),
        controller_communication_queue=VerboseMessageQueue(
            log_file_queue, mp.Queue(), "Controller Command Queue"
        ),
        data_in_queue=mp.Queue(),
        data_out_queue=mp.Queue(),
        log_file_queue=log_file_queue,
    )
    kwargs.update(overrides)
    return SkeletonSysIdQueues(**kwargs)


def skeleton_sysid_environment(**overrides):
    """Builds a SkeletonEnvironment (sysid environment) with fresh test defaults."""
    kwargs = dict(
        environment_name="Skeleton SysId Environment",
        queue_name="Skeleton SysId Queue",
        queue_container=skeleton_sysid_queues(),
        acquisition_active_event=mp.Event(),
        output_active_event=mp.Event(),
        active_event=mp.Event(),
        ready_event=mp.Event(),
        sysid_active_event=mp.Event(),
        sysid_stored_event=mp.Event(),
    )
    kwargs.update(overrides)
    return SkeletonSysIdEnvironment(**kwargs)


# endregion

# region System Id Helpers
"""Helpers for testing the system-identification environments.

These helpers build small, deterministic objects for testing the sys-id
environments (random vibration, transient, sine) without spawning any of
their sub-worker processes.
"""

COORD_DTYPE = np.dtype([("node", "<u8"), ("direction", "i1")])


def numeric_channel_list(num_response=1, num_drive=1):
    """Channel list with numeric node numbers for coordinate-based operations.

    Response channels are nodes 1..num_response, drive channels are nodes
    101..100+num_drive, all in the X+ direction.  A channel is a drive if its
    feedback_device is set.
    """
    channels = []
    for index in range(num_response):
        channels.append(
            Channel(
                node_number=index + 1,
                node_direction="X+",
                channel_type="Acceleration",
                physical_device="Virtual",
                physical_channel=str(index + 1),
            )
        )
    for index in range(num_drive):
        channels.append(
            Channel(
                node_number=index + 101,
                node_direction="X+",
                channel_type="Force",
                physical_device="Virtual",
                physical_channel=str(num_response + index + 1),
                feedback_device="Virtual",
                feedback_channel=str(index + 1),
            )
        )
    return channels


def numeric_hardware_metadata(num_response=1, num_drive=1):
    """SkeletonHardwareMetadata with a numeric channel list."""
    return skeleton_hardware_metadata(
        channel_list=numeric_channel_list(num_response, num_drive)
    )


def sysid_measurement_metadata(sample_rate=1000, frame_size=200):
    """Small SysIdMetadata (101 fft lines, 5 Hz spacing at the defaults)."""
    return SysIdMetadata(
        sample_rate=sample_rate,
        sysid_frame_size=frame_size,
        sysid_averaging_type="Linear",
        sysid_noise_averages=5,
        sysid_averages=10,
        sysid_exponential_averaging_coefficient=0.01,
        sysid_estimator="H1",
        sysid_level=0.01,
        sysid_level_ramp_time=0.5,
        sysid_signal_type="Random",
        sysid_window="Hann",
        sysid_overlap=0.5,
        sysid_burst_on=0.5,
        sysid_pretrigger=0.05,
        sysid_burst_ramp_fraction=0.05,
        sysid_low_frequency_cutoff=0,
        sysid_high_frequency_cutoff=int(sample_rate / 2),
        stream_file=None,
        auto_shutdown=False,
    )


def sysid_measurement_data_package(sysid_metadata, num_response=1, num_reference=1):
    """Deterministic SysIdDataPackage sized to the given SysIdMetadata."""
    fft_lines = sysid_metadata.sysid_fft_lines
    frequencies = sysid_metadata.sysid_frequency_spacing * np.arange(fft_lines)
    frf = np.ones((fft_lines, num_response, num_reference), dtype="complex128") * (
        1 - 0.5j
    )
    coherence = np.ones((fft_lines, num_response))
    response_cpsd = np.ones((fft_lines, num_response, num_response), dtype="complex128")
    reference_cpsd = np.ones(
        (fft_lines, num_reference, num_reference), dtype="complex128"
    )
    return SysIdDataPackage(
        sysid_frames=sysid_metadata.sysid_averages,
        frequencies=frequencies,
        sysid_frf=frf,
        sysid_coherence=coherence,
        sysid_response_cpsd=response_cpsd,
        sysid_reference_cpsd=reference_cpsd,
        sysid_condition=np.ones((fft_lines,)),
        sysid_response_noise=np.zeros_like(response_cpsd),
        sysid_reference_noise=np.zeros_like(reference_cpsd),
    )


def outer_coordinate(nodes, directions):
    """Outer-product coordinate array as written by sdynpy specifications."""
    num_channels = len(nodes)
    coordinate = np.empty((num_channels, num_channels, 2), dtype=COORD_DTYPE)
    for i in range(num_channels):
        for j in range(num_channels):
            coordinate[i, j, 0] = (nodes[i], directions[i])
            coordinate[i, j, 1] = (nodes[j], directions[j])
    return coordinate


def write_cpsd_spec_npz(
    path,
    frequencies,
    num_channels,
    warning=True,
    abort=True,
    coordinate=None,
):
    """Writes a random-vibration CPSD specification .npz file.

    The CPSD diagonal for channel ``i`` is ``i + 1`` at every frequency line.
    Warning levels are cpsd*2 (upper) and cpsd/2 (lower); abort levels are
    cpsd*4 and cpsd/4.  Returns the dictionary that was saved.
    """
    frequencies = np.asarray(frequencies, dtype="float64")
    num_lines = frequencies.size
    cpsd = np.zeros((num_lines, num_channels, num_channels), dtype="complex128")
    for i in range(num_channels):
        cpsd[:, i, i] = i + 1
    diagonal = np.real(np.einsum("fii->fi", cpsd))
    data = {"f": frequencies, "cpsd": cpsd}
    if warning:
        data["warning_upper"] = diagonal * 2
        data["warning_lower"] = diagonal / 2
    if abort:
        data["abort_upper"] = diagonal * 4
        data["abort_lower"] = diagonal / 4
    if coordinate is not None:
        data["coordinate"] = coordinate
    np.savez(path, **data)
    return data


def write_sine_spec_npz(
    path,
    name="Tone 1",
    num_channels=1,
    frequencies=(10.0, 20.0),
    amplitude=1.0,
    sweep_rate=10.0,
):
    """Writes a sine specification .npz file (a single linear sweep).

    Phases are in degrees per the sine specification file format.  Returns
    the dictionary that was saved.
    """
    frequencies = np.asarray(frequencies, dtype="float64")
    num_breakpoints = frequencies.size
    data = {
        "frequency": frequencies,
        "amplitude": np.full((num_breakpoints, num_channels), amplitude),
        "phase": np.zeros((num_breakpoints, num_channels)),
        "sweep_type": np.zeros(num_breakpoints - 1, dtype="u1"),
        "sweep_rate": np.full(num_breakpoints - 1, sweep_rate),
        "start_time": 0.0,
        "name": name,
    }
    np.savez(path, **data)
    return data


TRANSIENT_CONTROL_SCRIPT = '''
import numpy as np


def transient_control(
    sample_rate,
    control_signal,
    frequency_spacing,
    frf,
    response_noise,
    reference_noise,
    response_cpsd,
    reference_cpsd,
    coherence,
    frames,
    total_frames,
    output_oversample,
    extra_parameters,
    last_drive,
    last_response,
):
    """Deterministic type-0 transient control law: half the specification."""
    return np.repeat(control_signal * 0.5, output_oversample, axis=-1)
'''


def write_transient_control_script(path):
    """Writes a type-0 transient control law script, returning its function name.

    The control law deterministically returns ``control_signal * 0.5``
    upsampled by the output oversample factor.
    """
    with open(path, "w", encoding="utf-8") as script_file:
        script_file.write(TRANSIENT_CONTROL_SCRIPT)
    return "transient_control"


def get_queue_messages(queue, task_name, count, timeout=10):
    """Gets exactly ``count`` (message, data) tuples from a VerboseMessageQueue.

    Uses a timeout on every get so a missing message fails the test quickly
    instead of hanging it.
    """
    return [queue.get(task_name, timeout=timeout) for _ in range(count)]


def drain_queue_commands(queue, task_name, timeout=0.5):
    """Drains a VerboseMessageQueue, returning the list of commands received.

    Blocks at most ``timeout`` seconds after the last message, so it is safe
    against multiprocessing feeder-thread latency without risking a hang.
    """
    commands = []
    while True:
        try:
            message, _ = queue.get(task_name, timeout=timeout)
        except thqueue.Empty:
            return commands
        commands.append(message)


# endregion
