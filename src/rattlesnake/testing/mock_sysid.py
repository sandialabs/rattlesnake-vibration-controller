"""Test helpers for the system-identification environments.

These helpers build small, deterministic objects for testing the sys-id
environments (random vibration, transient, sine) without spawning any of
their sub-worker processes.
"""

import queue as thqueue

import numpy as np

from rattlesnake.hardware.hardware_utilities import Channel
from rattlesnake.process.abstract_sysid_data_analysis import (
    SysIdDataPackage,
    SysIdMetadata,
)
from rattlesnake.testing.mock_hardware import MockHardwareMetadata

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
    """MockHardwareMetadata with a numeric channel list."""
    hardware_metadata = MockHardwareMetadata()
    hardware_metadata.channel_list = numeric_channel_list(num_response, num_drive)
    return hardware_metadata


def mock_sysid_metadata(sample_rate=1000, frame_size=200):
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


def mock_sysid_data_package(sysid_metadata, num_response=1, num_reference=1):
    """Deterministic SysIdDataPackage sized to the given SysIdMetadata."""
    fft_lines = sysid_metadata.sysid_fft_lines
    frequencies = sysid_metadata.sysid_frequency_spacing * np.arange(fft_lines)
    frf = np.ones((fft_lines, num_response, num_reference), dtype="complex128") * (
        1 - 0.5j
    )
    coherence = np.ones((fft_lines, num_response))
    response_cpsd = np.ones(
        (fft_lines, num_response, num_response), dtype="complex128"
    )
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
