# -*- coding: utf-8 -*-
"""
Interactive debug viewer for Rattlesnake acquisition/output/environment queue traces.

This script assembles the saved debug files into continuous time histories and
plots:

1. Environment data_out queue payloads
2. Hardware output writes
3. Hardware acquisition reads
4. Environment data_in queue payloads

This version is intentionally NOT command-line driven. Edit the user variables
near the top and rerun in Spyder / an IDE.

This version allows input-channel and output-channel selection independently.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# =============================================================================
# USER SETTINGS
# =============================================================================

DEBUG_DIRECTORY = r"../debug_data"
ENVIRONMENT_NAME = "Environment_0"

# Optional sample rates for converting sample index to time axis.
# If set to None, sample indices are used on the x-axis.
OUTPUT_SAMPLE_RATE = None
ACQUISITION_SAMPLE_RATE = None
ENV_QUEUE_SAMPLE_RATE = None

# Independent channel selections
DEFAULT_OUTPUT_CHANNEL_INDEX = 0
DEFAULT_INPUT_CHANNEL_INDEX = 0

# If True, plot all channels faintly with selected channel highlighted.
PLOT_ALL_CHANNELS = False

FIND_ENV_IN_WITHIN_ACQUISITION = True

# Which environment data_in block to use for matching.
# Usually 0 is fine to start; you can change it and rerun.
ENV_IN_BLOCK_INDEX_TO_MATCH = 0

# If True, use correlation to match. If False, use exact/near-exact search.
USE_CORRELATION_MATCH = True

# If True, highlight the portions of time when the environments are active in
# the acquisition.
SHOW_ENVIRONMENT_STATE_INTERVALS = True

# If True, print summary information to console.
VERBOSE = True


# =============================================================================
# HELPERS
# =============================================================================

def normalize_environment_name(name):
    """
    Normalize environment names so that variants like 'Environment 0'
    and 'Environment_0' compare equal.
    """
    return str(name).strip().replace(" ", "_")

def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def load_npz_sequence(files):
    records = []
    for file in sorted(files, key=natural_sort_key):
        with np.load(file, allow_pickle=True) as data:
            record = {key: data[key] for key in data.files}
            record["_file"] = str(file)
            records.append(record)
    return records


def concat_field(records, field):
    arrays = [rec[field] for rec in records if field in rec]
    if not arrays:
        return None
    return np.concatenate(arrays, axis=-1)


def make_time_axis(n_samples, sample_rate=None):
    if sample_rate is None:
        return np.arange(n_samples), "Sample"
    return np.arange(n_samples) / sample_rate, "Time (s)"


def print_summary(name, data):
    if data is None:
        print(f"{name}: no data")
        return
    print(
        f"{name}: shape={data.shape}, "
        f"rms={np.sqrt(np.mean(data**2, axis=-1))}, "
        f"peak={np.max(np.abs(data), axis=-1)}"
    )


def find_files(debug_dir: Path, environment_name: str):
    safe_env = environment_name.replace(" ", "_")

    output_debug_files = list(debug_dir.glob("output_debug_*.npz"))
    acquisition_debug_files = list(debug_dir.glob("acquisition_debug_*.npz"))
    env_out_files = list(debug_dir.glob(f"{safe_env}_data_out_*.npz"))
    env_in_files = list(debug_dir.glob(f"{safe_env}_data_in_*.npz"))

    return output_debug_files, acquisition_debug_files, env_out_files, env_in_files


def records_last_flags(records, field_name):
    flags = []
    for rec in records:
        if field_name in rec:
            value = rec[field_name]
            try:
                flags.append(bool(np.atleast_1d(value)[0]))
            except Exception:
                flags.append(bool(value))
        else:
            flags.append(False)
    return np.array(flags, dtype=bool)


def block_lengths(records, field_name):
    lengths = []
    for rec in records:
        if field_name in rec:
            arr = rec[field_name]
            lengths.append(arr.shape[-1] if arr.ndim >= 1 else 0)
        else:
            lengths.append(0)
    return np.array(lengths, dtype=int)


def cumulative_block_edges(lengths):
    return np.concatenate(([0], np.cumsum(lengths)))


def shaded_intervals_from_lengths(lengths):
    edges = cumulative_block_edges(lengths)
    intervals = []
    for i, length in enumerate(lengths):
        if length > 0:
            intervals.append((edges[i], edges[i + 1]))
    return intervals

def find_env_in_within_acquisition(acquisition_data, env_in_records, block_index, channel_index):
    """
    Find a chosen environment data_in block inside the acquisition trace.

    Returns
    -------
    (start, end, score, method) or (None, None, None, None)
    """
    block = get_environment_in_block(env_in_records, block_index, channel_index)
    if block is None:
        return None, None, None, None

    reference = acquisition_data[channel_index]
    if block.size > reference.size:
        return None, None, None, None

    if USE_CORRELATION_MATCH:
        start, score = best_subsequence_correlation(reference, block)
        method = "corr"
    else:
        start, score = best_subsequence_match(reference, block)
        method = "mse"

    if start is None:
        return None, None, None, None

    end = start + block.size
    return start, end, score, method

def plot_selected_channel(
    ax,
    data,
    title,
    selected_channel,
    sample_rate=None,
    all_channels=True,
    intervals=None,
    interval_color="lightgray",
    highlight_interval=None,
    highlight_label=None,
    state_intervals=None,
):
    ax.clear()

    if data is None:
        ax.set_title(f"{title} (no data)")
        ax.grid(True, alpha=0.25)
        return

    n_channels, n_samples = data.shape
    selected_channel = min(selected_channel, n_channels - 1)

    x, xlabel = make_time_axis(n_samples, sample_rate)

    # Generic intervals
    if intervals is not None:
        for start, end in intervals:
            if sample_rate is None:
                ax.axvspan(start, end, color=interval_color, alpha=0.10)
            else:
                ax.axvspan(start / sample_rate, end / sample_rate, color=interval_color, alpha=0.10)

    # Environment state intervals
    if state_intervals is not None:
        colors = {
            "pending": "orange",
            "active": "green",
            "last": "red",
        }
        labels_drawn = set()
        for state_name, state_blocks in state_intervals.items():
            for start, end in state_blocks:
                label = None
                if state_name not in labels_drawn:
                    label = f"{state_name} interval"
                    labels_drawn.add(state_name)
                if sample_rate is None:
                    ax.axvspan(start, end, color=colors.get(state_name, "gray"), alpha=0.15, label=label)
                else:
                    ax.axvspan(
                        start / sample_rate,
                        end / sample_rate,
                        color=colors.get(state_name, "gray"),
                        alpha=0.15,
                        label=label,
                    )

    # Matched environment-data extent
    if highlight_interval is not None:
        start, end = highlight_interval
        if sample_rate is None:
            ax.axvspan(start, end, color="magenta", alpha=0.20, label=highlight_label)
        else:
            ax.axvspan(
                start / sample_rate,
                end / sample_rate,
                color="magenta",
                alpha=0.20,
                label=highlight_label,
            )

    if all_channels:
        for i in range(n_channels):
            color = "C0"
            alpha = 0.25
            lw = 0.8
            if i == selected_channel:
                color = "C3"
                alpha = 1.0
                lw = 1.5
            ax.plot(x, data[i], color=color, alpha=alpha, linewidth=lw)
    else:
        ax.plot(x, data[selected_channel], color="C3", linewidth=1.5)

    y = data[selected_channel]
    finite = np.isfinite(y)
    if np.any(finite):
        y_valid = y[finite]
        y_min = np.min(y_valid)
        y_max = np.max(y_valid)

        if y_min == y_max:
            margin = 1.0 if y_min == 0 else abs(y_min) * 0.1
        else:
            margin = 0.05 * (y_max - y_min)

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_title(f"{title} (channel {selected_channel})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)

    if (
        highlight_interval is not None and highlight_label is not None
    ) or (state_intervals is not None and any(state_intervals.values())):
        ax.legend(loc="upper right")

def best_subsequence_match(reference, query):
    """
    Find the location of `query` inside `reference` using least-squares matching.

    Parameters
    ----------
    reference : 1D np.ndarray
        Long signal.
    query : 1D np.ndarray
        Shorter signal to find inside reference.

    Returns
    -------
    best_start : int
        Starting sample index of best match.
    best_error : float
        Mean squared error of the best match.
    """
    reference = np.asarray(reference).ravel()
    query = np.asarray(query).ravel()

    if query.size > reference.size:
        return None, np.inf

    best_start = None
    best_error = np.inf

    qnorm = np.mean(query**2) + 1e-30

    for start in range(reference.size - query.size + 1):
        segment = reference[start : start + query.size]
        err = np.mean((segment - query) ** 2) / qnorm
        if err < best_error:
            best_error = err
            best_start = start

    return best_start, best_error


def best_subsequence_correlation(reference, query):
    """
    Find the location of `query` inside `reference` using normalized correlation.

    Parameters
    ----------
    reference : 1D np.ndarray
        Long signal.
    query : 1D np.ndarray
        Shorter signal to find inside reference.

    Returns
    -------
    best_start : int
        Starting sample index of best match.
    best_corr : float
        Correlation coefficient-like score at the best match.
    """
    reference = np.asarray(reference).ravel()
    query = np.asarray(query).ravel()

    if query.size > reference.size:
        return None, -np.inf

    q = query - np.mean(query)
    qnorm = np.linalg.norm(q)
    if qnorm == 0:
        return None, -np.inf

    best_start = None
    best_corr = -np.inf

    for start in range(reference.size - query.size + 1):
        segment = reference[start : start + query.size]
        s = segment - np.mean(segment)
        snorm = np.linalg.norm(s)
        if snorm == 0:
            corr = -np.inf
        else:
            corr = np.dot(s, q) / (snorm * qnorm)
        if corr > best_corr:
            best_corr = corr
            best_start = start

    return best_start, best_corr

def get_environment_in_block(records, block_index, channel_index):
    """
    Extract one saved environment data_in block for a given channel.

    Parameters
    ----------
    records : list[dict]
        Loaded environment data_in records.
    block_index : int
        Which record to use.
    channel_index : int
        Which channel from that record.

    Returns
    -------
    block : np.ndarray | None
        1D signal block.
    """
    if not records:
        return None
    if block_index < 0 or block_index >= len(records):
        return None

    rec = records[block_index]
    if "environment_data" not in rec:
        return None

    arr = rec["environment_data"]
    if arr.ndim != 2:
        return None
    if channel_index < 0 or channel_index >= arr.shape[0]:
        return None

    return arr[channel_index]

def parse_name_list_field(arr):
    """
    Parse an object array of strings into a normalized Python set.
    """
    if arr is None:
        return set()
    result = set()
    for val in np.atleast_1d(arr):
        try:
            text = str(val)
        except Exception:
            continue
        if text.strip() != "":
            result.add(normalize_environment_name(text))
    return result


def parse_flag_pairs(arr):
    """
    Parse arrays of strings like 'Environment 0:1' into a normalized dict.
    """
    result = {}
    if arr is None:
        return result
    for val in np.atleast_1d(arr):
        text = str(val)
        if ":" not in text:
            continue
        key, raw = text.split(":", 1)
        key = normalize_environment_name(key)
        try:
            result[key] = bool(int(float(raw)))
        except Exception:
            result[key] = False
    return result

def compute_environment_state_intervals_from_acquisition_records(records, environment_name):
    """
    Build intervals in concatenated acquisition sample space for a given environment.

    Returns a dict with keys:
        - "pending"
        - "active"
        - "last"
    Each value is a list of (start_sample, end_sample) tuples.
    """
    environment_name = normalize_environment_name(environment_name)

    lengths = block_lengths(records, "read_data")
    edges = cumulative_block_edges(lengths)

    state_blocks = {
        "pending": [],
        "active": [],
        "last": [],
    }

    for i, rec in enumerate(records):
        start = edges[i]
        end = edges[i + 1]

        active_envs = parse_name_list_field(rec.get("active_environments", None))
        pending_envs = parse_name_list_field(rec.get("first_data_pending", None))
        last_flags = parse_flag_pairs(rec.get("last_data_flags", None))

        if environment_name in pending_envs:
            state_blocks["pending"].append((start, end))
        if environment_name in active_envs:
            state_blocks["active"].append((start, end))
        if last_flags.get(environment_name, False):
            state_blocks["last"].append((start, end))

    return state_blocks

def merge_intervals(intervals):
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged

# =============================================================================
# LOAD DATA
# =============================================================================

debug_dir = Path(DEBUG_DIRECTORY)

output_debug_files, acquisition_debug_files, env_out_files, env_in_files = find_files(
    debug_dir, ENVIRONMENT_NAME
)

output_records = load_npz_sequence(output_debug_files)
acquisition_records = load_npz_sequence(acquisition_debug_files)
env_out_records = load_npz_sequence(env_out_files)
env_in_records = load_npz_sequence(env_in_files)

output_data = concat_field(output_records, "write_data")
acquisition_data = concat_field(acquisition_records, "read_data")
env_out_data = concat_field(env_out_records, "environment_data")
env_in_data = concat_field(env_in_records, "environment_data")

env_out_last_flags = records_last_flags(env_out_records, "last_run")
env_in_last_flags = records_last_flags(env_in_records, "environment_finished")

output_block_lengths = block_lengths(output_records, "write_data")
acquisition_block_lengths = block_lengths(acquisition_records, "read_data")
env_out_block_lengths = block_lengths(env_out_records, "environment_data")
env_in_block_lengths = block_lengths(env_in_records, "environment_data")

output_intervals = shaded_intervals_from_lengths(output_block_lengths)
acquisition_intervals = shaded_intervals_from_lengths(acquisition_block_lengths)
env_out_intervals = shaded_intervals_from_lengths(env_out_block_lengths)
env_in_intervals = shaded_intervals_from_lengths(env_in_block_lengths)

environment_state_intervals = compute_environment_state_intervals_from_acquisition_records(
    acquisition_records,
    ENVIRONMENT_NAME,
)
for key in environment_state_intervals:
    environment_state_intervals[key] = merge_intervals(environment_state_intervals[key])

if VERBOSE:
    print(f"DEBUG_DIRECTORY = {debug_dir.resolve()}")
    print(f"ENVIRONMENT_NAME = {ENVIRONMENT_NAME}")
    print(f"output_debug_files: {len(output_debug_files)}")
    print(f"acquisition_debug_files: {len(acquisition_debug_files)}")
    print(f"{ENVIRONMENT_NAME}_data_out_files: {len(env_out_files)}")
    print(f"{ENVIRONMENT_NAME}_data_in_files: {len(env_in_files)}")
    print_summary("output_data", output_data)
    print_summary("acquisition_data", acquisition_data)
    print_summary("env_out_data", env_out_data)
    print_summary("env_in_data", env_in_data)
    if len(env_out_last_flags) > 0:
        print(f"env_out last_run flags: {env_out_last_flags}")
    if len(env_in_last_flags) > 0:
        print(f"env_in environment_finished flags: {env_in_last_flags}")
    print(f"environment state intervals for {ENVIRONMENT_NAME}:")
    for key, intervals in environment_state_intervals.items():
        print(f"  {key}: {intervals}")

matched_env_in_block = None
matched_env_in_start = None
matched_env_in_score = None

if FIND_ENV_IN_WITHIN_ACQUISITION and acquisition_data is not None and env_in_records:
    matched_env_in_block = get_environment_in_block(
        env_in_records,
        ENV_IN_BLOCK_INDEX_TO_MATCH,
        min(DEFAULT_INPUT_CHANNEL_INDEX, env_in_records[ENV_IN_BLOCK_INDEX_TO_MATCH]["environment_data"].shape[0] - 1)
        if "environment_data" in env_in_records[ENV_IN_BLOCK_INDEX_TO_MATCH]
        else 0,
    )

def find_full_env_in_extent_within_acquisition(
    acquisition_data,
    env_in_records,
    full_env_in_data,
    block_index,
    channel_index,
):
    """
    Find the start of one chosen env_in block inside acquisition_data, then
    extend that match to the full concatenated env_in duration.

    Returns
    -------
    (start, end, score, method) or (None, None, None, None)
    """
    block = get_environment_in_block(env_in_records, block_index, channel_index)
    if block is None:
        return None, None, None, None

    reference = acquisition_data[channel_index]
    if block.size > reference.size:
        return None, None, None, None

    if USE_CORRELATION_MATCH:
        start, score = best_subsequence_correlation(reference, block)
        method = "corr"
    else:
        start, score = best_subsequence_match(reference, block)
        method = "mse"

    if start is None:
        return None, None, None, None

    if full_env_in_data is None:
        end = start + block.size
    else:
        end = start + full_env_in_data.shape[-1]

    return start, end, score, method


# =============================================================================
# PLOT
# =============================================================================

output_channel_counts = [
    arr.shape[0]
    for arr in [env_out_data, output_data]
    if arr is not None
]
input_channel_counts = [
    arr.shape[0]
    for arr in [acquisition_data, env_in_data]
    if arr is not None
]

max_output_channels = max(output_channel_counts) if output_channel_counts else 1
max_input_channels = max(input_channel_counts) if input_channel_counts else 1

selected_output_channel = min(DEFAULT_OUTPUT_CHANNEL_INDEX, max_output_channels - 1)
selected_input_channel = min(DEFAULT_INPUT_CHANNEL_INDEX, max_input_channels - 1)

fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
fig.suptitle(f"Rattlesnake Debug Trace Viewer: {ENVIRONMENT_NAME}", fontsize=14)

plot_selected_channel(
    axs[0],
    env_out_data,
    f"{ENVIRONMENT_NAME} data_out queue payloads",
    selected_channel=selected_output_channel,
    sample_rate=ENV_QUEUE_SAMPLE_RATE,
    all_channels=PLOT_ALL_CHANNELS,
    intervals=env_out_intervals,
    interval_color="lightblue",
)

plot_selected_channel(
    axs[1],
    output_data,
    "Hardware output writes",
    selected_channel=selected_output_channel,
    sample_rate=OUTPUT_SAMPLE_RATE,
    all_channels=PLOT_ALL_CHANNELS,
    intervals=output_intervals,
    interval_color="lightgreen",
)

plot_selected_channel(
    axs[2],
    acquisition_data,
    "Hardware acquisition reads",
    selected_channel=selected_input_channel,
    sample_rate=ACQUISITION_SAMPLE_RATE,
    all_channels=PLOT_ALL_CHANNELS,
    intervals=acquisition_intervals,
    interval_color="lightyellow",
)

plot_selected_channel(
    axs[3],
    env_in_data,
    f"{ENVIRONMENT_NAME} data_in queue payloads",
    selected_channel=selected_input_channel,
    sample_rate=ENV_QUEUE_SAMPLE_RATE,
    all_channels=PLOT_ALL_CHANNELS,
    intervals=env_in_intervals,
    interval_color="lightcoral",
)

plt.tight_layout(rect=[0, 0.1, 1, 0.96])

# Output channel slider
output_slider_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])
output_slider = Slider(
    output_slider_ax,
    "Output Ch",
    0,
    max_output_channels - 1,
    valinit=selected_output_channel,
    valstep=1,
)

# Input channel slider
input_slider_ax = fig.add_axes([0.15, 0.00, 0.7, 0.02])
input_slider = Slider(
    input_slider_ax,
    "Input Ch",
    0,
    max_input_channels - 1,
    valinit=selected_input_channel,
    valstep=1,
)


def update(_):
    out_ch = int(output_slider.val)
    in_ch = int(input_slider.val)

    plot_selected_channel(
        axs[0],
        env_out_data,
        f"{ENVIRONMENT_NAME} data_out queue payloads",
        selected_channel=min(out_ch, env_out_data.shape[0] - 1) if env_out_data is not None else 0,
        sample_rate=ENV_QUEUE_SAMPLE_RATE,
        all_channels=PLOT_ALL_CHANNELS,
        intervals=env_out_intervals,
        interval_color="lightblue",
    )

    plot_selected_channel(
        axs[1],
        output_data,
        "Hardware output writes",
        selected_channel=min(out_ch, output_data.shape[0] - 1) if output_data is not None else 0,
        sample_rate=OUTPUT_SAMPLE_RATE,
        all_channels=PLOT_ALL_CHANNELS,
        intervals=output_intervals,
        interval_color="lightgreen",
    )

    highlight_interval = None
    highlight_label = None
    if FIND_ENV_IN_WITHIN_ACQUISITION and acquisition_data is not None and env_in_records:
        start, end, score, method = find_full_env_in_extent_within_acquisition(
            acquisition_data,
            env_in_records,
            env_in_data,
            ENV_IN_BLOCK_INDEX_TO_MATCH,
            min(in_ch, acquisition_data.shape[0] - 1),
        )
        if VERBOSE and start is not None:
            print(
                f"[match] input channel {in_ch}, env_in block {ENV_IN_BLOCK_INDEX_TO_MATCH}: "
                f"start={start}, end={end}, method={method}, score={score}"
            )
        if start is not None:
            highlight_interval = (start, end)
            if method == "corr":
                highlight_label = f"matched env_in block ({method}={score:0.3f})"
            else:
                highlight_label = f"matched env_in block ({method}={score:0.3e})"

    plot_selected_channel(
        axs[2],
        acquisition_data,
        "Hardware acquisition reads",
        selected_channel=min(in_ch, acquisition_data.shape[0] - 1) if acquisition_data is not None else 0,
        sample_rate=ACQUISITION_SAMPLE_RATE,
        all_channels=PLOT_ALL_CHANNELS,
        intervals=acquisition_intervals,
        interval_color="lightyellow",
        highlight_interval=highlight_interval,
        highlight_label=highlight_label,
        state_intervals=environment_state_intervals if SHOW_ENVIRONMENT_STATE_INTERVALS else None,
    )

    plot_selected_channel(
        axs[3],
        env_in_data,
        f"{ENVIRONMENT_NAME} data_in queue payloads",
        selected_channel=min(in_ch, env_in_data.shape[0] - 1) if env_in_data is not None else 0,
        sample_rate=ENV_QUEUE_SAMPLE_RATE,
        all_channels=PLOT_ALL_CHANNELS,
        intervals=env_in_intervals,
        interval_color="lightcoral",
    )

    fig.canvas.draw_idle()


output_slider.on_changed(update)
input_slider.on_changed(update)

plt.show()