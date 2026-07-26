# -*- coding: utf-8 -*-
"""
Lightweight per-process timing instrumentation for the Rattlesnake controller.

Every process built on ``AbstractMessageProcess`` or ``AbstractEnvironment``
(Controller, Acquisition, Output, Streaming, each environment's
``DataCollectorProcess``, and every environment's own control-law process)
already runs the same loop: pull one ``(message, data)`` command off a
queue, dispatch it, repeat.  ``BenchmarkRecorder`` hooks that single spot in
each loop (see ``AbstractMessageProcess.run`` and ``AbstractEnvironment.run``)
to record, per iteration, how long the process spent waiting for its next
command versus how long it spent executing it.  Since ``RUN_HARDWARE``,
``ACQUIRE``, etc. re-queue themselves every frame, this reconstructs a
per-frame timing history for every stage of the
acquisition -> environment -> output pipeline with no per-process-specific
instrumentation required.

Collection is off by default -- set the ``RATTLESNAKE_BENCHMARK``
environment variable before launching Rattlesnake to enable it, so normal
runs and the test suite never touch the filesystem.  Samples are appended
incrementally to ``benchmark_data/<process>_<pid>.csv`` (rather than held in
memory for the whole run) so a run that is killed instead of shut down
gracefully still leaves most of its data on disk.

``record``/``timer`` are intentionally the only things call sites use.  How
samples are stored is an internal detail of ``BenchmarkRecorder``, so a
live/real-time consumer can be swapped in later (e.g. pushing samples to a
queue for a live plot, or for adaptive buffer tuning) without touching any
of the instrumented processes.
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_BENCHMARK_DIRECTORY = "benchmark_data"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "", "false", "no", "off")


# Read once at import time.  Tests and normal runs never set this, so
# benchmarking stays a no-op unless a user explicitly opts in.
# BENCHMARK_ENABLED = _env_flag("RATTLESNAKE_BENCHMARK")
BENCHMARK_ENABLED = True


def _safe_filename_part(name) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))


class BenchmarkRecorder:
    """Records wait/compute duration samples for one process.

    Parameters
    ----------
    process_name : str
        Name used to label samples and to build the output filename.
    directory : str or Path
        Directory that the recorder's CSV file will be created in.
    enabled : bool, optional
        Overrides the module-level ``BENCHMARK_ENABLED`` flag for this
        recorder specifically. Defaults to following the global flag.
    flush_every : int
        Number of samples to buffer in memory before appending them to disk.
    """

    def __init__(
        self,
        process_name: str,
        *,
        directory: "str | Path" = DEFAULT_BENCHMARK_DIRECTORY,
        enabled: Optional[bool] = None,
        flush_every: int = 50,
    ):
        self.process_name = process_name
        self.directory = Path(directory)
        self.enabled = BENCHMARK_ENABLED if enabled is None else enabled
        self.flush_every = max(1, flush_every)
        self._buffer: List[Tuple[float, str, float, float]] = []
        self._file = None
        self._writer = None

    def record(self, stage, *, wait: float = 0.0, duration: float = 0.0) -> None:
        """Record one (wait_time, compute_time) sample for ``stage``.

        Never raises: instrumentation must not be able to take down the
        real-time loop it is measuring, so any failure just disables this
        recorder silently.
        """
        if not self.enabled:
            return
        try:
            self._buffer.append(
                (time.time(), self.process_name, str(stage), float(wait), float(duration))
            )
            if len(self._buffer) >= self.flush_every:
                self.flush()
        except Exception:  # pylint: disable=broad-exception-caught
            self.enabled = False

    @contextmanager
    def timer(self, stage):
        """Context manager form of ``record`` that only measures compute time.

        Useful for timing a sub-step (e.g. just the hardware read/write
        call) rather than the whole command dispatch.
        """
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, duration=time.perf_counter() - start)

    def flush(self) -> None:
        """Append any buffered samples to this recorder's CSV file."""
        if not self.enabled or not self._buffer:
            return
        try:
            if self._file is None:
                self.directory.mkdir(parents=True, exist_ok=True)
                filename = (
                    self.directory
                    / f"{_safe_filename_part(self.process_name)}_{os.getpid()}.csv"
                )
                is_new = not filename.exists()
                self._file = open(filename, "a", newline="", encoding="utf-8")
                self._writer = csv.writer(self._file)
                if is_new:
                    self._writer.writerow(
                        ["timestamp", "process", "stage", "wait_time", "compute_time"]
                    )
            self._writer.writerows(self._buffer)
            self._file.flush()
            self._buffer.clear()
        except Exception:  # pylint: disable=broad-exception-caught
            self.enabled = False

    def close(self) -> None:
        """Flush any remaining samples and close the underlying file.

        Called from the ``finally`` block wrapping each process's ``run``
        loop, so this runs on any graceful shutdown path (``QUIT`` command
        or ``shutdown_event``) but not on a hard ``terminate()``.
        """
        try:
            self.flush()
        finally:
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
                self._file = None


@dataclass
class BenchmarkSample:
    process: str
    timestamp: float
    stage: str
    wait_time: float
    compute_time: float


def load_benchmark_samples(
    directory: "str | Path" = DEFAULT_BENCHMARK_DIRECTORY,
    *,
    since: Optional[float] = None,
    until: Optional[float] = None,
) -> List[BenchmarkSample]:
    """Loads every ``*.csv`` file in ``directory`` into a list of samples.

    Parameters
    ----------
    since, until : float, optional
        Unix timestamps used to restrict samples to a single run, since
        multiple runs' files can accumulate in the same directory.
    """
    directory = Path(directory)
    samples: List[BenchmarkSample] = []
    if not directory.exists():
        return samples
    for filepath in sorted(directory.glob("*.csv")):
        try:
            with open(filepath, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # header
                for row in reader:
                    if len(row) != 5:
                        continue
                    timestamp_str, process_name, stage, wait_str, compute_str = row
                    try:
                        timestamp = float(timestamp_str)
                        wait_time = float(wait_str)
                        compute_time = float(compute_str)
                    except ValueError:
                        continue
                    if since is not None and timestamp < since:
                        continue
                    if until is not None and timestamp > until:
                        continue
                    samples.append(
                        BenchmarkSample(
                            process=process_name,
                            timestamp=timestamp,
                            stage=stage,
                            wait_time=wait_time,
                            compute_time=compute_time,
                        )
                    )
        except OSError:
            continue
    return samples


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(sorted_values: List[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def summarize_samples(
    samples: Iterable[BenchmarkSample],
) -> Dict[Tuple[str, str], dict]:
    """Groups samples by (process, stage) and computes summary statistics.

    Returns a dict keyed by ``(process, stage)`` with count, mean/p95/max
    compute time, and mean/max wait time, in seconds.
    """
    groups: Dict[Tuple[str, str], List[BenchmarkSample]] = defaultdict(list)
    for sample in samples:
        groups[(sample.process, sample.stage)].append(sample)

    summary = {}
    for key, group_samples in groups.items():
        compute_times = sorted(s.compute_time for s in group_samples)
        wait_times = sorted(s.wait_time for s in group_samples)
        summary[key] = {
            "count": len(group_samples),
            "compute_mean": _mean(compute_times),
            "compute_p95": _percentile(compute_times, 0.95),
            "compute_max": compute_times[-1],
            "wait_mean": _mean(wait_times),
            "wait_max": wait_times[-1],
        }
    return summary


def _format_seconds(value: float) -> str:
    return f"{value * 1000:.2f} ms"


def _render_text_report(summary: Dict[Tuple[str, str], dict]) -> str:
    if not summary:
        return "No benchmark samples found.\n"
    lines = [
        f"{'Process':<20} {'Stage':<24} {'Count':>8} "
        f"{'Wait Mean':>12} {'Compute Mean':>14} {'Compute P95':>13} {'Compute Max':>13}"
    ]
    lines.append("-" * len(lines[0]))
    # Slowest average compute time first -- that is the most likely bottleneck.
    for (process, stage), stats in sorted(
        summary.items(), key=lambda item: item[1]["compute_mean"], reverse=True
    ):
        lines.append(
            f"{process:<20} {stage:<24} {stats['count']:>8} "
            f"{_format_seconds(stats['wait_mean']):>12} "
            f"{_format_seconds(stats['compute_mean']):>14} "
            f"{_format_seconds(stats['compute_p95']):>13} "
            f"{_format_seconds(stats['compute_max']):>13}"
        )
    return "\n".join(lines) + "\n"


def _render_html_report(summary: Dict[Tuple[str, str], dict]) -> str:
    rows = sorted(
        summary.items(), key=lambda item: item[1]["compute_mean"], reverse=True
    )
    max_value = max((stats["compute_max"] for _, stats in rows), default=0.0)
    max_value = max(max_value, 1e-9)

    row_height = 34
    chart_width = 480
    label_width = 260
    svg_height = row_height * len(rows) + 40

    bars = []
    for i, ((process, stage), stats) in enumerate(rows):
        y = i * row_height + 30
        wait_width = chart_width * stats["wait_mean"] / max_value
        compute_width = chart_width * stats["compute_mean"] / max_value
        p95_x = label_width + chart_width * stats["compute_p95"] / max_value
        max_x = label_width + chart_width * stats["compute_max"] / max_value
        bars.append(
            f'<text x="4" y="{y + 14}" class="label">{process} / {stage}</text>'
            f'<rect x="{label_width}" y="{y}" width="{wait_width:.1f}" height="10" class="wait-bar" />'
            f'<rect x="{label_width}" y="{y + 12}" width="{compute_width:.1f}" height="10" class="compute-bar" />'
            f'<line x1="{p95_x:.1f}" y1="{y}" x2="{p95_x:.1f}" y2="{y + 22}" class="p95-marker" />'
            f'<line x1="{max_x:.1f}" y1="{y}" x2="{max_x:.1f}" y2="{y + 22}" class="max-marker" />'
            f'<text x="{label_width + chart_width + 8}" y="{y + 14}" class="value">'
            f'{_format_seconds(stats["compute_mean"])} avg'
            f"</text>"
        )

    table_rows = "".join(
        f"<tr><td>{process}</td><td>{stage}</td><td>{stats['count']}</td>"
        f"<td>{_format_seconds(stats['wait_mean'])}</td>"
        f"<td>{_format_seconds(stats['compute_mean'])}</td>"
        f"<td>{_format_seconds(stats['compute_p95'])}</td>"
        f"<td>{_format_seconds(stats['compute_max'])}</td></tr>"
        for (process, stage), stats in rows
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Rattlesnake Benchmark Report</title>
<style>
  body {{ font-family: sans-serif; margin: 2rem; color: #1a1a1a; background: #fff; }}
  @media (prefers-color-scheme: dark) {{
    body {{ color: #e8e8e8; background: #1e1e1e; }}
    table {{ color: #e8e8e8; }}
    .label, .value {{ fill: #e8e8e8; }}
  }}
  h1 {{ font-size: 1.3rem; }}
  p.note {{ color: #777; font-size: 0.85rem; }}
  table {{ border-collapse: collapse; margin-top: 1rem; font-size: 0.85rem; }}
  th, td {{ padding: 4px 10px; text-align: right; border-bottom: 1px solid #8884; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  svg {{ overflow: visible; }}
  .label {{ font-size: 12px; fill: #1a1a1a; }}
  .value {{ font-size: 11px; fill: #777; }}
  .wait-bar {{ fill: #9aa5b1; }}
  .compute-bar {{ fill: #3b82c4; }}
  .p95-marker {{ stroke: #e08e2d; stroke-width: 2; }}
  .max-marker {{ stroke: #c0392b; stroke-width: 2; }}
  .legend span {{ display: inline-block; width: 12px; height: 12px; margin-right: 4px; vertical-align: middle; }}
</style>
</head>
<body>
<h1>Rattlesnake Benchmark Report</h1>
<p class="note">
  Per (process, stage) command-dispatch timing, grouped from every
  <code>benchmark_data/*.csv</code> file for this run. "Stage" is the
  command name each process was handling (e.g. RUN_HARDWARE for
  Acquisition/Output, ACQUIRE for a data collector) -- since these commands
  re-queue themselves every frame while active, each sample is
  approximately one frame's worth of work.
</p>
<p class="legend">
  <span style="background:#9aa5b1;"></span> mean wait time
  &nbsp;&nbsp;<span style="background:#3b82c4;"></span> mean compute time
  &nbsp;&nbsp;<span style="border-left:2px solid #e08e2d; height:12px; display:inline-block;"></span> p95 compute
  &nbsp;&nbsp;<span style="border-left:2px solid #c0392b; height:12px; display:inline-block;"></span> max compute
</p>
<svg width="{label_width + chart_width + 140}" height="{svg_height}">
{''.join(bars)}
</svg>
<table>
<tr><th>Process</th><th>Stage</th><th>Count</th><th>Wait Mean</th>
<th>Compute Mean</th><th>Compute P95</th><th>Compute Max</th></tr>
{table_rows}
</table>
</body>
</html>
"""


def generate_benchmark_report(
    directory: "str | Path" = DEFAULT_BENCHMARK_DIRECTORY,
    *,
    output_path: Optional["str | Path"] = None,
    since: Optional[float] = None,
    until: Optional[float] = None,
) -> Optional[Path]:
    """Loads benchmark samples and writes an HTML + console summary report.

    Intended to be called once, as a post-processing step, after Rattlesnake
    has fully shut down (so every process has already flushed its samples
    to disk). Pass ``since=<time.time() at startup>`` to restrict the
    report to a single run when ``directory`` may contain files from
    previous runs.

    Returns the path to the written report, or ``None`` if benchmarking was
    not enabled / no samples were found.
    """
    directory = Path(directory)
    samples = load_benchmark_samples(directory, since=since, until=until)
    if not samples:
        print(
            "No benchmark samples found. Set the RATTLESNAKE_BENCHMARK "
            "environment variable to a truthy value before launching "
            "Rattlesnake to collect timing data."
        )
        return None

    summary = summarize_samples(samples)
    print(_render_text_report(summary))

    if output_path is None:
        output_path = directory / f"benchmark_report_{int(time.time())}.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_html_report(summary), encoding="utf-8")
    return output_path


def plot_benchmark_data(
    directory: "str | Path" = DEFAULT_BENCHMARK_DIRECTORY,
    *,
    since: Optional[float] = None,
    until: Optional[float] = None,
    output_path: Optional["str | Path"] = None,
    show: bool = True,
):
    """Plots a per-process timeline of compute vs. wait periods with matplotlib.

    Draws one row per process along a shared time axis. Each recorded
    command dispatch is reconstructed as two adjacent intervals -- the time
    the process spent waiting for that command, immediately followed by the
    time it spent executing it -- and drawn as a Gantt-style timeline: the
    compute interval is filled in, the wait interval is left uncolored
    (outline only). A row that is mostly filled in is running close to
    back-to-back, with little slack before it would start missing its
    real-time budget; a row with large uncolored gaps has headroom.

    Requires matplotlib (``pip install matplotlib``, or the ``benchmark``
    extra: ``pip install rattlesnake-vibration-controller[benchmark]``).
    It is only imported here, not at module load time, so the recording
    path used by every controller process never needs matplotlib installed.

    Parameters
    ----------
    since, until : float, optional
        Unix timestamps used to restrict samples to a single run, since
        multiple runs' files can accumulate in the same directory. ``since``
        is also used as the timeline's zero point when given, so the x-axis
        reads as "time since the run started" -- pass the ``time.time()``
        value captured right before the controller was constructed.
    output_path : str or Path, optional
        Where to save the plot as a PNG. Defaults to a timestamped file
        inside ``directory``.
    show : bool
        Whether to also open an interactive matplotlib window
        (``plt.show()``). Set to ``False`` for non-interactive/headless use.

    Returns
    -------
    Path or None
        The path the plot was saved to, or ``None`` if no samples were found.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError(
            "plot_benchmark_data requires matplotlib. Install it with "
            "`pip install matplotlib`, or `pip install "
            "rattlesnake-vibration-controller[benchmark]`."
        ) from exc

    directory = Path(directory)
    samples = load_benchmark_samples(directory, since=since, until=until)
    if not samples:
        print(
            "No benchmark samples found. Set the RATTLESNAKE_BENCHMARK "
            "environment variable to a truthy value before launching "
            "Rattlesnake to collect timing data."
        )
        return None

    by_process: Dict[str, List[BenchmarkSample]] = defaultdict(list)
    for sample in samples:
        by_process[sample.process].append(sample)

    # Busiest process (most total compute time) first/top.
    processes = sorted(
        by_process,
        key=lambda name: sum(s.compute_time for s in by_process[name]),
        reverse=True,
    )

    # Each sample's timestamp is recorded right as its compute segment ends,
    # so walk backwards from there: [.. wait_time ..][.. compute_time ..]|<-timestamp
    if since is not None:
        t0 = since
    else:
        t0 = min(s.timestamp - s.compute_time - s.wait_time for s in samples)

    compute_color = "#3b82c4"
    bar_height = 0.8
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(processes) + 1.5))

    for row, process in enumerate(processes):
        compute_intervals = []
        wait_intervals = []
        for sample in sorted(by_process[process], key=lambda s: s.timestamp):
            compute_end = sample.timestamp - t0
            compute_start = compute_end - sample.compute_time
            wait_start = compute_start - sample.wait_time
            if sample.wait_time > 0:
                wait_intervals.append((wait_start, sample.wait_time))
            if sample.compute_time > 0:
                compute_intervals.append((compute_start, sample.compute_time))

        y = row - bar_height / 2
        if wait_intervals:
            ax.broken_barh(
                wait_intervals,
                (y, bar_height),
                facecolors="none",
                edgecolors=compute_color,
                linewidths=0.8,
            )
        if compute_intervals:
            ax.broken_barh(
                compute_intervals,
                (y, bar_height),
                facecolors=compute_color,
                edgecolors=compute_color,
                linewidths=0.8,
            )

    ax.set_yticks(range(len(processes)))
    ax.set_yticklabels(processes)
    ax.invert_yaxis()
    ax.set_xlabel("Time since start of run (s)" if since is not None else "Time (s)")
    ax.set_title("Rattlesnake Benchmark Timeline: Compute vs. Wait Time by Process")
    ax.legend(
        handles=[
            Patch(facecolor=compute_color, edgecolor=compute_color, label="Compute (busy)"),
            Patch(facecolor="none", edgecolor=compute_color, label="Wait (idle)"),
        ],
        loc="upper right",
    )
    fig.tight_layout()

    if output_path is None:
        output_path = directory / f"benchmark_timeline_{int(time.time())}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Benchmark timeline plot written to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path
