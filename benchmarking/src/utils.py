from __future__ import annotations

import json
import resource
import statistics
import time
import psutil
import matplotlib.pyplot as plt
import math
import subprocess

from matplotlib.ticker import (
    AutoMinorLocator,
    MaxNLocator,
)
from pathlib import Path
from typing import Any

from pyhs3 import jaxify
from pyhs3.model import Model
from pyhs3.transpile import JaxifiedGraph
from pyhs3.workspace import Workspace
from pytensor.tensor.variable import TensorVariable

from config import WORKSPACE_LABELS

def get_current_rss_mb() -> float:
    """
    Return current process RSS usage in MB.
    """

    process = psutil.Process()

    return process.memory_info().rss / (1024 * 1024)


def get_peak_rss_mb() -> float:
    """
    Return peak process RSS usage in MB.

    On Linux, ru_maxrss is reported in KB.
    """

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return rss_kb / 1024.0


def run_repeated_timing(func, n_runs: int = 5, warmup_runs: int = 1) -> tuple[Any, list[float]]:
    """
    Run the given function multiple times and return 
    the result of the last run along with the list of timings for each run.
    """

    timings = []

    for _ in range(warmup_runs):
        func()
    
    result = None

    for _ in range(n_runs):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()

        timings.append(end - start)

    return result, timings


def summarize_timings(timings):
    """
    Summarize the list of timings by calculating the mean and standard deviation.
    """

    return {
        "wall_time_seconds_mean": statistics.mean(timings),
        "wall_time_seconds_median": statistics.median(timings),
        "wall_time_seconds_std": (
            statistics.stdev(timings)
            if len(timings) > 1
            else 0.0
        ),
    }

def get_main_sha() -> str:
    """
    Return the short SHA of the pyHS3 main branch used as the benchmark baseline.
    """

    result = subprocess.run(
        ["git", "rev-parse", "--short", "origin/main"],
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def save_json(data, output_path: Path):
    """
    Save the given data as JSON to the specified output path.
    Creates parent directories if they do not exist.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w") as f:
        json.dump(
            data,
            f,
            indent=2,
            sort_keys=True,
        )

def should_plot_metric(
    results: list[dict[str, Any]],
    metric_key: str,
) -> bool:
    """Return True if a metric exists and has at least one non-zero value."""

    values = [result.get(metric_key, 0.0) for result in results]
    return any(value != 0 for value in values)



def _apply_style() -> None:
    """
    Apply a  matplotlib style.
    """

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.25",
            "axes.linewidth": 1.5,
            "axes.titlesize": 24,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "grid.color": "0.55",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.35,
            "legend.frameon": True,
            "legend.fontsize": 14,
            "savefig.dpi": 300,
        }
    )



def _result_label(result: dict[str, Any]) -> str:
    """
    Create a compact multi-line label for one benchmark result.
    """

    parts = []

    workspace = result.get("workspace")
    if workspace is not None:
        workspace_key = str(workspace).replace(".json", "")
        parts.append(WORKSPACE_LABELS.get(workspace_key, workspace_key))

    target = result.get("target")
    if target is not None:
        parts.append(str(target))

    mode = result.get("mode")
    if mode is not None:
        parts.append(str(mode))
    
    n_evaluations = result.get("n_evaluations")
    if n_evaluations is not None:
        parts.append(f"{n_evaluations} evals")

    return "\n".join(parts)



def _scaled_metric(
    results: list[dict[str, Any]],
    metric_key: str,
    metric_label: str,
) -> tuple[list[float], list[float] | None, str]:
    """
    Return values, optional errors, and an updated y-axis label.

    Timing means are stored in seconds but are usually easier to read in ms.
    If a matching *_std field exists, it is used as an error bar.
    """

    values = [float(result[metric_key]) for result in results]

    std_key = None
    if metric_key.endswith("_mean"):
        candidate = metric_key.removesuffix("_mean") + "_std"
        if any(candidate in result for result in results):
            std_key = candidate

    errors = (
        [float(result.get(std_key, 0.0)) for result in results]
        if std_key is not None
        else None
    )

    if metric_key == "wall_time_seconds_mean":
        values = [value * 1000.0 for value in values]
        if errors is not None:
            errors = [error * 1000.0 for error in errors]
        metric_label = "Mean wall time [ms]"

    return values, errors, metric_label



def _format_value(value: float, metric_label: str) -> str:
    """
    Format bar labels in readable units.
    """

    if not math.isfinite(value):
        return "nan"

    if "[ms]" in metric_label:
        return f"{value:.2f}"
    if "[MB]" in metric_label:
        return f"{value:.2f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"

def make_bar_plot(
    results: list[dict[str, Any]],
    output_path: Path,
    title: str,
    metric_key: str,
    metric_label: str,
    error_key: str | None = None,
) -> None:
    """
    Create a bar plot for a benchmark metric.
    """

    labels = [_result_label(result) for result in results]
    values = [result[metric_key] for result in results]

    errors = None
    if error_key is not None:
        errors = [result.get(error_key, 0.0) for result in results]

    fig_width = max(10, len(labels) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    bars = ax.bar(
        range(len(values)),
        values,
        yerr=errors,
        capsize=6 if errors is not None else 0,
    )

    ax.set_title(title, fontsize=18)
    ax.set_ylabel(metric_label)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.grid(axis="y", alpha=0.3)

    ymin = min(0.0, min(values))
    ymax = max(values)

    if errors is not None:
        ymax = max(
            value + error
            for value, error in zip(values, errors, strict=False)
        )

    span = ymax - ymin

    if span <= 0:
        span = 1.0

    ax.set_ylim(
        ymin - 0.08 * span,
        ymax + 0.25 * span,
    )

    for index, (bar, value) in enumerate(zip(bars, values, strict=False)):
        error = 0.0
        if errors is not None:
            error = errors[index]

        offset = 0.035 * span

        if value >= 0:
            y = value + error + offset
            va = "bottom"
        else:
            y = value - error - offset
            va = "top"

        if errors is not None and any(
            error_value != 0 for error_value in errors
        ):
            label = (
                f"{_format_value(value, metric_label)} ± "
                f"{_format_value(error, metric_label)}"
            )
        else:
            label = _format_value(value, metric_label)

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va=va,
            fontsize=11,
        )

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(output_path)
    plt.close(fig)


def load_workspace(workspace_path: Path) -> Workspace:
    """
    Load a workspace from the given path.
    """

    return Workspace.load(workspace_path)

def create_model(
    workspace: Workspace,
    target: str,
    mode: str,
) -> Model:
    """
    Create a model from the given workspace, target, and mode.
    """

    return workspace.model(
        target,
        progress=False,
        mode=mode,
    )

def build_log_prob(
    workspace_path: Path,
    target: str,
    mode: str,
) -> tuple[Model, TensorVariable]:
    """
    Build a log_prob graph from the given workspace, target, and mode.
    """

    workspace = load_workspace(workspace_path)
    model = create_model(
        workspace=workspace,
        target=target,
        mode=mode,
    )
    return model, model.log_prob
