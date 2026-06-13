from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable


def run_repeated_timing(
    func: Callable[[], Any],
    n_runs: int = 5,
) -> tuple[Any, list[float]]:
    """
    Run a function multiple times and return the last result
    together with all timing measurements.
    """

    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    timings: list[float] = []
    result: Any = None

    for _ in range(n_runs):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()

        timings.append(end - start)

    return result, timings


def summarize_timings(timings: list[float]) -> dict[str, float]:
    """
    Compute mean and standard deviation of timing measurements.
    """

    if not timings:
        raise ValueError("timings must not be empty")

    return {
        "wall_time_seconds_mean": statistics.mean(timings),
        "wall_time_seconds_std": (
            statistics.stdev(timings)
            if len(timings) > 1
            else 0.0
        ),
    }


def save_json(
    data: dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Save benchmark results as formatted JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            data,
            output_file,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")
