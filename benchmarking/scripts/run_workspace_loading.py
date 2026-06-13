from __future__ import annotations

from pathlib import Path

from utils import (
    run_repeated_timing,
    save_json,
    summarize_timings,
)

DEFAULT_MODE = "FAST_RUN"
N_RUNS = 5


def load_workspace_placeholder() -> dict[str, str]:
    """
    Placeholder workspace loading function.

    This will later be replaced by real pyHS3 workspace loading.
    """
    return {
        "workspace": "small",
        "status": "loaded",
    }


def main() -> None:
    result, timings = run_repeated_timing(
        load_workspace_placeholder,
        n_runs=N_RUNS,
    )

    timing_summary = summarize_timings(timings)

    benchmark_result = {
        "benchmark": "workspace_loading",
        "workspace": result["workspace"],
        "mode": DEFAULT_MODE,
        "n_runs": N_RUNS,
        **timing_summary,
        "rss_mb_peak": None,
        "status": "success",
    }

    output_path = Path(
        "benchmarking/results/workspace_loading_result.json"
    )

    save_json(
        benchmark_result,
        output_path,
    )

    print(f"Saved result to {output_path}")
    print(benchmark_result)


if __name__ == "__main__":
    main()