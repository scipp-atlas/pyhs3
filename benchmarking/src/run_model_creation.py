from __future__ import annotations

import argparse
import gc
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from pyhs3.model import Model
from pyhs3.workspace import Workspace

from config import (
    DEFAULT_MODE,
    DEFAULT_N_RUNS,
    DEFAULT_TARGET,
    DEFAULT_WORKSPACE,
    PLOTS_DIR,
    RESULTS_DIR,
)
from utils import (
    create_model,
    get_current_rss_mb,
    get_peak_rss_mb,
    load_workspace,
    make_bar_plot,
    save_json,
    summarize_timings,
    get_main_sha,
)


BENCHMARK_NAME = "model_creation"

DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = "model_creation_result.json"

DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME
DEFAULT_PLOT_NAME = "model_creation_wall_time.png"


def validate_model(model: Model) -> dict[str, Any]:
    """
    Validate that the created model exposes expected interfaces.
    """

    if model is None:
        raise ValueError("Model creation returned None")

    if not hasattr(model, "log_prob"):
        raise ValueError("Model does not expose log_prob")

    if not hasattr(model, "data"):
        raise ValueError("Model does not expose data")

    if not hasattr(model, "free_params"):
        raise ValueError("Model does not expose free_params")

    return {
        "model_type": type(model).__name__,
        "has_log_prob": hasattr(model, "log_prob"),
        "has_data": hasattr(model, "data"),
        "has_free_params": hasattr(model, "free_params"),
        "n_free_params": (
            len(model.free_params)
            if getattr(model, "free_params", None) is not None
            else 0
        ),
    }


def measure_model_creation_memory(
    workspace: Workspace,
    target: str,
    mode: str,
) -> tuple[Model, dict[str, float]]:
    """
    Measure memory change for a single model creation.

    This is intentionally separate from repeated timing runs to avoid reporting
    memory accumulation across many created PyTensor graphs as the memory cost of
    one ws.model(...) call.
    """

    gc.collect()

    warmup_model = create_model(
        workspace=workspace,
        target=target,
        mode=mode,
    )

    del warmup_model
    gc.collect()

    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()

    model = create_model(
        workspace=workspace,
        target=target,
        mode=mode,
    )

    current_rss_after_mb = get_current_rss_mb()
    peak_rss_after_mb = get_peak_rss_mb()

    return model, {
        "memory_n_runs": 1,
        "current_rss_before_mb": current_rss_before_mb,
        "current_rss_after_mb": current_rss_after_mb,
        "current_rss_delta_mb": current_rss_after_mb - current_rss_before_mb,
        "peak_rss_before_mb": peak_rss_before_mb,
        "peak_rss_after_mb": peak_rss_after_mb,
        "peak_rss_delta_mb": peak_rss_after_mb - peak_rss_before_mb,
    }


def measure_model_creation_timing(
    workspace: Workspace,
    target: str,
    mode: str,
    n_runs: int,
) -> list[float]:
    """
    Measure repeated ws.model(...) wall times.

    Created models are discarded after each run. Memory is measured separately
    using a single model creation.
    """

    timings: list[float] = []

    for _ in range(n_runs):
        start = time.perf_counter()
        model = create_model(
            workspace=workspace,
            target=target,
            mode=mode,
        )
        end = time.perf_counter()

        timings.append(end - start)

        del model
        gc.collect()

    return timings


def run_single_benchmark(
    workspace_path: Path,
    target: str,
    mode: str,
    n_runs: int,
) -> dict[str, Any]:
    """
    Run model creation benchmark for one workspace, target, and mode.

    Workspace loading is setup only and is intentionally excluded from timing.

    Timing measures repeated ws.model(...) calls.
    Memory measures one isolated ws.model(...) call.
    """

    workspace = load_workspace(workspace_path)

    model, memory_summary = measure_model_creation_memory(
        workspace=workspace,
        target=target,
        mode=mode,
    )
    validation_summary = validate_model(model)

    del model
    gc.collect()

    timings = measure_model_creation_timing(
        workspace=workspace,
        target=target,
        mode=mode,
        n_runs=n_runs,
    )
    timing_summary = summarize_timings(timings)

    return {
        "benchmark": BENCHMARK_NAME,
        "workspace": workspace_path.name,
        "workspace_path": str(workspace_path),
        "target": target,
        "mode": mode,
        "n_runs": n_runs,
        "wall_time_seconds_samples": timings,
        **timing_summary,
        **memory_summary,
        "status": "success",
        **validation_summary,
    }


def print_result(result: dict[str, Any]) -> None:
    """
    Print a readable benchmark summary.
    """

    print()
    print("=" * 72)
    print("Model creation benchmark")
    print("=" * 72)
    print(f"Workspace: {result['workspace']}")
    print(f"Target:    {result['target']}")
    print(f"Mode:      {result['mode']}")
    print(f"Runs:      {result['n_runs']}")
    print(f"Status:    {result['status']}")

    print()
    print("Timing")
    print(f"  mean:   {result['wall_time_seconds_mean'] * 1000:.3f} ms")
    print(f"  median: {result['wall_time_seconds_median'] * 1000:.3f} ms")
    print(f"  std:    {result['wall_time_seconds_std'] * 1000:.3f} ms")

    print()
    print("Memory")
    print(f"  memory runs:        {result['memory_n_runs']}")
    print(f"  current RSS before: {result['current_rss_before_mb']:.3f} MB")
    print(f"  current RSS after:  {result['current_rss_after_mb']:.3f} MB")
    print(f"  current RSS delta:  {result['current_rss_delta_mb']:.3f} MB")
    print(f"  peak RSS before:    {result['peak_rss_before_mb']:.3f} MB")
    print(f"  peak RSS after:     {result['peak_rss_after_mb']:.3f} MB")
    print(f"  peak RSS delta:     {result['peak_rss_delta_mb']:.3f} MB")

    print()
    print("Validation")
    print(f"  model type:       {result['model_type']}")
    print(f"  has log_prob:     {result['has_log_prob']}")
    print(f"  has data:         {result['has_data']}")
    print(f"  has free_params:  {result['has_free_params']}")
    print(f"  free parameters:  {result['n_free_params']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pyHS3 Model creation from already-loaded Workspaces."
    )

    parser.add_argument(
        "--workspaces",
        nargs="+",
        type=Path,
        default=[DEFAULT_WORKSPACE],
        help="Workspace JSON files to benchmark.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=[DEFAULT_TARGET],
        help="Workspace model targets, such as analysis or likelihood names.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[DEFAULT_MODE],
        help="PyTensor compilation modes passed to workspace.model(...).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help="Number of repeated timing runs per workspace/target/mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSON benchmark output will be saved.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Name of the JSON benchmark output file.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create comparison plots when multiple result entries are available.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--plot-name",
        default=DEFAULT_PLOT_NAME,
        help="Name of the wall-time plot output file.",
    )

    return parser.parse_args()


def make_plots(
    results: list[dict[str, Any]],
    plot_dir: Path,
    wall_time_plot_name: str,
) -> None:
    """
    Create standard plots for the model creation benchmark.
    """

    plot_dir.mkdir(parents=True, exist_ok=True)

    wall_time_plot_path = plot_dir / wall_time_plot_name
    make_bar_plot(
        results=results,
        output_path=wall_time_plot_path,
        title="Model creation wall time",
        metric_key="wall_time_seconds_mean",
        metric_label="Mean wall time [s]",
        error_key="wall_time_seconds_std",
    )
    print(f"Saved plot to {wall_time_plot_path}")

    current_rss_plot_path = plot_dir / "model_creation_current_rss_delta.png"
    make_bar_plot(
        results=results,
        output_path=current_rss_plot_path,
        title="Model creation current RSS delta",
        metric_key="current_rss_delta_mb",
        metric_label="Current RSS delta [MB]",
    )
    print(f"Saved plot to {current_rss_plot_path}")

    peak_rss_plot_path = plot_dir / "model_creation_peak_rss_delta.png"
    make_bar_plot(
        results=results,
        output_path=peak_rss_plot_path,
        title="Model creation peak RSS delta",
        metric_key="peak_rss_delta_mb",
        metric_label="Peak RSS delta [MB]",
    )
    print(f"Saved plot to {peak_rss_plot_path}")


def main() -> None:
    args = parse_args()

    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1")

    results = []

    ctx = get_context("spawn")

    for workspace_path in args.workspaces:
        for target in args.targets:
            for mode in args.modes:
                with ctx.Pool(processes=1) as pool:
                    result = pool.apply(
                        run_single_benchmark,
                        args=(workspace_path, target, mode, args.n_runs),
                    )

                results.append(result)
                print_result(result)

    output_data: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "n_results": len(results),
        "results": results,
        "pyhs3_main_sha": get_main_sha(),
    }

    output_path = args.output_dir / args.output_name
    save_json(output_data, output_path)

    print()
    print(f"Saved result to {output_path}")

    if args.plot:
        if len(results) < 2:
            print("Skipping plots: at least two result entries are needed.")
        else:
            make_plots(
                results=results,
                plot_dir=args.plot_dir,
                wall_time_plot_name=args.plot_name,
            )


if __name__ == "__main__":
    main()
