from __future__ import annotations

import argparse
import gc
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from pytensor.tensor.variable import TensorVariable

from pyhs3.model import Model
from pyhs3.workspace import Workspace
from utils import (
    get_current_rss_mb,
    get_peak_rss_mb,
    make_bar_plot,
    save_json,
    summarize_timings,
    should_plot_metric,
    load_workspace,
    create_model,
    get_main_sha,
)

from config import (
    DEFAULT_MODE,
    DEFAULT_N_RUNS,
    DEFAULT_TARGET,
    DEFAULT_WORKSPACE,
    PLOTS_DIR,
    RESULTS_DIR,
)


BENCHMARK_NAME = "log_prob_construction"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = "log_prob_construction_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME


def construct_log_prob(model: Model) -> TensorVariable:
    """
    Construct the symbolic log-probability expression.

    model.log_prob is a property. Accessing it builds the PyTensor likelihood
    expression but does not compile or evaluate it.
    """

    return model.log_prob


def validate_log_prob(log_prob: TensorVariable) -> dict[str, Any]:
    """
    Validate that the log-probability expression was constructed.
    """

    if log_prob is None:
        raise ValueError("log_prob construction returned None")

    if not isinstance(log_prob, TensorVariable):
        raise TypeError(
            f"Expected TensorVariable, got {type(log_prob).__name__}"
        )

    return {
        "log_prob_type": type(log_prob).__name__,
        "log_prob_name": str(log_prob),
        "log_prob_ndim": log_prob.ndim,
        "log_prob_dtype": str(log_prob.dtype),
        "can_proceed_to_compilation": True,
    }


def measure_log_prob_construction_memory(model: Model) -> tuple[TensorVariable, dict[str, float]]:
    """
    Measure memory change for a single log_prob construction.

    This is separate from repeated timing runs so memory is not reported as
    accumulation across many constructed PyTensor expressions.
    """

    gc.collect()

    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()

    log_prob = construct_log_prob(model)

    current_rss_after_mb = get_current_rss_mb()
    peak_rss_after_mb = get_peak_rss_mb()

    return log_prob, {
        "memory_n_runs": 1,
        "current_rss_before_mb": current_rss_before_mb,
        "current_rss_after_mb": current_rss_after_mb,
        "current_rss_delta_mb": current_rss_after_mb - current_rss_before_mb,
        "peak_rss_before_mb": peak_rss_before_mb,
        "peak_rss_after_mb": peak_rss_after_mb,
        "peak_rss_delta_mb": peak_rss_after_mb - peak_rss_before_mb,
    }


def measure_log_prob_construction_timing(
    workspace: Workspace,
    target: str,
    mode: str,
    n_runs: int,
) -> list[float]:
    """
    Measure repeated log_prob construction wall times.

    A fresh Model is created for each timing run so that each measured access to
    model.log_prob corresponds to constructing the likelihood expression from a
    fresh model. The time spent in ws.model(...) is excluded from the measured
    interval.
    """

    timings: list[float] = []

    for _ in range(n_runs):
        model = create_model(
            workspace=workspace,
            target=target,
            mode=mode,
        )

        start = time.perf_counter()
        log_prob = construct_log_prob(model)
        end = time.perf_counter()

        timings.append(end - start)

        del log_prob
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
    Run log_prob construction benchmark for one workspace, target, and mode.

    Workspace loading and model creation are setup only.
    The timed operation is only accessing model.log_prob.
    """

    workspace = load_workspace(workspace_path)

    model_for_memory = create_model(
        workspace=workspace,
        target=target,
        mode=mode,
    )

    log_prob, memory_summary = measure_log_prob_construction_memory(
        model_for_memory
    )
    validation_summary = validate_log_prob(log_prob)

    del log_prob
    del model_for_memory
    gc.collect()

    timings = measure_log_prob_construction_timing(
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
        "timings_seconds": timings,
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
    print("log_prob construction benchmark")
    print("=" * 72)
    print(f"Workspace: {result['workspace']}")
    print(f"Target:    {result['target']}")
    print(f"Mode:      {result['mode']}")
    print(f"Runs:      {result['n_runs']}")
    print(f"Status:    {result['status']}")

    print()
    print("Timing")
    print(f"  mean: {result['wall_time_seconds_mean'] * 1000:.3f} ms")
    print(f"  median: {result['wall_time_seconds_median'] * 1000:.3f} ms")
    print(f"  std:  {result['wall_time_seconds_std'] * 1000:.3f} ms")

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
    print(f"  log_prob type:     {result['log_prob_type']}")
    print(f"  log_prob name:     {result['log_prob_name']}")
    print(f"  log_prob ndim:     {result['log_prob_ndim']}")
    print(f"  log_prob dtype:    {result['log_prob_dtype']}")
    print(f"  compilable:        {result['can_proceed_to_compilation']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark symbolic log_prob construction from pyHS3 Models."
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
        help="Create benchmark plots when multiple result entries are available.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Directory where plots will be saved.",
    )

    return parser.parse_args()


def make_plots(
    results: list[dict[str, Any]],
    plot_dir: Path,
) -> None:
    """
    Create standard plots for the log_prob construction benchmark.
    """

    plot_dir.mkdir(parents=True, exist_ok=True)

    make_bar_plot(
        results=results,
        output_path=plot_dir / "log_prob_construction_wall_time.png",
        title="log_prob construction wall time",
        metric_key="wall_time_seconds_mean",
        metric_label="Mean wall time [s]",
    )

    if should_plot_metric(results, "current_rss_delta_mb"):
        make_bar_plot(
            results=results,
            output_path=plot_dir / "log_prob_construction_current_rss_delta.png",
            title="log_prob construction RSS increase",
            metric_key="current_rss_delta_mb",
            metric_label="RSS increase [MB]",
        )

    if should_plot_metric(results, "peak_rss_delta_mb"):
        make_bar_plot(
            results=results,
            output_path=plot_dir / "log_prob_construction_peak_rss_delta.png",
            title="log_prob construction peak RSS increase",
            metric_key="peak_rss_delta_mb",
            metric_label="Peak RSS increase [MB]",
        )


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
            make_plots(results=results, plot_dir=args.plot_dir)
            print(f"Saved plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
