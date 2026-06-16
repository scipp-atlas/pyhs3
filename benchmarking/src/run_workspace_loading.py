from __future__ import annotations

import argparse
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from pyhs3.workspace import Workspace

from config import (
    DEFAULT_N_RUNS,
    DEFAULT_WORKSPACE,
    PLOTS_DIR,
    RESULTS_DIR,
)
from utils import (
    get_current_rss_mb,
    get_peak_rss_mb,
    load_workspace,
    make_bar_plot,
    run_repeated_timing,
    save_json,
    summarize_timings,
)


BENCHMARK_NAME = "workspace_loading"

DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = "workspace_loading_result.json"

DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME
DEFAULT_PLOT_NAME = "workspace_loading_wall_time.png"


def validate_workspace(workspace: Workspace) -> dict[str, int | str]:
    """
    Validate that the loaded workspace contains expected top-level content.
    """

    if workspace.distributions is None or len(workspace.distributions) == 0:
        raise ValueError("Workspace does not contain distributions")

    if workspace.likelihoods is None or len(workspace.likelihoods) == 0:
        raise ValueError("Workspace does not contain likelihoods")

    if workspace.data is None or len(workspace.data) == 0:
        raise ValueError("Workspace does not contain data")

    return {
        "metadata_hs3_version": workspace.metadata.hs3_version,
        "n_distributions": len(workspace.distributions),
        "n_likelihoods": len(workspace.likelihoods),
        "n_data": len(workspace.data),
        "n_domains": len(workspace.domains) if workspace.domains is not None else 0,
        "n_parameter_points": (
            len(workspace.parameter_points)
            if workspace.parameter_points is not None
            else 0
        ),
    }


def run_single_benchmark(
    workspace_path: Path,
    n_runs: int,
) -> dict[str, Any]:
    """
    Run workspace loading benchmark for one workspace.
    """

    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()

    workspace, timings = run_repeated_timing(
        lambda: load_workspace(workspace_path),
        n_runs=n_runs,
    )

    current_rss_after_mb = get_current_rss_mb()
    peak_rss_after_mb = get_peak_rss_mb()

    validation_summary = validate_workspace(workspace)
    timing_summary = summarize_timings(timings)

    return {
        "benchmark": BENCHMARK_NAME,
        "workspace": workspace_path.name,
        "workspace_path": str(workspace_path),
        "n_runs": n_runs,
        "wall_time_seconds_samples": timings,
        **timing_summary,
        "current_rss_before_mb": current_rss_before_mb,
        "current_rss_after_mb": current_rss_after_mb,
        "current_rss_delta_mb": current_rss_after_mb - current_rss_before_mb,
        "peak_rss_before_mb": peak_rss_before_mb,
        "peak_rss_after_mb": peak_rss_after_mb,
        "peak_rss_delta_mb": peak_rss_after_mb - peak_rss_before_mb,
        "status": "success",
        **validation_summary,
    }


def print_result(result: dict[str, Any]) -> None:
    """
    Print a readable benchmark summary.
    """

    print()
    print("=" * 72)
    print("Workspace loading benchmark")
    print("=" * 72)
    print(f"Workspace: {result['workspace']}")
    print(f"Runs:      {result['n_runs']}")
    print(f"Status:    {result['status']}")

    print()
    print("Timing")
    print(f"  mean:   {result['wall_time_seconds_mean'] * 1000:.3f} ms")
    print(f"  median: {result['wall_time_seconds_median'] * 1000:.3f} ms")
    print(f"  std:    {result['wall_time_seconds_std'] * 1000:.3f} ms")

    print()
    print("Memory")
    print(f"  current RSS before: {result['current_rss_before_mb']:.3f} MB")
    print(f"  current RSS after:  {result['current_rss_after_mb']:.3f} MB")
    print(f"  current RSS delta:  {result['current_rss_delta_mb']:.3f} MB")
    print(f"  peak RSS before:    {result['peak_rss_before_mb']:.3f} MB")
    print(f"  peak RSS after:     {result['peak_rss_after_mb']:.3f} MB")
    print(f"  peak RSS delta:     {result['peak_rss_delta_mb']:.3f} MB")

    print()
    print("Validation")
    print(f"  HS3 version:      {result['metadata_hs3_version']}")
    print(f"  distributions:    {result['n_distributions']}")
    print(f"  likelihoods:      {result['n_likelihoods']}")
    print(f"  data:             {result['n_data']}")
    print(f"  domains:          {result['n_domains']}")
    print(f"  parameter points: {result['n_parameter_points']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark HS3 workspace loading into pyHS3 Workspace objects."
    )

    parser.add_argument(
        "--workspaces",
        nargs="+",
        type=Path,
        default=[DEFAULT_WORKSPACE],
        help="Workspace JSON files to benchmark.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help="Number of repeated runs per workspace.",
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
        help="Create comparison plots when multiple workspaces are benchmarked.",
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
    Create standard plots for the workspace loading benchmark.
    """

    plot_dir.mkdir(parents=True, exist_ok=True)

    wall_time_plot_path = plot_dir / wall_time_plot_name
    make_bar_plot(
        results=results,
        output_path=wall_time_plot_path,
        title="Workspace loading wall time",
        metric_key="wall_time_seconds_mean",
        metric_label="Mean wall time [s]",
        error_key="wall_time_seconds_std",
    )
    print(f"Saved plot to {wall_time_plot_path}")

    peak_rss_plot_path = plot_dir / "workspace_loading_peak_rss_delta.png"
    make_bar_plot(
        results=results,
        output_path=peak_rss_plot_path,
        title="Workspace loading peak RSS delta",
        metric_key="peak_rss_delta_mb",
        metric_label="Peak RSS delta [MB]",
    )
    print(f"Saved plot to {peak_rss_plot_path}")

    current_rss_plot_path = plot_dir / "workspace_loading_current_rss_delta.png"
    make_bar_plot(
        results=results,
        output_path=current_rss_plot_path,
        title="Workspace loading current RSS delta",
        metric_key="current_rss_delta_mb",
        metric_label="Current RSS delta [MB]",
    )
    print(f"Saved plot to {current_rss_plot_path}")


def main() -> None:
    args = parse_args()

    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1")

    results = []

    ctx = get_context("spawn")

    for workspace_path in args.workspaces:
        with ctx.Pool(processes=1) as pool:
            result = pool.apply(
                run_single_benchmark,
                args=(workspace_path, args.n_runs),
            )

        results.append(result)
        print_result(result)

    output_data: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "n_workspaces": len(results),
        "results": results,
    }

    output_path = args.output_dir / args.output_name
    save_json(output_data, output_path)

    print()
    print(f"Saved result to {output_path}")

    if args.plot:
        if len(results) < 2:
            print("Skipping plots: at least two workspaces are needed.")
        else:
            make_plots(
                results=results,
                plot_dir=args.plot_dir,
                wall_time_plot_name=args.plot_name,
            )


if __name__ == "__main__":
    main()
