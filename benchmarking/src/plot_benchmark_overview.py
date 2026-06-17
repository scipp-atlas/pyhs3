import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


WORKSPACE_ORDER = [
    "simple_workspace_nonp",
    "simple_workspace",
    "simple_workspace_generic_nonp",
    "simple_workspace_generic",
]

WORKSPACE_LABELS = {
    "simple_workspace_nonp": "Simple\n(nonp)",
    "simple_workspace": "Simple",
    "simple_workspace_generic_nonp": "Generic\n(nonp)",
    "simple_workspace_generic": "Generic",
}


def load_results(path: Path) -> list[dict]:
    """
    Load benchmark results from a JSON file and return the list of results.
    """

    with path.open() as f:
        data = json.load(f)

    return data["results"]


def workspace_key(result: dict) -> str:
    """
    Extract the workspace key from a benchmark result dictionary, which is used for grouping results by workspace.
    """

    return result["workspace"].replace(".json", "")


def sort_workspaces(workspaces: set[str]) -> list[str]:
    """
    Sort workspaces based on a predefined order, with any additional workspaces sorted alphabetically at the end.
    """

    ordered = [
        workspace
        for workspace in WORKSPACE_ORDER
        if workspace in workspaces
    ]

    remaining = sorted(workspaces - set(ordered))

    return ordered + remaining


def collect_by_workspace(
    results_by_stage: dict[str, list[dict]],
    metric: str,
) -> tuple[list[str], dict[str, list[float]]]:
    """
    Collect metric values by workspace for each stage, ensuring a consistent order of workspaces across stages.
    """

    all_workspaces = {
        workspace_key(result)
        for results in results_by_stage.values()
        for result in results
    }

    workspaces = sort_workspaces(all_workspaces)

    values_by_stage = {}

    for stage, results in results_by_stage.items():
        by_workspace = {
            workspace_key(result): result
            for result in results
        }

        values_by_stage[stage] = [
            float(by_workspace.get(workspace, {}).get(metric, 0.0))
            for workspace in workspaces
        ]

    return workspaces, values_by_stage


def collect_total_errors(
    results_by_stage: dict[str, list[dict]],
    workspaces: list[str],
    error_key: str,
    scale: float,
) -> list[float]:
    """
    Collect total error values by workspace across all stages, 
    combining them in quadrature to get the overall error for each workspace.
    """

    variances = [0.0] * len(workspaces)

    for results in results_by_stage.values():
        by_workspace = {
            workspace_key(result): result
            for result in results
        }

        for index, workspace in enumerate(workspaces):
            error = float(by_workspace.get(workspace, {}).get(error_key, 0.0))
            variances[index] += (error * scale) ** 2

    return [math.sqrt(variance) for variance in variances]


def format_value(value: float, ylabel: str) -> str:
    """
    Format a value for display in the plot, with formatting based on the ylabel units.
    """

    if "[ms]" in ylabel:
        return f"{value:.1f}"
    if "[MB]" in ylabel:
        return f"{value:.2f}"
    return f"{value:.3g}"


def add_segment_labels(
    ax,
    bars,
    values: list[float],
    bottoms: list[float],
    totals: list[float],
    ylabel: str,
) -> None:
    """
    Add labels to each segment of the stacked bars, with dynamic positioning
    based on the segment size relative to the total.
    """

    max_total = max(totals) if totals else 1.0
    small_segment_threshold = 0.05 * max_total

    for bar, value, bottom in zip(bars, values, bottoms, strict=False):
        if value <= 0:
            continue

        if value < small_segment_threshold:
            y = bottom + value + max_total * 0.015
            va = "bottom"
        else:
            y = bottom + value / 2
            va = "center"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            format_value(value, ylabel),
            ha="center",
            va=va,
            fontsize=9,
        )


def add_total_labels(
    ax,
    totals: list[float],
    ylabel: str,
) -> None:
    """
    Add total labels above the stacked bars, with dynamic positioning based on the maximum total value.
    """

    ymax = max(totals) if totals else 1.0

    for index, total in enumerate(totals):
        ax.text(
            index,
            total + ymax * 0.02,
            format_value(total, ylabel),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def make_stacked_plot(
    results_by_stage: dict[str, list[dict]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    scale: float = 1.0,
    error_key: str | None = None,
) -> None:
    """
    Create a stacked bar plot for the given metric across different stages and workspaces, with optional error bars.
    """

    workspaces, values_by_stage = collect_by_workspace(
        results_by_stage=results_by_stage,
        metric=metric,
    )

    x = list(range(len(workspaces)))
    bottoms = [0.0] * len(workspaces)

    totals = [0.0] * len(workspaces)
    for values in values_by_stage.values():
        scaled_values = [value * scale for value in values]
        totals = [
            total + value
            for total, value in zip(totals, scaled_values, strict=False)
        ]

    fig, ax = plt.subplots(figsize=(12, 7))

    for stage, values in values_by_stage.items():
        scaled_values = [value * scale for value in values]

        bars = ax.bar(
            x,
            scaled_values,
            bottom=bottoms,
            label=stage,
        )

        add_segment_labels(
            ax=ax,
            bars=bars,
            values=scaled_values,
            bottoms=bottoms,
            totals=totals,
            ylabel=ylabel,
        )

        bottoms = [
            bottom + value
            for bottom, value in zip(bottoms, scaled_values, strict=False)
        ]

    if error_key is not None:
        total_errors = collect_total_errors(
            results_by_stage=results_by_stage,
            workspaces=workspaces,
            error_key=error_key,
            scale=scale,
        )

        ax.errorbar(
            x,
            bottoms,
            yerr=total_errors,
            fmt="none",
            capsize=5,
            color="black",
            linewidth=1,
        )

    add_total_labels(
        ax=ax,
        totals=bottoms,
        ylabel=ylabel,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)

    display_labels = [
        WORKSPACE_LABELS.get(workspace, workspace)
        for workspace in workspaces
    ]
    ax.set_xticklabels(display_labels, fontsize=11)

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ymax = max(bottoms) if bottoms else 1.0
    ax.set_ylim(0, ymax * 1.15)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create overview benchmark plots from benchmark JSON outputs."
    )

    parser.add_argument(
        "--workspace-loading-result",
        type=Path,
        required=True,
        help="Path to the workspace loading benchmark JSON result.",
    )
    parser.add_argument(
        "--model-creation-result",
        type=Path,
        required=True,
        help="Path to the model creation benchmark JSON result.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarking/plots/overview"),
        help="Directory where overview plots will be saved.",
    )
    parser.add_argument(
        "--memory-metric",
        default="peak_rss_delta_mb",
        choices=["peak_rss_delta_mb", "current_rss_delta_mb"],
        help="Memory metric to use for the overview memory plot.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    workspace_loading_results = load_results(args.workspace_loading_result)
    model_creation_results = load_results(args.model_creation_result)

    results_by_stage = {
        "Load": workspace_loading_results,
        "Create model": model_creation_results,
    }

    make_stacked_plot(
        results_by_stage=results_by_stage,
        metric="wall_time_seconds_mean",
        ylabel="Mean wall time [ms]",
        title="Benchmark wall time by stage",
        output_path=args.output_dir / "benchmark_wall_time_overview.png",
        scale=1000.0,
        error_key="wall_time_seconds_std",
    )

    make_stacked_plot(
        results_by_stage=results_by_stage,
        metric=args.memory_metric,
        ylabel="RSS delta [MB]",
        title="Benchmark RSS delta by stage",
        output_path=args.output_dir / "benchmark_rss_delta_overview.png",
    )


if __name__ == "__main__":
    main()
