from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from config import WORKSPACE_LABELS


WORKSPACE_ORDER = [
    "simple_workspace_nonp",
    "simple_workspace",
    "simple_workspace_generic_nonp",
    "simple_workspace_generic",
]


def load_results(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)

    return data["results"]


def workspace_key(result: dict) -> str:
    return result["workspace"].replace(".json", "")


def sort_workspaces(workspaces: set[str]) -> list[str]:
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


def format_value(value: float, ylabel: str) -> str:
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
    max_total = max(totals) if totals else 1.0
    min_visible_segment = 0.04 * max_total

    for bar, value, bottom in zip(bars, values, bottoms, strict=False):
        if value <= 0:
            continue

        if value < min_visible_segment:
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bottom + value / 2,
            format_value(value, ylabel),
            ha="center",
            va="center",
            fontsize=9,
        )


def add_total_labels(
    ax,
    totals: list[float],
    ylabel: str,
) -> None:
    ymax = max(totals) if totals else 1.0
    offset = ymax * 0.025

    for index, total in enumerate(totals):
        ax.text(
            index,
            total + offset,
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
) -> None:
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
    ax.set_ylim(0, ymax * 1.18)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Saved plot to {output_path}")


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
        "--log-prob-construction-result",
        type=Path,
        required=True,
        help="Path to the log_prob construction benchmark JSON result.",
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
    log_prob_construction_results = load_results(
        args.log_prob_construction_result
    )

    results_by_stage = {
        "Load": workspace_loading_results,
        "Create model": model_creation_results,
        "Log-prob construction": log_prob_construction_results,
    }

    make_stacked_plot(
        results_by_stage=results_by_stage,
        metric="wall_time_seconds_mean",
        ylabel="Mean wall time [ms]",
        title="Benchmark wall time by stage",
        output_path=args.output_dir / "benchmark_wall_time_overview.png",
        scale=1000.0,
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
