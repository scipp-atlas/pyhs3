# ruff: noqa: T201
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
from aquarel import load_theme

mpl.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

# Layout: change these two lines to reshape the figure while keeping
# each cell at a fixed _CELL_W x _CELL_H inches.
_NROWS, _NCOLS = 3, 3
_CELL_W, _CELL_H = 5, 4  # inches per cell


def load_bundle(path: Path) -> dict:
    """Load benchmark bundle produced by exploring_minimization.py."""
    return json.loads(path.read_text())


def load_reference(path: Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Load optional reference scan JSON as (poi, delta_2nll) arrays.

    Expected format::

        {"points_scan_order": [{"poi": 0.8, "nll": 123.4}, ...]}
    """
    if path is None:
        return None
    data = json.loads(path.read_text())
    points = sorted(data["points_scan_order"], key=lambda p: p["poi"])
    poi = np.array([p["poi"] for p in points], dtype=float)
    nll = np.array([p["nll"] for p in points], dtype=float)
    finite = np.isfinite(nll)
    delta = np.full_like(nll, np.nan)
    if np.any(finite):
        delta[finite] = nll[finite] - np.min(nll[finite])
    return poi, delta


def _interpolate_reference(
    ref_poi: np.ndarray,
    ref_delta: np.ndarray,
    scan_poi: np.ndarray,
) -> np.ndarray:
    """Match scan POI values to reference by exact float equality (12 decimals)."""
    by_poi = {
        round(float(p), 12): float(d) for p, d in zip(ref_poi, ref_delta, strict=False)
    }
    return np.array([by_poi.get(round(float(p), 12), np.nan) for p in scan_poi])


def apply_filters(
    bundle: dict,
    methods: list[str] | None,
    tols: list[float] | None,
) -> list[dict]:
    """Return scans matching the optional method and tolerance filters."""
    scans = bundle.get("scans", [])
    if methods:
        scans = [s for s in scans if s["method"] in methods]
    if tols:
        scans = [s for s in scans if any(abs(s["tol"] - t) < 1e-15 for t in tols)]
    return scans


def _boxplot_panel(
    ax: plt.Axes,
    data: list[list[float]],
    x: np.ndarray,
    scans: list[dict],
    style_fn,
    ylabel: str,
    title: str,
    tick_labels: list[str],
) -> None:
    """Draw a single colour-coded boxplot panel."""
    bp = ax.boxplot(
        data,
        positions=x,
        widths=0.5,
        patch_artist=True,
        manage_ticks=False,
        showfliers=True,
        flierprops={"marker": "x", "markersize": 3, "alpha": 0.5},
    )
    for patch, scan in zip(bp["boxes"], scans, strict=False):
        color, _ = style_fn(scan)
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)


def plot(
    bundle: dict,
    ref: tuple[np.ndarray, np.ndarray] | None,
    scans: list[dict],
    output_pdf: Path,
    output_png: Path,
) -> dict:
    """Build 3x3 figure and return per-scan summary dict."""

    colormap = mpl.colors.ListedColormap(
        [
            "#d73027",
            "#fc8d59",
            "#fee090",
            "#e0f3f8",
            "#91bfdb",
            "#4575b4",
        ]
    )

    markers = ["o", "s", "^", "D"]

    # Colour by method, marker by tolerance rank
    all_methods = list(dict.fromkeys(s["method"] for s in scans))
    all_tols = sorted({s["tol"] for s in scans})

    def _style(scan: dict) -> tuple:
        mi = all_methods.index(scan["method"])
        ti = all_tols.index(scan["tol"])
        return colormap(mi), markers[ti % len(markers)]

    fig, axes = plt.subplots(
        _NROWS,
        _NCOLS,
        figsize=(_NCOLS * _CELL_W, _NROWS * _CELL_H),
        constrained_layout=True,
    )
    (
        ax_nll,
        ax_nll_raw,
        ax_nll_global,
        ax_time,
        ax_nfev,
        ax_nit,
        ax_mem,
        ax_cpu,
        ax_unused,
    ) = axes.flat
    ax_unused.set_visible(False)

    # ------------------------------------------------------------------ (0,0) ΔNLL
    n_poi = len(bundle.get("poi_grid", []))
    if n_poi == 0 and scans:
        n_poi = len(scans[0].get("points", []))

    if ref is not None:
        ref_poi, ref_delta = ref
        ax_nll.plot(
            ref_poi,
            ref_delta,
            color="0.4",
            linewidth=1.5,
            linestyle="--",
            zorder=0,
            label="reference",
        )

    for scan in scans:
        color, marker = _style(scan)
        points = scan.get("points", [])
        poi = np.array([p["poi"] for p in points], dtype=float)
        nll = np.array([p["nll"] for p in points], dtype=float)
        finite = np.isfinite(nll)
        delta = np.full_like(nll, np.nan)
        if np.any(finite):
            delta[finite] = nll[finite] - np.nanmin(nll[finite])
        ax_nll.plot(
            poi[finite],
            delta[finite],
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.2,
            label=scan["label"],
        )

    ax_nll.set_xlabel(bundle.get("poi", "POI"))
    ax_nll.set_ylabel(r"$\Delta(-2\ln L)$")
    ax_nll.set_title(f"Profile likelihood — per-scan min  (N={n_poi})", fontsize=8)
    ax_nll.grid(True, alpha=0.25)
    ax_nll.legend(fontsize=5, ncol=2)

    # ------------------------------------------------------------------ (0,1) raw NLL
    for scan in scans:
        color, marker = _style(scan)
        points = scan.get("points", [])
        poi = np.array([p["poi"] for p in points], dtype=float)
        nll = np.array([p["nll"] for p in points], dtype=float)
        finite = np.isfinite(nll)
        ax_nll_raw.plot(
            poi[finite],
            nll[finite],
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.2,
            label=scan["label"],
        )

    ax_nll_raw.set_xlabel(bundle.get("poi", "POI"))
    ax_nll_raw.set_ylabel(r"$-2\ln L$")
    ax_nll_raw.set_title(f"Raw NLL  (N={n_poi})", fontsize=8)
    ax_nll_raw.grid(True, alpha=0.25)
    ax_nll_raw.legend(fontsize=5, ncol=2)

    # ------------------------------------------------------------------ (0,2) global-min ΔNLL
    all_finite_nlls = [
        p["nll"]
        for scan in scans
        for p in scan.get("points", [])
        if np.isfinite(p.get("nll", np.nan))
    ]
    global_nll_min = min(all_finite_nlls) if all_finite_nlls else 0.0

    if ref is not None:
        ref_poi, ref_delta = ref
        ax_nll_global.plot(
            ref_poi,
            ref_delta,
            color="0.4",
            linewidth=1.5,
            linestyle="--",
            zorder=0,
            label="reference (own min)",
        )

    for scan in scans:
        color, marker = _style(scan)
        points = scan.get("points", [])
        poi = np.array([p["poi"] for p in points], dtype=float)
        nll = np.array([p["nll"] for p in points], dtype=float)
        finite = np.isfinite(nll)
        delta_global = np.full_like(nll, np.nan)
        if np.any(finite):
            delta_global[finite] = nll[finite] - global_nll_min
        ax_nll_global.plot(
            poi[finite],
            delta_global[finite],
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.2,
            label=scan["label"],
        )

    ax_nll_global.set_xlabel(bundle.get("poi", "POI"))
    ax_nll_global.set_ylabel(r"$\Delta(-2\ln L)$")
    ax_nll_global.set_title(f"Profile likelihood — global min  (N={n_poi})", fontsize=8)
    ax_nll_global.grid(True, alpha=0.25)
    ax_nll_global.legend(fontsize=5, ncol=2)

    # ------------------------------------------------------------------ boxplots
    tick_labels = [s["label"] for s in scans]
    x = np.arange(len(scans))

    def _extract(scan: dict, key: str) -> list[float]:
        return [
            p[key]
            for p in scan.get("points", [])
            if isinstance(p.get(key), (int, float))
        ]

    # (0,1) timing: overlaid wall (solid) and cpu (translucent, narrower)
    wall_data = [_extract(s, "wall_s") for s in scans]
    cpu_data = [_extract(s, "cpu_s") for s in scans]
    bp_wall = ax_time.boxplot(
        wall_data,
        positions=x,
        widths=0.5,
        patch_artist=True,
        manage_ticks=False,
        showfliers=True,
        flierprops={"marker": "x", "markersize": 3, "alpha": 0.5},
    )
    for patch, scan in zip(bp_wall["boxes"], scans, strict=False):
        color, _ = _style(scan)
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    bp_cpu = ax_time.boxplot(
        cpu_data,
        positions=x,
        widths=0.25,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
    )
    for patch, scan in zip(bp_cpu["boxes"], scans, strict=False):
        color, _ = _style(scan)
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=5)
    ax_time.set_ylabel("seconds")
    ax_time.set_title("Per-fit time  (wide=wall, narrow=cpu)", fontsize=8)
    ax_time.grid(True, axis="y", alpha=0.25)

    # (0,2) nfev
    nfev_data = [[v for v in _extract(s, "nfev") if v >= 0] for s in scans]
    _boxplot_panel(
        ax_nfev,
        nfev_data,
        x,
        scans,
        _style,
        "function evaluations",
        "Per-fit nfev",
        tick_labels,
    )

    # (1,0) nit — filter out -1 placeholders (methods that don't report nit)
    nit_data = [[v for v in _extract(s, "nit") if v >= 0] for s in scans]
    _boxplot_panel(
        ax_nit,
        nit_data,
        x,
        scans,
        _style,
        "iterations",
        "Per-fit nit",
        tick_labels,
    )

    # (1,1) peak-RSS delta
    def _rss_delta(scan: dict) -> list[float]:
        return [
            p["rss_peak_mb"] - p["rss_before_mb"]
            for p in scan.get("points", [])
            if isinstance(p.get("rss_peak_mb"), (int, float))
            and isinstance(p.get("rss_before_mb"), (int, float))
        ]

    _boxplot_panel(
        ax_mem,
        [_rss_delta(s) for s in scans],
        x,
        scans,
        _style,
        "MB",
        "Per-fit peak RSS delta",
        tick_labels,
    )

    # (1,2) cpu_percent_mean
    cpu_pct_data = [_extract(s, "cpu_percent_mean") for s in scans]
    _boxplot_panel(
        ax_cpu,
        cpu_pct_data,
        x,
        scans,
        _style,
        "% CPU",
        "Per-fit mean CPU %",
        tick_labels,
    )

    # ------------------------------------------------------------------ titles + annotation
    poi_key = bundle.get("poi", "")
    workspace = bundle.get("workspace", "")
    n_poi_str = f"N={n_poi} POI"
    fig.suptitle(
        f"Minimizer benchmark — {workspace}  POI={poi_key}  {n_poi_str}",
        fontsize=10,
    )

    machine = bundle.get("machine", {})
    if machine:
        proc = machine.get("processor") or machine.get("machine", "")
        phys = machine.get("cpu_count_physical", "?")
        logi = machine.get("cpu_count_logical", "?")
        ram = machine.get("total_ram_gb", "?")
        os_str = f"{machine.get('os', '')} {machine.get('os_release', '')}".strip()
        freq = machine.get("cpu_freq_mhz")
        freq_str = f"  {freq:.0f} MHz" if freq else ""
        machine_line = f"Machine: {proc}{freq_str} | {phys}P/{logi}L cores | {ram} GB RAM | {os_str}"
        fig.text(0.5, 0.0, machine_line, ha="center", fontsize=6.5, color="0.4")

    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------ summary dict
    summary: dict[str, dict] = {}
    for scan in scans:
        points = scan.get("points", [])
        n_success = sum(1 for p in points if p.get("success"))
        walls = [
            p["wall_s"] for p in points if isinstance(p.get("wall_s"), (int, float))
        ]
        cpus = [p["cpu_s"] for p in points if isinstance(p.get("cpu_s"), (int, float))]
        nits = [
            p["nit"] for p in points if isinstance(p.get("nit"), int) and p["nit"] >= 0
        ]
        nfevs = [
            p["nfev"]
            for p in points
            if isinstance(p.get("nfev"), int) and p["nfev"] >= 0
        ]
        rss_deltas = [
            p["rss_peak_mb"] - p["rss_before_mb"]
            for p in points
            if isinstance(p.get("rss_peak_mb"), (int, float))
            and isinstance(p.get("rss_before_mb"), (int, float))
        ]
        cpu_pcts = [
            p["cpu_percent_mean"]
            for p in points
            if isinstance(p.get("cpu_percent_mean"), (int, float))
        ]

        entry: dict = {
            "n_points": len(points),
            "n_success": n_success,
            "mean_wall_s": float(np.mean(walls)) if walls else None,
            "mean_cpu_s": float(np.mean(cpus)) if cpus else None,
            "mean_nit": float(np.mean(nits)) if nits else None,
            "mean_nfev": float(np.mean(nfevs)) if nfevs else None,
            "mean_rss_delta_mb": float(np.mean(rss_deltas)) if rss_deltas else None,
            "mean_cpu_percent": float(np.mean(cpu_pcts)) if cpu_pcts else None,
        }

        if ref is not None:
            ref_poi, ref_delta = ref
            poi_arr = np.array([p["poi"] for p in points], dtype=float)
            nll_arr = np.array([p["nll"] for p in points], dtype=float)
            finite = np.isfinite(nll_arr)
            delta_arr = np.full_like(nll_arr, np.nan)
            if np.any(finite):
                delta_arr[finite] = nll_arr[finite] - np.nanmin(nll_arr[finite])
            ref_at_scan = _interpolate_reference(ref_poi, ref_delta, poi_arr)
            comparable = np.isfinite(delta_arr) & np.isfinite(ref_at_scan)
            if np.any(comparable):
                diff = np.abs(delta_arr[comparable] - ref_at_scan[comparable])
                entry["max_abs_diff_vs_ref"] = float(np.nanmax(diff))
            else:
                entry["max_abs_diff_vs_ref"] = None

        summary[scan["label"]] = entry

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from exploring_minimization.py."
    )
    parser.add_argument(
        "--bundle", type=Path, required=True, help="Benchmark JSON bundle"
    )
    parser.add_argument(
        "--reference-json", type=Path, help="Optional reference scan JSON"
    )
    parser.add_argument(
        "--filter-method",
        help="Comma-separated method names to include (default: all)",
    )
    parser.add_argument(
        "--filter-tol",
        help="Comma-separated tolerance floats to include (default: all)",
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    ref = load_reference(args.reference_json)

    methods = (
        [m.strip() for m in args.filter_method.split(",")]
        if args.filter_method
        else None
    )
    tols = [float(t) for t in args.filter_tol.split(",")] if args.filter_tol else None
    scans = apply_filters(bundle, methods, tols)

    if not scans:
        msg = "No scans match the given filters."
        raise SystemExit(msg)

    with load_theme("scientific").set_overrides(
        {"xtick.major.size": 4, "xtick.minor.visible": False}
    ):
        summary = plot(bundle, ref, scans, args.output_pdf, args.output_png)

    if args.output_json:
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
