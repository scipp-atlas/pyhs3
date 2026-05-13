# ruff: noqa: T201
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
from matplotlib import pyplot as plt


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
        delta[finite] = 2.0 * (nll[finite] - np.min(nll[finite]))
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


def plot(
    bundle: dict,
    ref: tuple[np.ndarray, np.ndarray] | None,
    scans: list[dict],
    output_pdf: Path,
    output_png: Path,
) -> dict:
    """Build 2x2 figure and return per-scan summary dict."""
    tab10 = mpl.colormaps["tab10"]
    markers = ["o", "s", "^", "D"]

    # Assign colour by method, marker by tolerance rank
    all_methods = list(dict.fromkeys(s["method"] for s in scans))
    all_tols = sorted({s["tol"] for s in scans})

    def _style(scan: dict) -> tuple:
        mi = all_methods.index(scan["method"])
        ti = all_tols.index(scan["tol"])
        return tab10(mi), markers[ti % len(markers)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_nll, ax_time, ax_nit, ax_mem = axes.flat

    # ------------------------------------------------------------------ (0,0) ΔNLL
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
            delta[finite] = 2.0 * (nll[finite] - np.nanmin(nll[finite]))
        ax_nll.plot(
            poi[finite],
            delta[finite],
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.2,
            label=scan["label"],
        )

    ax_nll.set_ylabel(r"-2$\Delta$NLL")
    ax_nll.set_title("Profile likelihood")
    ax_nll.grid(True, alpha=0.25)
    ax_nll.legend(fontsize=6, ncol=2)

    # --------------------------------- (0,1) wall + CPU time, (1,0) nit, (1,1) ΔRSS
    tick_labels = [s["label"] for s in scans]
    x = np.arange(len(scans))

    def _extract(scan: dict, key: str) -> list[float]:
        return [
            p[key]
            for p in scan.get("points", [])
            if isinstance(p.get(key), (int, float))
        ]

    # (0,1) timing boxplot
    wall_data = [_extract(s, "wall_s") for s in scans]
    cpu_data = [_extract(s, "cpu_s") for s in scans]
    bp_wall = ax_time.boxplot(
        wall_data,
        positions=x,
        widths=0.35,
        patch_artist=True,
        manage_ticks=False,
    )
    for patch, scan in zip(bp_wall["boxes"], scans, strict=False):
        color, _ = _style(scan)
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    bp_cpu = ax_time.boxplot(
        cpu_data,
        positions=x,
        widths=0.2,
        patch_artist=True,
        manage_ticks=False,
    )
    for patch, scan in zip(bp_cpu["boxes"], scans, strict=False):
        color, _ = _style(scan)
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=6)
    ax_time.set_ylabel("seconds")
    ax_time.set_title("Per-fit time (solid=wall, translucent=cpu)")
    ax_time.grid(True, axis="y", alpha=0.25)

    # (1,0) iterations boxplot
    nit_data = [_extract(s, "nit") for s in scans]
    # filter out -1 placeholders (migrad/methods without nit)
    nit_data = [[v for v in d if v >= 0] for d in nit_data]
    bp_nit = ax_nit.boxplot(
        nit_data,
        positions=x,
        widths=0.35,
        patch_artist=True,
        manage_ticks=False,
    )
    for patch, scan in zip(bp_nit["boxes"], scans, strict=False):
        color, _ = _style(scan)
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax_nit.set_xticks(x)
    ax_nit.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=6)
    ax_nit.set_ylabel("iterations")
    ax_nit.set_title("Per-fit iteration count")
    ax_nit.grid(True, axis="y", alpha=0.25)

    # (1,1) peak-RSS-delta boxplot
    def _rss_delta(scan: dict) -> list[float]:
        return [
            p["rss_peak_mb"] - p["rss_before_mb"]
            for p in scan.get("points", [])
            if isinstance(p.get("rss_peak_mb"), (int, float))
            and isinstance(p.get("rss_before_mb"), (int, float))
        ]

    mem_data = [_rss_delta(s) for s in scans]
    bp_mem = ax_mem.boxplot(
        mem_data,
        positions=x,
        widths=0.35,
        patch_artist=True,
        manage_ticks=False,
    )
    for patch, scan in zip(bp_mem["boxes"], scans, strict=False):
        color, _ = _style(scan)
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=6)
    ax_mem.set_ylabel("MB")
    ax_mem.set_title("Per-fit peak RSS delta")
    ax_mem.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"Minimizer benchmark — {bundle.get('workspace', '')}  POI={bundle.get('poi', '')}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    # ---- summary dict ----
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
                delta_arr[finite] = 2.0 * (nll_arr[finite] - np.nanmin(nll_arr[finite]))
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

    summary = plot(bundle, ref, scans, args.output_pdf, args.output_png)

    if args.output_json:
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
