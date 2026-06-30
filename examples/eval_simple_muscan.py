#!/usr/bin/env python3
# ruff: noqa: T201
"""Evaluate the pyhs3 NLL at each scan point in quickFit/muscan.json.

Loads simple_workspace.json, builds one Model per per-channel likelihood
(L_ch0, L_ch1, L_ch2), and sums the -log(L) contributions.  Prints a
comparison table against the quickFit NLL values from muscan.json.

The ``diff = pyhs3_nll - qf_nll`` column is expected to be a constant offset
across the scan; how far it deviates from constant (the max absolute residual
about its mean) measures how well pyhs3 reproduces quickFit for a workspace.

Workspace names are fully-specified by workspace-scripts/workflow.sh: every
make_workspace.py option is spelled out in the stem, e.g.
``3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x``. Each scan is
``scans/<stem>_muscan.json`` and each workspace is ``workspaces/<stem>.json``.

Usage:
    # single workspace/scan (detailed table; defaults to the 3ch base variant)
    python eval_simple_muscan.py
    python eval_simple_muscan.py \\
        --workspace ../workspace-scripts/workspaces/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \\
        --scan      ../workspace-scripts/scans/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x_muscan.json

    # compare several workspaces, each with its own scan (run from the pyhs3
    # repo root, so ../workspace-scripts resolves to the sibling checkout):
    pixi run python examples/eval_simple_muscan.py \
        --pair ../workspace-scripts/workspaces/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json ../workspace-scripts/scans/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x_muscan.json \
        --pair ../workspace-scripts/workspaces/3ch_bkgGenExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json ../workspace-scripts/scans/3ch_bkgGenExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x_muscan.json \
        --pair ../workspace-scripts/workspaces/3ch_bkgRooExp_sigGauss_shapeFloat_npOff_constrGauss_yield1x.json ../workspace-scripts/scans/3ch_bkgRooExp_sigGauss_shapeFloat_npOff_constrGauss_yield1x_muscan.json \
        --pair ../workspace-scripts/workspaces/3ch_bkgGenExp_sigGauss_shapeFixed_npOn_constrGauss_yield1x.json ../workspace-scripts/scans/3ch_bkgGenExp_sigGauss_shapeFixed_npOn_constrGauss_yield1x_muscan.json \
        --pair ../workspace-scripts/workspaces/3ch_bkgGenExp_sigGauss_shapeFloat_npOff_constrGauss_yield1x.json ../workspace-scripts/scans/3ch_bkgGenExp_sigGauss_shapeFloat_npOff_constrGauss_yield1x_muscan.json \
        --pair ../workspace-scripts/workspaces/3ch_bkgGenPoly_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json ../workspace-scripts/scans/3ch_bkgGenPoly_sigGauss_shapeFloat_npOn_constrGauss_yield1x_muscan.json \
        --pair ../workspace-scripts/workspaces/3ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json ../workspace-scripts/scans/3ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield1x_muscan.json \
        --pair ../workspace-scripts/workspaces/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrPoisson_yield1x.json ../workspace-scripts/scans/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrPoisson_yield1x_muscan.json \
        --pair ../workspace-scripts/workspaces/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrNone_yield1x.json ../workspace-scripts/scans/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrNone_yield1x_muscan.json \
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pytensor
from pytensor.graph.traversal import explicit_graph_inputs

from pyhs3 import Workspace
from pyhs3.model import Model

from plot_residuals import FIELDS

_HERE = Path(__file__).resolve().parent
# workspace-scripts is a sibling of the pyhs3 repo (git-projects/workspace-scripts),
# so go up two levels from examples/ (examples -> pyhs3 -> git-projects).
_WS_SCRIPTS = _HERE.parent.parent / "workspace-scripts"
_DEFAULT_STEM = "3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x"
_DEFAULT_WS = _WS_SCRIPTS / "workspaces" / f"{_DEFAULT_STEM}.json"
_DEFAULT_SCAN = _WS_SCRIPTS / "scans" / f"{_DEFAULT_STEM}_muscan.json"

# Sentinel for a bare ``--plot`` flag: the output path is derived from the
# workspace name in main() (where the workspace is known).
_PLOT_DEFAULT = object()


def _default_nll_plot_path(results: list[dict]) -> Path:
    """Default PDF path: ``<workspace stem>_nlls.pdf`` next to this script.

    Uses the first workspace's name when several are plotted together.
    """
    return _HERE / f"all_nll_comparisons.pdf"

def _default_resid_plot_path(results: list[dict]) -> Path:
    return _HERE / f"residual_comparison.pdf"


def build_channel_models(ws_path: Path) -> list[tuple[Model, dict]]:
    """Load workspace and compile one NLL function per per-channel likelihood."""
    print(f"Loading workspace from {ws_path} ...")
    with ws_path.open() as fh:
        ws = Workspace(**json.load(fh))

    channel_analyses = sorted(
        (a for a in ws.analyses if re.match(r"L_ch\d+$", a.name)),
        key=lambda a: a.name,
    )
    print(f"  Per-channel analyses: {[a.name for a in channel_analyses]}")

    channel_models: list[tuple[Model, dict]] = []
    for analysis in channel_analyses:
        model = ws.model(analysis, progress=False)
        nll_expr = -model.log_prob
        inputs_map = {
            v.name: v for v in explicit_graph_inputs([nll_expr]) if v.name is not None
        }
        input_names = list(inputs_map.keys())
        fn = pytensor.function(
            list(inputs_map.values()), nll_expr, on_unused_input="ignore"
        )
        print(
            f"  {analysis.name}: compiled, {len(input_names)} inputs: {sorted(input_names)}"
        )
        channel_models.append((model, {"fn": fn, "input_names": input_names}))

    return channel_models


def eval_nll(
    channel_models: list[tuple[Model, dict]], params: dict[str, float]
) -> float:
    """Sum -log(L) over all channels at *params*."""
    total = 0.0
    for model, compiled in channel_models:
        # data arrays (e.g. x) take priority over any scalar fallback in params
        source = {**params, **model.data}
        fn = compiled["fn"]
        names = compiled["input_names"]
        args = [
            np.asarray(source[n], dtype=np.float64) if n in source else np.float64(0.0)
            for n in names
        ]
        total += float(np.asarray(fn(*args)).item())
    return total


def run_scan(ws_path: Path, scan_path: Path, *, verbose: bool = True) -> dict:
    """Evaluate one workspace against one scan and summarize the diff.

    Returns a dict with the per-point diffs and the constant-offset statistics:
    ``mean_offset`` (the average of the diff column) and ``max_abs_resid``
    (the largest absolute deviation of any point from that mean -- i.e. how
    far the diff strays from being a perfect constant).
    """
    channel_models = build_channel_models(ws_path)

    # Collect nominal free params from all channels (shared params are consistent)
    nominal: dict[str, float] = {}
    for model, _ in channel_models:
        nominal.update(model.free_params)

    if verbose:
        print(f"\nLoading scan points from {scan_path} ...")
    with scan_path.open() as fh:
        scan = json.load(fh)

    qf_nll_min = scan["metadata"]["nll_min"]
    bkg_type = scan["metadata"].get("bkg_type", "")
    points = scan["scan_points"]
    if verbose:
        print(f"  {len(points)} scan points, quickFit NLL_min = {qf_nll_min:.6f}\n")
        header = f"{'mu_sig':>8}  {'qf_nll':>14}  {'pyhs3_nll':>14}  {'diff':>16}"
        print(header)
        print("-" * len(header))

    mus: list[float] = []
    qf_nlls: list[float] = []
    pyhs3_nlls: list[float] = []
    diffs: list[float] = []
    for pt_data in sorted(points, key=lambda p: p["mu_sig"]):
        mu = pt_data["mu_sig"]
        qf_nll = pt_data["nll"]

        params: dict[str, float] = dict(nominal)
        params["mu_sig"] = mu
        for name, info in pt_data["parameters"].items():
            val = info["value"]
            # muscan.json uses ROOT's negative-tau convention for exponential_dist;
            # HS3/pyhs3 uses positive c — only flip for exponential backgrounds
            if bkg_type == "exponential" and name.startswith("tau_"):
                val = -val
            params[name] = val

        pyhs3_nll = eval_nll(channel_models, params)
        diff = pyhs3_nll - qf_nll
        mus.append(mu)
        qf_nlls.append(qf_nll)
        pyhs3_nlls.append(pyhs3_nll)
        diffs.append(diff)

        if verbose:
            print(f"{mu:>8.3f}  {qf_nll:>14.6f}  {pyhs3_nll:>14.6f}  {diff:>10.10f}")

    diff_arr = np.asarray(diffs, dtype=np.float64)
    mean_offset = float(diff_arr.mean())
    resid = diff_arr - mean_offset
    max_abs_resid = float(np.abs(resid).max())

    if verbose:
        pyhs3_nll_min = min(pyhs3_nlls)
        print(f"\npyhs3 NLL_min    = {pyhs3_nll_min:.6f}")
        print(f"quickFit NLL_min = {qf_nll_min:.6f}")
        print(f"Difference       = {pyhs3_nll_min - qf_nll_min:.6f}")
        print(f"\nmean offset      = {mean_offset:.6f}")
        print(
            f"max |residual|   = {max_abs_resid:.3e}  (deviation of diff from constant)"
        )

    return {
        "workspace": ws_path,
        "scan": scan_path,
        "mus": mus,
        "diffs": diffs,
        "mean_offset": mean_offset,
        "max_abs_resid": max_abs_resid,
        "qf_nlls": qf_nlls,
        "pyhs3_nlls": pyhs3_nlls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workspace", type=Path, default=_DEFAULT_WS)
    parser.add_argument("--scan", type=Path, default=_DEFAULT_SCAN)
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("WORKSPACE", "SCAN"),
        type=Path,
        help="A workspace/scan pair to evaluate. Repeat to compare several; "
        "when given, --workspace/--scan are ignored and a ranked "
        "constant-offset summary is printed.",
    )
    parser.add_argument(
        "--plot-nll",
        nargs="?",
        const=_PLOT_DEFAULT,
        default=None,
        type=Path,
        metavar="PATH",
        help="Plot the pyhs3 vs quickFit NLL curves into a single PDF, one "
        "panel per workspace. Pass a path to override the default "
        "(<workspace name>_nlls.pdf next to this script).",
    )
    parser.add_argument(
        "--plot-resid",
        nargs="?",
        const=_PLOT_DEFAULT,
        default=None,
        type=Path,
        metavar="PATH",
        help="Plot curves of the residuals and mean offsets of all evaluated workspaces",
    )
    parser.add_argument(
        "--plot-resid-field",
        nargs="+",
        default=["channels"],
        choices=FIELDS,
        metavar="FIELD",
        help= "specify the field for x labels on data"
    )
    args = parser.parse_args()

    if not args.pair:
        result = run_scan(args.workspace, args.scan, verbose=True)
        if args.plot_nll is not None:
            from plot_muscan_nll import plot_nll_curves  # noqa: PLC0415

            out = (
                _default_nll_plot_path([result])
                if args.plot_nll is _PLOT_DEFAULT
                else args.plot_nll
            )
            plot_nll_curves([result], out)
        if args.plot_resid is not None:
            from plot_residuals.py import plot_residual_and_offset

            out = (
                _default_resid_plot_path([result])
                if args.plot_resid is _PLOT_DEFAULT
                else args.plot_resid
            )
        return

    results = []
    for ws_path, scan_path in args.pair:
        print("=" * 72)
        print(f"WORKSPACE: {ws_path}")
        print(f"SCAN:      {scan_path}")
        print("=" * 72)
        results.append(run_scan(ws_path, scan_path, verbose=True))

    if args.plot_nll is not None:
        from plot_muscan_nll import plot_nll_curves  # noqa: PLC0415

        out = _default_nll_plot_path(results) if args.plot_nll is _PLOT_DEFAULT else args.plot_nll
        plot_nll_curves(results, out)

    if args.plot_resid is not None:
        from plot_residuals import plot_residual_and_offset 

        out = _default_resid_plot_path(results) if args.plot_resid is _PLOT_DEFAULT else args.plot_resid
        plot_residual_and_offset(results, out, label_field=args.plot_resid_field)

    # Rank by how flat the diff is: smaller max |residual| == closer to constant.
    results.sort(key=lambda r: r["max_abs_resid"])

    name_width = max(len("workspace"), *(len(r["workspace"].name) for r in results))
    off_width, res_width = 14, 11

    header = f"{'workspace':<{name_width}}  {'mean offset':>{off_width}}  {'max |resid|':>{res_width}}"
    print("\n" + "=" * len(header))
    print("CONSTANT-OFFSET SUMMARY  (sorted: flattest diff first)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        name = r["workspace"].name
        print(f"{name:<{name_width}}  {r['mean_offset']:>{off_width}.3f}  {r['max_abs_resid']:>{res_width}.3e}")


if __name__ == "__main__":
    main()
