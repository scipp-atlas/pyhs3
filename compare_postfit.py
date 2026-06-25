#!/usr/bin/env python3
# ruff: noqa: T201
"""Compare post-fit nuisance-parameter values between a pyhs3 results file
and the corresponding quickFit log.

Usage
-----
    python compare_postfit.py --mu -0.3
    python compare_postfit.py --mu 1.0 --sort name
    python compare_postfit.py --mu 0.5 --threshold 0.05
    python compare_postfit.py --mu -0.3 --sort pull

By default the script looks for:
  • pyhs3 run dir:   runs/latest/  (set with --pyhs3-dir to pick an older run)
  • quickFit log:    ../quickFit/output__workspace_FINAL_ISOBUGFIX/log__mu_<MU>.txt
  • workspace:       ../workspace_FINAL_ISOBUGFIX/WS-bbyy-non-resonant-non-param-isofix_unbinnedFix.json

Within the pyhs3 run directory the script reads:
  • results_mu_<MU>.txt   — post-fit parameter values
  • initial_values.txt    — starting-point values (written by minimization_dihiggs.py)
  • scan_summary.json     — per-point NLL / convergence / iteration counts

If initial_values.txt is absent, the script falls back to loading nominal
parameter values from the workspace JSON (requires pyhs3 to be installed).
Override with --workspace or suppress with --no-workspace.

Both file paths can be overridden with --pyhs3-file / --quickfit-file.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import sys
from pathlib import Path

# Optional pyhs3 import — used only for workspace-based initial values.
try:
    import pyhs3 as _pyhs3

    _HAS_PYHS3 = True
except ImportError:
    _HAS_PYHS3 = False

# Default workspace path (relative to this script's repo root).
_WS_DEFAULT = (
    Path(__file__).resolve().parent.parent
    / "workspace_FINAL_ISOBUGFIX"
    / "WS-bbyy-non-resonant-non-param-isofix_unbinnedFix.json"
).resolve()

# Parameter-point sets to read from the workspace, in priority order
# (later sets overwrite earlier ones for the same parameter name).
_WS_PSET_NAMES = [
    "default_values",
    "nominalGlobs",
    "nominalNuis",
    "unconditionalGlobs_muhat",
    "unconditionalNuis_muhat",
    "POI_muhat",
]

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_name_value_file(path: Path) -> dict[str, float]:
    """Parse a 'NAME: VALUE' text file (results_mu_*.txt, initial_values.txt)."""
    params: dict[str, float] = {}
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            name, sep, val = line.partition(": ")
            if not sep:
                continue
            with contextlib.suppress(ValueError):
                params[name.strip()] = float(val.strip())
    return params


def parse_quickfit(path: Path) -> dict[str, tuple[float, float]]:
    """Parse 'log__mu_*.txt' format:
        PARAM_NAME\\t  = VALUE\\t +/-  ERR\\t(limited)

    Returns dict mapping name -> (value, error).
    Only lines from the Minuit summary block are read.
    """
    params: dict[str, tuple[float, float]] = {}
    in_results = False
    with path.open() as f:
        for line in f:
            if "Minuit2Minimizer" in line and "Valid minimum" in line:
                in_results = True
            if not in_results:
                continue
            m = re.match(
                r"^(\S+)\s*=\s*([-+]?\d[\d.eE+\-]*)\s+\+/-\s+([\d.eE+\-]+)",
                line,
            )
            if m:
                name, val, err = m.group(1), m.group(2), m.group(3)
                with contextlib.suppress(ValueError):
                    params[name] = (float(val), float(err))
    return params


def parse_quickfit_summary(path: Path) -> dict:
    """Extract FVAL, Edm, Nfcn and wall-clock time from a quickFit log."""
    summary: dict = {}
    with path.open() as f:
        in_results = False
        for line in f:
            if "Minuit2Minimizer" in line and "Valid minimum" in line:
                in_results = True
            if not in_results:
                continue
            for key, pattern in [
                ("fval", r"^FVAL\s*=\s*([\d.eE+\-]+)"),
                ("edm", r"^Edm\s*=\s*([\d.eE+\-]+)"),
                ("nfcn", r"^Nfcn\s*=\s*(\d+)"),
            ]:
                m = re.match(pattern, line)
                if m:
                    summary[key] = float(m.group(1))
            # Wall-clock time from e.g. "Stop iterating after 18.339 s"
            m = re.search(r"Stop iterating after\s+([\d.]+)\s+s", line)
            if m:
                summary["time_s"] = float(m.group(1))
    return summary


def load_workspace_init_vals(ws_path: Path) -> dict[str, float]:
    """Load nominal/best-fit parameter values from the workspace JSON.

    Reads only the parameter_points (fast — no model compilation).
    Returns {} if pyhs3 is not importable or the file cannot be parsed.
    Later parameter sets in _WS_PSET_NAMES overwrite earlier ones, matching
    the priority order used by minimization_dihiggs.py.
    """
    if not _HAS_PYHS3:
        return {}
    try:
        ws = _pyhs3.Workspace.load(ws_path)
        vals: dict[str, float] = {}
        for pset_name in _WS_PSET_NAMES:
            if pset_name in ws.parameter_points:
                for p in ws.parameter_points[pset_name]:
                    vals[p.name] = float(p.value)
        return vals
    except Exception as exc:
        print(f"  Warning: could not load workspace ({exc}); skipping initial values.")
        return {}


def load_scan_summary(run_dir: Path, mu: float) -> dict | None:
    """Load the entry for *mu* from scan_summary.json, or None if not found."""
    summary_path = run_dir / "scan_summary.json"
    if not summary_path.exists():
        return None
    try:
        entries = json.loads(summary_path.read_text())
        for entry in entries:
            if abs(entry.get("mu", float("nan")) - mu) < 1e-9:
                return entry
    except (json.JSONDecodeError, KeyError):
        pass
    return None


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _mu_label(mu: float) -> str:
    """Canonical float→string used in pyhs3 filenames (e.g. 0.0, -0.3, 1.5)."""
    s = f"{mu:.10g}"
    if "." not in s and "e" not in s:
        s += ".0"
    return s


def find_pyhs3_file(mu: float, directory: Path) -> Path | None:
    """Find results_mu_<MU>.txt, trying a few numeric representations."""
    for candidate in [
        directory / f"results_mu_{_mu_label(mu)}.txt",
        directory / f"results_mu_{mu}.txt",
        directory / f"results_mu_{mu:g}.txt",
    ]:
        if candidate.exists():
            return candidate
    # Fallback: scan directory numerically
    for p in sorted(directory.glob("results_mu_*.txt")):
        m = re.search(r"results_mu_(.+)\.txt$", p.name)
        if m:
            try:
                if abs(float(m.group(1)) - mu) < 1e-9:
                    return p
            except ValueError:
                pass
    return None


def find_quickfit_file(mu: float, directory: Path) -> Path | None:
    """Find log__mu_<MU>.txt matching mu numerically."""
    for p in sorted(directory.glob("log__mu_*.txt")):
        m = re.search(r"log__mu_(.+)\.txt$", p.name)
        if m:
            try:
                if abs(float(m.group(1)) - mu) < 1e-9:
                    return p
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _fmt(v: float | None, width: int = 12) -> str:
    if v is None:
        return f"{'—':>{width}}"
    return f"{v:{width}.6g}"


def print_fit_summary(
    mu: float,
    pyhs3_entry: dict | None,
    qf_summary: dict,
) -> None:
    """Print a one-block header with NLL / convergence / iteration info."""
    print(f"Fit summary for mu_HH = {mu}")
    print()

    # pyhs3 row
    if pyhs3_entry:
        conv = "✓ converged" if pyhs3_entry.get("converged") else "✗ NOT converged"
        nit = pyhs3_entry.get("nit", "?")
        nfev = pyhs3_entry.get("nfev", "?")
        nll = pyhs3_entry.get("nll")
        t = pyhs3_entry.get("time_s")
        nll_s = f"{nll:.6f}" if nll is not None else "?"
        t_s = f"{t:.1f}s" if t is not None else "?"
        print(
            f"  pyhs3    : -2ln(L) = {nll_s}  {conv}  "
            f"{nit} iters  {nfev} fn calls  {t_s}"
        )
    else:
        print("  pyhs3    : (no scan_summary.json found in run directory)")

    # quickFit row
    fval = qf_summary.get("fval")
    edm = qf_summary.get("edm")
    nfcn = qf_summary.get("nfcn")
    t_qf = qf_summary.get("time_s")
    fval_s = f"{fval:.6f}" if fval is not None else "?"
    edm_s = f"{edm:.2e}" if edm is not None else "?"
    nfcn_s = str(int(nfcn)) if nfcn is not None else "?"
    t_qf_s = f"{t_qf:.1f}s" if t_qf is not None else "?"
    print(f"  quickFit : FVAL = {fval_s}  Edm = {edm_s}  {nfcn_s} fn calls  {t_qf_s}")
    print()


def print_table(
    rows: list[tuple],
    *,
    sort_by: str = "diff",
    threshold: float | None = None,
    has_initial: bool = False,
) -> None:
    """Print the parameter comparison table.

    Each row is either:
      (name, pyhs3_val, qf_val, abs_diff, pull)                         has_initial=False
      (name, init_val, pyhs3_val, p3_init_delta, qf_val, abs_diff, pull) has_initial=True

    p3_init_delta = pyhs3_val - init_val  (signed; None if init_val is None)
    """
    diff_idx = 5 if has_initial else 3
    pull_idx = 6 if has_initial else 4

    # Sort
    if sort_by == "diff":
        rows = sorted(rows, key=lambda r: r[diff_idx], reverse=True)
    elif sort_by == "pull":
        rows = sorted(rows, key=lambda r: abs(r[pull_idx]), reverse=True)
    elif sort_by == "name":
        rows = sorted(rows, key=lambda r: r[0])

    # Filter
    if threshold is not None:
        rows = [r for r in rows if r[diff_idx] >= threshold]

    if not rows:
        print("  (no rows to display — try a smaller --threshold)")
        return

    name_w = max(max(len(r[0]) for r in rows), len("Parameter"))  # noqa: PLW3301

    if has_initial:
        header = (
            f"{'Parameter':<{name_w}}  {'initial':>12}  {'pyhs3':>12}  "
            f"{'Δ(p3-init)':>12}  {'quickFit':>12}  {'|Δ pyhs3-qf|':>14}  {'|Δ|/σ':>8}"  # noqa: RUF001
        )
    else:
        header = (
            f"{'Parameter':<{name_w}}  {'pyhs3':>12}  {'quickFit':>12}  "
            f"{'|Δ pyhs3-qf|':>14}  {'|Δ|/σ':>8}"  # noqa: RUF001
        )

    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for row in rows:
        name = row[0]
        if has_initial:
            iv, pv, p3d, qv, ad, pull = row[1], row[2], row[3], row[4], row[5], row[6]
            pull_s = f"{pull:8.4f}" if not math.isnan(pull) else f"{'NaN':>8}"
            print(
                f"{name:<{name_w}}  {_fmt(iv)}  {_fmt(pv)}  "
                f"{_fmt(p3d)}  {_fmt(qv)}  "
                f"{ad:>14.6f}  {pull_s}"
            )
        else:
            pv, qv, ad, pull = row[1], row[2], row[3], row[4]
            pull_s = f"{pull:8.4f}" if not math.isnan(pull) else f"{'NaN':>8}"
            print(f"{name:<{name_w}}  {_fmt(pv)}  {_fmt(qv)}  {ad:>14.6f}  {pull_s}")

    print(sep)
    print(f"  {len(rows)} parameters shown")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    here = Path.cwd()
    default_qf_dir = here.parent / "quickFit" / "output__workspace_FINAL_ISOBUGFIX"

    # Default pyhs3 directory: runs/latest if it exists, otherwise cwd.
    latest_link = here / "runs" / "latest"
    default_pyhs3_dir = latest_link.resolve() if latest_link.exists() else here

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mu", type=float, required=True, help="mu_HH value to compare (e.g. -0.3)"
    )
    parser.add_argument(
        "--pyhs3-file",
        type=Path,
        default=None,
        help="Path to pyhs3 results file (overrides auto-discovery)",
    )
    parser.add_argument(
        "--quickfit-file",
        type=Path,
        default=None,
        help="Path to quickFit log file (overrides auto-discovery)",
    )
    parser.add_argument(
        "--pyhs3-dir",
        type=Path,
        default=default_pyhs3_dir,
        help="Directory containing results_mu_*.txt, "
        "initial_values.txt, scan_summary.json "
        "(default: runs/latest if it exists, otherwise cwd)",
    )
    parser.add_argument(
        "--quickfit-dir",
        type=Path,
        default=default_qf_dir,
        help="Directory containing log__mu_*.txt files",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=_WS_DEFAULT if _WS_DEFAULT.exists() else None,
        help="Workspace JSON to load initial parameter values from "
        "when initial_values.txt is absent from the run directory "
        f"(default: {_WS_DEFAULT.name} if present)",
    )
    parser.add_argument(
        "--no-workspace",
        action="store_true",
        help="Skip workspace loading even if --workspace is set",
    )
    parser.add_argument(
        "--sort",
        choices=["diff", "pull", "name"],
        default="diff",
        help="Sort order: diff (default, largest |Δ| first), pull (|Δ|/σ), or name",  # noqa: RUF001
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Only show rows where |Δ pyhs3-qf| >= threshold",
    )
    parser.add_argument(
        "--only-common",
        action="store_true",
        help="Suppress lists of parameters found in only one source",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve files
    # ------------------------------------------------------------------
    pyhs3_file = args.pyhs3_file or find_pyhs3_file(args.mu, args.pyhs3_dir)
    qf_file = args.quickfit_file or find_quickfit_file(args.mu, args.quickfit_dir)

    if pyhs3_file is None:
        sys.exit(
            f"ERROR: no pyhs3 results file found for mu={args.mu} in {args.pyhs3_dir}"
        )
    if qf_file is None:
        sys.exit(
            f"ERROR: no quickFit log file found for mu={args.mu} in {args.quickfit_dir}"
        )

    print(f"pyhs3    : {pyhs3_file}")
    print(f"quickFit : {qf_file}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    pyhs3_vals = parse_name_value_file(pyhs3_file)
    qf_vals = parse_quickfit(qf_file)
    qf_summary = parse_quickfit_summary(qf_file)
    pyhs3_entry = load_scan_summary(args.pyhs3_dir, args.mu)

    # Initial values: prefer initial_values.txt from the run directory,
    # fall back to loading parameter_points from the workspace JSON.
    init_path = args.pyhs3_dir / "initial_values.txt"
    if init_path.exists():
        init_vals = parse_name_value_file(init_path)
        print(f"initial  : {init_path}  ({len(init_vals)} values)")
    elif not args.no_workspace and args.workspace and args.workspace.exists():
        if not _HAS_PYHS3:
            print(
                "  (pyhs3 not importable — skipping workspace initial values; "
                "run with a pyhs3-enabled interpreter or provide initial_values.txt)"
            )
            init_vals = {}
        else:
            print(f"initial  : {args.workspace}  (from workspace)")
            init_vals = load_workspace_init_vals(args.workspace)
    else:
        init_vals = {}
    print()

    has_initial = bool(init_vals)

    if not pyhs3_vals:
        sys.exit("ERROR: parsed 0 parameters from pyhs3 file — check format")
    if not qf_vals:
        sys.exit("ERROR: parsed 0 parameters from quickFit file — check format")

    print(
        f"Parsed {len(pyhs3_vals)} post-fit parameters from pyhs3, "
        f"{len(qf_vals)} from quickFit"
        + (f", {len(init_vals)} initial values" if has_initial else "")
        + "."
    )
    print()

    # ------------------------------------------------------------------
    # Fit summary header
    # ------------------------------------------------------------------
    print_fit_summary(args.mu, pyhs3_entry, qf_summary)

    # ------------------------------------------------------------------
    # Parameter comparison
    # ------------------------------------------------------------------
    only_pyhs3 = sorted(set(pyhs3_vals) - set(qf_vals))
    only_qf = sorted(set(qf_vals) - set(pyhs3_vals))
    common = sorted(set(pyhs3_vals) & set(qf_vals))

    if not args.only_common:
        if only_pyhs3:
            print(f"Parameters only in pyhs3 ({len(only_pyhs3)}):")
            for n in only_pyhs3:
                iv_s = f"  initial={init_vals[n]:.6g}" if n in init_vals else ""
                print(f"  {n}: {pyhs3_vals[n]:.6g}{iv_s}")
            print()
        if only_qf:
            print(f"Parameters only in quickFit ({len(only_qf)}):")
            for n in only_qf:
                v, e = qf_vals[n]
                print(f"  {n}: {v:.6g}  +/- {e:.6g}")
            print()

    # Build rows
    rows: list[tuple] = []
    for name in common:
        pv = pyhs3_vals[name]
        qv, qe = qf_vals[name]
        ad = abs(pv - qv)
        pull = ad / qe if qe > 0 else float("nan")
        if has_initial:
            iv = init_vals.get(name)
            p3d = (pv - iv) if iv is not None else None
            rows.append((name, iv, pv, p3d, qv, ad, pull))
        else:
            rows.append((name, pv, qv, ad, pull))

    print(
        f"Comparison for mu_HH = {args.mu}  "
        f"(sort: {args.sort}"
        + (f", threshold |Δ|≥{args.threshold}" if args.threshold else "")
        + "):"
    )
    print()
    print_table(
        rows, sort_by=args.sort, threshold=args.threshold, has_initial=has_initial
    )


if __name__ == "__main__":
    main()
