# ruff: noqa: T201
"""Evaluate the pyhs3 NLL at postfit parameter values taken from quickFit logs.

Instead of minimising, this script:
  1. Reads quickFit/output__workspace_FINAL_ISOBUGFIX/nlls.txt for the
     reference NLL values (multiplied by 2 to match pyhs3's -2*log(L) convention).
  2. Parses the matching quickFit log file to extract the postfit NP values.
  3. Loads the pre-compiled pyhs3 log_prob function from the cache.
  4. Evaluates the pyhs3 NLL at those parameter values.
  5. Reports pyhs3 NLL vs 2*quickFit NLL directly (same units).

The reported difference is written as

    diff = pyhs3_nll - qf_nll + N_total * ln(n_channels)

where the ``N_total * ln(n_channels)`` term corrects for the known
RooSimultaneous category-normalization offset between the two NLL definitions.
``N_total`` (sum of dataset event weights) and ``n_channels`` (number of
distributions in the likelihood) are read from the HS3 workspace JSON passed via
``--workspace``; without that flag the offset is 0 and ``diff`` is the raw
``pyhs3_nll - qf_nll`` (the previous behaviour).

Usage:
    # evaluate a single mu point
    python examples/eval_nll_at_quickfit.py --mu 1.0

    # specify the log file explicitly
    python examples/eval_nll_at_quickfit.py \\
        --log quickFit/output__workspace_FINAL_ISOBUGFIX/log__mu_1.txt

    # sweep all mu points that appear in nlls.txt
    python examples/eval_nll_at_quickfit.py --all

    # apply the category-normalization offset read from the workspace JSON
    python examples/eval_nll_at_quickfit.py --all \\
        --workspace path/to/workspace.json

    # use a non-default cache or log directory
    python examples/eval_nll_at_quickfit.py --mu 1.0 \\
        --cache-dir pyhs3/cache/WS-bbyy-non-resonant-non-param-isofix_unbinnedFix_clean \\
        --log-dir quickFit/output__workspace_FINAL_ISOBUGFIX
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# Likelihood whose channel count / event weights define the offset term.
_ANALYSIS = "CombinedPdf_combData"

# ---------------------------------------------------------------------------
# Default paths (relative to the repo root — parent of the pyhs3/ package dir)
# ---------------------------------------------------------------------------
_REPO_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)  # examples/ -> pyhs3/ -> repo root
_DEFAULT_LOG_DIR = _REPO_ROOT / "quickFit" / "output__workspace_FINAL_ISOBUGFIX"
_DEFAULT_WORKSPACE_PATH = Path("/home/mhance/pyhs3/pyhs3/data/WS-bbyy-non-resonant-non-param-isofix_unbinnedFix_withaux_nobounds.json")
_DEFAULT_CACHE_DIR = (
    _REPO_ROOT
    / "pyhs3"
    / "cache"
    / "WS-bbyy-non-resonant-non-param-isofix_unbinnedFix_clean"
)


# ---------------------------------------------------------------------------
# nlls.txt parsing
# ---------------------------------------------------------------------------


def parse_nlls_file(path: Path) -> dict[float, float]:
    """Parse quickFit's nlls.txt into a {mu: nll} dict.

    Expected line format:
        mu = 1                         \tnll = 2102.16832136193

    quickFit stores NLL values in RooFit's -log(L) convention; pyhs3 uses
    -2*log(L).  The values are multiplied by 2 here so that the returned
    dict is in the same units as pyhs3's NLL output.
    """
    result: dict[float, float] = {}
    line_re = re.compile(r"mu\s*=\s*([-\d.eE+]+)\s*\tnll\s*=\s*([-\d.eE+]+)")
    with path.open() as fh:
        for line in fh:
            m = line_re.search(line)
            if m:
                result[float(m.group(1))] = 2.0 * float(m.group(2))
    if not result:
        msg = f"No mu/nll pairs found in {path}"
        raise ValueError(msg)
    return result


def lookup_qf_nll(
    nlls: dict[float, float], mu: float, tol: float = 1e-6
) -> float | None:
    """Return the quickFit NLL for *mu*, or None if not found."""
    if mu in nlls:
        return nlls[mu]
    # float key lookup with tolerance
    for k, v in nlls.items():
        if abs(k - mu) <= tol:
            return v
    return None


# ---------------------------------------------------------------------------
# Workspace offset term
# ---------------------------------------------------------------------------


def workspace_event_counts(ws_json: dict) -> tuple[int, float, int]:
    """(n_channels, sum-of-weights, raw entry count) for te analysis likelihood.

    ``n_channels`` is the number of distributions in the ``_ANALYSIS``
    likelihood; ``sum-of-weights`` is the total event weight across its datasets
    (the ~1e-9 weights are RooFit ghost/padding entries); the raw entry count is
    also returned for reference.
    """
    likelihood = next(lk for lk in ws_json["likelihoods"] if lk["name"] == _ANALYSIS)
    data = {d["name"]: d for d in ws_json["data"]}
    n_entries = 0
    sum_w = 0.0
    for name in likelihood["data"]:
        d = data[name]
        n_entries += len(d["entries"])
        sum_w += sum(d["weights"])
    return len(likelihood["distributions"]), sum_w, n_entries


def compute_offset(workspace_path: Path) -> float:
    """N_total * ln(n_channels) from the HS3 workspace JSON at *workspace_path*."""
    with workspace_path.open() as fh:
        ws_json = json.load(fh)
    n_channels, sum_w, n_entries = workspace_event_counts(ws_json)
    offset = sum_w * np.log(n_channels)
    print(f"\nOffset from {workspace_path}:")
    print(f"  n_channels          = {n_channels}")
    print(f"  N_total (weights)   = {sum_w:.6f}  (raw entries: {n_entries})")
    print(f"  N_total * ln(C)     = {offset:.6f}")
    return float(offset)


# ---------------------------------------------------------------------------
# Log file helpers
# ---------------------------------------------------------------------------


def find_log_for_mu(mu: float, log_dir: Path) -> Path:
    """Return the log__mu_*.txt file closest to *mu* in *log_dir*."""
    candidate = log_dir / f"log__mu_{mu}.txt"
    if candidate.exists():
        return candidate

    logs = sorted(log_dir.glob("log__mu_*.txt"))
    if not logs:
        msg = f"No log__mu_*.txt files in {log_dir}"
        raise FileNotFoundError(msg)

    def _mu_of(p: Path) -> float:
        m = re.search(r"log__mu_([-\d.]+)\.txt", p.name)
        return float(m.group(1)) if m else float("inf")

    best = min(logs, key=lambda p: abs(_mu_of(p) - mu))
    best_mu = _mu_of(best)
    if abs(best_mu - mu) > 1e-6:
        print(
            f"  Warning: no log for mu={mu}; using closest mu={best_mu} ({best.name})"
        )
    return best


def parse_quickfit_log(log_path: Path) -> dict:
    """Extract postfit NP values and convergence status from a quickFit log.

    Parameter lines appear after the 'FVAL = ...' line and have the format:
        NAME\\t  = VALUE\\t +/-  ERROR\\t(limited)

    Returns:
        mu_HH     : float - from the filename
        converged : bool
        params    : dict[str, float] - postfit NP values
    """
    params: dict[str, float] = {}
    converged = False
    in_param_block = False

    param_re = re.compile(r"^(\S+)\t\s+=\s+([-\d.eE+]+)\t")
    fval_re = re.compile(r"^FVAL\s+=\s+[-\d.eE+]+")

    with log_path.open() as fh:
        for line in fh:
            if "Valid minimum - status = 0" in line:
                converged = True
            if fval_re.match(line):
                in_param_block = True
                continue
            if in_param_block:
                m = param_re.match(line)
                if m:
                    params[m.group(1)] = float(m.group(2))

    mu_match = re.search(r"log__mu_([-\d.]+)\.txt", log_path.name)
    mu_HH = float(mu_match.group(1)) if mu_match else None

    return {"mu_HH": mu_HH, "converged": converged, "params": params}


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------


def load_compiled_fn(cache_dir: Path):
    """Load (log_prob_fn, grad_fn, input_names) from cache_dir/log_prob_fn.pkl."""
    pkl_path = cache_dir / "log_prob_fn.pkl"
    if not pkl_path.exists():
        msg = f"Cache file not found: {pkl_path}"
        raise FileNotFoundError(msg)

    size_mb = pkl_path.stat().st_size // 1024 // 1024
    print(f"Loading compiled function from {pkl_path} ({size_mb} MB) ...")
    t0 = time.perf_counter()
    with pkl_path.open("rb") as fh:
        cached = pickle.load(fh)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    if not isinstance(cached, tuple):
        msg = f"Unexpected cache type: {type(cached)}"
        raise ValueError(msg)
    if (
        len(cached) == 2
        and isinstance(cached[0], str)
        and cached[0] in ("jax", "jax_export")
    ):
        msg = (
            "Cache contains a JAX sentinel — re-run minimization_dihiggs.py "
            "with --no-jax to produce a PyTensor compiled function."
        )
        raise ValueError(msg)
    if len(cached) == 3:
        log_prob_fn, grad_fn, input_names = cached
        print(f"  {len(input_names)} free inputs.")
        return log_prob_fn, grad_fn, input_names

    msg = f"Unrecognised cache tuple length: {len(cached)}"
    raise ValueError(msg)


def load_model(cache_dir: Path):
    """Load the cached pyhs3 Model from cache_dir/ws.pkl."""
    pkl_path = cache_dir / "ws.pkl"
    if not pkl_path.exists():
        msg = f"Model cache not found: {pkl_path}"
        raise FileNotFoundError(msg)
    size_mb = pkl_path.stat().st_size // 1024 // 1024
    print(f"Loading model from {pkl_path} ({size_mb} MB) ...")
    t0 = time.perf_counter()
    with pkl_path.open("rb") as fh:
        model = pickle.load(fh)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# NLL evaluation
# ---------------------------------------------------------------------------


def evaluate_nll(
    log_prob_fn,
    input_names: list[str],
    nominal_params: dict[str, float],
    qf_params: dict[str, float],
    mu_HH: float,
) -> tuple[float, dict]:
    """Evaluate the pyhs3 NLL at the quickFit postfit parameters.

    Parameters
    ----------
    log_prob_fn    : compiled PyTensor function
    input_names    : ordered list of free-parameter names the function expects
    nominal_params : fallback values for any input not in qf_params
                     (typically model.free_params)
    qf_params      : postfit NP values from the quickFit log
    mu_HH          : mu_HH value to pin for this scan point

    Returns
    -------
    (nll_value, diagnostics)
    """
    # Priority: quickFit postfit > pyhs3 nominal; mu_HH always pinned.
    source: dict[str, float] = {}
    source.update(nominal_params)
    source.update(qf_params)
    source["mu_HH"] = mu_HH

    args: list[np.ndarray] = []
    missing_from_qf: list[str] = []
    for name in input_names:
        if name in source:
            args.append(np.asarray(source[name], dtype=np.float64))
        else:
            args.append(np.asarray(0.0, dtype=np.float64))
            missing_from_qf.append(name)

    in_qf_not_pyhs3 = sorted(set(qf_params) - set(input_names) - {"mu_HH"})

    nll_val = float(log_prob_fn(*args)[0])

    return nll_val, {
        "n_inputs": len(input_names),
        "missing_from_qf": missing_from_qf,
        "in_qf_not_pyhs3": in_qf_not_pyhs3,
    }


# ---------------------------------------------------------------------------
# Single-point driver
# ---------------------------------------------------------------------------


def run_one(
    log_path: Path,
    log_prob_fn,
    input_names: list[str],
    nominal_params: dict[str, float],
    nlls: dict[float, float],
    offset: float = 0.0,
) -> dict:
    """Parse one log, evaluate pyhs3 NLL, compare to quickFit NLL from nlls.txt.

    The reported ``diff`` is ``pyhs3_nll - qf_nll + offset``, where *offset* is
    the ``N_total * ln(n_channels)`` category-normalization correction (0 when no
    workspace is supplied).  ``raw_diff`` keeps the uncorrected difference.
    """
    parsed = parse_quickfit_log(log_path)
    mu = parsed["mu_HH"]
    converged = parsed["converged"]
    qf_params = parsed["params"]

    qf_nll = lookup_qf_nll(nlls, mu)

    t0 = time.perf_counter()
    pyhs3_nll, diag = evaluate_nll(
        log_prob_fn, input_names, nominal_params, qf_params, mu
    )
    dt = time.perf_counter() - t0

    raw_diff = pyhs3_nll - qf_nll if qf_nll is not None else None
    diff = raw_diff + (2 * offset) if raw_diff is not None else None

    return {
        "mu_HH": mu,
        "qf_nll": qf_nll,
        "pyhs3_nll": pyhs3_nll,
        "raw_diff": raw_diff,
        "diff": diff,
        "converged": converged,
        "eval_time_s": dt,
        **diag,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results: list[dict], out_path: Path) -> None:
    """Plot Δ(-2 ln L) vs mu_HH for both pyhs3 and quickFit.

    Both series are delta-shifted so their minimum is 0, matching the
    standard profile-likelihood plot convention.
    """
    # Only include points where both NLLs are available.
    valid = [
        r for r in results if r["qf_nll"] is not None and r["pyhs3_nll"] is not None
    ]
    if not valid:
        print("  No valid points to plot.")
        return

    mu = np.array([r["mu_HH"] for r in valid])
    qf_nll = np.array([r["qf_nll"] for r in valid])
    pyhs3_nll = np.array([r["pyhs3_nll"] for r in valid])

    # Sort by mu so the scatter points are in a sensible left-to-right order.
    order = np.argsort(mu)
    mu, qf_nll, pyhs3_nll = mu[order], qf_nll[order], pyhs3_nll[order]

    qf_shifted = qf_nll - qf_nll.min()
    pyhs3_shifted = pyhs3_nll - pyhs3_nll.min()

    fig, ax = plt.subplots()
    ax.scatter(mu, qf_shifted, label="quickFit", marker="o")
    ax.scatter(mu, pyhs3_shifted, label="pyhs3", marker="x")
    # Set y-axis ceiling just above the highest data point.
    data_max = max(qf_shifted.max(), pyhs3_shifted.max())
    ax.set_ylim(bottom=0, top=data_max * 1.1)

    ax.set_xlabel(r"$\mu_{HH}$")
    ax.set_ylabel(r"$-2\,\Delta(\ln\Lambda)$")
    ax.set_title("pyhs3 run with quickFit postfit params")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"\nPlot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--mu",
        "-m",
        type=float,
        metavar="MU",
        help="mu_HH value; finds log__mu_<MU>.txt automatically.",
    )
    group.add_argument(
        "--log",
        "-l",
        type=Path,
        metavar="LOG.txt",
        help="Explicit path to a quickFit log file.",
    )
    group.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Evaluate all mu points found in nlls.txt.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=_DEFAULT_LOG_DIR,
        metavar="DIR",
        help=f"Directory containing quickFit log files and nlls.txt.\n(default: {_DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE_DIR,
        metavar="DIR",
        help=f"pyhs3 cache directory (ws.pkl + log_prob_fn.pkl).\n(default: {_DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip loading ws.pkl; use 0 as fallback for params missing from the qf log.\n"
        "Faster startup; safe only if the qf log covers all pyhs3 free parameters.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        metavar="WS.json",
        help="HS3 workspace JSON used to compute the offset term\n"
        "N_total * ln(n_channels) added to every diff.\n"
        "(default: no offset — diff is the raw pyhs3_nll - qf_nll)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        metavar="FILE.pdf",
        help="Output path for the NLL comparison plot.\n"
        "(default: nll_comparison_quickfit.pdf in --log-dir)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load cache
    # ------------------------------------------------------------------
    print(f"\nCache directory: {args.cache_dir}")
    log_prob_fn, _grad_fn, input_names = load_compiled_fn(args.cache_dir)

    if args.no_model:
        nominal_params: dict[str, float] = {}
    else:
        model = load_model(args.cache_dir)
        nominal_params = dict(model.free_params)

    # ------------------------------------------------------------------
    # Load quickFit NLL reference
    # ------------------------------------------------------------------
    nlls_path = args.log_dir / "nlls.txt"
    print(f"\nLoading quickFit NLLs from {nlls_path} ...")
    nlls = parse_nlls_file(nlls_path)
    print(f"  {len(nlls)} mu points: {sorted(nlls)[:5]} ...")

    # ------------------------------------------------------------------
    # Offset term: N_total * ln(n_channels) from the workspace (if given)
    # ------------------------------------------------------------------
    offset = compute_offset(args.workspace) if args.workspace else compute_offset(_DEFAULT_WORKSPACE_PATH)

    # ------------------------------------------------------------------
    # Collect log files to process
    # ------------------------------------------------------------------
    if args.all:
        # Process every mu in nlls.txt for which a log file exists.
        log_files = []
        missing_logs = []
        for mu in sorted(nlls):
            try:
                lf = find_log_for_mu(mu, args.log_dir)
                log_files.append(lf)
            except FileNotFoundError:
                missing_logs.append(mu)
        if missing_logs:
            print(f"  Warning: no log file for mu={missing_logs}")
        print(f"\nProcessing {len(log_files)} mu points ...")
    elif args.log:
        log_files = [args.log]
    else:
        log_files = [find_log_for_mu(args.mu, args.log_dir)]

    # ------------------------------------------------------------------
    # Evaluate and print
    # ------------------------------------------------------------------
    hdr = (
        f"{'mu':>10}  {'2*qf_NLL':>18}  {'pyhs3_NLL':>18}  "
        f"{'raw_diff':>12}  {'diff':>12}  conv  time"
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))

    results = []
    warned_coverage = False
    for log_path in log_files:
        r = run_one(log_path, log_prob_fn, input_names, nominal_params, nlls, offset)
        results.append(r)

        conv = "✓" if r["converged"] else "✗"
        qf = f"{r['qf_nll']:.6f}" if r["qf_nll"] is not None else "  (not in nlls.txt)"
        raw = f"{r['raw_diff']:+.6f}" if r["raw_diff"] is not None else "          —"
        diff = f"{r['diff']:+.6f}" if r["diff"] is not None else "          —"

        print(
            f"{r['mu_HH']:>+10.6g}  "
            f"{qf:>18}  "
            f"{r['pyhs3_nll']:>18.6f}  "
            f"{raw:>12}  "
            f"{diff:>12}  "
            f"{conv:>4}  "
            f"{r['eval_time_s']:.2f}s"
        )

        # Print coverage warnings on the first point only (same for every mu).
        if not warned_coverage:
            if r["missing_from_qf"]:
                n = len(r["missing_from_qf"])
                print(
                    f"         ↳ {n} pyhs3 input(s) not in qf log (using nominal): "
                    f"{r['missing_from_qf'][:5]}" + (" ..." if n > 5 else "")
                )
            if r["in_qf_not_pyhs3"]:
                n = len(r["in_qf_not_pyhs3"])
                print(
                    f"         ↳ {n} qf param(s) not used by pyhs3: "
                    f"{r['in_qf_not_pyhs3'][:5]}" + (" ..." if n > 5 else "")
                )
            warned_coverage = True

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if len(results) > 1:
        out_path = (
            args.output if args.output else args.log_dir / "nll_comparison_quickfit.pdf"
        )
        plot_results(results, out_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if len(results) > 1:
        diffs = [r["diff"] for r in results if r["diff"] is not None]
        if diffs:
            label = (
                "pyhs3_NLL - qf_NLL + N_total*ln(n_channels)"
                if offset
                else "pyhs3_NLL - qf_NLL"
            )
            print()
            print(f"{label} across {len(diffs)} points:")
            print(f"  mean = {np.mean(diffs):+.6f}")
            print(f"  std  = {np.std(diffs):.6f}")
            print(
                f"  min  = {np.min(diffs):+.6f}  (mu={results[np.argmin(diffs)]['mu_HH']})"
            )
            print(
                f"  max  = {np.max(diffs):+.6f}  (mu={results[np.argmax(diffs)]['mu_HH']})"
            )
            print()
            if np.std(diffs) < 0.01:
                print("std < 0.01: the two likelihoods agree well across the scan.")
            else:
                print(
                    "std ≥ 0.01: notable variation in the difference — the likelihoods\n"
                    "may disagree on which NPs contribute, or some parameters fed to\n"
                    "pyhs3 differ from what quickFit optimised."
                )


if __name__ == "__main__":
    main()
