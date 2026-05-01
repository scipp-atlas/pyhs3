# ruff: noqa: T201
"""NLL validation for the ATLAS diHiggs bbyy workspace (issue #41).

Demonstrates two approaches to computing NLL over a mu_HH scan grid:

1. **Scalar (non-batched)**: compile ``model.log_prob`` with scalar mu_HH,
   then loop over the scan grid evaluating one point at a time.

2. **Vectorized (batched)**: set ``param_set["mu_HH"].kind = pt.vector``
   before building the model, then pass the entire scan grid in a single call.

Both approaches produce identical results and are validated against ROOT
reference values from ATLAS bbyy diHiggs (issue #41).

Install: pip install "pyhs3" matplotlib skhep-testdata
Run:     python examples/nll_validation_dihiggs.py
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytensor.tensor as pt
from pytensor.compile.function import function
from pytensor.graph.traversal import explicit_graph_inputs
from skhep_testdata import data_path as skhep_testdata_path

import pyhs3

# ROOT reference NLL values from issue #41 (Allex Wang / ATLAS bbyy).
# Index 0 is the unconditional best-fit; indices 1-31 are the mu_HH scan.
_REFERENCE = {
    "mu_HH": [
        0.9999909338901939,
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
    ],
    "nll": [
        2115.2146170568185,
        2116.5050528141624,
        2116.32024442325,
        2116.147536871291,
        2115.9898748809364,
        2115.8484135401295,
        2115.7234904744287,
        2115.614760462136,
        2115.5217651455596,
        2115.4428493502382,
        2115.3774140758096,
        2115.3244289405097,
        2115.282917807648,
        2115.2519751699165,
        2115.2307707411514,
        2115.2185475973124,
        2115.214617056965,
        2115.218352131681,
        2115.229180543526,
        2115.2465778591986,
        2115.270060985844,
        2115.299182173344,
        2115.333523579179,
        2115.372692531467,
        2115.4163176659454,
        2115.4640463133146,
        2115.5155437672224,
        2115.5704953916675,
        2115.6286127532935,
        2115.689167112463,
        2115.7526215376183,
        2115.8184195879385,
    ],
}

# Scan grid: skip index 0 (unconditional best-fit).
MU_GRID = _REFERENCE["mu_HH"][1:]
REF_NLL = _REFERENCE["nll"][1:]

_SCALAR_CACHE = Path("ws_scalar.pkl")
_BATCHED_CACHE = Path("ws_batched.pkl")


def load_workspace():
    """Load workspace and collect parameter points.

    Returns (workspace, analysis, param_set).
    """
    ws_path = skhep_testdata_path("test_hs3_unbinned_pyhs3_validation_issue41.json")
    print(f"Loading workspace from {Path(ws_path).name} ...")
    ws = pyhs3.Workspace.load(ws_path)
    analysis = ws.analyses["CombinedPdf_combData"]

    pset_names = [
        "default_values",
        "nominalGlobs",
        "nominalNuis",
        "unconditionalGlobs_muhat",
        "unconditionalNuis_muhat",
        "POI_muhat",
    ]
    param_set = pyhs3.parameter_points.ParameterSet(
        name="collected",
        parameters=[
            pp for pset_name in pset_names for pp in ws.parameter_points[pset_name]
        ],
    )
    return ws, analysis, param_set


def build_or_load_model(ws, analysis, param_set, cache_path):
    """Build model from workspace, or load from pickle cache."""
    if cache_path.exists():
        print(f"Loading cached model from {cache_path} ...")
        with cache_path.open("rb") as f:
            return pickle.load(f)
    print("Building symbolic model (this takes ~1 min) ...")
    model = ws.model(analysis, parameter_set=param_set, progress=True)
    with cache_path.open("wb") as f:
        pickle.dump(model, f)
    print(f"  Model cached to {cache_path}")
    return model


def compile_log_prob(model):
    """Compile model.log_prob into a callable pytensor function.

    Returns (log_prob_fn, inputs) where inputs is the ordered list of
    symbolic input variables the compiled function expects.
    """
    dist_expression = model.log_prob
    inputs = [
        var for var in explicit_graph_inputs([dist_expression]) if var.name is not None
    ]
    log_prob_fn = function(
        inputs=inputs,
        outputs=dist_expression,
        mode=model.mode,
        on_unused_input="ignore",
        name=model._likelihood.name,
        trust_input=True,
    )
    return log_prob_fn, inputs


def scan_scalar(log_prob_fn, inputs, model, mu_grid):
    """Evaluate -2*log_prob one mu_HH value at a time (non-batched).

    Loops over *mu_grid*, calling the compiled function once per point.
    Each call receives scalar parameter values wrapped via ``np.asarray``
    (pytensor's C VM rejects bare ``np.float64`` scalars).
    """
    base_vals = {**model.free_params, **model.data}
    nlls = []
    for mu in mu_grid:
        vals = {**base_vals, "mu_HH": mu}
        args = [np.asarray(vals[inp.name]) for inp in inputs]
        nll = float(-2.0 * log_prob_fn(*args)[0])
        nlls.append(nll)
    return nlls


def scan_batched(log_prob_fn, inputs, model, mu_grid):
    """Evaluate -2*log_prob for all mu_HH values in a single call (batched).

    Passes the entire *mu_grid* as a 1-D array.  The model must have been
    built with ``param_set["mu_HH"].kind = pt.vector`` so the symbolic
    graph broadcasts correctly over the scan dimension.
    """
    vals = {**model.free_params, **model.data, "mu_HH": np.array(mu_grid)}
    args = [np.asarray(vals[inp.name]) for inp in inputs]
    # log_prob shape: (M,) where M = len(mu_grid)
    log_probs = log_prob_fn(*args)[0]
    return (-2.0 * log_probs).tolist()


def main() -> None:
    ws, analysis, param_set = load_workspace()

    # ================================================================== #
    # Approach 1: Scalar (non-batched)                                    #
    # Build model with all parameters as scalars, then loop over mu_HH.  #
    # ================================================================== #
    print("\n=== Approach 1: Scalar (non-batched) ===")
    model_scalar = build_or_load_model(ws, analysis, param_set, _SCALAR_CACHE)

    print("Compiling log_prob ...")
    t0 = time.perf_counter()
    log_prob_fn, inputs = compile_log_prob(model_scalar)
    print(f"  Compiled in {time.perf_counter() - t0:.1f}s")
    print(f"  {len(inputs)} input variables, {len(model_scalar.free_params)} free")

    print(f"Running NLL scan over {len(MU_GRID)} mu_HH values ...")
    t0 = time.perf_counter()
    nlls_scalar = scan_scalar(log_prob_fn, inputs, model_scalar, MU_GRID)
    dt_scalar = time.perf_counter() - t0
    print(f"  {dt_scalar:.2f}s  ({1000 * dt_scalar / len(MU_GRID):.1f} ms/point)")

    # ================================================================== #
    # Approach 2: Vectorized (batched)                                    #
    # Set mu_HH to pt.vector so the entire grid is evaluated at once.     #
    # ================================================================== #
    print("\n=== Approach 2: Vectorized (batched) ===")
    param_set["mu_HH"].kind = pt.vector
    model_batched = build_or_load_model(ws, analysis, param_set, _BATCHED_CACHE)

    print("Compiling log_prob ...")
    t0 = time.perf_counter()
    log_prob_fn_b, inputs_b = compile_log_prob(model_batched)
    print(f"  Compiled in {time.perf_counter() - t0:.1f}s")
    print(f"  {len(inputs_b)} input variables, {len(model_batched.free_params)} free")

    print(f"Running NLL scan over {len(MU_GRID)} mu_HH values ...")
    t0 = time.perf_counter()
    nlls_batched = scan_batched(log_prob_fn_b, inputs_b, model_batched, MU_GRID)
    dt_batched = time.perf_counter() - t0
    print(f"  {dt_batched:.2f}s  (single call)")

    # ================================================================== #
    # Compare approaches and validate against ROOT reference              #
    # ================================================================== #
    scalar_arr = np.array(nlls_scalar)
    batched_arr = np.array(nlls_batched)
    ref_arr = np.array(REF_NLL)

    max_diff_approaches = float(np.max(np.abs(scalar_arr - batched_arr)))
    print(f"\nMax |NLL(scalar) - NLL(batched)| = {max_diff_approaches:.6e}")

    delta_scalar = scalar_arr - scalar_arr.min()
    delta_batched = batched_arr - batched_arr.min()
    delta_ref = ref_arr - ref_arr.min()

    max_diff_scalar = float(np.max(np.abs(delta_scalar - delta_ref)))
    max_diff_batched = float(np.max(np.abs(delta_batched - delta_ref)))
    print(f"Max |ΔNLL(scalar)  - ΔNLL(ROOT)| = {max_diff_scalar:.4f}")
    print(f"Max |ΔNLL(batched) - ΔNLL(ROOT)| = {max_diff_batched:.4f}")

    for i, mu in enumerate(MU_GRID):
        print(
            f"  mu_HH={mu:+.2f}  "
            f"scalar={nlls_scalar[i]:.4f}  "
            f"batched={nlls_batched[i]:.4f}  "
            f"ROOT={REF_NLL[i]:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Plot ΔNLL curves                                                     #
    # ------------------------------------------------------------------ #
    mu_arr = np.array(MU_GRID)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mu_arr, delta_ref, marker="o", zorder=3, label="ROOT reference")
    ax.scatter(
        mu_arr,
        delta_scalar,
        marker="x",
        s=80,
        label=f"pyhs3 scalar ({dt_scalar:.1f}s)",
    )
    ax.scatter(
        mu_arr,
        delta_batched,
        marker="+",
        s=80,
        label=f"pyhs3 batched ({dt_batched:.1f}s)",
    )
    ax.set_xlabel(r"$\mu_{HH}$")
    ax.set_ylabel(r"$\Delta\,(-2\ln\mathcal{L})$")
    ax.set_title("NLL validation: diHiggs bbyy (pyhs3 vs ROOT)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out = Path("nll_validation.pdf")
    fig.savefig(out)
    print(f"\nWrote {out}")

    # Save JSON for downstream comparison.
    out_json = Path("nll_validation.json")
    out_json.write_text(
        json.dumps(
            {
                "mu_HH": MU_GRID,
                "nll_scalar": nlls_scalar,
                "nll_batched": nlls_batched,
                "nll_ref": REF_NLL,
            },
            indent=2,
        )
    )
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
