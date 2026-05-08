# ruff: noqa: T201
"""Profile-likelihood minimization demo for the ATLAS diHiggs bbyy workspace.

Demonstrates how to:
1. Build a pyhs3 Model from a HistFactory workspace
2. Compile the joint log-probability into a fast pytensor function
3. Profile over nuisance parameters at a fixed mu_HH using scipy

Install: pip install "pyhs3" scipy skhep-testdata
Run:     python examples/minimization_dihiggs.py
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
import json
from matplotlib import pyplot as plt

import numpy as np
from pytensor.compile.function import function
from pytensor.graph.traversal import explicit_graph_inputs
from scipy.optimize import minimize
from skhep_testdata import data_path as skhep_testdata_path

import pyhs3

_MODEL_CACHE = Path("ws.pkl")

_REFERENCE = {
    "mu_HH": [
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

def build_model() -> pyhs3.Model:
    """Load workspace and build (or load cached) model."""
    ws_path = skhep_testdata_path("test_hs3_unbinned_pyhs3_validation_issue41.json")
    print(f"Loading workspace from {Path(ws_path).name} ...")
    ws = pyhs3.Workspace.load(ws_path)
    analysis = ws.analyses["CombinedPdf_combData"]

    # Collect parameter points from the workspace
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

    #if _MODEL_CACHE.exists():
    #    print(f"Loading cached model from {_MODEL_CACHE} ...")
    #    with _MODEL_CACHE.open("rb") as f:
    #        return pickle.load(f)

    print("Building symbolic model (this takes ~1 min) ...")
    model = ws.model(analysis, parameter_set=param_set, progress=True)
    with _MODEL_CACHE.open("wb") as f:
        pickle.dump(model, f)
    print(f"  Model cached to {_MODEL_CACHE}")
    return model


def compile_log_prob(model: pyhs3.Model):
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


def profile_nll(
    log_prob_fn, inputs, model, mu_val, method="SLSQP", ftol=1e-3, maxiter=1000
):
    """Minimize -2*log_prob over nuisance parameters with mu_HH fixed.

    Parameters
    ----------
    log_prob_fn : pytensor compiled function
    inputs : list of symbolic input variables (ordering matters)
    model : pyhs3.Model with .free_params and .data
    mu_val : float, the fixed value of mu_HH for this scan point
    method : str, scipy minimization method (default SLSQP)
    ftol : float, tolerance for minimization to stop minimizing
    maxiter : int, maximum number of iterations for minimization

    Returns
    -------
    scipy.optimize.OptimizeResult
    """
    pinned = {**model.data, "mu_HH": mu_val}

    # Build a template: pinned entries filled in, free entries left as None.
    # We track which indices are free so the optimizer can fill them.
    template = []
    free_names = []
    free_input_indices = []
    bounds = []

    model_bounds = {axis.name: (axis.min, axis.max) for axis in model.domain.axes}
    for i, inp in enumerate(inputs):
        if inp.name in pinned:
            template.append(np.asarray(pinned[inp.name]))
        else:
            template.append(None)
            free_names.append(inp.name)
            free_input_indices.append(i)
            bounds.append(model_bounds[inp.name])

    x0 = np.array([model.free_params[name] for name in free_names], dtype=float)

    def nll(x):
        vals = list(template)
        for idx, xi in zip(free_input_indices, x, strict=True):
            vals[idx] = xi
        # np.asarray is a no-op for existing ndarrays (data), but wraps
        # np.float64 scalars (from iterating over x) into 0-d ndarrays
        # that pytensor's C VM accepts.
        return float(-2.0 * log_prob_fn(*[np.asarray(v) for v in vals])[0])

    return minimize(
        nll,
        x0,
        method=method,
        options={"maxiter": maxiter, "ftol": ftol},
        bounds=bounds,
    )


def main() -> None:
    model = build_model()
    computed_nlls = []

    print("\nCompiling log_prob ...")
    t0 = time.perf_counter()
    log_prob_fn, inputs = compile_log_prob(model)
    print(f"  Compiled in {time.perf_counter() - t0:.1f}s")
    print(f"  {len(inputs)} input variables, {len(model.free_params)} free parameters")

    # --- Single mu_HH demo ---
    # mu_demo = 1.0
    # print(f"\nProfiling NLL at mu_HH = {mu_demo} ...")
    # t0 = time.perf_counter()
    # result = profile_nll(log_prob_fn, inputs, model, mu_val=mu_demo)
    # dt = time.perf_counter() - t0
    # print(f"  -2*ln(L) = {result.fun:.6f}")
    # print(
    #     f"  converged = {result.success}  ({result.nit} iterations, "
    #     f"{result.nfev} fn evals, {dt:.1f}s)"
    # )

    # --- Uncomment below to scan over a grid of mu_HH values ---
    MU_GRID = [
        -0.5, -0.4, -0.3, -0.2, -0.1,
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5,
        1.6, 1.7, 1.8, 1.9, 2.0,
        2.1, 2.2, 2.3, 2.4, 2.5,
    ]
    mu_arr = np.array(MU_GRID)

    for mu in MU_GRID:
        t0 = time.perf_counter()
        result = profile_nll(log_prob_fn, inputs, model, mu_val=mu)
        dt = time.perf_counter() - t0
        print(
            f"mu={mu:+.1f}  -2ln(L)={result.fun:.6f}  "
            f"converged={result.success} ({result.nit} iterations, "
            f"{result.nfev} fn evals, {dt:.1f}s)"
        )
        computed_nlls.append(result)

    computed_nlls = [optim.fun for optim in computed_nlls]
    
    breakpoint()

    provided_nll = _REFERENCE["nll"]
    provided_nll_shifted = [v - min(provided_nll) for v in provided_nll]
    computed_nll_shifted = [v - min(computed_nlls) for v in computed_nlls]
    plt.figure()
    plt.scatter(mu_arr, provided_nll_shifted, label="provided nll", marker="o")
    plt.scatter(mu_arr, computed_nll_shifted, label="computed nll", marker="x")
    plt.xlabel("mu_HH")
    plt.ylabel("nll")
    plt.title("NLL Comparison with minimization")
    plt.legend()
    plt.savefig("nll_comparison_minimization.pdf")

    breakpoint() 
    

if __name__ == "__main__":
    main()