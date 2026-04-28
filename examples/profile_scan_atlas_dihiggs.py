# ruff: noqa: T201
"""Profile-likelihood scan over mu_HH using pyhs3 + JAX + optimistix.

Reproduces the diHiggs bbyy validation from tests/test_manual.py at lower
runtime, demonstrating ws.model(analysis).log_prob + transpile + optimistix.

Install: pip install "pyhs3[jax]" optimistix matplotlib skhep-testdata
Run:     python examples/profile_scan_atlas_dihiggs.py
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx
import pytensor.tensor as pt
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import explicit_graph_inputs
from skhep_testdata import data_path as skhep_testdata_path

import pyhs3
from pyhs3.transpile import jaxify

# ROOT reference NLL values from issue #41 validation (Allex Wang / ATLAS bbyy).
# Index 0 is the unconditional best-fit (mu_HH ~ 1); indices 1-31 are the scan.
_REFERENCE_MU_HH = [
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
]
_REFERENCE_NLL = [
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
]

# Scan grid: skip index 0 (unconditional best-fit mu_HH ~ 1); use -0.5 to 2.5.
MU_GRID = _REFERENCE_MU_HH[1:]
REF_NLL = _REFERENCE_NLL[1:]


def _collect_init_values(ws: pyhs3.Workspace) -> dict[str, float]:
    """Collect initial parameter values, preferring unconditional-fit results."""
    vals: dict[str, float] = {}
    # Load in order of increasing priority (last write wins).
    for pset_name in [
        "default_values",
        "nominalGlobs",
        "nominalNuis",
        "unconditionalGlobs_muhat",
        "unconditionalNuis_muhat",
        "POI_muhat",
    ]:
        for p in ws.parameter_points[pset_name]:
            vals[p.name] = float(p.value)
    return vals


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Load workspace and build model from analysis                      #
    # ------------------------------------------------------------------ #
    ws_path = skhep_testdata_path("test_hs3_unbinned_pyhs3_validation_issue41.json")
    print(f"Loading workspace from {Path(ws_path).name} ...")
    ws = pyhs3.Workspace.load(ws_path)
    analysis = ws.analyses["CombinedPdf_combData"]

    print("Building symbolic model (may take a minute) ...")
    model = ws.model(analysis, progress=True)

    # ------------------------------------------------------------------ #
    # 2. Bake observed data as pt.constant, then transpile to JAX          #
    # ------------------------------------------------------------------ #
    # model.data returns {obs_name: event_array} for each channel.
    # Substituting the symbolic observable pt.vector inputs with pt.constant
    # nodes removes them from the jaxified function's inputs — data is then
    # fixed "for free" without any runtime overhead.
    data_np = model.data
    nll_expr = -2.0 * model.log_prob

    # Map symbolic variable name → pt.constant node.
    sym_inputs = list(explicit_graph_inputs([nll_expr]))
    data_substitutions: dict = {}
    for var in sym_inputs:
        if var.name in data_np:
            # TensorConstants created here are excluded from jaxify's input
            # list (explicit_graph_inputs skips Constant nodes), so the data
            # is embedded as a compile-time constant in the JAX function.
            data_substitutions[var] = pt.constant(
                np.asarray(data_np[var.name], dtype=np.float64)
            )

    nll_with_data = clone_replace(nll_expr, replace=data_substitutions)
    print(f"  Baked {len(data_substitutions)} observable channel(s) as constants.")

    print("Transpiling NLL expression to JAX ...")
    jg = jaxify(nll_with_data)
    print(f"  {len(jg.input_names)} free symbolic inputs (parameters only).")

    # ------------------------------------------------------------------ #
    # 3. Set up optimistix-compatible profile NLL                          #
    # ------------------------------------------------------------------ #
    # Parameters that remain after data substitution: mu_HH + nuisance.
    nuisance_names = sorted(set(jg.input_names) - {"mu_HH"})
    init_vals = _collect_init_values(ws)

    # Initial nuisance parameter vector (flat JAX array, for BFGS).
    nuisance_y0 = jnp.array([init_vals.get(n, 0.0) for n in nuisance_names])

    # Map nuisance name → index in free_values array (static at trace time).
    nuisance_idx = {n: i for i, n in enumerate(nuisance_names)}

    @jax.jit
    def profile_nll(free_values: jax.Array, mu_hh: jax.Array) -> jax.Array:
        """Joint -2 log L with data baked, nuisance free, mu_HH fixed."""
        # Build positional args in jg.input_names order.  All lookups use
        # Python-level static keys — JAX traces only the values.
        args = [
            mu_hh if name == "mu_HH" else free_values[nuisance_idx[name]]
            for name in jg.input_names
        ]
        return jg.fn(*args)[0]

    # Warm-up JIT at best-fit mu_HH to avoid measuring compile time in scan.
    _ = profile_nll(nuisance_y0, jnp.asarray(1.0))
    print("JIT warm-up done.")

    # ------------------------------------------------------------------ #
    # 4. Profile scan with BFGS and DFP                                   #
    # ------------------------------------------------------------------ #
    def run_scan(solver: optx.AbstractMinimiser, label: str) -> np.ndarray:
        nlls: list[float] = []
        n_ok = 0
        t0 = time.perf_counter()
        for mu in MU_GRID:
            mu_jax = jnp.asarray(float(mu))
            sol = optx.minimise(
                profile_nll,
                solver,
                y0=nuisance_y0,
                args=mu_jax,
                max_steps=2000,
                throw=False,
            )
            nll_val = float(profile_nll(sol.value, mu_jax))
            ok = bool(sol.result == optx.RESULTS.successful)
            nlls.append(nll_val)
            n_ok += ok
            print(
                f"  [{label}] mu={mu:+.2f}  NLL={nll_val:.4f}  "
                f"{'ok' if ok else 'WARN: did not converge'}"
            )
        dt = time.perf_counter() - t0
        print(
            f"{label}: {n_ok}/{len(MU_GRID)} converged in {dt:.2f}s "
            f"({1000 * dt / len(MU_GRID):.1f} ms/fit)\n"
        )
        return np.array(nlls)

    print("\n--- BFGS scan ---")
    bfgs_nll = run_scan(optx.BFGS(rtol=1e-6, atol=1e-6), "BFGS")

    print("--- DFP scan ---")
    dfp_nll = run_scan(optx.DFP(rtol=1e-6, atol=1e-6), "DFP")

    # ------------------------------------------------------------------ #
    # 5. Compare against ROOT reference and plot                          #
    # ------------------------------------------------------------------ #
    ref_arr = np.array(REF_NLL)
    mu_arr = np.array(MU_GRID)

    bfgs_delta = bfgs_nll - bfgs_nll.min()
    dfp_delta = dfp_nll - dfp_nll.min()
    ref_delta = ref_arr - ref_arr.min()

    max_bfgs_diff = float(np.max(np.abs(bfgs_delta - ref_delta)))
    max_dfp_diff = float(np.max(np.abs(dfp_delta - ref_delta)))
    print(f"Max |ΔNLL(pyhs3 BFGS) - ΔNLL(ROOT)| = {max_bfgs_diff:.4f}")
    print(f"Max |ΔNLL(pyhs3  DFP) - ΔNLL(ROOT)| = {max_dfp_diff:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mu_arr, ref_delta, marker="o", zorder=3, label="ROOT reference")
    ax.scatter(mu_arr, bfgs_delta, marker="x", s=80, label="pyhs3 + optx.BFGS")
    ax.scatter(mu_arr, dfp_delta, marker="+", s=80, label="pyhs3 + optx.DFP")
    ax.set_xlabel(r"$\mu_{HH}$")
    ax.set_ylabel(r"$\Delta\,(-2\ln\mathcal{L})$")
    ax.set_title("Profile-likelihood scan: diHiggs bbyy (pyhs3 validation)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out = Path("nll_profile_scan.pdf")
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
