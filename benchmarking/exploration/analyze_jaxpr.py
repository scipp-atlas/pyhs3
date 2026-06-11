# ruff: noqa: T201
"""Count JAXpr equations for the pyhs3 NLL function on the diHiggs bbyy workspace.

Mirrors the baking strategy from examples/profile_scan_atlas_dihiggs.py:
  - observable data baked as constants
  - fixed (non-floating) parameters baked at best-fit values
  - mu_HH + nominalNuis kept as free symbolic inputs

Reports:
  - number of free inputs after baking
  - number of JAXpr equations (jax.make_jaxpr(nll)(params).eqns)
  - primitive breakdown (top-20)

Usage:
    pixi run -e py312-jax -- python analyze_jaxpr.py
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import jax
import numpy as np
import pytensor.tensor as pt
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import explicit_graph_inputs
from skhep_testdata import data_path as skhep_testdata_path

import pyhs3
from pyhs3.transpile import jaxify

_MODEL_CACHE = Path("ws.pkl")


def _collect_init_values(ws: pyhs3.Workspace) -> dict[str, float]:
    vals: dict[str, float] = {}
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
    ws_path = skhep_testdata_path("test_hs3_unbinned_pyhs3_validation_issue41.json")
    print(f"Loading workspace from {Path(ws_path).name} ...")
    ws = pyhs3.Workspace.load(ws_path)
    analysis = ws.analyses["CombinedPdf_combData"]

    print(f"Loading cached model from {_MODEL_CACHE} ...")
    with _MODEL_CACHE.open("rb") as f:
        model = pickle.load(f)

    # ------------------------------------------------------------------ #
    # Build reduced NLL expression (same baking as profile_scan script)   #
    # ------------------------------------------------------------------ #
    data_np = model.data
    nll_expr = -2.0 * model.log_prob

    nuisance_param_names = {p.name for p in ws.parameter_points["nominalNuis"]}
    free_names = nuisance_param_names | {"mu_HH"}
    init_vals = _collect_init_values(ws)

    all_inputs = [v for v in explicit_graph_inputs([nll_expr]) if v.name is not None]

    all_subs: dict = {}
    free_vars: dict[str, pt.TensorVariable] = {}
    for var in all_inputs:
        if var.name in data_np:
            all_subs[var] = pt.constant(np.asarray(data_np[var.name], dtype=np.float64))
        elif var.name in free_names:
            if var.name not in free_vars:
                free_vars[var.name] = pt.scalar(var.name)
            all_subs[var] = free_vars[var.name]
        else:
            all_subs[var] = pt.constant(np.float64(init_vals.get(var.name, 0.0)))

    nll_fixed = clone_replace(nll_expr, replace=all_subs)
    n_free = len(free_vars)
    n_fixed = len(all_inputs) - len(data_np) - n_free
    print(
        f"  {len(data_np)} observable(s) baked, "
        f"{n_fixed} fixed params baked, "
        f"{n_free} free params remain."
    )

    # ------------------------------------------------------------------ #
    # Transpile to JAX and count JAXpr equations                          #
    # ------------------------------------------------------------------ #
    print("Transpiling NLL expression to JAX ...")
    jg = jaxify(nll_fixed)
    n_inputs = len(jg.input_names)
    print(f"  JAX function has {n_inputs} scalar inputs: {sorted(jg.input_names)[:5]} ...")

    # Build a sample inputs dict at nominal values.
    sample_args = [
        np.float64(init_vals.get(name, 0.0)) for name in jg.input_names
    ]

    print("Tracing JAXpr ...")
    jaxpr = jax.make_jaxpr(jg.fn)(*sample_args)

    n_eqns = len(jaxpr.eqns)
    print(f"\nJAXpr equations          : {n_eqns}")
    print(f"JAX free inputs          : {n_inputs}")

    # Primitive breakdown
    prim_counts: Counter[str] = Counter(
        str(eqn.primitive) for eqn in jaxpr.eqns
    )
    print(f"\nTop-20 JAXpr primitives (of {len(prim_counts)} distinct):")
    for prim, cnt in prim_counts.most_common(20):
        pct = 100 * cnt / n_eqns
        print(f"  {prim:<35s} {cnt:>6d}  ({pct:5.1f}%)")

    # Also trace the jit-compiled version to see after XLA optimizations
    print("\nTracing jit-lowered version ...")
    jitted = jax.jit(jg.fn)
    lowered = jitted.lower(*sample_args)
    try:
        compiled_text = lowered.as_text()
        # Count HLO instructions (rough proxy)
        hlo_lines = [l.strip() for l in compiled_text.splitlines() if l.strip() and not l.strip().startswith("//")]
        print(f"HLO text lines (rough instruction count): {len(hlo_lines)}")
    except Exception as e:
        print(f"Could not extract HLO text: {e}")


if __name__ == "__main__":
    main()
