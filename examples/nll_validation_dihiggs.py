# ruff: noqa: T201
"""Vectorized NLL validation for the ATLAS diHiggs bbyy workspace (issue #41).

Reproduces the NLL curve from tests/test_manual.py using vectorized
``model.logpdf_unsafe`` calls instead of a per-event Python loop.  No JAX
or optimistix required.

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

_MODEL_CACHE = Path("ws.pkl")


def _collect_params(ws: pyhs3.Workspace) -> dict[str, float]:
    """Collect parameter values; best-fit sets override nominal (last write wins)."""
    params: dict[str, float] = {}
    for pset_name in [
        "default_values",
        "nominalGlobs",
        "nominalNuis",
        "unconditionalGlobs_muhat",
        "unconditionalNuis_muhat",
        "POI_muhat",
    ]:
        for p in ws.parameter_points[pset_name]:
            params[p.name] = float(p.value)
    return params


def plot_dist(
    model: pyhs3.Model,
    parameters: dict,
    dist_name: str,
    data_set,
    *,
    factor: float = 1.0,
    plot_name: str | None = None,
    label: str | None = None,
    color: str = "red",
    linewidth: float = 2.5,
) -> None:
    """Plot a distribution PDF overlaid on its data points."""
    obs_name = data_set.axes[0].name
    xs = np.asarray([val[0] for val in data_set.entries], dtype=np.float64)
    sort_idx = np.argsort(xs)
    xs = xs[sort_idx]
    ys = model.pdf_unsafe(dist_name, **{**parameters, obs_name: xs}) * factor
    plt.figure(plot_name)
    plt.title(plot_name)
    kwargs = {"color": color, "linewidth": linewidth}
    if label is not None:
        kwargs["label"] = label
    plt.plot(xs, ys, **kwargs)
    plt.ylim(0, 18)


def main() -> None:
    ws_path = skhep_testdata_path("test_hs3_unbinned_pyhs3_validation_issue41.json")
    print(f"Loading workspace from {Path(ws_path).name} ...")
    ws = pyhs3.Workspace.load(ws_path)
    analysis = ws.analyses["CombinedPdf_combData"]
    likelihood = analysis.likelihood

    # ------------------------------------------------------------------ #
    # Build or load model                                                  #
    # ------------------------------------------------------------------ #
    if _MODEL_CACHE.exists():
        print(f"Loading cached model from {_MODEL_CACHE} ...")
        with _MODEL_CACHE.open("rb") as f:
            model = pickle.load(f)
    else:
        print("Building symbolic model (this takes ~1 min) ...")
        model = ws.model(analysis, progress=True)
        with _MODEL_CACHE.open("wb") as f:
            pickle.dump(model, f)
        print(f"  Model cached to {_MODEL_CACHE}")

    # Parameter values: collect from workspace sets (best-fit overrides nominal).
    parameters = _collect_params(ws)

    # Only keep unbinned data sets that are not the "binned-resampled" copies.
    unbinned = [
        d
        for d in ws.data.root
        if getattr(d, "type", None) == "unbinned" and "binned" not in d.name
    ]

    # ------------------------------------------------------------------ #
    # Vectorized NLL scan                                                  #
    # ------------------------------------------------------------------ #
    # For each channel, collect the weighted entries once (reused across mu).
    channel_data: list[tuple[str, str, np.ndarray]] = []  # (dist_name, obs_name, vals)
    for dist_obj, datum in zip(likelihood.distributions, unbinned, strict=False):
        dist_name = dist_obj if isinstance(dist_obj, str) else dist_obj.name
        obs_name = datum.axes[0].name
        obs_vals = datum.weighted_entries[:, 0]
        vals = np.sort(obs_vals[np.abs(obs_vals) > 1e-6])
        channel_data.append((dist_name, obs_name, vals))

    # Some workspace parameters are referenced by distributions but absent
    # from every named parameter set (e.g. auxiliary RNDM__ observables).
    # Supply 0.0 as a safe default so _reorder_params never hits a KeyError.
    all_dist_exprs = [model.distributions[dn] for dn, _, _ in channel_data]
    all_needed = {
        v.name for v in explicit_graph_inputs(all_dist_exprs) if v.name is not None
    }
    missing = all_needed - set(parameters)
    if missing:
        print(
            f"  Defaulting {len(missing)} undeclared parameter(s) to 0.0 (e.g. {sorted(missing)[0]})"
        )
        parameters.update(dict.fromkeys(missing, 0.0))

    print(f"\nRunning NLL scan over {len(MU_GRID)} mu_HH values ...")
    nll_given_mu: list[float] = []
    t0 = time.perf_counter()

    for i, mu in enumerate(MU_GRID, 1):
        parameters["mu_HH"] = mu
        total_nll = 0.0
        for dist_name, obs_name, vals in channel_data:
            # One compiled-function call per channel — no per-event Python loop.
            log_pdfs = model.logpdf_unsafe(dist_name, **{**parameters, obs_name: vals})
            total_nll += -2.0 * float(np.sum(log_pdfs)) / len(vals)
        nll_given_mu.append(total_nll)
        print(f"  ({i:2d}/{len(MU_GRID)}) mu_HH={mu:+.2f}  NLL={total_nll:.4f}")

    dt = time.perf_counter() - t0
    print(f"\nTotal: {dt:.2f}s  ({1000 * dt / len(MU_GRID):.1f} ms/mu)")

    # ------------------------------------------------------------------ #
    # Compare with ROOT reference and plot                                 #
    # ------------------------------------------------------------------ #
    computed = np.array(nll_given_mu)
    ref = np.array(REF_NLL)

    delta_computed = computed - computed.min()
    delta_ref = ref - ref.min()
    max_diff = float(np.max(np.abs(delta_computed - delta_ref)))
    print(f"\nMax |ΔNLL(pyhs3) - ΔNLL(ROOT)| = {max_diff:.4f}")

    mu_arr = np.array(MU_GRID)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mu_arr, delta_ref, marker="o", zorder=3, label="ROOT reference")
    ax.scatter(mu_arr, delta_computed, marker="x", s=80, label="pyhs3 (vectorized)")
    ax.set_xlabel(r"$\mu_{HH}$")
    ax.set_ylabel(r"$\Delta\,(-2\ln\mathcal{L})$")
    ax.set_title("NLL validation: diHiggs bbyy (pyhs3 vs ROOT)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out = Path("nll_validation.pdf")
    fig.savefig(out)
    print(f"Wrote {out}")

    # Save JSON for downstream comparison.
    out_json = Path("nll_validation.json")
    out_json.write_text(
        json.dumps(
            {"mu_HH": MU_GRID, "nll": nll_given_mu, "nll_ref": REF_NLL}, indent=2
        )
    )
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
