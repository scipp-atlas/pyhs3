"""Time each phase of fitting a HistFactory workspace with pyhs3 as it stands.

Phases: load, Model build, graph size, compile (value, then value+gradient), per-call cost, and an
L-BFGS-B fit. Results are rewritten to OUTDIR/pyhs3_<MODE>.json after every phase, so a run that
never finishes compiling still records how far it got.

Every graph input that is not observed data is fitted, including parameters the analysis domains
do not list (on the ROOT-exported HWW workspace: Lumi and 100 gamma_stat parameters that ROOT holds
constant); those get unbounded limits. pyhs3 itself exposes them as free inputs.

Usage: python bench_pyhs3.py WORKSPACE.json ANALYSIS OUTDIR [MODE]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytensor
import pytensor.tensor as pt
import scipy.optimize
from bfgs import penalize_undefined
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import applys_between, explicit_graph_inputs

import pyhs3

REPEATS = 50


def main(workspace: Path, analysis: str, outdir: Path, mode: str) -> None:
    out: dict[str, object] = {"mode": mode}
    record_path = outdir / f"pyhs3_{mode}.json"

    def record(key: str, value: object) -> None:
        out[key] = value
        print(f"{key}: {value}", flush=True)
        record_path.write_text(
            json.dumps(out, indent=1, default=float), encoding="utf-8"
        )

    t = time.perf_counter()
    ws = pyhs3.Workspace.load(workspace)
    record("load_s", time.perf_counter() - t)
    t = time.perf_counter()
    model = ws.model(analysis, progress=False, mode=mode)
    record("model_build_s", time.perf_counter() - t)
    log_prob = model.log_prob

    data = model.data
    inputs = {
        v.name: v for v in explicit_graph_inputs([log_prob]) if v.name is not None
    }
    par_names = [name for name in inputs if name not in data]
    record("n_free_inputs", len(par_names))
    record(
        "n_apply_nodes_raw",
        len(list(applys_between(list(explicit_graph_inputs([log_prob])), [log_prob]))),
    )

    raw = json.loads(workspace.read_text(encoding="utf-8"))
    spec = next(a for a in raw["analyses"] if a["name"] == analysis)
    bounds = {
        axis["name"]: (axis.get("min", -np.inf), axis.get("max", np.inf))
        for domain in raw["domains"]
        if domain["name"] in spec["domains"]
        for axis in domain["axes"]
    }
    lower = np.array([bounds.get(name, (-np.inf, np.inf))[0] for name in par_names])
    upper = np.array([bounds.get(name, (-np.inf, np.inf))[1] for name in par_names])
    # Starting values layered as in `pyhs3 nll`: the workspace's parameter_points, then the model's
    # free_params (empty when the analysis has no init set).
    start = {
        point.name: float(point.value)
        for pset in ws.parameter_points or []
        for point in pset
    }
    start |= model.free_params
    missing = [name for name in par_names if name not in start]
    if missing:
        msg = f"no starting value for {len(missing)} parameters, e.g. {missing[:3]}"
        raise ValueError(msg)
    x0 = np.clip(
        np.array([start[name] for name in par_names], dtype=float), lower, upper
    )

    # One (P,) vector input for all free parameters instead of P scalar keyword arguments per call.
    theta = pt.dvector("theta")
    replacements = {
        inputs[name]: theta[i].reshape(inputs[name].type.shape)
        if inputs[name].ndim
        else theta[i]
        for i, name in enumerate(par_names)
    }
    twice_nll = -2.0 * graph_replace(log_prob, replacements, strict=False).sum()
    data_vars = [inputs[name] for name in inputs if name in data]
    data_values = [np.asarray(data[v.name]) for v in data_vars]

    t = time.perf_counter()
    value_fn = pytensor.function(
        [theta, *data_vars], twice_nll, mode=mode, on_unused_input="ignore"
    )
    record("compile_value_s", time.perf_counter() - t)
    record("n_apply_nodes_compiled_value", len(value_fn.maker.fgraph.apply_nodes))

    t = time.perf_counter()
    value_grad_fn = pytensor.function(
        [theta, *data_vars],
        [twice_nll, pt.grad(twice_nll, theta)],
        mode=mode,
        on_unused_input="ignore",
    )
    record("compile_value_grad_s", time.perf_counter() - t)

    calls = 0

    def fg(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal calls
        calls += 1
        value, grad = value_grad_fn(x, *data_values)
        return float(value), np.asarray(grad, dtype=float)

    fg(x0)
    t = time.perf_counter()
    for _ in range(REPEATS):
        fg(x0)
    record("value_grad_ms", (time.perf_counter() - t) / REPEATS * 1e3)

    calls = 0
    t = time.perf_counter()
    result = scipy.optimize.minimize(
        penalize_undefined(fg),
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper, strict=True)),
        options={"maxiter": 20000, "maxfun": 50000, "ftol": 1e-12, "gtol": 1e-6},
    )
    record("lbfgsb_s", time.perf_counter() - t)
    record("lbfgsb_calls", calls)
    record("lbfgsb_message", str(result.message))
    record("lbfgsb_twice_nll", float(result.fun))
    record(
        "lbfgsb_poi_hat",
        float(result.x[par_names.index(spec["parameters_of_interest"][0])]),
    )


if __name__ == "__main__":
    main(
        Path(sys.argv[1]),
        sys.argv[2],
        Path(sys.argv[3]),
        sys.argv[4] if len(sys.argv) > 4 else "FAST_RUN",
    )
