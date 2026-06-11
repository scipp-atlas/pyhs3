# ruff: noqa: T201
"""Analyze PyTensor symbolic graph complexity for the ATLAS diHiggs bbyy workspace.

For each distribution used in the NLL scan, reports the symbolic Apply-node count
(pre-optimization).  This is fast — no compilation required.

Also compiles and inspects one background + one signal distribution to show the
optimizer's reduction ratio, and reports total function-call counts across a
full NLL scan.

Usage:
    pixi run -e py312-jax -- python analyze_graph_complexity.py
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import pytensor
from pytensor.graph.traversal import applys_between, explicit_graph_inputs
from skhep_testdata import data_path as skhep_testdata_path

import pyhs3

_MODEL_CACHE = Path("ws.pkl")
_MU_GRID_LEN = 31  # same as nll_validation_dihiggs.py

# Compile only a representative sample to show optimizer reduction ratio.
_SAMPLE_NAMES = [
    "pdf__background_Run3LM_4",          # simple background
    "pdf__ggFHH_kl1p0_mc23e_Run3LM_4",  # HH signal
    "pdf__ggFH_mc23e_Run3LM_4",          # single-H signal
    "constr__ATLAS_lumi_run3",           # constraint
]


def _symbolic_op_count(model: pyhs3.Model, name: str) -> tuple[int, int]:
    """Return (n_inputs, n_apply_nodes) for the symbolic graph of *name*."""
    dist = model.distributions[name]
    inputs = list(explicit_graph_inputs([dist]))
    applies = list(applys_between(inputs, [dist]))
    return len(inputs), len(applies)


def _compile_sample(
    model: pyhs3.Model, name: str
) -> tuple[int, Counter[str]]:
    """Compile *name* and return (optimized_op_count, op_type_counter)."""
    dist = model.distributions[name]
    inputs = [v for v in explicit_graph_inputs([dist]) if v.name is not None]
    fn = pytensor.function(
        inputs=inputs,
        outputs=dist,
        on_unused_input="ignore",
        trust_input=True,
    )
    nodes = fn.maker.fgraph.apply_nodes
    return len(nodes), Counter(type(n.op).__name__ for n in nodes)


def main() -> None:
    ws_path = skhep_testdata_path("test_hs3_unbinned_pyhs3_validation_issue41.json")
    print(f"Loading workspace from {Path(ws_path).name} ...")
    ws = pyhs3.Workspace.load(ws_path)
    analysis = ws.analyses["CombinedPdf_combData"]
    likelihood = analysis.likelihood

    print(f"Loading cached model from {_MODEL_CACHE} ...")
    with _MODEL_CACHE.open("rb") as f:
        model = pickle.load(f)

    # Mirror channel selection from nll_validation_dihiggs.py
    unbinned = [
        d
        for d in ws.data.root
        if getattr(d, "type", None) == "unbinned" and "binned" not in d.name
    ]
    channel_dist_names: list[str] = []
    for dist_obj, datum in zip(likelihood.distributions, unbinned, strict=False):
        dist_name = dist_obj if isinstance(dist_obj, str) else dist_obj.name
        channel_dist_names.append(dist_name)

    n_channels = len(channel_dist_names)
    print(f"\nChannels in NLL scan: {n_channels}")

    # ------------------------------------------------------------------ #
    # Symbolic graph analysis (all channels, no compilation)               #
    # ------------------------------------------------------------------ #
    print("\n{:<55s} {:>8s} {:>10s}".format("Distribution", "inputs", "sym ops"))
    print("-" * 77)

    total_inputs = 0
    total_sym = 0
    sym_counts: Counter[int] = Counter()
    input_counts: Counter[int] = Counter()

    for name in channel_dist_names:
        n_in, n_ops = _symbolic_op_count(model, name)
        total_inputs += n_in
        total_sym += n_ops
        sym_counts[n_ops] += 1
        input_counts[n_in] += 1
        print(f"  {name:<53s} {n_in:>8d} {n_ops:>10d}")

    print("\n" + "=" * 77)
    print(f"Channels                    : {n_channels}")
    print(f"Symbolic inputs total       : {total_inputs}  (avg {total_inputs/n_channels:.0f}/channel)")
    print(f"Symbolic ops total          : {total_sym}  (avg {total_sym/n_channels:.1f}/channel)")
    print(f"Distinct symbolic op counts : {dict(sorted(sym_counts.items()))}")
    print(f"Distinct input counts       : {dict(sorted(input_counts.most_common()))}")

    print(f"\nPer NLL evaluation ({n_channels} fn calls):")
    print(f"  Compiled fn invocations   : {n_channels}")
    print(f"  Symbolic ops in-flight    : {total_sym}")

    print(f"\nFull NLL scan ({_MU_GRID_LEN} mu values × {n_channels} channels = {_MU_GRID_LEN * n_channels} calls):")
    print(f"  Total fn invocations      : {_MU_GRID_LEN * n_channels}")
    print(f"  Total symbolic ops        : {_MU_GRID_LEN * total_sym}")

    # ------------------------------------------------------------------ #
    # Compile sample distributions to show optimizer reduction             #
    # ------------------------------------------------------------------ #
    print(f"\n\nCompiling {len(_SAMPLE_NAMES)} sample distributions (shows optimizer ratio) ...")
    print("{:<50s} {:>8s} {:>9s} {:>8s}  {:<s}".format(
        "Distribution", "sym ops", "opt ops", "ratio", "optimized op types"
    ))
    print("-" * 100)

    for name in _SAMPLE_NAMES:
        if name not in model.distributions:
            print(f"  {name}: NOT FOUND, skipping")
            continue
        _, n_sym = _symbolic_op_count(model, name)
        n_opt, op_types = _compile_sample(model, name)
        ratio = n_sym / max(n_opt, 1)
        print(
            f"  {name:<48s} {n_sym:>8d} {n_opt:>9d} {ratio:>7.1f}x"
            f"  {dict(op_types.most_common(5))}"
        )

    print("\nNote: run analyze_jaxpr.py for JAXpr equation count.")


if __name__ == "__main__":
    main()
