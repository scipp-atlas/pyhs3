"""Count PyTensor apply nodes in pyhs3's log_prob as HistFactory modifiers are added.

Builds single-channel HS3 workspaces in memory and reports the slope:
- nodes per normsys / histosys modifier, each with its own parameter and constraint term;
- nodes per additional normsys entry when every sample shares one parameter, so its
  constraint term is built once (the situation in large workspaces).

Usage: python node_counts.py
"""

from __future__ import annotations

from typing import Any

from pytensor.graph.traversal import applys_between, explicit_graph_inputs

import pyhs3

N_BINS = 4
AXIS = {"name": "x", "min": 0.0, "max": 1.0, "nbins": N_BINS}


def normsys(name: str, parameter: str) -> dict[str, Any]:
    return {
        "type": "normsys",
        "name": name,
        "parameter": parameter,
        "constraint": "Gauss",
        "data": {"hi": 1.1, "lo": 0.9},
    }


def histosys(name: str, parameter: str) -> dict[str, Any]:
    return {
        "type": "histosys",
        "name": name,
        "parameter": parameter,
        "constraint": "Gauss",
        "data": {
            "hi": {"contents": [11.0] * N_BINS},
            "lo": {"contents": [9.0] * N_BINS},
        },
    }


def workspace(
    samples: list[list[dict[str, Any]]], parameters: list[str]
) -> pyhs3.Workspace:
    """One channel whose samples carry the given modifier lists; every parameter gets a domain axis."""
    normfactor = {"type": "normfactor", "name": "mu", "parameter": "mu"}
    return pyhs3.Workspace(
        metadata={"hs3_version": "0.2"},
        distributions=[
            {
                "name": "SR",
                "type": "histfactory_dist",
                "axes": [AXIS],
                "samples": [
                    {
                        "name": f"s{i}",
                        "data": {"contents": [10.0] * N_BINS, "errors": [1.0] * N_BINS},
                        "modifiers": [normfactor, *modifiers],
                    }
                    for i, modifiers in enumerate(samples)
                ],
            }
        ],
        data=[
            {
                "name": "obs",
                "type": "binned",
                "axes": [AXIS],
                "contents": [10.0] * N_BINS,
            }
        ],
        likelihoods=[{"name": "L", "distributions": ["SR"], "data": ["obs"]}],
        domains=[
            {
                "name": "d",
                "type": "product_domain",
                "axes": [
                    {"name": p, "min": -5.0, "max": 5.0} for p in ["mu", *parameters]
                ],
            }
        ],
        analyses=[{"name": "A", "likelihood": "L", "domains": ["d"]}],
    )


def raw_nodes(ws: pyhs3.Workspace) -> int:
    log_prob = ws.model("A", progress=False).log_prob
    return len(
        list(applys_between(list(explicit_graph_inputs([log_prob])), [log_prob]))
    )


def main() -> None:
    base = raw_nodes(workspace([[]], []))
    for n in (10, 20, 40):
        params = [f"a{i}" for i in range(n)]
        ws = workspace([[normsys(f"ns{i}", p) for i, p in enumerate(params)]], params)
        print(f"{n} normsys modifiers: {(raw_nodes(ws) - base) / n:.1f} nodes each")
    for n in (10, 20):
        params = [f"h{i}" for i in range(n)]
        ws = workspace([[histosys(f"hs{i}", p) for i, p in enumerate(params)]], params)
        print(f"{n} histosys modifiers: {(raw_nodes(ws) - base) / n:.1f} nodes each")
    for n in (10, 40):
        without = raw_nodes(workspace([[] for _ in range(n)], []))
        shared = raw_nodes(workspace([[normsys("ns", "a")] for _ in range(n)], ["a"]))
        print(
            f"{n} samples sharing one normsys: {(shared - without) / n:.1f} nodes per entry"
        )


if __name__ == "__main__":
    main()
