"""Integration tests: Model builds batchable functions as families."""

from __future__ import annotations

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.graph.traversal import explicit_graph_inputs, io_toposort
from pytensor.tensor.basic import Split

import pyhs3
from pyhs3 import model as model_module


def _workspace() -> pyhs3.Workspace:
    """Two Gaussian channels whose means are products of interpolation functions.

    Each interpolation entry is a ProductFunction ``alpha * beta`` with constant
    beta, high, low and nominal values; channel 1 also uses its own NP alpha_c.
    """
    parameters = [
        {"name": "alpha_a", "value": 0.0},
        {"name": "alpha_b", "value": 0.0},
        {"name": "alpha_c", "value": 0.0},
    ]
    functions = []
    interps = {"g0": [], "g1": []}
    nps = {"g0": ["alpha_a", "alpha_b"], "g1": ["alpha_a", "alpha_c"]}
    for channel, codes_per_function in (
        ("g0", [[5, 1], [0, 0]]),
        ("g1", [[5, 5], [1]]),
    ):
        for k, codes in enumerate(codes_per_function):
            name = f"{channel}_interp{k}"
            entries = []
            for i, _ in enumerate(codes):
                beta = f"{name}_beta{i}"
                parameters += [
                    {"name": beta, "value": 1.0 + 0.1 * i, "const": True},
                    {
                        "name": f"{name}_hi{i}",
                        "value": 1.0 + 0.2 + 0.05 * i,
                        "const": True,
                    },
                    {
                        "name": f"{name}_lo{i}",
                        "value": 1.0 - 0.15 - 0.05 * i,
                        "const": True,
                    },
                ]
                product = f"{name}_entry{i}"
                functions.append(
                    {
                        "name": product,
                        "type": "product",
                        "factors": [nps[channel][i % 2], beta],
                    }
                )
                entries.append(product)
            parameters.append({"name": f"{name}_nom", "value": 2.0 + k, "const": True})
            functions.append(
                {
                    "name": name,
                    "type": "interpolation",
                    "high": [f"{name}_hi{i}" for i in range(len(codes))],
                    "low": [f"{name}_lo{i}" for i in range(len(codes))],
                    "nom": f"{name}_nom",
                    "interpolationCodes": codes,
                    "positiveDefinite": False,
                    "vars": entries,
                }
            )
            interps[channel].append(name)
        functions.append(
            {"name": f"{channel}_mean", "type": "product", "factors": interps[channel]}
        )
    distributions = [
        {
            "name": ch,
            "type": "gaussian_dist",
            "x": f"x_{ch}",
            "mean": f"{ch}_mean",
            "sigma": 2.0,
        }
        for ch in ("g0", "g1")
    ] + [
        {
            "name": f"constr_{a}",
            "type": "gaussian_dist",
            "x": a,
            "mean": 0.0,
            "sigma": 1.0,
        }
        for a in ("alpha_a", "alpha_b", "alpha_c")
    ]
    data = [
        {
            "name": f"d_{ch}",
            "type": "unbinned",
            "axes": [{"name": f"x_{ch}", "min": -20.0, "max": 20.0}],
            "entries": [[v] for v in values],
        }
        for ch, values in (("g0", [4.1, 5.0, 6.3]), ("g1", [6.0, 7.2]))
    ]
    return pyhs3.Workspace(
        metadata={"hs3_version": "0.2"},
        parameter_points=[{"name": "init", "parameters": parameters}],
        functions=functions,
        distributions=distributions,
        domains=[
            {
                "name": "main",
                "type": "product_domain",
                "axes": [
                    {"name": a, "min": -5.0, "max": 5.0}
                    for a in ("alpha_a", "alpha_b", "alpha_c")
                ],
            }
        ],
        data=data,
        likelihoods=[
            {
                "name": "L",
                "distributions": ["g0", "g1"],
                "data": ["d_g0", "d_g1"],
                "aux_distributions": [
                    "constr_alpha_a",
                    "constr_alpha_b",
                    "constr_alpha_c",
                ],
            }
        ],
        analyses=[
            {"name": "A", "likelihood": "L", "domains": ["main"], "init": "init"}
        ],
    )


@pytest.fixture
def unbatched(monkeypatch):
    """Build models without families, as the per-function reference."""
    monkeypatch.setattr(model_module, "plan_function_families", lambda *_a, **_k: [])


def _model():
    ws = _workspace()
    return ws.model(ws.analyses["A"], progress=False)


def _value_and_grad(model, point):
    inputs = [v for v in explicit_graph_inputs([model.log_prob]) if v.name]
    free = [v for v in inputs if v.name in point]
    total = pt.sum(model.log_prob)
    fn = pytensor.function(inputs, [total, *pt.grad(total, free)], mode="FAST_COMPILE")
    out = fn(**{**model.data, **point})
    return out[0], dict(zip([v.name for v in free], out[1:], strict=True))


_POINT = {"alpha_a": 0.6, "alpha_b": -1.3, "alpha_c": 0.9}


class TestModelBuildsFamilies:
    def test_log_prob_and_gradient_match_the_per_function_build(self, request):
        value, grad = _value_and_grad(_model(), _POINT)
        request.getfixturevalue("unbatched")
        ref_value, ref_grad = _value_and_grad(_model(), _POINT)

        assert value == pytest.approx(ref_value, rel=1e-12)
        for name, g in ref_grad.items():
            assert grad[name] == pytest.approx(g, rel=1e-10)

    def test_log_prob_uses_the_batched_path(self, request):
        """Each channel's interpolations are split from one batched vector."""

        def splits(model):
            nodes = io_toposort([], [model.log_prob])
            return sum(isinstance(node.op, Split) for node in nodes)

        assert splits(_model()) == 2  # one family per channel
        request.getfixturevalue("unbatched")
        assert splits(_model()) == 0

    def test_a_distribution_only_evaluates_its_own_channel(self):
        """g0's pdf does not depend on g1's NP, so families are scoped per channel."""
        model = _model()
        names = {v.name for v in explicit_graph_inputs([model.distributions["g0"]])}
        assert "alpha_a" in names
        assert "alpha_c" not in names

    def test_pdf_matches_the_per_function_build(self, request):
        point = {
            "x_g0": np.array([4.1, 5.0]),
            "alpha_a": np.array(0.6),
            "alpha_b": np.array(-1.3),
        }
        value = _model().logpdf("g0", **point)
        request.getfixturevalue("unbatched")
        np.testing.assert_allclose(value, _model().logpdf("g0", **point), rtol=1e-12)
