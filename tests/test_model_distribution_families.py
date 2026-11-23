"""Integration tests: Model builds batchable distributions as families."""

from __future__ import annotations

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.graph.traversal import explicit_graph_inputs

import pyhs3
from pyhs3 import model as model_module
from pyhs3.distributions import BatchableDistribution, GaussianDist


class BatchableGaussian(GaussianDist, BatchableDistribution):
    """GaussianDist opted into batching, for tests.

    No built-in distribution opts in; see docs/explanation/batched_families.
    """


def _workspace() -> pyhs3.Workspace:
    """Two channels, each an extended mixture of two Gaussians, plus constraints.

    The Gaussians in a channel share their observable; the three Gaussian
    constraints on alpha_a/b/c are product factors of both channels.
    """
    parameters = [{"name": n, "value": 0.0} for n in ("alpha_a", "alpha_b", "alpha_c")]
    parameters += [
        {"name": "mu_g0_s", "value": 5.0},
        {"name": "mu_g0_b", "value": 2.0},
        {"name": "mu_g1_s", "value": 6.0},
        {"name": "mu_g1_b", "value": 1.0},
        {"name": "c_s", "value": 3.0},
        {"name": "c_b", "value": 10.0},
    ]
    distributions = []
    for ch in ("g0", "g1"):
        distributions += [
            BatchableGaussian(
                name=f"{ch}_sig", x=f"x_{ch}", mean=f"mu_{ch}_s", sigma=1.0
            ),
            BatchableGaussian(
                name=f"{ch}_bkg", x=f"x_{ch}", mean=f"mu_{ch}_b", sigma=3.0
            ),
            {
                "name": f"{ch}_mix",
                "type": "mixture_dist",
                "summands": [f"{ch}_sig", f"{ch}_bkg"],
                "coefficients": ["c_s", "c_b"],
                "extended": True,
            },
            {
                "name": f"{ch}_model",
                "type": "product_dist",
                "factors": [f"{ch}_mix", "constr_a", "constr_b", "constr_c"],
            },
        ]
    distributions += [
        BatchableGaussian(name=f"constr_{s}", x=f"alpha_{s}", mean=0.0, sigma=1.0)
        for s in ("a", "b", "c")
    ]
    data = [
        {
            "name": f"d_{ch}",
            "type": "unbinned",
            "axes": [{"name": f"x_{ch}", "min": -10.0, "max": 20.0}],
            "entries": [[v] for v in values],
        }
        for ch, values in (("g0", [4.1, 5.0, 1.3, 2.2]), ("g1", [6.0, 0.7, 5.5]))
    ]
    return pyhs3.Workspace(
        metadata={"hs3_version": "0.2"},
        parameter_points=[{"name": "init", "parameters": parameters}],
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
                "distributions": ["g0_model", "g1_model"],
                "data": ["d_g0", "d_g1"],
            }
        ],
        analyses=[
            {"name": "A", "likelihood": "L", "domains": ["main"], "init": "init"}
        ],
    )


@pytest.fixture
def unbatched(monkeypatch):
    """Build models without families, as the per-member reference."""
    monkeypatch.setattr(model_module, "plan_families", lambda *_a, **_k: [])


def _model():
    ws = _workspace()
    return ws.model(ws.analyses["A"], progress=False)


_POINT = {
    "alpha_a": 0.6,
    "alpha_b": -1.3,
    "alpha_c": 0.9,
    "mu_g0_s": 4.7,
    "mu_g0_b": 2.5,
    "mu_g1_s": 6.2,
    "mu_g1_b": 0.8,
    "c_s": 3.5,
    "c_b": 9.0,
}


def _value_and_grad(model):
    inputs = [v for v in explicit_graph_inputs([model.log_prob]) if v.name]
    free = [v for v in inputs if v.name in _POINT]
    total = pt.sum(model.log_prob)
    fn = pytensor.function(inputs, [total, *pt.grad(total, free)], mode="FAST_COMPILE")
    out = fn(**{**model.data, **_POINT})
    return out[0], dict(zip([v.name for v in free], out[1:], strict=True))


class TestModelBuildsDistributionFamilies:
    def test_log_prob_and_gradient_match_the_per_member_build(self, request):
        value, grad = _value_and_grad(_model())
        request.getfixturevalue("unbatched")
        ref_value, ref_grad = _value_and_grad(_model())

        assert value == pytest.approx(ref_value, rel=1e-12)
        assert grad.keys() == ref_grad.keys()
        for name, g in ref_grad.items():
            assert grad[name] == pytest.approx(g, rel=1e-10, abs=1e-12)

    def test_logpdf_of_a_member_matches_the_per_member_build(self, request):
        """A member's logpdf takes the inputs of every member batched with it."""
        point = {
            "x_g0": np.array([4.1, 5.0]),
            "mu_g0_s": np.array(4.7),
            "mu_g0_b": np.array(2.5),
        }
        value = _model().logpdf("g0_sig", **point)
        request.getfixturevalue("unbatched")
        np.testing.assert_allclose(
            value, _model().logpdf("g0_sig", **point), rtol=1e-12
        )

    def test_members_are_named(self):
        model = _model()
        assert model.distributions["g0_sig"].name == "g0_sig"
        assert model.log_distributions["constr_b"].name == "constr_b"
