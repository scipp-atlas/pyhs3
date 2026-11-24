"""Integration tests: Model builds batchable distributions as families."""

from __future__ import annotations

from typing import ClassVar

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


def _workspace(duplicates: tuple[tuple[str, str], ...] = ()) -> pyhs3.Workspace:
    """Two channels, each an extended mixture of two Gaussians, plus constraints.

    The Gaussians in a channel share their observable; the three Gaussian
    constraints on alpha_a/b/c are product factors of both channels.  Each
    ``(channel, s)`` in ``duplicates`` adds ``constr_{s}_{channel}``, identical
    to ``constr_{s}`` but for its name, as another factor of that channel.
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
                "factors": [
                    f"{ch}_mix",
                    "constr_a",
                    "constr_b",
                    "constr_c",
                    *(f"constr_{s}_{c}" for c, s in duplicates if c == ch),
                ],
            },
        ]
    distributions += [
        BatchableGaussian(name=f"constr_{s}", x=f"alpha_{s}", mean=0.0, sigma=1.0)
        for s in ("a", "b", "c")
    ]
    distributions += [
        BatchableGaussian(name=f"constr_{s}_{ch}", x=f"alpha_{s}", mean=0.0, sigma=1.0)
        for ch, s in duplicates
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
                # HS3 section 2.7: every free parameter must be in a domain.
                "axes": [
                    {"name": p["name"], "min": -20.0, "max": 20.0}
                    if p["name"] not in ("alpha_a", "alpha_b", "alpha_c")
                    else {"name": p["name"], "min": -5.0, "max": 5.0}
                    for p in parameters
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


def _model(duplicates: tuple[tuple[str, str], ...] = ()):
    ws = _workspace(duplicates)
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


class TestStructuralSharingInFamilies:
    """Members identical but for their name reuse one expression, as without families."""

    @pytest.mark.parametrize(
        "duplicates",
        [
            pytest.param((("g0", "a"),), id="within-one-family"),
            pytest.param((("g1", "a"),), id="across-families"),
        ],
    )
    def test_log_prob_and_gradient_match_the_per_member_build(
        self, duplicates, request
    ):
        model = _model(duplicates)
        ((channel, s),) = duplicates
        assert (
            model.log_distributions[f"constr_{s}_{channel}"]
            is model.log_distributions[f"constr_{s}"]
        )
        value, grad = _value_and_grad(model)
        request.getfixturevalue("unbatched")
        ref_value, ref_grad = _value_and_grad(_model(duplicates))

        assert value == pytest.approx(ref_value, rel=1e-12)
        for name, g in ref_grad.items():
            assert grad[name] == pytest.approx(g, rel=1e-10, abs=1e-12)


class UnsharedBatchableGaussian(BatchableGaussian):
    """A batchable type that opts out of structural sharing."""

    share_identical_expressions: ClassVar[bool] = False


def _twin_products(cls: type[BatchableGaussian]) -> pyhs3.Workspace:
    """prod_A and prod_B multiply constraints identical but for their names.

    Without a likelihood every distribution is a root, so each product's two
    constraints form a family of their own.
    """
    distributions: list = []
    for p in ("A", "B"):
        distributions += [
            cls(name=f"c{p}_{s}", x=f"alpha_{s}", mean=0.0, sigma=1.0)
            for s in ("a", "b")
        ]
        distributions.append(
            {
                "name": f"prod_{p}",
                "type": "product_dist",
                "factors": [f"c{p}_a", f"c{p}_b"],
            }
        )
    return pyhs3.Workspace(
        metadata={"hs3_version": "0.2"},
        parameter_points=[
            {
                "name": "init",
                "parameters": [
                    {"name": "alpha_a", "value": 0.6},
                    {"name": "alpha_b", "value": -1.3},
                ],
            }
        ],
        distributions=distributions,
    )


class TestFamiliesWithoutRowsOfTheirOwn:
    @pytest.mark.parametrize(
        ("cls", "shared"),
        [
            pytest.param(BatchableGaussian, True, id="shared"),
            pytest.param(UnsharedBatchableGaussian, False, id="opted-out"),
        ],
    )
    def test_twin_families_match_the_per_member_build(self, cls, shared, request):
        model = _twin_products(cls).model(0, progress=False)
        assert (
            model.log_distributions["cB_a"] is model.log_distributions["cA_a"]
        ) == shared
        point = {"alpha_a": np.array(0.6), "alpha_b": np.array(-1.3)}
        value = model.logpdf("prod_B", **point)
        request.getfixturevalue("unbatched")
        reference = (
            _twin_products(cls).model(0, progress=False).logpdf("prod_B", **point)
        )
        np.testing.assert_allclose(value, reference, rtol=1e-12)
