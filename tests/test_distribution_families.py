"""Unit tests for evaluating many distributions of one type as a batched family."""

from __future__ import annotations

import logging

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.graph.traversal import io_toposort
from pytensor.tensor.basic import Split

from pyhs3.context import Context
from pyhs3.distributions import AsymmetricCrystalBallDist, Distributions, GaussianDist
from pyhs3.distributions.core import BatchableDistribution
from pyhs3.functions import Functions
from pyhs3.networks import build_dependency_graph, plan_families
from pyhs3.parameter_points import ParameterSet

_EVENTS = np.array([118.0, 122.5, 124.9, 125.2, 127.0, 131.0])


class BatchableCrystalBall(AsymmetricCrystalBallDist, BatchableDistribution):
    """AsymmetricCrystalBallDist opted into batching, for tests.

    No built-in distribution opts in (see docs/explanation/batched_families),
    but their math is elementwise, so they exercise the default
    batch_log_expressions.
    """


class BatchableGaussian(GaussianDist, BatchableDistribution):
    """GaussianDist opted into batching, for tests (see BatchableCrystalBall)."""


def _observable_context(names=("x",), bounds=(105.0, 160.0)):
    """Context with observables of shape (N, 1) views and their bounds."""
    leaves = {name: pt.dvector(name) for name in names}
    return (
        Context(
            parameters=dict(leaves),
            observables={
                n: (pt.constant(bounds[0]), pt.constant(bounds[1])) for n in names
            },
            views={n: leaf[:, None] for n, leaf in leaves.items()},
        ),
        leaves,
    )


def _crystal_ball(name, m="x"):
    fields = ("m0", "sigma_L", "sigma_R", "alpha_L", "alpha_R", "n_L", "n_R")
    return BatchableCrystalBall(name=name, m=m, **{f: f"{name}_{f}" for f in fields})


_CB_VALUES = {
    "m0": 125.0,
    "sigma_L": 1.6,
    "sigma_R": 1.9,
    "alpha_L": 1.4,
    "alpha_R": 1.7,
    "n_L": 5.0,
    "n_R": 9.0,
}


def _add_crystal_ball_parameters(context, members, rng, free=()):
    """Scalar shape parameters per member: free inputs for names in ``free``."""
    inputs, values = [], {}
    for member in members:
        for field, base in _CB_VALUES.items():
            name = getattr(member, field)
            value = base * (1.0 + 0.05 * rng.standard_normal())
            if field in free:
                var = pt.dscalar(name)
                inputs.append(var)
                values[name] = value
                context.add_parameter(name, var)
            else:
                context.add_parameter(name, pt.constant(value, name=name))
    return inputs, values


def _batched(members, context):
    outputs = type(members[0]).batch_log_expressions(members, context)
    return [outputs[m.name] for m in members]


def _per_member(members, context):
    return [m.log_expression(context) for m in members]


class TestBatchableDistributionInterface:
    @pytest.mark.parametrize("cls", [AsymmetricCrystalBallDist, GaussianDist])
    def test_built_in_distributions_do_not_opt_in(self, cls):
        """Batching them slows the C backend (see docs/explanation/batched_families)."""
        assert not issubclass(cls, BatchableDistribution)

    @pytest.mark.parametrize("cls", [BatchableCrystalBall, BatchableGaussian])
    def test_a_subclass_opts_in_by_inheritance(self, cls):
        assert issubclass(cls, BatchableDistribution)


class TestBatchedDistributionsMatchPerMember:
    def test_crystal_balls_sharing_an_observable(self):
        """Normalised log densities, shape (N, 1) per member, as built one by one."""
        context, leaves = _observable_context()
        members = [_crystal_ball(f"cb{k}") for k in range(3)]
        inputs, values = _add_crystal_ball_parameters(
            context, members, np.random.default_rng(0), free=("m0", "sigma_R")
        )
        fn_inputs = [leaves["x"], *inputs]

        def evaluate(outputs):
            fn = pytensor.function(fn_inputs, outputs, mode="FAST_COMPILE")
            return fn(_EVENTS, *(values[v.name] for v in inputs))

        batched = evaluate(_batched(members, context))
        expected = evaluate(_per_member(members, context))
        for got, want in zip(batched, expected, strict=True):
            assert got.shape == (len(_EVENTS), 1)
            np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_crystal_ball_gradient(self):
        context, leaves = _observable_context()
        members = [_crystal_ball(f"cb{k}") for k in range(2)]
        inputs, values = _add_crystal_ball_parameters(
            context,
            members,
            np.random.default_rng(1),
            free=("m0", "sigma_L", "alpha_R"),
        )

        def gradient(outputs):
            total = pt.sum([pt.sum(o) for o in outputs])
            fn = pytensor.function(
                [leaves["x"], *inputs], pt.grad(total, inputs), mode="FAST_COMPILE"
            )
            return fn(_EVENTS, *(values[v.name] for v in inputs))

        np.testing.assert_allclose(
            gradient(_batched(members, context)),
            gradient(_per_member(members, context)),
            rtol=1e-10,
        )

    def test_scalar_gaussian_constraints(self):
        """Constraints on parameters have no observable: one scalar per member."""
        alphas = [pt.dscalar(f"alpha_{k}") for k in range(4)]
        context = Context(parameters={a.name: a for a in alphas})
        members = []
        for k, alpha in enumerate(alphas):
            context.add_parameter(f"glob_{k}", pt.constant(0.1 * k, name=f"glob_{k}"))
            members.append(
                BatchableGaussian(
                    name=f"constr_{k}",
                    x=alpha.name,
                    mean=f"glob_{k}",
                    sigma=1.0 + 0.5 * k,
                )
            )
            for key, tensor in members[-1].constants.items():
                context.add_parameter(key, tensor)
        point = [0.3, -1.2, 2.0, 0.0]

        def evaluate(outputs):
            return pytensor.function(alphas, outputs, mode="FAST_COMPILE")(*point)

        batched = evaluate(_batched(members, context))
        assert all(np.shape(b) == () for b in batched)
        np.testing.assert_allclose(
            batched, evaluate(_per_member(members, context)), rtol=1e-12
        )

    def test_gaussians_sharing_an_observable(self):
        context, leaves = _observable_context()
        means = [pt.dscalar(f"mu_{k}") for k in range(3)]
        members = []
        for k, mean in enumerate(means):
            context.add_parameter(mean.name, mean)
            context.add_parameter(f"s_{k}", pt.constant(2.0 + k, name=f"s_{k}"))
            members.append(
                BatchableGaussian(name=f"g{k}", x="x", mean=mean.name, sigma=f"s_{k}")
            )

        def evaluate(outputs):
            fn = pytensor.function([leaves["x"], *means], outputs, mode="FAST_COMPILE")
            return fn(_EVENTS, 124.0, 126.0, 130.0)

        for got, want in zip(
            evaluate(_batched(members, context)),
            evaluate(_per_member(members, context)),
            strict=True,
        ):
            np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_identical_members(self):
        """Members that differ only in name share one expression."""
        context, leaves = _observable_context()
        members = [_crystal_ball("cb0"), _crystal_ball("cb0")]
        members[1] = members[1].model_copy(update={"name": "cb1"})
        _add_crystal_ball_parameters(context, members[:1], np.random.default_rng(2))
        batched = _batched(members, context)
        expected = _per_member(members, context)
        fn = pytensor.function(
            [leaves["x"]], [*batched, *expected], mode="FAST_COMPILE"
        )
        a, b, c, d = fn(_EVENTS)
        np.testing.assert_allclose([a, b], [c, d], rtol=1e-12)


class TestBatchedDistributionFallback:
    def test_different_observables_use_the_per_member_path(self, caplog):
        context, leaves = _observable_context(names=("x", "y"))
        members = [_crystal_ball("cb0", m="x"), _crystal_ball("cb1", m="y")]
        _add_crystal_ball_parameters(context, members, np.random.default_rng(3))
        with caplog.at_level(logging.DEBUG, logger="pyhs3.distributions.core"):
            batched = _batched(members, context)
        assert "cb0" in caplog.text
        fn = pytensor.function(
            [leaves["x"], leaves["y"]],
            [*batched, *_per_member(members, context)],
            mode="FAST_COMPILE",
        )
        a, b, c, d = fn(_EVENTS, _EVENTS + 1.0)
        np.testing.assert_allclose(a, c, rtol=1e-12)
        np.testing.assert_allclose(b, d, rtol=1e-12)

    def test_batched_parameter_uses_the_per_member_path(self, caplog):
        context, leaves = _observable_context()
        members = [_crystal_ball(f"cb{k}") for k in range(2)]
        _add_crystal_ball_parameters(context, members, np.random.default_rng(4))
        scan = pt.dvector("cb0_m0")
        context.add_parameter("cb0_m0", scan[None, :])  # a (1, M) view
        with caplog.at_level(logging.DEBUG, logger="pyhs3.distributions.core"):
            batched = _batched(members, context)
        assert "cb0" in caplog.text
        fn = pytensor.function(
            [leaves["x"], scan],
            [*batched, *_per_member(members, context)],
            mode="FAST_COMPILE",
        )
        a, b, c, d = fn(_EVENTS, np.array([124.0, 125.0, 126.0]))
        np.testing.assert_allclose(a, c, rtol=1e-12)
        np.testing.assert_allclose(b, d, rtol=1e-12)


class TestPlanDistributionFamilies:
    def test_distributions_feeding_one_root_form_a_family(self):
        distributions = Distributions(
            [
                BatchableGaussian(name="g0", x="x", mean="mu0", sigma=1.0),
                BatchableGaussian(name="g1", x="x", mean="mu1", sigma=2.0),
                {
                    "name": "mix",
                    "type": "mixture_dist",
                    "summands": ["g0", "g1"],
                    "coefficients": ["c0"],
                    "extended": False,
                },
            ]
        )
        functions = Functions([])
        graph, _, _ = build_dependency_graph(
            ParameterSet(name="p", parameters=[]), functions, distributions
        )
        families = plan_families(graph, [*functions, *distributions], ["mix"])

        assert [sorted(f.members) for f in families] == [["g0", "g1"]]
        assert families[0].kind is BatchableGaussian
        assert graph.predecessors("g0") == [families[0].name]


class TestBatchedDistributionSubBatches:
    def test_shapes_and_constraints_of_one_class_batch_separately(self, caplog):
        """Members using an observable and members of parameters only form two batches."""
        context, leaves = _observable_context()
        alphas = [pt.dscalar(f"alpha_{k}") for k in range(2)]
        means = [pt.dscalar(f"mu_{k}") for k in range(2)]
        for var in (*alphas, *means):
            context.add_parameter(var.name, var)
        members = [
            BatchableGaussian(name=f"shape_{k}", x="x", mean=f"mu_{k}", sigma=2.0 + k)
            for k in range(2)
        ] + [
            BatchableGaussian(name=f"constr_{k}", x=f"alpha_{k}", mean=0.0, sigma=1.0)
            for k in range(2)
        ]
        for member in members:
            for key, tensor in member.constants.items():
                context.add_parameter(key, tensor)

        with caplog.at_level(logging.DEBUG, logger="pyhs3.distributions.core"):
            batched = _batched(members, context)

        assert "built on its own" not in caplog.text
        splits = {
            id(node)
            for out in batched
            for node in io_toposort([], [out])
            if isinstance(node.op, Split)
        }
        assert len(splits) == 2  # one batch per shape signature
        inputs = [leaves["x"], *alphas, *means]
        fn = pytensor.function(
            inputs, [*batched, *_per_member(members, context)], mode="FAST_COMPILE"
        )
        out = fn(_EVENTS, 0.4, -1.1, 124.0, 127.0)
        for got, want in zip(out[:4], out[4:], strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-12)
