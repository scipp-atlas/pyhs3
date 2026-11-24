"""Unit tests for evaluating many InterpolationFunctions as one batched family."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.graph.traversal import io_toposort
from pytensor.tensor.subtensor import AdvancedIncSubtensor1, IncSubtensor

from pyhs3.compiled import default_nll_mode
from pyhs3.distributions import Distributions
from pyhs3.functions import Functions, InterpolationFunction, ProductFunction
from pyhs3.functions.core import BatchableFunction
from pyhs3.networks import build_dependency_graph, plan_families
from pyhs3.parameter_points import ParameterSet

_NOMINALS = [10.0, 3.0, 0.5, 7.0]


def _interp(name, codes, nom, entries, positive_definite=False):
    """InterpolationFunction whose i-th entry is ``entries[i]`` with hi/lo names."""
    return InterpolationFunction(
        name=name,
        high=[f"{name}_hi_{i}" for i in range(len(codes))],
        low=[f"{name}_lo_{i}" for i in range(len(codes))],
        nom=nom,
        interpolationCodes=codes,
        positiveDefinite=positive_definite,
        vars=entries,
    )


class _Setup:
    """Members, a context of tensors, and the free scalar inputs."""

    def __init__(self, members, products, *, symbolic_variations=False, seed=0):
        rng = np.random.default_rng(seed)
        self.members = members
        self.functions = {f.name: f for f in [*members, *products]}
        self.inputs: list[pt.TensorVariable] = []
        self.context: dict[str, pt.TensorVariable] = {}
        for k, member in enumerate(members):
            nom = _NOMINALS[k % len(_NOMINALS)]
            self.context.setdefault(member.nom, pt.constant(nom, name=member.nom))
            for i in range(len(member.vars)):
                hi = nom * (1.0 + rng.uniform(0.02, 0.3))
                lo = nom * (1.0 - rng.uniform(0.02, 0.3))
                for name, value in ((member.high[i], hi), (member.low[i], lo)):
                    if symbolic_variations:
                        var = pt.dscalar(name)
                        self.inputs.append(var)
                        self.context[name] = var
                        self.values = getattr(self, "values", {})
                        self.values[name] = value
                    else:
                        self.context[name] = pt.constant(value, name=name)
        self.values = getattr(self, "values", {})
        for product in products:
            for factor in product.factors:
                if isinstance(factor, str) and factor not in self.context:
                    self._add_free(factor, rng)
            for key, tensor in product.constants.items():
                self.context[key] = tensor
            self.context[product.name] = product.expression(self.context)
        for member in members:
            for var in member.vars:
                if var not in self.context:
                    self._add_free(var, rng)

    def _add_free(self, name, rng):
        var = pt.dscalar(name)
        self.inputs.append(var)
        self.context[name] = var
        self.values[name] = rng.uniform(-2.0, 2.0)

    def evaluate(self, outputs):
        fn = pytensor.function(
            self.inputs, outputs, mode="FAST_COMPILE", on_unused_input="ignore"
        )
        return fn(**self.values)


def _per_function(setup):
    return [m.expression(setup.context) for m in setup.members]


def _batched(setup):
    outputs = InterpolationFunction.batch_expressions(
        setup.members, setup.context, setup.functions
    )
    return [outputs[m.name] for m in setup.members]


def _product_entries(prefix, n, params, betas):
    """``n`` ProductFunctions ``param * beta`` named ``{prefix}_{i}``."""
    products = [
        ProductFunction(
            name=f"{prefix}_{i}", factors=[params[i % len(params)], betas[i]]
        )
        for i in range(n)
    ]
    return products, [p.name for p in products]


class TestBatchableInterface:
    def test_interpolation_function_is_batchable(self):
        assert issubclass(InterpolationFunction, BatchableFunction)


class TestBatchedInterpolationMatchesPerFunction:
    @pytest.mark.parametrize(
        "codes",
        [
            pytest.param([[5, 5, 1], [1], [5, 5, 5, 5]], id="multiplicative"),
            pytest.param([[0, 0], [2, 3], [4]], id="additive"),
            pytest.param([[6, 6], [0, 2], [1, 5, 5], []], id="mixed_kinds_and_empty"),
        ],
    )
    @pytest.mark.parametrize("positive_definite", [False, True])
    def test_product_entries(self, codes, positive_definite):
        """Entries ``NP * const`` across several functions, shared NPs."""
        params = ["alpha_a", "alpha_b", "alpha_c"]
        members, products = [], []
        for k, member_codes in enumerate(codes):
            prods, names = _product_entries(
                f"f{k}_entry",
                len(member_codes),
                params,
                [1.0 + 0.1 * i for i in range(len(member_codes))],
            )
            products += prods
            members.append(
                _interp(f"f{k}", member_codes, f"nom_{k}", names, positive_definite)
            )
        setup = _Setup(members, products)

        np.testing.assert_allclose(
            setup.evaluate(_batched(setup)),
            setup.evaluate(_per_function(setup)),
            rtol=1e-12,
        )

    def test_direct_parameter_entries(self):
        members = [
            _interp("f0", [5, 1], "nom_0", ["alpha_a", "alpha_b"]),
            _interp("f1", [5], "nom_1", ["alpha_a"]),
        ]
        setup = _Setup(members, [])
        np.testing.assert_allclose(
            setup.evaluate(_batched(setup)),
            setup.evaluate(_per_function(setup)),
            rtol=1e-12,
        )

    def test_entries_with_several_free_factors(self):
        """An entry that is not ``NP * const`` is stacked as its expression."""
        products = [ProductFunction(name="mixed", factors=["alpha_a", "alpha_b", 2.0])]
        members = [
            _interp("f0", [5, 5], "nom_0", ["mixed", "alpha_c"]),
            _interp("f1", [1], "nom_1", ["alpha_a"]),
        ]
        setup = _Setup(members, products)
        np.testing.assert_allclose(
            setup.evaluate(_batched(setup)),
            setup.evaluate(_per_function(setup)),
            rtol=1e-12,
        )

    def test_symbolic_variations(self):
        """Non-constant high/low values are stacked instead of tabulated."""
        members = [
            _interp("f0", [5, 1], "nom_0", ["alpha_a", "alpha_b"]),
            _interp("f1", [0], "nom_1", ["alpha_a"]),
        ]
        setup = _Setup(members, [], symbolic_variations=True)
        np.testing.assert_allclose(
            setup.evaluate(_batched(setup)),
            setup.evaluate(_per_function(setup)),
            rtol=1e-12,
        )

    def test_gradient_matches(self):
        params = ["alpha_a", "alpha_b"]
        prods0, names0 = _product_entries("e0", 3, params, [1.0, 0.5, 2.0])
        prods1, names1 = _product_entries("e1", 2, params, [1.5, 1.0])
        members = [
            _interp("f0", [5, 1, 5], "nom_0", names0),
            _interp("f1", [0, 0], "nom_1", names1),
        ]
        setup = _Setup(members, [*prods0, *prods1])
        free = [v for v in setup.inputs if v.name in params]

        def gradient(outputs):
            return pt.grad(pt.sum(pt.stack(outputs) ** 2), free)

        np.testing.assert_allclose(
            setup.evaluate(gradient(_batched(setup))),
            setup.evaluate(gradient(_per_function(setup))),
            rtol=1e-10,
        )


class TestBatchedInterpolationFallback:
    def test_interleaved_kinds_use_the_per_function_path(self, caplog):
        """Additive and multiplicative entries that interleave do not commute."""
        members = [
            _interp("f0", [0, 5, 0], "nom_0", ["alpha_a", "alpha_b", "alpha_c"]),
            _interp("f1", [5], "nom_1", ["alpha_a"]),
        ]
        setup = _Setup(members, [])
        with caplog.at_level(logging.DEBUG, logger="pyhs3.functions.standard"):
            batched = _batched(setup)
        assert "f0" in caplog.text
        np.testing.assert_allclose(
            setup.evaluate(batched), setup.evaluate(_per_function(setup)), rtol=1e-12
        )

    def test_batched_parameter_uses_the_per_function_path(self, caplog):
        members = [
            _interp("f0", [5], "nom_0", ["alpha_a"]),
            _interp("f1", [5], "nom_1", ["alpha_b"]),
        ]
        setup = _Setup(members, [])
        scan = pt.dvector("alpha_a")
        setup.inputs = [scan if v.name == "alpha_a" else v for v in setup.inputs]
        setup.context["alpha_a"] = scan
        setup.values["alpha_a"] = np.array([-1.0, 0.0, 1.5])
        with caplog.at_level(logging.DEBUG, logger="pyhs3.functions.standard"):
            batched = _batched(setup)
        assert "f0" in caplog.text
        result = setup.evaluate(batched)
        expected = setup.evaluate(_per_function(setup))
        for got, want in zip(result, expected, strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_every_member_on_the_per_function_path(self, caplog):
        members = [
            _interp("f0", [0, 5, 0], "nom_0", ["alpha_a", "alpha_b", "alpha_c"]),
            _interp("f1", [5, 0, 5], "nom_1", ["alpha_a", "alpha_b", "alpha_c"]),
        ]
        setup = _Setup(members, [])
        with caplog.at_level(logging.DEBUG, logger="pyhs3.functions.standard"):
            batched = _batched(setup)
        assert "f0" in caplog.text
        assert "f1" in caplog.text
        np.testing.assert_allclose(
            setup.evaluate(batched), setup.evaluate(_per_function(setup)), rtol=1e-12
        )

    @pytest.mark.parametrize("which", ["nominal", "variation"])
    def test_non_scalar_inputs_use_the_per_function_path(self, which, caplog):
        """A (3,) nominal or variation cannot sit in the (F,) or (F, E) tables."""
        members = [
            _interp("f0", [5], "nom_0", ["alpha_a"]),
            _interp("f1", [5], "nom_1", ["alpha_b"]),
        ]
        setup = _Setup(members, [])
        name = "nom_0" if which == "nominal" else "f0_hi_0"
        vector = pt.dvector(name)
        setup.inputs.append(vector)
        setup.context[name] = vector
        setup.values[name] = np.array([9.0, 10.0, 11.0])
        with caplog.at_level(logging.DEBUG, logger="pyhs3.functions.standard"):
            batched = _batched(setup)
        assert "'f0' is built on its own" in caplog.text
        result = setup.evaluate(batched)
        expected = setup.evaluate(_per_function(setup))
        for got, want in zip(result, expected, strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_product_entry_with_a_vector_factor_uses_the_per_function_path(
        self, caplog
    ):
        """``p0 = alpha_v * beta`` with alpha_v of shape (3,) is not a scalar entry."""
        product = ProductFunction(name="p0", factors=["alpha_v", "beta"])
        members = [
            _interp("f0", [5], "nom_0", ["p0"]),
            _interp("f1", [5], "nom_1", ["alpha_b"]),
        ]
        setup = _Setup(members, [])
        setup.inputs = [v for v in setup.inputs if v.name != "p0"]
        del setup.values["p0"]
        alpha_v, beta = pt.dvector("alpha_v"), pt.dscalar("beta")
        setup.inputs += [alpha_v, beta]
        setup.context.update(alpha_v=alpha_v, beta=beta)
        setup.values.update(alpha_v=np.array([-1.0, 0.0, 1.0]), beta=0.5)
        setup.context["p0"] = product.expression(setup.context)
        setup.functions["p0"] = product
        with caplog.at_level(logging.DEBUG, logger="pyhs3.functions.standard"):
            batched = _batched(setup)
        assert "'f0' is built on its own" in caplog.text
        result = setup.evaluate(batched)
        expected = setup.evaluate(_per_function(setup))
        for got, want in zip(result, expected, strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_members_without_entries_give_their_nominals(self):
        members = [_interp("f0", [], "nom_0", []), _interp("f1", [], "nom_1", [])]
        setup = _Setup(members, [])
        np.testing.assert_allclose(setup.evaluate(_batched(setup)), _NOMINALS[:2])


class TestBatchedInterpolationGraph:
    @staticmethod
    def _gradient_ops(n_functions):
        params = ["alpha_a", "alpha_b", "alpha_c"]
        members, products = [], []
        for k in range(n_functions):
            prods, names = _product_entries(f"e{k}", 4, params, [1.0, 0.9, 1.1, 1.2])
            products += prods
            members.append(_interp(f"f{k}", [5, 1, 5, 1], f"nom_{k}", names))
        setup = _Setup(members, products)
        outputs = _batched(setup)
        free = [v for v in setup.inputs if v.name in params]
        grads = pt.grad(pt.sum(pt.stack(outputs)), free)
        return io_toposort([], [*outputs, *grads])

    def test_graph_size_does_not_grow_per_function(self):
        """Each extra function adds O(1) nodes, not a copy of the formula."""
        small, large = len(self._gradient_ops(5)), len(self._gradient_ops(50))
        assert (large - small) / 45 <= 6

    def test_gradient_scatters_do_not_grow_per_function(self):
        """Outputs are split, not indexed, so the gradient has no per-function scatter.

        The remaining scatters come from the gather from the parameter vector
        and from slicing it into one block per interpolation code.
        """

        def scatters(n_functions):
            ops = [type(node.op) for node in self._gradient_ops(n_functions)]
            return ops.count(IncSubtensor) + ops.count(AdvancedIncSubtensor1)

        assert scatters(20) == scatters(5)
        assert scatters(5) <= 4


def _planning_inputs():
    """Two channels: f0/f1 feed g0, f2/f3 feed g1; f4 depends on f0 (a member)."""
    functions = Functions(
        [
            _interp("f0", [5], "nom_0", ["alpha_a"]),
            _interp("f1", [1], "nom_1", ["alpha_b"]),
            _interp("f2", [5], "nom_2", ["alpha_c"]),
            _interp("f3", [5], "nom_3", ["alpha_a"]),
            _interp("f4", [5], "nom_4", ["f0"]),
            ProductFunction(name="mean0", factors=["f0", "f1", "f4"]),
            ProductFunction(name="mean1", factors=["f2", "f3"]),
        ]
    )
    distributions = Distributions(
        [
            {
                "name": "g0",
                "type": "gaussian_dist",
                "x": "x0",
                "mean": "mean0",
                "sigma": 1.0,
            },
            {
                "name": "g1",
                "type": "gaussian_dist",
                "x": "x1",
                "mean": "mean1",
                "sigma": 1.0,
            },
        ]
    )
    graph, _, _ = build_dependency_graph(
        ParameterSet(name="p", parameters=[]), functions, distributions
    )
    return graph, functions


class TestPlanFunctionFamilies:
    def test_one_family_per_type_and_channel(self):
        graph, functions = _planning_inputs()
        families = plan_families(graph, functions, ["g0", "g1"])
        members = sorted(sorted(f.members) for f in families)
        assert members == [["f0", "f1"], ["f2", "f3"]]
        assert all(f.kind is InterpolationFunction for f in families)

    def test_members_depend_only_on_their_family(self):
        graph, functions = _planning_inputs()
        families = plan_families(graph, functions, ["g0", "g1"])
        for family in families:
            assert graph[family.name]["type"] == "family"
            for member in family.members:
                assert graph.predecessors(member) == [family.name]
            inputs = set(graph.predecessors(family.name))
            assert {"alpha_a"} <= inputs
            assert not inputs & set(family.members)

    def test_member_of_a_member_is_excluded(self, caplog):
        """f4 reads f0's output, so batching it with f0 would form a cycle."""
        graph, functions = _planning_inputs()
        with caplog.at_level(logging.DEBUG, logger="pyhs3.networks"):
            families = plan_families(graph, functions, ["g0", "g1"])
        assert all("f4" not in f.members for f in families)
        assert "f4" in caplog.text
        graph.topological_sort()  # still acyclic

    def test_functions_feeding_no_listed_root_are_left_alone(self):
        """f2 and f3 feed only g1, which is not a root here."""
        graph, functions = _planning_inputs()
        families = plan_families(graph, functions, ["g0"])
        assert [sorted(f.members) for f in families] == [["f0", "f1"]]
        assert graph.predecessors("f2") != []

    def test_single_instance_is_not_a_family(self):
        """A channel with one batchable function gains nothing from a family."""
        functions = Functions([_interp("f0", [5], "nom_0", ["alpha_a"])])
        distributions = Distributions(
            [
                {
                    "name": "g0",
                    "type": "gaussian_dist",
                    "x": "x0",
                    "mean": "f0",
                    "sigma": 1.0,
                }
            ]
        )
        graph, _, _ = build_dependency_graph(
            ParameterSet(name="p", parameters=[]), functions, distributions
        )
        assert plan_families(graph, functions, ["g0"]) == []
        assert graph.predecessors("f0") != []


class TestFamiliesUnderTheDefaultNLLMode:
    def test_gradient_of_two_families_sharing_parameters(self):
        """compile_nll's default C mode gives the same gradient as Python mode.

        With the Numba rewrite set on the C linker, an in-place IndexedElemwise
        made two families scatter their gradients into one shared buffer,
        doubling the result (fixed by building default_nll_mode from "CVM").
        """
        groups = [
            [_interp(f"g{g}f{k}", [0], f"nom_{g}_{k}", ["alpha_a"]) for k in range(2)]
            for g in range(2)
        ]
        setup = _Setup([m for group in groups for m in group], [])
        outputs = []
        for members in groups:
            batched = InterpolationFunction.batch_expressions(
                members, setup.context, setup.functions
            )
            outputs += [batched[m.name] for m in members]
        free = [v for v in setup.inputs if v.name == "alpha_a"]
        grads = pt.grad(pt.sum(pt.log(pt.stack(outputs))), free)
        args = [setup.values[v.name] for v in setup.inputs]

        def evaluate(mode):
            return pytensor.function(
                setup.inputs, grads, mode=mode, on_unused_input="ignore"
            )(*args)

        np.testing.assert_allclose(
            evaluate(default_nll_mode()), evaluate("FAST_COMPILE"), rtol=1e-10
        )


class TestBatchedInterpolationEdgeCases:
    def test_additive_member_with_contribution_minus_one_has_finite_gradient(self):
        """An additive term of exactly -1 must not leak a 0/0 from the product branch."""
        members = [
            # Code 0 with nominal 10 and high 9: at alpha = 1 the term is exactly -1.
            _interp("f0", [0], "nom_0", ["alpha_a"]),
            _interp("f1", [5], "nom_1", ["alpha_b"]),
        ]
        setup = _Setup(members, [])
        setup.context["nom_0"] = pt.constant(10.0, name="nom_0")
        setup.context["f0_hi_0"] = pt.constant(9.0, name="f0_hi_0")
        setup.values["alpha_a"] = 1.0
        free = [v for v in setup.inputs if v.name in ("alpha_a", "alpha_b")]

        def gradient(outputs):
            return pt.grad(pt.sum(pt.stack(outputs)), free)

        batched = setup.evaluate(gradient(_batched(setup)))
        assert np.all(np.isfinite(batched))
        np.testing.assert_allclose(
            batched, setup.evaluate(gradient(_per_function(setup))), rtol=1e-12
        )

    def test_mixed_but_not_interleaved_kinds_use_the_per_function_path(self, caplog):
        members = [
            _interp("f0", [0, 0, 5], "nom_0", ["alpha_a", "alpha_b", "alpha_c"]),
            _interp("f1", [5], "nom_1", ["alpha_a"]),
        ]
        setup = _Setup(members, [])
        with caplog.at_level(logging.DEBUG, logger="pyhs3.functions.standard"):
            batched = _batched(setup)
        assert "f0" in caplog.text
        np.testing.assert_allclose(
            setup.evaluate(batched), setup.evaluate(_per_function(setup)), rtol=1e-12
        )


class _Scaled(BatchableFunction):
    """A second batchable type for planner tests: ``x * factor``."""

    type: Literal["scaled"] = "scaled"
    x: str
    factor: float

    def _expression(self, context):
        return context[self.x] * self.factor

    @classmethod
    def batch_expressions(cls, members, context, _functions):
        return {m.name: m.expression(context) for m in members}


class TestPlanFamiliesOfSeveralTypes:
    def test_families_of_two_types_never_form_a_cycle(self, caplog):
        """b1 (scaled) feeds a1 (interp) while a2 (interp) feeds b2 (scaled).

        Checked against the original graph neither group has an internal
        dependency, yet the two families together would form the cycle
        family(interp) -> a2 -> family(scaled) -> b1 -> family(interp).  The
        planner checks each group against the graph as already rewired by
        earlier families, so one member stays out instead.
        """
        functions = [
            _Scaled(name="b1", x="alpha_a", factor=2.0),
            _Scaled(name="b2", x="a2", factor=3.0),
            _Scaled(name="b3", x="alpha_b", factor=4.0),
            _interp("a1", [5], "nom_1", ["b1"]),
            _interp("a2", [5], "nom_2", ["alpha_c"]),
            ProductFunction(name="mean", factors=["a1", "b2", "b3"]),
        ]
        distributions = Distributions(
            [
                {
                    "name": "g",
                    "type": "gaussian_dist",
                    "x": "x0",
                    "mean": "mean",
                    "sigma": 1.0,
                }
            ]
        )
        graph, _, _ = build_dependency_graph(
            ParameterSet(name="p", parameters=[]), functions, distributions
        )
        with caplog.at_level(logging.DEBUG, logger="pyhs3.networks"):
            families = plan_families(graph, functions, ["g"])

        graph.topological_sort()  # raises on a cycle
        members = {m for f in families for m in f.members}
        assert not {"a1", "a2", "b1", "b2"} <= members
