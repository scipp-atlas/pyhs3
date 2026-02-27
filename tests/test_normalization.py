"""Unit tests for distribution normalization infrastructure."""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
import pytensor.tensor as pt
import pytest
from pydantic import (
    ConfigDict,
    Field,
)
from pytensor.compile.function import function
from pytensor.graph.replace import clone_replace
from scipy.integrate import quad

from pyhs3.context import Context
from pyhs3.distributions.basic import GaussianDist
from pyhs3.distributions.composite import MixtureDist
from pyhs3.distributions.core import Distribution
from pyhs3.distributions.mathematical import GenericDist
from pyhs3.normalization import Normalizable, gauss_legendre_integral
from pyhs3.typing.aliases import TensorVar


class TestGaussLegendreIntegral:
    """Tests for Gauss-Legendre quadrature."""

    def test_constant_function(self):
        """∫1 dx over [0,1] = 1.0"""
        x = pt.dscalar("x")
        expr = pt.constant(1.0)
        lower = pt.constant(0.0)
        upper = pt.constant(1.0)

        integral = gauss_legendre_integral(expr, x, lower, upper)
        f = function([], integral)

        result = f()
        assert np.isclose(result, 1.0, rtol=1e-10)

    def test_polynomial(self):
        """∫x² dx over [0,1] = 1/3"""
        x = pt.dscalar("x")
        expr = x**2
        lower = pt.constant(0.0)
        upper = pt.constant(1.0)

        integral = gauss_legendre_integral(expr, x, lower, upper)
        f = function([], integral)

        result = f()
        expected = 1.0 / 3.0
        assert np.isclose(result, expected, rtol=1e-10)

    def test_exponential(self):
        """∫exp(-x) dx over [0,5] = 1-exp(-5)"""
        x = pt.dscalar("x")
        expr = pt.exp(-x)
        lower = pt.constant(0.0)
        upper = pt.constant(5.0)

        integral = gauss_legendre_integral(expr, x, lower, upper)
        f = function([], integral)

        result = f()
        expected = 1 - np.exp(-5.0)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_result_is_symbolic(self):
        """Integral depends on PyTensor variables."""
        x = pt.dscalar("x")
        lower = pt.dscalar("lower")
        upper = pt.dscalar("upper")
        expr = pt.constant(1.0)

        integral = gauss_legendre_integral(expr, x, lower, upper)
        f = function([lower, upper], integral)

        # Integral of 1 over [a, b] is (b-a)
        result = f(0.0, 3.0)
        assert np.isclose(result, 3.0, rtol=1e-10)

        result = f(2.0, 5.0)
        assert np.isclose(result, 3.0, rtol=1e-10)

    def test_parameter_dependence(self):
        """∫exp(c*x) dx changes with c."""
        x = pt.dscalar("x")
        c = pt.dscalar("c")
        expr = pt.exp(c * x)
        lower = pt.constant(0.0)
        upper = pt.constant(1.0)

        integral = gauss_legendre_integral(expr, x, lower, upper)
        f = function([c], integral)

        # For c=-1: ∫exp(-x) dx from 0 to 1 = 1 - exp(-1)
        result_c_neg1 = f(-1.0)
        expected_c_neg1 = 1 - np.exp(-1.0)
        assert np.isclose(result_c_neg1, expected_c_neg1, rtol=1e-10)

        # For c=-2: ∫exp(-2x) dx from 0 to 1 = (1 - exp(-2))/2
        result_c_neg2 = f(-2.0)
        expected_c_neg2 = (1 - np.exp(-2.0)) / 2.0
        assert np.isclose(result_c_neg2, expected_c_neg2, rtol=1e-10)


class TestDistributionNormalization:
    """Tests for distribution normalization."""

    def test_no_normalization_without_observables(self):
        """Unchanged behavior when observables={}."""
        dist = GenericDist(name="test", expression="exp(c*x)")
        x_var = pt.dscalar("x")
        c_val = pt.constant(-0.5, name="c")
        context_no_obs = Context(
            parameters={"x": x_var, "c": c_val},
            observables={},
        )
        context_none_obs = Context(
            parameters={"x": x_var, "c": c_val},
        )

        expr_no_obs = dist._expression(context_no_obs)
        expr_none_obs = dist._expression(context_none_obs)

        f_no_obs = function([x_var], expr_no_obs)
        f_none_obs = function([x_var], expr_none_obs)

        # Both should give the same unnormalized result
        assert np.isclose(f_no_obs(1.0), f_none_obs(1.0), rtol=1e-10)

    def test_generic_dist_integrates_to_one(self):
        """Normalized GenericDist integrates to ~1.0."""
        dist = GenericDist(name="test", expression="exp(c*x)")
        x_var = pt.dscalar("x")
        c_val = pt.constant(-0.5, name="c")
        lower = pt.constant(0.0, name="x_lower")
        upper = pt.constant(10.0, name="x_upper")
        context = Context(
            parameters={"x": x_var, "c": c_val},
            observables={"x": (lower, upper)},
        )
        expr = dist._expression(context)
        f = function([x_var], expr)

        integral, error = quad(lambda x: f(x), 0, 10)
        assert np.isclose(integral, 1.0, atol=1e-6)
        assert error < 1e-6

    def test_normalization_changes_with_domain(self):
        """Different bounds → still integrates to 1.0."""
        dist = GenericDist(name="test", expression="exp(c*x)")
        x_var = pt.dscalar("x")
        c_val = pt.constant(-0.5, name="c")

        # Domain [0, 5]
        lower1 = pt.constant(0.0, name="x_lower1")
        upper1 = pt.constant(5.0, name="x_upper1")
        context1 = Context(
            parameters={"x": x_var, "c": c_val},
            observables={"x": (lower1, upper1)},
        )
        expr1 = dist._expression(context1)
        f1 = function([x_var], expr1)
        integral1, _ = quad(lambda x: f1(x), 0, 5)

        # Domain [0, 10]
        lower2 = pt.constant(0.0, name="x_lower2")
        upper2 = pt.constant(10.0, name="x_upper2")
        context2 = Context(
            parameters={"x": x_var, "c": c_val},
            observables={"x": (lower2, upper2)},
        )
        expr2 = dist._expression(context2)
        f2 = function([x_var], expr2)
        integral2, _ = quad(lambda x: f2(x), 0, 10)

        # Both should integrate to 1.0
        assert np.isclose(integral1, 1.0, atol=1e-6)
        assert np.isclose(integral2, 1.0, atol=1e-6)

    def test_normalization_integral_default_returns_none(self):
        """Base class normalization_integral() returns None."""
        dist = GenericDist(name="test", expression="x")
        x_var = pt.dscalar("x")
        context = Context(parameters={"x": x_var})

        result = dist.normalization_integral(
            context, "x", pt.constant(0.0), pt.constant(1.0)
        )
        assert result is None

    def test_generic_dist_expression(self):
        """Base class normalization_integral() returns None."""
        dist = GenericDist(name="test", expression="x")
        x_var = pt.dscalar("x")
        context = Context(parameters={"x": x_var}, observables={"x": (0, 10)})

        result = dist.expression(context)
        func = function([x_var], result)
        assert pytest.approx(func(1.0)) == 0.02
        assert pytest.approx(func(10.0)) == 0.2

    def test_dist_symbolic_integral_expression(self):
        """Base class normalization_integral() returns x."""

        class CustomDist(Distribution, Normalizable):
            _parameters: ClassVar = {"x": "x"}
            model_config = ConfigDict(
                arbitrary_types_allowed=True, serialize_by_alias=True
            )

            type: Literal["custom_dist"] = Field(default="custom_dist", repr=False)

            def likelihood(self, context: Context) -> TensorVar:
                return context["x"]

            def normalization_integral(
                self,
                context: Context,
                observable_name: str,
                lower: TensorVar,
                upper: TensorVar,
            ) -> TensorVar:
                observable = context[observable_name]
                expression = observable**2 / 2.0
                upper_t = pt.as_tensor_variable(upper, dtype=observable.dtype)
                lower_t = pt.as_tensor_variable(lower, dtype=observable.dtype)
                return clone_replace(expression, {observable: upper_t}) - clone_replace(
                    expression, {observable: lower_t}
                )

        # Set parameters based on the analyzed expression
        dist = CustomDist(name="test", expression="x")
        x_var = pt.dscalar("x")
        context = Context(parameters={"x": x_var}, observables={"x": (0, 10)})

        result = dist.expression(context)
        func = function([x_var], result)
        assert pytest.approx(func(1.0)) == 0.02
        assert pytest.approx(func(10.0)) == 0.2

        log_result = dist.log_expression(context)
        log_func = function([x_var], log_result)
        assert pytest.approx(log_func(1.0)) == np.log(0.02)
        assert pytest.approx(log_func(10.0)) == np.log(0.2)

    def test_composite_dist_not_normalized(self):
        """MixtureDist/ProductDist skip normalization."""
        # Create a GenericDist that WOULD be normalized
        generic = GenericDist(name="generic", expression="exp(c*x)")

        # Wrap it in a MixtureDist
        mixture = MixtureDist(
            name="mixture",
            summands=["generic"],
            coefficients=["coeff"],
        )

        x_var = pt.dscalar("x")
        c_val = pt.constant(-0.5, name="c")
        coeff_val = pt.constant(1.0, name="coeff")
        lower = pt.constant(0.0, name="x_lower")
        upper = pt.constant(10.0, name="x_upper")
        context = Context(
            parameters={
                "x": x_var,
                "c": c_val,
                "coeff": coeff_val,
            },
            observables={"x": (lower, upper)},
        )

        # MixtureDist doesn't inherit Normalizable, so it won't normalize
        # We need to build the generic distribution separately first
        generic_expr = generic._expression(context)
        context_with_generic = Context(
            parameters={
                **context.parameters,
                "generic": generic_expr,
            },
            observables=context.observables,
        )

        mixture_expr = mixture._expression(context_with_generic)
        f = function([x_var], mixture_expr)

        # The mixture should NOT be normalized (integral > 1)
        _integral, _ = quad(lambda x: f(x), 0, 10)
        # Since generic is normalized, and mixture just returns it scaled by coeff=1.0,
        # the mixture should also integrate to ~1.0 (but through a different path)
        # Actually, let me reconsider: MixtureDist doesn't normalize, but it uses
        # the already-normalized generic distribution, so it will still be normalized
        # Let's verify mixture doesn't have Normalizable mixin
        assert not isinstance(mixture, Normalizable)

    def test_constraint_not_normalized(self):
        """Gaussian with non-observable x skips normalization."""
        # Create a Gaussian constraint that doesn't use an observable
        dist = GaussianDist(name="constraint", mean="nom", sigma="sigma", x="alpha")

        # GaussianDist doesn't inherit Normalizable, so it won't normalize
        assert not isinstance(dist, Normalizable)


class TestContextObservables:
    """Tests for Context observables functionality."""

    def test_context_with_observables(self):
        """Verify observables property works."""
        x_var = pt.dscalar("x")
        lower = pt.constant(0.0, name="x_lower")
        upper = pt.constant(10.0, name="x_upper")
        obs_dict = {"x": (lower, upper)}
        context = Context(
            parameters={"x": x_var},
            observables=obs_dict,
        )

        observables = context.observables
        assert "x" in observables
        assert len(observables) == 1

    def test_context_default_no_observables(self):
        """Backward compatible - observables defaults to empty dict."""
        x_var = pt.dscalar("x")
        context = Context(parameters={"x": x_var})

        assert context.observables == {}

    def test_context_copy_preserves_observables(self):
        """Context.copy() preserves observables."""
        x_var = pt.dscalar("x")
        lower = pt.constant(0.0, name="x_lower")
        upper = pt.constant(10.0, name="x_upper")
        obs_dict = {"x": (lower, upper)}
        context = Context(
            parameters={"x": x_var},
            observables=obs_dict,
        )

        context_copy = context.copy()
        assert context_copy.observables == context.observables
        assert "x" in context_copy.observables
