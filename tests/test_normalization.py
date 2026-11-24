"""Unit tests for distribution normalization infrastructure."""

from __future__ import annotations

from typing import ClassVar, Literal
from unittest.mock import patch

import numpy as np
import pytensor.tensor as pt
import pytensor_distributions.normal as Normal
import pytest
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
)
from pytensor.compile.maker import function
from pytensor.compile.mode import Mode, get_mode
from pytensor.graph.traversal import io_toposort
from scipy import stats
from scipy.stats import gennorm

from pyhs3.axes import RegularAxis
from pyhs3.context import Context
from pyhs3.distributions.basic import (
    ExponentialDist,
    GaussianDist,
    GenNormalDist,
    UniformDist,
)
from pyhs3.distributions.composite import MixtureDist
from pyhs3.distributions.core import Distribution
from pyhs3.distributions.histfactory import HistFactoryDistChannel
from pyhs3.distributions.mathematical import GenericDist
from pyhs3.distributions.physics import AsymmetricCrystalBallDist
from pyhs3.normalization import gauss_legendre_integral, substitute_variable
from pyhs3.typing.aliases import TensorVar


class TestGaussLegendreIntegral:
    """Tests for Gauss-Legendre quadrature."""

    def test_constant_function(self):
        """∫1 dx over [0,1] = 1.0"""
        x = pt.vector("x")
        expr = pt.constant(1.0)
        lower = pt.constant(0.0)
        upper = pt.constant(1.0)

        integral = gauss_legendre_integral(expr, x, lower, upper)
        f = function([], integral)

        result = f()
        assert np.isclose(result, 1.0, rtol=1e-10)

    def test_polynomial(self):
        """∫x² dx over [0,1] = 1/3"""
        x = pt.vector("x")
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
        x = pt.vector("x")
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
        x = pt.vector("x")
        lower = pt.scalar("lower")
        upper = pt.scalar("upper")
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
        x = pt.vector("x")
        c = pt.scalar("c")
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
        x_var = pt.scalar("x")
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
        x_var = pt.vector("x")
        c_val = pt.constant(-0.5, name="c")
        lower = pt.constant(0.0, name="x_lower")
        upper = pt.constant(10.0, name="x_upper")
        context = Context(
            parameters={"x": x_var, "c": c_val},
            observables={"x": (lower, upper)},
        )
        expr = dist._expression(context)

        compiled = function([x_var], expr)
        xs = np.linspace(0, 10, 10000)
        ys = compiled(xs)
        integral = np.trapezoid(ys, xs)
        assert np.isclose(integral, 1.0, atol=1e-6)

    def test_normalization_changes_with_domain(self):
        """Different bounds → still integrates to 1.0."""
        dist = GenericDist(name="test", expression="exp(c*x)")
        x_var = pt.vector("x")
        c_val = pt.constant(-0.5, name="c")

        # Domain [0, 5]
        lower1 = pt.constant(0.0, name="x_lower1")
        upper1 = pt.constant(5.0, name="x_upper1")
        context1 = Context(
            parameters={"x": x_var, "c": c_val},
            observables={"x": (lower1, upper1)},
        )
        expr1 = dist._expression(context1)
        compiled1 = function([x_var], expr1)
        xs1 = np.linspace(0, 5, 10000)
        ys1 = compiled1(xs1)
        integral1 = np.trapezoid(ys1, xs1)

        # Domain [0, 10]
        lower2 = pt.constant(0.0, name="x_lower2")
        upper2 = pt.constant(10.0, name="x_upper2")
        context2 = Context(
            parameters={"x": x_var, "c": c_val},
            observables={"x": (lower2, upper2)},
        )
        expr2 = dist._expression(context2)
        compiled2 = function([x_var], expr2)
        xs2 = np.linspace(0, 10, 10000)
        ys2 = compiled2(xs2)
        integral2 = np.trapezoid(ys2, xs2)

        # Both should integrate to 1.0
        assert np.isclose(integral1, 1.0, atol=1e-6)
        assert np.isclose(integral2, 1.0, atol=1e-6)

    def test_normalization_expression_default_returns_none(self):
        """Base class normalization_expression() returns None."""
        dist = GenericDist(name="test", expression="x")
        x_var = pt.scalar("x")
        context = Context(parameters={"x": x_var})

        result = dist.normalization_expression(context, "x")
        assert result is None

    def test_generic_dist_expression(self):
        """Base class normalization_integral() returns None."""
        dist = GenericDist(name="test", expression="x")
        x_var = pt.vector("x")
        context = Context(parameters={"x": x_var}, observables={"x": (0, 10)})

        result = dist.expression(context)
        func = function([x_var], result)
        assert pytest.approx(func(np.array([1.0]))[0]) == 0.02
        assert pytest.approx(func(np.array([10.0]))[0]) == 0.2

    def test_dist_symbolic_integral_expression(self):
        """Custom distribution with analytical normalization expression."""

        class CustomDist(Distribution):
            _parameters: ClassVar = {"x": "x"}
            model_config = ConfigDict(
                arbitrary_types_allowed=True, serialize_by_alias=True
            )

            type: Literal["custom_dist"] = Field(default="custom_dist", repr=False)

            def likelihood(self, context: Context) -> TensorVar:
                return context["x"]

            def normalization_expression(
                self, context: Context, observable_name: str
            ) -> TensorVar:
                observable = context[observable_name]
                return observable**2 / 2.0

        # Set parameters based on the analyzed expression
        dist = CustomDist(name="test", expression="x")
        x_var = pt.vector("x")
        context = Context(parameters={"x": x_var}, observables={"x": (0, 10)})

        result = dist.expression(context)
        func = function([x_var], result)
        assert pytest.approx(func(np.array([1.0]))[0]) == 0.02
        assert pytest.approx(func(np.array([10.0]))[0]) == 0.2

        log_result = dist.log_expression(context)
        log_func = function([x_var], log_result)
        assert pytest.approx(log_func(np.array([1.0]))[0]) == np.log(0.02)
        assert pytest.approx(log_func(np.array([10.0]))[0]) == np.log(0.2)

    def test_normalizable_false_skips_log_expression_normalization(self):
        """_normalizable = False: log_expression() skips normalization entirely,
        even with a matching observable in context -- it must not evaluate
        likelihood() a second time or attempt to build a normalization integral."""

        class NonNormalizableDist(Distribution):
            _parameters: ClassVar = {"x": "x"}
            _normalizable: bool = PrivateAttr(default=False)
            model_config = ConfigDict(
                arbitrary_types_allowed=True, serialize_by_alias=True
            )

            type: Literal["non_normalizable_dist"] = Field(
                default="non_normalizable_dist", repr=False
            )

            def likelihood(self, context: Context) -> TensorVar:
                return context["x"]

        dist = NonNormalizableDist(name="test", expression="x")
        x_var = pt.vector("x")
        # An observable matching "x" is present, but _normalizable=False must
        # short-circuit before any normalization machinery is invoked.
        context = Context(parameters={"x": x_var}, observables={"x": (0, 10)})

        log_result = dist.log_expression(context)
        log_func = function([x_var], log_result)
        assert pytest.approx(log_func(np.array([2.0]))[0]) == np.log(2.0)

    def test_composite_dist_not_normalized(self):
        """MixtureDist uses components but doesn't re-normalize."""
        # Create a GenericDist that gets normalized
        generic = GenericDist(name="generic", expression="exp(c*x)")

        # Wrap it in a MixtureDist
        mixture = MixtureDist(
            name="mixture",
            summands=["generic"],
            coefficients=["coeff"],
        )

        x_var = pt.vector("x")
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

        # Build the generic distribution separately first
        generic_expr = generic._expression(context)
        context_with_generic = Context(
            parameters={
                **context.parameters,
                "generic": generic_expr,
            },
            observables=context.observables,
        )

        mixture_expr = mixture._expression(context_with_generic)
        compiled = function([x_var], mixture_expr)

        # Since generic is normalized and mixture just returns it scaled by coeff=1.0,
        # the mixture should also integrate to ~1.0
        xs = np.linspace(0, 10, 10000)
        ys = compiled(xs)
        integral = np.trapezoid(ys, xs)
        assert np.isclose(integral, 1.0, atol=1e-6)

    def test_constraint_not_normalized(self):
        """Gaussian with non-observable x skips normalization."""
        # Create a Gaussian constraint that doesn't use an observable
        dist = GaussianDist(name="constraint", mean="nom", sigma="sigma", x="alpha")

        alpha_var = pt.vector("alpha")
        nom_val = pt.constant(1.0, name="nom")
        sigma_val = pt.constant(0.1, name="sigma")
        x_lower = pt.constant(0.0, name="x_lower")
        x_upper = pt.constant(10.0, name="x_upper")

        # Observable is "x" but the distribution uses "alpha"
        context = Context(
            parameters={"alpha": alpha_var, "nom": nom_val, "sigma": sigma_val},
            observables={"x": (x_lower, x_upper)},
        )

        # Since "alpha" is not in observables, normalization is skipped
        expr = dist._expression(context)
        compiled = function([alpha_var], expr)

        # Verify it's not normalized - the Gaussian doesn't integrate to 1
        # over all alpha values (it's a proper Gaussian PDF without domain restriction)
        alphas = np.linspace(-10, 10, 10000)
        ys = compiled(alphas)
        integral = np.trapezoid(ys, alphas)
        # A Gaussian with sigma=0.1 has very high peak, integral over infinite domain is 1
        # But over finite domain and normalized to [0,10] for x (which doesn't apply),
        # it should just be the standard Gaussian
        assert np.isclose(integral, 1.0, atol=1e-3)

    def test_gaussian_dist_integrates_to_one(self):
        """GaussianDist normalized over finite domain integrates to 1.0."""
        dist = GaussianDist(name="gauss", mean="mu", sigma="sigma", x="x")

        x_var = pt.vector("x")
        mu_val = pt.constant(130.0, name="mu")
        sigma_val = pt.constant(10.0, name="sigma")
        lower = pt.constant(100.0, name="x_lower")
        upper = pt.constant(160.0, name="x_upper")

        context = Context(
            parameters={"x": x_var, "mu": mu_val, "sigma": sigma_val},
            observables={"x": (lower, upper)},
        )

        expr = dist._expression(context)
        compiled = function([x_var], expr)

        xs = np.linspace(100, 160, 10000)
        ys = compiled(xs)
        integral = np.trapezoid(ys, xs)
        assert np.isclose(integral, 1.0, atol=1e-6)

    def test_exponential_dist_integrates_to_one(self):
        """ExponentialDist normalized over finite domain integrates to 1.0."""
        dist = ExponentialDist(name="exp", c="c", x="x")

        x_var = pt.vector("x")
        c_val = pt.constant(0.5, name="c")
        lower = pt.constant(0.0, name="x_lower")
        upper = pt.constant(10.0, name="x_upper")

        context = Context(
            parameters={"x": x_var, "c": c_val},
            observables={"x": (lower, upper)},
        )

        expr = dist._expression(context)
        compiled = function([x_var], expr)

        xs = np.linspace(0, 10, 10000)
        ys = compiled(xs)
        integral = np.trapezoid(ys, xs)
        assert np.isclose(integral, 1.0, atol=1e-6)

    def test_uniform_dist_integrates_to_one(self):
        """UniformDist over a finite domain integrates to 1.0 and equals
        ``1/(upper-lower)`` everywhere on its support.

        HS3 ``uniform_dist`` declares only ``name``/``type``/``x`` with PDF
        ``1/M``, where ``M`` is the domain measure. pyhs3 delegates to
        ``pytensor_distributions.uniform``, which returns the self-normalized
        ``1/(upper-lower)`` on the support (and 0 outside), deriving the bounds
        from the observable domain with no ``lower``/``upper`` distribution
        parameters.
        """
        dist = UniformDist(name="uniform", x=["x"])

        x_var = pt.vector("x", dtype="float64")
        lower = pt.constant(2.0, name="x_lower", dtype="float64")
        upper = pt.constant(7.0, name="x_upper", dtype="float64")

        context = Context(
            parameters={"x": x_var},
            observables={"x": (lower, upper)},
        )

        # The self-normalized density broadcasts to x's shape: 1/(upper-lower)
        # on the support, 0 outside it. Evaluated on a length-3 grid, every
        # on-support point equals 1/(7-2) = 0.2.
        expr = dist._expression(context)
        compiled = function([x_var], expr)
        density = compiled(np.array([2.0, 4.5, 7.0]))

        expected_density = 1.0 / (7.0 - 2.0)
        np.testing.assert_allclose(density, expected_density, rtol=1e-10)

        # Integrating the density over [2, 7] recovers 1.0.
        xs = np.linspace(2.0, 7.0, 10000)
        ys = compiled(xs)
        integral = np.trapezoid(ys, xs)
        assert np.isclose(integral, 1.0, atol=1e-6)

    def test_histfactory_not_normalizable(self):
        """Verify HistFactoryDistChannel._normalizable is False."""
        dist = HistFactoryDistChannel(
            name="channel",
            axes=[RegularAxis(name="x", min=0, max=10, nbins=10)],
            samples=[],
        )

        assert dist._normalizable is False

    def test_histfactory_normalization_skipped(self):
        """Use mock to verify normalization is NOT called for HistFactory."""
        dist = HistFactoryDistChannel(
            name="channel",
            axes=[RegularAxis(name="x", min=0, max=10, nbins=10)],
            samples=[],
        )

        x_var = pt.dvector("x")
        channel_observed = pt.dvector("channel_observed")
        lower = pt.constant(0.0, name="x_lower")
        upper = pt.constant(10.0, name="x_upper")

        context = Context(
            parameters={"x": x_var, "channel_observed": channel_observed},
            observables={"x": (lower, upper)},
        )

        # Patch gauss_legendre_integral to verify it's not called
        with patch("pyhs3.distributions.core.gauss_legendre_integral") as mock_gl:
            _expr = dist._expression(context)
            # gauss_legendre_integral should NOT be called
            mock_gl.assert_not_called()


class TestContextObservables:
    """Tests for Context observables functionality."""

    def test_context_with_observables(self):
        """Verify observables property works."""
        x_var = pt.vector("x")
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
        x_var = pt.scalar("x")
        context = Context(parameters={"x": x_var})

        assert context.observables == {}

    def test_context_copy_preserves_observables(self):
        """Context.copy() preserves observables."""
        x_var = pt.vector("x")
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


def _no_fusion_mode() -> Mode:
    """FAST_RUN rewrites without elemwise fusion, on the C linker.

    Without fusion every Elemwise checks its inputs' static broadcastable
    flags, so a length-1 axis typed as unknown fails at runtime.
    """
    return Mode(
        linker="cvm",
        optimizer=get_mode("FAST_RUN").excluding("fusion").provided_optimizer,
    )


def _observable_context(params: dict[str, TensorVar]) -> tuple[Context, TensorVar]:
    """Context with an unbinned observable ``x`` on [-5, 5] seen as an (N, 1) view."""
    x = pt.dvector("x")
    context = Context(
        parameters={"x": x, **params},
        observables={"x": (pt.constant(-5.0), pt.constant(5.0))},
        views={"x": x[:, None]},
    )
    return context, x


_ANALYTIC = [
    pytest.param(
        GenNormalDist(name="d", x="x", mean="mu", alpha="alpha", beta="beta"),
        {"mu": 0.3, "alpha": 1.2, "beta": 1.7},
        id="generalized_normal",
    ),
    pytest.param(
        AsymmetricCrystalBallDist(
            name="d",
            m="x",
            m0="mu",
            sigma_L="sL",
            sigma_R="sR",
            alpha_L="aL",
            alpha_R="aR",
            n_L="nL",
            n_R="nR",
        ),
        {"mu": 0.2, "sL": 1.1, "sR": 0.9, "aL": 1.4, "aR": 1.6, "nL": 3.0, "nR": 4.0},
        id="crystalball_doublesided",
    ),
]


class TestNormalizationStaticShapes:
    """Normalization integrals carry the static shape of the bounds, not of x."""

    @pytest.mark.parametrize(("dist", "values"), _ANALYTIC)
    def test_analytic_integral_is_broadcastable(self, dist, values):
        """F(upper) - F(lower) has only length-1 axes, statically known."""
        params = {name: pt.dscalar(name) for name in values}
        context, _ = _observable_context(params)
        lower, upper = context.observables["x"]

        integral = dist._normalization_integral(context, "x", lower, upper)

        assert integral is not None
        assert all(size == 1 for size in integral.type.shape)

    def test_batched_parameter_integral_has_static_unit_event_axis(self):
        """A parameter batch of M gives a (1, M) integral and an (N, M) density.

        Shapes: ``mu`` is (M,), the observable view is (N, 1), so the density is
        (N, M).  The integral substitutes a (1,) bound for x, so it is (1, M)
        with a statically length-1 event axis that broadcasts against N.
        """
        n_events, n_batch = 5, 3
        mu = pt.dvector("mu")
        params = {"mu": mu, "alpha": pt.dscalar("alpha"), "beta": pt.dscalar("beta")}
        context, x = _observable_context(params)
        lower, upper = context.observables["x"]
        dist = GenNormalDist(name="d", x="x", mean="mu", alpha="alpha", beta="beta")

        integral = dist._normalization_integral(context, "x", lower, upper)
        density = dist.expression(context)

        assert integral is not None
        assert integral.type.shape == (1, None)
        events = np.linspace(-1.0, 1.0, n_events)
        mus = np.array([0.1, 0.2, 0.3])
        alpha, beta = 1.2, 1.7
        inputs = [x, mu, params["alpha"], params["beta"]]
        integral_val = function(inputs[1:], integral)(mus, alpha, beta)
        density_val = function(inputs, density)(events, mus, alpha, beta)

        assert integral_val.shape == (1, n_batch)
        assert density_val.shape == (n_events, n_batch)
        expected_norm = gennorm.cdf(5.0, beta, loc=mus, scale=alpha) - gennorm.cdf(
            -5.0, beta, loc=mus, scale=alpha
        )
        np.testing.assert_allclose(integral_val[0], expected_norm, rtol=1e-10)
        expected_density = (
            gennorm.pdf(events[:, None], beta, loc=mus, scale=alpha) / expected_norm
        )
        np.testing.assert_allclose(density_val, expected_density, rtol=1e-10)

    @pytest.mark.parametrize(("dist", "values"), _ANALYTIC)
    def test_analytic_integral_shares_observable_independent_nodes(self, dist, values):
        """F(upper) and F(lower) share the antiderivative's parameter-only nodes.

        Only the nodes downstream of x are rebuilt per bound, so the integral
        has n_free + 2 * n_dependent + 1 nodes (the 1 is the subtraction)
        rather than two full copies of the antiderivative.
        """
        params = {name: pt.dscalar(name) for name in values}
        context, x = _observable_context(params)
        lower, upper = context.observables["x"]

        antiderivative = dist.normalization_expression(context, "x")
        depends_on_x = {x}
        n_free = n_dependent = 0
        for node in io_toposort([], [antiderivative]):
            if any(var in depends_on_x for var in node.inputs):
                depends_on_x.update(node.outputs)
                n_dependent += 1
            else:
                n_free += 1

        integral = dist._normalization_integral(context, "x", lower, upper)

        assert n_free > 0
        assert len(io_toposort([], [integral])) == n_free + 2 * n_dependent + 1

    @pytest.mark.parametrize(("dist", "values"), _ANALYTIC)
    def test_analytic_log_expression_evaluates_without_fusion(self, dist, values):
        """The normalized log-pdf over N > 1 events runs without elemwise fusion."""
        params = {name: pt.dscalar(name) for name in values}
        context, x = _observable_context(params)
        events = np.linspace(-4.0, 4.0, 7)

        log_pdf = dist.log_expression(context)
        inputs = [x, *params.values()]
        args = [events, *values.values()]
        unfused = function(inputs, log_pdf, mode=_no_fusion_mode())(*args)
        fused = function(inputs, log_pdf, mode="FAST_RUN")(*args)

        assert unfused.shape == (7, 1)
        np.testing.assert_allclose(unfused, fused, rtol=1e-12)

    def test_quadrature_log_expression_evaluates_without_fusion(self):
        """The Gauss-Legendre fallback also runs without elemwise fusion."""
        params = {"c": pt.dscalar("c")}
        context, x = _observable_context(params)
        dist = ExponentialDist(name="e", x="x", c="c")

        log_pdf = dist.log_expression(context)
        unfused = function([x, *params.values()], log_pdf, mode=_no_fusion_mode())(
            np.linspace(0.5, 4.0, 7), 0.5
        )

        assert unfused.shape == (7, 1)
        assert np.all(np.isfinite(unfused))

    def test_mixture_of_analytic_summands_evaluates_without_fusion(self):
        """A mixture subtracting log Z per summand matches the fused result."""
        values = {
            "mu": 0.2,
            "sL": 1.1,
            "sR": 0.9,
            "aL": 1.4,
            "aR": 1.6,
            "nL": 3.0,
            "nR": 4.0,
            "c1": 3.0,
            "c2": 1.5,
            "beta": 1.7,
            "alpha": 1.2,
        }
        params = {name: pt.dscalar(name) for name in values}
        context, x = _observable_context(params)
        summands = {
            "cb": AsymmetricCrystalBallDist(
                name="cb",
                m="x",
                m0="mu",
                sigma_L="sL",
                sigma_R="sR",
                alpha_L="aL",
                alpha_R="aR",
                n_L="nL",
                n_R="nR",
            ),
            "gn": GenNormalDist(
                name="gn", x="x", mean="mu", alpha="alpha", beta="beta"
            ),
        }
        log_terms = [
            pt.log(params[c]) + summands[s].log_expression(context)
            for s, c in (("cb", "c1"), ("gn", "c2"))
        ]
        mixture = pt.logsumexp(pt.stack(log_terms), axis=0)

        inputs = [x, *params.values()]
        args = [np.linspace(-4.0, 4.0, 7), *values.values()]
        unfused = function(inputs, mixture, mode=_no_fusion_mode())(*args)
        fused = function(inputs, mixture, mode="FAST_RUN")(*args)

        np.testing.assert_allclose(unfused, fused, rtol=1e-12)


class TestSubstituteVariable:
    """substitute_variable rebuilds only what depends on the replaced variable."""

    def test_downstream_types_are_reinferred(self):
        """Replacing an unknown-length vector by a length-1 one makes the result broadcastable."""
        x = pt.dvector("x")
        s = pt.dscalar("s")
        expr = pt.exp(x[:, None] * s)  # (?, 1)

        result = substitute_variable(expr, x, pt.as_tensor_variable([3.0]))

        assert result.type.shape == (1, 1)
        np.testing.assert_allclose(function([s], result)(0.5), [[np.exp(1.5)]])

    def test_independent_nodes_are_shared(self):
        """Nodes that do not depend on the replaced variable are reused, not copied."""
        x = pt.dvector("x")
        s = pt.dscalar("s")
        scale = pt.exp(s) * 2.0  # independent of x
        expr = x * scale

        result = substitute_variable(expr, x, pt.as_tensor_variable([3.0]))

        # x * scale broadcasts scale through an ExpandDims node, which is
        # independent of x and therefore the very same node in both graphs.
        assert result.owner.inputs[1] is expr.owner.inputs[1]

    def test_variable_not_in_graph_returns_output(self):
        """Without the variable in the graph the output is returned unchanged."""
        x = pt.dvector("x")
        s = pt.dscalar("s")
        expr = pt.exp(s)

        assert substitute_variable(expr, x, pt.as_tensor_variable([3.0])) is expr


class TestGaussianAnalyticNormalization:
    """GaussianDist normalizes with the exact CDF difference, not quadrature."""

    @staticmethod
    def _integral_functions():
        """Compile (lower, upper, mu, sigma) -> normalization integral and its gradients."""
        lower, upper = pt.dscalar("lower"), pt.dscalar("upper")
        mu, sigma = pt.dscalar("mu"), pt.dscalar("sigma")
        x = pt.dvector("x")
        context = Context(
            parameters={"x": x, "mu": mu, "sigma": sigma},
            observables={"x": (lower, upper)},
            views={"x": x[:, None]},
        )
        dist = GaussianDist(name="g", x="x", mean="mu", sigma="sigma")
        integral = dist._normalization_integral(context, "x", lower, upper)
        assert integral is not None
        # shape (1, 1) for scalar parameters: one bound pair, one parameter point
        scalar = integral[0, 0]
        inputs = [lower, upper, mu, sigma]
        return function(inputs, scalar), function(inputs, pt.grad(scalar, [mu, sigma]))

    @pytest.mark.parametrize("dependent", ["mean", "sigma"])
    def test_parameter_depending_on_the_observable_defers_to_quadrature(
        self, dependent
    ):
        """The CDF difference is only the integral when mean and sigma are constant in x.

        A mean or sigma built from the observable (shape (N, 1), like its view)
        varies with the integration variable, so both analytic entry points
        return None and the base class integrates by quadrature instead.
        """
        lower, upper = pt.dscalar("lower"), pt.dscalar("upper")
        x = pt.dvector("x")
        view = x[:, None]
        mean = 1.0 + 0.1 * view if dependent == "mean" else pt.dscalar("mu")
        sigma = 1.0 + 0.05 * view**2 if dependent == "sigma" else pt.dscalar("s")
        context = Context(
            parameters={"x": x, "mu": mean, "s": sigma},
            observables={"x": (lower, upper)},
            views={"x": view},
        )
        dist = GaussianDist(name="g", x="x", mean="mu", sigma="s")

        assert dist.normalization_expression(context, "x") is None
        assert dist._normalization_integral(context, "x", lower, upper) is None

    def test_normalization_expression_is_provided(self):
        """The antiderivative exists for x and defers to quadrature otherwise."""
        context, _ = _observable_context({"mu": pt.dscalar("mu"), "s": pt.dscalar("s")})
        dist = GaussianDist(name="g", x="x", mean="mu", sigma="s")

        assert dist.normalization_expression(context, "x") is not None
        assert dist.normalization_expression(context, "mu") is None

    @pytest.mark.parametrize(
        ("lower", "upper", "mu", "sigma"),
        [
            pytest.param(-5.0, 5.0, 0.0, 1.0, id="symmetric"),
            pytest.param(-1.0, 4.0, 0.3, 1.7, id="asymmetric"),
            pytest.param(100.0, 160.0, 125.0, 2.5, id="wide-window"),
            pytest.param(30.0, 31.0, 0.0, 1.0, id="right-far-tail"),
            pytest.param(-31.0, -30.0, 0.0, 1.0, id="left-far-tail"),
            pytest.param(8.0, 9.5, 1.0, 0.5, id="right-far-tail-scaled"),
        ],
    )
    def test_integral_matches_scipy_cdf(self, lower, upper, mu, sigma):
        """F(upper) - F(lower) equals the scipy CDF difference to 1e-10 relative."""
        value, _ = self._integral_functions()
        z_lo, z_hi = (lower - mu) / sigma, (upper - mu) / sigma
        # sf on the right tail, cdf on the left: each stays accurate where the
        # naive cdf(b) - cdf(a) cancels catastrophically
        expected = (
            stats.norm.sf(z_lo) - stats.norm.sf(z_hi)
            if z_lo + z_hi > 0
            else stats.norm.cdf(z_hi) - stats.norm.cdf(z_lo)
        )

        np.testing.assert_allclose(
            value(lower, upper, mu, sigma), expected, rtol=1e-10, atol=0
        )

    @pytest.mark.parametrize(
        ("lower", "upper", "mu", "sigma"),
        [(-5.0, 5.0, 0.0, 1.0), (-1.0, 4.0, 0.3, 1.7), (3.0, 9.0, 1.0, 2.0)],
    )
    def test_integral_matches_quadrature(self, lower, upper, mu, sigma):
        """The analytic integral agrees with the Gauss-Legendre fallback to 1e-10."""
        x = pt.dvector("x")
        mu_t, sigma_t = pt.dscalar("mu"), pt.dscalar("sigma")
        quadrature = gauss_legendre_integral(
            Normal.pdf(x, mu_t, sigma_t), x, pt.constant(lower), pt.constant(upper)
        )
        value, _ = self._integral_functions()

        np.testing.assert_allclose(
            value(lower, upper, mu, sigma),
            function([mu_t, sigma_t], quadrature)(mu, sigma).item(),
            rtol=1e-10,
        )

    @pytest.mark.parametrize(
        ("lower", "upper", "mu", "sigma"),
        [
            (-5.0, 5.0, 0.3, 1.2),
            (-1.0, 4.0, 0.3, 1.7),
            (3.0, 9.0, 1.0, 2.0),
            (-9.0, -3.0, 1.0, 2.0),
        ],
    )
    def test_gradient_matches_finite_differences(self, lower, upper, mu, sigma):
        """d(integral)/d(mu, sigma) agrees with central differences to rtol 1e-6."""
        value, grads = self._integral_functions()
        h = 1e-6
        d_mu = (
            value(lower, upper, mu + h, sigma) - value(lower, upper, mu - h, sigma)
        ) / (2 * h)
        d_sigma = (
            value(lower, upper, mu, sigma + h) - value(lower, upper, mu, sigma - h)
        ) / (2 * h)

        got_mu, got_sigma = grads(lower, upper, mu, sigma)
        np.testing.assert_allclose(got_mu, d_mu, rtol=1e-6)
        np.testing.assert_allclose(got_sigma, d_sigma, rtol=1e-6)

    def test_far_tail_density_matches_truncated_normal(self):
        """A window 30 sigma out normalizes: the integral is tiny but nonzero."""
        params = {"mu": pt.dscalar("mu"), "sigma": pt.dscalar("sigma")}
        x = pt.dvector("x")
        context = Context(
            parameters={"x": x, **params},
            observables={"x": (pt.constant(30.0), pt.constant(31.0))},
            views={"x": x[:, None]},
        )
        dist = GaussianDist(name="g", x="x", mean="mu", sigma="sigma")
        events = np.linspace(30.0, 31.0, 2001)

        pdf = function([x, *params.values()], dist._expression(context))(
            events, 0.0, 1.0
        )

        assert np.all(np.isfinite(pdf))
        expected = stats.norm.pdf(events) / (stats.norm.sf(30.0) - stats.norm.sf(31.0))
        np.testing.assert_allclose(pdf[:, 0], expected, rtol=1e-10)


class TestGenNormalDependentShape:
    """GenNormalDist's CDF difference is the integral only for parameters constant in x."""

    @pytest.mark.parametrize("dependent", ["mean", "alpha", "beta"])
    def test_parameter_depending_on_the_observable_defers_to_quadrature(
        self, dependent
    ):
        """A parameter built from the observable (shape (N, 1), like its view)
        makes both analytic entry points return None, so the base class
        integrates by quadrature instead."""
        lower, upper = pt.dscalar("lower"), pt.dscalar("upper")
        x = pt.dvector("x")
        view = x[:, None]
        values = {
            "mu": 1.0 + 0.1 * view if dependent == "mean" else pt.dscalar("mu"),
            "alpha": 1.0 + 0.05 * view**2
            if dependent == "alpha"
            else pt.dscalar("alpha"),
            "beta": 2.0 + 0.1 * view**2 if dependent == "beta" else pt.dscalar("beta"),
        }
        context = Context(
            parameters={"x": x, **values},
            observables={"x": (lower, upper)},
            views={"x": view},
        )
        dist = GenNormalDist(name="d", x="x", mean="mu", alpha="alpha", beta="beta")

        assert dist.normalization_expression(context, "x") is None
        assert dist._normalization_integral(context, "x", lower, upper) is None

    def test_independent_parameters_keep_the_analytic_integral(self):
        lower, upper = pt.dscalar("lower"), pt.dscalar("upper")
        x = pt.dvector("x")
        context = Context(
            parameters={
                "x": x,
                "mu": pt.dscalar("mu"),
                "alpha": pt.dscalar("alpha"),
                "beta": pt.dscalar("beta"),
            },
            observables={"x": (lower, upper)},
            views={"x": x[:, None]},
        )
        dist = GenNormalDist(name="d", x="x", mean="mu", alpha="alpha", beta="beta")

        assert dist.normalization_expression(context, "x") is not None
