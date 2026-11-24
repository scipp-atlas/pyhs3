"""
Basic HS3 Distribution implementations.

Provides classes for handling basic probability distributions including
Gaussian, Uniform, Poisson, Exponential, Log-Normal, and Landau distributions
as defined in the HS3 specification.
"""

from __future__ import annotations

import math
from typing import Literal, cast

import pytensor.tensor as pt
import pytensor_distributions.exponential as Exponential
import pytensor_distributions.lognormal as LogNormal
import pytensor_distributions.normal as Normal
import pytensor_distributions.poisson as Poisson
import pytensor_distributions.uniform as Uniform
from pydantic import PrivateAttr

from pyhs3.context import Context
from pyhs3.distributions.core import Distribution
from pyhs3.normalization import substitute_variable
from pyhs3.typing.aliases import TensorVar


class GaussianDist(Distribution):
    r"""
    Gaussian (normal) probability distribution.

    Implements the standard Gaussian probability density function:

    .. math::

        f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)

    Parameters:
        mean (str): Parameter name for the mean (μ).
        sigma (str): Parameter name for the standard deviation (sigma).
        x (str): Input variable name.

    HS3 Reference:
        :ref:`hs3:hs3.gaussiannormal-distribution`
    """

    type: Literal["gaussian_dist"] = "gaussian_dist"
    mean: str | float | int
    sigma: str | float | int
    x: str | float | int

    def likelihood(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the Gaussian PDF.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the Gaussian PDF.
        """
        return cast(
            TensorVar,
            Normal.pdf(
                context[self._parameters["x"]],
                context[self._parameters["mean"]],
                context[self._parameters["sigma"]],
            ),
        )

    def log_likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the Gaussian log-PDF.

        Delegates to pytensor-distributions' analytic log form of
        :meth:`likelihood`:

        .. math::

            \log f(x; \mu, \sigma) = -\frac{1}{2}z^2 - \log\sigma - \frac{1}{2}\log(2\pi),
            \quad z = \frac{x-\mu}{\sigma}

        Evaluating this directly (rather than ``pt.log(self.likelihood(...))``)
        avoids computing :math:`\exp(-z^2/2)` and re-logging it, which
        underflows to 0.0 (and then to ``-inf``) once :math:`|z|` exceeds
        roughly 38 in float64.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the Gaussian log-PDF.
        """
        return cast(
            TensorVar,
            Normal.logpdf(
                context[self._parameters["x"]],
                context[self._parameters["mean"]],
                context[self._parameters["sigma"]],
            ),
        )

    def normalization_expression(
        self, context: Context, observable_name: str
    ) -> TensorVar | None:
        r"""
        Analytic antiderivative of the Gaussian PDF.

        The antiderivative of the fully normalized :meth:`likelihood` density is
        the centered CDF

        .. math::

            F(x) = \frac{1}{2}\,\operatorname{erf}\!\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)

        so that :math:`F(+\infty) - F(-\infty) = 1`. Returns None (deferring to
        quadrature) when the matching observable is not this distribution's
        ``x``, since the antiderivative is only defined with respect to ``x``,
        and when ``mean`` or ``sigma`` depends on that observable, since the
        CDF difference then is not the integral.

        ``_normalization_integral`` does not subtract this expression at
        the two bounds: far from :math:`\mu` both values approach
        :math:`\pm 1/2` and the difference loses all of its significant digits.

        Args:
            context: Mapping of names to pytensor variables.
            observable_name: Name of the observable to integrate over.

        Returns:
            Symbolic antiderivative expression, or None for numerical fallback.
        """
        if observable_name != self._parameters["x"] or self._depends_on_observable(
            context, observable_name, ("mean", "sigma")
        ):
            return None

        x = context[self._parameters["x"]]
        mean = context[self._parameters["mean"]]
        sigma = context[self._parameters["sigma"]]
        return cast(TensorVar, 0.5 * pt.erf((x - mean) / (sigma * math.sqrt(2.0))))

    def _normalization_integral(
        self, context: Context, obs_name: str, lower: TensorVar, upper: TensorVar
    ) -> TensorVar | None:
        r"""
        Exact Gaussian probability between ``lower`` and ``upper``.

        Equals :math:`F(\text{upper}) - F(\text{lower})` for the antiderivative
        of :meth:`normalization_expression`, written with ``erfc`` on the tail
        side of the window so that no two nearly equal numbers are subtracted.
        With :math:`z = (x-\mu)/(\sigma\sqrt{2})`, the window is mirrored about
        :math:`\mu` so that its midpoint lies on the :math:`z > 0` side, where

        .. math::

            \frac{1}{2}\left[\operatorname{erfc}(z_\text{lo}) - \operatorname{erfc}(z_\text{hi})\right]

        keeps its relative precision for any distance from :math:`\mu` (the
        difference of two ``erf`` values underflows to 0 once both bounds are
        roughly :math:`6\sigma` out on one side, which would make the
        normalized density infinite or NaN).

        The bounds are substituted into the observable's view, so the result
        has the same static shape as the base-class integral: the view's shape
        with every axis of length 1.

        Returns None, deferring to quadrature, under the same conditions as
        :meth:`normalization_expression`.

        Args:
            context: Mapping of names to pytensor variables.
            obs_name: Name of the observable to integrate over.
            lower: Lower integration bound.
            upper: Upper integration bound.

        Returns:
            Symbolic integral expression, or None for numerical fallback.
        """
        if obs_name != self._parameters["x"] or self._depends_on_observable(
            context, obs_name, ("mean", "sigma")
        ):
            return None

        leaf = context.parameters[obs_name]
        view = context[self._parameters["x"]]
        mean = context[self._parameters["mean"]]
        sigma = context[self._parameters["sigma"]]
        scale = sigma * math.sqrt(2.0)

        # Bounds enter in the view's shape, e.g. (1, 1) for an (N, 1) view.
        lower_t = pt.as_tensor_variable([lower], dtype=leaf.dtype)
        upper_t = pt.as_tensor_variable([upper], dtype=leaf.dtype)
        z_lower = (substitute_variable(view, leaf, lower_t) - mean) / scale
        z_upper = (substitute_variable(view, leaf, upper_t) - mean) / scale

        # Mirror about mu when the window's midpoint is below it, so that both
        # arguments of erfc sit on the right tail, where erfc is accurate.
        mirrored = (z_lower + z_upper) < 0
        z_near = pt.switch(mirrored, -z_upper, z_lower)
        z_far = pt.switch(mirrored, -z_lower, z_upper)
        return cast(TensorVar, 0.5 * (pt.erfc(z_near) - pt.erfc(z_far)))


class UniformDist(Distribution):
    r"""
    Uniform (rectangular) probability distribution.

    Implements the continuous uniform probability density function with constant
    density over its support region, as defined in ROOT's RooUniform. The support
    of each axis is the observable's domain, so the density is the product over
    axes of the per-axis reciprocal measure:

    .. math::

        f(x_1, \ldots, x_D) = \prod_{i=1}^{D} \frac{1}{u_i - l_i}

    where :math:`(l_i, u_i)` are the domain bounds of axis :math:`x_i`.

    Delegates to ``pytensor_distributions.uniform``, which already returns the
    self-normalized :math:`1/(u_i - l_i)` on the support, so the distribution
    opts out of pyhs3's own normalization (``_normalizable = False``) to avoid
    dividing by the domain measure a second time. Because normalization is opted
    out, a multivariate ``x`` does not hit the single-observable normalization
    limit (see https://github.com/scipp-atlas/pyhs3/issues/214).

    Parameters:
        x (list[str]): Input variable names (one per axis of the box).

    Note:
        The bounds are derived from the observable domain, not from distribution
        parameters, matching both the HS3 specification and ROOT's RooUniform.
        An axis with no domain/observable bounds has no well-defined uniform
        density, so :meth:`likelihood` and :meth:`log_likelihood` raise
        ``ValueError`` naming that axis rather than returning a wrong density.

    HS3 Reference:
        :ref:`hs3:hs3.uniform-distribution`
    """

    type: Literal["uniform_dist"] = "uniform_dist"
    x: list[str]
    _normalizable: bool = PrivateAttr(default=False)

    # Docstring refs a private method autodoc never renders (known gap: #327).
    def _resolved_bounds(
        self, context: Context
    ) -> list[tuple[str, TensorVar, TensorVar]]:
        """
        Per-axis ``(name, lower, upper)`` bounds in the order of ``self.x``.

        Bounds come from the observable domain via
        :meth:`~pyhs3.distributions.core.Distribution._matching_observables`.

        Args:
            context: Mapping of names to pytensor variables (includes observables).

        Returns:
            List of ``(name, lower, upper)``, one per axis in ``self.x`` order.

        Raises:
            ValueError: If ``x`` is empty, or if any axis has no
                observable/domain bounds.
        """
        if not self.x:
            msg = (
                f"uniform_dist {self.name!r} requires at least one axis in 'x', "
                f"but 'x' is empty."
            )
            raise ValueError(msg)
        bounds = {
            name: (lower, upper)
            for name, lower, upper in self._matching_observables(context)
        }
        resolved: list[tuple[str, TensorVar, TensorVar]] = []
        for index in range(len(self.x)):
            name = self._parameters[f"x[{index}]"]
            if name not in bounds:
                msg = (
                    f"uniform_dist {self.name!r} requires domain/observable bounds "
                    f"for axis {name!r}, but none were found in the context. "
                    f"Provide the observable range via the model's domain "
                    f"(e.g. observables={{{name!r}: (lower, upper)}})."
                )
                raise ValueError(msg)
            lower, upper = bounds[name]
            resolved.append((name, lower, upper))
        return resolved

    def likelihood(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the uniform PDF.

        Delegates to ``pytensor_distributions.uniform``, returning the product
        over axes of :math:`1/(u_i - l_i)` on the support. The bounds are derived
        from the observable domain, so ``x`` itself does not appear in the density.

        Args:
            context: Mapping of names to pytensor variables (includes observables).

        Returns:
            pytensor.tensor.variable.TensorVariable: Self-normalized uniform density.

        Raises:
            ValueError: If any axis has no observable/domain bounds.
        """
        density: TensorVar | None = None
        for name, lower, upper in self._resolved_bounds(context):
            factor = Uniform.pdf(context[name], lower, upper)
            density = factor if density is None else density * factor
        return cast(TensorVar, density)

    def log_likelihood(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the uniform log-PDF.

        Analytic log form of :meth:`likelihood`: the sum over axes of
        ``pytensor_distributions.uniform.logpdf``, i.e.
        :math:`-\\sum_i \\log(u_i - l_i)` on the support. Each term is a constant
        independent of any parameter, so there is no underflow concern to guard
        against here.

        Args:
            context: Mapping of names to pytensor variables (includes observables).

        Returns:
            pytensor.tensor.variable.TensorVariable: Self-normalized uniform log-density.

        Raises:
            ValueError: If any axis has no observable/domain bounds.
        """
        log_density: TensorVar | None = None
        for name, lower, upper in self._resolved_bounds(context):
            term = Uniform.logpdf(context[name], lower, upper)
            log_density = term if log_density is None else log_density + term
        return cast(TensorVar, log_density)


class PoissonDist(Distribution):
    r"""
    Poisson probability distribution.

    Implements the Poisson probability mass function:

    .. math::

        P(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}

    Parameters:
        mean (str): Parameter name for the rate parameter (λ).
        x (str): Input variable name (discrete count).

    HS3 Reference:
        :ref:`hs3:hs3.dist:poisson`
    """

    type: Literal["poisson_dist"] = "poisson_dist"
    mean: str | float | int
    x: str | float | int

    def likelihood(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the Poisson PMF.

        Delegates to pytensor-distributions, which exponentiates the analytic
        log-pmf of :meth:`log_likelihood`.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the Poisson PMF.
        """
        return cast(
            TensorVar,
            Poisson.pdf(
                context[self._parameters["x"]],
                context[self._parameters["mean"]],
            ),
        )

    def log_likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the Poisson log-PMF.

        Delegates to pytensor-distributions' analytic log form:

        .. math::

            \log P(k; \lambda) = k \log\lambda - \lambda - \log\Gamma(k+1)

        Evaluating this directly (rather than ``pt.log(self.likelihood(...))``)
        avoids a ``pt.log(pt.exp(log_pmf))`` round-trip that underflows to
        ``-inf`` once the true log-pmf is a large negative number (e.g. far into
        the tail).

        pytensor-distributions writes the :math:`k \log\lambda` term as
        ``xlogy(k, lambda)``, which is ``0`` at ``k == 0`` (Poisson(k=0 |
        lambda=0) = 1, so log-pmf = ``-lambda`` = 0) and whose gradient with
        respect to ``lambda`` is guarded to stay finite there, avoiding the
        ``0 * log(0) = NaN`` value and ``k/lambda = 0/0`` NaN gradient that a
        naive ``k * pt.log(lambda)`` would produce.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the Poisson log-PMF.
        """
        return cast(
            TensorVar,
            Poisson.logpdf(
                context[self._parameters["x"]],
                context[self._parameters["mean"]],
            ),
        )


class ExponentialDist(Distribution):
    r"""
    Exponential probability distribution.

    Implements the exponential probability density function. The raw density,
    delegated to ``pytensor_distributions.exponential``, is rate-normalized
    and supported on :math:`x \ge 0`:

    .. math::

        f(x; c) = c \exp(-c \cdot x), \quad x \ge 0, \qquad f(x; c) = 0, \quad x < 0

    Parameters:
        x (str): Input variable name.
        c (str): Rate/decay parameter (coefficient).

    Note:
        The HS3 specification defines the density as
        :math:`(1/\mathcal{M}) \exp(-c \cdot x)`, where :math:`\mathcal{M}` is the
        domain measure. This distribution is normalized over its observable
        domain, so the constant rate factor :math:`c` in the delegated
        :math:`c \exp(-c \cdot x)` divides out and the normalized density equals
        the HS3 form. The sign convention matches ROOT's RooExponential with the
        negateCoefficient flag set to True.

    HS3 Reference:
        :external+hs3:ref:`exponential_dist <hs3.exponential-distribution>`

    ROOT Reference:
        :root:`RooExponential`
    """

    type: Literal["exponential_dist"] = "exponential_dist"
    x: str | float | int
    c: str | float | int

    def likelihood(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the exponential PDF.

        Delegates to pytensor-distributions' rate-normalized form
        :math:`c\\,\\exp(-c x)`.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of exponential PDF.
        """
        return cast(
            TensorVar,
            Exponential.pdf(
                context[self._parameters["x"]],
                context[self._parameters["c"]],
            ),
        )

    def log_likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the exponential log-PDF.

        Delegates to pytensor-distributions' analytic log form of
        :meth:`likelihood`:

        .. math::

            \log f(x; c) = \log c - c x

        Evaluating this directly (rather than ``pt.log(self.likelihood(...))``)
        avoids computing :math:`\exp(-cx)` and re-logging it, which underflows
        to 0.0 (and then to ``-inf``) once :math:`cx` exceeds roughly 745 in
        float64.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of exponential log-PDF.
        """
        return cast(
            TensorVar,
            Exponential.logpdf(
                context[self._parameters["x"]],
                context[self._parameters["c"]],
            ),
        )


class LogNormalDist(Distribution):
    r"""
    Log-normal probability distribution.

    Implements the log-normal probability density function:

    .. math::

        f(x; \mu, \sigma) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln(x)-\mu)^2}{2\sigma^2}\right)

    Parameters:
        x (str): Input variable name (must be > 0).
        mu (str): Location parameter (log-scale mean).
        sigma (str): Scale parameter (log-scale standard deviation).

    Note:
        This implementation uses the standard parametrization where mu and sigma
        are the mean and standard deviation of the underlying normal distribution
        in log-space. ROOT handles parameter transformations automatically for
        compatibility with median/shape parametrization.

    HS3 Reference:
        :external+hs3:ref:`lognormal_dist <hs3.log-normal-distribution>`
    """

    type: Literal["lognormal_dist"] = "lognormal_dist"
    x: str | float | int
    mu: str | float | int
    sigma: str | float | int

    def likelihood(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the log-normal PDF.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of log-normal PDF.
        """
        return cast(
            TensorVar,
            LogNormal.pdf(
                context[self._parameters["x"]],
                context[self._parameters["mu"]],
                context[self._parameters["sigma"]],
            ),
        )

    def log_likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the log-normal log-PDF.

        Delegates to pytensor-distributions' analytic log form of
        :meth:`likelihood`:

        .. math::

            \log f(x; \mu, \sigma) = -\log x - \log\sigma - \frac{1}{2}\log(2\pi)
            - \frac{1}{2}z^2, \quad z = \frac{\ln x - \mu}{\sigma}

        Evaluating this directly (rather than ``pt.log(self.likelihood(...))``)
        avoids computing :math:`\exp(-z^2/2)` and re-logging it, which
        underflows to 0.0 (and then to ``-inf``) once :math:`|z|` exceeds
        roughly 38 in float64.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of log-normal log-PDF.
        """
        return cast(
            TensorVar,
            LogNormal.logpdf(
                context[self._parameters["x"]],
                context[self._parameters["mu"]],
                context[self._parameters["sigma"]],
            ),
        )


# Coefficients of the rational approximation to the Landau density by
# K.S. Koelbig and B. Schorr, CPC 31 (1984) 97 (CERNLIB
# G110 DENLAN), as used by ROOT's ``landau_pdf`` in
# ``math/mathcore/src/PdfFuncMathCore.cxx``. Each ``_LANDAU_P*`` / ``_LANDAU_Q*``
# pair holds numerator / denominator polynomial coefficients in increasing power.
_LANDAU_P1 = (
    0.4259894875,
    -0.1249762550,
    0.03984243700,
    -0.006298287635,
    0.001511162253,
)
_LANDAU_Q1 = (1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063)
_LANDAU_P2 = (
    0.1788541609,
    0.1173957403,
    0.01488850518,
    -0.001394989411,
    0.0001283617211,
)
_LANDAU_Q2 = (1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714)
_LANDAU_P3 = (
    0.1788544503,
    0.09359161662,
    0.006325387654,
    0.00006611667319,
    -0.000002031049101,
)
_LANDAU_Q3 = (1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675)
_LANDAU_P4 = (0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186)
_LANDAU_Q4 = (1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511)
_LANDAU_P5 = (1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910)
_LANDAU_Q5 = (1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357)
_LANDAU_P6 = (1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109)
_LANDAU_Q6 = (1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939)
_LANDAU_A1 = (0.04166666667, -0.01996527778, 0.02709538966)
_LANDAU_A2 = (-1.845568670, -4.284640743)
_LANDAU_INV_SQRT_2PI = 0.3989422803  # 1/sqrt(2*pi), truncated as in ROOT


def _horner(coefficients: tuple[float, ...], t: TensorVar) -> TensorVar:
    """Evaluate ``sum_k coefficients[k] * t**k`` by Horner's rule."""
    result = pt.as_tensor_variable(coefficients[-1]) + 0.0 * t
    for c in reversed(coefficients[:-1]):
        result = result * t + c
    return cast(TensorVar, result)


def _log_rational(
    p: tuple[float, ...], q: tuple[float, ...], t: TensorVar
) -> TensorVar:
    """Log of the rational function ``P(t) / Q(t)`` of the given coefficients."""
    return cast(TensorVar, pt.log(_horner(p, t)) - pt.log(_horner(q, t)))


def _landau_log_density(v: TensorVar) -> TensorVar:
    r"""
    Log of the standard (location 0, scale 1) Landau density at ``v``.

    Piecewise approximation of Koelbig and Schorr, selected by ``v``. Every
    region is evaluated on an argument clipped into that region (the
    "double-where" pattern), so the unselected branches of ``pt.switch`` never
    produce inf or NaN that would poison gradients. ``v`` may have any shape;
    the result has the same shape.
    """
    # v < -5.5: exp(-1/u) / sqrt(u) * (1 + a1 polynomial in u), u = exp(v + 1).
    v_left = pt.minimum(v, -5.5)
    u_left = pt.exp(v_left + 1.0)
    log_left = (
        math.log(_LANDAU_INV_SQRT_2PI)
        - 1.0 / u_left
        - 0.5 * (v_left + 1.0)
        + pt.log(_horner(_LANDAU_A1, u_left) * u_left + 1.0)
    )

    # -5.5 <= v < -1: exp(-u) sqrt(u) P1(v) / Q1(v), u = exp(-v - 1).
    v_a = pt.clip(v, -5.5, -1.0)
    log_a = (
        -pt.exp(-v_a - 1.0)
        + 0.5 * (-v_a - 1.0)
        + _log_rational(_LANDAU_P1, _LANDAU_Q1, v_a)
    )

    # -1 <= v < 1 and 1 <= v < 5: rational functions of v.
    v_b = pt.clip(v, -1.0, 1.0)
    log_b = _log_rational(_LANDAU_P2, _LANDAU_Q2, v_b)
    v_c = pt.clip(v, 1.0, 5.0)
    log_c = _log_rational(_LANDAU_P3, _LANDAU_Q3, v_c)

    # v >= 5: u**2 * P(u) / Q(u) with u = 1/v, split at 12, 50, 300.
    def log_inverse_branch(
        p: tuple[float, ...], q: tuple[float, ...], lower: float, upper: float
    ) -> TensorVar:
        u = 1.0 / pt.clip(v, lower, upper)
        return cast(TensorVar, 2.0 * pt.log(u) + _log_rational(p, q, u))

    log_d = log_inverse_branch(_LANDAU_P4, _LANDAU_Q4, 5.0, 12.0)
    log_e = log_inverse_branch(_LANDAU_P5, _LANDAU_Q5, 12.0, 50.0)
    log_f = log_inverse_branch(_LANDAU_P6, _LANDAU_Q6, 50.0, 300.0)

    # v >= 300: u**2 * (1 + a2 polynomial in u), u = 1 / (v - v log(v) / (v + 1)).
    v_tail = pt.maximum(v, 300.0)
    u_tail = 1.0 / (v_tail - v_tail * pt.log(v_tail) / (v_tail + 1.0))
    log_tail = 2.0 * pt.log(u_tail) + pt.log(
        1.0 + (_LANDAU_A2[0] + _LANDAU_A2[1] * u_tail) * u_tail
    )

    return cast(
        TensorVar,
        pt.switch(
            v < -5.5,
            log_left,
            pt.switch(
                v < -1.0,
                log_a,
                pt.switch(
                    v < 1.0,
                    log_b,
                    pt.switch(
                        v < 5.0,
                        log_c,
                        pt.switch(
                            v < 12.0,
                            log_d,
                            pt.switch(
                                v < 50.0,
                                log_e,
                                pt.switch(v < 300.0, log_f, log_tail),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


class LandauDist(Distribution):
    r"""
    Landau probability distribution.

    Implements the Landau probability density function as defined in ROOT's
    :root:`RooLandau`, used in high-energy physics to model energy loss. The
    density is

    .. math::

        f(x; \mu, \sigma) = \frac{1}{\sigma}\,\phi(v), \qquad
        v = \frac{x-\mu}{\sigma},

    where :math:`\phi(v) = \frac{1}{2\pi i}\int_{c-i\infty}^{c+i\infty}
    \exp(s\ln s + vs)\,ds` is the standard Landau density. :math:`\phi` is
    evaluated with the piecewise rational approximation of Koelbig and Schorr
    (CPC 31 (1984) 97; CERNLIB G110 ``DENLAN``), the same
    one ROOT's ``TMath::Landau`` uses (``landau_pdf`` in
    ``math/mathcore/src/PdfFuncMathCore.cxx``). The approximation uses
    different expressions on ``v < -5.5``, ``[-5.5, -1)``, ``[-1, 1)``,
    ``[1, 5)``, ``[5, 12)``, ``[12, 50)``, ``[50, 300)`` and ``v >= 300``.

    Parameter mapping: ``mean`` is the location parameter of the distribution,
    not its mean (the Landau distribution has no finite mean) and not its most
    probable value. The mode lies at :math:`\mu - 0.22278\,\sigma`. The pyhs3
    parameters map onto ROOT's ``RooLandau(x, mean, sigma)`` and
    ``TMath::Landau(x, mean, sigma)`` unchanged. ``scipy.stats.landau`` uses a
    different standardization (its variable is rescaled by :math:`2/\pi` and
    shifted by :math:`\ln(\pi/2)`), so the mapping is
    ``loc = mean + sigma * log(pi / 2)`` and ``scale = sigma * pi / 2``.

    Parameters:
        x (str): Input variable name.
        mean (str): Location parameter.
        sigma (str): Scale parameter, must be positive.

    Note:
        Array shapes follow the inputs: ``x`` of shape ``(N,)`` or an ``(N, 1)``
        observable view broadcast against parameter batches in the usual way.

    HS3 Reference:
        Note: Landau distribution is not explicitly defined in the current HS3 specification.

    ROOT Reference:
        :root:`RooLandau`
    """

    type: Literal["landau_dist"] = "landau_dist"
    x: str | float | int
    mean: str | float | int
    sigma: str | float | int

    def likelihood(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the Landau PDF.

        Evaluated as the exponential of :meth:`log_likelihood`, which underflows
        to 0.0 deep in the left tail.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of Landau PDF.
        """
        return cast(TensorVar, pt.exp(self.log_likelihood(context)))

    def log_likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the Landau log-PDF.

        Analytic log of the piecewise approximation described in the class
        docstring, taken region by region. In the left tail the log density
        is :math:`-e^{-(v+1)} + \dots`, which stays finite long after the
        density itself underflows.

        .. math::

            \log f(x; \mu, \sigma) = \log\phi(v) - \log\sigma,
            \quad v = \frac{x-\mu}{\sigma}

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the
            Landau log-PDF.
        """
        x = context[self._parameters["x"]]
        mean = context[self._parameters["mean"]]
        sigma = context[self._parameters["sigma"]]

        return cast(TensorVar, _landau_log_density((x - mean) / sigma) - pt.log(sigma))


class GenNormalDist(Distribution):
    r"""
    Generalized normal (Subbotin) probability distribution, version 1.

    Implements the symmetric generalized normal probability density function:

    .. math::

        f(x; \mu, \alpha, \beta) = \frac{\beta}{2\alpha\,\Gamma(1/\beta)}
        \exp\left\{ -\left( \frac{|x-\mu|}{\alpha} \right)^\beta \right\}

    :math:`\mu` is the location, :math:`\alpha > 0` the scale, and
    :math:`\beta > 0` the shape. :math:`\beta = 2` recovers a Gaussian with
    :math:`\sigma = \alpha/\sqrt{2}`; :math:`\beta = 1` recovers a Laplace with
    scale :math:`\alpha`. Large :math:`\beta` (e.g. :math:`\beta \sim 8`)
    produces a flat-topped shape used for "2 point" parameter constraints
    between models with no clear preference.

    The pyhs3 parameter names (``mean``, ``alpha``, ``beta``) map onto
    :func:`scipy.stats.gennorm` as ``loc = mean``, ``scale = alpha``,
    ``beta = beta`` with no additional scale factor.

    Parameters:
        x (str): Input variable name.
        mean (str): Location parameter (μ).
        alpha (str): Scale parameter (alpha), corresponding to sigma = alpha/sqrt(2) at beta = 2.
        beta (str): Shape/power parameter (beta), with beta = 2 the Gaussian shape.

    HS3 Reference:
        :ref:`hs3:hs3.generalized-normal-distribution`
    """

    type: Literal["generalized_normal_dist"] = "generalized_normal_dist"
    x: str | float | int
    mean: str | float | int
    alpha: str | float | int
    beta: str | float | int

    def likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the generalized normal PDF.

        Returns the fully normalized density (over infinite support)

        .. math::

            f(x; \mu, \alpha, \beta) = \frac{\beta}{2\alpha\,\Gamma(1/\beta)}
            \exp\left\{ -\left( \frac{|x-\mu|}{\alpha} \right)^\beta \right\}

        matching :func:`scipy.stats.gennorm`. When the model carries a finite
        observable domain, :meth:`normalization_expression` renormalizes this
        over that domain exactly.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the
            generalized normal PDF.
        """
        x = context[self._parameters["x"]]
        mu = context[self._parameters["mean"]]
        alpha = context[self._parameters["alpha"]]
        beta = context[self._parameters["beta"]]

        norm = beta / (2.0 * alpha * pt.gamma(1.0 / beta))
        return cast(
            TensorVar,
            norm * pt.exp(-((pt.abs(x - mu) / alpha) ** beta)),
        )

    def log_likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the generalized normal log-PDF.

        Analytic log form of :meth:`likelihood`:

        .. math::

            \log f(x; \mu, \alpha, \beta) = \log\beta - \log 2 - \log\alpha
            - \log\Gamma(1/\beta) - \left( \frac{|x-\mu|}{\alpha} \right)^\beta

        Evaluating this directly (rather than ``pt.log(self.likelihood(...))``)
        avoids computing :math:`\exp(-(|x-\mu|/\alpha)^\beta)` and re-logging it,
        which underflows to 0.0 (and then to ``-inf``) once the exponent grows
        large in the tail. ``pt.gammaln`` is used for :math:`\log\Gamma(1/\beta)`.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the
            generalized normal log-PDF.
        """
        x = context[self._parameters["x"]]
        mu = context[self._parameters["mean"]]
        alpha = context[self._parameters["alpha"]]
        beta = context[self._parameters["beta"]]

        log_norm = pt.log(beta) - math.log(2.0) - pt.log(alpha) - pt.gammaln(1.0 / beta)
        return cast(
            TensorVar,
            log_norm - (pt.abs(x - mu) / alpha) ** beta,
        )

    def normalization_expression(
        self, context: Context, observable_name: str
    ) -> TensorVar | None:
        r"""
        Analytic antiderivative of the generalized normal PDF.

        The antiderivative of the fully normalized :meth:`likelihood` density is
        the centered CDF

        .. math::

            F(x) = \frac{1}{2}\,\operatorname{sign}(x-\mu)\,
            P\!\left( \frac{1}{\beta}, \left(\frac{|x-\mu|}{\alpha}\right)^\beta \right)

        where :math:`P(a, z)` is the regularized lower incomplete gamma function
        (``pt.gammainc``). The :math:`1/M` normalization constant of the density
        collapses the antiderivative's prefactor to exactly :math:`1/2`, so
        :math:`F(+\infty) - F(-\infty) = 1`. The framework evaluates
        :math:`F(\text{upper}) - F(\text{lower})` to renormalize over a finite
        observable domain, which is exact and avoids the Gauss-Legendre
        quadrature fallback.

        Returns None (deferring to quadrature) when the matching observable is
        not this distribution's ``x``, since the antiderivative is only defined
        with respect to ``x``, and when ``mean``, ``alpha`` or ``beta`` depends on
        that observable, since the CDF difference then is not the integral.

        Args:
            context: Mapping of names to pytensor variables.
            observable_name: Name of the observable to integrate over.

        Returns:
            Symbolic antiderivative expression, or None for numerical fallback.
        """
        if observable_name != self._parameters["x"] or self._depends_on_observable(
            context, observable_name, ("mean", "alpha", "beta")
        ):
            return None

        x = context[self._parameters["x"]]
        mu = context[self._parameters["mean"]]
        alpha = context[self._parameters["alpha"]]
        beta = context[self._parameters["beta"]]

        # F(x) = 0.5 * sign(x - mu) * P(1/beta, (|x-mu|/alpha)^beta); the
        # regularized lower incomplete gamma P is pt.gammainc.
        return cast(
            TensorVar,
            0.5
            * pt.sign(x - mu)
            * pt.gammainc(1.0 / beta, (pt.abs(x - mu) / alpha) ** beta),
        )


# Registry of basic distributions
distributions: dict[str, type[Distribution]] = {
    "gaussian_dist": GaussianDist,
    "uniform_dist": UniformDist,
    "poisson_dist": PoissonDist,
    "exponential_dist": ExponentialDist,
    "lognormal_dist": LogNormalDist,
    "generalized_normal_dist": GenNormalDist,
    "landau_dist": LandauDist,
}

# Define what should be exported from this module
__all__ = [
    "ExponentialDist",
    "GaussianDist",
    "GenNormalDist",
    "LandauDist",
    "LogNormalDist",
    "PoissonDist",
    "UniformDist",
    "distributions",
]
