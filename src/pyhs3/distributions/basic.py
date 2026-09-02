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
        :ref:`hs3:hs3.gaussian-normal-distribution`
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

    Delegates to :mod:`pytensor_distributions.uniform`, which already returns the
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

        Delegates to :mod:`pytensor_distributions.uniform`, returning the product
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
    delegated to :mod:`pytensor_distributions.exponential`, is rate-normalized
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


class LandauDist(Distribution):
    r"""
    Landau probability distribution.

    Implements the Landau probability density function as defined in ROOT's
    RooLandau. Used primarily in high-energy physics for modeling energy
    loss distributions.

    Approximation using modified Gaussian with asymmetric tails:

    .. math::

        f(x; \mu, \sigma) = \frac{1}{\sigma} \exp\left(-\frac{1}{2}z^2 - \frac{1}{10}(z-1)^2\right)

    where $z = \frac{x-\mu}{\sigma}$ for $z > 1$.

    Parameters:
        x (str): Input variable name.
        mean (str): Location parameter.
        sigma (str): Scale parameter.

    Note:
        The Landau distribution is asymmetric with a long tail towards larger values.
        This implementation uses an approximation since the exact Landau function
        is not available in PyTensor.

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

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of Landau PDF.

        Note:
            This implementation uses a Gaussian approximation. In practice,
            ROOT uses more sophisticated approximations or numerical methods.
        """
        x = context[self._parameters["x"]]
        mean = context[self._parameters["mean"]]
        sigma = context[self._parameters["sigma"]]

        # Normalized variable
        z = (x - mean) / sigma

        # Landau approximation using a modified Gaussian with asymmetric tails
        # This is a simplified approximation - ROOT uses more sophisticated methods
        gaussian_core = pt.exp(-0.5 * z**2)
        asymmetric_factor = pt.exp(-0.1 * pt.maximum(0.0, z - 1) ** 2)
        gaussian_term_integral = pt.sqrt(math.pi / 2) * (1 + pt.erf(1 / pt.sqrt(2.0)))
        asymmetric_factor_integral = (
            pt.exp(-1 / 12)
            * (pt.sqrt(5 * math.pi / 3) / 2)
            * pt.erfc((5 / 6) * pt.sqrt(3 / 5))
        )
        normalization = gaussian_term_integral + asymmetric_factor_integral

        return cast(
            TensorVar,
            (1.0 / normalization) * (1.0 / sigma) * gaussian_core * asymmetric_factor,
        )

    def log_likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the Landau approximation log-PDF.

        Analytic log form of :meth:`likelihood`. The raw density there is a
        product of two exponentials and a constant normalization factor
        (``gaussian_core * asymmetric_factor / (sigma * normalization)``), so
        its log is a plain sum -- no product term can partially cancel another:

        .. math::

            \log f(x; \mu, \sigma) = -\log\sigma - \frac{1}{2}z^2
            - \frac{1}{10}\max(0, z-1)^2 - \log(\mathcal{N}),
            \quad z = \frac{x-\mu}{\sigma}

        where :math:`\mathcal{N}` is the same constant (parameter-independent)
        normalization computed in :meth:`likelihood`. Evaluating this directly
        avoids computing the two exponentials and re-logging their product,
        which underflows to 0.0 (and then to ``-inf``) for large :math:`|z|`.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the Landau
            approximation log-PDF.
        """
        x = context[self._parameters["x"]]
        mean = context[self._parameters["mean"]]
        sigma = context[self._parameters["sigma"]]

        # Normalized variable
        z = (x - mean) / sigma

        # Log of the same Gaussian-core / asymmetric-tail approximation used in likelihood()
        gaussian_log_term = -0.5 * z**2
        asymmetric_log_term = -0.1 * pt.maximum(0.0, z - 1) ** 2
        gaussian_term_integral = pt.sqrt(math.pi / 2) * (1 + pt.erf(1 / pt.sqrt(2.0)))
        asymmetric_factor_integral = (
            pt.exp(-1 / 12)
            * (pt.sqrt(5 * math.pi / 3) / 2)
            * pt.erfc((5 / 6) * pt.sqrt(3 / 5))
        )
        normalization = gaussian_term_integral + asymmetric_factor_integral

        return cast(
            TensorVar,
            -pt.log(sigma)
            - pt.log(normalization)
            + gaussian_log_term
            + asymmetric_log_term,
        )


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
        with respect to ``x``.

        Args:
            context: Mapping of names to pytensor variables.
            observable_name: Name of the observable to integrate over.

        Returns:
            Symbolic antiderivative expression, or None for numerical fallback.
        """
        if observable_name != self._parameters["x"]:
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
