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
    density over its support region, as defined in ROOT's RooUniform.

    .. math::

        f(x) = \frac{1}{\mathcal{M}}

    where the normalization constant $\mathcal{M}$ is determined by the domain bounds.

    Parameters:
        x (str): Input variable name.

    Note:
        The actual bounds are defined by the domain, not by distribution parameters.
        This matches both the HS3 specification and ROOT's RooUniform implementation.

    HS3 Reference:
        :ref:`hs3:hs3.uniform-distribution`
    """

    type: Literal["uniform_dist"] = "uniform_dist"
    x: list[str]

    def likelihood(self, _context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the uniform PDF.

        Args:
            _context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Constant value representing uniform density.

        Note:
            Returns a constant value of 1.0. The actual normalization is handled
            by the domain bounds during integration/sampling. The variables in self.x
            are used to define the domain but don't affect the constant density.
        """
        # Uniform distribution has constant density over its support
        # The actual normalization factor is handled by domain bounds
        # The variables in self.x define the domain but don't change the constant density
        return cast(TensorVar, pt.constant(1.0))

    def log_likelihood(self, _context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the uniform log-PDF.

        Analytic log form of :meth:`likelihood`: ``log(1.0) == 0.0``, a
        constant independent of any parameter, so there is no underflow
        concern to guard against here.

        Args:
            _context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Constant value (0.0) representing
            the uniform log-density.
        """
        return cast(TensorVar, pt.constant(0.0))


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
        :hs3:label:`exponential_dist <hs3.exponential-distribution>`

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
        :hs3:label:`lognormal_dist <hs3.log-normal-distribution>`
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


# Registry of basic distributions
distributions: dict[str, type[Distribution]] = {
    "gaussian_dist": GaussianDist,
    "uniform_dist": UniformDist,
    "poisson_dist": PoissonDist,
    "exponential_dist": ExponentialDist,
    "lognormal_dist": LogNormalDist,
    "landau_dist": LandauDist,
}

# Define what should be exported from this module
__all__ = [
    "ExponentialDist",
    "GaussianDist",
    "LandauDist",
    "LogNormalDist",
    "PoissonDist",
    "UniformDist",
    "distributions",
]
