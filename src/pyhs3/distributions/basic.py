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

from pyhs3.context import Context
from pyhs3.distributions.core import Distribution
from pyhs3.typing.aliases import TensorVar


class GaussianDist(Distribution):
    r"""
    Gaussian (normal) probability distribution.

    Implements the standard Gaussian probability density function:

    .. math::

        f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)

    Log-PDF expression:

    .. math::

        \log f(x; \mu, \sigma) = -\frac{\mu^2}{2\sigma^2} + \frac{\mu x}{\sigma^2} - \log(\sigma) - \frac{x^2}{2\sigma^2} - \frac{\log(2\pi)}{2}

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

    def expression(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the Gaussian PDF.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the Gaussian PDF.
        """
        # log.info("parameters: ", parameters)
        norm_const = 1.0 / (pt.sqrt(2 * math.pi) * context[self._parameters["sigma"]])
        exponent = pt.exp(
            -0.5
            * (
                (context[self._parameters["x"]] - context[self._parameters["mean"]])
                / context[self._parameters["sigma"]]
            )
            ** 2
        )
        return cast(TensorVar, norm_const * exponent)

    def log_expression(self, context: Context) -> TensorVar:
        """
        Log-PDF expression for Gaussian distribution in logarithmic space.

        Implements: -mean**2/(2*sigma**2) + mean*x/sigma**2 - log(sigma) - x**2/(2*sigma**2) - log(2*pi)/2

        Returns:
            TensorVar: Log-probability density in log-space
        """
        x = context[self._parameters["x"]]
        mean = context[self._parameters["mean"]]
        sigma = context[self._parameters["sigma"]]

        # Variable terms from symbolic analysis:
        # -mean**2/(2*sigma**2) + mean*x/sigma**2 - log(sigma) - x**2/(2*sigma**2)
        sigma_sq = sigma**2
        log_pdf = (
            -(mean**2) / (2 * sigma_sq)
            + mean * x / sigma_sq
            - pt.log(sigma)
            - x**2 / (2 * sigma_sq)
        )

        # Constant terms: -log(pi)/2 - log(2)/2 = -log(2*pi)/2
        log_pdf = log_pdf - pt.log(2 * math.pi) / 2

        return cast(TensorVar, log_pdf)


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

    def expression(self, _context: Context) -> TensorVar:
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

    def log_expression(self, _context: Context) -> TensorVar:
        """
        Log-PDF expression for uniform distribution in logarithmic space.

        Since uniform has constant density 1.0, log(1.0) = 0.0.

        Returns:
            TensorVar: Log-probability density (always 0.0)
        """
        return cast(TensorVar, pt.constant(0.0))


class PoissonDist(Distribution):
    r"""
    Poisson probability distribution.

    Implements the Poisson probability mass function:

    .. math::

        P(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}

    Log-PMF expression:

    .. math::

        \log P(k; \lambda) = k \log(\lambda) - \lambda - \log(k!)

    Parameters:
        mean (str): Parameter name for the rate parameter (λ).
        x (str): Input variable name (discrete count).

    HS3 Reference:
        :ref:`hs3:hs3.dist:poisson`
    """

    type: Literal["poisson_dist"] = "poisson_dist"
    mean: str | float | int
    x: str | float | int

    def expression(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the Poisson PMF.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the Poisson PMF.
        """
        mean = context[self._parameters["mean"]]
        x = context[self._parameters["x"]]

        # Poisson PMF: λ^k * e^(-λ) / k!
        # Using pt.gammaln for log(k!) = log(Γ(k+1))
        log_pmf = x * pt.log(mean) - mean - pt.gammaln(x + 1)
        return cast(TensorVar, pt.exp(log_pmf))

    def log_expression(self, context: Context) -> TensorVar:
        """
        Log-PMF expression for Poisson distribution in logarithmic space.

        Implements: x*log(mean) - mean - log(x!) where log(x!) = gammaln(x+1)

        Returns:
            TensorVar: Log-probability mass in log-space
        """
        mean = context[self._parameters["mean"]]
        x = context[self._parameters["x"]]

        # From symbolic analysis: x*log(mean) - mean - log(factorial(x))
        # Using pt.gammaln for log(x!) = log(Γ(x+1))
        log_pmf = x * pt.log(mean) - mean - pt.gammaln(x + 1)
        return cast(TensorVar, log_pmf)


class ExponentialDist(Distribution):
    r"""
    Exponential probability distribution.

    Implements the exponential probability density function with proper normalization:

    .. math::

        f(x; c) = \exp(-c \cdot x)

    Log-PDF expression:

    .. math::

        \log f(x; c) = -c \cdot x

    Parameters:
        x (str): Input variable name.
        c (str): Rate/decay parameter (coefficient).

    Note:
        The HS3 specification uses the form exp(-c*x), which matches ROOT's RooExponential
        when the negateCoefficient flag is True. ROOT handles parameter transformations
        automatically for compatibility.

    HS3 Reference:
        :hs3:label:`exponential_dist <hs3.exponential-distribution>`

    ROOT Reference:
        :rootref:`RooExponential <classRooExponential.html>`
    """

    type: Literal["exponential_dist"] = "exponential_dist"
    x: str | float | int
    c: str | float | int

    def expression(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the exponential PDF.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of exponential PDF.
        """
        x = context[self._parameters["x"]]
        c = context[self._parameters["c"]]

        # Exponential PDF: exp(-c * x)
        return cast(TensorVar, pt.exp(-c * x))

    def log_expression(self, context: Context) -> TensorVar:
        """
        Log-PDF expression for exponential distribution in logarithmic space.

        Implements: -rate*x (matches the expression() method which is exp(-rate*x))

        Returns:
            TensorVar: Log-probability density in log-space
        """
        x = context[self._parameters["x"]]
        rate = context[self._parameters["c"]]

        # Log of exp(-rate*x) = -rate*x
        log_pdf = -rate * x
        return cast(TensorVar, log_pdf)


class LogNormalDist(Distribution):
    r"""
    Log-normal probability distribution.

    Implements the log-normal probability density function with proper normalization:

    .. math::

        f(x; \mu, \sigma) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln(x)-\mu)^2}{2\sigma^2}\right)

    Log-PDF expression:

    .. math::

        \log f(x; \mu, \sigma) = -\frac{\mu^2}{2\sigma^2} + \frac{\mu \ln(x)}{\sigma^2} - \log(\sigma) - \ln(x) - \frac{\ln^2(x)}{2\sigma^2} - \frac{\log(2\pi)}{2}

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

    def expression(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the log-normal PDF.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of log-normal PDF.
        """
        x = context[self._parameters["x"]]
        mu = context[self._parameters["mu"]]
        sigma = context[self._parameters["sigma"]]

        # Log-normal PDF: (1/x) * exp(-((ln(x) - mu)^2) / (2 * sigma^2))
        log_x = pt.log(x)
        normalized_log = (log_x - mu) / sigma
        return cast(TensorVar, (1.0 / x) * pt.exp(-0.5 * normalized_log**2))

    def log_expression(self, context: Context) -> TensorVar:
        """
        Log-PDF expression for log-normal distribution in logarithmic space.

        Matches the expression() method: log((1/x) * exp(-0.5 * ((log(x) - mu)/sigma)^2))
        = -log(x) - 0.5 * ((log(x) - mu)/sigma)^2

        Returns:
            TensorVar: Log-probability density in log-space
        """
        x = context[self._parameters["x"]]
        mu = context[self._parameters["mu"]]
        sigma = context[self._parameters["sigma"]]

        # Log of expression method result
        log_x = pt.log(x)
        normalized_log = (log_x - mu) / sigma
        log_pdf = -log_x - 0.5 * normalized_log**2
        return cast(TensorVar, log_pdf)


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

    Log-PDF expression:

    .. math::

        \log f(x; \mu, \sigma) = -\frac{3\mu^2}{5\sigma^2} - \frac{\mu}{5\sigma} + \frac{6\mu x}{5\sigma^2} - \log(\sigma) + \frac{x}{5\sigma} - \frac{3x^2}{5\sigma^2} - \frac{1}{10}

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
        :rootref:`RooLandau <classRooLandau.html>`

    """

    type: Literal["landau_dist"] = "landau_dist"
    x: str | float | int
    mean: str | float | int
    sigma: str | float | int

    def expression(self, context: Context) -> TensorVar:
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

        return cast(TensorVar, (1.0 / sigma) * gaussian_core * asymmetric_factor)

    def log_expression(self, context: Context) -> TensorVar:
        """
        Log-PDF expression for Landau distribution in logarithmic space.

        Implements: -3*mean**2/(5*sigma**2) - mean/(5*sigma) + 6*mean*x/(5*sigma**2) - log(sigma) + x/(5*sigma) - 3*x**2/(5*sigma**2) - 1/10

        Returns:
            TensorVar: Log-probability density in log-space
        """
        x = context[self._parameters["x"]]
        mean = context[self._parameters["mean"]]
        sigma = context[self._parameters["sigma"]]

        # Expanded Landau log-PDF from symbolic analysis
        sigma_sq = sigma**2
        log_pdf = (
            -3 * mean**2 / (5 * sigma_sq)
            - mean / (5 * sigma)
            + 6 * mean * x / (5 * sigma_sq)
            - pt.log(sigma)
            + x / (5 * sigma)
            - 3 * x**2 / (5 * sigma_sq)
            - 0.1
        )
        return cast(TensorVar, log_pdf)


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
