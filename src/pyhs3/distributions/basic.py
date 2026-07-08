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

    def log_likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the Gaussian log-PDF.

        Analytic log form of :meth:`likelihood`:

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
        sigma = context[self._parameters["sigma"]]
        z = (context[self._parameters["x"]] - context[self._parameters["mean"]]) / sigma
        return cast(
            TensorVar,
            -0.5 * z**2 - pt.log(sigma) - 0.5 * math.log(2.0 * math.pi),
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

        The analytic log-pmf in :meth:`log_likelihood` is the primary form;
        the probability-space pmf is its exponential.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the Poisson PMF.
        """
        # Poisson PMF: λ^k * e^(-λ) / k!
        return cast(TensorVar, pt.exp(self.log_likelihood(context)))

    def log_likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the Poisson log-PMF.

        Primary analytic form of the distribution (:meth:`likelihood` is its
        exponential):

        .. math::

            \log P(k; \lambda) = k \log\lambda - \lambda - \log\Gamma(k+1)

        using pt.gammaln for :math:`\log(k!) = \log\Gamma(k+1)`. Returning the
        log-pmf directly avoids a ``pt.log(pt.exp(log_pmf))`` round-trip that
        underflows to ``-inf`` once the true log-pmf is a large negative
        number (e.g. far into the tail).

        Guards the log's argument (not just the switch's output) against
        ``0 * log(0) = NaN`` when both ``k == 0`` and ``lambda == 0``: pt.switch
        differentiates both branches, so the untaken "full" branch would
        otherwise contribute a NaN gradient at this point (d/dlambda of
        ``k * log(lambda)`` is ``k/lambda = 0/0``). Poisson(k=0 | lambda=0) = 1,
        so the switch's true branch (``-lambda = 0``) already gives the correct
        log-pmf; substituting a nonzero placeholder for lambda inside the
        untaken branch's log keeps both its value and its gradient finite there,
        matching the guard pattern used by
        :meth:`~pyhs3.distributions.histfactory.HistFactoryDistChannel._bin_log_probs`.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the Poisson log-PMF.
        """
        mean = context[self._parameters["mean"]]
        x = context[self._parameters["x"]]

        safe_mean = pt.switch(pt.eq(x, 0), 1.0, mean)
        full = x * pt.log(safe_mean) - mean - pt.gammaln(x + 1)
        return cast(TensorVar, pt.switch(pt.eq(x, 0), -mean, full))


class ExponentialDist(Distribution):
    r"""
    Exponential probability distribution.

    Implements the exponential probability density function:

    .. math::

        f(x; c) = \exp(-c \cdot x)

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
        :root:`RooExponential`
    """

    type: Literal["exponential_dist"] = "exponential_dist"
    x: str | float | int
    c: str | float | int

    def likelihood(self, context: Context) -> TensorVar:
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
        return cast(TensorVar, (c) * pt.exp(-c * x))

    def log_likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the exponential log-PDF.

        Analytic log form of :meth:`likelihood`:

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
        x = context[self._parameters["x"]]
        c = context[self._parameters["c"]]

        return cast(TensorVar, pt.log(c) - c * x)


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
        x = context[self._parameters["x"]]
        mu = context[self._parameters["mu"]]
        sigma = context[self._parameters["sigma"]]

        # Log-normal PDF: (1/x) * exp(-((ln(x) - mu)^2) / (2 * sigma^2))
        log_x = pt.log(x)
        normalized_log = (log_x - mu) / sigma
        return cast(
            TensorVar,
            (1.0 / (x * sigma * pt.sqrt(2.0 * math.pi)))
            * pt.exp(-0.5 * normalized_log**2),
        )

    def log_likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the log-normal log-PDF.

        Analytic log form of :meth:`likelihood`:

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
        x = context[self._parameters["x"]]
        mu = context[self._parameters["mu"]]
        sigma = context[self._parameters["sigma"]]

        log_x = pt.log(x)
        normalized_log = (log_x - mu) / sigma
        return cast(
            TensorVar,
            -log_x
            - pt.log(sigma)
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * normalized_log**2,
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
