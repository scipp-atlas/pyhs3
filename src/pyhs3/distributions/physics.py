"""
Physics-specific distribution implementations.

Provides classes for handling probability distributions commonly used
in high-energy physics analysis, including Crystal Ball distributions
and ARGUS background models.
"""

from __future__ import annotations

from typing import Literal, cast

import pytensor.tensor as pt

from pyhs3.context import Context
from pyhs3.distributions.core import Distribution
from pyhs3.typing.aliases import TensorVar


class CrystalBallDist(Distribution):
    r"""
    Single-sided Crystal Ball distribution implementation.

    Implements the ROOT RooCrystalBall lineshape with a single power-law tail.
    This is the standard Crystal Ball distribution with shared parameters for
    both sides of the Gaussian core, but only one tail (usually on the left).

    Mathematical Form:

    .. math::

        f(m; m_0, \sigma, \alpha, n) = \begin{cases}
        A \cdot \left(B - \frac{m - m_0}{\sigma}\right)^{-n}, & \text{for } \frac{m - m_0}{\sigma} < -\alpha \\
        \exp\left(-\frac{1}{2} \cdot \left[\frac{m - m_0}{\sigma}\right]^2\right), & \text{otherwise}
        \end{cases}

    where:

    .. math::

        \begin{align}
        A &= \left(\frac{n}{\alpha}\right)^{n} \cdot \exp\left(-\frac{\alpha^2}{2}\right) \\
        B &= \frac{n}{\alpha} - \alpha
        \end{align}

    Parameters:
        m: Observable variable
        m0: Peak position (mean)
        sigma: Width parameter (must be > 0)
        alpha: Transition point (must be > 0)
        n: Power law exponent (must be > 0)

    Note:
        All parameters except m and m0 must be positive. This is the standard
        single-sided Crystal Ball used widely in high-energy physics.
    """

    type: Literal["crystalball_dist"] = "crystalball_dist"
    alpha: str
    m: str
    m0: str
    n: str
    sigma: str

    def expression(self, distributionsandparameters: Context) -> TensorVar:
        """
        Evaluate the single-sided Crystal Ball distribution.

        Implements the ROOT RooCrystalBall formula with a single tail.
        All shape parameters (alpha, n, sigma) are assumed to be positive.
        """
        alpha = distributionsandparameters[self.alpha]
        m = distributionsandparameters[self.m]
        m0 = distributionsandparameters[self.m0]
        n = distributionsandparameters[self.n]
        sigma = distributionsandparameters[self.sigma]

        # Calculate A and B per ROOT formula
        # Note: alpha, n, sigma are assumed to be positive
        A = (n / alpha) ** n * pt.exp(-(alpha**2) / 2)
        B = (n / alpha) - alpha

        # Calculate normalized distance from peak
        t = (m - m0) / sigma

        # Calculate each region per ROOT formula
        tail = A * ((B - t) ** (-n))
        core = pt.exp(-(t**2) / 2)

        # Apply ROOT conditions: tail for t < -alpha, core otherwise
        return cast(
            TensorVar,
            pt.switch(
                t < -alpha,
                tail,
                core,
            ),
        )


class AsymmetricCrystalBallDist(Distribution):
    r"""
    Crystal Ball distribution implementation.

    Implements the generalized asymmetrical double-sided Crystal Ball line shape
    as defined in ROOT's RooCrystalBall.

    The probability density function is defined as:

    .. math::

        f(m; m_0, \sigma_L, \sigma_R, \alpha_L, \alpha_R, n_L, n_R) = \begin{cases}
        A_L \cdot \left(B_L - \frac{m - m_0}{\sigma_L}\right)^{-n_L}, & \text{for } \frac{m - m_0}{\sigma_L} < -\alpha_L \\
        \exp\left(-\frac{1}{2} \cdot \left[\frac{m - m_0}{\sigma_L}\right]^2\right), & \text{for } \frac{m - m_0}{\sigma_L} \leq 0 \\
        \exp\left(-\frac{1}{2} \cdot \left[\frac{m - m_0}{\sigma_R}\right]^2\right), & \text{for } \frac{m - m_0}{\sigma_R} \leq \alpha_R \\
        A_R \cdot \left(B_R + \frac{m - m_0}{\sigma_R}\right)^{-n_R}, & \text{otherwise}
        \end{cases}

    where:

    .. math::

        \begin{align}
        A_i &= \left(\frac{n_i}{\alpha_i}\right)^{n_i} \cdot \exp\left(-\frac{\alpha_i^2}{2}\right) \\
        B_i &= \frac{n_i}{\alpha_i} - \alpha_i
        \end{align}

    Parameters:
        m: Observable variable
        m0: Peak position (mean)
        sigma_L: Left-side width parameter (must be > 0)
        sigma_R: Right-side width parameter (must be > 0)
        alpha_L: Left-side transition point (must be > 0)
        alpha_R: Right-side transition point (must be > 0)
        n_L: Left-side power law exponent (must be > 0)
        n_R: Right-side power law exponent (must be > 0)

    Note:
        All parameters except m and m0 must be positive. The distribution
        reduces to a single-sided Crystal Ball when one of the alpha parameters
        is set to zero.
    """

    type: Literal["crystalball_doublesided_dist"] = "crystalball_doublesided_dist"
    alpha_L: str
    alpha_R: str
    m: str
    m0: str
    n_L: str
    n_R: str
    sigma_R: str
    sigma_L: str

    def expression(self, distributionsandparameters: Context) -> TensorVar:
        """
        Evaluate the Crystal Ball distribution.

        Implements the ROOT RooCrystalBall formula with proper parameter validation.
        All shape parameters (alpha, n, sigma) are assumed to be positive.
        """
        alpha_L = distributionsandparameters[self.alpha_L]
        alpha_R = distributionsandparameters[self.alpha_R]
        m = distributionsandparameters[self.m]
        m0 = distributionsandparameters[self.m0]
        n_L = distributionsandparameters[self.n_L]
        n_R = distributionsandparameters[self.n_R]
        sigma_L = distributionsandparameters[self.sigma_L]
        sigma_R = distributionsandparameters[self.sigma_R]

        # Calculate A_i and B_i per ROOT formula
        # Note: alpha, n, sigma are assumed to be positive
        A_L = (n_L / alpha_L) ** n_L * pt.exp(-(alpha_L**2) / 2)
        A_R = (n_R / alpha_R) ** n_R * pt.exp(-(alpha_R**2) / 2)
        B_L = (n_L / alpha_L) - alpha_L
        B_R = (n_R / alpha_R) - alpha_R

        # Calculate normalized distance from peak for each side
        t_L = (m - m0) / sigma_L
        t_R = (m - m0) / sigma_R

        # Calculate each region per ROOT formula
        left_tail = A_L * ((B_L - t_L) ** (-n_L))
        core_left = pt.exp(-(t_L**2) / 2)
        core_right = pt.exp(-(t_R**2) / 2)
        right_tail = A_R * ((B_R + t_R) ** (-n_R))

        # Apply ROOT conditions
        return cast(
            TensorVar,
            pt.switch(
                t_L < -alpha_L,
                left_tail,
                pt.switch(
                    t_L <= 0,
                    core_left,
                    pt.switch(t_R <= alpha_R, core_right, right_tail),
                ),
            ),
        )


class ArgusDist(Distribution):
    r"""
    ARGUS probability distribution.

    Implements the ARGUS background distribution as defined in ROOT's RooArgusBG
    and the HS3 specification. Used extensively in B physics for modeling
    combinatorial backgrounds.

    .. math::

        f(m; m_0, c, p) = \frac{1}{\mathcal{M}} \cdot m \cdot \left[ 1 - \left( \frac{m}{m_0} \right)^2 \right]^p \cdot \exp\left[ c \cdot \left(1 - \left(\frac{m}{m_0}\right)^2 \right) \right]

    Parameters:
        mass (str): Input variable name (invariant mass).
        resonance (str): Kinematic endpoint parameter (m₀).
        slope (str): Slope parameter (c).
        power (str): Power parameter (p).

    Note:
        The ARGUS distribution is used to model the invariant mass spectrum of
        combinatorial backgrounds in B meson decays. The resonance parameter
        typically corresponds to a kinematic endpoint.
    """

    type: Literal["argus_dist"] = "argus_dist"
    mass: str | float | int
    resonance: str | float | int
    slope: str | float | int
    power: str | float | int

    def expression(self, distributionsandparameters: Context) -> TensorVar:
        """
        Builds a symbolic expression for the ARGUS PDF.

        Args:
            distributionsandparameters (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of ARGUS PDF.
        """
        m = distributionsandparameters[self._parameters["mass"]]
        m0 = distributionsandparameters[self._parameters["resonance"]]
        c = distributionsandparameters[self._parameters["slope"]]
        p = distributionsandparameters[self._parameters["power"]]

        # ARGUS PDF: m * [1 - (m/m0)^2]^p * exp[c * (1 - (m/m0)^2)]
        ratio_squared = (m / m0) ** 2
        bracket_term = 1.0 - ratio_squared
        power_term = bracket_term**p
        exp_term = pt.exp(c * bracket_term)

        return cast(TensorVar, m * power_term * exp_term)


# Export list of physics distribution classes
__all__ = [
    "ArgusDist",
    "AsymmetricCrystalBallDist",
    "CrystalBallDist",
    "distributions",
]

# Registry dict mapping type strings to classes
distributions: dict[str, type[Distribution]] = {
    "crystalball_dist": CrystalBallDist,
    "crystalball_doublesided_dist": AsymmetricCrystalBallDist,
    "argus_dist": ArgusDist,
}
