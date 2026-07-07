"""
Physics-specific distribution implementations.

Provides classes for handling probability distributions commonly used
in high-energy physics analysis, including Crystal Ball distributions
and ARGUS background models.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
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

    HS3 Reference:
        :hs3:label:`crystalball_dist <hs3.crystalball-distribution>`
    """

    type: Literal["crystalball_dist"] = "crystalball_dist"
    alpha: str
    m: str
    m0: str
    n: str
    sigma: str

    def likelihood(self, context: Context) -> TensorVar:
        """
        Evaluate the single-sided Crystal Ball distribution.

        Implements the ROOT RooCrystalBall formula with a single tail.
        All shape parameters (alpha, n, sigma) are assumed to be positive.
        """
        alpha = context[self.alpha]
        m = context[self.m]
        m0 = context[self.m0]
        n = context[self.n]
        sigma = context[self.sigma]

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

    HS3 Reference:
        Note: Asymmetric Crystal Ball distribution is not explicitly defined in the current HS3 specification.
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

    def likelihood(self, context: Context) -> TensorVar:
        """
        Evaluate the Crystal Ball distribution.

        Implements the ROOT RooCrystalBall formula with proper parameter validation.
        All shape parameters (alpha, n, sigma) are assumed to be positive.
        """
        alpha_L = context[self.alpha_L]
        alpha_R = context[self.alpha_R]
        m = context[self.m]
        m0 = context[self.m0]
        n_L = context[self.n_L]
        n_R = context[self.n_R]
        sigma_L = context[self.sigma_L]
        sigma_R = context[self.sigma_R]

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

    def normalization_expression(
        self, context: Context, observable_name: str
    ) -> TensorVar | None:
        r"""
        Analytic antiderivative of the double-sided Crystal Ball.

        The DSCB is only piecewise-smooth (its second derivative is
        discontinuous at the core/tail junctions), which degrades the
        single-interval Gauss-Legendre fallback to ~1e-4..1e-3 relative
        error that shifts with the shape parameters.  The antiderivative is
        exact: Gaussian error functions in the core and closed-form power-law
        integrals in the tails, with matching constants at the junctions so
        that F is continuous and ``F(b) - F(a)`` is the integral for any
        bounds.  This matches ROOT's ``RooCrystalBall::analyticalIntegral``.

        The ``n == 1`` tails use the logarithmic antiderivative; the power
        form's denominator is masked to 1 in that branch so the unused branch
        stays finite.
        """
        if observable_name != self.m:
            return None

        alpha_L = context[self.alpha_L]
        alpha_R = context[self.alpha_R]
        m = context[self.m]
        m0 = context[self.m0]
        n_L = context[self.n_L]
        n_R = context[self.n_R]
        sigma_L = context[self.sigma_L]
        sigma_R = context[self.sigma_R]

        # Same A_i and B_i as likelihood(); alpha, n, sigma assumed positive.
        A_L = (n_L / alpha_L) ** n_L * pt.exp(-(alpha_L**2) / 2)
        A_R = (n_R / alpha_R) ** n_R * pt.exp(-(alpha_R**2) / 2)
        B_L = (n_L / alpha_L) - alpha_L
        B_R = (n_R / alpha_R) - alpha_R

        t_L = (m - m0) / sigma_L
        t_R = (m - m0) / sigma_R

        sqrt_half_pi = np.sqrt(np.pi / 2.0)
        sqrt_two = np.sqrt(2.0)

        # Antiderivative of A (B - t)^(-n) w.r.t. m on the left tail, with the
        # n == 1 logarithmic special case.  ``u`` is B_L - t_L, which equals
        # n_L/alpha_L at the junction t_L = -alpha_L and stays positive on the
        # tail; the power-branch denominator is masked to 1 when n == 1 so the
        # unused branch never divides by zero.
        denom_L = pt.switch(pt.eq(n_L, 1.0), 1.0, n_L - 1.0)
        denom_R = pt.switch(pt.eq(n_R, 1.0), 1.0, n_R - 1.0)

        def tail_left(u: TensorVar) -> TensorVar:
            return cast(
                TensorVar,
                sigma_L
                * A_L
                * pt.switch(pt.eq(n_L, 1.0), -pt.log(u), u ** (1.0 - n_L) / denom_L),
            )

        def tail_right(u: TensorVar) -> TensorVar:
            return cast(
                TensorVar,
                sigma_R
                * A_R
                * pt.switch(pt.eq(n_R, 1.0), pt.log(u), -(u ** (1.0 - n_R)) / denom_R),
            )

        core_left = sigma_L * sqrt_half_pi * pt.erf(t_L / sqrt_two)
        core_right = sigma_R * sqrt_half_pi * pt.erf(t_R / sqrt_two)

        # Matching constants: shift the core so it continues the left tail at
        # t_L = -alpha_L (continuity at t = 0 is automatic, erf(0) = 0), then
        # shift the right tail so it continues the core at t_R = alpha_R.
        core_offset = tail_left(n_L / alpha_L) - sigma_L * sqrt_half_pi * pt.erf(
            -alpha_L / sqrt_two
        )
        tail_right_offset = (
            sigma_R * sqrt_half_pi * pt.erf(alpha_R / sqrt_two)
            + core_offset
            - tail_right(n_R / alpha_R)
        )

        return cast(
            TensorVar,
            pt.switch(
                t_L < -alpha_L,
                tail_left(B_L - t_L),
                pt.switch(
                    t_L <= 0,
                    core_left + core_offset,
                    pt.switch(
                        t_R <= alpha_R,
                        core_right + core_offset,
                        tail_right(B_R + t_R) + tail_right_offset,
                    ),
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

        f(m; m_0, c, p) = m \cdot \left[ 1 - \left( \frac{m}{m_0} \right)^2 \right]^p \cdot \exp\left[ c \cdot \left(1 - \left(\frac{m}{m_0}\right)^2 \right) \right]

    Parameters:
        mass (str): Input variable name (invariant mass).
        resonance (str): Kinematic endpoint parameter (m₀).
        slope (str): Slope parameter (c).
        power (str): Power parameter (p).

    Note:
        The ARGUS distribution is used to model the invariant mass spectrum of
        combinatorial backgrounds in B meson decays. The resonance parameter
        typically corresponds to a kinematic endpoint.

    HS3 Reference:
        :hs3:label:`argus_dist <hs3.argus-distribution>`
    """

    type: Literal["argus_dist"] = "argus_dist"
    mass: str | float | int
    resonance: str | float | int
    slope: str | float | int
    power: str | float | int

    def likelihood(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the ARGUS PDF.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of ARGUS PDF.
        """
        m = context[self._parameters["mass"]]
        m0 = context[self._parameters["resonance"]]
        c = context[self._parameters["slope"]]
        p = context[self._parameters["power"]]

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
