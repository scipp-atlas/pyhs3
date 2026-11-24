"""
Physics-specific distribution implementations.

Provides classes for handling probability distributions commonly used
in high-energy physics analysis, including Crystal Ball distributions,
ARGUS background models, and the relativistic Breit-Wigner resonance shape.

``CrystalBallDist``, ``AsymmetricCrystalBallDist``, ``ArgusDist``, and
``RelativisticBreitWignerDist`` are hand-rolled symbolic PyTensor expressions
and stay that way: ``pytensor_distributions`` provides no equivalent for any of
them, so they cannot be delegated the way the basic distributions are.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pytensor.tensor as pt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyhs3.base import balanced_sum
from pyhs3.context import Context
from pyhs3.distributions.core import Distribution
from pyhs3.typing.aliases import TensorVar

# Lower bound applied to Crystal Ball shape parameters (alpha, n, sigma) inside
# the symbolic expression. These are mathematically required to be positive --
# sigma and alpha appear as divisors, n is a power-law exponent -- but the input
# parameter may be a shared bounded scalar or a compound expression whose bounds
# are set for a different distribution. Clipping here, rather than constraining
# the input, guards against division-by-zero / NaN (e.g. a negative alpha with a
# non-integer n makes the power-law tail NaN) without touching the shared
# parameter. See issue #57.
_SHAPE_CLIP_MIN = 1e-12


def _clip_shape(value: TensorVar) -> TensorVar:
    """Lower-bound a Crystal Ball shape parameter at ``_SHAPE_CLIP_MIN`` > 0.

    Applied element-wise; preserves the input shape. For any valid (positive)
    parameter this is a no-op, so it does not change the pdf for in-range
    inputs -- it only rescues out-of-range (<= 0) values from producing NaN or a
    negative pseudo-pdf.
    """
    return cast(TensorVar, pt.maximum(value, _SHAPE_CLIP_MIN))


def _crystalball_log_tail(n: TensorVar, alpha: TensorVar, u: TensorVar) -> TensorVar:
    r"""
    Analytic log of a Crystal Ball power-law tail :math:`A \cdot u^{-n}`.

    ``u`` is ``B - t`` (left tail) or ``B + t`` (right tail); ``n``/``alpha``
    are that tail's own shape parameters (assumed positive, as throughout
    this module). :math:`\log A = n \log(n/\alpha) - \alpha^2/2` is computed
    directly rather than as ``pt.log(A)``, since :math:`A` is itself an
    exponential (:math:`\exp(-\alpha^2/2)`) that underflows to 0.0 for large
    :math:`\alpha`, which would turn ``pt.log(A)`` into ``-inf`` even though
    the true value is finite.

    Shared by :class:`CrystalBallDist` and :class:`AsymmetricCrystalBallDist`
    since both use the same power-law tail form on each side.

    Guards the argument of ``pt.log(u)`` (not just the caller's switch
    output): both callers evaluate this helper for every ``m``, including
    core/other-tail regions where ``u`` is zero or negative. PyTensor
    differentiates every branch of a switch, so ``pt.log(u)`` at a
    non-positive ``u`` would otherwise contribute a NaN gradient there that
    survives multiplication by the switch's zero mask (``0 * NaN = NaN``) --
    the same guard idiom as :class:`AsymmetricCrystalBallDist`'s
    ``normalization_expression`` ``tail_left``/``tail_right`` helpers.

    Args:
        n: Power-law exponent (assumed > 0).
        alpha: Transition point (assumed > 0).
        u: ``B - t`` (left tail) or ``B + t`` (right tail); positive within
            this tail's own region, but evaluated for every ``m`` by the
            caller's outer switch.

    Returns:
        Symbolic log of the tail value, safe to evaluate for any real ``u``.
    """
    safe_u = pt.switch(u > 0, u, 1.0)
    log_A = n * pt.log(n / alpha) - (alpha**2) / 2.0
    return cast(TensorVar, log_A - n * pt.log(safe_u))


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
        :external+hs3:ref:`crystalball_dist <hs3.crystal-ball-distribution>`
    """

    type: Literal["crystalball_dist"] = "crystalball_dist"
    alpha: str | float
    m: str
    m0: str | float
    n: str | float
    sigma: str | float

    def likelihood(self, context: Context) -> TensorVar:
        """
        Evaluate the single-sided Crystal Ball distribution.

        Implements the ROOT RooCrystalBall formula with a single tail.
        All shape parameters (alpha, n, sigma) are assumed to be positive.
        """
        # Clip shape parameters at _SHAPE_CLIP_MIN > 0 (see _clip_shape): guards
        # the sigma/alpha divisors and the power-law exponent n against <= 0
        # inputs. m and m0 are positions and stay unclipped. Element-wise; keeps
        # each input's shape.
        alpha = _clip_shape(context[self._parameters["alpha"]])
        m = context[self.m]
        m0 = context[self._parameters["m0"]]
        n = _clip_shape(context[self._parameters["n"]])
        sigma = _clip_shape(context[self._parameters["sigma"]])

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

    def log_likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the single-sided Crystal Ball log-PDF.

        Analytic log form of :meth:`likelihood`, piecewise over the same
        core/tail split:

        .. math::

            \log f(m) = \begin{cases}
            \log A - n \log(B - t), & t < -\alpha \\
            -\frac{1}{2} t^2, & \text{otherwise}
            \end{cases}

        via ``_crystalball_log_tail`` for the tail branch. Evaluating
        this directly (rather than ``pt.log(self.likelihood(...))``) avoids
        computing the power-law tail :math:`A(B-t)^{-n}` and re-logging it,
        which underflows to 0.0 (and then to ``-inf``) once the tail decays
        far enough, and avoids computing :math:`\exp(-t^2/2)` in the core and
        re-logging that, which underflows for large :math:`|t|`.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation
            of the single-sided Crystal Ball log-PDF.
        """
        # Clip shape parameters at _SHAPE_CLIP_MIN > 0 (see _clip_shape): guards
        # the sigma/alpha divisors and the power-law exponent n against <= 0
        # inputs. m and m0 are positions and stay unclipped. Element-wise; keeps
        # each input's shape.
        alpha = _clip_shape(context[self._parameters["alpha"]])
        m = context[self.m]
        m0 = context[self._parameters["m0"]]
        n = _clip_shape(context[self._parameters["n"]])
        sigma = _clip_shape(context[self._parameters["sigma"]])

        B = (n / alpha) - alpha
        t = (m - m0) / sigma

        log_tail = _crystalball_log_tail(n, alpha, B - t)
        log_core = -(t**2) / 2

        return cast(TensorVar, pt.switch(t < -alpha, log_tail, log_core))


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
    alpha_L: str | float
    alpha_R: str | float
    m: str
    m0: str | float
    n_L: str | float
    n_R: str | float
    sigma_R: str | float
    sigma_L: str | float

    def likelihood(self, context: Context) -> TensorVar:
        """
        Evaluate the Crystal Ball distribution.

        Implements the ROOT RooCrystalBall formula with proper parameter validation.
        All shape parameters (alpha, n, sigma) are assumed to be positive.
        """
        # Clip shape parameters at _SHAPE_CLIP_MIN > 0 (see _clip_shape): guards
        # the sigma/alpha divisors and the power-law exponents n against <= 0
        # inputs on either side. m and m0 are positions and stay unclipped.
        # Element-wise; keeps each input's shape.
        alpha_L = _clip_shape(context[self._parameters["alpha_L"]])
        alpha_R = _clip_shape(context[self._parameters["alpha_R"]])
        m = context[self.m]
        m0 = context[self._parameters["m0"]]
        n_L = _clip_shape(context[self._parameters["n_L"]])
        n_R = _clip_shape(context[self._parameters["n_R"]])
        sigma_L = _clip_shape(context[self._parameters["sigma_L"]])
        sigma_R = _clip_shape(context[self._parameters["sigma_R"]])

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

    def log_likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the double-sided Crystal Ball log-PDF.

        Analytic log form of :meth:`likelihood`, piecewise over the same
        four regions (left tail, left core, right core, right tail):

        .. math::

            \log f(m) = \begin{cases}
            \log A_L - n_L \log(B_L - t_L), & t_L < -\alpha_L \\
            -\frac{1}{2} t_L^2, & -\alpha_L \leq t_L \leq 0 \\
            -\frac{1}{2} t_R^2, & 0 < t_L,\ t_R \leq \alpha_R \\
            \log A_R - n_R \log(B_R + t_R), & \text{otherwise}
            \end{cases}

        via ``_crystalball_log_tail`` for both tail branches (shared with
        :class:`CrystalBallDist`, which uses the same power-law tail form on
        its single tail). Evaluating this directly (rather than
        ``pt.log(self.likelihood(...))``) avoids computing each region's
        probability-space value and re-logging it, which underflows to 0.0
        (and then to ``-inf``) far enough into either tail or core.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation
            of the double-sided Crystal Ball log-PDF.
        """
        # Clip shape parameters at _SHAPE_CLIP_MIN > 0 (see _clip_shape): guards
        # the sigma/alpha divisors and the power-law exponents n against <= 0
        # inputs on either side. m and m0 are positions and stay unclipped.
        # Element-wise; keeps each input's shape.
        alpha_L = _clip_shape(context[self._parameters["alpha_L"]])
        alpha_R = _clip_shape(context[self._parameters["alpha_R"]])
        m = context[self.m]
        m0 = context[self._parameters["m0"]]
        n_L = _clip_shape(context[self._parameters["n_L"]])
        n_R = _clip_shape(context[self._parameters["n_R"]])
        sigma_L = _clip_shape(context[self._parameters["sigma_L"]])
        sigma_R = _clip_shape(context[self._parameters["sigma_R"]])

        B_L = (n_L / alpha_L) - alpha_L
        B_R = (n_R / alpha_R) - alpha_R

        t_L = (m - m0) / sigma_L
        t_R = (m - m0) / sigma_R

        log_left_tail = _crystalball_log_tail(n_L, alpha_L, B_L - t_L)
        log_core_left = -(t_L**2) / 2
        log_core_right = -(t_R**2) / 2
        log_right_tail = _crystalball_log_tail(n_R, alpha_R, B_R + t_R)

        return cast(
            TensorVar,
            pt.switch(
                t_L < -alpha_L,
                log_left_tail,
                pt.switch(
                    t_L <= 0,
                    log_core_left,
                    pt.switch(t_R <= alpha_R, log_core_right, log_right_tail),
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

        # Clip shape parameters at _SHAPE_CLIP_MIN > 0 (see _clip_shape): guards
        # the sigma/alpha divisors and the power-law exponents n against <= 0
        # inputs on either side. m and m0 are positions and stay unclipped.
        # Element-wise; keeps each input's shape.
        alpha_L = _clip_shape(context[self._parameters["alpha_L"]])
        alpha_R = _clip_shape(context[self._parameters["alpha_R"]])
        m = context[self.m]
        m0 = context[self._parameters["m0"]]
        n_L = _clip_shape(context[self._parameters["n_L"]])
        n_R = _clip_shape(context[self._parameters["n_R"]])
        sigma_L = _clip_shape(context[self._parameters["sigma_L"]])
        sigma_R = _clip_shape(context[self._parameters["sigma_R"]])

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
            # Guard the argument (not just the caller's switch output): u is
            # only positive within the left tail region (t_L < -alpha_L); the
            # outer switch below evaluates tail_left(B_L - t_L) unconditionally
            # for every m, including the core/right regions where B_L - t_L is
            # zero or negative. PyTensor differentiates every branch of a
            # switch, so log(u) / u ** (1 - n_L) at a non-positive u would
            # otherwise contribute a NaN gradient there that survives
            # multiplication by the switch's zero mask (0 * NaN = NaN).
            # Substituting a safe positive placeholder keeps both the value
            # and the gradient finite; the outer switch discards this
            # branch's result wherever u would have been invalid.
            safe_u = pt.switch(u > 0, u, 1.0)
            return cast(
                TensorVar,
                sigma_L
                * A_L
                * pt.switch(
                    pt.eq(n_L, 1.0), -pt.log(safe_u), safe_u ** (1.0 - n_L) / denom_L
                ),
            )

        def tail_right(u: TensorVar) -> TensorVar:
            # See tail_left's guard above: tail_right(B_R + t_R) is likewise
            # evaluated unconditionally for every m by the outer switch below,
            # including the left/core regions where B_R + t_R is non-positive.
            safe_u = pt.switch(u > 0, u, 1.0)
            return cast(
                TensorVar,
                sigma_R
                * A_R
                * pt.switch(
                    pt.eq(n_R, 1.0), pt.log(safe_u), -(safe_u ** (1.0 - n_R)) / denom_R
                ),
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
        :external+hs3:ref:`argus_dist <hs3.argus-distribution>`
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

    def log_likelihood(self, context: Context) -> TensorVar:
        r"""
        Builds a symbolic expression for the ARGUS log-PDF.

        Analytic log form of :meth:`likelihood`, restricted to the ARGUS
        kinematic domain :math:`1 - r^2 > 0` (i.e. :math:`m < m_0`), where
        :math:`r = m/m_0`:

        .. math::

            \log f(m) = \log m + p \log(1 - r^2) + c (1 - r^2)

        Outside the kinematic bound the ARGUS density is 0 by definition
        (ROOT's ``RooArgusBG`` restricts support to ``m < m0``), so ``-inf``
        is the correct log-density there -- not merely a numerical artifact
        to suppress. This is why this method intentionally diverges from
        ``pt.log(self.likelihood(...))`` outside the bound: :meth:`likelihood`
        computes ``bracket_term**p`` unconditionally, which is NaN there for
        the non-integer ``p`` values ARGUS normally uses (a real, non-integer
        power of a negative base), rather than 0. Inside the bound the two
        agree exactly.

        Guards the argument of both logs (not just the switch's output): the
        bracket term is clamped to a safe positive placeholder before either
        ``pt.log`` call, so the in-range branch's gradient stays finite when
        PyTensor differentiates it at points where that branch is not
        selected (bracket_term <= 0) -- the same guard idiom as
        :class:`CrystalBallDist`'s tail guard (``_crystalball_log_tail``).
        Without the clamp, the in-range branch's un-selected-branch gradient
        would multiply an infinite/NaN local derivative of ``pt.log`` at a
        non-positive argument by the switch's zero mask (``0 * NaN = NaN``).

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation
            of the ARGUS log-PDF.
        """
        m = context[self._parameters["mass"]]
        m0 = context[self._parameters["resonance"]]
        c = context[self._parameters["slope"]]
        p = context[self._parameters["power"]]

        ratio_squared = (m / m0) ** 2
        bracket_term = 1.0 - ratio_squared

        safe_bracket = pt.switch(bracket_term > 0, bracket_term, 1.0)
        log_valid = pt.log(m) + p * pt.log(safe_bracket) + c * safe_bracket

        return cast(TensorVar, pt.switch(bracket_term > 0, log_valid, -np.inf))


def _barrier_factor_squared(w: TensorVar, ell: int) -> TensorVar:
    r"""
    Blatt-Weisskopf factor :math:`n_l^2 = w^l h_l^2` as a function of :math:`w = z^2`.

    HS3 defines :math:`h_l^2(z) = 1 / (P_l(z) P_l(-z))` with
    :math:`P_0 = 1`, :math:`P_1 = 1 - iz` and
    :math:`P_{l+1} = (2l+1) P_l - z^2 P_{l-1}`. The product
    :math:`P_l(z) P_l(-z)` is even in :math:`z` with real coefficients, so it is
    a real polynomial in :math:`w`. Its coefficients are computed here (``ell``
    is a plain integer fixed when the graph is built) and evaluated in real
    arithmetic, which stays valid for :math:`w < 0`, i.e. an imaginary breakup
    momentum below threshold.

    Args:
        w: :math:`(qR)^2`, real, of any broadcastable shape.
        ell: Orbital angular momentum :math:`l \geq 0`.

    Returns:
        Symbolic :math:`n_l^2`, with the shape of ``w``.
    """
    # Coefficients of P_l(z), lowest power of z first (complex).
    p_prev = np.array([1.0 + 0j])
    p_curr = np.array([1.0 + 0j, -1.0j])
    for k in range(1, ell):
        # P_{k+1} = (2k+1) P_k - z^2 P_{k-1}, padded to a common length
        z2_p_prev = np.concatenate([[0j, 0j], p_prev])
        scaled = np.concatenate(
            [(2 * k + 1) * p_curr, np.zeros(len(z2_p_prev) - len(p_curr))]
        )
        p_prev, p_curr = p_curr, scaled - z2_p_prev
    poly = p_prev if ell == 0 else p_curr
    # P_l(-z) flips the sign of the odd powers of z.
    flipped = poly * (-1.0) ** np.arange(len(poly))
    # P_l(z) P_l(-z) is even in z: keep the even powers, i.e. powers of w = z^2.
    denominator_coeffs = np.convolve(poly, flipped)[::2].real

    denominator = pt.constant(denominator_coeffs[-1])
    for coeff in denominator_coeffs[-2::-1]:  # Horner's scheme
        denominator = denominator * w + coeff
    return cast(TensorVar, w**ell / denominator)


class RelativisticBreitWignerChannel(BaseModel):
    r"""
    One two-body decay channel of a :class:`RelativisticBreitWignerDist`.

    Field names follow HS3. The angular momentum is stored as ``ell`` and
    serialized as ``l``, avoiding an ambiguous single-letter attribute name.

    Parameters:
        name (str | None): Name of the final state (optional).
        Gamma (str | float): Non-negative width coefficient
            :math:`\Gamma_{\text{BW},i}`, in mass units.
        m1 (str | float): Mass of the first daughter (default 0).
        m2 (str | float): Mass of the second daughter (default 0).
        ell (int): Orbital angular momentum :math:`l_i \geq 0` (``l`` in HS3,
            default 0).
        R (str | float): Radius parameter :math:`R_i`, in inverse mass units
            (default 3, i.e. 3 GeV\ :sup:`-1`).
    """

    model_config = ConfigDict(serialize_by_alias=True, populate_by_name=True)

    name: str | None = None
    Gamma: str | float | int
    m1: str | float | int = 0.0
    m2: str | float | int = 0.0
    ell: int = Field(default=0, ge=0, alias="l")
    R: str | float | int = 3.0


class RelativisticBreitWignerDist(Distribution):
    r"""
    Relativistic Breit-Wigner distribution for a resonance in a two-body mass spectrum.

    Implements the HS3 line shape for the invariant mass :math:`m` of the
    observed two-particle system, with the resonance coupled to a list of
    two-body channels (the first is the observed final state). For
    :math:`m > m_{1,1} + m_{2,1}`:

    .. math::

        f(m) \propto \frac{m\,\Gamma_1(m)}
        {\left| m_\text{BW}^2 - m^2 - i\,m_\text{BW}\,\Gamma(m) \right|^2},
        \qquad \Gamma(m) = \sum_i \Gamma_i(m)

    and :math:`f(m) = 0` below the observed channel's threshold. Each channel
    contributes

    .. math::

        \Gamma_i(m) = \Gamma_{\text{BW},i}\; n_{l_i}^2(m)\; \rho_i(m), \quad
        \rho_i = \frac{2 q_i}{m}, \quad
        n_{l}^2 = z^{2l} h_l^2(z), \quad z = q_i R_i,

    with breakup momentum

    .. math::

        q_i(m) = \frac{\sqrt{\left(m^2 - (m_{1i}+m_{2i})^2\right)
        \left(m^2 - (m_{1i}-m_{2i})^2\right)}}{2m}

    and Blatt-Weisskopf factors :math:`h_l^2(z) = 1/(P_l(z)P_l(-z))`,
    :math:`P_0 = 1`, :math:`P_1 = 1 - iz`,
    :math:`P_{l+1} = (2l+1)P_l - z^2 P_{l-1}` (for example
    :math:`h_1^2 = 1/(1+z^2)`).

    Channels that are closed at :math:`m` still enter the denominator: their
    breakup momentum is continued analytically to :math:`q = +i\sqrt{-q^2}`
    between :math:`|m_{1}-m_{2}|` and :math:`m_{1}+m_{2}`, and :math:`q` is
    negative real below :math:`|m_{1}-m_{2}|`. This implementation evaluates
    the complex width in real arithmetic: open channels contribute to
    :math:`\mathrm{Re}\,\Gamma` and continued channels to
    :math:`\mathrm{Im}\,\Gamma`, so the denominator is
    :math:`(m_\text{BW}^2 - m^2 + m_\text{BW}\,\mathrm{Im}\,\Gamma)^2 +
    (m_\text{BW}\,\mathrm{Re}\,\Gamma)^2`.

    The density is normalized over the observable's domain by the standard
    automatic normalization (64-point Gauss-Legendre quadrature, see
    :doc:`/normalization`). The line shape has a square-root onset at the
    observed channel's threshold, which the quadrature resolves poorly if the
    threshold lies inside the domain; start the domain at or above the
    threshold.

    Parameters:
        mass (str): Name of the observable mass variable :math:`m`.
        resonance (str | float): Nominal mass :math:`m_\text{BW}` (strictly
            positive; not in general the pole mass).
        channels (list[RelativisticBreitWignerChannel]): Non-empty list of
            channels, observed channel first. The first channel's ``Gamma``
            must be strictly positive.

    Note:
        Following HS3, ``Gamma`` is a width coefficient, not the partial width
        at the nominal mass: no division by the phase space at
        :math:`m_\text{BW}` is made, unlike the nominal-partial-width
        convention of Eq. (50.35) in the PDG review. The density carries the
        factor :math:`m` (the Jacobian from :math:`m^2` to :math:`m`) and no
        mass-dependent production factors, so it differs from ROOT's
        ``RooBreitWigner`` (a constant-width, non-relativistic Cauchy shape)
        and from ROOT's ``RooRelBreitWigner``, which uses its own
        parametrization of the running width and barrier factors.

    HS3 Reference:
        :external+hs3:ref:`relativistic_breit_wigner_dist <hs3.relativistic-breit-wigner-distribution>`
    """

    type: Literal["relativistic_breit_wigner_dist"] = "relativistic_breit_wigner_dist"
    mass: str | float | int
    resonance: str | float | int
    channels: list[RelativisticBreitWignerChannel] = Field(
        min_length=1, json_schema_extra={"preprocess": False}
    )

    @model_validator(mode="after")
    def _register_channel_parameters(self) -> RelativisticBreitWignerDist:
        """Register channel fields as parameters, mirroring the flat-field processing."""
        for index, channel in enumerate(self.channels):
            for field_name in ("Gamma", "m1", "m2", "R"):
                key = f"channels[{index}].{field_name}"
                value = getattr(channel, field_name)
                if isinstance(value, str):
                    self._parameters[key] = value
                    continue
                constant_name = f"constant_{self.name}_{key}"
                self._parameters[key] = constant_name
                self._constants_values[constant_name] = value
        return self

    def _channel_widths(
        self, context: Context, index: int, m: TensorVar
    ) -> tuple[TensorVar, TensorVar]:
        r"""
        Real and imaginary parts of :math:`\Gamma_i(m)` for one channel.

        Channel parameters are scalars and ``m`` has any shape, so both
        returned tensors have the shape of ``m``.

        With :math:`q^2 = (m^2 - s_+^2)(m^2 - s_-^2) / (4m^2)`, :math:`q^2 > 0`
        gives a real :math:`q` (negative below :math:`|m_1 - m_2|`) and
        :math:`q^2 < 0` gives :math:`q = i\sqrt{-q^2}`. The Blatt-Weisskopf
        factor is real in both cases (it depends on :math:`q^2` only), so
        :math:`\Gamma_i` is real for :math:`q^2 > 0` and purely imaginary for
        :math:`q^2 < 0`.

        Args:
            context: Mapping of names to pytensor variables.
            index: Channel position in ``channels``.
            m: Invariant mass.

        Returns:
            ``(Re Gamma_i, Im Gamma_i)``.
        """
        gamma, m1, m2, radius = (
            context[self._parameters[f"channels[{index}].{name}"]]
            for name in ("Gamma", "m1", "m2", "R")
        )
        above_threshold = m**2 - (m1 + m2) ** 2
        above_pseudothreshold = m**2 - (m1 - m2) ** 2
        q_squared = above_threshold * above_pseudothreshold / (4.0 * m**2)

        # |q| = sqrt(|q^2|), with the argument of sqrt guarded so its gradient
        # stays finite at q^2 == 0 (the switch's zero mask would otherwise
        # multiply an infinite local derivative).
        abs_q2 = pt.abs(q_squared)
        safe_abs_q2 = pt.switch(abs_q2 > 0, abs_q2, 1.0)
        abs_q = pt.switch(abs_q2 > 0, pt.sqrt(safe_abs_q2), 0.0)

        barrier = _barrier_factor_squared(
            q_squared * radius**2, self.channels[index].ell
        )
        magnitude = gamma * barrier * 2.0 * abs_q / m
        # q is negative real only below |m1 - m2|, where both factors are negative
        sign = pt.switch(above_threshold > 0, 1.0, -1.0)
        real = pt.switch(q_squared > 0, sign * magnitude, 0.0)
        imag = pt.switch(q_squared < 0, magnitude, 0.0)
        return cast(TensorVar, real), cast(TensorVar, imag)

    def likelihood(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the unnormalized relativistic Breit-Wigner density.

        The density is zero at and below the observed channel's threshold.
        Normalization over the observable's domain is applied by
        :meth:`pyhs3.distributions.Distribution.expression`.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the
            unnormalized density, with the shape of the mass variable.
        """
        m = context[self._parameters["mass"]]
        m_bw = context[self._parameters["resonance"]]

        widths = [
            self._channel_widths(context, index, m)
            for index in range(len(self.channels))
        ]
        total_real = balanced_sum([real for real, _ in widths], pt.constant(0.0))
        total_imag = balanced_sum([imag for _, imag in widths], pt.constant(0.0))

        # |m_BW^2 - m^2 - i m_BW (Re + i Im)|^2, in real arithmetic
        denominator = (m_bw**2 - m**2 + m_bw * total_imag) ** 2 + (
            m_bw * total_real
        ) ** 2

        observed_real, _ = widths[0]
        m1 = context[self._parameters["channels[0].m1"]]
        m2 = context[self._parameters["channels[0].m2"]]
        numerator = pt.switch(m > m1 + m2, m * observed_real, 0.0)

        return cast(TensorVar, numerator / denominator)


# Export list of physics distribution classes
__all__ = [
    "ArgusDist",
    "AsymmetricCrystalBallDist",
    "CrystalBallDist",
    "RelativisticBreitWignerChannel",
    "RelativisticBreitWignerDist",
    "distributions",
]

# Registry dict mapping type strings to classes
distributions: dict[str, type[Distribution]] = {
    "crystalball_dist": CrystalBallDist,
    "crystalball_doublesided_dist": AsymmetricCrystalBallDist,
    "argus_dist": ArgusDist,
    "relativistic_breit_wigner_dist": RelativisticBreitWignerDist,
}
