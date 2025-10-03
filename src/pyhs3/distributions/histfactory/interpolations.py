"""
HistFactory Distribution implementation.

Provides the HistFactoryDist class for handling binned statistical models
with samples and modifiers as defined in the HS3 specification.
"""

from __future__ import annotations

from typing import cast

import pytensor.tensor as pt

# Import existing distributions for constraint terms
from pyhs3.typing.aliases import TensorVar


def interpolate_lin(
    alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    r"""
    Linear interpolation between hi/lo values.

    .. math::

        f(\alpha) = \begin{cases}
        \text{nom} + \alpha \cdot (\text{hi} - \text{nom}) & \text{if } \alpha \geq 0 \\
        \text{nom} + \alpha \cdot (\text{nom} - \text{lo}) & \text{if } \alpha < 0
        \end{cases}
    """
    return cast(
        TensorVar,
        pt.where(alpha >= 0, nom + alpha * (hi - nom), nom + alpha * (nom - lo)),  # type: ignore[no-untyped-call]
    )


def interpolate_log(
    alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    r"""
    Logarithmic interpolation between hi/lo values.

    .. math::

        f(\alpha) = \begin{cases}
        \text{nom} \cdot \left(\frac{\text{hi}}{\text{nom}}\right)^{\alpha} & \text{if } \alpha \geq 0 \\
        \text{nom} \cdot \left(\frac{\text{lo}}{\text{nom}}\right)^{-\alpha} & \text{if } \alpha < 0
        \end{cases}
    """
    return cast(
        TensorVar,
        pt.where(  # type: ignore[no-untyped-call]
            alpha >= 0,
            nom * pt.power(hi / nom, alpha),  # type: ignore[no-untyped-call]
            nom * pt.power(lo / nom, -alpha),  # type: ignore[no-untyped-call]
        ),
    )


def interpolate_parabolic(
    alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    r"""
    Parabolic interpolation between hi/lo values.

    .. math::

        f(\alpha) = \begin{cases}
        \text{nom} + (2s+d)(\alpha-1) + (\text{hi} - \text{nom}) & \text{if } \alpha > 1 \\
        \text{nom} - (2s-d)(\alpha+1) + (\text{lo} - \text{nom}) & \text{if } \alpha < -1 \\
        \text{nom} + s \alpha^2 + d \alpha & \text{otherwise}
        \end{cases}

    where :math:`s = \frac{1}{2}(\text{hi} + \text{lo}) - \text{nom}` and :math:`d = \frac{1}{2}(\text{hi} - \text{lo})`.
    """
    s = 0.5 * (hi + lo) - nom
    d = 0.5 * (hi - lo)

    # Central parabolic region
    central = nom + s * alpha**2 + d * alpha

    # Linear extensions for |alpha| > 1
    high_ext = nom + (2 * s + d) * (alpha - 1) + (hi - nom)
    low_ext = nom - (2 * s - d) * (alpha + 1) + (lo - nom)

    return cast(
        TensorVar,
        pt.where(alpha > 1, high_ext, pt.where(alpha < -1, low_ext, central)),  # type: ignore[no-untyped-call]
    )


def interpolate_poly6(
    alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    r"""
    6th-order polynomial interpolation between hi/lo values.

    .. math::

        f(\alpha) = \begin{cases}
        \text{nom} + \alpha (\text{hi} - \text{nom}) & \text{if } \alpha > 1 \\
        \text{nom} + \alpha (\text{nom} - \text{lo}) & \text{if } \alpha < -1 \\
        \text{nom} + \alpha (S + \alpha A (15 + \alpha^2 (3\alpha^2 - 10))) & \text{otherwise}
        \end{cases}

    where :math:`S = \frac{1}{2}(\text{hi} - \text{lo})` and :math:`A = \frac{1}{16}(\text{hi} + \text{lo} - 2\text{nom})`.
    """
    S = 0.5 * (hi - lo)
    A = (1.0 / 16.0) * (hi + lo - 2 * nom)

    # 6th-order polynomial for |alpha| <= 1
    poly_term = S + alpha * A * (15 + alpha**2 * (3 * alpha**2 - 10))
    central = nom + alpha * poly_term

    # Linear extensions for |alpha| > 1
    high_ext = nom + alpha * (hi - nom)
    low_ext = nom + alpha * (nom - lo)

    return cast(
        TensorVar,
        pt.where(alpha > 1, high_ext, pt.where(alpha < -1, low_ext, central)),  # type: ignore[no-untyped-call]
    )


def interpolate_exp(
    alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    r"""
    Exponential interpolation between hi/lo values (pyhf code1 compatible).

    .. math::

        f(\alpha) = \begin{cases}
        \text{nom} \cdot \left(\frac{\text{hi}}{\text{nom}}\right)^{\alpha} & \text{if } \alpha \geq 0 \\
        \text{nom} \cdot \left(\frac{\text{lo}}{\text{nom}}\right)^{-\alpha} & \text{if } \alpha < 0
        \end{cases}
    """
    return cast(
        TensorVar,
        pt.where(  # type: ignore[no-untyped-call]
            alpha >= 0,
            nom * pt.power(hi / nom, alpha),  # type: ignore[no-untyped-call]
            nom * pt.power(lo / nom, -alpha),  # type: ignore[no-untyped-call]
        ),
    )


def interpolate_code4(
    alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar, alpha0: float = 1.0
) -> TensorVar:
    r"""
    pyhf code4 polynomial interpolation with exponential extrapolation (pyhf compatible).

    .. math::

        f(\alpha) = \begin{cases}
        \text{nom} \cdot \left(\frac{\text{hi}}{\text{nom}}\right)^{\alpha} & \text{if } \alpha \geq \alpha_0 \\
        \text{nom} \cdot \left(1 + \sum_{i=1}^6 a_i \alpha^i\right) & \text{if } |\alpha| < \alpha_0 \\
        \text{nom} \cdot \left(\frac{\text{lo}}{\text{nom}}\right)^{-\alpha} & \text{if } \alpha \leq -\alpha_0
        \end{cases}

    The polynomial coefficients are determined by boundary conditions for continuity.
    """

    # Ratios
    hi_ratio = hi / nom
    lo_ratio = lo / nom

    # Polynomial coefficients for code4 (alpha0=1)
    # These come from the pyhf implementation
    A_inv = [
        [15.0 / 16, -15.0 / 16, -7.0 / 16, -7.0 / 16, 1.0 / 16, -1.0 / 16],
        [3.0 / 2, 3.0 / 2, -9.0 / 16, 9.0 / 16, 1.0 / 16, 1.0 / 16],
        [-5.0 / 8, 5.0 / 8, 5.0 / 8, 5.0 / 8, -1.0 / 8, 1.0 / 8],
        [-3.0 / 2, -3.0 / 2, 7.0 / 8, -7.0 / 8, -1.0 / 8, -1.0 / 8],
        [3.0 / 16, -3.0 / 16, -3.0 / 16, -3.0 / 16, 1.0 / 16, -1.0 / 16],
        [1.0 / 2, 1.0 / 2, -5.0 / 16, 5.0 / 16, 1.0 / 16, 1.0 / 16],
    ]

    # Boundary values at alpha0
    hi_at_alpha0 = pt.power(hi_ratio, alpha0)  # type: ignore[no-untyped-call]
    lo_at_alpha0 = pt.power(lo_ratio, alpha0)  # type: ignore[no-untyped-call]

    # RHS vector b
    b = [
        hi_at_alpha0 - 1.0,
        lo_at_alpha0 - 1.0,
        pt.log(hi_ratio) * hi_at_alpha0,
        -pt.log(lo_ratio) * lo_at_alpha0,
        pt.power(pt.log(hi_ratio), 2) * hi_at_alpha0,  # type: ignore[no-untyped-call]
        pt.power(pt.log(lo_ratio), 2) * lo_at_alpha0,  # type: ignore[no-untyped-call]
    ]

    # Calculate polynomial coefficients a_i = A^(-1) * b
    coeffs = []
    for i in range(6):
        coeff = sum(A_inv[i][j] * b[j] for j in range(6))
        coeffs.append(coeff)

    # Polynomial evaluation: 1 + sum(a_i * alpha^i for i=1..6)
    alpha_powers = [alpha, alpha**2, alpha**3, alpha**4, alpha**5, alpha**6]
    poly_sum = sum(coeffs[i] * alpha_powers[i] for i in range(6))
    poly_result = nom * (1.0 + poly_sum)

    # Exponential extrapolation
    exp_hi = nom * pt.power(hi_ratio, alpha)  # type: ignore[no-untyped-call]
    exp_lo = nom * pt.power(lo_ratio, -alpha)  # type: ignore[no-untyped-call]

    # Combine based on alpha value
    result = pt.where(  # type: ignore[no-untyped-call]
        alpha >= alpha0,
        exp_hi,
        pt.where(alpha <= -alpha0, exp_lo, poly_result),  # type: ignore[no-untyped-call]
    )

    return cast(TensorVar, result)


def interpolate_code0(
    alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    r"""
    pyhf code0 piecewise-linear interpolation (additive deltas).

    .. math::

        f(\alpha) = I^0 + I_{\text{lin}}(\alpha; I^0, I^+, I^-)

    where

    .. math::

        I_{\text{lin}}(\alpha; I^0, I^+, I^-) = \begin{cases}
        \alpha(I^+ - I^0) & \text{if } \alpha \geq 0 \\
        \alpha(I^0 - I^-) & \text{if } \alpha < 0
        \end{cases}
    """
    return cast(
        TensorVar,
        pt.where(  # type: ignore[no-untyped-call]
            alpha >= 0,
            nom + alpha * (hi - nom),
            nom + alpha * (nom - lo),
        ),
    )


def interpolate_code1(
    alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    r"""
    pyhf code1 piecewise-exponential interpolation (multiplicative factors).

    .. math::

        f(\alpha) = I^0 \cdot I_{\text{exp}}(\alpha; I^0, I^+, I^-)

    where

    .. math::

        I_{\text{exp}}(\alpha; I^0, I^+, I^-) = \begin{cases}
        \left(\frac{I^+}{I^0}\right)^{\alpha} & \text{if } \alpha \geq 0 \\
        \left(\frac{I^-}{I^0}\right)^{-\alpha} & \text{if } \alpha < 0
        \end{cases}
    """
    return cast(
        TensorVar,
        pt.where(  # type: ignore[no-untyped-call]
            alpha >= 0,
            nom * pt.power(hi / nom, alpha),  # type: ignore[no-untyped-call]
            nom * pt.power(lo / nom, -alpha),  # type: ignore[no-untyped-call]
        ),
    )


def interpolate_code2(
    alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    r"""
    pyhf code2 quadratic interpolation with linear extrapolation (additive deltas).

    .. math::

        f(\alpha) = I^0 + I_{\text{quad|lin}}(\alpha; I^0, I^+, I^-)

    where

    .. math::

        I_{\text{quad|lin}}(\alpha; I^0, I^+, I^-) = \begin{cases}
        (b + 2a)(\alpha - 1) & \text{if } \alpha \geq 1 \\
        a\alpha^2 + b\alpha & \text{if } |\alpha| < 1 \\
        (b - 2a)(\alpha + 1) & \text{if } \alpha < -1
        \end{cases}

    with :math:`a = \frac{1}{2}((I^+ - I^0) + (I^- - I^0))` and :math:`b = \frac{1}{2}((I^+ - I^0) - (I^- - I^0))`.
    """
    # Calculate quadratic coefficients
    hi_delta = hi - nom
    lo_delta = lo - nom

    a = 0.5 * (hi_delta + lo_delta)
    b = 0.5 * (hi_delta - lo_delta)

    # Quadratic interpolation for |alpha| < 1
    quad_result = nom + a * alpha**2 + b * alpha

    # Linear extrapolation for |alpha| >= 1
    high_ext = nom + (b + 2 * a) * (alpha - 1)
    low_ext = nom + (b - 2 * a) * (alpha + 1)

    return cast(
        TensorVar,
        pt.where(  # type: ignore[no-untyped-call]
            alpha > 1,
            high_ext,
            pt.where(alpha < -1, low_ext, quad_result),  # type: ignore[no-untyped-call]
        ),
    )


def interpolate_code4p(
    alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar, alpha0: float = 1.0
) -> TensorVar:
    r"""
    pyhf code4p piecewise-linear with polynomial interpolation for |alpha| < 1 (additive deltas).

    .. math::

        f(\alpha) = I^0 + I_{\text{lin|poly}}(\alpha; I^0, I^+, I^-)

    Linear extrapolation for :math:`|\alpha| \geq \alpha_0`, polynomial interpolation for :math:`|\alpha| < \alpha_0`.
    """
    hi_delta = hi - nom
    lo_delta = lo - nom

    # For |alpha| >= alpha0, use linear extrapolation
    linear_hi = nom + alpha * hi_delta
    linear_lo = nom + alpha * lo_delta

    # For |alpha| < alpha0, use polynomial (similar to code4 but additive)
    # Calculate polynomial coefficients for boundary matching
    A_inv = [
        [15.0 / 16, -15.0 / 16, -7.0 / 16, -7.0 / 16, 1.0 / 16, -1.0 / 16],
        [3.0 / 2, 3.0 / 2, -9.0 / 16, 9.0 / 16, 1.0 / 16, 1.0 / 16],
        [-5.0 / 8, 5.0 / 8, 5.0 / 8, 5.0 / 8, -1.0 / 8, 1.0 / 8],
        [-3.0 / 2, -3.0 / 2, 7.0 / 8, -7.0 / 8, -1.0 / 8, -1.0 / 8],
        [3.0 / 16, -3.0 / 16, -3.0 / 16, -3.0 / 16, 1.0 / 16, -1.0 / 16],
        [1.0 / 2, 1.0 / 2, -5.0 / 16, 5.0 / 16, 1.0 / 16, 1.0 / 16],
    ]

    # Boundary values at alpha0 (additive form)
    hi_at_alpha0 = alpha0 * hi_delta
    lo_at_alpha0 = alpha0 * lo_delta

    # RHS vector b (for additive deltas)
    b = [
        hi_at_alpha0,
        lo_at_alpha0,
        hi_delta,
        -lo_delta,
        0.0,  # Second derivatives
        0.0,
    ]

    # Calculate polynomial coefficients
    coeffs = []
    for i in range(6):
        coeff = sum(A_inv[i][j] * b[j] for j in range(6))
        coeffs.append(coeff)

    # Polynomial evaluation: sum(a_i * alpha^i for i=1..6)
    alpha_powers = [alpha, alpha**2, alpha**3, alpha**4, alpha**5, alpha**6]
    poly_result = nom + sum(coeffs[i] * alpha_powers[i] for i in range(6))

    # Combine based on alpha value
    result = pt.where(  # type: ignore[no-untyped-call]
        alpha >= alpha0,
        linear_hi,
        pt.where(alpha <= -alpha0, linear_lo, poly_result),  # type: ignore[no-untyped-call]
    )

    return cast(TensorVar, result)


def apply_interpolation(
    method: str, alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    """Apply the specified interpolation method."""
    if method == "lin":
        return interpolate_lin(alpha, nom, hi, lo)
    if method == "log":
        return interpolate_log(alpha, nom, hi, lo)
    if method == "exp":
        return interpolate_exp(alpha, nom, hi, lo)
    if method == "code0":
        return interpolate_code0(alpha, nom, hi, lo)
    if method == "code1":
        return interpolate_code1(alpha, nom, hi, lo)
    if method == "code2":
        return interpolate_code2(alpha, nom, hi, lo)
    if method == "code4":
        return interpolate_code4(alpha, nom, hi, lo)
    if method == "code4p":
        return interpolate_code4p(alpha, nom, hi, lo)
    if method == "parabolic":
        return interpolate_parabolic(alpha, nom, hi, lo)
    if method == "poly6":
        return interpolate_poly6(alpha, nom, hi, lo)
    # Default to linear interpolation
    return interpolate_lin(alpha, nom, hi, lo)
