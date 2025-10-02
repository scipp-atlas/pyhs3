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


def apply_interpolation(
    method: str, alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    """Apply the specified interpolation method."""
    if method == "lin":
        return interpolate_lin(alpha, nom, hi, lo)
    if method == "log":
        return interpolate_log(alpha, nom, hi, lo)
    if method == "parabolic":
        return interpolate_parabolic(alpha, nom, hi, lo)
    if method == "poly6":
        return interpolate_poly6(alpha, nom, hi, lo)
    # Default to linear interpolation
    return interpolate_lin(alpha, nom, hi, lo)
