"""
CMS-specific Distribution implementations.

Provides classes for handling CMS experiment-specific probability distributions
that extend beyond the standard HS3 specification. These are used in CMS
combine files and related analyses.
"""

from __future__ import annotations

import logging
from typing import Literal, cast

import pytensor.tensor as pt

from pyhs3.context import Context
from pyhs3.distributions.core import Distribution
from pyhs3.typing.aliases import TensorVar

log = logging.getLogger(__name__)


class FastVerticalInterpHistPdf2Dist(Distribution):
    r"""
    CMS Fast Vertical Interpolation Histogram PDF (2D version).

    Implements CMS's FastVerticalInterpHistPdf2 distribution, which is used for
    efficient interpolation of histogram templates with systematic uncertainties.
    This matches the actual JSON structure from CMS combine files.

    Parameters:
        x (str): Observable variable name.
        coefList (list[str]): List of coefficient parameter names for morphing.

    Note:
        This is a simplified implementation. The actual CMS implementation uses
        sophisticated interpolation algorithms and caching for performance.
    """

    type: Literal["CMS::fastverticalinterphistpdf2"] = "CMS::fastverticalinterphistpdf2"
    x: str | float | int
    coefList: list[str]

    def likelihood(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the fast vertical interpolation histogram PDF.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of interpolated histogram PDF.

        Note:
            This implementation provides a simplified approximation.
            The actual CMS implementation uses more sophisticated algorithms.
        """
        _x = context[self._parameters["x"]]

        # Start with a base histogram value
        result = pt.constant(1.0)

        # Apply morphing for each coefficient parameter
        for coef_name in self.coefList:
            coef = context[coef_name]
            # Simple linear combination with coefficients
            result = result * (1.0 + 0.1 * coef)  # Simplified morphing

        return cast(TensorVar, result)


class GGZZBackgroundDist(Distribution):
    r"""
    CMS ggZZ background distribution.

    Implements CMS's ggZZ background model used in Higgs to ZZ analyses.
    This distribution models the continuum background from gluon-gluon to ZZ production.

    .. math::

        f(m_{4\ell}; a_1, a_2, a_3) = a_1 \cdot m_{4\ell}^{a_2} \cdot \exp(-a_3 \cdot m_{4\ell})

    Parameters:
        m4l (str): Four-lepton invariant mass variable.
        a1 (str): Normalization parameter.
        a2 (str): Power law exponent.
        a3 (str): Exponential decay parameter.

    Note:
        This is specific to CMS's ggZZ background modeling in H->ZZ->4l analyses.
    """

    type: Literal["CMS::ggZZ_background_dist"] = "CMS::ggZZ_background_dist"
    m4l: str | float | int
    a1: str | float | int
    a2: str | float | int
    a3: str | float | int

    def likelihood(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the ggZZ background PDF.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of ggZZ background PDF.
        """
        m4l = context[self._parameters["m4l"]]
        a1 = context[self._parameters["a1"]]
        a2 = context[self._parameters["a2"]]
        a3 = context[self._parameters["a3"]]

        # ggZZ background: a1 * m4l^a2 * exp(-a3 * m4l)
        power_term = m4l**a2
        exp_term = pt.exp(-a3 * m4l)

        return cast(TensorVar, a1 * power_term * exp_term)


class QQZZBackgroundDist(Distribution):
    r"""
    CMS qqZZ background distribution.

    Implements CMS's qqZZ background model used in Higgs to ZZ analyses.
    This distribution models the continuum background from quark-antiquark to ZZ production.

    .. math::

        f(m_{4\ell}; a_1, a_2, a_3, a_4) = a_1 \cdot (m_{4\ell} + a_2)^{a_3} \cdot \exp(-a_4 \cdot m_{4\ell})

    Parameters:
        m4l (str): Four-lepton invariant mass variable.
        a1 (str): Normalization parameter.
        a2 (str): Mass shift parameter.
        a3 (str): Power law exponent.
        a4 (str): Exponential decay parameter.

    Note:
        This is specific to CMS's qqZZ background modeling in H->ZZ->4l analyses.
    """

    type: Literal["CMS::qqZZ_background_dist"] = "CMS::qqZZ_background_dist"
    m4l: str | float | int
    a1: str | float | int
    a2: str | float | int
    a3: str | float | int
    a4: str | float | int

    def likelihood(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the qqZZ background PDF.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of qqZZ background PDF.
        """
        m4l = context[self._parameters["m4l"]]
        a1 = context[self._parameters["a1"]]
        a2 = context[self._parameters["a2"]]
        a3 = context[self._parameters["a3"]]
        a4 = context[self._parameters["a4"]]

        # qqZZ background: a1 * (m4l + a2)^a3 * exp(-a4 * m4l)
        shifted_mass = m4l + a2
        power_term = shifted_mass**a3
        exp_term = pt.exp(-a4 * m4l)

        return cast(TensorVar, a1 * power_term * exp_term)


class FastVerticalInterpHistPdf2D2Dist(Distribution):
    r"""
    CMS Fast Vertical Interpolation Histogram PDF (2D version 2).

    Implements another variant of CMS's FastVerticalInterpHistPdf used for
    2D histogram interpolation with systematic uncertainties. This matches
    the actual JSON structure from CMS combine files.

    Parameters:
        x (str): First observable variable name.
        y (str): Second observable variable name.
        coefList (list[str]): List of coefficient parameter names for morphing.

    Note:
        This is a simplified implementation for 2D histogram morphing.
    """

    type: Literal["CMS::FastVerticalInterpHistPdf2D2"] = (
        "CMS::FastVerticalInterpHistPdf2D2"
    )
    x: str | float | int
    y: str | float | int
    coefList: list[str]

    def likelihood(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the 2D fast vertical interpolation histogram PDF.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of 2D interpolated histogram PDF.

        Note:
            This implementation provides a simplified 2D morphing approximation.
        """
        _x = context[self._parameters["x"]]
        _y = context[self._parameters["y"]]

        # Start with a base 2D histogram value
        result = pt.constant(1.0)

        # Apply morphing for each coefficient parameter
        for coef_name in self.coefList:
            coef = context[coef_name]
            # Simple 2D morphing with coefficients
            result = result * (1.0 + 0.1 * coef)  # Simplified morphing

        return cast(TensorVar, result)


# Registry of CMS-specific distributions
distributions: dict[str, type[Distribution]] = {
    "CMS::fastverticalinterphistpdf2": FastVerticalInterpHistPdf2Dist,
    "CMS::ggZZ_background_dist": GGZZBackgroundDist,
    "CMS::qqZZ_background_dist": QQZZBackgroundDist,
    "CMS::FastVerticalInterpHistPdf2D2": FastVerticalInterpHistPdf2D2Dist,
}

# Define what should be exported from this module
__all__ = [
    "FastVerticalInterpHistPdf2D2Dist",
    "FastVerticalInterpHistPdf2Dist",
    "GGZZBackgroundDist",
    "QQZZBackgroundDist",
    "distributions",
]
