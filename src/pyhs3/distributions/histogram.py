"""
Histogram-based distribution implementations.

Provides classes for handling histogram-based probability distributions, including
the HistogramDist class for non-parametric modeling with binned data and related
helper classes for managing histogram data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from pyhs3.axes import BinnedAxis
from pyhs3.distributions.core import Distribution


class HistogramData(BaseModel):
    """
    Histogram data implementation for the HistogramFunction.

    Parameters:
        axes: list of BinnedAxis used to describe the binning
        contents: list of bin content parameter values
    """

    axes: list[BinnedAxis] = Field(..., repr=False)
    contents: list[float] = Field(..., repr=False)


class HistogramDist(Distribution):  # pylint: disable=abstract-method
    r"""
    Histogram probability distribution.

    Implements a histogram-based probability density function as defined in ROOT's
    RooHistPdf. Used for non-parametric modeling with binned data.

    .. math::

        f(x) = \frac{h_i}{\mathcal{M}}

    where $h_i$ is the bin content for the bin containing $x$.

    Parameters:
        x (str): Input variable name.
        bin_contents (list[str]): Array of bin content parameter names.

    Note:
        The bin boundaries are typically defined by the domain. This implementation
        assumes uniform binning over the domain range.
    """

    type: Literal["histogram_dist"] = "histogram_dist"
    data: HistogramData = Field(..., json_schema_extra={"preprocess": False})


# Registry of histogram distributions
# NOTE: HistogramDist is intentionally NOT registered here because it has no
# likelihood() implementation. Workspaces referencing "histogram_dist" will get
# the normal clean unknown-type validation error from the discriminated union.
distributions: dict[str, type[Distribution]] = {}

# Define what should be exported from this module
__all__ = [
    "HistogramData",
    "HistogramDist",
    "distributions",
]
