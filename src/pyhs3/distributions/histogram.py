"""
Histogram-based distribution implementations.

Provides classes for handling histogram-based probability distributions, including
the HistogramDist class for non-parametric modeling with binned data and related
helper classes for managing histogram data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pyhs3.context import Context
from pyhs3.data import Axis
from pyhs3.distributions.core import Distribution
from pyhs3.typing.aliases import TensorVar


class HistogramData(BaseModel):
    """
    Histogram data implementation for the HistogramFunction.

    Parameters:
        axes: list of Axis used to describe the binning
        contents: list of bin content parameter values
    """

    model_config = ConfigDict()

    axes: list[Axis] = Field(..., repr=False)
    contents: list[float] = Field(..., repr=False)


class HistogramDist(Distribution):
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

    def log_expression(self, _context: Context) -> TensorVar:
        """Log-PDF expression - NEEDS IMPLEMENTATION."""
        msg = f"log_expression not implemented for {self.type}"
        raise NotImplementedError(msg)


# Registry of histogram distributions
distributions: dict[str, type[Distribution]] = {
    "histogram_dist": HistogramDist,
}

# Define what should be exported from this module
__all__ = [
    "HistogramData",
    "HistogramDist",
    "distributions",
]
