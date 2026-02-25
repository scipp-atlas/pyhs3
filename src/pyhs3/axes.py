"""
HS3 Axis implementations.

Provides Pydantic classes for handling HS3 axis specifications
including unbinned axes (with min/max bounds) and binned axes
(with regular or irregular binning).
"""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

import hist
import numpy as np
from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    model_validator,
)

from pyhs3.collections import NamedCollection, NamedModel
from pyhs3.exceptions import custom_error_msg


class Axis(NamedModel):
    """
    Base axis specification for data coordinates.

    Attributes:
        name: Name of the axis/variable
    """

    model_config = ConfigDict()


class UnbinnedAxis(Axis):
    """
    Axis for unbinned data.

    Alias for Axis with required min/max bounds.

    Attributes:
        name: Name of the axis/variable
        min: Minimum value (required)
        max: Maximum value (required)
    """

    min: float = Field(..., repr=False)
    max: float = Field(..., repr=False)

    @model_validator(mode="after")
    def check_min_le_max(self) -> UnbinnedAxis:
        """Validate that max >= min when both are provided."""
        if self.max < self.min:
            msg = f"UnbinnedAxis '{self.name}': max ({self.max}) must be >= min ({self.min})"
            raise ValueError(msg)
        return self

    def to_hist(self) -> Any:
        """
        Convert this axis to a hist.axis object.

        This is a base implementation that should be overridden by subclasses
        that have specific binning information (like BinnedAxis).

        Returns:
            A hist.axis object

        Raises:
            ValueError: If axis has insufficient binning information
        """
        msg = f"UnbinnedAxis '{self.name}' does not have binning information for histogram conversion"
        raise ValueError(msg)


class ConstantAxis(Axis):
    """
    Axis for constant data.

    Alias for Axis with required const field.

    Attributes:
        name: Name of the axis/variable
        const: true (required)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    const: Literal[True] = Field(True, repr=False, init=False)


class RegularAxis(Axis):
    """
    Attributes:
        name: Name of the axis/variable
        min: Minimum value
        max: Maximum value
        nbins: Number of bins (for regular binning)
    """

    min: float = Field(..., repr=False)
    max: float = Field(..., repr=False)
    nbins: int = Field(repr=False)

    @model_validator(mode="after")
    def check_min_le_max(self) -> RegularAxis:
        """Validate that max >= min."""
        if self.max < self.min:
            msg = f"Axis '{self.name}': max ({self.max}) must be >= min ({self.min})"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_binning(self) -> RegularAxis:
        """Validate that max >= min."""
        if self.nbins <= 0:
            msg = f"RegularAxis '{self.name}' must have positive number of bins, got {self.nbins}"
            raise ValueError(msg)
        return self

    @property
    def edges(self) -> list[float]:
        """Get the bin edges for this axis.

        Returns:
            List of bin edges. Generates edges using linspace.
        """
        return list(np.linspace(self.min, self.max, self.nbins + 1))

    def to_hist(self) -> hist.axis.Regular:
        """
        Convert this axis to a hist.axis object.

        Returns:
            A hist.axis.Regular object
        """
        return hist.axis.Regular(self.nbins, self.min, self.max, name=self.name)


class IrregularAxis(Axis):
    """
    Attributes:
        name: Name of the axis/variable
        edges: Bin edges array (length n+1)
    """

    edges: list[float] = Field(repr=False)

    @model_validator(mode="after")
    def validate_binning(self) -> IrregularAxis:
        """Ensure proper binning specification for binned data."""
        if len(self.edges) < 2:
            msg = f"IrregularAxis '{self.name}' must have at least 2 edges"
            raise ValueError(msg)
        # Check that edges are in ascending order
        for prev, curr in pairwise(self.edges):
            if curr <= prev:
                msg = f"IrregularAxis '{self.name}' edges must be in ascending order"
                raise ValueError(msg)
        return self

    @property
    def min(self) -> float:
        """Return lower edge."""
        return self.edges[0]

    @property
    def max(self) -> float:
        """Return upper edge."""
        return self.edges[-1]

    @property
    def nbins(self) -> int:
        """Get the nbins for this axis.

        Returns:
            Number of bins.
        """
        return len(self.edges) - 1

    def to_hist(self) -> hist.axis.Variable:
        """
        Convert this axis to a hist.axis object.

        Returns:
            A hist.axis.Variable object
        """
        return hist.axis.Variable(self.edges, name=self.name)


def _binned_axis_discriminator(v: Any) -> str | None:
    if isinstance(v, dict):
        if "edges" in v and "nbins" not in v:
            return "irregular"
        if "nbins" in v and "edges" not in v:
            return "regular"
        return None

    # Already-constructed model case
    if isinstance(v, IrregularAxis):
        return "irregular"
    if isinstance(v, RegularAxis):
        return "regular"

    return None


BinnedAxis = Annotated[
    (
        Annotated[RegularAxis, Tag("regular")]
        | Annotated[IrregularAxis, Tag("irregular")]
    ),
    Discriminator(_binned_axis_discriminator),
    custom_error_msg(
        {
            "union_tag_not_found": "Unknown axis {input}'. You must specify either regular binning (nbins/min/max) or irregular binning (edges).",
            "missing": "{input_value['name']} is missing {loc}",
        }
    ),
]

TAxis = TypeVar("TAxis", bound=Axis)


class BinnedAxes(NamedCollection[BinnedAxis]):
    """
    Collection of binned axis.
    """

    def get_total_bins(self) -> int:
        """Calculate total number of bins across all axes."""
        total = 1
        for axis in self:
            total *= axis.nbins
        return total


DomainAxis: TypeAlias = UnbinnedAxis | ConstantAxis


class UnbinnedAxes(NamedCollection[UnbinnedAxis]):
    """
    Collection of UnbinnedAxis.
    """

    root: list[UnbinnedAxis] = Field(default_factory=list)


class Axes(NamedCollection[BinnedAxis | UnbinnedAxis]):
    """
    Collection of BinnedAxis | UnbinnedAxis.
    """

    root: list[BinnedAxis | UnbinnedAxis] = Field(default_factory=list)


class DomainAxes(NamedCollection[DomainAxis]):
    """
    Collection of BinnedAxis | UnbinnedAxis.
    """

    root: list[DomainAxis] = Field(default_factory=list)
