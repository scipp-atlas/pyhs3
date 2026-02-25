"""
HS3 Axis implementations.

Provides Pydantic classes for handling HS3 axis specifications
including unbinned axes (with min/max bounds) and binned axes
(with regular or irregular binning).
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import pairwise
from typing import Annotated, Any

import hist
import numpy as np
from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
    RootModel,
    Tag,
    model_validator,
)

from pyhs3.collections import NamedModel
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

    Alias for Axis with required min/max bounds. Provided for backward
    compatibility and semantic clarity.

    Attributes:
        name: Name of the axis/variable
        min: Minimum value (required)
        max: Maximum value (required)
    """

    min: float = Field(..., repr=False)
    max: float = Field(..., repr=False)


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


def binned_axis_discriminator(v: Any) -> str | None:
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


BinnedAxisUnion = Annotated[
    (
        Annotated[RegularAxis, Tag("regular")]
        | Annotated[IrregularAxis, Tag("irregular")]
    ),
    Discriminator(binned_axis_discriminator),
    custom_error_msg(
        {
            "union_tag_not_found": "Unknown axis {input}'. You must specify either regular binning (nbins/min/max) or irregular binning (edges).",
            "missing": "{input_value['name']} is missing {loc}",
        }
    ),
]


class BinnedAxis(RootModel[BinnedAxisUnion]):
    """
    Binned axis specification.

    Supports both regular binning (min/max/nbins) and irregular binning (edges)
    through a discriminated union. The discriminator automatically selects the
    correct type based on the presence of 'nbins' or 'edges' fields.
    """

    root: BinnedAxisUnion

    @property
    def name(self) -> str:
        """Get the axis name."""
        return self.root.name

    @property
    def nbins(self) -> int:
        """Get the number of bins."""
        return self.root.nbins

    @property
    def min(self) -> float:
        """Get the min."""
        return self.root.min

    @property
    def max(self) -> float:
        """Get the max."""
        return self.root.max

    @property
    def edges(self) -> list[float]:
        """Get the edges."""
        return self.root.edges

    def to_hist(self) -> hist.axis.Variable | hist.axis.Regular:
        """
        Convert this axis to a hist.axis object.

        Returns:
            A hist.axis.Variable object
        """
        return self.root.to_hist()


class BinnedAxes(RootModel[list[BinnedAxis]]):
    """
    Collection of binned axes.

    Manages a list of BinnedAxis instances, providing list-like access and
    validation. Each axis can use either regular or irregular binning.
    """

    root: list[BinnedAxis] = Field(default_factory=list)

    def __getitem__(self, index: int) -> BinnedAxis:
        """Get axis by index."""
        return self.root[index]

    def __len__(self) -> int:
        """Get number of axes."""
        return len(self.root)

    def __iter__(self) -> Iterator[BinnedAxis]:  # type: ignore[override]  # https://github.com/pydantic/pydantic/issues/8872
        """Iterate over axes."""
        return iter(self.root)

    def get_total_bins(self) -> int:
        """Calculate total number of bins across all axes."""
        total_bins = 1
        for axis in self.root:
            total_bins *= axis.nbins
        return total_bins


class UnbinnedAxes(RootModel[list[UnbinnedAxis]]):
    """
    Collection of unbinned axes.

    Manages a list of UnbinnedAxis instances, providing list-like access and
    validation. Each axis represents an unbinned observable with min/max bounds.
    """

    root: list[UnbinnedAxis] = Field(default_factory=list)

    def __getitem__(self, index: int) -> UnbinnedAxis:
        """Get axis by index."""
        return self.root[index]

    def __len__(self) -> int:
        """Get number of axes."""
        return len(self.root)

    def __iter__(self) -> Iterator[UnbinnedAxis]:  # type: ignore[override]  # https://github.com/pydantic/pydantic/issues/8872
        """Iterate over axes."""
        return iter(self.root)


class Axes(RootModel[list[BinnedAxis | UnbinnedAxis]]):
    """
    Collection of axes.

    Manages a list of BinnedAxis and UnbinnedAxis instances, providing list-like
    access and validation. Each axis can be binned or unbinned.
    """

    root: list[BinnedAxis | UnbinnedAxis] = Field(default_factory=list)

    def __getitem__(self, index: int) -> BinnedAxis | UnbinnedAxis:
        """Get axis by index."""
        return self.root[index]

    def __len__(self) -> int:
        """Get number of axes."""
        return len(self.root)

    def __iter__(self) -> Iterator[BinnedAxis | UnbinnedAxis]:  # type: ignore[override]  # https://github.com/pydantic/pydantic/issues/8872
        """Iterate over axes."""
        return iter(self.root)
