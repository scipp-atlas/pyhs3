"""
HistFactory axes implementation.

Provides axis specifications for HistFactory distributions including both
regular binning (min/max/nbins) and irregular binning (edges) through
discriminated unions.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import pairwise
from typing import Annotated, Any

import hist
from pydantic import Discriminator, Field, RootModel, Tag, model_validator

from pyhs3.domains import Axis

TYPE_CHECKING = False


class BinnedAxisRange(Axis):
    """
    Binned axis specification using regular binning (min, max, nbins).

    Extends the base Axis class to support regular binning for HistFactory
    distributions where bins are uniformly spaced between min and max.
    """

    nbins: int = Field(description="Number of bins")

    @model_validator(mode="after")
    def validate_range_binning(self) -> BinnedAxisRange:
        """Validate regular binning parameters."""
        if self.min is None or self.max is None:
            msg = f"BinnedAxisRange '{self.name}' must specify both 'min' and 'max'"
            raise ValueError(msg)
        if self.nbins <= 0:
            msg = f"BinnedAxisRange '{self.name}' must have positive number of bins, got {self.nbins}"
            raise ValueError(msg)
        return self

    def get_nbins(self) -> int:
        """Get the number of bins."""
        return self.nbins

    def to_hist(self) -> Any:
        """
        Convert this axis to a hist.axis.Regular object.

        Returns:
            A hist.axis.Regular object for regular binning
        """
        if TYPE_CHECKING:
            assert self.min is not None
            assert self.max is not None
        return hist.axis.Regular(self.nbins, self.min, self.max, name=self.name)


class BinnedAxisEdges(Axis):
    """
    Binned axis specification using explicit bin edges.

    Extends the base Axis class to support irregular binning for HistFactory
    distributions where bin boundaries are explicitly specified.
    """

    edges: list[float] = Field(description="Explicit bin edges")

    @model_validator(mode="after")
    def validate_edges_binning(self) -> BinnedAxisEdges:
        """Validate explicit bin edges."""
        if len(self.edges) < 2:
            msg = f"BinnedAxisEdges '{self.name}' must have at least 2 edges"
            raise ValueError(msg)
        # Check that edges are in ascending order
        for a, b in pairwise(self.edges):
            if b <= a:
                msg = f"BinnedAxisEdges '{self.name}' edges must be in ascending order"
                raise ValueError(msg)
        return self

    def get_nbins(self) -> int:
        """Get the number of bins."""
        return len(self.edges) - 1

    def to_hist(self) -> Any:
        """
        Convert this axis to a hist.axis.Variable object.

        Returns:
            A hist.axis.Variable object for irregular binning
        """
        return hist.axis.Variable(self.edges, name=self.name)


def get_binned_axis_discriminator(v: Any) -> str:
    """Discriminator function for BinnedAxis union."""
    if isinstance(v, dict):
        if "nbins" in v:
            return "range"
        if "edges" in v:
            return "edges"
        # Default to range if neither is specified explicitly
        # This will let the validation happen in the specific class
        return "range"
    # For object instances
    if hasattr(v, "nbins"):
        return "range"
    if hasattr(v, "edges"):
        return "edges"
    return "range"


# Define the discriminated union type
BinnedAxisUnion = Annotated[
    Annotated[BinnedAxisRange, Tag("range")] | Annotated[BinnedAxisEdges, Tag("edges")],
    Discriminator(get_binned_axis_discriminator),
]


class BinnedAxis(RootModel[BinnedAxisUnion]):
    """
    Binned axis specification for HistFactory distributions.

    Supports both regular binning (min/max/nbins) and irregular binning (edges)
    through a discriminated union. The discriminator automatically selects the
    correct type based on the presence of 'nbins' or 'edges' fields.
    """

    root: BinnedAxisUnion

    def get_nbins(self) -> int:
        """Get the number of bins."""
        return self.root.get_nbins()

    @property
    def name(self) -> str:
        """Get the axis name."""
        return self.root.name


class Axes(RootModel[list[BinnedAxis]]):
    """
    Collection of binned axes for HistFactory distributions.

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
            total_bins *= axis.get_nbins()
        return total_bins


__all__ = [
    "Axes",
    "BinnedAxis",
    "BinnedAxisEdges",
    "BinnedAxisRange",
]
