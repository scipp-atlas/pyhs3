"""
HS3 Data implementations.

Provides Pydantic classes for handling HS3 data specifications
including point data, unbinned data, and binned data with uncertainties.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from pyhs3.exceptions import custom_error_msg


class Axis(BaseModel):
    """
    Axis specification for data coordinates.

    Defines coordinate system for unbinned or binned data.
    For binned data, can use either regular binning (min/max/nbins)
    or irregular binning (edges).

    Attributes:
        name: Name of the axis/variable
        min: Minimum value (for regular binning or unbinned data bounds)
        max: Maximum value (for regular binning or unbinned data bounds)
        nbins: Number of bins (for regular binning)
        edges: Bin edges array (for irregular binning, length n+1)
    """

    model_config = ConfigDict()

    name: str = Field(..., repr=True)
    min: float | None = Field(default=None, repr=False)
    max: float | None = Field(default=None, repr=False)
    nbins: int | None = Field(default=None, repr=False)
    edges: list[float] | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def validate_binning(self) -> Axis:
        """Ensure proper binning specification for binned data."""
        # For regular binning, need min, max, and nbins
        has_regular = all(x is not None for x in [self.min, self.max, self.nbins])
        # For irregular binning, need edges
        has_irregular = self.edges is not None

        # Either both regular binning or irregular binning, but not mixed
        if has_regular and has_irregular:
            msg = "Cannot specify both regular binning (min/max/nbins) and irregular binning (edges)"
            raise ValueError(msg)

        # For irregular binning, validate edges
        if has_irregular and self.edges is not None:
            if len(self.edges) < 2:
                msg = "Edges array must have at least 2 elements"
                raise ValueError(msg)
            if not all(
                self.edges[i] <= self.edges[i + 1]  # pylint: disable=unsubscriptable-object
                for i in range(len(self.edges) - 1)
            ):
                msg = "Edges must be in non-decreasing order"
                raise ValueError(msg)

        return self

    @property
    def bin_edges(self) -> list[float] | None:
        """Get the bin edges for this axis.

        Returns:
            List of bin edges. For regular binning, generates edges using linspace.
            For irregular binning, returns the provided edges. Empty list if
            insufficient information is provided.
        """
        if self.edges is not None:
            return self.edges

        if self.min is not None and self.max is not None and self.nbins is not None:
            return list(np.linspace(self.min, self.max, self.nbins + 1))
        return []


class GaussianUncertainty(BaseModel):
    """
    Gaussian uncertainty specification for data.

    Attributes:
        type: Must be "gaussian_uncertainty"
        sigma: Standard deviations for each data point
        correlation: Correlation matrix or 0 for no correlation
    """

    model_config = ConfigDict()

    type: Literal["gaussian_uncertainty"] = Field(..., repr=False)
    sigma: list[float] = Field(..., repr=False)
    correlation: list[list[float]] | Literal[0] = Field(default=0, repr=False)

    @model_validator(mode="after")
    def validate_correlation(self) -> GaussianUncertainty:
        """Validate correlation matrix dimensions."""
        if self.correlation != 0:
            n = len(self.sigma)
            if len(self.correlation) != n:
                msg = f"Correlation matrix must be {n}x{n} to match sigma length"
                raise ValueError(msg)
            for row in self.correlation:  # pylint: disable=not-an-iterable
                if len(row) != n:
                    msg = f"Correlation matrix must be {n}x{n} to match sigma length"
                    raise ValueError(msg)
        return self


class Datum(BaseModel):
    """
    Base class for HS3 data specifications.

    Provides the foundation for all data implementations,
    handling common properties like name and type identification.

    Attributes:
        name: Custom string identifier for the data
        type: Type identifier for the data format
    """

    model_config = ConfigDict()

    name: str = Field(..., repr=True)
    type: str = Field(..., repr=False)


class PointData(Datum):
    """
    Point data specification for single measurements.

    Represents a single measured value with optional uncertainty.

    Attributes:
        name: Custom string identifier
        type: Must be "point"
        value: Measured value
        uncertainty: Optional uncertainty/error
    """

    type: Literal["point"]
    value: float
    uncertainty: float | None = Field(default=None)


class UnbinnedData(Datum):
    """
    Unbinned data specification for multiple data points.

    Represents individual data points in multi-dimensional space
    with optional weights and uncertainties.

    Attributes:
        name: Custom string identifier
        type: Must be "unbinned"
        entries: Array of coordinate arrays for each data point
        axes: Axis specifications defining coordinate system
        weights: Optional weights for each entry
        entries_uncertainties: Optional uncertainties for each coordinate
    """

    type: Literal["unbinned"]
    entries: list[list[float]]
    axes: list[Axis]
    weights: list[float] | None = Field(default=None)
    entries_uncertainties: list[list[float]] | None = Field(default=None)

    @model_validator(mode="after")
    def validate_unbinned_data(self) -> UnbinnedData:
        """Validate consistency of unbinned data arrays."""
        n_entries = len(self.entries)

        # Check weights length
        if self.weights is not None and len(self.weights) != n_entries:
            msg = f"Weights array length ({len(self.weights)}) must match entries length ({n_entries})"
            raise ValueError(msg)

        # Check uncertainties shape
        if self.entries_uncertainties is not None:
            if len(self.entries_uncertainties) != n_entries:
                msg = f"Uncertainties array length ({len(self.entries_uncertainties)}) must match entries length ({n_entries})"
                raise ValueError(msg)

            # Check each entry has same dimensionality
            if n_entries > 0:
                expected_dims = len(self.entries[0])
                for i, entry_unc in enumerate(self.entries_uncertainties):
                    if len(entry_unc) != expected_dims:
                        msg = f"Entry uncertainties[{i}] has {len(entry_unc)} dimensions, expected {expected_dims}"
                        raise ValueError(msg)

        # Check entries dimensionality matches axes
        if n_entries > 0:
            entry_dims = len(self.entries[0])
            if entry_dims != len(self.axes):
                msg = f"Entry dimensionality ({entry_dims}) must match number of axes ({len(self.axes)})"
                raise ValueError(msg)

            # Check all entries have same dimensionality
            for i, entry in enumerate(self.entries):
                if len(entry) != entry_dims:
                    msg = (
                        f"Entry[{i}] has {len(entry)} dimensions, expected {entry_dims}"
                    )
                    raise ValueError(msg)

        return self


class BinnedData(Datum):
    """
    Binned data specification for histogram data.

    Represents binned/histogram data in multi-dimensional space
    with optional uncertainties and correlations.

    Attributes:
        name: Custom string identifier
        type: Must be "binned"
        contents: Bin contents array
        axes: Axis specifications defining binning
        uncertainty: Optional uncertainty specification
    """

    type: Literal["binned"]
    contents: list[float]
    axes: list[Axis]
    uncertainty: GaussianUncertainty | None = Field(default=None)

    @model_validator(mode="after")
    def validate_binned_data(self) -> BinnedData:
        """Validate binned data consistency."""
        # Calculate expected number of bins
        expected_bins = 1
        for axis in self.axes:
            if axis.nbins is not None:
                # Regular binning
                expected_bins *= axis.nbins
            elif axis.edges is not None:
                # Irregular binning
                expected_bins *= len(axis.edges) - 1
            else:
                msg = f"Axis '{axis.name}' must specify either regular binning (nbins/min/max) or irregular binning (edges)"
                raise ValueError(msg)

        # Check contents length
        if len(self.contents) != expected_bins:
            msg = f"Contents array length ({len(self.contents)}) must match expected number of bins ({expected_bins})"
            raise ValueError(msg)

        # Check uncertainty consistency
        if self.uncertainty is not None and len(self.uncertainty.sigma) != len(
            self.contents
        ):
            msg = f"Uncertainty sigma length ({len(self.uncertainty.sigma)}) must match contents length ({len(self.contents)})"
            raise ValueError(msg)

        return self


# Type alias for all data types using discriminated union
DataType = Annotated[PointData | UnbinnedData | BinnedData, Field(discriminator="type")]


class Data(
    RootModel[
        Annotated[
            list[DataType],
            custom_error_msg(
                {
                    "union_tag_not_found": "Data entry missing required 'type' field. Expected one of: 'point', 'unbinned', 'binned'",
                    "union_tag_invalid": "Unknown data type '{tag}' does not match any of the expected types: {expected_tags}",
                }
            ),
        ]
    ]
):
    """
    Collection of HS3 data specifications.

    Manages a set of data instances that define observed data
    for likelihood evaluations. Provides dict-like access to data by name.
    """

    root: Annotated[
        list[DataType],
        custom_error_msg(
            {
                "union_tag_not_found": "Data entry missing required 'type' field. Expected one of: 'point', 'unbinned', 'binned'",
                "union_tag_invalid": "Unknown data type '{tag}' does not match any of the expected types: {expected_tags}",
            }
        ),
    ] = Field(default_factory=list)

    @property
    def data_map(self) -> dict[str, Datum]:
        """Mapping from data names to Datum instances."""
        return {data.name: data for data in self.root}

    def __len__(self) -> int:
        """Number of data sets in this collection."""
        return len(self.root)

    def __contains__(self, data_name: str) -> bool:
        """Check if a data set with the given name exists."""
        return data_name in self.data_map

    def __getitem__(self, item: str | int) -> Datum:
        """Get a data set by name or index."""
        if isinstance(item, int):
            return self.root[item]
        return self.data_map[item]

    def get(self, data_name: str, default: Datum | None = None) -> Datum | None:
        """Get a data set by name, returning default if not found."""
        return self.data_map.get(data_name, default)

    def __iter__(self) -> Iterator[Datum]:  # type: ignore[override]
        """Iterate over the data sets."""
        return iter(self.root)
