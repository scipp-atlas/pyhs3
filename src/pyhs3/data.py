"""
HS3 Data implementations.

Provides Pydantic classes for handling HS3 data specifications
including point data, unbinned data, and binned data with uncertainties.
"""

from __future__ import annotations

from typing import Annotated, Literal

import hist
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pyhs3.axes import Axis, BinnedAxis, UnbinnedAxis
from pyhs3.collections import NamedCollection, NamedModel
from pyhs3.exceptions import custom_error_msg

TYPE_CHECKING = False


class GaussianUncertainty(BaseModel):
    """
    Gaussian uncertainty specification for data.

    Attributes:
        type: Must be "gaussian_uncertainty"
        sigma: Standard deviations for each data point
        correlation: Correlation matrix or 0 for no correlation
    """

    model_config = ConfigDict()

    type: Literal["gaussian_uncertainty"] = Field(
        default="gaussian_uncertainty", repr=False
    )
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


class Datum(NamedModel):
    """
    Base class for HS3 data specifications.

    Provides the foundation for all data implementations,
    handling common properties like name and type identification.

    Attributes:
        name: Custom string identifier for the data
        type: Type identifier for the data format
    """

    model_config = ConfigDict()

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
        axes: Optional axes for observable bounds (for normalization)
    """

    type: Literal["point"] = Field(default="point", repr=False)
    value: float = Field(..., repr=False)
    uncertainty: float | None = Field(default=None, repr=False)
    axes: list[Axis] | None = Field(default=None, repr=False)


class UnbinnedData(Datum):
    """
    Unbinned data specification for multiple data points.

    Represents individual data points in multi-dimensional space
    with optional weights and uncertainties.

    Attributes:
        name: Custom string identifier
        type: Must be "unbinned"
        entries: Array of coordinate arrays for each data point
        axes: Axis specifications defining coordinate system (UnbinnedAxis with required min/max)
        weights: Optional weights for each entry
        entries_uncertainties: Optional uncertainties for each coordinate
    """

    type: Literal["unbinned"] = Field(default="unbinned", repr=False)
    entries: list[list[float]] = Field(..., repr=False)
    axes: list[UnbinnedAxis] = Field(..., repr=False)
    weights: list[float] | None = Field(default=None, repr=False)
    entries_uncertainties: list[list[float]] | None = Field(default=None, repr=False)

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

    def to_hist(
        self, nbins: int = 50
    ) -> hist.Hist[hist.storage.Weight | hist.storage.Double]:
        """
        Convert to scikit-hep hist.Hist object by binning entries.

        Creates a hist.Hist histogram by binning the unbinned entries according
        to the axis specifications. The resulting histogram can be plotted using
        matplotlib or other visualization tools.

        Args:
            nbins: Number of bins to use for each axis (default: 50)

        Returns:
            hist.Hist: Histogram representation with:
                - Axes matching the data axes
                - Values from binned entries
                - Weights if provided

        Examples:
            >>> entries = [[0.5], [1.2], [1.8]]
            >>> axes = [UnbinnedAxis(name="x", min=0, max=3)]
            >>> data = UnbinnedData(
            ...     name="example",
            ...     type="unbinned",
            ...     entries=entries,
            ...     axes=axes
            ... )
            >>> data.to_hist(nbins=3)
            Hist(Regular(3, 0, 3, name='x'), storage=Double()) # Sum: 3.0
        """
        # Convert axes to hist.axis objects
        # UnbinnedAxis doesn't have to_hist(), so create Regular axes manually
        hist_axes = [
            hist.axis.Regular(nbins, axis.min, axis.max, name=axis.name)
            for axis in self.axes
        ]

        # Create histogram with appropriate storage
        storage = (
            hist.storage.Weight() if self.weights is not None else hist.storage.Double()
        )
        h = hist.Hist(*hist_axes, storage=storage)

        # Transpose entries from [[x1, y1], [x2, y2]] to [[x1, x2], [y1, y2]]
        if len(self.entries) > 0:
            entries_transposed = list(zip(*self.entries, strict=True))
            # Convert to numpy arrays for filling
            fill_args = [np.array(coord_list) for coord_list in entries_transposed]

            # Fill the histogram
            if self.weights is not None:
                h.fill(*fill_args, weight=np.array(self.weights))
            else:
                h.fill(*fill_args)

        return h


class BinnedData(Datum):
    """
    Binned data specification for histogram data.

    Represents binned/histogram data in multi-dimensional space
    with optional uncertainties and correlations.

    Attributes:
        name: Custom string identifier
        type: Must be "binned"
        contents: Bin contents array
        axes: Axis specifications defining binning (BinnedAxis with binning info)
        uncertainty: Optional uncertainty specification
    """

    type: Literal["binned"] = Field(default="binned", repr=False)
    contents: list[float] = Field(..., repr=False)
    axes: list[BinnedAxis] = Field(..., repr=False)
    uncertainty: GaussianUncertainty | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def validate_binned_data(self) -> BinnedData:
        """Validate binned data consistency."""
        # Calculate expected number of bins
        # BinnedAxis.validate_binning already ensures each axis has valid binning
        expected_bins = 1
        for axis in self.axes:
            expected_bins *= axis.nbins

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

    def to_hist(self) -> hist.Hist[hist.storage.Weight | hist.storage.Double]:
        """
        Convert to scikit-hep hist.Hist object for visualization.

        Creates a hist.Hist histogram from this binned data. The resulting
        histogram can be plotted using matplotlib or other visualization tools.

        Note:
            Correlation matrices in uncertainties are not preserved. Only the
            sigma values (standard deviations) are included as histogram variances.

        Returns:
            hist.Hist: Histogram representation with:
                - Axes matching the data axes
                - Values from contents
                - Variances from uncertainties if present

        Examples:
            >>> data = BinnedData(
            ...     name="example",
            ...     type="binned",
            ...     contents=[10, 20, 15],
            ...     axes=[BinnedAxis(name="x", min=0, max=3, nbins=3)]
            ... )
            >>> data.to_hist()
            Hist(Regular(3, 0, 3, name='x'), storage=Double()) # Sum: 45.0
        """
        # Convert axes to hist.axis objects
        hist_axes = [axis.to_hist() for axis in self.axes]

        # Create histogram with appropriate storage
        storage = (
            hist.storage.Weight()
            if self.uncertainty is not None
            else hist.storage.Double()
        )
        h = hist.Hist(*hist_axes, storage=storage)

        # Calculate shape from axes
        shape = tuple(axis.nbins for axis in self.axes)

        # Reshape contents for assignment
        if self.uncertainty is not None:
            # Reshape both contents and variances
            contents_nd = np.array(self.contents).reshape(shape)
            variances_nd = np.square(self.uncertainty.sigma).reshape(shape)

            stacked = np.stack([contents_nd, variances_nd], axis=-1)
            h[...] = stacked
        else:
            # Reshape and set contents using view
            contents_nd = np.array(self.contents).reshape(shape)
            h[...] = contents_nd

        return h


# Type alias for all data types using discriminated union
DataType = Annotated[PointData | UnbinnedData | BinnedData, Field(discriminator="type")]


class Data(NamedCollection[DataType]):
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
