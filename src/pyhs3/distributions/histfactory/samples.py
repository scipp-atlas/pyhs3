"""
HistFactory Distribution implementation.

Provides the HistFactoryDist class for handling binned statistical models
with samples and modifiers as defined in the HS3 specification.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, RootModel, model_validator

# Import existing distributions for constraint terms
from pyhs3.distributions.histfactory.modifiers import Modifiers
from pyhs3.lazy import get_hist

if TYPE_CHECKING:
    import hist

    from pyhs3.distributions.histfactory.axes import Axes


class SampleData(BaseModel):
    """Sample data containing bin contents and errors."""

    contents: list[float]
    errors: list[float]

    @model_validator(mode="after")
    def validate_lengths(self) -> SampleData:
        """Ensure contents and errors have same length."""
        if len(self.contents) != len(self.errors):
            msg = f"Sample data contents ({len(self.contents)}) and errors ({len(self.errors)}) must have same length"
            raise ValueError(msg)
        return self


class Sample(BaseModel):
    """HistFactory sample specification."""

    name: str
    data: SampleData
    modifiers: Modifiers = Field(default_factory=Modifiers)

    def to_hist(self, axes: Axes) -> hist.Hist:
        """
        Convert to scikit-hep hist.Hist object for visualization.

        Creates a hist.Hist histogram from this sample's contents and errors.
        The axes must be provided since SampleData doesn't contain axis information.

        Note:
            Requires the hist package. Install with: python -m pip install 'pyhs3[visualization]'
            or python -m pip install hist

        Args:
            axes: Axes specification defining the binning

        Returns:
            hist.Hist: Histogram representation with:
                - Axes matching the provided axes
                - Values from sample contents
                - Variances from sample errors (squared)

        Raises:
            ImportError: If hist package is not installed

        Examples:
            >>> from pyhs3.distributions.histfactory.axes import Axes
            >>> sample = Sample(
            ...     name="signal",
            ...     data={"contents": [10, 20, 15], "errors": [3, 4, 2.5]}
            ... )
            >>> axes = Axes([{"name": "x", "min": 0, "max": 3, "nbins": 3}])
            >>> h = sample.to_hist(axes)
            >>> h.plot()  # Plot with matplotlib
        """
        hist = get_hist()

        # Convert axes to hist.axis objects
        # Access the root to get the actual axis (BinnedAxisRange or BinnedAxisEdges)
        hist_axes = [axis.root.to_hist() for axis in axes]

        # Create histogram with Weight storage since we always have errors
        h = hist.Hist(*hist_axes, storage=hist.storage.Weight())

        # Calculate shape from axes
        shape = tuple(axis.get_nbins() for axis in axes)

        # Reshape contents and variances (errors squared)
        contents_nd = np.array(self.data.contents).reshape(shape)
        variances_nd = np.square(self.data.errors).reshape(shape)

        # Set values with variances using view
        h.view(flow=False)["value"] = contents_nd
        h.view(flow=False)["variance"] = variances_nd

        return h  # type: ignore[no-any-return]


class Samples(RootModel[list[Sample]]):
    """
    Collection of samples for a HistFactory distribution.

    Manages a set of sample instances, providing dict-like access by sample name
    and list-like iteration. Handles sample validation and maintains name uniqueness.
    """

    root: list[Sample] = Field(default_factory=list)
    _map: dict[str, Sample] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        """Initialize computed collections after Pydantic validation."""
        self._map = {sample.name: sample for sample in self.root}

    def __getitem__(self, item: str | int) -> Sample:
        if isinstance(item, int):
            return self.root[item]
        return self._map[item]

    def __contains__(self, item: str) -> bool:
        return item in self._map

    def __iter__(self) -> Iterator[Sample]:  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)


__all__ = ("Sample", "Samples")
