"""
HistFactory Distribution implementation.

Provides the HistFactoryDist class for handling binned statistical models
with samples and modifiers as defined in the HS3 specification.
"""

from __future__ import annotations

import hist
import numpy as np
from pydantic import Field

from pyhs3.collections import NamedCollection, NamedModel
from pyhs3.data import BinnedAxes

# Import existing distributions for constraint terms
from pyhs3.distributions.histfactory.data import SampleData
from pyhs3.distributions.histfactory.modifiers import Modifiers


class Sample(NamedModel):
    """HistFactory sample specification."""

    data: SampleData
    modifiers: Modifiers = Field(default_factory=Modifiers)

    def to_hist(self, axes: BinnedAxes) -> hist.Hist[hist.storage.Weight]:
        """
        Convert to scikit-hep hist.Hist object for visualization.

        Creates a hist.Hist histogram from this sample's contents and errors.
        The axes must be provided since SampleData doesn't contain axis information.

        Args:
            axes: BinnedAxes specification defining the binning

        Returns:
            hist.Hist: Histogram representation with:
                - Axes matching the provided axes
                - Values from sample contents
                - Variances from sample errors (squared)

        Examples:
            >>> from pyhs3.distributions.histfactory.axes import BinnedAxes
            >>> sample = Sample(
            ...     name="signal",
            ...     data={"contents": [10, 20, 15], "errors": [3, 4, 2.5]}
            ... )
            >>> axes = BinnedAxes([{"name": "x", "min": 0, "max": 3, "nbins": 3}])
            >>> sample.to_hist(axes)
            Hist(Regular(3, 0, 3, name='x'), storage=Weight()) # Sum: WeightedSum(value=45, variance=31.25)
        """
        # Convert axes to hist.axis objects
        # Access the root to get the actual axis (BinnedAxisRange or BinnedAxisEdges)
        hist_axes = [axis.root.to_hist() for axis in axes]

        # Create histogram with Weight storage since we always have errors
        h = hist.Hist(*hist_axes, storage=hist.storage.Weight())

        # Calculate shape from axes
        shape = tuple(axis.nbins for axis in axes)

        # Reshape contents and variances (errors squared)
        contents_nd = np.array(self.data.contents).reshape(shape)
        variances_nd = np.square(self.data.errors).reshape(shape)

        stacked = np.stack([contents_nd, variances_nd], axis=-1)
        h[...] = stacked

        return h


class Samples(NamedCollection[Sample]):
    """
    Collection of samples for a HistFactory distribution.

    Manages a set of sample instances, providing dict-like access by sample name
    and list-like iteration. Handles sample validation and maintains name uniqueness.
    """

    root: list[Sample] = Field(default_factory=list)


__all__ = ("Sample", "Samples")
