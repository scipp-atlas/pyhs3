"""
Unit tests for data implementations.

Tests for Datum subclasses (PointData, UnbinnedData, BinnedData),
Data collection class, and related components like Axis and GaussianUncertainty.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import TypeAdapter

from pyhs3.axes import BinnedAxis, UnbinnedAxis
from pyhs3.data import (
    BinnedData,
    Data,
    GaussianUncertainty,
    PointData,
    UnbinnedData,
)

hist = pytest.importorskip("hist", reason="hist not installed")

binned_axis_adapter = TypeAdapter(BinnedAxis)


def make_binned_axis(**kwargs):
    return binned_axis_adapter.validate_python(kwargs)


class TestGaussianUncertainty:
    """Tests for the GaussianUncertainty class."""

    def test_gaussian_uncertainty_creation_basic(self):
        """Test basic GaussianUncertainty creation."""
        unc = GaussianUncertainty(type="gaussian_uncertainty", sigma=[1.0, 2.0, 1.5])
        assert unc.type == "gaussian_uncertainty"
        assert unc.sigma == [1.0, 2.0, 1.5]
        assert unc.correlation == 0

    def test_gaussian_uncertainty_with_correlation_matrix(self):
        """Test GaussianUncertainty with correlation matrix."""
        correlation = [[1.0, 0.5], [0.5, 1.0]]
        unc = GaussianUncertainty(
            type="gaussian_uncertainty", sigma=[1.0, 2.0], correlation=correlation
        )
        assert unc.sigma == [1.0, 2.0]
        assert unc.correlation == correlation

    def test_gaussian_uncertainty_correlation_size_mismatch(self):
        """Test that correlation matrix size must match sigma length."""
        with pytest.raises(ValueError, match="Correlation matrix must be 3x3"):
            GaussianUncertainty(
                type="gaussian_uncertainty",
                sigma=[1.0, 2.0, 1.5],
                correlation=[[1.0, 0.5], [0.5, 1.0]],
            )

    def test_gaussian_uncertainty_correlation_not_square(self):
        """Test that correlation matrix must be square."""
        with pytest.raises(ValueError, match="Correlation matrix must be 2x2"):
            GaussianUncertainty(
                type="gaussian_uncertainty",
                sigma=[1.0, 2.0],
                correlation=[[1.0, 0.5, 0.1], [0.5, 1.0, 0.2]],
            )


class TestPointData:
    """Tests for the PointData class."""

    def test_point_data_creation_basic(self):
        """Test basic PointData creation."""
        data = PointData(name="measurement1", type="point", value=42.5)
        assert data.name == "measurement1"
        assert data.type == "point"
        assert data.value == 42.5
        assert data.uncertainty is None

    def test_point_data_creation_with_uncertainty(self):
        """Test PointData creation with uncertainty."""
        data = PointData(name="measurement2", type="point", value=10.0, uncertainty=0.5)
        assert data.name == "measurement2"
        assert data.type == "point"
        assert data.value == 10.0
        assert data.uncertainty == 0.5

    def test_point_data_validation_requires_name(self):
        """Test that PointData validation requires name field."""
        with pytest.raises(ValueError, match="Field required"):
            PointData(type="point", value=1.0)

    def test_point_data_validation_requires_value(self):
        """Test that PointData validation requires value field."""
        with pytest.raises(ValueError, match="Field required"):
            PointData(name="test", type="point")

    def test_point_data_without_axes(self):
        """Test PointData without axes (default None)."""
        data = PointData(name="measurement", type="point", value=42.5)
        assert data.axes is None

    def test_point_data_with_axes(self):
        """Test PointData with axes for observable bounds."""
        axes = [UnbinnedAxis(name="x", min=0.0, max=10.0)]
        data = PointData(name="measurement", type="point", value=5.0, axes=axes)
        assert data.axes is not None
        assert len(data.axes) == 1
        assert data.axes[0].name == "x"
        assert data.axes[0].min == 0.0
        assert data.axes[0].max == 10.0

    def test_point_data_with_multiple_axes(self):
        """Test PointData with multiple axes."""
        axes = [
            UnbinnedAxis(name="x", min=0.0, max=10.0),
            UnbinnedAxis(name="y", min=-5.0, max=5.0),
        ]
        data = PointData(name="measurement", type="point", value=1.0, axes=axes)
        assert data.axes is not None
        assert len(data.axes) == 2


class TestUnbinnedData:
    """Tests for the UnbinnedData class."""

    def test_unbinned_data_creation_basic(self):
        """Test basic UnbinnedData creation."""
        entries = [[1.0, 2.0], [3.0, 4.0]]
        axes = [
            UnbinnedAxis(name="x", min=0.0, max=5.0),
            UnbinnedAxis(name="y", min=0.0, max=10.0),
        ]
        data = UnbinnedData(name="events", type="unbinned", entries=entries, axes=axes)

        assert data.name == "events"
        assert data.type == "unbinned"
        assert data.entries == entries
        assert len(data.axes) == 2
        assert data.weights is None
        assert data.entries_uncertainties is None

    def test_unbinned_data_creation_with_weights(self):
        """Test UnbinnedData creation with weights."""
        entries = [[1.0], [2.0], [3.0]]
        axes = [UnbinnedAxis(name="x", min=0.0, max=5.0)]
        weights = [0.8, 1.2, 0.9]
        data = UnbinnedData(
            name="weighted_events",
            type="unbinned",
            entries=entries,
            axes=axes,
            weights=weights,
        )

        assert data.entries == entries
        assert data.weights == weights

    def test_unbinned_data_creation_with_uncertainties(self):
        """Test UnbinnedData creation with entry uncertainties."""
        entries = [[1.0, 2.0], [3.0, 4.0]]
        axes = [
            UnbinnedAxis(name="x", min=0.0, max=5.0),
            UnbinnedAxis(name="y", min=0.0, max=5.0),
        ]
        uncertainties = [[0.1, 0.2], [0.3, 0.4]]
        data = UnbinnedData(
            name="events_with_errors",
            type="unbinned",
            entries=entries,
            axes=axes,
            entries_uncertainties=uncertainties,
        )

        assert data.entries_uncertainties == uncertainties

    def test_unbinned_data_validation_weights_length_mismatch(self):
        """Test that weights array length must match entries length."""
        entries = [[1.0], [2.0], [3.0]]
        axes = [UnbinnedAxis(name="x", min=0.0, max=5.0)]
        weights = [0.8, 1.2]  # Wrong length

        with pytest.raises(
            ValueError, match=r"Weights array length .* must match entries length"
        ):
            UnbinnedData(
                name="test",
                type="unbinned",
                entries=entries,
                axes=axes,
                weights=weights,
            )

    def test_unbinned_data_validation_uncertainties_length_mismatch(self):
        """Test that uncertainties array length must match entries length."""
        entries = [[1.0], [2.0]]
        axes = [UnbinnedAxis(name="x", min=0.0, max=5.0)]
        uncertainties = [[0.1], [0.2], [0.3]]  # Wrong length

        with pytest.raises(
            ValueError, match=r"Uncertainties array length .* must match entries length"
        ):
            UnbinnedData(
                name="test",
                type="unbinned",
                entries=entries,
                axes=axes,
                entries_uncertainties=uncertainties,
            )

    def test_unbinned_data_validation_entry_dimensionality_mismatch(self):
        """Test that entry dimensionality must match number of axes."""
        entries = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # 3D entries
        axes = [
            UnbinnedAxis(name="x", min=0.0, max=5.0),
            UnbinnedAxis(name="y", min=0.0, max=5.0),
        ]  # 2 axes

        with pytest.raises(
            ValueError, match=r"Entry dimensionality .* must match number of axes"
        ):
            UnbinnedData(name="test", type="unbinned", entries=entries, axes=axes)

    def test_unbinned_data_validation_inconsistent_entry_dimensions(self):
        """Test that all entries must have same dimensionality."""
        entries = [[1.0, 2.0], [3.0]]  # Inconsistent dimensions
        axes = [
            UnbinnedAxis(name="x", min=0.0, max=5.0),
            UnbinnedAxis(name="y", min=0.0, max=5.0),
        ]

        with pytest.raises(ValueError, match=r"Entry.*has.*dimensions, expected"):
            UnbinnedData(name="test", type="unbinned", entries=entries, axes=axes)

    def test_unbinned_data_validation_inconsistent_uncertainty_dimensions(self):
        """Test that uncertainty entries must have same dimensionality as data entries."""
        entries = [[1.0, 2.0], [3.0, 4.0]]  # 2D entries
        axes = [
            UnbinnedAxis(name="x", min=0.0, max=5.0),
            UnbinnedAxis(name="y", min=0.0, max=5.0),
        ]
        uncertainties = [[0.1, 0.2], [0.3]]  # Inconsistent uncertainty dimensions

        with pytest.raises(
            ValueError, match="Entry uncertainties\\[1\\] has 1 dimensions, expected 2"
        ):
            UnbinnedData(
                name="test",
                type="unbinned",
                entries=entries,
                axes=axes,
                entries_uncertainties=uncertainties,
            )

    def test_unbinned_data_empty_entries_with_weights(self):
        """Test that empty entries with weights validates correctly."""
        entries = []  # Empty entries (n_entries = 0)
        axes = [
            UnbinnedAxis(name="x", min=0.0, max=5.0),
            UnbinnedAxis(name="y", min=0.0, max=5.0),
        ]
        weights = []  # Empty weights to match

        # Should not raise any validation errors
        data = UnbinnedData(
            name="empty_data",
            type="unbinned",
            entries=entries,
            axes=axes,
            weights=weights,
        )
        assert data.entries == []
        assert data.weights == []

    def test_unbinned_data_empty_entries_with_uncertainties(self):
        """Test that empty entries with uncertainties validates correctly."""
        entries = []  # Empty entries (n_entries = 0)
        axes = [
            UnbinnedAxis(name="x", min=0.0, max=5.0),
            UnbinnedAxis(name="y", min=0.0, max=5.0),
        ]
        uncertainties = []  # Empty uncertainties to match

        # Should not raise any validation errors
        data = UnbinnedData(
            name="empty_data_with_unc",
            type="unbinned",
            entries=entries,
            axes=axes,
            entries_uncertainties=uncertainties,
        )
        assert data.entries == []
        assert data.entries_uncertainties == []

    def test_unbinned_data_empty_entries_basic(self):
        """Test that empty entries validates correctly without weights or uncertainties."""
        entries = []  # Empty entries (n_entries = 0)
        axes = [
            UnbinnedAxis(name="x", min=0.0, max=5.0),
            UnbinnedAxis(name="y", min=0.0, max=5.0),
        ]

        # Should not raise any validation errors
        data = UnbinnedData(
            name="empty_basic", type="unbinned", entries=entries, axes=axes
        )
        assert data.entries == []
        assert data.weights is None
        assert data.entries_uncertainties is None


class TestBinnedData:
    """Tests for the BinnedData class."""

    def test_binned_data_creation_basic(self):
        """Test basic BinnedData creation."""
        contents = [10.0, 20.0, 15.0, 5.0]
        axes = [make_binned_axis(name="mass", min=100.0, max=200.0, nbins=4)]
        data = BinnedData(name="histogram", type="binned", contents=contents, axes=axes)

        assert data.name == "histogram"
        assert data.type == "binned"
        assert data.contents == contents
        assert len(data.axes) == 1
        assert data.uncertainty is None

    def test_binned_data_creation_2d(self):
        """Test BinnedData creation with 2D histogram."""
        contents = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 2x3 = 6 bins
        axes = [
            make_binned_axis(name="x", min=0.0, max=2.0, nbins=2),
            make_binned_axis(name="y", min=0.0, max=3.0, nbins=3),
        ]
        data = BinnedData(name="hist2d", type="binned", contents=contents, axes=axes)

        assert len(data.contents) == 6
        assert len(data.axes) == 2

    def test_binned_data_creation_irregular_binning(self):
        """Test BinnedData creation with irregular binning."""
        contents = [10.0, 25.0, 5.0]  # 3 bins
        axes = [make_binned_axis(name="pt", edges=[0.0, 10.0, 50.0, 100.0])]  # 3 bins
        data = BinnedData(
            name="irregular_hist", type="binned", contents=contents, axes=axes
        )

        assert data.contents == contents
        assert data.axes[0].edges == [0.0, 10.0, 50.0, 100.0]

    def test_binned_data_creation_with_uncertainty(self):
        """Test BinnedData creation with uncertainty."""
        contents = [10.0, 20.0]
        axes = [make_binned_axis(name="x", min=0.0, max=2.0, nbins=2)]
        uncertainty = GaussianUncertainty(type="gaussian_uncertainty", sigma=[3.0, 4.0])
        data = BinnedData(
            name="hist_with_errors",
            type="binned",
            contents=contents,
            axes=axes,
            uncertainty=uncertainty,
        )

        assert data.uncertainty == uncertainty

    def test_binned_data_validation_contents_length_mismatch(self):
        """Test that contents length must match expected number of bins."""
        contents = [10.0, 20.0]  # 2 values
        axes = [
            make_binned_axis(name="x", min=0.0, max=3.0, nbins=3)
        ]  # 3 bins expected

        with pytest.raises(
            ValueError,
            match=r"Contents array length .* must match expected number of bins",
        ):
            BinnedData(name="test", type="binned", contents=contents, axes=axes)

    def test_binned_data_validation_axis_missing_binning(self):
        """Test that BinnedAxis must specify binning."""
        # BinnedAxis validation happens at axis creation, not at BinnedData creation
        with pytest.raises(
            ValueError,
            match=r"must specify either regular binning .* or irregular binning",
        ):
            make_binned_axis(name="x", min=0.0, max=2.0)  # No nbins or edges

    def test_binned_data_validation_uncertainty_sigma_length_mismatch(self):
        """Test that uncertainty sigma length must match contents length."""
        contents = [10.0, 20.0, 15.0]
        axes = [make_binned_axis(name="x", min=0.0, max=3.0, nbins=3)]
        uncertainty = GaussianUncertainty(
            type="gaussian_uncertainty", sigma=[3.0, 4.0]
        )  # Wrong length

        with pytest.raises(
            ValueError, match=r"Uncertainty sigma length .* must match contents length"
        ):
            BinnedData(
                name="test",
                type="binned",
                contents=contents,
                axes=axes,
                uncertainty=uncertainty,
            )


class TestBinnedDataHistConversion:
    """Tests for BinnedData.to_hist() method."""

    def test_to_hist_1d_regular_binning(self):
        """Test BinnedData.to_hist() with 1D regular binning."""
        contents = [10.0, 20.0, 15.0]
        axes = [make_binned_axis(name="x", min=0.0, max=3.0, nbins=3)]
        data = BinnedData(name="test", type="binned", contents=contents, axes=axes)

        h = data.to_hist()

        # Check that values match
        assert np.array_equal(h.values(), contents)
        # Check that axis is correctly configured
        assert len(h.axes) == 1
        assert h.axes[0].name == "x"
        assert len(h.axes[0].edges) == 4  # 3 bins = 4 edges
        assert h.axes[0].edges[0] == pytest.approx(0.0)
        assert h.axes[0].edges[-1] == pytest.approx(3.0)

    def test_to_hist_1d_irregular_binning(self):
        """Test BinnedData.to_hist() with 1D irregular binning."""
        contents = [10.0, 25.0, 5.0]
        edges = [0.0, 10.0, 50.0, 100.0]  # 3 bins with variable widths
        axes = [make_binned_axis(name="pt", edges=edges)]
        data = BinnedData(
            name="test_irregular", type="binned", contents=contents, axes=axes
        )

        h = data.to_hist()

        # Check that values match
        assert np.array_equal(h.values(), contents)
        # Check that axis is correctly configured with irregular binning
        assert len(h.axes) == 1
        assert h.axes[0].name == "pt"
        assert len(h.axes[0].edges) == 4  # 3 bins = 4 edges
        assert np.array_equal(h.axes[0].edges, edges)

    def test_to_hist_1d_with_uncertainties(self):
        """Test BinnedData.to_hist() with gaussian uncertainties."""
        contents = [10.0, 20.0, 15.0]
        sigma = [3.0, 4.0, 2.5]
        axes = [make_binned_axis(name="x", min=0.0, max=3.0, nbins=3)]
        uncertainty = GaussianUncertainty(type="gaussian_uncertainty", sigma=sigma)
        data = BinnedData(
            name="test_unc",
            type="binned",
            contents=contents,
            axes=axes,
            uncertainty=uncertainty,
        )

        h = data.to_hist()

        # Check that values match
        assert np.array_equal(h.values(), contents)
        # Check that variances match (sigma^2)
        expected_variances = np.square(sigma)
        assert np.allclose(h.variances(), expected_variances)

    def test_to_hist_2d_regular_binning(self):
        """Test BinnedData.to_hist() with 2D regular binning."""
        # 2x3 = 6 bins, flattened in C-order (row-major)
        contents = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        axes = [
            make_binned_axis(name="x", min=0.0, max=2.0, nbins=2),
            make_binned_axis(name="y", min=0.0, max=3.0, nbins=3),
        ]
        data = BinnedData(name="test_2d", type="binned", contents=contents, axes=axes)

        h = data.to_hist()

        # Check dimensions
        assert len(h.axes) == 2
        assert h.axes[0].name == "x"
        assert h.axes[1].name == "y"

        # Check values - hist should reshape to (2, 3) in C-order
        expected_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert np.array_equal(h.values(), expected_2d)

    def test_to_hist_empty_contents(self):
        """Test BinnedData.to_hist() with empty contents raises error."""
        # This will fail validation before to_hist() is called
        with pytest.raises(ValueError, match=r"Contents array length.*must match"):
            BinnedData(
                name="empty",
                type="binned",
                contents=[],
                axes=[make_binned_axis(name="x", min=0.0, max=1.0, nbins=1)],
            )

    def test_to_hist_3d_binning(self):
        """Test BinnedData.to_hist() with 3D histogram."""
        # 2x3x2 = 12 bins
        contents = [float(i) for i in range(1, 13)]
        axes = [
            make_binned_axis(name="x", min=0.0, max=2.0, nbins=2),
            make_binned_axis(name="y", min=0.0, max=3.0, nbins=3),
            make_binned_axis(name="z", min=0.0, max=2.0, nbins=2),
        ]
        data = BinnedData(name="test_3d", type="binned", contents=contents, axes=axes)

        h = data.to_hist()

        # Check dimensions
        assert len(h.axes) == 3
        assert h.axes[0].name == "x"
        assert h.axes[1].name == "y"
        assert h.axes[2].name == "z"

        # Check that total is preserved
        assert h.values().sum() == sum(contents)


class TestUnbinnedDataHistConversion:
    """Tests for UnbinnedData.to_hist() method."""

    def test_to_hist_1d_no_weights(self):
        """Test UnbinnedData.to_hist() with 1D data without weights."""
        # Create some unbinned data points
        entries = [[0.5], [1.2], [1.8], [2.3], [0.9]]
        axes = [UnbinnedAxis(name="x", min=0.0, max=3.0)]
        data = UnbinnedData(
            name="test_unbinned", type="unbinned", entries=entries, axes=axes
        )

        h = data.to_hist(nbins=3)

        # Check that axis is configured correctly
        assert len(h.axes) == 1
        assert h.axes[0].name == "x"

        # Check that entries were binned correctly
        # Bin 0: [0, 1) -> 0.5, 0.9 = 2 entries
        # Bin 1: [1, 2) -> 1.2, 1.8 = 2 entries
        # Bin 2: [2, 3) -> 2.3 = 1 entry
        expected_values = [2.0, 2.0, 1.0]
        assert np.array_equal(h.values(), expected_values)

    def test_to_hist_1d_with_weights(self):
        """Test UnbinnedData.to_hist() with 1D data with weights."""
        entries = [[0.5], [1.2], [1.8]]
        weights = [2.0, 3.0, 1.5]
        axes = [UnbinnedAxis(name="x", min=0.0, max=3.0)]
        data = UnbinnedData(
            name="test_weighted",
            type="unbinned",
            entries=entries,
            axes=axes,
            weights=weights,
        )

        h = data.to_hist(nbins=3)

        # Check that weighted entries were binned correctly
        # Bin 0: [0, 1) -> 0.5 with weight 2.0 = 2.0
        # Bin 1: [1, 2) -> 1.2 (w=3.0) + 1.8 (w=1.5) = 4.5
        # Bin 2: [2, 3) -> empty = 0.0
        expected_values = [2.0, 4.5, 0.0]
        assert np.allclose(h.values(), expected_values)

    def test_to_hist_2d(self):
        """Test UnbinnedData.to_hist() with 2D data."""
        entries = [[0.5, 0.3], [1.2, 1.8], [0.8, 2.5]]
        axes = [
            UnbinnedAxis(name="x", min=0.0, max=2.0),
            UnbinnedAxis(name="y", min=0.0, max=3.0),
        ]
        data = UnbinnedData(
            name="test_2d_unbinned", type="unbinned", entries=entries, axes=axes
        )

        h = data.to_hist(nbins=3)

        # Check dimensions
        assert len(h.axes) == 2
        assert h.axes[0].name == "x"
        assert h.axes[1].name == "y"

        # Verify that entries were binned (exact values depend on binning logic)
        assert h.values().sum() == 3.0  # Total entries

    def test_to_hist_empty_entries(self):
        """Test UnbinnedData.to_hist() with empty entries."""
        entries = []
        axes = [UnbinnedAxis(name="x", min=0.0, max=1.0)]
        data = UnbinnedData(
            name="empty_unbinned", type="unbinned", entries=entries, axes=axes
        )

        h = data.to_hist(nbins=10)

        # Should create histogram with all zeros
        assert len(h.axes) == 1
        assert h.values().sum() == 0.0

    def test_to_hist_3d_unbinned(self):
        """Test UnbinnedData.to_hist() with 3D unbinned data."""
        # Create some 3D points
        entries = [
            [0.5, 1.5, 0.5],
            [1.5, 0.5, 1.5],
            [0.5, 2.5, 0.5],
        ]
        axes = [
            UnbinnedAxis(name="x", min=0.0, max=2.0),
            UnbinnedAxis(name="y", min=0.0, max=3.0),
            UnbinnedAxis(name="z", min=0.0, max=2.0),
        ]
        data = UnbinnedData(
            name="test_3d_unbinned", type="unbinned", entries=entries, axes=axes
        )

        h = data.to_hist(nbins=2)

        # Check dimensions
        assert len(h.axes) == 3
        assert h.axes[0].name == "x"
        assert h.axes[1].name == "y"
        assert h.axes[2].name == "z"

        # Check that total entries is correct
        assert h.values().sum() == len(entries)


class TestData:
    """Tests for the Data collection class."""

    def test_data_creation_empty(self):
        """Test empty Data creation."""
        data = Data([])
        assert len(data) == 0
        assert list(data) == []

    def test_data_creation_with_mixed_types(self):
        """Test Data creation with mixed data types."""
        point = PointData(name="point1", type="point", value=42.0)
        unbinned = UnbinnedData(
            name="events1",
            type="unbinned",
            entries=[[1.0]],
            axes=[UnbinnedAxis(name="x", min=0.0, max=5.0)],
        )
        binned = BinnedData(
            name="hist1",
            type="binned",
            contents=[10.0, 20.0],
            axes=[make_binned_axis(name="mass", min=100.0, max=200.0, nbins=2)],
        )

        data = Data([point, unbinned, binned])

        assert len(data) == 3
        assert "point1" in data
        assert "events1" in data
        assert "hist1" in data
        assert data["point1"] == point
        assert data["events1"] == unbinned
        assert data["hist1"] == binned

    def test_data_get_by_name(self):
        """Test getting data by name."""
        datum = PointData(name="test_point", type="point", value=1.0)
        data = Data([datum])

        assert data.get("test_point") == datum
        assert data.get("nonexistent") is None

        default_datum = PointData(name="default", type="point", value=0.0)
        assert data.get("nonexistent", default_datum) == default_datum

    def test_data_get_by_index(self):
        """Test getting data by index."""
        datum1 = PointData(name="point1", type="point", value=1.0)
        datum2 = PointData(name="point2", type="point", value=2.0)
        data = Data([datum1, datum2])

        assert data[0] == datum1
        assert data[1] == datum2

    def test_data_iteration(self):
        """Test iteration over data."""
        datum1 = PointData(name="point1", type="point", value=1.0)
        datum2 = PointData(name="point2", type="point", value=2.0)
        data = Data([datum1, datum2])

        result = list(data)
        assert result == [datum1, datum2]

    def test_data_contains_operator(self):
        """Test 'in' operator for data."""
        datum = PointData(name="test_point", type="point", value=1.0)
        data = Data([datum])

        assert "test_point" in data
        assert "nonexistent_point" not in data

    def test_data_keyerror_on_missing_name(self):
        """Test KeyError when accessing non-existent data by name."""
        data = Data([])

        with pytest.raises(KeyError):
            _ = data["nonexistent"]

    def test_data_indexerror_on_missing_index(self):
        """Test IndexError when accessing non-existent data by index."""
        data = Data([])

        with pytest.raises(IndexError):
            _ = data[0]


class TestDataRepr:
    """Tests for Data.__repr__() method."""

    def test_data_repr(self):
        """Test Data.__repr__() returns expected format."""
        datum1 = PointData(name="obs1", value=1.5)
        datum2 = PointData(name="obs2", value=2.5)
        data = Data([datum1, datum2])

        repr_str = repr(data)
        assert repr_str == "Data(['obs1', 'obs2'])"

    def test_data_repr_empty(self):
        """Test Data.__repr__() with empty collection."""
        data = Data([])
        repr_str = repr(data)
        assert repr_str == "Data([])"
