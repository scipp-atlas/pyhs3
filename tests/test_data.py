"""
Unit tests for data implementations.

Tests for Datum subclasses (PointData, UnbinnedData, BinnedData),
Data collection class, and related components like Axis and GaussianUncertainty.
"""

from __future__ import annotations

import pytest

from pyhs3.data import (
    Axis,
    BinnedData,
    Data,
    GaussianUncertainty,
    PointData,
    UnbinnedData,
)


class TestAxis:
    """Tests for the Axis class."""

    def test_axis_creation_basic(self):
        """Test basic Axis creation with name only."""
        axis = Axis(name="test_var")
        assert axis.name == "test_var"
        assert axis.min is None
        assert axis.max is None
        assert axis.nbins is None
        assert axis.edges is None

    def test_axis_creation_regular_binning(self):
        """Test Axis creation with regular binning."""
        axis = Axis(name="mass", min=100.0, max=200.0, nbins=50)
        assert axis.name == "mass"
        assert axis.min == 100.0
        assert axis.max == 200.0
        assert axis.nbins == 50
        assert axis.edges is None

    def test_axis_creation_irregular_binning(self):
        """Test Axis creation with irregular binning."""
        edges = [0.0, 1.0, 3.0, 10.0, 100.0]
        axis = Axis(name="pt", edges=edges)
        assert axis.name == "pt"
        assert axis.min is None
        assert axis.max is None
        assert axis.nbins is None
        assert axis.edges == edges

    def test_axis_validation_mixed_binning_fails(self):
        """Test that specifying both regular and irregular binning fails."""
        with pytest.raises(ValueError, match="Cannot specify both regular binning"):
            Axis(name="test", min=0.0, max=10.0, nbins=5, edges=[0, 5, 10])

    def test_axis_validation_edges_too_short(self):
        """Test that edges array must have at least 2 elements."""
        with pytest.raises(
            ValueError, match="Edges array must have at least 2 elements"
        ):
            Axis(name="test", edges=[1.0])

    def test_axis_validation_edges_not_ordered(self):
        """Test that edges must be in non-decreasing order."""
        with pytest.raises(ValueError, match="Edges must be in non-decreasing order"):
            Axis(name="test", edges=[1.0, 3.0, 2.0, 4.0])

    def test_axis_validation_edges_equal_allowed(self):
        """Test that equal adjacent edges are allowed."""
        axis = Axis(name="test", edges=[1.0, 2.0, 2.0, 3.0])
        assert axis.edges == [1.0, 2.0, 2.0, 3.0]

    def test_axis_bin_edges_regular_binning(self):
        """Test bin_edges property with regular binning."""
        axis = Axis(name="mass", min=0.0, max=10.0, nbins=5)
        edges = axis.bin_edges

        # Should return 6 edges for 5 bins: [0, 2, 4, 6, 8, 10]
        expected_edges = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        assert len(edges) == 6
        assert edges == pytest.approx(expected_edges)

    def test_axis_bin_edges_irregular_binning(self):
        """Test bin_edges property with irregular binning."""
        custom_edges = [0.0, 1.0, 5.0, 12.0, 25.0]
        axis = Axis(name="pt", edges=custom_edges)
        edges = axis.bin_edges

        # Should return the provided edges exactly
        assert edges == custom_edges

    def test_axis_bin_edges_no_binning_info(self):
        """Test bin_edges property when no binning information is provided."""
        axis = Axis(name="var")
        edges = axis.bin_edges

        # Should return empty list when no binning info
        assert edges == []

    def test_axis_bin_edges_partial_regular_info(self):
        """Test bin_edges property with incomplete regular binning info."""
        # Missing nbins
        axis = Axis(name="var", min=0.0, max=5.0)
        assert axis.bin_edges == []

        # Missing max
        axis = Axis(name="var", min=0.0, nbins=3)
        assert axis.bin_edges == []

        # Missing min
        axis = Axis(name="var", max=5.0, nbins=3)
        assert axis.bin_edges == []

    def test_axis_bin_edges_single_bin(self):
        """Test bin_edges property with single bin."""
        axis = Axis(name="single", min=1.0, max=2.0, nbins=1)
        edges = axis.bin_edges

        # Should return 2 edges for 1 bin: [1.0, 2.0]
        assert len(edges) == 2
        assert edges == pytest.approx([1.0, 2.0])

    def test_axis_bin_edges_zero_range(self):
        """Test bin_edges property with zero range (min=max)."""
        axis = Axis(name="zero_range", min=5.0, max=5.0, nbins=1)
        edges = axis.bin_edges

        # Should return [5.0, 5.0] for zero range
        assert len(edges) == 2
        assert edges == pytest.approx([5.0, 5.0])

    def test_axis_bin_edges_negative_range(self):
        """Test bin_edges property with negative values."""
        axis = Axis(name="negative", min=-10.0, max=-2.0, nbins=4)
        edges = axis.bin_edges

        # Should handle negative values correctly: [-10, -8, -6, -4, -2]
        expected_edges = [-10.0, -8.0, -6.0, -4.0, -2.0]
        assert len(edges) == 5
        assert edges == pytest.approx(expected_edges)

    def test_axis_bin_edges_large_number_of_bins(self):
        """Test bin_edges property with large number of bins."""
        axis = Axis(name="many_bins", min=0.0, max=100.0, nbins=1000)
        edges = axis.bin_edges

        # Should return 1001 edges for 1000 bins
        assert len(edges) == 1001
        assert edges[0] == pytest.approx(0.0)
        assert edges[-1] == pytest.approx(100.0)
        assert edges[500] == pytest.approx(50.0)  # Middle should be 50.0


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


class TestUnbinnedData:
    """Tests for the UnbinnedData class."""

    def test_unbinned_data_creation_basic(self):
        """Test basic UnbinnedData creation."""
        entries = [[1.0, 2.0], [3.0, 4.0]]
        axes = [Axis(name="x", min=0.0, max=5.0), Axis(name="y", min=0.0, max=10.0)]
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
        axes = [Axis(name="x")]
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
        axes = [Axis(name="x"), Axis(name="y")]
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
        axes = [Axis(name="x")]
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
        axes = [Axis(name="x")]
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
        axes = [Axis(name="x"), Axis(name="y")]  # 2 axes

        with pytest.raises(
            ValueError, match=r"Entry dimensionality .* must match number of axes"
        ):
            UnbinnedData(name="test", type="unbinned", entries=entries, axes=axes)

    def test_unbinned_data_validation_inconsistent_entry_dimensions(self):
        """Test that all entries must have same dimensionality."""
        entries = [[1.0, 2.0], [3.0]]  # Inconsistent dimensions
        axes = [Axis(name="x"), Axis(name="y")]

        with pytest.raises(ValueError, match=r"Entry.*has.*dimensions, expected"):
            UnbinnedData(name="test", type="unbinned", entries=entries, axes=axes)

    def test_unbinned_data_validation_inconsistent_uncertainty_dimensions(self):
        """Test that uncertainty entries must have same dimensionality as data entries."""
        entries = [[1.0, 2.0], [3.0, 4.0]]  # 2D entries
        axes = [Axis(name="x"), Axis(name="y")]
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
        axes = [Axis(name="x"), Axis(name="y")]
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
        axes = [Axis(name="x"), Axis(name="y")]
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
        axes = [Axis(name="x"), Axis(name="y")]

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
        axes = [Axis(name="mass", min=100.0, max=200.0, nbins=4)]
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
            Axis(name="x", min=0.0, max=2.0, nbins=2),
            Axis(name="y", min=0.0, max=3.0, nbins=3),
        ]
        data = BinnedData(name="hist2d", type="binned", contents=contents, axes=axes)

        assert len(data.contents) == 6
        assert len(data.axes) == 2

    def test_binned_data_creation_irregular_binning(self):
        """Test BinnedData creation with irregular binning."""
        contents = [10.0, 25.0, 5.0]  # 3 bins
        axes = [Axis(name="pt", edges=[0.0, 10.0, 50.0, 100.0])]  # 3 bins
        data = BinnedData(
            name="irregular_hist", type="binned", contents=contents, axes=axes
        )

        assert data.contents == contents
        assert data.axes[0].edges == [0.0, 10.0, 50.0, 100.0]

    def test_binned_data_creation_with_uncertainty(self):
        """Test BinnedData creation with uncertainty."""
        contents = [10.0, 20.0]
        axes = [Axis(name="x", min=0.0, max=2.0, nbins=2)]
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
        axes = [Axis(name="x", min=0.0, max=3.0, nbins=3)]  # 3 bins expected

        with pytest.raises(
            ValueError,
            match=r"Contents array length .* must match expected number of bins",
        ):
            BinnedData(name="test", type="binned", contents=contents, axes=axes)

    def test_binned_data_validation_axis_missing_binning(self):
        """Test that axis must specify binning for binned data."""
        contents = [10.0, 20.0]
        axes = [Axis(name="x", min=0.0, max=2.0)]  # No nbins or edges

        with pytest.raises(
            ValueError,
            match=r"must specify either regular binning .* or irregular binning",
        ):
            BinnedData(name="test", type="binned", contents=contents, axes=axes)

    def test_binned_data_validation_uncertainty_sigma_length_mismatch(self):
        """Test that uncertainty sigma length must match contents length."""
        contents = [10.0, 20.0, 15.0]
        axes = [Axis(name="x", min=0.0, max=3.0, nbins=3)]
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
            name="events1", type="unbinned", entries=[[1.0]], axes=[Axis(name="x")]
        )
        binned = BinnedData(
            name="hist1",
            type="binned",
            contents=[10.0, 20.0],
            axes=[Axis(name="mass", min=100.0, max=200.0, nbins=2)],
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

    def test_data_data_map_property(self):
        """Test data_map property."""
        datum1 = PointData(name="point1", type="point", value=1.0)
        datum2 = PointData(name="point2", type="point", value=2.0)
        data = Data([datum1, datum2])

        data_map = data.data_map
        assert data_map["point1"] == datum1
        assert data_map["point2"] == datum2
        assert len(data_map) == 2

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
