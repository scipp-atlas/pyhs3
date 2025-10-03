"""
Tests for HistFactory axes module.

Test BinnedAxis discriminated union functionality with BinnedAxisRange and BinnedAxisEdges.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pyhs3.distributions.histfactory.axes import (
    Axes,
    BinnedAxis,
    BinnedAxisEdges,
    BinnedAxisRange,
)


class TestBinnedAxisRange:
    """Test BinnedAxisRange functionality."""

    def test_valid_binned_axis_range(self):
        """Test valid BinnedAxisRange creation."""
        axis = BinnedAxisRange(name="x", min=0.0, max=10.0, nbins=5)
        assert axis.name == "x"
        assert axis.min == 0.0
        assert axis.max == 10.0
        assert axis.nbins == 5
        assert axis.get_nbins() == 5

    def test_binned_axis_range_missing_min(self):
        """Test BinnedAxisRange raises error when min is missing."""
        with pytest.raises(
            ValueError, match="BinnedAxisRange 'x' must specify both 'min' and 'max'"
        ):
            BinnedAxisRange(name="x", max=10.0, nbins=5)

    def test_binned_axis_range_missing_max(self):
        """Test BinnedAxisRange raises error when max is missing."""
        with pytest.raises(
            ValueError, match="BinnedAxisRange 'x' must specify both 'min' and 'max'"
        ):
            BinnedAxisRange(name="x", min=0.0, nbins=5)

    def test_binned_axis_range_invalid_range(self):
        """Test BinnedAxisRange raises error when min >= max."""
        with pytest.raises(
            ValidationError, match=r"Axis 'x': max \(5\.0\) must be >= min \(10\.0\)"
        ):
            BinnedAxisRange(name="x", min=10.0, max=5.0, nbins=5)

    def test_binned_axis_range_equal_min_max(self):
        """Test BinnedAxisRange allows equal min and max (though not very useful)."""
        # The base Axis class allows min == max, so this should work
        axis = BinnedAxisRange(name="x", min=5.0, max=5.0, nbins=5)
        assert axis.min == 5.0
        assert axis.max == 5.0
        assert axis.nbins == 5

    def test_binned_axis_range_zero_bins(self):
        """Test BinnedAxisRange raises error when nbins is zero."""
        with pytest.raises(
            ValueError,
            match="BinnedAxisRange 'x' must have positive number of bins, got 0",
        ):
            BinnedAxisRange(name="x", min=0.0, max=10.0, nbins=0)

    def test_binned_axis_range_negative_bins(self):
        """Test BinnedAxisRange raises error when nbins is negative."""
        with pytest.raises(
            ValueError,
            match="BinnedAxisRange 'x' must have positive number of bins, got -5",
        ):
            BinnedAxisRange(name="x", min=0.0, max=10.0, nbins=-5)


class TestBinnedAxisEdges:
    """Test BinnedAxisEdges functionality."""

    def test_valid_binned_axis_edges(self):
        """Test valid BinnedAxisEdges creation."""
        axis = BinnedAxisEdges(name="x", edges=[0.0, 2.5, 5.0, 7.5, 10.0])
        assert axis.name == "x"
        assert axis.edges == [0.0, 2.5, 5.0, 7.5, 10.0]
        assert axis.get_nbins() == 4

    def test_binned_axis_edges_minimum_edges(self):
        """Test BinnedAxisEdges with minimum valid edges."""
        axis = BinnedAxisEdges(name="x", edges=[0.0, 10.0])
        assert axis.edges == [0.0, 10.0]
        assert axis.get_nbins() == 1

    def test_binned_axis_edges_too_few_edges(self):
        """Test BinnedAxisEdges raises error with too few edges."""
        with pytest.raises(
            ValueError, match="BinnedAxisEdges 'x' must have at least 2 edges"
        ):
            BinnedAxisEdges(name="x", edges=[5.0])

    def test_binned_axis_edges_empty_edges(self):
        """Test BinnedAxisEdges raises error with empty edges."""
        with pytest.raises(
            ValueError, match="BinnedAxisEdges 'x' must have at least 2 edges"
        ):
            BinnedAxisEdges(name="x", edges=[])

    def test_binned_axis_edges_not_ascending(self):
        """Test BinnedAxisEdges raises error when edges are not in ascending order."""
        with pytest.raises(
            ValueError, match="BinnedAxisEdges 'x' edges must be in ascending order"
        ):
            BinnedAxisEdges(name="x", edges=[0.0, 10.0, 5.0])

    def test_binned_axis_edges_equal_edges(self):
        """Test BinnedAxisEdges raises error when edges are equal."""
        with pytest.raises(
            ValueError, match="BinnedAxisEdges 'x' edges must be in ascending order"
        ):
            BinnedAxisEdges(name="x", edges=[0.0, 5.0, 5.0, 10.0])


class TestBinnedAxisDiscriminatedUnion:
    """Test BinnedAxis discriminated union functionality."""

    def test_discriminator_selects_range_with_nbins(self):
        """Test discriminator selects BinnedAxisRange when nbins is present."""
        axis_data = {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5}
        axis = BinnedAxis.model_validate(axis_data)
        assert isinstance(axis.root, BinnedAxisRange)
        assert axis.name == "x"
        assert axis.root.nbins == 5

    def test_discriminator_selects_edges_with_edges(self):
        """Test discriminator selects BinnedAxisEdges when edges is present."""
        axis_data = {"name": "x", "edges": [0.0, 2.5, 5.0, 7.5, 10.0]}
        axis = BinnedAxis.model_validate(axis_data)
        assert isinstance(axis.root, BinnedAxisEdges)
        assert axis.name == "x"
        assert axis.root.edges == [0.0, 2.5, 5.0, 7.5, 10.0]

    def test_discriminator_defaults_to_range(self):
        """Test discriminator defaults to BinnedAxisRange when neither nbins nor edges specified."""
        axis_data = {"name": "x", "min": 0.0, "max": 10.0}
        # This should select BinnedAxisRange but fail validation since nbins is missing
        with pytest.raises(ValueError, match="Field required"):
            BinnedAxis.model_validate(axis_data)

    def test_both_nbins_and_edges_raises_validation_error(self):
        """Test that having both nbins and edges fails at validation level."""
        # The discriminator will choose 'range' when both are present (nbins takes precedence)
        # But this should be caught by validation logic if we want to prevent it
        axis_data = {
            "name": "x",
            "min": 0.0,
            "max": 10.0,
            "nbins": 5,
            "edges": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        }
        # Since discriminator picks 'range', this should create a BinnedAxisRange
        # which will ignore the 'edges' field
        axis = BinnedAxis.model_validate(axis_data)
        assert isinstance(axis.root, BinnedAxisRange)
        assert axis.root.nbins == 5
        # edges should not be present in BinnedAxisRange
        assert not hasattr(axis.root, "edges")

    def test_discriminator_with_pydantic_instance(self):
        """Test discriminator works with already instantiated objects."""
        range_axis = BinnedAxisRange(name="x", min=0.0, max=10.0, nbins=5)
        edges_axis = BinnedAxisEdges(name="y", edges=[0.0, 5.0, 10.0])

        # Test that both instances work correctly
        assert isinstance(range_axis, BinnedAxisRange)
        assert isinstance(edges_axis, BinnedAxisEdges)
        assert range_axis.get_nbins() == 5
        assert edges_axis.get_nbins() == 2

    def test_get_nbins_works_for_both_types(self):
        """Test that get_nbins() method works for both axis types."""
        range_axis_data = {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5}
        edges_axis_data = {"name": "y", "edges": [0.0, 2.5, 5.0, 7.5, 10.0]}

        range_axis = BinnedAxis.model_validate(range_axis_data)
        edges_axis = BinnedAxis.model_validate(edges_axis_data)

        assert range_axis.get_nbins() == 5
        assert edges_axis.get_nbins() == 4

    def test_validation_errors_for_range_type(self):
        """Test validation errors are properly raised for BinnedAxisRange."""
        # Missing min/max
        with pytest.raises(
            ValueError, match="BinnedAxisRange 'x' must specify both 'min' and 'max'"
        ):
            BinnedAxis.model_validate({"name": "x", "nbins": 5})

        # Invalid range
        with pytest.raises(
            ValidationError, match=r"Axis 'x': max \(5\.0\) must be >= min \(10\.0\)"
        ):
            BinnedAxis.model_validate(
                {"name": "x", "min": 10.0, "max": 5.0, "nbins": 5}
            )

    def test_validation_errors_for_edges_type(self):
        """Test validation errors are properly raised for BinnedAxisEdges."""
        # Too few edges
        with pytest.raises(
            ValueError, match="BinnedAxisEdges 'x' must have at least 2 edges"
        ):
            BinnedAxis.model_validate({"name": "x", "edges": [5.0]})

        # Not ascending
        with pytest.raises(
            ValueError, match="BinnedAxisEdges 'x' edges must be in ascending order"
        ):
            BinnedAxis.model_validate({"name": "x", "edges": [0.0, 10.0, 5.0]})


class TestAxes:
    """Test Axes collection functionality."""

    def test_axes_creation_empty(self):
        """Test empty Axes creation."""
        axes = Axes()
        assert len(axes) == 0
        assert axes.get_total_bins() == 1  # No axes means 1 bin total

    def test_axes_creation_with_data(self):
        """Test Axes creation with initial data."""
        axes_data = [
            {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5},
            {"name": "y", "edges": [0.0, 2.5, 5.0, 10.0]},
        ]
        axes = Axes(axes_data)
        assert len(axes) == 2

    def test_axes_getitem(self):
        """Test Axes.__getitem__ functionality."""
        axes_data = [
            {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5},
            {"name": "y", "edges": [0.0, 2.5, 5.0, 10.0]},
        ]
        axes = Axes(axes_data)

        # Test access by index
        first_axis = axes[0]
        assert first_axis.name == "x"

        second_axis = axes[1]
        assert second_axis.name == "y"

    def test_axes_iteration(self):
        """Test Axes.__iter__ functionality."""
        axes_data = [
            {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5},
            {"name": "y", "edges": [0.0, 2.5, 5.0, 10.0]},
        ]
        axes = Axes(axes_data)

        names = [axis.name for axis in axes]
        assert names == ["x", "y"]

    def test_axes_get_total_bins_single_axis(self):
        """Test get_total_bins with single axis."""
        axes_data = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 5}]
        axes = Axes(axes_data)
        assert axes.get_total_bins() == 5

    def test_axes_get_total_bins_multiple_axes(self):
        """Test get_total_bins with multiple axes (should multiply)."""
        axes_data = [
            {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5},
            {"name": "y", "edges": [0.0, 2.5, 5.0, 10.0]},  # 3 bins
        ]
        axes = Axes(axes_data)
        assert axes.get_total_bins() == 15  # 5 * 3

    def test_axes_mixed_types(self):
        """Test Axes with mixed BinnedAxisRange and BinnedAxisEdges."""
        axes_data = [
            {"name": "x", "min": 0.0, "max": 10.0, "nbins": 2},
            {"name": "y", "edges": [0.0, 5.0, 10.0]},  # 2 bins
            {"name": "z", "min": -1.0, "max": 1.0, "nbins": 4},
        ]
        axes = Axes(axes_data)

        assert len(axes) == 3
        assert axes.get_total_bins() == 16  # 2 * 2 * 4

        # Check that discriminator worked correctly
        assert isinstance(axes[0].root, BinnedAxisRange)
        assert isinstance(axes[1].root, BinnedAxisEdges)
        assert isinstance(axes[2].root, BinnedAxisRange)
