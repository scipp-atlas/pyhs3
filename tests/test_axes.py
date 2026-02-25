"""
Tests for axis module.

Test axis classes including Axis, UnbinnedAxis, BinnedAxis discriminated union
functionality with RegularAxis and IrregularAxis.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from pyhs3.axes import (
    Axes,
    Axis,
    BinnedAxes,
    BinnedAxis,
    IrregularAxis,
    RegularAxis,
    UnbinnedAxis,
    _binned_axis_discriminator,
)

hist = pytest.importorskip("hist", reason="hist not installed")

binned_axis_adapter = TypeAdapter(BinnedAxis)


def make_binned_axis(**kwargs):
    return binned_axis_adapter.validate_python(kwargs)


class TestAxis:
    """Tests for the base Axis class."""

    def test_axis_creation_with_bounds(self):
        """Test Axis creation with required min/max bounds."""
        axis = Axis(name="test_var", min=0.0, max=10.0)
        assert axis.name == "test_var"
        assert not hasattr(axis, "min")
        assert not hasattr(axis, "max")


class TestUnbinnedAxis:
    """Tests for the UnbinnedAxis class."""

    def test_unbinned_axis_creation(self):
        """Test UnbinnedAxis creation with required min/max."""
        axis = UnbinnedAxis(name="x", min=0.0, max=5.0)
        assert axis.name == "x"
        assert axis.min == 0.0
        assert axis.max == 5.0

    def test_unbinned_axis_min_required(self):
        """Test that UnbinnedAxis requires min."""
        with pytest.raises(ValidationError, match="Field required"):
            UnbinnedAxis(name="x", max=5.0)

    def test_unbinned_axis_max_required(self):
        """Test that UnbinnedAxis requires max."""
        with pytest.raises(ValidationError, match="Field required"):
            UnbinnedAxis(name="x", min=0.0)

    def test_unbinned_axis_both_required(self):
        """Test that UnbinnedAxis requires both min and max."""
        with pytest.raises(ValidationError):
            UnbinnedAxis(name="x")


class TestRegularAxis:
    """Test RegularAxis functionality."""

    def test_valid_binned_axis_range(self):
        """Test valid RegularAxis creation."""
        axis = RegularAxis(name="x", min=0.0, max=10.0, nbins=5)
        assert axis.name == "x"
        assert axis.min == 0.0
        assert axis.max == 10.0
        assert axis.nbins == 5

    def test_binned_axis_range_missing_min(self):
        """Test RegularAxis raises error when min is missing."""
        with pytest.raises(ValidationError):
            RegularAxis(name="x", max=10.0, nbins=5)

    def test_binned_axis_range_missing_max(self):
        """Test RegularAxis raises error when max is missing."""
        with pytest.raises(ValidationError):
            RegularAxis(name="x", min=0.0, nbins=5)

    def test_binned_axis_range_invalid_range(self):
        """Test RegularAxis raises error when min >= max."""
        with pytest.raises(
            ValidationError, match=r"Axis 'x': max \(5\.0\) must be >= min \(10\.0\)"
        ):
            RegularAxis(name="x", min=10.0, max=5.0, nbins=5)

    def test_binned_axis_range_equal_min_max(self):
        """Test RegularAxis allows equal min and max (though not very useful)."""
        # The base Axis class allows min == max, so this should work
        axis = RegularAxis(name="x", min=5.0, max=5.0, nbins=5)
        assert axis.min == 5.0
        assert axis.max == 5.0
        assert axis.nbins == 5

    def test_binned_axis_range_zero_bins(self):
        """Test RegularAxis raises error when nbins is zero."""
        with pytest.raises(
            ValueError,
            match="RegularAxis 'x' must have positive number of bins, got 0",
        ):
            RegularAxis(name="x", min=0.0, max=10.0, nbins=0)

    def test_binned_axis_range_negative_bins(self):
        """Test RegularAxis raises error when nbins is negative."""
        with pytest.raises(
            ValueError,
            match="RegularAxis 'x' must have positive number of bins, got -5",
        ):
            RegularAxis(name="x", min=0.0, max=10.0, nbins=-5)


class TestIrregularAxis:
    """Test IrregularAxis functionality."""

    def test_valid_binned_axis_edges(self):
        """Test valid IrregularAxis creation."""
        axis = IrregularAxis(name="x", edges=[0.0, 2.5, 5.0, 7.5, 10.0])
        assert axis.name == "x"
        assert axis.edges == [0.0, 2.5, 5.0, 7.5, 10.0]
        assert axis.nbins == 4

    def test_binned_axis_edges_minimum_edges(self):
        """Test IrregularAxis with minimum valid edges."""
        axis = IrregularAxis(name="x", edges=[0.0, 10.0])
        assert axis.edges == [0.0, 10.0]
        assert axis.nbins == 1

    def test_binned_axis_edges_too_few_edges(self):
        """Test IrregularAxis raises error with too few edges."""
        with pytest.raises(
            ValueError, match="IrregularAxis 'x' must have at least 2 edges"
        ):
            IrregularAxis(name="x", edges=[5.0])

    def test_binned_axis_edges_empty_edges(self):
        """Test IrregularAxis raises error with empty edges."""
        with pytest.raises(
            ValueError, match="IrregularAxis 'x' must have at least 2 edges"
        ):
            IrregularAxis(name="x", edges=[])

    def test_binned_axis_edges_not_ascending(self):
        """Test IrregularAxis raises error when edges are not in ascending order."""
        with pytest.raises(
            ValueError, match="IrregularAxis 'x' edges must be in ascending order"
        ):
            IrregularAxis(name="x", edges=[0.0, 10.0, 5.0])

    def test_binned_axis_edges_equal_edges(self):
        """Test IrregularAxis raises error when edges are equal."""
        with pytest.raises(
            ValueError, match="IrregularAxis 'x' edges must be in ascending order"
        ):
            IrregularAxis(name="x", edges=[0.0, 5.0, 5.0, 10.0])


class TestBinnedAxisDiscriminatedUnion:
    """Test BinnedAxis discriminated union functionality."""

    def test_discriminator_selects_range_with_nbins(self):
        """Test discriminator selects RegularAxis when nbins is present."""
        axis_data = {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5}
        axis = binned_axis_adapter.validate_python(axis_data)
        assert isinstance(axis, RegularAxis)
        assert axis.name == "x"
        assert axis.nbins == 5

    def test_discriminator_selects_edges_with_edges(self):
        """Test discriminator selects IrregularAxis when edges is present."""
        axis_data = {"name": "x", "edges": [0.0, 2.5, 5.0, 7.5, 10.0]}
        axis = binned_axis_adapter.validate_python(axis_data)
        assert isinstance(axis, IrregularAxis)
        assert axis.name == "x"
        assert axis.edges == [0.0, 2.5, 5.0, 7.5, 10.0]

    def test_discriminator_errors_when_nothing_specified(self):
        """Test discriminator does not default to RegularAxis when neither nbins nor edges specified."""
        axis_data = {"name": "x", "min": 0.0, "max": 10.0}
        with pytest.raises(ValidationError, match="must specify either"):
            binned_axis_adapter.validate_python(axis_data)

    def test_binned_axis_validation_no_binning_with_bounds_fails(self):
        """Test that BinnedAxis requires binning, not just bounds."""
        with pytest.raises(
            ValidationError, match="must specify either regular binning"
        ):
            make_binned_axis(name="test", min=0.0, max=10.0)

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
        with pytest.raises(ValidationError, match="must specify either"):
            binned_axis_adapter.validate_python(axis_data)

    def test_discriminator_with_pydantic_instance(self):
        """Test discriminator works with already instantiated objects."""
        range_axis = RegularAxis(name="x", min=0.0, max=10.0, nbins=5)
        edges_axis = IrregularAxis(name="y", edges=[0.0, 5.0, 10.0])

        # Test that both instances work correctly
        assert isinstance(range_axis, RegularAxis)
        assert isinstance(edges_axis, IrregularAxis)
        assert range_axis.nbins == 5
        assert edges_axis.nbins == 2

    def test_nbins_works_for_both_types(self):
        """Test that nbins method works for both axis types."""
        range_axis_data = {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5}
        edges_axis_data = {"name": "y", "edges": [0.0, 2.5, 5.0, 7.5, 10.0]}

        range_axis = binned_axis_adapter.validate_python(range_axis_data)
        edges_axis = binned_axis_adapter.validate_python(edges_axis_data)

        assert range_axis.nbins == 5
        assert edges_axis.nbins == 4

    def test_validation_errors_for_range_type(self):
        """Test validation errors are properly raised for RegularAxis."""
        # Missing min/max
        with pytest.raises(ValidationError):
            binned_axis_adapter.validate_python({"name": "x", "nbins": 5})

        # Invalid range
        with pytest.raises(
            ValidationError, match=r"Axis 'x': max \(5\.0\) must be >= min \(10\.0\)"
        ):
            binned_axis_adapter.validate_python(
                {"name": "x", "min": 10.0, "max": 5.0, "nbins": 5}
            )

    def test_validation_errors_for_edges_type(self):
        """Test validation errors are properly raised for IrregularAxis."""
        # Too few edges
        with pytest.raises(
            ValueError, match="IrregularAxis 'x' must have at least 2 edges"
        ):
            binned_axis_adapter.validate_python({"name": "x", "edges": [5.0]})

        # Not ascending
        with pytest.raises(
            ValueError, match="IrregularAxis 'x' edges must be in ascending order"
        ):
            binned_axis_adapter.validate_python(
                {"name": "x", "edges": [0.0, 10.0, 5.0]}
            )


class TestBinnedAxisEdges:
    """Test BinnedAxis edges property."""

    def test_binned_axis_edges_regular_binning(self):
        """Test edges property with regular binning."""
        axis = make_binned_axis(name="mass", min=0.0, max=10.0, nbins=5)

        # Should return 6 edges for 5 bins: [0, 2, 4, 6, 8, 10]
        expected_edges = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        assert len(axis.edges) == 6
        assert axis.edges == pytest.approx(expected_edges)

    def test_binned_axis_edges_irregular_binning(self):
        """Test edges property with irregular binning."""
        custom_edges = [0.0, 1.0, 5.0, 12.0, 25.0]
        axis = make_binned_axis(name="pt", edges=custom_edges)

        # Should return the provided edges exactly
        assert axis.edges == custom_edges

    def test_binned_axis_edges_single_bin(self):
        """Test edges property with single bin."""
        axis = make_binned_axis(name="single", min=1.0, max=2.0, nbins=1)

        # Should return 2 edges for 1 bin: [1.0, 2.0]
        assert len(axis.edges) == 2
        assert axis.edges == pytest.approx([1.0, 2.0])

    def test_binned_axis_edges_zero_range(self):
        """Test edges property with zero range (min=max)."""
        axis = make_binned_axis(name="zero_range", min=5.0, max=5.0, nbins=1)

        # Should return [5.0, 5.0] for zero range
        assert len(axis.edges) == 2
        assert axis.edges == pytest.approx([5.0, 5.0])

    def test_binned_axis_edges_negative_range(self):
        """Test edges property with negative values."""
        axis = make_binned_axis(name="negative", min=-10.0, max=-2.0, nbins=4)

        # Should handle negative values correctly: [-10, -8, -6, -4, -2]
        expected_edges = [-10.0, -8.0, -6.0, -4.0, -2.0]
        assert len(axis.edges) == 5
        assert axis.edges == pytest.approx(expected_edges)

    def test_binned_axis_edges_large_number_of_bins(self):
        """Test edges property with large number of bins."""
        axis = make_binned_axis(name="many_bins", min=0.0, max=100.0, nbins=1000)

        # Should return 1001 edges for 1000 bins
        assert len(axis.edges) == 1001
        assert axis.edges[0] == pytest.approx(0.0)
        assert axis.edges[-1] == pytest.approx(100.0)
        assert axis.edges[500] == pytest.approx(50.0)  # Middle should be 50.0


class TestBinnedAxisHistConversion:
    """Test BinnedAxis to_hist() conversion."""

    def test_binned_axis_to_hist_regular(self):
        """Test to_hist() method with regular binning."""
        axis = make_binned_axis(name="x", min=0.0, max=10.0, nbins=5)
        hist_axis = axis.to_hist()

        assert hist_axis.name == "x"
        assert len(hist_axis.edges) == 6  # 5 bins + 1

    def test_binned_axis_to_hist_irregular(self):
        """Test to_hist() method with irregular binning."""
        edges = [0.0, 1.0, 5.0, 10.0]
        axis = make_binned_axis(name="y", edges=edges)
        hist_axis = axis.to_hist()

        assert hist_axis.name == "y"
        assert len(hist_axis.edges) == 4


class TestAxes:
    """Test Axes collection functionality."""

    def test_axes_creation_empty(self):
        """Test empty Axes creation."""
        axes = BinnedAxes()
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
        axes = BinnedAxes(axes_data)
        assert axes.get_total_bins() == 5

    def test_axes_get_total_bins_multiple_axes(self):
        """Test get_total_bins with multiple axes (should multiply)."""
        axes_data = [
            {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5},
            {"name": "y", "edges": [0.0, 2.5, 5.0, 10.0]},  # 3 bins
        ]
        axes = BinnedAxes(axes_data)
        assert axes.get_total_bins() == 15  # 5 * 3

    def test_axes_mixed_types(self):
        """Test Axes with mixed RegularAxis and IrregularAxis."""
        axes_data = [
            {"name": "x", "min": 0.0, "max": 10.0, "nbins": 2},
            {"name": "y", "edges": [0.0, 5.0, 10.0]},  # 2 bins
            {"name": "z", "min": -1.0, "max": 1.0, "nbins": 4},
        ]
        axes = BinnedAxes(axes_data)

        assert len(axes) == 3
        assert axes.get_total_bins() == 16  # 2 * 2 * 4

        # Check that discriminator worked correctly
        assert isinstance(axes[0], RegularAxis)
        assert isinstance(axes[1], IrregularAxis)
        assert isinstance(axes[2], RegularAxis)


class TestBinnedAxisDiscriminator:
    """Test binned_axis_discriminator function."""

    def test_discriminator_with_dict_inputs(self):
        """Test discriminator function with dictionary inputs."""
        # Test with nbins present - should select 'range'
        dict_with_nbins = {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5}
        assert _binned_axis_discriminator(dict_with_nbins) == "regular"

        # Test with edges present - should select 'edges'
        dict_with_edges = {"name": "x", "edges": [0.0, 5.0, 10.0]}
        assert _binned_axis_discriminator(dict_with_edges) == "irregular"

        # Test with both present - nbins takes precedence, should select 'range'
        dict_with_both = {
            "name": "x",
            "min": 0.0,
            "max": 10.0,
            "nbins": 5,
            "edges": [0.0, 5.0, 10.0],
        }
        assert _binned_axis_discriminator(dict_with_both) is None

        # Test with neither present - should return None
        dict_with_neither = {"name": "x", "min": 0.0, "max": 10.0}
        assert _binned_axis_discriminator(dict_with_neither) is None

    def test_discriminator_with_object_instances(self):
        """Test discriminator function with actual object instances."""
        # Test with RegularAxis instance
        regular_instance = RegularAxis(name="x", min=0.0, max=10.0, nbins=5)
        assert _binned_axis_discriminator(regular_instance) == "regular"

        # Test with IrregularAxis instance
        irregular_instance = IrregularAxis(name="x", edges=[0.0, 5.0, 10.0])
        assert _binned_axis_discriminator(irregular_instance) == "irregular"

    def test_discriminator_serialization_roundtrip(self):
        """Test that discriminator works correctly during serialization round-trips."""
        # Test RegularAxis round-trip
        range_data = {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5}
        range_axis = binned_axis_adapter.validate_python(range_data)

        # Serialize and deserialize
        serialized = range_axis.model_dump()
        reconstructed = binned_axis_adapter.validate_python(serialized)

        # Verify the discriminator works during both validation and serialization
        assert isinstance(range_axis, RegularAxis)
        assert isinstance(reconstructed, RegularAxis)
        assert range_axis.nbins == reconstructed.nbins == 5

        # Test IrregularAxis round-trip
        edges_data = {"name": "y", "edges": [0.0, 2.5, 5.0, 10.0]}
        edges_axis = binned_axis_adapter.validate_python(edges_data)

        # Serialize and deserialize
        serialized = edges_axis.model_dump()
        reconstructed = binned_axis_adapter.validate_python(serialized)

        # Verify the discriminator works during both validation and serialization
        assert isinstance(edges_axis, IrregularAxis)
        assert isinstance(reconstructed, IrregularAxis)
        assert edges_axis.edges == reconstructed.edges == [0.0, 2.5, 5.0, 10.0]

    def test_discriminator_handles_serialization_input(self):
        """Test that discriminator specifically handles model instances during serialization."""
        # Create instances
        regular_instance = make_binned_axis(name="x", min=0.0, max=10.0, nbins=5)
        irregular_instance = make_binned_axis(name="y", edges=[0.0, 5.0, 10.0])

        # Test that discriminator correctly identifies them as model instances
        # (This simulates what happens during serialization)
        assert _binned_axis_discriminator(regular_instance) == "regular"
        assert _binned_axis_discriminator(irregular_instance) == "irregular"

        # Test serialization works correctly
        range_serialized = regular_instance.model_dump()
        edges_serialized = irregular_instance.model_dump()

        # Verify serialized data has correct structure
        assert "nbins" in range_serialized
        assert range_serialized["nbins"] == 5
        assert "edges" in edges_serialized
        assert edges_serialized["edges"] == [0.0, 5.0, 10.0]
