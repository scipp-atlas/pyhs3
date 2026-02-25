"""
Tests for HistFactory axes module.

Test BinnedAxis discriminated union functionality with RegularAxis and IrregularAxis.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pyhs3.data import (
    Axes,
    BinnedAxes,
    BinnedAxis,
    IrregularAxis,
    RegularAxis,
    binned_axis_discriminator,
)


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
        axis = BinnedAxis.model_validate(axis_data)
        assert isinstance(axis.root, RegularAxis)
        assert axis.name == "x"
        assert axis.root.nbins == 5

    def test_discriminator_selects_edges_with_edges(self):
        """Test discriminator selects IrregularAxis when edges is present."""
        axis_data = {"name": "x", "edges": [0.0, 2.5, 5.0, 7.5, 10.0]}
        axis = BinnedAxis.model_validate(axis_data)
        assert isinstance(axis.root, IrregularAxis)
        assert axis.name == "x"
        assert axis.root.edges == [0.0, 2.5, 5.0, 7.5, 10.0]

    def test_discriminator_errors_when_nothing_specified(self):
        """Test discriminator does not default to RegularAxis when neither nbins nor edges specified."""
        axis_data = {"name": "x", "min": 0.0, "max": 10.0}
        with pytest.raises(ValidationError, match="must specify either"):
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
        with pytest.raises(ValidationError, match="must specify either"):
            BinnedAxis.model_validate(axis_data)

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

        range_axis = BinnedAxis.model_validate(range_axis_data)
        edges_axis = BinnedAxis.model_validate(edges_axis_data)

        assert range_axis.nbins == 5
        assert edges_axis.nbins == 4

    def test_validation_errors_for_range_type(self):
        """Test validation errors are properly raised for RegularAxis."""
        # Missing min/max
        with pytest.raises(ValidationError):
            BinnedAxis.model_validate({"name": "x", "nbins": 5})

        # Invalid range
        with pytest.raises(
            ValidationError, match=r"Axis 'x': max \(5\.0\) must be >= min \(10\.0\)"
        ):
            BinnedAxis.model_validate(
                {"name": "x", "min": 10.0, "max": 5.0, "nbins": 5}
            )

    def test_validation_errors_for_edges_type(self):
        """Test validation errors are properly raised for IrregularAxis."""
        # Too few edges
        with pytest.raises(
            ValueError, match="IrregularAxis 'x' must have at least 2 edges"
        ):
            BinnedAxis.model_validate({"name": "x", "edges": [5.0]})

        # Not ascending
        with pytest.raises(
            ValueError, match="IrregularAxis 'x' edges must be in ascending order"
        ):
            BinnedAxis.model_validate({"name": "x", "edges": [0.0, 10.0, 5.0]})


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
        assert isinstance(axes[0].root, RegularAxis)
        assert isinstance(axes[1].root, IrregularAxis)
        assert isinstance(axes[2].root, RegularAxis)


class TestBinnedAxisDiscriminator:
    """Test binned_axis_discriminator function."""

    def test_discriminator_with_dict_inputs(self):
        """Test discriminator function with dictionary inputs."""
        # Test with nbins present - should select 'range'
        dict_with_nbins = {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5}
        assert binned_axis_discriminator(dict_with_nbins) == "regular"

        # Test with edges present - should select 'edges'
        dict_with_edges = {"name": "x", "edges": [0.0, 5.0, 10.0]}
        assert binned_axis_discriminator(dict_with_edges) == "irregular"

        # Test with both present - nbins takes precedence, should select 'range'
        dict_with_both = {
            "name": "x",
            "min": 0.0,
            "max": 10.0,
            "nbins": 5,
            "edges": [0.0, 5.0, 10.0],
        }
        assert binned_axis_discriminator(dict_with_both) is None

        # Test with neither present - should default to 'range'
        dict_with_neither = {"name": "x", "min": 0.0, "max": 10.0}
        # with pytest.raises(ValidationError, match="Field required"):
        binned_axis_discriminator(dict_with_neither)

    def test_discriminator_with_object_instances(self):
        """Test discriminator function with actual object instances."""
        # Test with RegularAxis instance
        range_instance = RegularAxis(name="x", min=0.0, max=10.0, nbins=5)
        assert binned_axis_discriminator(range_instance) == "regular"

        # Test with IrregularAxis instance
        edges_instance = IrregularAxis(name="x", edges=[0.0, 5.0, 10.0])
        assert binned_axis_discriminator(edges_instance) == "irregular"

    def test_discriminator_serialization_roundtrip(self):
        """Test that discriminator works correctly during serialization round-trips."""
        # Test RegularAxis round-trip
        range_data = {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5}
        range_axis = BinnedAxis.model_validate(range_data)

        # Serialize and deserialize
        serialized = range_axis.model_dump()
        reconstructed = BinnedAxis.model_validate(serialized)

        # Verify the discriminator works during both validation and serialization
        assert isinstance(range_axis.root, RegularAxis)
        assert isinstance(reconstructed.root, RegularAxis)
        assert range_axis.root.nbins == reconstructed.root.nbins == 5

        # Test IrregularAxis round-trip
        edges_data = {"name": "y", "edges": [0.0, 2.5, 5.0, 10.0]}
        edges_axis = BinnedAxis.model_validate(edges_data)

        # Serialize and deserialize
        serialized = edges_axis.model_dump()
        reconstructed = BinnedAxis.model_validate(serialized)

        # Verify the discriminator works during both validation and serialization
        assert isinstance(edges_axis.root, IrregularAxis)
        assert isinstance(reconstructed.root, IrregularAxis)
        assert (
            edges_axis.root.edges == reconstructed.root.edges == [0.0, 2.5, 5.0, 10.0]
        )

    def test_discriminator_handles_serialization_input(self):
        """Test that discriminator specifically handles model instances during serialization."""
        # Create instances
        range_instance = RegularAxis(name="x", min=0.0, max=10.0, nbins=5)
        edges_instance = IrregularAxis(name="y", edges=[0.0, 5.0, 10.0])

        # Test that discriminator correctly identifies them as model instances
        # (This simulates what happens during serialization)
        assert binned_axis_discriminator(range_instance) == "regular"
        assert binned_axis_discriminator(edges_instance) == "irregular"

        # Verify that these instances can be wrapped in BinnedAxis for serialization
        wrapped_range = BinnedAxis(root=range_instance)
        wrapped_edges = BinnedAxis(root=edges_instance)

        # Test serialization works correctly
        range_serialized = wrapped_range.model_dump()
        edges_serialized = wrapped_edges.model_dump()

        # Verify serialized data has correct structure
        assert "nbins" in range_serialized
        assert range_serialized["nbins"] == 5
        assert "edges" in edges_serialized
        assert edges_serialized["edges"] == [0.0, 5.0, 10.0]
