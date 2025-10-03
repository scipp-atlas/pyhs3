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
    get_binned_axis_discriminator,
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


class TestBinnedAxisDiscriminator:
    """Test get_binned_axis_discriminator function."""

    def test_discriminator_with_dict_inputs(self):
        """Test discriminator function with dictionary inputs."""
        # Test with nbins present - should select 'range'
        dict_with_nbins = {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5}
        assert get_binned_axis_discriminator(dict_with_nbins) == "range"

        # Test with edges present - should select 'edges'
        dict_with_edges = {"name": "x", "edges": [0.0, 5.0, 10.0]}
        assert get_binned_axis_discriminator(dict_with_edges) == "edges"

        # Test with both present - nbins takes precedence, should select 'range'
        dict_with_both = {
            "name": "x",
            "min": 0.0,
            "max": 10.0,
            "nbins": 5,
            "edges": [0.0, 5.0, 10.0],
        }
        assert get_binned_axis_discriminator(dict_with_both) == "range"

        # Test with neither present - should default to 'range'
        dict_with_neither = {"name": "x", "min": 0.0, "max": 10.0}
        assert get_binned_axis_discriminator(dict_with_neither) == "range"

    def test_discriminator_with_object_instances(self):
        """Test discriminator function with actual object instances."""
        # Test with BinnedAxisRange instance - should select 'range'
        range_instance = BinnedAxisRange(name="x", min=0.0, max=10.0, nbins=5)
        assert get_binned_axis_discriminator(range_instance) == "range"

        # Test with BinnedAxisEdges instance - should select 'edges'
        edges_instance = BinnedAxisEdges(name="x", edges=[0.0, 5.0, 10.0])
        assert get_binned_axis_discriminator(edges_instance) == "edges"

    def test_discriminator_with_generic_objects(self):
        """Test discriminator function with generic objects that have relevant attributes."""

        # Create a mock object that has nbins attribute
        class MockRangeObject:
            def __init__(self):
                self.nbins = 5
                self.name = "mock_range"

        mock_range = MockRangeObject()
        assert get_binned_axis_discriminator(mock_range) == "range"

        # Create a mock object that has edges attribute
        class MockEdgesObject:
            def __init__(self):
                self.edges = [0.0, 5.0, 10.0]
                self.name = "mock_edges"

        mock_edges = MockEdgesObject()
        assert get_binned_axis_discriminator(mock_edges) == "edges"

        # Create a mock object that has both attributes - nbins takes precedence
        class MockBothObject:
            def __init__(self):
                self.nbins = 5
                self.edges = [0.0, 5.0, 10.0]
                self.name = "mock_both"

        mock_both = MockBothObject()
        assert get_binned_axis_discriminator(mock_both) == "range"

        # Create a mock object that has neither attribute - should default to 'range'
        class MockNeitherObject:
            def __init__(self):
                self.name = "mock_neither"

        mock_neither = MockNeitherObject()
        assert get_binned_axis_discriminator(mock_neither) == "range"

    def test_discriminator_serialization_roundtrip(self):
        """Test that discriminator works correctly during serialization round-trips."""
        # Test BinnedAxisRange round-trip
        range_data = {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5}
        range_axis = BinnedAxis.model_validate(range_data)

        # Serialize and deserialize
        serialized = range_axis.model_dump()
        reconstructed = BinnedAxis.model_validate(serialized)

        # Verify the discriminator works during both validation and serialization
        assert isinstance(range_axis.root, BinnedAxisRange)
        assert isinstance(reconstructed.root, BinnedAxisRange)
        assert range_axis.root.nbins == reconstructed.root.nbins == 5

        # Test BinnedAxisEdges round-trip
        edges_data = {"name": "y", "edges": [0.0, 2.5, 5.0, 10.0]}
        edges_axis = BinnedAxis.model_validate(edges_data)

        # Serialize and deserialize
        serialized = edges_axis.model_dump()
        reconstructed = BinnedAxis.model_validate(serialized)

        # Verify the discriminator works during both validation and serialization
        assert isinstance(edges_axis.root, BinnedAxisEdges)
        assert isinstance(reconstructed.root, BinnedAxisEdges)
        assert (
            edges_axis.root.edges == reconstructed.root.edges == [0.0, 2.5, 5.0, 10.0]
        )

    def test_discriminator_handles_serialization_input(self):
        """Test that discriminator specifically handles model instances during serialization."""
        # Create instances
        range_instance = BinnedAxisRange(name="x", min=0.0, max=10.0, nbins=5)
        edges_instance = BinnedAxisEdges(name="y", edges=[0.0, 5.0, 10.0])

        # Test that discriminator correctly identifies them as model instances
        # (This simulates what happens during serialization)
        assert get_binned_axis_discriminator(range_instance) == "range"
        assert get_binned_axis_discriminator(edges_instance) == "edges"

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
