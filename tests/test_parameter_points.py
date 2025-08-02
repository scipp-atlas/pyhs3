"""
Unit tests for parameter_points module.

Tests for ParameterPoint, ParameterSet, and ParameterPoints collections,
including validation and access methods.
"""

from __future__ import annotations

import pytest

from pyhs3.parameter_points import ParameterPoint, ParameterPoints, ParameterSet


class TestParameterSet:
    """Tests for the ParameterSet class."""

    def test_parameter_set_creation(self):
        """Test basic ParameterSet creation."""
        param1 = ParameterPoint(name="mu", value=0.0)
        param2 = ParameterPoint(name="sigma", value=1.0)
        param_set = ParameterSet(name="test_set", parameters=[param1, param2])

        assert param_set.name == "test_set"
        assert len(param_set.parameters) == 2
        assert param_set.parameters[0].name == "mu"
        assert param_set.parameters[1].name == "sigma"

    def test_parameter_set_getitem_by_string(self):
        """Test ParameterSet.__getitem__ with string (parameter name)."""
        param1 = ParameterPoint(name="mu", value=0.0)
        param2 = ParameterPoint(name="sigma", value=1.0)
        param_set = ParameterSet(name="test_set", parameters=[param1, param2])

        # Test string access
        retrieved_mu = param_set["mu"]
        assert retrieved_mu.name == "mu"
        assert retrieved_mu.value == 0.0

        retrieved_sigma = param_set["sigma"]
        assert retrieved_sigma.name == "sigma"
        assert retrieved_sigma.value == 1.0

    def test_parameter_set_getitem_by_integer(self):
        """Test ParameterSet.__getitem__ with integer (index)."""
        param1 = ParameterPoint(name="mu", value=0.0)
        param2 = ParameterPoint(name="sigma", value=1.0)
        param_set = ParameterSet(name="test_set", parameters=[param1, param2])

        # Test integer access
        retrieved_first = param_set[0]
        assert retrieved_first.name == "mu"
        assert retrieved_first.value == 0.0

        retrieved_second = param_set[1]
        assert retrieved_second.name == "sigma"
        assert retrieved_second.value == 1.0

    def test_parameter_set_getitem_nonexistent_name_raises_keyerror(self):
        """Test that accessing non-existent parameter name raises KeyError."""
        param1 = ParameterPoint(name="mu", value=0.0)
        param_set = ParameterSet(name="test_set", parameters=[param1])

        with pytest.raises(KeyError):
            _ = param_set["nonexistent"]

    def test_parameter_set_getitem_invalid_index_raises_indexerror(self):
        """Test that accessing invalid index raises IndexError."""
        param1 = ParameterPoint(name="mu", value=0.0)
        param_set = ParameterSet(name="test_set", parameters=[param1])

        with pytest.raises(IndexError):
            _ = param_set[10]  # Out of bounds

    def test_parameter_set_get_method(self):
        """Test ParameterSet.get method with default values."""
        param1 = ParameterPoint(name="mu", value=0.0)
        param_set = ParameterSet(name="test_set", parameters=[param1])

        # Existing parameter
        retrieved = param_set.get("mu")
        assert retrieved.name == "mu"
        assert retrieved.value == 0.0

        # Non-existing parameter with default
        default_param = ParameterPoint(name="default", value=999.0)
        result = param_set.get("nonexistent", default_param)
        assert result.name == "default"
        assert result.value == 999.0

        # Non-existing parameter without default (should return None)
        result = param_set.get("nonexistent")
        assert result is None

    def test_parameter_set_len(self):
        """Test ParameterSet.__len__ method."""
        param1 = ParameterPoint(name="mu", value=0.0)
        param2 = ParameterPoint(name="sigma", value=1.0)
        param_set = ParameterSet(name="test_set", parameters=[param1, param2])

        assert len(param_set) == 2

    def test_parameter_set_contains(self):
        """Test ParameterSet.__contains__ method."""
        param1 = ParameterPoint(name="mu", value=0.0)
        param_set = ParameterSet(name="test_set", parameters=[param1])

        assert "mu" in param_set
        assert "nonexistent" not in param_set

    def test_parameter_set_iter(self):
        """Test ParameterSet.__iter__ method."""
        param1 = ParameterPoint(name="mu", value=0.0)
        param2 = ParameterPoint(name="sigma", value=1.0)
        param_set = ParameterSet(name="test_set", parameters=[param1, param2])

        param_names = [param.name for param in param_set]
        assert param_names == ["mu", "sigma"]


class TestParameterPoints:
    """Tests for the ParameterPoints collection class."""

    def test_parameter_points_creation(self):
        """Test basic ParameterPoints creation."""
        param_set1 = ParameterSet(
            name="set1", parameters=[ParameterPoint(name="mu", value=0.0)]
        )
        param_set2 = ParameterSet(
            name="set2", parameters=[ParameterPoint(name="sigma", value=1.0)]
        )

        param_points = ParameterPoints([param_set1, param_set2])
        assert len(param_points) == 2

    def test_parameter_points_getitem_by_string(self):
        """Test ParameterPoints.__getitem__ with string (set name)."""
        param_set1 = ParameterSet(
            name="set1", parameters=[ParameterPoint(name="mu", value=0.0)]
        )
        param_set2 = ParameterSet(
            name="set2", parameters=[ParameterPoint(name="sigma", value=1.0)]
        )

        param_points = ParameterPoints([param_set1, param_set2])

        # Test string access
        retrieved_set1 = param_points["set1"]
        assert retrieved_set1.name == "set1"

        retrieved_set2 = param_points["set2"]
        assert retrieved_set2.name == "set2"

    def test_parameter_points_getitem_by_integer(self):
        """Test ParameterPoints.__getitem__ with integer (index)."""
        param_set1 = ParameterSet(
            name="set1", parameters=[ParameterPoint(name="mu", value=0.0)]
        )
        param_set2 = ParameterSet(
            name="set2", parameters=[ParameterPoint(name="sigma", value=1.0)]
        )

        param_points = ParameterPoints([param_set1, param_set2])

        # Test integer access
        retrieved_first = param_points[0]
        assert retrieved_first.name == "set1"

        retrieved_second = param_points[1]
        assert retrieved_second.name == "set2"

    def test_parameter_points_get_method(self):
        """Test ParameterPoints.get method."""
        param_set1 = ParameterSet(
            name="set1", parameters=[ParameterPoint(name="mu", value=0.0)]
        )
        param_points = ParameterPoints([param_set1])

        # Existing set
        retrieved = param_points.get("set1")
        assert retrieved.name == "set1"

        # Non-existing set with default
        default_set = ParameterSet(name="default", parameters=[])
        result = param_points.get("nonexistent", default_set)
        assert result.name == "default"

        # Non-existing set without default
        result = param_points.get("nonexistent")
        assert result is None

    def test_parameter_points_contains(self):
        """Test ParameterPoints.__contains__ method."""
        param_set1 = ParameterSet(
            name="set1", parameters=[ParameterPoint(name="mu", value=0.0)]
        )
        param_points = ParameterPoints([param_set1])

        assert "set1" in param_points
        assert "nonexistent" not in param_points

    def test_parameter_points_iter(self):
        """Test ParameterPoints.__iter__ method."""
        param_set1 = ParameterSet(
            name="set1", parameters=[ParameterPoint(name="mu", value=0.0)]
        )
        param_set2 = ParameterSet(
            name="set2", parameters=[ParameterPoint(name="sigma", value=1.0)]
        )
        param_points = ParameterPoints([param_set1, param_set2])

        set_names = [param_set.name for param_set in param_points]
        assert set_names == ["set1", "set2"]
