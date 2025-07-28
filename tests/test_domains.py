"""
Unit tests for domain implementations.

Tests for Axis, ProductDomain, and related domain functionality,
including validation of range constraints and parameter bounds.
"""

from __future__ import annotations

import pytest

from pyhs3.domains import Axis, Domains, ProductDomain


class TestAxis:
    """Tests for the Axis class."""

    def test_axis_creation_basic(self):
        """Test basic Axis creation with name only."""
        axis = Axis(name="test_param")
        assert axis.name == "test_param"
        assert axis.min is None
        assert axis.max is None

    def test_axis_creation_with_range(self):
        """Test Axis creation with min and max values."""
        axis = Axis(name="test_param", min=0.0, max=10.0)
        assert axis.name == "test_param"
        assert axis.min == 0.0
        assert axis.max == 10.0

    def test_axis_creation_min_only(self):
        """Test Axis creation with only min value."""
        axis = Axis(name="test_param", min=0.0)
        assert axis.name == "test_param"
        assert axis.min == 0.0
        assert axis.max is None

    def test_axis_creation_max_only(self):
        """Test Axis creation with only max value."""
        axis = Axis(name="test_param", max=10.0)
        assert axis.name == "test_param"
        assert axis.min is None
        assert axis.max == 10.0

    def test_axis_validation_max_less_than_min_raises_error(self):
        """Test that Axis validation raises ValueError when max < min."""
        with pytest.raises(
            ValueError,
            match=r"Axis 'test_param': max \(5\.0\) must be >= min \(10\.0\)",
        ):
            Axis(name="test_param", min=10.0, max=5.0)

    def test_axis_validation_max_equal_min_allowed(self):
        """Test that Axis allows max == min."""
        axis = Axis(name="test_param", min=5.0, max=5.0)
        assert axis.min == 5.0
        assert axis.max == 5.0

    def test_axis_validation_max_greater_than_min_allowed(self):
        """Test that Axis allows max > min."""
        axis = Axis(name="test_param", min=5.0, max=10.0)
        assert axis.min == 5.0
        assert axis.max == 10.0

    def test_axis_validation_negative_values(self):
        """Test Axis validation with negative values."""
        # Valid: both negative, max > min
        axis1 = Axis(name="param1", min=-10.0, max=-5.0)
        assert axis1.min == -10.0
        assert axis1.max == -5.0

        # Valid: min negative, max positive
        axis2 = Axis(name="param2", min=-5.0, max=5.0)
        assert axis2.min == -5.0
        assert axis2.max == 5.0

        # Invalid: max < min with negative values
        with pytest.raises(
            ValueError, match=r"Axis 'param3': max \(-10\.0\) must be >= min \(-5\.0\)"
        ):
            Axis(name="param3", min=-5.0, max=-10.0)

    def test_axis_from_dict(self):
        """Test Axis creation from dictionary."""
        config = {"name": "test_param", "min": 0.0, "max": 10.0}
        axis = Axis(**config)
        assert axis.name == "test_param"
        assert axis.min == 0.0
        assert axis.max == 10.0

    def test_axis_from_dict_validation_error(self):
        """Test that Axis.from_dict also validates range constraints."""
        config = {"name": "test_param", "min": 10.0, "max": 5.0}
        with pytest.raises(
            ValueError,
            match=r"Axis 'test_param': max \(5\.0\) must be >= min \(10\.0\)",
        ):
            Axis(**config)


class TestProductDomain:
    """Tests for the ProductDomain class."""

    def test_product_domain_creation(self):
        """Test basic ProductDomain creation."""
        axes = [
            Axis(name="param1", min=0.0, max=1.0),
            Axis(name="param2", min=-5.0, max=5.0),
        ]
        domain = ProductDomain(name="test_domain", axes=axes)
        assert domain.name == "test_domain"
        assert len(domain.axes) == 2
        assert domain.axes[0].name == "param1"
        assert domain.axes[1].name == "param2"

    def test_product_domain_empty_axes(self):
        """Test ProductDomain creation with empty axes list."""
        domain = ProductDomain(name="empty_domain", axes=[])
        assert domain.name == "empty_domain"
        assert len(domain.axes) == 0

    def test_product_domain_from_dict(self):
        """Test ProductDomain creation from dictionary."""
        config = {
            "name": "test_domain",
            "type": "product_domain",
            "axes": [
                {"name": "param1", "min": 0.0, "max": 1.0},
                {"name": "param2", "min": -5.0, "max": 5.0},
            ],
        }
        domain = ProductDomain(**config)
        assert domain.name == "test_domain"
        assert len(domain.axes) == 2
        assert domain.axes[0].name == "param1"
        assert domain.axes[1].name == "param2"

    def test_product_domain_validation_propagates_axis_errors(self):
        """Test that ProductDomain propagates axis validation errors."""
        config = {
            "name": "test_domain",
            "type": "product_domain",
            "axes": [
                {"name": "param1", "min": 0.0, "max": 1.0},
                {"name": "param2", "min": 10.0, "max": 5.0},  # Invalid: max < min
            ],
        }
        with pytest.raises(
            ValueError, match=r"Axis 'param2': max \(5\.0\) must be >= min \(10\.0\)"
        ):
            ProductDomain(**config)


class TestDomains:
    """Tests for the Domains collection class."""

    def test_domains_creation_empty(self):
        """Test Domains creation with empty list."""
        domains = Domains([])
        assert len(domains.root) == 0

    def test_domains_creation_with_domains(self):
        """Test Domains creation with domain list."""
        domain1 = ProductDomain(
            name="domain1", axes=[Axis(name="param1", min=0.0, max=1.0)]
        )
        domain2 = ProductDomain(
            name="domain2", axes=[Axis(name="param2", min=-1.0, max=1.0)]
        )

        domains = Domains([domain1, domain2])
        assert len(domains.root) == 2
        assert domains.root[0].name == "domain1"
        assert domains.root[1].name == "domain2"

    def test_domains_iteration(self):
        """Test that Domains can be iterated."""
        domain1 = ProductDomain(name="domain1", axes=[])
        domain2 = ProductDomain(name="domain2", axes=[])

        domains = Domains([domain1, domain2])
        domain_names = [domain.name for domain in domains]
        assert domain_names == ["domain1", "domain2"]

    def test_domains_length(self):
        """Test that len() works on Domains."""
        domain1 = ProductDomain(name="domain1", axes=[])
        domains = Domains([domain1])
        assert len(domains) == 1

    def test_domains_validation_propagates_domain_errors(self):
        """Test that Domains propagates domain validation errors."""
        # This should fail when creating the ProductDomain due to invalid axis
        with pytest.raises(
            ValueError, match=r"Axis 'param1': max \(5\.0\) must be >= min \(10\.0\)"
        ):
            ProductDomain(
                name="invalid_domain", axes=[Axis(name="param1", min=10.0, max=5.0)]
            )
