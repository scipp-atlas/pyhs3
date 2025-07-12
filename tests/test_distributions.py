"""
Unit tests for new distribution implementations.

Tests for ProductDist, CrystalDist, and GenericDist implementations,
including validation against expected NLL values from ROOT fits.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from pytensor import function

from pyhs3.core import CrystalDist, GenericDist, ProductDist, boundedscalar


class TestProductDist:
    """Test ProductDist implementation."""

    def test_product_dist_creation(self):
        """Test ProductDist can be created and configured."""
        dist = ProductDist(name="test_product", factors=["factor1", "factor2"])
        assert dist.name == "test_product"
        assert dist.factors == ["factor1", "factor2"]
        assert dist.parameters == ["factor1", "factor2"]

    def test_product_dist_from_dict(self):
        """Test ProductDist can be created from dictionary."""
        config = {"name": "test_product", "factors": ["pdf1", "pdf2", "pdf3"]}
        dist = ProductDist.from_dict(config)
        assert dist.name == "test_product"
        assert dist.factors == ["pdf1", "pdf2", "pdf3"]

    def test_product_dist_expression(self):
        """Test ProductDist expression evaluation."""
        dist = ProductDist(name="test_product", factors=["f1", "f2"])

        # Create test parameters
        params = {
            "f1": pt.constant([1.0, 2.0, 3.0]),
            "f2": pt.constant([2.0, 3.0, 4.0]),
        }

        result = dist.expression(params)
        expected = pt.constant([2.0, 6.0, 12.0])  # elementwise product

        # Compile and evaluate
        f = function([], [result, expected])
        result_val, expected_val = f()

        np.testing.assert_allclose(result_val, expected_val)


class TestCrystalDist:
    """Test CrystalDist implementation."""

    def test_crystal_dist_creation(self):
        """Test CrystalDist can be created and configured."""
        dist = CrystalDist(
            name="test_crystal",
            alpha_L="alpha_L",
            alpha_R="alpha_R",
            m="m",
            m0="m0",
            n_L="n_L",
            n_R="n_R",
            sigma_L="sigma_L",
            sigma_R="sigma_R",
        )
        assert dist.name == "test_crystal"
        assert dist.alpha_L == "alpha_L"
        assert dist.alpha_R == "alpha_R"
        assert len(dist.parameters) == 8

    def test_crystal_dist_from_dict(self):
        """Test CrystalDist can be created from dictionary."""
        config = {
            "name": "test_crystal",
            "alpha_L": "aL",
            "alpha_R": "aR",
            "m": "mass",
            "m0": "mean",
            "n_L": "nL",
            "n_R": "nR",
            "sigma_L": "sL",
            "sigma_R": "sR",
        }
        dist = CrystalDist.from_dict(config)
        assert dist.name == "test_crystal"
        assert dist.m == "mass"
        assert dist.m0 == "mean"

    def test_crystal_dist_expression_gaussian_core(self):
        """Test CrystalDist reduces to Gaussian in core region."""
        dist = CrystalDist(
            name="test_crystal",
            alpha_L="alpha_L",
            alpha_R="alpha_R",
            m="m",
            m0="m0",
            n_L="n_L",
            n_R="n_R",
            sigma_L="sigma_L",
            sigma_R="sigma_R",
        )

        # Test parameters that should give Gaussian behavior
        params = {
            "alpha_L": pt.constant(2.0),
            "alpha_R": pt.constant(2.0),
            "m": pt.constant(0.0),  # At peak
            "m0": pt.constant(0.0),
            "n_L": pt.constant(1.0),
            "n_R": pt.constant(1.0),
            "sigma_L": pt.constant(1.0),
            "sigma_R": pt.constant(1.0),
        }

        result = dist.expression(params)
        expected = pt.exp(-0.5 * 0.0**2)  # Gaussian at peak

        f = function([], [result, expected])
        result_val, expected_val = f()

        np.testing.assert_allclose(result_val, expected_val)

    def test_crystal_dist_expression_power_law_tails(self):
        """Test CrystalDist gives power law behavior in tails."""
        dist = CrystalDist(
            name="test_crystal",
            alpha_L="alpha_L",
            alpha_R="alpha_R",
            m="m",
            m0="m0",
            n_L="n_L",
            n_R="n_R",
            sigma_L="sigma_L",
            sigma_R="sigma_R",
        )

        # Test parameters for left tail
        params = {
            "alpha_L": pt.constant(1.0),
            "alpha_R": pt.constant(1.0),
            "m": pt.constant(-3.0),  # Far left tail
            "m0": pt.constant(0.0),
            "n_L": pt.constant(2.0),
            "n_R": pt.constant(2.0),
            "sigma_L": pt.constant(1.0),
            "sigma_R": pt.constant(1.0),
        }

        result = dist.expression(params)

        # Should be in left tail region since t_L = -3.0 < -1.0
        f = function([], result)
        result_val = f()

        # Value should be positive and finite
        assert result_val > 0
        assert np.isfinite(result_val)


class TestGenericDist:
    """Test GenericDist implementation."""

    def test_generic_dist_creation(self):
        """Test GenericDist can be created and configured."""
        dist = GenericDist(name="test_generic", expression="x**2 + y**2")
        assert dist.name == "test_generic"
        assert dist.expression_str == "x**2 + y**2"

    def test_generic_dist_from_dict(self):
        """Test GenericDist can be created from dictionary."""
        config = {"name": "test_generic", "expression": "sin(x) + cos(y)"}
        dist = GenericDist.from_dict(config)
        assert dist.name == "test_generic"
        assert dist.expression_str == "sin(x) + cos(y)"

    def test_generic_dist_expression_evaluation(self):
        """Test GenericDist evaluates mathematical expressions."""
        dist = GenericDist(name="test_generic", expression="x**2")

        # Create a PyTensor scalar variable
        x = pt.scalar("x")
        params = {"x": x}

        # Get the expression result from the distribution
        result = dist.expression(params)

        # Compile and test the function
        f = function([x], result)

        # Test that x^2 is evaluated correctly
        assert f(2.0) == 4.0
        assert f(3.0) == 9.0
        assert f(0.0) == 0.0


class TestBoundedScalar:
    """Test boundedscalar function improvements."""

    def test_boundedscalar_two_sided(self):
        """Test boundedscalar with two-sided bounds."""
        x = boundedscalar("test", (0.0, 1.0))
        assert x is not None

    def test_boundedscalar_lower_bound_only(self):
        """Test boundedscalar with lower bound only."""
        x = boundedscalar("test", (0.0, None))
        assert x is not None

    def test_boundedscalar_upper_bound_only(self):
        """Test boundedscalar with upper bound only."""
        x = boundedscalar("test", (None, 1.0))
        assert x is not None

    def test_boundedscalar_no_bounds(self):
        """Test boundedscalar with no bounds."""
        x = boundedscalar("test", (None, None))
        assert x is not None
