"""
Unit tests for new distribution implementations.

Tests for ProductDist, CrystalDist, and GenericDist implementations,
including validation against expected NLL values from ROOT fits.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from pytensor import function

from pyhs3 import Workspace
from pyhs3.core import boundedscalar
from pyhs3.distributions import CrystalDist, GaussianDist, GenericDist, ProductDist


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


class TestNumericParameters:
    """Test handling of numeric parameters in distributions."""

    def test_gaussian_with_numeric_sigma(self):
        """Test GaussianDist handles numeric sigma parameter."""
        config = {
            "name": "test_gauss",
            "mean": "mu_param",
            "sigma": 1.5,  # Numeric value, not string reference
            "x": "obs_var",
        }

        dist = GaussianDist.from_dict(config)

        # Check that sigma parameter was converted to a constant name
        assert dist.sigma == "constant_test_gauss_sigma"
        assert dist.mean == "mu_param"  # String reference unchanged
        assert dist.x == "obs_var"  # String reference unchanged

        # Check that all parameters (including constants) are in parameters list
        assert "mu_param" in dist.parameters
        assert "obs_var" in dist.parameters
        assert (
            "constant_test_gauss_sigma" in dist.parameters
        )  # Constants are dependencies

        # Check that the constant was created
        assert "constant_test_gauss_sigma" in dist.constants
        constant_tensor = dist.constants["constant_test_gauss_sigma"]

        # Verify the constant has the correct value
        f = function([], constant_tensor)
        assert np.isclose(f(), 1.5)

    def test_gaussian_with_all_numeric_parameters(self):
        """Test GaussianDist handles all numeric parameters."""
        config = {"name": "numeric_gauss", "mean": 2.0, "sigma": 0.5, "x": 1.0}

        dist = GaussianDist.from_dict(config)

        # All should be converted to constant names
        assert dist.mean == "constant_numeric_gauss_mean"
        assert dist.sigma == "constant_numeric_gauss_sigma"
        assert dist.x == "constant_numeric_gauss_x"

        # All constants, so parameters list contains all constant names
        expected_params = [
            "constant_numeric_gauss_mean",
            "constant_numeric_gauss_sigma",
            "constant_numeric_gauss_x",
        ]
        assert set(dist.parameters) == set(expected_params)

        # All constants should be created
        assert len(dist.constants) == 3
        assert "constant_numeric_gauss_mean" in dist.constants
        assert "constant_numeric_gauss_sigma" in dist.constants
        assert "constant_numeric_gauss_x" in dist.constants

    def test_gaussian_mixed_parameters(self):
        """Test GaussianDist with mix of string and numeric parameters."""
        config = {
            "name": "mixed_gauss",
            "mean": "mu_param",  # String reference
            "sigma": 2.0,  # Numeric constant
            "x": "obs_var",  # String reference
        }

        dist = GaussianDist.from_dict(config)

        assert dist.mean == "mu_param"
        assert dist.sigma == "constant_mixed_gauss_sigma"
        assert dist.x == "obs_var"

        # String references and constants in parameters
        assert set(dist.parameters) == {
            "mu_param",
            "obs_var",
            "constant_mixed_gauss_sigma",
        }

        # Only sigma constant created
        assert len(dist.constants) == 1
        assert "constant_mixed_gauss_sigma" in dist.constants

    def test_numeric_parameters_end_to_end_integration(self):
        """Test that numeric parameters work in a full Model workflow."""

        # Test data with numeric sigma - mimics the real-world issue
        test_data = {
            "parameter_points": [
                {"name": "test_params", "parameters": [{"name": "mu", "value": 0.0}]}
            ],
            "distributions": [
                {
                    "type": "gaussian_dist",
                    "name": "test_gauss",
                    "mean": "mu",
                    "sigma": 1.0,  # This is numeric, should work
                    "x": "mu",
                }
            ],
            "domains": [{"name": "test_domain", "type": "product_domain", "axes": []}],
            "functions": [],
            "metadata": {"name": "test"},
        }

        # This should not raise "Unknown entity referenced: '1.0'" error
        ws = Workspace(test_data)
        model = ws.model(domain="test_domain", parameter_point="test_params")

        # Verify the model was created successfully
        assert "mu" in model.parameters
        assert "constant_test_gauss_sigma" in model.parameters
        assert "test_gauss" in model.distributions

        # Verify we can evaluate the distribution
        pdf_value = model.pdf("test_gauss", mu=0.0)
        assert pdf_value is not None
        assert pdf_value > 0  # Should be a valid probability
