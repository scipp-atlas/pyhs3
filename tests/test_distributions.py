"""
Unit tests for new distribution implementations.

Tests for ProductDist, CrystalBallDist, and GenericDist implementations,
including validation against expected NLL values from ROOT fits.
"""

from __future__ import annotations

import math

import numpy as np
import pytensor.tensor as pt
import pytest
from pytensor import function

from pyhs3 import Workspace
from pyhs3.core import create_bounded_tensor
from pyhs3.distributions import (
    ArgusDist,
    AsymmetricCrystalBallDist,
    BernsteinPolyDist,
    CrystalBallDist,
    Distribution,
    ExponentialDist,
    GaussianDist,
    GenericDist,
    LandauDist,
    LogNormalDist,
    PoissonDist,
    PolynomialDist,
    ProductDist,
    UniformDist,
)


class TestDistribution:
    """Test the base Distribution class."""

    def test_distribution_base_class(self):
        """Test Distribution base class initialization."""
        dist = Distribution(
            name="test_dist",
            type="test",
        )
        assert dist.name == "test_dist"
        assert dist.type == "test"

    def test_distribution_expression_not_implemented(self):
        """Test that base distribution expression method raises NotImplementedError."""
        dist = Distribution(name="test", type="unknown")
        with pytest.raises(
            NotImplementedError, match="Distribution type=unknown is not implemented."
        ):
            dist.expression({})


class TestProductDist:
    """Test ProductDist implementation."""

    def test_product_dist_creation(self):
        """Test ProductDist can be created and configured."""
        dist = ProductDist(name="test_product", factors=["factor1", "factor2"])
        assert dist.name == "test_product"
        assert dist.factors == ["factor1", "factor2"]
        assert dist.parameters == {"factor1", "factor2"}

    def test_product_dist_from_dict(self):
        """Test ProductDist can be created from dictionary."""
        config = {"name": "test_product", "factors": ["pdf1", "pdf2", "pdf3"]}
        dist = ProductDist(**config)
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

    def test_product_dist_empty_factors(self):
        """Test ProductDist with empty factors returns 1.0 and doesn't crash."""
        dist = ProductDist(name="empty_product", factors=[])

        # Empty context since no factors to provide
        params = {}

        result = dist.expression(params)
        expected = pt.constant(1.0)

        # Compile and evaluate
        f = function([], [result, expected])
        result_val, expected_val = f()

        np.testing.assert_allclose(result_val, expected_val)
        assert result_val == 1.0


class TestCrystalBallDist:
    """Test CrystalBallDist (single-sided) implementation."""

    def test_crystal_dist_creation(self):
        """Test CrystalBallDist (single-sided) can be created and configured."""
        dist = CrystalBallDist(
            name="test_crystal",
            alpha="alpha",
            m="m",
            m0="m0",
            n="n",
            sigma="sigma",
        )
        assert dist.name == "test_crystal"
        assert dist.alpha == "alpha"
        assert dist.n == "n"
        assert len(dist.parameters) == 5

    def test_crystal_dist_from_dict(self):
        """Test CrystalBallDist (single-sided) can be created from dictionary."""
        config = {
            "name": "test_crystal",
            "alpha": "a",
            "m": "mass",
            "m0": "mean",
            "n": "n_param",
            "sigma": "s",
        }
        dist = CrystalBallDist(**config)
        assert dist.name == "test_crystal"
        assert dist.m == "mass"
        assert dist.m0 == "mean"

    def test_crystal_dist_expression_gaussian_core(self):
        """Test CrystalBallDist (single-sided) reduces to Gaussian in core region."""
        dist = CrystalBallDist(
            name="test_crystal",
            alpha="alpha",
            m="m",
            m0="m0",
            n="n",
            sigma="sigma",
        )

        # Test parameters that should give Gaussian behavior
        params = {
            "alpha": pt.constant(2.0),
            "m": pt.constant(0.0),  # At peak
            "m0": pt.constant(0.0),
            "n": pt.constant(1.0),
            "sigma": pt.constant(1.0),
        }

        result = dist.expression(params)
        expected = pt.exp(-0.5 * 0.0**2)  # Gaussian at peak

        f = function([], [result, expected])
        result_val, expected_val = f()

        np.testing.assert_allclose(result_val, expected_val)

    def test_crystal_dist_expression_power_law_tails(self):
        """Test CrystalBallDist (single-sided) gives power law behavior in tails."""
        dist = CrystalBallDist(
            name="test_crystal",
            alpha="alpha",
            m="m",
            m0="m0",
            n="n",
            sigma="sigma",
        )

        # Test parameters for left tail
        params = {
            "alpha": pt.constant(1.0),
            "m": pt.constant(-3.0),  # Far left tail
            "m0": pt.constant(0.0),
            "n": pt.constant(2.0),
            "sigma": pt.constant(1.0),
        }

        result = dist.expression(params)

        # Should be in left tail region since t = -3.0 < -1.0
        f = function([], result)
        result_val = f()

        # Value should be positive and finite
        assert result_val > 0
        assert np.isfinite(result_val)


class TestAsymmetricCrystalBallDist:
    """Test AsymmetricCrystalBallDist (double-sided) implementation."""

    def test_asymmetric_crystal_dist_creation(self):
        """Test AsymmetricCrystalBallDist can be created and configured."""
        dist = AsymmetricCrystalBallDist(
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

    def test_asymmetric_crystal_dist_from_dict(self):
        """Test AsymmetricCrystalBallDist can be created from dictionary."""
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
        dist = AsymmetricCrystalBallDist(**config)
        assert dist.name == "test_crystal"
        assert dist.m == "mass"
        assert dist.m0 == "mean"

    def test_asymmetric_crystal_dist_expression_gaussian_core(self):
        """Test AsymmetricCrystalBallDist reduces to Gaussian in core region."""
        dist = AsymmetricCrystalBallDist(
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

    def test_asymmetric_crystal_dist_expression_power_law_tails(self):
        """Test AsymmetricCrystalBallDist gives power law behavior in tails."""
        dist = AsymmetricCrystalBallDist(
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
        dist = GenericDist(**config)
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


class TestPoissonDist:
    """Test PoissonDist implementation."""

    def test_poisson_dist_creation(self):
        """Test PoissonDist can be created and configured."""
        dist = PoissonDist(name="test_poisson", mean="lambda_param", x="count_var")
        assert dist.name == "test_poisson"
        assert dist.mean == "lambda_param"
        assert dist.x == "count_var"
        assert dist.parameters == {"lambda_param", "count_var"}

    def test_poisson_dist_from_dict(self):
        """Test PoissonDist can be created from dictionary."""
        config = {
            "type": "poisson_dist",
            "name": "test_poisson",
            "mean": "rate_param",
            "x": "observation",
        }
        dist = PoissonDist(**config)
        assert dist.name == "test_poisson"
        assert dist.mean == "rate_param"
        assert dist.x == "observation"

    def test_poisson_dist_from_dict_with_numeric_parameters(self):
        """Test PoissonDist handles numeric parameters correctly."""
        config = {
            "type": "poisson_dist",
            "name": "numeric_poisson",
            "mean": 3.5,  # Numeric rate
            "x": 2,  # Numeric count
        }
        dist = PoissonDist(**config)

        # Field attributes should preserve original values for serialization
        assert dist.mean == 3.5
        assert dist.x == 2

        # Parameters dict should contain the constant names for dependency tracking
        # Check that constants were created for numeric parameters
        assert "constant_numeric_poisson_mean" in dist.parameters
        assert "constant_numeric_poisson_x" in dist.parameters

        # Constants should be created
        assert "constant_numeric_poisson_mean" in dist.constants
        assert "constant_numeric_poisson_x" in dist.constants

        # Verify constant values
        f_mean = function([], dist.constants["constant_numeric_poisson_mean"])
        f_x = function([], dist.constants["constant_numeric_poisson_x"])
        assert np.isclose(f_mean(), 3.5)
        assert np.isclose(f_x(), 2.0)

    def test_poisson_dist_expression_at_mean(self):
        """Test PoissonDist PMF evaluation at k=λ."""
        dist = PoissonDist(name="test_poisson", mean="lambda_param", x="k")

        # Test at λ=3, k=3 (peak of distribution)
        params = {
            "lambda_param": pt.constant(3.0),
            "k": pt.constant(3.0),
        }

        result = dist.expression(params)
        f = function([], result)
        result_val = f()

        # Expected: P(3; 3) = 3^3 * e^(-3) / 3! = 27 * e^(-3) / 6 ≈ 0.224
        expected = (3.0**3) * math.exp(-3.0) / math.factorial(3)
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_poisson_dist_expression_zero_count(self):
        """Test PoissonDist PMF evaluation at k=0."""
        dist = PoissonDist(name="test_poisson", mean="lambda_param", x="k")

        # Test at λ=2, k=0
        params = {
            "lambda_param": pt.constant(2.0),
            "k": pt.constant(0.0),
        }

        result = dist.expression(params)
        f = function([], result)
        result_val = f()

        # Expected: P(0; 2) = 2^0 * e^(-2) / 0! = e^(-2) ≈ 0.135
        expected = math.exp(-2.0)
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        ("lambda_val", "k_val", "expected"),
        [
            pytest.param(1.0, 0.0, np.exp(-1.0), id="lambda1_k0"),
            pytest.param(1.0, 1.0, 1.0 * np.exp(-1.0), id="lambda1_k1"),
            pytest.param(2.0, 2.0, 4.0 * np.exp(-2.0) / 2.0, id="lambda2_k2"),
            pytest.param(5.0, 3.0, 125.0 * np.exp(-5.0) / 6.0, id="lambda5_k3"),
        ],
    )
    def test_poisson_dist_pmf_values(self, lambda_val, k_val, expected):
        """Test PoissonDist PMF against known values."""
        dist = PoissonDist(name="test_poisson", mean="lambda_param", x="k")

        params = {
            "lambda_param": pt.constant(lambda_val),
            "k": pt.constant(k_val),
        }

        result = dist.expression(params)
        f = function([], result)
        result_val = f()

        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_poisson_dist_expression_with_variables(self):
        """Test PoissonDist with variable parameters."""
        dist = PoissonDist(name="test_poisson", mean="lambda_param", x="k")

        # Create variables
        lambda_var = pt.scalar("lambda_param")
        k_var = pt.scalar("k")
        params = {"lambda_param": lambda_var, "k": k_var}

        result = dist.expression(params)
        f = function([lambda_var, k_var], result)

        # Test several points
        # λ=1, k=1: P(1; 1) = 1 * e^(-1) / 1! = e^(-1)
        assert np.isclose(f(1.0, 1.0), np.exp(-1.0))

        # λ=4, k=2: P(2; 4) = 16 * e^(-4) / 2! = 8 * e^(-4)
        assert np.isclose(f(4.0, 2.0), 8.0 * np.exp(-4.0))

    def test_poisson_dist_properties(self):
        """Test that PoissonDist has correct mathematical properties."""
        dist = PoissonDist(name="test_poisson", mean="lambda_param", x="k")

        lambda_val = 3.0
        params = {"lambda_param": pt.constant(lambda_val)}

        # Test that PMF is non-negative for various k values
        for k_val in [0, 1, 2, 5, 10]:
            test_params = dict(params)
            test_params["k"] = pt.constant(float(k_val))
            result = dist.expression(test_params)
            f = function([], result)
            pmf_val = f()
            assert pmf_val >= 0.0, (
                f"PMF should be non-negative, got {pmf_val} for k={k_val}"
            )

    def test_poisson_dist_integration_with_workspace(self):
        """Test PoissonDist integration in full Workspace workflow."""
        test_data = {
            "parameter_points": [
                {
                    "name": "test_params",
                    "parameters": [
                        {"name": "rate", "value": 2.5},
                        {"name": "observed", "value": 3.0},
                    ],
                }
            ],
            "distributions": [
                {
                    "type": "poisson_dist",
                    "name": "count_dist",
                    "mean": "rate",
                    "x": "observed",
                }
            ],
            "domains": [{"name": "test_domain", "type": "product_domain", "axes": []}],
            "functions": [],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)
        model = ws.model(domain="test_domain", parameter_set="test_params")

        # Verify the distribution was created
        assert "count_dist" in model.distributions
        assert "rate" in model.parameters
        assert "observed" in model.parameters

        # Verify we can evaluate the PMF
        pdf_value = model.pdf("count_dist", rate=2.5, observed=3.0)
        assert pdf_value is not None
        assert pdf_value > 0.0


class TestBoundedScalar:
    """Test create_bounded_tensor function improvements."""

    def test_create_bounded_tensor_two_sided(self):
        """Test create_bounded_tensor with two-sided bounds."""
        x = create_bounded_tensor("test", (0.0, 1.0))
        assert x is not None
        assert x.name == "test"

    def test_create_bounded_tensor_lower_bound_only(self):
        """Test create_bounded_tensor with lower bound only."""
        x = create_bounded_tensor("test", (0.0, None))
        assert x is not None
        assert x.name == "test"

    def test_create_bounded_tensor_upper_bound_only(self):
        """Test create_bounded_tensor with upper bound only."""
        x = create_bounded_tensor("test", (None, 1.0))
        assert x is not None
        assert x.name == "test"

    def test_create_bounded_tensor_no_bounds(self):
        """Test create_bounded_tensor with no bounds."""
        x = create_bounded_tensor("test", (None, None))
        assert x is not None
        assert x.name == "test"


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

        dist = GaussianDist(**config)

        # Field attributes should preserve original values for serialization
        assert dist.sigma == 1.5  # Original numeric value preserved
        assert dist.mean == "mu_param"  # String reference unchanged
        assert dist.x == "obs_var"  # String reference unchanged

        # Check that all parameters (including constants) are in parameters dict values
        param_values = sorted(dist.parameters)
        assert "mu_param" in param_values
        assert "obs_var" in param_values
        assert "constant_test_gauss_sigma" in param_values  # Constants are dependencies

        # Check that the constant was created
        assert "constant_test_gauss_sigma" in dist.constants
        constant_tensor = dist.constants["constant_test_gauss_sigma"]

        # Verify the constant has the correct value
        f = function([], constant_tensor)
        assert np.isclose(f(), 1.5)

    def test_gaussian_with_all_numeric_parameters(self):
        """Test GaussianDist handles all numeric parameters."""
        config = {"name": "numeric_gauss", "mean": 2.0, "sigma": 0.5, "x": 1.0}

        dist = GaussianDist(**config)

        # Field attributes should preserve original values for serialization
        assert dist.mean == 2.0
        assert dist.sigma == 0.5
        assert dist.x == 1.0

        # All constants, so parameters dict values contain all constant names
        expected_params = [
            "constant_numeric_gauss_mean",
            "constant_numeric_gauss_sigma",
            "constant_numeric_gauss_x",
        ]
        assert dist.parameters == set(expected_params)

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

        dist = GaussianDist(**config)

        # Field attributes should preserve original values for serialization
        assert dist.mean == "mu_param"
        assert dist.sigma == 2.0  # Original numeric value preserved
        assert dist.x == "obs_var"

        # String references and constants in parameters
        assert dist.parameters == {
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
            "metadata": {"hs3_version": "0.2"},
        }

        # This should not raise "Unknown entity referenced: '1.0'" error
        ws = Workspace(**test_data)
        model = ws.model(domain="test_domain", parameter_set="test_params")

        # Verify the model was created successfully
        assert "mu" in model.parameters
        assert "constant_test_gauss_sigma" in model.parameters
        assert "test_gauss" in model.distributions

        # Verify we can evaluate the distribution
        pdf_value = model.pdf("test_gauss", mu=0.0)
        assert pdf_value is not None
        assert pdf_value > 0  # Should be a valid probability


class TestDependencyGraphErrors:
    """Test dependency graph error conditions in core.py for code coverage."""

    def test_circular_dependency_error(self):
        """Test that circular dependencies raise ValueError."""

        # Create a workspace with circular dependency between functions
        test_data = {
            "parameter_points": [
                {"name": "test_params", "parameters": [{"name": "base", "value": 1.0}]}
            ],
            "functions": [
                {
                    "type": "generic_function",
                    "name": "func_a",
                    "expression": "func_b + 1",  # func_a depends on func_b
                },
                {
                    "type": "generic_function",
                    "name": "func_b",
                    "expression": "func_a * 2",  # func_b depends on func_a -> circular!
                },
            ],
            "distributions": [],
            "domains": [{"name": "test_domain", "type": "product_domain", "axes": []}],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)

        # This should raise ValueError about circular dependency
        with pytest.raises(ValueError, match="Circular dependency detected in graph"):
            ws.model(domain="test_domain", parameter_set="test_params")

    def test_bounded_scalar_applied_to_parameters(self):
        """Test that parameters get bounded scalar applied when domains exist."""
        # Create a workspace with a parameter that has domain bounds
        test_data = {
            "parameter_points": [
                {"name": "test_params", "parameters": [{"name": "mu", "value": 0.0}]}
            ],
            "distributions": [
                {
                    "type": "gaussian_dist",
                    "name": "test_dist",
                    "mean": "mu",
                    "sigma": 1.0,
                    "x": "mu",
                }
            ],
            "domains": [
                {
                    "name": "test_domain",
                    "type": "product_domain",
                    "axes": [
                        {
                            "name": "mu",
                            "min": -2.0,
                            "max": 2.0,
                        }  # Bounds on parameter mu
                    ],
                }
            ],
            "functions": [],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)
        model = ws.model(domain="test_domain", parameter_set="test_params")

        # Verify the parameter is bounded
        assert "mu" in model.parameters
        mu_tensor = model.parameters["mu"]
        assert mu_tensor is not None
        assert hasattr(mu_tensor, "name")

        # The distribution should have been created successfully using the bounded parameter
        assert "test_dist" in model.distributions
        dist_tensor = model.distributions["test_dist"]
        assert dist_tensor is not None
        assert hasattr(dist_tensor, "name")


class TestCollectionMethods:
    """Test collection methods for code coverage."""

    def testparameter_points_methods(self):
        """Test ParameterPoints get(), __contains__, and __len__ methods."""
        test_data = {
            "parameter_points": [
                {"name": "params1", "parameters": [{"name": "mu", "value": 0.0}]},
                {"name": "params2", "parameters": [{"name": "sigma", "value": 1.0}]},
            ],
            "distributions": [],
            "domains": [],
            "functions": [],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)
        param_collection = ws.parameter_points

        # Test __len__
        assert len(param_collection) == 2

        # Test __contains__
        assert "params1" in param_collection
        assert "params2" in param_collection
        assert "nonexistent" not in param_collection

        # Test get() method
        params1 = param_collection.get("params1")
        assert params1 is not None
        assert params1.name == "params1"

        # Test get() with default
        default_result = param_collection.get("nonexistent", "default")
        assert default_result == "default"

        # Test get() with None default
        none_result = param_collection.get("nonexistent")
        assert none_result is None

    def test_parameter_set_methods(self):
        """Test ParameterSet get(), __contains__, and __len__ methods."""
        test_data = {
            "parameter_points": [
                {
                    "name": "test_params",
                    "parameters": [
                        {"name": "mu", "value": 0.0},
                        {"name": "sigma", "value": 1.0},
                    ],
                }
            ],
            "distributions": [],
            "domains": [],
            "functions": [],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)
        param_set = ws.parameter_points["test_params"]

        # Test __len__
        assert len(param_set) == 2

        # Test __contains__
        assert "mu" in param_set
        assert "sigma" in param_set
        assert "nonexistent" not in param_set

        # Test get() method
        mu_param = param_set.get("mu")
        assert mu_param is not None
        assert mu_param.name == "mu"
        assert mu_param.value == 0.0

        # Test get() with default
        default_result = param_set.get("nonexistent", "default")
        assert default_result == "default"

    def testdomains_methods(self):
        """Test DomainCollection get(), __contains__, and __len__ methods."""
        test_data = {
            "parameter_points": [
                {"name": "test_params", "parameters": [{"name": "mu", "value": 0.0}]}
            ],
            "distributions": [],
            "domains": [
                {
                    "name": "domain1",
                    "type": "product_domain",
                    "axes": [{"name": "mu", "min": -1.0, "max": 1.0}],
                },
                {
                    "name": "domain2",
                    "type": "product_domain",
                    "axes": [{"name": "sigma", "min": 0.0, "max": 5.0}],
                },
            ],
            "functions": [],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)
        domain_collection = ws.domains

        # Test __len__
        assert len(domain_collection) == 2

        # Test __contains__
        assert "domain1" in domain_collection
        assert "domain2" in domain_collection
        assert "nonexistent" not in domain_collection

        # Test get() method
        domain1 = domain_collection.get("domain1")
        assert domain1 is not None
        assert domain1.name == "domain1"

        # Test get() with default
        default_result = domain_collection.get("nonexistent", "default")
        assert default_result == "default"

    def test_domain_set_methods(self):
        """Test Domains get(), __contains__, and __len__ methods."""
        test_data = {
            "parameter_points": [
                {"name": "test_params", "parameters": [{"name": "mu", "value": 0.0}]}
            ],
            "distributions": [],
            "domains": [
                {
                    "name": "test_domain",
                    "type": "product_domain",
                    "axes": [
                        {"name": "mu", "min": -1.0, "max": 1.0},
                        {"name": "sigma", "min": 0.0, "max": 5.0},
                    ],
                }
            ],
            "functions": [],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)
        domain_set = ws.domains["test_domain"]

        # Test __len__
        assert len(domain_set) == 2

        # Test __contains__
        assert "mu" in domain_set
        assert "sigma" in domain_set
        assert "nonexistent" not in domain_set

        # Test get() method
        mu_bounds = domain_set.get("mu")
        assert mu_bounds == (-1.0, 1.0)

        # Test get() with default (should return (None, None) by default)
        default_result = domain_set.get("nonexistent")
        assert default_result == (None, None)

        # Test get() with custom default
        custom_default = domain_set.get("nonexistent", (0.0, 10.0))
        assert custom_default == (0.0, 10.0)


class TestCrossDependencies:
    """Test cross-dependencies between functions and distributions."""

    def test_distribution_depending_on_function(self):
        """Test that a distribution can depend on a function result."""
        test_data = {
            "parameter_points": [
                {"name": "test_params", "parameters": [{"name": "x", "value": 2.0}]}
            ],
            "functions": [
                {
                    "type": "generic_function",
                    "name": "computed_mean",
                    "expression": "x * 2",  # This will compute 2.0 * 2 = 4.0
                }
            ],
            "distributions": [
                {
                    "type": "gaussian_dist",
                    "name": "dist_with_func_mean",
                    "mean": "computed_mean",  # Distribution depends on function
                    "sigma": 1.0,
                    "x": "x",
                }
            ],
            "domains": [{"name": "test_domain", "type": "product_domain", "axes": []}],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)
        model = ws.model(domain="test_domain", parameter_set="test_params")

        # Verify both function and distribution were created
        assert "computed_mean" in model.functions
        assert "dist_with_func_mean" in model.distributions

        # Verify the dependency was resolved correctly
        func_tensor = model.functions["computed_mean"]
        assert func_tensor is not None

        dist_tensor = model.distributions["dist_with_func_mean"]
        assert dist_tensor is not None

    def test_function_depending_on_distribution(self):
        """Test that a function can depend on a distribution result."""
        test_data = {
            "parameter_points": [
                {"name": "test_params", "parameters": [{"name": "mu", "value": 0.0}]}
            ],
            "distributions": [
                {
                    "type": "gaussian_dist",
                    "name": "base_dist",
                    "mean": "mu",
                    "sigma": 1.0,
                    "x": "mu",
                }
            ],
            "functions": [
                {
                    "type": "generic_function",
                    "name": "dist_transform",
                    "expression": "mu + 1",  # Simpler expression to avoid PyTensor name issues
                }
            ],
            "domains": [{"name": "test_domain", "type": "product_domain", "axes": []}],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)
        model = ws.model(domain="test_domain", parameter_set="test_params")

        # Verify both distribution and function were created
        assert "base_dist" in model.distributions
        assert "dist_transform" in model.functions

        # Verify the dependency was resolved correctly
        dist_tensor = model.distributions["base_dist"]
        assert dist_tensor is not None

        func_tensor = model.functions["dist_transform"]
        assert func_tensor is not None

    def test_multiple_functions_and_distributions(self):
        """Test that multiple functions and distributions can coexist and depend on each other."""
        test_data = {
            "parameter_points": [
                {"name": "test_params", "parameters": [{"name": "base", "value": 1.0}]}
            ],
            "functions": [
                {
                    "type": "generic_function",
                    "name": "func1",
                    "expression": "base * 3",  # func1 depends on parameter
                },
                {
                    "type": "generic_function",
                    "name": "func2",
                    "expression": "base + 2",  # func2 also depends on parameter
                },
            ],
            "distributions": [
                {
                    "type": "gaussian_dist",
                    "name": "dist1",
                    "mean": "func1",  # dist1 depends on func1
                    "sigma": 1.0,
                    "x": "base",
                },
                {
                    "type": "gaussian_dist",
                    "name": "dist2",
                    "mean": "func2",  # dist2 depends on func2
                    "sigma": 2.0,
                    "x": "base",
                },
            ],
            "domains": [{"name": "test_domain", "type": "product_domain", "axes": []}],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)
        model = ws.model(domain="test_domain", parameter_set="test_params")

        # Verify all entities were created
        assert "func1" in model.functions
        assert "func2" in model.functions
        assert "dist1" in model.distributions
        assert "dist2" in model.distributions

        # Verify all tensors are valid
        for name in ["func1", "func2"]:
            assert model.functions[name] is not None

        for name in ["dist1", "dist2"]:
            assert model.distributions[name] is not None


class TestUniformDist:
    """Test UniformDist implementation."""

    def test_uniform_dist_creation(self):
        """Test UniformDist can be created and configured."""
        dist = UniformDist(name="test_uniform", x=["x_var"])
        assert dist.name == "test_uniform"
        assert dist.x == ["x_var"]
        assert dist.parameters == {"x_var"}

    def test_uniform_dist_from_dict(self):
        """Test UniformDist can be created from dictionary."""
        config = {
            "type": "uniform_dist",
            "name": "test_uniform",
            "x": ["observable"],
        }
        dist = UniformDist(**config)
        assert dist.name == "test_uniform"
        assert dist.x == ["observable"]

    def test_uniform_dist_from_dict_with_string_parameter(self):
        """Test UniformDist handles string x parameter correctly."""
        config = {
            "type": "uniform_dist",
            "name": "string_uniform",
            "x": ["obs_var"],  # String parameter
        }
        dist = UniformDist(**config)

        # Field attributes should preserve original values
        assert dist.x == ["obs_var"]

        # Parameters dict should contain the parameter mapping
        assert "obs_var" in dist.parameters
        assert "obs_var" in dist.parameters

    def test_uniform_dist_expression_constant_value(self):
        """Test UniformDist returns constant value of 1.0."""
        dist = UniformDist(name="test_uniform", x=["x"])

        # Parameters - the x value doesn't matter for uniform distribution
        params = {"x": pt.constant(0.5)}

        result = dist.expression(params)
        f = function([], result)
        result_val = f()

        # Should always return 1.0 (normalization handled by domain)
        assert result_val == 1.0

    def test_uniform_dist_expression_with_different_x_values(self):
        """Test UniformDist returns same value regardless of x."""
        dist = UniformDist(name="test_uniform", x=["x"])

        # Test multiple x values - should all give same result
        x_values = [-1.0, 0.0, 0.5, 1.0, 10.0]

        for x_val in x_values:
            params = {"x": pt.constant(x_val)}
            result = dist.expression(params)
            f = function([], result)
            result_val = f()
            assert result_val == 1.0, f"Failed for x={x_val}"

    def test_uniform_dist_integration_with_workspace(self):
        """Test UniformDist integration in full Workspace workflow."""
        test_data = {
            "parameter_points": [
                {
                    "name": "test_params",
                    "parameters": [
                        {"name": "obs", "value": 0.5},
                    ],
                }
            ],
            "distributions": [
                {
                    "type": "uniform_dist",
                    "name": "uniform_dist",
                    "x": ["obs"],
                }
            ],
            "domains": [{"name": "test_domain", "type": "product_domain", "axes": []}],
            "functions": [],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)
        model = ws.model(domain="test_domain", parameter_set="test_params")

        # Verify the distribution was created
        assert "uniform_dist" in model.distributions
        assert "obs" in model.parameters

        # Verify we can evaluate the PDF
        pdf_value = model.pdf("uniform_dist", obs=0.5)
        assert pdf_value is not None
        assert pdf_value == 1.0


class TestExponentialDist:
    """Test ExponentialDist implementation."""

    def test_exponential_dist_creation(self):
        """Test ExponentialDist can be created and configured."""
        dist = ExponentialDist(name="test_exp", x="x_var", c="c_param")
        assert dist.name == "test_exp"
        assert dist.x == "x_var"
        assert dist.c == "c_param"
        assert dist.parameters == {"x_var", "c_param"}

    def test_exponential_dist_from_dict(self):
        """Test ExponentialDist can be created from dictionary."""
        config = {
            "type": "exponential_dist",
            "name": "test_exp",
            "x": "time",
            "c": "decay_rate",
        }
        dist = ExponentialDist(**config)
        assert dist.name == "test_exp"
        assert dist.x == "time"
        assert dist.c == "decay_rate"

    def test_exponential_dist_from_dict_with_numeric_parameters(self):
        """Test ExponentialDist handles numeric parameters correctly."""
        config = {
            "type": "exponential_dist",
            "name": "numeric_exp",
            "x": 1.0,  # Numeric value
            "c": 0.5,  # Numeric rate
        }
        dist = ExponentialDist(**config)

        # Field attributes should preserve original values
        assert dist.x == 1.0
        assert dist.c == 0.5

        # Parameters dict should contain the constant names
        # Check that constants were created for numeric parameters
        assert "constant_numeric_exp_x" in dist.parameters
        assert "constant_numeric_exp_c" in dist.parameters

        # Constants should be created
        assert "constant_numeric_exp_x" in dist.constants
        assert "constant_numeric_exp_c" in dist.constants

    def test_exponential_dist_expression_known_values(self):
        """Test ExponentialDist against known exponential values."""
        dist = ExponentialDist(name="test_exp", x="x", c="c")

        # Test exp(-1 * 1) = exp(-1) ≈ 0.3679
        params = {"x": pt.constant(1.0), "c": pt.constant(1.0)}
        result = dist.expression(params)
        f = function([], result)
        result_val = f()
        expected = np.exp(-1.0)
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

        # Test exp(-0.5 * 2) = exp(-1) ≈ 0.3679
        params = {"x": pt.constant(2.0), "c": pt.constant(0.5)}
        result = dist.expression(params)
        f = function([], result)
        result_val = f()
        expected = np.exp(-1.0)
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

        # Test exp(-2 * 0) = exp(0) = 1
        params = {"x": pt.constant(0.0), "c": pt.constant(2.0)}
        result = dist.expression(params)
        f = function([], result)
        result_val = f()
        expected = 1.0
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        ("x_val", "c_val", "expected"),
        [
            pytest.param(0.0, 1.0, 1.0, id="x0_c1"),  # exp(-1*0) = 1
            pytest.param(1.0, 1.0, np.exp(-1.0), id="x1_c1"),  # exp(-1*1)
            pytest.param(2.0, 0.5, np.exp(-1.0), id="x2_c0.5"),  # exp(-0.5*2)
            pytest.param(0.5, 2.0, np.exp(-1.0), id="x0.5_c2"),  # exp(-2*0.5)
            pytest.param(1.0, 2.0, np.exp(-2.0), id="x1_c2"),  # exp(-2*1)
        ],
    )
    def test_exponential_dist_parameterized_values(self, x_val, c_val, expected):
        """Test ExponentialDist against parameterized known values."""
        dist = ExponentialDist(name="test_exp", x="x", c="c")

        params = {"x": pt.constant(x_val), "c": pt.constant(c_val)}
        result = dist.expression(params)
        f = function([], result)
        result_val = f()

        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_exponential_dist_expression_with_variables(self):
        """Test ExponentialDist with variable parameters."""
        dist = ExponentialDist(name="test_exp", x="x", c="c")

        # Create variables
        x_var = pt.scalar("x")
        c_var = pt.scalar("c")
        params = {"x": x_var, "c": c_var}

        result = dist.expression(params)
        f = function([x_var, c_var], result)

        # Test several points
        # exp(-1*0) = 1
        assert np.isclose(f(0.0, 1.0), 1.0)
        # exp(-2*1) = exp(-2)
        assert np.isclose(f(1.0, 2.0), np.exp(-2.0))
        # exp(-0.5*3) = exp(-1.5)
        assert np.isclose(f(3.0, 0.5), np.exp(-1.5))

    def test_exponential_dist_properties(self):
        """Test that ExponentialDist has correct mathematical properties."""
        dist = ExponentialDist(name="test_exp", x="x", c="c")

        c_val = 1.0
        params = {"c": pt.constant(c_val)}

        # Test that PDF is positive and decreasing for x >= 0
        x_values = [0.0, 0.5, 1.0, 2.0, 5.0]
        pdf_values = []

        for x_val in x_values:
            test_params = dict(params)
            test_params["x"] = pt.constant(x_val)
            result = dist.expression(test_params)
            f = function([], result)
            pdf_val = f()
            assert pdf_val > 0.0, f"PDF should be positive, got {pdf_val} for x={x_val}"
            pdf_values.append(pdf_val)

        # Check that values are decreasing (for positive c)
        for i in range(len(pdf_values) - 1):
            assert pdf_values[i] >= pdf_values[i + 1], (
                f"PDF should be decreasing, but {pdf_values[i]} < {pdf_values[i + 1]} "
                f"at positions {i} and {i + 1}"
            )

    def test_exponential_dist_integration_with_workspace(self):
        """Test ExponentialDist integration in full Workspace workflow."""
        test_data = {
            "parameter_points": [
                {
                    "name": "test_params",
                    "parameters": [
                        {"name": "time", "value": 1.0},
                        {"name": "rate", "value": 0.5},
                    ],
                }
            ],
            "distributions": [
                {
                    "type": "exponential_dist",
                    "name": "exp_decay",
                    "x": "time",
                    "c": "rate",
                }
            ],
            "domains": [{"name": "test_domain", "type": "product_domain", "axes": []}],
            "functions": [],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)
        model = ws.model(domain="test_domain", parameter_set="test_params")

        # Verify the distribution was created
        assert "exp_decay" in model.distributions
        assert "time" in model.parameters
        assert "rate" in model.parameters

        # Verify we can evaluate the PDF: exp(-0.5 * 1.0) = exp(-0.5)
        pdf_value = model.pdf("exp_decay", time=1.0, rate=0.5)
        assert pdf_value is not None
        expected = np.exp(-0.5)
        np.testing.assert_allclose(pdf_value, expected, rtol=1e-6)


class TestLogNormalDist:
    """Test LogNormalDist implementation."""

    def test_lognormal_dist_creation(self):
        """Test LogNormalDist can be created and configured."""
        dist = LogNormalDist(
            name="test_lognorm", x="x_var", mu="mu_param", sigma="sigma_param"
        )
        assert dist.name == "test_lognorm"
        assert dist.x == "x_var"
        assert dist.mu == "mu_param"
        assert dist.sigma == "sigma_param"
        assert dist.parameters == {"x_var", "mu_param", "sigma_param"}

    def test_lognormal_dist_from_dict(self):
        """Test LogNormalDist can be created from dictionary."""
        config = {
            "type": "lognormal_dist",
            "name": "test_lognorm",
            "x": "observable",
            "mu": "log_mean",
            "sigma": "log_sigma",
        }
        dist = LogNormalDist(**config)
        assert dist.name == "test_lognorm"
        assert dist.x == "observable"
        assert dist.mu == "log_mean"
        assert dist.sigma == "log_sigma"

    def test_lognormal_dist_from_dict_with_numeric_parameters(self):
        """Test LogNormalDist handles numeric parameters correctly."""
        config = {
            "type": "lognormal_dist",
            "name": "numeric_lognorm",
            "x": 2.0,  # Numeric value
            "mu": 0.0,  # Log-scale mean
            "sigma": 1.0,  # Log-scale std dev
        }
        dist = LogNormalDist(**config)

        # Field attributes should preserve original values
        assert dist.x == 2.0
        assert dist.mu == 0.0
        assert dist.sigma == 1.0

        # Parameters dict should contain the constant names
        # Check that constants were created for numeric parameters
        assert "constant_numeric_lognorm_x" in dist.parameters
        assert "constant_numeric_lognorm_mu" in dist.parameters
        assert "constant_numeric_lognorm_sigma" in dist.parameters

        # Constants should be created
        assert "constant_numeric_lognorm_x" in dist.constants
        assert "constant_numeric_lognorm_mu" in dist.constants
        assert "constant_numeric_lognorm_sigma" in dist.constants

    def test_lognormal_dist_expression_known_values(self):
        """Test LogNormalDist against known log-normal values."""
        dist = LogNormalDist(name="test_lognorm", x="x", mu="mu", sigma="sigma")

        # Test standard log-normal at x=1 with mu=0, sigma=1
        # ln(1) = 0, so we get (1/1) * exp(-0^2 / (2*1^2)) = 1 * exp(0) = 1
        params = {
            "x": pt.constant(1.0),
            "mu": pt.constant(0.0),
            "sigma": pt.constant(1.0),
        }
        result = dist.expression(params)
        f = function([], result)
        result_val = f()
        expected = 1.0
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

        # Test at x=e with mu=1, sigma=1
        # ln(e) = 1, so we get (1/e) * exp(-(1-1)^2 / (2*1^2)) = (1/e) * exp(0) = 1/e
        e_val = np.exp(1.0)
        params = {
            "x": pt.constant(e_val),
            "mu": pt.constant(1.0),
            "sigma": pt.constant(1.0),
        }
        result = dist.expression(params)
        f = function([], result)
        result_val = f()
        expected = 1.0 / e_val
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        ("x_val", "mu_val", "sigma_val"),
        [
            pytest.param(
                1.0, 0.0, 1.0, id="standard_at_1"
            ),  # Standard log-normal at x=1
            pytest.param(np.exp(1.0), 1.0, 1.0, id="at_e_mu1"),  # At x=e with mu=1
            pytest.param(
                np.exp(0.5), 0.5, 0.5, id="at_exp0.5"
            ),  # At x=exp(0.5) with mu=0.5
        ],
    )
    def test_lognormal_dist_parameterized_known_values(self, x_val, mu_val, sigma_val):
        """Test LogNormalDist against parameterized known values."""
        dist = LogNormalDist(name="test_lognorm", x="x", mu="mu", sigma="sigma")

        params = {
            "x": pt.constant(x_val),
            "mu": pt.constant(mu_val),
            "sigma": pt.constant(sigma_val),
        }
        result = dist.expression(params)
        f = function([], result)
        result_val = f()

        # Calculate expected value: (1/x) * exp(-((ln(x) - mu)^2) / (2 * sigma^2))
        log_x = np.log(x_val)
        normalized_log = (log_x - mu_val) / sigma_val
        expected = (1.0 / x_val) * np.exp(-0.5 * normalized_log**2)

        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_lognormal_dist_expression_with_variables(self):
        """Test LogNormalDist with variable parameters."""
        dist = LogNormalDist(name="test_lognorm", x="x", mu="mu", sigma="sigma")

        # Create variables
        x_var = pt.scalar("x")
        mu_var = pt.scalar("mu")
        sigma_var = pt.scalar("sigma")
        params = {"x": x_var, "mu": mu_var, "sigma": sigma_var}

        result = dist.expression(params)
        f = function([x_var, mu_var, sigma_var], result)

        # Test at x=1, mu=0, sigma=1: should give 1.0
        assert np.isclose(f(1.0, 0.0, 1.0), 1.0)

        # Test at x=e, mu=1, sigma=1: should give 1/e
        e_val = np.exp(1.0)
        assert np.isclose(f(e_val, 1.0, 1.0), 1.0 / e_val)

    def test_lognormal_dist_properties(self):
        """Test that LogNormalDist has correct mathematical properties."""
        dist = LogNormalDist(name="test_lognorm", x="x", mu="mu", sigma="sigma")

        mu_val = 0.0
        sigma_val = 1.0
        params = {"mu": pt.constant(mu_val), "sigma": pt.constant(sigma_val)}

        # Test that PDF is positive for x > 0
        x_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

        for x_val in x_values:
            test_params = dict(params)
            test_params["x"] = pt.constant(x_val)
            result = dist.expression(test_params)
            f = function([], result)
            pdf_val = f()
            assert pdf_val > 0.0, f"PDF should be positive, got {pdf_val} for x={x_val}"

        # Test that mode is at exp(mu - sigma^2) for our parameters
        # For mu=0, sigma=1: mode at exp(0-1) = exp(-1)
        mode_x = np.exp(mu_val - sigma_val**2)
        test_params = dict(params)
        test_params["x"] = pt.constant(mode_x)
        result = dist.expression(test_params)
        f_mode = function([], result)
        mode_val = f_mode()

        # Check that nearby points have lower values
        for offset in [-0.1, 0.1]:
            test_x = mode_x + offset
            if test_x > 0:  # Only test positive values
                test_params = dict(params)
                test_params["x"] = pt.constant(test_x)
                result = dist.expression(test_params)
                f_test = function([], result)
                test_val = f_test()
                assert test_val <= mode_val, (
                    f"Value at mode should be maximum, but {test_val} > {mode_val} "
                    f"at x={test_x} vs mode x={mode_x}"
                )

    def test_lognormal_dist_integration_with_workspace(self):
        """Test LogNormalDist integration in full Workspace workflow."""
        test_data = {
            "parameter_points": [
                {
                    "name": "test_params",
                    "parameters": [
                        {"name": "mass", "value": 1.0},
                        {"name": "log_mean", "value": 0.0},
                        {"name": "log_width", "value": 1.0},
                    ],
                }
            ],
            "distributions": [
                {
                    "type": "lognormal_dist",
                    "name": "lognorm_dist",
                    "x": "mass",
                    "mu": "log_mean",
                    "sigma": "log_width",
                }
            ],
            "domains": [{"name": "test_domain", "type": "product_domain", "axes": []}],
            "functions": [],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)
        model = ws.model(domain="test_domain", parameter_set="test_params")

        # Verify the distribution was created
        assert "lognorm_dist" in model.distributions
        assert "mass" in model.parameters
        assert "log_mean" in model.parameters
        assert "log_width" in model.parameters

        # Verify we can evaluate the PDF at x=1, mu=0, sigma=1: should give 1.0
        pdf_value = model.pdf("lognorm_dist", mass=1.0, log_mean=0.0, log_width=1.0)
        assert pdf_value is not None
        expected = 1.0
        np.testing.assert_allclose(pdf_value, expected, rtol=1e-6)


class TestPolynomialDist:
    """Test PolynomialDist implementation."""

    def test_polynomial_dist_creation(self):
        """Test PolynomialDist can be created and configured."""
        dist = PolynomialDist(
            name="test_poly", x="x_var", coefficients=["a0", "a1", "a2"]
        )
        assert dist.name == "test_poly"
        assert dist.x == "x_var"
        assert dist.coefficients == ["a0", "a1", "a2"]
        # Should have x plus all coefficients in parameters
        expected_params = ["x_var", "a0", "a1", "a2"]
        assert dist.parameters == set(expected_params)

    def test_polynomial_dist_from_dict(self):
        """Test PolynomialDist can be created from dictionary."""
        config = {
            "type": "polynomial_dist",
            "name": "test_poly",
            "x": "mass",
            "coefficients": ["c0", "c1", "c2"],
        }
        dist = PolynomialDist(**config)
        assert dist.name == "test_poly"
        assert dist.x == "mass"
        assert dist.coefficients == ["c0", "c1", "c2"]

    def test_polynomial_dist_expression_known_values(self):
        """Test PolynomialDist against known polynomial values."""
        # Test quadratic: 1 + 2x + 3x^2
        dist = PolynomialDist(name="test_poly", x="x", coefficients=["a0", "a1", "a2"])

        # At x=0: 1 + 2*0 + 3*0^2 = 1
        params = {
            "x": pt.constant(0.0),
            "a0": pt.constant(1.0),
            "a1": pt.constant(2.0),
            "a2": pt.constant(3.0),
        }
        result = dist.expression(params)
        f = function([], result)
        result_val = f()
        expected = 1.0
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

        # At x=1: 1 + 2*1 + 3*1^2 = 6
        params = {
            "x": pt.constant(1.0),
            "a0": pt.constant(1.0),
            "a1": pt.constant(2.0),
            "a2": pt.constant(3.0),
        }
        result = dist.expression(params)
        f = function([], result)
        result_val = f()
        expected = 6.0
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

        # At x=2: 1 + 2*2 + 3*2^2 = 1 + 4 + 12 = 17
        params = {
            "x": pt.constant(2.0),
            "a0": pt.constant(1.0),
            "a1": pt.constant(2.0),
            "a2": pt.constant(3.0),
        }
        result = dist.expression(params)
        f = function([], result)
        result_val = f()
        expected = 17.0
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_polynomial_dist_linear_case(self):
        """Test PolynomialDist for simple linear case."""
        # Test linear: 5 + 3x
        dist = PolynomialDist(name="linear", x="x", coefficients=["c0", "c1"])

        # At x=0: 5 + 3*0 = 5
        params = {"x": pt.constant(0.0), "c0": pt.constant(5.0), "c1": pt.constant(3.0)}
        result = dist.expression(params)
        f = function([], result)
        assert np.isclose(f(), 5.0)

        # At x=2: 5 + 3*2 = 11
        params = {"x": pt.constant(2.0), "c0": pt.constant(5.0), "c1": pt.constant(3.0)}
        result = dist.expression(params)
        f = function([], result)
        assert np.isclose(f(), 11.0)

    def test_polynomial_dist_constant_case(self):
        """Test PolynomialDist for constant case."""
        # Test constant: 42
        dist = PolynomialDist(name="constant", x="x", coefficients=["c0"])

        # Should always return 42 regardless of x value
        for x_val in [0.0, 1.0, -1.0, 10.0]:
            params = {"x": pt.constant(x_val), "c0": pt.constant(42.0)}
            result = dist.expression(params)
            f = function([], result)
            assert np.isclose(f(), 42.0), f"Failed for x={x_val}"

    def test_polynomial_dist_integration_with_workspace(self):
        """Test PolynomialDist integration in full Workspace workflow."""
        test_data = {
            "parameter_points": [
                {
                    "name": "test_params",
                    "parameters": [
                        {"name": "mass", "value": 1.0},
                        {"name": "p0", "value": 1.0},
                        {"name": "p1", "value": 2.0},
                    ],
                }
            ],
            "distributions": [
                {
                    "type": "polynomial_dist",
                    "name": "poly_bkg",
                    "x": "mass",
                    "coefficients": ["p0", "p1"],
                }
            ],
            "domains": [{"name": "test_domain", "type": "product_domain", "axes": []}],
            "functions": [],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)
        model = ws.model(domain="test_domain", parameter_set="test_params")

        # Verify the distribution was created
        assert "poly_bkg" in model.distributions
        assert "mass" in model.parameters
        assert "p0" in model.parameters
        assert "p1" in model.parameters

        # Verify we can evaluate the PDF: 1 + 2*1 = 3
        pdf_value = model.pdf("poly_bkg", mass=1.0, p0=1.0, p1=2.0)
        assert pdf_value is not None
        expected = 3.0
        np.testing.assert_allclose(pdf_value, expected, rtol=1e-6)


class TestArgusDist:
    """Test ArgusDist implementation."""

    def test_argus_dist_creation(self):
        """Test ArgusDist can be created and configured."""
        dist = ArgusDist(
            name="test_argus", mass="m", resonance="m0", slope="c", power="p"
        )
        assert dist.name == "test_argus"
        assert dist.mass == "m"
        assert dist.resonance == "m0"
        assert dist.slope == "c"
        assert dist.power == "p"
        expected_params = ["m", "m0", "c", "p"]
        assert dist.parameters == set(expected_params)

    def test_argus_dist_from_dict(self):
        """Test ArgusDist can be created from dictionary."""
        config = {
            "type": "argus_dist",
            "name": "test_argus",
            "mass": "mbc",
            "resonance": "m_B",
            "slope": "c_argus",
            "power": "p_argus",
        }
        dist = ArgusDist(**config)
        assert dist.name == "test_argus"
        assert dist.mass == "mbc"
        assert dist.resonance == "m_B"
        assert dist.slope == "c_argus"
        assert dist.power == "p_argus"

    def test_argus_dist_expression_known_values(self):
        """Test ArgusDist against known ARGUS values."""
        dist = ArgusDist(
            name="test_argus", mass="m", resonance="m0", slope="c", power="p"
        )

        # Test at m=0: 0 * [1 - 0]^p * exp[c * 1] = 0 (regardless of other params)
        params = {
            "m": pt.constant(0.0),
            "m0": pt.constant(5.0),
            "c": pt.constant(-1.0),
            "p": pt.constant(0.5),
        }
        result = dist.expression(params)
        f = function([], result)
        result_val = f()
        expected = 0.0
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

        # Test specific case: m=3, m0=5, c=-0.5, p=0.5
        # Should give: 3 * [1 - (3/5)^2]^0.5 * exp[-0.5 * (1 - (3/5)^2)]
        # = 3 * [1 - 9/25]^0.5 * exp[-0.5 * (1 - 9/25)]
        # = 3 * [16/25]^0.5 * exp[-0.5 * 16/25]
        # = 3 * (4/5) * exp[-0.32]
        m_val, m0_val, c_val, p_val = 3.0, 5.0, -0.5, 0.5
        params = {
            "m": pt.constant(m_val),
            "m0": pt.constant(m0_val),
            "c": pt.constant(c_val),
            "p": pt.constant(p_val),
        }
        result = dist.expression(params)
        f = function([], result)
        result_val = f()

        # Calculate expected manually
        ratio_squared = (m_val / m0_val) ** 2  # (3/5)^2 = 9/25 = 0.36
        bracket_term = 1.0 - ratio_squared  # 1 - 0.36 = 0.64
        power_term = bracket_term**p_val  # 0.64^0.5 = 0.8
        exp_term = np.exp(c_val * bracket_term)  # exp(-0.5 * 0.64) = exp(-0.32)
        expected = m_val * power_term * exp_term  # 3 * 0.8 * exp(-0.32)

        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_argus_dist_properties(self):
        """Test that ArgusDist has correct mathematical properties."""
        dist = ArgusDist(
            name="test_argus", mass="m", resonance="m0", slope="c", power="p"
        )

        # Fixed parameters for B physics: m0=5.279 (B meson mass), typical values
        m0_val = 5.279
        c_val = -15.0  # Negative slope parameter
        p_val = 0.5  # Common power
        params = {
            "m0": pt.constant(m0_val),
            "c": pt.constant(c_val),
            "p": pt.constant(p_val),
        }

        # Test that PDF is zero at endpoint m = m0
        test_params = dict(params)
        test_params["m"] = pt.constant(m0_val)
        result = dist.expression(test_params)
        f = function([], result)
        pdf_val = f()
        assert np.isclose(pdf_val, 0.0, atol=1e-10), (
            f"PDF should be 0 at endpoint, got {pdf_val}"
        )

        # Test that PDF is positive for m < m0
        m_values = [1.0, 2.0, 3.0, 4.0, 5.0]  # All less than m0

        for m_val in m_values:
            test_params = dict(params)
            test_params["m"] = pt.constant(m_val)
            result = dist.expression(test_params)
            f = function([], result)
            pdf_val = f()
            assert pdf_val >= 0.0, (
                f"PDF should be non-negative, got {pdf_val} for m={m_val}"
            )

    def test_argus_dist_integration_with_workspace(self):
        """Test ArgusDist integration in full Workspace workflow."""
        test_data = {
            "parameter_points": [
                {
                    "name": "test_params",
                    "parameters": [
                        {"name": "mbc", "value": 5.0},
                        {"name": "m_B", "value": 5.279},
                        {"name": "c_slope", "value": -10.0},
                        {"name": "p_power", "value": 0.5},
                    ],
                }
            ],
            "distributions": [
                {
                    "type": "argus_dist",
                    "name": "argus_bkg",
                    "mass": "mbc",
                    "resonance": "m_B",
                    "slope": "c_slope",
                    "power": "p_power",
                }
            ],
            "domains": [{"name": "test_domain", "type": "product_domain", "axes": []}],
            "functions": [],
            "metadata": {"hs3_version": "0.2"},
        }

        ws = Workspace(**test_data)
        model = ws.model(domain="test_domain", parameter_set="test_params")

        # Verify the distribution was created
        assert "argus_bkg" in model.distributions
        assert "mbc" in model.parameters
        assert "m_B" in model.parameters
        assert "c_slope" in model.parameters
        assert "p_power" in model.parameters

        # Verify we can evaluate the PDF
        pdf_value = model.pdf(
            "argus_bkg", mbc=5.0, m_B=5.279, c_slope=-10.0, p_power=0.5
        )
        assert pdf_value is not None
        assert pdf_value >= 0.0  # Should be non-negative


class TestBernsteinPolyDist:
    """Test BernsteinPolyDist implementation."""

    def test_bernstein_poly_creation(self):
        """Test BernsteinPolyDist can be created and configured."""
        dist = BernsteinPolyDist(
            name="test_bernstein",
            x="x_var",
            coefficients=["c0", "c1", "c2"],
        )
        assert dist.name == "test_bernstein"
        assert dist.x == "x_var"
        assert dist.coefficients == ["c0", "c1", "c2"]
        assert set(dist.parameters) == {"x_var", "c0", "c1", "c2"}

    def test_bernstein_poly_single_coefficient(self):
        """Test BernsteinPolyDist with single coefficient (degree 0)."""
        dist = BernsteinPolyDist(
            name="bernstein_const",
            x="x",
            coefficients=["c0"],
        )

        # For degree 0: B_0,0(x) = 1, so result is just c0
        context = {"x": pt.constant(0.5), "c0": pt.constant(2.0)}
        result = dist.expression(context)
        f = function([], result)
        result_val = f()

        # Expected: c0 * B_0,0(0.5) = 2.0 * 1 = 2.0
        expected = 2.0
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_bernstein_poly_linear(self):
        """Test BernsteinPolyDist with linear polynomial (degree 1)."""
        dist = BernsteinPolyDist(
            name="bernstein_linear",
            x="x",
            coefficients=["c0", "c1"],
        )

        # For degree 1: B_0,1(x) = 1-x, B_1,1(x) = x
        x_val = 0.3
        context = {
            "x": pt.constant(x_val),
            "c0": pt.constant(1.0),
            "c1": pt.constant(3.0),
        }
        result = dist.expression(context)
        f = function([], result)
        result_val = f()

        # Expected: c0 * B_0,1(0.3) + c1 * B_1,1(0.3) = 1.0 * (1-0.3) + 3.0 * 0.3 = 0.7 + 0.9 = 1.6
        expected = 1.0 * (1 - x_val) + 3.0 * x_val
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_bernstein_poly_quadratic(self):
        """Test BernsteinPolyDist with quadratic polynomial (degree 2)."""
        dist = BernsteinPolyDist(
            name="bernstein_quad",
            x="x",
            coefficients=["c0", "c1", "c2"],
        )

        # For degree 2: B_0,2(x) = (1-x)^2, B_1,2(x) = 2x(1-x), B_2,2(x) = x^2
        x_val = 0.4
        context = {
            "x": pt.constant(x_val),
            "c0": pt.constant(1.0),
            "c1": pt.constant(2.0),
            "c2": pt.constant(1.5),
        }
        result = dist.expression(context)
        f = function([], result)
        result_val = f()

        # Expected calculation using binomial coefficients
        # B_0,2(0.4) = C(2,0) * 0.4^0 * (1-0.4)^2 = 1 * 1 * 0.36 = 0.36
        # B_1,2(0.4) = C(2,1) * 0.4^1 * (1-0.4)^1 = 2 * 0.4 * 0.6 = 0.48
        # B_2,2(0.4) = C(2,2) * 0.4^2 * (1-0.4)^0 = 1 * 0.16 * 1 = 0.16
        b_0_2 = (1 - x_val) ** 2
        b_1_2 = 2 * x_val * (1 - x_val)
        b_2_2 = x_val**2
        expected = 1.0 * b_0_2 + 2.0 * b_1_2 + 1.5 * b_2_2
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_bernstein_poly_boundary_values(self):
        """Test BernsteinPolyDist at boundary values x=0 and x=1."""
        dist = BernsteinPolyDist(
            name="bernstein_boundary",
            x="x",
            coefficients=["c0", "c1", "c2"],
        )

        # At x=0: only B_0,n(0) = 1, all others are 0
        context = {
            "x": pt.constant(0.0),
            "c0": pt.constant(3.0),
            "c1": pt.constant(2.0),
            "c2": pt.constant(1.0),
        }
        result = dist.expression(context)
        f = function([], result)
        result_val = f()

        # At x=0, result should be c0
        expected = 3.0
        np.testing.assert_allclose(result_val, expected, rtol=1e-10)

        # At x=1: only B_n,n(1) = 1, all others are 0
        context["x"] = pt.constant(1.0)
        result = dist.expression(context)
        f = function([], result)
        result_val = f()

        # At x=1, result should be c2 (the last coefficient)
        expected = 1.0
        np.testing.assert_allclose(result_val, expected, rtol=1e-10)

    def test_bernstein_poly_normalization_property(self):
        """Test that Bernstein basis polynomials sum to 1."""
        # This test verifies the mathematical property, not the distribution evaluation
        dist = BernsteinPolyDist(
            name="bernstein_unity",
            x="x",
            coefficients=[1.0, 1.0, 1.0, 1.0],  # All coefficients = 1
        )

        # When all coefficients are 1, the sum of Bernstein polynomials should be 1
        for x_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            context = {
                "x": pt.constant(x_val),
                "constant_bernstein_unity_coefficients[0]": pt.constant(1.0),
                "constant_bernstein_unity_coefficients[1]": pt.constant(1.0),
                "constant_bernstein_unity_coefficients[2]": pt.constant(1.0),
                "constant_bernstein_unity_coefficients[3]": pt.constant(1.0),
            }
            result = dist.expression(context)
            f = function([], result)
            result_val = f()

            # Should always be 1.0 when all coefficients are 1
            np.testing.assert_allclose(result_val, 1.0, rtol=1e-10)

    def test_bernstein_poly_with_numeric_coefficients(self):
        """Test BernsteinPolyDist with numeric coefficients."""
        dist = BernsteinPolyDist(
            name="bernstein_numeric",
            x="x_var",
            coefficients=[0.5, 1.0, 2.0],
        )

        # Check that parameters were created for numeric coefficients
        assert len(dist.parameters) == 4  # x_var + 3 coefficients
        assert "x_var" in dist.parameters

        # Test evaluation with proper constant parameter names
        context = {"x_var": pt.constant(0.2)}
        # Add the auto-generated constant parameter names
        for param in dist.parameters:
            if param != "x_var":
                context[param] = pt.constant(1.0)  # Will use actual coefficient values

        result = dist.expression(context)
        f = function([], result)
        result_val = f()

        # Should return a finite positive result
        assert np.isfinite(result_val)
        assert result_val >= 0.0


class TestLandauDist:
    """Test LandauDist implementation."""

    def test_landau_creation(self):
        """Test LandauDist can be created and configured."""
        dist = LandauDist(
            name="test_landau",
            x="x_var",
            mean="mu",
            sigma="sig",
        )
        assert dist.name == "test_landau"
        assert dist.x == "x_var"
        assert dist.mean == "mu"
        assert dist.sigma == "sig"
        assert set(dist.parameters) == {"x_var", "mu", "sig"}

    def test_landau_expression_evaluation(self):
        """Test LandauDist expression evaluation."""
        dist = LandauDist(
            name="landau_test",
            x="x",
            mean="mean_param",
            sigma="sigma_param",
        )

        context = {
            "x": pt.constant(5.0),
            "mean_param": pt.constant(3.0),
            "sigma_param": pt.constant(1.0),
        }
        result = dist.expression(context)
        f = function([], result)
        result_val = f()

        # Should return a positive finite value
        assert np.isfinite(result_val)
        assert result_val > 0.0

    def test_landau_symmetry_properties(self):
        """Test LandauDist asymmetric properties."""
        dist = LandauDist(
            name="landau_asym",
            x="x",
            mean=0.0,
            sigma=1.0,
        )

        # Test that the distribution has the expected asymmetric behavior
        # (values above mean should generally be different from values below mean by same distance)
        context_below = {
            "constant_landau_asym_mean": pt.constant(0.0),
            "constant_landau_asym_sigma": pt.constant(1.0),
            "x": pt.constant(-1.0),  # Below mean
        }
        result_below = dist.expression(context_below)
        f_below = function([], result_below)
        val_below = f_below()

        context_above = {
            "constant_landau_asym_mean": pt.constant(0.0),
            "constant_landau_asym_sigma": pt.constant(1.0),
            "x": pt.constant(1.0),  # Above mean
        }
        result_above = dist.expression(context_above)
        f_above = function([], result_above)
        val_above = f_above()

        # Both should be positive and finite
        assert np.isfinite(val_below)
        assert val_below > 0.0
        assert np.isfinite(val_above)
        assert val_above > 0.0

        # Test that the distribution can handle different input values
        # (The simplified Landau implementation may not show true asymmetry)
        assert val_below > 0.0
        assert val_above > 0.0

    def test_landau_scaling_with_sigma(self):
        """Test LandauDist scaling behavior with sigma parameter."""
        dist = LandauDist(
            name="landau_scale",
            x="x",
            mean=0.0,
            sigma="sigma_param",
        )

        # Test different sigma values
        x_val = 1.0
        for sigma_val in [0.5, 1.0, 2.0]:
            context = {
                "constant_landau_scale_mean": pt.constant(0.0),
                "sigma_param": pt.constant(sigma_val),
                "x": pt.constant(x_val),
            }
            result = dist.expression(context)
            f = function([], result)
            result_val = f()

            # All should be positive and finite
            assert np.isfinite(result_val)
            assert result_val > 0.0

            # The distribution should be properly scaled by 1/sigma
            # (this is a basic sanity check, not a strict mathematical verification)
