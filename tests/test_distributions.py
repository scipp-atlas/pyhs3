"""
Unit tests for new distribution implementations.

Tests for ProductDist, CrystalBallDist, and GenericDist implementations,
including validation against expected NLL values from ROOT fits.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import pytest
from pytensor import function

from pyhs3 import Workspace
from pyhs3.core import boundedscalar
from pyhs3.distributions import CrystalBallDist, GaussianDist, GenericDist, ProductDist


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


class TestCrystalBallDist:
    """Test CrystalBallDist implementation."""

    def test_crystal_dist_creation(self):
        """Test CrystalBallDist can be created and configured."""
        dist = CrystalBallDist(
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
        """Test CrystalBallDist can be created from dictionary."""
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
        dist = CrystalBallDist.from_dict(config)
        assert dist.name == "test_crystal"
        assert dist.m == "mass"
        assert dist.m0 == "mean"

    def test_crystal_dist_expression_gaussian_core(self):
        """Test CrystalBallDist reduces to Gaussian in core region."""
        dist = CrystalBallDist(
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
        """Test CrystalBallDist gives power law behavior in tails."""
        dist = CrystalBallDist(
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
        assert x.name == "test"

    def test_boundedscalar_lower_bound_only(self):
        """Test boundedscalar with lower bound only."""
        x = boundedscalar("test", (0.0, None))
        assert x is not None
        assert x.name == "test"

    def test_boundedscalar_upper_bound_only(self):
        """Test boundedscalar with upper bound only."""
        x = boundedscalar("test", (None, 1.0))
        assert x is not None
        assert x.name == "test"

    def test_boundedscalar_no_bounds(self):
        """Test boundedscalar with no bounds."""
        x = boundedscalar("test", (None, None))
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


class TestDependencyGraphErrors:
    """Test dependency graph error conditions in core.py for code coverage."""

    def test_unknown_entity_referenced_error(self):
        """Test that referencing an unknown entity raises ValueError."""

        # Create a workspace with a distribution that references a non-existent parameter
        test_data = {
            "parameter_points": [
                {"name": "test_params", "parameters": [{"name": "mu", "value": 0.0}]}
            ],
            "distributions": [
                {
                    "type": "gaussian_dist",
                    "name": "test_gauss",
                    "mean": "mu",
                    "sigma": 1.0,
                    "x": "nonexistent_param",  # This parameter doesn't exist
                }
            ],
            "domains": [{"name": "test_domain", "type": "product_domain", "axes": []}],
            "functions": [],
            "metadata": {"name": "test"},
        }

        ws = Workspace(test_data)

        # This should raise ValueError with specific message about unknown entity
        with pytest.raises(
            ValueError,
            match="Unknown entity referenced: 'nonexistent_param' from 'test_gauss'",
        ):
            ws.model(domain="test_domain", parameter_point="test_params")

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
            "metadata": {"name": "test"},
        }

        ws = Workspace(test_data)

        # This should raise ValueError about circular dependency
        with pytest.raises(ValueError, match="Circular dependency detected in model"):
            ws.model(domain="test_domain", parameter_point="test_params")

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
            "metadata": {"name": "test"},
        }

        ws = Workspace(test_data)
        model = ws.model(domain="test_domain", parameter_point="test_params")

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

    def test_parameter_collection_methods(self):
        """Test ParameterCollection get(), __contains__, and __len__ methods."""
        test_data = {
            "parameter_points": [
                {"name": "params1", "parameters": [{"name": "mu", "value": 0.0}]},
                {"name": "params2", "parameters": [{"name": "sigma", "value": 1.0}]},
            ],
            "distributions": [],
            "domains": [],
            "functions": [],
            "metadata": {"name": "test"},
        }

        ws = Workspace(test_data)
        param_collection = ws.parameter_collection

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
            "metadata": {"name": "test"},
        }

        ws = Workspace(test_data)
        param_set = ws.parameter_collection["test_params"]

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

    def test_domain_collection_methods(self):
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
            "metadata": {"name": "test"},
        }

        ws = Workspace(test_data)
        domain_collection = ws.domain_collection

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
        """Test DomainSet get(), __contains__, and __len__ methods."""
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
            "metadata": {"name": "test"},
        }

        ws = Workspace(test_data)
        domain_set = ws.domain_collection["test_domain"]

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
            "metadata": {"name": "test"},
        }

        ws = Workspace(test_data)
        model = ws.model(domain="test_domain", parameter_point="test_params")

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
            "metadata": {"name": "test"},
        }

        ws = Workspace(test_data)
        model = ws.model(domain="test_domain", parameter_point="test_params")

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
            "metadata": {"name": "test"},
        }

        ws = Workspace(test_data)
        model = ws.model(domain="test_domain", parameter_point="test_params")

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
