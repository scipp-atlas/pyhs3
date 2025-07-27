"""
Test roundtripping for all distributions and functions.

This module tests that every distribution and function can be:
1. Created from a dictionary configuration
2. Serialized back to a dictionary
3. The serialized dictionary can recreate an equivalent instance
"""

from __future__ import annotations

from pyhs3.distributions import (
    CrystalBallDist,
    GaussianDist,
    GenericDist,
    MixtureDist,
    PoissonDist,
    ProductDist,
)
from pyhs3.functions import GenericFunction, ProductFunction, SumFunction


class TestDistributionRoundtripping:
    """Test roundtripping for all distribution types."""

    def test_gaussian_dist_roundtrip_string_params(self):
        """Test GaussianDist roundtripping with string parameters."""
        config = {
            "type": "gaussian_dist",
            "name": "test_gaussian",
            "mean": "mu",
            "sigma": "sigma",
            "x": "obs",
        }

        # Create from dict
        dist1 = GaussianDist.from_dict(config)

        # Serialize back to dict
        serialized = dist1.model_dump()

        # Create again from serialized dict
        dist2 = GaussianDist.from_dict(serialized)

        # Should be equivalent
        assert dist1.name == dist2.name
        assert dist1.type == dist2.type
        assert dist1.mean == dist2.mean
        assert dist1.sigma == dist2.sigma
        assert dist1.x == dist2.x
        assert serialized == config

    def test_gaussian_dist_roundtrip_numeric_params(self):
        """Test GaussianDist roundtripping with numeric parameters."""
        config = {
            "type": "gaussian_dist",
            "name": "test_gaussian",
            "mean": 1.5,
            "sigma": "sigma",
            "x": 2.0,
        }

        # Create from dict
        dist1 = GaussianDist.from_dict(config)

        # Serialize back to dict
        serialized = dist1.model_dump()

        # Create again from serialized dict
        dist2 = GaussianDist.from_dict(serialized)

        # Should be equivalent
        assert dist1.name == dist2.name
        assert dist1.type == dist2.type
        assert serialized == config

        # Check that constants were generated correctly
        assert len(dist1._constants_values) == 2
        assert len(dist2._constants_values) == 2

    def test_poisson_dist_roundtrip(self):
        """Test PoissonDist roundtripping."""
        config = {
            "type": "poisson_dist",
            "name": "test_poisson",
            "mean": 5.0,
            "x": "count",
        }

        # Create from dict
        dist1 = PoissonDist.from_dict(config)

        # Serialize back to dict
        serialized = dist1.model_dump()

        # Create again from serialized dict
        dist2 = PoissonDist.from_dict(serialized)

        # Should be equivalent
        assert dist1.name == dist2.name
        assert dist1.type == dist2.type
        assert serialized == config

    def test_mixture_dist_roundtrip(self):
        """Test MixtureDist roundtripping."""
        config = {
            "type": "mixture_dist",
            "name": "test_mixture",
            "summands": ["comp1", "comp2", "comp3"],
            "coefficients": ["c1", "c2"],
            "extended": True,
        }

        # Create from dict
        dist1 = MixtureDist.from_dict(config)

        # Serialize back to dict
        serialized = dist1.model_dump()

        # Create again from serialized dict
        dist2 = MixtureDist.from_dict(serialized)

        # Should be equivalent
        assert dist1.name == dist2.name
        assert dist1.type == dist2.type
        assert dist1.summands == dist2.summands
        assert dist1.coefficients == dist2.coefficients
        assert dist1.extended == dist2.extended
        assert serialized == config

    def test_product_dist_roundtrip(self):
        """Test ProductDist roundtripping."""
        config = {
            "type": "product_dist",
            "name": "test_product",
            "factors": ["factor1", "factor2"],
        }

        # Create from dict
        dist1 = ProductDist.from_dict(config)

        # Serialize back to dict
        serialized = dist1.model_dump()

        # Create again from serialized dict
        dist2 = ProductDist.from_dict(serialized)

        # Should be equivalent
        assert dist1.name == dist2.name
        assert dist1.type == dist2.type
        assert dist1.factors == dist2.factors
        assert serialized == config

    def test_crystal_ball_dist_roundtrip(self):
        """Test CrystalBallDist roundtripping."""
        config = {
            "type": "crystalball_doublesided_dist",
            "name": "test_crystalball",
            "alpha_L": "alpha_L",
            "alpha_R": "alpha_R",
            "m": "mass",
            "m0": "mass0",
            "n_L": "n_L",
            "n_R": "n_R",
            "sigma_L": "sigma_L",
            "sigma_R": "sigma_R",
        }

        # Create from dict
        dist1 = CrystalBallDist.from_dict(config)

        # Serialize back to dict
        serialized = dist1.model_dump()

        # Create again from serialized dict
        dist2 = CrystalBallDist.from_dict(serialized)

        # Should be equivalent
        assert dist1.name == dist2.name
        assert dist1.type == dist2.type
        assert dist1.alpha_L == dist2.alpha_L
        assert dist1.alpha_R == dist2.alpha_R
        assert dist1.m == dist2.m
        assert dist1.m0 == dist2.m0
        assert dist1.n_L == dist2.n_L
        assert dist1.n_R == dist2.n_R
        assert dist1.sigma_L == dist2.sigma_L
        assert dist1.sigma_R == dist2.sigma_R
        assert serialized == config

    def test_generic_dist_roundtrip(self):
        """Test GenericDist roundtripping."""
        config = {
            "type": "generic_dist",
            "name": "test_generic",
            "expression": "x**2 + y*sin(z)",
        }

        # Create from dict
        dist1 = GenericDist.from_dict(config)

        # Serialize back to dict
        serialized = dist1.model_dump()

        # Create again from serialized dict
        dist2 = GenericDist.from_dict(serialized)

        # Should be equivalent
        assert dist1.name == dist2.name
        assert dist1.type == dist2.type
        assert dist1.expression_str == dist2.expression_str
        assert serialized == config


class TestFunctionRoundtripping:
    """Test roundtripping for all function types."""

    def test_sum_function_roundtrip(self):
        """Test SumFunction roundtripping."""
        config = {
            "type": "sum",
            "name": "test_sum",
            "summands": ["a", "b", "c"],
        }

        # Create from dict
        func1 = SumFunction.from_dict(config)

        # Serialize back to dict
        serialized = func1.model_dump()

        # Create again from serialized dict
        func2 = SumFunction.from_dict(serialized)

        # Should be equivalent
        assert func1.name == func2.name
        assert func1.type == func2.type
        assert func1.summands == func2.summands
        assert serialized == config

    def test_product_function_roundtrip(self):
        """Test ProductFunction roundtripping."""
        config = {
            "type": "product",
            "name": "test_product",
            "factors": ["x", "y", "z"],
        }

        # Create from dict
        func1 = ProductFunction.from_dict(config)

        # Serialize back to dict
        serialized = func1.model_dump()

        # Create again from serialized dict
        func2 = ProductFunction.from_dict(serialized)

        # Should be equivalent
        assert func1.name == func2.name
        assert func1.type == func2.type
        assert func1.factors == func2.factors
        assert serialized == config

    def test_generic_function_roundtrip(self):
        """Test GenericFunction roundtripping."""
        config = {
            "type": "generic_function",
            "name": "test_generic_func",
            "expression": "a + b*log(c)",
        }

        # Create from dict
        func1 = GenericFunction.from_dict(config)

        # Serialize back to dict
        serialized = func1.model_dump()

        # Create again from serialized dict
        func2 = GenericFunction.from_dict(serialized)

        # Should be equivalent
        assert func1.name == func2.name
        assert func1.type == func2.type
        assert func1.expression_str == func2.expression_str
        assert serialized == config


class TestParametersHandling:
    """Test that parameters are handled correctly for dependency tracking."""

    def test_gaussian_parameters_dict_structure(self):
        """Test that GaussianDist has correct parameters dict structure."""
        config = {
            "type": "gaussian_dist",
            "name": "test",
            "mean": "mu",
            "sigma": 1.0,
            "x": "obs",
        }

        dist = GaussianDist.from_dict(config)

        # Parameters should be a dict mapping logical names to actual parameter names
        assert isinstance(dist.parameters, dict)
        assert "mean" in dist.parameters
        assert "sigma" in dist.parameters
        assert "x" in dist.parameters

        # list(parameters) should give dependency names for the dependency graph
        param_names = list(dist.parameters.values())
        assert "mu" in param_names
        assert "obs" in param_names
        # Should have generated constant name for sigma
        assert any("constant_test_sigma" in name for name in param_names)

    def test_mixture_parameters_dict_structure(self):
        """Test that MixtureDist has correct parameters dict structure."""
        config = {
            "type": "mixture_dist",
            "name": "test",
            "summands": ["s1", "s2"],
            "coefficients": ["c1"],
            "extended": False,
        }

        dist = MixtureDist.from_dict(config)

        # Parameters should map each summand and coefficient to itself
        assert isinstance(dist.parameters, dict)
        param_values = list(dist.parameters.values())
        assert "s1" in param_values
        assert "s2" in param_values
        assert "c1" in param_values

    def test_generic_function_parameters_from_expression(self):
        """Test that GenericFunction extracts parameters from expression."""
        config = {
            "type": "generic_function",
            "name": "test_func",
            "expression": "x*y + sin(z)",
        }

        func = GenericFunction.from_dict(config)

        # Should extract x, y, z as parameters
        param_values = list(func.parameters.values())
        assert "x" in param_values
        assert "y" in param_values
        assert "z" in param_values
        assert len(param_values) == 3
