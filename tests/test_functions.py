"""
Unit tests for the functions module.

Tests for ProductFunction, GenericFunction, InterpolationFunction,
and FunctionSet implementations.
"""

from __future__ import annotations

import logging

import numpy as np
import pytensor.tensor as pt
import pytest
from pytensor import function

from pyhs3.exceptions import UnknownInterpolationCodeError
from pyhs3.functions import (
    Function,
    FunctionSet,
    GenericFunction,
    InterpolationFunction,
    ProductFunction,
    registered_functions,
)


class TestFunction:
    """Test the base Function class."""

    def test_function_base_class(self):
        """Test Function base class initialization."""
        func = Function(
            name="test_func",
            kind="test",
            parameters=["param1", "param2"],
        )
        assert func.name == "test_func"
        assert func.kind == "test"
        assert func.parameters == ["param1", "param2"]

    def test_function_expression_not_implemented(self):
        """Test that base Function expression raises NotImplementedError."""
        func = Function(
            name="test_func",
            kind="test",
            parameters=["param1"],
        )
        with pytest.raises(
            NotImplementedError, match="Function type test not implemented"
        ):
            func.expression({})

    def test_function_from_dict_not_implemented(self):
        """Test that base Function from_dict raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            Function.from_dict({})


class TestProductFunction:
    """Test ProductFunction implementation."""

    def test_product_function_creation(self):
        """Test ProductFunction can be created and configured."""
        func = ProductFunction(name="test_product", factors=["factor1", "factor2"])
        assert func.name == "test_product"
        assert func.kind == "product"
        assert func.factors == ["factor1", "factor2"]
        assert func.parameters == ["factor1", "factor2"]

    def test_product_function_from_dict(self):
        """Test ProductFunction can be created from dictionary."""
        config = {"name": "test_product", "factors": ["f1", "f2", "f3"]}
        func = ProductFunction.from_dict(config)
        assert func.name == "test_product"
        assert func.factors == ["f1", "f2", "f3"]
        assert func.parameters == ["f1", "f2", "f3"]

    def test_product_function_expression_empty_factors(self):
        """Test ProductFunction with empty factors returns 1.0."""
        func = ProductFunction(name="test_product", factors=[])
        context = {}
        result = func.expression(context)

        # Compile and evaluate
        f = function([], result)
        assert np.isclose(f(), 1.0)

    @pytest.mark.parametrize(
        ("factors", "values", "expected"),
        [
            pytest.param(["factor1"], [5.0], 5.0, id="single_factor"),
            pytest.param(["f1", "f2"], [2.0, 3.0], 6.0, id="two_factors"),
            pytest.param(["f1", "f2", "f3"], [2.0, 3.0, 4.0], 24.0, id="three_factors"),
            pytest.param(
                ["a", "b", "c", "d"], [1.0, 2.0, 3.0, 4.0], 24.0, id="four_factors"
            ),
        ],
    )
    def test_product_function_expression_multiple_factors(
        self, factors, values, expected
    ):
        """Test ProductFunction with different numbers of factors."""
        func = ProductFunction(name="test_product", factors=factors)
        context = {factor: pt.constant(value) for factor, value in zip(factors, values)}
        result = func.expression(context)

        # Compile and evaluate
        f = function([], result)
        assert np.isclose(f(), expected)

    @pytest.mark.parametrize(
        ("x_val", "y_val", "expected"),
        [
            pytest.param(3.0, 7.0, 21.0, id="positive_values"),
            pytest.param(-2.0, 4.0, -8.0, id="negative_positive"),
            pytest.param(0.0, 5.0, 0.0, id="zero_factor"),
            pytest.param(1.5, 2.5, 3.75, id="decimal_values"),
        ],
    )
    def test_product_function_expression_with_variables(self, x_val, y_val, expected):
        """Test ProductFunction with variable factors."""
        func = ProductFunction(name="test_product", factors=["x", "y"])
        x = pt.scalar("x")
        y = pt.scalar("y")
        context = {"x": x, "y": y}
        result = func.expression(context)

        # Compile and evaluate
        f = function([x, y], result)
        assert np.isclose(f(x_val, y_val), expected)


class TestGenericFunction:
    """Test GenericFunction implementation."""

    def test_generic_function_creation(self):
        """Test GenericFunction can be created and configured."""
        func = GenericFunction(name="test_generic", expression="x + y")
        assert func.name == "test_generic"
        assert func.kind == "generic_function"
        assert func.expression_str == "x + y"
        assert set(func.parameters) == {"x", "y"}

    def test_generic_function_from_dict(self):
        """Test GenericFunction can be created from dictionary."""
        config = {"name": "test_generic", "expression": "sin(x) * cos(y)"}
        func = GenericFunction.from_dict(config)
        assert func.name == "test_generic"
        assert func.expression_str == "sin(x) * cos(y)"
        assert set(func.parameters) == {"x", "y"}

    @pytest.mark.parametrize(
        ("expression", "x_val", "expected"),
        [
            pytest.param("x**2 + 1", 3.0, 10.0, id="polynomial"),
            pytest.param("2*x", 5.0, 10.0, id="linear"),
            pytest.param("exp(x)", 0.0, 1.0, id="exponential_zero"),
            pytest.param("exp(x)", 1.0, np.e, id="exponential_one"),
            pytest.param("sin(x)", 0.0, 0.0, id="sine_zero"),
            pytest.param("cos(x)", 0.0, 1.0, id="cosine_zero"),
            pytest.param("sqrt(x)", 4.0, 2.0, id="square_root"),
            pytest.param("abs(x)", -5.0, 5.0, id="absolute_value"),
        ],
    )
    def test_generic_function_single_variable_expressions(
        self, expression, x_val, expected
    ):
        """Test GenericFunction with various single-variable expressions."""
        func = GenericFunction(name="test_generic", expression=expression)
        x = pt.scalar("x")
        context = {"x": x}
        result = func.expression(context)

        # Compile and evaluate
        f = function([x], result)
        assert np.isclose(f(x_val), expected, atol=1e-10)

    @pytest.mark.parametrize(
        ("expression", "inputs", "expected"),
        [
            pytest.param(
                "x * y + z",
                {"x": 2.0, "y": 3.0, "z": 1.0},
                7.0,
                id="linear_combination",
            ),
            pytest.param(
                "x**2 + y**2", {"x": 3.0, "y": 4.0}, 25.0, id="sum_of_squares"
            ),
            pytest.param(
                "sin(x) + cos(y)", {"x": 0.0, "y": 0.0}, 1.0, id="trig_combination"
            ),
            pytest.param(
                "exp(x) * log(y)", {"x": 1.0, "y": np.e}, np.e, id="exp_log_product"
            ),
            pytest.param(
                "(x + y) / z", {"x": 6.0, "y": 3.0, "z": 3.0}, 3.0, id="division"
            ),
        ],
    )
    def test_generic_function_multi_variable_expressions(
        self, expression, inputs, expected
    ):
        """Test GenericFunction with multi-variable expressions."""
        func = GenericFunction(name="test_generic", expression=expression)

        # Create PyTensor variables for all parameters
        variables = {}
        var_list = []
        for param in func.parameters:
            var = pt.scalar(param)
            variables[param] = var
            var_list.append(var)

        result = func.expression(variables)

        # Compile and evaluate
        f = function(var_list, result)
        input_values = [inputs[param] for param in func.parameters]
        assert np.isclose(f(*input_values), expected, atol=1e-10)

    @pytest.mark.parametrize(
        ("expression", "expected_params"),
        [
            pytest.param("x**2", ["x"], id="single_variable"),
            pytest.param(
                "a*x + b*y + c", ["a", "b", "c", "x", "y"], id="multiple_variables"
            ),
            pytest.param("42", [], id="constants_only"),
            pytest.param("sin(theta) + cos(phi)", ["phi", "theta"], id="greek_letters"),
            pytest.param(
                "x_1 + x_2 + x_3", ["x_1", "x_2", "x_3"], id="numbered_variables"
            ),
        ],
    )
    def test_generic_function_parameter_detection(self, expression, expected_params):
        """Test that GenericFunction correctly detects parameters."""
        func = GenericFunction(name="test", expression=expression)
        assert set(func.parameters) == set(expected_params)


class TestInterpolationFunction:
    """Test InterpolationFunction implementation."""

    def test_interpolation_function_creation(self):
        """Test InterpolationFunction can be created and configured."""
        func = InterpolationFunction(
            name="test_interp",
            high=["high1", "high2"],
            low=["low1", "low2"],
            nom="nominal",
            interpolationCodes=[0, 1],
            positiveDefinite=True,
            parameters=["var1", "var2"],
        )
        assert func.name == "test_interp"
        assert func.kind == "interpolation"
        assert func.high == ["high1", "high2"]
        assert func.low == ["low1", "low2"]
        assert func.nom == "nominal"
        assert func.interpolationCodes == [0, 1]
        assert func.positiveDefinite is True
        assert func.parameters == ["var1", "var2"]

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "name": "test_interp",
                    "high": ["h1"],
                    "low": ["l1"],
                    "nom": "n1",
                    "interpolationCodes": [0],
                    "positiveDefinite": False,
                    "vars": ["x"],
                },
                id="single_variable",
            ),
            pytest.param(
                {
                    "name": "multi_interp",
                    "high": ["h1", "h2", "h3"],
                    "low": ["l1", "l2", "l3"],
                    "nom": "nominal",
                    "interpolationCodes": [0, 1, 2],
                    "positiveDefinite": True,
                    "vars": ["x", "y", "z"],
                },
                id="multiple_variables",
            ),
            pytest.param(
                {
                    "name": "empty_interp",
                    "high": [],
                    "low": [],
                    "nom": "nominal",
                    "interpolationCodes": [],
                    "positiveDefinite": False,
                    "vars": [],
                },
                id="empty_lists",
            ),
        ],
    )
    def test_interpolation_function_from_dict(self, config):
        """Test InterpolationFunction can be created from various dictionaries."""
        func = InterpolationFunction.from_dict(config)
        assert func.name == config["name"]
        assert func.high == config["high"]
        assert func.low == config["low"]
        assert func.nom == config["nom"]
        assert func.interpolationCodes == config["interpolationCodes"]
        assert func.positiveDefinite == config["positiveDefinite"]
        assert func.parameters == config["vars"]

    def test_interpolation_function_expression_nominal_only(self):
        """Test InterpolationFunction returns nominal value when no parameters."""
        func = InterpolationFunction(
            name="test_interp",
            high=[],
            low=[],
            nom="nominal",
            interpolationCodes=[],
            positiveDefinite=True,
            parameters=[],
        )
        context = {"nominal": pt.constant(5.0)}

        result = func.expression(context)
        f = function([], result)
        assert np.isclose(f(), 5.0)

    def test_interpolation_function_linear_interpolation(self):
        """Test InterpolationFunction with linear interpolation (code 0)."""
        func = InterpolationFunction(
            name="test_interp",
            high=["high_var"],
            low=["low_var"],
            nom="nominal",
            interpolationCodes=[0],
            positiveDefinite=False,
            parameters=["nuisance_param"],
        )

        # Test with positive parameter value
        context = {
            "nominal": pt.constant(10.0),
            "high_var": pt.constant(12.0),
            "low_var": pt.constant(8.0),
            "nuisance_param": pt.constant(0.5),
        }

        result = func.expression(context)
        f = function([], result)
        result_val = f()

        # Expected: 10.0 + 0.5 * (12.0 - 10.0) = 11.0
        np.testing.assert_allclose(result_val, 11.0, rtol=1e-10)

    def test_interpolation_function_exponential_interpolation(self):
        """Test InterpolationFunction with exponential interpolation (code 1)."""
        func = InterpolationFunction(
            name="test_interp",
            high=["high_var"],
            low=["low_var"],
            nom="nominal",
            interpolationCodes=[1],
            positiveDefinite=False,
            parameters=["nuisance_param"],
        )

        # Test with positive parameter value
        context = {
            "nominal": pt.constant(10.0),
            "high_var": pt.constant(20.0),  # ratio = 2.0
            "low_var": pt.constant(5.0),  # ratio = 0.5
            "nuisance_param": pt.constant(0.5),
        }

        result = func.expression(context)
        f = function([], result)
        result_val = f()

        # Expected: 10.0 * (1 + (2.0^0.5 - 1)) = 10.0 * sqrt(2) ≈ 14.14
        expected = 10.0 * np.sqrt(2.0)
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_interpolation_function_positive_definite(self):
        """Test InterpolationFunction with positive definite constraint."""
        func = InterpolationFunction(
            name="test_interp",
            high=["high_var"],
            low=["low_var"],
            nom="nominal",
            interpolationCodes=[0],
            positiveDefinite=True,
            parameters=["nuisance_param"],
        )

        # Create context that would give negative result without constraint
        context = {
            "nominal": pt.constant(5.0),
            "high_var": pt.constant(2.0),
            "low_var": pt.constant(8.0),
            "nuisance_param": pt.constant(2.0),  # Large positive value
        }

        result = func.expression(context)
        f = function([], result)
        result_val = f()

        # Without constraint: 5.0 + 2.0 * (2.0 - 5.0) = 5.0 - 6.0 = -1.0
        # With constraint: max(-1.0, 0.0) = 0.0
        np.testing.assert_allclose(result_val, 0.0, rtol=1e-10)

    @pytest.mark.parametrize("interp_code", [0, 1, 2, 3, 4, 5, 6])
    def test_interpolation_function_all_codes_at_zero(self, interp_code):
        """Test InterpolationFunction with all interpolation codes at theta=0."""
        func = InterpolationFunction(
            name="test_interp",
            high=["high_var"],
            low=["low_var"],
            nom="nominal",
            interpolationCodes=[interp_code],
            positiveDefinite=False,
            parameters=["nuisance_param"],
        )

        context = {
            "nominal": pt.constant(10.0),
            "high_var": pt.constant(12.0),
            "low_var": pt.constant(8.0),
            "nuisance_param": pt.constant(0.0),  # At nominal
        }

        result = func.expression(context)
        f = function([], result)
        result_val = f()

        # At theta=0, all codes should return nominal value
        np.testing.assert_allclose(result_val, 10.0, rtol=1e-10)

    def test_interpolation_function_code_6_polynomial_multiplicative(self):
        """Test InterpolationFunction with code 6 (polynomial + linear multiplicative)."""
        func = InterpolationFunction(
            name="test_interp",
            high=["high_var"],
            low=["low_var"],
            nom="nominal",
            interpolationCodes=[6],
            positiveDefinite=False,
            parameters=["nuisance_param"],
        )

        # Test with positive parameter value in polynomial region (|theta| < 1)
        context = {
            "nominal": pt.constant(10.0),
            "high_var": pt.constant(20.0),  # ratio = 2.0
            "low_var": pt.constant(5.0),  # ratio = 0.5
            "nuisance_param": pt.constant(0.5),  # |0.5| < 1, polynomial region
        }

        result = func.expression(context)
        f = function([], result)
        result_val = f()

        # Code 6: polynomial multiplicative mode
        # Expected: 10.0 * (1 + 0.5 * (2.0 - 1.0) * (1 + 0.5^2 * (-3 + 0.5^2) / 16))
        # = 10.0 * (1 + 0.5 * 1.0 * (1 + 0.25 * (-3 + 0.25) / 16))
        # = 10.0 * (1 + 0.5 * (1 + 0.25 * (-2.75) / 16))
        # = 10.0 * (1 + 0.5 * (1 - 0.04296875))
        # = 10.0 * (1 + 0.5 * 0.95703125) = 10.0 * 1.478515625 ≈ 14.785
        expected = 10.0 * (1 + 0.5 * (2.0 - 1.0) * (1 + 0.25 * (-3 + 0.25) / 16))
        np.testing.assert_allclose(result_val, expected, rtol=1e-10)

    def test_interpolation_function_parameter_index_warning(self, caplog):
        """Test InterpolationFunction logs warning when parameter index exceeds lists."""
        func = InterpolationFunction(
            name="test_interp",
            high=["high_var"],  # Only one element
            low=["low_var"],  # Only one element
            nom="nominal",
            interpolationCodes=[0],  # Only one element
            positiveDefinite=False,
            parameters=["param1", "param2"],  # Two parameters - exceeds lists!
        )

        context = {
            "nominal": pt.constant(10.0),
            "high_var": pt.constant(12.0),
            "low_var": pt.constant(8.0),
            "param1": pt.constant(0.5),
            "param2": pt.constant(0.3),  # This will trigger the warning
        }

        with caplog.at_level(logging.WARNING):
            result = func.expression(context)
            f = function([], result)
            result_val = f()

        # Should log warning about parameter index exceeding lists
        assert (
            "Parameter index 1 exceeds variation lists for function test_interp"
            in caplog.text
        )

        # Result should still be computed (only first parameter processed)
        # Expected: 10.0 + 0.5 * (12.0 - 10.0) = 11.0
        np.testing.assert_allclose(result_val, 11.0, rtol=1e-10)

    def test_interpolation_function_unknown_code_raises_exception(self):
        """Test InterpolationFunction raises exception for unknown interpolation codes."""
        # Test invalid code during initialization
        with pytest.raises(
            UnknownInterpolationCodeError,
            match="Unknown interpolation code 99 in function 'bad_interp'. Valid codes are 0-6.",
        ):
            InterpolationFunction(
                name="bad_interp",
                high=["high_var"],
                low=["low_var"],
                nom="nominal",
                interpolationCodes=[99],  # Invalid code
                positiveDefinite=False,
                parameters=["param"],
            )

        # Test mix of valid and invalid codes
        with pytest.raises(
            UnknownInterpolationCodeError,
            match="Unknown interpolation code -1 in function 'bad_interp2'. Valid codes are 0-6.",
        ):
            InterpolationFunction(
                name="bad_interp2",
                high=["high1", "high2"],
                low=["low1", "low2"],
                nom="nominal",
                interpolationCodes=[0, -1],  # Mix of valid and invalid
                positiveDefinite=False,
                parameters=["param1", "param2"],
            )

    def test_interpolation_function_integration(self):
        """Test InterpolationFunction integration with FunctionSet."""

        # Create function configuration matching your example structure
        functions_config = [
            {
                "type": "interpolation",
                "name": "test_shape_interp",
                "high": ["high_variation"],
                "low": ["low_variation"],
                "nom": "nominal_shape",
                "interpolationCodes": [0],
                "positiveDefinite": False,
                "vars": ["shape_param"],
            }
        ]

        function_set = FunctionSet(functions_config)
        assert len(function_set) == 1

        interp_func = function_set["test_shape_interp"]
        assert isinstance(interp_func, InterpolationFunction)
        assert interp_func.nom == "nominal_shape"
        assert interp_func.high == ["high_variation"]
        assert interp_func.low == ["low_variation"]
        assert interp_func.interpolationCodes == [0]
        assert interp_func.parameters == ["shape_param"]

        # Test evaluation
        context = {
            "nominal_shape": pt.constant(100.0),
            "high_variation": pt.constant(110.0),
            "low_variation": pt.constant(90.0),
            "shape_param": pt.constant(0.2),
        }

        result = interp_func.expression(context)
        f = function([], result)
        result_val = f()

        # Expected: 100.0 + 0.2 * (110.0 - 100.0) = 102.0
        np.testing.assert_allclose(result_val, 102.0, rtol=1e-10)


class TestRegisteredFunctions:
    """Test the registered_functions registry."""

    def test_registry_contains_expected_functions(self):
        """Test that registry contains all expected function types."""
        expected_types = {"product", "generic_function", "interpolation"}
        assert set(registered_functions.keys()) == expected_types

    @pytest.mark.parametrize(
        ("func_type", "expected_class"),
        [
            pytest.param("product", ProductFunction, id="product_function"),
            pytest.param("generic_function", GenericFunction, id="generic_function"),
            pytest.param(
                "interpolation", InterpolationFunction, id="interpolation_function"
            ),
        ],
    )
    def test_registry_maps_to_correct_classes(self, func_type, expected_class):
        """Test that registry maps to correct function classes."""
        assert registered_functions[func_type] is expected_class


class TestFunctionSet:
    """Test FunctionSet collection."""

    def test_function_set_empty(self):
        """Test FunctionSet with empty function list."""
        function_set = FunctionSet([])
        assert len(function_set) == 0

    def test_function_set_product_functions(self):
        """Test FunctionSet with product functions."""
        functions_config = [
            {"name": "prod1", "type": "product", "factors": ["a", "b"]},
            {"name": "prod2", "type": "product", "factors": ["x", "y", "z"]},
        ]
        function_set = FunctionSet(functions_config)
        assert len(function_set) == 2
        assert "prod1" in function_set
        assert "prod2" in function_set

        func1 = function_set["prod1"]
        assert isinstance(func1, ProductFunction)
        assert func1.factors == ["a", "b"]

        func2 = function_set["prod2"]
        assert isinstance(func2, ProductFunction)
        assert func2.factors == ["x", "y", "z"]

    def test_function_set_generic_functions(self):
        """Test FunctionSet with generic functions."""
        functions_config = [
            {"name": "gen1", "type": "generic_function", "expression": "x + y"},
            {"name": "gen2", "type": "generic_function", "expression": "sin(t)"},
        ]
        function_set = FunctionSet(functions_config)
        assert len(function_set) == 2

        func1 = function_set["gen1"]
        assert isinstance(func1, GenericFunction)
        assert func1.expression_str == "x + y"

        func2 = function_set["gen2"]
        assert isinstance(func2, GenericFunction)
        assert func2.expression_str == "sin(t)"

    def test_function_set_interpolation_functions(self):
        """Test FunctionSet with interpolation functions."""
        functions_config = [
            {
                "name": "interp1",
                "type": "interpolation",
                "high": ["h1"],
                "low": ["l1"],
                "nom": ["n1"],
                "interpolationCodes": ["linear"],
                "positiveDefinite": True,
                "vars": ["x"],
            }
        ]
        function_set = FunctionSet(functions_config)
        assert len(function_set) == 1

        func = function_set["interp1"]
        assert isinstance(func, InterpolationFunction)
        assert func.high == ["h1"]
        assert func.parameters == ["x"]

    def test_function_set_mixed_types(self):
        """Test FunctionSet with mixed function types."""
        functions_config = [
            {"name": "prod", "type": "product", "factors": ["a", "b"]},
            {"name": "gen", "type": "generic_function", "expression": "x**2"},
            {
                "name": "interp",
                "type": "interpolation",
                "high": [],
                "low": [],
                "nom": [],
                "interpolationCodes": [],
                "positiveDefinite": False,
                "vars": [],
            },
        ]
        function_set = FunctionSet(functions_config)
        assert len(function_set) == 3

        assert isinstance(function_set["prod"], ProductFunction)
        assert isinstance(function_set["gen"], GenericFunction)
        assert isinstance(function_set["interp"], InterpolationFunction)

    def test_function_set_unknown_type(self):
        """Test FunctionSet raises error for unknown function types."""
        functions_config = [
            {"name": "good", "type": "product", "factors": ["a"]},
            {"name": "bad", "type": "unknown_type", "param": "value"},
            {"name": "good2", "type": "generic_function", "expression": "x"},
        ]

        # Should raise ValueError for unknown types
        with pytest.raises(ValueError, match="Unknown function type: unknown_type"):
            FunctionSet(functions_config)

    def test_function_set_malformed_config(self):
        """Test FunctionSet raises error for malformed function configs."""
        functions_config = [
            {"name": "good", "type": "product", "factors": ["a"]},
            {"name": "bad", "type": "product"},  # Missing factors
            {"name": "good2", "type": "generic_function", "expression": "x"},
        ]

        # Should raise KeyError for missing required fields
        with pytest.raises(KeyError, match="factors"):
            FunctionSet(functions_config)

    def test_function_set_getitem_keyerror(self):
        """Test FunctionSet raises KeyError for missing functions."""
        function_set = FunctionSet([])
        with pytest.raises(KeyError):
            _ = function_set["nonexistent"]

    def test_function_set_contains(self):
        """Test FunctionSet __contains__ method."""
        functions_config = [
            {"name": "test_func", "type": "product", "factors": ["a"]},
        ]
        function_set = FunctionSet(functions_config)

        assert "test_func" in function_set
        assert "nonexistent" not in function_set


class TestFunctionIntegration:
    """Test integration between different function types."""

    def test_product_of_generic_functions(self):
        """Test ProductFunction that depends on GenericFunction results."""
        # Create a generic function that computes x^2
        generic_func = GenericFunction(name="square", expression="x**2")

        # Create a product function that multiplies two results
        product_func = ProductFunction(name="product", factors=["val1", "val2"])

        # Set up context
        x = pt.scalar("x")
        context = {"x": x}

        # Evaluate generic function
        square_result = generic_func.expression(context)

        # Use results in product
        product_context = {"val1": square_result, "val2": pt.constant(3.0)}
        product_result = product_func.expression(product_context)

        # Compile and test
        f = function([x], product_result)
        # x=4: 4^2 * 3 = 16 * 3 = 48
        assert np.isclose(f(4.0), 48.0)

    def test_function_parameter_dependencies(self):
        """Test that function parameters are correctly identified."""
        # Product function depends on two factors
        prod_func = ProductFunction(name="prod", factors=["a", "b"])
        assert set(prod_func.parameters) == {"a", "b"}

        # Generic function depends on variables in expression
        gen_func = GenericFunction(name="gen", expression="x * y + z")
        assert set(gen_func.parameters) == {"x", "y", "z"}

        # Interpolation function depends on vars
        interp_func = InterpolationFunction(
            name="interp",
            high=[],
            low=[],
            nom="nominal",
            interpolationCodes=[],
            positiveDefinite=True,
            parameters=["param1", "param2"],
        )
        assert set(interp_func.parameters) == {"param1", "param2"}
