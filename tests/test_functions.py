"""
Unit tests for the functions module.

Tests for ProductFunction, GenericFunction, InterpolationFunction,
and FunctionSet implementations.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import pytest
from pytensor import function

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
            dtype="test",
            parameters=["param1", "param2"],
        )
        assert func.name == "test_func"
        assert func.dtype == "test"
        assert func.parameters == ["param1", "param2"]

    def test_function_expression_not_implemented(self):
        """Test that base Function expression raises NotImplementedError."""
        func = Function(
            name="test_func",
            dtype="test",
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
        assert func.dtype == "product"
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
        assert func.dtype == "generic_function"
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
            nom=["nom1", "nom2"],
            interpolationCodes=["code1", "code2"],
            positiveDefinite=True,
            vars=["var1", "var2"],
        )
        assert func.name == "test_interp"
        assert func.dtype == "interpolation"
        assert func.high == ["high1", "high2"]
        assert func.low == ["low1", "low2"]
        assert func.nom == ["nom1", "nom2"]
        assert func.interpolationCodes == ["code1", "code2"]
        assert func.positiveDefinite is True
        assert func.vars == ["var1", "var2"]
        assert func.parameters == ["var1", "var2"]

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "name": "test_interp",
                    "high": ["h1"],
                    "low": ["l1"],
                    "nom": ["n1"],
                    "interpolationCodes": ["linear"],
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
                    "nom": ["n1", "n2", "n3"],
                    "interpolationCodes": ["linear", "polynomial", "exponential"],
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
                    "nom": [],
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
        assert func.vars == config["vars"]
        assert func.parameters == config["vars"]

    def test_interpolation_function_expression_placeholder(self):
        """Test InterpolationFunction returns constant 1.0 (placeholder)."""
        func = InterpolationFunction(
            name="test_interp",
            high=[],
            low=[],
            nom=[],
            interpolationCodes=[],
            positiveDefinite=True,
            vars=[],
        )
        context = {}

        result = func.expression(context)
        f = function([], result)
        assert np.isclose(f(), 1.0)


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
        assert func.vars == ["x"]

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
        """Test FunctionSet handles unknown function types gracefully."""
        functions_config = [
            {"name": "good", "type": "product", "factors": ["a"]},
            {"name": "bad", "type": "unknown_type", "param": "value"},
            {"name": "good2", "type": "generic_function", "expression": "x"},
        ]

        # Should log warnings for unknown types but continue
        function_set = FunctionSet(functions_config)

        # Should only have the valid functions
        assert len(function_set) == 2
        assert "good" in function_set
        assert "good2" in function_set
        assert "bad" not in function_set

    def test_function_set_malformed_config(self):
        """Test FunctionSet handles malformed function configs gracefully."""
        functions_config = [
            {"name": "good", "type": "product", "factors": ["a"]},
            {"name": "bad", "type": "product"},  # Missing factors
            {"name": "good2", "type": "generic_function", "expression": "x"},
        ]

        # Should log warnings for malformed configs but continue
        function_set = FunctionSet(functions_config)

        # Should only have the valid functions
        assert len(function_set) == 2
        assert "good" in function_set
        assert "good2" in function_set
        assert "bad" not in function_set

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
            nom=[],
            interpolationCodes=[],
            positiveDefinite=True,
            vars=["param1", "param2"],
        )
        assert set(interp_func.parameters) == {"param1", "param2"}
