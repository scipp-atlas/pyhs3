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

from pyhs3 import Workspace
from pyhs3.exceptions import UnknownInterpolationCodeError
from pyhs3.functions import (
    Function,
    FunctionSet,
    GenericFunction,
    InterpolationFunction,
    ProcessNormalizationFunction,
    ProductFunction,
    SumFunction,
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


class TestSumFunction:
    """Test SumFunction implementation."""

    def test_sum_function_creation(self):
        """Test SumFunction can be created and configured."""
        func = SumFunction(name="test_sum", summands=["term1", "term2"])
        assert func.name == "test_sum"
        assert func.kind == "sum"
        assert func.summands == ["term1", "term2"]
        assert func.parameters == ["term1", "term2"]

    def test_sum_function_from_dict(self):
        """Test SumFunction can be created from dictionary."""
        config = {"name": "test_sum", "summands": ["a", "b", "c"]}
        func = SumFunction.from_dict(config)
        assert func.name == "test_sum"
        assert func.summands == ["a", "b", "c"]
        assert func.parameters == ["a", "b", "c"]

    def test_sum_function_expression_empty_summands(self):
        """Test SumFunction with empty summands returns 0.0."""
        func = SumFunction(name="test_sum", summands=[])
        context = {}
        result = func.expression(context)

        # Compile and evaluate
        f = function([], result)
        assert np.isclose(f(), 0.0)

    @pytest.mark.parametrize(
        ("summands", "values", "expected"),
        [
            pytest.param(["term1"], [5.0], 5.0, id="single_summand"),
            pytest.param(["a", "b"], [2.0, 3.0], 5.0, id="two_summands"),
            pytest.param(["x", "y", "z"], [1.0, 2.0, 3.0], 6.0, id="three_summands"),
            pytest.param(
                ["a", "b", "c", "d"], [1.5, 2.5, 3.5, 4.5], 12.0, id="four_summands"
            ),
        ],
    )
    def test_sum_function_expression_multiple_summands(
        self, summands, values, expected
    ):
        """Test SumFunction with different numbers of summands."""
        func = SumFunction(name="test_sum", summands=summands)
        context = {
            summand: pt.constant(value) for summand, value in zip(summands, values)
        }
        result = func.expression(context)

        # Compile and evaluate
        f = function([], result)
        assert np.isclose(f(), expected)

    @pytest.mark.parametrize(
        ("x_val", "y_val", "expected"),
        [
            pytest.param(3.0, 7.0, 10.0, id="positive_values"),
            pytest.param(-2.0, 4.0, 2.0, id="negative_positive"),
            pytest.param(0.0, 5.0, 5.0, id="zero_summand"),
            pytest.param(1.5, 2.5, 4.0, id="decimal_values"),
            pytest.param(-3.0, -2.0, -5.0, id="negative_values"),
        ],
    )
    def test_sum_function_expression_with_variables(self, x_val, y_val, expected):
        """Test SumFunction with variable summands."""
        func = SumFunction(name="test_sum", summands=["x", "y"])
        x = pt.scalar("x")
        y = pt.scalar("y")
        context = {"x": x, "y": y}
        result = func.expression(context)

        # Compile and evaluate
        f = function([x, y], result)
        assert np.isclose(f(x_val, y_val), expected)

    def test_sum_function_single_summand(self):
        """Test SumFunction with single summand returns that value."""
        func = SumFunction(name="test_sum", summands=["single"])

        test_value = 42.0
        context = {"single": pt.constant(test_value)}
        result = func.expression(context)

        f = function([], result)
        assert np.isclose(f(), test_value)

    def test_sum_function_with_many_terms(self):
        """Test SumFunction with many summands."""
        # Create function with 10 summands
        summands = [f"term_{i}" for i in range(10)]
        func = SumFunction(name="many_sum", summands=summands)

        # All terms equal to 1.0, so sum should be 10.0
        context = {term: pt.constant(1.0) for term in summands}
        result = func.expression(context)

        f = function([], result)
        assert np.isclose(f(), 10.0)

    def test_sum_function_integration_with_workspace(self):
        """Test SumFunction integration in full Workspace workflow."""
        test_data = {
            "parameter_points": [
                {
                    "name": "test_params",
                    "parameters": [
                        {"name": "param1", "value": 2.0},
                        {"name": "param2", "value": 3.0},
                        {"name": "param3", "value": 5.0},
                    ],
                }
            ],
            "functions": [
                {
                    "type": "sum",
                    "name": "total_sum",
                    "summands": ["param1", "param2", "param3"],
                }
            ],
            "distributions": [],
            "domains": [{"name": "test_domain", "type": "product_domain", "axes": []}],
            "metadata": {"name": "test"},
        }

        ws = Workspace(test_data)
        model = ws.model(domain="test_domain", parameter_point="test_params")

        # Verify the function was created
        assert "total_sum" in model.functions
        assert "param1" in model.parameters
        assert "param2" in model.parameters
        assert "param3" in model.parameters

        # Evaluate function - should be 2.0 + 3.0 + 5.0 = 10.0
        # Get the function expression and compile with parameter values
        param_values = {"param1": 2.0, "param2": 3.0, "param3": 5.0}

        # Use model's evaluation with parameter values
        func_expr = model.functions["total_sum"]
        param_tensors = [
            model.parameters[name] for name in ["param1", "param2", "param3"]
        ]
        param_vals = [param_values[name] for name in ["param1", "param2", "param3"]]

        f = function(param_tensors, func_expr)
        result = f(*param_vals)
        assert np.isclose(result, 10.0)


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


class TestProcessNormalizationFunction:
    """Test ProcessNormalizationFunction implementation."""

    def test_process_normalization_creation(self):
        """Test ProcessNormalizationFunction can be created and configured."""
        func = ProcessNormalizationFunction(
            name="test_norm",
            expression="test_expr",
            nominalValue=1.0,
            thetaList=["theta1", "theta2"],
            logKappa=[0.1, 0.2],
            asymmThetaList=["asym1"],
            logAsymmKappa=[[0.05, 0.15]],
            otherFactorList=["factor1"],
        )
        assert func.name == "test_norm"
        assert func.expression_name == "test_expr"
        assert func.nominalValue == 1.0
        assert func.thetaList == ["theta1", "theta2"]
        assert func.logKappa == [0.1, 0.2]
        assert func.asymmThetaList == ["asym1"]
        assert func.logAsymmKappa == [[0.05, 0.15]]
        assert func.otherFactorList == ["factor1"]
        assert set(func.parameters) == {"theta1", "theta2", "asym1", "factor1"}

    def test_process_normalization_from_dict(self):
        """Test ProcessNormalizationFunction can be created from dictionary."""
        config = {
            "name": "test_norm",
            "expression": "test_expr",
            "nominalValue": 2.5,
            "thetaList": ["sym1", "sym2"],
            "logKappa": [0.3, 0.4],
            "asymmThetaList": ["asym1", "asym2"],
            "logAsymmKappa": [[0.1, 0.2], [0.15, 0.25]],
            "otherFactorList": ["other1", "other2"],
        }
        func = ProcessNormalizationFunction.from_dict(config)
        assert func.name == "test_norm"
        assert func.expression_name == "test_expr"
        assert func.nominalValue == 2.5
        assert func.thetaList == ["sym1", "sym2"]
        assert func.logKappa == [0.3, 0.4]
        assert func.asymmThetaList == ["asym1", "asym2"]
        assert func.logAsymmKappa == [[0.1, 0.2], [0.15, 0.25]]
        assert func.otherFactorList == ["other1", "other2"]

    def test_process_normalization_nominal_only(self):
        """Test ProcessNormalizationFunction with only nominal value."""
        func = ProcessNormalizationFunction(
            name="nominal_only",
            expression="test",
            nominalValue=5.0,
            thetaList=[],
            logKappa=[],
            asymmThetaList=[],
            logAsymmKappa=[],
            otherFactorList=[],
        )

        context = {}
        result = func.expression(context)
        f = function([], result)

        # Should return nominal value
        assert np.isclose(f(), 5.0)

    def test_process_normalization_symmetric_variations(self):
        """Test ProcessNormalizationFunction with symmetric variations only."""
        func = ProcessNormalizationFunction(
            name="sym_test",
            expression="test",
            nominalValue=2.0,
            thetaList=["theta1", "theta2"],
            logKappa=[0.1, 0.2],  # log-normal variations
            asymmThetaList=[],
            logAsymmKappa=[],
            otherFactorList=[],
        )

        # Test with specific theta values
        context = {
            "theta1": pt.constant(1.0),  # +1 sigma
            "theta2": pt.constant(-0.5),  # -0.5 sigma
        }

        result = func.expression(context)
        f = function([], result)
        result_val = f()

        # Expected: 2.0 * exp(0.1 * 1.0 + 0.2 * (-0.5)) = 2.0 * exp(0.1 - 0.1) = 2.0 * exp(0) = 2.0
        expected = 2.0 * np.exp(0.1 * 1.0 + 0.2 * (-0.5))
        np.testing.assert_allclose(result_val, expected, rtol=1e-10)

    def test_process_normalization_asymmetric_interpolation_at_zero(self):
        """Test ProcessNormalizationFunction asymmetric interpolation at theta=0."""
        func = ProcessNormalizationFunction(
            name="asym_test",
            expression="test",
            nominalValue=3.0,
            thetaList=[],
            logKappa=[],
            asymmThetaList=["asym_theta"],
            logAsymmKappa=[[0.1, 0.3]],  # [low, high] kappa values
            otherFactorList=[],
        )

        # Test at theta=0 (should return nominal)
        context = {"asym_theta": pt.constant(0.0)}

        result = func.expression(context)
        f = function([], result)
        result_val = f()

        # At theta=0, asymmetric interpolation polynomial gives 15.0/8.0 = 1.875
        # So morph = 0.5 * (0.2 * 0 + 0.4 * 1.875) = 0.5 * 0.75 = 0.375
        # Expected: 3.0 * exp(0.375)
        expected = 3.0 * np.exp(0.375)
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_process_normalization_asymmetric_interpolation_positive(self):
        """Test ProcessNormalizationFunction asymmetric interpolation with positive theta."""
        func = ProcessNormalizationFunction(
            name="asym_test",
            expression="test",
            nominalValue=1.0,
            thetaList=[],
            logKappa=[],
            asymmThetaList=["asym_theta"],
            logAsymmKappa=[[0.1, 0.3]],  # [low, high] kappa values
            otherFactorList=[],
        )

        # Test with theta=0.5 (polynomial region: |theta| < 1)
        theta_val = 0.5
        context = {"asym_theta": pt.constant(theta_val)}

        result = func.expression(context)
        f = function([], result)
        result_val = f()

        # Manual calculation of asymmetric interpolation
        kappa_lo, kappa_hi = 0.1, 0.3
        kappa_sum = kappa_hi + kappa_lo  # 0.4
        kappa_diff = kappa_hi - kappa_lo  # 0.2

        # Polynomial interpolation for |theta| < 1
        poly_result = (3.0 * theta_val**4 - 10.0 * theta_val**2 + 15.0) / 8.0
        smooth_function = poly_result  # since |0.5| < 1

        # Asymmetric interpolation
        morph = 0.5 * (kappa_diff * theta_val + kappa_sum * smooth_function)
        expected = 1.0 * np.exp(morph)

        np.testing.assert_allclose(result_val, expected, rtol=1e-8)

    def test_process_normalization_asymmetric_interpolation_extrapolation(self):
        """Test ProcessNormalizationFunction asymmetric interpolation in extrapolation region."""
        func = ProcessNormalizationFunction(
            name="asym_test",
            expression="test",
            nominalValue=1.0,
            thetaList=[],
            logKappa=[],
            asymmThetaList=["asym_theta"],
            logAsymmKappa=[[0.2, 0.4]],
            otherFactorList=[],
        )

        # Test with theta=2.0 (extrapolation region: |theta| > 1)
        theta_val = 2.0
        context = {"asym_theta": pt.constant(theta_val)}

        result = func.expression(context)
        f = function([], result)
        result_val = f()

        # Manual calculation
        kappa_lo, kappa_hi = 0.2, 0.4
        kappa_sum = kappa_hi + kappa_lo  # 0.6
        kappa_diff = kappa_hi - kappa_lo  # 0.2

        # Linear extrapolation for |theta| > 1
        abs_theta = abs(theta_val)  # 2.0
        smooth_function = abs_theta  # since |2.0| > 1

        # Asymmetric interpolation
        morph = 0.5 * (kappa_diff * theta_val + kappa_sum * smooth_function)
        # = 0.5 * (0.2 * 2.0 + 0.6 * 2.0) = 0.5 * (0.4 + 1.2) = 0.5 * 1.6 = 0.8
        expected = 1.0 * np.exp(morph)

        np.testing.assert_allclose(result_val, expected, rtol=1e-8)

    def test_process_normalization_other_factors(self):
        """Test ProcessNormalizationFunction with additional multiplicative factors."""
        func = ProcessNormalizationFunction(
            name="factor_test",
            expression="test",
            nominalValue=2.0,
            thetaList=[],
            logKappa=[],
            asymmThetaList=[],
            logAsymmKappa=[],
            otherFactorList=["factor1", "factor2"],
        )

        context = {
            "factor1": pt.constant(1.5),
            "factor2": pt.constant(3.0),
        }

        result = func.expression(context)
        f = function([], result)
        result_val = f()

        # Expected: 2.0 * 1.5 * 3.0 = 9.0
        expected = 2.0 * 1.5 * 3.0
        np.testing.assert_allclose(result_val, expected, rtol=1e-10)

    def test_process_normalization_combined_all_features(self):
        """Test ProcessNormalizationFunction with all features combined."""
        func = ProcessNormalizationFunction(
            name="combined_test",
            expression="test",
            nominalValue=1.0,
            thetaList=["sym1"],
            logKappa=[0.1],
            asymmThetaList=["asym1"],
            logAsymmKappa=[[0.05, 0.15]],
            otherFactorList=["factor1"],
        )

        context = {
            "sym1": pt.constant(0.5),  # Symmetric variation
            "asym1": pt.constant(0.3),  # Asymmetric variation (polynomial region)
            "factor1": pt.constant(2.0),  # Additional factor
        }

        result = func.expression(context)
        f = function([], result)
        result_val = f()

        # Manual calculation
        # 1. Nominal: 1.0
        # 2. Symmetric: exp(0.1 * 0.5) = exp(0.05)
        # 3. Asymmetric: need to compute interpolation for theta=0.3
        kappa_sum = 0.15 + 0.05  # 0.2
        kappa_diff = 0.15 - 0.05  # 0.1
        poly_result = (3.0 * 0.3**4 - 10.0 * 0.3**2 + 15.0) / 8.0
        asym_morph = 0.5 * (kappa_diff * 0.3 + kappa_sum * poly_result)
        # 4. Factor: 2.0

        expected = 1.0 * np.exp(0.05 + asym_morph) * 2.0
        np.testing.assert_allclose(result_val, expected, rtol=1e-8)

    def test_process_normalization_negative_theta_asymmetric(self):
        """Test ProcessNormalizationFunction with negative theta for asymmetric interpolation."""
        func = ProcessNormalizationFunction(
            name="neg_asym_test",
            expression="test",
            nominalValue=1.0,
            thetaList=[],
            logKappa=[],
            asymmThetaList=["asym_theta"],
            logAsymmKappa=[[0.1, 0.3]],
            otherFactorList=[],
        )

        # Test with negative theta in polynomial region
        theta_val = -0.7
        context = {"asym_theta": pt.constant(theta_val)}

        result = func.expression(context)
        f = function([], result)
        result_val = f()

        # The asymmetric interpolation should handle negative values correctly
        # The smooth function uses abs(theta) for the polynomial, but theta itself for the difference term
        assert result_val > 0.0  # Should be positive
        assert np.isfinite(result_val)  # Should be finite

    def test_process_normalization_integration_with_workspace(self):
        """Test ProcessNormalizationFunction integration in full Workspace workflow."""
        test_data = {
            "parameter_points": [
                {
                    "name": "test_params",
                    "parameters": [
                        {"name": "sym_nuisance", "value": 0.5},
                        {"name": "asym_nuisance", "value": 0.3},
                        {"name": "other_factor", "value": 1.2},
                    ],
                }
            ],
            "functions": [
                {
                    "type": "CMS::process_normalization",
                    "name": "process_norm",
                    "expression": "norm_expr",
                    "nominalValue": 100.0,
                    "thetaList": ["sym_nuisance"],
                    "logKappa": [0.2],
                    "asymmThetaList": ["asym_nuisance"],
                    "logAsymmKappa": [[0.1, 0.3]],
                    "otherFactorList": ["other_factor"],
                }
            ],
            "distributions": [],
            "domains": [{"name": "test_domain", "type": "product_domain", "axes": []}],
            "metadata": {"name": "test"},
        }

        ws = Workspace(test_data)
        model = ws.model(domain="test_domain", parameter_point="test_params")

        # Verify the function was created
        assert "process_norm" in model.functions
        assert "sym_nuisance" in model.parameters
        assert "asym_nuisance" in model.parameters
        assert "other_factor" in model.parameters

        # Evaluate function with parameter values
        param_values = {"sym_nuisance": 0.5, "asym_nuisance": 0.3, "other_factor": 1.2}

        func_expr = model.functions["process_norm"]
        param_tensors = [model.parameters[name] for name in param_values]
        param_vals = [param_values[name] for name in param_values]

        f = function(param_tensors, func_expr)
        result = f(*param_vals)

        # Should be positive and finite
        assert result > 0.0
        assert np.isfinite(result)
        # Should be close to nominal * factor since variations are small
        assert 100.0 < result < 200.0  # Rough sanity check


class TestRegisteredFunctions:
    """Test the registered_functions registry."""

    def test_registry_contains_expected_functions(self):
        """Test that registry contains all expected function types."""
        expected_types = {
            "product",
            "sum",
            "generic_function",
            "interpolation",
            "CMS::process_normalization",
        }
        assert set(registered_functions.keys()) == expected_types

    @pytest.mark.parametrize(
        ("func_type", "expected_class"),
        [
            pytest.param("product", ProductFunction, id="product_function"),
            pytest.param("sum", SumFunction, id="sum_function"),
            pytest.param("generic_function", GenericFunction, id="generic_function"),
            pytest.param(
                "interpolation", InterpolationFunction, id="interpolation_function"
            ),
            pytest.param(
                "CMS::process_normalization",
                ProcessNormalizationFunction,
                id="process_normalization_function",
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
                "nom": "n1",
                "interpolationCodes": [0],
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
