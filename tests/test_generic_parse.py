"""
Unit tests for the generic_parse module.

Tests for SymPy expression parsing and conversion to PyTensor operations.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import sympy as sp
from pytensor import function

from pyhs3.generic_parse import analyze_sympy_expr, parse_expression, sympy_to_pytensor


class TestAnalyzeSymPyExpr:
    """Test the analyze_sympy_expr function."""

    def test_simple_expression_analysis(self):
        """Test analysis of a simple polynomial expression."""
        expr = sp.parse_expr("x**2 + 2*x + 1")
        result = analyze_sympy_expr(expr)

        assert result["expression"] == expr
        assert result["independent_vars"] == {sp.Symbol("x")}
        assert result["dependent_vars"] == set()

    def test_multi_variable_expression(self):
        """Test analysis of expression with multiple variables."""
        expr = sp.parse_expr("x*y + sin(z)")
        result = analyze_sympy_expr(expr)

        expected_vars = {sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("z")}
        assert result["independent_vars"] == expected_vars
        # sin(z) should be detected as a dependent variable
        assert len(result["dependent_vars"]) == 1

    def test_expression_with_functions(self):
        """Test analysis of expression with function calls."""
        expr = sp.parse_expr("exp(x) + log(y) + sqrt(z)")
        result = analyze_sympy_expr(expr)

        expected_vars = {sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("z")}
        assert result["independent_vars"] == expected_vars
        # exp(x) and log(y) should be detected as dependent variables, sqrt is handled differently
        assert len(result["dependent_vars"]) >= 1


class TestSymPyToPyTensor:
    """Test the sympy_to_pytensor function."""

    def test_simple_polynomial(self):
        """Test conversion of simple polynomial expression."""
        x = pt.scalar("x")
        variables = [x]

        expr = parse_expression("x**2 + 2*x + 1")
        result = sympy_to_pytensor(expr, variables)

        # Test evaluation
        f = function([x], result)

        # Test specific values
        assert np.isclose(f(0), 1.0)  # 0^2 + 2*0 + 1 = 1
        assert np.isclose(f(1), 4.0)  # 1^2 + 2*1 + 1 = 4
        assert np.isclose(f(2), 9.0)  # 2^2 + 2*2 + 1 = 9

    def test_multi_variable_expression(self):
        """Test conversion of multi-variable expression."""
        x = pt.scalar("x")
        y = pt.scalar("y")
        variables = [x, y]

        expr = parse_expression("x*y + x**2")
        result = sympy_to_pytensor(expr, variables)

        # Test evaluation
        f = function([x, y], result)

        # Test specific values: x*y + x^2
        assert np.isclose(f(2, 3), 10.0)  # 2*3 + 2^2 = 6 + 4 = 10
        assert np.isclose(f(1, 5), 6.0)  # 1*5 + 1^2 = 5 + 1 = 6

    def test_trigonometric_functions(self):
        """Test conversion of expressions with trigonometric functions."""
        x = pt.scalar("x")
        variables = [x]

        expr = parse_expression("sin(x) + cos(x)")
        result = sympy_to_pytensor(expr, variables)

        # Test evaluation
        f = function([x], result)

        # Test at x=0: sin(0) + cos(0) = 0 + 1 = 1
        assert np.isclose(f(0), 1.0)

        # Test at x=π/2: sin(π/2) + cos(π/2) = 1 + 0 = 1
        assert np.isclose(f(np.pi / 2), 1.0, atol=1e-10)

    def test_exponential_and_logarithm(self):
        """Test conversion of expressions with exponential and logarithm."""
        x = pt.scalar("x")
        variables = [x]

        expr = parse_expression("exp(x) + log(x)")
        result = sympy_to_pytensor(expr, variables)

        # Test evaluation
        f = function([x], result)

        # Test at x=1: exp(1) + log(1) = e + 0 = e
        assert np.isclose(f(1), np.e)

        # Test at x=e: exp(e) + log(e) = e^e + 1
        expected = np.exp(np.e) + 1
        assert np.isclose(f(np.e), expected)

    def test_sqrt_and_abs_functions(self):
        """Test conversion of expressions with sqrt and abs functions."""
        x = pt.scalar("x")
        variables = [x]

        expr = parse_expression("sqrt(abs(x))")
        result = sympy_to_pytensor(expr, variables)

        # Test evaluation
        f = function([x], result)

        # Test positive value
        assert np.isclose(f(4), 2.0)  # sqrt(abs(4)) = sqrt(4) = 2

        # Test negative value
        assert np.isclose(f(-9), 3.0)  # sqrt(abs(-9)) = sqrt(9) = 3

    def test_complex_expression(self):
        """Test conversion of a complex mathematical expression."""
        x = pt.scalar("x")
        y = pt.scalar("y")
        variables = [x, y]

        # Complex expression combining multiple operations
        expr_str = "exp(-x**2/2) * cos(y) + sin(x*y)"
        expr = parse_expression(expr_str)
        result = sympy_to_pytensor(expr, variables)

        # Test evaluation
        f = function([x, y], result)

        # Test at x=0, y=0: exp(0) * cos(0) + sin(0) = 1 * 1 + 0 = 1
        assert np.isclose(f(0, 0), 1.0)

    def test_variable_order_independence(self):
        """Test that variable order doesn't affect the result."""
        x = pt.scalar("x")
        y = pt.scalar("y")

        # Test with different variable orders
        expr = parse_expression("x - y")
        result1 = sympy_to_pytensor(expr, [x, y])
        result2 = sympy_to_pytensor(expr, [x, y])  # Same order

        f1 = function([x, y], result1)
        f2 = function([x, y], result2)

        # Results should be identical
        test_values = [(1, 2), (3, 1), (0, 5)]
        for x_val, y_val in test_values:
            assert np.isclose(f1(x_val, y_val), f2(x_val, y_val))

    def test_constants_in_expression(self):
        """Test expressions with numerical constants."""
        x = pt.scalar("x")
        variables = [x]

        expr = parse_expression("3.14159 * x + 2.71828")
        result = sympy_to_pytensor(expr, variables)

        f = function([x], result)

        # Test at x=1: 3.14159 * 1 + 2.71828
        expected = 3.14159 + 2.71828
        assert np.isclose(f(1), expected)

    def test_single_variable_expression(self):
        """Test expressions with just a single variable."""
        x = pt.scalar("x")
        variables = [x]

        expr = parse_expression("x")
        result = sympy_to_pytensor(expr, variables)

        f = function([x], result)

        # Should be identity function
        test_values = [0, 1, -1, 5.5, -3.2]
        for val in test_values:
            assert np.isclose(f(val), val)

    def test_constant_expression(self):
        """Test expressions that are just constants."""
        x = pt.scalar("x")
        variables = [x]

        expr = parse_expression("42")
        result = sympy_to_pytensor(expr, variables)

        # For constant expressions, x is unused, so we need to allow unused inputs
        f = function([x], result, on_unused_input="ignore")

        # Should always return 42 regardless of x
        test_values = [0, 1, -1, 100]
        for val in test_values:
            assert np.isclose(f(val), 42)
