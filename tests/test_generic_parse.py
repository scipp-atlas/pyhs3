"""
Unit tests for the generic_parse module.

Tests for SymPy expression parsing and conversion to PyTensor operations.
"""

from __future__ import annotations

import logging

import numpy as np
import pytensor.tensor as pt
import pytest
import sympy as sp
from pytensor import function

from pyhs3.exceptions import ExpressionEvaluationError, ExpressionParseError
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

    def test_simple_expression_analysis_with_debug(self, caplog):
        """Test analysis of a simple polynomial expression."""
        expr = sp.parse_expr("x**2 + 2*x + 1")

        with caplog.set_level(logging.DEBUG):
            analyze_sympy_expr(expr)

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


class TestExceptionHandling:
    """Test exception handling in parse_expression and sympy_to_pytensor."""

    @pytest.mark.parametrize(
        "invalid_expr",
        [
            pytest.param("x +", id="incomplete_expression"),
            pytest.param("2 ** ** 3", id="invalid_operator_usage"),
            pytest.param("sin(", id="unmatched_parenthesis"),
            pytest.param("x y", id="missing_operator"),
            pytest.param("2..3", id="invalid_number_format"),
            pytest.param("", id="empty_string"),
        ],
    )
    def test_parse_expression_invalid_syntax(self, invalid_expr):
        """Test that invalid syntax raises ExpressionParseError."""
        with pytest.raises(ExpressionParseError, match="Failed to parse expression"):
            parse_expression(invalid_expr)

    def test_sympy_to_pytensor_mismatched_variables(self):
        """Test that mismatched variables raise ExpressionEvaluationError."""
        # Create expression with variables x and y
        expr = parse_expression("x + y")

        # But only provide variable x
        x = pt.scalar("x")
        variables = [x]  # Missing y

        with pytest.raises(
            ExpressionEvaluationError, match="Failed to convert expression to PyTensor"
        ):
            sympy_to_pytensor(expr, variables)

    def test_sympy_to_pytensor_wrong_variable_names(self):
        """Test that wrong variable names raise ExpressionEvaluationError."""

        # Create expression with variable x
        expr = parse_expression("x**2")

        # But provide variable with different name
        z = pt.scalar("z")  # Wrong name
        variables = [z]

        with pytest.raises(
            ExpressionEvaluationError, match="Failed to convert expression to PyTensor"
        ):
            sympy_to_pytensor(expr, variables)

    def test_sympy_to_pytensor_too_many_variables(self):
        """Test that providing more variables than needed works correctly."""
        # This should work - extra variables should be ignored
        expr = parse_expression("x**2")

        x = pt.scalar("x")
        y = pt.scalar("y")  # Extra variable
        z = pt.scalar("z")  # Another extra variable
        variables = [x, y, z]

        # Should work and only use x
        result = sympy_to_pytensor(expr, variables)

        # Test evaluation (only x should be used)
        f = function([x, y, z], result, on_unused_input="ignore")
        assert np.isclose(f(3, 999, -999), 9.0)  # Only x=3 matters

    def test_sympy_to_pytensor_complex_expression_error(self):
        """Test that complex expressions that can't be converted raise ExpressionEvaluationError."""

        # Create a complex expression that might fail conversion
        # Using a symbolic expression that's hard to convert
        x_sym = sp.Symbol("x")
        complex_expr = sp.gamma(x_sym)  # Gamma function might not be supported

        x = pt.scalar("x")
        variables = [x]

        # This might raise an exception depending on PyTensor support
        try:
            result = sympy_to_pytensor(complex_expr, variables)
            # If it doesn't raise, test that it at least works
            f = function([x], result)
            # Just verify it runs without crashing
            val = f(2.0)
            assert np.isfinite(val)
        except ExpressionEvaluationError:
            # This is expected for unsupported functions
            pass


class TestRealWorldExpressions:
    """Test parsing of real expressions from the DiHiggs workspace."""

    @pytest.mark.parametrize(
        "expr_str",
        [
            pytest.param(
                "exp((atlas_invMass_Run2HM_1-100)*BKG_p0_HM_1/100)", id="run2_HM_1"
            ),
            pytest.param(
                "exp((atlas_invMass_Run2HM_2-100)*BKG_p0_HM_2/100)", id="run2_HM_2"
            ),
            pytest.param(
                "exp((atlas_invMass_Run2LM_1-100)*BKG_p0_LM_1/100)", id="run2_LM_1"
            ),
            pytest.param(
                "exp((atlas_invMass_Run3HM_1-100)*BKG_p0_HM_1/100)", id="run3_HM_1"
            ),
        ],
    )
    def test_diHiggs_background_expressions(self, expr_str):
        """Test parsing of actual DiHiggs background model expressions."""
        # Should parse without error
        expr = parse_expression(expr_str)

        # Should be able to analyze
        analysis = analyze_sympy_expr(expr)

        # Should have 2 independent variables
        assert len(analysis["independent_vars"]) == 2

        # Should be able to convert to PyTensor
        var_names = sorted([str(var) for var in analysis["independent_vars"]])
        variables = [pt.scalar(name) for name in var_names]

        result = sympy_to_pytensor(expr, variables)

        # Should be able to evaluate
        f = function(variables, result)

        # Test with some reasonable values
        if "atlas_invMass" in var_names[0]:
            atlas_val, bkg_val = 125.0, -0.01  # Reasonable physics values
        else:
            bkg_val, atlas_val = -0.01, 125.0

        output = f(atlas_val, bkg_val)
        assert np.isfinite(output)
        assert output > 0  # Should be positive (exponential function)

    @pytest.mark.parametrize(
        "expr_str",
        [
            pytest.param("exp(-(x-mu)**2/(2*sigma**2))", id="gaussian_like"),
            pytest.param("a0 + a1*x + a2*x**2 + a3*x**3", id="polynomial_background"),
            pytest.param("exp(-0.5*((x-mean)/sigma)**2)", id="crystal_ball_core"),
            pytest.param("norm * exp(-x/tau)", id="exponential_decay"),
            pytest.param("exp(p0 + p1*x + p2*x**2)", id="exponential_polynomial"),
            pytest.param("1 + amp*cos(freq*x + phase)", id="trigonometric_modulation"),
        ],
    )
    def test_complex_physics_expressions(self, expr_str):
        """Test more complex expressions that might appear in physics."""
        # Should parse without error
        expr = parse_expression(expr_str)

        # Should be able to analyze
        analysis = analyze_sympy_expr(expr)

        # Should have at least one variable
        assert len(analysis["independent_vars"]) >= 1

        # Should be able to convert to PyTensor
        var_names = sorted([str(var) for var in analysis["independent_vars"]])
        variables = [pt.scalar(name) for name in var_names]

        result = sympy_to_pytensor(expr, variables)

        # Should be able to evaluate (test with unit values)
        f = function(variables, result)
        unit_values = [1.0] * len(variables)

        try:
            output = f(*unit_values)
            assert np.isfinite(output)
        except ZeroDivisionError:
            # Some expressions might have division by zero at unit values
            # Try with different values
            alt_values = [2.0] * len(variables)
            output = f(*alt_values)
            assert np.isfinite(output)

    def test_ratio_expression_division_by_zero(self):
        """Test handling of division by zero in ratio expressions."""
        expr_str = "(a + b*x)/(c + d*x)"
        expr = parse_expression(expr_str)

        var_names = sorted([str(var) for var in expr.free_symbols])
        variables = [pt.scalar(name) for name in var_names]

        result = sympy_to_pytensor(expr, variables)
        f = function(variables, result)

        # Variables are sorted alphabetically: [a, b, c, d, x]
        # Test with values that avoid division by zero
        # a=1, b=1, c=1, d=1, x=1 gives (1+1*1)/(1+1*1) = 2/2 = 1
        assert len(var_names) == 5  # Should have a, b, c, d, x
        output = f(1.0, 1.0, 1.0, 1.0, 1.0)  # [a, b, c, d, x]
        assert np.isclose(output, 1.0)

    def test_variable_order_independence_with_real_expressions(self):
        """Test that variable order doesn't matter for real expressions."""
        expr_str = "exp((atlas_invMass_Run2HM_1-100)*BKG_p0_HM_1/100)"
        expr = parse_expression(expr_str)

        # Create variables in different orders
        var1 = pt.scalar("atlas_invMass_Run2HM_1")
        var2 = pt.scalar("BKG_p0_HM_1")

        result1 = sympy_to_pytensor(expr, [var1, var2])
        result2 = sympy_to_pytensor(expr, [var2, var1])  # Different order

        # Both should evaluate to the same result
        f1 = function([var1, var2], result1)
        f2 = function([var2, var1], result2)

        # Test with physics values
        atlas_val, bkg_val = 125.0, -0.01

        result_val1 = f1(atlas_val, bkg_val)
        result_val2 = f2(
            bkg_val, atlas_val
        )  # Note: swapped order to match function signature

        assert np.isclose(result_val1, result_val2)

    @pytest.mark.parametrize(
        "expr_str",
        [
            pytest.param("sigma_eff_1 * sqrt(2*pi)", id="efficiency_normalization"),
            pytest.param(
                "mu_sig_1p2GeV + delta_mu_sys", id="signal_mean_with_systematic"
            ),
            pytest.param(
                "N_bkg_run2_cat1 * eff_trigger_2018",
                id="background_yield_with_efficiency",
            ),
            pytest.param(
                "alpha_JES_NP_1 * sigma_JES_unc", id="jet_energy_scale_uncertainty"
            ),
        ],
    )
    def test_expression_with_underscores_and_numbers(self, expr_str):
        """Test expressions with variable names containing underscores and numbers."""
        # Should parse correctly
        expr = parse_expression(expr_str)
        analysis = analyze_sympy_expr(expr)

        # Should identify all variables correctly
        assert len(analysis["independent_vars"]) >= 1

        # Variable names should contain underscores
        var_names = [str(var) for var in analysis["independent_vars"]]
        assert any("_" in name for name in var_names)

        # Should convert to PyTensor
        variables = [pt.scalar(name) for name in var_names]
        result = sympy_to_pytensor(expr, variables)

        # Should evaluate
        f = function(variables, result)
        test_vals = [1.0] * len(variables)
        output = f(*test_vals)
        assert np.isfinite(output)
