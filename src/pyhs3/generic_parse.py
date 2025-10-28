from __future__ import annotations

import logging
from typing import Any, cast

import pytensor.tensor as pt
import sympy as sp
from pytensor.tensor.type import TensorType
from sympy.parsing import sympy_parser
from sympy.parsing.sympy_parser import (
    auto_number,
    auto_symbol,
    factorial_notation,
    lambda_notation,
    repeated_decimals,
)

from pyhs3.exceptions import ExpressionEvaluationError, ExpressionParseError

log = logging.getLogger(__name__)


def analyze_sympy_expr(sympy_expr: sp.Expr) -> dict[str, Any]:
    """
    Analyzes a SymPy expression and logs its independent variables,
    dependent variables, and structure for debugging.

    Args:
        sympy_expr: The SymPy expression to analyze.

    Returns:
        Dictionary containing analysis results with keys:
        - 'expression': The original expression
        - 'independent_vars': Set of independent variables (symbols)
        - 'dependent_vars': Set of dependent variables (functions)
    """
    # Independent variables (symbols in the expression)
    independent_vars = sympy_expr.free_symbols

    # Dependent variables (functions of symbols)
    dependent_vars = sympy_expr.atoms(sp.Function)

    # Log information for debugging
    log.debug("Expression: %s", sympy_expr)
    log.debug("Independent Variables: %s", independent_vars)
    log.debug("Dependent Variables: %s", dependent_vars)
    log.debug("Expression Structure:\n%s", sp.pretty(sympy_expr))

    return {
        "expression": sympy_expr,
        "independent_vars": independent_vars,
        "dependent_vars": dependent_vars,
    }


def parse_expression(expr_str: str) -> sp.Expr:
    """
    Parse a mathematical expression string into a SymPy expression.

    Args:
        expr_str: The mathematical expression as a string.

    Returns:
        SymPy expression object.

    Raises:
        ExpressionParseError: If the expression cannot be parsed.
    """
    transformations = (
        auto_symbol,
        lambda_notation,
        repeated_decimals,
        auto_number,
        factorial_notation,
    )

    # Provide global_dict with Symbol, Integer, etc. so that:
    # 1. auto_number transformation can work properly
    # 2. Unknown variables become Symbols (not Functions)
    global_dict = {
        "Symbol": sp.Symbol,
        "Integer": sp.Integer,
        "Float": sp.Float,
        "Rational": sp.Rational,
    }

    try:
        return sympy_parser.parse_expr(
            expr_str,
            transformations=transformations,
            global_dict=global_dict,
        )
    except Exception as exc:
        msg = f"Failed to parse expression '{expr_str}': {exc}"
        raise ExpressionParseError(msg) from exc


def sympy_to_pytensor(
    sympy_expr: sp.Expr,
    variables: list[pt.variable.TensorVariable[TensorType, Any]],
) -> pt.variable.TensorVariable[TensorType, Any]:
    """
    Converts a SymPy expression into a PyTensor computational graph using lambdify.

    Args:
        sympy_expr: The SymPy expression object.
        variables: List of PyTensor variables.

    Returns:
        PyTensor expression.

    Raises:
        ExpressionEvaluationError: If the expression cannot be converted or contains unsupported operations.
    """
    try:
        # Define the mapping for SymPy functions to PyTensor functions (using pt.math)
        custom_modules = {
            "sin": pt.math.sin,
            "cos": pt.math.cos,
            "tan": pt.math.tan,
            "exp": pt.math.exp,
            "log": pt.math.log,
            "sqrt": pt.math.sqrt,
            "abs": pt.math.abs,
            "erf": pt.math.erf,
            "min": pt.math.min,
            "max": pt.math.max,
        }

        # Convert variable names to SymPy symbols
        sympy_vars = {var.name: sp.Symbol(var.name) for var in variables}

        # Log the expression for debugging
        analyze_sympy_expr(sympy_expr)

        # Convert SymPy expression to a PyTensor-compatible function
        pytensor_func = sp.lambdify(
            list(sympy_vars.values()), sympy_expr, modules=custom_modules
        )

        # Apply the function to PyTensor variables
        result = pytensor_func(*variables)

        # Handle case where result is a constant (not a PyTensor variable)
        if not isinstance(result, pt.variable.TensorVariable):
            result = pt.constant(result)

        return cast(pt.variable.TensorVariable[TensorType, Any], result)

    except Exception as exc:
        msg = f"Failed to convert expression to PyTensor: {sympy_expr}. {exc}"
        raise ExpressionEvaluationError(msg) from exc
