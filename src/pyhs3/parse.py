from __future__ import annotations

from typing import Any, cast

import pytensor
import pytensor.tensor as pt
import sympy as sp
from pytensor.tensor.type import TensorType
from sympy.parsing import sympy_parser
from sympy.parsing.sympy_parser import (
    auto_number,
    factorial_notation,
    lambda_notation,
    repeated_decimals,
)


def analyze_sympy_expr(sympy_expr: sp.Expr) -> None:
    """
    Analyzes a SymPy expression and prints its independent variables,
    dependent variables, and structure.

    Args:
        sympy_expr (sympy.Expr): The mathematical expression as a string.
    """
    # Independent variables (symbols in the expression)
    independent_vars = sympy_expr.free_symbols

    # Dependent variables (functions of symbols)
    dependent_vars = sympy_expr.atoms(sp.Function)

    # Print information
    print(f"Expression: {sympy_expr}")
    print(f"Independent Variables: {independent_vars}")
    print(f"Dependent Variables: {dependent_vars}")

    # Print expression tree
    print("\nExpression Structure (AST-like dump):")
    sp.pprint(sympy_expr)


def sympy_to_pytensor(
    expr_str: str,
    variables: list[pt.variable.TensorVariable[TensorType, Any]],
) -> pt.variable.TensorVariable[TensorType, Any]:
    """
    Converts a SymPy expression string into a PyTensor computational graph using lambdify.

    Args:
        expr_str (str): The mathematical expression as a string.
        variables (dict): Mapping of variable names to PyTensor variables.

    Returns:
        pytensor.tensor.variable.TensorVariable: PyTensor expression.
    """
    # Define the mapping for SymPy functions to PyTensor functions (using pt.math)
    custom_modules = {
        "sin": pt.math.sin,
        "cos": pt.math.cos,
        "tan": pt.math.tan,
        "exp": pt.math.exp,
        "log": pt.math.log,
        "sqrt": pt.math.sqrt,
        "abs": pt.math.abs,
    }

    # Convert variable names to SymPy symbols
    sympy_vars = {var.name: sp.Symbol(var.name) for var in variables}

    # Parse the expression using SymPy
    transformations = (
        lambda_notation,
        repeated_decimals,
        auto_number,
        factorial_notation,
    )  # no auto_symbol
    sympy_expr = sympy_parser.parse_expr(
        expr_str, local_dict=sympy_vars, transformations=transformations
    )
    analyze_sympy_expr(sympy_expr)

    # Convert SymPy expression to a PyTensor-compatible function
    pytensor_func = sp.lambdify(
        list(sympy_vars.values()), sympy_expr, modules=custom_modules
    )

    # Apply the function to PyTensor variables
    return cast(pt.variable.TensorVariable[TensorType, Any], pytensor_func(*variables))


# Define PyTensor variables
x = pt.scalar("x")
y = pt.scalar("y")

# Map variable names to PyTensor variables
var_map = {"x": x, "y": y}

# Convert the parsed expression to PyTensor
pytensor_expr = sympy_to_pytensor("x**2 + 2*x + sin(y)", [x, y])
print(pytensor.printing.pprint(pytensor_expr))
