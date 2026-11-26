from __future__ import annotations

import ast
import logging
import re
from collections.abc import Callable
from functools import lru_cache, reduce
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytensor.tensor as pt
import sympy as sp
from pydantic import ConfigDict, Field, PrivateAttr, model_validator
from pytensor.tensor.type import TensorType
from sympy.logic.boolalg import BooleanAtom, BooleanFunction

from pyhs3.exceptions import ExpressionEvaluationError, ExpressionParseError

if TYPE_CHECKING:
    from pyhs3.context import Context
    from pyhs3.typing.aliases import TensorVar

log = logging.getLogger(__name__)

# HS3 generic expression grammar:
# https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard/blob/master/docs/chapters/3.1_generic_expressions.md
#
# The expression is rewritten token by token into a Python expression (see
# ``_to_python``), parsed with :mod:`ast`, and the Python AST is walked into
# SymPy objects (see ``_SympyBuilder``). Only the nodes handled there are
# accepted. Every name that is not a reserved constant becomes a ``Symbol`` and
# every call must be one of the whitelisted functions, so a variable sharing a
# name with a SymPy special function (``beta``, ``gamma``, ``zeta``, ...)
# stays a free symbol (issue #117).

# C operators with no Python spelling. ``!`` becomes ``~`` rather than ``not``
# because ``~`` binds tighter than a comparison as in C, so ``!x>0`` is
# ``(!x)>0``. ``==`` and ``===`` both mean exact equality: the spec only says
# that ``==`` SHOULD compare within a few ULP.
_OPERATOR_REWRITES = {
    "!==": "!=",
    "===": "==",
    "!=": "!=",
    "&&": " and ",
    "||": " or ",
    "!": "~",
    "^": "**",
}
_OPERATOR_PATTERN = re.compile("|".join(re.escape(op) for op in _OPERATOR_REWRITES))

_CONSTANTS = {"PI": sp.pi, "EULER": sp.E, "TRUE": sp.true, "FALSE": sp.false}

_COMPARISONS = {
    ast.Eq: sp.Eq,
    ast.NotEq: sp.Ne,
    ast.Lt: sp.Lt,
    ast.LtE: sp.Le,
    ast.Gt: sp.Gt,
    ast.GtE: sp.Ge,
}

# name -> (number of arguments, builder)
_FUNCTIONS: dict[str, tuple[int, Any]] = {
    "exp": (1, sp.exp),
    "log": (1, sp.log),
    "sqrt": (1, sp.sqrt),
    "abs": (1, sp.Abs),
    "floor": (1, sp.floor),
    "ceil": (1, sp.ceiling),
    "pow": (2, lambda base, exponent: base**exponent),
    "min": (2, sp.Min),
    "max": (2, sp.Max),
    "sin": (1, sp.sin),
    "cos": (1, sp.cos),
    "tan": (1, sp.tan),
    "asin": (1, sp.asin),
    "acos": (1, sp.acos),
    "atan": (1, sp.atan),
    "erf": (1, sp.erf),
    "ifelse": (3, lambda cond, a, b: sp.Piecewise((a, _as_boolean(cond)), (b, True))),
}


def _is_boolean(expr: sp.Basic) -> bool:
    return bool(expr.is_Relational) or isinstance(expr, (BooleanFunction, BooleanAtom))


def _as_boolean(expr: sp.Basic) -> sp.Basic:
    """Return *expr* if it is a boolean; a real-valued expression is not one."""
    if not _is_boolean(expr):
        msg = f"expected a boolean expression, got '{expr}'"
        raise ValueError(msg)
    return expr


def _top_level(text: str) -> list[tuple[int, str]]:
    """Return ``(index, character)`` for every character outside brackets.

    Raises:
        ValueError: If the brackets in *text* are unbalanced.
    """
    depth = 0
    chars = []
    for i, char in enumerate(text):
        if char == ")":
            depth -= 1
            if depth < 0:
                msg = "unbalanced brackets"
                raise ValueError(msg)
        elif depth == 0 and char != "(":
            chars.append((i, char))
        if char == "(":
            depth += 1
    if depth:
        msg = "unbalanced brackets"
        raise ValueError(msg)
    return chars


def _matching_bracket(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return i
    msg = "unbalanced brackets"
    raise ValueError(msg)


def _rewrite_ternary(text: str) -> str:
    """Rewrite every ``c ? a : b`` in *text* to ``ifelse(c, a, b)``.

    ``?:`` has the lowest precedence and is right-associative, so ``text`` is
    split at the first ``?`` outside brackets and at the ``:`` that matches it.
    Bracketed groups are rewritten first and comma-separated arguments
    separately, so a conditional inside a function argument stays inside it.
    """
    pieces = []
    i = 0
    while i < len(text):
        if text[i] == "(":
            end = _matching_bracket(text, i)
            pieces.append(f"({_rewrite_ternary(text[i + 1 : end])})")
            i = end + 1
        else:
            pieces.append(text[i])
            i += 1
    text = "".join(pieces)

    commas = [i for i, char in _top_level(text) if char == ","]
    bounds = [-1, *commas, len(text)]
    return ",".join(
        _rewrite_conditional(text[start + 1 : stop]) for start, stop in pairwise(bounds)
    )


def _rewrite_conditional(text: str) -> str:
    """Rewrite one comma-free ``c ? a : b`` whose bracketed groups are done."""
    question = next((i for i, char in _top_level(text) if char == "?"), None)
    if question is None:
        return text

    nested = 0
    for i, char in _top_level(text):
        if i < question:
            continue
        if char == "?":
            nested += 1
        elif char == ":":
            nested -= 1
            if nested == 0:
                condition = text[:question]
                if_true = _rewrite_conditional(text[question + 1 : i])
                if_false = _rewrite_conditional(text[i + 1 :])
                return f"ifelse({condition}, {if_true}, {if_false})"
    msg = "'?' without a matching ':'"
    raise ValueError(msg)


def _to_python(expr_str: str) -> str:
    """Rewrite the HS3 generic expression *expr_str* to Python syntax."""
    python = _OPERATOR_PATTERN.sub(
        lambda match: _OPERATOR_REWRITES[match.group()], expr_str
    )
    return _rewrite_ternary(python).strip()


class _SympyBuilder:
    """Build a SymPy expression from a Python AST, accepting only HS3 grammar."""

    def build(self, node: ast.AST) -> sp.Basic:
        try:
            method = getattr(self, f"_build_{type(node).__name__}")
        except AttributeError as exc:
            msg = f"unsupported syntax '{type(node).__name__}'"
            raise ValueError(msg) from exc
        return cast(sp.Basic, method(node))

    def _build_Expression(self, node: ast.Expression) -> sp.Basic:
        return self.build(node.body)

    def _build_Constant(self, node: ast.Constant) -> sp.Basic:
        # bool is a subclass of int, but TRUE/FALSE are the only boolean literals
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return sp.Integer(node.value)
        if isinstance(node.value, float):
            return sp.Float(node.value)
        msg = f"unsupported literal {node.value!r}"
        raise ValueError(msg)

    def _build_Name(self, node: ast.Name) -> sp.Basic:
        # ``sp.false`` is falsy, so test membership rather than use ``or``
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        return sp.Symbol(node.id)

    def _build_BinOp(self, node: ast.BinOp) -> sp.Basic:
        left, right = self.build(node.left), self.build(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        msg = f"unsupported operator '{type(node.op).__name__}'"
        raise ValueError(msg)

    def _build_UnaryOp(self, node: ast.UnaryOp) -> sp.Basic:
        operand = self.build(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.Invert):
            return sp.Not(_as_boolean(operand))
        msg = f"unsupported operator '{type(node.op).__name__}'"
        raise ValueError(msg)

    def _build_BoolOp(self, node: ast.BoolOp) -> sp.Basic:
        operands = [_as_boolean(self.build(value)) for value in node.values]
        return sp.And(*operands) if isinstance(node.op, ast.And) else sp.Or(*operands)

    def _build_Compare(self, node: ast.Compare) -> sp.Basic:
        operands = [self.build(node.left), *map(self.build, node.comparators)]
        pairs = []
        for op, (left, right) in zip(node.ops, pairwise(operands), strict=True):
            if type(op) not in _COMPARISONS:
                msg = f"unsupported comparison '{type(op).__name__}'"
                raise ValueError(msg)
            pairs.append(_COMPARISONS[type(op)](left, right))
        return sp.And(*pairs)

    def _build_Call(self, node: ast.Call) -> sp.Basic:
        if not isinstance(node.func, ast.Name) or node.keywords:
            msg = "unsupported function call"
            raise ValueError(msg)
        if node.func.id not in _FUNCTIONS:
            msg = f"unknown function '{node.func.id}'"
            raise ValueError(msg)
        nargs, builder = _FUNCTIONS[node.func.id]
        if len(node.args) != nargs:
            msg = f"{node.func.id}() takes {nargs} argument(s), got {len(node.args)}"
            raise ValueError(msg)
        return cast(sp.Basic, builder(*map(self.build, node.args)))


# SymPy node type -> PyTensor function taking the printed arguments.
# Mapping adapted from sympytensor
# (https://github.com/jessegrabowski/sympytensor/blob/main/sympytensor/pytensor.py,
# MIT License). Every function is elementwise, so scalar and vector ``(N,)``
# inputs broadcast against each other.
_PYTENSOR_MAPPING: dict[type, Callable[..., Any]] = {
    sp.Add: pt.add,
    sp.Mul: pt.mul,
    sp.Pow: pt.pow,
    sp.Abs: pt.abs,
    sp.floor: pt.floor,
    sp.ceiling: pt.ceil,
    sp.exp: pt.exp,
    sp.log: pt.log,
    sp.sin: pt.sin,
    sp.cos: pt.cos,
    sp.tan: pt.tan,
    sp.asin: pt.arcsin,
    sp.acos: pt.arccos,
    sp.atan: pt.arctan,
    sp.erf: pt.erf,
    sp.Min: lambda *args: reduce(pt.minimum, args),
    sp.Max: lambda *args: reduce(pt.maximum, args),
    sp.And: lambda *args: reduce(pt.and_, args),
    sp.Or: lambda *args: reduce(pt.or_, args),
    sp.Not: pt.invert,
    sp.Eq: pt.eq,
    sp.Ne: pt.neq,
    sp.Lt: pt.lt,
    sp.Le: pt.le,
    sp.Gt: pt.gt,
    sp.Ge: pt.ge,
}


def _to_pytensor(node: sp.Basic, symbols: dict[str, Any]) -> Any:
    """
    Walk a SymPy expression into PyTensor, binding symbols by name.

    Args:
        node: The SymPy node to convert.
        symbols: Mapping from symbol name to the PyTensor variable it stands for.
            A variable may be a scalar or a vector of shape ``(N,)``; every
            operation is elementwise, so a ``Piecewise`` over a vector yields
            a vector of shape ``(N,)``.

    Raises:
        KeyError: If a symbol has no entry in *symbols*.
        NotImplementedError: If *node* has no PyTensor equivalent.
    """
    if isinstance(node, sp.Symbol):
        return symbols[node.name]
    if node is sp.true or node is sp.false:
        return pt.constant(bool(node))
    if isinstance(node, (sp.Number, sp.NumberSymbol)):
        # float64 explicitly: PyTensor infers int8 or float32 for a number
        # those represent exactly, so sqrt(2) would evaluate in float32
        return pt.constant(float(node), dtype="float64")
    if isinstance(node, sp.Piecewise):
        # Built from the last piece outward; NaN covers inputs that match no
        # condition.
        result: Any = np.nan
        for value, condition in reversed(node.args):
            if condition is sp.true:
                result = _to_pytensor(value, symbols)
            else:
                result = pt.switch(
                    _to_pytensor(condition, symbols),
                    _to_pytensor(value, symbols),
                    result,
                )
        return result
    if type(node) not in _PYTENSOR_MAPPING:
        msg = f"SymPy type {type(node).__name__} has no PyTensor mapping"
        raise NotImplementedError(msg)
    return _PYTENSOR_MAPPING[type(node)](
        *(_to_pytensor(arg, symbols) for arg in node.args)
    )


def analyze_sympy_expr(sympy_expr: sp.Expr) -> dict[str, Any]:
    """
    Analyzes a SymPy expression and logs its independent variables,
    dependent variables, and structure for debugging.

    Args:
        sympy_expr: The SymPy expression to analyze.

    Returns:
        The analysis, keyed by ``'expression'`` (the original expression),
        ``'independent_vars'`` (the set of independent variables, as symbols)
        and ``'dependent_vars'`` (the set of dependent variables, as
        functions).
    """
    # Independent variables (symbols in the expression)
    independent_vars = sympy_expr.free_symbols

    # Dependent variables (functions of symbols)
    dependent_vars = sympy_expr.atoms(sp.Function)

    # Log information for debugging
    log.debug("Expression: %s", sympy_expr)
    log.debug("Independent Variables: %s", independent_vars)
    log.debug("Dependent Variables: %s", dependent_vars)
    if log.isEnabledFor(logging.DEBUG):
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

    try:
        return _parse_expr_cached(expr_str)
    except Exception as exc:
        msg = f"Failed to parse expression '{expr_str}': {exc}"
        raise ExpressionParseError(msg) from exc


def sympy_to_pytensor(
    sympy_expr: sp.Expr,
    variables: list[pt.variable.TensorVariable[TensorType, Any]],
) -> pt.variable.TensorVariable[TensorType, Any]:
    """
    Converts a SymPy expression into a PyTensor computational graph by walking its nodes.

    Args:
        sympy_expr: The SymPy expression object.
        variables: List of PyTensor variables.

    Returns:
        PyTensor expression.

    Raises:
        ExpressionEvaluationError: If the expression cannot be converted or contains unsupported operations.
    """
    try:
        # Bind each symbol to the caller's variable by name
        symbols = {var.name: var for var in variables if var.name is not None}

        # Log the expression for debugging
        analyze_sympy_expr(sympy_expr)

        result = _to_pytensor(sympy_expr, symbols)

        # Handle case where result is a constant (not a PyTensor variable);
        # float64 explicitly, since PyTensor infers float32 or int8 for a
        # number those represent exactly (1.5, 3)
        if not isinstance(result, pt.variable.TensorVariable):
            result = pt.constant(result, dtype="float64")

        return cast(pt.variable.TensorVariable[TensorType, Any], result)

    except Exception as exc:
        msg = f"Failed to convert expression to PyTensor: {sympy_expr}. {exc}"
        raise ExpressionEvaluationError(msg) from exc


@lru_cache(maxsize=2048)
def _parse_expr_cached(expr_str: str) -> sp.Expr:
    # Results are cached to avoid repeated work
    tree = ast.parse(_to_python(expr_str), mode="eval")
    return cast(sp.Expr, _SympyBuilder().build(tree))


class GenericExpressionMixin:
    """
    Mixin for pydantic models that evaluate a user-supplied math expression string.

    Inherit this mixin **before** the pydantic ``BaseModel`` (or subclass such as
    :class:`~pyhs3.base.Evaluable`) in the MRO, e.g.::

        class MyDist(GenericExpressionMixin, Distribution): ...

    The mixin automatically provides:

    - ``model_config`` — allows ``sp.Expr`` private attributes and alias-based
      serialisation.
    - ``expression_str`` — the raw expression string, aliased to ``"expression"``
      in serialised form.
    - ``setup_expression`` — a ``@model_validator(mode="after")`` that parses and
      analyses the expression at construction time.

    Subclasses do **not** need to redeclare any of these.

    Attributes:
        model_config: Pydantic config that permits ``sp.Expr`` private attrs.
        expression_str: The raw mathematical expression string (aliased to
            ``"expression"`` in serialised form).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, serialize_by_alias=True)

    expression_str: str = Field(alias="expression", repr=False)
    _sympy_expr: sp.Expr = PrivateAttr(default=None)

    @model_validator(mode="after")
    def setup_expression(self) -> GenericExpressionMixin:
        """Parse and analyze the expression during initialization."""
        self._sympy_expr = parse_expression(self.expression_str)

        analysis = analyze_sympy_expr(self._sympy_expr)
        independent_vars = [str(symbol) for symbol in analysis["independent_vars"]]

        # _parameters is provided by Evaluable (for distributions/functions)
        self._parameters = {var: var for var in independent_vars}
        return self

    def _eval_expression(self, context: Context) -> TensorVar:
        """Evaluate the parsed expression using variables from *context*."""
        variables = [context[name] for name in self._parameters.values()]
        return cast("TensorVar", sympy_to_pytensor(self._sympy_expr, variables))
