"""
Exception classes for pyhs3.

Custom exception hierarchy for better error handling and debugging.
"""

from __future__ import annotations


class HS3Exception(Exception):
    """
    Base exception class for all pyhs3-related errors.

    This serves as the root exception that all other pyhs3 exceptions inherit from,
    allowing users to catch all pyhs3-specific errors with a single except clause.
    """


class ExpressionParseError(HS3Exception):
    """
    Exception raised when a mathematical expression cannot be parsed.

    This typically occurs when:
    - The expression contains invalid syntax
    - Unsupported mathematical operations are used
    - Variable names are malformed
    """


class ExpressionEvaluationError(HS3Exception):
    """
    Exception raised when a parsed expression cannot be evaluated.

    This typically occurs when:
    - Required variables are missing from the evaluation context
    - The expression results in mathematical errors (division by zero, etc.)
    - PyTensor conversion fails
    """
