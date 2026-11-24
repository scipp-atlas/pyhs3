"""
Exception classes for pyhs3.

Custom exception hierarchy for better error handling and debugging.
"""

from __future__ import annotations

from typing import Any

from pydantic import (
    ValidationError,
    ValidationInfo,
    WrapValidator,
)
from pydantic_core import ErrorDetails, InitErrorDetails, PydanticCustomError


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


class WorkspaceValidationError(HS3Exception):
    """
    Raised when a workspace fails to validate.

    This typically occurs when:

    - Foreign key references cannot be resolved (e.g. a likelihood references
      a distribution that doesn't exist in the workspace)
    - A JSON file cannot be parsed into a valid workspace
    """


class UnlistedParameterError(WorkspaceValidationError):
    """
    Raised when an analysis violates HS3 section 2.7.

    Every parameter of an analysis's likelihood must be listed in one of the
    analysis domains or be ``const`` in the referenced parameter point. ROOT
    exports commonly reference no parameter point (no ``init``) while marking
    parameters such as ``Lumi`` const in every parameter point.

    Attributes:
        parameters: The offending parameter names, sorted.
        const_parameter_sets: Names of the workspace parameter points in which
            every parameter outside the analysis domains and data is ``const``;
            selecting one of them as the parameter set (which replaces ``init``)
            makes the analysis valid.
    """

    def __init__(
        self, message: str, parameters: list[str], const_parameter_sets: list[str]
    ) -> None:
        """Store the message and the data a caller needs to fix the analysis.

        Args:
            message: Human-readable description of the violation.
            parameters: The offending parameter names, sorted.
            const_parameter_sets: Parameter points in which every parameter
                outside the analysis domains and data is ``const``.
        """
        super().__init__(message)
        self.parameters = parameters
        self.const_parameter_sets = const_parameter_sets


class DuplicateEntityError(HS3Exception):
    """
    Raised when two distinct entities share a single name in the dependency graph.

    Distributions, functions, parameters, constants, and HistFactory modifiers
    all live in one shared name namespace when the computation graph is built.
    Two distinct entities with the same name (for example a function and a
    distribution both named ``shared``, or two distributions named ``model``)
    would silently shadow one another — only the last one built would survive,
    and references to that name would be wired to the wrong node.  This error is
    raised instead so the collision surfaces loudly at model-construction time.

    A parameter declared in ``parameter_points`` that also appears as a graph
    parameter is the *same* logical entity and is not a collision.
    """


def custom_error_msg(custom_messages: dict[str, str]) -> Any:
    r"""
    Customize an error message for pydantic validation errors.

    See https://github.com/pydantic/pydantic/discussions/8468.

    Example:

    >>> from typing import Annotated
    >>> from pydantic import BaseModel
    >>> from pydantic.types import StringConstraints
    >>> NameString = Annotated[
    ...     str,
    ...     StringConstraints(pattern=r"^[a-zA-Z0-9]*$"),
    ...     custom_error_msg({"string_pattern_mismatch": "The field {field_name} can only contain letters and numbers."}),
    ... ]
    >>> class Model(BaseModel):
    ...     name: NameString
    >>> Model(name="dog@123")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    pydantic_core._pydantic_core.ValidationError: 1 validation error for Model
    name
      The field name can only contain letters and numbers. ...
    """

    def _validator(v: Any, next_: Any, ctx: ValidationInfo) -> Any:
        try:
            return next_(v, ctx)
        except ValidationError as exc:
            new_errors: list[InitErrorDetails | ErrorDetails] = []
            for error in exc.errors():
                error["loc"] = error["loc"][1:]  # to skip current location
                custom_message = custom_messages.get(error["type"])

                if custom_message:
                    err_ctx = error.get("ctx", {}).copy()

                    # Add input and ValidationInfo data to context
                    err_ctx["input"] = error["input"]
                    if ctx.data:
                        err_ctx.update(ctx.data)

                    new_error = InitErrorDetails(
                        type=PydanticCustomError(
                            error["type"], custom_message, err_ctx
                        ),
                        loc=error["loc"],
                        input=error["input"],
                    )

                    new_errors.append(new_error)
                else:
                    new_errors.append(error)

            raise ValidationError.from_exception_data(
                title=exc.title,
                line_errors=new_errors,  # type: ignore[arg-type]
            ) from None

    return WrapValidator(_validator)
