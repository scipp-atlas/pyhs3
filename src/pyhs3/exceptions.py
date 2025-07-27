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


class UnknownInterpolationCodeError(HS3Exception):
    """
    Exception raised when an unknown interpolation code is used.

    This occurs when an InterpolationFunction is configured with an
    interpolation code outside the valid range (0-6).
    """


def custom_error_msg(custom_messages: dict[str, str]) -> Any:
    r"""
    Customize an error message for pydantic validation errors.

    See https://github.com/pydantic/pydantic/discussions/8468.

    Example:

    >>> NameString = Annotated[
    ...     str,
    ...     StringConstraints(pattern=r"^[0-9~`!@#$%^&*()_+={\[}\]|\:;\"<>\/?]*$"),
    ...     custom_error_msg({"str_error", "The field {field_name} can not contain special symbols."}),
    ... ]
    ...
    >>> class Model(BaseModel):
    ...     name: NameString
    ...
    >>> dog = Model(name="dog123")
    """

    def _validator(v: Any, next_: Any, ctx: ValidationInfo) -> Any:
        try:
            return next_(v, ctx)
        except ValidationError as exc:
            new_errors: list[InitErrorDetails | ErrorDetails] = []
            for error in exc.errors():
                custom_message = custom_messages.get(error["type"])
                if custom_message:
                    err_ctx = error.get("ctx", {})

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
