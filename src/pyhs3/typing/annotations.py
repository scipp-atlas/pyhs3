"""Type annotations for foreign key relationships in HS3 models.

Provides Pydantic v2 validators and serializers for FK fields that:
- Accept string references from JSON
- Accept model instances from Python
- Reject embedded dicts
- Serialize back to strings for JSON output
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, PlainSerializer, WithJsonSchema, WrapValidator
from pydantic_core import core_schema


def _validate_fk(v: Any, handler: core_schema.ValidatorFunctionWrapHandler) -> Any:
    """Accept string references or model instances. Reject dicts.

    Uses handler (Pydantic's core validator) for model instances, providing
    type-specific validation based on the Annotated type.
    """
    if isinstance(v, str):
        return v  # Skip core validation for strings
    if isinstance(v, dict):
        msg = "Embedded objects not allowed in JSON - use string reference"
        raise TypeError(msg)
    return handler(v)  # Validates against the annotated type (e.g. Likelihood)


def _serialize_fk(v: Any) -> str | None:
    """Serialize FK to string name."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    # Assume it's a BaseModel with a name attribute
    return str(v.name)


def _serialize_fk_list(v: Any) -> list[str]:
    """Serialize FK list to list of string names."""
    result: list[str] = []
    for item in v:
        if isinstance(item, str):
            result.append(item)
        else:
            # Assume it's a BaseModel with a name attribute
            result.append(str(item.name))
    return result


def make_fk_list_validator(model_type: type[BaseModel]) -> WrapValidator:
    """Create a type-specific WrapValidator for list FK fields.

    Unlike the single FK validator which delegates to handler for type checking,
    list validators handle items individually and need an explicit type parameter
    since handler validates list[T], not individual T items.
    """

    def _validate(
        v: Any, _handler: core_schema.ValidatorFunctionWrapHandler
    ) -> list[str | BaseModel]:
        if not isinstance(v, list):
            msg = "Expected a list"
            raise TypeError(msg)
        result: list[str | BaseModel] = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                msg = "Embedded objects not allowed in JSON - use string reference"
                raise TypeError(msg)
            elif isinstance(item, model_type):
                result.append(item)
            else:
                msg = f"Expected string reference or {model_type.__name__}, got {type(item)}"
                raise TypeError(msg)
        return result

    return WrapValidator(_validate)


# Pre-built annotation components for single FK fields
FKValidator = WrapValidator(_validate_fk)
FKSerializer = PlainSerializer(_serialize_fk)
FKSchema = WithJsonSchema({"type": "string"})

# Pre-built annotation components for list FK fields (validator created per-field via factory)
FKListSerializer = PlainSerializer(_serialize_fk_list)
FKListSchema = WithJsonSchema({"type": "array", "items": {"type": "string"}})
