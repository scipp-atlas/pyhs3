"""
HistFactory data classes.

Provides data structures for HistFactory distributions including sample data
with bin contents and errors.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class SampleData(BaseModel):
    """Sample data containing bin contents and errors."""

    model_config = ConfigDict(serialize_by_alias=True)

    contents: list[float]
    v_errors: list[float] | None = Field(
        default=None, alias="errors", repr=False, exclude_if=lambda v: v is None
    )
    _errors: list[float] = PrivateAttr()

    @model_validator(mode="after")
    def set_default_and_validate(self) -> SampleData:
        """Ensure contents and errors have same length."""
        if self.v_errors is None:
            self._errors = [0.0] * len(self.contents)
        else:
            self._errors = self.v_errors

            if len(self.contents) != len(self.v_errors):
                msg = f"Sample data contents ({len(self.contents)}) and errors ({len(self.v_errors)}) must have same length"
                raise ValueError(msg)
        return self

    @property
    def errors(self) -> list[float]:
        return self._errors


__all__ = ["SampleData"]
