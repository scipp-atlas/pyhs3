"""
HistFactory data classes.

Provides data structures for HistFactory distributions including sample data
with bin contents and errors.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class SampleData(BaseModel):
    """Sample data containing bin contents and errors."""

    contents: list[float]
    errors: list[float]

    @model_validator(mode="after")
    def validate_lengths(self) -> SampleData:
        """Ensure contents and errors have same length."""
        if len(self.contents) != len(self.errors):
            msg = f"Sample data contents ({len(self.contents)}) and errors ({len(self.errors)}) must have same length"
            raise ValueError(msg)
        return self


__all__ = ["SampleData"]
