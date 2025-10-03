"""
HistFactory Distribution implementation.

Provides the HistFactoryDist class for handling binned statistical models
with samples and modifiers as defined in the HS3 specification.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, RootModel, model_validator

# Import existing distributions for constraint terms
from pyhs3.distributions.histfactory.modifiers import Modifiers


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


class Sample(BaseModel):
    """HistFactory sample specification."""

    name: str
    data: SampleData
    modifiers: Modifiers = Field(default_factory=Modifiers)


class Samples(RootModel[list[Sample]]):
    """
    Collection of samples for a HistFactory distribution.

    Manages a set of sample instances, providing dict-like access by sample name
    and list-like iteration. Handles sample validation and maintains name uniqueness.
    """

    root: list[Sample] = Field(default_factory=list)
    _map: dict[str, Sample] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        """Initialize computed collections after Pydantic validation."""
        self._map = {sample.name: sample for sample in self.root}

    def __getitem__(self, item: str | int) -> Sample:
        if isinstance(item, int):
            return self.root[item]
        return self._map[item]

    def __contains__(self, item: str) -> bool:
        return item in self._map

    def __iter__(self) -> Iterator[Sample]:  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)


__all__ = ("Sample", "Samples")
