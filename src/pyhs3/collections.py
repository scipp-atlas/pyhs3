"""Generic collection classes for named items."""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterator
from typing import Any, TypeVar

from pydantic import BaseModel, PrivateAttr, RootModel


class NamedModel(BaseModel, ABC):
    """ABC for objects that have a name attribute."""

    name: str


T = TypeVar("T", bound=NamedModel)


class NamedCollection(RootModel[list[T]]):
    """Generic collection providing dict-like access to named items.

    Wraps a list of items with a .name attribute. Builds an internal
    name-to-item mapping once at init for O(1) lookups.
    """

    _map: dict[str, T] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        """Initialize computed collections after Pydantic validation."""
        self._map = {item.name: item for item in self.root}

    def __getitem__(self, item: str | int) -> T:
        if isinstance(item, int):
            return self.root[item]
        return self._map[item]

    def get(self, name: str, default: T | None = None) -> T | None:
        """Get an item by name, returning default if not found."""
        return self._map.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self._map

    def __iter__(self) -> Iterator[T]:  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({[item.name for item in self]})"
