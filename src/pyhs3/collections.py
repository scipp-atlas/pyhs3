"""Generic collection classes for named items."""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterator, Mapping
from typing import Any, TypeVar

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    RootModel,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)
from pydantic_core import PydanticCustomError


class NamedModel(BaseModel, ABC):
    """ABC for objects that have a name attribute."""

    name: str = Field(..., repr=True)


T = TypeVar("T", bound=NamedModel)


def dispatch_on_type(
    base: type[BaseModel], registry: Mapping[str, type[BaseModel]]
) -> WrapValidator:
    """Validate a mapping as the class that ``registry`` names for its ``type``.

    The registry is read at validation time, so a class added to it after
    import is loadable from JSON without redefining any type alias. Instances of
    ``base`` pass through unchanged.

    An unknown ``type`` raises a ``union_tag_invalid`` error carrying ``tag``
    and ``expected_tags``, the same error a pydantic discriminated union raises,
    so :func:`~pyhs3.exceptions.custom_error_msg` can reword it.
    """

    def _validator(value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        if isinstance(value, base):
            return value
        if isinstance(value, Mapping):
            tag = value.get("type")
            cls = registry.get(tag) if isinstance(tag, str) else None
            if cls is None:
                error_type = "union_tag_invalid"
                raise PydanticCustomError(
                    error_type,
                    "Input tag '{tag}' found using 'type' does not match any of the expected tags: {expected_tags}",
                    {"tag": tag, "expected_tags": ", ".join(map(repr, registry))},
                )
            return cls.model_validate(value)
        return handler(value)

    return WrapValidator(_validator)


class NamedCollection(RootModel[list[T]]):
    """Generic collection providing dict-like access to named items.

    Wraps a list of items with a .name attribute. Builds an internal
    name-to-item mapping once at init for O(1) lookups.
    """

    _map: dict[str, T] = PrivateAttr(default_factory=dict)

    #: Subclasses set this to ``True`` to reject duplicate item names at
    #: validation time (instead of the default silent last-wins behaviour).
    #: Left ``False`` on the base class so collections that perform their own,
    #: more specific duplicate handling (e.g. :class:`~pyhs3.domains.ProductDomain`
    #: axes) keep their existing error messages.
    _enforce_unique_names: bool = False

    def model_post_init(self, __context: Any, /) -> None:
        """Initialize computed collections after Pydantic validation.

        Raises:
            ValueError: If ``_enforce_unique_names`` is set and two items share a
                name. Duplicate names would otherwise silently shadow one another
                (last-wins), so collections that opt in fail loudly instead.
        """
        if self._enforce_unique_names:
            seen: set[str] = set()
            duplicates: list[str] = []
            for item in self.root:
                if item.name in seen and item.name not in duplicates:
                    duplicates.append(item.name)
                seen.add(item.name)
            if duplicates:
                names = ", ".join(repr(name) for name in duplicates)
                msg = (
                    f"{type(self).__name__} contains duplicate item name(s): "
                    f"{names}. Names must be unique within a collection."
                )
                raise ValueError(msg)
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
