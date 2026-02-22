"""Tests for generic collection classes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pyhs3.collections import NamedCollection, NamedModel


# Test fixtures - create concrete implementations for testing
class DummyItem(NamedModel):
    """Test item with a name."""

    name: str
    value: int = 0


class DummyCollection(NamedCollection[DummyItem]):
    """Test collection of named items."""


@pytest.fixture
def items() -> list[DummyItem]:
    """Create test items."""
    return [
        DummyItem(name="first", value=1),
        DummyItem(name="second", value=2),
        DummyItem(name="third", value=3),
    ]


@pytest.fixture
def collection(items: list[DummyItem]) -> DummyCollection:
    """Create test collection."""
    return DummyCollection(items)


class TestNamedModel:
    """Test the NamedModel ABC."""

    def test_named_requires_name_attribute(self) -> None:
        """NamedModel ABC requires a name attribute."""
        # DummyItem should be valid since it has a name
        item = DummyItem(name="test")
        assert isinstance(item, NamedModel)
        assert item.name == "test"

    def test_named_is_abstract(self) -> None:
        """NamedModel is an ABC that requires a name attribute."""
        # NamedModel requires the name field
        with pytest.raises(ValidationError):
            NamedModel()


class TestNamedCollection:
    """Test the NamedCollection base class."""

    def test_initialization(self, items: list[DummyItem]) -> None:
        """Collection can be initialized with a list."""
        collection = DummyCollection(items)
        assert len(collection) == 3

    def test_getitem_by_name(self, collection: DummyCollection) -> None:
        """Collection supports lookup by name."""
        assert collection["first"].value == 1
        assert collection["second"].value == 2
        assert collection["third"].value == 3

    def test_getitem_by_index(self, collection: DummyCollection) -> None:
        """Collection supports lookup by integer index."""
        assert collection[0].name == "first"
        assert collection[1].name == "second"
        assert collection[2].name == "third"

    def test_getitem_by_negative_index(self, collection: DummyCollection) -> None:
        """Collection supports negative indexing."""
        assert collection[-1].name == "third"
        assert collection[-2].name == "second"

    def test_getitem_missing_name(self, collection: DummyCollection) -> None:
        """Collection raises KeyError for missing names."""
        with pytest.raises(KeyError):
            collection["missing"]

    def test_getitem_out_of_bounds(self, collection: DummyCollection) -> None:
        """Collection raises IndexError for out of bounds indices."""
        with pytest.raises(IndexError):
            collection[10]

    def test_get_with_existing_name(self, collection: DummyCollection) -> None:
        """get() returns item for existing name."""
        item = collection.get("second")
        assert item is not None
        assert item.value == 2

    def test_get_with_missing_name(self, collection: DummyCollection) -> None:
        """get() returns None for missing name."""
        assert collection.get("missing") is None

    def test_get_with_default(self, collection: DummyCollection) -> None:
        """get() returns default for missing name."""
        default = DummyItem(name="default", value=99)
        result = collection.get("missing", default)
        assert result is default

    def test_contains_existing(self, collection: DummyCollection) -> None:
        """__contains__ returns True for existing names."""
        assert "first" in collection
        assert "second" in collection
        assert "third" in collection

    def test_contains_missing(self, collection: DummyCollection) -> None:
        """__contains__ returns False for missing names."""
        assert "missing" not in collection

    def test_iter(self, collection: DummyCollection, items: list[DummyItem]) -> None:
        """Collection is iterable."""
        result = list(collection)
        assert result == items

    def test_len(self, collection: DummyCollection) -> None:
        """Collection has correct length."""
        assert len(collection) == 3

    def test_repr(self, collection: DummyCollection) -> None:
        """Collection has meaningful repr."""
        result = repr(collection)
        assert result == "DummyCollection(['first', 'second', 'third'])"

    def test_empty_collection(self) -> None:
        """Empty collection works correctly."""
        empty = DummyCollection([])
        assert len(empty) == 0
        assert list(empty) == []
        assert "anything" not in empty
        assert empty.get("anything") is None
        assert repr(empty) == "DummyCollection([])"

    def test_map_is_built_at_init(self, items: list[DummyItem]) -> None:
        """Internal _map is built once at initialization."""
        collection = DummyCollection(items)
        # Access the private attribute for testing
        assert len(collection._map) == 3
        assert "first" in collection._map
        assert "second" in collection._map
        assert "third" in collection._map

    def test_duplicate_names(self) -> None:
        """Duplicate names - last one wins in the map."""
        items = [
            DummyItem(name="duplicate", value=1),
            DummyItem(name="duplicate", value=2),
        ]
        collection = DummyCollection(items)
        # The map will have the last item with that name
        assert collection["duplicate"].value == 2
        # But the full list is preserved
        assert len(collection) == 2

    def test_serialization_roundtrip(self, collection: DummyCollection) -> None:
        """Collection can be serialized and deserialized."""
        # Serialize to dict
        data = collection.model_dump()
        # Deserialize back
        restored = DummyCollection.model_validate(data)
        assert len(restored) == len(collection)
        assert list(restored) == list(collection)
        assert "first" in restored
        assert restored["second"].value == 2

    def test_json_roundtrip(self, collection: DummyCollection) -> None:
        """Collection can be serialized to JSON and back."""
        # Serialize to JSON
        json_str = collection.model_dump_json()
        # Deserialize from JSON
        restored = DummyCollection.model_validate_json(json_str)
        assert len(restored) == len(collection)
        assert list(restored) == list(collection)
        assert restored["third"].value == 3
