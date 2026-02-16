"""
Unit tests for FK annotation functions.

Tests for the low-level serialization and validation functions
in pyhs3.typing.annotations that power FK field behavior.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pyhs3.likelihoods import Likelihood
from pyhs3.typing.annotations import _serialize_fk, _serialize_fk_list


class TestSerializeFKSingle:
    """Tests for _serialize_fk function covering None and string branches."""

    def test_serialize_fk_none_returns_none(self):
        """Test that _serialize_fk returns None when given None."""
        result = _serialize_fk(None)
        assert result is None

    def test_serialize_fk_string_passthrough(self):
        """Test that _serialize_fk passes through string values unchanged."""
        result = _serialize_fk("my_lk")
        assert result == "my_lk"


class TestSerializeFKList:
    """Tests for _serialize_fk_list function covering string item branch."""

    def test_serialize_fk_list_string_items(self):
        """Test that _serialize_fk_list handles string items correctly."""
        result = _serialize_fk_list(["dist1", "dist2"])
        assert result == ["dist1", "dist2"]


class TestFKListValidatorErrors:
    """Tests for FK list validator error branches."""

    def test_fk_list_validator_rejects_non_list(self):
        """Test that FK list validator rejects non-list inputs."""
        # Pydantic v2 may wrap TypeError as ValidationError
        with pytest.raises((TypeError, ValidationError)) as exc_info:
            Likelihood(name="t", distributions="not_a_list", data=["o"])

        # Verify the error message mentions list expectation
        error_msg = str(exc_info.value)
        assert "Expected a list" in error_msg or "list" in error_msg.lower()

    def test_fk_list_validator_rejects_wrong_type_item(self):
        """Test that FK list validator rejects wrong-type items in list."""
        # Pydantic v2 may wrap TypeError as ValidationError
        with pytest.raises((TypeError, ValidationError)) as exc_info:
            Likelihood(name="t", distributions=[42], data=["o"])

        # Verify the error message mentions type mismatch
        error_msg = str(exc_info.value)
        assert (
            "Expected string reference or Distribution" in error_msg
            or "string" in error_msg.lower()
        )
