"""
Unit tests for Evaluable base class and auto-preprocessing logic.

Tests the core parameter processing functionality that eliminates the need
for manual @model_validator methods in distributions and functions.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import Field

from pyhs3.base import Evaluable
from pyhs3.context import Context


class TestEvaluable:
    """Test the base Evaluable class functionality."""

    def test_evaluable_base_creation(self):
        """Test basic Evaluable creation."""
        evaluable = Evaluable(name="test", type="test")
        assert evaluable.name == "test"
        assert evaluable.type == "test"
        assert evaluable.parameters == set()
        assert evaluable.constants == {}

    def test_evaluable_expression_not_implemented(self):
        """Test that base evaluable expression method raises NotImplementedError."""
        evaluable = Evaluable(name="test", type="unknown")
        with pytest.raises(
            NotImplementedError,
            match="Component type unknown expression not implemented",
        ):
            evaluable.expression({})


class TestAutoParameterProcessing:
    """Test automatic parameter processing functionality."""

    def test_string_parameter_processing(self):
        """Test processing of string parameters."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            param1: str
            param2: str

        dist = TestDistribution(name="test", param1="alpha", param2="beta")

        # Check parameters were processed
        assert dist.parameters == {"alpha", "beta"}
        assert dist._parameters == {"param1": "alpha", "param2": "beta"}
        assert dist.constants == {}  # No constants for string parameters
        assert dist._constants_values == {}

    def test_numeric_parameter_processing(self):
        """Test processing of numeric parameters."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            param1: float
            param2: int

        dist = TestDistribution(name="test", param1=1.5, param2=42)

        # Check constants were created
        expected_name1 = "constant_test_param1"
        expected_name2 = "constant_test_param2"

        assert dist.parameters == {expected_name1, expected_name2}
        assert dist._parameters == {"param1": expected_name1, "param2": expected_name2}
        assert dist._constants_values == {expected_name1: 1.5, expected_name2: 42}
        assert set(dist.constants.keys()) == {expected_name1, expected_name2}

    def test_mixed_parameter_processing(self):
        """Test processing of mixed str/numeric parameters."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            param1: str | float
            param2: str | int

        # Test with string values
        dist1 = TestDistribution(name="test1", param1="alpha", param2="beta")
        assert dist1.parameters == {"alpha", "beta"}
        assert dist1._constants_values == {}
        assert dist1.constants == {}

        # Test with numeric values
        dist2 = TestDistribution(name="test2", param1=1.5, param2=42)
        expected_name1 = "constant_test2_param1"
        expected_name2 = "constant_test2_param2"
        assert dist2.parameters == {expected_name1, expected_name2}
        assert dist2._constants_values == {expected_name1: 1.5, expected_name2: 42}
        assert set(dist2.constants.keys()) == {expected_name1, expected_name2}

    def test_union_type_ordering(self):
        """Test that union type ordering doesn't matter (float | str, int | str, etc.)."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            param1: float | str  # Different order
            param2: int | str  # Different order
            param3: str | int | float  # Multiple types

        # Test with string values
        dist1 = TestDistribution(
            name="test1", param1="alpha", param2="beta", param3="gamma"
        )
        assert dist1.parameters == {"alpha", "beta", "gamma"}

        # Test with numeric values
        dist2 = TestDistribution(name="test2", param1=1.5, param2=42, param3=3.14)
        expected_names = {
            "constant_test2_param1",
            "constant_test2_param2",
            "constant_test2_param3",
        }
        assert dist2.parameters == expected_names
        assert set(dist2.constants.keys()) == expected_names

    def test_string_list_parameter_processing(self):
        """Test processing of list[str] parameters."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            factors: list[str]

        dist = TestDistribution(name="test", factors=["f1", "f2", "f3"])

        # Check indexed parameters were created
        assert dist.parameters == {"f1", "f2", "f3"}
        assert dist._parameters == {
            "factors[0]": "f1",
            "factors[1]": "f2",
            "factors[2]": "f3",
        }
        assert dist.constants == {}

    def test_mixed_list_parameter_processing(self):
        """Test processing of list[str | float] parameters."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            coeffs: list[str | float]

        dist = TestDistribution(name="test", coeffs=["alpha", 1.5, "beta", 2.0])

        # Check mixed processing
        expected_const1 = "constant_test_coeffs[1]"
        expected_const2 = "constant_test_coeffs[3]"

        assert dist.parameters == {"alpha", expected_const1, "beta", expected_const2}
        assert dist._parameters == {
            "coeffs[0]": "alpha",
            "coeffs[1]": expected_const1,
            "coeffs[2]": "beta",
            "coeffs[3]": expected_const2,
        }
        assert dist._constants_values == {expected_const1: 1.5, expected_const2: 2.0}
        assert set(dist.constants.keys()) == {expected_const1, expected_const2}

    def test_get_parameter_list_reconstruction(self):
        """Test reconstruction of parameter lists from flattened storage."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            factors: list[str | float]

        dist = TestDistribution(name="test", factors=["a", 1.0, "b", 2.0])

        # Create mock context
        context: Context = {
            "a": "tensor_a",
            "constant_test_factors[1]": "tensor_1",
            "b": "tensor_b",
            "constant_test_factors[3]": "tensor_2",
        }

        # Test reconstruction
        result = dist.get_parameter_list(context, "factors")
        expected = ["tensor_a", "tensor_1", "tensor_b", "tensor_2"]
        assert result == expected


class TestPreprocessExclusions:
    """Test the preprocess=False exclusion logic."""

    def test_base_class_field_exclusion(self):
        """Test that base Evaluable fields are excluded from preprocessing."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            param1: str  # This should be processed

        dist = TestDistribution(name="test", param1="alpha")

        # Only param1 should be processed (name and type are marked preprocess=False)
        assert dist.parameters == {"alpha"}
        assert dist._parameters == {"param1": "alpha"}
        assert dist.constants == {}

    def test_explicit_preprocess_false(self):
        """Test fields explicitly marked with preprocess=False."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            param1: str  # Should be processed
            config_field: float = Field(
                default=1.0, json_schema_extra={"preprocess": False}
            )
            param2: str  # Should be processed

        dist = TestDistribution(
            name="test", param1="alpha", config_field=2.5, param2="beta"
        )

        # Only param1 and param2 should be processed
        assert dist.parameters == {"alpha", "beta"}
        assert dist._parameters == {"param1": "alpha", "param2": "beta"}
        assert dist._constants_values == {}  # No constants from config_field
        assert dist.constants == {}

    def test_boolean_field_exclusion(self):
        """Test that boolean fields are automatically excluded."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            param1: str
            enabled: bool  # Should be excluded automatically

        dist = TestDistribution(name="test", param1="alpha", enabled=True)

        # Only param1 should be processed
        assert dist.parameters == {"alpha"}
        assert dist._parameters == {"param1": "alpha"}
        assert dist.constants == {}

    def test_none_value_exclusion(self):
        """Test that fields with None values are skipped."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            param1: str
            param2: str | None = None

        dist = TestDistribution(name="test", param1="alpha", param2=None)

        # Only param1 should be processed
        assert dist.parameters == {"alpha"}
        assert dist._parameters == {"param1": "alpha"}
        assert dist.constants == {}

    def test_manual_override_skips_auto_processing(self):
        """Test that manually set _parameters skips auto-processing."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            param1: str
            param2: str

            def __init__(self, **data):
                super().__init__(**data)
                # Manually set parameters before auto-processing
                self._parameters = {"manual": "override"}

        dist = TestDistribution(name="test", param1="alpha", param2="beta")

        # Auto-processing should be skipped
        assert dist._parameters == {"manual": "override"}
        assert dist.parameters == {"override"}
        assert dist.constants == {}


class TestErrorConditions:
    """Test error conditions and edge cases."""

    def test_unsupported_field_type_error(self):
        """Test error for unsupported field types."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            param1: str
            unsupported_field: dict  # Unsupported type

        with pytest.raises(RuntimeError) as exc_info:
            TestDistribution(
                name="test", param1="alpha", unsupported_field={"key": "value"}
            )

        error_msg = str(exc_info.value)
        assert (
            "Unable to handle `unsupported_field` with type `builtins.dict`"
            in error_msg
        )
        assert "TestDistribution" in error_msg
        assert 'add `json_schema_extra={"preprocess": False}`' in error_msg

    def test_unsupported_field_with_location_info(self):
        """Test error includes file location when available."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            unsupported_field: complex  # Unsupported type

        with pytest.raises(RuntimeError) as exc_info:
            TestDistribution(name="test", unsupported_field=1 + 2j)

        error_msg = str(exc_info.value)
        assert (
            "Unable to handle `unsupported_field` with type `builtins.complex`"
            in error_msg
        )

    def test_parameter_processing_methods(self):
        """Test individual parameter processing methods."""

        class TestEvaluable(Evaluable):
            type: Literal["test"] = "test"
            test_param: str
            numeric_param: float

        # Test process_parameter with string
        evaluable = TestEvaluable(name="test", test_param="alpha", numeric_param=42.0)
        result_name, result_value = evaluable.process_parameter("test_param")
        assert result_name == "alpha"
        assert result_value is None

        # Test process_parameter with numeric
        result_name, result_value = evaluable.process_parameter("numeric_param")
        assert result_name == "constant_test_numeric_param"
        assert result_value == 42.0

    def test_process_parameter_list_methods(self):
        """Test parameter list processing methods."""

        class TestEvaluable(Evaluable):
            type: Literal["test"] = "test"
            test_list: list[str | float]

        evaluable = TestEvaluable(name="test", test_list=["alpha", 1.5, "beta"])

        names, values = evaluable.process_parameter_list("test_list")
        expected_names = ["alpha", "constant_test_test_list[1]", "beta"]
        expected_values = [None, 1.5, None]

        assert names == expected_names
        assert values == expected_values

    def test_empty_parameter_list(self):
        """Test processing empty parameter lists."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            empty_list: list[str] = Field(default_factory=list)

        dist = TestDistribution(name="test", empty_list=[])

        # Empty list should not add any parameters
        assert dist.parameters == set()
        assert dist._parameters == {}

    def test_numeric_only_lists_skipped(self):
        """Test that list[float] and list[int] are skipped."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            param1: str
            float_list: list[float] = Field(default_factory=list)
            int_list: list[int] = Field(default_factory=list)

        dist = TestDistribution(
            name="test", param1="alpha", float_list=[1.0, 2.0, 3.0], int_list=[1, 2, 3]
        )

        # Only param1 should be processed
        assert dist.parameters == {"alpha"}
        assert dist._parameters == {"param1": "alpha"}
        assert dist.constants == {}


class TestConstants:
    """Test constant management functionality."""

    def test_constants_property_conversion(self):
        """Test that constants property converts stored values to PyTensor constants."""

        class TestDistribution(Evaluable):
            type: Literal["test"] = "test"
            param1: float
            param2: int

        dist = TestDistribution(name="test", param1=3.14, param2=42)

        # Check that constants are created with correct keys
        const1_name = "constant_test_param1"
        const2_name = "constant_test_param2"

        assert set(dist.constants.keys()) == {const1_name, const2_name}
        assert dist._constants_values == {const1_name: 3.14, const2_name: 42}
        assert dist.parameters == {const1_name, const2_name}
