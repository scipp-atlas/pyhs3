"""
Unit tests for Evaluable base class and auto-preprocessing logic.

Tests the core parameter processing functionality that eliminates the need
for manual @model_validator methods in distributions and functions.
"""

from __future__ import annotations

import inspect
from typing import Literal
from unittest.mock import patch

import pytensor.tensor as pt
import pytest
from pydantic import Field

from pyhs3.base import Evaluable, find_field_definition_line
from pyhs3.context import Context
from pyhs3.typing.aliases import TensorVar


# Test helper: concrete Evaluable subclass for testing
class ConcreteEvaluable(Evaluable):
    """Concrete Evaluable implementation for testing."""

    def _expression(self, _: Context) -> TensorVar:
        """Dummy implementation for testing."""
        return pt.constant(1.0)


class TestEvaluable:
    """Test the base Evaluable class functionality."""

    def test_evaluable_base_creation(self):
        """Test basic Evaluable creation with concrete subclass."""
        evaluable = ConcreteEvaluable(name="test", type="test")
        assert evaluable.name == "test"
        assert evaluable.type == "test"
        assert evaluable.parameters == set()
        assert evaluable.constants == {}

    def test_evaluable_base_class_is_abstract(self):
        """Test that Evaluable base class cannot be instantiated directly."""
        # Evaluable is now abstract with _expression() as abstract method
        with pytest.raises(
            TypeError,
            match=r"Can't instantiate abstract class Evaluable.*_expression",
        ):
            Evaluable(name="test", type="unknown")


class TestAutoParameterProcessing:
    """Test automatic parameter processing functionality."""

    def test_string_parameter_processing(self):
        """Test processing of string parameters."""

        class TestDistribution(ConcreteEvaluable):
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

        class TestDistribution(ConcreteEvaluable):
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

        class TestDistribution(ConcreteEvaluable):
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

        class TestDistribution(ConcreteEvaluable):
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

        class TestDistribution(ConcreteEvaluable):
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

        class TestDistribution(ConcreteEvaluable):
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

        class TestDistribution(ConcreteEvaluable):
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

        class TestDistribution(ConcreteEvaluable):
            type: Literal["test"] = "test"
            param1: str  # This should be processed

        dist = TestDistribution(name="test", param1="alpha")

        # Only param1 should be processed (name and type are marked preprocess=False)
        assert dist.parameters == {"alpha"}
        assert dist._parameters == {"param1": "alpha"}
        assert dist.constants == {}

    def test_explicit_preprocess_false(self):
        """Test fields explicitly marked with preprocess=False."""

        class TestDistribution(ConcreteEvaluable):
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

        class TestDistribution(ConcreteEvaluable):
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

        class TestDistribution(ConcreteEvaluable):
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

        class TestDistribution(ConcreteEvaluable):
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

        class TestDistribution(ConcreteEvaluable):
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

        class TestDistribution(ConcreteEvaluable):
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

        class TestEvaluable(ConcreteEvaluable):
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

        class TestEvaluable(ConcreteEvaluable):
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

        class TestDistribution(ConcreteEvaluable):
            type: Literal["test"] = "test"
            empty_list: list[str] = Field(default_factory=list)

        dist = TestDistribution(name="test", empty_list=[])

        # Empty list should not add any parameters
        assert dist.parameters == set()
        assert dist._parameters == {}

    def test_numeric_only_lists_skipped(self):
        """Test that list[float] and list[int] are skipped."""

        class TestDistribution(ConcreteEvaluable):
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

        class TestDistribution(ConcreteEvaluable):
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


class TestFindFieldDefinitionLine:
    """Test find_field_definition_line utility function."""

    def test_find_field_definition_line_success(self):
        """Test finding field definition line for a valid field."""

        class TestClass:
            field1: str
            field2: int = 42

        result = find_field_definition_line(TestClass, "field1")

        # Should return a string in format "filepath:line_number"
        assert result is not None
        assert ":" in result
        parts = result.split(":")
        assert len(parts) >= 2  # filepath:line, may have windows drive letter
        assert parts[-1].isdigit()  # line number should be numeric

    def test_find_field_definition_line_field_not_found(self):
        """Test behavior when field is not found in class."""

        class TestClass:
            existing_field: str

        result = find_field_definition_line(TestClass, "nonexistent_field")

        # Should return None when field is not found
        assert result is None

    def test_find_field_definition_line_no_source_available(self):
        """Test behavior when source is not available (built-in classes)."""

        # Test with a built-in class that has no source available
        result = find_field_definition_line(int, "real")

        # Should return None when source is not available
        assert result is None

    def test_find_field_definition_line_io_error(self):
        """Test behavior when inspect raises OSError."""

        class MockClass:
            field: str

        # Mock inspect to raise OSError
        with patch.object(inspect, "getsourcelines", side_effect=OSError("No source")):
            result = find_field_definition_line(MockClass, "field")
            assert result is None

    def test_find_field_definition_line_type_error(self):
        """Test behavior when inspect raises TypeError."""

        class MockClass:
            field: str

        # Mock inspect to raise TypeError
        with patch.object(
            inspect, "getsourcelines", side_effect=TypeError("Not supported")
        ):
            result = find_field_definition_line(MockClass, "field")
            assert result is None

    def test_find_field_definition_line_with_complex_field_names(self):
        """Test finding fields with complex names and annotations."""

        class ComplexClass:
            simple_field: str
            _private_field: int
            fieldWithCamelCase: float
            field_with_annotation: list[str] = Field(default_factory=list)

        # Test simple field
        result = find_field_definition_line(ComplexClass, "simple_field")
        assert result is not None

        # Test private field
        result = find_field_definition_line(ComplexClass, "_private_field")
        assert result is not None

        # Test camelCase field
        result = find_field_definition_line(ComplexClass, "fieldWithCamelCase")
        assert result is not None

        # Test field with annotation
        result = find_field_definition_line(ComplexClass, "field_with_annotation")
        assert result is not None

    def test_find_field_definition_line_empty_class(self):
        """Test behavior with empty class."""

        class EmptyClass:
            pass

        result = find_field_definition_line(EmptyClass, "any_field")
        assert result is None


class TestContext:
    """Test Context class functionality."""

    def test_context_creation(self):
        """Test Context creation with dictionary data."""

        data = {
            "param1": pt.constant(1.0),
            "param2": pt.constant(2.0),
            "param3": pt.constant(3.0),
        }
        context = Context(data)

        # Test __getitem__
        assert context["param1"] is data["param1"]
        assert context["param2"] is data["param2"]
        assert context["param3"] is data["param3"]

    def test_context_contains(self):
        """Test Context __contains__ method."""

        data = {"existing_param": pt.constant(1.0), "another_param": pt.constant(2.0)}
        context = Context(data)

        # Test __contains__
        assert "existing_param" in context
        assert "another_param" in context
        assert "nonexistent_param" not in context

    def test_context_keys(self):
        """Test Context keys() method."""

        data = {
            "alpha": pt.constant(1.0),
            "beta": pt.constant(2.0),
            "gamma": pt.constant(3.0),
        }
        context = Context(data)

        # Test keys()
        keys = context.keys()
        assert set(keys) == {"alpha", "beta", "gamma"}
        assert len(keys) == 3

    def test_context_values(self):
        """Test Context values() method."""

        tensor1 = pt.constant(1.0)
        tensor2 = pt.constant(2.0)
        data = {"param1": tensor1, "param2": tensor2}
        context = Context(data)

        # Test values()
        values = list(context.values())
        assert len(values) == 2
        assert tensor1 in values
        assert tensor2 in values

    def test_context_items(self):
        """Test Context items() method."""

        tensor1 = pt.constant(1.0)
        tensor2 = pt.constant(2.0)
        data = {"x": tensor1, "y": tensor2}
        context = Context(data)

        # Test items()
        items = list(context.items())
        assert len(items) == 2
        assert ("x", tensor1) in items
        assert ("y", tensor2) in items

    def test_context_empty(self):
        """Test Context with empty data."""
        context = Context({})

        # Test all methods with empty context
        assert len(context.keys()) == 0
        assert len(context.values()) == 0
        assert len(context.items()) == 0
        assert "anything" not in context

    def test_context_getitem_keyerror(self):
        """Test Context __getitem__ raises KeyError for missing key."""

        context = Context({"param1": pt.constant(1.0)})

        with pytest.raises(KeyError):
            _ = context["nonexistent_key"]


class TestEvaluableAdvanced:
    """Test advanced Evaluable functionality and edge cases."""

    def test_evaluable_manual_parameter_override(self):
        """Test Evaluable when _parameters is manually set (skips auto-processing)."""

        class TestEvaluableOverride(ConcreteEvaluable):
            type: Literal["test"] = "test"
            param1: str
            param2: float

        # Create evaluable normally first
        evaluable = TestEvaluableOverride(
            name="test_override", param1="alpha", param2=3.14
        )

        # Verify it was auto-processed normally first
        assert evaluable._parameters == {
            "param1": "alpha",
            "param2": "constant_test_override_param2",
        }

        # Now test the manual override by calling the auto-processing method directly
        # after manually setting parameters
        evaluable._parameters = {"param1": "manual_param1", "param2": "manual_param2"}
        result = evaluable._auto_process_parameters()

        # Should return self unchanged (early return due to _parameters being set)
        assert result is evaluable
        assert evaluable._parameters == {
            "param1": "manual_param1",
            "param2": "manual_param2",
        }
        assert evaluable.parameters == {"manual_param1", "manual_param2"}

    def test_evaluable_unsupported_field_error_with_location(self):
        """Test error message includes location when find_field_definition_line succeeds."""

        class TestUnsupportedField(ConcreteEvaluable):
            type: Literal["test"] = "test"
            unsupported_field: set[str]  # set is not supported

        # This should raise RuntimeError with location info
        with pytest.raises(RuntimeError) as exc_info:
            TestUnsupportedField(name="test", unsupported_field={"a", "b"})

        error_msg = str(exc_info.value)
        assert "Unable to handle `unsupported_field`" in error_msg
        assert "with type `builtins.set`" in error_msg
        assert "TestUnsupportedField" in error_msg
        assert "Declared at" in error_msg  # Location should be included
        assert 'add `json_schema_extra={"preprocess": False}`' in error_msg

    def test_evaluable_union_single_typing_arg(self):
        """Test union type handling with a simple single union type."""

        class TestUnionSingle(ConcreteEvaluable):
            type: Literal["test"] = "test"
            # Test a regular supported union type
            param1: str | float

        evaluable = TestUnionSingle(name="test", param1="alpha")

        # Should handle union correctly - string value should be used as parameter name
        assert "alpha" in evaluable.parameters
        assert evaluable._parameters["param1"] == "alpha"

    def test_evaluable_str_float_union_handling(self):
        """Test str/float union type handling."""

        class TestStrFloatUnion(ConcreteEvaluable):
            type: Literal["test"] = "test"
            mixed_param: str | float

        # Test with string value
        evaluable_str = TestStrFloatUnion(name="test1", mixed_param="alpha")
        assert evaluable_str._parameters["mixed_param"] == "alpha"
        assert "alpha" in evaluable_str.parameters

        # Test with float value
        evaluable_float = TestStrFloatUnion(name="test2", mixed_param=3.14)
        assert "constant_test2_mixed_param" in evaluable_float.parameters
        assert evaluable_float.constants["constant_test2_mixed_param"] is not None

    def test_evaluable_union_single_typing_arg_coverage(self):
        """Test union type handling code path with integer/string union."""

        class TestIntStrUnion(ConcreteEvaluable):
            type: Literal["test"] = "test"
            # Test int | str union type
            mixed_param: int | str

        # Test with string value
        evaluable_str = TestIntStrUnion(name="test1", mixed_param="alpha")
        assert evaluable_str._parameters["mixed_param"] == "alpha"
        assert "alpha" in evaluable_str.parameters

        # Test with int value
        evaluable_int = TestIntStrUnion(name="test2", mixed_param=42)
        assert "constant_test2_mixed_param" in evaluable_int.parameters
        assert evaluable_int.constants["constant_test2_mixed_param"] is not None


class TestProcessListFieldCoverage:
    """Test _process_list_field method falsey cases for coverage."""

    def test_process_list_field_union_without_str_and_numeric(self):
        """Test union subargs that don't contain both str and numeric types."""

        class TestUnionWithoutStrNumeric(ConcreteEvaluable):
            type: Literal["test"] = "test"
            # Create a list with union that doesn't match the condition
            # This tests the falsey case for: str in subargs and (float in subargs or int in subargs)
            mixed_list: list[str | bool]  # bool is not numeric

        # This should not be processed because bool is not int/float
        # The condition `str in subargs and (float in subargs or int in subargs)` is False
        # So the list is ignored (not processed, no parameters added)
        evaluable = TestUnionWithoutStrNumeric(name="test", mixed_list=["alpha", True])

        # Should have no parameters because the list was not processed
        assert evaluable._parameters == {}
        assert evaluable.parameters == set()
        assert evaluable._constants_values == {}


class TestErrorLocationCoverage:
    """Test error location falsey case coverage."""

    def test_error_without_location(self):
        """Test error message when find_field_definition_line returns None."""

        class TestNoLocation(ConcreteEvaluable):
            type: Literal["test"] = "test"
            unsupported_field: dict  # unsupported type

        # Mock find_field_definition_line to return None
        with (
            patch("pyhs3.base.find_field_definition_line", return_value=None),
            pytest.raises(RuntimeError) as exc_info,
        ):
            TestNoLocation(name="test", unsupported_field={"key": "value"})

        error_msg = str(exc_info.value)
        assert "Unable to handle `unsupported_field`" in error_msg
        assert "with type `builtins.dict`" in error_msg
        assert "TestNoLocation" in error_msg
        # Should NOT contain "Declared at" when location is None
        assert "Declared at" not in error_msg
        assert 'add `json_schema_extra={"preprocess": False}`' in error_msg
