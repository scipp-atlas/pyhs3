"""
Base classes for HS3 distributions and functions.

Provides shared functionality for parameter processing and constant management.
"""

from __future__ import annotations

import inspect
import types
from typing import Any, cast, get_args, get_origin

import pytensor.tensor as pt
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from pyhs3.context import Context
from pyhs3.typing.aliases import TensorVar


def find_field_definition_line(cls: type, field_name: str) -> str | None:
    """Find the source file and line number where a field is defined.

    Args:
        cls: The class to search in.
        field_name: The name of the field to locate.

    Returns:
        String in format "filepath:line_number" if found, None otherwise.
    """
    try:
        lines, start_line = inspect.getsourcelines(cls)
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{field_name}:"):
                return f"{inspect.getsourcefile(cls)}:{start_line + i}"
    except (OSError, TypeError):
        return None
    return None


class Evaluable(BaseModel):
    """Base class for HS3 distributions and functions with automatic parameter preprocessing.

    This class provides automatic parameter processing that eliminates the need for manual
    @model_validator methods in subclasses. It automatically converts field values into
    parameter names and generates constants for numeric values.

    The class automatically processes all field annotations during initialization:

    - **String fields** (``str``) → Direct parameter mapping
    - **Numeric fields** (``float``, ``int``) → Generate unique constant names
    - **Union fields** (``str | float``, ``int | str``, etc.) → Runtime type detection
    - **List fields** (``list[str]``, ``list[str | float]``) → Indexed parameter mapping
    - **Boolean fields** (``bool``) → Automatically excluded from processing
    - **Excluded fields** → Fields marked with ``json_schema_extra={"preprocess": False}``

    Examples:
        Basic usage with string and numeric parameters::

            from typing import Literal

            class MyDistribution(Evaluable):
                type: Literal["gaussian"] = "gaussian"
                mean: str | float  # Can be parameter name or numeric value
                sigma: str | float

            # With parameter references
            dist1 = MyDistribution(name="gauss1", mean="mu_param", sigma="sigma_param")
            print(dist1.parameters)  # {'mu_param', 'sigma_param'}
            print(dist1.constants)   # {}

            # With numeric values - constants are generated automatically
            dist2 = MyDistribution(name="gauss2", mean=1.5, sigma=0.5)
            print(dist2.parameters)  # {'constant_gauss2_mean', 'constant_gauss2_sigma'}
            print(list(dist2.constants.keys()))  # ['constant_gauss2_mean', 'constant_gauss2_sigma']

        List parameter processing::

            class ProductFunction(Evaluable):
                type: Literal["product"] = "product"
                factors: list[str | float]  # Mixed list of names and values

            func = ProductFunction(name="prod", factors=["param1", 2.0, "param2", 1.5])
            print(sorted(func.parameters))
            # ['constant_prod_factors[1]', 'constant_prod_factors[3]', 'param1', 'param2']

            # Reconstruct parameter list in context
            context = {
                "param1": "tensor1",
                "constant_prod_factors[1]": "tensor2",
                "param2": "tensor3",
                "constant_prod_factors[3]": "tensor4"
            }
            func.get_parameter_list(context, "factors")  # ['tensor1', 'tensor2', 'tensor3', 'tensor4']

        Excluding fields from preprocessing::

            from pydantic import Field

            class ConfigurableDistribution(Evaluable):
                type: Literal["configurable"] = "configurable"
                param: str | float                    # Will be processed
                enabled: bool                         # Automatically excluded
                config_val: float = Field(           # Explicitly excluded
                    default=1.0,
                    json_schema_extra={"preprocess": False}
                )

            dist = ConfigurableDistribution(name="test", param="alpha", enabled=True, config_val=2.0)
            print(dist.parameters)  # {'alpha'} - Only param is processed

    Note:
        If you need custom parameter processing, set ``_parameters`` manually before
        the auto-processing runs, or provide a custom ``@model_validator``.

        Unsupported field types raise ``RuntimeError`` with helpful guidance about
        using ``json_schema_extra={"preprocess": False}`` for non-parameter fields.

    Attributes:
        name (str): Name of the component.
        type (str): Type identifier for the component.
        parameters (set[str]): Set of parameter names this component depends on.
        constants (dict[str, TensorVar]): Generated PyTensor constants for numeric values.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    name: str = Field(..., json_schema_extra={"preprocess": False})
    type: str = Field(..., json_schema_extra={"preprocess": False})
    _parameters: dict[str, str] = PrivateAttr(default_factory=dict)
    _constants_values: dict[str, float | int] = PrivateAttr(default_factory=dict)

    @property
    def parameters(self) -> set[str]:
        """Set of parameter names this component depends on.

        Returns:
            Set of parameter names, including both string references and
            generated constant names for numeric values.
        """
        return set(self._parameters.values())

    @property
    def constants(self) -> dict[str, TensorVar]:
        """Dictionary of PyTensor constants generated from numeric field values.

        Returns:
            Mapping from generated constant names to PyTensor constant tensors.
            Empty if all fields are string references.
        """
        return {
            name: cast(TensorVar, pt.constant(value))
            for name, value in self._constants_values.items()
        }

    def process_parameter(self, param_key: str) -> tuple[str, float | int | None]:
        """Process a single parameter that can be either a string reference or numeric value.

        For numeric values, generates a unique constant name. For string values,
        returns the value as-is.

        Args:
            param_key: The parameter field name to process (e.g., "mean", "sigma").

        Returns:
            Tuple containing:
                - processed_name: Either the original string value or a generated constant name
                - numeric_value: The numeric value if input was numeric, None otherwise

        Example:
            >>> from typing import Literal
            >>> class TestEvaluable(Evaluable):
            ...     type: Literal["test"] = "test"
            ...     some_param: str | float
            >>>
            >>> # String parameter
            >>> eval1 = TestEvaluable(name="test1", some_param="alpha")
            >>> eval1.process_parameter("some_param")
            ('alpha', None)

            >>> # Numeric parameter
            >>> eval2 = TestEvaluable(name="test2", some_param=1.5)
            >>> eval2.process_parameter("some_param")
            ('constant_test2_some_param', 1.5)
        """
        param_value = getattr(self, param_key)
        if isinstance(param_value, int | float):
            # Generate unique constant name
            constant_name = f"constant_{self.name}_{param_key}"
            return constant_name, param_value
        # It's a string reference - return as-is with no numeric value
        return param_value, None

    def process_parameter_list(
        self, param_key: str
    ) -> tuple[list[str], list[float | int | None]]:
        """Process a list parameter containing mixed string references and numeric values.

        For numeric values, generates indexed unique names and stores the values.
        For string values, returns the values as-is. Also updates internal parameter
        mapping with indexed keys.

        Args:
            param_key: The parameter field name to process (e.g., "factors", "coefficients").

        Returns:
            Tuple containing:
                - processed_names: List of parameter names (original strings or generated constant names)
                - numeric_values: List of numeric values (None for string entries)

        Example:
            >>> from typing import Literal
            >>> class TestEvaluable(Evaluable):
            ...     type: Literal["test"] = "test"
            ...     factors: list[str | float]
            >>>
            >>> eval1 = TestEvaluable(name="test", factors=["param1", 2.0, "param2"])
            >>> names, values = eval1.process_parameter_list("factors")
            >>> names
            ['param1', 'constant_test_factors[1]', 'param2']
            >>> values
            [None, 2.0, None]
        """
        result: list[tuple[str, float | int | None]] = []
        param_values = getattr(self, param_key)
        for param_index, param_value in enumerate(param_values):
            if isinstance(param_value, int | float):
                # Generate unique constant name with indexing
                constant_name = f"constant_{self.name}_{param_key}[{param_index}]"
                result.append((constant_name, param_value))
                # Store in flattened _parameters
                self._parameters[f"{param_key}[{param_index}]"] = constant_name
                continue
            # It's a string reference - return as-is with no numeric value
            result.append((param_value, None))
            # Store in flattened _parameters
            self._parameters[f"{param_key}[{param_index}]"] = param_value

        if not result:
            return [], []
        names, values = zip(*result, strict=False)
        return list(names), list(values)

    def get_parameter_list(self, context: Context, param_key: str) -> list[TensorVar]:
        """Reconstruct a parameter list from flattened indexed keys.

        Used to recover the original list structure from the indexed parameter mapping
        created by process_parameter_list().

        Args:
            context: The context containing parameter values mapped by name.
            param_key: The base parameter key (e.g., "factors").

        Returns:
            List of parameter values in original order.

        Example:
            >>> from typing import Literal
            >>> class TestEvaluable(Evaluable):
            ...     type: Literal["test"] = "test"
            ...     factors: list[str | float]
            >>>
            >>> eval1 = TestEvaluable(name="test", factors=["a", 1.0, "b"])
            >>> context = {
            ...     "a": "tensor_a",
            ...     "constant_test_factors[1]": "tensor_1",
            ...     "b": "tensor_b"
            ... }
            >>> eval1.get_parameter_list(context, "factors")
            ['tensor_a', 'tensor_1', 'tensor_b']
        """
        result = []
        i = 0
        while f"{param_key}[{i}]" in self._parameters:
            param_name = self._parameters[f"{param_key}[{i}]"]
            result.append(context[param_name])
            i += 1
        return result

    @model_validator(mode="after")
    def _auto_process_parameters(self) -> Evaluable:
        """
        Automatically process parameters based on model fields.

        This eliminates the need for manual @model_validator methods in subclasses.
        Processes all fields that aren't part of the base Evaluable class.
        """
        # Skip if already processed (allow manual override)
        if self._parameters:
            return self

        # Build excluded fields from base Evaluable class
        excluded_fields = {
            name
            for name, info in Evaluable.model_fields.items()
            if info.json_schema_extra
            and isinstance(info.json_schema_extra, dict)
            and info.json_schema_extra.get("preprocess") is False
        }

        # Process all fields except excluded ones
        for field_name, field_info in self.__class__.model_fields.items():
            if self._should_skip_field(field_name, field_info, excluded_fields):
                continue

            field_value = getattr(self, field_name)
            if field_value is None:
                continue

            self._process_field(field_name, field_info, field_value)

        return self

    def _should_skip_field(
        self, field_name: str, field_info: Any, excluded_fields: set[str]
    ) -> bool:
        """Check if a field should be skipped during auto-processing."""
        # Skip fields marked as non-preprocessable in base class
        if field_name in excluded_fields:
            return True

        # Also check json_schema_extra for explicit exclusion in subclass
        if (
            field_info.json_schema_extra
            and isinstance(field_info.json_schema_extra, dict)
            and field_info.json_schema_extra.get("preprocess") is False
        ):
            return True

        # Check if field exists and has a value
        if not hasattr(self, field_name):
            return True

        # Skip boolean fields - they're not parameters
        return field_info.annotation is bool

    def _is_processable_single_field(self, annotation: Any) -> bool:
        """Check if a field annotation represents a processable single parameter."""
        # Handle simple types
        if annotation in (str, float, int):
            return True

        # Handle union types (e.g., str | float, int | str, etc.)
        if get_origin(annotation) is types.UnionType:
            union_args = get_args(annotation)
            # Check if union contains str and only numeric types
            has_str = str in union_args
            has_numeric = any(t in (int, float) for t in union_args)
            non_processable = any(t not in (str, int, float) for t in union_args)
            return has_str and has_numeric and not non_processable

        return False

    def _process_field(
        self, field_name: str, field_info: Any, field_value: Any
    ) -> None:
        """Process a single field during auto-processing."""
        # Let process_parameter* methods handle the type detection
        if get_origin(field_info.annotation) is list:
            self._process_list_field(field_name, field_info, field_value)
        elif self._is_processable_single_field(field_info.annotation):
            self._process_single_field(field_name)
        else:
            self._raise_unsupported_field_error(field_name, field_info)

    def _process_list_field(
        self, field_name: str, field_info: Any, field_value: list[Any]
    ) -> None:
        """Process a list field during auto-processing."""
        typing_args = get_args(field_info.annotation)
        # Process the list based on its inner type (list[T] always has exactly one type argument)
        if str in typing_args:
            # handle list[str]
            for index, name in enumerate(field_value):
                self._parameters[f"{field_name}[{index}]"] = name
        elif float in typing_args or int in typing_args:
            # skip list[float] and list[int]
            pass
        elif isinstance(typing_args[0], types.UnionType):
            subargs = get_args(typing_args[0])
            if str in subargs and (float in subargs or int in subargs):
                # handle list[int | float | str] or similar mixed versions
                processed_names, processed_values = self.process_parameter_list(
                    field_name
                )
                # Add constants for numeric values
                for name, value in zip(processed_names, processed_values, strict=False):
                    if value is not None:
                        self._constants_values[name] = value

    def _process_single_field(self, field_name: str) -> None:
        """Process a single parameter field during auto-processing."""
        # Handle single parameters - process_parameter handles all cases
        processed_name, processed_value = self.process_parameter(field_name)
        self._parameters[field_name] = processed_name

        # Add constant if numeric
        if processed_value is not None:
            self._constants_values[processed_name] = processed_value

    def _raise_unsupported_field_error(self, field_name: str, field_info: Any) -> None:
        """Raise an error for unsupported field types."""
        location = find_field_definition_line(self.__class__, field_name)
        type_name = getattr(
            field_info.annotation, "__qualname__", str(field_info.annotation)
        )
        module_name = getattr(field_info.annotation, "__module__", "")
        full_type_name = f"{module_name}.{type_name}" if module_name else type_name

        msg = f"Unable to handle `{field_name}` with type `{full_type_name}` on {self.__class__.__name__}."

        if location:
            msg += f" Declared at {location}."

        msg += ' If this is not a parameter to preprocess, add `json_schema_extra={"preprocess": False}`.'
        raise RuntimeError(msg)

    def expression(self, _: Context) -> TensorVar:
        """
        Base expression method - should be overridden by subclasses.

        Args:
            context: Mapping of names to PyTensor variables

        Returns:
            PyTensor expression representing the component
        """
        msg = f"Component type {self.type} expression not implemented"
        raise NotImplementedError(msg)
