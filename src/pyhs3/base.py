"""
Base classes for HS3 distributions and functions.

Provides shared functionality for parameter processing and constant management.
"""

from __future__ import annotations

from typing import cast

import pytensor.tensor as pt
from pydantic import BaseModel, ConfigDict, PrivateAttr

from pyhs3.context import Context
from pyhs3.typing.aliases import TensorVar


class Evaluable(BaseModel):
    """
    Base class for HS3 distributions and functions.

    Provides shared functionality for parameter management, constant generation,
    and parameter processing for mixed string/numeric values.

    Attributes:
        name (str): Name of the component.
        type (str): Type identifier for the component.
        parameters (dict): Parameter mapping (can contain strings or lists of strings).
        constants (dict[str, TensorVar]): Generated PyTensor constants for numeric values.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    name: str
    type: str
    _parameters: dict[str, str] = PrivateAttr(default_factory=dict)
    _constants_values: dict[str, float | int] = PrivateAttr(default_factory=dict)

    @property
    def parameters(self) -> set[str]:
        """Access to parameter names this component depends on."""
        return set(self._parameters.values())

    @property
    def constants(self) -> dict[str, TensorVar]:
        """Convert stored numeric constants to PyTensor constants."""
        return {
            name: cast(TensorVar, pt.constant(value))
            for name, value in self._constants_values.items()
        }

    def process_parameter(self, param_key: str) -> tuple[str, float | int | None]:
        """
        Process a parameter that can be either a string reference or a numeric value.

        For numeric values, generates a unique name and returns the numeric value.
        For string values, returns the value as-is with None for the numeric value.

        Args:
            param_key: The parameter key to process (e.g., "mean", "sigma")

        Returns:
            Tuple of (processed_name, numeric_value_or_none)
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
        """
        Process a parameter from a list of values that can be either string references or numeric values.

        For numeric values, generates indexed unique names and returns the numeric values.
        For string values, returns the values as-is with None for the numeric values.

        Args:
            param_key: The parameter key to process (e.g., "factors", "coefficients")

        Returns:
            Tuple of (processed_names_list, numeric_values_list)
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
        """
        Reconstruct a parameter list from flattened indexed keys.

        Args:
            context: The context containing parameter values
            param_key: The base parameter key (e.g., "factors")

        Returns:
            list[TensorVar]: List of parameter values in order
        """
        result = []
        i = 0
        while f"{param_key}[{i}]" in self._parameters:
            param_name = self._parameters[f"{param_key}[{i}]"]
            result.append(context[param_name])
            i += 1
        return result

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
