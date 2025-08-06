"""
Context class for HS3 parameter resolution.

Provides a dictionary-like interface that can handle both single parameter names
and lists of parameter names for accessing PyTensor variables.
"""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, ValuesView

from pyhs3.typing.aliases import TensorVar


class Context:
    """
    Context for parameter resolution in HS3 expressions.

    Wraps a dictionary of parameter names to PyTensor variables and provides
    extended indexing that can handle both single strings and lists of strings.
    """

    def __init__(self, data: dict[str, TensorVar]) -> None:
        """
        Initialize context with parameter data.

        Args:
            data: Dictionary mapping parameter names to PyTensor variables
        """
        self._data = data

    def __getitem__(self, key: str) -> TensorVar:
        """
        Get parameter value from context.

        Args:
            key: Parameter name (str)

        Returns:
            TensorVar: The parameter value
        """
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        """Check if parameter name exists in context."""
        return key in self._data

    def keys(self) -> KeysView[str]:
        """Get parameter names."""
        return self._data.keys()

    def values(self) -> ValuesView[TensorVar]:
        """Get parameter values."""
        return self._data.values()

    def items(self) -> ItemsView[str, TensorVar]:
        """Get parameter name-value pairs."""
        return self._data.items()
