"""
Context class for HS3 parameter resolution.

Provides a dictionary-like interface that can handle both single parameter names
and lists of parameter names for accessing PyTensor variables.
"""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, Mapping, ValuesView

from pyhs3.typing.aliases import TensorVar


class Context:
    """
    Context for parameter resolution in HS3 expressions.

    Provides access to both user-provided parameters and model-computed values
    (such as auxiliary parameters used by constraints). Distinguishes between
    parameters and computed values while providing a unified access interface.
    """

    def __init__(
        self,
        parameters: Mapping[str, TensorVar],
        auxiliaries: Mapping[str, TensorVar] | None = None,
        observables: Mapping[str, tuple[TensorVar, TensorVar]] | None = None,
    ) -> None:
        """
        Initialize context with parameter data.

        Args:
            parameters: Dictionary of user-provided parameters
            auxiliaries: Dictionary of model-computed auxiliary values
            observables: Dictionary mapping observable names to (lower, upper) bound tuples

        Raises:
            ValueError: If there's any overlap between parameter and auxiliary names
        """
        self._parameters = dict(parameters)
        self._auxiliaries = dict(auxiliaries) if auxiliaries else {}
        self._observables = dict(observables) if observables else {}

        # Validate no key duplication between parameters and auxiliaries
        parameter_keys = set(self._parameters.keys())
        auxiliary_keys = set(self._auxiliaries.keys())
        overlap = parameter_keys & auxiliary_keys

        if overlap:
            msg = f"Parameter names cannot overlap between parameters and auxiliaries. Overlapping names: {sorted(overlap)}"
            raise ValueError(msg)

    def __getitem__(self, key: str) -> TensorVar:
        """
        Get parameter value from context.

        Checks parameters first, then auxiliaries.

        Args:
            key: Parameter name (str)

        Returns:
            TensorVar: The parameter value

        Raises:
            KeyError: If parameter is not found in either parameters or auxiliaries
        """
        if key in self._parameters:
            return self._parameters[key]
        if key in self._auxiliaries:
            return self._auxiliaries[key]
        msg = f"Parameter '{key}' not found in context"
        raise KeyError(msg)

    def __contains__(self, key: str) -> bool:
        """Check if parameter name exists in context."""
        return key in self._parameters or key in self._auxiliaries

    @property
    def parameters(self) -> dict[str, TensorVar]:
        """Get read-only view of parameters."""
        return self._parameters.copy()

    @property
    def auxiliaries(self) -> dict[str, TensorVar]:
        """Get read-only view of auxiliaries."""
        return self._auxiliaries.copy()

    @property
    def observables(self) -> dict[str, tuple[TensorVar, TensorVar]]:
        """Observable names mapped to (lower, upper) bound PyTensor expressions."""
        return self._observables.copy()

    def keys(self) -> KeysView[str]:
        """Get all parameter names (both parameters and auxiliaries)."""
        # Create a view of combined keys
        combined = {**self._parameters, **self._auxiliaries}
        return combined.keys()

    def values(self) -> ValuesView[TensorVar]:
        """Get all parameter values (both parameters and auxiliaries)."""
        # Create a view of combined values
        combined = {**self._parameters, **self._auxiliaries}
        return combined.values()

    def items(self) -> ItemsView[str, TensorVar]:
        """Get all parameter name-value pairs (both parameters and auxiliaries)."""
        # Create a view of combined items
        combined = {**self._parameters, **self._auxiliaries}
        return combined.items()

    def copy(self) -> Context:
        """Create a copy of this context."""
        return Context(
            parameters=self._parameters.copy(),
            auxiliaries=self._auxiliaries.copy(),
            observables=self._observables.copy(),
        )
