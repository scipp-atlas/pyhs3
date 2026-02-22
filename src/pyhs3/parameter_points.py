"""
HS3 Parameter Point implementations.

Provides Pydantic classes for handling HS3 parameter point specifications including
individual parameters and parameter sets for defining model parameter values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytensor.tensor as pt
from pydantic import ConfigDict, Field

from pyhs3.collections import NamedCollection, NamedModel
from pyhs3.typing.aliases import TensorVar


class ParameterPoint(NamedModel):
    """
    Individual parameter specification with name and value.

    Represents a single parameter specification with its value and optional configuration.
    Used within parameter sets to specify concrete parameter values for model
    evaluation and fitting.

    Parameters:
        name: Name identifier for the parameter
        value: Numeric value of the parameter
        const: Whether parameter is constant (optional, defaults to False)
        nbins: Number of bins for binned parameters (optional)
        kind: Type of tensor to create (optional, defaults to pt.scalar)
    """

    model_config = ConfigDict()

    value: float = Field(..., repr=False)
    const: bool = Field(default=False, repr=False)
    nbins: int | None = Field(default=None, repr=False)
    kind: Callable[..., TensorVar] = Field(default=pt.scalar, exclude=True, repr=False)


class ParameterSet(NamedModel):
    """
    Named collection of parameter specifications (matches HS3Spec structure).

    Represents a complete set of parameter values that can be used to
    evaluate a model. Each parameter set contains multiple individual
    parameter points with their names and values.

    Parameters:
        name: Name identifier for the parameter set
        parameters: List of ParameterPoint specifications
    """

    model_config = ConfigDict()

    parameters: list[ParameterPoint] = Field(default_factory=list, repr=False)

    @property
    def points(self) -> dict[str, ParameterPoint]:
        """Compatibility property for core.py access."""
        return {param.name: param for param in self.parameters}

    def __len__(self) -> int:
        """Number of parameters in this set."""
        return len(self.parameters)

    def __contains__(self, param_name: str) -> bool:
        """Check if a parameter with the given name exists in this set."""
        return param_name in self.points

    def __getitem__(self, item: str | int) -> ParameterPoint:
        """Get a parameter by name or index."""
        if isinstance(item, int):
            return self.parameters[item]
        return self.points[item]

    def get(
        self, param_name: str, default: ParameterPoint | None = None
    ) -> ParameterPoint | None:
        """Get a parameter by name, returning default if not found."""
        return self.points.get(param_name, default)

    def __iter__(self) -> Iterator[ParameterPoint]:  # type: ignore[override]
        """Iterate over the parameters."""
        return iter(self.parameters)


class ParameterPoints(NamedCollection[ParameterSet]):
    """
    Collection of HS3 parameter sets for model configuration.

    Manages a set of parameter set instances that define parameter values
    for model evaluation. Provides dict-like access to parameter sets by
    name and handles parameter set creation from configuration dictionaries.

    Attributes:
        parameter_sets: Mapping from parameter set names to ParameterSet instances.
    """

    root: list[ParameterSet] = Field(default_factory=list)
