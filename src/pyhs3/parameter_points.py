"""
HS3 Parameter Point implementations.

Provides Pydantic classes for handling HS3 parameter point specifications including
individual parameters and parameter sets for defining model parameter values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytensor.tensor as pt
from pydantic import BaseModel, Field, PrivateAttr, RootModel

from pyhs3.typing.aliases import TensorVar


class ParameterPoint(BaseModel):
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

    name: str
    value: float
    const: bool = False
    nbins: int | None = None
    kind: Callable[..., TensorVar] = Field(default=pt.scalar, exclude=True)


class ParameterSet(BaseModel):
    """
    Named collection of parameter specifications (matches HS3Spec structure).

    Represents a complete set of parameter values that can be used to
    evaluate a model. Each parameter set contains multiple individual
    parameter points with their names and values.

    Parameters:
        name: Name identifier for the parameter set
        parameters: List of ParameterPoint specifications
    """

    name: str
    parameters: list[ParameterPoint] = Field(default_factory=list)

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


class ParameterPoints(RootModel[list[ParameterSet]]):
    """
    Collection of HS3 parameter sets for model configuration.

    Manages a set of parameter set instances that define parameter values
    for model evaluation. Provides dict-like access to parameter sets by
    name and handles parameter set creation from configuration dictionaries.

    Attributes:
        parameter_sets: Mapping from parameter set names to ParameterSet instances.
    """

    root: list[ParameterSet] = Field(default_factory=list)
    _map: dict[str, ParameterSet] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        """Initialize computed collections after Pydantic validation."""
        self._map = {param_set.name: param_set for param_set in self.root}

    def __getitem__(self, item: str | int) -> ParameterSet:
        if isinstance(item, int):
            return self.root[item]
        return self._map[item]

    def get(
        self, item: str, default: ParameterSet | None = None
    ) -> ParameterSet | None:
        """Get a parameter set by name, returning default if not found."""
        return self._map.get(item, default)

    def __contains__(self, item: str) -> bool:
        return item in self._map

    def __iter__(self) -> Iterator[ParameterSet]:  # type: ignore[override]  # https://github.com/pydantic/pydantic/issues/8872
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)
