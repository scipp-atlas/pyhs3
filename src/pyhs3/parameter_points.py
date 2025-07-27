"""
HS3 Parameter Point implementations.

Provides Pydantic classes for handling HS3 parameter point specifications including
individual parameters and parameter sets for defining model parameter values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytensor.tensor as pt
from pydantic import BaseModel, Field

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
    """

    name: str
    value: float
    const: bool = False
    nbins: int | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ParameterPoint:
        """
        Creates a ParameterPoint from a dictionary configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            ParameterPoint: The created ParameterPoint instance.
        """
        return cls(**config)


class ParameterSet(BaseModel):
    """
    Named collection of parameter specifications (matches HS3Spec structure).

    Represents a complete set of parameter values that can be used to
    evaluate a model. Each parameter set contains multiple individual
    parameter points with their names and values.

    Parameters:
        name: Name identifier for the parameter set
        parameters: List of ParameterPoint specifications
        kind: Type of tensor to create (optional, defaults to pt.scalar)
    """

    name: str
    parameters: list[ParameterPoint]
    kind: Callable[..., TensorVar] = Field(default=pt.scalar, exclude=True)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ParameterSet:
        """
        Creates a ParameterSet from a dictionary configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            ParameterSet: The created ParameterSet instance.
        """
        # Convert parameter dicts to ParameterPoint objects
        if "parameters" in config:
            config = config.copy()
            config["parameters"] = [
                ParameterPoint.from_dict(param) if isinstance(param, dict) else param
                for param in config["parameters"]
            ]
        return cls(**config)

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


class ParameterCollection:
    """
    Collection of HS3 parameter sets for model configuration.

    Manages a set of parameter set instances that define parameter values
    for model evaluation. Provides dict-like access to parameter sets by
    name and handles parameter set creation from configuration dictionaries.

    Attributes:
        parameter_sets: Mapping from parameter set names to ParameterSet instances.
    """

    def __init__(self, parameter_points: list[dict[str, Any]]) -> None:
        """
        Collection of parameter sets that define parameter values.

        Args:
            parameter_points: List of parameter set configurations from HS3 spec (called parameter_points in spec)
        """
        self.parameter_sets: dict[str, ParameterSet] = {}
        for set_config in parameter_points:
            param_set = ParameterSet.from_dict(set_config)
            self.parameter_sets[param_set.name] = param_set

    def __getitem__(self, item: str | int) -> ParameterSet:
        if isinstance(item, int):
            key = list(self.parameter_sets.keys())[item]
            return self.parameter_sets[key]
        return self.parameter_sets[item]

    def get(
        self, item: str, default: ParameterSet | None = None
    ) -> ParameterSet | None:
        """Get a parameter set by name, returning default if not found."""
        return self.parameter_sets.get(item, default)

    def __contains__(self, item: str) -> bool:
        return item in self.parameter_sets

    def __iter__(self) -> Iterator[ParameterSet]:
        return iter(self.parameter_sets.values())

    def __len__(self) -> int:
        return len(self.parameter_sets)
