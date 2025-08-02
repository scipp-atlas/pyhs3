"""
HS3 Domain implementations.

Provides Pydantic classes for handling HS3 domain specifications including
axes and product domains for defining parameter spaces and integration regions.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, RootModel, model_validator

from pyhs3.exceptions import custom_error_msg


class Axis(BaseModel):
    """
    Axis specification for parameter domains.

    Defines a single axis of a parameter space with a name and numeric range.
    Used within domains to specify multi-dimensional parameter spaces
    for integration, likelihood evaluation, and parameter constraints.

    Parameters:
        name: Name identifier for the axis
        min: Minimum value for the axis range (optional)
        max: Maximum value for the axis range (optional)
    """

    name: str
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def validate_range(self) -> Axis:
        """Validate that max >= min when both are provided."""
        if self.max is not None and self.min is not None and self.max < self.min:
            msg = f"Axis '{self.name}': max ({self.max}) must be >= min ({self.min})"
            raise ValueError(msg)
        return self


class Domain(BaseModel):
    """
    Base class for HS3 domain specifications.

    Provides the foundation for all domain implementations,
    handling common properties like name and type identification.
    Domains define parameter spaces for integration, constraints,
    and likelihood evaluation.

    Parameters:
        name: Name identifier for the domain
        type: Domain type identifier
    """

    name: str
    type: str

    @property
    def dimension(self) -> int:
        """Number of dimensions in this domain."""
        raise NotImplementedError

    @property
    def axis_names(self) -> list[str]:
        """List of axis names in this domain. Note: may not be implemented for all domain types."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of axes in this domain."""
        return 0

    def __contains__(self, _axis_name: str) -> bool:
        """Check if an axis with the given name exists in this domain."""
        return False

    def get(self, _axis_name: str, default: Any = None) -> Any:
        """Get axis bounds for a parameter name. Note: may not be implemented for all domain types."""
        return default

    def __getitem__(self, axis_name: str) -> Any:
        """Get axis bounds for a parameter name (dict-like access)."""
        raise KeyError(axis_name)


class ProductDomain(Domain):
    """
    Product domain specification for multi-dimensional parameter spaces.

    Defines a Cartesian product of axes to create multi-dimensional parameter
    domains. Used for specifying integration regions, parameter constraints,
    and likelihood evaluation domains in HS3 specifications.

    The domain represents the Cartesian product: axis₁ x axis₂ x ... x axisₙ
    where each axis defines a one-dimensional range.

    Parameters:
        name: Name identifier for the domain
        type: Domain type identifier (always "product_domain")
        axes: List of Axis specifications defining each dimension
    """

    type: Literal["product_domain"] = "product_domain"
    axes: list[Axis] = Field(default_factory=list)
    _axes_map: dict[str, Axis] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def initialize_axes_map(self) -> ProductDomain:
        """Initialize the internal axes mapping for fast lookup."""
        self._axes_map = {axis.name: axis for axis in self.axes}
        return self

    @model_validator(mode="after")
    def validate_unique_axis_names(self) -> ProductDomain:
        """Validate that all axis names are unique within the domain."""
        axis_names = [axis.name for axis in self.axes]
        if len(axis_names) != len(set(axis_names)):
            duplicates = [name for name in axis_names if axis_names.count(name) > 1]
            msg = (
                f"Domain '{self.name}' contains duplicate axis names: {set(duplicates)}"
            )
            raise ValueError(msg)
        return self

    @property
    def dimension(self) -> int:
        """Number of dimensions (axes) in this domain."""
        return len(self.axes)

    @property
    def axis_names(self) -> list[str]:
        """List of axis names in this domain."""
        return [axis.name for axis in self.axes]

    def __len__(self) -> int:
        """Number of axes in this domain."""
        return len(self.axes)

    def __contains__(self, axis_name: str) -> bool:
        """Check if an axis with the given name exists in this domain."""
        return axis_name in self._axes_map

    def get(
        self, axis_name: str, default: tuple[float | None, float | None] = (None, None)
    ) -> tuple[float | None, float | None]:
        """
        Get axis bounds for a parameter name.

        Args:
            axis_name: Name of the axis to get bounds for.
            default: Default value to return if axis not found.

        Returns:
            Tuple of (min, max) bounds if axis exists, otherwise default.
        """
        axis = self._axes_map.get(axis_name)
        return (axis.min, axis.max) if axis is not None else default

    def __getitem__(self, axis_name: str) -> tuple[float | None, float | None]:
        """Get axis bounds for a parameter name (dict-like access)."""
        axis = self._axes_map.get(axis_name)
        if axis is not None:
            return (axis.min, axis.max)
        msg = f"No axis named '{axis_name}' found in domain '{self.name}'"
        raise KeyError(msg)


# Define the union type for all domain configurations
DomainConfig = ProductDomain

# Registry for domain types
registered_domains: dict[str, type[Domain]] = {
    "product_domain": ProductDomain,
}

# Type alias for all domain types using discriminated union
# Currently only ProductDomain exists, but this allows for future domain types
DomainType = Annotated[ProductDomain, Field(discriminator="type")]


class Domains(RootModel[list[DomainType]]):
    """
    Collection of HS3 domains for parameter space definitions.

    Manages a set of domain instances that define parameter spaces,
    integration regions, and constraints. Provides dict-like access
    to domains by name and handles domain creation from configuration
    dictionaries.

    Attributes:
        domains: Mapping from domain names to Domain instances.
    """

    root: Annotated[
        list[DomainType],
        custom_error_msg(
            {
                "union_tag_invalid": "Unknown domain type '{tag}' does not match any of the expected domains: {expected_tags}"
            }
        ),
    ]
    _map: dict[str, Domain] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        """Initialize computed collections after Pydantic validation."""
        self._map = {domain.name: domain for domain in self.root}

    def __getitem__(self, item: str | int) -> Domain:
        if isinstance(item, int):
            return self.root[item]
        return self._map[item]

    def get(self, item: str, default: Domain | None = None) -> Domain | None:
        """Get a domain by name, returning default if not found."""
        return self._map.get(item, default)

    def __contains__(self, item: str) -> bool:
        return item in self._map

    def __iter__(self) -> Iterator[Domain]:  # type: ignore[override]  # https://github.com/pydantic/pydantic/issues/8872
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)
