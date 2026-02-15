"""
HS3 Likelihood implementations.

Provides Pydantic classes for handling HS3 likelihood specifications
including likelihood mappings between distributions and data.
"""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict, Field, RootModel


class Likelihood(BaseModel):
    """
    Likelihood specification mapping distributions to observations.

    Represents a likelihood function that combines parameterized distributions
    with observations to generate a likelihood function L(θ₁, θ₂, ...).
    The likelihood is the product of PDFs evaluated at observed data points.

    Attributes:
        name: Custom string identifier for the likelihood
        distributions: Array of strings referencing distributions
        data: Array of strings referencing data or inline values for constraints
        aux_distributions: Optional array of auxiliary distributions for regularization
    """

    model_config = ConfigDict()

    name: str = Field(..., repr=True)
    distributions: list[str] = Field(..., repr=False)
    data: list[str] = Field(..., repr=False)
    aux_distributions: list[str] | None = Field(default=None, repr=False)


class Likelihoods(RootModel[list[Likelihood]]):
    """
    Collection of HS3 likelihood specifications.

    Manages a set of likelihood instances that define mappings between
    distributions and observations for statistical inference.
    Provides dict-like access to likelihoods by name.
    """

    root: list[Likelihood] = Field(default_factory=list)

    @property
    def likelihood_map(self) -> dict[str, Likelihood]:
        """Mapping from likelihood names to Likelihood instances."""
        return {likelihood.name: likelihood for likelihood in self.root}

    def __len__(self) -> int | float:
        """Number of likelihoods in this collection."""
        return len(self.root)

    def __contains__(self, likelihood_name: str) -> bool:
        """Check if a likelihood with the given name exists."""
        return likelihood_name in self.likelihood_map

    def __getitem__(self, item: str | int) -> Likelihood:
        """Get a likelihood by name or index."""
        if isinstance(item, int):
            return self.root[item]
        return self.likelihood_map[item]

    def get(
        self, likelihood_name: str, default: Likelihood | None = None
    ) -> Likelihood | None:
        """Get a likelihood by name, returning default if not found."""
        return self.likelihood_map.get(likelihood_name, default)

    def __iter__(self) -> Iterator[Likelihood]:  # type: ignore[override]
        """Iterate over the likelihoods."""
        return iter(self.root)
