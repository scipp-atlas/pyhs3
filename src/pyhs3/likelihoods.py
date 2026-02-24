"""
HS3 Likelihood implementations.

Provides Pydantic classes for handling HS3 likelihood specifications
including likelihood mappings between distributions and data.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field, model_validator

from pyhs3.collections import NamedCollection, NamedModel
from pyhs3.data import Data, Datum
from pyhs3.distributions import Distributions
from pyhs3.distributions.core import Distribution
from pyhs3.typing.annotations import (
    FKListSchema,
    FKListSerializer,
    make_fk_list_validator,
)


class Likelihood(NamedModel):
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

    distributions: Annotated[
        list[str] | Distributions,
        make_fk_list_validator(Distribution),
        FKListSerializer,
        FKListSchema,
    ] = Field(..., repr=False)
    data: Annotated[
        list[str] | Data,
        make_fk_list_validator(Datum),
        FKListSerializer,
        FKListSchema,
    ] = Field(..., repr=False)
    aux_distributions: list[str] | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def validate_distributions_data_pairing(self) -> Likelihood:
        """Validate that distributions and data are properly paired."""
        if len(self.distributions) != len(self.data):
            msg = (
                f"Likelihood '{self.name}': distributions and data must have the same length, "
                f"got {len(self.distributions)} distributions and {len(self.data)} data entries"
            )
            raise ValueError(msg)
        if len(self.distributions) == 0 and not self.aux_distributions:
            msg = (
                f"Likelihood '{self.name}': must have at least one distribution/data pair "
                f"or provide aux_distributions"
            )
            raise ValueError(msg)
        return self


class Likelihoods(NamedCollection[Likelihood]):
    """
    Collection of HS3 likelihood specifications.

    Manages a set of likelihood instances that define mappings between
    distributions and observations for statistical inference.
    Provides dict-like access to likelihoods by name.
    """

    root: list[Likelihood] = Field(default_factory=list)
