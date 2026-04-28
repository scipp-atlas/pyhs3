"""
HS3 Likelihood implementations.

Provides Pydantic classes for handling HS3 likelihood specifications
including likelihood mappings between distributions and data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict, Field, model_validator

from pyhs3.collections import NamedCollection, NamedModel
from pyhs3.data import Data, Datum

if TYPE_CHECKING:
    from pyhs3.workspace import Workspace
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

    def validate_unique_axis_names(self, workspace: Workspace | None = None) -> None:
        """Raise ValueError if any observable axis name appears more than once.

        When *workspace* is provided, unresolved string FK references in ``data``
        are resolved via ``workspace.data`` before checking.  Without a workspace,
        string entries are skipped.
        """
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        for entry in self.data:
            if isinstance(entry, str):
                if workspace is None or workspace.data is None:
                    continue
                datum = workspace.data.get(entry)
                if datum is None:
                    continue
            else:
                datum = entry
            for axis in datum.axes or []:
                if axis.name in seen:
                    duplicates.append(
                        f"'{axis.name}' in '{datum.name}' and '{seen[axis.name]}'"
                    )
                else:
                    seen[axis.name] = datum.name
        if duplicates:
            msg = (
                f"Likelihood '{self.name}' has duplicate observable axis names: "
                + ", ".join(duplicates)
            )
            raise ValueError(msg)

    def data_arrays(self) -> dict[str, npt.NDArray[np.float64]]:
        """Observable data as numpy arrays keyed by axis name.

        Returns a dict mapping each observable axis name to a 1-D float64 array
        of event values.  Only data entries with both ``axes`` and ``entries``
        are included (i.e. :class:`~pyhs3.data.UnbinnedData`).

        Suitable for passing directly to compiled or JAX functions::

            fn(**likelihood.data_arrays(), **params)
        """
        result: dict[str, npt.NDArray[np.float64]] = {}
        # self.data is guaranteed FK-resolved (no string entries after workspace construction).
        for datum in cast(Data, self.data):
            if datum.axes is None:
                continue
            entries = getattr(datum, "entries", None)
            if entries is None:
                continue
            entries_arr = np.asarray(entries, dtype=np.float64)
            for ax_idx, axis in enumerate(datum.axes):
                result[axis.name] = entries_arr[:, ax_idx]
        return result

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
