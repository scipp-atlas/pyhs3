"""
HS3 Likelihood implementations.

Provides Pydantic classes for handling HS3 likelihood specifications
including likelihood mappings between distributions and data.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Annotated, cast

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

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
    aux_distributions: Annotated[
        list[str] | Distributions | None,
        make_fk_list_validator(Distribution),
        FKListSerializer,
        FKListSchema,
    ] = Field(default=None, repr=False)

    def validate_unique_axis_names(self, workspace: Workspace | None = None) -> None:
        """Validate that repeated observable names have compatible bounds.

        When *workspace* is provided, unresolved string FK references in ``data``
        are resolved via ``workspace.data`` before checking. Repeated names are
        supported because model inputs are bound per distribution/data pair;
        their ranges must still agree because the distribution graph has one
        normalization range per serialized observable name.
        """
        seen: dict[str, tuple[float, float, str]] = {}
        conflicts: list[str] = []
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
                    previous_min, previous_max, previous_datum = seen[axis.name]
                    if (axis.min, axis.max) != (previous_min, previous_max):
                        conflicts.append(
                            f"'{axis.name}' in '{datum.name}' has bounds "
                            f"({axis.min}, {axis.max}), but '{previous_datum}' "
                            f"has ({previous_min}, {previous_max})"
                        )
                else:
                    seen[axis.name] = (axis.min, axis.max, datum.name)
        if conflicts:
            msg = (
                f"Likelihood '{self.name}' has repeated observable axis names "
                f"with conflicting bounds: " + ", ".join(conflicts)
            )
            raise ValueError(msg)

    def observable_bindings(self) -> list[dict[str, str]]:
        """Return graph-input names for every distribution/data pair.

        A bare observable name remains the public input when it belongs to one
        datum. If distinct data objects reuse the name, each input is
        namespaced by its datum. Reusing the same datum is explicit sharing and
        therefore reuses the same graph input.
        """
        resolved_data = list(self.data)
        owners: dict[str, set[str]] = {}
        for datum in resolved_data:
            if isinstance(datum, str) or getattr(datum, "entries", None) is None:
                continue
            for axis in datum.axes or []:
                owners.setdefault(axis.name, set()).add(datum.name)

        bindings: list[dict[str, str]] = []
        owner_binding: dict[tuple[str, str], str] = {}
        used_names: Counter[str] = Counter()
        for pair_index, datum in enumerate(resolved_data):
            pair_bindings: dict[str, str] = {}
            if isinstance(datum, str) or getattr(datum, "entries", None) is None:
                bindings.append(pair_bindings)
                continue
            for axis in datum.axes or []:
                owner_key = (datum.name, axis.name)
                bound_name = owner_binding.get(owner_key)
                if bound_name is None:
                    bound_name = (
                        axis.name
                        if len(owners[axis.name]) == 1
                        else f"{datum.name}__{axis.name}"
                    )
                    if used_names[bound_name]:
                        bound_name = f"{bound_name}__pair{pair_index}"
                    owner_binding[owner_key] = bound_name
                    used_names[bound_name] += 1
                pair_bindings[axis.name] = bound_name
            bindings.append(pair_bindings)
        return bindings

    def data_arrays(self) -> dict[str, npt.NDArray[np.float64]]:
        """Observable data as numpy arrays keyed by axis name.

        Returns a dict mapping each observable axis name to a 1-D float64 array
        of event values.  Only data entries with both ``axes`` and ``entries``
        are included (i.e. :class:`~pyhs3.data.UnbinnedData`).

        Suitable for passing directly to compiled or JAX functions::

            fn(**likelihood.data_arrays(), **params)
        """
        result: dict[str, npt.NDArray[np.float64]] = {}
        bindings = self.observable_bindings()
        # self.data is guaranteed FK-resolved (no string entries after workspace construction).
        for pair_index, datum in enumerate(cast(Data, self.data)):
            if datum.axes is None:
                continue
            entries = getattr(datum, "entries", None)
            if entries is None:
                continue
            entries_arr = np.asarray(entries, dtype=np.float64)
            n_axes = len(datum.axes)
            if entries_arr.size == 0:
                entries_arr = entries_arr.reshape(0, n_axes)
            for ax_idx, axis in enumerate(datum.axes):
                result[bindings[pair_index][axis.name]] = entries_arr[:, ax_idx]
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
