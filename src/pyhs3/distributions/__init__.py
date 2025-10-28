"""
HS3 Distribution implementations.

Provides classes for handling various probability distributions including
Gaussian, Mixture, Product, Crystal Ball, and Generic distributions.
Includes both standard HS3 distributions and CMS-specific extensions.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Any, TypeVar

from pydantic import Field, PrivateAttr, RootModel

# Import modules instead of individual classes
from pyhs3.distributions import (
    basic,
    cms,
    composite,
    histfactory,
    histogram,
    mathematical,
    physics,
)
from pyhs3.distributions.core import Distribution
from pyhs3.exceptions import custom_error_msg

# Export distribution classes for backwards compatibility
# Basic distributions
GaussianDist = basic.GaussianDist
UniformDist = basic.UniformDist
PoissonDist = basic.PoissonDist
ExponentialDist = basic.ExponentialDist
LogNormalDist = basic.LogNormalDist
LandauDist = basic.LandauDist

# Composite distributions
MixtureDist = composite.MixtureDist
ProductDist = composite.ProductDist

# Histogram distributions
HistogramDist = histogram.HistogramDist

# HistFactory distributions
HistFactoryDistChannel = histfactory.HistFactoryDistChannel

# Mathematical distributions
GenericDist = mathematical.GenericDist
PolynomialDist = mathematical.PolynomialDist
BernsteinPolyDist = mathematical.BernsteinPolyDist

# Physics distributions
CrystalBallDist = physics.CrystalBallDist
AsymmetricCrystalBallDist = physics.AsymmetricCrystalBallDist
ArgusDist = physics.ArgusDist

# CMS-specific distributions
FastVerticalInterpHistPdf2Dist = cms.FastVerticalInterpHistPdf2Dist
GGZZBackgroundDist = cms.GGZZBackgroundDist
QQZZBackgroundDist = cms.QQZZBackgroundDist
FastVerticalInterpHistPdf2D2Dist = cms.FastVerticalInterpHistPdf2D2Dist

# Export all distribution classes
__all__ = [
    "ArgusDist",
    "AsymmetricCrystalBallDist",
    "BernsteinPolyDist",
    "CrystalBallDist",
    "Distribution",
    "Distributions",
    "ExponentialDist",
    "FastVerticalInterpHistPdf2D2Dist",
    "FastVerticalInterpHistPdf2Dist",
    "GGZZBackgroundDist",
    "GaussianDist",
    "GenericDist",
    "HistFactoryDistChannel",
    "HistogramDist",
    "LandauDist",
    "LogNormalDist",
    "MixtureDist",
    "PoissonDist",
    "PolynomialDist",
    "ProductDist",
    "QQZZBackgroundDist",
    "UniformDist",
    "registered_distributions",
]

DistT = TypeVar("DistT", bound="Distribution")

# Combine all distribution registries
registered_distributions: dict[str, type[Distribution]] = {
    **basic.distributions,
    **composite.distributions,
    **histfactory.distributions,
    **histogram.distributions,
    **mathematical.distributions,
    **physics.distributions,
    **cms.distributions,
}

# Type alias for all distribution types using discriminated union
DistributionType = Annotated[
    # Basic distributions
    basic.GaussianDist
    | basic.UniformDist
    | basic.PoissonDist
    | basic.ExponentialDist
    | basic.LogNormalDist
    | basic.LandauDist
    # Composite distributions
    | composite.MixtureDist
    | composite.ProductDist
    # Histogram distributions
    | histogram.HistogramDist
    # HistFactory distributions
    | histfactory.HistFactoryDistChannel
    # Mathematical distributions
    | mathematical.GenericDist
    | mathematical.PolynomialDist
    | mathematical.BernsteinPolyDist
    # Physics distributions
    | physics.CrystalBallDist
    | physics.AsymmetricCrystalBallDist
    | physics.ArgusDist
    # CMS distributions
    | cms.FastVerticalInterpHistPdf2Dist
    | cms.GGZZBackgroundDist
    | cms.QQZZBackgroundDist
    | cms.FastVerticalInterpHistPdf2D2Dist,
    Field(discriminator="type"),
]


class Distributions(RootModel[list[DistributionType]]):
    """
    Collection of distributions for a probabilistic model.

    Manages a set of distribution instances, providing dict-like access
    by distribution name. Handles distribution creation from configuration
    dictionaries and maintains a registry of available distribution types.

    Attributes:
        dists: Mapping from distribution names to Distribution instances.

    HS3 Reference:
        :hs3:label:`distributions <hs3.sec:distributions>`
    """

    root: Annotated[
        list[DistributionType],
        custom_error_msg(
            {
                "union_tag_invalid": "Unknown distribution type '{tag}' does not match any of the expected distributions: {expected_tags}"
            }
        ),
    ] = Field(default_factory=list)
    _map: dict[str, Distribution] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        """Initialize computed collections after Pydantic validation."""
        self._map = {dist.name: dist for dist in self.root}

    def __getitem__(self, item: str) -> Distribution:
        return self._map[item]

    def __contains__(self, item: str) -> bool:
        return item in self._map

    def __iter__(self) -> Iterator[Distribution]:  # type: ignore[override]  # https://github.com/pydantic/pydantic/issues/8872
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)
