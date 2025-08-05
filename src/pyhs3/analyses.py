"""
HS3 Analysis implementations.

Provides Pydantic classes for handling HS3 analysis specifications
including analysis configurations with parameters of interest and domains.
"""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel, Field, RootModel


class Analysis(BaseModel):
    """
    Analysis specification defining automated analysis parameters.

    Represents a complete analysis configuration that specifies which likelihood
    to use, parameters of interest, parameter domains, and optional priors.
    All parameters from the likelihood distributions must either be in the
    referenced domain or set to 'const' in the parameter point.

    Attributes:
        name: Custom string identifier for the analysis
        likelihood: Name referencing a likelihood in the likelihoods component
        parameters_of_interest: Optional array of parameter names of interest
        domains: Array of domain names from domains component
        init: Optional name of initial values from parameter_points component
        prior: Optional name of prior distribution from distributions component
    """

    name: str
    likelihood: str
    parameters_of_interest: list[str] | None = Field(default=None)
    domains: list[str]
    init: str | None = Field(default=None)
    prior: str | None = Field(default=None)


class Analyses(RootModel[list[Analysis]]):
    """
    Collection of HS3 analysis specifications.

    Manages a set of analysis instances that define automated analysis
    configurations with likelihoods, parameters of interest, and domains.
    Provides dict-like access to analyses by name.
    """

    root: list[Analysis] = Field(default_factory=list)

    @property
    def analysis_map(self) -> dict[str, Analysis]:
        """Mapping from analysis names to Analysis instances."""
        return {analysis.name: analysis for analysis in self.root}

    def __len__(self) -> int:
        """Number of analyses in this collection."""
        return len(self.root)

    def __contains__(self, analysis_name: str) -> bool:
        """Check if an analysis with the given name exists."""
        return analysis_name in self.analysis_map

    def __getitem__(self, item: str | int) -> Analysis:
        """Get an analysis by name or index."""
        if isinstance(item, int):
            return self.root[item]
        return self.analysis_map[item]

    def get(
        self, analysis_name: str, default: Analysis | None = None
    ) -> Analysis | None:
        """Get an analysis by name, returning default if not found."""
        return self.analysis_map.get(analysis_name, default)

    def __iter__(self) -> Iterator[Analysis]:  # type: ignore[override]
        """Iterate over the analyses."""
        return iter(self.root)
