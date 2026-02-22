"""
HS3 Analysis implementations.

Provides Pydantic classes for handling HS3 analysis specifications
including analysis configurations with parameters of interest and domains.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from pyhs3.collections import NamedCollection, NamedModel


class Analysis(NamedModel):
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

    model_config = ConfigDict()

    name: str = Field(..., repr=True)
    likelihood: str = Field(..., repr=False)
    parameters_of_interest: list[str] | None = Field(default=None, repr=False)
    domains: list[str] = Field(..., repr=False)
    init: str | None = Field(default=None, repr=False)
    prior: str | None = Field(default=None, repr=False)


class Analyses(NamedCollection[Analysis]):
    """
    Collection of HS3 analysis specifications.

    Manages a set of analysis instances that define automated analysis
    configurations with likelihoods, parameters of interest, and domains.
    Provides dict-like access to analyses by name.
    """

    root: list[Analysis] = Field(default_factory=list)
