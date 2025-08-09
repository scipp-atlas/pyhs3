"""
HS3 Metadata implementations.

Provides Pydantic classes for handling HS3 metadata specifications including
package information, authorship, and publication details following the HS3 specification.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PackageInfo(BaseModel):
    """
    Package information for tracking software dependencies.

    Represents a software package with its name and version that was used
    in the creation or analysis of the HS3 specification.

    Parameters:
        name: Name of the package/software
        version: Version string of the package
    """

    model_config = ConfigDict()

    name: str = Field(repr=True)
    version: str = Field(repr=False)


class Metadata(BaseModel):
    """
    HS3 metadata containing version information and optional attribution.

    Contains required HS3 version information along with optional details about
    packages used, authors, publications, and descriptions following the HS3
    metadata specification.

    Parameters:
        hs3_version: Version of the HS3 specification used (required)
        packages: List of software packages used (optional)
        authors: List of authors or collaborations (optional)
        publications: List of publication identifiers (optional)
        description: Short description or abstract (optional)
    """

    model_config = ConfigDict()

    hs3_version: str = Field(..., repr=False)
    packages: list[PackageInfo] | None = Field(default=None, repr=False)
    authors: list[str] | None = Field(default=None, repr=False)
    publications: list[str] | None = Field(default=None, repr=False)
    description: str | None = Field(default=None, repr=False)
