"""
HS3 Metadata implementations.

Provides Pydantic classes for handling HS3 metadata specifications including
package information, authorship, and publication details following the HS3 specification.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PackageInfo(BaseModel):
    """
    Package information for tracking software dependencies.

    Represents a software package with its name and version that was used
    in the creation or analysis of the HS3 specification.

    Parameters:
        name: Name of the package/software
        version: Version string of the package
    """

    name: str
    version: str

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> PackageInfo:
        """
        Creates a PackageInfo instance from a dictionary configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            PackageInfo: The created PackageInfo instance.
        """
        return cls(**config)


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

    hs3_version: str
    packages: list[PackageInfo] | None = None
    authors: list[str] | None = None
    publications: list[str] | None = None
    description: str | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Metadata:
        """
        Creates a Metadata instance from a dictionary configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Metadata: The created Metadata instance.
        """
        # Convert package dicts to PackageInfo objects if present
        if "packages" in config and config["packages"] is not None:
            config = config.copy()
            config["packages"] = [
                PackageInfo.from_dict(pkg) if isinstance(pkg, dict) else pkg
                for pkg in config["packages"]
            ]
        return cls(**config)
