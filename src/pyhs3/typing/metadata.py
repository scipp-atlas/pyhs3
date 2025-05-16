"""
typing metadata
"""

from __future__ import annotations

from typing import TypedDict


class PackageInfo(TypedDict):
    """
    PackageInfo
    """

    name: str
    version: str


class Metadata(TypedDict):
    """
    Metadata
    """

    hs3_version: str
    packages: list[PackageInfo]
