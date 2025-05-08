"""
typing misc
"""

from __future__ import annotations

from typing import TypedDict


class CombinedDistributions(TypedDict):
    """
    CombinedDistributions
    """

    distributions: list[str]
    index_cat: str
    indices: list[int]
    labels: list[str]


class ROOTInternal(TypedDict):
    """
    ROOTInternal
    """

    combined_distributions: dict[str, CombinedDistributions]


class Misc(TypedDict):
    """
    Misc
    """

    ROOT_internal: ROOTInternal
