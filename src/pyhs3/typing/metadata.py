from __future__ import annotations

from typing import TypedDict


class PackageInfo(TypedDict):
    name: str
    version: str


class Metadata(TypedDict):
    hs3_version: str
    packages: list[PackageInfo]
