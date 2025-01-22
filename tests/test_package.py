from __future__ import annotations

import importlib.metadata

import pyhs3 as m


def test_version():
    assert importlib.metadata.version("pyhs3") == m.__version__
