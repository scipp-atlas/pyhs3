import math
import json
import pytensor.tensor as pt
from pytensor.compile.function import function
from pytensor import printing
from pyhs3 import typing as T
import pyhs3 as hs3
# from pyhs3.core import GaussianDist, boundedscalar  # import from wherever you defined them
import pytest
import argparse

def summarize(node, max_depth=3, _depth=0):

    indent = "  " * _depth
    if _depth >= max_depth:
        print(f"{indent}…")
        return

    if isinstance(node, dict):
        for key, value in node.items():
            print(f"{indent}{key} -> {type(value).__name__}")
            summarize(value, max_depth, _depth+1)

    elif isinstance(node, list):
        print(f"{indent}[list of {len(node)} items]")
        if node:
            summarize(node[0], max_depth, _depth+1)

    else:
        print(f"{indent}{node!r} ({type(node).__name__})")

with open("WS.json", "r", encoding="utf-8") as f:
    json_content = f.read()

summarize(json.loads(json_content), max_depth=3)

workspace = hs3.Workspace(json.loads(json_content))