from __future__ import annotations

import json
from pathlib import Path

import pytest

import pyhs3 as hs3


def summarize(node, max_depth=3, _depth=0):
    indent = "  " * _depth
    if _depth >= max_depth:
        print(f"{indent}…")
        return

    if isinstance(node, dict):
        for key, value in node.items():
            print(f"{indent}{key} -> {type(value).__name__}")
            summarize(value, max_depth, _depth + 1)

    elif isinstance(node, list):
        print(f"{indent}[list of {len(node)} items]")
        if node:
            summarize(node[0], max_depth, _depth + 1)

    else:
        print(f"{indent}{node!r} ({type(node).__name__})")


@pytest.fixture
def ws_json(request):
    """Load issue41_diHiggs_workspace.json file and return parsed JSON content.

    This workspace is from Alex Wang for the diHiggs gamgam bb analysis,
    related to GitHub issue #41.
    """
    json_path = Path(request.module.__file__).parent / "issue41_diHiggs_workspace.json"
    return json.loads(json_path.read_text(encoding="utf-8"))


@pytest.fixture
def ws_workspace(ws_json):
    """Create workspace from WS.json content."""
    return hs3.Workspace(ws_json)


def test_workspace_structure(ws_json):
    """Test and display workspace structure."""
    summarize(ws_json, max_depth=3)
    assert isinstance(ws_json, dict)


def test_workspace_loading(ws_workspace):
    """Test loading workspace from WS.json."""
    assert ws_workspace is not None
