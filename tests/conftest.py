from __future__ import annotations

import pathlib

# shutil is nicer, but doesn't work: https://bugs.python.org/issue20849
from functools import partial
from shutil import copytree as _copytree

import pytest

# ignore specific files from being collected
collect_ignore = ["test_pdf/rf501_simultaneouspdf.py"]

copytree = partial(_copytree, dirs_exist_ok=True)


def pytest_addoption(parser):
    """Add command line option to run slow tests."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --runslow option is given."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def datadir(tmp_path, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    """
    # this gets the module name (e.g. /path/to/module-qc-analysis-tools/tests/test_cli.py)
    # and then gets the directory by removing the suffix (e.g. /path/to/module-qc-analysis-tools/tests/test_cli)
    test_dir = pathlib.Path(request.module.__file__).with_suffix("")

    if test_dir.is_dir():
        copytree(test_dir, str(tmp_path))

    return tmp_path
