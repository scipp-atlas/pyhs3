from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from pyhs3.model import Model

sys.path.insert(0, str(Path(__file__).parent))

import hs3testsuite_pyhs3_backend as backend


@dataclass
class _Named:
    name: str


class _Collection(list[_Named]):
    def get(self, name: str) -> _Named | None:
        return next((item for item in self if item.name == name), None)


def _workspace_stub() -> Any:
    return SimpleNamespace(
        distributions=_Collection([_Named("model"), _Named("extra_pdf")]),
        functions=_Collection([_Named("mean")]),
        data=_Collection([_Named("observed")]),
    )


def test_structure_check_requires_subset_and_allows_extras() -> None:
    adapter = backend.PyHS3Backend()
    adapter.run_structure_check(
        _workspace_stub(),
        {
            "target": {
                "pdfs": ["model"],
                "functions": ["mean"],
                "data": ["observed"],
            }
        },
    )


def test_structure_check_reports_missing_object_category() -> None:
    adapter = backend.PyHS3Backend()
    with pytest.raises(
        backend.BackendFailure, match=r"missing pdfs: \['absent'\]"
    ) as exc_info:
        adapter.run_structure_check(
            _workspace_stub(),
            {"target": {"pdfs": ["absent"], "functions": [], "data": []}},
        )
    assert exc_info.value.stage == "structure_check"
    assert str(exc_info.value).startswith(
        "pyhs3_failure_stage=structure_check: missing pdfs"
    )


def test_workspace_import_failure_has_stable_stage_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(*_args: object, **_kwargs: object) -> None:
        msg = "invalid workspace"
        raise ValueError(msg)

    monkeypatch.setattr(backend.Workspace, "load", fail_load)

    with pytest.raises(backend.BackendFailure, match="invalid workspace") as exc_info:
        backend.PyHS3Backend().load_workspace(Path("invalid.json"))

    assert exc_info.value.stage == "workspace_import"
    assert str(exc_info.value).startswith(
        "pyhs3_failure_stage=workspace_import: invalid workspace"
    )
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_evaluation_failure_has_stable_stage_marker() -> None:
    with pytest.raises(
        backend.BackendFailure, match="distribution 'absent' not found"
    ) as exc_info:
        backend.PyHS3Backend().run_twice_delta_nll_scan(
            _workspace_stub(),
            {"target": {"pdf": "absent", "data": "observed"}},
        )

    assert exc_info.value.stage == "evaluation"
    assert str(exc_info.value).startswith(
        "pyhs3_failure_stage=evaluation: distribution 'absent' not found"
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([1.0, 2.0], "expected exactly one scalar"),
        (np.nan, "non-finite value"),
        ("not numeric", "did not return a numeric value"),
    ],
)
def test_as_scalar_rejects_invalid_evaluator_output(
    value: object, message: str
) -> None:
    with pytest.raises(AssertionError, match=message):
        backend._as_scalar(value)


def test_base_values_apply_reference_data_and_literal_precedence() -> None:
    model = cast(
        Model,
        SimpleNamespace(
            free_params={"mu": 1.0, "x": 999.0},
            data={"x": np.asarray([0.5, 1.5])},
        ),
    )
    inputs = [
        SimpleNamespace(name="mu"),
        SimpleNamespace(name="x"),
        SimpleNamespace(name="3.5"),
    ]

    values = backend.PyHS3Backend._base_values(
        model,
        {"reference_point": {"mu": 2.0, "x": 0.0}},
        inputs,
    )

    assert values["mu"] == 2.0
    np.testing.assert_array_equal(values["x"], [0.5, 1.5])
    assert values["3.5"] == 3.5
