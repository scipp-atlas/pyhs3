"""Unit tests for the pyhs3 command-line interface.

These exercise the real ``Workspace`` / ``Model`` / log-prob path (no mocks):
the CLI is a thin wrapper and its behaviour is only meaningful against the
genuine loading, validation, and NLL computation.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

import pytest
import typer
from scipy.stats import truncnorm
from typer.testing import CliRunner

from pyhs3.cli import app
from pyhs3.cli._shared import _stdin_is_tty, read_spec, stdout_is_interactive
from pyhs3.cli.infer import _select_target
from pyhs3.workspace import Workspace

runner = CliRunner()


# ---------------------------------------------------------------------------
# Minimal two-Gaussian-channel workspace (mirrors tests/test_model_likelihood.py)
# ---------------------------------------------------------------------------

_WS_DICT: dict = {
    "metadata": {"hs3_version": "0.2"},
    "distributions": [
        {
            "name": "gauss1",
            "type": "gaussian_dist",
            "x": "x_obs",
            "mean": "mean",
            "sigma": 1.0,
        },
        {
            "name": "gauss2",
            "type": "gaussian_dist",
            "x": "y_obs",
            "mean": "mean",
            "sigma": 2.0,
        },
    ],
    "domains": [
        {
            "name": "main",
            "type": "product_domain",
            "axes": [{"name": "mean", "min": -10.0, "max": 10.0}],
        }
    ],
    "data": [
        {
            "name": "data1",
            "type": "unbinned",
            "axes": [{"name": "x_obs", "min": -10.0, "max": 10.0}],
            "entries": [[1.0], [2.0], [3.0], [4.0], [5.0]],
        },
        {
            "name": "data2",
            "type": "unbinned",
            "axes": [{"name": "y_obs", "min": -10.0, "max": 10.0}],
            "entries": [[0.5], [1.5], [2.5], [3.5], [4.5]],
        },
    ],
    "likelihoods": [
        {"name": "L", "distributions": ["gauss1", "gauss2"], "data": ["data1", "data2"]}
    ],
    "analyses": [
        {"name": "A", "likelihood": "L", "domains": ["main"], "init": "params"}
    ],
    "parameter_points": [
        {"name": "params", "parameters": [{"name": "mean", "value": 2.0}]}
    ],
}


def _truncnorm_logpdf(
    x: float, loc: float, scale: float, low: float, high: float
) -> float:
    a = (low - loc) / scale
    b = (high - loc) / scale
    return float(truncnorm.logpdf(x, a, b, loc=loc, scale=scale))


def _expected_nll(mean: float) -> float:
    """NLL = -2 * joint log-prob for the two-Gaussian workspace at ``mean``."""
    events1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    events2 = [0.5, 1.5, 2.5, 3.5, 4.5]
    lp = sum(_truncnorm_logpdf(x, mean, 1.0, -10.0, 10.0) for x in events1) + sum(
        _truncnorm_logpdf(y, mean, 2.0, -10.0, 10.0) for y in events2
    )
    return -2.0 * lp


@pytest.fixture
def good_workspace(tmp_path):
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps(_WS_DICT), encoding="utf-8")
    return path


@pytest.fixture
def bad_workspace(tmp_path):
    """A workspace whose likelihood references a non-existent distribution."""
    spec = json.loads(json.dumps(_WS_DICT))
    spec["likelihoods"][0]["distributions"] = ["does_not_exist", "gauss2"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_good_workspace(good_workspace):
    result = runner.invoke(app, ["validate", str(good_workspace)])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output.lower()


def test_validate_bad_workspace(bad_workspace):
    result = runner.invoke(app, ["validate", str(bad_workspace)])
    assert result.exit_code != 0
    # The unresolved-reference error must be surfaced to the user.
    assert "does_not_exist" in result.output


def test_validate_from_stdin_dash(good_workspace):
    result = runner.invoke(app, ["validate", "-"], input=good_workspace.read_text())
    assert result.exit_code == 0, result.output
    assert "valid" in result.output.lower()


def test_validate_from_stdin_default(good_workspace):
    """Omitting the path reads from stdin when stdin is not a TTY."""
    result = runner.invoke(app, ["validate"], input=good_workspace.read_text())
    assert result.exit_code == 0, result.output
    assert "valid" in result.output.lower()


def test_validate_bad_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code != 0


def test_validate_schema_error(tmp_path):
    """A distribution missing required fields fails pydantic schema validation
    (distinct from the FK-resolution WorkspaceValidationError path)."""
    spec = {
        "metadata": {"hs3_version": "0.2"},
        "distributions": [{"name": "g", "type": "gaussian_dist", "x": "x"}],
    }
    path = tmp_path / "schema_bad.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1
    assert "validation failed" in result.output.lower()


@pytest.mark.parametrize("root", ["[]", "42", '"just a string"'])
def test_validate_non_object_json_root(tmp_path, root):
    """A JSON root that isn't an object fails cleanly, not with a raw TypeError."""
    path = tmp_path / "root.json"
    path.write_text(root, encoding="utf-8")
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1
    assert "object" in result.output.lower()


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def test_inspect_json_when_piped(good_workspace):
    """CliRunner stdout is not a TTY, so inspect emits JSON by default."""
    result = runner.invoke(app, ["inspect", str(good_workspace)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    dist_names = {d["name"] for d in payload["distributions"]}
    assert dist_names == {"gauss1", "gauss2"}
    assert {d["type"] for d in payload["distributions"]} == {"gaussian_dist"}
    data_by_name = {d["name"]: d for d in payload["data"]}
    assert data_by_name["data1"]["entries"] == 5
    assert [lk["name"] for lk in payload["likelihoods"]] == ["L"]
    assert payload["analyses"][0]["likelihood"] == "L"


def test_inspect_json_forced_flag(good_workspace):
    result = runner.invoke(app, ["inspect", "--json", str(good_workspace)])
    assert result.exit_code == 0, result.output
    json.loads(result.output)  # parses cleanly


def test_inspect_table_when_forced(good_workspace):
    result = runner.invoke(app, ["inspect", "--no-json", str(good_workspace)])
    assert result.exit_code == 0, result.output
    # Human table names the distributions but is not machine JSON.
    assert "gauss1" in result.output
    assert "gauss2" in result.output
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.output)


def test_inspect_from_stdin(good_workspace):
    result = runner.invoke(app, ["inspect", "-"], input=good_workspace.read_text())
    assert result.exit_code == 0, result.output
    json.loads(result.output)


# ---------------------------------------------------------------------------
# nll
# ---------------------------------------------------------------------------


def test_nll_default_params(good_workspace):
    """No overrides -> uses the workspace init parameter set (mean=2.0)."""
    result = runner.invoke(app, ["nll", str(good_workspace)])
    assert result.exit_code == 0, result.output
    value = float(result.output.strip().splitlines()[-1])
    assert value == pytest.approx(_expected_nll(2.0), rel=1e-6)


def test_nll_param_override(good_workspace):
    result = runner.invoke(app, ["nll", str(good_workspace), "--param", "mean=1.5"])
    assert result.exit_code == 0, result.output
    value = float(result.output.strip().splitlines()[-1])
    assert value == pytest.approx(_expected_nll(1.5), rel=1e-6)


def test_nll_params_file(good_workspace, tmp_path):
    params_file = tmp_path / "params.json"
    params_file.write_text(json.dumps({"mean": 3.0}), encoding="utf-8")
    result = runner.invoke(
        app,
        ["nll", str(good_workspace), "--params-file", str(params_file)],
    )
    assert result.exit_code == 0, result.output
    value = float(result.output.strip().splitlines()[-1])
    assert value == pytest.approx(_expected_nll(3.0), rel=1e-6)


def test_nll_analysis_selection(good_workspace):
    result = runner.invoke(
        app, ["nll", str(good_workspace), "--analysis", "A", "--param", "mean=2.0"]
    )
    assert result.exit_code == 0, result.output
    value = float(result.output.strip().splitlines()[-1])
    assert value == pytest.approx(_expected_nll(2.0), rel=1e-6)


def test_nll_bad_param_format(good_workspace):
    result = runner.invoke(app, ["nll", str(good_workspace), "--param", "mean"])
    assert result.exit_code != 0


def test_nll_no_init_uses_parameter_points(tmp_path):
    """An analysis without an `init` set falls back to workspace parameter_points."""
    spec = json.loads(json.dumps(_WS_DICT))
    del spec["analyses"][0]["init"]  # analysis no longer names a parameter set
    path = tmp_path / "no_init.json"
    path.write_text(json.dumps(spec), encoding="utf-8")

    result = runner.invoke(app, ["nll", str(path)])
    assert result.exit_code == 0, result.output
    value = float(result.output.strip().splitlines()[-1])
    # parameter_points declares mean=2.0, so the NLL matches the mean=2.0 case.
    assert value == pytest.approx(_expected_nll(2.0), rel=1e-6)


def test_nll_from_stdin(good_workspace):
    result = runner.invoke(
        app, ["nll", "-", "--param", "mean=2.0"], input=good_workspace.read_text()
    )
    assert result.exit_code == 0, result.output
    value = float(result.output.strip().splitlines()[-1])
    assert value == pytest.approx(_expected_nll(2.0), rel=1e-6)


# ---------------------------------------------------------------------------
# _shared helpers (unit-level: real fds/streams, no CliRunner substitution)
# ---------------------------------------------------------------------------


def test_stdout_is_interactive_regular_file(tmp_path):
    path = tmp_path / "out.txt"
    with path.open("w") as f, pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "stdout", f)
        assert stdout_is_interactive() is False


def test_stdout_is_interactive_character_device():
    """Neither a FIFO nor a regular file falls through to isatty().

    Whether the null device itself reports as a tty is a platform CRT
    detail (Windows' isatty() returns True for NUL, unlike POSIX), so this
    checks the delegation itself rather than hardcoding a truth value.
    """
    with (
        Path(os.devnull).open("w", encoding="utf-8") as f,
        pytest.MonkeyPatch.context() as mp,
    ):
        mp.setattr(sys, "stdout", f)
        assert stdout_is_interactive() == f.isatty()


def test_stdin_is_tty_handles_broken_isatty():
    class BrokenStdin:
        def isatty(self) -> bool:
            msg = "no tty"
            raise OSError(msg)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "stdin", BrokenStdin())
        assert _stdin_is_tty() is False


def test_read_spec_stdin_tty_raises():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("pyhs3.cli._shared._stdin_is_tty", lambda: True)
        with pytest.raises(typer.BadParameter, match="stdin is a terminal"):
            read_spec(None)


# ---------------------------------------------------------------------------
# python -m pyhs3
# ---------------------------------------------------------------------------


def test_main_module_entrypoint(good_workspace):
    """``python -m pyhs3`` wires up the same Typer app as the console script."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["pyhs3", "validate", str(good_workspace)])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("pyhs3.__main__", run_name="__main__")
    assert exc_info.value.code == 0


def test_main_module_import_does_not_invoke_app():
    """A plain import (__name__ != "__main__") must not call app()."""
    runpy.run_module("pyhs3.__main__", run_name="pyhs3.__main__")


def test_nll_param_override_rejects_observable_name(good_workspace):
    """--param naming an observable must be ignored, never clobber workspace data."""
    result = runner.invoke(app, ["nll", str(good_workspace), "--param", "x_obs=999"])
    assert result.exception is None, result.output
    assert result.exit_code == 0, result.output
    assert "x_obs" in result.output
    value = float(result.output.strip().splitlines()[-1])
    assert value == pytest.approx(_expected_nll(2.0), rel=1e-6)


def test_nll_param_override_unknown_name_ignored(good_workspace):
    result = runner.invoke(
        app, ["nll", str(good_workspace), "--param", "totally_unknown=1.0"]
    )
    assert result.exit_code == 0, result.output
    assert "totally_unknown" in result.output
    value = float(result.output.strip().splitlines()[-1])
    assert value == pytest.approx(_expected_nll(2.0), rel=1e-6)


def test_nll_param_value_not_a_number(good_workspace):
    result = runner.invoke(
        app, ["nll", str(good_workspace), "--param", "mean=notanumber"]
    )
    assert result.exit_code != 0


def test_nll_params_file_not_object(good_workspace, tmp_path):
    params_file = tmp_path / "bad_params.json"
    params_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    result = runner.invoke(
        app, ["nll", str(good_workspace), "--params-file", str(params_file)]
    )
    assert result.exit_code != 0


def test_nll_analysis_selects_likelihood_by_name(good_workspace):
    """--analysis can also name a likelihood directly, bypassing analyses."""
    result = runner.invoke(app, ["nll", str(good_workspace), "--analysis", "L"])
    assert result.exit_code == 0, result.output
    value = float(result.output.strip().splitlines()[-1])
    assert value == pytest.approx(_expected_nll(2.0), rel=1e-6)


def test_nll_analysis_not_found(good_workspace):
    result = runner.invoke(
        app, ["nll", str(good_workspace), "--analysis", "nonexistent"]
    )
    assert result.exit_code != 0
    assert "nonexistent" in result.output


def test_nll_multiple_analyses_requires_selection(tmp_path):
    spec = json.loads(json.dumps(_WS_DICT))
    spec["analyses"].append({**spec["analyses"][0], "name": "B"})
    path = tmp_path / "two_analyses.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    result = runner.invoke(app, ["nll", str(path)])
    assert result.exit_code != 0
    assert "multiple analyses" in result.output


def test_nll_no_analyses_uses_sole_likelihood(tmp_path):
    """With no analyses declared at all, the sole likelihood is used directly."""
    spec = json.loads(json.dumps(_WS_DICT))
    del spec["analyses"]
    path = tmp_path / "no_analyses.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    result = runner.invoke(app, ["nll", str(path)])
    assert result.exit_code == 0, result.output
    value = float(result.output.strip().splitlines()[-1])
    assert value == pytest.approx(_expected_nll(2.0), rel=1e-6)


def test_nll_multiple_likelihoods_without_analyses_requires_selection(tmp_path):
    spec = json.loads(json.dumps(_WS_DICT))
    del spec["analyses"]
    spec["likelihoods"] = [
        {"name": "L1", "distributions": ["gauss1"], "data": ["data1"]},
        {"name": "L2", "distributions": ["gauss2"], "data": ["data2"]},
    ]
    path = tmp_path / "two_likelihoods.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    result = runner.invoke(app, ["nll", str(path)])
    assert result.exit_code != 0
    assert "multiple likelihoods" in result.output


def test_nll_no_targets_available(tmp_path):
    spec = json.loads(json.dumps(_WS_DICT))
    del spec["analyses"]
    del spec["likelihoods"]
    path = tmp_path / "no_targets.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    result = runner.invoke(app, ["nll", str(path)])
    assert result.exit_code != 0
    assert "no analyses or likelihoods" in result.output


def test_select_target_no_analyses_or_likelihoods():
    """Direct unit test: a workspace with no distributions/likelihoods/analyses
    at all is still a legal (if useless) Workspace; _select_target must still
    report a clean error rather than crash."""
    ws = Workspace(metadata={"hs3_version": "0.2"})
    with pytest.raises(typer.BadParameter, match="no analyses or likelihoods"):
        _select_target(ws, None)


def test_nll_missing_free_parameter_value(tmp_path):
    """A free parameter absent from parameter_points, free_params, and
    --param overrides alike has no value available anywhere."""
    spec = json.loads(json.dumps(_WS_DICT))
    spec["parameter_points"][0]["parameters"] = []
    path = tmp_path / "no_mean_value.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    result = runner.invoke(app, ["nll", str(path)])
    assert result.exit_code != 0
    assert "mean" in result.output
