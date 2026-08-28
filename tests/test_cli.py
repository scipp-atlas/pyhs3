"""Unit tests for the pyhs3 command-line interface.

These exercise the real ``Workspace`` / ``Model`` / log-prob path (no mocks):
the CLI is a thin wrapper and its behaviour is only meaningful against the
genuine loading, validation, and NLL computation.
"""

from __future__ import annotations

import json

import pytest
from scipy.stats import truncnorm
from typer.testing import CliRunner

from pyhs3.cli import app

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
