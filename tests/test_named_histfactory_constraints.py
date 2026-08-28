"""Current-ROOT named HistFactory constraint references."""

from __future__ import annotations

import math

import numpy as np
import pytensor
import pytest

from pyhs3 import Workspace
from pyhs3.exceptions import WorkspaceValidationError


def _base_spec(
    *,
    constraint: str | None = "mu_constraint",
    target: dict | None = None,
    extra_distributions: list[dict] | None = None,
) -> dict:
    modifier = {
        "name": "mu",
        "type": "normfactor",
        "parameter": "mu",
    }
    if constraint is not None:
        modifier["constraint"] = constraint

    distributions = [
        {
            "name": "channel",
            "type": "histfactory_dist",
            "axes": [{"name": "x", "min": 0.0, "max": 1.0, "nbins": 1}],
            "samples": [
                {
                    "name": "sample",
                    "data": {"contents": [10.0], "errors": [0.0]},
                    "modifiers": [modifier],
                }
            ],
        }
    ]
    if target is not None:
        distributions.append(target)
    distributions.extend(extra_distributions or [])

    return {
        "metadata": {"hs3_version": "0.2"},
        "distributions": distributions,
        "data": [
            {
                "name": "observed",
                "type": "binned",
                "axes": [{"name": "x", "min": 0.0, "max": 1.0, "nbins": 1}],
                "contents": [10.0],
            }
        ],
        "domains": [
            {
                "name": "default_domain",
                "type": "product_domain",
                "axes": [{"name": "mu", "min": 0.0, "max": 3.0}],
            }
        ],
        "parameter_points": [
            {
                "name": "nominal",
                "parameters": [{"name": "mu", "value": 1.0}],
            }
        ],
        "likelihoods": [
            {
                "name": "likelihood",
                "distributions": ["channel"],
                "data": ["observed"],
            }
        ],
    }


def _gaussian_target(
    *,
    name: str = "mu_constraint",
    x: str = "mu",
    mean: str | float = 1.0,
    sigma: float = 0.2,
) -> dict:
    return {
        "name": name,
        "type": "gaussian_dist",
        "x": x,
        "mean": mean,
        "sigma": sigma,
    }


def _evaluate_log_prob(workspace: Workspace) -> float:
    likelihood = workspace.likelihoods["likelihood"]
    model = workspace.model(likelihood, progress=False)
    expression = model.log_prob
    inputs = {
        var.name: var
        for var in pytensor.graph.traversal.explicit_graph_inputs([expression])
        if var.name is not None
    }
    function = pytensor.function(list(inputs.values()), expression)
    values = {**model.data, **model.nominal_params}
    return float(np.asarray(function(**{name: values[name] for name in inputs})).item())


def test_named_constraint_roundtrips_and_uses_exact_target_logpdf() -> None:
    """A non-unit Gaussian is evaluated as serialized, not synthesized."""
    workspace = Workspace.model_validate(_base_spec(target=_gaussian_target()))

    modifier = workspace.distributions["channel"].samples[0].modifiers[0]
    assert modifier.constraint == "mu_constraint"
    assert "mu_constraint" in modifier.dependencies
    dumped = workspace.model_dump(mode="json", exclude_none=True)
    dumped_modifier = dumped["distributions"][0]["samples"][0]["modifiers"][0]
    assert dumped_modifier["constraint"] == "mu_constraint"
    assert dumped["metadata"]["hs3_version"] == "0.2"

    value = _evaluate_log_prob(workspace)
    poisson = 10.0 * math.log(10.0) - 10.0 - math.lgamma(11.0)
    gaussian = -math.log(0.2 * math.sqrt(2.0 * math.pi))
    assert value == pytest.approx(poisson + gaussian, rel=1e-12)


def _dedup_spec(*, repeat_constraint_in_product_and_aux: bool) -> dict:
    spec = _base_spec(target=_gaussian_target())
    channel = spec["distributions"][0]
    channel["samples"].append(
        {
            "name": "second_sample",
            "data": {"contents": [0.0], "errors": [0.0]},
            "modifiers": [
                {
                    "name": "mu_again",
                    "type": "normfactor",
                    "parameter": "mu",
                    "constraint": "mu_constraint",
                }
            ],
        }
    )
    spec["distributions"].extend(
        [
            {
                "name": "shape",
                "type": "gaussian_dist",
                "x": "y",
                "mean": 0.0,
                "sigma": 1.0,
            },
            {
                "name": "product",
                "type": "product_dist",
                "factors": (
                    ["shape", "mu_constraint"]
                    if repeat_constraint_in_product_and_aux
                    else ["shape"]
                ),
            },
        ]
    )
    spec["data"].append(
        {
            "name": "events",
            "type": "unbinned",
            "axes": [{"name": "y", "min": -5.0, "max": 5.0}],
            "entries": [[0.0]],
        }
    )
    likelihood = spec["likelihoods"][0]
    likelihood["distributions"].append("product")
    likelihood["data"].append("events")
    if repeat_constraint_in_product_and_aux:
        likelihood["aux_distributions"] = ["mu_constraint"]
    return spec


def test_named_constraint_is_deduplicated_across_all_likelihood_sources() -> None:
    """HF references, ProductDist factors, and aux roots share one name registry."""
    baseline = Workspace.model_validate(
        _dedup_spec(repeat_constraint_in_product_and_aux=False)
    )
    repeated = Workspace.model_validate(
        _dedup_spec(repeat_constraint_in_product_and_aux=True)
    )
    assert _evaluate_log_prob(repeated) == pytest.approx(
        _evaluate_log_prob(baseline), rel=1e-12
    )

    model = repeated.model(repeated.likelihoods["likelihood"], progress=False)
    assert list(model._hfdc_named_log_constraints) == [  # pylint: disable=protected-access
        "mu_constraint"
    ]


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (
            _base_spec(target=None),
            "unknown constraint distribution 'mu_constraint'",
        ),
        (
            _base_spec(
                target=_gaussian_target(x="other_parameter"),
            ),
            "does not depend on modifier parameter 'mu'",
        ),
        (
            _base_spec(
                target=_gaussian_target(x="x", mean="mu"),
            ),
            "depends on event observable",
        ),
        (
            _base_spec(
                constraint="other_channel",
                target={
                    "name": "other_channel",
                    "type": "histfactory_dist",
                    "axes": [{"name": "z", "min": 0.0, "max": 1.0, "nbins": 1}],
                    "samples": [],
                },
            ),
            "must not reference a histfactory_dist",
        ),
        (
            _base_spec(
                constraint="cycle",
                target={
                    "name": "cycle",
                    "type": "product_dist",
                    "factors": ["channel"],
                },
            ),
            "creates a circular dependency",
        ),
    ],
)
def test_invalid_named_constraints_fail_workspace_validation(
    spec: dict, message: str
) -> None:
    with pytest.raises(WorkspaceValidationError, match=message):
        Workspace.model_validate(spec)


def test_absent_constraint_remains_unconstrained() -> None:
    workspace = Workspace.model_validate(_base_spec(constraint=None, target=None))
    modifier = workspace.distributions["channel"].samples[0].modifiers[0]
    assert modifier.constraint is None
    assert modifier.dependencies == {"mu"}
    dumped = workspace.model_dump(mode="json", exclude_none=True)
    assert "constraint" not in dumped["distributions"][0]["samples"][0]["modifiers"][0]
