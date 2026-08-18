"""pyhs3-owned backend adapter for the external HS3TestSuite runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

import numpy as np
import numpy.typing as npt
import pytensor
from pytensor.graph.traversal import explicit_graph_inputs

from pyhs3 import Workspace
from pyhs3.likelihoods import Likelihood
from pyhs3.model import Model

type Check = Mapping[str, Any]
type GraphInput = Any
type Evaluator = Callable[..., Any]
type FailureStage = Literal["workspace_import", "structure_check", "evaluation"]

FAILURE_STAGE_MARKER = "pyhs3_failure_stage="


class BackendFailure(AssertionError):
    """Failure carrying a stable stage through the upstream string-only runner."""

    def __init__(self, stage: FailureStage, detail: str) -> None:
        self.stage = stage
        self.detail = detail
        super().__init__(f"{FAILURE_STAGE_MARKER}{stage}: {detail}")


class PyHS3Backend:
    """Adapt pyhs3 to the backend protocol used by HS3TestSuite."""

    name = "pyhs3"
    mode = "FAST_COMPILE"

    @staticmethod
    def load_workspace(path: Path) -> Workspace:
        """Load an HS3 document and preserve its full validation traceback."""
        try:
            return Workspace.load(path, suppress_traceback=False)
        except BackendFailure:
            raise
        except Exception as exc:
            _raise_backend_failure("workspace_import", exc)

    @staticmethod
    def structure(workspace: Workspace) -> dict[str, list[str]]:
        """Return the object categories understood by structure-import checks."""
        return {
            "pdfs": _names(workspace.distributions),
            "functions": _names(workspace.functions),
            "data": _names(workspace.data),
        }

    def run_structure_check(self, workspace: Workspace, check: Check) -> None:
        """Require the requested objects while allowing additional objects."""
        try:
            actual = self.structure(workspace)
            target = _require_mapping(check.get("target", {}), "structure target")

            for key in ("pdfs", "functions", "data"):
                required = set(_string_sequence(target.get(key, ()), f"target {key}"))
                missing = required.difference(actual[key])
                if missing:
                    msg = f"missing {key}: {sorted(missing)}"
                    raise AssertionError(msg)
        except BackendFailure:
            raise
        except Exception as exc:
            _raise_backend_failure("structure_check", exc)

    def run_twice_delta_nll_scan(
        self, workspace: Workspace, check: Check
    ) -> list[float]:
        """Evaluate the suite's twice-delta-NLL scan for one PDF/data pair."""
        try:
            return self._run_twice_delta_nll_scan(workspace, check)
        except BackendFailure:
            raise
        except Exception as exc:
            _raise_backend_failure("evaluation", exc)

    def _run_twice_delta_nll_scan(
        self, workspace: Workspace, check: Check
    ) -> list[float]:
        target = _require_mapping(check.get("target"), "scan target")
        pdf_name = _required_string(target, "pdf", "scan target")
        data_name = _required_string(target, "data", "scan target")

        distribution = _get_named(workspace.distributions, pdf_name, "distribution")
        data = _get_named(workspace.data, data_name, "data")
        likelihood = Likelihood(
            name=f"hs3suite_{pdf_name}_{data_name}",
            distributions=[distribution],
            data=[data],
        )
        model = workspace.model(likelihood, progress=False, mode=self.mode)

        twice_nll = -2.0 * model.log_prob
        inputs = _named_graph_inputs(twice_nll)
        evaluator: Evaluator = pytensor.function(
            inputs=inputs,
            outputs=twice_nll,
            mode=self.mode,
            on_unused_input="ignore",
        )

        reference_point = _require_mapping(
            check.get("reference_point"), "reference point"
        )
        scan_parameter = _required_string(check, "scan_parameter", "scan check")
        if scan_parameter not in reference_point:
            msg = f"reference point does not define scan parameter {scan_parameter!r}"
            raise AssertionError(msg)

        base_values = self._base_values(model, check, inputs)
        reference_values = dict(base_values)
        reference_values[scan_parameter] = reference_point[scan_parameter]
        reference = _evaluate_scalar(
            evaluator,
            inputs,
            reference_values,
            context="reference point",
        )

        scan_points = _number_sequence(check.get("scan_points"), "scan points")
        results: list[float] = []
        for index, point in enumerate(scan_points):
            point_values = dict(base_values)
            point_values[scan_parameter] = point
            value = _evaluate_scalar(
                evaluator,
                inputs,
                point_values,
                context=f"scan point {index} ({scan_parameter}={point:.17g})",
            )
            delta = value - reference
            if not np.isfinite(delta):
                msg = (
                    f"scan point {index} ({scan_parameter}={point:.17g}) produced "
                    f"non-finite twice-delta-NLL {delta!r}"
                )
                raise AssertionError(msg)
            results.append(delta)

        return results

    @staticmethod
    def _base_values(
        model: Model, check: Check, inputs: Sequence[GraphInput]
    ) -> dict[str, Any]:
        """Assemble defaults, reference overrides, data, and literal inputs."""
        reference_point = _require_mapping(
            check.get("reference_point"), "reference point"
        )

        # ``free_params`` contains the model's default values for every symbolic
        # parameter input. The suite reference point overrides those defaults,
        # and the selected dataset overrides observable placeholders such as x=0.
        values: dict[str, Any] = dict(model.free_params)
        values.update(reference_point)
        values.update(model.data)

        for variable in inputs:
            name = _graph_input_name(variable)
            if name not in values:
                literal = _numeric_literal(name)
                if literal is not None:
                    values[name] = literal

        missing = [
            _graph_input_name(variable)
            for variable in inputs
            if _graph_input_name(variable) not in values
        ]
        if missing:
            msg = f"missing pyhs3 graph inputs: {missing}"
            raise AssertionError(msg)
        return values


def _raise_backend_failure(stage: FailureStage, exc: Exception) -> NoReturn:
    detail = str(exc) or type(exc).__name__
    raise BackendFailure(stage, detail) from exc


def _names(collection: Any | None) -> list[str]:
    if collection is None:
        return []
    return sorted(item.name for item in collection)


def _get_named(collection: Any | None, name: str, label: str) -> Any:
    if collection is None:
        msg = f"{label} {name!r} not found"
        raise AssertionError(msg)
    item = collection.get(name)
    if item is None:
        msg = f"{label} {name!r} not found"
        raise AssertionError(msg)
    return item


def _named_graph_inputs(expression: Any) -> list[GraphInput]:
    inputs: list[GraphInput] = []
    names: set[str] = set()
    for variable in explicit_graph_inputs([expression]):
        name = variable.name
        if not isinstance(name, str):
            msg = f"pyhs3 graph contains an unnamed free input of type {variable.type}"
            raise AssertionError(msg)
        if name in names:
            msg = f"pyhs3 graph contains duplicate input name {name!r}"
            raise AssertionError(msg)
        names.add(name)
        inputs.append(variable)
    return inputs


def _graph_input_name(variable: GraphInput) -> str:
    name = variable.name
    if not isinstance(name, str):
        msg = "pyhs3 graph input is unnamed"
        raise AssertionError(msg)
    return name


def _numeric_literal(name: str | None) -> float | None:
    if name is None:
        return None
    try:
        return float(name)
    except ValueError:
        return None


def _ordered_values(
    inputs: Sequence[GraphInput], values: Mapping[str, Any]
) -> list[npt.NDArray[np.float64]]:
    ordered: list[npt.NDArray[np.float64]] = []
    for variable in inputs:
        name = _graph_input_name(variable)
        if name not in values:
            msg = f"missing pyhs3 graph input {name!r}"
            raise AssertionError(msg)
        try:
            value = np.asarray(values[name], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            msg = f"pyhs3 graph input {name!r} cannot be converted to float64"
            raise AssertionError(msg) from exc
        ordered.append(value)
    return ordered


def _evaluate_scalar(
    evaluator: Evaluator,
    inputs: Sequence[GraphInput],
    values: Mapping[str, Any],
    *,
    context: str,
) -> float:
    try:
        output = evaluator(*_ordered_values(inputs, values))
    except Exception as exc:
        msg = f"pyhs3 failed to evaluate {context}: {exc}"
        raise AssertionError(msg) from exc
    return _as_scalar(output, context=context)


def _as_scalar(value: Any, *, context: str = "NLL evaluation") -> float:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"pyhs3 {context} did not return a numeric value"
        raise AssertionError(msg) from exc
    if array.size != 1:
        msg = (
            f"pyhs3 {context} returned shape {array.shape}; "
            "expected exactly one scalar value"
        )
        raise AssertionError(msg)
    result = float(array.item())
    if not np.isfinite(result):
        msg = f"pyhs3 {context} returned non-finite value {result!r}"
        raise AssertionError(msg)
    return result


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"{label} must be a mapping"
        raise AssertionError(msg)
    return cast(Mapping[str, Any], value)


def _required_string(values: Check, key: str, label: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{label} must define a non-empty string {key!r}"
        raise AssertionError(msg)
    return value


def _string_sequence(value: object, label: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{label} must be a sequence of strings"
        raise AssertionError(msg)
    if not all(isinstance(item, str) for item in value):
        msg = f"{label} must contain only strings"
        raise AssertionError(msg)
    return list(cast(Sequence[str], value))


def _number_sequence(value: object, label: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{label} must be a sequence of numbers"
        raise AssertionError(msg)
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        msg = f"{label} must contain only numbers"
        raise AssertionError(msg) from exc
