from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

CONFIG_DIR = Path(__file__).with_name("hs3testsuite")
TESTS_DIR = Path(__file__).parent


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


PIN = _load_json(CONFIG_DIR / "pin.json")
KNOWN_FAILURE_CONFIG = _load_json(CONFIG_DIR / "known_failures.json")
KNOWN_FAILURES: dict[str, dict[str, str]] = KNOWN_FAILURE_CONFIG["failures"]
AUDIT_CANDIDATE = os.environ.get("PYHS3_HS3TESTSUITE_AUDIT_CANDIDATE") == "1"
EXPECTED_STAGES_BY_CATEGORY = {
    "evaluation_error": "evaluation",
    "numerical_mismatch": "numerical_comparison",
    "unsupported_distribution": "workspace_import",
    "unsupported_function": "workspace_import",
}

suite_root_value = os.environ.get("HS3TESTSUITE_ROOT")
if suite_root_value is None:
    pytest.skip(
        "set HS3TESTSUITE_ROOT to run the external HS3 conformance suite",
        allow_module_level=True,
    )

SUITE_ROOT = Path(suite_root_value).resolve()
if not (SUITE_ROOT / "manifest.json").is_file():
    msg = f"HS3TESTSUITE_ROOT does not contain manifest.json: {SUITE_ROOT}"
    raise RuntimeError(msg)


def _suite_identity() -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "-C", str(SUITE_ROOT), "rev-parse", "HEAD^{commit}"],
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(SUITE_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest_bytes = (SUITE_ROOT / "manifest.json").read_bytes()
    return {
        "repository": PIN["repository"],
        "commit": completed.stdout.strip(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "checkout_clean": not bool(status.stdout.strip()),
    }


def _suite_matches_pin(actual: Mapping[str, Any]) -> bool:
    return (
        actual["commit"] == PIN["commit"]
        and actual["manifest_sha256"] == PIN["manifest_sha256"]
    )


ACTUAL_SUITE = _suite_identity()
if not AUDIT_CANDIDATE:
    setup_errors = []
    if not _suite_matches_pin(ACTUAL_SUITE):
        setup_errors.append(
            "checkout identity does not match tests/hs3testsuite/pin.json"
        )
    if not ACTUAL_SUITE["checkout_clean"]:
        setup_errors.append("checkout has modified or untracked files")
    if setup_errors:
        msg = "Invalid pinned HS3TestSuite checkout: " + "; ".join(setup_errors)
        raise RuntimeError(msg)

sys.path.insert(0, str(SUITE_ROOT))
sys.path.insert(0, str(TESTS_DIR))

from hs3suite.backends import _BACKENDS  # noqa: E402
from hs3suite.runner import CheckResult, run_suite  # noqa: E402

_BACKENDS["pyhs3"] = ("hs3testsuite_pyhs3_backend", "PyHS3Backend")

MANIFEST = _load_json(SUITE_ROOT / "manifest.json")
CASES = [
    (fixture["test_id"], check_id)
    for fixture in MANIFEST["fixtures"]
    for check_id in fixture["checks"]
]
CASE_KEYS = {f"{test_id}::{check_id}" for test_id, check_id in CASES}


def _case_parameter(test_id: str, check_id: str) -> Any:
    key = f"{test_id}::{check_id}"
    failure = KNOWN_FAILURES.get(key)
    if failure is None:
        return pytest.param(test_id, check_id, id=key)
    reason = f"{failure['category']}: {failure['reason']}"
    return pytest.param(
        test_id,
        check_id,
        marks=pytest.mark.xfail(reason=reason, strict=True),
        id=key,
    )


CASE_PARAMETERS = [_case_parameter(*case) for case in CASES]


def _result_key(result: CheckResult) -> str:
    return f"{result.test_id}::{result.check_id}"


def _expected_failure_stage(failure: Mapping[str, str]) -> str:
    return EXPECTED_STAGES_BY_CATEGORY.get(failure["category"], "unconfigured")


def _failure_stage(result: CheckResult) -> str | None:
    if result.status != "failed":
        return None

    marker = "pyhs3_failure_stage="
    if result.message.startswith(marker):
        stage, separator, _ = result.message[len(marker) :].partition(":")
        if separator and stage:
            return stage
    if result.check_id == "twice_delta_nll_scan" and result.message.startswith(
        "point "
    ):
        return "numerical_comparison"
    return "unclassified"


def _classify(result: CheckResult) -> str:
    failure = KNOWN_FAILURES.get(_result_key(result))
    if result.status == "passed":
        return "unexpected_pass" if failure is not None else "passed"
    if result.status == "failed":
        if failure is None:
            return "unexpected_failure"
        if _failure_stage(result) != _expected_failure_stage(failure):
            return "known_failure_stage_changed"
        return "known_failure"
    return "unexpected_status"


def _result_diagnostics(results: list[CheckResult]) -> dict[str, Any]:
    keys = [_result_key(result) for result in results]
    result_keys = set(keys)
    return {
        "pin_enforced": not AUDIT_CANDIDATE,
        "suite_matches_pin": _suite_matches_pin(ACTUAL_SUITE),
        "ledger_matches_pin": KNOWN_FAILURE_CONFIG["suite_commit"] == PIN["commit"],
        "unknown_ledger_entries": sorted(set(KNOWN_FAILURES).difference(CASE_KEYS)),
        "duplicate_results": sorted(
            key for key, count in Counter(keys).items() if count > 1
        ),
        "missing_results": sorted(CASE_KEYS.difference(result_keys)),
        "extra_results": sorted(result_keys.difference(CASE_KEYS)),
        "unexpected_statuses": {
            _result_key(result): result.status
            for result in results
            if result.status not in {"passed", "failed"}
        },
    }


def _pyhs3_version() -> str:
    try:
        return importlib.metadata.version("pyhs3")
    except importlib.metadata.PackageNotFoundError:
        return "uninstalled"


def _markdown_cell(value: str, limit: int = 300) -> str:
    normalized = " ".join(value.split()).replace("|", "\\|")
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1]}…"


def _write_reports(results: list[CheckResult], error: str | None = None) -> None:
    report_dir_value = os.environ.get("HS3TESTSUITE_REPORT_DIR")
    if report_dir_value is None:
        return

    report_dir = Path(report_dir_value)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "test_id": result.test_id,
            "check_id": result.check_id,
            "key": _result_key(result),
            "status": result.status,
            "classification": _classify(result),
            "failure_stage": _failure_stage(result),
            "message": result.message,
            "known_failure": KNOWN_FAILURES.get(_result_key(result)),
        }
        for result in results
    ]
    counts = Counter(row["classification"] for row in rows)
    diagnostics = _result_diagnostics(results)
    summary = (
        ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        or "no check results"
    )
    payload = {
        "schema_version": 1,
        "suite": ACTUAL_SUITE,
        "expected_suite": PIN,
        "pyhs3_version": _pyhs3_version(),
        "error": error,
        "harness": diagnostics,
        "summary": dict(sorted(counts.items())),
        "results": rows,
    }
    with (report_dir / "hs3suite-results.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")

    lines = [
        "# HS3TestSuite compatibility report",
        "",
        f"- Suite commit: `{ACTUAL_SUITE['commit']}`",
        f"- Expected pinned commit: `{PIN['commit']}`",
        f"- Exact pin enforced: `{'no (candidate audit)' if AUDIT_CANDIDATE else 'yes'}`",
        f"- pyhs3 version: `{payload['pyhs3_version']}`",
        "- Reference: frozen RooFit values and tolerances from the selected suite revision",
        f"- Summary: {summary}",
        "- Result-set diagnostics: "
        f"missing={len(diagnostics['missing_results'])}, "
        f"extra={len(diagnostics['extra_results'])}, "
        f"duplicates={len(diagnostics['duplicate_results'])}, "
        f"invalid_statuses={len(diagnostics['unexpected_statuses'])}",
    ]
    if error:
        lines.extend((f"- Runner error: `{_markdown_cell(error)}`", ""))
    else:
        lines.append("")
    lines.extend(
        (
            "| Fixture | Check | Classification | Detail |",
            "| --- | --- | --- | --- |",
        )
    )
    for row in rows:
        failure = row["known_failure"]
        detail = row["message"]
        if failure is not None:
            detail = (
                f"{failure['category']} "
                f"(expected stage={_expected_failure_stage(failure)}, "
                f"actual stage={row['failure_stage']}): {failure['reason']} Raw: {detail}"
            )
        lines.append(
            "| "
            + " | ".join(
                (
                    row["test_id"],
                    row["check_id"],
                    row["classification"],
                    _markdown_cell(detail),
                )
            )
            + " |"
        )
    (report_dir / "hs3suite-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


@pytest.fixture(scope="session")
def hs3_results() -> list[CheckResult]:
    try:
        results = run_suite(SUITE_ROOT, "pyhs3")
    except Exception as exc:
        _write_reports([], error=f"{type(exc).__name__}: {exc}")
        raise
    _write_reports(results)
    return results


@pytest.fixture(scope="session")
def hs3_result_map(hs3_results: list[CheckResult]) -> Mapping[str, CheckResult]:
    keys = [_result_key(result) for result in hs3_results]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    assert not duplicates, f"duplicate HS3TestSuite results: {duplicates}"
    return dict(zip(keys, hs3_results, strict=True))


def test_suite_checkout_matches_pin() -> None:
    assert ACTUAL_SUITE["commit"]
    if not AUDIT_CANDIDATE:
        assert _suite_matches_pin(ACTUAL_SUITE)
        assert ACTUAL_SUITE["checkout_clean"]


def test_known_failure_ledger_matches_pin_and_manifest() -> None:
    assert KNOWN_FAILURE_CONFIG["suite_commit"] == PIN["commit"]
    unknown = sorted(set(KNOWN_FAILURES).difference(CASE_KEYS))
    assert not unknown, f"known-failure entries are absent from the manifest: {unknown}"
    unknown_categories = sorted(
        {
            failure["category"]
            for failure in KNOWN_FAILURES.values()
            if failure["category"] not in EXPECTED_STAGES_BY_CATEGORY
        }
    )
    assert not unknown_categories, (
        f"known-failure categories have no expected stage: {unknown_categories}"
    )


def test_runner_returns_exact_manifest_checks(
    hs3_result_map: Mapping[str, CheckResult],
) -> None:
    actual = set(hs3_result_map)
    missing = sorted(CASE_KEYS.difference(actual))
    extra = sorted(actual.difference(CASE_KEYS))
    assert not missing, f"HS3TestSuite results are missing checks: {missing}"
    assert not extra, f"HS3TestSuite returned unexpected checks: {extra}"


def test_runner_returns_only_passed_or_failed(
    hs3_result_map: Mapping[str, CheckResult],
) -> None:
    unexpected = {
        key: result.status
        for key, result in hs3_result_map.items()
        if result.status not in {"passed", "failed"}
    }
    assert not unexpected, f"unexpected HS3TestSuite statuses: {unexpected}"


def test_known_failures_match_expected_stages(
    hs3_result_map: Mapping[str, CheckResult],
) -> None:
    mismatches = {}
    for key, failure in KNOWN_FAILURES.items():
        result = hs3_result_map.get(key)
        if result is None or result.status != "failed":
            continue
        expected = _expected_failure_stage(failure)
        actual = _failure_stage(result)
        if actual != expected:
            mismatches[key] = {"expected": expected, "actual": actual}
    assert not mismatches, f"known failures changed stage: {mismatches}"


@pytest.mark.parametrize(("test_id", "check_id"), CASE_PARAMETERS)
def test_hs3_conformance_check(
    test_id: str,
    check_id: str,
    hs3_result_map: Mapping[str, CheckResult],
) -> None:
    key = f"{test_id}::{check_id}"
    assert key in hs3_result_map, f"HS3TestSuite did not return {key}"
    result = hs3_result_map[key]
    assert result.status == "passed", result.message
