"""Run the pyhs3 compatibility checks against an explicit suite checkout."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)
REPORT_FILENAMES = (
    "hs3suite-junit.xml",
    "hs3suite-results.json",
    "hs3suite-summary.md",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a candidate HS3TestSuite checkout without changing the tracked "
            "suite pin or known-failure ledger."
        )
    )
    parser.add_argument(
        "--suite-root",
        type=Path,
        required=True,
        help="Path to an explicit HS3TestSuite checkout",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Report directory (defaults to a new directory under the system temp dir)",
    )
    return parser


def _write_wrapper_failure(output_dir: Path, returncode: int) -> None:
    message = (
        "pytest ended before the HS3TestSuite harness could write its reports "
        f"(exit code {returncode})"
    )
    payload = {
        "schema_version": 1,
        "error": message,
        "results": [],
        "summary": {},
    }
    with (output_dir / "hs3suite-results.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    (output_dir / "hs3suite-summary.md").write_text(
        f"# HS3TestSuite compatibility report\n\n- Audit error: `{message}`\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    suite_root = args.suite_root.resolve()
    if not (suite_root / "manifest.json").is_file():
        _parser().error(f"suite root does not contain manifest.json: {suite_root}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(tempfile.mkdtemp(prefix="pyhs3-hs3suite-audit-"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in REPORT_FILENAMES:
        (output_dir / filename).unlink(missing_ok=True)

    env = os.environ.copy()
    env["HS3TESTSUITE_ROOT"] = str(suite_root)
    env["HS3TESTSUITE_REPORT_DIR"] = str(output_dir)
    env["PYHS3_HS3TESTSUITE_AUDIT_CANDIDATE"] = "1"
    env.setdefault("PYTENSOR_FLAGS", f"base_compiledir={output_dir / 'pytensor'}")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_hs3testsuite.py",
        "-ra",
        f"--junitxml={output_dir / 'hs3suite-junit.xml'}",
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
    )
    reports = (
        output_dir / "hs3suite-results.json",
        output_dir / "hs3suite-summary.md",
    )
    if not all(report.is_file() for report in reports):
        _write_wrapper_failure(output_dir, completed.returncode)
        log.info("HS3TestSuite audit reports: %s", output_dir)
        return completed.returncode or 1
    log.info("HS3TestSuite audit reports: %s", output_dir)
    return completed.returncode


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
