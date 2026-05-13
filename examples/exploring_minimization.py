# ruff: noqa: T201
"""Benchmark multiple minimizers on any HS3 workspace POI scan.

Demonstrates how to:
1. Build a pyhs3 Model from any HS3 workspace
2. Cache both the Model and compiled log_prob in a single pickle
3. Profile over nuisance parameters at each fixed POI value
4. Capture per-fit wall time, CPU time, and memory usage
5. Write results to a bundled JSON suitable for plot_comparisons.py

Install: pixi run -e minimize python examples/exploring_minimization.py --help
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import pickle
import platform
import threading
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from iminuit import minimize as iminuit_minimize
from pytensor import function
from pytensor.graph.traversal import explicit_graph_inputs
from scipy.optimize import Bounds
from scipy.optimize import minimize as scipy_minimize

import pyhs3

_ALL_METHODS = ["SLSQP", "L-BFGS-B", "TNC", "trust-constr", "migrad"]
_ALL_TOLERANCES = [1e-3, 1e-4, 1e-5, 1e-6]


def _collect_machine_info() -> dict:
    """Collect CPU / OS metadata for the bundle header."""
    freq = psutil.cpu_freq()
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        # processor() is empty on some platforms (e.g. macOS arm64); fall back to machine
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": round(freq.current, 0) if freq else None,
        "total_ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark minimizers on a pyhs3 workspace POI scan."
    )
    p.add_argument(
        "--workspace", type=Path, required=True, help="Path to HS3 JSON workspace"
    )
    p.add_argument("--analysis", required=True, help="Analysis name in the workspace")
    p.add_argument("--poi", required=True, help="Parameter of interest name")

    poi_group = p.add_mutually_exclusive_group(required=True)
    poi_group.add_argument(
        "--poi-points",
        help="Comma-separated POI values, e.g. '0.5,1.0,1.5'",
    )
    poi_group.add_argument(
        "--poi-range",
        nargs=3,
        metavar=("MIN", "MAX", "N"),
        help="POI linspace: MIN MAX N",
    )

    p.add_argument(
        "--parameter-points",
        help="Comma-separated ParameterSet names to merge for initial values (default: default_values)",
    )
    p.add_argument(
        "--methods",
        help=f"Comma-separated minimizer methods (default: {','.join(_ALL_METHODS)})",
    )
    p.add_argument(
        "--tolerances",
        help=f"Comma-separated tolerance floats (default: {','.join(str(t) for t in _ALL_TOLERANCES)})",
    )
    p.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Max iterations per fit (default: 1000)",
    )
    p.add_argument(
        "--cache",
        type=Path,
        help="Cache pickle path (default: <workspace_stem>.cache.pkl)",
    )
    p.add_argument(
        "--rebuild", action="store_true", help="Ignore existing cache and rebuild"
    )
    p.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Worker processes (default: 1 = serial)",
    )
    p.add_argument(
        "--output", type=Path, default=Path("benchmark.json"), help="Output JSON path"
    )
    p.add_argument(
        "--resume", action="store_true", help="Skip already-completed scans in --output"
    )
    p.add_argument(
        "--sampler-hz", type=float, default=20.0, help="CPU/RSS sampler frequency in Hz"
    )
    p.add_argument(
        "--label-prefix", default="", help="Optional prefix prepended to scan labels"
    )
    return p


def parse_poi_points(args: argparse.Namespace) -> list[float]:
    """Return the POI grid as a plain list of floats."""
    if args.poi_points is not None:
        return [float(v) for v in args.poi_points.split(",")]
    mn, mx, n = args.poi_range
    return list(np.linspace(float(mn), float(mx), int(n)))


# ---------------------------------------------------------------------------
# Workspace + model loading
# ---------------------------------------------------------------------------


def load_workspace(
    path: Path,
    analysis_name: str,
    parameter_points_names: list[str],
) -> tuple[Any, Any, Any]:
    """Load workspace, return (ws, analysis, param_set).

    Raises ValueError if analysis_name or any parameter_points_names is absent.
    """
    print(f"Loading workspace from {path.name} ...")
    ws = pyhs3.Workspace.load(str(path))

    if analysis_name not in ws.analyses:
        available = list(ws.analyses.keys())
        msg = f"Analysis {analysis_name!r} not found. Available: {available}"
        raise ValueError(msg)
    analysis = ws.analyses[analysis_name]

    available_psets = [ps.name for ps in ws.parameter_points.root]
    missing = [n for n in parameter_points_names if n not in ws.parameter_points]
    if missing:
        msg = f"Parameter point sets not found: {missing}. Available: {sorted(available_psets)}"
        raise ValueError(msg)

    param_set = pyhs3.parameter_points.ParameterSet(
        name="collected",
        parameters=[
            pp
            for pset_name in parameter_points_names
            for pp in ws.parameter_points[pset_name]
        ],
    )
    return ws, analysis, param_set


def compile_log_prob(model: Any) -> tuple[Any, list[Any], float]:
    """Compile model.log_prob into a pytensor function.

    Returns (log_prob_fn, inputs, compile_time_s).
    """
    dist_expression = model.log_prob
    inputs = [
        var for var in explicit_graph_inputs([dist_expression]) if var.name is not None
    ]
    t0 = time.perf_counter()
    log_prob_fn = function(
        inputs=inputs,
        outputs=dist_expression,
        mode=model.mode,
        on_unused_input="ignore",
        name=model._likelihood.name,
        trust_input=True,
    )
    compile_time_s = time.perf_counter() - t0
    return log_prob_fn, inputs, compile_time_s


def _atomic_pickle(payload: dict, path: Path) -> None:
    """Write payload to path atomically via a temp file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f)
    tmp.replace(path)


def load_or_build_cache(
    ws_path: Path,
    analysis_name: str,
    parameter_points_names: list[str],
    cache_path: Path,
    force_rebuild: bool,
) -> tuple[dict, float, float]:
    """Return (cache_dict, build_time_s, compile_time_s).

    cache_dict keys: "model", "log_prob", "input_names".
    If loading from disk, build_time_s and compile_time_s are 0.0.
    """
    if cache_path.exists() and not force_rebuild:
        print(f"Loading model + compiled log_prob from cache {cache_path} ...")
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        print(
            f"  Loaded {len(payload['input_names'])} inputs, "
            f"{len(payload['model'].free_params)} free params"
        )
        return payload, 0.0, 0.0

    ws, analysis, param_set = load_workspace(
        ws_path, analysis_name, parameter_points_names
    )

    print("Building symbolic model ...")
    t0 = time.perf_counter()
    model = ws.model(analysis, parameter_set=param_set, progress=True)
    build_time_s = time.perf_counter() - t0
    print(f"  Built in {build_time_s:.1f}s")

    print("Compiling log_prob ...")
    log_prob_fn, inputs, compile_time_s = compile_log_prob(model)
    print(f"  Compiled in {compile_time_s:.1f}s  ({len(inputs)} inputs)")

    payload = {
        "model": model,
        "log_prob": log_prob_fn,
        "input_names": [v.name for v in inputs],
    }
    _atomic_pickle(payload, cache_path)
    print(f"  Cache written to {cache_path}")

    return payload, build_time_s, compile_time_s


# ---------------------------------------------------------------------------
# ResourceSampler
# ---------------------------------------------------------------------------


class ResourceSampler:
    """Context manager that polls psutil in a daemon thread at sampler_hz Hz."""

    def __init__(self, sampler_hz: float = 20.0) -> None:
        self._hz = sampler_hz
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self.rss_before_mb: float = 0.0
        self.rss_after_mb: float = 0.0
        self.rss_peak_mb: float = 0.0
        self.cpu_percent_mean: float = 0.0
        self.n_samples: int = 0
        self._rss_samples: list[float] = []
        self._cpu_samples: list[float] = []

    def _sample_loop(self) -> None:
        interval = 1.0 / self._hz
        # Prime cpu_percent so first real sample is meaningful
        self._proc.cpu_percent(interval=None)
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss / 2**20
            cpu = self._proc.cpu_percent(interval=None)
            self._rss_samples.append(rss)
            self._cpu_samples.append(cpu)
            self._stop.wait(interval)

    def __enter__(self) -> ResourceSampler:
        self.rss_before_mb = self._proc.memory_info().rss / 2**20
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.rss_after_mb = self._proc.memory_info().rss / 2**20
        if self._rss_samples:
            self.rss_peak_mb = float(max(self._rss_samples))
            self.cpu_percent_mean = float(np.mean(self._cpu_samples))
            self.n_samples = len(self._rss_samples)
        else:
            self.rss_peak_mb = self.rss_before_mb
            self.cpu_percent_mean = 0.0
            self.n_samples = 0


# ---------------------------------------------------------------------------
# Single-point fit
# ---------------------------------------------------------------------------


def profile_nll(
    log_prob_fn: Any,
    input_names: list[str],
    model: Any,
    poi_name: str,
    poi_value: float,
    method: str,
    tol: float,
    maxiter: int,
    sampler_hz: float,
) -> dict:
    """Minimise -2*log_prob over nuisance params with poi_name fixed at poi_value.

    Returns a per-point record matching the bundle JSON schema.
    """
    pinned = {**model.data, poi_name: poi_value}

    template: list[Any] = []
    free_names: list[str] = []
    free_input_indices: list[int] = []
    bounds_list: list[tuple[float, float]] = []

    model_bounds = {axis.name: (axis.min, axis.max) for axis in model.domain.axes}
    for i, name in enumerate(input_names):
        if name in pinned:
            template.append(np.asarray(pinned[name]))
        else:
            template.append(None)
            free_names.append(name)
            free_input_indices.append(i)
            bounds_list.append(model_bounds[name])

    x0 = np.array([model.free_params[n] for n in free_names], dtype=float)

    def nll(x: np.ndarray) -> float:
        vals = list(template)
        for idx, xi in zip(free_input_indices, x, strict=True):
            vals[idx] = xi
        return float(-2.0 * log_prob_fn(*[np.asarray(v) for v in vals]))

    t_wall_start = time.perf_counter()
    t_cpu_start = time.thread_time()
    error_msg = None
    result = None

    with ResourceSampler(sampler_hz) as rs:
        try:
            if method == "migrad":
                result = iminuit_minimize(
                    nll,
                    x0,
                    bounds=bounds_list,
                    tol=tol,
                    options={"maxfun": maxiter},
                )
            elif method == "trust-constr":
                lb = [b[0] for b in bounds_list]
                ub = [b[1] for b in bounds_list]
                result = scipy_minimize(
                    nll,
                    x0,
                    method=method,
                    bounds=Bounds(lb, ub),
                    tol=tol,
                    options={"maxiter": maxiter},
                )
            elif method == "TNC":
                # TNC uses maxfun (function evaluations), not maxiter
                result = scipy_minimize(
                    nll,
                    x0,
                    method=method,
                    bounds=bounds_list,
                    tol=tol,
                    options={"maxfun": maxiter, "ftol": tol},
                )
            else:
                # ftol intentionally omitted: tol= maps to the method's primary
                # criterion (gtol for L-BFGS-B, ftol for SLSQP) via scipy's
                # unified interface. Explicitly passing ftol here would override
                # L-BFGS-B's tight default (2.22e-9) with a loose value and
                # cause premature convergence to a suboptimal minimum.
                result = scipy_minimize(
                    nll,
                    x0,
                    method=method,
                    bounds=bounds_list,
                    tol=tol,
                    options={"maxiter": maxiter},
                )
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"

    wall_s = time.perf_counter() - t_wall_start
    cpu_s = time.thread_time() - t_cpu_start

    if result is not None:
        nll_val = (
            float(result.fun)
            if result.success or result.fun is not None
            else float("nan")
        )
        nit = int(getattr(result, "nit", getattr(result, "niter", -1)))
        nfev = int(getattr(result, "nfev", -1))
        success = bool(result.success)
        message = str(getattr(result, "message", ""))
    else:
        nll_val = float("nan")
        nit = -1
        nfev = -1
        success = False
        message = ""

    return {
        "poi": poi_value,
        "nll": nll_val,
        "success": success,
        "nfev": nfev,
        "nit": nit,
        "wall_s": round(wall_s, 4),
        "cpu_s": round(cpu_s, 4),
        "rss_before_mb": round(rs.rss_before_mb, 2),
        "rss_after_mb": round(rs.rss_after_mb, 2),
        "rss_peak_mb": round(rs.rss_peak_mb, 2),
        "cpu_percent_mean": round(rs.cpu_percent_mean, 2),
        "n_samples": rs.n_samples,
        "message": message,
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Scan loop
# ---------------------------------------------------------------------------


def _scan_label(method: str, tol: float, prefix: str) -> str:
    label = f"{method}, tol={tol:.0e}"
    return f"{prefix}{label}" if prefix else label


def run_scan(
    log_prob_fn: Any,
    input_names: list[str],
    model: Any,
    poi_name: str,
    poi_grid: list[float],
    method: str,
    tol: float,
    maxiter: int,
    sampler_hz: float,
    label: str,
) -> dict:
    """Run a full scan over poi_grid for a single (method, tol) combination."""
    points = []
    scan_wall_start = time.perf_counter()
    scan_cpu_start = time.thread_time()
    peak_rss = 0.0

    for mu in poi_grid:
        try:
            pt_record = profile_nll(
                log_prob_fn,
                input_names,
                model,
                poi_name,
                mu,
                method,
                tol,
                maxiter,
                sampler_hz,
            )
        except Exception as exc:
            pt_record = {
                "poi": mu,
                "nll": float("nan"),
                "success": False,
                "nfev": -1,
                "nit": -1,
                "wall_s": 0.0,
                "cpu_s": 0.0,
                "rss_before_mb": 0.0,
                "rss_after_mb": 0.0,
                "rss_peak_mb": 0.0,
                "cpu_percent_mean": 0.0,
                "n_samples": 0,
                "message": "",
                "error": f"{type(exc).__name__}: {exc}",
            }

        peak_rss = max(peak_rss, pt_record.get("rss_peak_mb", 0.0))
        status = "ok" if pt_record["success"] else "FAILED"
        extra = (
            f"  error={pt_record['error']}"
            if pt_record["error"]
            else f"  nfev={pt_record['nfev']}  nit={pt_record['nit']}"
        )
        print(
            f"[{label}] poi={mu:+.3f}  nll={pt_record['nll']:.6f}  {status}"
            f"{extra}  ({pt_record['wall_s']:.1f}s, peak +{pt_record['rss_peak_mb'] - pt_record['rss_before_mb']:.1f} MB)"
        )
        points.append(pt_record)

    wall_s = time.perf_counter() - scan_wall_start
    cpu_s = time.thread_time() - scan_cpu_start

    return {
        "label": label,
        "method": method,
        "tol": tol,
        "maxiter": maxiter,
        "wall_s": round(wall_s, 4),
        "cpu_s": round(cpu_s, 4),
        "peak_rss_mb": round(peak_rss, 2),
        "points": points,
        # points_scan_order alias for plot_comparisons.py reference loader
        "points_scan_order": [{"poi": p["poi"], "nll": p["nll"]} for p in points],
    }


# ---------------------------------------------------------------------------
# Atomic JSON write
# ---------------------------------------------------------------------------


def _atomic_write_bundle(bundle: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(bundle, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Multiprocessing worker globals
# ---------------------------------------------------------------------------

_WORKER_FN = None
_WORKER_INPUT_NAMES = None
_WORKER_MODEL = None


def _worker_init(cache_path: str) -> None:
    global _WORKER_FN, _WORKER_INPUT_NAMES, _WORKER_MODEL  # noqa: PLW0603
    with Path(cache_path).open("rb") as f:
        payload = pickle.load(f)
    _WORKER_FN = payload["log_prob"]
    _WORKER_INPUT_NAMES = payload["input_names"]
    _WORKER_MODEL = payload["model"]


def _worker_run_scan(
    method: str,
    tol: float,
    poi_name: str,
    poi_grid: list[float],
    maxiter: int,
    sampler_hz: float,
    label: str,
) -> dict:
    return run_scan(
        _WORKER_FN,
        _WORKER_INPUT_NAMES,
        _WORKER_MODEL,
        poi_name,
        poi_grid,
        method,
        tol,
        maxiter,
        sampler_hz,
        label,
    )


def _failed_scan_dict(
    method: str, tol: float, maxiter: int, label: str, exc: Exception
) -> dict:
    return {
        "label": label,
        "method": method,
        "tol": tol,
        "maxiter": maxiter,
        "wall_s": 0.0,
        "cpu_s": 0.0,
        "peak_rss_mb": 0.0,
        "points": [],
        "points_scan_order": [],
        "error": f"{type(exc).__name__}: {exc}",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    poi_grid = parse_poi_points(args)

    methods = (
        [m.strip() for m in args.methods.split(",")] if args.methods else _ALL_METHODS
    )
    tolerances = (
        [float(t) for t in args.tolerances.split(",")]
        if args.tolerances
        else _ALL_TOLERANCES
    )
    pset_names = (
        [n.strip() for n in args.parameter_points.split(",")]
        if args.parameter_points
        else ["default_values"]
    )

    cache_path = (
        args.cache
        if args.cache
        else args.workspace.with_name(args.workspace.stem + ".cache.pkl")
    )

    payload, build_time_s, compile_time_s = load_or_build_cache(
        args.workspace, args.analysis, pset_names, cache_path, args.rebuild
    )
    model = payload["model"]
    log_prob_fn = payload["log_prob"]
    input_names = payload["input_names"]

    proc = psutil.Process()
    rss_baseline_mb = proc.memory_info().rss / 2**20

    print(f"\n{len(input_names)} inputs, {len(model.free_params)} free params")
    print(
        f"POI grid: {len(poi_grid)} points from {poi_grid[0]:.3g} to {poi_grid[-1]:.3g}"
    )
    print(f"Methods: {methods}")
    print(f"Tolerances: {tolerances}")

    bundle: dict = {
        "workspace": args.workspace.name,
        "analysis": args.analysis,
        "poi": args.poi,
        "poi_grid": poi_grid,
        "n_inputs": len(input_names),
        "n_free_params": len(model.free_params),
        "compile_time_s": compile_time_s,
        "model_build_s": build_time_s,
        "rss_baseline_mb": round(rss_baseline_mb, 2),
        "sampler_hz": args.sampler_hz,
        "machine": _collect_machine_info(),
        "scans": [],
    }

    # --resume: load existing bundle and skip already-completed scans
    completed_labels: set[str] = set()
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text())
        if (
            existing.get("workspace") == bundle["workspace"]
            and existing.get("analysis") == bundle["analysis"]
            and existing.get("poi") == bundle["poi"]
            and existing.get("poi_grid") == bundle["poi_grid"]
        ):
            bundle["scans"] = existing["scans"]
            completed_labels = {s["label"] for s in bundle["scans"]}
            print(f"\nResuming: {len(completed_labels)} scans already done.")
        else:
            msg = (
                "ERROR: --resume requested but existing bundle header does not match. "
                "Remove the file or omit --resume."
            )
            raise SystemExit(msg)

    scan_configs = [
        (m, t, _scan_label(m, t, args.label_prefix))
        for m, t in product(methods, tolerances)
        if _scan_label(m, t, args.label_prefix) not in completed_labels
    ]

    print(f"\nRunning {len(scan_configs)} scans ...\n")

    try:
        if args.processes <= 1:
            for method, tol, label in scan_configs:
                scan = run_scan(
                    log_prob_fn,
                    input_names,
                    model,
                    args.poi,
                    poi_grid,
                    method,
                    tol,
                    args.maxiter,
                    args.sampler_hz,
                    label,
                )
                bundle["scans"].append(scan)
                bundle["scans"].sort(key=lambda s: (s["method"], s["tol"]))
                _atomic_write_bundle(bundle, args.output)
                print(f"  → Checkpoint written ({len(bundle['scans'])} scans total)\n")
        else:
            ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.processes,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(str(cache_path),),
            ) as pool:
                futures = {
                    pool.submit(
                        _worker_run_scan,
                        m,
                        t,
                        args.poi,
                        poi_grid,
                        args.maxiter,
                        args.sampler_hz,
                        label,
                    ): (m, t, label)
                    for m, t, label in scan_configs
                }
                for fut in concurrent.futures.as_completed(futures):
                    m, t, label = futures[fut]
                    try:
                        scan = fut.result()
                    except Exception as exc:
                        scan = _failed_scan_dict(m, t, args.maxiter, label, exc)
                    bundle["scans"].append(scan)
                    bundle["scans"].sort(key=lambda s: (s["method"], s["tol"]))
                    _atomic_write_bundle(bundle, args.output)
                    print(
                        f"  → Checkpoint written ({len(bundle['scans'])} scans total)\n"
                    )
    except KeyboardInterrupt:
        print("\nInterrupted — writing partial bundle ...")
        _atomic_write_bundle(bundle, args.output)
        raise

    print(f"\nDone. Results written to {args.output}")
    print(
        f"  {len(bundle['scans'])} scans, {sum(len(s['points']) for s in bundle['scans'])} total fits"
    )


if __name__ == "__main__":
    main()
