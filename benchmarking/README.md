# pyHS3 benchmarking

This directory contains benchmarking scripts for pyHS3 performance studies.

The goal is to provide a simple and reproducible structure for measuring runtime
and memory usage of pyHS3 operations, starting with small benchmark scripts and
expanding over time.

## Structure

```text
benchmarking/
  inputs/     # input workspaces or generated benchmark inputs
  results/    # JSON benchmark outputs
  scripts/    # runnable benchmark scripts and shared helpers
  plots/      # optional plots generated from benchmark results
```

## Default settings

The default PyTensor compilation mode for benchmarks is `FAST_RUN`.

Benchmark scripts should be easy to rerun and should save structured output,
preferably as JSON files under `benchmarking/results/`.

## Result format

A minimal benchmark result should look like this:
```
{
  "benchmark": "workspace_loading",
  "workspace": "small",
  "mode": "FAST_RUN",
  "n_runs": 5,
  "wall_time_seconds_mean": 0.0,
  "wall_time_seconds_std": 0.0,
  "rss_mb_peak": 0.0,
  "status": "success"
}
```

## Running the first benchmark

From the repository root, run:

```
pixi run python benchmarking/scripts/run_workspace_loading.py
```

The result will be written to:

```
benchmarking/results/workspace_loading_result.json
```

## Notes

The first version of this benchmarking suite is intentionally simple. The goal is
to establish a clean structure before adding more complete pyHS3, ROOT, pyhf,
zfit, or numba-stats comparisons.