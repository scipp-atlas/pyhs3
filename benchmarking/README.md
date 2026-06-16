# pyHS3 Benchmarking

This directory contains benchmarking infrastructure and benchmark scripts for measuring pyHS3 performance.

The goal is to provide a simple, reproducible, and extensible framework for evaluating runtime and memory usage of different pyHS3 workflow stages.

The benchmarking suite is intentionally developed incrementally, starting with small benchmark components and expanding toward larger performance studies and comparisons.

# Directory Structure

```text
benchmarking/
├── inputs/     # benchmark input workspaces
├── results/    # JSON benchmark outputs
├── plots/      # generated benchmark plots
├── src/        # benchmark implementations and shared utilities
└── README.md
```

The benchmarking suite focuses on measuring pyHS3 workflow stages individually rather than combining multiple operations into a single benchmark.

# Current Benchmarks

## Benchmark 1 — Workspace Loading

This benchmark measures the runtime and memory usage of loading an HS3 JSON workspace into a pyHS3 Workspace object. It also validates that the loaded workspace contains the expected top-level components, such as distributions, likelihoods, and data.

Measures:

```text
HS3 workspace JSON
        ↓
pyHS3 Workspace object
```

This benchmark evaluates:

* workspace loading;
* JSON parsing;
* Pydantic model construction;
* workspace validation;
* runtime characteristics;
* memory usage characteristics.

It does **not** include:

* workspace generation;
* workspace export;
* model construction (`ws.model()`);
* likelihood compilation;
* likelihood evaluation.

These stages are benchmarked separately.

# Metrics

Current benchmarks may report:

* wall time (mean);
* wall time (standard deviation);
* current RSS before execution;
* current RSS after execution;
* current RSS delta;
* peak RSS before execution;
* peak RSS after execution;
* peak RSS delta;
* validation statistics.

Example result:

```json
{
  "benchmark": "workspace_loading",
  "workspace": "simple_workspace_nonp.json",
  "n_runs": 5,

  "wall_time_seconds_mean": 0.0021,
  "wall_time_seconds_std": 0.0007,

  "current_rss_before_mb": 206.8,
  "current_rss_after_mb": 207.2,
  "current_rss_delta_mb": 0.4,

  "peak_rss_before_mb": 206.3,
  "peak_rss_after_mb": 206.7,
  "peak_rss_delta_mb": 0.4,

  "status": "success"
}
```

# Running Benchmarks

## Workspace Loading

### Command Line Arguments

| Argument | Description | Default |
|-----------|-------------|----------|
| `--workspaces` | One or more HS3 workspace JSON files to benchmark. | `benchmarking/inputs/simple_workspace_nonp.json` |
| `--n-runs` | Number of repeated benchmark runs per workspace. | `5` |
| `--output-dir` | Directory where benchmark JSON results will be stored. | `benchmarking/results/workspace_loading` |
| `--output-name` | Name of the benchmark JSON output file. | `workspace_loading_result.json` |
| `--plot` | Generate a wall-time comparison plot. Requires at least two workspaces. | Disabled |
| `--plot-dir` | Directory where generated plots will be stored. | `benchmarking/plots/workspace_loading` |
| `--plot-name` | Name of the generated plot file. | `workspace_loading_wall_time.png` |

```bash
pixi run python benchmarking/src/run_workspace_loading.py
```

```bash
pixi run python benchmarking/src/run_workspace_loading.py \
  --n-runs 20
```

```bash
pixi run python benchmarking/src/run_workspace_loading.py \
  --workspaces \
  benchmarking/inputs/simple_workspace_nonp.json \
  benchmarking/inputs/simple_workspace.json
```

```bash
pixi run python benchmarking/src/run_workspace_loading.py \
  --workspaces \
  benchmarking/inputs/simple_workspace_nonp.json \
  benchmarking/inputs/simple_workspace.json \
  --plot
```

# Outputs

Benchmark results are saved under:

```text
benchmarking/results/<benchmark_name>/
```

For example:

```text
benchmarking/results/workspace_loading/
└── workspace_loading_result.json
```

Generated plots are saved under:

```text
benchmarking/plots/<benchmark_name>/
```

For example:

```text
benchmarking/plots/workspace_loading/
└── workspace_loading_wall_time.png
```

Memory plots are generated only when the measured memory metric contains non-zero values. This avoids producing empty plots for benchmarks where memory changes are below the measurement resolution.
