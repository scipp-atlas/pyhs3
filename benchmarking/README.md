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

## Benchmark 2 — Model Creation

This benchmark measures the time and memory required to construct a pyHS3 model from a selected likelihood target. The benchmark validates that the resulting model exposes the expected interfaces (log_prob, data access, and free parameters) and reports runtime and memory characteristics of the model creation stage.

Measures:

```text
Loaded Workspace
        ↓
ws.model(...)
        ↓
Model
```

This benchmark evaluates:

* likelihood model construction;
* runtime characteristics;
* memory usage characteristics;
* model validation.

Workspace loading is intentionally excluded from the timed section so that the benchmark isolates the cost of `ws.model(...)`.

Validation checks include:

* model object creation;
* availability of `log_prob`;
* availability of model data;
* availability of free parameters.

This benchmark does **not** include:

* likelihood graph construction;
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

## Model Creation

### Command Line Arguments

| Argument | Description | Default |
|-----------|-------------|----------|
| `--workspaces`  | One or more HS3 workspace JSON files to benchmark.| `simple_workspace_nonp.json`|
| `--targets`     | One or more likelihood targets for model creation.| `L_ch0`|
| `--n-runs`      | Number of repeated benchmark runs per target.   | `5`|
| `--output-dir`  | Directory where benchmark JSON results will be stored.| `benchmarking/results/model_creation` |
| `--output-name` | Name of the benchmark JSON output file. | `model_creation_result.json` |
| `--plot`        | Generate comparison plots for wall time and memory usage. Requires at least two benchmark results. | Disabled   |
| `--plot-dir`    | Directory where generated plots will be stored.| `benchmarking/plots/model_creation`   |

```bash
pixi run python benchmarking/src/run_model_creation.py
```

```bash
pixi run python benchmarking/src/run_model_creation.py \
  --n-runs 20
```

```bash
pixi run python benchmarking/src/run_model_creation.py \
  --workspaces \
  benchmarking/inputs/simple_workspace_nonp.json \
  benchmarking/inputs/simple_workspace.json
```

```bash
pixi run python benchmarking/src/run_model_creation.py \
  --targets L_ch0 L_ch1 L_ch2
```

```bash
pixi run python benchmarking/src/run_model_creation.py \
  --workspaces \
  benchmarking/inputs/simple_workspace_nonp.json \
  benchmarking/inputs/simple_workspace.json \
  --targets L_ch0 L_ch1 L_ch2 \
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

benchmarking/results/model_creation/
└── model_creation_result.json
```

Generated plots are saved under:

```text
benchmarking/plots/<benchmark_name>/
```

For example:

```text
benchmarking/plots/workspace_loading/
└── workspace_loading_wall_time.png

benchmarking/plots/model_creation/
└── model_creation_wall_time.png
```

Memory plots are generated only when the measured memory metric contains non-zero values. This avoids producing empty plots for benchmarks where memory changes are below the measurement resolution.
