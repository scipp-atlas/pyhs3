# pyHS3 Benchmarking

This directory contains benchmarking infrastructure and benchmark scripts for measuring pyHS3 performance.

The goal is to provide a simple, reproducible, and extensible framework for evaluating runtime and memory usage of different pyHS3 workflow stages.

The benchmarking suite is intentionally developed incrementally, starting with small benchmark components and expanding toward larger performance studies and comparisons.

# Benchmark Baseline

Benchmark results should always be interpreted relative to a specific
pyHS3 version.

Current benchmark development is based on:

```text
pyHS3 main SHA: 326aadd
```

When collecting benchmark results, the short SHA of the `main` commit
used as the baseline should be recorded together with the benchmark
outputs and reports.

# Directory Structure

```text
benchmarking/
├── inputs/     # benchmark input workspaces
├── results/    # JSON benchmark outputs
├── plots/      # generated benchmark plots
├── src/        # benchmark implementations and shared utilities
├── reports/    # reports benchmark outputs
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

## Benchmark 3 — log_prob Construction

Measures the runtime and memory usage required to construct the symbolic likelihood expression (`model.log_prob`) for a given likelihood target in pyHS3.

Evaluates:

computational graph construction;
runtime characteristics;
memory usage characteristics;
correctness of the constructed expression.

Measures:

```text
Model
   ↓
model.log_prob
   ↓
Likelihood expression
```

This benchmark evaluates:

* symbolic likelihood expression construction;
* PyTensor graph construction;
* runtime characteristics;
* memory usage characteristics;
* expression validation.

Workspace loading and model creation are intentionally excluded from the timed section so that the benchmark isolates the cost of constructing the likelihood expression.

Validation checks include:

* successful construction of `model.log_prob`;
* expression is not `None`;
* expression type is `TensorVariable`;
* expression can proceed to compilation.

This benchmark does **not** include:

* likelihood compilation;
* likelihood evaluation;
* batched evaluation.

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

## log_prob Construction

### Command Line Arguments

| Argument | Description | Default |
|-----------|-------------|----------|
| `--workspaces` | One or more HS3 workspace JSON files to benchmark. | `simple_workspace_nonp.json` |
| `--targets` | One or more likelihood targets for log-probability construction. | `L_ch0` |
| `--modes` | PyTensor compilation modes passed to `workspace.model(...)`. | `FAST_RUN` |
| `--n-runs` | Number of repeated benchmark runs per target. | `5` |
| `--output-dir` | Directory where benchmark JSON results will be stored. | `benchmarking/results/log_prob_construction` |
| `--output-name` | Name of the benchmark JSON output file. | `log_prob_construction_result.json` |
| `--plot` | Generate comparison plots for wall time and memory usage. Requires at least two benchmark results. | Disabled |
| `--plot-dir` | Directory where generated plots will be stored. | `benchmarking/plots/log_prob_construction` |

```bash
pixi run python benchmarking/src/run_log_prob_construction.py
```

```bash
pixi run python benchmarking/src/run_log_prob_construction.py \
  --n-runs 20
```

```bash
pixi run python benchmarking/src/run_log_prob_construction.py \
  --workspaces \
  benchmarking/inputs/simple_workspace_nonp.json \
  benchmarking/inputs/simple_workspace.json
```

```bash
pixi run python benchmarking/src/run_log_prob_construction.py \
  --targets L_ch0 L_ch1 L_ch2
```

```bash
pixi run python benchmarking/src/run_log_prob_construction.py \
  --workspaces \
  benchmarking/inputs/simple_workspace_nonp.json \
  benchmarking/inputs/simple_workspace.json \
  --targets L_ch0 L_ch1 L_ch2 \
  --plot
```

## Benchmark Overview Plots

The `plot_benchmark_overview.py` script creates combined overview plots from existing benchmark JSON outputs. Unlike individual benchmark plots, this script compares multiple pyHS3 workflow stages in a single stacked plot.

It is useful for understanding how much each stage contributes to the total runtime and memory usage.

Currently, the overview plot can combine:

* workspace loading;
* model creation;
* log_prob construction.

Measures:

```text
HS3 workspace JSON
        ↓
Workspace loading
        ↓
Model creation
        ↓
log_prob construction
```

### Command Line Arguments

| Argument | Description | Default |
|-----------|-------------|----------|
| `--workspace-loading-result` | Path to the workspace loading benchmark JSON result. | Required |
| `--model-creation-result` | Path to the model creation benchmark JSON result. | Required |
| `--log-prob-construction-result` | Path to the log_prob construction benchmark JSON result. | Required |
| `--output-dir` | Directory where overview plots will be stored. | `benchmarking/plots/overview` |
| `--memory-metric` | Memory metric used for the overview memory plot. | `peak_rss_delta_mb` |

### Running

```bash
pixi run python benchmarking/src/plot_benchmark_overview.py \
  --workspace-loading-result benchmarking/results/workspace_loading/workspace_loading_result.json \
  --model-creation-result benchmarking/results/model_creation/model_creation_result.json \
  --log-prob-construction-result benchmarking/results/log_prob_construction/log_prob_construction_result.json
```

This creates overview plots under:

```text
benchmarking/plots/overview/
├── benchmark_wall_time_overview.png
└── benchmark_rss_delta_overview.png
```

The wall-time overview plot uses stacked bars to show the cumulative cost of the measured workflow stages for each workspace. The memory overview plot uses the selected RSS delta metric and is intended to provide a high-level comparison of memory usage across stages.

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

benchmarking/results/log_prob_construction/
└── log_prob_construction_result.json
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

benchmarking/plots/log_prob_construction/
└── log_prob_construction_wall_time.png
```

Memory plots are generated only when the measured memory metric contains non-zero values. This avoids producing empty plots for benchmarks where memory changes are below the measurement resolution.
