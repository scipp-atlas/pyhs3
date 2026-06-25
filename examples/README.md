# pyhs3 Examples

> **Branch note:** This directory exists only on the `profile-scan-script`
> branch and is **not** part of any release. It is a living sandbox — commits
> accumulate here; no PR to `main` is ever opened from this branch.

## `minimization_dihiggs.py`

Profile-likelihood minimization demo: compile `model.log_prob` into a pytensor
function and minimize over nuisance parameters at a fixed `mu_HH` using scipy.

### What it demonstrates

| Feature                                              | API used                                |
| ---------------------------------------------------- | --------------------------------------- |
| Build a joint symbolic NLL from a workspace Analysis | `ws.model(analysis)` → `model.log_prob` |
| Compile the PyTensor expression                      | `pytensor.compile.function.function`    |
| Extract symbolic free inputs                         | `explicit_graph_inputs([expression])`   |
| Profile minimization with scipy                      | `scipy.optimize.minimize` (SLSQP)       |

### Run

```bash
pixi run -e py312 python examples/minimization_dihiggs.py
```

The script profiles the NLL at a single `mu_HH = 1.0` value. A commented-out
section shows how to loop over a full `MU_GRID` scan.

---

## `nll_validation_dihiggs.py`

NLL validation against the ROOT reference for the ATLAS bbyy workspace (pyhs3
issue #41). Demonstrates **two approaches** to computing an NLL scan over a
`mu_HH` grid:

1. **Scalar (non-batched)** — compile `model.log_prob` with scalar `mu_HH`, loop
   over the scan grid evaluating one point at a time.
2. **Vectorized (batched)** — set `param_set["mu_HH"].kind = pt.vector` before
   building the model, pass the entire grid in a single compiled function call.

### What it demonstrates

| Feature                                              | API used                                 |
| ---------------------------------------------------- | ---------------------------------------- |
| Build a joint symbolic NLL from a workspace Analysis | `ws.model(analysis)` → `model.log_prob`  |
| Compile the PyTensor expression                      | `pytensor.compile.function.function`     |
| Scalar NLL evaluation (one point at a time)          | loop with `np.asarray` wrapping per call |
| Batched NLL evaluation (all points at once)          | `param_set["mu_HH"].kind = pt.vector`    |
| Comparison against ROOT reference values             | embedded `_REFERENCE` dict               |

### Run

```bash
pixi run -e py312 python examples/nll_validation_dihiggs.py
```

The script builds two models (one scalar, one batched — each cached separately
as `ws_scalar.pkl` / `ws_batched.pkl`), runs both scans, compares them against
each other and the ROOT reference, and writes `nll_validation.pdf` +
`nll_validation.json`.

---

## `profile_scan_atlas_dihiggs.py`

A profile-likelihood scan over the diHiggs signal-strength parameter `mu_HH` for
the ATLAS bbyy non-resonant analysis (pyhs3 issue #41 validation workspace).

### What it demonstrates

| Feature                                              | API used                                |
| ---------------------------------------------------- | --------------------------------------- |
| Build a joint symbolic NLL from a workspace Analysis | `ws.model(analysis)` → `model.log_prob` |
| Transpile the PyTensor expression to JAX             | `pyhs3.jaxify(nll_expr)`                |
| Profile scan with nuisance minimisation              | `optimistix.BFGS` / `optimistix.DFP`    |
| Comparison against ROOT reference values             | embedded `_REFERENCE` dict              |

### Run

```bash
pixi run -e py312-jax python -m pip install optimistix
pixi run -e py312-jax python examples/profile_scan_atlas_dihiggs.py
```

`optimistix` is **not** a pyhs3 dependency — it is only needed to run this
script.

The script:

1. Downloads / finds the validation workspace via `skhep-testdata`.
2. Builds the symbolic model and transpiles the NLL to JAX (~1–2 min first run).
3. Runs two 31-point profile scans (BFGS + DFP), printing per-point NLL values.
4. Writes `nll_profile_scan.pdf` showing ΔNLL vs the ROOT reference.

### Expected output

```
Loading workspace from test_hs3_unbinned_pyhs3_validation_issue41.json ...
Building model (this compiles the symbolic graph) ...
Transpiling NLL expression to JAX ...
  N symbolic inputs: ...
JIT warm-up done.

--- BFGS scan ---
  [BFGS] mu=-0.50  NLL=...  [ok]
  ...
BFGS: 31/31 converged in Xs (Yms/fit)

--- DFP scan ---
  ...

Max |ΔNLL(pyhs3 BFGS) - ΔNLL(ROOT)| = ...
Max |ΔNLL(pyhs3  DFP) - ΔNLL(ROOT)| = ...

Wrote nll_profile_scan.pdf
```

---

## `exploring_minimization.py`

Benchmark multiple minimizers on any HS3 workspace + POI scan, capturing per-fit
wall time, CPU time, and peak RSS. Designed for producing benchmark plots for
posters and talks.

### What it demonstrates

| Feature                                       | API / approach used                             |
| --------------------------------------------- | ----------------------------------------------- |
| Load any HS3 workspace from a CLI path        | `pyhs3.Workspace.load` + argparse               |
| Cache model + compiled log_prob in one pickle | `{"model": …, "log_prob": …, "input_names": …}` |
| Benchmark 5 minimizers × 4 tolerances         | SLSQP, L-BFGS-B, TNC, trust-constr, migrad      |
| Capture per-fit timing and memory             | `ResourceSampler` daemon thread + psutil        |
| Atomic checkpointing after every scan         | tempfile + `Path.replace()`                     |
| Resume an interrupted run                     | `--resume` flag skips already-completed scans   |
| Optional multiprocessing                      | `ProcessPoolExecutor` with spawn context        |

### Run (smoke test — fast)

```bash
pixi run -e minimize python examples/exploring_minimization.py \
  --workspace tests/test_histfactory/simplemodel_uncorrelated-background_hs3.json \
  --analysis simPdf_obsData \
  --poi mu \
  --poi-range 0.5 1.5 11 \
  --output benchmark_simplemodel.json
```

Completes in under a minute. On the second run, the combined cache
(`simplemodel_uncorrelated-background_hs3.cache.pkl`) is reloaded from disk —
both the model build and the compile step are skipped.

### Run (full benchmark — diHiggs workspace)

```bash
WORKSPACE=$(python -c '
from skhep_testdata import data_path
print(data_path("test_hs3_unbinned_pyhs3_validation_issue41.json"))
')

pixi run -e minimize python examples/exploring_minimization.py \
  --workspace "$WORKSPACE" \
  --analysis CombinedPdf_combData \
  --poi mu_HH \
  --parameter-points default_values,nominalGlobs,nominalNuis,unconditionalGlobs_muhat,unconditionalNuis_muhat,POI_muhat \
  --poi-range -0.5 2.5 31 \
  --output benchmark_dihiggs.json
```

The first run builds and compiles the model (~1–8 min depending on hardware) and
writes `test_hs3_unbinned_pyhs3_validation_issue41.cache.pkl`. Subsequent runs
load from the cache in seconds.

### Notes

- **Serial by default.** Results are checkpointed atomically after every scan. A
  Ctrl-C or per-fit crash never loses already-collected data. Use `--resume` to
  pick up where a previous run left off.
- **Optional `--processes N`.** Uses a `spawn`-context `ProcessPoolExecutor`.
  Each worker loads the combined cache once at startup — no recompilation.
  `fork` is unsafe with PyTensor's compiled C extensions, so `spawn` is
  required.
- **Cache is machine-local.** The pickle references the PyTensor compile cache
  directory (`~/.pytensor`). It does not transfer to another machine and will
  break if the compile cache is wiped. Treat `*.cache.pkl` as a per-machine
  build artefact, not a portable bundle.

Plot the resulting JSON with
`plot_comparisons.py --bundle benchmark_dihiggs.json` (see below).

---

## `plot_comparisons.py`

Consume a benchmark bundle produced by `exploring_minimization.py` and render a
2×2 poster-quality figure comparing minimizers.

### What it demonstrates

| Panel        | Content                                            |
| ------------ | -------------------------------------------------- |
| Top-left     | ΔNLL curves vs POI per `(method, tol)` combination |
| Top-right    | Per-fit wall + CPU time boxplots                   |
| Bottom-left  | Per-fit iteration count boxplots                   |
| Bottom-right | Per-fit peak RSS delta boxplots                    |

Colour is indexed by method; marker shape is indexed by tolerance. An optional
`--reference-json` overlays a dashed grey reference curve on the ΔNLL panel.

### Run

```bash
pixi run -e minimize python examples/plot_comparisons.py \
  --bundle benchmark_dihiggs.json \
  --output-pdf benchmark_dihiggs.pdf \
  --output-png benchmark_dihiggs.png \
  --output-json benchmark_dihiggs_summary.json
```

Restrict to a subset of methods or tolerances:

```bash
pixi run -e minimize python examples/plot_comparisons.py \
  --bundle benchmark_dihiggs.json \
  --filter-method SLSQP,migrad \
  --filter-tol 1e-3,1e-4 \
  --output-pdf filtered.pdf \
  --output-png filtered.png
```

Overlay a reference scan (format:
`{"points_scan_order": [{"poi": …, "nll": …}, …]}`):

```bash
pixi run -e minimize python examples/plot_comparisons.py \
  --bundle benchmark_dihiggs.json \
  --reference-json reference_nll.json \
  --output-pdf comparison.pdf \
  --output-png comparison.png
```
