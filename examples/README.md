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
