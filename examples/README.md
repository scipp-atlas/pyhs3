# pyhs3 Examples

> **Branch note:** This directory exists only on the `profile-scan-script`
> branch and is **not** part of any release. It is a living sandbox — commits
> accumulate here; no PR to `main` is ever opened from this branch.

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

### Install

```bash
pip install "pyhs3[jax]" optimistix matplotlib skhep-testdata
```

`pyhs3[jax]` pulls in `pytensor[jax]` which transitively installs JAX.
`optimistix` is **not** a pyhs3 dependency — it is only needed to run this
script.

### Run

```bash
python examples/profile_scan_atlas_dihiggs.py
```

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
