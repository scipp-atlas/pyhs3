# Scripts for the HistFactory performance note

Reproduces every number in
[`../2026-11-16-histfactory-performance-lessons.md`](../2026-11-16-histfactory-performance-lessons.md).
Run all commands from the repository root.

## Inputs

- `aheld/hww_ws_fixed.json` and `aheld/workspace.json`: the HWW workspace in pyhf and HS3 format
  (not in the repository).
- `ENGINE_DIR`: a directory holding the histfactory_js engine files `numerics.js`, `workspace.js`,
  `model.js`, `minimize.js` and `infer.js` (from the page's `js/` folder; not redistributed here).
- Node 24 for the `.mjs` harnesses; the `py312` pixi environment (`py312-jax` for the JAX backend).

## Commands

```bash
OUT=/path/to/scratch
D=docs/plans/2026-11-16-histfactory-performance

# one seeded Poisson toy, written into both formats
pixi run -e py312 python $D/make_toy.py aheld/hww_ws_fixed.json aheld/workspace.json $OUT

# histfactory_js: calls by phase, and a warm-started 21-point POI scan
node $D/js_fit_phases.mjs $ENGINE_DIR $OUT/hww_toy_pyhf.json
# seed-Hessian check, starting-diagonal and finite-difference-gradient ablations (about 3 min)
node $D/js_ablations.mjs $ENGINE_DIR $OUT/hww_toy_pyhf.json
# SciPy and iminuit on the identical JS likelihood (about 5 min)
pixi run -e py312 python $D/drive_js_optimizers.py $ENGINE_DIR $OUT/hww_toy_pyhf.json

# the JS model as arrays, plus reference values; then the PyTensor kernel on them
node $D/js_dump_model.mjs $ENGINE_DIR $OUT/hww_toy_pyhf.json $OUT/hww_toy_coo.json
pixi run -e py312 python $D/bench_kernel.py $OUT/hww_toy_coo.json FAST_RUN,NUMBA prod
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 pixi run -e py312-jax python $D/bench_kernel.py $OUT/hww_toy_coo.json JAX unrolled
# kernel + ported BFGS fit, covariance by gradient differences and by autodiff (about 4 min)
pixi run -e py312 python $D/fit_kernel.py $OUT/hww_toy_coo.json

# pyhs3 as it stands (on HWW the compile does not finish within an hour)
pixi run -e py312 python $D/bench_pyhs3.py $OUT/hww_toy_hs3.json simPdf_obsData $OUT
pixi run -e py312 pyhs3 nll $OUT/hww_toy_hs3.json --analysis simPdf_obsData
```

## Files

| file                     | what it does                                                         |
| ------------------------ | -------------------------------------------------------------------- |
| `make_toy.py`            | seeded Poisson toy written into the pyhf and HS3 files               |
| `js_engine.mjs`          | shared loader for the JS engine files                                |
| `js_fit_phases.mjs`      | one JS fit with calls split by phase; warm-started scan              |
| `js_ablations.mjs`       | seed-Hessian check; starting diagonal; finite-difference gradient    |
| `js_serve.mjs`           | the JS likelihood over stdin/stdout                                  |
| `js_bridge.py`           | Python client for `js_serve.mjs`                                     |
| `drive_js_optimizers.py` | L-BFGS-B and MIGRAD variants on the JS likelihood                    |
| `js_dump_model.mjs`      | the JS model as index arrays, with reference 2NLL and gradients      |
| `hf_kernel.py`           | vectorised HistFactory 2NLL as a fixed-size PyTensor graph           |
| `bench_kernel.py`        | compile and time the kernel per backend; compare with the JS         |
| `bfgs.py`                | NumPy port of the JS dense BFGS; gradient-difference Hessian         |
| `fit_kernel.py`          | kernel + BFGS fit; covariance two ways                               |
| `bench_pyhs3.py`         | pyhs3 phase timings: build, graph size, compile, L-BFGS-B fit        |
