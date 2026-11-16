# HistFactory performance: lessons from histfactory_js

Date: 2026-11-16. Scripts and exact commands:
[`2026-11-16-histfactory-performance/`](2026-11-16-histfactory-performance/README.md).

## Question

Alex Held's histfactory_js fits a large ATLAS HistFactory workspace in the browser in about a
second, while pyhf with JAX and Minuit takes minutes and pyhs3 does not finish compiling. Where does
the difference come from, and what carries over to pyhs3?

Following Lukas Heinrich's framing, the cost of a fit splits into three factors:

1. the cost of one likelihood evaluation,
2. how many evaluations the gradient method needs per iteration (analytic or autodiff gradient: about
   one; finite differences: 2n), and
3. how many iterations the optimiser needs.

This note measures each factor separately.

## Setup

- **Model.** The HWW workspace in `aheld/`, which ships in both formats: `hww_ws_fixed.json` (pyhf)
  and `workspace.json` (HS3, exported by ROOT 6.40). It has 55 channels, 121 bins, 1,716
  (channel, sample) blocks, 3,778 (sample, bin) cells, and 49,737 normsys, 1,194 normfactor, 616
  staterror and 450 histosys modifier entries.
- **Data.** Both files carry blinded Asimov data, so a fit from the nominal values starts at the
  minimum (the JS converges in 0 iterations). `make_toy.py` writes one seeded Poisson toy of the main
  measurement into both files.
- **Engines.** histfactory_js runs under Node 24 with the page's own engine files. To compare
  optimisers without any other difference, `js_serve.mjs` exposes the JS likelihood over a pipe and
  SciPy and iminuit drive the identical function (`drive_js_optimizers.py`). pyhs3 is timed as it
  stands (`bench_pyhs3.py`, `pyhs3 nll`).
- **Machine.** Apple silicon laptop, CPU only. Call counts and iteration counts are exact and
  reproducible; wall times vary by about 20% between runs.

**The two formats are not quite the same model.** ROOT holds `Lumi` and 100 `gamma_stat_*`
parameters constant (they are outside the analysis domains). The pyhf file fits those 100 gammas;
pyhs3 exposes all 101 as free inputs. The kernel comparisons below use the JS model's arrays, so
they are exact comparisons; pyhs3 numbers are timings only.

## Results

### Where the JS fit spends its calls

| phase (`infer.fit`)                         |     calls |  time |
| ------------------------------------------- | --------: | ----: |
| seed Hessian at the start (forward diff.)   |       643 | 0.3 s |
| BFGS minimisation (84 iterations)           |       279 | 0.3 s |
| covariance Hessian at the minimum (central) |     1,280 | 0.5 s |
| Newton polish                               |         3 |       |
| **total**                                   | **2,205** | 1.2 s |

One value plus gradient costs 0.4 to 0.85 ms. The seed Hessian at the start is not positive definite
on this workspace, so `fit()` discards it and BFGS starts from its diagonal fallback: 25% of the fit
time buys nothing. The covariance is the most expensive part of the fit, not the minimisation.

A 21-point profile scan of the POI, warm-started from the fit, costs 236 calls in total (about 11 per
point): each conditional fit starts from a linear prediction along the covariance and from the
conditional inverse Hessian.

### The optimiser, on the identical function

| optimiser                                          | iterations | calls     | converged     |
| -------------------------------------------------- | ---------: | --------: | ------------- |
| JS dense BFGS, start `diag(0.5 sigma_prefit^2)`     |         84 |       279 | yes           |
| JS dense BFGS, start identity                      |        259 |       894 | yes           |
| SciPy L-BFGS-B, m=10, raw parameters               |    20,000  |   21,409  | no (limit)    |
| SciPy L-BFGS-B, m=10, sigma-scaled                 |    10,261  |   11,205  | yes           |
| SciPy L-BFGS-B, m=30, sigma-scaled                 |     5,450  |    5,863  | yes           |
| SciPy L-BFGS-B, m=100, sigma-scaled                |       865  |      914  | yes           |
| SciPy L-BFGS-B, m=300, sigma-scaled                |       365  |      385  | yes           |
| MIGRAD strategy 0, analytic gradient               | 72 grads   |    1,815  | yes           |
| MIGRAD strategy 1, analytic gradient               | 76 grads   |  422,277  | yes (208 s)   |
| MIGRAD strategy 0, Minuit's numerical gradient     |            |   94,209  | yes (48 s)    |

The likelihood is badly conditioned (strongly correlated normalisation factors). A limited-memory
quasi-Newton method with a handful of correction pairs cannot represent that curvature; the dense
642 x 642 inverse Hessian can. Scaling each parameter by its prefit width removes a further factor
of three.

The dense matrix is not free. Each iteration does several O(n^2) operations on it (the search
direction, the rank-two update, the projection of active bounds), on the order of a million
floating-point operations at n = 642, which is comparable to one likelihood call. In the NumPy port
below the minimiser spends about 0.4 s of its 0.9 s outside the likelihood. histopresto's README
states the same limit: the dense update dominates above a few hundred parameters. At larger n the
trade-off between iterations saved and O(n^2) work per iteration has to be measured, for example
against L-BFGS with a large memory (m = 300 needed 365 iterations here).

MIGRAD itself is not the problem: with the analytic gradient and strategy 0 it needs about as many
gradient iterations as the JS BFGS (72 against 84) and lands within 0.002 of the same 2NLL. Its
extra calls are its numerical seed of the Hessian diagonal (about 2n). The slow configurations in
Alex's pyhf table come from the numerical gradient (each gradient costs 2n values) and from strategy
1, which spends 422,000 value calls on numerical second derivatives even with the analytic gradient.

Replacing the analytic gradient by central differences in the JS BFGS leaves the path unchanged (84
iterations) but multiplies the cost of every gradient by 2n + 1: 358,515 value calls and 108 s
instead of 279 calls and 0.28 s.

### pyhs3 as it stands

| phase                                  | result                                |
| -------------------------------------- | ------------------------------------- |
| `Workspace.load`                       | 7 s                                   |
| `ws.model(analysis)` (graph build)     | 265 s                                 |
| graph size (`log_prob`)                | 1,879,256 apply nodes                 |
| compile of the value function          | not finished after 51 min (stopped)   |
| `pyhs3 nll` (build, compile, one call) | not finished after 60 min (timeout)   |

The graph holds the model's structure. `interpolate_code4` builds about 30 nodes per normsys entry,
including the solve for the six polynomial coefficients from the constant `hi`/`lo` values, so the
49,737 normsys entries alone account for about 1.5 million nodes. PyTensor's rewrites then
constant-fold them one by one.

### The same model as data: a vectorised kernel

`hf_kernel.py` builds the 2NLL from the arrays the JS uses (index arrays, precomputed code4
coefficients, per-cell nominal yields) with gather, a polynomial, scatter-add and padded products.
The graph size does not depend on the number of modifiers.

| backend                  | raw nodes | compile | value + gradient | max. rel. error vs JS     |
| ------------------------ | --------: | ------: | ---------------: | ------------------------- |
| C (`FAST_RUN`)           |       315 |  0.6 s  |          1.9 ms  | 3e-11 value, 4e-12 grad   |
| Numba                    |       315 |  0.2 s  |          1.9 ms  | 3e-11 value, 4e-12 grad   |
| C, unrolled products     |       953 |  2.2 s  |          2.1 ms  | 3e-11 value, 6e-12 grad   |
| JAX, unrolled products   |       953 |  2.1 s  |          1.3 ms  | 3e-11 value, 4e-13 grad   |

The JS evaluates the same model in 0.4 to 0.85 ms. JAX needs the unrolled products because of the
PyTensor gaps listed under Implications.

With a NumPy port of the JS BFGS (`bfgs.py`), the Python fit follows the JS path exactly (84
iterations, 279 calls, 2NLL 1021.859467, POI 0.69776 +- 0.65582) in 0.9 s; the covariance from
central differences of the gradient takes 2.8 s (1,280 calls). End to end, including compilation:
about 4.7 s against the JS's 1.2 to 1.6 s.

The covariance from PyTensor's autodiff Hessian agrees to 7e-10 but costs 115 s to compile and 3.3 s
to evaluate (`pytensor.gradient.hessian` builds a scan over 642 reverse passes).

## Lessons

1. **Keep the model's size in data, not in the graph.** Compile time grows with graph nodes;
   evaluation time grows with the number of nonzero entries. A HistFactory model is a sparse set of
   index arrays and coefficients evaluated by one fixed kernel. Building one subgraph per modifier
   turns a 3,778-cell model into a 1.9-million-node graph. This is also the root of what PR #287
   works around: constants baked into the graph prevent sharing compiled functions.
2. **Precompute what depends only on constants.** Interpolation coefficients, log ratios, log
   factorials of the data and constraint normalisations are numbers, computed once in NumPy; leaving
   them to the compiler's constant folding costs build and rewrite time proportional to the model.
3. **The optimiser matters as much as the evaluation speed.** On a badly conditioned likelihood a
   dense quasi-Newton method with prefit-width scaling needs 84 iterations where L-BFGS with default
   memory does not converge in 20,000. A numerical gradient multiplies the cost of every iteration
   by 2n, and Minuit's strategy 1 adds numerical second derivatives on top.
4. **Gradients by autodiff, Hessians by differences of the gradient.** 2k gradient calls cost
   2.8 s; the autodiff Hessian gives the same matrix but takes 115 s to compile.
5. **Reuse what the last fit learned.** Warm starts from the covariance make each profile-scan point
   cost about 11 calls.
6. **Guard the domain.** Where the expected yields are not positive the 2NLL is undefined. The JS
   backtracks out of that region; SciPy's L-BFGS-B stops (`ABNORMAL`) unless the objective returns a
   large finite value there, even on the two-bin `simplemodel` test workspace.

## Implications for pyhs3

In order of payoff:

1. **Vectorise `HistFactoryDist`'s expression** (issue #126). Build the index and coefficient arrays
   in NumPy at model construction and emit a fixed-size graph per channel. This fits the existing
   PyTensor architecture and addresses the build and compile time directly. Every interpolation code
   pyhs3 supports (`code0`, `code1`, `code2`, `code4`, `code4p`) needs a vectorised form; the
   prototype covers `code4` normsys and `code4p` histosys only.
2. **Offer a fit utility**: dense BFGS from prefit widths (from the constraint terms), projected
   bounds, backtracking out of the undefined region, covariance from differences of the gradient,
   warm-started scans. At minimum, document "pass the gradient to Minuit and use strategy 0".
3. **Pass parameters as one vector** rather than one scalar input per parameter, which is what
   optimisers work with. The prototype does this; the per-call overhead of pyhs3's 643 keyword
   arguments was not measured here.
4. **Report upstream (PyTensor):** the JAX backend cannot differentiate `pt.prod` (it dispatches to
   `jax.lax.mul_without_zeros`, which the JAX installed in `py312-jax` does not provide); the
   gradient of `pt.concatenate`
   lowers to a `Split` that fails under JAX with traced sizes; `pt.prod` has no second derivative.
5. **Decide the semantics of parameters outside the analysis domains.** pyhs3 makes them free
   inputs; ROOT holds them constant. On this workspace that is 101 parameters, and on the
   `simplemodel` test workspace a free `Lumi` makes the fit degenerate (POI runs to its bound).

For histfactory_js: the seed Hessian at the starting point is not positive definite on this
workspace and costs a quarter of the fit. histopresto's README already lists this as a known
limitation ("the case on all larger models tried").

## The published code: histopresto

The engine is published as [histopresto](https://github.com/alexander-held/histopresto) (initial
commit `e64de97`). Its JavaScript core is algorithmically identical to the engine files measured
here (only the module packaging differs), so the JS numbers above apply to it.

The Python package parses the workspace and builds the same flat arrays in NumPy, then hands them
to a C kernel (`kernel.c`, loaded with ctypes) that is a statement-by-statement port of the JS:
`hf_nll` (value and analytic gradient), `hf_minimize` (the dense BFGS) and `hf_hessian` (gradient
differences) all run in C, so no Python is involved per likelihood call or per iteration. Alex
reports about a further factor of two over the JS; not measured here.

That is the other end of the design space from item 1 below. histopresto hand-writes the model,
its gradient and the optimiser for the HistFactory modifier set; pyhs3 has to evaluate arbitrary HS3
graphs, so it keeps a compiled graph and gets gradients from autodiff. The vectorised kernel shows
that the graph approach reaches the same per-call order of magnitude (1.3 to 1.9 ms against 0.4 to
0.85 ms) once the model's structure is data, leaving the optimiser loop as the remaining Python
cost.

## Caveats

- One workspace and one toy. The conditioning, and so the optimiser comparison, depends on the model.
- The kernel is validated against the JS model built from the pyhf file, not against pyhs3's model
  from the HS3 file; the two differ in which parameters float (see Setup).
- `bench_pyhs3.py` replaces 643 graph inputs by one vector with `graph_replace` before compiling, so
  its compile timing includes that replacement; the `pyhs3 nll` timing does not.
- The kernel uses a single global graph. Per-channel kernels would add some tens of nodes per
  channel; not measured.
