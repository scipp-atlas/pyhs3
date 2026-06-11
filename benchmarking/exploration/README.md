# Benchmarking Scripts

Scripts for profiling and analyzing pyhs3 / PyTensor performance. They fall into
three groups: end-to-end PDF timing, Gaussian shape-variant microbenchmarks, and
a vectorized-vs-individual Poisson investigation.

## End-to-end timing

- **`timing_pdf_evals.py`** — Times PyHS3's Gaussian distribution PDF evaluation
  across compile modes (FAST_RUN, FAST_COMPILE, JAX) against a numba-stats
  `norm.pdf` baseline. Uses the pyhs3 API directly, 1000-point grid, 100k evaluations.

- **`timing_comparison.py`** — Head-to-head logpdf timing of ROOT/RooFit vs pyhs3
  in FAST_RUN, NUMBA, and JAX modes, using the `rf501_simultaneouspdf` example workspace.

## Gaussian microbenchmarks (pure PyTensor, input-shape variants)

All four time a Gaussian PDF across FAST_RUN/JAX/NUMBA modes vs numba-stats;
they differ only in the symbolic input shapes.

- **`pytensor_timing_pdf_evals.py`** — All three inputs (`x`, `mean`, `sigma`) as vectors.

- **`pytensor_timing_pdf_evals_scalar.py`** — All inputs as scalars (plus
  `trust_input=True`), measuring per-call overhead in the fully scalar case.

- **`pytensor_timing_pdf_evals_ricardoV94.py`** — Variant following ricardoV94's
  (PyTensor maintainer) suggestion: vector `x` with scalar `mean`/`sigma` and
  `trust_input=True` — the "natural" shape for PDF evaluation.

- **`pytensor_timing_pdf_evals_ricardoV94_batched.py`** — The inverse batching:
  scalar `x` with vector `mean`/`sigma` (1000 parameter values at once), testing
  batched-over-parameters evaluation.

## Graph analysis (ATLAS diHiggs bbyy workspace)

- **`analyze_graph_complexity.py`** — Reports symbolic Apply-node counts per
  distribution, optimizer reduction ratios on a sample of compiled graphs, and
  total function-call counts over an NLL scan.

- **`analyze_jaxpr.py`** — Counts JAXpr equations for the NLL after baking
  observables and fixed parameters as constants, reporting free-input count and
  a top-20 primitive breakdown.

## Vectorized vs individual Poisson investigation

- **`debug_pytensor_poisson_optimization.py`** — Explores vectorized vs individual
  Poisson calculations: numerical equivalence, graph differences, and performance.

- **`debug_optimized_graphs.py`** — Prints before/after-optimization graphs
  (`debugprint`) comparing vectorized vs per-element Poisson product constructions
  on a small case.

- **`debug_scalability_limits.py`** — Stress-tests how the vectorized vs individual
  approaches scale (100–10000 dimensions), finding where compilation breaks down.

- **`pytensor_optimization_findings.md`** — Write-up of the Poisson experiments:
  all approaches numerically equivalent, individual ops hit C++ compilation limits
  at ~500 dims, and `exp(sum(log_probs))` is the most stable/efficient form.
