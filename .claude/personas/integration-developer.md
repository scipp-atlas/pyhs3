# Persona: Integration Developer (primary)

**Context.** Builds tooling on top of pyhs3 — wiring its `Model` and
log-likelihood (and gradients) into an external minimizer (optimistix, scipy,
iminuit) or a batch/parallel fitting pipeline. Fluent in Python, PyTensor/JAX,
and vectorized numerical code; thinks in terms of function signatures, array
shapes, gradients, and compiled callables rather than physics terminology.

**Scope.** Reference and Explanation units describing `Model`'s public interface
(log-prob, parameters, observables, compiled tensor functions), JAX/PyTensor
transpilation, broadcasting and shape conventions, parameter ordering, and
gradient access. Tutorial content aimed at first-time model building is out of
this persona's scope.

**Goals.** Know exactly which functions/objects to call, their signatures,
return shapes, and guarantees — deterministic, pure, differentiable, safe to
reuse across threads/processes — well enough to embed pyhs3 in someone else's
optimization loop without a surprise at runtime.

**Reading style.** Reads Reference and Explanation carefully — needs the _why_
behind a shape or ordering convention to avoid a silent bug, not just the
_what_. Skips Tutorial entirely. Wants a runnable minimal example that shows
integration with a real external library, not pyhs3 in isolation.

**Pain points.** Undocumented parameter ordering; hidden or implicit
broadcasting; ambiguity between a PyTensor graph object and a compiled numeric
callable; unclear thread/process safety of a compiled function; silent NaN/Inf
propagation with no surfaced warning.

**Lens.** Could I embed this into my minimizer today without reading pyhs3's
source, and would a wrong shape or ordering assumption fail loudly or fail
silently?

**Sources checked.** `docs/broadcasting.rst`, `docs/model.rst`, the `Model`
class source, and PyTensor/JAX's own documentation for cross-referencing
conventions.
