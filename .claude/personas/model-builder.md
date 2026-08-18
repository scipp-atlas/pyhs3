# Persona: Model Builder (primary)

**Context.** A physicist (e.g. an ATLAS/CMS analyzer) who needs a statistical
model to produce a physics result — a fit, a limit, a p-value — starting either
from an HS3 JSON workspace (often ported from a RooFit/HistFactory/combine
export) or by constructing a model directly from pyhs3 Python objects.
Comfortable with Python and fluent in ROOT/RooFit/HistFactory concepts and
terminology; not necessarily comfortable with PyTensor internals, computation
graphs, or software architecture. Using pyhs3 to get an analysis result, not to
learn pyhs3 for its own sake.

**Scope.** Tutorial, How-to, and Reference units aimed at building a model and
getting a number out of it: loading a workspace, constructing parameters and
observables, evaluating a likelihood, running a fit, reading off a result.
Architecture, contribution workflow, and custom-component authoring are out of
this persona's scope.

**Goals.** Get a working statistical model as fast as possible, with confidence
that a copied example actually runs as shown; find the exact API/parameter
needed for the task at hand without reading pyhs3's source.

**Reading style.** Follows a Tutorial start to finish exactly once, in order.
Otherwise skims: jumps straight to the How-to or Reference section matching the
task, copy-pastes the example, and only reads surrounding prose if the example
doesn't work.

**Pain points.** Not knowing whether pyhs3 supports a construct their existing
workspace uses (a specific HistFactory modifier, an interpolation code);
ROOT/RooFit terminology that doesn't map cleanly onto pyhs3's naming; an example
that renders but doesn't actually run; unclear guidance on how observable
bounds/domains are supposed to be set.

**Lens.** Can I go from "I have an HS3 JSON (or want to build one from scratch)"
to "I have a trustworthy number" without reading pyhs3's source code?

**Sources checked.** `README.rst`'s Hello World examples; any existing
getting-started material under `docs/`; pyhf's public docs, only as a frame of
reference for concepts a HEP physicist likely already knows.
