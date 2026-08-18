# Persona: Core Contributor

**Context.** Maintains or extends pyhs3 itself — adds features, fixes bugs,
reviews PRs, touches CI. Knows Python well; is building or refreshing a mental
model of pyhs3's internal architecture (module layout, PyTensor graph
construction, the HistFactory modifier system).

**Scope.** The Contributing toctree: `docs/architecture.rst`,
`docs/development.rst`, `docs/testing.rst`, `docs/CONTRIBUTING.rst`. User-facing
Tutorial/How-to content is out of this persona's scope — already knows how to
use pyhs3 as a library.

**Goals.** Understand how pyhs3's pieces fit together well enough to make a
change without breaking an invariant; know the dev workflow (pixi environments,
pre-commit, test markers, how to run the slow real-world suite) well enough to
follow it without asking.

**Reading style.** Reads Explanation for architecture and design rationale;
reads Reference for exact commands and file layout; skips anything user-facing.

**Pain points.** Architecture docs describing a module layout that's since been
refactored (e.g. a stale reference to a monolithic module that has since been
split); missing rationale for a non-obvious design choice (why normalization
defaults to Gauss-Legendre quadrature, why observables auto-vectorize); a
documented dev command that no longer matches what the project's environment
config actually defines.

**Lens.** If I made the change this document describes, would I break an
invariant the docs never warned me about?

**Sources checked.** `docs/architecture.rst`, the project's environment/build
config, `docs/CONTRIBUTING.rst`, and the actual module layout under
`src/pyhs3/`.
