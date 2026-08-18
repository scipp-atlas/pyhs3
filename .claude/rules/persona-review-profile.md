# Persona review profile — pyhs3 docs

## Document

The document is the pyhs3 Sphinx site under `docs/`, treated as one document
whose section set is an output of the run: files may be split, merged, or newly
created as content is reorganized into Diátaxis quadrants.

In scope: the main toctree — the "building and running a model" pilot's current
section set (`getting_started`, `howto_load_workspace`, `howto_evaluate_model`,
`howto_fit_model`, `howto_debug_model`, `workspace_reference`,
`model_reference`, `data_reference`, `building_a_model`) plus the
not-yet-reorganized `broadcasting`, `defining_components`, `normalization`,
`visualization`, `api` — and the Contributing toctree (`CONTRIBUTING`,
`development`, `testing`, `architecture`).

Out of scope: the Academics toctree (`acknowledgements`, `abstracts`, `talks`)
and `CODE_OF_CONDUCT` — archival and governance content, not
Diátaxis-classifiable. A run never restructures a document outside its scope.

## Personas

Five persona head files under `.claude/personas/`:

- `model-builder.md` — **primary**. A physicist building and running statistical
  models with pyhs3 day to day.
- `integration-developer.md` — **primary**. Wires pyhs3's `Model`/likelihood
  into external minimizers (optimistix, scipy, iminuit) or other libraries.
- `core-contributor.md` — extends pyhs3's internals; reads architecture,
  testing, and development docs.
- `component-extender.md` — writes new `Distribution`/`Function`/`Axis`
  subclasses to extend pyhs3 for their own analysis.
- `numerical-validator.md` — checks numerical agreement between pyhs3 and
  ROOT/HistFactory/RooFit.

The two personas marked primary above have their verdicts outweigh the other
three's when fixes conflict. Every run launches all five.

## Declaration mechanism

Marker syntax, unit granularity, and document shapes are defined in
`.claude/rules/diataxis-declaration.md`. Out-of-quadrant content with no home in
its current document is routed to a new same-file section per the Diátaxis
rules; content whose true home is a different file is reported as a suggested
cross-document move, never auto-relocated.

## Premise to pin

None. No single fact or design decision needs pinning before drafting begins.

## Sources

- The pyhs3 source tree (`src/pyhs3/`): docstrings, type hints, and behavior.
- `README.rst`'s Hello World examples.
- The existing `docs/*.rst` content being reorganized — read before removing.
- `tests/`, as executable ground truth for behavior.
- The public HS3 spec (cite by URL — never the local untracked clone of
  `hep-statistics-serialization-standard/`).
- pyhf's public docs (cite by URL) for prior-art framing only — never as an
  authority on pyhs3's own behavior.

## Fact-check targets

- Code/API claims (signatures, defaults, return types, parameter names):
  verified against the actual source.
- Behavioral claims (numerical results, broadcasting shapes, normalization):
  verified against `tests/`, and by actually running the example where feasible.
- Spec-compliance claims: verified against the public HS3 spec text; a claim
  citing no spec section is flagged unverifiable, not assumed correct.
- Checking note: doctests must actually execute
  (`JAX_PLATFORMS=cpu pixi run --environment py312-jax test-docs` runs them —
  `pixi run docs-build` only builds HTML) — a code block that renders but was
  never run does not pass. An example using a library that isn't a pyhs3
  dependency (e.g. optimistix) is a `code-block` rather than a `doctest`, since
  no environment has it installed to run against; verify it by hand before
  shipping instead.

## Status dimension

Enabled. "Implemented" units are verified against current behavior per
Fact-check targets. "Spec" units describe intended-but-not-yet-built behavior
(e.g., the planned JAX transpile path or NLL rewrite) and are verified against a
design brief supplied at invocation, plus implementability on the platform:
**pyhs3's PyTensor-based computation-graph architecture** (as described in
`docs/architecture.rst`). A spec claim requiring something that architecture
cannot express is a blocking defect.

## Verification

One-time wiring: `reviewed-writer` pinned in `.claude/settings.json`
(`extraKnownMarketplaces` + `enabledPlugins`); the profile, declaration file,
persona files, and voice-rules file committed.

Re-run every round:
`JAX_PLATFORMS=cpu pixi run --environment py312-jax test-docs` must succeed
(this repo's actual doctest runner — `pytest README.rst docs`, only executes
literal `>>>` lines; `.. testsetup::` blocks are a Sphinx-only construct this
runner ignores entirely, so setup code must be written as real `>>>` examples).
`pixi run --environment docs docs-build -W` must also succeed with zero warnings
(HTML build only — it does not run doctests, despite the two often being
conflated). `pre-commit run --files <changed docs>` must pass; note markers use
a single colon (`.. diataxis: how-to`, not `.. diataxis:: how-to`) since the
double-colon form is real RST directive-invocation syntax and fails
`rstcheck`/`sphinx -W` on the unregistered directive name.

## Record

No separate run log. The record is the git history: each shipped revision lands
as a normal commit with a Conventional Commits message (`docs: ...`) describing
the driving change, following this repository's existing commit conventions.

## Voice rules

`.claude/rules/docs-voice.md`.

## Extra guidelines

Don't duplicate content across quadrants — link instead of restating. A
Reference unit is the single source of truth for any parameter or default also
mentioned in a How-to or Explanation unit.
