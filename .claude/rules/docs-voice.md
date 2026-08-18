# Docs voice rules — pyhs3

Editorial rules for `docs/*.rst`, applied by every persona reviewer and by the
synthesis step of a `write-doc` run. These describe how content should read, not
what it should say — the Diátaxis rules and the profile's rubric govern content;
this file governs prose and code style.

## Naming

Write the package name **`pyhs3`**, lowercase, everywhere — in prose, headings,
and titles. Some existing docs write `PyHS3`; treat that as a defect to fix when
touching a unit, not a style to match. Refer to the spec it implements as
**HS3**, and never as the package name.

## Address and voice

Second person, active voice: "you build a model," not "one builds a model" or "a
model is built." Avoid the passive voice specifically where it hides who does
the acting — "the workspace validates the JSON" beats "the JSON is validated."
Never use "we" to mean the pyhs3 maintainers or the reader collectively; each
sentence's actor is either the reader ("you") or the code ("the `Model`
compiles...").

## Tone

State facts and instructions directly. No promotional language ("powerful,"
"seamless," "simply," "just") and no hedging ("should," "typically," "in most
cases") where the actual behavior is knowable and unconditional — if behavior
really is conditional, state the condition instead of hedging around it. Don't
oversell what a feature does; don't undersell a real limitation by burying it in
a subordinate clause.

No filler transitions ("It's worth noting that...", "Importantly,..."). No
inflated symbolism or grandiosity about pyhs3's role ("revolutionizes,"
"transforms how physicists...") — it's a library that does a specific job.

## Code examples

- Use `.. doctest::` blocks with the `>>>` REPL style already used throughout
  `docs/*.rst`, matching `README.rst`'s Hello World examples — every example
  must actually run (`pixi run docs-build` executes doctests; an example that
  renders but was never checked does not pass).
- Prefer Scikit-HEP ecosystem tools in examples (`awkward`, `hist`,
  `boost-histogram`, `pyhf`, `uproot`, `vector`) over generic alternatives when
  a choice exists.
- Comment only the non-obvious step in an example, not every line — match the
  existing pattern of one inline `#` comment marking a step, not narrating each
  line of output.
- State array shapes and broadcasting behavior explicitly in prose or a comment
  wherever an example's inputs or outputs aren't scalars — never leave a shape
  implicit for the reader to infer from output.

## Structure

- Use `.. warning::` and `.. note::` admonitions sparingly, for a genuine
  correctness hazard or a fact a reader is likely to miss — not as a substitute
  for stating the fact in prose.
- Bold (`**term**`) a term only the first time a bullet list defines it, not on
  every subsequent mention.
- Prefer double backticks (` ``like_this`` `) for identifiers, file paths, and
  literal values, matching existing usage.
- No section restates a fact already stated in a Reference unit — link to it
  instead (see the profile's Extra guidelines).

## Things to avoid

- Em dash overuse — prefer a period, a comma, or a colon.
- The "rule of three" list pattern used as a rhetorical crutch (three
  adjectives, three examples) where one precise word or a real, non-padded list
  would do.
- Vague attribution ("many users report...", "it is well known that...") for a
  claim that needs either a citation (the HS3 spec, a test, a benchmark) or
  should not be made at all.
