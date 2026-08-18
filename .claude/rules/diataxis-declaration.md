# Diátaxis declaration — pyhs3 docs

## Marker syntax

Every unit of content — a whole `.rst` file when the file serves one quadrant,
or an individual heading when a file mixes modes — carries its quadrant as a
true RST comment immediately above the unit's title:

```rst
.. diataxis: how-to
```

Legend: `tutorial`, `how-to`, `reference`, `explanation`. This is a single
colon, not a directive invocation: RST only parses `.. name::` (double colon) as
an attempt to invoke a registered directive, and errors on an unregistered name
(confirmed against `rstcheck` — the double-colon form fails the build). The
single-colon form doesn't match the directive pattern, so docutils falls through
to treating the whole block as an inert comment: it renders as nothing and
passes `rstcheck` cleanly, and is read by grep and by the persona-reviewer
agent, never by a reader of the built docs.

When the status dimension is enabled (it is, per the profile), a second comment
follows immediately after the quadrant marker:

```rst
.. status: implemented
```

Legend: `implemented`, `spec`.

## Document shapes

- **File-as-unit**: a file whose entire content serves one quadrant carries its
  markers once, immediately above the file's top-level title — consistent with
  Marker syntax above, and with the existing convention of a leading RST
  comment/directive before the title (e.g. `.. testsetup:: *` in
  `docs/howto_evaluate_model.rst`).
- **Heading-as-unit**: a file that mixes quadrants carries markers on each
  second-level heading that starts a new unit; the file's top-level title
  carries no marker of its own — the file is a container, not a unit.
- No unit may nest inside another unit of a different quadrant. A container file
  may hold sibling units of different quadrants side by side, never one inside
  another's section.

## Record

The markers themselves are the record — there is no separate table mapping files
or headings to quadrants. `grep -rn "^\.\. diataxis: " docs/` lists every
declared unit and its quadrant at any point in time.
