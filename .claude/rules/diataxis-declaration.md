# Diátaxis declaration — pyhs3 docs

## Marker syntax

Every unit of content — a whole `.rst` file when the file serves one quadrant,
or an individual heading when a file mixes modes — carries its quadrant as an
RST comment directive immediately above the unit's title:

```rst
.. diataxis:: how-to
```

Legend: `tutorial`, `how-to`, `reference`, `explanation`. The directive is an
RST comment (`.. `) and renders as nothing; it is read by grep and by the
persona-reviewer agent, never by a reader of the built docs.

When the status dimension is enabled (it is, per the profile), a second
directive follows immediately after the quadrant marker:

```rst
.. status:: implemented
```

Legend: `implemented`, `spec`.

## Document shapes

- **File-as-unit**: a file whose entire content serves one quadrant carries its
  markers once, directly below the file's top-level title.
- **Heading-as-unit**: a file that mixes quadrants carries markers on each
  second-level heading that starts a new unit; the file's top-level title
  carries no marker of its own — the file is a container, not a unit.
- No unit may nest inside another unit of a different quadrant. A container file
  may hold sibling units of different quadrants side by side, never one inside
  another's section.

## Record

The markers themselves are the record — there is no separate table mapping files
or headings to quadrants. `grep -rn "^\.\. diataxis::" docs/` lists every
declared unit and its quadrant at any point in time.
