# CDS 2026 Poster

Poster for the Cognitive Development Society 2026 conference (April 9-11, Montreal).

**Title:** How Infant Looking Patterns to Trivalent Events Change from Object to Person Interaction

**Authors:** Gushiken, M., Li, Y., & Gordon, P.

## Build

Requires MiKTeX (or TeX Live) with `beamerposter` package.

```bash
cd poster_space/poster
pdflatex poster.tex
pdflatex poster.tex   # run twice for cross-references
```

## Style Notes

- **Never use em dashes or en dashes.** Use a regular hyphen `-` or rephrase. This applies to the LaTeX source, captions, and all text.
- **Do not use the terms "motion tracking" or "social attention"** as labels for transition types. These were internal working labels not approved for publication. Instead, refer to the transitions directly (e.g., "Toy-Body transitions" or "Face-Face transitions").
- Template: Gemini beamer poster theme, adapted for Teachers College Columbia University colors.
- Poster size: 48 x 36 inches (landscape).

## Files

- `poster.tex` - main LaTeX source
- `beamerthemegemini.sty` - Gemini theme (modified for pdflatex, no Raleway/Lato fonts)
- `beamercolorthemetccolumbia.sty` - TC Columbia color theme
- `figures/` - all images used in the poster
