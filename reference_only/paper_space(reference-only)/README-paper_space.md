# Paper Space (CCN 2026)

Local workspace for the CCN 2026 manuscript.

## Paper Overview

- Title: `Neural Similarity Patterns Support a Categorical Boundary Between 4 and 5 in Numerosity (1-6)`
- Dataset: ALL trials (correct + incorrect), 24 participants
- Main temporal window: `-100 to 700 ms`
- Main claim: converging evidence for a PI/ANS boundary at `4-5`, with numerosity `1` inflating ratio-based fits in with-1 analyses

## Manuscript Style Rules

- No en dashes or em dashes in manuscript prose.
- No semicolons in text (split into separate sentences).
- Colons are allowed.
- No qualitative adjectives (e.g., "robust", "striking").
- Use plain hyphen for ranges, for example `0-700 ms`, `4-5`, `2-6`.
- Active voice throughout. Measured, evidence-first language.
- Results: present evidence before conclusions.

## Build

From `paper_space/`:

```powershell
.\build.ps1
.\build.ps1 -Clean
```

Quick rebuild without bibliography edits:

```powershell
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

`build.ps1` uses `latexmk` when available, then falls back to `pdflatex + biber + pdflatex + pdflatex`.

## Output Naming

After build, always copy `main.pdf` to the titled deliverable:

```powershell
Copy-Item .\main.pdf ".\Neural Similarity Patterns Support a Categorical Boundary Between 4 and 5 in Numerosity (1-6) (2026).pdf" -Force
```

Do not distribute `main.pdf` directly.

## Directory Map

```text
paper_space/
|-- main.tex
|-- sections/
|   |-- introduction.tex
|   |-- methods.tex
|   |-- results.tex
|   |-- discussion.tex
|   `-- appendix.tex
|-- figures/
|   |-- condition_matrix.tex
|   |-- table_static_model_comparison.tex
|   |-- static_model_rdms.tex
|   |-- static_change6_mds.tex
|   |-- temporal_distance.tex
|   |-- temporal_change24_model_rdms_compact.tex
|   |-- temporal_change24_rdm_snapshots.tex
|   |-- temporal_change24_mds.tex
|   |-- temporal_rdm_evolution.tex
|   `-- table_model_differences.tex
|-- media/
|-- bib/references.bib
|-- ccn-template-main/
|-- build.ps1
`-- README-paper_space.md
```

## Current Figure Wiring

Wrappers are included from `sections/methods.tex` and `sections/results.tex`.

- `figures/condition_matrix.tex`
  - Uses `media/trial_structure.png`, `media/conditions-6.png`, and `media/conditions-24.png`
- `figures/table_static_model_comparison.tex`
  - Source: static tables in `results/analysis-lda/static-change6/.../tables-pairwise/` and `results/analysis-lda/static-change5/.../tables-pairwise/`
- `figures/static_model_rdms.tex`
  - Source: static compact row composites for change6 and change5
- `figures/static_change6_mds.tex`
  - Source: static change6 MDS image
- `figures/temporal_distance.tex`
  - Source: `temporal_change24_all-100-700ms/figures/temporal_distance_timecourse.png` and no1 counterpart
- `figures/temporal_change24_model_rdms_compact.tex`
  - Source: `temporal_model_rdms_all_compact_col-...-RT.png` (with and no1)
- `figures/temporal_change24_rdm_snapshots.tex`
  - Source: `temporal_rdm_snapshots-...png`
- `figures/temporal_change24_mds.tex`
  - Source: `temporal_mds_static_0-700ms-change_code-...png`
- `figures/temporal_rdm_evolution.tex`
  - Source: `temporal_model_fits_with_significance-...png` (with and no1)
- `figures/table_model_differences.tex`
  - Source: model-difference outputs from `temporal_change24_all-100-700ms/tables*` and `stats*`

Note: `media/option1_trial_schematic.png` is a design asset and is not currently the manuscript figure source.

## Active Analysis Directories

- `results/analysis-lda/static-change6/static_change6_all`
- `results/analysis-lda/static-change5/static_change5_all`
- `results/analysis-lda/temporal_change24_all-100-700ms`
- `results/analysis-lda/temporal_change6_all-100-700ms` (supporting)
- `results/analysis-lda/temporal_change5_all-100-700ms` (supporting)
- `results/analysis-lda/temporal_cardinality_loso-100-700ms` (supporting)

## Statistical Workflow Summary

- Whole-epoch static model tests:
  - one Spearman correlation per subject per model
  - one-sample t-tests against zero
  - Holm correction across models
  - boundary contrast: paired t-test on Fisher-z differences, two-sided
- Temporal model-fit tests:
  - cluster-based permutation over time windows (5,000 permutations)
- Temporal model-difference tests:
  - permutation on Fisher-z correlation differences over time (10,000 permutations)
  - two-sided for all model-difference comparisons

## Naming Conventions for Manuscript Text

Use descriptive names in prose. Keep internal codes only when needed.

- static change6: target-only whole-epoch analysis (1-6)
- static change5: target-only whole-epoch analysis (2-6, no numerosity 1 in training/evaluation)
- temporal change24: 24-condition prime-target analysis
- temporal change24 no1: 24-condition analysis excluding pairs containing 1

## Plot Regeneration Commands

Run from repo root with the active project environment.

```bash
python -m scripts.rsa.bundle_temporal_change24 \
  --run-root results/runs-lda/temporal_change24_all \
  --time-min -100 --time-max 700 \
  --skip-temporal-generalization

python -m scripts.rsa.bundle_static_change6 \
  --run-root results/runs-lda/static_change6_all \
  --analysis-dir results/analysis-lda/static-change6/static_change6_all

python -m scripts.rsa.bundle_static_change5 \
  --run-root results/runs-lda/static_change5_all \
  --analysis-dir results/analysis-lda/static-change5/static_change5_all
```

If only static model-comparison outputs need refresh:

```bash
python -m scripts.rsa.analyze_rsa_model_comparison \
  --run-dir results/runs-lda/static_change6_all \
  --output-dir results/analysis-lda/static-change6/static_change6_all

python -m scripts.rsa.analyze_rsa_model_comparison \
  --run-dir results/runs-lda/static_change5_all \
  --output-dir results/analysis-lda/static-change5/static_change5_all
```

## Results Section Order (Current)

1. Distance-Dependent Decoding
2. Whole-Epoch Model Comparison
3. Time-Resolved Analysis

## Two-Column LaTeX Layout Guide (CCN Template)

Hard-won knowledge for tight, professional figure placement in a two-column conference paper.

### Float Types and Behavior

- `figure` / `table` = single-column float (fits in one column)
- `figure*` / `table*` = full-width float (spans both columns)
- Full-width floats (`figure*`) can ONLY appear at the top of a page. LaTeX queues them for the NEXT page after they are defined. Define them early so they land on earlier pages.
- Single-column floats can appear at top, bottom, or on float-only pages.

### Placement Specifiers

Use `[!tbp]` consistently for all floats. The `!` overrides LaTeX's conservative defaults. Avoid `[h]` or `[htbp]` (the `h` option can cause mid-text placement that breaks reading flow in two-column layouts). Never use `[H]` (it disables float placement entirely and causes overflow).

### Key Preamble Settings

```latex
% Allow more floats per page and reduce wasted space.
\setcounter{topnumber}{3}        % up to 3 floats at top of column
\setcounter{dbltopnumber}{2}     % up to 2 full-width floats at top
\setcounter{bottomnumber}{2}     % up to 2 floats at bottom
\setcounter{totalnumber}{5}      % up to 5 floats per page total
\renewcommand{\topfraction}{0.95}      % float can use 95% of top
\renewcommand{\dbltopfraction}{0.95}   % same for full-width
\renewcommand{\bottomfraction}{0.5}    % float can use 50% of bottom
\renewcommand{\textfraction}{0.05}     % only 5% of page must be text
\renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dblfloatpagefraction}{0.85}

% Tight spacing between floats and text.
\setlength{\textfloatsep}{6pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{6pt plus 2pt minus 2pt}
\setlength{\floatsep}{4pt plus 2pt minus 2pt}
\setlength{\dblfloatsep}{4pt plus 2pt minus 2pt}
\setlength{\intextsep}{4pt plus 2pt minus 2pt}
\setlength{\abovecaptionskip}{3pt}
\setlength{\belowcaptionskip}{0pt}

% Prevent float-only pages from centering vertically (kills whitespace).
\makeatletter
\setlength{\@fptop}{0pt}
\setlength{\@dblfptop}{0pt}
\makeatother
```

### Figure Ordering Strategy

1. **Define `figure*` (full-width) environments as early as possible** in the source. They queue for the next page top, so early definition = earlier placement. Group them at the start of a section if their exact position relative to text is flexible.
2. **Define single-column figures near the text that references them.** LaTeX usually places them in the same column or the next available slot.
3. **Avoid `\clearpage` unless absolutely necessary.** It forces all queued floats to be placed immediately, often creating partially-empty pages. Let LaTeX handle float placement naturally.

### Handling Tall Figures

If a single-column figure is taller than the text area, LaTeX will warn "Float too large for page." Fix with a height cap:

```latex
\includegraphics[width=\columnwidth,height=0.88\textheight,keepaspectratio]{...}
```

The `keepaspectratio` flag ensures the image scales uniformly. Use 0.85-0.90 of `\textheight` to leave room for the caption.

### Packages

- `dblfloatfix` -- allows `figure*` to appear at the bottom of a page (not just top). Essential for two-column layouts with many full-width figures.
- `placeins` -- provides `\FloatBarrier` to prevent floats from drifting past section boundaries. Use sparingly.

### Common Mistakes

- Using `\clearpage` to "fix" float placement. It usually creates more whitespace than it solves.
- Defining `figure*` environments late in the source. They always go to the NEXT page, so late definition = late placement = float pile-up.
- Setting `\textfraction` too high (default 0.2). This prevents LaTeX from placing floats on pages that don't have enough text, creating float-only pages.
- Not using `!` in placement specifiers. Without it, LaTeX applies conservative defaults that reject valid placements.
- Using `[H]` from the float package. It disables all float placement logic and causes overflow.

## PI Model Naming Convention

- **With numerosity 1 (1-6)**: Target PI(1-4), Target PI(1-3), ANS
- **Without numerosity 1 (2-6, no-1)**: Target PI(2-4), Target PI(2-3), ANS
- The PI model is redefined when numerosity 1 is excluded (boundary range shifts)

## Change Log

### 2026-02-13

- Removed Prime x Target model framing from active manuscript text and figure captions.
- Removed prime-only and activity-silent working-memory interpretation language from Discussion.
- Updated Appendix temporal model-fit cluster table to include only active reported models (Target PI, ANS, RT).
- Updated README inventory to match current section files and Results subsection titles.

### 2026-02-11

- Fixed PI model labels in no-1 context: PI(1-4) to PI(2-4) in results.tex and temporal_rdm_evolution.tex caption.
- Fixed Table 1 boundary contrast wording for two-sided test (neutral "tests for a difference" instead of directional "correlates more", row label "vs" instead of ">").
- Comprehensive Introduction restructure (theory-driven 5-paragraph flow).
- Discussion rewrite (Limitations as 4 paragraphs, Summary scoped).
- Methods improvements (analysis strategy rationale, model description framing, Statistics rewrite).
- Results improvements (subsection titles, bridge sentences, figure order optimization, caption rewrites, Table 2 column rename).
- Added style rules to README.

### 2026-02-10

- Synced documentation to current manuscript wrapper usage.
- Updated build notes to match `biber` fallback behavior.
- Added static bundle commands (`bundle_static_change5`, `bundle_static_change6`).
- Updated results section order and figure source mapping.

### 2026-02-09

- Added whole-epoch static RSA section and corresponding table/figures.
- Updated manuscript title and summary language to the 4-5 boundary framing.

### 2026-02-07

- Added temporal model-difference tests and two-sided/one-sided reporting conventions.

### 2026-02-06

- Migrated manuscript to ALL-data temporal outputs (`-100 to 700 ms`).
