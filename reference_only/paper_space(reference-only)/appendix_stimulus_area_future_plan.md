# Appendix Stimulus Area Figure Future Implementation Plan

## Scope
This document records a future manuscript update to add the stimulus area characterization figure to the Appendix. This is a planning note only and does not modify manuscript text or figures yet.

## Goal
Add `stimuli/output/stimuli_analysis_individual_plot_absolute.png` as an Appendix figure that documents low-level white-pixel area variation used for the Pixel control model.

## Requested framing to use
Use the convention that `a`-`d` variants are prime stimuli and `e` variants are target stimuli.

## Proposed Appendix Caption (requested version)
Appendix Figure X. Stimulus-level white-pixel area characterization for the dot-array set used to construct the Pixel control model. The top panel shows absolute white-pixel counts for each exemplar (`1a`-`6e`) computed from grayscale images using a fixed threshold (`pixel intensity > 200`). The bottom panel shows the corresponding stimulus images arranged by numerosity and exemplar. In the task implementation, `a`-`d` variants were used as prime exemplars and `e` variants as target exemplars. This panel documents low-level area variation across exemplars. See `stimuli/output/stimuli_analysis.csv` for per-image values.

## Planned manuscript touchpoints
1. `paper_space/sections/appendix.tex`
- Add a new figure environment after existing appendix table.
- Include image path.
- Add a stable label, for example `fig:appendix-stimulus-area`.

2. `paper_space/sections/methods.tex` (Pixel model paragraph)
- Add one short sentence referencing the new appendix figure.
- Suggested addition:
  - See Appendix Figure~\ref{fig:appendix-stimulus-area} for stimulus-level white-pixel area values across exemplars.

3. `paper_space/sections/discussion.tex` (Low-level visual properties paragraph)
- Add one short sentence referencing the new appendix figure.
- Suggested addition:
  - Stimulus-level area variation is shown in Appendix Figure~\ref{fig:appendix-stimulus-area}.

## Implementation checklist
1. Add figure block in `paper_space/sections/appendix.tex`.
2. Add one-line reference in Methods Pixel paragraph.
3. Add one-line reference in Discussion low-level visual properties paragraph.
4. Rebuild PDF (`paper_space/build.ps1`).
5. Copy rebuilt output to:
- `D:\eeg_nn\paper_space\Neural Similarity Patterns Support a Categorical Boundary Between 4 and 5 in Numerosity (1-6) (2026).pdf`
6. Check figure placement and reference resolution in final PDF.

## Validation checklist
1. Figure appears in Appendix and caption wraps cleanly.
2. In-text references resolve correctly.
3. No float overlap or caption truncation.
4. Build logs show no unresolved references.

## Notes
- This plan intentionally records the requested variant-role wording.
- If needed later, a verification pass can be run against trial metadata before final submission.
