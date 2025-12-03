## Tri-Argument Fixation Analysis

This analysis investigates whether participants in each age cohort produce
fixations to every essential argument (giver, recipient, object) while viewing
`gw` (give-with-toy) events. Failing to fixate all three AOI groups may indicate
an incomplete event representation for GIVE.

### Research framing
- **Research Question**: Do participants in each cohort fixate all relevant
  arguments (man, woman, toy) during GIVE-with-toy events?
- **Hypothesis**: Infants younger than 10 months will fail to fixate all three
  arguments more often than older infants (10–11 months) and adults.
- **Prediction**: 10+ month infants and adults will show a higher proportion of
  GIVE trials with tri-argument coverage than 7–9 month infants.

### Files
- `gw_min4.yaml`, `gwo_min4.yaml`: Configuration files (threshold directory, AOI
  groups, cohorts, report text).
- `run.py`: Executable analysis pipeline.
- Outputs:
  - `tables/` for aggregated CSV summaries.
  - `figures/` for visualizations.
  - `reports/` for `.txt`, `.html`, `.pdf` deliverables.

### Running
```
conda activate ier_analysis
python -m analyses.tri_argument_fixation.run --config analyses/tri_argument_fixation/gw_min4.yaml
python -m analyses.tri_argument_fixation.run --config analyses/tri_argument_fixation/gwo_min4.yaml
python -m analyses.tri_argument_fixation.run --config analyses/tri_argument_fixation/gw_min4_35-90.yaml
python -m analyses.tri_argument_fixation.run --config analyses/tri_argument_fixation/gwo_min4_35-90.yaml
python -m analyses.tri_argument_fixation.run --config analyses/tri_argument_fixation/gw_min3.yaml
python -m analyses.tri_argument_fixation.run --config analyses/tri_argument_fixation/gwo_min3.yaml
python -m analyses.tri_argument_fixation.run --config analyses/tri_argument_fixation/sw_min3.yaml
python -m analyses.tri_argument_fixation.run --config analyses/tri_argument_fixation/swo_min3.yaml
```

Key configuration fields (per YAML):
- `input_threshold_dir` + `input_filename`: Select which gaze-fixation CSV set
  (e.g., `min4`) to analyze. By default, the combined cohort file is used.
- `condition_codes`: Event codes to include (currently `[gw]`).
- `aoi_groups`: Explicit AOI labels that must be observed for each argument
  (customizable if you want body+face grouped, etc.).
- `min_trials_per_participant`: Drop participants with fewer GIVE trials than
  this threshold.
- `cohorts`: Ordered list of age bins; results respect this order.
- `report.*`: Research Question, Hypothesis, and Prediction text inserted into
  reports. You can also add `plot_title` here for custom chart headings.
- `frame_window`: Optional `start`/`end` frame indices; if supplied, only
  fixations whose `gaze_start_frame`–`gaze_end_frame` overlap that interval are
  included, and the window is appended to the chart title. Output directories
  are auto-derived from the config filename (e.g., running `gw_min4.yaml` writes
  to `.../gw_min4/{tables,figures,reports}`).

-Outputs per config (e.g., `gw_min4/`):
- `tables/tri_argument_summary.csv` summarizing per-cohort coverage.
- `figures/tri_argument_success.png` (bar chart of coverage by cohort, now with
  significance annotations when GEE is enabled).
- `figures/trials_per_participant.png` showing how many valid trials each
  participant contributed (after all filters).
- `reports/tri_argument_report.{txt,html,pdf}` mirroring the key findings and
  embedding the visualization.
- If a `gee` block is enabled in the config:
  - `reports/gee_results.txt` contains the statsmodels GEE coefficient table
    (clustered on participant_id) plus descriptive/model diagnostics.
  - `figures/forest_plot_odds_ratios.png` plots odds ratios with 95% CIs.

### Notes
- No synthetic data is used; tests rely on curated slices from the real GIVE
  datasets.
- AOI groups are defined explicitly in the config to avoid ambiguity about what
  counts as “man”, “woman”, or “toy”.

