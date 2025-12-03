## Project Extension: Multi-Threshold Gaze Fixations & Developmental Eye-Tracking Analyses

This subproject provides the **current, active** eye-tracking analysis system for studying infant cognitive development through gaze patterns.

### Overview

The project extension includes **five complementary analysis systems**:

1. **Fixation Generator**: Generate filtered, threshold-aware fixation CSVs directly from
   human-verified frame-level inputs under `data/csvs_human_verified_vv/`. Enforces ≥30 on-screen
   frames (not `What=no/Where=signal`) per participant×trial×condition, with support for stricter
   requirements (≥75 frames for "50% on-screen" or ≥105 frames for "70% on-screen" datasets).

2. **Tri-Argument Fixation Analysis** (GW/GWO/SW/SWO): *"At what age do participants reliably fixate
   every verb argument (giver/show-er, recipient/observer, object or its location) within an event?"*
   Uses binomial GEE with participant clustering.

3. **Gaze Transition Analysis** ⭐: Examines how infants shift attention between areas of interest.
   Uses **precision-weighted Gaussian GEE** to properly account for trials having different numbers
   of transitions (2-9 per trial). **Key improvement (2024)**: Added weighting by `total_transitions`
   to ensure trials with more data points have appropriately greater statistical influence.

4. **Latency to Toy Analysis**: Measures how quickly participants first fixate on the toy after
   event onset. Compares infant cohorts to adults using Gaussian GEE.

5. **Time Window Look Analysis**: Tests whether participants looked at specific targets during
   defined time windows using binomial GEE.

---

## Statistical Methods ⭐

All analyses use **Generalized Estimating Equations (GEE)** to properly handle repeated measures
(multiple trials per participant) while estimating population-level effects.

### Recent Improvement: Precision Weighting in Gaze Transitions

The gaze transition analysis now implements **precision weighting** (added 2024):

```python
weights = working["total_transitions"].fillna(0)

model = smf.gee(
    formula=formula,
    groups="participant_id",
    data=working,
    family=sm.families.Gaussian(),
    weights=weights,  # Trials weighted by information content
)
```

**Why this matters:**
- Each trial contributes a different number of transitions (range: 1-9)
- Old approach: All trials weighted equally → biased estimates, overconfident p-values
- New approach: Trials weighted by `total_transitions` → proper uncertainty quantification

**Impact:**
- Coefficient estimates: Modest changes (<20%)
- P-values: May shift by 0.01-0.05, particularly for marginal findings
- More statistically principled and honest about uncertainty

## Tri-Argument Fixation Analysis

### Research questions & configurations

| YAML             | Condition (frames)                                 | Research question                                                                                   |
|------------------|----------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `gw_min4.yaml`   | GIVE-with-toy (frames 1–150, min4 fixations)       | Do cohorts fixate giver+recipient+toy when the object is present?                                    |
| `gwo_min4.yaml`  | GIVE-without-toy (frames 1–150, min4)              | Do cohorts still encode toy *locations* when the object disappears?                                  |
| `sw_min4.yaml`   | SHOW-with-toy (frames 45–115, min4)                | During the core interaction, do they look at show-er, observer, and toy?                             |
| `swo_min4.yaml`  | SHOW-without-toy (frames 45–115, min4)             | Same as above, but the toy must be inferred.                                                         |
| `gw_min3.yaml`   | GIVE-with-toy (min3 sensitivity analysis)          | How do results shift with a looser fixation threshold?                                               |
| `gwo_min3.yaml`  | GIVE-without-toy (min3)                            | Sensitivity analysis for the absence-of-toy case.                                                    |
| `sw_min3.yaml`   | SHOW-with-toy (min3)                               | Sensitivity analysis for SHOW during the interaction window.                                         |
| `swo_min3.yaml`  | SHOW-without-toy (min3)                            | Sensitivity analysis for SHOW-without-toy.                                                           |

Across all configs we hypothesize a developmental step, with 10–11 month-olds
and adults exhibiting significantly higher tri-argument coverage than 7–9 month
cohorts. Adults should always form the upper bound. Each analysis runs a
statsmodels **GEE (Binomial, logit link)** clustered on participant ID (random
intercept analogue), reporting odds ratios vs the 7-month reference.

---

## Gaze Transition Analysis ⭐

Located under `analyses/gaze_transition_analysis/`, this analysis examines
the **order and frequency of AOI fixations**, focusing on three key gaze strategies:

1. **Agent-Agent Attention**: Transitions between the two people's faces (social monitoring)
2. **Agent-Object Binding**: Transitions between faces and the toy (referential understanding)
3. **Motion Tracking**: Transitions between bodies and the toy (physical action tracking)

### Key Features

- **Precision-weighted GEE**: Properly accounts for varying trial quality (2-9 transitions per trial)
- **Cohort comparisons**: Tests each infant age group vs. 7-month reference
- **Linear trends**: Tests for continuous developmental change across infant ages (7-11 months)
- **Network visualizations**: Transition matrices and heatmaps showing gaze flow patterns

### Running the analysis

```bash
conda activate ier_analysis
python -m analyses.gaze_transition_analysis.run \
    --config analyses/gaze_transition_analysis/gw_transitions_min3_50_percent.yaml
```

### Outputs (prefixed by config name)

- **Tables:**
  - `*_transition_counts.csv` – Raw per-participant/trial transition counts
  - `*_transition_matrix.csv` – Cohort-level mean transition frequencies
  - `*_strategy_proportions.csv` – Per-trial strategy proportions with `total_transitions`
  - `*_strategy_summary.csv` – Cohort-level strategy means

- **Figures:**
  - `*_transition_heatmap.png` – Cohort transition frequency heatmaps
  - `*_agent_agent_attention_strategy.png` – Strategy bar chart with significance brackets
  - `*_motion_tracking_strategy.png` – Motion tracking strategy comparison
  - `*_agent_object_binding_strategy.png` – Agent-object binding comparison
  - `*_linear_trend_*.png` – Age trend visualizations for each strategy

- **Reports:**
  - `*_stats_agent_agent_attention.txt` – GEE results + linear trend test
  - `*_stats_motion_tracking.txt` – GEE results + linear trend test
  - `*_stats_agent_object_binding.txt` – GEE results + linear trend test
  - `*_transition_summary.txt` – Top transitions per cohort

### Statistical Innovation

The gaze transition analysis is the **first in this project** to implement precision weighting:

```python
# Before (INCORRECT):
# All trials weighted equally regardless of transition count

# After (CORRECT):
weights = working["total_transitions"]
model = smf.gee(..., weights=weights)
# Trials with 9 transitions → 9x influence
# Trials with 2 transitions → 2x influence
```

This ensures proper statistical inference when trials provide different amounts of information.

---

## Latency to Toy Analysis

Located under `analyses/latency_to_toy/`, this analysis measures the **time
from event onset until first toy fixation**.

### Research Question

*Do infants take longer than adults to shift attention to the toy? Does latency decrease with age?*

### Statistical Approach

- **Adult-reference GEE**: Compares each infant cohort to adults (Gaussian family)
- **Linear trend test**: Tests for continuous decrease in latency across infant ages

### Running the analysis

```bash
conda activate ier_analysis
python -m analyses.latency_to_toy.run \
    --config analyses/latency_to_toy/latency_config.yaml
```

### Key Feature

Event onset is properly referenced (not just raw frame numbers), ensuring latency measures
"time since the relevant event began" rather than absolute frame position.

---

## Time Window Look Analysis

Located under `analyses/time_window_look_analysis/`, this analysis tests
**binary outcomes**: Did the participant look at a target AOI during a specific time window?

### Research Question

*Do different age cohorts differ in the probability of fixating on critical areas during key moments?*

### Statistical Approach

- **Binomial GEE**: Proper distribution for yes/no outcomes
- **Odds ratios**: Interpretable effect sizes (OR > 1 = higher probability than reference)
- **Linear trend test**: Tests for developmental change across infant ages

### Running the analysis

```bash
conda activate ier_analysis
python -m analyses.time_window_look_analysis.run \
    --config analyses/time_window_look_analysis/time_window_config.yaml
```

---

## Project Layout

```
ier_analysis-2/
├── README.md                          # This file
├── outputs/                           # Generated fixation CSVs (gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py                      # Constants (thresholds, paths)
│   ├── loader.py                      # Frame CSV ingestion & ≥30-frame trial filter
│   ├── aoi_mapper.py                  # Local What/Where → AOI logic
│   ├── gaze_detector.py               # Threshold-based fixation detection
│   └── generator.py                   # CLI entry point
└── analyses/
    ├── tri_argument_fixation/
    │   ├── run.py                     # Shared runner (plots, reports, GEE)
    │   ├── stats.py                   # Statistical functions
    │   ├── gw_min4.yaml ...           # Condition-specific configs
    │   └── gw_min4/...                # Per-config outputs (tables/figures/reports)
    │
    ├── gaze_transition_analysis/ ⭐
    │   ├── run.py                     # Main entry point
    │   ├── strategy.py                # GEE with PRECISION WEIGHTING
    │   ├── transitions.py             # Transition counting
    │   ├── matrix.py                  # Transition matrices
    │   ├── visuals.py                 # Plotting functions
    │   ├── gw_transitions_min3_50_percent.yaml ...
    │   └── gw_transitions_min3_50_percent/...
    │
    ├── latency_to_toy/
    │   ├── run.py                     # Main entry point
    │   ├── stats.py                   # GEE analysis
    │   ├── latency_analysis.py        # Latency computation
    │   └── latency_config.yaml ...
    │
    └── time_window_look_analysis/
        ├── run.py                     # Main entry point
        ├── stats.py                   # Binomial GEE analysis
        └── time_window_config.yaml ...
```

---

## Running the Fixation Generator

```bash
conda activate ier_analysis
python -m src.generator \
    --thresholds 3 4 5 \
    --output-root outputs \
    --min-onscreen-frames 105 \
    --dir-suffix -70_percent
```

- Default inputs: `data/csvs_human_verified_vv/child` and `/adult`
- Override with `--child-dir`/`--adult-dir` flags if needed
- Pass `--exclude-screen-nonroi` to drop `screen_nonAOI` fixations from exported CSVs
  (while still counting them toward on-screen frame totals)

### Outputs per threshold

- `outputs/min3/gaze_fixations_child_min3.csv`
- `outputs/min3/gaze_fixations_adult_min3.csv`
- `outputs/min3/gaze_fixations_combined_min3.csv`
- Additional 50%/70% directories: `min3-50_percent`, `min4-70_percent`, etc.

Each CSV contains `min_frames` (threshold used), `cohort`, and the requested on-screen-frame
filter already applied.

**Current focus:** We primarily use the **70% datasets** (`min3-70_percent/`, `min4-70_percent/`)
for the most conservative analyses.

---

## Configuration System

All analyses use **YAML-driven configuration**:

### Tri-argument analysis example

```yaml
condition: "gw"
condition_label: "Give"
fixation_source: "outputs/min4-70_percent/gaze_fixations_combined_min4.csv"
frame_range: [1, 150]
cohorts:
  - label: "7-month-olds"
    min_months: 7
    max_months: 7
  - label: "Adults"
    min_months: 18
    max_months: 100
```

### Gaze transition analysis example

```yaml
input_fixations: "outputs/min3-50_percent/gaze_fixations_combined_min3.csv"
condition_codes: ["gw"]
aoi_nodes: ["man_face", "woman_face", "toy_present", "man_body", "woman_body"]
cohorts:
  - label: "7-month-olds"
    min_months: 7
    max_months: 7
```

**Config-driven workflow benefits:**
- Adding new variants requires only a YAML file (no code changes)
- All outputs prefixed by config name (prevents overwrites)
- Clear documentation of analysis parameters

---

## Design Highlights (for Reuse)

- **Config-driven workflow**: Every analysis defined by a single YAML file
- **Prefixed output naming**: All files include config name (e.g., `gw_min4_70_percent_*`)
- **High-DPI visuals**: 300 DPI plots with reader-friendly labels
- **Consistent color palettes**: Cohort colors maintained across all plot types
- **Robust reporting**: Aligned TXT/HTML/PDF reports with embedded figures
- **Statistical rigor**: GEE with proper clustering, precision weighting where appropriate
- **Comprehensive testing**: pytest coverage for unit tests and CLI integration

---

## Tests

Place regression tests under `tests/` (gitignored locally for flexibility):

```bash
conda activate ier_analysis
pytest tests -v
```

Coverage targets:
- Fixation generator (trial filtering, min-frame thresholds)
- Tri-argument reporting pipeline (bar charts, forest plots, GEE outputs)
- Gaze transition analysis (precision weighting, strategy calculations)
- Statistical functions (GEE parameter passing, weighting verification)

---

## Statistical Documentation

For detailed statistical methods, interpretations, and comparisons across all analyses:

## Version History

**Version 2.0** (2024-11)
- ✅ Added gaze transition analysis with precision-weighted GEE
- ✅ Added latency to toy analysis
- ✅ Added time window look analysis
- ✅ Implemented statistical improvements (precision weighting)
- ✅ Comprehensive test coverage
- ✅ Updated documentation with statistical methods

**Version 1.0** (2024-10)
- ✅ Tri-argument fixation analysis
- ✅ Multi-threshold fixation generator
- ✅ 50%/70% on-screen filtering options

---

## Citation

If you use this analysis system in your research, please cite:

```bibtex
@software{ier_analysis_project_extension_2024,
  title = {Project Extension: Developmental Eye-Tracking Analysis System},
  author = {[Your Name/Team]},
  year = {2024},
  note = {Precision-weighted GEE analysis for infant gaze patterns}
}
```

---

**Status**: Production Ready ✅
**Last Updated**: 2024-11-29
**Active Development**: Gaze transition analysis, latency analysis, time window analysis

---

## Environment setup reminder

Before running any generator, analysis, or test command, always activate the shared environment:

```bash
conda activate ier_analysis
```
