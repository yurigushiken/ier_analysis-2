# Look Who's Giving: Developmental Eye-Tracking Analysis Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research Abstract**: [Look Who's Giving: Developmental Shift in Attention From Object Movement to the Faces](https://docs.google.com/document/d/1z3gO30xdPWrcsX5qYWvljeWNdxxwHL-vcqHfk6ucMcI/edit?tab=t.0#heading=h.f134jajyzrzy)
> Gushiken, M., Li, Y., Tang, J. E., & Gordon, P. (2024)

> **This work continues**: Gordon, P. (2003). The origin of argument structure in infant event representations. *ResearchGate*.

---

## Scientific Motivation

*The following is adapted from Dr. Peter Gordon's project proposal for this research.*

### The Origins of Language

The project investigates how pre-linguistic infants represent events (e.g., "giving" vs. "hugging") to determine if these concepts naturally undergird the acquisition of verb-argument structure in language.

### Inspiration from "Home Signers"

The hypothesis is driven by evidence that deaf children invent gestural communication systems with complex argument structures even without linguistic input, suggesting these concepts are innate to the human conceptual system.

### The Role of Intentionality

The study aims to understand how infants use intentionality to parse the "main act" of an event (e.g., distinguishing the goal of "giving" from the motion of "moving").

### Broader Impacts

**Clinical Applications**: Identifying the cognitive underpinnings of language can assist in helping individuals with neuropsychological conditions that diminish language learning or social understanding.

**Diversity & Community**: Situated at Teachers College near Harlem, the project is explicitly committed to training students from underrepresented minority populations and engaging the local community in scientific research.

---

## Research Questions

This pipeline addresses three core questions:

1. **Developmental Trajectory**: At what age do infants reliably fixate all three event arguments (giver/show-er, recipient/observer, object)?
   - **Finding**: Significant improvement at **10 months** (67% success vs. 35-48% in younger infants)

2. **Gaze Strategy Shifts**: How do scanning patterns change across development?
   - **Finding**: 7-month-olds track physical motion (Toy↔Body), while 10-11-month-olds and adults prioritize faces (Face↔Face, Face↔Toy)

3. **Semantic vs. Visual Processing**: Are these shifts driven by low-level visual features or emerging semantic understanding?
   - **Finding**: The face-prioritization effect **diminishes in inverted "Give" events**, suggesting semantic rather than purely visual processing

---

## Study Background

### Participants
- **Infants**: 41 participants aged 7-11 months (stratified by month)
- **Adults**: 13 adult controls (18+ years)

### Stimuli
Naturalistic videos of dyadic social interactions:
- **GIVE-with-toy** (primary condition): Person A hands object to Person B
- **SHOW-with-toy**: Person A shows object to Person B
- **GIVE-without-toy**: Giving gesture with empty hands (object absent)
- **Upside-down GIVE** (control): Inverted version of Give events

### Data Collection
- Eye-tracking at 30 Hz (frame-level AOI coding)
- Human-verified gaze coding (`What × Where` → AOI mapping)
- Minimum trial quality: ≥70% on-screen frames per participant×trial×condition

---

## What This Pipeline Does

This pipeline provides a **complete, reproducible workflow** from raw eye-tracking frames to publication-ready statistical analyses:

### 1. **Fixation Generation** ([src/](src/))
Converts frame-level gaze data into fixation events with configurable thresholds:
- Minimum fixation duration (3, 4, or 5 consecutive frames)
- On-screen attention filters (50% or 70% thresholds)
- AOI mapping (What/Where descriptors → semantic categories)

### 2. **Five Complementary Analyses** ([analyses/](analyses/))

| Analysis | Research Question | Statistical Method | Output |
|----------|------------------|-------------------|------------|
| **Tri-Argument Fixation** | Do infants fixate all three arguments? | Binomial GEE | Success rates, odds ratios |
| **Gaze Transitions** ⭐ | What scanning strategies do infants use? | Precision-weighted Gaussian GEE | Transition matrices, strategy proportions |
| **Latency to Toy** | How quickly do infants shift to the object? | Gaussian GEE | Mean latency by cohort |
| **Time Window Looks** | Do infants look at critical AOIs during moments? | Binomial GEE | Binary outcomes by time window |
| **Event Structure** | How does trial complexity affect coverage? | Descriptive statistics | Breakdown by event type |

### 3. **Statistical Rigor**
- **GEE (Generalized Estimating Equations)**: Properly handles repeated measures with participant clustering
- **Precision Weighting** (gaze transitions): Trials weighted by information content (2-9 transitions per trial)
- **Linear Trend Testing**: Continuous developmental trajectories across infant ages
- **Effect Sizes**: Odds ratios with 95% confidence intervals

### 4. **Publication-Ready Outputs**
For each analysis configuration:
- **Tables** (CSV): Summary statistics, trial-level data, GEE coefficients
- **Figures** (300 DPI PNG): Bar charts, forest plots, heatmaps, network diagrams, trend lines
- **Reports** (TXT/HTML/PDF): Statistical results, model diagnostics, interpretations

---

## Preliminary Results

### Finding 1: Developmental Step at 10 Months

**Tri-argument fixation success rates** (GIVE-with-toy, 70% on-screen threshold):

| Age Group | Success Rate | vs. 7-month-olds |
|-----------|-------------|------------------|
| 7-month-olds | 35% | Reference |
| 8-month-olds | 48% | OR=1.7, p=.41 (ns) |
| 9-month-olds | 44% | OR=1.5, p=.53 (ns) |
| **10-month-olds** | **67%** | **OR=3.8, p=.012*** |
| **11-month-olds** | **67%** | **OR=3.8, p=.025*** |
| **Adults** | **83%** | **OR=8.8, p<.001*** |

**Linear trend**: Each additional month increases success odds by 43% (p=.007)

### Finding 2: Shift From Motion-Tracking to Social-Semantic Processing

**Gaze transition strategies** (percentage of all transitions):

| Strategy | 7mo | 8mo | 9mo | 10mo | 11mo | Adults | Trend |
|----------|-----|-----|-----|------|------|--------|-------|
| **Motion Tracking** (Toy↔Body) | High | → | → | → | ↓ | Low | Decreasing** |
| **Social Monitoring** (Face↔Face) | Low | → | → | ↑ | ↑ | High | Increasing*** |
| **Agent-Object Binding** (Face↔Toy) | Low | → | ↑ | ↑ | ↑ | High | Increasing*** |

**Interpretation**: 7-month-olds follow the physical path of the object. By 10-11 months, infants prioritize faces, connecting people with objects—a strategy matching adult patterns.

### Finding 3: Semantic Understanding, Not Just Visual Features

The face-prioritization effect **disappears in inverted "Give" events**, suggesting infants are not simply responding to low-level visual salience but actively constructing semantic event representations.

---

## Features

✅ **Configuration-Driven Workflow**: Every analysis variant defined by a single YAML file (no code changes needed)
✅ **Multi-Threshold Support**: Generate fixations at different durations (min3, min4, min5) and quality levels (50%, 70% on-screen)
✅ **Precision-Weighted GEE**: Properly accounts for varying trial quality in transition analyses
✅ **Comprehensive Testing**: pytest coverage for data processing and statistical functions
✅ **High-DPI Visuals**: Publication-ready 300 DPI plots with consistent color palettes
✅ **Reproducible**: Version-controlled configs, deterministic outputs, documented statistical methods

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ier_analysis-2.git
cd ier_analysis-2

# Create conda environment
conda create -n ier_analysis python=3.8
conda activate ier_analysis

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

**Step 1: Generate Fixations**

```bash
python -m src.generator \
    --thresholds 3 4 \
    --output-root outputs \
    --min-onscreen-frames 105 \
    --dir-suffix -70_percent
```

**Step 2: Run Tri-Argument Analysis**

```bash
python -m analyses.tri_argument_fixation.run \
    --config analyses/tri_argument_fixation/gw_min3_70_percent.yaml
```

**Step 3: Run Gaze Transition Analysis**

```bash
python -m analyses.gaze_transition_analysis.run \
    --config analyses/gaze_transition_analysis/gw_transitions_min3_70_percent.yaml
```

**Step 4: View Results**

```bash
# Tables
ls analyses/tri_argument_fixation/gw_min3_70_percent/tables/

# Figures
ls analyses/tri_argument_fixation/gw_min3_70_percent/figures/

# Statistical reports
cat analyses/tri_argument_fixation/gw_min3_70_percent/reports/gw_min3_70_percent_gee_results.txt
```

---

## Data Requirements

### Input Format

Frame-level CSV files with human-verified AOI coding:

**Required Columns**:
- `Participant`, `participant_type`, `participant_age_months`
- `trial_number`, `condition`, `segment`
- `Frame Number`, `Onset`, `Offset`
- `What`, `Where` (AOI descriptors, e.g., "woman", "face")
- `event_verified`, `frame_count_trial_number`

**Directory Structure**:
```
data/csvs_human_verified_vv/
├── child/
│   ├── participant_001.csv
│   ├── participant_002.csv
│   └── ...
└── adult/
    ├── adult_001.csv
    └── ...
```

### AOI Mapping

The pipeline maps `(What, Where)` pairs to semantic AOI categories:

| What | Where | AOI Category |
|------|-------|-------------|
| woman | face | woman_face |
| man | face | man_face |
| toy | other | toy_present |
| toy2 | other | toy_location |
| woman/man | body | woman_body / man_body |
| woman/man | hands | woman_hands / man_hands |
| no | signal | off_screen |
| screen | other | screen_nonAOI |

---

## Documentation

### Analysis Configuration Examples

**Tri-Argument Analysis** ([gw_min3_70_percent.yaml](analyses/tri_argument_fixation/gw_min3_70_percent.yaml)):
```yaml
input_threshold_dir: "outputs/min3-70_percent"
input_filename: "gaze_fixations_combined_min3.csv"
condition_codes: ["gw"]
frame_window:
  start: 1
  end: 150
aoi_groups:
  man: ["man_face", "man_body"]
  woman: ["woman_face", "woman_body"]
  toy: ["toy_present"]
```

**Gaze Transition Analysis** ([gw_transitions_min3_70_percent.yaml](analyses/gaze_transition_analysis/gw_transitions_min3_70_percent.yaml)):
```yaml
input_fixations: "outputs/min3-70_percent/gaze_fixations_combined_min3.csv"
condition_codes: ["gw"]
aoi_nodes: ["man_face", "woman_face", "toy_present", "man_body", "woman_body"]
strategies:
  agent_agent_attention:
    - ["man_face", "woman_face"]
  agent_object_binding:
    - ["man_face", "toy_present"]
    - ["woman_face", "toy_present"]
  motion_tracking:
    - ["man_body", "toy_present"]
    - ["woman_body", "toy_present"]
```

### Statistical Methods

All analyses use **Generalized Estimating Equations (GEE)** to properly handle:
- Repeated measures (multiple trials per participant)
- Participant-level clustering (random intercept analogue)
- Appropriate distributions (Binomial for success/failure, Gaussian for continuous measures)

**Innovation**: The gaze transition analysis implements **precision weighting**, where each trial is weighted by its `total_transitions` count (range: 2-9), ensuring proper statistical inference when trials provide different amounts of information.

---

## Reproducibility

### Version Control
- All analysis configurations tracked in YAML files
- Statistical parameters explicitly documented
- Random seeds set where applicable

### Output Naming
All outputs prefixed by configuration name to prevent overwrites:
- `gw_min3_70_percent_tri_argument_summary.csv`
- `gw_min3_70_percent_transition_heatmap.png`
- `gw_min3_70_percent_gee_results.txt`

### Testing
```bash
pytest tests -v
```

Coverage includes:
- Fixation detection algorithms
- AOI mapping logic
- Statistical function parameter passing
- GEE model specifications

---

## Repository Structure

```
ier_analysis-2/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/                              # Input data (gitignored)
│   └── csvs_human_verified_vv/
│       ├── child/
│       └── adult/
├── outputs/                           # Generated fixations (gitignored)
│   ├── min3-50_percent/
│   ├── min3-70_percent/
│   ├── min4-50_percent/
│   └── min4-70_percent/
├── src/                               # Core data processing pipeline
│   ├── config.py                      # Project constants and paths
│   ├── loader.py                      # Frame CSV ingestion
│   ├── gaze_detector.py               # Fixation detection algorithm
│   ├── aoi_mapper.py                  # What/Where → AOI mapping
│   └── generator.py                   # CLI fixation generator
├── analyses/                          # Five analysis systems
│   ├── tri_argument_fixation/
│   │   ├── run.py                     # Main entry point
│   │   ├── pipeline.py                # Trial metric computation
│   │   ├── stats.py                   # GEE statistical models
│   │   ├── visuals.py                 # Plotting functions
│   │   ├── reports.py                 # Report generation
│   │   ├── gw_min3_70_percent.yaml    # Configuration file
│   │   └── gw_min3_70_percent/        # Output directory
│   │       ├── tables/
│   │       ├── figures/
│   │       └── reports/
│   ├── gaze_transition_analysis/
│   │   ├── run.py
│   │   ├── transitions.py             # Transition counting
│   │   ├── matrix.py                  # Transition matrices
│   │   ├── strategy.py                # Precision-weighted GEE
│   │   └── visuals.py
│   ├── latency_to_toy/
│   ├── time_window_look_analysis/
│   └── __init__.py
├── tests/                             # Unit and integration tests
└── presentation/                      # Presentation utilities
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Status**: Production Ready ✅
**Last Updated**: 2024-12-03
**Contact**: [Your contact information]

---

*This pipeline was developed to support reproducible research in developmental cognitive science. We hope it serves as a resource for the broader research community studying infant attention and social cognition.*
