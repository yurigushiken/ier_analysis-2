# Look Who's Giving: Developmental Eye-Tracking Analysis Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research**: Look Who's Giving: Developmental Shift in Attention From Object Movement to the Faces
> Gushiken, M., Li, Y., Tang, J. E., & Gordon, P. (2024)

> **This work continues**: Gordon, P. (2003). The origin of argument structure in infant event representations. *ResearchGate*.

---

## Scientific Motivation

*The following 'scientific motivation' text is adapted from Dr. Peter Gordon's project proposal for this research.*

### The Origins of Language

The project investigates how pre-linguistic infants represent events (e.g., "giving", "showing") to determine if these concepts naturally undergird the acquisition of verb-argument structure in language.

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

2. **Gaze Strategy Shifts**: How do scanning patterns change across development?

3. **Semantic vs. Visual Processing**: Are these shifts driven by low-level visual features or emerging semantic understanding?

---

## Study Background

### Participants
- **Infants**: Multiple age cohorts of infants (7-11 months)
- **Adults**: Adult control participants (18+ years)

### Stimuli
Video recordings of actors performing social interaction events:

**Upright videos**:
- **GIVE-with-toy**: Person A hands object to Person B
- **GIVE-without-toy**: Giving gesture with empty hands
- **HUG-with-toy**: Person A and Person B embrace while holding object
- **HUG-without-toy**: Person A and Person B embrace
- **SHOW-with-toy**: Person A shows object to Person B
- **SHOW-without-toy**: Showing gesture without object present
- **FLOATING-toy**: Object moves without human interaction

**Inverted control videos** (to test semantic vs. low-level visual processing):
- **Inverted GIVE-with-toy**: Upside-down version of GIVE-with-toy
- **Inverted GIVE-without-toy**: Upside-down version of GIVE-without-toy
- **Inverted HUG-with-toy**: Upside-down version of HUG-with-toy
- **Inverted HUG-without-toy**: Upside-down version of HUG-without-toy

### Areas of Interest (AOI) Coding System

Infant gaze was coded frame-by-frame into the following AOI categories:
- `man_face` - Face of the male actor
- `woman_face` - Face of the female actor
- `man_body` - Body of the male actor
- `woman_body` - Body of the female actor
- `man_hands` - Hands of the male actor
- `woman_hands` - Hands of the female actor
- `toy_present` - The object itself (when present)
- `toy_location` - Where the object was or will be (when object is absent)

This coding system allows the pipeline to track which event arguments (agents, recipients, objects) infants fixate during social interactions.

### Analysis Parameters

The analyses presented here use the **min3-50_percent** dataset:
- **min3**: A fixation is defined as ≥3 consecutive frames on the same AOI
- **50_percent**: Only trials where the participant looked on-screen for ≥50% of frames are included

This ensures data quality while maintaining sufficient statistical power.

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
- **Advanced statistical methods**: Properly handles repeated measures with participant clustering
- **Precision Weighting**: Trials weighted by information content
- **Developmental trajectory analysis**: Continuous trends across infant ages
- **Effect size estimation**: Comprehensive statistical inference

### 4. **Publication-Ready Outputs**
For each analysis configuration:
- **Tables** (CSV): Summary statistics, trial-level data, GEE coefficients
- **Figures** (300 DPI PNG): Bar charts, forest plots, heatmaps, network diagrams, trend lines
- **Reports** (TXT/HTML/PDF): Statistical results, model diagnostics, interpretations

---

## Findings

### Finding 1: Developmental Trajectory in Event Processing

Analysis of tri-argument fixation patterns reveals significant developmental changes across the studied age range, with notable improvements observed in older infant cohorts.

### Finding 2: Shift From Motion-Tracking to Social-Semantic Processing

Gaze transition analysis indicates developmental shifts in attention strategies:
- Younger infants tend to track physical motion and object movement
- Older infants increasingly prioritize social cues and face-to-face monitoring
- Developmental patterns show progression toward adult-like attention strategies

### Finding 3: Evidence for Semantic Understanding

Comparison of upright and inverted event conditions suggests that observed attention patterns reflect semantic event understanding rather than purely visual feature processing.

---

## Features

✅ **Configuration-Driven Workflow**: Every analysis variant defined by a single YAML file (no code changes needed)
✅ **Multi-Threshold Support**: Generate fixations at different durations (min3, min4, min5) and quality levels (50%, 70% on-screen)
✅ **Precision-Weighted GEE**: Properly accounts for varying trial quality in transition analyses
✅ **Comprehensive Testing**: pytest coverage for data processing and statistical functions
✅ **High-DPI Visuals**: Publication-ready 300 DPI plots with consistent color palettes
✅ **Reproducible**: Version-controlled configs, deterministic outputs, documented statistical methods

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

Tri-argument fixation, time-window looks, latency-to-toy, and gaze-strategy outcomes are fit with statsmodels Generalized Estimating Equations that cluster on `participant_id` and choose Binomial or Gaussian families to match binary vs. continuous targets. Gaze-transition strategy GEEs additionally pass each trial’s total transition count as a precision weight so high-information trials contribute proportionally while repeated measures remain properly handled.

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Status**: Production Ready ✅
**Last Updated**: 2024-12-03
**Contact**: [Your contact information]

---

*This pipeline was developed to support reproducible research in developmental cognitive science. We hope it serves as a resource for the broader research community studying infant attention and social cognition.*
