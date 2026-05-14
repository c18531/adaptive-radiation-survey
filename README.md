# Forecast-informed adaptive radiation surveying with residual Gaussian process regression for rapid radiological contamination assessment

This repository contains the code, reproducibility workflow, and supporting files for the study:

> **Wooseok Choi, Myeongsik Shin, and Ku Kang**  
> *Forecast-informed adaptive radiation surveying with residual Gaussian process regression for rapid radiological contamination assessment*

## Overview

Rapid radiological contamination assessment after a nuclear or radiological release requires faster decision support than conventional structured surveys alone can provide. This repository implements a forecast-informed adaptive radiation survey framework that combines:

- atmospheric-dispersion forecast priors,
- residual Gaussian process regression (GPR),
- path-aware / risk-aware adaptive sampling, and
- FEMA PAG-based Hot/Warm/Cold zone delineation.

In the simulation benchmark reported in the manuscript, the proposed adaptive design achieved **74.9% zone-classification accuracy with 83 measurements**, compared with approximately **480 measurements** for the structured MARSSIM/VSP baseline. This corresponds to an **82.7% reduction in sampling effort** and an **84.4% reduction in mission time**.

## Main capabilities

- Forecast-informed contamination assessment using atmospheric-dispersion priors
- Residual GPR updating in measurement space
- Adaptive waypoint selection with uncertainty, boundary, and travel-cost balancing
- Structured MARSSIM/VSP-style baseline comparison
- Monte Carlo evaluation and weighted risk-cost analysis
- Prefix diagnostics for fatal-miss and Pareto-front analysis
- Payload export / rebuild workflow for reproducible result generation
- Footprint sensitivity analysis and VSP-track experiments

## Repository structure

```text
.
├── Radiological_Reconnaissance.py
├── raw_data/
│   ├── Areas_dxf/
│   ├── Areas_shp/
│   ├── Areas_shx/
│   ├── Case2_DXF/
│   └── baseline_csv/
├── outputs/
├── paper/
│   └── manuscript.pdf
└── README.md
```

## Requirements

Python 3.10 or higher is recommended.

Main Python packages used by the script:

- numpy
- pandas
- matplotlib
- scipy
- shapely
- geopandas
- scikit-learn

Example installation:

```bash
pip install numpy pandas matplotlib scipy shapely geopandas scikit-learn
```

## Input files

The main script expects the following study inputs:

- `--Case2_dxf`: DXF contamination prior / warm-hot geometry (**required**)
- `--Areas_shp`: survey boundary shapefile (optional but recommended)
- `--baseline_csv`: baseline station-plan CSV

Optional shapefile companion paths (`--survey_shx`, `--survey_dbf`) can also be supplied when needed.

## Quick start

### 1. Standard evaluation

```bash
python Radiological_Reconnaissance.py \
  --mode eval \
  --zone_dxf raw_data/zone_dxf/prior.dxf \
  --survey_shp raw_data/survey_shp/survey.shp \
  --baseline_csv raw_data/baseline_csv/baseline.csv \
  --out_dir outputs/eval_run \
  --seed 42
```

### 2. Export payloads only

```bash
python Radiological_Reconnaissance.py \
  --mode eval \
  --run_stage payload_only \
  --zone_dxf raw_data/zone_dxf/prior.dxf \
  --survey_shp raw_data/survey_shp/survey.shp \
  --baseline_csv raw_data/baseline_csv/baseline.csv \
  --out_dir outputs/eval_payloads \
  --seed 42
```

### 3. Rebuild figures/results from saved payloads

```bash
python Radiological_Reconnaissance.py \
  --mode eval \
  --run_stage rebuild_only \
  --payload_source outputs/eval_payloads \
  --zone_dxf raw_data/zone_dxf/prior.dxf \
  --survey_shp raw_data/survey_shp/survey.shp \
  --baseline_csv raw_data/baseline_csv/baseline.csv \
  --out_dir outputs/eval_rebuilt \
  --seed 42
```

### 4. Footprint sweep experiment

```bash
python Radiological_Reconnaissance.py \
  --mode footprint_sweep \
  --measurement_diameter_sweep_m 5,10,15,22 \
  --zone_dxf raw_data/zone_dxf/prior.dxf \
  --survey_shp raw_data/survey_shp/survey.shp \
  --baseline_csv raw_data/baseline_csv/baseline.csv \
  --out_dir outputs/footprint_sweep \
  --seed 42
```

### 5. VSP track mode

```bash
python Radiological_Reconnaissance.py \
  --mode vsp_track \
  --zone_dxf raw_data/zone_dxf/prior.dxf \
  --survey_shp raw_data/survey_shp/survey.shp \
  --vsp_csv raw_data/vsp_csv/vsp_points.csv \
  --out_dir outputs/vsp_track \
  --seed 42
```

## Important options

Key command-line options include:

- `--mode {paper_figures, eval, vsp_track, footprint_sweep}`
- `--run_stage {legacy, payload_only, rebuild_only, full_pipeline}`
- `--truth_model {legacy, puff}`
- `--forecast_model {none, puff}`
- `--measurement_diameter_m`
- `--gp_n0`
- `--gp_max_samples`
- `--eval_mc`
- `--w_t`, `--w_fr`, `--w_fh`

## Key outputs

Typical outputs produced by the evaluation pipeline include:

- `truth_field.png`
- `baseline.png`
- `adaptive.png`
- `mc_summary.csv`
- `mc_summary_aggregated.csv`
- `mc_weighted_summary.csv`
- `adaptive_prefix_curve.csv`
- `adaptive_prefix_curve_aggregated.csv`
- `adaptive_risk_trace.csv`
- `fatal_miss_curve.png`
- `pareto_front.png`
- `lambda_heatmap_summary.csv`
- `footprint_sweep_summary.csv`
- `footprint_sweep_manifest.csv`
- `vsp_track.png`
- `ai_waypoints.csv`

## Method summary

The framework does **not** directly retrain the forecast field itself. Instead, it models the **residual** between forecast-consistent measurements and observed measurements using GPR, and then selects the next measurement location based on:

- threshold-sensitive classification relevance,
- posterior uncertainty,
- hot-prior relevance,
- frontier / boundary sensitivity, and
- route-aware travel penalties.

This turns adaptive surveying into a constrained operational routing problem rather than a pure interpolation task.

## Reproducibility notes

For manuscript-oriented reproduction, the following settings are especially important:

- operational zone labels: `Cold`, `Warm`, `Hot`
- default PAG thresholds in the code: `1.0, 5.0`
- effective measurement diameter: **22 m**
- reference operational weights in the manuscript: **wmiss = 1.0, wt = 1.0, wovercall = 0.3**
- reported manuscript protocol: **5 random seeds × 5 Monte Carlo replications = 25 paired trials**

**Note:** the script default for `--eval_mc` is `20`, so if you want to align the repository example with the manuscript protocol, explicitly set the Monte Carlo setting used in your published experiment.

## Data availability

If all inputs can be shared publicly:

> The data and code supporting the findings of this study are publicly available in this GitHub repository.

If some original inputs cannot be redistributed:

> The code and reproducibility workflow are publicly available in this GitHub repository. Shareable study inputs are provided under `raw_data/`. Restricted or non-public source files are described in this repository and are available from the corresponding author subject to institutional and security constraints.

## Code availability

> The code used to generate and analyze the results presented in this study is available in this repository.

## Citation

If this repository is associated with a publication, please cite both:

1. the journal article, and
2. the versioned repository release.

## Contact

For questions regarding the manuscript or repository, please contact the corresponding author listed in the paper.
