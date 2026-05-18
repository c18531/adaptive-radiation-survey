# Forecast-informed adaptive radiation surveying with residual Gaussian process regression for rapid radiological contamination assessment

This repository provides the code and reproducibility workflow for the study:

> **Wooseok Choi, Myeongsik Shin, and Ku Kang**  
> *Forecast-informed adaptive radiation surveying with residual Gaussian process regression for rapid radiological contamination assessment*

## Overview

Rapid radiological contamination assessment after a nuclear or radiological release requires efficient use of sparse measurements under uncertainty. This repository implements a forecast-informed adaptive radiation survey framework that combines:

- atmospheric-dispersion forecast priors,
- residual Gaussian process regression (GPR),
- risk-aware and path-aware adaptive sampling,
- PAG-based Hot/Warm/Cold zone classification, and
- comparison with a structured MARSSIM/VSP-style survey baseline.

The workflow is intended for simulation-based research on nuclear emergency monitoring and radiation-protection survey planning. It is not intended to provide operational emergency-response instructions without site-specific validation and regulatory review.

## Main capabilities

- Forecast-informed radiological contamination assessment using atmospheric-dispersion priors
- Residual GPR updating in measurement space
- Adaptive next-measurement selection using uncertainty, threshold relevance, and travel-cost terms
- Structured MARSSIM/VSP-style baseline comparison
- Monte Carlo evaluation of zone accuracy, Hot-zone recall, Action recall, sample count, path length, and survey time
- Prefix-based diagnostics for adaptive survey trajectories
- Payload export and rebuild workflow for reproducible figure/result generation
- Footprint sensitivity and VSP-track experiments

## Repository structure

```text
.
├── Radiological_Reconnaissance.py     # Main simulation and evaluation script
├── raw_data/
│   ├── Areas_dxf/                     # DXF prior/zone geometry files
│   ├── Areas_shp/                     # Optional survey-domain shapefiles
│   ├── Areas_shx/                     # Optional shapefile companion files
│   ├── Case2_DXF/                     # Case-specific DXF files
│   └── baseline_csv/                  # MARSSIM/VSP-style baseline station plans
├── outputs/                           # Generated outputs; ignored by git when possible
├── paper/                             # Manuscript and supplementary files, if shared
└── README.md
```

## Requirements

Python 3.10 or higher is recommended.

Main Python packages:

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

For a cleaner environment, use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS/WSL
# .venv\Scripts\activate       # Windows PowerShell
pip install numpy pandas matplotlib scipy shapely geopandas scikit-learn
```

## Required input files

The main script expects the following study inputs:

- `--zone_dxf`: DXF file used for the contamination prior or Warm/Hot geometry
- `--baseline_csv`: structured baseline station-plan CSV
- `--survey_shp`: optional survey-domain shapefile

Optional shapefile companion files can be passed through:

- `--survey_shx`
- `--survey_dbf`

The code also accepts `--vsp_csv` for VSP-track experiments or as a fallback baseline input when `--baseline_csv` is omitted.

## Quick start

### Standard evaluation

```bash
python Radiological_Reconnaissance.py \
  --mode eval \
  --zone_dxf raw_data/Case2_DXF/Case2.DXF \
  --baseline_csv raw_data/baseline_csv/baseline.csv \
  --measurement_diameter_m 22 \
  --eval_mc 25 \
  --baseline_max_samples 480 \
  --seed 7 \
  --out_dir outputs/eval_case2
```

### Manuscript-style evaluation example

The command below follows the main simulation setting used for the manuscript-style Case 2 evaluation. File paths should be changed to match your local repository layout.

```bash
python Radiological_Reconnaissance.py \
  --zone_dxf raw_data/Case2_DXF/Case2.DXF \
  --baseline_csv raw_data/baseline_csv/baseline.csv \
  --mode eval \
  --eval_mc 25 \
  --measurement_diameter_m 22 \
  --baseline_candidate_source vsp \
  --baseline_vsp_serpentine_axis col \
  --baseline_vsp_line_tol_m 22 \
  --baseline_vsp_end_corner upper_right \
  --baseline_max_samples 480 \
  --truth_model legacy \
  --forecast_model puff \
  --truth_weather_mode warp \
  --truth_meteo_advect_scale 0 \
  --truth_meteo_blur_scale 0.0002 \
  --truth_value_scale 1.0 \
  --truth_sources_max 1 \
  --truth_shape_fill_fraction 0.08 \
  --truth_blur_sigma 1.2 \
  --truth_prior_softness 0.35 \
  --truth_shape_follow_strength 0.65 \
  --truth_lock_hot_anchor \
  --truth_hot_anchor_x 0 \
  --truth_hot_anchor_y 0 \
  --truth_meteo_tilt_max_deg 5 \
  --truth_theta_jitter_deg 0 \
  --truth_preserve_hot_core_during_warp \
  --seed 7 \
  --forecast_members 16 \
  --forecast_dt_min 30 \
  --gp_acq_mode posterior \
  --gp_stop_acq -1 \
  --no-gp_metric_stop \
  --grid_n 121 \
  --gp_n0 3 \
  --gp_max_samples 80 \
  --gp_candidate_source grid \
  --gp_candidate_spacing_m 20 \
  --gp_min_spacing_m 20 \
  --gp_hot_prior_weight 0.80 \
  --gp_frontier_weight 1.20 \
  --gp_travel_weight 0.03 \
  --gp_turn_weight 0.0 \
  --gp_max_step_m 160 \
  --gp_global_every 1 \
  --gp_gamma 0.03 \
  --meteo_dirs_deg "260,250,245,240,235,230,225,220" \
  --meteo_speeds_mps "4.5,5.0,6.0,5.5,4.0,3.5,4.5,5.0" \
  --meteo_stabilities "D,D,D,E,E,D,D,D" \
  --meteo_rain_mmph "0,0,0,0.5,1.2,0.5,0,0" \
  --out_dir outputs/case2_eval_mc25
```

### Payload-only export

```bash
python Radiological_Reconnaissance.py \
  --mode eval \
  --run_stage payload_only \
  --zone_dxf raw_data/Case2_DXF/Case2.DXF \
  --baseline_csv raw_data/baseline_csv/baseline.csv \
  --eval_mc 25 \
  --seed 7 \
  --out_dir outputs/payloads_case2
```

### Rebuild figures and summaries from saved payloads

```bash
python Radiological_Reconnaissance.py \
  --mode eval \
  --run_stage rebuild_only \
  --payload_source outputs/payloads_case2 \
  --zone_dxf raw_data/Case2_DXF/Case2.DXF \
  --baseline_csv raw_data/baseline_csv/baseline.csv \
  --out_dir outputs/rebuilt_case2
```

### Footprint sensitivity experiment

```bash
python Radiological_Reconnaissance.py \
  --mode footprint_sweep \
  --measurement_diameter_sweep_m 5,10,15,22 \
  --zone_dxf raw_data/Case2_DXF/Case2.DXF \
  --baseline_csv raw_data/baseline_csv/baseline.csv \
  --eval_mc 25 \
  --seed 7 \
  --out_dir outputs/footprint_sweep
```

### VSP-track experiment

```bash
python Radiological_Reconnaissance.py \
  --mode vsp_track \
  --zone_dxf raw_data/Case2_DXF/Case2.DXF \
  --vsp_csv raw_data/baseline_csv/baseline.csv \
  --seed 7 \
  --out_dir outputs/vsp_track
```

## Important options

Key command-line options include:

- `--mode {paper_figures, eval, vsp_track, footprint_sweep}`
- `--run_stage {legacy, payload_only, rebuild_only, full_pipeline}`
- `--truth_model {legacy, puff}`
- `--forecast_model {none, puff}`
- `--zone_dxf`
- `--baseline_csv`
- `--survey_shp`
- `--measurement_diameter_m`
- `--baseline_max_samples`
- `--gp_n0`
- `--gp_max_samples`
- `--gp_acq_mode {posterior, oracle_onestep}`
- `--eval_mc`
- `--w_t`, `--w_fr`, `--w_fh`

For publishable or deployable policy comparisons, use `--gp_acq_mode posterior`. The `oracle_onestep` mode uses truth-field information and is intended only as a simulation diagnostic.

## Key outputs

Typical outputs include:

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
- `zone_boundary_summary.csv`
- `baseline_cross_phase_prune_stats.csv`
- `phase2_removed_within_22m.csv`, `near_removed_within_22m.csv`, `all_removed_within_22m.csv`, when available

## Hot-zone recall post-processing

When HotRecall is reported only for trials containing a true Hot zone, trials without any true Hot-zone grid cells should be excluded because HotRecall is undefined for those cases.

Example post-processing:

```bash
python summarize_hot_recall.py outputs/case2_eval_mc25
```

The script should read `mc_summary.csv`, filter runs with `truth_has_hot == 1`, and compute the mean and standard deviation of `hot_recall` for `baseline` and `adaptive` separately.

## Method summary

The framework does not directly retrain the forecast field itself. Instead, it models the residual between forecast-consistent measurements and observed measurements using GPR. The posterior field is combined with PAG-based thresholds to support Hot/Warm/Cold zone classification. The next measurement location is selected based on threshold relevance, posterior uncertainty, Hot-prior relevance, boundary sensitivity, and travel-cost terms.

This turns adaptive surveying into a constrained survey-planning problem rather than a pure interpolation task.

## Reproducibility notes

For manuscript-oriented reproduction, the following settings are important:

- zone labels: `Cold`, `Warm`, `Hot`
- default PAG thresholds in the code: `1.0, 5.0`
- effective measurement diameter: `22 m`
- default script value for `--eval_mc`: `20`
- manuscript-style Monte Carlo setting: `--eval_mc 25`
- baseline limit used in the main comparison: `--baseline_max_samples 480`
- adaptive sample setting in the efficient-survey comparison: `--gp_n0 3`, `--gp_max_samples 80`

The script exports `mc_summary.csv`, which includes per-run `truth_has_hot`, `pred_has_hot`, `hot_recall`, `n_used`, and time metrics. These columns are useful for reproducing Hot-zone recall analyses and trial-level diagnostic checks.

## Data and code availability

The source code and reproducibility workflow supporting this study are provided in this repository for manuscript review and research reproducibility.

Shareable input files, example configurations, and representative output files are provided where redistribution is permitted. Some original source files, geospatial inputs, scenario configurations, or institutional datasets used in the manuscript may not be publicly redistributed because of organizational or security restrictions. These restricted inputs are described in the repository, and access may be requested from the corresponding author subject to institutional approval.

## Citation

If you use this repository, please cite the repository. A manuscript associated with this repository is currently under preparation/submission, and the final journal citation will be added after publication.

```bibtex
@software{choi_adaptive_radiation_survey_code,
  title  = {Forecast-informed adaptive radiation surveying with residual Gaussian process regression},
  author = {Choi, Wooseok and Shin, Myeongsik and Kang, Ku},
  year   = {2026},
  url    = {https://github.com/c18531/adaptive-radiation-survey},
  note   = {Code repository}
}
```

```bibtex
@unpublished{choi_adaptive_radiation_survey_manuscript,
  title  = {Forecast-informed adaptive radiation surveying with residual Gaussian process regression for rapid radiological contamination assessment},
  author = {Choi, Wooseok and Shin, Myeongsik and Kang, Ku},
  year   = {2026},
  note   = {Manuscript under preparation/submission}
}
```

After publication, this section will be updated with the final journal citation.

## License

A repository license will be added after institutional and co-author review.

Until a license is added, the source code is provided for manuscript review and reproducibility inspection only. Redistribution, modification, commercial use, or incorporation into other projects should not be assumed without written permission from the authors.

Some input files, scenario configurations, geospatial data, or institutional datasets used in the manuscript may not be publicly redistributable because of organizational or security restrictions. Where possible, representative examples and command-line templates are provided to support reproducibility.

## Contact

For questions about the code, experiments, or manuscript, please contact:

**Wooseok Choi**  
Department of Nuclear and Quantum Engineering  
Korea Advanced Institute of Science and Technology (KAIST)  
Daejeon, Republic of Korea  

**Corresponding author:**  
**Ku Kang**  
CBRN Defense Research Institute  
Seoul, Republic of Korea  
Email: bisu9082@gmail.com
