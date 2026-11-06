# Revision R1 (Annals of Nuclear Energy, ANUCENE-D-26-01000): reproduction package

All synthetic and field-data numbers in the revised manuscript are produced by the scripts in `scripts/`.
`Radiological_Reconnaissance.py` (repository root) and `raw_data/` are used unchanged.

## Environment
Python 3.10+, packages pinned in `requirements.txt`. `pdftotext` (poppler) is needed only for the field data.

## Synthetic study (Sections 5.1 to 5.6)
| Step | Command | Output |
|---|---|---|
| Reference, Hot-zone and budget runs (25 paired trials each; seeds 42, 123, 456, 789, 1000 x 5) | `python scripts/run_all.py` | `work/<experiment>_seed_<seed>.csv` |
| Ablation (no path penalties, no prior guidance) | `python scripts/run_ablation.py` | `work/no_path_*.csv`, `work/no_prior_*.csv` |
| Statistics (Tables 3 and 4) | `python scripts/synth_stats.py "hot=results/synthetic/hot_seed_*.csv"` (one label=glob per condition) | console |
| Ablation statistics (Table 6) | `python scripts/ablation_stats.py` | console |
| Representative trial (Fig. 4) | `python scripts/fig_representative.py` then `python scripts/make_figures.py` | `figures/` |
| Truth-field montages (Figs. S1, S3) | `python scripts/make_si_figures.py` (needs the saved payloads in `work/out_*`) | `figures/` |

Runs are resumable: a finished (experiment, seed) file is skipped. The per-trial outputs used in the manuscript are in `results/synthetic/` and `results/ablation/`.
Settings that matter: `--forecast_model puff --truth_model legacy --truth_weather_mode none --truth_scale 0.6`; reference scenario `--truth_value_scale 0.5`, Hot-zone scenario `--truth_value_scale 1.0`; total budgets 83/120/160/240 (`--gp_n0 3`, `--gp_max_samples` 80/117/157/237).
Note (stated in the manuscript, Section 2.2): with these settings the puff forecast ensemble yields no concentration above background inside the survey domain, so the prior is constant in every trial.

## Field validation (Section 6)
The Kamenna data (IAEA-TECDOC-2001, Appendix I, Tables 29 and 30) are not redistributed. Download
https://www-pub.iaea.org/MTCD/Publications/PDF/TE-2001web.pdf and run
```
python scripts/extract_kamenna.py TE-2001web.pdf
python scripts/kamenna_validation.py --threshold_mode fulldata
python scripts/kamenna_validation.py --threshold_mode pilot
```
Results used in the manuscript are in `results/field_validation/`.
