"""Representative single trial (Hot-zone scenario, seed 42 representative run), same settings as run_all.py."""
import sys, pickle
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]          # repository root
R1 = REPO / "revision_R1"
RAW = REPO / "raw_data"
WORK = R1 / "work"; WORK.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))
W = RAW
import Radiological_Reconnaissance as R
a = R.build_argparser().parse_args([
    "--zone_dxf", str(W/"Case2.DXF"), "--survey_shp", str(W/"Areas.shp"), "--baseline_csv", str(W/"baseline.csv"),
    "--out_dir", "/tmp/fig_rep", "--seed", "42", "--mode", "eval", "--eval_mc", "1", "--grid_n", "101",
    "--gp_max_samples", "80", "--gp_n0", "3", "--truth_scale", "0.6", "--truth_model", "legacy",
    "--forecast_model", "puff", "--truth_weather_mode", "none", "--truth_value_scale", "1.0"])
geom = R.build_geometry(a); ops = R.build_ops(a)
Z, meta, base, adap = R.run_single(geom, ops, R.build_truth_cfg(a), R.build_baseline_cfg(a), R.build_adaptive_cfg(a), a.seed,
    R.load_baseline_plan(a), None, source_term=R.build_source_term(a), meteo_steps=R.build_meteo_schedule(a),
    forecast_cfg=R.build_forecast_cfg(a), timing_cfg=R.build_timing_cfg(a), truth_weather_mode="none",
    truth_meteo_advect_scale=a.truth_meteo_advect_scale, truth_meteo_blur_scale=a.truth_meteo_blur_scale,
    truth_model="legacy", weight_cfg=R.build_weight_cfg(a))
out = {"Z": Z, "bounds": geom.cfg.bounds, "thr": ops.thresholds,
       "base": {k: getattr(base, k) for k in ["pred_zone", "truth_zone", "X_obs", "zone_accuracy", "hot_recall", "action_recall", "total_time_min"]},
       "adap": {k: getattr(adap, k) for k in ["pred_zone", "truth_zone", "X_obs", "zone_accuracy", "hot_recall", "action_recall", "total_time_min"]}}
pickle.dump(out, open(R1/"figures"/"representative_trial.pkl", "wb"))
print({k: (out['base'][k], out['adap'][k]) for k in ["zone_accuracy", "hot_recall", "action_recall", "total_time_min"]}, len(out['base']['X_obs']), len(out['adap']['X_obs']))
