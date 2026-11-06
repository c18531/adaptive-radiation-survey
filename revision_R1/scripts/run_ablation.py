"""Ablation, Hot scenario, budget 83, forecast puff. Resumable. Core code unchanged."""
import sys, time
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[2]          # repository root
R1 = REPO / "revision_R1"
RAW = REPO / "raw_data"
WORK = R1 / "work"; WORK.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))
D = WORK
from Radiological_Reconnaissance import build_argparser, run_eval_mode
SEEDS = [42, 123, 456, 789, 1000]
VARS = [("no_path", ["--gp_travel_weight", "0", "--gp_turn_weight", "0"]),
        ("no_prior", ["--gp_hot_prior_weight", "0", "--gp_frontier_weight", "0"])]
for name, extra in VARS:
    for s in SEEDS:
        out = D / f"{name}_seed_{s}.csv"
        if out.exists(): continue
        od = D / f"out_{name}_{s}"; od.mkdir(exist_ok=True)
        a = build_argparser().parse_args([
            "--zone_dxf", str(RAW/"Case2.DXF"), "--survey_shp", str(RAW/"Areas.shp"),
            "--baseline_csv", str(RAW/"baseline.csv"), "--out_dir", str(od),
            "--seed", str(s), "--mode", "eval", "--eval_mc", "5", "--grid_n", "101",
            "--gp_max_samples", "80", "--gp_n0", "3", "--truth_scale", "0.6",
            "--truth_model", "legacy", "--forecast_model", "puff",
            "--truth_weather_mode", "none", "--truth_value_scale", "1.0"] + extra)
        t = time.time(); run_eval_mode(a)
        mc = pd.read_csv(od / "mc_summary.csv"); mc["seed"] = s; mc["variant"] = name
        mc.to_csv(out, index=False)  # copy to revision_R1/results/ to reuse with the statistics scripts; print(f"{name} seed {s} done {time.time()-t:.0f}s", flush=True)
print("ALL DONE", flush=True)
