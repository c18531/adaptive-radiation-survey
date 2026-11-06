"""
Retrospective field validation on Kamenna Tests 5/3P and 6/4P (IAEA-TECDOC-2001, Tables 29-30).

Design: claude_research/choi_R1/evidence/step_s6_kamenna/DESIGN_PROPOSAL.md (approved 2026-11-03).

Survey replay
  - Candidates are all measured locations of one test. A design "visits" locations one at a time;
    a visit reveals the tabulated deposition at that location (log10 Bq/m2). Nothing else is revealed.
  - After the budget is spent, every unvisited location is classified from the posterior and compared
    with its own measured class. No interpolated field is used as reference.

Method under test (reused unchanged from Radiological_Reconnaissance.py)
  - fit_gp: residual GP on (observation - prior)
  - choose_next_point_mile_pag_residual: path-aware acquisition of Section 3.4
  - classify_pag_from_posterior: confidence-based zoning of Section 3.3 (Cold is the fallback label)

Site-specific settings (fixed before running)
  - Response: log10(deposition). Thresholds: per-test median (Warm) and 90th percentile (Hot) of all
    measurements, used as fixed constants. Sensitivity run: thresholds from the 8 pilot readings.
  - Prior: inverse-distance weighting (power 2) of 8 maximin pilot readings; prior std = 0.
  - Point measurements (measurement_diameter 2 m), meas_sigma 0.05 log10 units.
  - Spatial scaling: candidate spacing 5 m (90th pct nearest-neighbour distance), max step 8.5 m
    (= 34/20 x 5, the synthetic ratio), min spacing 0.5 m (only exact revisits are excluded).
  - Budgets: 15, 30, 50 % of locations (pilots and start point included).
  - Starts: 3 random start locations per test (seed 0..2). Random comparator: 5 seeds per start.
  - Time model: 1 min per observation + path length / 250 m/min.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "revision_R1"
sys.path.insert(0, str(REPO))
import Radiological_Reconnaissance as R  # noqa: E402

DATA = ROOT / "data_kamenna"   # created by extract_kamenna.py (IAEA data are not redistributed)
TESTS = {"5_3P": DATA / "kamenna_5_3P.csv", "6_4P": DATA / "kamenna_6_4P.csv"}
BUDGET_FRACS = (0.15, 0.30, 0.50)
N_PILOT = 8
N_STARTS = 3
N_RANDOM = 5
MEAS_SIGMA = 0.05
CONFIDENCE = 0.975
SPEED_M_PER_MIN = 250.0
STATION_MIN = 1.0


# ----------------------------------------------------------------------------- helpers
def maximin_pilots(xy: np.ndarray, k: int) -> list[int]:
    """Coordinate-only maximin selection, starting from the location nearest the centroid."""
    c = xy.mean(axis=0)
    chosen = [int(np.argmin(np.linalg.norm(xy - c, axis=1)))]
    while len(chosen) < k:
        d = np.min(np.linalg.norm(xy[:, None, :] - xy[chosen][None, :, :], axis=2), axis=1)
        d[chosen] = -1.0
        chosen.append(int(np.argmax(d)))
    return chosen


def idw(xy_obs: np.ndarray, z_obs: np.ndarray, xy: np.ndarray, power: float = 2.0) -> np.ndarray:
    d = np.linalg.norm(xy[:, None, :] - xy_obs[None, :, :], axis=2)
    out = np.empty(len(xy))
    exact = d < 1e-9
    for i in range(len(xy)):
        if exact[i].any():
            out[i] = z_obs[exact[i]][0]
        else:
            w = 1.0 / d[i] ** power
            out[i] = float(np.sum(w * z_obs) / np.sum(w))
    return out


class PointPrior:
    """Callable with the interface expected by the acquisition: input columns (y, x)."""

    def __init__(self, xy: np.ndarray, values: np.ndarray):
        self.lookup = {(round(float(x), 4), round(float(y), 4)): float(v) for (x, y), v in zip(xy, values)}

    def __call__(self, yx: np.ndarray) -> np.ndarray:
        return np.array([self.lookup[(round(float(x), 4), round(float(y), 4))] for y, x in np.asarray(yx)])


def path_length(xy_seq: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(xy_seq, axis=0), axis=1))) if len(xy_seq) > 1 else 0.0


def nn_order(xy: np.ndarray, start: int, pool: list[int]) -> list[int]:
    order, cur, left = [], start, list(pool)
    while left:
        j = min(left, key=lambda k: np.linalg.norm(xy[k] - xy[cur]))
        order.append(j); left.remove(j); cur = j
    return order


def posterior(xy, z, visited, prior, cfg):
    X = xy[visited]
    res = z[visited] - prior[visited]
    gp = R.fit_gp(X, res, cfg, n_restarts=2)
    mu_r, sd_r = gp.predict(xy, return_std=True)
    return gp, prior + mu_r, sd_r


def evaluate(xy, z, visited, prior, cfg, thr):
    _, mu, sd = posterior(xy, z, visited, prior, cfg)
    pred = R.classify_pag_from_posterior(mu, sd, thr, CONFIDENCE)
    truth = np.digitize(z, thr)  # 0 Cold, 1 Warm, 2 Hot
    un = np.setdiff1d(np.arange(len(z)), visited)
    t, p = truth[un], pred[un]
    hot_t, hot_p = t == 2, p == 2
    act_t, act_p = t >= 1, p >= 1
    return {
        "n_unvisited": int(len(un)),
        "n_true_hot_unvisited": int(hot_t.sum()),
        "n_true_action_unvisited": int(act_t.sum()),
        "hot_recall": float(np.mean(hot_p[hot_t])) if hot_t.any() else np.nan,
        "action_recall": float(np.mean(act_p[act_t])) if act_t.any() else np.nan,
        "hot_precision": float(np.mean(hot_t[hot_p])) if hot_p.any() else np.nan,
        "action_precision": float(np.mean(act_t[act_p])) if act_p.any() else np.nan,
        "zone_accuracy": float(np.mean(t == p)),
        "pred_any_hot": int(hot_p.any()),
    }


# ----------------------------------------------------------------------------- designs
def run_adaptive(xy, z, prior, cfg, adapt, ops, start, pilots, budget):
    seq = [start] + [p for p in nn_order(xy, start, [p for p in pilots if p != start])]
    prior_f = PointPrior(xy, prior)
    zero_f = PointPrior(xy, np.zeros(len(xy)))
    hot_prior_mask = prior >= ops.thresholds[-1]
    while len(seq) < budget:
        gp, _, _ = posterior(xy, z, seq, prior, cfg)
        x_next, _ = R.choose_next_point_mile_pag_residual(
            gp=gp, X_obs=xy[seq], domain_xy=xy, hot_prior_mask=hot_prior_mask, ops=ops, cfg=cfg,
            adapt=adapt, forecast_mean_interp=prior_f, forecast_std_interp=zero_f, route_rank=None)
        j = int(np.argmin(np.linalg.norm(xy - x_next[None, :], axis=1)))
        if j in seq:  # safety: should not happen with min_spacing > 0
            raise RuntimeError("acquisition revisited a location")
        seq.append(j)
    return seq


def run_nn(xy, start, pilots, budget):
    seq = [start] + nn_order(xy, start, [p for p in pilots if p != start])
    rest = [i for i in range(len(xy)) if i not in seq]
    seq += nn_order(xy, seq[-1], rest)[: budget - len(seq)]
    return seq


def run_random(xy, start, pilots, budget, rng):
    seq = [start] + nn_order(xy, start, [p for p in pilots if p != start])
    rest = np.array([i for i in range(len(xy)) if i not in seq])
    picks = list(rng.choice(rest, size=budget - len(seq), replace=False))
    # visiting order for path length: nearest-neighbour through the random set
    seq += nn_order(xy, seq[-1], [int(p) for p in picks])
    return seq


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "field_validation"))
    ap.add_argument("--threshold_mode", choices=["fulldata", "pilot"], default="fulldata")
    ap.add_argument("--tests", default="5_3P,6_4P")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for test in a.tests.split(","):
        df = pd.read_csv(TESTS[test])
        xy = df[["x_m", "y_m"]].to_numpy(float)
        z = np.log10(df["deposition_Bq_m2"].to_numpy(float))
        n = len(z)
        pilots = maximin_pilots(xy, N_PILOT)
        prior = idw(xy[pilots], z[pilots], xy)
        if a.threshold_mode == "fulldata":
            thr = (float(np.median(z)), float(np.quantile(z, 0.90)))
        else:
            thr = (float(np.median(z[pilots])), float(np.quantile(z[pilots], 0.90)))
        ops = R.OperationalCriteria(thresholds=thr, confidence=CONFIDENCE, labels=("Cold", "Warm", "Hot"))
        bounds = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
        cfg = R.FieldConfig(bounds=bounds, meas_sigma=MEAS_SIGMA, measurement_diameter_m=2.0, measurement_model="point")
        adapt = R.AdaptiveConfig(n0=N_PILOT + 1, candidate_spacing_m=5.0, min_spacing_m=0.5, max_step_m=8.5)

        rng0 = np.random.default_rng(0)
        non_pilot = [i for i in range(n) if i not in pilots]
        starts = [int(s) for s in rng0.choice(non_pilot, size=N_STARTS, replace=False)]

        for frac in BUDGET_FRACS:
            budget = int(round(frac * n))
            for si, start in enumerate(starts):
                designs = {"adaptive": run_adaptive(xy, z, prior, cfg, adapt, ops, start, pilots, budget),
                           "nearest_neighbour": run_nn(xy, start, pilots, budget)}
                for rs in range(N_RANDOM):
                    designs[f"random_{rs}"] = run_random(xy, start, pilots, budget, np.random.default_rng(100 + 10 * si + rs))
                for name, seq in designs.items():
                    assert len(seq) == budget and len(set(seq)) == budget
                    m = evaluate(xy, z, seq, prior, cfg, thr)
                    pl = path_length(xy[seq])
                    rows.append({"test": test, "threshold_mode": a.threshold_mode, "budget_frac": frac,
                                 "budget": budget, "start_index": si, "design": name.split("_")[0] if name.startswith("random") else name,
                                 "replicate": name, "n_obs": len(seq), "path_m": pl,
                                 "time_min": STATION_MIN * len(seq) + pl / SPEED_M_PER_MIN, **m})
                print(test, frac, si, "done", flush=True)
        meta = {"test": test, "n": n, "pilots": pilots, "starts": starts, "thresholds_log10": thr,
                "threshold_mode": a.threshold_mode}
        (out / f"meta_{test}_{a.threshold_mode}.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame(rows).to_csv(out / f"kamenna_results_{a.threshold_mode}.csv", index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
