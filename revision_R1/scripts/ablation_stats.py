"""Paired ablation comparison against FULL (Hot scenario, 83 obs, forecast puff)."""
import sys, glob, numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
R=str(Path(__file__).resolve().parents[1])+'/'
def load(pat, label):
    d=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))]); d=d[d.design=='adaptive'].sort_values(['seed','run']).reset_index(drop=True); d['v']=label; return d
full=load(R+'results/synthetic/hot_seed_*.csv','FULL')
variants={'NO_PATH':R+'results/ablation/no_path_seed_*.csv','NO_PRIOR':R+'results/ablation/no_prior_seed_*.csv'}
mets=['path_length_m','travel_time_min','total_time_min','hot_recall','action_recall','zone_accuracy']
print('FULL', {m: f"{full[m].mean():.3f}±{full[m].std(ddof=1):.3f}" for m in mets}, 'fatal', int(((full.truth_has_hot==1)&(full.pred_has_hot==0)).sum()))
for k,pat in variants.items():
    if not glob.glob(pat): print(k,'missing'); continue
    v=load(pat,k)
    assert (v[['seed','run']].values==full[['seed','run']].values).all()
    out=[]
    for m in mets:
        diff=v[m].values-full[m].values
        p=stats.wilcoxon(v[m],full[m]).pvalue if np.any(diff!=0) else 1.0
        out.append(f"{m}: {v[m].mean():.3f}±{v[m].std(ddof=1):.3f} (diff {diff.mean():+.3f}, p={p:.3g})")
    print(k, 'n',len(v),'fatal',int(((v.truth_has_hot==1)&(v.pred_has_hot==0)).sum())); print('   '+'\n   '.join(out))
    pt=100*(v.travel_time_min.mean()-full.travel_time_min.mean())/v.travel_time_min.mean()
    print(f"   FULL travel time shorter than {k} by {pt:.1f}% of {k} travel time; total time shorter by {100*(v.total_time_min.mean()-full.total_time_min.mean())/v.total_time_min.mean():.1f}%")
