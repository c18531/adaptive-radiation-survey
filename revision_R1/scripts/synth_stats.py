"""Statistics for synthetic experiments (same tests as manuscript Table 4)."""
import sys, glob
import numpy as np, pandas as pd
from scipy import stats
def cd(x,y):
    p=np.sqrt(((len(x)-1)*np.var(x,ddof=1)+(len(y)-1)*np.var(y,ddof=1))/(len(x)+len(y)-2)); return (np.mean(x)-np.mean(y))/p
def cliff(x,y): return (sum(xi>yj for xi in x for yj in y)-sum(xi<yj for xi in x for yj in y))/(len(x)*len(y))
def cp(k,n):
    return (stats.beta.ppf(0.025,k,n-k+1) if k>0 else 0.0, stats.beta.ppf(0.975,k+1,n-k) if k<n else 1.0)
def summarize(files, label):
    d=pd.concat([pd.read_csv(f) for f in files])
    a=d[d.design=='adaptive'].sort_values(['seed','run']).reset_index(drop=True)
    b=d[d.design=='baseline'].sort_values(['seed','run']).reset_index(drop=True)
    print(f'=== {label}: trials/design {len(a)}, truth has Hot {int(a.truth_has_hot.sum())}')
    for m in ['zone_accuracy','action_recall','n_used','total_time_min']:
        x,y=a[m].values,b[m].values; u=stats.mannwhitneyu(x,y,alternative='two-sided')
        print(f"  {m}: adapt {x.mean():.3f}±{x.std(ddof=1):.3f} | base {y.mean():.3f}±{y.std(ddof=1):.3f} | MWU p={u.pvalue:.2e} d={cd(x,y):.2f} cliff={cliff(x,y):.2f}")
    m=(a.truth_has_hot==1).values; n=int(m.sum())
    if n>1:
        x,y=a.hot_recall.values[m],b.hot_recall.values[m]; diff=x-y
        w=stats.wilcoxon(x,y) if np.any(diff!=0) else None
        rng=np.random.default_rng(42); bs=[rng.choice(diff,len(diff)).mean() for _ in range(10000)]
        print(f"  hot_recall (n={n}): adapt {x.mean():.3f}±{x.std(ddof=1):.3f} | base {y.mean():.3f}±{y.std(ddof=1):.3f} | Wilcoxon p={w.pvalue:.2e} | boot95 diff [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]")
    fa=int(((a.truth_has_hot==1)&(a.pred_has_hot==0)).sum()); fb=int(((b.truth_has_hot==1)&(b.pred_has_hot==0)).sum())
    if n>0:
        f=stats.fisher_exact([[fa,n-fa],[fb,n-fb]])
        print(f"  fatal miss: adapt {fa}/{n} CP95 {cp(fa,n)[0]:.3f}-{cp(fa,n)[1]:.3f} | base {fb}/{n} | Fisher p={f.pvalue:.3f}")
if __name__=='__main__':
    for arg in sys.argv[1:]:
        label,pat=arg.split('=',1); summarize(sorted(glob.glob(pat)),label)
