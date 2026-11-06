"""Revision figures from deposited run outputs (no new simulation except representative_trial.pkl)."""
import glob, pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
R1 = Path(__file__).resolve().parents[1]; FIG = R1/"figures"; EV = R1/"results"/"synthetic"
matplotlib.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
  'legend.fontsize':7,'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'axes.spines.top':False,
  'axes.spines.right':False,'axes.linewidth':0.8,'legend.frameon':False,'savefig.dpi':600,'savefig.bbox':'tight','savefig.pad_inches':0.05})
OI = ['#E69F00','#56B4E9','#009E73','#0072B2','#D55E00','#CC79A7']
ZC = ListedColormap(['#DDEEDD', '#F2C14E', '#C0392B']); ZN = BoundaryNorm([-0.5,0.5,1.5,2.5], 3)

def fig_representative():
    d = pickle.load(open(FIG/"representative_trial.pkl","rb")); x0,x1,y0,y1 = d["bounds"]; Z = d["Z"]
    fig, axs = plt.subplots(3,1, figsize=(7.2,6.4), constrained_layout=True)
    im = axs[0].imshow(Z, origin="lower", extent=[x0,x1,y0,y1], cmap="viridis", aspect="auto")
    axs[0].contour(np.linspace(x0,x1,Z.shape[1]), np.linspace(y0,y1,Z.shape[0]), np.nan_to_num(Z), levels=list(d["thr"]), colors=['white','black'], linewidths=0.8)
    fig.colorbar(im, ax=axs[0], label="Field value (Warm-threshold units)")
    axs[0].set_title("(a) Truth field with Warm (white) and Hot (black) contours", loc="left")
    for ax, key, lab in [(axs[1],"base","(b) Structured baseline"),(axs[2],"adap","(c) Adaptive design")]:
        r = d[key]; ax.imshow(r["pred_zone"], origin="lower", extent=[x0,x1,y0,y1], cmap=ZC, norm=ZN, aspect="auto")
        ax.contour(np.linspace(x0,x1,Z.shape[1]), np.linspace(y0,y1,Z.shape[0]), (r["truth_zone"]==2).astype(float), levels=[0.5], colors='k', linewidths=0.9, linestyles='--')
        X = np.asarray(r["X_obs"]); ax.plot(X[:,0], X[:,1], '-', color=OI[3], lw=0.6, alpha=0.8); ax.plot(X[:,0], X[:,1], 'o', ms=1.6, color=OI[3])
        hr = r["hot_recall"]; ax.set_title(f"{lab}: {len(X)} observations, {r['total_time_min']:.0f} min, ZCA {r['zone_accuracy']:.2f}, Hot Recall {hr:.2f}", loc="left")
    for ax in axs: ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    h = [plt.Rectangle((0,0),1,1,color=c) for c in ['#DDEEDD','#F2C14E','#C0392B']]+[plt.Line2D([],[],color='k',ls='--')]
    axs[2].legend(h, ["Predicted Cold","Predicted Warm","Predicted Hot","True Hot boundary"], ncol=4, loc="upper center", bbox_to_anchor=(0.5,-0.32))
    fig.savefig(FIG/"fig_04_representative_trial.png"); plt.close(fig)

def fig_boxplot():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(EV/"default_seed_*.csv"))])
    mets = [("zone_accuracy","Zone classification accuracy"),("action_recall","Action Recall"),("n_used","Observations"),("total_time_min","Modeled mission time (min)")]
    fig, axs = plt.subplots(1,4, figsize=(7.2,2.6), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax,(m,lab) in zip(axs,mets):
        data = [d[d.design=="baseline"][m].values, d[d.design=="adaptive"][m].values]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color="black", lw=1.5))
        for b,c in zip(bp["boxes"],[OI[1],OI[0]]): b.set_facecolor(c); b.set_alpha(0.8)
        for i,v in enumerate(data): ax.plot(np.full(len(v),i+1)+rng.uniform(-0.12,0.12,len(v)), v, 'o', ms=2, color='k', alpha=0.6)
        ax.set_xticks([1,2]); ax.set_xticklabels(["Baseline","Adaptive"]); ax.set_title(lab)
    fig.savefig(FIG/"fig_05_boxplot_reference.png"); plt.close(fig)

def fig_budget():
    rows = []
    for name, n in [("hot",83),("hot_b120",120),("hot_b160",160),("hot_b240",240)]:
        d = pd.concat([pd.read_csv(f) for f in glob.glob(str(EV/f"{name}_seed_*.csv"))])
        a = d[d.design=="adaptive"]; b = d[d.design=="baseline"]
        rows.append(dict(n=n, hr=a.hot_recall.mean(), hrs=a.hot_recall.std(ddof=1), ar=a.action_recall.mean(), ars=a.action_recall.std(ddof=1),
                         miss=int(((a.truth_has_hot==1)&(a.pred_has_hot==0)).sum()), bhr=b.hot_recall.mean(), bar=b.action_recall.mean(), bn=b.n_used.mean()))
    t = pd.DataFrame(rows); t.to_csv(FIG/"fig_06_budget_values.csv", index=False)
    fig, axs = plt.subplots(1,3, figsize=(7.2,2.5), constrained_layout=True)
    for ax,(k,ks,bk,lab) in zip(axs[:2],[("hr","hrs","bhr","Hot Recall"),("ar","ars","bar","Action Recall")]):
        ax.errorbar(t.n, t[k], yerr=t[ks], fmt='o-', color=OI[0], capsize=3, lw=1.5, ms=4, label="Adaptive (mean ± SD)")
        ax.axhline(t[bk].iloc[0], color=OI[3], ls='--', lw=1.2, label=f"Baseline ({t.bn.iloc[0]:.0f} obs.)")
        ax.set_ylim(0,1.05); ax.set_xlabel("Adaptive observation budget"); ax.set_ylabel(lab); ax.set_xticks(t.n)
    axs[0].legend(loc="lower right"); axs[0].set_title("(a)", loc="left"); axs[1].set_title("(b)", loc="left")
    axs[2].bar(t.n.astype(str), t.miss, color=OI[4]); axs[2].set_ylim(0, 4); axs[2].set_yticks([0,1,2,3,4]); axs[2].set_ylabel("Complete Hot misses (of 25)")
    axs[2].set_xlabel("Adaptive observation budget"); axs[2].set_title("(c)", loc="left")
    fig.savefig(FIG/"fig_06_budget_curve.png"); plt.close(fig)

if __name__ == "__main__":
    fig_representative(); fig_boxplot(); fig_budget(); print("ok")
