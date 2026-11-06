"""SI figures: truth fields of all 25 trials per scenario (from saved payloads) and two examples."""
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R1=Path(__file__).resolve().parents[1]; W=R1/"work"; FIG=R1/"figures"
matplotlib.rcParams.update({'font.size':7,'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'savefig.dpi':400,'savefig.bbox':'tight'})
SEEDS=[42,123,456,789,1000]
def load(scen,s,r): return np.load(W/f"out_{scen}_{s}"/f"truth_payload_run_{r:04d}.npz")["Z_true"]
for scen,label in [("default","Reference scenario"),("hot","Hot-zone scenario")]:
    fig,axs=plt.subplots(5,5,figsize=(7.2,4.2),constrained_layout=True)
    vmax=max(np.nanmax(load(scen,s,r)) for s in SEEDS for r in range(1,6))
    for i,s in enumerate(SEEDS):
        for r in range(1,6):
            ax=axs[i,r-1]; Z=load(scen,s,r)
            im=ax.imshow(Z,origin="lower",cmap="viridis",vmin=0,vmax=vmax,aspect="auto")
            ax.contour(np.nan_to_num(Z),levels=[1.0,5.0],colors=["white","red"],linewidths=0.5)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"seed {s}, run {r}"+(" (Hot)" if np.nanmax(Z)>=5 else ""),fontsize=6)
    fig.colorbar(im,ax=axs,shrink=0.6,label="Field value (Warm-threshold units)")
    fig.savefig(FIG/f"si_truth_montage_{scen}.png"); plt.close(fig)
print("ok")
