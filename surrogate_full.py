import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""Full surrogate experiment: beta dose-response plus two control arms."""
import itertools, numpy as np, pandas as pd
from multiprocessing import Pool
from surrogate_loop import episode, plant_matrix

BETAS = (0.00, 0.25, 0.50, 0.75, 0.90, 1.10, 1.50)
LANDS = ("linear", "smooth", "multimodal", "saturating")
PLANTS = tuple(range(8))
SEEDS = tuple(range(int(os.environ.get("SUR_SEEDS", "40"))))

def one(t):
    pi, land, s = t
    W = plant_matrix(1000 + pi)
    trng = np.random.default_rng(90000 + 37*pi + 7919*s)
    target = trng.uniform(-0.75, 0.75, size=2)
    sd = 5000 + 31*s + 101*pi
    out = []
    for b in BETAS:
        r = episode(b, W, land, target, seed=sd)
        out.append((f"beta_{b:.2f}", b, pi, land, s, r["distance_auc"], r["success"],
                    r["repair_rate"], r["mean_step"]))
    r = episode(0.0, W, land, target, seed=sd, max_slew=True)
    out.append(("maxslew", 0.0, pi, land, s, r["distance_auc"], r["success"],
                r["repair_rate"], r["mean_step"]))
    r = episode(0.90, W, land, target, seed=sd, carry="all")
    out.append(("carry_all", 0.90, pi, land, s, r["distance_auc"], r["success"],
                r["repair_rate"], r["mean_step"]))
    return out

if __name__ == "__main__":
    tasks = list(itertools.product(PLANTS, LANDS, SEEDS))
    print(f"{len(tasks)*(len(BETAS)+2):,} episodes", flush=True)
    with Pool(int(os.environ.get("SUR_WORKERS", "10"))) as p:
        rows = [r for c in p.imap_unordered(one, tasks, chunksize=8) for r in c]
    pd.DataFrame(rows, columns=["arm","beta","plant","landscape","seed","distance_auc",
                                "success","repair_rate","mean_step"]
                 ).to_csv("surrogate_full.csv", index=False)
    print("wrote", len(rows))
