"""Does a maximum-slew policy reproduce the displacement term's advantage?

Three arms in the second loop, sharing the harness and the paired seed:
  creo_b090   beta = 0.90, applied step as the host proposes it  (the deployed arm)
  creo_b000   beta = 0.00, applied step as the host proposes it  (matched baseline)
  maxslew     beta = 0.00, applied step rescaled so its largest component reaches the
              rate limit: saturation WITHOUT the displacement term

If maxslew matches creo_b090 the mechanism reduces to saturating the limiter; if it does
not, the term supplies something a saturating policy cannot.
"""
import os, sys, copy, itertools, numpy as np, pandas as pd
from dataclasses import replace
from multiprocessing import Pool

ROOT = "/gpfs/scratch/hdw517/PhD-experiments-e07/repo"
sys.path.insert(0, ROOT)
REPEATS = int(os.environ.get("E07_REPEATS", "50"))
ARMS = ("creo_b090", "creo_b000", "maxslew")

def _setup(arm):
    import E07_End_to_End_Simulation.src.chain as chain
    from E07_End_to_End_Simulation.src.controller_adapter import frozen_e06_config
    from E06_Robustness_Ablation.src.controllers import make_controller, controller_cycle
    cfg6 = copy.deepcopy(frozen_e06_config())
    cfg6["creo"] = dict(cfg6["creo"])
    cfg6["creo"]["beta"] = 0.90 if arm == "creo_b090" else 0.0
    chain.make_frozen_controller = lambda name, seed, pop_size=20: make_controller(
        name, int(seed), int(pop_size), cfg6)
    if arm == "maxslew":
        delta = None
        def wrapped(state, objective_fn, u_prev, delta_max, budget=20):
            r = controller_cycle(state, objective_fn, u_prev, delta_max, budget)
            step = np.asarray(r.best_control, float) - np.asarray(u_prev, float)
            m = float(np.max(np.abs(step)))
            if m > 1e-12:
                step = step * (float(delta_max) / m)
            step = np.clip(step, -float(delta_max), float(delta_max))
            return replace(r, best_control=np.clip(np.asarray(u_prev, float) + step, 0.0, 1.0))
        chain.controller_cycle = wrapped
    else:
        chain.controller_cycle = controller_cycle
    return chain

def _task(args):
    arm, cond_i, landscape, target_id = args
    from E07_End_to_End_Simulation.src.config import load_config
    from E07_End_to_End_Simulation.src.matrices import build_primary_conditions
    chain = _setup(arm)
    cfg = load_config(os.path.join(ROOT, "E07_End_to_End_Simulation/config/e07.yaml"))
    cond = build_primary_conditions(cfg)[cond_i]
    out = []
    for rep in range(REPEATS):
        e = chain.run_chain_episode(cfg, cond, landscape, target_id, rep, "creo_de").episode
        out.append((arm, e["condition_id"], landscape, target_id, rep, e["paired_seed"],
                    e["true_distance_auc"], e["success"], e["repair_rate"],
                    e["control_path_length"]))
    return out

if __name__ == "__main__":
    from E07_End_to_End_Simulation.src.config import load_config
    from E07_End_to_End_Simulation.src.matrices import build_primary_conditions
    cfg = load_config(os.path.join(ROOT, "E07_End_to_End_Simulation/config/e07.yaml"))
    nc = len(build_primary_conditions(cfg))
    tasks = [(a, ci, ls, t) for a in ARMS for ci in range(nc)
             for ls in cfg["landscapes"] for t in range(len(cfg["targets"]))]
    n = nc*len(cfg['landscapes'])*len(cfg['targets'])*REPEATS*len(ARMS)
    print(f"{len(ARMS)} arms, {n:,} episodes", flush=True)
    with Pool(int(os.environ.get("E07_WORKERS", "10"))) as p:
        rows = [r for c in p.imap_unordered(_task, tasks, chunksize=4) for r in c]
    pd.DataFrame(rows, columns=["arm","condition_id","landscape","target_id","repeat",
                                "paired_seed","distance_auc","success","repair_rate","path_length"]
                 ).to_csv("e07_maxslew.csv", index=False)
    print("wrote", len(rows))
