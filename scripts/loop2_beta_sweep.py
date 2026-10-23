"""Beta dose-response in the second closed-loop environment.

Reads the E07 harness read-only and overrides creo.beta in memory. Gamma is held
at its configured value, so beta=0 differs from the deployed arm by the beta term
alone - the same isolation used for the first environment. The paired seed depends
on (seed, landscape, target, repeat, provider) and NOT on beta, so every beta level
sees an exactly matched episode.
"""
import os, sys, itertools, numpy as np, pandas as pd
from multiprocessing import Pool

ROOT = "/Users/vallava/Documents/PhD-experiments-local"
sys.path.insert(0, ROOT)

BETAS = (0.00, 0.25, 0.50, 0.75, 0.90, 1.10, 1.50)
REPEATS = int(os.environ.get("E07_REPEATS", "50"))
CONTROLLER = "creo_de"

_CFG = None

def _init():
    global _CFG
    import E07_End_to_End_Simulation.src.chain as chain
    from E06_Robustness_Ablation.src.controllers import make_controller
    def patched(name, seed, pop_size=20):
        return make_controller(name, int(seed), int(pop_size), _CFG)
    chain.make_frozen_controller = patched

def _task(args):
    beta, cond_i, landscape, target_id = args
    global _CFG
    import copy
    import E07_End_to_End_Simulation.src.chain as chain
    from E07_End_to_End_Simulation.src.controller_adapter import frozen_e06_config
    from E07_End_to_End_Simulation.src.config import load_config
    from E07_End_to_End_Simulation.src.matrices import build_primary_conditions
    _init()
    cfg = load_config(os.path.join(ROOT, "E07_End_to_End_Simulation/config/e07.yaml"))
    _CFG = copy.deepcopy(frozen_e06_config())
    _CFG["creo"] = dict(_CFG["creo"]); _CFG["creo"]["beta"] = float(beta)
    cond = build_primary_conditions(cfg)[cond_i]
    out = []
    for rep in range(REPEATS):
        r = chain.run_chain_episode(cfg, cond, landscape, target_id, rep, CONTROLLER)
        e = r.episode
        out.append((beta, e["condition_id"], landscape, target_id, rep, e["paired_seed"],
                    e["true_distance_auc"], e["success"], e["repair_rate"]))
    return out

if __name__ == "__main__":
    from E07_End_to_End_Simulation.src.config import load_config
    from E07_End_to_End_Simulation.src.matrices import build_primary_conditions
    cfg = load_config(os.path.join(ROOT, "E07_End_to_End_Simulation/config/e07.yaml"))
    nc = len(build_primary_conditions(cfg))
    tasks = [(b, ci, ls, t) for b in BETAS for ci in range(nc)
             for ls in cfg["landscapes"] for t in range(len(cfg["targets"]))]
    print(f"{nc} conditions x {len(cfg['landscapes'])} landscapes x {len(cfg['targets'])} targets "
          f"x {REPEATS} repeats x {len(BETAS)} betas = {nc*len(cfg['landscapes'])*len(cfg['targets'])*REPEATS*len(BETAS):,} episodes",
          flush=True)
    with Pool(int(os.environ.get("E07_WORKERS", "10"))) as p:
        rows = [r for chunk in p.imap_unordered(_task, tasks, chunksize=4) for r in chunk]
    pd.DataFrame(rows, columns=["beta","condition_id","landscape","target_id","repeat",
                                "paired_seed","distance_auc","success","repair_rate"]
                 ).to_csv("e07_beta_sweep.csv", index=False)
    print("wrote", len(rows), "episodes")
