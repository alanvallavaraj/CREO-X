"""Structural replication: rank-deficient objective and plant lag.

The closed-loop application that motivated this mechanism does NOT optimise a
landscape with a point optimum. Verified from its source, it maps a
six-dimensional bounded control u in [0,1]^6 through

    x = 2u - 1,     y = W x  (+ landscape nonlinearity),     W of shape (2, 6)

to a TWO-dimensional observable, then drives a first-order plant

    z <- (1 - kappa) z + kappa * y,

and scores the distance from z to a target. Two structural features follow that
no BBOB landscape has: the objective is RANK-DEFICIENT, with a four-dimensional
null space in which movement is free, and the plant has MEMORY, so the score
depends on the control history rather than on the current point alone.

This campaign isolates those two features. `rank` selects a (rank x 6) map, so
rank 6 is full rank with a point optimum and rank 2 reproduces the application's
degeneracy. `kappa` selects the lag, with 1.0 memoryless. Everything else is held
at the application's values: u in [0,1]^6, delta_max 0.10, budget 300.
"""
import os, sys, zlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BETA = 0.90
BUDGET = int(os.environ.get("CREO_BUDGET", "300"))
DIM = 6
RANKS = (2, 6)
KAPPAS = (0.5, 1.0)
DRIFTS = (0.0, 0.05)
Z0 = ('zero', 'opposite')   # plant start: at origin, or at -0.5*target as the loop does
SEEDS = tuple(range(1, 11))
OPS = ['rate limit (E06)', 'box projection', 'random resample (null)']
LANDSCAPES = ('linear', 'nonlinear')


def make_problem(rank, kappa, amp, landscape, seed, z0):
    from hosts import Problem
    rng = np.random.default_rng(zlib.crc32(b"W%d%s" % (rank, landscape.encode())) % (2 ** 31))
    W = rng.normal(size=(rank, DIM))
    W /= np.sum(np.abs(W), axis=1, keepdims=True)
    tgt_rng = np.random.default_rng(seed)
    target0 = tgt_rng.uniform(-0.6, 0.6, size=rank)
    d = tgt_rng.normal(size=rank); d /= np.linalg.norm(d)
    T = BUDGET / 4.0
    lo, hi = np.zeros(DIM), np.ones(DIM)
    z_init = np.zeros(rank) if z0 == 'zero' else np.clip(-0.5 * target0, -1.0, 1.0)
    st = {'t': 0, 'z': z_init.astype(float), 'err': 0.0, 'n': 0}

    def obj(u):
        x = 2.0 * np.clip(np.asarray(u, dtype=float), 0.0, 1.0) - 1.0
        y = W @ x
        if landscape == 'nonlinear':
            y = y.copy()
            y[0] += 0.18 * x[3] * x[4] - 0.12 * x[5] ** 2
            if rank > 1:
                y[1] += 0.20 * x[1] * x[2] + 0.15 * x[3] ** 2
        y = np.clip(y, -1.0, 1.0)
        st['z'] = np.clip((1.0 - kappa) * st['z'] + kappa * y, -1.0, 1.0)
        tgt = np.clip(target0 + amp * np.sin(2.0 * np.pi * st['t'] / T) * d, -1.0, 1.0)
        st['t'] += 1
        e = float(np.linalg.norm(st['z'] - tgt)) / np.sqrt(rank)
        st['err'] += e; st['n'] += 1
        return e

    return Problem("r%dk%s" % (rank, kappa), DIM, lo, hi, obj), st


def make_operator(op_name, problem, run_seed):
    from repair_ops import BoxProjection, RateLimit, RandomResample
    lo, hi = problem.lo, problem.hi
    if op_name == 'rate limit (E06)':
        return RateLimit(lo, hi, h_frac=0.10)        # the application's delta_max
    if op_name == 'box projection':
        return BoxProjection(lo, hi)
    if op_name == 'random resample (null)':
        return RandomResample(lo, hi, rng=np.random.default_rng(run_seed))
    raise KeyError(op_name)


def work(task):
    from hosts import run
    host, op_name, rank, kappa, amp, landscape, seed, z0, beta = task
    problem, st = make_problem(rank, kappa, amp, landscape, seed, z0)
    op = make_operator(op_name, problem, seed)
    r = run(host, problem, op, beta=beta, seed=seed, budget=BUDGET)
    r.pop('trace', None)
    r.update(rank=rank, kappa=kappa, drift=amp, landscape=landscape, z0=z0,
             mean_err=st['err'] / max(st['n'], 1), n_scored=st['n'])
    return r


def main():
    from multiprocessing import Pool
    from hosts import HOSTS
    tasks = [(h, o, rk, kp, a, ls, s, z, b)
             for h in HOSTS for o in OPS for rk in RANKS for kp in KAPPAS
             for a in DRIFTS for ls in LANDSCAPES for s in SEEDS for z in Z0
             for b in (0.0, BETA)]
    print("%d runs at budget %d" % (len(tasks), BUDGET), flush=True)
    out = []
    with Pool(int(os.environ.get("CREO_WORKERS", "11"))) as pool:
        for k, r in enumerate(pool.imap_unordered(work, tasks, chunksize=16), 1):
            out.append(r)
            if k % 2000 == 0:
                print("  %d/%d" % (k, len(tasks)), flush=True)
    df = pd.DataFrame(out)
    assert len(df) == len(tasks), "expected %d rows, got %d" % (len(tasks), len(df))
    assert set(df['rank']) == set(RANKS) and set(df['kappa']) == set(KAPPAS)
    assert set(df['z0']) == set(Z0), 'plant-start coverage incomplete'
    assert df['beta'].nunique() == 2 and (df['n_scored'] > 0).all()
    try:
        df.to_parquet('setpoint_pairs.parquet', index=False); nm = 'setpoint_pairs.parquet'
    except Exception as exc:
        print("parquet unavailable (%s); CSV" % exc, flush=True)
        df.to_csv('setpoint_pairs.csv', index=False); nm = 'setpoint_pairs.csv'
    print("SETPOINT_DONE saved %s %s" % (nm, df.shape), flush=True)


if __name__ == '__main__':
    main()
