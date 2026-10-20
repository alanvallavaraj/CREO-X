"""Moving-target control: does displacement reuse pay off when the optimum drifts?

The static campaigns in this repository show the beta*dr term is net harmful on
fixed landscapes under every repair geometry tested. The remaining conjecture is
that the mechanism belongs to TRACKING rather than to repair geometry: when a
controller repeatedly proposes a step, a limiter refuses part of it, and the
target itself moves, the refused component may describe a constraint the search
is actively fighting rather than a fixed boundary it is bumping into.

This campaign tests that directly. Each BBOB landscape is optimised with its
optimum displaced along a deterministic trajectory

    s(t) = A * (hi - lo)/2 * sin(2*pi*t / T) * d        T = budget/4

where t is the evaluation index, d is a unit direction fixed per problem, and A
is the drift amplitude as a fraction of the domain half-width. A = 0 reproduces
the static case and is the control: the campaign therefore contains its own
null.

The drift schedule is a deterministic function of t alone, so it is identical in
both arms of a matched pair even though the two arms evaluate different points.

Reported metric is TRACKING ERROR, the mean distance from the evaluated point to
the optimum's current location, normalised by the domain width. Best-so-far
objective value is meaningless when the optimum moves, so it is not used.
"""
import os, sys, zlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BETA = 0.90
BUDGET = 20000
DRIFTS = (0.0, 0.01, 0.05, 0.20)          # A = 0 is the static control
FIDS = (1, 8, 15, 17, 20, 21)             # sphere, Rosenbrock, Rastrigin, Schaffer,
                                          # Schwefel, Gallagher: separable to multimodal
DIMS = (10, 30)
SEEDS = (1, 2, 3, 4, 5)
OPS = ['radial shrinkage (CEC2017)', 'box projection', 'ellipsoid projection',
       'rate limit (E06)', 'random resample (null)']


def make_problem(fid, D, amp):
    """BBOB landscape whose optimum follows s(t). The objective advances its own
    clock, so drift is per evaluation and identical across arms."""
    import ioh
    from hosts import Problem
    pr = ioh.get_problem(fid, instance=1, dimension=D,
                         problem_class=ioh.ProblemClass.BBOB)
    xstar = np.array(pr.optimum.x, dtype=float)
    lo, hi = np.full(D, -5.0), np.full(D, 5.0)
    half = (hi - lo) / 2.0
    rng = np.random.default_rng(zlib.crc32(b"drift%d" % fid) % (2 ** 31))
    d = rng.normal(size=D)
    d /= np.linalg.norm(d)
    T = BUDGET / 4.0
    state = {'t': 0, 'err': 0.0, 'n': 0}
    width = float(np.linalg.norm(hi - lo))

    def shift(t):
        return amp * half * np.sin(2.0 * np.pi * t / T) * d

    def obj(x):
        s = shift(state['t'])
        state['t'] += 1
        xa = np.asarray(x, dtype=float)
        # distance from the evaluated point to where the optimum currently sits
        state['err'] += float(np.linalg.norm(xa - (xstar + s))) / width
        state['n'] += 1
        return float(pr(list(xa - s)))       # landscape translated by -s(t)

    p = Problem("f%d" % fid, D, lo, hi, obj, optimum_x=xstar)
    return p, state


def make_operator(op_name, problem, run_seed):
    from repair_ops import (RadialShrinkage, BoxProjection, EllipsoidProjection,
                            RateLimit, RandomResample)
    lo, hi = problem.lo, problem.hi
    geom_seed = zlib.crc32(problem.name.encode()) % (2 ** 31)
    if op_name == 'radial shrinkage (CEC2017)':
        return RadialShrinkage(lo, hi)
    if op_name == 'box projection':
        return BoxProjection(lo, hi)
    if op_name == 'ellipsoid projection':
        return EllipsoidProjection(lo, hi, np.random.default_rng(geom_seed))
    if op_name == 'rate limit (E06)':
        return RateLimit(lo, hi)
    if op_name == 'random resample (null)':
        return RandomResample(lo, hi, rng=np.random.default_rng(run_seed))
    raise KeyError(op_name)


def work(task):
    from hosts import run
    host, op_name, fid, D, amp, seed, beta = task
    problem, state = make_problem(fid, D, amp)
    op = make_operator(op_name, problem, seed)
    r = run(host, problem, op, beta=beta, seed=seed, budget=BUDGET)
    r.pop('trace', None)
    r.update(fid=fid, drift=amp,
             tracking_error=state['err'] / max(state['n'], 1),
             n_scored=state['n'])
    return r


def main():
    from multiprocessing import Pool
    from hosts import HOSTS
    tasks = [(h, o, f, D, a, s, b)
             for h in HOSTS for o in OPS for f in FIDS for D in DIMS
             for a in DRIFTS for s in SEEDS for b in (0.0, BETA)]
    print("%d runs" % len(tasks), flush=True)
    out = []
    with Pool(int(os.environ.get("CREO_WORKERS", "11"))) as pool:
        for k, r in enumerate(pool.imap_unordered(work, tasks, chunksize=32), 1):
            out.append(r)
            if k % 5000 == 0:
                print("  %d/%d" % (k, len(tasks)), flush=True)
    df = pd.DataFrame(out)
    assert len(df) == len(tasks), "expected %d rows, got %d" % (len(tasks), len(df))
    assert df['beta'].nunique() == 2, "arm coverage incomplete"
    assert sorted(df['drift'].unique()) == sorted(DRIFTS), "drift coverage incomplete"
    assert df['repair_op'].nunique() == len(OPS), "operator coverage incomplete"
    assert (df['n_scored'] > 0).all(), "tracking error was never accumulated"
    try:
        df.to_parquet('dynamic_pairs.parquet', index=False)
        name = 'dynamic_pairs.parquet'
    except Exception as exc:
        print("parquet unavailable (%s); writing CSV" % exc, flush=True)
        df.to_csv('dynamic_pairs.csv', index=False)
        name = 'dynamic_pairs.csv'
    print("DYNAMIC_DONE saved %s %s" % (name, df.shape), flush=True)


if __name__ == '__main__':
    main()
