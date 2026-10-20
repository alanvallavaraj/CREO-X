"""Constrained matched-pair campaign.

Grid: 8 hosts x 5 repair operators x 12 constrained problems x 31 paired seeds
      x 2 arms (beta = 0 zero-term baseline, beta = 0.90 CREO arm).

beta is held at the value pre-specified in the ASOC submission (0.90) and is NOT
tuned here. The repair operator acts on the search domain, exactly as the
author-defined preferred region does in the CEC2017 protocol; the problem's own
nonlinear constraints are handled by feasibility-aware selection in every host,
identically in both arms.
"""
import os, sys, zlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BETA = 0.90
BUDGET = 20000
N_SEEDS = 31
G_NAMES = ['G1', 'G2', 'G4', 'G7', 'G9', 'G10', 'G16', 'G18', 'G19', 'G24']
ENG_NAMES = ['PressureVessel', 'CantileveredBeam']
PROBLEMS = G_NAMES + ENG_NAMES


def make_problem(name):
    import pymoo.problems.single as ps
    from hosts import Problem
    cls = getattr(ps, name)
    p = cls()
    lo = np.array(p.xl, dtype=float)
    hi = np.array(p.xu, dtype=float)
    cache = {}

    def both(x, _p=p):
        o = _p.evaluate(np.asarray(x, dtype=float)[None, :], return_values_of=["F", "G"])
        return float(np.ravel(o[0])[0]), float(np.sum(np.maximum(0.0, o[1])))

    def obj(x):
        k = x.tobytes()
        if k not in cache:
            cache[k] = both(x)
        return cache[k][0]

    def viol(x):
        k = x.tobytes()
        if k not in cache:
            cache[k] = both(x)
        return cache[k][1]

    return Problem(name, p.n_var, lo, hi, obj, viol)


def make_operator(op_name, problem, run_seed):
    """Deterministic per problem: the ellipsoid is identical in both arms and
    across seeds, so it is part of the problem rather than part of the run."""
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
        # seeded from the run seed so both arms share the same resample stream
        return RandomResample(lo, hi, rng=np.random.default_rng(run_seed))
    raise KeyError(op_name)


_CACHE = {}


def work(task):
    from hosts import run
    host, op_name, prob_name, seed, beta = task
    if prob_name not in _CACHE:
        _CACHE[prob_name] = make_problem(prob_name)
    problem = make_problem(prob_name)        # fresh objective cache per run
    op = make_operator(op_name, problem, seed)
    r = run(host, problem, op, beta=beta, seed=seed, budget=BUDGET)
    tr = np.asarray(r.pop('trace'), dtype=float)
    idx = np.linspace(0, len(tr) - 1, 21).astype(int)
    r['trace_21'] = tr[idx].tolist()
    return r


def main():
    from multiprocessing import Pool
    from repair_ops import OPERATOR_NAMES
    from hosts import HOSTS
    tasks = [(h, o, p, s, b)
             for h in HOSTS for o in OPERATOR_NAMES for p in PROBLEMS
             for s in range(1, N_SEEDS + 1) for b in (0.0, BETA)]
    print(f"{len(tasks):,} runs", flush=True)
    out = []
    with Pool(int(os.environ.get("CREO_WORKERS", "11"))) as pool:
        for i, r in enumerate(pool.imap_unordered(work, tasks, chunksize=16), 1):
            out.append(r)
            if i % 2000 == 0:
                print(f"  {i:,}/{len(tasks):,}", flush=True)
    df = pd.DataFrame(out)
    assert len(df) == len(tasks), "expected %d rows, got %d" % (len(tasks), len(df))
    assert df['beta'].nunique() == 2 and df['repair_op'].nunique() == 5, "arm/operator coverage incomplete"
    try:
        df.to_parquet('constrained_pairs.parquet', index=False)
        out_name = 'constrained_pairs.parquet'
    except Exception as exc:                      # no pyarrow/fastparquet on the host
        print("parquet unavailable (%s); writing CSV" % exc, flush=True)
        df.to_csv('constrained_pairs.csv', index=False)
        out_name = 'constrained_pairs.csv'
    print("CONSTRAINED_DONE saved %s %s" % (out_name, df.shape), flush=True)


if __name__ == '__main__':
    main()
