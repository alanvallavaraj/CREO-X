"""Shifted-optimum control campaign.

The decisive test of the centre-pull confound. Each BBOB function/instance is run
in two variants that differ only in where the optimum sits:

    translated  f(x)        - the optimum is at x*, away from the domain centre
    centred     f(x + x*)   - the same landscape with the optimum at the origin,
                              which is the domain centre

A repair operator whose displacement points at the domain centre (radial shrinkage,
the CEC2017 operator) must help in the centred variant and not in the translated one.
If it does, the D=10 benefit reported on CEC2017 is a centre-seeking artefact rather
than a repair-guidance effect.
"""

import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BETA = 0.90
BUDGET = 20000
DIMS = (10, 30)
INSTANCES = (1, 2, 3, 4, 5)
SEEDS = (1, 2)
FIDS = tuple(range(1, 25))                     # the 24 BBOB functions
OPS = ['radial shrinkage (CEC2017)', 'box projection', 'ellipsoid projection']


def make_problem(fid, inst, D, variant):
    import ioh
    from hosts import Problem
    pr = ioh.get_problem(fid, instance=inst, dimension=D,
                         problem_class=ioh.ProblemClass.BBOB)
    xstar = np.array(pr.optimum.x, dtype=float)
    lo = np.full(D, -5.0)
    hi = np.full(D, 5.0)
    if variant == 'translated':
        obj = lambda x, _p=pr: float(_p(list(np.asarray(x, dtype=float))))
        opt = xstar
    elif variant == 'centred':
        obj = lambda x, _p=pr, _s=xstar: float(_p(list(np.asarray(x, dtype=float) + _s)))
        opt = np.zeros(D)
    else:
        raise KeyError(variant)
    return Problem("f%d" % fid, D, lo, hi, obj, optimum_x=opt), float(pr.optimum.y)


def make_operator(op_name, problem, run_seed):
    from repair_ops import RadialShrinkage, BoxProjection, EllipsoidProjection
    lo, hi = problem.lo, problem.hi
    if op_name == 'radial shrinkage (CEC2017)':
        return RadialShrinkage(lo, hi)
    if op_name == 'box projection':
        return BoxProjection(lo, hi)
    if op_name == 'ellipsoid projection':
        return EllipsoidProjection(lo, hi, np.random.default_rng(900 + len(lo)))
    raise KeyError(op_name)


def work(task):
    from hosts import run
    host, op_name, fid, inst, D, variant, seed, beta = task
    problem, ystar = make_problem(fid, inst, D, variant)
    op = make_operator(op_name, problem, seed)
    r = run(host, problem, op, beta=beta, seed=seed, budget=BUDGET)
    r.pop('trace', None)
    r.update(fid=fid, instance=inst, variant=variant, gap=r['best_f'] - ystar)
    return r


def main():
    from multiprocessing import Pool
    from hosts import HOSTS
    tasks = [(h, o, f, i, D, v, s, b)
             for h in HOSTS for o in OPS for f in FIDS for i in INSTANCES
             for D in DIMS for v in ('translated', 'centred') for s in SEEDS
             for b in (0.0, BETA)]
    print("%d runs" % len(tasks), flush=True)
    out = []
    with Pool(int(os.environ.get("CREO_WORKERS", "11"))) as pool:
        for k, r in enumerate(pool.imap_unordered(work, tasks, chunksize=32), 1):
            out.append(r)
            if k % 5000 == 0:
                print("  %d/%d" % (k, len(tasks)), flush=True)
    df = pd.DataFrame(out)
    assert len(df) == len(tasks), "expected %d rows, got %d" % (len(tasks), len(df))
    assert set(df['variant']) == {'translated', 'centred'}, "variant coverage incomplete"
    assert df['beta'].nunique() == 2, "arm coverage incomplete"
    try:
        df.to_parquet('bbob_pairs.parquet', index=False)
        out_name = 'bbob_pairs.parquet'
    except Exception as exc:                      # no pyarrow/fastparquet on the host
        print("parquet unavailable (%s); writing CSV" % exc, flush=True)
        df.to_csv('bbob_pairs.csv', index=False)
        out_name = 'bbob_pairs.csv'
    print("BBOB_DONE saved %s %s" % (out_name, df.shape), flush=True)


if __name__ == '__main__':
    main()
