"""Verify the drift actually moves the optimum and the metric responds to it."""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_dynamic as cd
from hosts import HOSTS


def main():
    from multiprocessing import Pool
    hl = list(HOSTS)[:3]
    tasks = [(h, o, f, 10, a, s, b)
             for h in hl for o in ['rate limit (E06)', 'radial shrinkage (CEC2017)']
             for f in (1, 15) for a in cd.DRIFTS for s in (1, 2) for b in (0.0, cd.BETA)]
    print('smoke grid:', len(tasks), flush=True)
    with Pool(int(os.environ.get('CREO_WORKERS', '8'))) as pool:
        res = list(pool.imap_unordered(cd.work, tasks, chunksize=4))
    df = pd.DataFrame(res)
    assert len(df) == len(tasks)
    assert (df.n_scored == cd.BUDGET).all(), df.n_scored.unique()

    print('\n--- tracking error by drift amplitude (mean over hosts/ops/fids/seeds) ---')
    piv = df.pivot_table(index='drift', columns='beta', values='tracking_error')
    piv['beta_effect'] = piv[0.0] - piv[cd.BETA]      # positive => term helps
    print(piv.round(5).to_string())

    # the drift must actually change the problem: error at A=0.20 > error at A=0
    e0 = df[df.drift == 0.0].tracking_error.mean()
    e2 = df[df.drift == 0.20].tracking_error.mean()
    print(f'\nmean tracking error  A=0.00 {e0:.5f}   A=0.20 {e2:.5f}')
    assert e2 > e0, 'drift did not change the tracking problem'

    # the term must alter the search somewhere
    w = df.pivot_table(index=['host','repair_op','fid','drift','seed'],
                       columns='beta', values='tracking_error')
    print('fraction of matched pairs where the term changed tracking error: %.3f'
          % float((w[0.0] != w[cd.BETA]).mean()))
    print('DYNAMIC_SMOKE_OK')


if __name__ == '__main__':
    main()
