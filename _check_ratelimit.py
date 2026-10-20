"""Confirm the rate-limit arm now fires on every host, and that beta=0 remains
the unmodified host. One cheap problem, 2 seeds, all 8 hosts, 2 operators."""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_constrained as cc
from hosts import HOSTS


def main():
    from multiprocessing import Pool
    hl = list(HOSTS)
    ops = ['rate limit (E06)', 'radial shrinkage (CEC2017)']
    tasks = [(h, o, 'G24', s, b) for h in hl for o in ops
             for s in (1, 2) for b in (0.0, cc.BETA)]
    with Pool(int(os.environ.get('CREO_WORKERS', '8'))) as pool:
        res = list(pool.imap_unordered(cc.work, tasks, chunksize=2))
    df = pd.DataFrame(res)
    piv = df.pivot_table(index=['repair_op', 'host'], columns='beta',
                         values='repair_fire_rate', aggfunc='mean')
    print(piv.round(4).to_string())
    rl = df[df.repair_op == 'rate limit (E06)']
    dead = sorted(rl[rl.repair_fire_rate <= 1e-9].host.unique())
    print('\nhosts where rate limit still never fires:', dead if dead else 'NONE')
    assert not dead, 'rate-limit arm is still a no-op on: %s' % dead
    # the term must change the outcome somewhere, else beta is inert
    w = df.pivot_table(index=['repair_op', 'host', 'seed'], columns='beta', values='best_f')
    changed = float((w[0.0] != w[cc.BETA]).mean())
    print('fraction of matched pairs where the term changed best_f: %.3f' % changed)
    assert changed > 0.5, 'beta term barely alters the search'
    print('RATE_LIMIT_LIVE_OK')


if __name__ == '__main__':
    main()
