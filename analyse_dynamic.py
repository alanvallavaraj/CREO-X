"""Analysis of the moving-target control.

Two questions:
  D1  At each drift amplitude, does the beta term help or hurt tracking?
  D2  THE CONJECTURE: does the term's effect improve as the target moves faster?
      A = 0 is the static control inside this same campaign, so the comparison
      is internal and matched rather than across campaigns.
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

OUT = 'results'
os.makedirs(OUT, exist_ok=True)
BETA = 0.90


def load():
    for ext, fn in (('parquet', pd.read_parquet), ('csv', pd.read_csv)):
        p = 'dynamic_pairs.%s' % ext
        if os.path.exists(p):
            print('loading %s' % p, flush=True)
            return fn(p)
    raise FileNotFoundError('dynamic_pairs.parquet/.csv not found')


def main():
    df = load()
    keys = ['host', 'repair_op', 'fid', 'dim', 'drift', 'seed']
    a = df[df.beta == 0.0].set_index(keys).sort_index()
    b = df[df.beta == BETA].set_index(keys).sort_index()
    assert a.index.equals(b.index), 'arms are not exactly paired'

    pair = pd.DataFrame({k: a.index.get_level_values(k) for k in keys})
    pair['err_base'] = a.tracking_error.values
    pair['err_creo'] = b.tracking_error.values
    # positive => the term REDUCES tracking error, i.e. helps
    pair['benefit'] = pair.err_base - pair.err_creo
    pair['fire_rate'] = a.repair_fire_rate.values
    pair.to_csv(os.path.join(OUT, 'dynamic_pairs_matched.csv'), index=False)

    # ---- D1: effect at each drift level
    rows = []
    for (op, dr, D), g in pair.groupby(['repair_op', 'drift', 'dim']):
        stat, p = wilcoxon(g.err_base, g.err_creo)
        rows.append(dict(repair_op=op, drift=dr, dim=D, n=len(g),
                         median_benefit=float(np.median(g.benefit)),
                         helped_frac=float((g.benefit > 0).mean()),
                         mean_fire_rate=float(g.fire_rate.mean()), p=p))
    eff = pd.DataFrame(rows)
    eff['p_holm'] = multipletests(eff.p.values, method='holm')[1]
    eff.to_csv(os.path.join(OUT, 'dynamic_effect_by_drift.csv'), index=False)

    # ---- D2: the conjecture. For each matched landscape, compare the benefit at
    # each moving drift against the benefit of the SAME configuration when static.
    idx = ['host', 'repair_op', 'fid', 'dim', 'seed']
    static = pair[pair.drift == 0.0].set_index(idx).benefit
    rows = []
    for dr in sorted(pair.drift.unique()):
        if dr == 0.0:
            continue
        moving = pair[pair.drift == dr].set_index(idx).benefit
        j = static.index.intersection(moving.index)
        assert len(j) == len(static) == len(moving), 'drift pairing incomplete'
        d = moving.loc[j] - static.loc[j]       # positive => moving target helps the term
        stat, p = wilcoxon(moving.loc[j], static.loc[j])
        rows.append(dict(drift=dr, n=len(j),
                         median_benefit_static=float(np.median(static.loc[j])),
                         median_benefit_moving=float(np.median(moving.loc[j])),
                         drift_advantage=float(np.median(d)),
                         frac_moving_better=float((d > 0).mean()), p=p))
    conj = pd.DataFrame(rows)
    conj['p_holm'] = multipletests(conj.p.values, method='holm')[1]
    conj.to_csv(os.path.join(OUT, 'dynamic_tracking_conjecture.csv'), index=False)

    # ---- D2 restricted to the rate-limit operator, the closed-loop geometry
    rl = pair[pair.repair_op == 'rate limit (E06)']
    st = rl[rl.drift == 0.0].set_index(idx).benefit
    rows = []
    for dr in sorted(rl.drift.unique()):
        if dr == 0.0:
            continue
        mv = rl[rl.drift == dr].set_index(idx).benefit
        j = st.index.intersection(mv.index)
        d = mv.loc[j] - st.loc[j]
        stat, p = wilcoxon(mv.loc[j], st.loc[j])
        rows.append(dict(drift=dr, n=len(j), drift_advantage=float(np.median(d)),
                         frac_moving_better=float((d > 0).mean()),
                         median_benefit_moving=float(np.median(mv.loc[j])), p=p))
    rlc = pd.DataFrame(rows)
    rlc['p_holm'] = multipletests(rlc.p.values, method='holm')[1]
    rlc.to_csv(os.path.join(OUT, 'dynamic_ratelimit_conjecture.csv'), index=False)

    print('\n=== D1 effect of the term at each drift amplitude ===')
    print(eff.to_string(index=False))
    print('\n=== D2 TRACKING CONJECTURE (moving minus static benefit, all operators) ===')
    print(conj.to_string(index=False))
    print('\n=== D2 restricted to rate limit (the closed-loop geometry) ===')
    print(rlc.to_string(index=False))
    print('\nDYNAMIC_ANALYSIS_DONE')


if __name__ == '__main__':
    main()
