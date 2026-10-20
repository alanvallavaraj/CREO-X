"""Analysis: does objective degeneracy or plant lag explain the closed-loop benefit?"""
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
OUT = 'results'; os.makedirs(OUT, exist_ok=True); BETA = 0.90

def load():
    for ext, fn in (('parquet', pd.read_parquet), ('csv', pd.read_csv)):
        p = 'setpoint_pairs.%s' % ext
        if os.path.exists(p):
            return fn(p)
    raise FileNotFoundError('setpoint_pairs not found')

def main():
    df = load()
    keys = ['host','repair_op','rank','kappa','drift','landscape','seed','z0']
    a = df[df.beta == 0.0].set_index(keys).sort_index()
    b = df[df.beta == BETA].set_index(keys).sort_index()
    assert a.index.equals(b.index), 'arms not exactly paired'
    pair = pd.DataFrame({k: a.index.get_level_values(k) for k in keys})
    pair['err_base'] = a.mean_err.values
    pair['err_creo'] = b.mean_err.values
    pair['benefit'] = pair.err_base - pair.err_creo          # positive => term helps
    pair.to_csv(os.path.join(OUT,'setpoint_pairs_matched.csv'), index=False)

    rows = []
    for (rk, kp, op), g in pair.groupby(['rank','kappa','repair_op']):
        stat, p = wilcoxon(g.err_base, g.err_creo)
        rows.append(dict(rank=rk, kappa=kp, repair_op=op, n=len(g),
                         median_benefit=float(np.median(g.benefit)),
                         helped_frac=float((g.benefit > 0).mean()), p=p))
    eff = pd.DataFrame(rows)
    eff['p_holm'] = multipletests(eff.p.values, method='holm')[1]
    eff.to_csv(os.path.join(OUT,'setpoint_effect.csv'), index=False)

    # Isolate degeneracy: same host/op/kappa/drift/landscape/seed, rank 2 vs rank 6
    idx = ['host','repair_op','kappa','drift','landscape','seed']
    r2 = pair[pair['rank'] == 2].set_index(idx).benefit
    r6 = pair[pair['rank'] == 6].set_index(idx).benefit
    j = r2.index.intersection(r6.index)
    stat, p = wilcoxon(r2.loc[j], r6.loc[j])
    deg = pd.DataFrame([dict(n=len(j),
                             median_benefit_rank2=float(np.median(r2.loc[j])),
                             median_benefit_rank6=float(np.median(r6.loc[j])),
                             degeneracy_advantage=float(np.median(r2.loc[j]-r6.loc[j])),
                             frac_rank2_better=float((r2.loc[j] > r6.loc[j]).mean()), p=p)])
    deg.to_csv(os.path.join(OUT,'setpoint_degeneracy_test.csv'), index=False)

    idx2 = ['host','repair_op','rank','drift','landscape','seed']
    k5 = pair[pair.kappa == 0.5].set_index(idx2).benefit
    k1 = pair[pair.kappa == 1.0].set_index(idx2).benefit
    j2 = k5.index.intersection(k1.index)
    stat, p2 = wilcoxon(k5.loc[j2], k1.loc[j2])
    lag = pd.DataFrame([dict(n=len(j2),
                             median_benefit_lag=float(np.median(k5.loc[j2])),
                             median_benefit_nolag=float(np.median(k1.loc[j2])),
                             lag_advantage=float(np.median(k5.loc[j2]-k1.loc[j2])),
                             frac_lag_better=float((k5.loc[j2] > k1.loc[j2]).mean()), p=p2)])
    lag.to_csv(os.path.join(OUT,'setpoint_lag_test.csv'), index=False)

    # THE TRAVERSE TEST: same configuration, plant starting at the origin vs
    # opposite the target, which is what the closed-loop application does.
    idx3 = ['host','repair_op','rank','kappa','drift','landscape','seed']
    zz = pair[pair.z0 == 'zero'].set_index(idx3).benefit
    op_ = pair[pair.z0 == 'opposite'].set_index(idx3).benefit
    j3 = zz.index.intersection(op_.index)
    stat, p3 = wilcoxon(op_.loc[j3], zz.loc[j3])
    trav = pd.DataFrame([dict(n=len(j3),
                              median_benefit_zero=float(np.median(zz.loc[j3])),
                              median_benefit_opposite=float(np.median(op_.loc[j3])),
                              traverse_advantage=float(np.median(op_.loc[j3]-zz.loc[j3])),
                              frac_opposite_better=float((op_.loc[j3] > zz.loc[j3]).mean()), p=p3)])
    trav.to_csv(os.path.join(OUT,'setpoint_traverse_test.csv'), index=False)

    by_z = []
    for (z, op2), g in pair.groupby(['z0','repair_op']):
        stat, pz = wilcoxon(g.err_base, g.err_creo)
        by_z.append(dict(z0=z, repair_op=op2, n=len(g),
                         median_benefit=float(np.median(g.benefit)),
                         helped_frac=float((g.benefit > 0).mean()), p=pz))
    bz = pd.DataFrame(by_z)
    bz['p_holm'] = multipletests(bz.p.values, method='holm')[1]
    bz.to_csv(os.path.join(OUT,'setpoint_effect_by_start.csv'), index=False)
    print('\n=== TRAVERSE TEST (plant opposite the target minus plant at origin) ===')
    print(trav.to_string(index=False))
    print('\n=== effect by plant start and operator ===')
    print(bz.to_string(index=False))

    print('\n=== effect by rank, lag and operator ===')
    print(eff.to_string(index=False))
    print('\n=== DEGENERACY TEST (rank 2 minus rank 6 benefit) ===')
    print(deg.to_string(index=False))
    print('\n=== LAG TEST (kappa 0.5 minus kappa 1.0 benefit) ===')
    print(lag.to_string(index=False))
    print('\nSETPOINT_ANALYSIS_DONE')

if __name__ == '__main__':
    main()
