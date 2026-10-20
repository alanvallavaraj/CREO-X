"""Analysis for the repair-geometry campaigns.

Consumes constrained_pairs.{parquet,csv} and bbob_pairs.{parquet,csv} and writes
the tables the preprint reports. Two questions, kept separate:

  Q1 (constrained)  Does the beta term help, and does the answer depend on the
                    GEOMETRY of the repair operator rather than its fire rate?
  Q2 (bbob)         Is the radial-shrinkage benefit a centre-seeking artefact?
                    Each landscape is run translated and centred, so the
                    centre-pull confound is measured, not argued.

Pairing is exact: the beta=0 and beta=0.90 arms share host, operator, problem,
dimension and seed, and beta=0 is stream-identical to the unmodified host.
"""
import os, sys, glob, itertools, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

OUT = 'results'
os.makedirs(OUT, exist_ok=True)
BETA = 0.90


def load(stem):
    for ext, fn in (('parquet', pd.read_parquet), ('csv', pd.read_csv)):
        p = '%s.%s' % (stem, ext)
        if os.path.exists(p):
            print('loading %s' % p, flush=True)
            return fn(p)
    raise FileNotFoundError('%s.parquet/.csv not found in %s' % (stem, os.getcwd()))


def holm(pvals):
    if len(pvals) == 0:
        return np.array([])
    return multipletests(pvals, method='holm')[1]


# ----------------------------------------------------------------- Q1
def analyse_constrained():
    df = load('constrained_pairs')
    keys = ['host', 'repair_op', 'problem', 'seed']
    a = df[df.beta == 0.0].set_index(keys)
    b = df[df.beta == BETA].set_index(keys)
    common = a.index.intersection(b.index)
    assert len(common) == len(a) == len(b), \
        'unpaired rows: %d vs %d vs %d' % (len(common), len(a), len(b))
    a, b = a.loc[common], b.loc[common]

    # Feasibility-aware score: infeasible runs are ordered by violation above all
    # feasible ones, matching Deb's rules used for selection inside every host.
    def score(fr):
        return np.where(fr.best_viol > 1e-12, np.inf, fr.best_f), fr.best_viol.values

    fa, va = score(a); fb, vb = score(b)
    pair = pd.DataFrame({
        'host': common.get_level_values('host'),
        'repair_op': common.get_level_values('repair_op'),
        'problem': common.get_level_values('problem'),
        'seed': common.get_level_values('seed'),
        'f_base': a.best_f.values, 'f_creo': b.best_f.values,
        'viol_base': va, 'viol_creo': vb,
        'fire_base': a.repair_fire_rate.values, 'fire_creo': b.repair_fire_rate.values,
    })
    # win = CREO arm strictly better on (feasible, violation, objective)
    key_base = list(zip(va > 1e-12, va, a.best_f.values))
    key_creo = list(zip(vb > 1e-12, vb, b.best_f.values))
    pair['creo_better'] = [kc < kb for kc, kb in zip(key_creo, key_base)]
    pair['tie'] = [kc == kb for kc, kb in zip(key_creo, key_base)]
    pair.to_csv(os.path.join(OUT, 'constrained_pairs_matched.csv'), index=False)

    # Per (operator, host, problem) Wilcoxon over the 31 paired seeds, on the
    # objective among runs feasible in both arms; cells with no feasible pairs
    # are reported as such rather than dropped silently.
    rows = []
    for (op, h, prob), g in pair.groupby(['repair_op', 'host', 'problem']):
        ok = (g.viol_base <= 1e-12) & (g.viol_creo <= 1e-12)
        n_ok = int(ok.sum())
        d = (g.f_base - g.f_creo)[ok]          # positive = CREO better (minimisation)
        if n_ok >= 6 and np.any(d != 0):
            stat, p = wilcoxon(g.f_base[ok], g.f_creo[ok])
        else:
            p = np.nan
        rows.append(dict(repair_op=op, host=h, problem=prob, n_feasible_pairs=n_ok,
                         median_delta_f=float(np.median(d)) if n_ok else np.nan,
                         wins=int(g.creo_better.sum()), ties=int(g.tie.sum()),
                         losses=int((~g.creo_better & ~g.tie).sum()),
                         mean_fire_rate=float(g.fire_base.mean()), p=p))
    cells = pd.DataFrame(rows)
    m = cells.p.notna()
    cells.loc[m, 'p_holm'] = holm(cells.loc[m, 'p'].values)
    cells.to_csv(os.path.join(OUT, 'constrained_cells.csv'), index=False)

    # The headline table: operator geometry, aggregated over hosts and problems.
    by_op = cells.groupby('repair_op').apply(lambda g: pd.Series({
        'cells': len(g),
        'sig_better': int(((g.p_holm < 0.05) & (g.median_delta_f > 0)).sum()),
        'sig_worse': int(((g.p_holm < 0.05) & (g.median_delta_f < 0)).sum()),
        'not_sig': int((g.p_holm >= 0.05).sum()),
        'untested_no_feasible_pairs': int(g.p_holm.isna().sum()),
        'pair_wins': int(g.wins.sum()), 'pair_ties': int(g.ties.sum()),
        'pair_losses': int(g.losses.sum()),
        'mean_fire_rate': float(g.mean_fire_rate.mean()),
    }), include_groups=False).reset_index()
    by_op.to_csv(os.path.join(OUT, 'constrained_by_operator.csv'), index=False)

    by_host = cells.groupby('host').apply(lambda g: pd.Series({
        'sig_better': int(((g.p_holm < 0.05) & (g.median_delta_f > 0)).sum()),
        'sig_worse': int(((g.p_holm < 0.05) & (g.median_delta_f < 0)).sum()),
        'pair_wins': int(g.wins.sum()), 'pair_losses': int(g.losses.sum()),
    }), include_groups=False).reset_index()
    by_host.to_csv(os.path.join(OUT, 'constrained_by_host.csv'), index=False)

    print('\n=== Q1 operator geometry (cells = host x problem, 31 paired seeds each) ===')
    print(by_op.to_string(index=False))
    print('\n=== Q1 by host ===')
    print(by_host.to_string(index=False))
    return by_op


# ----------------------------------------------------------------- Q2
def analyse_bbob():
    df = load('bbob_pairs')
    keys = ['host', 'repair_op', 'fid', 'instance', 'dim', 'variant', 'seed']
    a = df[df.beta == 0.0].set_index(keys)
    b = df[df.beta == BETA].set_index(keys)
    common = a.index.intersection(b.index)
    assert len(common) == len(a) == len(b), \
        'unpaired rows: %d vs %d vs %d' % (len(common), len(a), len(b))
    a, b = a.loc[common], b.loc[common]

    pair = pd.DataFrame({k: common.get_level_values(k) for k in keys})
    pair['gap_base'] = a.gap.values
    pair['gap_creo'] = b.gap.values
    # log-ratio improvement: positive = CREO closer to the optimum
    eps = 1e-12
    pair['log_ratio'] = np.log10((a.gap.values + eps) / (b.gap.values + eps))
    pair['fire_rate'] = a.repair_fire_rate.values
    pair.to_csv(os.path.join(OUT, 'bbob_pairs_matched.csv'), index=False)

    # Per (operator, variant, dim): does the term help at all?
    rows = []
    for (op, var, D), g in pair.groupby(['repair_op', 'variant', 'dim']):
        stat, p = wilcoxon(g.gap_base, g.gap_creo)
        rows.append(dict(repair_op=op, variant=var, dim=D, n=len(g),
                         median_log_ratio=float(np.median(g.log_ratio)),
                         better_frac=float((g.gap_creo < g.gap_base).mean()),
                         mean_fire_rate=float(g.fire_rate.mean()), p=p))
    eff = pd.DataFrame(rows)
    eff['p_holm'] = holm(eff.p.values)
    eff.to_csv(os.path.join(OUT, 'bbob_effect_by_variant.csv'), index=False)

    # THE CONFOUND TEST: the same landscape translated vs centred is a matched
    # pair, so the difference of improvements is tested directly.
    idx = ['host', 'repair_op', 'fid', 'instance', 'dim', 'seed']
    t = pair[pair.variant == 'translated'].set_index(idx).log_ratio
    c = pair[pair.variant == 'centred'].set_index(idx).log_ratio
    j = t.index.intersection(c.index)
    assert len(j) == len(t) == len(c), 'variant pairing incomplete'
    conf = []
    for (op, D), g in pd.DataFrame({'translated': t.loc[j], 'centred': c.loc[j]}) \
            .reset_index().groupby(['repair_op', 'dim']):
        stat, p = wilcoxon(g.centred, g.translated)
        conf.append(dict(repair_op=op, dim=D, n=len(g),
                         median_improvement_centred=float(np.median(g.centred)),
                         median_improvement_translated=float(np.median(g.translated)),
                         centre_advantage=float(np.median(g.centred - g.translated)),
                         frac_centred_larger=float((g.centred > g.translated).mean()), p=p))
    conf = pd.DataFrame(conf)
    conf['p_holm'] = holm(conf.p.values)
    conf.to_csv(os.path.join(OUT, 'bbob_centre_pull_test.csv'), index=False)

    print('\n=== Q2 effect of the term, by operator and optimum location ===')
    print(eff.to_string(index=False))
    print('\n=== Q2 CENTRE-PULL CONFOUND TEST (centred minus translated improvement) ===')
    print(conf.to_string(index=False))
    return conf


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('both', 'constrained'):
        analyse_constrained()
    if which in ('both', 'bbob'):
        analyse_bbob()
    print('\nANALYSIS_DONE ->', sorted(os.listdir(OUT)))
