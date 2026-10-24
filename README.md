# CREO-X

Code, campaigns and results for:

> **CREO: a repair-displacement term for population-based search, and the boundary of its
> usefulness**
> Alan Immanuel Benjamin Vallavaraj, School of Computer Science and Engineering,

The manuscript is in [`paper/creo_x.pdf`](paper/creo_x.pdf) and builds from
[`paper/creo_x.tex`](paper/creo_x.tex).

## The component

A single scalar term is added to the update rule of eight population-based optimisers:

    x_i  =  <unmodified host update>  +  beta * dr_i

where `dr_i` is the displacement the repair operator applied to that individual at its most
recent evaluation. CREO stands for Continuous Repair Evolutionary Optimiser, and the name
denotes this term and nothing else: it is not an algorithm, and a host carrying it is a
configuration of that host.

At `beta = 0` the arm is *stream-identical* to the unmodified host, because the term
consumes no random numbers. That is what makes this a within-host ablation rather than a
comparison between algorithms, and it is asserted for all eight hosts by
`test_zero_term_equivalence.py`.

## Findings

**1. On static problems the term is net harmful under every repair geometry tested.**
Cells are host x problem, 96 per operator, Wilcoxon signed-rank over 31 paired seeds,
Holm-corrected.

| repair_op                  |   cells |   sig_better |   sig_worse |   not_sig |   untested_no_feasible_pairs |   pair_wins |   pair_ties |   pair_losses |   mean_fire_rate |
|:---------------------------|--------:|-------------:|------------:|----------:|-----------------------------:|------------:|------------:|--------------:|-----------------:|
| box projection             |      96 |            6 |          17 |        44 |                           29 |         900 |         454 |          1622 |            0.83  |
| ellipsoid projection       |      96 |            5 |          31 |        13 |                           47 |         519 |          30 |          2427 |            0.997 |
| radial shrinkage (CEC2017) |      96 |            3 |          38 |        31 |                           24 |         560 |          78 |          2338 |            0.867 |
| random resample (null)     |      96 |            4 |          14 |        40 |                           38 |         656 |        1205 |          1115 |            0.637 |
| rate limit (E06)           |      96 |            0 |          14 |        61 |                           21 |         970 |         218 |          1788 |            0.721 |

The rate-limit operator, the geometry predicted to carry genuine directional information,
is never significantly better in any of its 96 cells.

**2. A benefit reported on a centre-biased suite is a centre-pull artefact.** Each BBOB
landscape is optimised twice, differing *only* in whether its optimum sits at the domain
centre.

| repair_op                  |   dim |    n |   median_improvement_centred |   median_improvement_translated |   centre_advantage |   frac_centred_larger |    p_holm |
|:---------------------------|------:|-----:|-----------------------------:|--------------------------------:|-------------------:|----------------------:|----------:|
| box projection             |    10 | 1920 |                     0        |                        0        |           0        |              0.472396 | 0.717     |
| box projection             |    30 | 1920 |                     0        |                       -0.00053  |           0.018582 |              0.553125 | 0.000274  |
| ellipsoid projection       |    10 | 1920 |                    -0.213489 |                       -0.007618 |          -0.143182 |              0.278646 | 8.12e-96  |
| ellipsoid projection       |    30 | 1920 |                    -0.127632 |                       -0.026121 |          -0.060832 |              0.248438 | 6.11e-105 |
| radial shrinkage (CEC2017) |    10 | 1920 |                     0.003305 |                       -0.074754 |           0.178466 |              0.664062 | 7.34e-58  |
| radial shrinkage (CEC2017) |    30 | 1920 |                     0.070265 |                       -0.109061 |           0.368694 |              0.797396 | 2.51e-175 |

Radial shrinkage, whose displacement always points at the domain centre, helps only in the
`centred` variant and hurts in `translated`. Information does not buy benefit: the null
control, whose displacement is pure noise by construction, is the *least* harmful of the
five (losing 37.5% of matched pairs), while the most informative operator, ellipsoid
projection, loses the most (81.6%) and is the only one significantly harmful whether or
not the optimum sits at the domain centre. By significantly-harmful cells the worst
operator is radial shrinkage (38 of 96), not the ellipsoid (31).

**3. On a deployed rate-constrained tracking loop the term does work.** Sweeping `beta`
within one controller, all other coefficients fixed, with `beta = 0` an exact zero-term
baseline:

|   beta |      auc |     succ |   repair |
|-------:|---------:|---------:|---------:|
|   0    | 0.421479 | 0.578438 | 0.876975 |
|   0.25 | 0.421147 | 0.573125 | 0.873916 |
|   0.5  | 0.41702  | 0.590312 | 0.879507 |
|   0.75 | 0.412328 | 0.592188 | 0.89702  |
|   0.9  | 0.409232 | 0.608438 | 0.92128  |
|   1.1  | 0.403673 | 0.638438 | 0.975349 |
|   1.5  | 0.397247 | 0.659062 | 0.999878 |

Distance AUC falls strictly monotonically across all seven levels. Decomposing the deployed
configuration's advantage over its bare host on this condition: 3.72% total, of which
0.84% is attributable to the other differences between arms and **2.91% to `beta` alone**,
so the term accounts for 78.1% of the effect.

It improves every host in every one of the 21 profiles, by 3.14% to 21.31%. It does not
reorder the field: four of the six equipped arms still rank below the strongest bare host,
and the gain is inversely related to host quality (Spearman +0.83).

| controller   |   mean_rank |
|:-------------|------------:|
| creo_gwo     |      1      |
| creo_dfo     |      2.0476 |
| gwo          |      2.9524 |
| creo_pso     |      4      |
| creo_de      |      5.1905 |
| dfo          |      6.4762 |
| pso          |      6.7619 |
| creo_ga      |      7.5714 |
| creo_jso     |      9      |
| jso          |     10      |
| ga           |     11      |
| de           |     12      |

**4. The dissociation is unexplained.** Seven structural features of that loop were
replicated on standard problems as eight conditions: the rate-limit geometry and its limit
magnitude, the 300-evaluation budget, the six-dimensional bounded control space, a
rank-deficient objective matching the loop's six-to-two map, the first-order plant lag, a
moving target at four rates, and a plant initialised opposite its target. In none did the
term become beneficial. One difference remains untested and is named in the paper: the loop
rate-limits about a *single shared* applied control, making every individual's `dr` a
component of the same refused step.

## Layout

    repair_ops.py                   five box-parameterised repair operators
    hosts.py                        eight hosts behind one component interface
    campaign_constrained.py         8 hosts x 5 operators x 12 problems x 31 seeds x 2 arms
    campaign_bbob.py                shifted-optimum control (translated vs centred)
    campaign_dynamic.py             moving-target control (four drift rates)
    campaign_setpoint.py            rank-deficiency, plant lag and traverse replication
    analyse_campaigns.py            matched-pair analysis and the centre-pull test
    analyse_dynamic.py              drift analysis and the tracking-conjecture test
    analyse_setpoint.py             degeneracy, lag and traverse tests
    test_zero_term_equivalence.py   beta=0 must be the unmodified host
    _check_ratelimit.py             guards against a silently dead rate-limit arm
    _check_dynamic.py               guards that the drift schedule changes the problem
    run_*.sbatch                    Slurm submission, gates before compute
    results/                        all analysis output
    paper/                          manuscript, bibliography and figures

## Reproducing

    python -m pip install -r requirements.txt
    python -m pytest test_zero_term_equivalence.py -q    # must pass before anything else
    python _check_ratelimit.py                           # must print RATE_LIMIT_LIVE_OK
    CREO_WORKERS=32 python campaign_constrained.py
    CREO_WORKERS=32 python campaign_bbob.py
    python analyse_campaigns.py

`CREO_BUDGET` and `CREO_DIMS` override the evaluation budget and dimensionality for the
replication campaigns. Every campaign asserts full row count and arm coverage before
writing output, and the submission scripts run the gates first, so a dead arm fails in
seconds rather than after an hour of compute.

## Two implementation notes that matter

- The rate-limit operator is a no-op unless every host passes the individual's predecessor
  into the evaluator. The predecessor used here is the incumbent the candidate is generated
  from, except for CMA-ES, which has no per-individual predecessor and uses the
  distribution mean. Omitting this silently produces an all-ties arm that reads as a null
  result.
- Keep `multiprocessing.Pool` construction inside `if __name__ == '__main__'`. macOS spawns
  rather than forks, so an unguarded module-level pool makes every worker re-import the
  module and create another pool.

## Data availability

`results/` holds the analysis tables. The raw per-run tables
(`constrained_pairs.parquet`, `bbob_pairs.parquet`, and the matched-pair expansions of
14,880 and 23,040 rows) are excluded from git by `.gitignore` and are deposited
separately; see `.zenodo.json`.

## Licence

Code is MIT (`LICENSE`). Results, figures and manuscript content are CC BY 4.0
(`LICENSE-DATA`).

## Citing

See `CITATION.cff`.
