# CREO-X

Repair-displacement reuse as an algorithm component: a multi-geometry ablation and a
shifted-optimum control.

## What this is

A single scalar term is added to the update rule of eight population-based optimisers:

    x_i  =  <unmodified host update>  +  beta * dr_i

where `dr_i` is the displacement the repair operator applied to that individual at its
most recent evaluation. At `beta = 0` the arm is *stream-identical* to the unmodified
host, because the term consumes no random numbers. That is what makes this a within-host
ablation rather than a comparison between algorithms, and it is asserted for all eight
hosts by `test_zero_term_equivalence.py`.

The question the campaigns answer: **is a repair displacement a usable search signal, and
does the answer depend on the geometry of the repair operator?**

Measured answer: no, and no.

## Headline results

One job, completed in 1h17m49s on 32 cores: 29,760 + 46,080 runs, `beta = 0.90` held
fixed and untuned, 20,000 function evaluations per run.

### 1. The term is net harmful on the constrained suite, under every geometry

Cells are host x problem, 96 per operator. Significance is Wilcoxon signed-rank over the
31 paired seeds within a cell, Holm-corrected across the family. Cells with fewer than
six pairs feasible in both arms are reported as untested rather than dropped.

| repair_op                  |   cells |   sig_better |   sig_worse |   not_sig |   untested_no_feasible_pairs |   pair_wins |   pair_ties |   pair_losses |   mean_fire_rate |
|:---------------------------|--------:|-------------:|------------:|----------:|-----------------------------:|------------:|------------:|--------------:|-----------------:|
| box projection             |      96 |            6 |          17 |        44 |                           29 |         900 |         454 |          1622 |            0.83  |
| ellipsoid projection       |      96 |            5 |          31 |        13 |                           47 |         519 |          30 |          2427 |            0.997 |
| radial shrinkage (CEC2017) |      96 |            3 |          38 |        31 |                           24 |         560 |          78 |          2338 |            0.867 |
| random resample (null)     |      96 |            4 |          14 |        40 |                           38 |         656 |        1205 |          1115 |            0.637 |
| rate limit (E06)           |      96 |            0 |          14 |        61 |                           21 |         970 |         218 |          1788 |            0.721 |

The rate-limit operator - the geometry predicted to carry genuine directional
information, being the component of the requested step a limiter refused - is **never**
significantly better in any of its 96 cells. Every one of the eight hosts is net harmed:

| host                  |   sig_better |   sig_worse |   pair_wins |   pair_losses |
|:----------------------|-------------:|------------:|------------:|--------------:|
| CMA-ES                |            0 |          14 |         478 |          1181 |
| DE/current-to-pbest/1 |            1 |          22 |         224 |          1444 |
| DE/rand/1/bin         |            0 |          21 |         182 |          1300 |
| GA (SBX + polynomial) |            3 |           4 |         839 |          1006 |
| GWO                   |            3 |          18 |         371 |          1294 |
| L-SHADE               |            3 |          15 |         496 |          1125 |
| SPSO-2011             |            5 |           9 |         611 |           880 |
| eps-constrained DE    |            3 |          11 |         404 |          1060 |

### 2. A benefit on a centre-biased suite is a centre-pull artefact

Each BBOB landscape is run twice, differing *only* in where the optimum sits:
`translated` f(x), with the optimum away from the domain centre, and `centred`
f(x + x*), the same landscape with the optimum at the origin, which is the domain
centre. The paired quantity is log10 of the ratio of optimality gaps; positive favours
the term. `centre_advantage` is the median of (centred - translated) over matched
landscapes.

| repair_op                  |   dim |    n |   median_improvement_centred |   median_improvement_translated |   centre_advantage |   frac_centred_larger |    p_holm |
|:---------------------------|------:|-----:|-----------------------------:|--------------------------------:|-------------------:|----------------------:|----------:|
| box projection             |    10 | 1920 |                     0        |                        0        |           0        |              0.472396 | 0.717     |
| box projection             |    30 | 1920 |                     0        |                       -0.00053  |           0.018582 |              0.553125 | 0.000274  |
| ellipsoid projection       |    10 | 1920 |                    -0.213489 |                       -0.007618 |          -0.143182 |              0.278646 | 8.12e-96  |
| ellipsoid projection       |    30 | 1920 |                    -0.127632 |                       -0.026121 |          -0.060832 |              0.248438 | 6.11e-105 |
| radial shrinkage (CEC2017) |    10 | 1920 |                     0.003305 |                       -0.074754 |           0.178466 |              0.664062 | 7.34e-58  |
| radial shrinkage (CEC2017) |    30 | 1920 |                     0.070265 |                       -0.109061 |           0.368694 |              0.797396 | 2.51e-175 |

Radial shrinkage, whose displacement always points at the domain centre, helps only in
the `centred` variant and actively hurts in `translated`. Box projection does nothing
either way. Ellipsoid projection, whose displacement is a true surface normal carrying D
degrees of freedom and is therefore the most geometrically informative operator in the
battery, is harmful in both variants.

The displacement's usefulness therefore does not increase with the geometric information
it carries. The ellipsoid arm falsifies that hypothesis directly.

## Layout

    repair_ops.py                   five box-parameterised repair operators
    hosts.py                        eight hosts behind one component interface
    campaign_constrained.py         8 hosts x 5 operators x 12 problems x 31 seeds x 2 arms
    campaign_bbob.py                the shifted-optimum control
    analyse_campaigns.py            matched-pair analysis and the centre-pull test
    test_zero_term_equivalence.py   beta=0 must be the unmodified host
    _check_ratelimit.py             guards against a silently dead rate-limit arm
    run_creo_campaigns.sbatch       Slurm submission, with both gates before compute
    results/                        analysis output
    paper/creo_methods.tex          methods section

## Reproducing

    python -m pip install numpy pandas scipy statsmodels pymoo ioh cma pyarrow pytest
    python -m pytest test_zero_term_equivalence.py -q    # must pass before anything else
    python _check_ratelimit.py                           # must print RATE_LIMIT_LIVE_OK
    CREO_WORKERS=32 python campaign_constrained.py
    CREO_WORKERS=32 python campaign_bbob.py
    python analyse_campaigns.py

Both campaigns assert full row count and arm/operator coverage before writing their
output, and the submission script runs the two gates first, so a dead arm fails in
seconds rather than after an hour of compute.

## Two implementation notes that matter

- The rate-limit operator is a no-op unless every host passes the individual's
  predecessor into the evaluator. The predecessor used here is the incumbent the
  candidate is generated from, except for CMA-ES, which has no per-individual
  predecessor and uses the distribution mean. Omitting this silently produces an
  all-ties arm that looks like a null result.
- On macOS, build archives with `COPYFILE_DISABLE=1 tar ...`, or AppleDouble sidecars
  (`._hosts.py`) travel with the bundle.

## Not in this repository

The two raw per-run tables (`constrained_pairs.parquet`, 3.5 MB; `bbob_pairs.parquet`,
899 KB) and the two matched-pair expansions (`constrained_pairs_matched.csv`, 14,880
rows; `bbob_pairs_matched.csv`, 23,040 rows) are held on the compute cluster and are
intended for the Zenodo companion deposit rather than for git.
