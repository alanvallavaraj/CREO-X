# Provenance

All results in `results/` come from one Slurm job on a university HPC cluster
(32 cores, 32 GB), completed in 1h17m49s with exit status 0.

    campaign bundle sha256  f409c0f567b8d950abd57526a6a12dbf5315bb403d27d15f3e6dee8ee4017adc
    constrained campaign    29,760 runs, 39m50s
    bbob campaign           46,080 runs, 36m55s
    peak RSS                4.9 GB

Gates that ran before any compute, both passing:

    test_zero_term_equivalence.py   2 passed
    _check_ratelimit.py             RATE_LIMIT_LIVE_OK

Row-count checks on the analysis output, which confirm nothing was dropped:

    results/constrained_cells.csv          480 rows = 5 operators x 8 hosts x 12 problems
    constrained_pairs_matched.csv       14,880 rows = 29,760 runs / 2 arms
    bbob_pairs_matched.csv              23,040 rows = 46,080 runs / 2 arms

Environment: Python 3.11.7; numpy, pandas, scipy, statsmodels, pymoo 0.6.2, ioh,
cma 4.5.0, pyarrow. `beta = 0.90` throughout, untuned, with a 20,000-evaluation budget.

One cross-platform note: seven of the eight hosts reproduce their repair fire rates
bit-identically between a local macOS run and the cluster run. CMA-ES does not, because
the reference `cma` implementation differs in version between the two; its rate-limit
fire rate was 0.7358/0.6833 locally against 0.6226/0.3557 on the cluster for the two
arms. Treat the CMA-ES arm as reproducible only against a pinned `cma` version.
