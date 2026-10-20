#!/bin/bash -l
#SBATCH --job-name=CREO_REPAIR_GEOM
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=creo_campaigns_%j.log
# Repair-geometry campaigns for the CREO component preprint.
#   campaign_constrained.py : 29,760 runs (8 hosts x 5 operators x 12 problems x 31 seeds x 2 arms)
#   campaign_bbob.py        : 46,080 runs (translated vs centred optimum: the centre-pull control)
# Measured cost on 11 workers: 63 min + 56 min. At 32 workers expect roughly 25 min + 20 min.
set -euo pipefail

module load python 2>/dev/null || true
PY="$(command -v python3 || command -v python)"
echo "interpreter: $PY ($("$PY" --version 2>&1))"
"$PY" -m venv "$TMPDIR/creoenv"
source "$TMPDIR/creoenv/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy pandas scipy statsmodels pymoo ioh cma pyarrow pytest
python -c "import numpy, pandas, scipy, statsmodels, pymoo, ioh, cma; print('deps ok')"


cd "$SLURM_SUBMIT_DIR"
export CREO_WORKERS="${SLURM_CPUS_PER_TASK:-11}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== zero-term equivalence (beta=0 must be the unmodified host) ==="
python -m pytest test_zero_term_equivalence.py -q

echo "=== rate-limit liveness (the arm that was dead on 2026-10-20) ==="
python _check_ratelimit.py

echo "=== constrained campaign ==="
time python campaign_constrained.py

echo "=== bbob centre-pull campaign ==="
time python campaign_bbob.py

ls -la constrained_pairs.* bbob_pairs.*
echo "ALL_CAMPAIGNS_DONE"
