#!/bin/bash -l
#SBATCH --job-name=CREO_D2
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=creo_d2_%j.log
# Final controlled variation. The closed-loop application steers a 2-dimensional control
# space on a 300-evaluation budget with a per-coordinate step limit of 0.10. The campaigns
# in this paper used D = 10 and 30 at 20,000 evaluations, and a 300-evaluation replication
# showed the term inert rather than beneficial. This job matches BOTH the application's
# dimensionality and its budget, which is the last structural difference left untested.
set -euo pipefail
module load python 2>/dev/null || true
PY="$(command -v python3 || command -v python)"
"$PY" -m venv "$TMPDIR/d2env"
source "$TMPDIR/d2env/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy pandas scipy statsmodels ioh cma pyarrow pytest
cd "$SLURM_SUBMIT_DIR"
export CREO_WORKERS="${SLURM_CPUS_PER_TASK:-11}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CREO_BUDGET=300
export CREO_DIMS=2

python -c "import campaign_dynamic as c; print('BUDGET',c.BUDGET,'DIMS',c.DIMS); assert c.BUDGET==300 and c.DIMS==(2,)"
time python campaign_dynamic.py
mv dynamic_pairs.parquet dynamic_pairs_d2b300.parquet 2>/dev/null || mv dynamic_pairs.csv dynamic_pairs_d2b300.csv
ln -sf dynamic_pairs_d2b300.parquet dynamic_pairs.parquet 2>/dev/null || true
python analyse_dynamic.py
mkdir -p results_d2b300 && cp results/dynamic_*.csv results_d2b300/
echo "ALL_D2_DONE"
