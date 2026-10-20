#!/bin/bash -l
#SBATCH --job-name=CREO_SHORTH
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=creo_shorth_%j.log
# Short-horizon replication. The closed-loop application that motivated this mechanism
# runs a 300-evaluation budget with a per-coordinate step limit of 0.10 and a population
# of 20. The campaigns in this paper used 20,000 evaluations: a 67x longer horizon. This
# job repeats the moving-target grid at the application's budget to test whether the
# horizon, rather than the repair geometry, is what separates the two findings.
set -euo pipefail
module load python 2>/dev/null || true
PY="$(command -v python3 || command -v python)"
"$PY" -m venv "$TMPDIR/shenv"
source "$TMPDIR/shenv/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy pandas scipy statsmodels ioh cma pyarrow pytest
cd "$SLURM_SUBMIT_DIR"
export CREO_WORKERS="${SLURM_CPUS_PER_TASK:-11}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CREO_BUDGET=300

echo "=== budget in force ==="
python -c "import campaign_dynamic as c; print('BUDGET =', c.BUDGET); assert c.BUDGET == 300"

echo "=== moving-target campaign at the application budget ==="
time python campaign_dynamic.py
mv dynamic_pairs.parquet dynamic_pairs_b300.parquet 2>/dev/null || mv dynamic_pairs.csv dynamic_pairs_b300.csv

echo "=== analysis ==="
ln -sf dynamic_pairs_b300.parquet dynamic_pairs.parquet 2>/dev/null || true
python analyse_dynamic.py
mkdir -p results_b300 && cp results/dynamic_*.csv results_b300/ 2>/dev/null || true
echo "ALL_SHORTH_DONE"
