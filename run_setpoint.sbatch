#!/bin/bash -l
#SBATCH --job-name=CREO_SETPT
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=creo_setpoint_%j.log
set -euo pipefail
module load python 2>/dev/null || true
PY="$(command -v python3 || command -v python)"
"$PY" -m venv "$TMPDIR/spenv"; source "$TMPDIR/spenv/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy pandas scipy statsmodels cma pyarrow pytest
cd "$SLURM_SUBMIT_DIR"
export CREO_WORKERS="${SLURM_CPUS_PER_TASK:-11}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest test_zero_term_equivalence.py -q
time python campaign_setpoint.py
python analyse_setpoint.py
echo "ALL_SETPOINT_DONE"
