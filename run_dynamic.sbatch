#!/bin/bash -l
#SBATCH --job-name=CREO_DYNAMIC
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=creo_dynamic_%j.log
# Moving-target control: 19,200 runs (8 hosts x 5 operators x 6 functions x 2 dims
# x 4 drift amplitudes x 5 seeds x 2 arms). A = 0 is the static control inside the grid.
set -euo pipefail
module load python 2>/dev/null || true
PY="$(command -v python3 || command -v python)"
echo "interpreter: $PY ($("$PY" --version 2>&1))"
"$PY" -m venv "$TMPDIR/dynenv"
source "$TMPDIR/dynenv/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy pandas scipy statsmodels pymoo ioh cma pyarrow pytest
python -c "import numpy,pandas,scipy,statsmodels,ioh,cma; print('deps ok')"
cd "$SLURM_SUBMIT_DIR"
export CREO_WORKERS="${SLURM_CPUS_PER_TASK:-11}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=== gates ==="
python -m pytest test_zero_term_equivalence.py -q
python _check_dynamic.py

echo "=== moving-target campaign ==="
time python campaign_dynamic.py

echo "=== analysis ==="
python analyse_dynamic.py

echo "ALL_DYNAMIC_DONE"
