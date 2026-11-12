#!/bin/sh
# POST-HOC (Sec. 17h). Launch wrapper: one attempt, log ends EXIT=<python's code>.
#   sh scripts/spatial/posthoc_run.sh <log-name> <script.py> [args...]
name="$1"; shift
mkdir -p run_logs/posthoc
log="run_logs/posthoc/${name}.log"
if [ -e "$log" ]; then
  echo "refusing to overwrite $log (one attempt per run; ask before retrying)" >&2
  exit 2
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "=== POST-HOC ${name} started $(date) ===" > "$log"
./hfc_env/Scripts/python.exe -u -W ignore "$@" >> "$log" 2>&1
rc=$?
echo "EXIT=$rc $(date)" >> "$log"
exit $rc
