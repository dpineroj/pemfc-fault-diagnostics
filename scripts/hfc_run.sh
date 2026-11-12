#!/bin/sh
# Run an hfc command with a log that ends EXIT=<the interpreter's own code>.
#   sh scripts/hfc_run.sh <log-name> <hfc args...>
# e.g. sh scripts/hfc_run.sh gen_extensive generate --preset extensive --out hfc_data/extensive
name="$1"; shift
mkdir -p run_logs/hfc
log="run_logs/hfc/${name}.log"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "=== hfc ${name} started $(date) :: $* ===" > "$log"
start=$(date +%s)
./hfc_env/Scripts/python.exe -u -W ignore -m hfc "$@" >> "$log" 2>&1
rc=$?
echo "elapsed_s=$(( $(date +%s) - start ))" >> "$log"
echo "EXIT=$rc $(date)" >> "$log"
exit $rc
