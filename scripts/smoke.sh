#!/bin/bash
# End-to-end smoke test: proves a fresh checkout is usable without any pre-existing dataset.
#   ./scripts/smoke.sh            (about 3-4 minutes on CPU)
# It runs: the IMU model verification, the analytic observer verification, a tiny randomised
# collection, a short training run, and the held-out evaluation (which prints the declared metrics).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="/home/liiil/Downloads/gym-pybullet-drones:$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-/home/liiil/Downloads/gym-pybullet-drones/.venv-repro/bin/python}"
TMP="results/smoke"

say() { echo; echo "=== $* ==="; }

say "1/5 IMU model vs pybullet finite differences"
$PY scripts/verify_imu.py 0 1000 | tail -8

say "2/5 analytic observer (in-range identification + beyond-range reconstruction)"
$PY scripts/verify_observer.py 2 1000 150 | tail -6

say "3/5 collect a tiny randomised dataset (12 episodes x 700 steps)"
rm -rf "$TMP" && $PY scripts/collect_dr.py --episodes 12 --workers 4 --steps 700 --out "$TMP" | tail -3

say "4/5 train for 300 iterations (GPU if available)"
$PY scripts/train_e2e.py --shards "$TMP" --iters 300 --stage2 100 --batch 64 --out "$TMP/e2e.pt" \
    | grep -E "device|frames|it |per-flight|stable bias|jitter|rate err|INS tilt|anchor|saved"

say "5/5 closed-loop sanity check (the estimator drives controller + INS)"
$PY scripts/e4_closed_loop.py --ckpt "$TMP/e2e.pt" --ranges 1000 --seeds 1 --steps 600 | tail -6

echo; echo "smoke OK"