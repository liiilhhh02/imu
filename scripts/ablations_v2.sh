#!/bin/bash
# E3 ablations for the fixed pipeline (v2), on a 2.4e6-step subset of dr_1e7_v2.
# Ten runs: six loss drops (bias/att/phys/torque/spec/prior) + no-slow-head + unmasked-residual
# + no-w_alg + the full model on the same subset as the reference row.
#   `anchor` is gone: it was identically zero under the physics parameterisation and has been deleted.
# Run:  bash scripts/ablations_v2.sh            (about 2 h; logs to results/ablations_v2.log)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=/home/liiil/Downloads/gym-pybullet-drones:$(pwd)
PY="${PY:-/home/liiil/Downloads/gym-pybullet-drones/.venv-repro/bin/python}"
LOG=results/ablations_v2.log
DATA=results/dr_1e7_v2
SUB="--limit 1200"
say() { echo "=== $* ==="; echo "=== $* ===" >> $LOG; }

for ab in bias att phys torque spec prior; do
    say "ablation: drop L_$ab"
    $PY scripts/train_e2e.py --shards $DATA $SUB --iters 4000 --stage2 1200 --batch 512 \
        --ablate "$ab" --out results/abl2_no_$ab.pt >> $LOG 2>&1
done
say "ablation: no slow parameter head"
$PY scripts/train_e2e.py --shards $DATA $SUB --iters 4000 --stage2 1200 --batch 512 \
    --no_slow 1 --out results/abl2_noslow.pt >> $LOG 2>&1
say "ablation: unmasked residual output (no physics parameterisation)"
$PY scripts/train_e2e.py --shards $DATA $SUB --iters 4000 --stage2 1200 --batch 512 \
    --mode residual --out results/abl2_residual.pt >> $LOG 2>&1
say "ablation: no w_alg anywhere (fully black-box end to end)"
$PY scripts/train_e2e.py --shards $DATA $SUB --iters 4000 --stage2 1200 --batch 512 \
    --mode noalg --out results/abl2_noalg.pt >> $LOG 2>&1
say "reference: full model on the same subset"
$PY scripts/train_e2e.py --shards $DATA $SUB --iters 4000 --stage2 1200 --batch 512 \
    --out results/abl2_full.pt >> $LOG 2>&1

say "ABLATIONS V2 DONE"
grep -E "^=== |per-flight lag" $LOG | tail -40