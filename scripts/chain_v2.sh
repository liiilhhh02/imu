#!/bin/bash
# Chain v2 — the run that produces the numbers quoted in README/docs after the review fixes.
#
# Differences from chain_run.sh (v1):
#   * the dataset is collected by the *fixed* collector: priors are identified with the
#     command-driven thrust model (identified g_T + identified actuator lag tau), so no offline
#     `fix_priors.py` repair pass is needed or wanted
#   * `anchor` is gone (it was structurally zero and has been deleted)
#   * `torque` is added (it is now properly skipped on episodes where G/T were unidentifiable)
#   * E4 runs after the estimator exists, with the repaired closed-loop setup
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=/home/liiil/Downloads/gym-pybullet-drones:$(pwd)
PY="${PY:-/home/liiil/Downloads/gym-pybullet-drones/.venv-repro/bin/python}"
LOG=results/chain_v2.log
DATA=results/dr_1e7_v2
say() { echo "=== $* ==="; echo "=== $* ===" >> $LOG; }

say "waiting for the in-flight collection to finish"
while pgrep -f "collect_dr.py" > /dev/null; do sleep 20; done
N=$(ls $DATA/*.npz 2>/dev/null | wc -l)
say "collection done: $N shards in $DATA"

say "1) main training (10^7 steps, 10k iters, stage2 at 3k)"
$PY scripts/train_e2e.py --shards $DATA --iters 10000 --stage2 3000 --batch 512 \
    --out results/e2e_v6.pt >> $LOG 2>&1
say "   main training finished"

say "2) E4 closed loop (adjacent-pair dual failure; w_hat drives controller AND INS)"
$PY scripts/e4_closed_loop.py --ckpt results/e2e_v6.pt --flag 3 --ranges 400,1000,4000 \
    --seeds 3 --steps 2000 >> $LOG 2>&1

say "3) E3 ablations on a 2.4e6-step subset"
SUB="--limit 1200"
for ab in bias att phys torque spec prior; do
    say "   ablation: drop L_$ab"
    $PY scripts/train_e2e.py --shards $DATA $SUB --iters 4000 --stage2 1200 --batch 512 \
        --ablate "$ab" --out results/abl2_no_$ab.pt >> $LOG 2>&1
done
say "   ablation: no slow parameter head"
$PY scripts/train_e2e.py --shards $DATA $SUB --iters 4000 --stage2 1200 --batch 512 \
    --no_slow 1 --out results/abl2_noslow.pt >> $LOG 2>&1
say "   ablation: unmasked residual output (no physics parameterisation)"
$PY scripts/train_e2e.py --shards $DATA $SUB --iters 4000 --stage2 1200 --batch 512 \
    --mode residual --out results/abl2_residual.pt >> $LOG 2>&1
say "   ablation: no w_alg anywhere (fully black-box end to end)"
$PY scripts/train_e2e.py --shards $DATA $SUB --iters 4000 --stage2 1200 --batch 512 \
    --mode noalg --out results/abl2_noalg.pt >> $LOG 2>&1
say "   ablation: full model on the same subset (reference row)"
$PY scripts/train_e2e.py --shards $DATA $SUB --iters 4000 --stage2 1200 --batch 512 \
    --out results/abl2_full.pt >> $LOG 2>&1

say "4) per-range stratified check of the main model on the held-out split"
$PY scripts/train_e2e.py --shards $DATA --limit 400 --iters 0 --load results/e2e_v6.pt \
    >> $LOG 2>&1

say "CHAIN V2 DONE"