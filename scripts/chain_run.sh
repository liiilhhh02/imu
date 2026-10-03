#!/bin/bash
# Chained overnight run: repair the priors, retrain, run E4 closed loop, run the E3 ablations.
# Waits for any in-flight collector/trainer first (they write/read the same shards).
set -u
cd /home/liiil/Downloads/me
export PYTHONPATH=/home/liiil/Downloads/gym-pybullet-drones:/home/liiil/Downloads/me
PY=/home/liiil/Downloads/gym-pybullet-drones/.venv-repro/bin/python
LOG=results/chain.log
say() { echo "=== $* ==="; echo "=== $* ===" >> $LOG; }

say "waiting for in-flight collect/train"
while pgrep -f "collect_dr.py|train_e2e.py|e4_closed_loop.py" > /dev/null; do sleep 20; done

say "1) repair priors in dr_1e7 (accelerometer DC was not removed during collection)"
$PY scripts/fix_priors.py --dir results/dr_1e7 --workers 12 >> $LOG 2>&1

say "2) main training on the repaired 1e7 data (bias-penalised, 60 ms tilt metric)"
$PY scripts/train_e2e.py --shards results/dr_1e7 --iters 10000 --stage2 3000 --batch 512 \
    --out results/e2e_v5.pt >> $LOG 2>&1

say "3) E4 closed loop (adjacent-pair dual failure, w_hat drives controller + INS)"
$PY scripts/e4_closed_loop.py --ckpt results/e2e_v5.pt --flag 3 --ranges 400,700,1000 \
    --seeds 3 --steps 2000 >> $LOG 2>&1

say "4) E3 ablations (subset of shards for speed: 1200 episodes = 2.4e6 steps)"
SUB="--limit 1200"
for ab in att phys spec bias anchor prior; do
    say "   ablation: drop L_$ab"
    $PY scripts/train_e2e.py --shards results/dr_1e7 $SUB --iters 4000 --stage2 1200 --batch 512 \
        --ablate "$ab" --out results/abl_no_$ab.pt >> $LOG 2>&1
done
say "   ablation: no slow parameter head"
$PY scripts/train_e2e.py --shards results/dr_1e7 $SUB --iters 4000 --stage2 1200 --batch 512 \
    --no_slow 1 --out results/abl_noslow.pt >> $LOG 2>&1
say "   ablation: unmasked residual output (no physics parameterisation)"
$PY scripts/train_e2e.py --shards results/dr_1e7 $SUB --iters 4000 --stage2 1200 --batch 512 \
    --mode residual --out results/abl_residual.pt >> $LOG 2>&1
say "   ablation: no w_alg anywhere (fully black-box end to end)"
$PY scripts/train_e2e.py --shards results/dr_1e7 $SUB --iters 4000 --stage2 1200 --batch 512 \
    --mode noalg --out results/abl_noalg.pt >> $LOG 2>&1
say "   ablation: full model on the same subset (reference for the ablation table)"
$PY scripts/train_e2e.py --shards results/dr_1e7 $SUB --iters 4000 --stage2 1200 --batch 512 \
    --out results/abl_full.pt >> $LOG 2>&1

say "CHAIN DONE"