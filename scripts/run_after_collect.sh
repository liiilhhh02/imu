#!/bin/bash
# waits for the in-flight (inert-randomisation) collection, parks it as an ablation set, then
# re-collects with the corrected randomised plant and runs the full chain on it.
set -u
cd /home/liiil/Downloads/me
export PYTHONPATH=/home/liiil/Downloads/gym-pybullet-drones:/home/liiil/Downloads/me
PY=/home/liiil/Downloads/gym-pybullet-drones/.venv-repro/bin/python
LOG=results/chain.log
say(){ echo "=== $* ==="; echo "=== $* ===" >> $LOG; }
while pgrep -f "collect_dr.py" > /dev/null; do sleep 15; done
say "0) parking the inert-randomisation collection as results/dr_norand (an ablation set)"
rm -rf results/dr_norand && mv results/dr_1e7 results/dr_norand
say "1) collecting 1e7 with the CORRECTED plant randomisation"
$PY scripts/collect_dr.py --episodes 5000 --workers 14 --steps 2000 --out results/dr_1e7 >> $LOG 2>&1
say "2) repairing the priors with the corrected identifiers"
$PY scripts/fix_priors.py --dir results/dr_1e7 --workers 12 >> $LOG 2>&1
say "3) main training"
$PY scripts/train_e2e.py --shards results/dr_1e7 --iters 10000 --stage2 3000 --batch 512 \
    --out results/e2e_v5.pt >> $LOG 2>&1
say "4) E4 closed loop"
$PY scripts/e4_closed_loop.py --ckpt results/e2e_v5.pt --flag 3 --ranges 400,700,1000 \
    --seeds 3 --steps 2000 >> $LOG 2>&1
say "5) E3 ablations (2.4e6-step subset)"
SUB="--limit 1200"
for ab in bias att phys spec anchor prior; do
  say "   drop L_$ab"
  $PY scripts/train_e2e.py --shards results/dr_1e7 $SUB --iters 4000 --stage2 1200 --batch 512 \
      --ablate "$ab" --out results/abl_no_$ab.pt >> $LOG 2>&1
done
say "   no slow head"
$PY scripts/train_e2e.py --shards results/dr_1e7 $SUB --iters 4000 --stage2 1200 --batch 512 \
    --no_slow 1 --out results/abl_noslow.pt >> $LOG 2>&1
say "   unmasked residual output"
$PY scripts/train_e2e.py --shards results/dr_1e7 $SUB --iters 4000 --stage2 1200 --batch 512 \
    --mode residual --out results/abl_residual.pt >> $LOG 2>&1
say "   no w_alg (black-box end to end)"
$PY scripts/train_e2e.py --shards results/dr_1e7 $SUB --iters 4000 --stage2 1200 --batch 512 \
    --mode noalg --out results/abl_noalg.pt >> $LOG 2>&1
say "   full model on the same subset (ablation reference)"
$PY scripts/train_e2e.py --shards results/dr_1e7 $SUB --iters 4000 --stage2 1200 --batch 512 \
    --out results/abl_full.pt >> $LOG 2>&1
say "CHAIN DONE"
