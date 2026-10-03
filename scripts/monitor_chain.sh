#!/bin/bash
# Watches the chain and, when it ends, writes results/SUMMARY.md with a per-stage health verdict.
# Needed because the chain itself runs without `set -e`: a crashed stage does not stop the later
# stages, so a silent NaN/failed artifact must be *detected* after the fact rather than assumed away.
set -u
cd /home/liiil/Downloads/me
LOG=results/chain.log
OUT=results/SUMMARY.md

while true; do
    if grep -q "CHAIN DONE" "$LOG" 2>/dev/null; then status="CHAIN DONE"; break; fi
    if ! pgrep -f run_after_collect.sh > /dev/null 2>&1; then status="LAUNCHER GONE (without CHAIN DONE)"; break; fi
    sleep 60
done

{
  echo "# Chain summary — $(date)"
  echo
  echo "final status: **$status**"
  echo
  echo "## stage markers"
  grep -n "^=== " "$LOG" 2>/dev/null | sed 's/^/    /'
  echo
  echo "## artifacts"
  for f in results/e2e_v5.pt results/abl_full.pt results/abl_no_bias.pt results/abl_no_att.pt \
           results/abl_no_phys.pt results/abl_no_spec.pt results/abl_no_anchor.pt \
           results/abl_no_prior.pt results/abl_noslow.pt results/abl_residual.pt results/abl_noalg.pt; do
      if [ -f "$f" ]; then printf "    OK   %-34s %s\n" "$f" "$(du -h "$f" | cut -f1)";
      else                  printf "    MISS %-34s\n" "$f"; fi
  done
  echo
  echo "## errors / NaN in the log (should be empty)"
  grep -nE "Traceback|RuntimeError|CUDA|out of memory|non-finite" "$LOG" 2>/dev/null | head -20 | sed 's/^/    /'
  echo
  echo "## main training: last evaluation block"
  awk '/^=== 3\) main training/,/^=== 4\)/' "$LOG" 2>/dev/null | tail -30
  echo
  echo "## E4 closed loop"
  awk '/^=== 4\) E4/,/^=== 5\)/' "$LOG" 2>/dev/null | tail -20
  echo
  echo "## ablation: final held-out lines (per run)"
  grep -E "^===    (drop|no|unmasked|full)|stable bias|jitter|rate err|INS tilt" "$LOG" 2>/dev/null \
      | awk '/^=== /{print "\n"$0} !/^=== /{print "    "$0}'
  echo
  echo "## collection rates"
  grep -E "steps/s\)" "$LOG" 2>/dev/null | tail -4 | sed 's/^/    /'
} > "$OUT" 2>&1
echo "wrote $OUT ($status)"