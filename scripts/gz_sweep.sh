#!/bin/bash
# Paired, resumable 4-src sweep for the Gazebo acceptance measurement (task 3).
#
#   truth   : ground-truth body rate (the denominator)
#   clipped : the saturated gyro (the status quo)
#   alg     : the analytic lever-arm front end alone (isolates the estimator *pipeline*)
#   net     : the trained end-to-end estimator (drives both the controller and the INS)
#
# Every src is flown from the *same* fault draw (the draw is `--seed`), and each finished run appends one
# line to the CSV, so a timeout cannot destroy the sweep: rerun and it skips what is already recorded.
#
# The net/alg rows need the slowed world (me.world real_time_update_rate=200, max_step_size=0.001) so the
# ~18 ms estimator step still corresponds to 5 ms of simulated time; truth/clipped keep up in any world.
# The runner paces on /model_states with auto-detected --substeps, so the control interval is sim-locked.
#
# Usage:  bash scripts/gz_sweep.sh [first_seed] [last_seed] [out_csv]
#         RL_ARGS="--rl_ckpt <name> --rl_dir <dir>" bash scripts/gz_sweep.sh ...   # new policy
set -u
cd "$(dirname "$0")/.."
FIRST=${1:-300}; LAST=${2:-311}
OUT=${3:-results/gz_sweep.csv}
PY="${PY:-/home/liiil/Downloads/gym-pybullet-drones/.venv-repro/bin/python}"
[ -f "$OUT" ] || echo "seed,src,held,z_final,z_med,z_std,att,tilt,rate_err,wall_s" > "$OUT"

run_one() {
    local seed=$1 src=$2
    grep -q "^$seed,$src," "$OUT" && { echo "  skip seed=$seed src=$src (already recorded)"; return; }
    local t0=$SECONDS
    local line
    mkdir -p results/gz_runs
    local rlog="results/gz_runs/seed${seed}_${src}.log"
    line=$(bash -c "source /opt/ros/humble/setup.bash; export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/home/liiil/Downloads/gym-pybullet-drones:$PWD:\$PYTHONPATH; timeout 600 $PY gazebo/gz_runner.py --flag 3 --dps 1000 --steps 2000 --dt 0.005 --src $src --seed $seed ${RL_ARGS:-}" > "$rlog" 2>&1; grep '^RESULT gz' "$rlog" | tail -1)
    if [ -z "$line" ]; then
        echo "$seed,$src,ERROR,,,,,,," >> "$OUT"
        echo "  seed=$seed src=$src ERROR -> $rlog"; tail -3 "$rlog" | sed 's/^/     /'
        return
    fi
    # RESULT gz flag=3 dps=1000 src=truth: z_last= ... (std ...) xy_last= ... |w|= ... wz= ... sat= ... rate_err= ... tilt_med= ...
    zf=$(echo "$line" | sed -n 's/.*z_last= *\([-0-9.]*\).*/\1/p')
    zs=$(echo "$line" | sed -n 's/.*(std *\([-0-9.]*\)).*/\1/p')
    att=$(echo "$line" | sed -n 's/.*att_med= *\([-0-9.]*\).*/\1/p')
    tilt=$(echo "$line" | sed -n 's/.*tilt_med= *\([-0-9.]*\).*/\1/p')
    re=$(echo "$line" | sed -n 's/.*rate_err= *\([-0-9.]*\).*/\1/p')
    held=0
    awk -v z="$zf" -v s="$zs" 'BEGIN{exit !(z>0.3 && s<3)}' && held=1
    echo "$seed,$src,$held,$zf,,$zs,$att,$tilt,$re,$((SECONDS-t0))" >> "$OUT"
    echo "  seed=$seed src=$src held=$held z=$zf std=$zs att=$att tilt=$tilt ($((SECONDS-t0))s)"
}

for seed in $(seq "$FIRST" "$LAST"); do
    for src in truth clipped alg net; do
        run_one "$seed" "$src"
    done
done
echo "=== sweep complete -> $OUT ==="
$PY - "$OUT" <<'EOF'
import sys, csv, math
rows = list(csv.DictReader(open(sys.argv[1])))
def wilson(k, n):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n; z = 1.96
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))
print(f"{'src':>8} {'held':>8} {'rate':>6} {'wilson95':>18} {'z_med(surv)':>12}")
for src in ("truth", "clipped", "alg", "net"):
    r = [x for x in rows if x['src'] == src and x['held'] != 'ERROR']
    if not r: continue
    k = sum(int(x['held']) for x in r); n = len(r)
    zz = sorted(float(x['z_final']) for x in r if int(x['held']))
    lo, hi = wilson(k, n)
    print(f"{src:>8} {k:>3}/{n:<4} {100*k/n:5.0f}% [{lo*100:5.1f}, {hi*100:5.1f}] "
          f"{zz[len(zz)//2] if zz else float('nan'):12.2f}")
EOF