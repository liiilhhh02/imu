"""MAIN EXPERIMENT — adjacent-pair dual-rotor failure under full IMU saturation.

Scenario: mask [0,0,1,1] (props 2 and 3 alive).  Chosen because
  * it needs a high spin (measured |w| = 43 rad/s, |wz| = 38.5 rad/s = 2205 dps),
  * the spin axis sits 24.6 deg off body z, so clipping the gyro distorts the rate *direction*
    (~20.5 deg) and tilts the attitude estimate,
  * no classical controller exists for it (3x2 allocation, tau_x slaved to the thrust), so the
    learned policy is the only working controller.

"Full saturation" = a saturated axis is never observable: every consumer gets the range value.

Configurations (all at the same four gyro ranges):

  A  truth/truth            attitude INS driven by the TRUE rate, controller fed the TRUE rate
                            -> upper bound (still shows how much the INS itself costs)
  B  clipped INS / true rate attitude INS driven by the CLIPPED gyro, controller fed the true rate
                            -> isolates the attitude-propagation damage
  C  all clipped            attitude INS + controller both driven by the CLIPPED gyro
                            -> the baseline failure case
  D  estimator drives INS   attitude INS driven by the RECONSTRUCTED rate, controller also
                            -> the proposal, used consistently
  E  estimator -> controller only   INS still clipped, only the controller gets the estimate
                            -> the half-measure that the first matrix run exposed as insufficient

Metrics: tilt-estimate error (it decides the thrust direction), position accuracy, and the rate
error restricted to the saturated steps.

Run:  PYTHONPATH=<repo>:<me> python scripts/main_experiment.py
"""
import os
import sys

import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, os.path.join(ME, "scripts"))
from verify_ins_attitude import run  # noqa: E402

FLAG = 3
RANGES = (1000.0, 2000.0, 3000.0, 4000.0)
STEPS = 2400
CONFIGS = [
    ("A truth/truth",        dict(att="truth", src="truth",    ins_rate="estimate")),
    ("B ins(clip)/true",     dict(att="ins",   src="truth",    ins_rate="gyro")),
    ("C all clipped",        dict(att="ins",   src="measured", ins_rate="gyro")),
    ("D estimator everywhere", dict(att="ins", src="override", ins_rate="estimate")),
    ("E estimator -> ctrl only", dict(att="ins", src="override", ins_rate="gyro")),
]


def main():
    print("=== MAIN EXPERIMENT: adjacent-pair dual-rotor failure (mask [0,0,1,1]) ===")
    print("both halves of the IMU saturation modelled; 'clipped' = the only value available\n")

    rows = {}
    for dps in RANGES:
        print(f"--- gyro range {dps:.0f} dps "
              f"({np.rad2deg(1.0) * 0 + np.deg2rad(dps):.1f} rad/s) ---")
        print(f"{'config':<26} {'z':>8} {'std':>6} {'xy':>7} {'tilt err':>9} "
              f"{'rate err(sat)':>14} {'sat%':>5}  verdict")
        for name, kw in CONFIGS:
            r = run(stack="rl", flag=FLAG, dps=dps, steps=STEPS, **kw)
            rows[(dps, name)] = r
            ok = "holds" if (r["z"] > 0.3 and abs(r["z"] - 3.7) < 0.8) else "LOST"
            print(f"{name:<26} {r['z']:>8.2f} {r['zstd']:>6.2f} {r['xy']:>7.3f} "
                  f"{r['tilt']:>9.1f} {r['rate_err']:>14.2f} {r['sat']:>5.0f}  {ok}")
        print()

    print("=== does the estimator help, and where? (deltas vs the clipped baseline C) ===")
    print(f"{'dps':>6} {'config':<26} {'d tilt err':>11} {'d xy':>9} {'d z':>9}  verdict")
    for dps in RANGES:
        c = rows[(dps, "C all clipped")]
        for name, _ in CONFIGS:
            r = rows[(dps, name)]
            print(f"{dps:>6.0f} {name:<26} {r['tilt'] - c['tilt']:>+11.1f} "
                  f"{r['xy'] - c['xy']:>+9.3f} {r['z'] - c['z']:>+9.2f}  "
                  f"{'holds' if (r['z'] > 0.3 and abs(r['z'] - 3.7) < 0.8) else 'LOST'}")
        print()


if __name__ == "__main__":
    main()