# Chain summary — 2026年 10月 03日 星期六 23:01:38 CST

final status: **CHAIN DONE**

## stage markers
    1:=== 0) parking the inert-randomisation collection as results/dr_norand (an ablation set) ===
    2:=== 1) collecting 1e7 with the CORRECTED plant randomisation ===
    784:=== 2) repairing the priors with the corrected identifiers ===
    788:=== 3) main training ===
    800:=== training (stage 1: physics head only; stage 2: + parameter head) ===
    854:=== held-out (episode split), saturated frames ===
    2507:=== 4) E4 closed loop ===
    2515:=== E4 closed loop: w_hat drives BOTH the attitude INS and the controller rate input ===
    2555:=== 5) E3 ablations (2.4e6-step subset) ===
    2556:===    drop L_bias ===
    2568:=== training (stage 1: physics head only; stage 2: + parameter head) ===
    2592:=== held-out (episode split), saturated frames ===
    2613:===    drop L_att ===
    2625:=== training (stage 1: physics head only; stage 2: + parameter head) ===
    2649:=== held-out (episode split), saturated frames ===
    2670:===    drop L_phys ===
    2682:=== training (stage 1: physics head only; stage 2: + parameter head) ===
    2706:=== held-out (episode split), saturated frames ===
    2727:===    drop L_spec ===
    2739:=== training (stage 1: physics head only; stage 2: + parameter head) ===
    2748:===    drop L_anchor ===
    2760:=== training (stage 1: physics head only; stage 2: + parameter head) ===
    2784:=== held-out (episode split), saturated frames ===
    2806:===    drop L_prior ===
    2818:=== training (stage 1: physics head only; stage 2: + parameter head) ===
    2842:=== held-out (episode split), saturated frames ===
    2864:===    no slow head ===
    2876:=== training (stage 1: physics head only; stage 2: + parameter head) ===
    2900:=== held-out (episode split), saturated frames ===
    2922:===    unmasked residual output ===
    2934:=== training (stage 1: physics head only; stage 2: + parameter head) ===
    2958:=== held-out (episode split), saturated frames ===
    2980:===    no w_alg (black-box end to end) ===
    2992:=== training (stage 1: physics head only; stage 2: + parameter head) ===
    3016:=== held-out (episode split), saturated frames ===
    3038:===    full model on the same subset (ablation reference) ===
    3050:=== training (stage 1: physics head only; stage 2: + parameter head) ===
    3074:=== held-out (episode split), saturated frames ===
    3096:=== CHAIN DONE ===

## artifacts
    OK   results/e2e_v5.pt                  552K
    OK   results/abl_full.pt                552K
    OK   results/abl_no_bias.pt             552K
    OK   results/abl_no_att.pt              552K
    OK   results/abl_no_phys.pt             552K
    MISS results/abl_no_spec.pt            
    OK   results/abl_no_anchor.pt           552K
    OK   results/abl_no_prior.pt            552K
    OK   results/abl_noslow.pt              552K
    OK   results/abl_residual.pt            552K
    OK   results/abl_noalg.pt               552K

## errors / NaN in the log (should be empty)
    2495:Traceback (most recent call last):
    2502:RuntimeError: CUDA error: device-side assert triggered
    2503:CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
    2504:For debugging consider passing CUDA_LAUNCH_BLOCKING=1
    2505:Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
    2525:Traceback (most recent call last):
    2554:RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
    2740:Traceback (most recent call last):
    2747:RuntimeError: The size of tensor a (47) must match the size of tensor b (512) at non-singleton dimension 1

## main training: last evaluation block
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [47,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [48,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [49,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [50,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [51,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [52,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [53,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [54,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [55,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [56,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [57,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [58,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [59,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [60,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [61,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [62,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94: operator(): block: [6,0,0], thread: [63,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
Traceback (most recent call last):
  File "/home/liiil/Downloads/me/scripts/train_e2e.py", line 330, in <module>
    ee = e[m]
  File "/home/liiil/Downloads/me/scripts/train_e2e.py", line 325, in main
    for dps in sorted(set(dpsf.tolist())):
  File "/home/liiil/Downloads/me/scripts/train_e2e.py", line 324, in bjt
    print(f"    {'dps':>6} {'n_frame':>8} | {'clipped':>18} | {'algebraic':>18} | {'network':>18}")
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

=== 4) E4 closed loop ===

## E4 closed loop
  File "/home/liiil/Downloads/gym-pybullet-drones/.venv-repro/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/liiil/Downloads/gym-pybullet-drones/.venv-repro/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/liiil/Downloads/me/gpd_me/e2e.py", line 149, in forward
    corr = self.slow(torch.cat([h[:, -1], coarse, prior_norm], dim=-1))
  File "/home/liiil/Downloads/gym-pybullet-drones/.venv-repro/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/liiil/Downloads/gym-pybullet-drones/.venv-repro/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/liiil/Downloads/gym-pybullet-drones/.venv-repro/lib/python3.10/site-packages/torch/nn/modules/container.py", line 250, in forward
    input = module(input)
  File "/home/liiil/Downloads/gym-pybullet-drones/.venv-repro/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/liiil/Downloads/gym-pybullet-drones/.venv-repro/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/liiil/Downloads/gym-pybullet-drones/.venv-repro/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
=== 5) E3 ablations (2.4e6-step subset) ===

## ablation: final held-out lines (per run)
      stable bias |mean_t err| : clipped  10.43 | algebraic   2.13 | network   1.10 rad/s
      jitter      (std)        : clipped  29.47 | algebraic  24.63 | network  17.67 rad/s
      rate err : clipped  37.361 | algebraic  34.801 | network  24.866 rad/s
      INS tilt @60 ms : clipped   16.93 | algebraic   15.00 | network   14.55 deg
      by gyro range (saturated frames): bias / jitter / tilt
       dps src            z    std      xy   bias  jitter    tilt   k_hat  verdict

===    drop L_bias ===
      stable bias |mean_t err| : clipped   8.36 | algebraic   2.63 | network   0.69 rad/s
      jitter      (std)        : clipped  26.90 | algebraic  18.91 | network  15.93 rad/s
      rate err : clipped  34.702 | algebraic  25.843 | network  21.651 rad/s
      INS tilt @60 ms : clipped   13.81 | algebraic   12.76 | network   13.06 deg
      by gyro range: frame-level bias/jitter (saturated frames) and window-level tilt@60ms

===    drop L_att ===
      stable bias |mean_t err| : clipped   8.36 | algebraic   2.63 | network   1.08 rad/s
      jitter      (std)        : clipped  26.90 | algebraic  18.91 | network  16.07 rad/s
      rate err : clipped  34.702 | algebraic  25.843 | network  21.870 rad/s
      INS tilt @60 ms : clipped   13.81 | algebraic   12.76 | network   13.00 deg
      by gyro range: frame-level bias/jitter (saturated frames) and window-level tilt@60ms

===    drop L_phys ===
      stable bias |mean_t err| : clipped   8.36 | algebraic   2.63 | network   0.85 rad/s
      jitter      (std)        : clipped  26.90 | algebraic  18.91 | network  16.22 rad/s
      rate err : clipped  34.702 | algebraic  25.843 | network  21.911 rad/s
      INS tilt @60 ms : clipped   13.81 | algebraic   12.76 | network   13.11 deg
      by gyro range: frame-level bias/jitter (saturated frames) and window-level tilt@60ms

===    drop L_spec ===

===    drop L_anchor ===
      stable bias |mean_t err| : clipped   8.82 | algebraic   2.84 | network   0.78 rad/s
      jitter      (std)        : clipped  27.35 | algebraic  18.93 | network  16.00 rad/s
      rate err : clipped  35.293 | algebraic  25.894 | network  21.645 rad/s
      INS tilt @60 ms : clipped   12.65 | algebraic    8.17 | network    7.13 deg
      by gyro range: frame-level bias/jitter (saturated frames) and window-level tilt@60ms

===    drop L_prior ===
      stable bias |mean_t err| : clipped   8.82 | algebraic   2.84 | network   1.04 rad/s
      jitter      (std)        : clipped  27.35 | algebraic  18.93 | network  17.19 rad/s
      rate err : clipped  35.293 | algebraic  25.894 | network  23.462 rad/s
      INS tilt @60 ms : clipped   12.65 | algebraic    8.17 | network    7.20 deg
      by gyro range: frame-level bias/jitter (saturated frames) and window-level tilt@60ms

===    no slow head ===
      stable bias |mean_t err| : clipped   8.82 | algebraic   2.84 | network   0.88 rad/s
      jitter      (std)        : clipped  27.35 | algebraic  18.93 | network  16.64 rad/s
      rate err : clipped  35.293 | algebraic  25.894 | network  22.578 rad/s
      INS tilt @60 ms : clipped   12.65 | algebraic    8.17 | network    7.30 deg
      by gyro range: frame-level bias/jitter (saturated frames) and window-level tilt@60ms

===    unmasked residual output ===
      stable bias |mean_t err| : clipped   8.82 | algebraic   2.84 | network   0.80 rad/s
      jitter      (std)        : clipped  27.35 | algebraic  18.93 | network  15.61 rad/s
      rate err : clipped  35.293 | algebraic  25.894 | network  21.618 rad/s
      INS tilt @60 ms : clipped   12.65 | algebraic    8.17 | network    7.29 deg
      by gyro range: frame-level bias/jitter (saturated frames) and window-level tilt@60ms

===    no w_alg (black-box end to end) ===
      stable bias |mean_t err| : clipped   8.82 | algebraic   2.84 | network   0.61 rad/s
      jitter      (std)        : clipped  27.35 | algebraic  18.93 | network  17.41 rad/s
      rate err : clipped  35.293 | algebraic  25.894 | network  24.901 rad/s
      INS tilt @60 ms : clipped   12.65 | algebraic    8.17 | network   16.14 deg
      by gyro range: frame-level bias/jitter (saturated frames) and window-level tilt@60ms

===    full model on the same subset (ablation reference) ===
      stable bias |mean_t err| : clipped   8.82 | algebraic   2.84 | network   1.03 rad/s
      jitter      (std)        : clipped  27.35 | algebraic  18.93 | network  16.23 rad/s
      rate err : clipped  35.293 | algebraic  25.894 | network  21.814 rad/s
      INS tilt @60 ms : clipped   12.65 | algebraic    8.17 | network    7.40 deg
      by gyro range: frame-level bias/jitter (saturated frames) and window-level tilt@60ms

## collection rates
      4976/5000 episodes  9952000 steps    516s  ( 19278 steps/s)
      4984/5000 episodes  9968000 steps    517s  ( 19296 steps/s)
      4992/5000 episodes  9984000 steps    518s  ( 19292 steps/s)
      5000/5000 episodes  10000000 steps    519s  ( 19286 steps/s)
