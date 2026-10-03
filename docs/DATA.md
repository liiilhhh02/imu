# 数据与产物政策

## 1. 仓库里**不放**数据

域随机化数据集每个约 570 MB（5000 个 npz 分片），**不进 git**（`.gitignore` 已排除 `results/dr_*/`）。
它们可以由一条命令在 ~10 分钟内重生成：

```bash
export PYTHONPATH=/path/to/gym-pybullet-drones:$(pwd)
python scripts/collect_dr.py --episodes 5000 --workers 14 --steps 2000 --out results/dr_1e7
# 实测：10^6 步 53 s（18.8 k 步/s）；10^7 步 488 s
```

每个分片记录：`gyro, accel, u, tom, omega, quat, sat, mask_t, mask, dps, dt, prior, true, mode, flag,
diag_keys, diag_vals`。`true` 里是**实际生效**的机体参数（`M` 回读自 `getDynamicsInfo`，不是请求值）。

## 2. 保留 / 删除的判断（已执行）

| 对象 | 决定 | 理由 |
|---|---|---|
| `results/dr_1e7`（568 MB，10⁷ 步） | **保留（本地）** | 当前唯一的正确随机化数据集；正在被消融读取，不能删 |
| `results/dr_norand`（562 MB） | **已删** | 采集时机身随机化失效（`reset()` 写回）+ `tom` 用了请求质量而实际质量恒为 1.0 → **残差本身是错的**，无保留价值；其教训已写入 `docs/STATUS.md` 负结果第 7 条 |
| `results/dr_smoke / dr_dbg / dr_val / dr_w / dr_1e6`（82 MB） | **已删** | 被 `dr_1e7` 取代的中间数据集 |
| `results/e2e_v2/v3/v4/e2e_gru/e2e_e2e.pt` | **已删** | 被 `e2e_v5.pt` 取代的中间 checkpoint |
| `results/e2e_v5.pt`（552 KB） | **保留并提交** | 当前主 checkpoint（受 BLOCKER 影响，见 STATUS） |
| `results/estimator_gru.pt`（84 KB） | **保留并提交** | 早期"残差式"估计器，作为对照保留 |
| `results/abl_*.pt`（9 × 552 KB） | **保留并提交** | E3 消融表的产物 |
| `results/chain.log` | **不提交**（在 `.gitignore`） | 运行日志体积大且随时变动；其关键数字已摘入 `docs/STATUS.md` |
| `review/*.md` | **提交** | 评审文本是本研究的一部分 |
| `reference/senior/`（1.2 MB） | **不提交**（git-ignored） | 学长的未发表代码，无分发许可；见 `reference/README.md` |

## 3. 产物含义速查

| 文件 | 含义 |
|---|---|
| `e2e_<mode>_*.pt` | `train_e2e.py` 的 checkpoint：`state`（网络权重）+ 四组归一化常量（`xf_m/xf_s/xp_m/xp_s`）。加载时需 `weights_only=False` |
| `estimator_gru.pt` | 早期残差式估计器（`train_estimator.py`） |
| `abl_*.pt` | 消融：`no_bias/no_att/no_phys/no_spec/no_anchor/no_prior`（去掉对应损失）、`noslow`（关慢头）、`residual`（不做饱和轴掩码）、`noalg`（完全不用 `w_alg`）、`full`（参考） |
| `dr_*/` | 域随机化数据集分片 |

## 4. 复现任意一个实验

```bash
# 端到端：采集 → 训练 → 评测
python scripts/collect_dr.py --episodes 5000 --workers 14 --steps 2000 --out results/dr_1e7
python scripts/train_e2e.py --shards results/dr_1e7 --iters 10000 --stage2 3000 --batch 512 \
       --out results/e2e.pt

# 指定消融
python scripts/train_e2e.py --shards results/dr_1e7 --limit 1200 --iters 4000 --ablate att --out results/abl_no_att.pt

# 闭环 E4（ω̂ 同时驱动控制器与 INS）
python scripts/e4_closed_loop.py --ckpt results/e2e.pt --flag 3 --ranges 400,700,1000 --seeds 3 --steps 2000

# 验证层
python scripts/verify_imu.py 0 1000          # IMU 模型 vs pybullet 有限差分
python scripts/verify_observer.py 2 1000 150 # 解析观测器
python scripts/verify_ins_attitude.py        # 完整饱和（含姿态由削顶陀螺积分）
```

## 5. 如果要重采集（例如修完 BLOCKER 3/4 之后）

注意两条硬约束，否则会得到**看似合理但错误**的数据：

1. 机体随机化必须在**最后一次 `reset()` 之后**应用（`shut_down_rotors()` 内部会 reset）；
2. `tom` 必须用**实际生效**的质量（`getDynamicsInfo` 回读），否则加计残差 `s` 会被系统性污染。