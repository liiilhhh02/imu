# 数据与产物政策

## 1. 仓库里**不放**数据

域随机化数据集每个约 570 MB（5000 个 npz 分片），**不进 git**（`.gitignore` 已排除 `results/dr_*/`）。
它们可以由一条命令在 ~10 分钟内重生成：

```bash
export PYTHONPATH=/path/to/gym-pybullet-drones:$(pwd)
python scripts/collect_dr.py --episodes 5000 --workers 14 --steps 2000 --out results/dr_1e7_v2 --seed 1
# 实测：10^7 步 1390 s（7.2 k 步/s，12 workers；v1 时 488 s，多的开销来自每回合多辨识 τ/扫描）
```

每个分片记录：`gyro, accel, u, tom, omega, quat, sat, mask_t, mask, dps, dt, prior, true, mode, flag,
diag_keys, diag_vals`。`true` 里是**实际生效**的机体参数（`M` 回读自 `getDynamicsInfo`，不是请求值）。

## 2. 保留 / 删除的判断（已执行）

| 对象 | 决定 | 理由 |
|---|---|---|
| `results/dr_1e7_v2`（568 MB，10⁷ 步） | **保留（本地，被链式脚本读取）** | 评审修好后重采的唯一数据集（9 先验、命令驱动推力、一拍延迟） |
| `results/dr_1e7`（568 MB，10⁷ 步，**旧管线**） | **已删** | 先验是 7 元组、推力残差用真值、无 τ；与新训练器不兼容，其"修复前"对照数字已写入 `docs/STATUS.md` |
| `results/dr_norand`（562 MB） | **已删** | 采集时机身随机化失效（`reset()` 写回）+ `tom` 用了请求质量而实际质量恒为 1.0 → **残差本身是错的**，无保留价值；其教训已写入 `docs/STATUS.md` 负结果第 7 条 |
| `results/dr_smoke / dr_dbg / dr_val / dr_w / dr_1e6`（82 MB） | **已删** | 被后续数据集取代的中间产物 |
| `results/e2e_v2/v3/v4/e2e_gru/e2e_e2e.pt` | **已删** | 被后续 checkpoint 取代的中间产物 |
| `results/e2e_v5.pt` + `results/abl_*.pt` | **保留并提交** | v1（旧管线）产物：为历史对照保留，**数字不可与 v2 并列引用** |
| `gpd_me/estimator.py` + `scripts/train_estimator.py` + `results/estimator_gru.pt` | **已删** | 早期"残差式"估计器：被 `train_e2e.py` 取代，且其划分是按**窗口**而非按航次，"跨航次泛化"的说法不成立（评审指出）→ 整块删除，结论已在 `docs/STATUS.md` 留字 |
| `results/chain.log` | **不提交**（在 `.gitignore`） | 运行日志体积大且随时变动；其关键数字已摘入 `docs/STATUS.md` |
| `review/*.md` | **提交** | 评审文本是本研究的一部分 |
| `reference/senior/`（1.2 MB） | **不提交**（git-ignored） | 学长的未发表代码，无分发许可；见 `reference/README.md` |

## 3. 产物含义速查

| 文件 | 含义 |
|---|---|
| `e2e_<mode>_*.pt` | `train_e2e.py` 的 checkpoint：`state`（网络权重）+ 四组归一化常量（`xf_m/xf_s/xp_m/xp_s`）。加载时需 `weights_only=False` |
| `abl_*.pt` | 消融：`no_bias/no_att/no_phys/no_torque/no_spec/no_prior`（去掉对应损失）、`noslow`（关慢头）、`residual`（不做饱和轴掩码）、`noalg`（完全不用 `w_alg`）、`full`（参考）。v2 一轮的产物前缀是 `abl2_` |
| `dr_*/` | 域随机化数据集分片 |

## 4. 复现任意一个实验

```bash
# 端到端：采集 → 训练 → 评测
python scripts/collect_dr.py --episodes 5000 --workers 14 --steps 2000 --out results/dr_1e7_v2 --seed 1
python scripts/train_e2e.py --shards results/dr_1e7_v2 --iters 10000 --stage2 3000 --batch 512 \
       --out results/e2e_v6.pt

# 指定消融
python scripts/train_e2e.py --shards results/dr_1e7_v2 --limit 1200 --iters 4000 --ablate att \
       --out results/abl2_no_att.pt

# 闭环 E4（ω̂ 同时驱动控制器与 INS；--seeds = 独立故障抽签数，报的是分布）
python scripts/e4_closed_loop.py --ckpt results/e2e_v6.pt --flag 3 --ranges 1000 --seeds 20 --steps 2000

# 验证层
python scripts/verify_imu.py 0 1000               # IMU 模型 vs pybullet 有限差分
python scripts/verify_observer.py 2 1000 150      # 解析观测器
python scripts/verify_ins_attitude.py             # 完整饱和（含姿态由削顶陀螺积分）
python scripts/verify_priors.py --dir results/dr_1e7_v2 --n 400   # 先验 vs 分片里保存的真值
```

## 5. 如果要重采集（例如修完 BLOCKER 3/4 之后）

注意三条硬约束，否则会得到**看似合理但错误**的数据：

1. 机体随机化必须在**最后一次 `reset()` 之后**应用（`shut_down_rotors()` 内部会 reset）；
2. `tom` 必须用**实际生效**的质量（`getDynamicsInfo` 回读），否则加计残差 `s` 会被系统性污染；
3. 采集器与训练器用的执行器模型必须含**一拍输入延迟**（被控对象在第 k 拍用第 k−1 拍的指令）。

修完评审项之后的正式一轮由 `scripts/chain_v2.sh` 一条命令跑完：

```bash
bash scripts/chain_v2.sh     # 等采集结束 → 主训练(10^7/10k iters) → E4 → 10 组消融 → 分层复核
```

产物命名：`results/e2e_v6.pt`（主模型）、`results/abl2_*.pt`（v2 消融）、`results/chain_v2.log`（全过程日志）。
v1 的 `results/e2e_v5.pt` + `results/abl_*.pt` 是**旧管线**产物，数字不可与 v2 并列引用。

数据集 `results/dr_1e7_v2`（5000 分片，568 MB）同样本地保留、不提交。