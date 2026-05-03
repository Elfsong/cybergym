# GRPO 稳定性分析与训练目标可行性审查

报告时间：2026-04-29 23:00 UTC+8（22:30 启动 → ~30 min Claude+Codex 双向辩论 → 共识）
报告作者：Claude (Opus 4.7) + Codex (gpt-5.4) 的两轮辩论结论。
触发：`394089dd` 训练在跑（当前 R3 完成，validation 0.391 → 0.391 无变化），用户问"为什么 GRPO 老是出问题"+"训练目标到底靠不靠谱"。

---

## TL;DR（给只看一段的人）

**所有历史 GRPO run（90b2ebc4 起，至少包括 9cc99030、c4f76f38、394089dd）都在跑一个被静默打破的 PPO 损失函数。** 根因是 `dual_loops/config.py:127-128` + `planner.py:680-683`：把 `clip_low_threshold/clip_high_threshold` 当成 ε（0.2）传给 Tinker，但 Tinker 把这两个字段解释成**绝对 ratio 上下界**（文档示例 0.9/1.1）。`torch.clamp(ratio, 0.2, 0.2)` 把每个 token 的概率比强制夹到 0.2 这一个点上，产生**asymmetric one-sided unlikelihood training**：A>0 的 token 梯度恒为 0（正向激励被 kill），A<0 的 token 梯度照常流（继续压低概率）。

后果链条：每个 round 模型只学"压低差的 strategy 概率"、从不学"提升好的 strategy 概率" → policy 持续收缩 → 9cc99030 R3 全部 256 rollouts 滚到 milestone 0 的"全 0 collapse"是这个机制走到尽头的必然结果。所有"monotonic decline"和"policy collapse"的历史观察都被这个 bug 污染，**不能用作"训练目标本身不可行"的证据**。

**修法（一行）**：`config.py:127-128` 改 `0.2 / 0.2` → `0.8 / 1.2`，加 assert `0.0 < low < 1.0 < high`，跑 1-2 round sanity pass 验证 substep 0 的 `ppo_clipped_fraction ≈ 0`、`ppo_mean_ratio ≈ 1`、`ppo_kl_div ≈ 0`，再决定下一步。

**修了之后能涨多少**：双方共识 **+3 到 +7 pp 验证 pass_rate（点估 +5 pp）**，**不会是 +20 pp**。原因：planner LoRA 在和 frozen executor 同一个 base（Qwen3.5-27B）上、rank 32 自由度有限、72-iter agent loop + temp 0.7 把 strategy 信号大量稀释、reward 双峰（60% m=0 + ~20% m=7）credit-assignment 模糊。

---

## 1. 致命发现：PPO clip threshold 语义错配（**所有历史 run 公共的 bug**）

### 代码位置

`dual_loops/config.py:127-128`：
```python
ppo_clip_low_threshold: float = 0.2    # ε_low, passed to loss_fn_config
ppo_clip_high_threshold: float = 0.2   # ε_high, passed to loss_fn_config
```
注释里的"ε_low"是误解。

`dual_loops/planner.py:680-683`：
```python
loss_fn_config = {
    "clip_low_threshold":  self.config.ppo_clip_low_threshold,
    "clip_high_threshold": self.config.ppo_clip_high_threshold,
}
```
原值直接转交给 Tinker，没有做 `(1-ε, 1+ε)` 转换。

### Tinker 实际语义（已用 WebFetch 验证 https://tinker-docs.thinkingmachines.ai/tinker/losses/ppo/）

> **Implementation:** `clipped_ratio = torch.clamp(prob_ratio, clip_low_threshold, clip_high_threshold)`
> **Example:** `loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}`

字段是**绝对的 ratio 上下界**，不是 ε。我们传 `(0.2, 0.2)` 等于让 Tinker 做 `torch.clamp(ratio, 0.2, 0.2)`，每个 ratio 都被强制夹到单点 0.2。

### 经验证据：bug 在所有最近 run 中一致出现

读 `metrics.json/substep_metrics`，substep 0（应当 old_policy = current_policy、ratio ≈ 1.0、KL ≈ 0）的实测：

| run | round | substep | mean_ratio | kl_div | clip_fraction |
|---|---|---|---|---|---|
| `9cc99030` | 0 | 0 | 0.897 | 1.383 | **1.000** |
| `9cc99030` | 1 | 0 | 0.896 | 1.381 | **1.000** |
| `9cc99030` | 2 | 0 | 0.912 | 1.137 | **1.000** |
| `9cc99030` | 3 | 0 | 0.896 | 1.386 | **1.000** |
| `394089dd` | 0 | 0 | 0.883 | 1.594 | **1.000** |
| `394089dd` | 1 | 0 | 0.891 | 1.464 | **1.000** |
| `394089dd` | 2 | 0 | 0.893 | 1.441 | **1.000** |
| `394089dd` | 3 | 0 | 0.890 | 1.489 | **1.000** |

**`mean_ratio ≈ 0.88-0.91` 是 PRE-clip 的真实分布**（说明 sampling 端和 training 端的 logp 实际差距只有 ~10%，bf16 + 1660 token 累积下符合预期），**`clip_fraction = 1.0` 是 clamp 范围 [0.2, 0.2] 必然吃掉所有点的结果**。这两个数字的并存恰好是 bug 的典型指纹。

我（Claude）v1 把 ratio = 0.88 + KL = 1.5 误读成"sampling backend numerical drift"，**错了**。Codex 第一轮就指出真正的原因是参数语义错配。

### 推导后果：asymmetric one-sided unlikelihood training

修复前的 PPO 目标在每个 token 上：
```
loss_token = -min(ratio * A, clamp(ratio, 0.2, 0.2) * A) = -min(ratio * A, 0.2 * A)
```

A > 0（高于组均值的 token，应该被强化）：
- `ratio * A ≈ A` > `0.2 * A`
- min 取后者，loss = `-0.2 * A`，**对策略参数梯度恒为 0**（clamp 项常量）。
- **正向激励信号完全 kill**。

A < 0（低于组均值的 token，应该被压低）：
- `ratio * A ≈ A` < `0.2 * A`（A 负，所以 ratio*A 更负）
- min 取前者，loss = `-ratio * A` > 0
- 梯度照常流，方向是降低 ratio = 降低这个 token 的策略概率。
- **负向 unlikelihood 训练正常工作**。

机制结论：**模型只学"压低差的 strategy"，从不学"提升好的 strategy"**。Codex 同意此推导是正确的，但提醒"entropy collapse 不是数学保证，是 plausible 的下游后果"——精确说法应该是"primary proof 是 no targeted positive learning；entropy collapse 是观察到的一致下游表现"。

### 这个 bug 解释了哪些历史观察

| 历史观察 | bug 解释 |
|---|---|
| 6+ 个 run 全部 pass_rate 单调下降 | 一边倒压低 token 概率 → policy 持续收缩 |
| `9cc99030` R3 全部 256 rollouts → milestone 0 | one-sided erosion 走到尽头 |
| `9cc99030` R0 +8.2 pp lift | 偶然把一些"灾难性 base mode"（refusal-loop 之类）压低，milestone 0 下降 → pass_rate 上升；之后没有正向 anchor 可学，慢慢退回 |
| `c4f76f38` 4 round 0.145→0.031 | 同上，慢速版本 |
| `7e91a68e` 噪声 floor [0.018, 0.133] 方差大 | 这个用 lr=0 + skip_grpo，**不受 bug 影响**；测的是真实 task-sample 噪声。它与 bug 无关，但其测出的高方差让历史的"GRPO 在恶化"信号难以与噪声区分 |
| 当前 `394089dd` validation pretrain 0.391 → R001 0.391 | bug 让训练近乎 no-op（4 round 累计的 negative-only 梯度还没积累出明显效果），验证集所以基本不动 |

**重大含义**：`experiment_report.md` 里所有"我们试了 X 但没用"的论断（survivor-bias filter、log1p、clipped_std、γ_strategy=0、lr 减半、skip_uniform_milestone 等），都不能用来证明这些 stabilizer 真的没用 —— 它们是叠加在一个本来就坏的 PPO 损失上的。

---

## 2. 训练目标可行性审查（Q1）：Claude vs Codex 共识

### 训练目标

Planner (Qwen3.5-27B + LoRA r32) → 生成 strategy 文本 → 注入 frozen Executor (Qwen3.5-27B base, OpenHands 72-iter, temp 0.7) → 跑 trajectory → milestone (0..7) → reward → GRPO 反向更新 Planner。

### 双方共识：salvageable as prompt optimization, NOT 强 vulnerability-strategy learning

修了 PPO bug 之后，**目标本身不是 fundamentally broken**。但有几个**和 bug 无关的结构性约束**会把上限拉得不高：

1. **Self-prompting paradox**（缓解版）。Planner LoRA 加在和 frozen executor 同一个 base 上。能学到的东西被限制在"base + rank-32 LoRA 的 prompt-engineering 自由度"内。理论上限是"base model 在最优 prompt 下的能力"。从已知数据看：base + 朴素 OpenHands prompt 在 EVAL 上 26.5%；同 base 用 DashScope Qwen3.6-Max（更强 base）能到 45.5%；Sonnet 4.6 是 33.5%。Planner 训练能往上走的空间存在，但不大。

2. **Credit-assignment 稀释**。Strategy ~1660 tokens（实测均值），但只有 ~50-200 tokens 是真正"导致成功"的（具体 file:line、payload bytes、submit 路径）。advantage 通过 `per_token_adv = adv / n_gen`（`planner.py:451`）均匀广播到所有 strategy token —— 这是**正确的 sum-loss 归一化**（Tinker 在累加，再除一次让每个 datum 贡献接近 advantage 而非 advantage × n_gen），但仍意味着 80%+ 的 token 在接收"我对结果有功"的虚假信号，gradient direction 被信号-噪声比稀释。

3. **Reward 双峰 + executor 噪声**。394089dd R3 milestone 直方图 `{0:155, 1:0, 2:3, 3:0, 4:37, 5:15, 6:5, 7:41}`：60% m=0（Docker CRASH / NO_TRAJ / APRIL-cancel）、几乎不存在 m=1-3、其余主要 m=4-7。组内 advantage 信号 90% 来自"这个 8 人组里有没有 m=7 outlier"。executor temp 0.7 + 72-iter loop 让"同一 strategy 跑两次"在 milestone 上完全可以从 0 跳到 7。

4. **Strategy 长度违反 prompt 指令**。`prompts.py:15` 写 "Output 200-500 tokens"，实测均值 1660，这是 **base Qwen 的 instruction-following failure，不是 LoRA 训出来的**（pretrain 验证就 1660）。Codex 还指出可能伴随的副作用：`planner.py:67-68` 的 `_split_thinking` 在没找到 `</think>` 标记时会把整段当成 strategy —— 也就是说"1660 tokens"里很可能包含**未被剥离的 thinking 泄漏**，进一步稀释了 strategy 的可执行性密度。这是独立的可行性顾虑，不致命但需要 ablation 验证。

### 修了 bug 后的合理预期（双方共识）

**+3 pp 到 +7 pp 验证 pass_rate，点估 +5 pp**。理由：

* 现在 394089dd pretrain 是 0.391（16 task × 4 sample 上的窄基线）；同条件下用 7e91a68e 的方法测的 train pool 噪声 floor 是 0.080 ± 0.041 —— 数量级差很大（因为 pool 和 batch 不同），但证实 base 能力已经被"开发到大部分"。
* 修复后 PPO 真正做"把 m=7 outlier 的 token 概率推上去"的工作，能学到的主要是：剥掉 base 的"啰嗦+refusal-loop"模式、固化"submit-first 节奏 + 具体 payload 描述模板"。这是正向但有限的 prompt-engineering，不是新的能力。
* 不是 +20 pp。同 base 的天花板（Qwen3.6-Max 45.5%）需要换 base，不是 LoRA 能涨到的。

### "loss 单调下降"不是合理目标

GRPO/PPO 的 loss 是 surrogate；advantage 是 group-centered（每个 group 内零均值），所以 loss 大致围绕 0 振荡是健康的。**应该追的指标**：

1. `validation pass_rate` 在 fixed eval 集上的滑动均值（噪声占比由 sample 数控制）
2. `ppo_clipped_fraction:mean` —— 修了 bug 后应当 substep 0 ≈ 0、随 substep 增加缓慢上升到 0.05-0.20 是健康
3. `ppo_kl_div:mean` —— substep 0 应当 ≈ 0；如果在 1 round 内涨到 > 0.5 就是 ratio 漂移过快
4. `advantage_stats.std` —— 0.3-1.5 是健康；接近 0 是组内 reward 没区分度，> 3 是 outlier 主导
5. `used / total_groups` 利用率 —— 现在 16-28%，目标 >50%
6. `unclipped_grad_l2:mean` —— 不发散到 10+ 即可

---

## 3. 不稳定根因的全清单（按严重度排序）

| # | 根因 | 影响 | 来源 |
|--:|---|---|---|
| 1 | PPO clip 语义错配 → asymmetric one-sided training | **致命，主因** | `config.py:127-128`、`planner.py:680-683` |
| 2 | Effective batch 极小（5-9 trainable groups / 32），utilization 16-28% | 高 | `planner.py:415-419` skip_uniform_milestone + bimodal milestone 分布 |
| 3 | 60% rollout 卡 milestone 0（CRASH/NO_TRAJ/APRIL-cancel）, milestone 1-3 几乎空 | 高 | docker contention、APRIL 早停、milestone detector 离散化（`milestones.py:308-328`） |
| 4 | strategy 长度撞 2048 cap，可能含 thinking 泄漏 | 中 | `_split_thinking`（`planner.py:67-89`）在缺 `</think>` 时把全文当 strategy；prompt 写 200-500 但实测 1660 |
| 5 | `kl_beta=0.01` 注释 "reserved (not currently wired into Tinker loss)"（`config.py:123`），无 KL-to-ref 锚 | 中（**修 bug 后再评估，目前 KL=1.5 是 bug 副作用，不是真实 drift**） | 同左 |
| 6 | 4-round 短 run + K=8 在 task-sample 噪声 floor 内，单 round 信号 SNR < 1 | 中 | 7e91a68e 测出的 [0.018, 0.133] 噪声 envelope |
| 7 | Validation 16 task × 4 sample = 64 rollouts，单点 σ 大 | 中 | `config.py:216-223` |
| 8 | reward bimodal → group advantage 几乎都被 m=7 outlier 主导 | 低（修 bug + log1p 即可缓解） | `reward.py:31` log1p 当前在 394089dd 里被关掉了 |

#3 是**底层数据质量问题**，不是 GRPO 算法问题：太多 rollout 因为 docker / 早停而没产生 trajectory。即使 PPO 修好，你仍然只有 ~40% 的 rollout 提供有意义的 milestone 信号。这块的 fix 是工程类的（docker 资源、APRIL 阈值），与 GRPO 训练算法无关。

#4 是另一条独立的 bug 线索 —— 不致命但应该被独立调查（见后面"次优先级动作"）。

---

## 4. 修复优先级与方案（双方共识序列）

### 立即（不需要新 run）

1. **改 PPO clip 边界**：
   - `config.py:127-128`：`ppo_clip_low_threshold = 0.8`、`ppo_clip_high_threshold = 1.2`，注释更新为"absolute ratio bounds (per Tinker docs); equivalent to ε=0.2 PPO."
   - `planner.py:680-683` 加一行 assert：`assert 0.0 < self.config.ppo_clip_low_threshold < 1.0 < self.config.ppo_clip_high_threshold, "Tinker PPO clip thresholds are absolute ratio bounds, not epsilons"`

2. **修 `experiment_report.md` 的解读**：在 TL;DR 顶部加一条 NOTICE，说明 90b2ebc4 起的所有 GRPO run 都受 PPO clip bug 影响，历史结论需要重测。

### Sanity-pass run（1-2 round）

3. **跑一个 1-2 round sanity-pass**，配置完全保持 394089dd 当前状态、**只**改 #1 的 clip 边界。验收标准：
   - substep 0 `ppo_clipped_fraction:mean` 应当 ≤ 0.05
   - substep 0 `ppo_mean_ratio:mean` 应当 ∈ [0.95, 1.05]
   - substep 0 `ppo_kl_div:mean` 应当 ≤ 0.1
   - 任一不通过 → 还有第二个 bug，不要继续 full run。

### 修复后的 first full run（4-12 round）

4. **One variable at a time** 回退保守 default，依次评估：
   - 重新打开 `reward_compression="log1p"`（当前 394089dd 默认 "none"，会让 m=7 advantage 主导组）
   - LR 从 2e-6 提到 5e-6（9cc99030 用过的；修了 PPO 后不再需要那么保守）
   - `skip_uniform_milestone_groups` 保留 ON（这个 guard 仍然合理）
   - **不要**回 `gamma_strategy=0.1` —— 9cc99030 collapse 的诱因之一；length penalty 在修了 PPO 后再不需要
   - **不要**回 `clipped_std`（mean_only 的 Dr.GRPO 路径在修了 PPO 后稳定性更好）

5. **加大 validation 信号**：`validation_batch_size 16 → 32`、`validation_samples_per_task 4 → 8`，每点估的 σ 从 ~0.06 降到 ~0.03。

### 次优先级（独立线索）

6. **strategy 长度 + thinking 泄漏 ablation**：固定 32 task，跑 4 个 condition：(a) 全文 strategy（~1660），(b) 截到 500，(c) blank strategy（只给 task），(d) shuffled strategy（同句不同 task）。如果 (b) ≥ (a) 或 (c) ≈ (a)，说明长 strategy 没用、planner 的 conditioning 信号薄弱，需要重新设计 prompt + 加 instruction tuning 阶段。

7. **改 `_split_thinking`**（`planner.py:67-89`）：当 `</think>` 缺失时，**应当 reject 整个 sample 而不是把整段当成 strategy**。当前的 fallback 行为会让"thinking 模式失败"的 rollout 变成 1600 token 的"长 strategy"污染训练。

### 暂不动（用户 memory 标记 deferred）

8. **KL-to-ref 锚**：用户 memory `feedback_grpo_kl_deferred.md` 写的是"don't propose KL-to-ref wiring; revisit only if PPO clip_fraction/approx_kl show runaway drift"。**修了 #1 之后**，substep 0 KL 应当 ≈ 0，整 round KL 累积应当 ≤ 0.3 —— 如果不是，再 revisit；目前没有证据需要它。

---

## 5. 双方立场记录与最终共识

### Claude v1（被推翻或修正的部分）

- ❌ "ppo_clipped_fraction=1.0 是 sampling backend numerical drift" —— **错了**，是 clip 边界 bug。
- ❌ "训练目标几乎不可能让 loss 稳定下降" —— **过悲观**：所有支持这个论断的历史 run 都被 PPO bug 污染了。
- ❌ "随机 LoRA init 让某些灾难性 strategy 概率偏高，所以 R0 lift" —— **错机制**：LoRA 起点是 base + adapter 初值，不是独立随机 policy。正确机制是"asymmetric gradient 在 R0 一次性把 sampled batch 中的 below-mean token 压下去"。
- ✅ "`per_token_adv = adv / n_gen` 是合理的归一化" —— 正确。Codex 也同意。
- ✅ "Loss 单调下降不是合理目标" —— 正确。Codex 也同意。

### Codex 的关键贡献

- ✅ 第一轮就定位到 PPO clip 边界 bug（`config.py:127` + `planner.py:683`），并引用 Tinker 文档证据。**这是整份报告的 anchor。**
- ✅ 修了 bug 后的合理增益估计 +3 到 +7 pp。
- ✅ 提醒 entropy collapse 不是数学保证，是 plausible 下游后果（精确化我的论断）。
- ✅ 推 sanity-pass-first 而不是 aggressive-config-revert：避免 confounding。
- ✅ 指出 `_split_thinking` 在缺 `</think>` 时把全文当 strategy 的 fallback 行为可能造成 thinking 泄漏，是 strategy 长度异常的 secondary explanation。

### 仍然存在的细微分歧（不影响行动）

- Claude 倾向把"asymmetric gradient → entropy collapse"当成强机制；Codex 坚持"primary proof 是 no positive learning，collapse 是 consistent 但非保证的下游"。**最终采用 Codex 的精确表述**。

---

## 6. 一句话行动建议（One thing to do）

**改 `config.py:127-128` 两个值（0.2 → 0.8/1.2），加 assert，跑 1 round sanity-pass，验证 substep-0 metrics 健康，然后回 4-round one-variable-at-a-time iteration**。

修这一行之前，所有关于"GRPO 训练目标到底能不能 work"的讨论都是无效的 —— 历史所有"它不 work"证据都来自一个被打破的损失函数。
