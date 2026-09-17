# Week 6 总结 — Context Sensitivity（Thunder Zone 3）

> **状态：已完成。** 全量实验一次跑通，证据经本地独立验证器（`--require-full`）逐项复核通过。
> **最终结论（verifier 重算，非解读）：`moderate_context_adapter_candidate` —— 9/9 数据集均为 moderate；建议 CONTEXT_ADAPTER_CANDIDATE，即先用一个简单的 context adapter，再考虑完整的 context-conditioned 模型。**

本文档是 Week 6 的导航页：结论、证据位置、版本演进史、每个目录是什么、怎么复核。
各版本目录本身是冻结历史记录，**不要修改**。

---

## 1. 结论（数字原文呈现）

- 全局决策：`moderate_context_adapter_candidate`
- 弱：[]（无）｜ 强：[]（无）｜ 中等：9/9（breast_cancer、wine、digits、diabetes、vehicle、spambase、blood、ionosphere、segment）
- 27/27 个 support-order 控制全部通过；同序重复控制位级为 0。

| 数据集 | regime | raw Kendall | raw Spearman | Top-3 Jaccard | 偏好翻转率 |
|---|---|---|---|---|---|
| sklearn_breast_cancer | moderate | 0.840 | 0.922 | 0.809 | 0.081 |
| sklearn_wine | moderate | 0.829 | 0.927 | 1.000 | 0.073 |
| sklearn_digits | moderate | 0.593 | 0.737 | 0.589 | 0.206 |
| openml_diabetes_37_v1 | moderate | 0.541 | 0.659 | 0.577 | 0.193 |
| openml_vehicle_54_v1 | moderate | 0.663 | 0.783 | 0.761 | 0.167 |
| openml_spambase_44_v1 | moderate | 0.764 | 0.872 | 0.794 | 0.121 |
| openml_blood_1464_v1 | moderate | 0.555 | 0.679 | 0.588 | 0.228 |
| openml_ionosphere_59_v1 | moderate | 0.717 | 0.841 | 0.687 | 0.138 |
| openml_segment_40984_v3 | moderate | 0.730 | 0.840 | 0.720 | 0.131 |

含义（一句话）：换背景组成后，锚点之间的预测关系**确实会移动**（大约每 5–13 对判断里翻转一对），但骨架仍在——不是固定不变的，也不是天翻地覆。科学解读归你；本目录只保证数字可复核。

---

## 2. 最终证据（v4 全量，64MB，本地已验证）

目录：`evidence/week06-full-537520d59eaa-b05bb854/`

| 文件 | 内容 |
|---|---|
| `metrics.json` | 全部数字证据：科学路线、各数据集 regime、27 组控制记录、`order_control_revision` 绑定 |
| `context_profiles.jsonl` | 216 条固定锚 profile（9 数据集 × 3 seeds × 8 queries；每行含完整 raw 面、alpha、pairwise 差、排名） |
| `week06_context.md` | 人类可读报告（含上表） |
| `dataset_provenance.json` / `dataset_lock.json` | 9 数据集内容 pin 与全量锁 |
| `run_input.json` / `run_status.json` / `runner.json` / `runtime_environment.json` | 运行输入、状态（技术 PASS）、源码 SHA、运行环境 |

run_id：`week06-full-537520d59eaa-b05bb854`；v4 源码 SHA：`537520d59eaa62aa04125e844d3991b9414ab7dbbc2fdda67c2cc174eb71f139`；
数据集清单 SHA：`9cfd2ea23411e1daba281a9eb918be92828bd97c290a374f47e8e964487b0176`。

---

## 3. 版本演进史（为什么有 v1/v2/v3/v4）

科学 gate（weak/moderate/strong 阈值）**全程一字未动**；变的只是“确认计算机没出鬼”的 order control。每次变更都写在对应树的 `provenance/order_control_revision.json` 里，v4 版含完整证据链。

| 版本 | 改了什么 | 实测结果 |
|---|---|---|
| v1（`Week 6`） | 冻结原协议：ID 对齐锚点 raw 逐元素 ≤1e-5 | smoke 两次熔断；runner 失败时不落盘，无测量值 |
| order-audit（`Week 6 order-audit`） | 独立诊断树，不碰冻结协议；同一 smoke 目标 11 种顺序各跑一遍，全部持久化 | 冻结控制重算：alpha 误差 5.2e-08（过）、raw 误差 1.72e-05（超 1.72×）；同序重复精确 0；跨序 raw 差异 0.76e-5…2.43e-5，Kendall/Top-3=1.0、零翻转、预测不变。**结论：阈值低于浮点噪声底，关系完好** |
| v2（`Week 6 v2`） | raw 改关系级控制（零翻转、Kendall=1.0、Top-K=1.0、argmax 不变）+ 绝对护栏 1e-4 | smoke 通过；**全尺寸探针抓到外推风险**：512 行下绝对噪声 2.3–2.7e-4（超护栏），尺度相对仅 7–8e-6，关系恒等 |
| v3（`Week 6 v3`） | 护栏改尺度相对 ≤1e-4（去掉绝对界） | smoke+探针通过；**全量中途熔断**：alpha 逐元素 1e-6（32 行标定）在 512 行下超限（digits:29=1.97e-06，vehicle:17=1.15e-06），关系全部恒等 |
| v4（`Week 6 v4`） | alpha 侧也改关系级：归一化份额死区（ε=1e-4）翻转率、条件 argmax、总质量漂移 ≤1e-3、粗错护栏 ≤1e-4 | **smoke 通过 → 27/27 全覆盖扫描通过 → 探针通过 → 全量一次跑通并验证** |

---

## 4. 本目录结构（唯一 Week 6 文件夹）

```text
Week 6/
├── README.md                    ← 本文件：结论、证据位置、版本史、复核指南
├── EXECUTION_AGENT_PROMPT.md    ← 执行交接（历史记录）
├── configs/ src/ tests/ scripts/← v4 最终冻结协议（scripts/history/ 内的诊断脚本仅历史存档）
├── provenance/                  ← v4 清单 + order_control_revision.json（v1→v4 完整证据链）
├── pyproject.toml
├── reports/week06_context.template.md
├── evidence/                    ← ★ 全部有效证据（核心交付物）
│   ├── week06-full-537520d59eaa-b05bb854/   ← v4 全量（--require-full 已验证）
│   ├── week06-smoke-537520d59eaa-fd57e057/  ← v4 smoke
│   ├── week06-v3-probe-537520d59eaa-fd33973b/ ← v4 探针（前缀 v3 字样是文案残留）
│   ├── week06-order-audit-aa60b3b62082-2e6591d4/ ← order-audit 完整证据
│   ├── route_summary.json / per_dataset_regimes.csv / version_chain.json
│   └── v3_sweep_27combos.json/.csv + alpha_probe_4cases.json
├── history/                     ← 被取代版本的可复核存档
│   ├── v1-frozen/               ← v1 冻结清单（dataset_manifest.v1.json 等）
│   ├── order-audit/             ← 审计源码清单 + 精简审计摘要
│   └── superseded-evidence/     ← v2/v3 中间 smoke 与 probe 证据
└── diagnosis/                   ← v3/v4 诊断脚本快照与本地日志（HISTORY-ONLY）
```

服务器端（`<SSH_USER>@<SERVER_HOST>:<SSH_PORT>`）仍保留各版本远端快照与 run 目录（历史记录，不再使用）；有效证据已全部下载到本目录 `evidence/` 并验证。

---

## 5. 复核指南（怎么确认结论是真的）

1. **重跑验证器**（无需服务器，纯本地）：在本目录下运行 `python scripts/verify_week06_evidence.py --evidence-dir evidence/week06-full-537520d59eaa-b05bb854 --require-runner --require-full --expected-source-sha 537520d59eaa62aa04125e844d3991b9414ab7dbbc2fdda67c2cc174eb71f139`，应输出 `{"status": "pass", "mode": "full", ...}`。验证器会从 profile 行重算全部指标、重算 gate、核对控制记录与源码 SHA。
2. **看报告**：`evidence/week06-full-537520d59eaa-b05bb854/week06_context.md` 即上表的人类可读版。
3. **看变更依据**：`provenance/order_control_revision.json` 含 v1→v4 完整证据链（audit run、探针 run、v3 失败 run、sweep、alpha probe 的精确数值）。
4. **科学 gate 未被动过**：对比任意版本 `provenance/dataset_manifest.json` 的 `gate` 块即可确认。
