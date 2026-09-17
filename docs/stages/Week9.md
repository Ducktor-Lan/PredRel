# Week 9 — Target / Loss / Dimension 消融（计划 §9）

> **状态：已完成。** 全量实验一次跑通（18 格 × 9 数据集 × 3 seeds），证据经本地独立验证器
> 逐项复核通过。本地 28/28 测试通过（含 6 subtests）。
> **最终结论（verifier 重算，非解读）：`ablation_v1_complete` —— 描述性完成门；
> ranking 家族（beta-rank @64）是 9 格 beta/alpha 网格内全局最优，但在 9/9 数据集上仍大幅落后于同预算 SupCon；
> adapter 在两个对照格上均未带来一致增益。**

## 1. 结论（数字原文呈现）

- 全局门：`ablation_v1_complete`（描述性完成门，不做路线断言、不移动阈值）
- 全局均值（9 数据集，spearman 排序）：
  supcon 0.4823 ＞ beta-rank_qk64_static 0.3541 ＞ beta-rank_qk128 0.3277 ＞
  beta-rank_qk32 0.3044 ＞ centered-MSErank_qk64 0.2682 ＞ centered-MSE_qk64 0.2681 ＞
  raw-MSErank_qk64 0.2300 ＞ raw-MSE_qk64 0.2277 ＞ alpha-KL_qk64 0.1982 ＞
  raw-MSErank_qk64_adapter 0.1934 ＞ beta-KLrank_qk64_static（Week 8 对照）0.1930 ＞
  alpha-KL_qk128 0.1866 ＞ beta-KL_qk64 0.1779 ＞ beta-KLrank_qk128 0.1757 ＞
  beta-KLrank_qk32 0.1714 ＞ alpha-KL_qk32 0.1632 ＞ beta-KL_qk128 0.1578 ＞
  beta-KL_qk32 0.1564 ＞ beta-KLrank_qk64_adapter 0.1275。
- Week 8 claim 复现：对照格 `beta-KLrank_qk64_static` 的 R@1/R@5/R@10 =
  0.0188/0.0777/0.1509、spearman 0.1930，与冻结值逐位一致（Week 8 全量：0.0188/0.0777/0.1509、0.1930）。
- 含义（一句话）：纯 ranking 目标学真类排序最好，KL（含 alpha-KL）与 MSE 均不如 ranking；
  qk 维度无一致增益（32/64/128 交错）；adapter 两个对照格均为同家族最差或次差。
  科学解读归你；本目录只保证数字可复核。

| 数据集 | 最优学生格（spearman） | 基线 beta-KLrank_qk64 | SupCon |
|---|---|---|---|
| sklearn_breast_cancer | raw-MSErank_qk64_adapter 0.2167 | 0.1385 | 0.4673 |
| sklearn_wine | beta-rank_qk64_static 0.2552 | 0.0785 | 0.5719 |
| sklearn_digits | beta-rank_qk128_static 0.3968 | 0.2351 | 0.5843 |
| openml_diabetes_37_v1 | beta-rank_qk32_static 0.4541 | 0.1680 | 0.5378 |
| openml_vehicle_54_v1 | centered-MSErank_qk64_static 0.5500 | 0.4015 | 0.4872 |
| openml_spambase_44_v1 | beta-rank_qk64_static 0.2886 | 0.1564 | 0.3271 |
| openml_blood_1464_v1 | beta-rank_qk32_static 0.5105 | 0.0583 | 0.4192 |
| openml_ionosphere_59_v1 | beta-rank_qk128_static 0.3425 | 0.1920 | 0.3602 |
| openml_segment_40984_v3 | centered-MSErank_qk64_static 0.4068 | 0.3085 | 0.5852 |

注：上表“最优学生格”按 spearman 取 18 格内最优；7/9 数据集最优为 beta-rank 家族，
vehicle/segment 为 centered-MSErank，breast_cancer 为 raw-MSErank adapter。
9/9 数据集上 SupCon 仍为全局最优（vehicle 上 centered-MSErank 最接近，blood 上 beta-rank 反超 SupCon
为唯一例外——原话：blood 上 beta-rank_qk32 0.5105 ＞ supcon 0.4192）。

## 2. 最终证据（v1 全量，本地已验证）

目录：`reports/evidence/week09-full-d63cbfce4790-03b7674e/`

| 文件 | 内容 |
|---|---|
| `metrics.json`（约 108MB） | 全部数字证据：各数据集/seed/格指标、27 组格指纹、全局门 |
| `eval_rows.jsonl`（约 865MB） | 每行含 Teacher 真类块、各格 student 分数与全套指标（无特征矩阵、无参数张量） |
| `week09_ablation.md` | 人类可读报告（18 格全局均值 + 按 spearman 排序） |
| `dataset_provenance.json` / `dataset_lock.json` | 9 数据集内容 pin 与全量锁 |
| `run_input.json` / `run_status.json` / `runner.json` / `runtime_environment.json` | 运行输入、状态（技术 PASS）、源码 SHA、运行环境 |

run_id：`week09-full-d63cbfce4790-03b7674e`；源码 SHA：
`d63cbfce4790bf1c2063d8a61a57894992c0f3514439e2ed8604f70e7f0a0104`；
数据集清单 SHA：`51ac4b99545177f865669127ade91ff880f0bd2bb338e5073f7348a5a6a56eb7`；
Teacher SHA：`b6575e5b280e134b121d69c5b09d19d1dee3c72e0a231a7a8e6e6048ee643dc4`。

## 3. 版本演进史（为什么本地树 SHA 变了两次）

科学门（描述性完成门）、网格 18 格、学生超参数**全程一字未动**；变的是工程实现与验证口径。

| 本地树 | 改了什么 | 实测结果 |
|---|---|---|
| `d63cbfce…`（远端冻结） | 首版全量协议：verifier 全字段复核 | smoke 通过；full 训练 18 格×27 组全部跑完（865MB rows 落盘），远端 verifier 在全局重聚段 `KeyError: 'teacher_topk_recall_mean'` 失败——verifier 误读了 dataset 级 `_median` 键（`median`） |
| `51f54c2d…`（当前本地） | 仅 verifier 一处键名修正（`mean`→`median`，与 runner `_dataset_summary` 一致）；runner、trainer、manifest、网格、超参数零改动 | full 证据本地验证 **pass**；smoke 证据同步复核 pass；28/28 测试通过 |

证据绑定的是远端冻结源码 `d63cbfce…`（runner.json + metrics 重算均不依赖本地 verifier 文本修正）；
本地树 `51f54c2d…` 仅用于复核。下次如需远端重跑，应先同步本地树并以新 SHA 为准。

## 4. 本目录结构

```text
Week 9/
├── README.md                        ← 本文件：结论、证据位置、版本史、复核指南
├── configs/server_windows.example.yaml      ← 服务器目标（<SSH_USER>@<SERVER_HOST>:<SSH_PORT>, conda base）
├── provenance/                      ← dataset_manifest.json（week09-ablation-v1）+ week01/week09 依赖 + source_manifest.json
├── src/predrel_week9/               ← data/label_anatomy/teacher_bridge/metrics/models/supcon（冻结逻辑）+ losses/train_student/runner（消融推广）
├── scripts/                         ← manifest/preflight/freeze/run/remote_run/verify + 4 个 ps1 包装
├── tests/                           ← 损失网格/训练一致性/mock 端到端/远端契约/manifest/teacher 桥测试（28/28）
├── reports/evidence/                ← ★ 全部有效证据（核心交付物）
│   ├── week09-full-d63cbfce4790-03b7674e/   ← 全量（本地已验证 pass）
│   └── week09-smoke-d63cbfce4790-911fd481/  ← smoke（本地已验证 pass）
└── reports/week09_ablation.template.md      ← 仅模板
```

服务器端（`<SSH_USER>@<SERVER_HOST>:<SSH_PORT>`，`<REMOTE_ROOT>\Week 9`）保留不可变快照
`d63cbfce…` 与运行目录（含 smoke/full）；有效证据已全部下载到本目录 `reports/evidence/` 并验证。

## 5. 复核指南（怎么确认结论是真的）

1. **重跑验证器**（无需服务器，纯本地）：在本目录下运行
   `python scripts/verify_week09_evidence.py --evidence-dir reports/evidence/week09-full-d63cbfce4790-03b7674e --require-runner --require-full --expected-source-sha d63cbfce4790bf1c2063d8a61a57894992c0f3514439e2ed8604f70e7f0a0104`，
   应输出 `{"status": "pass", "mode": "full", ...}`。
   验证器会重推伪切分与真类列映射、逐行重算全部学生/SupCon 指标（含 top-K 精确检查），
   并重聚 seeds/datasets/global；`--retrain`（需 `WEEK09_VERIFY_CACHE` + `WEEK09_VERIFY_WEEK01`）
   可逐格重训比对参数指纹。
2. **看报告**：`reports/evidence/week09-full-d63cbfce4790-03b7674e/week09_ablation.md`
   即 18 格全局均值 + spearman 排序的人类可读版。
3. **看冻结值**：`provenance/dataset_manifest.json`（`week09-ablation-v1`）与
   `provenance/week09_dependency.json`（含 Week 8 claim 冻结值）是全部超参数、网格与授权的唯一来源。
4. **Week 8 复现 check**：对照格 `beta-KLrank_qk64_static` 全局值与
   `week09_dependency.json → week08_claim_baseline` 逐位一致（R@1/R@5/R@10 0.0188/0.0777/0.1509，spearman 0.1930）。
