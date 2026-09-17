# Week 8 — Route B Student v1（主模型 v1，计划 §8）

> **状态：已完成。** 全量实验一次跑通，证据经本地独立验证器（`--require-full`）
> 逐项复核通过。本地 25/25 测试通过。
> **最终结论（verifier 重算，非解读）：`student_v1_complete` —— 描述性完成门；
> 学生 v1 在 9/9 数据集上均大幅落后于同预算 SupCon；adapter 未带来一致增益。**

## 1. 结论（数字原文呈现）

- 全局门：`student_v1_complete`（描述性完成门，不做路线断言、不移动阈值）
- 全局均值（9 数据集）：static R@1/R@5/R@10 = 0.019/0.078/0.151（spearman 0.193）；
  adapter = 0.009/0.076/0.130（spearman 0.128）；
  supcon = 0.283/0.382/0.436（spearman 0.482）。
- 含义（一句话）：v1 学生没学会老师的真类排序，简单的 mean-pool adapter
  没帮上忙，同预算对称 SupCon 全面更好。科学解读归你；本目录只保证数字可复核。

| 数据集 | static R@1/R@5/R@10 | adapter R@1/R@5/R@10 | supcon R@1/R@5/R@10 | spearman s/a/c |
|---|---|---|---|---|
| sklearn_breast_cancer | 0.0175/0.0386/0.0728 | 0.0088/0.0509/0.0816 | 0.1316/0.2246/0.2711 | 0.1385/0.1119/0.4673 |
| sklearn_wine | 0.0278/0.1278/0.2472 | 0.0000/0.1500/0.2417 | 0.2500/0.3833/0.5278 | 0.0785/0.0957/0.5719 |
| sklearn_digits | 0.0469/0.1609/0.3063 | 0.0391/0.1562/0.2859 | 0.3359/0.5141/0.5508 | 0.2351/0.2065/0.5843 |
| openml_diabetes_37_v1 | 0.0000/0.0281/0.0555 | 0.0000/0.0281/0.0492 | 0.2812/0.3547/0.4125 | 0.1680/0.1293/0.5378 |
| openml_vehicle_54_v1 | 0.0312/0.0938/0.1805 | 0.0078/0.0563/0.0930 | 0.1875/0.3391/0.4031 | 0.4015/0.1441/0.4872 |
| openml_spambase_44_v1 | 0.0078/0.0406/0.0812 | 0.0000/0.0484/0.0719 | 0.2500/0.2484/0.2789 | 0.1564/0.1076/0.3271 |
| openml_blood_1464_v1 | 0.0000/0.0141/0.0312 | 0.0000/0.0219/0.0336 | 0.4766/0.5359/0.5516 | 0.0583/-0.0166/0.4192 |
| openml_ionosphere_59_v1 | 0.0141/0.0732/0.1352 | 0.0141/0.0845/0.1310 | 0.2535/0.3493/0.3944 | 0.1920/0.1711/0.3602 |
| openml_segment_40984_v3 | 0.0234/0.1219/0.2484 | 0.0156/0.0891/0.1813 | 0.3828/0.4844/0.5367 | 0.3085/0.1979/0.5852 |

## 2. 最终证据（v6 全量，本地已验证）

目录：`reports/evidence/week08-full-f086b0996a45-4e22a4d9/`

| 文件 | 内容 |
|---|---|
| `metrics.json`（约 104MB） | 全部数字证据：各数据集/seed 指标、27 组学生参数、全局门 |
| `eval_profiles.jsonl`（约 329MB，2967 行） | 每行含特征矩阵、Teacher 真类块、三变体全套指标与 top-K 集合 |
| `week08_student.md` | 人类可读报告（含上表） |
| `dataset_provenance.json` / `dataset_lock.json` | 9 数据集内容 pin 与全量锁 |
| `run_input.json` / `run_status.json` / `runner.json` / `runtime_environment.json` | 运行输入、状态（技术 PASS）、源码 SHA、运行环境 |

run_id：`week08-full-f086b0996a45-4e22a4d9`；源码 SHA：
`f086b0996a45ebb1e31d712ec35f125f35d858eac3cea2a66cb1a4b288b6db3f`；
数据集清单 SHA：`4bef0e5ab5b35b0721a9572640dc55ad430ae1c75b42d6d497fa8d1312cc423d`；
Teacher SHA：`b6575e5b280e134b121d69c5b09d19d1dee3c72e0a231a7a8e6e6048ee643dc4`。

## 3. 版本演进史（为什么有多次快照）

科学门（描述性完成门）与学生超参数**全程一字未动**；变的是工程实现与验证口径。

| 快照 | 改了什么 | 实测结果 |
|---|---|---|
| `e6ce25cc…`（v1） | 冻结原协议：Python 双层循环 ranking | smoke 通过；full 在 CPU 上卡死 21 小时无落盘（pair 爆炸），已作废 |
| `1e875c79…`（v2） | ranking 全向量化（padding+mask，数学同 pair 集） | full 约 10 分钟跑完；本地验证失败：spambase spearman 差 5.6e-8（跨进程 residue） |
| `c1f7facb…`（v4） | 训练期 RNG 加固 + verifier 连续值容差 1e-6 | smoke 通过时发现 `use_deterministic_algorithms` 干崩 Teacher（CuBLAS SVD），已 revert 该项 |
| `8863b577…`（v5） | verifier 双层比较（连续值 1e-6 + top-K 精确）+ profile 存 top-K 集合 | full 跑完；验证失败：旧代码跑的 top-K 集合与新 tie-aware 排序差 1–2 个成员（spambase/segment 近 ties） |
| `f086b099…`（v6，当前） | runner/verifier 统一 tie-aware 排序（`tie_tolerance=1e-9`） | **smoke 通过 → full 一次跑通并验证** |

## 4. 本目录结构

```text
Week 8/
├── README.md                        ← 本文件：结论、证据位置、版本史、复核指南
├── configs/server_windows.example.yaml      ← 服务器目标（<SSH_USER>@<SERVER_HOST>:<SSH_PORT>, conda base）
├── provenance/                      ← dataset_manifest.json（week08-student-v1）+ week01/week07 依赖 + source_manifest.json
├── src/predrel_week8/               ← data/label_anatomy/supcon（复用冻结逻辑）+ models/losses/metrics/train_student/runner
├── scripts/                         ← manifest/preflight/freeze/run/remote_run/verify + 4 个 ps1 包装
├── tests/                           ← 损失/指标/训练一致性/mock 端到端/远端契约/manifest/teacher 桥测试（25/25）
├── reports/evidence/                ← ★ 全部有效证据（核心交付物）
│   ├── week08-full-f086b0996a45-4e22a4d9/   ← v6 全量（--require-full 已验证）
│   ├── week08-smoke-f086b0996a45-c571b6b9/  ← v6 smoke
│   └── week08-{full,smoke}-…/               ← 中间版本证据（历史记录，未验证或已取代）
└── reports/week08_student.template.md       ← 仅模板
```

服务器端（`<SSH_USER>@<SERVER_HOST>:<SSH_PORT>`，`<REMOTE_ROOT>\Week 8`）仍保留各版本远端快照与 run 目录（历史记录）；有效证据已全部下载到本目录 `reports/evidence/` 并验证。

## 5. 复核指南（怎么确认结论是真的）

1. **重跑验证器**（无需服务器，纯本地）：在本目录下运行 `python scripts/verify_week08_evidence.py --evidence-dir reports/evidence/week08-full-f086b0996a45-4e22a4d9 --require-runner --require-full --expected-source-sha f086b0996a45ebb1e31d712ec35f125f35d858eac3cea2a66cb1a4b288b6db3f`，应输出 `{"status": "pass", "mode": "full", ...}`。验证器会重推伪切分与真类列映射、恢复冻结学生参数、用 NumPy 重算全部学生/SupCon 分数与指标（含 top-K 精确检查），并重聚 seeds/datasets/global。
2. **看报告**：`reports/evidence/week08-full-f086b0996a45-4e22a4d9/week08_student.md` 即上表的人类可读版。
3. **看冻结值**：`provenance/dataset_manifest.json`（`week08-student-v1`）与 `provenance/week07_dependency.json` 是全部超参数与授权的唯一来源。
