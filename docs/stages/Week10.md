# Week 10 — 正式 Benchmark（计划 §10）

> **状态：已完成。** 全量实验一次跑通（20 数据集 × 8 方法 × 3 seeds），证据经本地独立验证器
> 逐项复核通过（`--require-runner --require-full`）。本地 33/33 测试通过。
> **最终结论（verifier 重算，非解读）：`benchmark_v1_complete` —— 描述性完成门；
> hidden 在 20/20 数据集上为 spearman 最优；readout_profile 全局第二；
> 学生（Week 9 winner beta-rank static qk64）全局最差。**

## 1. 结论（数字原文呈现）

- 全局门：`benchmark_v1_complete`（描述性完成门，不做路线断言、不移动阈值）
- 全局均值（20 数据集，spearman 排序）：
  hidden 0.9527 ＞ readout_profile 0.8449 ＞ pca 0.5414 ＞
  raw 0.5410 ＞ mlp 0.5079 ＞ supcon 0.4965 ＞
  fusion 0.4716 ＞ student 0.3386。
- 全局 R@1/R@5/R@10：hidden 0.6106/0.7545/0.8134；
  readout_profile 0.5036/0.6292/0.6755；pca 0.2985/0.4195/0.4795；
  raw 0.2978/0.4206/0.4811；mlp 0.2896/0.4075/0.4597；
  supcon 0.3022/0.4096/0.4673；fusion 0.1108/0.2499/0.3367；
  student 0.0351/0.1271/0.2092。
- Win counts（按数据集 spearman_vs_beta_median，tie 容差 1e-9）：
  hidden 20，其余 7 方法均为 0。
- 含义（一句话）：Teacher 自身隐向量与 readout 画像与真类排序高度一致，
  而蒸馏学生（含与 SupCon 的 late-fusion）在同预算下全面落后；
  SupCon 略优于学生但不如 raw/PCA/MLP。科学解读归你；本目录只保证数字可复核。

| 数据集 | student | supcon | fusion | hidden | profile | winner |
|---|---|---|---|---|---|---|
| sklearn_breast_cancer | 0.2158 | 0.4673 | 0.3794 | 0.9752 | 0.6846 | hidden |
| sklearn_wine | 0.2552 | 0.5719 | 0.4384 | 0.9248 | 0.7479 | hidden |
| sklearn_digits | 0.3391 | 0.5843 | 0.4559 | 0.9457 | 0.8943 | hidden |
| openml_diabetes_37_v1 | 0.3964 | 0.5378 | 0.5521 | 0.9734 | 0.6469 | hidden |
| openml_vehicle_54_v1 | 0.5136 | 0.4872 | 0.5908 | 0.9484 | 0.9002 | hidden |
| openml_spambase_44_v1 | 0.2886 | 0.3271 | 0.3659 | 0.9486 | 0.8552 | hidden |
| openml_blood_1464_v1 | 0.5056 | 0.4192 | 0.5845 | 0.9790 | 0.9585 | hidden |
| openml_ionosphere_59_v1 | 0.3068 | 0.3602 | 0.3658 | 0.9628 | 0.8427 | hidden |
| openml_segment_40984_v3 | 0.3659 | 0.5852 | 0.5523 | 0.9432 | 0.9104 | hidden |
| openml_iris_61_v1 | 0.0586 | 0.5420 | 0.4106 | 0.9227 | 0.8564 | hidden |
| openml_balance-scale_11_v1 | 0.0202 | 0.5589 | 0.2916 | 0.9543 | 0.9540 | hidden |
| openml_banknote_1462_v1 | 0.3260 | 0.4386 | 0.4702 | 0.9202 | 0.9091 | hidden |
| openml_mfeat-fourier_14_v1 | 0.3441 | 0.4952 | 0.4334 | 0.9467 | 0.8249 | hidden |
| openml_mfeat-factors_12_v1 | 0.4000 | 0.5507 | 0.4821 | 0.9392 | 0.7487 | hidden |
| openml_optdigits_28_v1 | 0.3396 | 0.5639 | 0.4503 | 0.9433 | 0.8229 | hidden |
| openml_page-blocks_30_v1 | 0.5011 | 0.5042 | 0.5908 | 0.9596 | 0.8752 | hidden |
| openml_kc1_1067_v1 | 0.3342 | 0.4307 | 0.4347 | 0.9790 | 0.8157 | hidden |
| openml_magictelescope_1120_v1 | 0.4408 | 0.4225 | 0.5427 | 0.9588 | 0.8168 | hidden |
| openml_pendigits_32_v1 | 0.4978 | 0.6952 | 0.6146 | 0.9528 | 0.9127 | hidden |
| openml_cmc_23_v1 | 0.3235 | 0.3876 | 0.4258 | 0.9753 | 0.9213 | hidden |

注：上表为各数据集 spearman_vs_beta_median（student/supcon/fusion/hidden/profile 五列；
raw/pca/mlp 见 `week10_benchmark.md` 与 `metrics.json`）；winner 列为 verifier 重聚的
`per_dataset_winners`（20/20 hidden）。

## 2. 最终证据（v1 全量，本地已验证）

目录：`reports/evidence/week10-full-f5cd52e969ba-c48ac2ed/`

| 文件 | 内容 |
|---|---|
| `metrics.json`（约 47MB） | 全部数字证据：各数据集/seed/方法指标、方法指纹、全局门、win counts |
| `eval_rows.jsonl`（约 917MB，6888 行） | 每行含 Teacher 真类块、各方法分数与全套指标（无特征矩阵、无参数张量） |
| `week10_benchmark.md` | 人类可读报告（8 方法全局均值 + spearman 排序 + win counts） |
| `dataset_provenance.json` / `dataset_lock.json` | 20 数据集内容 pin 与全量锁 |
| `run_input.json` / `run_status.json` / `runner.json` / `runtime_environment.json` | 运行输入、状态（技术 PASS）、源码 SHA、运行环境 |

run_id：`week10-full-f5cd52e969ba-c48ac2ed`；源码 SHA（远端冻结）：
`f5cd52e969bad81d3762070c1a7b4429553f238defd3f8f8580c3c5d82e5ff45`；
数据集清单 SHA：`f3b8c8a258ac82e2bcfe6c6584e5affb7fdf85eb758a95fddbea67597807d582`；
Teacher SHA：`b6575e5b280e134b121d69c5b09d19d1dee3c72e0a231a7a8e6e6048ee643dc4`。

## 3. 版本演进史（为什么本地树 SHA 与远端冻结不同）

科学门（描述性完成门）、20 数据集、8 方法、全部超参数**全程一字未动**；
本地树在证据回传后只改了本 README（定稿结论），源码零改动。

| 本地树 | 说明 |
|---|---|
| `f5cd52e9…`（远端冻结） | 首版全量协议：smoke 通过 → full 一次跑通并经远端 verifier 通过 |
| 当前本地（`provenance/source_manifest.json`） | 仅本 README 定稿；src/scripts/tests/configs/provenance 冻结文件零改动 |

证据绑定的是远端冻结源码 `f5cd52e9…`（runner.json + metrics 重算均不依赖本 README 文本）；
复核证据时以 `--expected-source-sha f5cd52e969bad81d3762070c1a7b4429553f238defd3f8f8580c3c5d82e5ff45` 为准。

## 4. 本目录结构

```text
Week 10/
├── README.md                        ← 本文件：结论、证据位置、版本史、复核指南
├── configs/server_windows.example.yaml      ← 服务器目标（<SSH_USER>@<SERVER_HOST>:<SSH_PORT>, conda base）
├── provenance/                      ← dataset_manifest.json（week10-benchmark-v1）+ week01/week10 依赖 + source_manifest.json
├── src/predrel_week10/              ← data/label_anatomy/metrics/models/supcon（冻结逻辑复用）+ baselines/fusion/teacher_bridge（含 hidden）/train_student/runner（8 方法）
├── scripts/                         ← manifest/preflight/freeze/run/remote_run/verify + 4 个 ps1 包装
├── tests/                           ← 基线/融合/指标/伪管线/远端契约/manifest链/teacher桥测试（33/33）
├── reports/evidence/                ← ★ 全部有效证据（核心交付物）
│   ├── week10-full-f5cd52e969ba-c48ac2ed/   ← 全量（本地已验证 pass）
│   └── week10-smoke-f5cd52e969ba-feda6689/  ← smoke（本地已验证 pass）
└── reports/week10_benchmark.template.md     ← 仅模板
```

服务器端（`<SSH_USER>@<SERVER_HOST>:<SSH_PORT>`，`<REMOTE_ROOT>\Week 10`）保留不可变快照
`f5cd52e9…` 与运行目录（含 smoke/full）；有效证据已全部下载到本目录 `reports/evidence/` 并验证。

## 5. 复核指南（怎么确认结论是真的）

1. **重跑验证器**（无需服务器，纯本地）：在本目录下运行
   `python scripts/verify_week10_evidence.py --evidence-dir reports/evidence/week10-full-f5cd52e969ba-c48ac2ed --require-runner --require-full --expected-source-sha f5cd52e969bad81d3762070c1a7b4429553f238defd3f8f8580c3c5d82e5ff45`，
   应输出 `{"status": "pass", "mode": "full", ...}`。
   验证器会重推伪切分、逐行重算全部 8 方法指标（含 fusion 重推、readout-profile 伪列限制、
   top-K 精确检查与全部诊断指标），并重聚 seeds/datasets/global（含 win counts）。
2. **看报告**：`reports/evidence/week10-full-f5cd52e969ba-c48ac2ed/week10_benchmark.md`
   即 8 方法全局均值 + spearman 排序的人类可读版。
3. **看冻结值**：`provenance/dataset_manifest.json`（`week10-benchmark-v1`）与
   `provenance/week10_dependency.json`（含 Week 9 winner 与 Week 8 claim 冻结值）是全部超参数、方法与授权的唯一来源。
4. **交叉验证**：本轮结论经 3 路独立 agent 交叉验证——A 路重算全局表与 win counts
   （最大偏差 ≤1.2e-16）、B 路审计泄漏边界（6/6 通过，Teacher 桥无 query-label 参数、
   训练仅见伪支持集、证据无特征矩阵、下载开关隔离）、C 路复核来源链。
