# PredRel Week 1–10 结果包（给导师）

> 解读声明：本包只呈现冻结证据与 verifier 重算数字；科学解读请导师判断。
> 最新结论见 `Week 10/README.md` §1；复核指南见同文件 §5。

## 0. 一页总览（白话）

- 目标：把 TabPFN 老师心里的"新生和谁像"（readout 打分），提炼成一本以后能直接翻的"相似手册"（向量空间）。
- 现状（Week 10，20 数据集 × 8 方法 × 3 seeds，冻结门 `benchmark_v1_complete`）：
  hidden（老师自己印象）0.9527，20/20 数据集全胜 ＞ readout 画像 0.8449 ＞
  pca 0.5414 ≈ raw 0.5410 ＞ mlp 0.5079 ＞ supcon 0.4965 ＞ fusion 0.4716 ＞
  student（我们训的小徒弟，Week 9 冠军 beta-rank_qk64_static）0.3386，全局垫底。
- 人话：好手册存在（老师自己手里），但徒弟没学会，连量身高体重的土办法（raw/PCA）都不如。
- 前史：Week 8 学生 0.193 vs supcon 0.482；Week 9 十八格最优 beta-rank 0.3541 仍远低于 supcon 0.4823。
  Week 4 曾冻结 `stop_readout2rep_route`（仅 1/6 数据集超出 SupCon），经 investigator override 才继续——导师可重点审这条决策链。

## 1. 本包内容

| 文件 | 说明 | 大小 |
|---|---|---|
| `week10_global_summary.json` | 全局 8 方法均值 + win counts + 每集 winner（verifier 重算值） | 约 2KB |
| `week10_per_dataset_table.json` | 20 数据集 × 8 方法全指标（spearman_vs_beta/raw、NDCG、top1rank、R@1/5/10） | 约 30KB |
| `week10_per_seed_spearman.json` | 60 格每 seed spearman（看方差用） | 约 8KB |
| `week10_benchmark.md` | Week 10 人类可读报告（8 方法全局表 + 排序 + 门） | 约 3KB |
| `week09_ablation.md` | Week 9 十八格消融报告 | 约 4KB |
| `week08_student.md` | Week 8 主模型 v1 报告 | 约 4KB |
| `READOUT_ANATOMY_REPORT.md` | Week 7 七字段解剖总结 | 约 10KB |
| `manifests/` | Week 7/8/9/10 清单哈希 + 依赖链 + 复核命令 | 约 10KB |
| `Week 10/README.md` | Week 10 导航页（结论、证据位置、版本史、复核指南） | 约 6KB |

## 2. 复现（可选，需服务器）

```bash
cd "Week 10"
python scripts/verify_week10_evidence.py --evidence-dir reports/evidence/week10-full-f5cd52e969ba-c48ac2ed --require-runner --require-full --expected-source-sha f5cd52e969bad81d3762070c1a7b4429553f238defd3f8f8580c3c5d82e5ff45
```

应输出 `{"status": "pass", "mode": "full", ...}`。

## 3. 想请导师判断的三件事

1. 当前蒸馏路线（Route B 学生）是否还值得继续，还是转 Outcome C（机制分析论文）？
2. hidden 0.95 vs 学生 0.34 的鸿沟，是否支持"加 hidden 辅助目标"的对照实验？
3. 学生在小数据集（iris/balance-scale/wine）崩盘、大数据集打平，是否指向"伪查询训练信号太少"？
