# Week 10 - Formal Benchmark on Frozen Real Data (20 datasets x 8 methods)

Technical status: PASS
Run ID: week10-full-f5cd52e969ba-c48ac2ed
Completed at UTC: 2026-09-17T10:27:17.502940+00:00
Mode: full
Week 1 Teacher source SHA-256: b6575e5b280e134b121d69c5b09d19d1dee3c72e0a231a7a8e6e6048ee643dc4
Dataset manifest SHA-256: f3b8c8a258ac82e2bcfe6c6584e5affb7fdf85eb758a95fddbea67597807d582
Methods: student, supcon, fusion, raw, pca, mlp, hidden, readout_profile
Student epochs: 50, SupCon epochs=50, MLP epochs=50

## Executed methods

student, supcon, fusion, raw, pca, mlp, hidden, readout_profile

## Dataset coverage

Executed: sklearn_breast_cancer, sklearn_wine, sklearn_digits, openml_diabetes_37_v1, openml_vehicle_54_v1, openml_spambase_44_v1, openml_blood_1464_v1, openml_ionosphere_59_v1, openml_segment_40984_v3, openml_iris_61_v1, openml_balance-scale_11_v1, openml_banknote_1462_v1, openml_mfeat-fourier_14_v1, openml_mfeat-factors_12_v1, openml_optdigits_28_v1, openml_page-blocks_30_v1, openml_kc1_1067_v1, openml_magictelescope_1120_v1, openml_pendigits_32_v1, openml_cmc_23_v1
Frozen full benchmark: sklearn_breast_cancer, sklearn_wine, sklearn_digits, openml_diabetes_37_v1, openml_vehicle_54_v1, openml_spambase_44_v1, openml_blood_1464_v1, openml_ionosphere_59_v1, openml_segment_40984_v3, openml_iris_61_v1, openml_balance-scale_11_v1, openml_banknote_1462_v1, openml_mfeat-fourier_14_v1, openml_mfeat-factors_12_v1, openml_optdigits_28_v1, openml_page-blocks_30_v1, openml_kc1_1067_v1, openml_magictelescope_1120_v1, openml_pendigits_32_v1, openml_cmc_23_v1

## Global method means (teacher-top-K recall / spearman vs beta)

| Method | R@1/R@5/R@10 | spearman |
| --- | --- | --- |
| student | 0.0351/0.1271/0.2092 | 0.3386 |
| supcon | 0.3022/0.4096/0.4673 | 0.4965 |
| fusion | 0.1108/0.2499/0.3367 | 0.4716 |
| raw | 0.2978/0.4206/0.4811 | 0.5410 |
| pca | 0.2985/0.4195/0.4795 | 0.5414 |
| mlp | 0.2896/0.4075/0.4597 | 0.5079 |
| hidden | 0.6106/0.7545/0.8134 | 0.9527 |
| readout_profile | 0.5036/0.6292/0.6755 | 0.8449 |

## Method ranking by global spearman (descriptive)

hidden (0.9527), readout_profile (0.8449), pca (0.5414), raw (0.5410), mlp (0.5079), supcon (0.4965), fusion (0.4716), student (0.3386)

## Win counts (per-dataset spearman winner, tie tolerance 1e-9)

student: 0, supcon: 0, fusion: 0, raw: 0, pca: 0, mlp: 0, hidden: 20, readout_profile: 0

## Scientific gate (descriptive completion)

Decision: benchmark_v1_complete
Recommendation: BENCHMARK_V1_COMPLETE Proceed to Week 11 faithfulness scoping with the Week 10 table as input

A technical PASS validates execution and provenance only. Method interpretation belongs to the investigator.
