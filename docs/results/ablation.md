# Week 09 - Target/Loss/Dimension Ablation on Frozen Real Data

Technical status: PASS
Run ID: week09-full-d63cbfce4790-03b7674e
Completed at UTC: 2026-09-15T16:33:03.249062+00:00
Mode: full
Week 1 Teacher source SHA-256: b6575e5b280e134b121d69c5b09d19d1dee3c72e0a231a7a8e6e6048ee643dc4
Dataset manifest SHA-256: 51ac4b99545177f865669127ade91ff880f0bd2bb338e5073f7348a5a6a56eb7
Student: hidden=[128], epochs=50, lr=0.001, lambda_ranking=0.5

## Executed cells

beta-KL_qk32_static, beta-KL_qk64_static, beta-KL_qk128_static, beta-rank_qk32_static, beta-rank_qk64_static, beta-rank_qk128_static, beta-KLrank_qk32_static, beta-KLrank_qk64_static, beta-KLrank_qk128_static, alpha-KL_qk32_static, alpha-KL_qk64_static, alpha-KL_qk128_static, raw-MSE_qk64_static, raw-MSErank_qk64_static, centered-MSE_qk64_static, centered-MSErank_qk64_static, beta-KLrank_qk64_adapter, raw-MSErank_qk64_adapter

## Dataset coverage

Executed: sklearn_breast_cancer, sklearn_wine, sklearn_digits, openml_diabetes_37_v1, openml_vehicle_54_v1, openml_spambase_44_v1, openml_blood_1464_v1, openml_ionosphere_59_v1, openml_segment_40984_v3
Frozen full benchmark: sklearn_breast_cancer, sklearn_wine, sklearn_digits, openml_diabetes_37_v1, openml_vehicle_54_v1, openml_spambase_44_v1, openml_blood_1464_v1, openml_ionosphere_59_v1, openml_segment_40984_v3

## Global cell means (teacher-top-K recall / spearman)

| Cell | R@1/R@5/R@10 | spearman |
| --- | --- | --- |
| beta-KL_qk32_static | 0.0147/0.0807/0.1418 | 0.1564 |
| beta-KL_qk64_static | 0.0205/0.0782/0.1511 | 0.1779 |
| beta-KL_qk128_static | 0.0159/0.0868/0.1430 | 0.1578 |
| beta-rank_qk32_static | 0.0243/0.1075/0.1829 | 0.3044 |
| beta-rank_qk64_static | 0.0266/0.1147/0.1895 | 0.3541 |
| beta-rank_qk128_static | 0.0237/0.1068/0.1866 | 0.3277 |
| beta-KLrank_qk32_static | 0.0130/0.0832/0.1463 | 0.1714 |
| beta-KLrank_qk64_static | 0.0188/0.0777/0.1509 | 0.1930 |
| beta-KLrank_qk128_static | 0.0159/0.0894/0.1469 | 0.1757 |
| alpha-KL_qk32_static | 0.0216/0.0873/0.1519 | 0.1632 |
| alpha-KL_qk64_static | 0.0196/0.0886/0.1564 | 0.1982 |
| alpha-KL_qk128_static | 0.0186/0.0905/0.1540 | 0.1866 |
| raw-MSE_qk64_static | 0.0204/0.0968/0.1590 | 0.2277 |
| raw-MSErank_qk64_static | 0.0204/0.0971/0.1595 | 0.2300 |
| centered-MSE_qk64_static | 0.0238/0.0983/0.1664 | 0.2681 |
| centered-MSErank_qk64_static | 0.0229/0.0983/0.1664 | 0.2682 |
| beta-KLrank_qk64_adapter | 0.0095/0.0762/0.1299 | 0.1275 |
| raw-MSErank_qk64_adapter | 0.0196/0.0891/0.1538 | 0.1934 |
| supcon | 0.2832/0.3815/0.4363 | 0.4823 |

## Cell ranking by global spearman (descriptive)

supcon (0.4823), beta-rank_qk64_static (0.3541), beta-rank_qk128_static (0.3277), beta-rank_qk32_static (0.3044), centered-MSErank_qk64_static (0.2682), centered-MSE_qk64_static (0.2681), raw-MSErank_qk64_static (0.2300), raw-MSE_qk64_static (0.2277), alpha-KL_qk64_static (0.1982), raw-MSErank_qk64_adapter (0.1934), beta-KLrank_qk64_static (0.1930), alpha-KL_qk128_static (0.1866), beta-KL_qk64_static (0.1779), beta-KLrank_qk128_static (0.1757), beta-KLrank_qk32_static (0.1714), alpha-KL_qk32_static (0.1632), beta-KL_qk128_static (0.1578), beta-KL_qk32_static (0.1564), beta-KLrank_qk64_adapter (0.1275)

## Scientific gate (descriptive completion)

Decision: ablation_v1_complete
Recommendation: ABLATION_V1_COMPLETE Proceed to Week 10 benchmark scoping with the Week 9 ranking as input

A technical PASS validates execution and provenance only. Target/loss/dimension interpretation belongs to the investigator.
