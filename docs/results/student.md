# Week 08 - Route B Student v1 on Frozen Real Data

Technical status: PASS
Run ID: week08-full-f086b0996a45-4e22a4d9
Completed at UTC: 2026-09-15T12:38:40.126949+00:00
Mode: full
Week 1 Teacher source SHA-256: b6575e5b280e134b121d69c5b09d19d1dee3c72e0a231a7a8e6e6048ee643dc4
Dataset manifest SHA-256: 4bef0e5ab5b35b0721a9572640dc55ad430ae1c75b42d6d497fa8d1312cc423d
Student: hidden=[128], qk_dim=64, epochs=50, lr=0.001, lambda_ranking=0.5

## Dataset coverage

Executed: sklearn_breast_cancer, sklearn_wine, sklearn_digits, openml_diabetes_37_v1, openml_vehicle_54_v1, openml_spambase_44_v1, openml_blood_1464_v1, openml_ionosphere_59_v1, openml_segment_40984_v3
Frozen full benchmark: sklearn_breast_cancer, sklearn_wine, sklearn_digits, openml_diabetes_37_v1, openml_vehicle_54_v1, openml_spambase_44_v1, openml_blood_1464_v1, openml_ionosphere_59_v1, openml_segment_40984_v3

## Per-dataset medians (teacher-top-K recall / spearman)

| Dataset | static R@1/R@5/R@10 | adapter R@1/R@5/R@10 | supcon R@1/R@5/R@10 | static/../supcon spearman |
| --- | --- | --- | --- | --- |
| sklearn_breast_cancer | 0.0175/0.0386/0.0728 | 0.0088/0.0509/0.0816 | 0.1316/0.2246/0.2711 | 0.1385/0.1119/0.4673 |
| sklearn_wine | 0.0278/0.1278/0.2472 | 0.0000/0.1500/0.2417 | 0.2500/0.3833/0.5278 | 0.0785/0.0957/0.5719 |
| sklearn_digits | 0.0469/0.1609/0.3063 | 0.0391/0.1562/0.2859 | 0.3359/0.5141/0.5508 | 0.2351/0.2065/0.5843 |
| openml_diabetes_37_v1 | 0.0000/0.0281/0.0555 | 0.0000/0.0281/0.0492 | 0.2812/0.3547/0.4125 | 0.1680/0.1293/0.5378 |
| openml_vehicle_54_v1 | 0.0312/0.0938/0.1805 | 0.0078/0.0563/0.0930 | 0.1875/0.3391/0.4031 | 0.4015/0.1441/0.4872 |
| openml_spambase_44_v1 | 0.0078/0.0406/0.0812 | 0.0000/0.0484/0.0719 | 0.2500/0.2484/0.2789 | 0.1564/0.1076/0.3271 |
| openml_blood_1464_v1 | 0.0000/0.0141/0.0312 | 0.0000/0.0219/0.0336 | 0.4766/0.5359/0.5516 | 0.0583/-0.0166/0.4192 |
| openml_ionosphere_59_v1 | 0.0141/0.0732/0.1352 | 0.0141/0.0845/0.1310 | 0.2535/0.3493/0.3944 | 0.1920/0.1711/0.3602 |
| openml_segment_40984_v3 | 0.0234/0.1219/0.2484 | 0.0156/0.0891/0.1813 | 0.3828/0.4844/0.5367 | 0.3085/0.1979/0.5852 |

## Scientific gate (descriptive completion)

Decision: student_v1_complete
Recommendation: STUDENT_V1_COMPLETE Proceed to Week 9 target/loss/dimension ablations

A technical PASS validates execution and provenance only. Preserve the evidence for Week 9 ablations.
