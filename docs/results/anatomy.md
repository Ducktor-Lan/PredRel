# READOUT_ANATOMY_REPORT

Status: `FROZEN` (recompute-only; no new Teacher fits, no model training).

Run ID: `week07-anatomy-9f9544511d03-5d54a8a3`
Week 7 source SHA-256: `9f9544511d03c646d0c68b76ec7adda8158eb81a031179fa2cd85358468c7e15`
Frozen inputs manifest: `provenance/anatomy_manifest.json` (`week07-readout-anatomy-v1`).

## Seven anatomy fields (plan Section 7)

```text
Beyond Label: Strong — 0/6 datasets approximately uniform (Week 3 frozen gate continue_to_week04)
Beyond SupCon: Weak-at-margin — frozen gate 1/6 beyond-SupCon at 0.02 (stop_readout2rep_route); investigator override continued on 18/18 positive seed signs (p ~ 7.6e-06), Week 5 extension recheck adds 1/3
Directionality: Mixed — strong 4/9, weak 1/9, neither majority; dual-space not authorized, symmetric not sufficient alone (Week 5 frozen gate continue_to_week06_context)
Context Dependence: Moderate — 9/9 moderate, weak 0/9, strong 0/9 (Week 6 frozen gate moderate_context_adapter_candidate)
Recommended Model: Route B (Static Directed, Query/Key dual-space) with a simple context adapter first — adapter before any full context-conditioned model
Recommended Teacher Target: beta-KL first (class-residual relation), raw-score ranking second; alpha-KL retains softmax-competition confound (plan Sections 7-8)
Go / Pivot / Stop: Go Route B with adapter — not a pivot to Retrieval Anatomy, not a stop; Week 6 moderate majority is the authorizing gate
```

Recommended route: **Route B: Static Directed with simple context adapter first**.

## Recomputed frozen gates (verifier output, no reinterpretation)

- Week 3: `continue_to_week04`; approximately-uniform IDs: `[]` (0/6).
- Week 4: `stop_readout2rep_route`; faithful IDs (6/6): `["sklearn_breast_cancer", "sklearn_wine", "sklearn_digits", "openml_diabetes_37_v1", "openml_vehicle_54_v1", "openml_spambase_44_v1"]`; beyond-SupCon IDs (1/6): `["openml_spambase_44_v1"]`; investigator override: `CONTINUE_TO_WEEK05_DIRECTIONALITY (Week 4 conclusion Section 6; frozen STOP preserved)`.
- Week 5: `continue_to_week06_context`; strong IDs (4/9): `["sklearn_digits", "openml_spambase_44_v1", "openml_ionosphere_59_v1", "openml_segment_40984_v3"]`; weak IDs (1/9): `["openml_blood_1464_v1"]`; extension beyond-SupCon (1/3): `["openml_ionosphere_59_v1"]`.
- Week 6: `moderate_context_adapter_candidate`; moderate IDs (9/9): `["sklearn_breast_cancer", "sklearn_wine", "sklearn_digits", "openml_diabetes_37_v1", "openml_vehicle_54_v1", "openml_spambase_44_v1", "openml_blood_1464_v1", "openml_ionosphere_59_v1", "openml_segment_40984_v3"]`.

## Per-dataset numbers (copied from inputs/frozen_weeks1to6.json)

### Week 3 — beyond labels (median JS / entropy / top-beta excess)

| dataset | median JS (nats) | median class entropy | median top-beta excess |
|---|---|---|---|
| sklearn_breast_cancer | 0.1469076 | 0.89425089 | 0.06474959 |
| sklearn_wine | 0.11230467 | 0.87644616 | 0.19988595 |
| sklearn_digits | 0.25222617 | 0.70088254 | 0.41799611 |
| openml_diabetes_37_v1 | 0.12369743 | 0.90948943 | 0.05485892 |
| openml_vehicle_54_v1 | 0.28301639 | 0.73420019 | 0.26869718 |
| openml_spambase_44_v1 | 0.2367992 | 0.77593035 | 0.18287514 |

### Week 4 — removal faithfulness at K=3 (median TV diffs)

| dataset | TVdiff(top-random) | win rate vs random | TVdiff(top-supcon) |
|---|---|---|---|
| sklearn_breast_cancer | 0.026549 | 0.781 | 0.014293 |
| sklearn_wine | 0.033691 | 0.875 | 0.010283 |
| sklearn_digits | 0.064893 | 0.938 | 0.017957 |
| openml_diabetes_37_v1 | 0.040505 | 1.0 | 0.009304 |
| openml_vehicle_54_v1 | 0.053019 | 0.969 | 0.018766 |
| openml_spambase_44_v1 | 0.039781 | 0.875 | 0.020755 |

### Week 5 — directionality (median A_F / reciprocity@5 / reversal)

| dataset | median A_F | median recip@5 | median reversal |
|---|---|---|---|
| sklearn_breast_cancer | 0.411923 | 0.679167 | 0.089231 |
| sklearn_wine | 0.402787 | 0.766667 | 0.110796 |
| sklearn_digits | 0.519678 | 0.733333 | 0.1912 |
| openml_diabetes_37_v1 | 0.318142 | 0.641667 | 0.207929 |
| openml_vehicle_54_v1 | 0.425269 | 0.691667 | 0.174684 |
| openml_spambase_44_v1 | 0.495101 | 0.575 | 0.160326 |
| openml_blood_1464_v1 | 0.291567 | 0.754167 | 0.164354 |
| openml_ionosphere_59_v1 | 0.55186 | 0.641667 | 0.190718 |
| openml_segment_40984_v3 | 0.489798 | 0.8125 | 0.167052 |

### Week 6 — context sensitivity (raw-score ranking stability)

| dataset | Kendall | Spearman | Top-K Jaccard | flip rate |
|---|---|---|---|---|
| sklearn_breast_cancer | 0.8400072150072151 | 0.9216269841269842 | 0.8087121212121211 | 0.08062770562770562 |
| sklearn_wine | 0.8285533910533911 | 0.9268278018278019 | 1.0 | 0.07278138528138528 |
| sklearn_digits | 0.5933441558441558 | 0.7372835497835498 | 0.5886363636363636 | 0.2058982683982684 |
| openml_diabetes_37_v1 | 0.5406746031746033 | 0.659361471861472 | 0.5766414141414141 | 0.19291125541125542 |
| openml_vehicle_54_v1 | 0.6627886002886002 | 0.7825577200577201 | 0.7612373737373738 | 0.1672077922077922 |
| openml_spambase_44_v1 | 0.7643398268398269 | 0.8716630591630592 | 0.7936868686868686 | 0.12094155844155843 |
| openml_blood_1464_v1 | 0.5551948051948052 | 0.6786315536315537 | 0.5876262626262627 | 0.22754329004329005 |
| openml_ionosphere_59_v1 | 0.7169913419913421 | 0.8413299663299665 | 0.6869949494949495 | 0.13825757575757575 |
| openml_segment_40984_v3 | 0.7302489177489178 | 0.8399170274170276 | 0.7196969696969697 | 0.1312229437229437 |

## Week 8 hand-off (plan Section 8, single route)

- Backbone: `numerical standardized MLP + categorical embedding, hidden 128, output 64 (plan Section 8)`.
- Loss priority: `["beta-KL", "raw-score ranking"]`.
- Context adapter first: `True`.
- DeepSets only if the adapter is insufficient: `True`.

## Reproducibility record

| Field | Value |
|---|---|
| run_input | `{"schema_version": 1, "mode": "anatomy", "run_id": "week07-anatomy-9f9544511d03-5d54a8a3", "anatomy_manifest": "<REMOTE_ROOT>\\Week 7\\snapshots\\9f9544511d03c646d0c68b76ec7adda8158eb81a031179fa2cd85358468c7e15\\provenance\\anatomy_manifest.json", "teacher_report": "<REMOTE_ROOT>\\Week 7\\snapshots\\Week 1\\reports\\week01_teacher_validation.md", "allow_download": false}` |
| runner | `{"schema_version": 1, "run_id": "week07-anatomy-9f9544511d03-5d54a8a3", "source_sha256": "9f9544511d03c646d0c68b76ec7adda8158eb81a031179fa2cd85358468c7e15", "week01_source_sha256": "b6575e5b280e134b121d69c5b09d19d1dee3c72e0a231a7a8e6e6048ee643dc4", "mode": "anatomy", "snapshot_path": "<REMOTE_ROOT>\\Week 7\\snapshots\\9f9544511d03c646d0c68b76ec7adda8158eb81a031179fa2cd85358468c7e15", "run_root": "<REMOTE_ROOT>\\Week 7\\runs\\week07-anatomy-9f9544511d03-5d54a8a3", "evidence_path": "<REMOTE_ROOT>\\Week 7\\runs\\week07-anatomy-9f9544511d03-5d54a8a3\\evidence", "week06_evidence_root": "<REMOTE_ROOT>\\Week 6 v4\\runs\\week06-full-537520d59eaa-b05bb854\\evidence", "allow_download": false, "completed_at_utc": "2026-09-13T15:42:24.403604+00:00"}` |
| anatomy manifest | `provenance/anatomy_manifest.json` |
| frozen values | `inputs/frozen_weeks1to6.json` |
| verifier | `python scripts/verify_week07_evidence.py --evidence-dir <dir> --require-runner` |
