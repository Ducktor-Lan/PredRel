# Module sources (frozen weekly trees → unified `predrel` package)

Numeric behavior is unchanged; only the import namespace moved from
`predrel_weekN.*` to `predrel.*`. Week labels below are stage names, not
packages.

| Unified module | Source tree | Change vs source |
|---|---|---|
| `predrel/contracts.py`, `fixtures.py`, `provenance.py`, `teacher/*` | Week 1 | verbatim (namespace only) |
| `predrel/synthetic.py` (+`synthetic_types.py`), `synthetic_anatomy.py` | Week 2 `synthetic.py`, `types.py`, `anatomy.py` | `types` renamed to `synthetic_types`; docstrings de-weeked |
| `predrel/data.py` | Week 10 `data.py` (= Week 8 + `5<=n<=30` cap) | docstring notes 6/9/20-dataset reuse |
| `predrel/label_anatomy.py` | Week 10 (= Week 4 `analyze_beyond_supcon`) | adds `analyze_beyond_labels` alias (Stage 3 name, identical numerics) |
| `predrel/teacher_bridge.py` | Week 10 readout+hidden bridge | binds in-package `predrel.teacher`; `week01_source_root` deprecated (audit-only); adds `estimator_role` |
| `predrel/supcon.py` | Week 10 (= Week 8 + RNG/threading hardening) | docstring only |
| `predrel/models.py` | Week 8/9/10 identical | docstring only |
| `predrel/losses.py` | Week 9/10 ablation grid | docstring only |
| `predrel/train_student.py` | Week 10 (= Week 9 + `winner_cell_spec`) | adds `train_route_b_student` alias (Stage 8 beta-KL+rank) |
| `predrel/metrics.py` | Week 10 (`score_benchmark_query` + NDCG/top1rank) | adds `score_eval_query` alias (Stage 8/9 key names) |
| `predrel/baselines.py`, `fusion.py` | Week 10 | verbatim |
| `predrel/removal.py` | Week 4/5 identical | docstring only |
| `predrel/directionality.py` | Week 5 | verbatim |
| `predrel/context_sensitivity.py` | Week 6 | docstring only |
| `predrel/experiments/beyond_labels.py` | Week 3 `runner.py` | imports `..`; `week01_source_root` optional |
| `predrel/experiments/removal.py` | Week 4 `runner.py` | imports `..`; `week01_source_root` optional |
| `predrel/experiments/directionality.py` | Week 5 `runner.py` | imports `..`; `week01_source_root` optional |
| `predrel/experiments/context.py` | Week 6 `runner.py` | provenance → unified `provenance.json`; `context.md` report name |
| `predrel/experiments/anatomy.py`, `gates.py` | Week 7 | verbatim (imports already local) |
| `predrel/experiments/student.py` | Week 8 `runner.py` | provenance → unified; manifest `predrel-student-v1` |
| `predrel/experiments/ablation.py` | Week 9 `runner.py` | provenance → unified; manifest `predrel-ablation-v1` |
| `predrel/experiments/benchmark.py` | Week 10 `runner.py` | provenance → unified; manifest `predrel-benchmark-v1`; `benchmark.md` report name |

## Frozen manifests (unified ids)

The four dataset manifests are byte-identical to their weekly sources
except `manifest_id`:

- `dataset_manifest_context.json`: `predrel-context-v1` (was `week06-context-sensitivity-real-v4`)
- `dataset_manifest_student.json`: `predrel-student-v1` (was `week08-student-v1`)
- `dataset_manifest_ablation.json`: `predrel-ablation-v1` (was `week09-ablation-v1`)
- `dataset_manifest_benchmark.json`: `predrel-benchmark-v1` (was `week10-benchmark-v1`)

Hashes: see `provenance/provenance.json`.
