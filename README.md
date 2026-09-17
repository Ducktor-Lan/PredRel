# PredRel — From Prediction to Representation

Learning predictive relations from a tabular foundation model (TabPFN v3):
does the decoder readout contain instance-level relational structure worth
distilling, and if so, what representation can faithfully compress it?

> **Outcome: the distillation route failed (Outcome C).** The frozen
> 20-dataset × 8-method benchmark (`benchmark_v1_complete`) shows the
> distilled student finishing **last** (spearman 0.3386), behind raw
> Euclidean / PCA / MLP / SupCon baselines — while the Teacher's own hidden
> embeddings score 0.9527. The readout structure is real (Stages 1–5), but
> the Route B student does not learn it. This repo publishes the full
> pipeline, frozen evidence, and the negative result.

## Results (frozen, verifier-recomputed)

- Benchmark (20 datasets × 8 methods × 3 seeds, `benchmark_v1_complete`):
  hidden 0.9527 > readout_profile 0.8449 > pca 0.5414 ≈ raw 0.5410 >
  mlp 0.5079 > supcon 0.4965 > fusion 0.4716 > **student 0.3386** (last).
  Hidden wins 20/20 datasets (tie tolerance 1e-9).
- Route: Go Route B (Static Directed Query/Key dual-space) + simple context
  adapter first; beta-KL first, raw-score ranking second (Stage 5 anatomy).
- History: student v1 0.193 vs SupCon 0.482 (Stage 6); ablation winner
  beta-rank_qk64_static 0.3541 still far below SupCon 0.4823 (Stage 7).
- Full tables: `docs/results/benchmark.md`, `docs/results/student.md`,
  `docs/results/ablation.md`, `docs/results/anatomy.md`.

## Repo layout

```text
predrel_unified/
├── README.md / LICENSE / .gitignore / .gitattributes
├── pyproject.toml                  ← package `predrel` (pip install -e .)
├── src/predrel/                    ← ONE package (no WeekN packages)
│   ├── teacher/                    ← frozen TabPFN v3 extraction
│   ├── data.py / teacher_bridge.py ← frozen dataset/split + readout bridge
│   ├── label_anatomy.py / synthetic*.py / removal.py / directionality.py
│   ├── context_sensitivity.py / models.py / losses.py / train_student.py
│   ├── supcon.py / metrics.py / baselines.py / fusion.py
│   └── experiments/                ← one runner per stage
│       ├── beyond_labels / removal / directionality / context
│       ├── anatomy / student / ablation / benchmark
├── scripts/                        ← manifest / env-preflight / freeze /
│                                      remote_run / run_benchmark
├── tests/                          ← 23 tests, no server needed
├── provenance/                     ← manifests + provenance.json +
│                                      MODULE_SOURCES.md + source_manifest.json
├── docs/                           ← paper idea / stage reports / advisor pack
└── configs/server_windows.example.yaml     ← server target (no secrets)
```

## Quickstart (local, no server)

```bash
pip install -e .
python -m unittest discover -s tests -v        # 23 tests
python scripts/source_manifest.py verify --root .
```

Smoked benchmark with a mock Teacher (exercises all 8 methods, no GPU):

```bash
python - <<'EOF'
import sys; sys.path.insert(0, 'src')
from unittest import mock
import numpy as np, tests.test_mock_pipeline as t
EOF
```

See `tests/test_mock_pipeline.py` for the full mock-benchmark example.

## Server run (frozen)

```bash
python scripts/check_server_environment.py --output env.json
python scripts/freeze_datasets.py --cache-dir <cache> --seed-cache-dir <seed> [--allow-download]
python scripts/run_benchmark.py --mode smoke --run-id smoke-01 --output-dir ./evidence \
  --dataset-cache-dir <cache> --model-cache-dir <tabpfn-v3>
python scripts/run_benchmark.py --mode full --run-id full-01 --output-dir ./evidence \
  --dataset-cache-dir <cache> --model-cache-dir <tabpfn-v3>
```

Remote (immutable snapshot): `scripts/remote_run.py --snapshot-path … --run-root …`.
Leakage boundary: query labels are metric-only, never in Teacher/train.
`TABPFN_TOKEN` lives in the server environment only — never in this repo.

## Provenance

- `provenance/MODULE_SOURCES.md`: every unified module ← frozen weekly source.
- `provenance/provenance.json`: route, winner cell, manifest hashes.
- `provenance/dataset_manifest_{context,student,ablation,benchmark}.json`:
  frozen dataset pins (6/9/20 datasets).
- `provenance/source_manifest.json`: content hash of this source tree
  (`b9991c43…`; regenerates after doc-only edits — numerics unaffected).
