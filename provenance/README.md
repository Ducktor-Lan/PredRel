# Provenance

- `source_manifest.json`: content hash of this source tree (regenerate after
  any source change: `python scripts/source_manifest.py write --root .`).
- `provenance.json`: unified route authorization, winner cell, and the four
  dataset-manifest hashes.
- `dataset_manifest_{context,student,ablation,benchmark}.json`: frozen
  dataset pins per stage (6/9/20 datasets; benchmark extends the frozen nine
  with 11 pre-registered extension datasets).
- `MODULE_SOURCES.md`: every unified module ← its frozen weekly source.

Stage evidence (metrics/reports) lives in `docs/results/` (small files).
Full row-level evidence (`eval_rows.jsonl`, full `metrics.json`) is archived
outside git (see `.gitattributes`/releases) — the frozen hashes above pin it.
