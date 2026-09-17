"""PredRel unified package: TabPFN predictive-relation analysis and distillation.

Single import namespace for the whole project. Week structure is preserved
as experiment stages (not separate packages):

- ``predrel.teacher``: frozen TabPFN v3 Teacher extraction (Week 1 snapshot).
- ``predrel.synthetic`` / ``predrel.synthetic_anatomy``: synthetic fixtures and
  method checks (Week 2).
- ``predrel.data`` / ``predrel.label_anatomy`` / ``predrel.teacher_bridge``:
  shared frozen dataset/split contracts and readout bridge (Weeks 3-10).
- ``predrel.removal``: same-class removal faithfulness (Week 4; reused Week 5).
- ``predrel.directionality``: directed-matrix R vs symmetric S (Week 5).
- ``predrel.context_sensitivity``: context-profile stability (Week 6).
- ``predrel.models`` / ``predrel.losses`` / ``predrel.train_student``:
  Route B static-directed student, ablation loss grid, torch training
  (Weeks 8-10; Week 10 adds ``winner_cell_spec``).
- ``predrel.supcon``: matched supervised-contrastive baseline (Weeks 4-10).
- ``predrel.metrics``: frozen eval metrics incl. 8-method benchmark scoring
  (Weeks 8-10; ``score_benchmark_query`` supersedes ``score_eval_query``).
- ``predrel.baselines`` / ``predrel.fusion``: raw/PCA/MLP baselines and
  student+SupCon late fusion (Week 10).
- ``predrel.experiments.*``: one runner per stage (beyond_labels, removal,
  directionality, context, anatomy, student, ablation, benchmark).

Provenance note: this package consolidates the ten frozen weekly trees
(source SHAs recorded in ``provenance/MODULE_SOURCES.md``). Frozen numeric
behavior is unchanged; only the import namespace moved from
``predrel_weekN.*`` to ``predrel.*``.
"""

__all__ = [
    "baselines",
    "context_sensitivity",
    "contracts",
    "data",
    "directionality",
    "fixtures",
    "fusion",
    "label_anatomy",
    "losses",
    "metrics",
    "models",
    "provenance",
    "removal",
    "supcon",
    "synthetic",
    "synthetic_anatomy",
    "teacher_bridge",
    "train_student",
]
