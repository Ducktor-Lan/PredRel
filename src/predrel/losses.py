"""Ablation loss grid (pure NumPy, no torch import).

Target family (plan Section 7, Week 7 hand-off):
  alpha-KL:   KL(teacher full-row alpha || softmax(student full-row scores))
  beta-KL:    KL(teacher true-class beta || softmax(student true-class scores))
  raw-MSE:    mean squared error of student true-class scores against the
              teacher mean-raw true-class block (verbatim plan Week 10 wording
              "raw l"). Scores train in the teacher's native pre-softmax scale;
              NO teacher-side rescaling, NO per-block normalization.
  centered-MSE: mean squared error against the query-centered teacher block
              (verbatim plan Week 10: "centered l", i.e. plan Section 7 T4
              l'_qi = l_qi - mean_j(l_qj) over the true-class block). Student
              scores are centered identically before the comparison.

Loss family (plan Section 8: KL, MSE, ranking, KL + ranking):
  KL       = alpha-KL or beta-KL according to the ablation target
  MSE      = raw-MSE or centered-MSE according to the ablation target
  ranking  = pairwise logistic ranking on the teacher mean-raw true-class
             block (Week 8 frozen definition)
  KL+rank  = KL + lambda_ranking * ranking (lambda_ranking = 0.5)
  MSE+rank = MSE + lambda_ranking * ranking (lambda_ranking = 0.5)

The torch training loop (train_student.py) uses the same formulas through
autograd; parity is covered by tests comparing forward scores and loss
values between the NumPy references here and one torch training epoch.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import EPSILON, StudentError


def _probability_vector(values: ArrayLike, *, name: str, epsilon: float = EPSILON) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size < 2 or not np.all(np.isfinite(vector)):
        raise StudentError(name + " must be a finite vector with at least two entries")
    if np.any(vector < -float(epsilon)):
        raise StudentError(name + " must be non-negative")
    total = float(vector.sum())
    if total <= 0.0:
        raise StudentError(name + " has no probability mass")
    return np.ascontiguousarray(np.clip(vector, 0.0, None) / total)


def _finite_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size < 2 or not np.all(np.isfinite(vector)):
        raise StudentError(name + " must be a finite vector with at least two entries")
    return np.ascontiguousarray(vector)


def softmax(values: ArrayLike) -> NDArray[np.float64]:
    """Numerically stable softmax over one score vector."""

    vector = _finite_vector(values, name="scores")
    shifted = vector - vector.max()
    exp = np.exp(shifted)
    total = float(exp.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise StudentError("softmax denominator is not positive")
    return np.ascontiguousarray(exp / total)


def kl_loss(teacher_probs: ArrayLike, student_scores: ArrayLike, *, epsilon: float = EPSILON) -> float:
    """KL(teacher_probs || softmax(student_scores)) with epsilon-clipped logs."""

    if float(epsilon) <= 0.0:
        raise StudentError("epsilon must be positive")
    teacher = _probability_vector(teacher_probs, name="teacher_probs", epsilon=epsilon)
    student = softmax(_finite_vector(student_scores, name="student_scores"))
    if teacher.shape != student.shape:
        raise StudentError("teacher distribution and student scores do not align")
    clipped_teacher = np.clip(teacher, float(epsilon), None)
    clipped_student = np.clip(student, float(epsilon), None)
    return float(np.sum(clipped_teacher * (np.log(clipped_teacher) - np.log(clipped_student))))


def beta_kl(teacher_beta: ArrayLike, student_scores: ArrayLike, *, epsilon: float = EPSILON) -> float:
    """Alias kept for the Week 8 claim comparison: beta-KL on the true block."""

    return kl_loss(teacher_beta, student_scores, epsilon=epsilon)


def alpha_kl(teacher_alpha_row: ArrayLike, student_full_scores: ArrayLike, *, epsilon: float = EPSILON) -> float:
    """Alias kept for the Week 10 grid: alpha-KL on the full support row."""

    return kl_loss(teacher_alpha_row, student_full_scores, epsilon=epsilon)


def regression_mse(teacher_values: ArrayLike, student_values: ArrayLike) -> float:
    """Mean squared error of student values against teacher values, native scale."""

    teacher = _finite_vector(teacher_values, name="teacher_values")
    student = _finite_vector(student_values, name="student_values")
    if teacher.shape != student.shape:
        raise StudentError("teacher values and student values do not align")
    return float(np.mean((student - teacher) ** 2))


def center_block(values: ArrayLike) -> NDArray[np.float64]:
    """Query-center one true-class block (plan Section 7 T4)."""

    vector = _finite_vector(values, name="values")
    return np.ascontiguousarray(vector - vector.mean())


def centered_mse(teacher_raw_block: ArrayLike, student_score_block: ArrayLike) -> float:
    """MSE between identically centered teacher and student true-class blocks."""

    teacher = center_block(_finite_vector(teacher_raw_block, name="teacher_raw_block"))
    student = center_block(_finite_vector(student_score_block, name="student_score_block"))
    return regression_mse(teacher, student)


def raw_mse(teacher_raw_block: ArrayLike, student_score_block: ArrayLike) -> float:
    """MSE between student scores and the native-scale teacher raw block."""

    return regression_mse(
        _finite_vector(teacher_raw_block, name="teacher_raw_block"),
        _finite_vector(student_score_block, name="student_score_block"),
    )


def ranking_loss(
    teacher_raw_block: ArrayLike, student_score_block: ArrayLike, *, epsilon: float = EPSILON
) -> float:
    """Mean pairwise logistic ranking loss over unordered true-class pairs."""

    if float(epsilon) <= 0.0:
        raise StudentError("epsilon must be positive")
    teacher = _finite_vector(teacher_raw_block, name="teacher_raw_block")
    student = _finite_vector(student_score_block, name="student_score_block")
    if teacher.shape != student.shape:
        raise StudentError("teacher raw block and student scores do not align")
    count = int(teacher.size)
    terms: list[float] = []
    for first in range(count):
        for second in range(first + 1, count):
            teacher_diff = float(teacher[first] - teacher[second])
            if abs(teacher_diff) <= float(epsilon):
                continue
            student_diff = float(student[first] - student[second])
            sign = 1.0 if teacher_diff > 0.0 else -1.0
            # softplus(-sign * ds) = log1p(exp(-sign * ds)); stable for large inputs.
            value = float(-sign * student_diff)
            terms.append(float(np.logaddexp(0.0, value)))
    if not terms:
        raise StudentError("ranking loss has no comparable teacher pair (all ties)")
    return float(np.mean(np.asarray(terms, dtype=np.float64)))


def total_loss(
    *,
    target: str,
    loss: str,
    teacher_alpha_row: ArrayLike | None = None,
    teacher_beta: ArrayLike | None = None,
    teacher_raw_block: ArrayLike | None = None,
    student_full_scores: ArrayLike | None = None,
    student_scores: ArrayLike | None = None,
    lambda_ranking: float = 0.5,
    epsilon: float = EPSILON,
) -> dict[str, float]:
    """Frozen Week 10 total for one train pseudo-query given a grid cell.

    target selects the regression/distribution view; loss selects the family:
      (alpha, KL)     -> alpha-KL on the full row
      (beta, KL)      -> beta-KL on the true block
      (beta, KL+rank) -> beta-KL + lambda_ranking * ranking
      (raw, MSE)      -> raw-MSE on the true block
      (centered, MSE) -> centered-MSE on the true block
      (raw, MSE+rank) -> raw-MSE + lambda_ranking * ranking
      (centered, MSE+rank) -> centered-MSE + lambda_ranking * ranking
      (beta, rank)    -> ranking on the true-class raw block
    """

    if not np.isfinite(float(lambda_ranking)) or float(lambda_ranking) < 0.0:
        raise StudentError("lambda_ranking must be a finite non-negative number")
    if target not in ("alpha", "beta", "raw", "centered"):
        raise StudentError("target must be one of alpha/beta/raw/centered")
    if loss not in ("KL", "MSE", "rank", "KL+rank", "MSE+rank"):
        raise StudentError("loss must be one of KL/MSE/rank/KL+rank/MSE+rank")
    # Alpha is a distribution target: only KL applies. Raw/centered are
    # regression targets: only MSE applies. Beta supports KL and ranking.
    if target == "alpha" and loss != "KL":
        raise StudentError("alpha target requires the KL loss")
    if target in ("raw", "centered") and loss not in ("MSE", "MSE+rank"):
        raise StudentError(target + " target requires the MSE loss family")
    if target == "beta" and loss not in ("KL", "rank", "KL+rank"):
        raise StudentError("beta target requires the KL/ranking loss family")
    main = 0.0
    rank = 0.0
    if loss in ("KL", "KL+rank"):
        if target == "alpha":
            if teacher_alpha_row is None or student_full_scores is None:
                raise StudentError("alpha-KL needs the teacher full row and student full scores")
            main = alpha_kl(teacher_alpha_row, student_full_scores, epsilon=epsilon)
        else:
            if teacher_beta is None or student_scores is None:
                raise StudentError("beta-KL needs the teacher beta block and student true scores")
            main = beta_kl(teacher_beta, student_scores, epsilon=epsilon)
    elif loss in ("MSE", "MSE+rank"):
        if teacher_raw_block is None or student_scores is None:
            raise StudentError("MSE needs the teacher raw block and student true scores")
        if target == "raw":
            main = raw_mse(teacher_raw_block, student_scores)
        else:
            main = centered_mse(teacher_raw_block, student_scores)
    if loss in ("rank", "KL+rank", "MSE+rank"):
        if teacher_raw_block is None or student_scores is None:
            raise StudentError("ranking needs the teacher raw block and student true scores")
        rank = ranking_loss(teacher_raw_block, student_scores, epsilon=epsilon)
    if loss == "rank":
        total = float(rank)
    elif loss in ("KL+rank", "MSE+rank"):
        total = float(main + float(lambda_ranking) * rank)
    else:
        total = float(main)
    if not np.isfinite(total):
        raise StudentError("total loss is non-finite")
    return {"main": main, "ranking": rank, "lambda_ranking": float(lambda_ranking), "total": total}


__all__ = [
    "beta_kl",
    "alpha_kl",
    "kl_loss",
    "raw_mse",
    "centered_mse",
    "center_block",
    "regression_mse",
    "ranking_loss",
    "total_loss",
    "softmax",
]
