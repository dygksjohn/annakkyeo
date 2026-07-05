"""이진 분류(스미싱 탐지) 평가 지표.

파인튜닝 모델 평가에도 그대로 재사용한다.
y_true, y_pred 는 1=smishing(양성) / 0=ham(음성).
"""
from __future__ import annotations

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)


def binary_report(y_true: list[int], y_pred: list[int]) -> dict:
    """양성 클래스(스미싱) 기준 precision/recall/f1 + 혼동행렬."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    macro_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[2]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def format_report(name: str, rep: dict) -> str:
    c = rep["confusion"]
    return (
        f"[{name}]  n={rep['n']}\n"
        f"  Accuracy : {rep['accuracy']:.4f}\n"
        f"  Precision: {rep['precision']:.4f}  (스미싱이라 판정한 것 중 실제 스미싱 비율)\n"
        f"  Recall   : {rep['recall']:.4f}  (실제 스미싱 중 잡아낸 비율)\n"
        f"  F1       : {rep['f1']:.4f}\n"
        f"  Macro-F1 : {rep['macro_f1']:.4f}\n"
        f"  Confusion: TP={c['tp']} FN={c['fn']} | FP={c['fp']} TN={c['tn']}"
    )
