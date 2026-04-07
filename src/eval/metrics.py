
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if not y_true:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def precision_recall_f1(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Tuple[float, float, float]:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a and b)
    fp = sum(1 for a, b in zip(y_true, y_pred) if not a and b)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a and not b)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    scores = []
    for label in labels:
        yt = [x == label for x in y_true]
        yp = [x == label for x in y_pred]
        _, _, f1 = precision_recall_f1(yt, yp)
        scores.append(f1)
    return sum(scores) / len(scores) if scores else 0.0


def jaccard_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    a = set(y_true)
    b = set(y_pred)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def set_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    a = set(y_true)
    b = set(y_pred)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    p = len(a & b) / len(b)
    r = len(a & b) / len(a)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def detection_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Dict[str, float]:
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    return {
        'accuracy': accuracy(y_true, y_pred),
        'precision': p,
        'recall': r,
        'f1': f1,
    }


def type_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    labels = ['none', 'self', 'pair', 'conditional']
    return {
        'accuracy': accuracy(y_true, y_pred),
        'macro_f1': macro_f1(y_true, y_pred, labels=labels),
    }


def segmentation_metrics(y_true_sets: Sequence[Sequence[int]], y_pred_sets: Sequence[Sequence[int]]) -> Dict[str, float]:
    js = [jaccard_score(a, b) for a, b in zip(y_true_sets, y_pred_sets)]
    fs = [set_f1(a, b) for a, b in zip(y_true_sets, y_pred_sets)]
    return {
        'jaccard': sum(js) / len(js) if js else 0.0,
        'f1': sum(fs) / len(fs) if fs else 0.0,
    }
