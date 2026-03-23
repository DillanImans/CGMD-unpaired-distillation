from __future__ import annotations
from collections import defaultdict
import numpy as np


def confusion_and_balanced_acc(y_true: np.ndarray, y_pred: np.ndarray):
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    recall0 = cm[0, 0] / max(cm[0].sum(), 1)
    recall1 = cm[1, 1] / max(cm[1].sum(), 1)
    bal_acc = 0.5 * (recall0 + recall1)
    acc = (y_true == y_pred).mean() if len(y_true) else 0.0
    return acc, bal_acc, cm , np.array([recall0, recall1], dtype=np.float32)


def aggregate_patient_probs(patient_ids, probs, labels):
    # mean prob per patient
    pid_to_probs = defaultdict(list)
    pid_to_y = {}
    for pid, p, y in zip(patient_ids, probs, labels):
        pid_to_probs[pid].append(float(p))
        pid_to_y[pid] = int(y)

    pids = sorted(pid_to_probs.keys())
    p_patient = np.array([np.mean(pid_to_probs[pid]) for pid in pids], dtype=np.float32)
    y_patient = np.array([pid_to_y[pid] for pid in pids], dtype=np.int64)
    return pids, p_patient, y_patient


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_score = np.asarray(y_score).astype(np.float64).ravel()
    if y_true.size == 0:
        return float("nan")
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def best_threshold_by_youden(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_score = np.asarray(y_score).astype(np.float64).ravel()
    if y_true.size == 0:
        return 0.5, 0.0, 0.0, np.zeros((2, 2), dtype=np.int64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        preds = (y_score >= 0.5).astype(np.int64)
        acc, bal, cm, _ = confusion_and_balanced_acc(y_true, preds)
        return 0.5, bal, acc, cm

    thresholds = np.unique(y_score)
    best_t = 0.5
    best_j = -1.0
    best_bal = 0.0
    best_acc = 0.0
    best_cm = np.zeros((2, 2), dtype=np.int64)
    for t in thresholds:
        preds = (y_score >= t).astype(np.int64)
        acc, bal, cm, recalls = confusion_and_balanced_acc(y_true, preds)
        j = float(recalls[1] - (1.0 - recalls[0]))
        if j > best_j:
            best_j = j
            best_t = float(t)
            best_bal = float(bal)
            best_acc = float(acc)
            best_cm = cm
    return best_t, best_bal, best_acc, best_cm
