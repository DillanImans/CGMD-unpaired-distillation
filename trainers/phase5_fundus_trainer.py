from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils.metrics import best_threshold_by_youden


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cos = F.cosine_similarity(pred, target, dim=1)
    return 1.0 - cos


def compute_distill_loss(pred: torch.Tensor, target: torch.Tensor, metric: str, mse_weight: float) -> torch.Tensor:
    if metric == "cosine":
        return cosine_loss(pred, target)
    if metric == "mse":
        return (pred - target).pow(2).mean(dim=1)
    if metric == "cosine+mse":
        return cosine_loss(pred, target) + mse_weight * (pred - target).pow(2).mean(dim=1)
    raise ValueError(f"Unknown distill metric: {metric}")


def _auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(y_score)) + 1
    rank_sum = ranks[y_true == 1].sum()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _balanced_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return float((tpr + tnr) / 2.0)


def _auprc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


def patient_level_metrics(
    patient_ids: List[str],
    probs: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    pid_to_probs: Dict[str, List[float]] = {}
    pid_to_label: Dict[str, int] = {}
    for pid, p, y in zip(patient_ids, probs, labels):
        pid_to_probs.setdefault(pid, []).append(float(p))
        pid_to_label[pid] = int(y)

    uniq_pids = list(pid_to_probs.keys())
    y_true = np.array([pid_to_label[pid] for pid in uniq_pids])
    y_score = np.array([np.mean(pid_to_probs[pid]) for pid in uniq_pids])

    youden_t, _, _, cm = best_threshold_by_youden(y_true, y_score)
    y_pred = (y_score >= youden_t).astype(int)
    acc = float((y_pred == y_true).mean())
    bal = _balanced_acc(y_true, y_pred)
    auc = _auc_score(y_true, y_score)
    auprc = _auprc_score(y_true, y_score)

    tp = int(cm[1, 1])
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
    youden_j = sensitivity + specificity - 1.0

    return {
        "auc": auc,
        "auprc": auprc,
        "accuracy": acc,
        "balanced_accuracy": bal,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "youden_threshold": float(youden_t),
        "youden_j": float(youden_j),
    }


def scan_level_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    y_true = labels.astype(int)
    y_score = probs
    youden_t, _, _, cm = best_threshold_by_youden(y_true, y_score)
    y_pred = (y_score >= youden_t).astype(int)

    acc = float((y_pred == y_true).mean())
    bal = _balanced_acc(y_true, y_pred)
    auc = _auc_score(y_true, y_score)
    auprc = _auprc_score(y_true, y_score)

    tp = int(cm[1, 1])
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
    youden_j = sensitivity + specificity - 1.0

    return {
        "auc": auc,
        "auprc": auprc,
        "accuracy": acc,
        "balanced_accuracy": bal,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "youden_threshold": float(youden_t),
        "youden_j": float(youden_j),
    }


def subgroup_metrics(
    patient_ids: List[str],
    probs: np.ndarray,
    labels: np.ndarray,
    groups: List[str],
) -> Dict[str, Dict[str, float]]:
    pid_group = {}
    for pid, g in zip(patient_ids, groups):
        pid_group[pid] = g

    result: Dict[str, Dict[str, float]] = {}
    for g in sorted(set(groups)):
        mask = [pid_group[pid] == g for pid in patient_ids]
        if not any(mask):
            continue
        pid_sub = [pid for pid, m in zip(patient_ids, mask) if m]
        probs_sub = probs[mask]
        labels_sub = labels[mask]
        result[g] = patient_level_metrics(pid_sub, probs_sub, labels_sub)
    return result


@dataclass
class TrainResult:
    best_epoch: int
    best_val: float
    history: list


def train_fundus(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    train_cfg: dict,
    losses_cfg: dict,
    device: torch.device,
    ckpt_path: Path,
) -> TrainResult:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    pos_weight = train_cfg.get("pos_weight")
    if pos_weight is not None:
        pos_weight = torch.tensor([float(pos_weight)], device=device)
    loss_cls_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    cls_w = float(losses_cfg.get("cls_weight", 1.0))
    distill_w = float(losses_cfg.get("distill_weight", 1.0))
    anchor_w = float(losses_cfg.get("anchor_weight", 1.0))
    distill_metric = losses_cfg.get("distill_metric", "cosine")
    distill_mse_w = float(losses_cfg.get("distill_mse_weight", 1.0))

    best_val = -1e9
    best_epoch = -1
    bad = 0
    patience = int(train_cfg.get("patience", 10))
    history = []

    for epoch in range(1, int(train_cfg.get("epochs", 50)) + 1):
        model.train()
        running = {"loss": 0.0, "loss_cls": 0.0, "loss_anchor": 0.0, "loss_distill": 0.0}
        n_batches = 0

        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            clinical = batch.get("clinical")
            if clinical is not None:
                clinical = clinical.to(device)

            logits, emb = model(image, clinical)
            loss_cls = loss_cls_fn(logits, label)
            loss = cls_w * loss_cls

            loss_anchor = torch.tensor(0.0, device=device)
            loss_distill = torch.tensor(0.0, device=device)

            if "anchor" in batch and "is_anchor" in batch:
                anchor = batch["anchor"].to(device)
                is_anchor = batch["is_anchor"].to(device)
                if is_anchor.sum() > 0:
                    anchor_loss = compute_distill_loss(emb, anchor, distill_metric, distill_mse_w)
                    loss_anchor = (anchor_loss * is_anchor.squeeze(1)).sum() / is_anchor.sum()
                    loss = loss + anchor_w * loss_anchor

            if "prior" in batch and "confidence" in batch:
                prior = batch["prior"].to(device)
                conf = batch["confidence"].to(device)
                if "is_anchor" in batch:
                    is_anchor = batch["is_anchor"].to(device)
                    conf = conf * (1.0 - is_anchor)
                if conf.sum() > 0:
                    distill_loss = compute_distill_loss(emb, prior, distill_metric, distill_mse_w)
                    loss_distill = (distill_loss * conf.squeeze(1)).sum() / conf.sum()
                    loss = loss + distill_w * loss_distill

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running["loss"] += float(loss.item())
            running["loss_cls"] += float(loss_cls.item())
            running["loss_anchor"] += float(loss_anchor.item())
            running["loss_distill"] += float(loss_distill.item())
            n_batches += 1

        for k in running:
            running[k] /= max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_metrics = evaluate_loader(model, val_loader, device)

        record = {
            "epoch": epoch,
            "train": running,
            "val": val_metrics,
        }
        history.append(record)

        val_score = val_metrics.get(train_cfg.get("val_metric", "auc"), float("nan"))
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            bad = 0
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_val": best_val,
            }
            torch.save(ckpt, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                break

    return TrainResult(best_epoch=best_epoch, best_val=best_val, history=history)


def evaluate_loader(model: torch.nn.Module, loader, device: torch.device) -> Dict[str, float]:
    probs = []
    labels = []
    patient_ids = []

    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        clinical = batch.get("clinical")
        if clinical is not None:
            clinical = clinical.to(device)

        logits, _ = model(image, clinical)
        prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        probs.append(prob)
        labels.append(label.squeeze(1).cpu().numpy())
        patient_ids.extend(batch["patient_id"])

    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0).astype(int)

    metrics = patient_level_metrics(patient_ids, probs, labels)
    scan_metrics = scan_level_metrics(probs, labels)
    metrics.update({
        "scan_auc": scan_metrics["auc"],
        "scan_auprc": scan_metrics["auprc"],
        "scan_accuracy": scan_metrics["accuracy"],
        "scan_balanced_accuracy": scan_metrics["balanced_accuracy"],
        "scan_sensitivity": scan_metrics["sensitivity"],
        "scan_specificity": scan_metrics["specificity"],
        "scan_f1": scan_metrics["f1"],
        "scan_confusion": scan_metrics["confusion"],
    })
    return metrics


def save_history(path: Path, history: list) -> None:
    path.write_text(json.dumps(history, indent=2))
