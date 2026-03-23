from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cos = F.cosine_similarity(pred, target, dim=1)
    return 1.0 - cos


def compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str, mse_weight: float) -> torch.Tensor:
    if loss_type == "cosine":
        return cosine_loss(pred, target).mean()
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "cosine+mse":
        return cosine_loss(pred, target).mean() + mse_weight * F.mse_loss(pred, target)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def eval_cos_mse(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    cos = F.cosine_similarity(pred, target, dim=1).mean().item()
    mse = F.mse_loss(pred, target).item()
    return {"cosine": cos, "mse": mse}


def degree_stats(edge_index: torch.Tensor, n: int) -> Dict[str, Dict[str, float]]:
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    out_deg = np.bincount(src, minlength=n)
    in_deg = np.bincount(dst, minlength=n)

    def summ(x):
        return {
            "min": int(x.min()),
            "p25": float(np.percentile(x, 25)),
            "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75)),
            "max": int(x.max()),
            "mean": float(x.mean()),
        }

    return {"out_deg": summ(out_deg), "in_deg": summ(in_deg)}


def neighbor_retrieval_metrics(true_emb: np.ndarray, pred_emb: np.ndarray, k: int) -> Dict[str, float]:
    def topk_neighbors(emb):
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        sim = emb @ emb.T
        np.fill_diagonal(sim, -np.inf)
        idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
        return idx

    true_idx = topk_neighbors(true_emb)
    pred_idx = topk_neighbors(pred_emb)

    recalls = []
    precisions = []
    jaccs = []
    for i in range(true_idx.shape[0]):
        t = set(true_idx[i].tolist())
        p = set(pred_idx[i].tolist())
        inter = len(t & p)
        union = len(t | p)
        recalls.append(inter / max(len(t), 1))
        precisions.append(inter / max(k, 1))
        jaccs.append(inter / max(union, 1))

    return {
        "recall_at_k_mean": float(np.mean(recalls)),
        "recall_at_k_std": float(np.std(recalls)),
        "precision_at_k_mean": float(np.mean(precisions)),
        "precision_at_k_std": float(np.std(precisions)),
        "jaccard_mean": float(np.mean(jaccs)),
        "jaccard_std": float(np.std(jaccs)),
    }


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


def linear_probe_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, float]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

        clf = LogisticRegression(max_iter=200, class_weight="balanced")
        clf.fit(x_train, y_train)
        prob = clf.predict_proba(x_val)[:, 1]
        pred = (prob >= 0.5).astype(int)
        return {
            "auc": float(roc_auc_score(y_val, prob)),
            "accuracy": float(accuracy_score(y_val, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
        }
    except Exception:
        x_train_t = torch.tensor(x_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        x_val_t = torch.tensor(x_val, dtype=torch.float32)

        model = torch.nn.Linear(x_train.shape[1], 1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        for _ in range(200):
            opt.zero_grad()
            logits = model(x_train_t)
            loss = loss_fn(logits, y_train_t)
            loss.backward()
            opt.step()

        with torch.no_grad():
            logits = model(x_val_t).numpy().reshape(-1)
            prob = 1 / (1 + np.exp(-logits))
            pred = (prob >= 0.5).astype(int)

        return {
            "auc": _auc_score(y_val, prob),
            "accuracy": float((pred == y_val).mean()),
            "balanced_accuracy": _balanced_acc(y_val, pred),
        }


def evaluate_all(
    pred: torch.Tensor,
    true: torch.Tensor,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    htn: np.ndarray,
    k_retrieval: int,
    seed: int,
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}

    pred_train = pred[train_mask]
    true_train = true[train_mask]
    pred_val = pred[val_mask]
    true_val = true[val_mask]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(pred_val.shape[0])
    pred_val_shuffled = pred_val[perm]

    out["embedding_similarity"] = {
        "val": eval_cos_mse(pred_val, true_val),
    }

    mean_emb = true_train.mean(dim=0, keepdim=True)
    mean_pred = mean_emb.repeat(true_val.shape[0], 1)
    out["embedding_similarity"]["baseline_mean"] = eval_cos_mse(mean_pred, true_val)

    rand = rng.normal(size=true_val.shape)
    rand = rand / (np.linalg.norm(rand, axis=1, keepdims=True) + 1e-12)
    rand = torch.tensor(rand, dtype=true_val.dtype)
    out["embedding_similarity"]["baseline_random"] = eval_cos_mse(rand, true_val)

    out["embedding_similarity"]["baseline_shuffled"] = eval_cos_mse(pred_val_shuffled, true_val)

    out["retrieval"] = {
        "imputed": neighbor_retrieval_metrics(true_val.numpy(), pred_val.numpy(), k_retrieval)
    }
    rand_metrics = neighbor_retrieval_metrics(true_val.numpy(), rand.numpy(), k_retrieval)
    out["retrieval"]["baseline_random"] = rand_metrics

    shuffled_metrics = neighbor_retrieval_metrics(true_val.numpy(), pred_val_shuffled.numpy(), k_retrieval)
    out["retrieval"]["baseline_shuffled"] = shuffled_metrics

    htn_train = htn[train_mask]
    htn_val = htn[val_mask]
    train_ok = ~np.isnan(htn_train)
    val_ok = ~np.isnan(htn_val)
    if train_ok.all() and val_ok.all():
        htn_train = htn_train.astype(int)
        htn_val = htn_val.astype(int)
        out["linear_probe"] = {
            "true_brain": linear_probe_metrics(true_train.numpy(), htn_train, true_val.numpy(), htn_val),
            "imputed": linear_probe_metrics(pred_train.numpy(), htn_train, pred_val.numpy(), htn_val),
            "imputed_shuffled": linear_probe_metrics(pred_train.numpy(), htn_train, pred_val_shuffled.numpy(), htn_val),
        }
        if set(np.unique(htn_train)).issuperset({0, 1}):
            mean_neg = true_train[htn_train == 0].mean(dim=0, keepdim=True)
            mean_pos = true_train[htn_train == 1].mean(dim=0, keepdim=True)
            class_mean = torch.where(
                torch.tensor(htn_val.reshape(-1, 1), dtype=torch.bool),
                mean_pos,
                mean_neg,
            )
            out["embedding_similarity"]["baseline_class_mean"] = eval_cos_mse(class_mean, true_val)
    else:
        out["linear_probe"] = {"warning": "htn labels missing for some train/val rows"}

    return out


@dataclass
class TrainResult:
    best_epoch: int
    best_val_metric: float
    best_metric_name: str
    history: list


def train_imputer(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    target: torch.Tensor,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    loss_type: str,
    mse_weight: float,
    patience: int,
    device: torch.device,
    ckpt_path: Path,
    probe_labels: np.ndarray | None = None,
    select_metric: str = "probe_auc",
) -> TrainResult:
    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    target = target.to(device)

    train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)
    val_mask_t = torch.tensor(val_mask, dtype=torch.bool, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1e9
    best_epoch = -1
    best_metric_name = "val_cosine"
    bad = 0
    history = []
    val_has_samples = bool(val_mask_t.any().item())

    for epoch in range(1, epochs + 1):
        model.train()
        pred = model(x, edge_index, edge_weight)
        loss = compute_loss(pred[train_mask_t], target[train_mask_t], loss_type, mse_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index, edge_weight)
            train_metrics = eval_cos_mse(pred[train_mask_t], target[train_mask_t])
            if val_has_samples:
                val_metrics = eval_cos_mse(pred[val_mask_t], target[val_mask_t])
                val_cos = val_metrics["cosine"]
                pred_val = pred[val_mask_t]
            else:
                val_metrics = {"cosine": float("nan"), "mse": float("nan")}
                val_cos = train_metrics["cosine"]
                pred_val = pred[train_mask_t]

            val_probe_auc = float("nan")
            if probe_labels is not None:
                labels = np.asarray(probe_labels)
                if np.isfinite(labels).all():
                    y_train = labels[train_mask].astype(int)
                    y_val = labels[val_mask].astype(int) if val_has_samples else y_train
                    if set(np.unique(y_train)).issuperset({0, 1}) and set(np.unique(y_val)).issuperset({0, 1}):
                        pred_train_np = pred[train_mask_t].detach().cpu().numpy()
                        pred_val_np = pred_val.detach().cpu().numpy()
                        val_probe_auc = linear_probe_metrics(
                            pred_train_np,
                            y_train,
                            pred_val_np,
                            y_val,
                        )["auc"]

        if select_metric == "probe_auc" and np.isfinite(val_probe_auc):
            val_score = val_probe_auc
            metric_name = "probe_auc"
        else:
            val_score = val_cos
            metric_name = "val_cosine"

        record = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "train_cosine": train_metrics["cosine"],
            "train_mse": train_metrics["mse"],
            "val_cosine": val_metrics["cosine"],
            "val_mse": val_metrics["mse"],
            "val_probe_auc": val_probe_auc,
            "val_metric": val_score,
            "val_metric_name": metric_name,
        }
        history.append(record)

        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            best_metric_name = metric_name
            bad = 0
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_val_metric": best_val,
                "best_metric_name": best_metric_name,
            }
            torch.save(ckpt, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                break

    if best_epoch < 0:
        ckpt = {
            "epoch": epochs,
            "model_state": model.state_dict(),
            "best_val_metric": best_val,
            "best_metric_name": best_metric_name,
        }
        torch.save(ckpt, ckpt_path)

    return TrainResult(
        best_epoch=best_epoch,
        best_val_metric=best_val,
        best_metric_name=best_metric_name,
        history=history,
    )


def save_history(path: Path, history: list) -> None:
    path.write_text(json.dumps(history, indent=2))
