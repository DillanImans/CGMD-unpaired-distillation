from __future__ import annotations


import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.graphsage_imputer import GraphSAGEImputer
from trainers.phase3_imputer_trainer import (
    degree_stats,
    evaluate_all,
    train_imputer,
    save_history,
)
from utils.phase3_imputer_data import (
    align_brain_embeddings,
    load_brain_embeddings,
    load_clinical_features,
    load_graph_npz,
    load_splits,
    save_json,
    seed_all,
)
from utils.run_config import (
    get_run_root,
    is_unified_config,
    load_yaml,
    phase_dir,
    resolve_path,
    write_run_config,
)


def inspect_npz(path: str) -> dict:
    npz = np.load(path, allow_pickle=True)
    info = {"keys": []}
    for k in npz.files:
        arr = npz[k]
        info["keys"].append({
            "name": k,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        })
    return info


def inspect_brain_embeddings(path: str) -> dict:
    info: dict = {"path": str(path)}
    obj = None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError:
        from utils.phase3_imputer_data import _ensure_monai_metatensor_stub
        _ensure_monai_metatensor_stub()
        obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        info["keys"] = list(obj.keys())
        for k, v in obj.items():
            if isinstance(v, dict):
                info[k] = {"keys": list(v.keys())}
                for kk, vv in v.items():
                    if hasattr(vv, "shape"):
                        info[k][kk] = {"shape": list(vv.shape), "dtype": str(vv.dtype)}
            elif hasattr(v, "shape"):
                info[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
    else:
        info["type"] = str(type(obj))
    return info


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def _load_edge_distances_flat(graph_path: str | Path, edge_index: torch.Tensor) -> np.ndarray | None:
    npz = np.load(graph_path, allow_pickle=True)
    if "edge_distance" in npz.files:
        edge_distance = npz["edge_distance"]
        if edge_distance.ndim == 1 and edge_distance.shape[0] == int(edge_index.shape[1]):
            return edge_distance.astype(np.float32)

    if "distances" not in npz.files:
        return None
    dists = npz["distances"]
    if dists.ndim != 2 or dists.size == 0:
        return None

    flat = dists.astype(np.float32).reshape(-1)
    if flat.shape[0] != int(edge_index.shape[1]):
        return None

    src = edge_index[0].cpu().numpy().astype(np.int64)
    expected_src = np.repeat(np.arange(dists.shape[0], dtype=np.int64), dists.shape[1])
    if src.shape[0] != expected_src.shape[0] or not np.array_equal(src, expected_src):
        return None
    return flat


def _extract_anchor_subgraph(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    anchor_mask: np.ndarray,
    edge_distance: np.ndarray | None = None,
) -> dict:
    src = edge_index[0].cpu().numpy().astype(np.int64)
    dst = edge_index[1].cpu().numpy().astype(np.int64)
    n_total = int(anchor_mask.shape[0])
    anchor_idx = np.where(anchor_mask)[0].astype(np.int64)

    glob_to_loc = np.full(n_total, -1, dtype=np.int64)
    glob_to_loc[anchor_idx] = np.arange(anchor_idx.shape[0], dtype=np.int64)

    keep = np.logical_and(anchor_mask[src], anchor_mask[dst])
    src_loc = glob_to_loc[src[keep]]
    dst_loc = glob_to_loc[dst[keep]]

    if edge_weight is None:
        edge_weight_np = np.ones(src.shape[0], dtype=np.float32)
    else:
        edge_weight_np = edge_weight.detach().cpu().numpy().astype(np.float32)
    edge_weight_loc = edge_weight_np[keep]

    edge_distance_loc = None
    if edge_distance is not None and edge_distance.shape[0] == src.shape[0]:
        edge_distance_loc = edge_distance[keep].astype(np.float32)

    return {
        "anchor_idx": anchor_idx,
        "edge_index": np.stack([src_loc, dst_loc], axis=0).astype(np.int64),
        "edge_weight": edge_weight_loc.astype(np.float32),
        "edge_distance": edge_distance_loc,
        "n_anchor": int(anchor_idx.shape[0]),
        "n_edges": int(src_loc.shape[0]),
    }


def _coalesce_undirected_weighted_edges(
    src: np.ndarray,
    dst: np.ndarray,
    w: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.logical_and.reduce([src >= 0, src < n_nodes, dst >= 0, dst < n_nodes, src != dst])
    src = src[valid].astype(np.int64)
    dst = dst[valid].astype(np.int64)
    w = w[valid].astype(np.float32)
    if src.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    u = np.minimum(src, dst)
    v = np.maximum(src, dst)
    keys = u.astype(np.int64) * np.int64(n_nodes) + v.astype(np.int64)
    order = np.argsort(keys, kind="mergesort")
    u = u[order]
    v = v[order]
    keys = keys[order]
    w = w[order]

    _, starts = np.unique(keys, return_index=True)
    ends = np.append(starts[1:], len(keys))
    w_pair = np.array([float(w[s:e].mean()) for s, e in zip(starts, ends)], dtype=np.float32)
    u_pair = u[starts].astype(np.int64)
    v_pair = v[starts].astype(np.int64)

    src_u = np.concatenate([u_pair, v_pair]).astype(np.int64)
    dst_u = np.concatenate([v_pair, u_pair]).astype(np.int64)
    w_u = np.concatenate([w_pair, w_pair]).astype(np.float32)
    return src_u, dst_u, w_u


def smooth_brain_embeddings(
    z: np.ndarray,
    edge_index: np.ndarray,
    *,
    alpha: float,
    steps: int,
    neighbor_dir: str,
    weight_mode: str,
    rbf_sigma: float,
    edge_weight: np.ndarray | None = None,
    edge_distance: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    if z.ndim != 2:
        raise ValueError("z must be shape (N, D)")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("teacher_smoothing.alpha must be in [0, 1]")
    if steps < 0:
        raise ValueError("teacher_smoothing.steps must be >= 0")
    if neighbor_dir not in {"out", "in", "undirected"}:
        raise ValueError("teacher_smoothing.neighbor_dir must be 'out', 'in', or 'undirected'")
    if weight_mode not in {"uniform", "rbf"}:
        raise ValueError("teacher_smoothing.weight_mode must be 'uniform' or 'rbf'")
    if rbf_sigma <= 0:
        raise ValueError("teacher_smoothing.rbf_sigma must be > 0")

    n, _ = z.shape
    if steps == 0 or edge_index.size == 0:
        return z.copy(), {
            "steps": int(steps),
            "alpha": float(alpha),
            "neighbor_dir": neighbor_dir,
            "weight_mode": weight_mode,
            "rbf_sigma": float(rbf_sigma),
            "n_nodes": int(n),
            "n_edges": int(edge_index.shape[1]) if edge_index.ndim == 2 else 0,
            "nodes_without_neighbors": int(n),
            "used_edge_distance": False,
            "used_edge_weight_as_affinity": False,
        }

    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    if neighbor_dir == "in":
        src, dst = dst, src

    if weight_mode == "uniform":
        raw_w = np.ones(src.shape[0], dtype=np.float32)
        used_edge_distance = False
        used_edge_weight_as_affinity = False
    else:
        if edge_distance is not None and edge_distance.shape[0] == src.shape[0]:
            raw_w = np.exp(-(edge_distance.astype(np.float32) ** 2) / (rbf_sigma ** 2)).astype(np.float32)
            used_edge_distance = True
            used_edge_weight_as_affinity = False
        elif edge_weight is not None and edge_weight.shape[0] == src.shape[0]:
            raw_w = np.clip(edge_weight.astype(np.float32), a_min=0.0, a_max=None)
            used_edge_distance = False
            used_edge_weight_as_affinity = True
        else:
            raw_w = np.ones(src.shape[0], dtype=np.float32)
            used_edge_distance = False
            used_edge_weight_as_affinity = False

    symmetrized_undirected = False
    if neighbor_dir == "undirected":
        src, dst, raw_w = _coalesce_undirected_weighted_edges(src, dst, raw_w, n)
        symmetrized_undirected = True

    row_sum = np.bincount(src, weights=raw_w.astype(np.float64), minlength=n).astype(np.float32)
    denom = np.maximum(row_sum[src], 1e-12)
    p = raw_w / denom
    has_neighbors = row_sum > 0

    z0 = z.astype(np.float32)
    z_prev = z0.copy()
    for _ in range(steps):
        agg = np.zeros_like(z0)
        np.add.at(agg, src, p[:, None] * z_prev[dst])
        z_next = alpha * z0 + (1.0 - alpha) * agg
        z_next[~has_neighbors] = z0[~has_neighbors]
        z_prev = z_next

    meta = {
        "steps": int(steps),
        "alpha": float(alpha),
        "neighbor_dir": neighbor_dir,
        "weight_mode": weight_mode,
        "rbf_sigma": float(rbf_sigma),
        "n_nodes": int(n),
        "n_edges": int(src.shape[0]),
        "nodes_without_neighbors": int((~has_neighbors).sum()),
        "used_edge_distance": bool(used_edge_distance),
        "used_edge_weight_as_affinity": bool(used_edge_weight_as_affinity),
        "symmetrized_undirected": bool(symmetrized_undirected),
    }
    return z_prev, meta


def apply_teacher_smoothing(
    emb: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    anchor_mask: np.ndarray,
    graph_path: str | Path,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    if emb.ndim != 2:
        raise ValueError("emb must be shape (N, D)")

    enabled = bool(cfg.get("enabled", False))
    meta = {
        "enabled": enabled,
        "n_total_nodes": int(emb.shape[0]),
        "n_anchor_nodes": int(anchor_mask.sum()),
    }
    if not enabled:
        meta["reason"] = "disabled"
        return emb.copy(), meta

    sub = _extract_anchor_subgraph(
        edge_index=edge_index,
        edge_weight=edge_weight,
        anchor_mask=anchor_mask,
        edge_distance=_load_edge_distances_flat(graph_path, edge_index),
    )
    if sub["n_anchor"] == 0:
        meta["reason"] = "no_anchor_nodes"
        return emb.copy(), meta

    alpha = float(cfg.get("alpha", 0.9))
    steps = int(cfg.get("steps", 1))
    neighbor_dir = str(cfg.get("neighbor_dir", "out")).lower()
    weight_mode = str(cfg.get("weight_mode", "rbf")).lower()
    rbf_sigma = float(cfg.get("rbf_sigma", 1.0))

    z_anchor = emb[sub["anchor_idx"]].astype(np.float32)
    z_smooth, smooth_meta = smooth_brain_embeddings(
        z_anchor,
        sub["edge_index"],
        alpha=alpha,
        steps=steps,
        neighbor_dir=neighbor_dir,
        weight_mode=weight_mode,
        rbf_sigma=rbf_sigma,
        edge_weight=sub["edge_weight"],
        edge_distance=sub["edge_distance"],
    )

    out = emb.copy()
    out[sub["anchor_idx"]] = z_smooth

    meta.update({
        "reason": "ok",
        "n_anchor_edges": int(sub["n_edges"]),
        "config": {
            "alpha": alpha,
            "steps": steps,
            "neighbor_dir": neighbor_dir,
            "weight_mode": weight_mode,
            "rbf_sigma": rbf_sigma,
        },
        "smoothing": smooth_meta,
    })
    return out, meta


def _compute_global_means(
    emb: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    anchor_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.logical_and(train_mask, anchor_mask)
    valid = np.logical_and(valid, np.isfinite(labels))
    if valid.any():
        global_all = emb[valid].mean(axis=0)
    elif anchor_mask.any():
        global_all = emb[anchor_mask].mean(axis=0)
    else:
        global_all = np.zeros((emb.shape[1],), dtype=np.float32)

    pos_mask = np.logical_and(valid, labels == 1)
    neg_mask = np.logical_and(valid, labels == 0)
    global_pos = emb[pos_mask].mean(axis=0) if pos_mask.any() else global_all
    global_neg = emb[neg_mask].mean(axis=0) if neg_mask.any() else global_all
    return global_pos, global_neg, global_all


def compute_knn_mean_priors(
    emb: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    anchor_mask: np.ndarray,
    normalize_output: bool,
) -> np.ndarray:
    n, d = emb.shape
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    if edge_weight is None:
        w = np.ones_like(src, dtype=np.float32)
    else:
        w = edge_weight.detach().cpu().numpy().astype(np.float32)

    order = np.argsort(src)
    src = src[order]
    dst = dst[order]
    w = w[order]

    counts = np.bincount(src, minlength=n)
    offsets = np.concatenate(([0], np.cumsum(counts)))

    if anchor_mask.any():
        global_mean = emb[anchor_mask].mean(axis=0)
    else:
        global_mean = np.zeros((d,), dtype=np.float32)

    out = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        start = offsets[i]
        end = offsets[i + 1]
        nbrs = dst[start:end]
        w_i = w[start:end]

        if nbrs.size == 0:
            out[i] = global_mean
            continue

        mask = anchor_mask[nbrs]
        if not mask.any():
            out[i] = global_mean
            continue

        nbrs = nbrs[mask]
        w_i = w_i[mask]
        w_sum = float(w_i.sum())
        if w_sum > 0:
            w_i = w_i / w_sum
            out[i] = (emb[nbrs] * w_i[:, None]).sum(axis=0)
        else:
            out[i] = emb[nbrs].mean(axis=0)

    if normalize_output:
        out = _normalize_rows(out)
    return out


def compute_local_class_prototypes(
    emb: np.ndarray,
    labels: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    anchor_mask: np.ndarray,
    train_mask: np.ndarray,
    min_count: int,
    smooth_alpha: float,
    use_edge_weight: bool,
    normalize_output: bool,
) -> dict:
    n, d = emb.shape
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    if edge_weight is None:
        w = np.ones_like(src, dtype=np.float32)
    else:
        w = edge_weight.detach().cpu().numpy().astype(np.float32)

    order = np.argsort(src)
    src = src[order]
    dst = dst[order]
    w = w[order]

    counts = np.bincount(src, minlength=n)
    offsets = np.concatenate(([0], np.cumsum(counts)))

    valid_label = np.isfinite(labels)
    label_int = np.full(n, -1, dtype=np.int64)
    label_int[valid_label] = labels[valid_label].astype(int)

    global_pos, global_neg, _ = _compute_global_means(emb, labels, train_mask, anchor_mask)

    pos_emb = np.zeros((n, d), dtype=np.float32)
    neg_emb = np.zeros((n, d), dtype=np.float32)
    pos_count = np.zeros(n, dtype=np.int64)
    neg_count = np.zeros(n, dtype=np.int64)
    pos_weight = np.zeros(n, dtype=np.float32)
    neg_weight = np.zeros(n, dtype=np.float32)
    pos_fallback = np.zeros(n, dtype=np.int8)
    neg_fallback = np.zeros(n, dtype=np.int8)

    for i in range(n):
        start = offsets[i]
        end = offsets[i + 1]
        nbrs = dst[start:end]
        w_i = w[start:end]

        if nbrs.size == 0:
            pos_emb[i] = global_pos
            neg_emb[i] = global_neg
            pos_fallback[i] = 1
            neg_fallback[i] = 1
            continue

        mask = np.logical_and(anchor_mask[nbrs], valid_label[nbrs])
        if not mask.any():
            pos_emb[i] = global_pos
            neg_emb[i] = global_neg
            pos_fallback[i] = 1
            neg_fallback[i] = 1
            continue

        nbrs = nbrs[mask]
        w_i = w_i[mask]
        lbl = label_int[nbrs]

        pos_mask = lbl == 1
        neg_mask = lbl == 0
        pos_count[i] = int(pos_mask.sum())
        neg_count[i] = int(neg_mask.sum())

        if pos_count[i] >= min_count:
            if use_edge_weight:
                w_pos = w_i[pos_mask]
                w_sum = float(w_pos.sum())
                pos_weight[i] = w_sum
                if w_sum > 0:
                    w_pos = w_pos / w_sum
                    mu_pos = (emb[nbrs[pos_mask]] * w_pos[:, None]).sum(axis=0)
                else:
                    mu_pos = global_pos
                    pos_fallback[i] = 1
            else:
                pos_weight[i] = float(pos_count[i])
                mu_pos = emb[nbrs[pos_mask]].mean(axis=0)
            if smooth_alpha < 1.0:
                mu_pos = smooth_alpha * mu_pos + (1.0 - smooth_alpha) * global_pos
            pos_emb[i] = mu_pos
        else:
            pos_emb[i] = global_pos
            pos_fallback[i] = 1

        if neg_count[i] >= min_count:
            if use_edge_weight:
                w_neg = w_i[neg_mask]
                w_sum = float(w_neg.sum())
                neg_weight[i] = w_sum
                if w_sum > 0:
                    w_neg = w_neg / w_sum
                    mu_neg = (emb[nbrs[neg_mask]] * w_neg[:, None]).sum(axis=0)
                else:
                    mu_neg = global_neg
                    neg_fallback[i] = 1
            else:
                neg_weight[i] = float(neg_count[i])
                mu_neg = emb[nbrs[neg_mask]].mean(axis=0)
            if smooth_alpha < 1.0:
                mu_neg = smooth_alpha * mu_neg + (1.0 - smooth_alpha) * global_neg
            neg_emb[i] = mu_neg
        else:
            neg_emb[i] = global_neg
            neg_fallback[i] = 1

    if normalize_output:
        pos_emb = _normalize_rows(pos_emb)
        neg_emb = _normalize_rows(neg_emb)

    return {
        "pos": pos_emb,
        "neg": neg_emb,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_weight": pos_weight,
        "neg_weight": neg_weight,
        "pos_fallback": pos_fallback,
        "neg_fallback": neg_fallback,
    }


def main(config_path: str) -> None:
    cfg = load_yaml(config_path)
    unified = is_unified_config(cfg)
    if unified:
        run_cfg = cfg["run"]
        phase_cfg = cfg["phase3"]
        run_root = get_run_root(cfg)
        run_dir = phase_dir(run_root, "phase3")
    else:
        run_cfg = cfg["run"]
        phase_cfg = cfg
        run_root = None
        ts = time.strftime("%Y%m%d_%H%M%S")
        base_name = run_cfg.get("name")
        run_name = f"{base_name}_{ts}" if base_name else f"run_{ts}"
        run_dir = Path(run_cfg.get("output_root", "logs/phase3_imputer")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if unified and run_root is not None:
        write_run_config(run_root, config_path)
    (run_dir / "config.yaml").write_text(Path(config_path).read_text())

    data_cfg = dict(phase_cfg["data"])
    model_cfg = phase_cfg["model"]
    train_cfg = phase_cfg["train"]
    eval_cfg = phase_cfg["eval"]

    if unified and run_root is not None:
        data_cfg["brain_teacher_embeddings"] = resolve_path(
            data_cfg.get("brain_teacher_embeddings"),
            phase_dir(run_root, "embeddings") / "brain_teacher_embeddings.pt",
        )
        data_cfg["graph_npz"] = resolve_path(
            data_cfg.get("graph_npz"),
            phase_dir(run_root, "graphs") / "graph_trainval_inductive.npz",
        )

    seed_all(int(run_cfg.get("seed", 0)))

    graph_path = data_cfg["graph_npz"]
    graph_info = inspect_npz(graph_path)
    save_json(run_dir / "graph_inspection.json", graph_info)

    emb_info = inspect_brain_embeddings(data_cfg["brain_teacher_embeddings"])
    save_json(run_dir / "brain_embeddings_inspection.json", emb_info)

    graph = load_graph_npz(graph_path)
    split_masks = load_splits(data_cfg["splits_csv"], graph.patient_ids)

    clinical = load_clinical_features(
        path=data_cfg["clinical_csv"],
        patient_ids=graph.patient_ids,
        train_mask_all=split_masks.train_mask_all,
        clinical_cols=data_cfg.get("clinical_cols"),
        exclude_cols=data_cfg.get("clinical_exclude_cols"),
        binary_cols=data_cfg.get("binary_cols"),
        cont_cols=data_cfg.get("cont_cols"),
        label_col=data_cfg.get("label_col", "htn"),
    )

    emb_pids, emb = load_brain_embeddings(data_cfg["brain_teacher_embeddings"])
    brain = align_brain_embeddings(graph.patient_ids, emb_pids, emb, split_masks.has_brain)
    pid_to_row = {str(pid).strip(): i for i, pid in enumerate(emb_pids)}
    has_embedding = np.array([pid in pid_to_row for pid in graph.patient_ids], dtype=bool)
    anchor_mask = np.logical_and(has_embedding, split_masks.has_brain)

    teacher_smoothing_cfg = dict(phase_cfg.get("teacher_smoothing", {}))
    teacher_emb_np = brain.embeddings.cpu().numpy().astype(np.float32)
    teacher_emb_smooth_np, smooth_meta = apply_teacher_smoothing(
        emb=teacher_emb_np,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        anchor_mask=anchor_mask,
        graph_path=graph_path,
        cfg=teacher_smoothing_cfg,
    )
    save_json(run_dir / "teacher_smoothing_summary.json", smooth_meta)
    if smooth_meta.get("enabled", False):
        print(
            "[SMOOTH] enabled "
            f"anchor_nodes={smooth_meta.get('n_anchor_nodes', 0)} "
            f"anchor_edges={smooth_meta.get('n_anchor_edges', 0)} "
            f"mode={smooth_meta.get('config', {}).get('weight_mode', 'unknown')} "
            f"steps={smooth_meta.get('config', {}).get('steps', 0)} "
            f"alpha={smooth_meta.get('config', {}).get('alpha', 0.0):.3f}"
        )
        sm = smooth_meta.get("smoothing", {})
        if sm.get("used_edge_weight_as_affinity", False):
            print("[SMOOTH][WARN] distances unavailable; used edge_weight as affinity")
    else:
        print(f"[SMOOTH] disabled ({smooth_meta.get('reason', 'disabled')})")
    teacher_target = torch.tensor(teacher_emb_smooth_np, dtype=brain.embeddings.dtype)

    deg = degree_stats(graph.edge_index, len(graph.patient_ids))
    print("[GRAPH] degree stats:", deg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphSAGEImputer(
        in_dim=clinical.x.shape[1],
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        out_dim=int(emb.shape[1]),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        use_mlp=bool(model_cfg.get("use_mlp", False)),
        mlp_dim=model_cfg.get("mlp_dim"),
        normalize_output=bool(model_cfg.get("normalize_output", True)),
    )

    ckpt_path = run_dir / "best.pt"
    train_res = train_imputer(
        model=model,
        x=clinical.x,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        target=teacher_target,
        train_mask=split_masks.train_brain_mask,
        val_mask=split_masks.val_brain_mask,
        epochs=int(train_cfg.get("epochs", 200)),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        loss_type=train_cfg.get("loss_type", "cosine"),
        mse_weight=float(train_cfg.get("mse_weight", 1.0)),
        patience=int(train_cfg.get("patience", 20)),
        device=device,
        probe_labels=clinical.htn,
        select_metric=str(train_cfg.get("select_metric", "probe_auc")),
        ckpt_path=ckpt_path,
    )

    save_history(run_dir / "train_history.json", train_res.history)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        pred = model(clinical.x.to(device), graph.edge_index.to(device), graph.edge_weight.to(device))

    pred_cpu = pred.cpu()
    eval_summary = evaluate_all(
        pred=pred_cpu,
        true=teacher_target,
        train_mask=split_masks.train_brain_mask,
        val_mask=split_masks.val_brain_mask,
        htn=clinical.htn,
        k_retrieval=int(eval_cfg.get("retrieval_k", 10)),
        seed=int(run_cfg.get("seed", 0)),
    )

    save_json(run_dir / "eval_summary.json", eval_summary)

    proto_cfg = dict(phase_cfg.get("local_prototypes", {}))
    min_count = max(1, int(proto_cfg.get("min_count", 1)))
    smooth_alpha = float(proto_cfg.get("smooth_alpha", 1.0))
    if not 0.0 <= smooth_alpha <= 1.0:
        raise ValueError("local_prototypes.smooth_alpha must be in [0,1]")
    use_edge_weight = bool(proto_cfg.get("use_edge_weight", True))
    normalize_proto = bool(proto_cfg.get("normalize_output", True))

    emb_np = teacher_emb_smooth_np
    local_proto = compute_local_class_prototypes(
        emb=emb_np,
        labels=clinical.htn,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        anchor_mask=anchor_mask,
        train_mask=split_masks.train_brain_mask,
        min_count=min_count,
        smooth_alpha=smooth_alpha,
        use_edge_weight=use_edge_weight,
        normalize_output=normalize_proto,
    )

    np.savez_compressed(
        run_dir / "imputed_brain_priors_local_pos.npz",
        patient_ids=np.array(graph.patient_ids, dtype=str),
        embeddings=local_proto["pos"].astype(np.float32),
    )
    np.savez_compressed(
        run_dir / "imputed_brain_priors_local_neg.npz",
        patient_ids=np.array(graph.patient_ids, dtype=str),
        embeddings=local_proto["neg"].astype(np.float32),
    )

    print(f"[DONE] run_dir={run_dir}")
    print(
        f"[DONE] best_epoch={train_res.best_epoch} "
        f"best_{train_res.best_metric_name}={train_res.best_val_metric:.4f}"
    )
    print("[DONE] saved priors: imputed_brain_priors_local_pos.npz, imputed_brain_priors_local_neg.npz")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
