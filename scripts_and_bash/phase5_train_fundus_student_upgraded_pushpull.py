from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.fundus_student import FundusStudent
from trainers.phase5_fundus_trainer import TrainResult, compute_distill_loss, save_history, evaluate_loader
from utils.phase5_fundus_data import (
    FundusStudentDataset,
    build_clinical_map,
    build_samples,
    debug_fundus_paths,
    load_anchors,
    load_fundus_index,
    load_priors,
    load_splits,
)
from utils.phase3_imputer_data import seed_all, save_json
from utils.run_config import (
    get_run_root,
    is_unified_config,
    load_yaml,
    phase_dir,
    resolve_path,
    write_run_config,
)


def _compute_pos_weight(samples: list) -> float:
    ys = [s["label"] for s in samples]
    n_pos = sum(ys)
    n_neg = len(ys) - n_pos
    return float(n_neg / max(n_pos, 1))


def _batch_opposite_priors(
    patient_ids: list[str],
    labels: torch.Tensor,
    pos_prior_map: dict,
    neg_prior_map: dict,
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    zeros = np.zeros(embed_dim, dtype=np.float32)
    labels_np = labels.detach().cpu().numpy().reshape(-1).astype(int)
    neg_list = []
    for pid, y in zip(patient_ids, labels_np):
        if int(y) == 1:
            neg_list.append(neg_prior_map.get(pid, zeros))
        else:
            neg_list.append(pos_prior_map.get(pid, zeros))
    neg = np.stack(neg_list, axis=0).astype(np.float32)
    return torch.tensor(neg, device=device, dtype=dtype)


def _build_label_conditional_priors(
    label_map: dict,
    pos_prior_map: dict,
    pos_conf_map: dict,
    neg_prior_map: dict,
    neg_conf_map: dict,
    embed_dim: int,
    use_confidence: bool,
) -> tuple[dict, dict]:
    prior_map: dict = {}
    conf_map: dict = {}
    zeros = np.zeros(embed_dim, dtype=np.float32)

    for pid, label in label_map.items():
        if int(label) == 1:
            prior_map[pid] = pos_prior_map.get(pid, zeros)
            conf = pos_conf_map.get(pid) if pos_conf_map else None
        else:
            prior_map[pid] = neg_prior_map.get(pid, zeros)
            conf = neg_conf_map.get(pid) if neg_conf_map else None
        if conf is None:
            conf = 1.0
        conf_map[pid] = float(conf)

    if not use_confidence:
        conf_map = {pid: 1.0 for pid in prior_map.keys()}

    return prior_map, conf_map


def _load_rel_graph(npz_path: str | Path) -> dict:
    npz = np.load(npz_path, allow_pickle=True)
    if not {"patient_ids", "edge_index"}.issubset(npz.files):
        raise ValueError(f"Invalid relational graph npz: {npz_path}")
    pids = [str(pid).strip() for pid in npz["patient_ids"].astype(str).tolist()]
    edge_index = npz["edge_index"].astype(np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edge_index.shape}")
    if "edge_weight" in npz.files:
        edge_weight = npz["edge_weight"].astype(np.float32)
        if edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_weight length mismatch with edge_index")
    else:
        edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    return {"patient_ids": pids, "edge_index": edge_index, "edge_weight": edge_weight}


def _build_rel_teacher_vectors(
    patient_ids: list[str],
    mode: str,
    embed_dim: int,
    pos_prior_map: dict | None,
    neg_prior_map: dict | None,
    label_map: dict | None,
) -> np.ndarray:
    zeros = np.zeros(embed_dim, dtype=np.float32)
    out = np.zeros((len(patient_ids), embed_dim), dtype=np.float32)
    mode = str(mode).lower()

    for i, pid in enumerate(patient_ids):
        pos = pos_prior_map.get(pid, zeros) if pos_prior_map is not None else zeros
        neg = neg_prior_map.get(pid, zeros) if neg_prior_map is not None else zeros
        if mode == "pos_minus_neg":
            out[i] = pos - neg
        elif mode == "pos":
            out[i] = pos
        elif mode == "neg":
            out[i] = neg
        elif mode == "label_conditional":
            if label_map is None:
                out[i] = zeros
            else:
                y = int(label_map.get(pid, 0))
                out[i] = pos if y == 1 else neg
        else:
            raise ValueError(
                "relational_kd.teacher_prior_mode must be one of: "
                "pos_minus_neg, pos, neg, label_conditional"
            )
    return out


def _prepare_relational_kd_state(
    rel_cfg: dict,
    samples_train: list[dict],
    split_info,
    pos_prior_map: dict | None,
    neg_prior_map: dict | None,
    label_map: dict,
    embed_dim: int,
    device: torch.device,
) -> tuple[dict | None, dict]:
    enabled = bool(rel_cfg.get("enabled", False))
    summary = {
        "enabled": enabled,
        "lambda_rel": float(rel_cfg.get("lambda_rel", 0.0)),
        "edge_source": str(rel_cfg.get("edge_source", "fundus_train_graph")),
        "same_label_only": bool(rel_cfg.get("same_label_only", False)),
    }
    if not enabled or summary["lambda_rel"] <= 0.0:
        summary["reason"] = "disabled_or_zero_lambda"
        return None, summary

    similarity = str(rel_cfg.get("similarity", "cosine")).lower()
    if similarity != "cosine":
        raise ValueError("relational_kd.similarity currently supports only: cosine")

    graph_npz = rel_cfg.get("graph_npz")
    if not graph_npz:
        raise ValueError("relational_kd.graph_npz is required when relational_kd is enabled")

    graph = _load_rel_graph(graph_npz)
    graph_pids = graph["patient_ids"]
    pid_to_graph = {pid: i for i, pid in enumerate(graph_pids)}

    # Only keep edges over patients that are actually used by the train loader.
    train_pid_set = {str(s["patient_id"]).strip() for s in samples_train}
    keep_node = np.array([pid in train_pid_set for pid in graph_pids], dtype=bool)
    src = graph["edge_index"][0]
    dst = graph["edge_index"][1]
    edge_keep = np.logical_and(keep_node[src], keep_node[dst])

    # Optional: only keep relational pairs with matching HTN label.
    if bool(rel_cfg.get("same_label_only", False)):
        node_label = np.full(len(graph_pids), -1, dtype=np.int64)
        for i, pid in enumerate(graph_pids):
            if pid in label_map:
                node_label[i] = int(label_map[pid])
        same_lbl = np.logical_and(node_label[src] >= 0, node_label[src] == node_label[dst])
        edge_keep = np.logical_and(edge_keep, same_lbl)

    src = src[edge_keep].astype(np.int64)
    dst = dst[edge_keep].astype(np.int64)
    edge_w = graph["edge_weight"][edge_keep].astype(np.float32)

    teacher_mode = str(rel_cfg.get("teacher_prior_mode", "pos_minus_neg")).lower()
    t_np = _build_rel_teacher_vectors(
        patient_ids=graph_pids,
        mode=teacher_mode,
        embed_dim=embed_dim,
        pos_prior_map=pos_prior_map,
        neg_prior_map=neg_prior_map,
        label_map=label_map,
    )
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    t = F.normalize(t, dim=1, eps=1e-8).detach()

    state = {
        "pid_to_graph": pid_to_graph,
        "src": src,
        "dst": dst,
        "w": edge_w,
        "teacher_norm": t,
        "use_weighted_edges": bool(rel_cfg.get("use_weighted_edges", True)),
        "lambda_rel": float(rel_cfg.get("lambda_rel", 0.0)),
        "similarity": similarity,
    }
    summary.update({
        "reason": "ok",
        "graph_npz": str(graph_npz),
        "teacher_prior_mode": teacher_mode,
        "use_weighted_edges": state["use_weighted_edges"],
        "n_graph_nodes": int(len(graph_pids)),
        "n_edges_kept": int(src.shape[0]),
        "n_train_patients": int(len(train_pid_set)),
        "n_split_train_patients": int(len(split_info.train_patients)),
    })
    return state, summary


def _compute_relational_loss(
    emb: torch.Tensor,
    batch_patient_ids: list[str],
    rel_state: dict | None,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    zero = emb.new_zeros(())
    if rel_state is None:
        return zero, 0
    if emb.shape[0] == 0:
        return zero, 0

    pid_to_graph = rel_state["pid_to_graph"]
    src = rel_state["src"]
    dst = rel_state["dst"]
    w = rel_state["w"]
    if src.size == 0:
        return zero, 0

    n_nodes = rel_state["teacher_norm"].shape[0]
    lookup = np.full(n_nodes, -1, dtype=np.int64)
    for bidx, pid in enumerate(batch_patient_ids):
        gidx = pid_to_graph.get(str(pid).strip())
        if gidx is None:
            continue
        if lookup[gidx] < 0:
            lookup[gidx] = bidx

    mask = np.logical_and(lookup[src] >= 0, lookup[dst] >= 0)
    if not mask.any():
        return zero, 0

    src_sel = src[mask]
    dst_sel = dst[mask]
    bu = lookup[src_sel]
    bv = lookup[dst_sel]

    bu_t = torch.as_tensor(bu, dtype=torch.long, device=device)
    bv_t = torch.as_tensor(bv, dtype=torch.long, device=device)
    src_t = torch.as_tensor(src_sel, dtype=torch.long, device=device)
    dst_t = torch.as_tensor(dst_sel, dtype=torch.long, device=device)

    s_norm = F.normalize(emb, dim=1, eps=1e-8)
    su = s_norm[bu_t]
    sv = s_norm[bv_t]
    cos_s = (su * sv).sum(dim=1)

    t_norm = rel_state["teacher_norm"]
    tu = t_norm[src_t]
    tv = t_norm[dst_t]
    cos_t = (tu * tv).sum(dim=1).detach()

    loss_vec = (cos_s - cos_t).pow(2)
    if rel_state["use_weighted_edges"]:
        w_t = torch.as_tensor(w[mask], dtype=loss_vec.dtype, device=device)
        loss_rel = (w_t * loss_vec).mean()
    else:
        loss_rel = loss_vec.mean()
    return loss_rel, int(mask.sum())


def train_fundus_pushpull(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    train_cfg: dict,
    losses_cfg: dict,
    device: torch.device,
    ckpt_path: Path,
    pos_prior_map: dict | None,
    neg_prior_map: dict | None,
    embed_dim: int,
    rel_state: dict | None,
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
    push_w = float(losses_cfg.get("push_weight", 1.0))
    push_margin = float(losses_cfg.get("push_margin", 0.2))
    push_temp = losses_cfg.get("push_temperature", None)
    push_temp = float(push_temp) if push_temp is not None else None
    push_trigger = float(losses_cfg.get("push_trigger", 0.0))
    push_gate = str(losses_cfg.get("push_gate", "hard")).lower()
    push_gate_temp = float(losses_cfg.get("push_gate_temp", 0.1))
    lambda_rel = float(rel_state["lambda_rel"]) if rel_state is not None else 0.0

    best_val = -1e9
    best_epoch = -1
    bad = 0
    patience = int(train_cfg.get("patience", 10))
    history = []
    push_applied_batches = []

    for epoch in range(1, int(train_cfg.get("epochs", 50)) + 1):
        model.train()
        running = {
            "loss": 0.0,
            "loss_cls": 0.0,
            "loss_anchor": 0.0,
            "loss_pull": 0.0,
            "loss_push": 0.0,
            "loss_rel": 0.0,
            "push_applied": 0.0,
            "rel_edges": 0.0,
            "rel_applied": 0.0,
        }
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
            loss_pull = torch.tensor(0.0, device=device)
            loss_push = torch.tensor(0.0, device=device)
            loss_rel = torch.tensor(0.0, device=device)

            if "anchor" in batch and "is_anchor" in batch:
                anchor = batch["anchor"].to(device)
                is_anchor = batch["is_anchor"].to(device)
                if is_anchor.sum() > 0:
                    anchor_loss = compute_distill_loss(emb, anchor, distill_metric, distill_mse_w)
                    loss_anchor = (anchor_loss * is_anchor.squeeze(1)).sum() / is_anchor.sum()
                    loss = loss + anchor_w * loss_anchor

            if "prior" in batch and "confidence" in batch and pos_prior_map is not None and neg_prior_map is not None:
                prior = batch["prior"].to(device)
                conf = batch["confidence"].to(device)
                if "is_anchor" in batch:
                    is_anchor = batch["is_anchor"].to(device)
                    conf = conf * (1.0 - is_anchor)
                conf_sum = conf.sum()
                if conf_sum > 0:
                    pull = compute_distill_loss(emb, prior, distill_metric, distill_mse_w)
                    loss_pull = (pull * conf.squeeze(1)).sum() / conf_sum
                    loss = loss + distill_w * loss_pull

                    if push_w > 0:
                        neg_prior = _batch_opposite_priors(
                            patient_ids=batch["patient_id"],
                            labels=label,
                            pos_prior_map=pos_prior_map,
                            neg_prior_map=neg_prior_map,
                            embed_dim=embed_dim,
                            device=device,
                            dtype=emb.dtype,
                        )
                        emb_n = F.normalize(emb, dim=1)
                        pos_n = F.normalize(prior, dim=1)
                        neg_n = F.normalize(neg_prior, dim=1)
                        s_pos = (emb_n * pos_n).sum(dim=1)
                        s_neg = (emb_n * neg_n).sum(dim=1)
                        if push_temp is not None and push_temp > 0:
                            s_pos = s_pos / push_temp
                            s_neg = s_neg / push_temp
                        gap = s_pos - s_neg
                        if push_gate == "soft":
                            gate = torch.sigmoid((push_trigger - gap) / max(push_gate_temp, 1e-6))
                        else:
                            gate = (gap < push_trigger).float()
                        push = torch.relu(push_margin + s_neg - s_pos)
                        loss_push = (push * gate * conf.squeeze(1)).sum() / conf_sum
                        loss = loss + push_w * loss_push
                        running["push_applied"] += float(gate.mean().item())

            if lambda_rel > 0.0:
                loss_rel, n_rel_edges = _compute_relational_loss(
                    emb=emb,
                    batch_patient_ids=batch["patient_id"],
                    rel_state=rel_state,
                    device=device,
                )
                loss = loss + lambda_rel * loss_rel
                running["rel_edges"] += float(n_rel_edges)
                if n_rel_edges > 0:
                    running["rel_applied"] += 1.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running["loss"] += float(loss.item())
            running["loss_cls"] += float(loss_cls.item())
            running["loss_anchor"] += float(loss_anchor.item())
            running["loss_pull"] += float(loss_pull.item())
            running["loss_push"] += float(loss_push.item())
            running["loss_rel"] += float(loss_rel.item())
            n_batches += 1

        for k in running:
            running[k] /= max(n_batches, 1)
        push_applied_batches.append(running.get("push_applied", 0.0))

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

    push_applied_mean = float(np.mean(push_applied_batches)) if push_applied_batches else 0.0
    return TrainResult(best_epoch=best_epoch, best_val=best_val, history=history)


def _run_once(run_dir: Path, run_cfg: dict, phase_cfg: dict, data_cfg: dict) -> dict:
    mode_cfg = phase_cfg["mode"]
    model_cfg = phase_cfg["model"]
    losses_cfg = phase_cfg["losses"]
    train_cfg = dict(phase_cfg["train"])
    rel_cfg = dict(phase_cfg.get("relational_kd", {}))

    seed_all(int(train_cfg.get("seed", 0)))

    split_info = load_splits(data_cfg["splits_csv"])
    fundus_df = load_fundus_index(data_cfg["fundus_index_csv"], data_cfg.get("fundus_root"))
    if not fundus_df.empty:
        first_path = fundus_df.iloc[0]["fundus_path"]
        if not Path(first_path).is_file():
            raise FileNotFoundError(f"fundus_path not found: {first_path}")
    debug_fundus_paths(fundus_df, n=3)

    clinical_map, label_map, clinical_stats = build_clinical_map(
        clinical_csv=data_cfg["clinical_csv"],
        patient_ids=fundus_df["patient_id"].astype(str).tolist(),
        train_patients=split_info.train_patients,
        label_col=data_cfg.get("label_col", "htn"),
        clinical_cols=data_cfg.get("clinical_cols"),
        exclude_cols=data_cfg.get("clinical_exclude_cols"),
        binary_cols=data_cfg.get("binary_cols"),
        cont_cols=data_cfg.get("cont_cols"),
    )
    save_json(run_dir / "clinical_stats.json", clinical_stats.__dict__)

    embed_dim = int(model_cfg.get("embed_dim", 128))

    prior_map = None
    conf_map = None
    pos_prior_map = None
    neg_prior_map = None
    if mode_cfg.get("use_priors", False):
        pos_prior_map, pos_conf_map = load_priors(data_cfg["priors_pos_npz"])
        neg_prior_map, neg_conf_map = load_priors(data_cfg["priors_neg_npz"])
        prior_map, conf_map = _build_label_conditional_priors(
            label_map=label_map,
            pos_prior_map=pos_prior_map,
            pos_conf_map=pos_conf_map,
            neg_prior_map=neg_prior_map,
            neg_conf_map=neg_conf_map,
            embed_dim=embed_dim,
            use_confidence=bool(data_cfg.get("priors_use_confidence", True)),
        )

    anchor_map = None
    if mode_cfg.get("use_anchor", False):
        anchor_map = load_anchors(data_cfg["brain_teacher_embeddings"])

    samples_train, samples_val = build_samples(
        fundus_df=fundus_df,
        split_info=split_info,
        clinical_map=clinical_map,
        label_map=label_map,
        prior_map=prior_map,
        conf_map=conf_map,
        anchor_map=anchor_map,
        embed_dim=embed_dim,
        priors_for_val=bool(data_cfg.get("priors_for_val", True)),
    )

    if not samples_train:
        raise ValueError("No training samples found. Check splits and fundus_index.csv")
    if not samples_val:
        raise ValueError("No validation samples found. Check splits and fundus_index.csv")

    sampler = None
    if data_cfg.get("sampling", "none") == "weighted":
        ys = [s["label"] for s in samples_train]
        n0 = ys.count(0)
        n1 = ys.count(1)
        w0 = 1.0 / max(n0, 1)
        w1 = 1.0 / max(n1, 1)
        weights = [w1 if y == 1 else w0 for y in ys]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ds = FundusStudentDataset(
        samples=samples_train,
        image_size=int(data_cfg.get("image_size", 224)),
        augment=bool(data_cfg.get("augment", True)),
        use_clinical=bool(mode_cfg.get("use_clinical", False)),
        use_priors=bool(mode_cfg.get("use_priors", False)),
        use_anchor=bool(mode_cfg.get("use_anchor", False)),
    )

    val_ds = FundusStudentDataset(
        samples=samples_val,
        image_size=int(data_cfg.get("image_size", 224)),
        augment=False,
        use_clinical=bool(mode_cfg.get("use_clinical", False)),
        use_priors=bool(mode_cfg.get("use_priors", False)),
        use_anchor=bool(mode_cfg.get("use_anchor", False)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=max(1, int(data_cfg.get("num_workers", 4)) // 2),
        pin_memory=True,
    )

    model = FundusStudent(
        backbone=model_cfg.get("backbone", "resnet18"),
        pretrained=bool(model_cfg.get("pretrained", True)),
        embed_dim=embed_dim,
        use_clinical=bool(mode_cfg.get("use_clinical", False)),
        clinical_in_dim=len(clinical_stats.feature_cols),
        clinical_mlp_dim=int(model_cfg.get("clinical_mlp_dim", 128)),
        fusion=model_cfg.get("fusion", "concat"),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )

    train_cfg = dict(train_cfg)
    if train_cfg.get("pos_weight") is None and data_cfg.get("sampling", "none") != "weighted":
        train_cfg["pos_weight"] = _compute_pos_weight(samples_train)

    device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    ckpt_path = run_dir / "best.pt"
    rel_state, rel_summary = _prepare_relational_kd_state(
        rel_cfg=rel_cfg,
        samples_train=samples_train,
        split_info=split_info,
        pos_prior_map=pos_prior_map,
        neg_prior_map=neg_prior_map,
        label_map=label_map,
        embed_dim=embed_dim,
        device=device,
    )
    save_json(run_dir / "relational_kd_summary.json", rel_summary)

    train_res = train_fundus_pushpull(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        losses_cfg=losses_cfg,
        device=device,
        ckpt_path=ckpt_path,
        pos_prior_map=pos_prior_map,
        neg_prior_map=neg_prior_map,
        embed_dim=embed_dim,
        rel_state=rel_state,
    )

    save_history(run_dir / "train_history.json", train_res.history)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        val_metrics = evaluate_loader(model, val_loader, device)

    save_json(run_dir / "eval_summary.json", val_metrics)

    return {
        "best_epoch": train_res.best_epoch,
        "best_val": float(train_res.best_val),
        "eval_summary": val_metrics,
        "push_applied": float(np.mean([h["train"].get("push_applied", 0.0) for h in train_res.history])) if train_res.history else 0.0,
        "rel_edges_per_batch": float(np.mean([h["train"].get("rel_edges", 0.0) for h in train_res.history])) if train_res.history else 0.0,
        "rel_applied": float(np.mean([h["train"].get("rel_applied", 0.0) for h in train_res.history])) if train_res.history else 0.0,
    }


def main(config_path: str) -> None:
    cfg = load_yaml(config_path)
    unified = is_unified_config(cfg)
    if unified:
        run_cfg = cfg["run"]
        phase_cfg = cfg["phase5"]
        run_root = get_run_root(cfg)
        subname = phase_cfg.get("run_subname")
        run_dir = phase_dir(run_root, "phase5")
        if subname:
            run_dir = run_dir / str(subname)
    else:
        run_cfg = cfg["run"]
        phase_cfg = cfg
        run_root = None
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = run_cfg.get("name", f"phase5_fundus_{ts}")
        run_dir = Path(run_cfg.get("output_root", "logs/phase5_fundus")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if unified and run_root is not None:
        write_run_config(run_root, config_path)
    (run_dir / "config.yaml").write_text(Path(config_path).read_text())

    data_cfg = dict(phase_cfg["data"])
    rel_cfg = dict(phase_cfg.get("relational_kd", {}))

    if unified and run_root is not None:
        data_cfg["priors_pos_npz"] = resolve_path(
            data_cfg.get("priors_pos_npz"),
            phase_dir(run_root, "phase4") / "brain_priors_local_pos.npz",
        )
        data_cfg["priors_neg_npz"] = resolve_path(
            data_cfg.get("priors_neg_npz"),
            phase_dir(run_root, "phase4") / "brain_priors_local_neg.npz",
        )
        data_cfg["brain_teacher_embeddings"] = resolve_path(
            data_cfg.get("brain_teacher_embeddings"),
            phase_dir(run_root, "embeddings") / "brain_teacher_embeddings.pt",
        )
        if str(rel_cfg.get("edge_source", "fundus_train_graph")).lower() == "fundus_train_graph":
            rel_cfg["graph_npz"] = resolve_path(
                rel_cfg.get("graph_npz"),
                phase_dir(run_root, "graphs") / "graph_fundus_train.npz",
            )
        phase_cfg = dict(phase_cfg)
        phase_cfg["relational_kd"] = rel_cfg

    res = _run_once(run_dir, run_cfg, phase_cfg, data_cfg)
    save_json(run_dir / "eval_summary_all.json", {"local_posneg_pushpull": res})

    print(f"[DONE] run_dir={run_dir}")
    print(f"[DONE] wrote eval_summary_all.json")


    print(f"[DONE] push_applied={res['push_applied']:.4f}")
    print(f"[DONE] push_applied_percent={res['push_applied'] * 100:.2f}%")
    print(f"[DONE] rel_applied={res['rel_applied']:.4f}")
    print(f"[DONE] rel_applied_percent={res['rel_applied'] * 100:.2f}%")
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
