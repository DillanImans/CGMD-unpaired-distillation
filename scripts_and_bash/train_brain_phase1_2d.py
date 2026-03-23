from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.brain_teacher_slice_dataset import BrainTeacherSliceDataset
from models.brain_teacher.backbone_resnet2d import ResNet18Backbone2D, ResNet34Backbone2D
from models.brain_teacher.heads import EmbeddingHead, ClassifierHead
from models.brain_teacher.brain_teacher import BrainTeacher
from trainers.brain_teacher_trainer import BrainTeacherTrainer
from utils.splits import make_patient_split
from utils.run_config import (
    get_run_root,
    is_unified_config,
    load_yaml,
    phase_dir,
    write_run_config,
)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def slice_collate(batch):
    slices, ys, pids, scan_ids = zip(*batch)
    ys = torch.stack(ys, dim=0)
    slices = torch.stack(slices, dim=0)
    return slices, ys, list(pids), list(scan_ids)


def _model_param_stats(model: nn.Module) -> tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (total * 4) / (1024 ** 2)  # fp32 estimate
    return total, trainable, size_mb


def _backbone_depth_str(backbone: nn.Module) -> str:
    net = getattr(backbone, "net", None)
    if net is None:
        return "n/a"
    stage_counts = []
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(net, stage_name, None)
        if layer is not None:
            stage_counts.append(len(layer))
    if not stage_counts:
        return "n/a"
    return f"{stage_counts} (sum={sum(stage_counts)})"


def main(config_path: str):
    cfg = load_yaml(config_path)
    unified = is_unified_config(cfg)
    if unified:
        run_cfg = cfg["run"]
        phase_cfg = cfg["phase1_2d"]
        run_root = get_run_root(cfg)
        run_dir = phase_dir(run_root, "phase1_2d")
    else:
        run_cfg = cfg["run"]
        phase_cfg = cfg
        run_root = None
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(run_cfg["output_root"]) / f"run_{ts}"

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if unified and run_root is not None:
        write_run_config(run_root, config_path)
    (run_dir / "config.yaml").write_text(Path(config_path).read_text())

    set_seed(int(run_cfg["seed"]))

    splits_json = phase_cfg.get("split", {}).get("splits_json")
    if splits_json:
        with open(splits_json, "r") as f:
            split_blob = json.load(f)
        train_pids = split_blob.get("train_patient_ids", [])
        val_pids = split_blob.get("val_patient_ids", [])
        split_meta = {
            "splits_json": splits_json,
            "n_train_patients": len(train_pids),
            "n_val_patients": len(val_pids),
        }
    else:
        train_pids, val_pids, split_meta = make_patient_split(
            splits_csv=phase_cfg["data"]["splits_csv"],
            train_imaging_group=phase_cfg["split"]["train_imaging_group"],
            val_imaging_group=phase_cfg["split"].get("val_imaging_group", "both"),
            patient_index_csv=phase_cfg["data"]["patient_index_csv"],
        )
    (run_dir / "splits.json").write_text(json.dumps({
        "train_patient_ids": train_pids,
        "val_patient_ids": val_pids,
        "meta": split_meta,
    }, indent=2))

    train_pid_set = set(train_pids)
    val_pid_set = set(val_pids)

    df_pat_stats = pd.read_csv(phase_cfg["data"]["patient_index_csv"])
    df_pat_train = df_pat_stats[df_pat_stats["patient_id"].isin(train_pid_set)]
    df_pat_val = df_pat_stats[df_pat_stats["patient_id"].isin(val_pid_set)]
    n_train = len(train_pid_set)
    n_val = len(val_pid_set)
    n_train_pos = int((df_pat_train["htn"] == 1).sum())
    n_val_pos = int((df_pat_val["htn"] == 1).sum())
    train_prev = n_train_pos / max(n_train, 1)
    val_prev = n_val_pos / max(n_val, 1)
    print(f"[SPLIT] train patients={n_train} htn_pos={n_train_pos} prev={train_prev:.4f}")
    print(f"[SPLIT] val patients={n_val} htn_pos={n_val_pos} prev={val_prev:.4f}")

    label_col = phase_cfg["data"].get("label_col", "htn")

    num_slices = phase_cfg["slices"].get("num_slices", None)
    if num_slices is None or int(num_slices) <= 0:
        raise ValueError("phase1_2d.slices.num_slices must be set to a positive integer")
    slice_strategy = phase_cfg["slices"].get("strategy", "even")

    train_ds = BrainTeacherSliceDataset(
        patient_index_csv=phase_cfg["data"]["patient_index_csv"],
        brain_index_csv=phase_cfg["data"]["brain_index_csv"],
        brain_root=phase_cfg["data"]["brain_root"],
        patient_ids=train_pid_set,
        intensity_scale=bool(phase_cfg["transforms"]["intensity_scale"]),
        label_col=label_col,
        num_slices=int(num_slices),
        slice_strategy=slice_strategy,
        seed=int(run_cfg["seed"]),
    )

    val_ds = BrainTeacherSliceDataset(
        patient_index_csv=phase_cfg["data"]["patient_index_csv"],
        brain_index_csv=phase_cfg["data"]["brain_index_csv"],
        brain_root=phase_cfg["data"]["brain_root"],
        patient_ids=val_pid_set,
        intensity_scale=bool(phase_cfg["transforms"]["intensity_scale"]),
        label_col=label_col,
        num_slices=int(num_slices),
        slice_strategy=slice_strategy,
        seed=int(run_cfg["seed"]) + 1,
    )

    train_scan_pids = {s.patient_id for s in train_ds.samples}
    val_scan_pids = {s.patient_id for s in val_ds.samples}
    first_shape = None
    if len(train_ds) > 0:
        first_shape = tuple(train_ds[0][0].shape)  # [num_slices, 1, H, W]
    print(
        f"[DATA] train_scans={len(train_ds)} val_scans={len(val_ds)} "
        f"train_scan_patients={len(train_scan_pids)} val_scan_patients={len(val_scan_pids)} "
        f"slice_tensor_shape={first_shape} num_slices={int(num_slices)} strategy={slice_strategy}"
    )

    sampler = None
    if bool(phase_cfg["sampler"]["enabled"]):
        df_pat = pd.read_csv(phase_cfg["data"]["patient_index_csv"])
        pid_to_y = {r["patient_id"]: int(r[label_col]) for _, r in df_pat.iterrows()}

        pid_counts = {}
        for s in train_ds.samples:
            pid_counts[s.patient_id] = pid_counts.get(s.patient_id, 0) + 1

        ys = np.array([pid_to_y[s.patient_id] for s in train_ds.samples], dtype=np.int64)
        n0 = int((ys == 0).sum())
        n1 = int((ys == 1).sum())

        w0 = 1.0 / max(n0, 1)
        w1 = 1.0 / max(n1, 1)

        sample_weights = []
        for s in train_ds.samples:
            cls_w = w1 if pid_to_y[s.patient_id] == 1 else w0
            sample_weights.append(cls_w / pid_counts[s.patient_id])

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(phase_cfg["train"]["batch_size"]),
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=int(phase_cfg["train"]["num_workers"]),
        pin_memory=True,
        collate_fn=slice_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(phase_cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=max(1, int(phase_cfg["train"]["num_workers"]) // 2),
        pin_memory=True,
        collate_fn=slice_collate,
    )

    backbone_name = str(phase_cfg["model"].get("backbone", "resnet18_2d")).lower()
    if backbone_name == "resnet18_2d":
        backbone = ResNet18Backbone2D(
            in_ch=1,
            pretrained=bool(phase_cfg["model"].get("pretrained", False)),
        )
    elif backbone_name == "resnet34_2d":
        backbone = ResNet34Backbone2D(
            in_ch=1,
            pretrained=bool(phase_cfg["model"].get("pretrained", False)),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    embed_head = EmbeddingHead(
        in_dim=backbone.feat_dim,
        embed_dim=int(phase_cfg["model"]["embed_dim"]),
        normalize=bool(phase_cfg["model"]["normalize_embedding"]),
    )
    classifier = ClassifierHead(embed_dim=int(phase_cfg["model"]["embed_dim"]), out_dim=1)
    model = BrainTeacher(backbone, embed_head, classifier)
    total_params, trainable_params, model_size_mb = _model_param_stats(model)
    print(
        f"[MODEL] backbone={backbone_name} depth={_backbone_depth_str(backbone)} "
        f"feat_dim={backbone.feat_dim} embed_dim={int(phase_cfg['model']['embed_dim'])} "
        f"params_total={total_params:,} params_trainable={trainable_params:,} "
        f"size_fp32_mb={model_size_mb:.2f}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and bool(phase_cfg["train"]["multi_gpu_dataparallel"]):
        model = nn.DataParallel(model)

    model = model.to(device)

    df_pat_train = pd.read_csv(phase_cfg["data"]["patient_index_csv"])
    df_pat_train = df_pat_train[df_pat_train["patient_id"].isin(train_pid_set)]
    n_pos = int((df_pat_train[label_col] == 1).sum())
    n_neg = int((df_pat_train[label_col] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device, dtype=torch.float32)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(phase_cfg["train"]["lr"]),
        weight_decay=float(phase_cfg["train"]["weight_decay"]),
    )

    trainer = BrainTeacherTrainer(
        model=model,
        device=device,
        amp=bool(phase_cfg["train"]["amp"]),
        grad_clip_norm=float(phase_cfg["train"]["grad_clip_norm"]),
    )

    best_pat_bal = -1.0
    best_pat_auroc = -1.0
    start_epoch = 1

    resume = phase_cfg["train"].get("resume", None)
    if resume:
        ckpt = torch.load(resume, map_location=device)
        state = ckpt["model_state"]
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_pat_bal = float(ckpt.get("best_patient_bal_acc", best_pat_bal))
        best_pat_auroc = float(ckpt.get("best_patient_auroc", best_pat_auroc))

    use_youden = bool(phase_cfg.get("eval", {}).get("use_youden_threshold", True))

    metrics_path = run_dir / "metrics.json"

    for epoch in range(start_epoch, int(phase_cfg["train"]["epochs"]) + 1):
        train_loss = trainer.train_one_epoch_slice_agg(train_loader, optimizer, loss_fn)

        do_val = (epoch % int(phase_cfg["train"]["val_every"]) == 0)
        eval_res = None
        if do_val:
            eval_res = trainer.evaluate_slice_agg(val_loader, use_youden=use_youden)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": None if eval_res is None else {
                "scan_acc": eval_res.scan_acc,
                "scan_bal_acc": eval_res.scan_bal_acc,
                "scan_auroc": eval_res.scan_auroc,
                "scan_threshold": eval_res.scan_thresh,
                "patient_acc": eval_res.patient_acc,
                "patient_bal_acc": eval_res.patient_bal_acc,
                "patient_auroc": eval_res.patient_auroc,
                "patient_threshold": eval_res.patient_thresh,
                "scan_cm": eval_res.scan_cm.tolist(),
                "patient_cm": eval_res.patient_cm.tolist(),
            }
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        last_ckpt = {
            "epoch": epoch,
            "model_state": (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "best_patient_bal_acc": best_pat_bal,
            "best_patient_auroc": best_pat_auroc,
            "config": phase_cfg,
        }
        torch.save(last_ckpt, ckpt_dir / "last.pt")

        if eval_res is not None:
            if eval_res.patient_bal_acc > best_pat_bal:
                best_pat_bal = float(eval_res.patient_bal_acc)
            if eval_res.patient_auroc > best_pat_auroc:
                best_pat_auroc = float(eval_res.patient_auroc)
                best_ckpt = dict(last_ckpt)
                best_ckpt["best_patient_bal_acc"] = best_pat_bal
                best_ckpt["best_patient_auroc"] = best_pat_auroc
                torch.save(best_ckpt, ckpt_dir / "best.pt")

    print(f"[DONE] run_dir={run_dir}")
    print(f"[DONE] best_patient_bal_acc={best_pat_bal:.4f}")
    print(f"[DONE] best_patient_auroc={best_pat_auroc:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
