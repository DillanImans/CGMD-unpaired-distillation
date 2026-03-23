import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from datasets.brain_teacher_slice_dataset import BrainTeacherSliceDataset
from models.brain_teacher.brain_teacher import BrainTeacher
from models.brain_teacher.backbone_resnet2d import ResNet18Backbone2D, ResNet34Backbone2D
from models.brain_teacher.heads import EmbeddingHead, ClassifierHead
from utils.run_config import (
    get_run_root,
    is_unified_config,
    load_yaml,
    phase_dir,
    resolve_path,
    write_run_config,
)


def slice_collate(batch):
    slices, ys, pids, scan_ids = zip(*batch)
    ys = torch.stack(ys, dim=0)
    slices = torch.stack(slices, dim=0)
    return slices, ys, list(pids), list(scan_ids)


@torch.no_grad()
def main(config_path):
    cfg = load_yaml(config_path)
    unified = is_unified_config(cfg)
    if unified:
        run_root = get_run_root(cfg)
        phase_cfg = cfg["phase1_2d"]
        ecfg = dict(phase_cfg.get("export", {}))
        data_cfg = phase_cfg.get("data", {})
    else:
        run_root = None
        ecfg = cfg["export"]
        data_cfg = cfg.get("data", {})

    if unified and run_root is not None:
        default_ckpt = phase_dir(run_root, "phase1_2d") / "checkpoints" / "best.pt"
        default_out = phase_dir(run_root, "embeddings") / "brain_teacher_embeddings.pt"
        ecfg["checkpoint"] = resolve_path(ecfg.get("checkpoint"), default_ckpt)
        ecfg["out_path"] = resolve_path(ecfg.get("out_path"), default_out)
        if "patient_index_csv" not in ecfg:
            ecfg["patient_index_csv"] = data_cfg.get("patient_index_csv")
        if "brain_index_csv" not in ecfg:
            ecfg["brain_index_csv"] = data_cfg.get("brain_index_csv")
        if "brain_root" not in ecfg:
            ecfg["brain_root"] = data_cfg.get("brain_root")

    patient_index_csv = ecfg["patient_index_csv"]
    brain_index_csv = ecfg["brain_index_csv"]
    brain_root = ecfg["brain_root"]
    ckpt_path = ecfg["checkpoint"]
    out_path = ecfg["out_path"]
    batch_size = ecfg.get("batch_size", 8)
    num_workers = ecfg.get("num_workers", 4)
    splits_csv = ecfg.get("splits_csv") or data_cfg.get("splits_csv")
    export_splits = ecfg.get("export_splits", ["train", "val"])

    if unified and run_root is not None:
        write_run_config(run_root, config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_patient_ids = None
    if splits_csv:
        df_splits = pd.read_csv(splits_csv)
        split_col = "split"
        if "brain_graph_split" in df_splits.columns:
            split_col = "brain_graph_split"
        if not {"patient_id", split_col, "imaging_group"}.issubset(df_splits.columns):
            raise ValueError("splits_csv must include patient_id, imaging_group, and split column")
        mask = df_splits[split_col].isin(export_splits)
        mask &= df_splits["imaging_group"].isin(["brain_only", "both"])
        train_patient_ids = df_splits.loc[mask, "patient_id"].astype(str).tolist()
        print(f"[EXPORT] using splits={export_splits} with brain_only/both: {len(train_patient_ids)}")

    num_slices = int(phase_cfg["slices"]["num_slices"])
    slice_strategy = phase_cfg["slices"].get("strategy", "even")

    ds = BrainTeacherSliceDataset(
        patient_index_csv=patient_index_csv,
        brain_index_csv=brain_index_csv,
        brain_root=brain_root,
        patient_ids=None if train_patient_ids is None else set(train_patient_ids),
        intensity_scale=True,
        num_slices=num_slices,
        slice_strategy=slice_strategy,
        seed=int(cfg["run"]["seed"]) if unified else 0,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
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

    model = BrainTeacher(backbone, embed_head, classifier).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scan_z = []
    scan_patient_ids = []
    scan_ids = []

    for slices, _, patient_id, scan_id in loader:
        slices = slices.to(device, non_blocking=True)

        bsz, nslices = slices.shape[:2]
        x = slices.view(bsz * nslices, *slices.shape[2:])

        z, _ = model(x)
        z = z.view(bsz, nslices, -1).mean(dim=1)

        scan_z.append(z.cpu())
        scan_patient_ids.extend(patient_id)
        scan_ids.extend(scan_id)

    scan_z = torch.cat(scan_z, dim=0)

    pid_to_rows = defaultdict(list)
    for i, pid in enumerate(scan_patient_ids):
        pid_to_rows[pid].append(i)

    patient_ids = sorted(pid_to_rows.keys())
    patient_z = []
    patient_n_scans = []

    for pid in patient_ids:
        rows = pid_to_rows[pid]
        z = scan_z[rows].mean(dim=0)
        z = torch.nn.functional.normalize(z, dim=0)
        patient_z.append(z)
        patient_n_scans.append(len(rows))

    patient_z = torch.stack(patient_z, dim=0)

    pid_to_prow = {pid: i for i, pid in enumerate(patient_ids)}

    out = {
        "meta": {
            "teacher_ckpt": str(ckpt_path),
            "embed_dim": patient_z.shape[1],
            "pooling": "mean",
            "normalized": True,
            "num_slices": num_slices,
            "slice_strategy": slice_strategy,
        },
        "scan": {
            "z": scan_z,
            "patient_ids": scan_patient_ids,
            "scan_ids": scan_ids,
        },
        "patient": {
            "z": patient_z,
            "patient_ids": patient_ids,
            "n_scans": torch.tensor(patient_n_scans),
        },
        "index": {
            "patient_id_to_scan_rows": dict(pid_to_rows),
            "patient_id_to_patient_row": pid_to_prow,
        },
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)

    print(f"[DONE] saved embeddings to {out_path}")
    print(f"Scans: {scan_z.shape[0]}, Patients: {patient_z.shape[0]}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    main(args.config)
