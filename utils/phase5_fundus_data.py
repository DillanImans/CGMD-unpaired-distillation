from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.phase3_imputer_data import load_brain_embeddings


@dataclass
class SplitInfo:
    train_patients: List[str]
    val_patients: List[str]
    imaging_group: Dict[str, str]
    anchor_flag: Optional[Dict[str, float]] = None


@dataclass
class ClinicalStats:
    feature_cols: List[str]
    binary_cols: List[str]
    cont_cols: List[str]
    stats: Dict[str, Dict[str, float]]


class FundusStudentDataset(Dataset):
    def __init__(
        self,
        samples: List[dict],
        image_size: int,
        augment: bool,
        use_clinical: bool,
        use_priors: bool,
        use_anchor: bool,
    ):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.use_clinical = use_clinical
        self.use_priors = use_priors
        self.use_anchor = use_anchor
        self.transform = _build_transforms(image_size, augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = Image.open(s["fundus_path"]).convert("RGB")
        x = self.transform(img)

        out = {
            "image": x,
            "patient_id": s["patient_id"],
            "label": torch.tensor([s["label"]], dtype=torch.float32),
            "imaging_group": s["imaging_group"],
        }

        if self.use_clinical:
            out["clinical"] = torch.tensor(s["clinical"], dtype=torch.float32)
        if self.use_priors:
            out["prior"] = torch.tensor(s["prior"], dtype=torch.float32)
            out["confidence"] = torch.tensor([s["confidence"]], dtype=torch.float32)
        if self.use_anchor:
            out["anchor"] = torch.tensor(s["anchor"], dtype=torch.float32)
            out["is_anchor"] = torch.tensor([s["is_anchor"]], dtype=torch.float32)

        return out


def _build_transforms(image_size: int, augment: bool):
    try:
        import torchvision.transforms as T
    except Exception as exc:
        raise ImportError("torchvision is required for fundus transforms") from exc

    if augment:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_splits(path: str | Path) -> SplitInfo:
    df = pd.read_csv(path)
    split_col = "split"
    if "fundus_split" in df.columns:
        split_col = "fundus_split"
    if not {"patient_id", split_col, "imaging_group"}.issubset(df.columns):
        raise ValueError("splits.csv must include patient_id, imaging_group, and split column")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df[split_col] = df[split_col].astype(str).str.strip()
    df["imaging_group"] = df["imaging_group"].astype(str).str.strip()

    imaging_group = dict(zip(df["patient_id"], df["imaging_group"]))
    anchor_flag = None
    if "anchor_flag" in df.columns:
        anchor_flag = dict(zip(df["patient_id"], df["anchor_flag"].astype(float)))
    train_patients = df[df[split_col] == "train"]["patient_id"].tolist()
    val_patients = df[df[split_col] == "val"]["patient_id"].tolist()

    return SplitInfo(
        train_patients=train_patients,
        val_patients=val_patients,
        imaging_group=imaging_group,
        anchor_flag=anchor_flag,
    )


def load_fundus_index(path: str | Path, fundus_root: Optional[str | Path] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "patient_id" not in df.columns:
        raise ValueError("fundus_index.csv must include patient_id")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()

    root = Path(fundus_root).expanduser().resolve() if fundus_root is not None else None

    if "fundus_path" in df.columns:
        df["fundus_path"] = df["fundus_path"].astype(str).str.strip()
        if root is not None:
            df["fundus_path"] = df["fundus_path"].apply(
                lambda p: str(root / p) if not Path(p).is_absolute() else p
            )
        return _filter_missing_fundus(df)

    if "filename" in df.columns:
        if fundus_root is None:
            raise ValueError("fundus_root is required when fundus_index.csv uses filename")
        df["fundus_path"] = df["filename"].astype(str).str.strip().apply(lambda x: str(root / x))
        return _filter_missing_fundus(df)

    raise ValueError("fundus_index.csv must include fundus_path or filename")
    return df


def _check_fundus_paths(paths: pd.Series) -> None:
    for p in paths.head(10).tolist():
        if not Path(p).is_file():
            raise FileNotFoundError(f"fundus file not found: {p}")


def debug_fundus_paths(df: pd.DataFrame, n: int = 5) -> None:
    print(f"[FUNDUS] sample paths (n={n}):")
    for p in df["fundus_path"].head(n).tolist():
        print(f"  {p}")


def _filter_missing_fundus(df: pd.DataFrame) -> pd.DataFrame:
    paths = df["fundus_path"].tolist()
    exists = [Path(p).is_file() for p in paths]
    missing = [p for p, ok in zip(paths, exists) if not ok]
    if missing:
        print(f"[WARN] missing fundus files: {len(missing)} (dropping)")
        for p in missing[:5]:
            print(f"  missing: {p}")
    return df[exists].reset_index(drop=True)


def _infer_binary_cols(df: pd.DataFrame, cols: List[str], train_mask: np.ndarray) -> List[str]:
    binary_cols = []
    for c in cols:
        vals = df.loc[train_mask, c].dropna().unique().tolist()
        if vals and set(vals).issubset({0, 1}):
            binary_cols.append(c)
    return binary_cols


def build_clinical_map(
    clinical_csv: str | Path,
    patient_ids: List[str],
    train_patients: List[str],
    label_col: str,
    clinical_cols: Optional[List[str]],
    exclude_cols: Optional[List[str]],
    binary_cols: Optional[List[str]],
    cont_cols: Optional[List[str]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], ClinicalStats]:
    df = pd.read_csv(clinical_csv)
    if "patient_id" not in df.columns:
        raise ValueError("clinical.csv must include patient_id")

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df = df.set_index("patient_id", drop=False)

    exclude_cols = exclude_cols or []
    if clinical_cols:
        feature_cols = list(clinical_cols)
    else:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in exclude_cols]
    if not feature_cols:
        raise ValueError("No clinical feature columns selected")

    train_mask = df.index.isin(train_patients)
    if train_mask.sum() == 0:
        raise ValueError("No train patients found for clinical preprocessing")

    if binary_cols is None:
        binary_cols = _infer_binary_cols(df, feature_cols, train_mask)
    if cont_cols is None:
        cont_cols = [c for c in feature_cols if c not in binary_cols]

    stats: Dict[str, Dict[str, float]] = {}
    df_feat = df[feature_cols].copy()

    for c in feature_cols:
        col = df_feat[c]
        if c in binary_cols:
            mode = col[train_mask].mode(dropna=True)
            fill = float(mode.iloc[0]) if not mode.empty else 0.0
            df_feat[c] = col.fillna(fill).astype(float)
            stats[c] = {"impute": fill}
        else:
            median = float(col[train_mask].median(skipna=True))
            df_feat[c] = col.fillna(median)
            stats[c] = {"impute": median}

    for c in cont_cols:
        vals = df_feat.loc[train_mask, c]
        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        if not std or std <= 1e-12:
            std = 1.0
        df_feat[c] = (df_feat[c] - mean) / std
        stats[c].update({"mean": mean, "std": std})

    clinical_map = {pid: df_feat.loc[pid].to_numpy(dtype=np.float32) for pid in patient_ids if pid in df_feat.index}
    label_map = {pid: int(df.loc[pid, label_col]) for pid in patient_ids if pid in df.index}

    return clinical_map, label_map, ClinicalStats(feature_cols, binary_cols, cont_cols, stats)


def load_priors(priors_npz: str | Path) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    npz = np.load(priors_npz, allow_pickle=True)
    if not {"patient_ids", "embeddings"}.issubset(npz.files):
        raise ValueError("priors_npz must include patient_ids and embeddings")
    patient_ids = npz["patient_ids"].astype(str)
    embeddings = npz["embeddings"].astype(np.float32)
    confidence = npz["confidence"].astype(np.float32) if "confidence" in npz.files else None

    prior_map = {pid: emb for pid, emb in zip(patient_ids, embeddings)}
    conf_map = {pid: float(c) for pid, c in zip(patient_ids, confidence)} if confidence is not None else {}
    return prior_map, conf_map


def load_anchors(teacher_path: str | Path) -> Dict[str, np.ndarray]:
    pids, emb = load_brain_embeddings(teacher_path)
    emb = emb.cpu().numpy().astype(np.float32)
    return {pid: vec for pid, vec in zip(pids, emb)}


def build_samples(
    fundus_df: pd.DataFrame,
    split_info: SplitInfo,
    clinical_map: Dict[str, np.ndarray],
    label_map: Dict[str, int],
    prior_map: Optional[Dict[str, np.ndarray]],
    conf_map: Optional[Dict[str, float]],
    anchor_map: Optional[Dict[str, np.ndarray]],
    embed_dim: int,
    priors_for_val: bool = True,
) -> Tuple[List[dict], List[dict]]:
    samples_train: List[dict] = []
    samples_val: List[dict] = []

    for _, row in fundus_df.iterrows():
        pid = str(row["patient_id"]).strip()
        if pid not in label_map:
            continue
        if pid not in clinical_map:
            continue

        imaging_group = split_info.imaging_group.get(pid, "missing")
        if imaging_group == "brain_only":
            continue

        s = {
            "patient_id": pid,
            "fundus_path": row["fundus_path"],
            "label": int(label_map[pid]),
            "clinical": clinical_map[pid],
            "imaging_group": imaging_group,
        }

        if prior_map is not None:
            prior = prior_map.get(pid, np.zeros(embed_dim, dtype=np.float32))
            conf = conf_map.get(pid, 0.0) if conf_map is not None else 1.0
            if not priors_for_val and pid in split_info.val_patients:
                prior = np.zeros(embed_dim, dtype=np.float32)
                conf = 0.0
            s["prior"] = prior
            s["confidence"] = float(conf)

        if anchor_map is not None:
            s["anchor"] = anchor_map.get(pid, np.zeros(embed_dim, dtype=np.float32))
            if split_info.anchor_flag is not None:
                s["is_anchor"] = float(split_info.anchor_flag.get(pid, 0.0))
            else:
                s["is_anchor"] = float(imaging_group == "both")

        if pid in split_info.train_patients:
            samples_train.append(s)
        elif pid in split_info.val_patients:
            samples_val.append(s)

    return samples_train, samples_val
