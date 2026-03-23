from __future__ import annotations

import json
import types
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class GraphData:
    patient_ids: List[str]
    edge_index: torch.Tensor
    edge_weight: torch.Tensor


@dataclass
class SplitMasks:
    train_mask_all: np.ndarray
    val_mask_all: np.ndarray
    train_brain_mask: np.ndarray
    val_brain_mask: np.ndarray
    has_brain: np.ndarray
    imaging_group: np.ndarray


@dataclass
class ClinicalData:
    x: torch.Tensor
    feature_cols: List[str]
    binary_cols: List[str]
    cont_cols: List[str]
    stats: Dict[str, Dict[str, float]]
    htn: np.ndarray


@dataclass
class BrainEmbeddings:
    embeddings: torch.Tensor
    has_brain: np.ndarray


def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_graph_npz(path: str | Path) -> GraphData:
    npz = np.load(path, allow_pickle=True)
    if "patient_ids" not in npz.files:
        raise ValueError("graph npz missing patient_ids")
    if "edge_index" not in npz.files:
        raise ValueError("graph npz missing edge_index")

    patient_ids = [pid.strip() for pid in npz["patient_ids"].astype(str).tolist()]
    edge_index = torch.from_numpy(npz["edge_index"].astype(np.int64))
    edge_weight = None
    if "edge_weight" in npz.files:
        edge_weight = torch.from_numpy(npz["edge_weight"].astype(np.float32))
    else:
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

    return GraphData(patient_ids=patient_ids, edge_index=edge_index, edge_weight=edge_weight)


def _ensure_monai_metatensor_stub() -> None:
    if "monai" in sys.modules:
        return
    monai = types.ModuleType("monai")
    data = types.ModuleType("monai.data")
    meta_tensor = types.ModuleType("monai.data.meta_tensor")

    class MetaTensor(torch.Tensor):
        pass

    meta_tensor.MetaTensor = MetaTensor
    data.meta_tensor = meta_tensor
    monai.data = data
    sys.modules["monai"] = monai
    sys.modules["monai.data"] = data
    sys.modules["monai.data.meta_tensor"] = meta_tensor


def load_brain_embeddings(path: str | Path) -> Tuple[List[str], torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        if "monai" not in str(exc):
            raise
        _ensure_monai_metatensor_stub()
        obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        if "patient" in obj and isinstance(obj["patient"], dict):
            pids = obj["patient"].get("patient_ids")
            emb = obj["patient"].get("z")
            if pids is not None and emb is not None:
                return [str(pid).strip() for pid in pids], torch.as_tensor(emb)
        if "patient_ids" in obj and "embeddings" in obj:
            return [str(pid).strip() for pid in obj["patient_ids"]], torch.as_tensor(obj["embeddings"])
        if "patient_ids" in obj and "z" in obj:
            return [str(pid).strip() for pid in obj["patient_ids"]], torch.as_tensor(obj["z"])

    raise ValueError("Unrecognized brain embeddings format")


def load_splits(path: str | Path, patient_ids: List[str]) -> SplitMasks:
    df = pd.read_csv(path)
    split_col = "split"
    if "brain_graph_split" in df.columns:
        split_col = "brain_graph_split"
    if not {"patient_id", split_col, "imaging_group"}.issubset(df.columns):
        raise ValueError("splits.csv must include patient_id, imaging_group, and split column")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df[split_col] = df[split_col].astype(str).str.strip()
    df["imaging_group"] = df["imaging_group"].astype(str).str.strip()

    split_map = dict(zip(df["patient_id"], df[split_col].astype(str)))
    group_map = dict(zip(df["patient_id"], df["imaging_group"].astype(str)))

    split = np.array([split_map.get(pid, "missing") for pid in patient_ids], dtype=object)
    imaging_group = np.array([group_map.get(pid, "missing") for pid in patient_ids], dtype=object)
    has_brain = np.isin(imaging_group, ["brain_only", "both"])

    train_mask_all = split == "train"
    val_mask_all = split == "val"
    train_brain_mask = np.logical_and(train_mask_all, has_brain)
    val_brain_mask = np.logical_and(val_mask_all, has_brain)

    n_train = int(train_mask_all.sum())
    n_val = int(val_mask_all.sum())
    n_missing = int((split == "missing").sum())
    print(f"[SPLITS] train={n_train} val={n_val} missing={n_missing}")
    if n_train == 0:
        raise ValueError("No train rows found after applying splits.csv to graph patient_ids")

    return SplitMasks(
        train_mask_all=train_mask_all,
        val_mask_all=val_mask_all,
        train_brain_mask=train_brain_mask,
        val_brain_mask=val_brain_mask,
        has_brain=has_brain,
        imaging_group=imaging_group,
    )


def _infer_feature_cols(
    df: pd.DataFrame,
    clinical_cols: Optional[List[str]],
    exclude_cols: List[str],
) -> List[str]:
    if clinical_cols:
        cols = list(clinical_cols)
    else:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in cols if c not in exclude_cols]
    if not cols:
        raise ValueError("No clinical feature columns selected")
    return cols


def _infer_binary_cols(df: pd.DataFrame, cols: List[str], train_mask: np.ndarray) -> List[str]:
    binary_cols = []
    for c in cols:
        vals = df.loc[train_mask, c].dropna().unique().tolist()
        if vals and set(vals).issubset({0, 1}):
            binary_cols.append(c)
    return binary_cols


def load_clinical_features(
    path: str | Path,
    patient_ids: List[str],
    train_mask_all: np.ndarray,
    clinical_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    binary_cols: Optional[List[str]] = None,
    cont_cols: Optional[List[str]] = None,
    label_col: str = "htn",
) -> ClinicalData:
    df = pd.read_csv(path)
    if "patient_id" not in df.columns:
        raise ValueError("clinical.csv must include patient_id")

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df = df.set_index("patient_id", drop=False)

    missing = [pid for pid in patient_ids if pid not in df.index]
    if missing:
        print(f"[WARN] {len(missing)} patient_ids missing in clinical.csv; imputing zeros")
        for pid in missing:
            df.loc[pid] = {"patient_id": pid}

    df = df.reindex(patient_ids)

    exclude_cols = exclude_cols or []
    feature_cols = _infer_feature_cols(df, clinical_cols, exclude_cols)

    if binary_cols is None:
        binary_cols = _infer_binary_cols(df, feature_cols, train_mask_all)
    if cont_cols is None:
        cont_cols = [c for c in feature_cols if c not in binary_cols]

    stats: Dict[str, Dict[str, float]] = {}
    df_feat = df[feature_cols].copy()

    train_mask = train_mask_all
    if train_mask.sum() == 0:
        raise ValueError("train_mask_all is empty; cannot compute clinical stats")

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

    x = torch.tensor(df_feat[feature_cols].to_numpy(dtype=np.float32))

    htn = None
    if label_col in df.columns:
        htn = df[label_col].to_numpy()
    else:
        htn = np.full(len(df), np.nan)

    return ClinicalData(
        x=x,
        feature_cols=feature_cols,
        binary_cols=binary_cols,
        cont_cols=cont_cols,
        stats=stats,
        htn=htn,
    )


def align_brain_embeddings(
    patient_ids: List[str],
    emb_pids: List[str],
    emb: torch.Tensor,
    has_brain: np.ndarray,
) -> BrainEmbeddings:
    pid_to_row = {str(pid).strip(): i for i, pid in enumerate(emb_pids)}
    n = len(patient_ids)
    d = emb.shape[1]
    out = torch.zeros((n, d), dtype=emb.dtype)

    found = 0
    for i, pid in enumerate(patient_ids):
        j = pid_to_row.get(pid)
        if j is not None:
            out[i] = emb[j]
            found += 1

    if found == 0:
        raise ValueError("No matching patient_ids between graph and brain embeddings")

    missing_brain = np.logical_and(has_brain, np.array([pid not in pid_to_row for pid in patient_ids]))
    if missing_brain.any():
        print(f"[WARN] {missing_brain.sum()} has_brain patients missing brain embeddings")

    return BrainEmbeddings(embeddings=out, has_brain=has_brain)


def save_json(path: str | Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))
