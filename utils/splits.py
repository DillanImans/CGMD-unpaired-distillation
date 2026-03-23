from __future__ import annotations
import pandas as pd
from typing import Dict, List, Tuple


def make_patient_split(
    splits_csv: str,
    train_imaging_group: str | List[str],
    val_imaging_group: str | List[str] = "both",
    patient_index_csv: str | None = None,
    label_col: str = "htn",
    split_col: str | None = None,
    imaging_group_col: str = "imaging_group",
    patient_id_col: str = "patient_id",
) -> Tuple[List[str], List[str], Dict]:
    df = pd.read_csv(splits_csv)
    if split_col is None:
        if "brain_graph_split" in df.columns:
            split_col = "brain_graph_split"
        elif "split" in df.columns:
            split_col = "split"
    if split_col is None:
        raise ValueError("splits_csv missing split column (expected brain_graph_split or split)")
    missing_cols = {split_col, imaging_group_col, patient_id_col} - set(df.columns)
    if missing_cols:
        raise ValueError(f"splits_csv missing columns: {sorted(missing_cols)}")

    if isinstance(train_imaging_group, str) and train_imaging_group.lower() == "all":
        train_groups = df[imaging_group_col].dropna().unique().tolist()
    else:
        train_groups = [train_imaging_group] if isinstance(train_imaging_group, str) else train_imaging_group
    val_groups = [val_imaging_group] if isinstance(val_imaging_group, str) else val_imaging_group

    train_df = df[
        (df[split_col] == "train")
        & (df[imaging_group_col].isin(train_groups))
    ]
    val_df = df[
        (df[split_col] == "val")
        & (df[imaging_group_col].isin(val_groups))
    ]

    train_pids = train_df[patient_id_col].astype(str).tolist()
    val_pids = val_df[patient_id_col].astype(str).tolist()

    meta: Dict[str, object] = {
        "splits_csv": splits_csv,
        "train_imaging_group": train_groups,
        "val_imaging_group": val_groups,
        "n_train_patients": len(train_pids),
        "n_val_patients": len(val_pids),
    }

    if patient_index_csv:
        df_pat = pd.read_csv(patient_index_csv)
        if label_col in df_pat.columns:
            df_train = df_pat[df_pat[patient_id_col].isin(train_pids)]
            df_val = df_pat[df_pat[patient_id_col].isin(val_pids)]
            meta["train_label_counts"] = {
                "0": int((df_train[label_col] == 0).sum()),
                "1": int((df_train[label_col] == 1).sum()),
            }
            meta["val_label_counts"] = {
                "0": int((df_val[label_col] == 0).sum()),
                "1": int((df_val[label_col] == 1).sum()),
            }

    return train_pids, val_pids, meta
