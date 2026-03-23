from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord


@dataclass
class BrainSliceSample:
    patient_id: str
    scan_id: str
    path: str
    date: Optional[str]
    y: int


class BrainTeacherSliceDataset(Dataset):
    def __init__(
        self,
        patient_index_csv: str,
        brain_index_csv: str,
        brain_root: str,
        patient_ids: Optional[set[str]] = None,
        intensity_scale: bool = True,
        label_col: str = "htn",
        num_slices: Optional[int] = None,
        slice_strategy: str = "even",
        seed: int = 0,
    ):
        self.patient_df = pd.read_csv(patient_index_csv)
        self.brain_df = pd.read_csv(brain_index_csv)

        assert "patient_id" in self.patient_df.columns
        if label_col not in self.patient_df.columns:
            raise ValueError(f"Label column not found in patient_index_csv: {label_col}")
        assert "patient_id" in self.brain_df.columns
        assert "filename" in self.brain_df.columns

        pid_to_y: Dict[str, int] = {
            r["patient_id"]: int(r[label_col])
            for _, r in self.patient_df.iterrows()
        }

        brain_root = Path(brain_root)

        samples: List[BrainSliceSample] = []
        for _, r in self.brain_df.iterrows():
            pid = str(r["patient_id"])
            if pid not in pid_to_y:
                continue
            if patient_ids is not None and pid not in patient_ids:
                continue

            fname = str(r["filename"])
            fpath = brain_root / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing scan file: {fpath}")

            date = str(r["date"]) if "date" in r and pd.notna(r["date"]) else None
            samples.append(
                BrainSliceSample(
                    patient_id=pid,
                    scan_id=fname,
                    path=str(fpath),
                    date=date,
                    y=pid_to_y[pid],
                )
            )

        self.samples = samples
        self.num_slices = num_slices
        self.slice_strategy = str(slice_strategy).lower()
        self.seed = seed

        tx = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
        ]
        if intensity_scale:
            tx.append(ScaleIntensityd(keys=["image"]))
        tx.append(ToTensord(keys=["image"]))
        self.transform = Compose(tx)

    def __len__(self) -> int:
        return len(self.samples)

    def _pick_indices(self, depth: int, idx: int) -> np.ndarray:
        if self.num_slices is None or self.num_slices <= 0 or depth <= self.num_slices:
            return np.arange(depth, dtype=np.int64)
        if self.slice_strategy == "center":
            start = max((depth - self.num_slices) // 2, 0)
            end = start + self.num_slices
            return np.arange(start, min(end, depth), dtype=np.int64)
        if self.slice_strategy == "random":
            rng = np.random.RandomState(self.seed + idx)
            replace = depth < self.num_slices
            return rng.choice(depth, size=self.num_slices, replace=replace)
        return np.linspace(0, depth - 1, self.num_slices, dtype=np.int64)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        data = self.transform({"image": s.path})
        img = data["image"]
        img = img.permute(0, 3, 1, 2)

        depth = img.shape[1]
        slice_idx = self._pick_indices(depth, idx)
        slices = img[:, slice_idx, :, :].permute(1, 0, 2, 3)

        y = torch.tensor(float(s.y), dtype=torch.float32)
        return slices, y, s.patient_id, s.scan_id
