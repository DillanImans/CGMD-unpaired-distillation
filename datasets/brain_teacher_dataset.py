from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord


# schema
@dataclass
class BrainSample:
    patient_id: str
    scan_id: str
    path: str
    date: Optional[str]
    y: int


class BrainTeacherScanDataset(Dataset):
    def __init__(
        self,
        patient_index_csv: str,
        brain_index_csv: str,
        brain_root: str,
        patient_ids: Optional[set[str]] = None,
        intensity_scale: bool = True,
        label_col: str = "htn",
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

        samples: List[BrainSample] = []
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
                BrainSample(
                    patient_id = pid,
                    scan_id = fname,
                    path = str(fpath),
                    date = date,
                    y = pid_to_y[pid],
                )
            )

        self.samples = samples

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
    
    def __getitem__(self, idx: int):
        s = self.samples[idx]
        data = self.transform({"image": s.path})
        img = data["image"]
        img = img.permute(0, 3, 1, 2)

        y = torch.tensor(float(s.y), dtype=torch.float32)
        return img, y, s.patient_id, s.scan_id

class BrainTeacherPatientDataset(Dataset):
    def __init__(
        self,
        patient_index_csv: str,
        brain_index_csv: str,
        brain_root: str,
        patient_ids: Optional[set[str]] = None,
        intensity_scale: bool = True,
        label_col: str = "htn",
        max_scans: Optional[int] = None,
        sample_strategy: str = "all",
        seed: int = 0,
    ):
        self.patient_df = pd.read_csv(patient_index_csv)
        self.brain_df = pd.read_csv(brain_index_csv)

        assert "patient_id" in self.patient_df.columns
        if label_col not in self.patient_df.columns:
            raise ValueError(f"Label column not found in patient_index_csv: {label_col}")
        assert "patient_id" in self.brain_df.columns
        assert "filename" in self.brain_df.columns

        self.pid_to_y: Dict[str, int] = {
            r["patient_id"]: int(r[label_col])
            for _, r in self.patient_df.iterrows()
        }

        brain_root = Path(brain_root)

        pid_to_samples: Dict[str, List[BrainSample]] = {}
        for _, r in self.brain_df.iterrows():
            pid = str(r["patient_id"])
            if pid not in self.pid_to_y:
                continue
            if patient_ids is not None and pid not in patient_ids:
                continue

            fname = str(r["filename"])
            fpath = brain_root / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing scan file: {fpath}")

            date = str(r["date"]) if "date" in r and pd.notna(r["date"]) else None
            sample = BrainSample(
                patient_id=pid,
                scan_id=fname,
                path=str(fpath),
                date=date,
                y=self.pid_to_y[pid],
            )
            pid_to_samples.setdefault(pid, []).append(sample)

        self.patient_ids = sorted(pid_to_samples.keys())
        self.pid_to_samples = pid_to_samples
        self.patient_labels = [self.pid_to_y[pid] for pid in self.patient_ids]
        self.max_scans = max_scans
        self.sample_strategy = str(sample_strategy).lower()
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
        return len(self.patient_ids)

    def _select_samples(self, idx: int, samples: List[BrainSample]) -> List[BrainSample]:
        if self.max_scans is None or self.max_scans <= 0 or len(samples) <= self.max_scans:
            return samples
        if self.sample_strategy == "random":
            rng = np.random.RandomState(self.seed + idx)
            idx = rng.choice(len(samples), size=self.max_scans, replace=False)
            return [samples[i] for i in idx]
        return samples[: self.max_scans]

    def __getitem__(self, idx: int):
        pid = self.patient_ids[idx]
        samples = self._select_samples(idx, self.pid_to_samples[pid])
        imgs = []
        scan_ids = []
        for s in samples:
            data = self.transform({"image": s.path})
            img = data["image"]
            img = img.permute(0, 3, 1, 2)
            imgs.append(img)
            scan_ids.append(s.scan_id)

        x = torch.stack(imgs, dim=0)
        y = torch.tensor(float(self.pid_to_y[pid]), dtype=torch.float32)
        return x, y, pid, scan_ids
