from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils.metrics import confusion_and_balanced_acc, aggregate_patient_probs, roc_auc_score, best_threshold_by_youden



@dataclass
class EvalResult:
    scan_acc: float
    scan_bal_acc: float
    scan_auroc: float
    patient_acc: float
    patient_bal_acc: float
    patient_auroc: float
    scan_cm: np.ndarray
    patient_cm: np.ndarray
    scan_thresh: float | None = None
    patient_thresh: float | None = None


class BrainTeacherTrainer:
    def __init__(self, model: nn.Module, device: torch.device, amp: bool, grad_clip_norm: float):
        self.model = model
        self.device = device
        self.amp = amp
        self.grad_clip_norm = grad_clip_norm
        self.scaler = GradScaler(enabled=amp)

    def train_one_epoch(self, loader, optimizer, loss_fn):
        self.model.train()
        running, total = 0.0, 0

        pbar = tqdm(loader, desc="Train", leave=False)
        for x, y, _, _ in pbar:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.amp):
                _, logits = self.model(x)
                loss = loss_fn(logits, y)

            self.scaler.scale(loss).backward()

            if self.grad_clip_norm and self.grad_clip_norm > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(optimizer)
            self.scaler.update()

            bs = x.size(0)
            running += float(loss.item()) * bs
            total += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return running / max(total, 1)

    def train_one_epoch_patient_agg(self, loader, optimizer, loss_fn):
        self.model.train()
        running, total = 0.0, 0

        pbar = tqdm(loader, desc="Train", leave=False)
        for scans_list, y, _, _ in pbar:
            y = y.to(self.device, non_blocking=True)
            scans_list = [s.to(self.device, non_blocking=True) for s in scans_list]

            optimizer.zero_grad(set_to_none=True)

            counts = [s.shape[0] for s in scans_list]
            x = torch.cat(scans_list, dim=0)

            with autocast(enabled=self.amp):
                _, logits = self.model(x)
                agg_logits = []
                start = 0
                for c in counts:
                    agg_logits.append(logits[start:start + c].mean(dim=0))
                    start += c
                agg_logits = torch.stack(agg_logits, dim=0)
                loss = loss_fn(agg_logits, y)

            self.scaler.scale(loss).backward()

            if self.grad_clip_norm and self.grad_clip_norm > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(optimizer)
            self.scaler.update()

            bs = y.size(0)
            running += float(loss.item()) * bs
            total += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return running / max(total, 1)
    def train_one_epoch_slice_agg(self, loader, optimizer, loss_fn):
        self.model.train()
        running, total = 0.0, 0

        pbar = tqdm(loader, desc="Train", leave=False)
        for slices, y, _, _ in pbar:
            y = y.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if isinstance(slices, torch.Tensor):
                slices = slices.to(self.device, non_blocking=True)
                bsz, nslices = slices.shape[:2]
                x = slices.view(bsz * nslices, *slices.shape[2:])
                counts = [nslices] * bsz
            else:
                slices = [s.to(self.device, non_blocking=True) for s in slices]
                counts = [s.shape[0] for s in slices]
                x = torch.cat(slices, dim=0)

            with autocast(enabled=self.amp):
                _, logits = self.model(x)
                agg_logits = []
                start = 0
                for c in counts:
                    agg_logits.append(logits[start:start + c].mean(dim=0))
                    start += c
                agg_logits = torch.stack(agg_logits, dim=0)
                loss = loss_fn(agg_logits, y)

            self.scaler.scale(loss).backward()

            if self.grad_clip_norm and self.grad_clip_norm > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(optimizer)
            self.scaler.update()

            bs = y.size(0)
            running += float(loss.item()) * bs
            total += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return running / max(total, 1)


    @torch.no_grad()
    def evaluate_slice_agg(self, loader, use_youden: bool = True):
        self.model.eval()

        scan_probs = []
        scan_labels = []
        scan_pids = []

        for slices, y, patient_id, _ in loader:
            y = y.to(self.device, non_blocking=True)
            slices = slices.to(self.device, non_blocking=True)

            bsz, nslices = slices.shape[:2]
            x = slices.view(bsz * nslices, *slices.shape[2:])

            _, logits = self.model(x)
            probs = torch.sigmoid(logits).view(bsz, nslices, -1).mean(dim=1)

            scan_probs.append(probs.detach().cpu().numpy())
            scan_labels.append(y.detach().cpu().numpy())
            scan_pids.extend(list(patient_id))

        probs = np.concatenate(scan_probs).astype(np.float32)
        labels = np.concatenate(scan_labels).astype(np.int64)

        if use_youden:
            scan_thresh, scan_bal, scan_acc, scan_cm = best_threshold_by_youden(labels, probs)
        else:
            scan_thresh = 0.5
            pred_scan = (probs >= scan_thresh).astype(np.int64)
            scan_acc, scan_bal, scan_cm, _ = confusion_and_balanced_acc(labels, pred_scan)
        scan_auroc = roc_auc_score(labels, probs)

        _, p_patient, y_patient = aggregate_patient_probs(scan_pids, probs, labels)
        if use_youden:
            patient_thresh, pat_bal, pat_acc, pat_cm = best_threshold_by_youden(y_patient, p_patient)
        else:
            patient_thresh = 0.5
            pred_patient = (p_patient >= patient_thresh).astype(np.int64)
            pat_acc, pat_bal, pat_cm, _ = confusion_and_balanced_acc(y_patient, pred_patient)
        patient_auroc = roc_auc_score(y_patient, p_patient)

        return EvalResult(
            scan_acc=scan_acc,
            scan_bal_acc=scan_bal,
            scan_auroc=scan_auroc,
            patient_acc=pat_acc,
            patient_bal_acc=pat_bal,
            patient_auroc=patient_auroc,
            scan_cm=scan_cm,
            patient_cm=pat_cm,
            scan_thresh=scan_thresh,
            patient_thresh=patient_thresh,
        )


    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        probs = []
        labels = []
        pids = []

        for x, y, patient_id, _ in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            _, logits = self.model(x)
            p = torch.sigmoid(logits).detach().cpu().numpy()

            probs.append(p)
            labels.append(y.detach().cpu().numpy())
            pids.extend(list(patient_id))

        probs = np.concatenate(probs).astype(np.float32)
        labels = np.concatenate(labels).astype(np.int64)

        # scan-level
        scan_thresh, scan_bal, scan_acc, scan_cm = best_threshold_by_youden(labels, probs)
        scan_auroc = roc_auc_score(labels, probs)

        # patient-level (mean prob per patient
        _, p_patient, y_patient = aggregate_patient_probs(pids, probs, labels)
        patient_thresh, pat_bal, pat_acc, pat_cm = best_threshold_by_youden(y_patient, p_patient)
        patient_auroc = roc_auc_score(y_patient, p_patient)

        return EvalResult(
            scan_acc = scan_acc,
            scan_bal_acc = scan_bal,
            scan_auroc = scan_auroc,
            patient_acc = pat_acc,
            patient_bal_acc = pat_bal,
            patient_auroc = patient_auroc,
            scan_cm = scan_cm,
            patient_cm = pat_cm,
            scan_thresh = scan_thresh,
            patient_thresh = patient_thresh
        )
    
    