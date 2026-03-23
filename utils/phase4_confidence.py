from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def mc_predict_embeddings(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    k: int,
    device: torch.device,
    normalize_each_pass: bool = True,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if k < 2:
        raise ValueError("K must be >= 2 for Monte Carlo dropout")

    model.train()
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    sum_z = None
    sum_norm2 = None

    with torch.no_grad():
        for _ in range(k):
            z = model(x, edge_index, edge_weight)
            if normalize_each_pass:
                z = F.normalize(z, dim=1)
            if sum_z is None:
                sum_z = torch.zeros_like(z)
                sum_norm2 = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
            sum_z += z
            sum_norm2 += (z * z).sum(dim=1)

    mu = sum_z / float(k)
    mean_norm2 = sum_norm2 / float(k)
    u = mean_norm2 - (mu * mu).sum(dim=1)
    u = torch.clamp(u, min=0.0)

    if torch.isnan(mu).any() or torch.isnan(u).any():
        raise ValueError("NaNs detected in MC outputs")

    return mu, u


def uncertainty_to_confidence(u: np.ndarray, p_high: float = 95.0, eps: float = 1e-12) -> Tuple[np.ndarray, Dict[str, float]]:
    u_min = float(np.min(u))
    u_p = float(np.percentile(u, p_high))
    denom = max(u_p - u_min, eps)
    norm = np.clip((u - u_min) / denom, 0.0, 1.0)
    conf = 1.0 - norm
    meta = {
        "u_min": u_min,
        "u_p": u_p,
        "p_high": p_high,
        "eps": eps,
    }
    return conf.astype(np.float32), meta


def summary_stats(x: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(x)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }
