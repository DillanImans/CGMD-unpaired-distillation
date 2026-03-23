from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_neighbor_mean(x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    src = edge_index[0]
    dst = edge_index[1]
    if edge_weight is None:
        edge_weight = torch.ones_like(src, dtype=x.dtype)
    w = edge_weight.to(x.dtype)

    agg = torch.zeros_like(x)
    wsum = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    messages = x[dst] * w.unsqueeze(1)
    agg.index_add_(0, src, messages)
    wsum.index_add_(0, src, w)

    wsum = wsum.clamp_min(1e-12).unsqueeze(1)
    return agg / wsum


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        neigh = weighted_neighbor_mean(x, edge_index, edge_weight)
        h = torch.cat([x, neigh], dim=1)
        h = self.lin(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h


class GraphSAGEImputer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_mlp: bool = False,
        mlp_dim: int | None = None,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.normalize_output = normalize_output

        mlp_dim = mlp_dim or hidden_dim
        if use_mlp:
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, mlp_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            in_dim = mlp_dim
        else:
            self.encoder = None

        layers = []
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        if num_layers == 1:
            layers.append(GraphSAGELayer(in_dim, out_dim, dropout))
        else:
            layers.append(GraphSAGELayer(in_dim, hidden_dim, dropout))
            for _ in range(num_layers - 2):
                layers.append(GraphSAGELayer(hidden_dim, hidden_dim, dropout))
            layers.append(GraphSAGELayer(hidden_dim, out_dim, dropout))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        if self.encoder is not None:
            x = self.encoder(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        if self.normalize_output:
            x = F.normalize(x, dim=1)
        return x
