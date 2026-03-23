import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingHead(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, normalize: bool = True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.normalize = normalize

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        z = self.proj(feat)
        if self.normalize:
            z = F.normalize(z, dim = 1)
        return z
    
class ClassifierHead(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(embed_dim, out_dim)

    def forward(self, z):
        return self.fc(z).squeeze(1)