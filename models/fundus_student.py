from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    try:
        import torchvision.models as models
    except Exception as exc:
        raise ImportError("torchvision is required for fundus backbones") from exc

    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim

    if name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim

    if name == "vit":
        if not hasattr(models, "vit_b_16"):
            raise ValueError("vit_b_16 not available in torchvision version")
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        feat_dim = model.heads.head.in_features
        model.heads.head = nn.Identity()
        return model, feat_dim

    raise ValueError(f"Unsupported backbone: {name}")


class FundusStudent(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        embed_dim: int,
        use_clinical: bool,
        clinical_in_dim: int,
        clinical_mlp_dim: int,
        fusion: str = "concat",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_clinical = use_clinical
        self.fusion = fusion

        self.backbone, feat_dim = _build_backbone(backbone, pretrained)
        self.feat_dim = feat_dim
        self.embed_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if use_clinical:
            self.clinical_mlp = nn.Sequential(
                nn.Linear(clinical_in_dim, clinical_mlp_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            if fusion != "concat":
                raise ValueError("Only concat fusion is supported")
            clf_in = embed_dim + clinical_mlp_dim
        else:
            self.clinical_mlp = None
            clf_in = embed_dim

        self.classifier = nn.Linear(clf_in, 1)

    def forward(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor | None = None,
        return_feat: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(image)
        emb = self.embed_proj(feat)
        emb = F.normalize(emb, dim=1)

        if self.use_clinical and clinical is not None:
            c = self.clinical_mlp(clinical)
            fused = torch.cat([emb, c], dim=1)
        else:
            fused = emb

        logits = self.classifier(fused)
        if return_feat:
            return logits, emb, feat
        return logits, emb
