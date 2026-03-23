import torch.nn as nn


class BrainTeacher(nn.Module):
    def __init__(self, backbone: nn.Module, embed_head: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.embed_head = embed_head
        self.classifier = classifier

    def forward(self, x):
        feat = self.backbone(x)
        z_scan = self.embed_head(feat)
        logits = self.classifier(z_scan)
        return z_scan, logits