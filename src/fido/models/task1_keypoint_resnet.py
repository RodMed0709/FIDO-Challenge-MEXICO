from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from fido.heatmap_decode import soft_argmax_2d


class Task1KeypointResNet18FPN(nn.Module):
    """ResNet-18 con FPN C2-C5 y heatmap de stride 4."""

    def __init__(self, pretrained: bool = False, freeze_backbone: bool = False,
                 fpn_channels: int = 64, heatmap_temperature: float = 1.0) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.fpn = nn.ModuleDict({
            "c2": nn.Conv2d(64, fpn_channels, 1),
            "c3": nn.Conv2d(128, fpn_channels, 1),
            "c4": nn.Conv2d(256, fpn_channels, 1),
            "c5": nn.Conv2d(512, fpn_channels, 1),
        })
        self.smooth = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            nn.GroupNorm(8, fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(fpn_channels, 1, 1)
        self.heatmap_temperature = heatmap_temperature
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> Task1KeypointResNet18FPN:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _features(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        backbone = self.backbone
        x = backbone.relu(backbone.bn1(backbone.conv1(image)))
        x = backbone.maxpool(x)
        c2 = backbone.layer1(x)
        c3 = backbone.layer2(c2)
        c4 = backbone.layer3(c3)
        c5 = backbone.layer4(c4)
        return c2, c3, c4, c5

    def forward(self, fundus: torch.Tensor, fundus_size: int | None = None) -> dict:
        image_size = float(fundus_size or fundus.shape[-1])
        normalized = (fundus - self.imagenet_mean) / self.imagenet_std
        c2, c3, c4, c5 = self._features(normalized)
        p5 = self.fpn["c5"](c5)
        p4 = self.fpn["c4"](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.fpn["c3"](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.fpn["c2"](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        heatmap_logits = self.heatmap_head(self.smooth(p2))
        coordinates = soft_argmax_2d(heatmap_logits, temperature=self.heatmap_temperature)
        stride = image_size / heatmap_logits.shape[-1]
        return {"keypoint": coordinates[:, 0] * stride, "heatmap_logits": heatmap_logits}
