from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _DenseEncoder(nn.Module):
    def __init__(self, in_channels: int, descriptor_dim: int = 128) -> None:
        super().__init__()
        channels = (32, 64, 128)
        layers: list[nn.Module] = []
        current = in_channels
        for output in channels:
            layers.extend([
                nn.Conv2d(current, output, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(_groups(output), output),
                nn.ReLU(inplace=True),
                nn.Conv2d(output, output, 3, padding=1, bias=False),
                nn.GroupNorm(_groups(output), output),
                nn.ReLU(inplace=True),
            ])
            current = output
        self.backbone = nn.Sequential(*layers)
        self.projection = nn.Conv2d(current, descriptor_dim, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.backbone(image)), dim=1, eps=1e-6)


class FundusDenseEncoder(_DenseEncoder):
    def __init__(self, descriptor_dim: int = 128) -> None:
        super().__init__(3, descriptor_dim)


class OctDenseEncoder(_DenseEncoder):
    def __init__(self, descriptor_dim: int = 128) -> None:
        super().__init__(1, descriptor_dim)


class Task2CommonModel(nn.Module):
    def __init__(self, descriptor_dim: int = 128) -> None:
        super().__init__()
        self.fundus_encoder = FundusDenseEncoder(descriptor_dim)
        self.oct_encoder = OctDenseEncoder(descriptor_dim)

    def forward(self, fundus: torch.Tensor, enface: torch.Tensor,
                valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if valid_mask is None:
            valid_mask = torch.ones_like(fundus[:, :1], dtype=torch.bool)
        if valid_mask.shape != fundus[:, :1].shape:
            raise ValueError("valid_mask must have shape (B,1,H,W)")
        fundus_desc = self.fundus_encoder(fundus * valid_mask.to(fundus.dtype))
        oct_desc = self.oct_encoder(enface)
        descriptor_valid = F.interpolate(valid_mask.float(), size=fundus_desc.shape[-2:],
                                         mode="nearest").bool()
        return {"fundus_desc": fundus_desc, "oct_desc": oct_desc,
                "descriptor_valid_mask": descriptor_valid}
